//
//  CoreMLExecutor.mm
//  MNN
//
//  Created by MNN on 2021/03/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "CoreMLDefine.h"
#import "CoreMLExecutor.h"

#import <CommonCrypto/CommonDigest.h>
#import <mach/mach.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <limits>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

bool isAvailable() {
#if !defined(__APPLE__)
    return false;
#endif
#if (TARGET_OS_IPHONE)
    if (@available(iOS 11, *)) {
        return true;
    } else {
        return false;
    }
#else
    return true;
#endif
}

namespace {
NSURL* createTemporaryFile() {
    NSURL* temporaryDirectoryURL = [NSURL fileURLWithPath:NSTemporaryDirectory() isDirectory:YES];
    NSString* temporaryFilename = [[NSProcessInfo processInfo] globallyUniqueString];
    NSURL* temporaryFileURL = [temporaryDirectoryURL URLByAppendingPathComponent:temporaryFilename];
    return temporaryFileURL;
}

std::string lowerAndDash(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    std::replace(value.begin(), value.end(), '_', '-');
    return value;
}

bool envFlagEnabled(const char* name) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return false;
    }
    const auto value = lowerAndDash(raw);
    return !(value == "0" || value == "false" || value == "off" || value == "none");
}

struct CoreMLProfileCounter {
    double total_ms = 0.0;
    int calls = 0;
};

std::mutex& coreMLProfileMutex() {
    static auto* mutex = new std::mutex;
    return *mutex;
}

std::unordered_map<std::string, CoreMLProfileCounter>& coreMLProfileCounters() {
    static auto* counters = new std::unordered_map<std::string, CoreMLProfileCounter>;
    return *counters;
}

void printCoreMLProfile() {
    std::vector<std::pair<std::string, CoreMLProfileCounter>> rows;
    {
        std::lock_guard<std::mutex> lock(coreMLProfileMutex());
        const auto& counters = coreMLProfileCounters();
        rows.reserve(counters.size());
        for (const auto& item : counters) {
            rows.push_back(item);
        }
    }
    if (rows.empty()) {
        return;
    }
    std::sort(rows.begin(), rows.end(), [](const std::pair<std::string, CoreMLProfileCounter>& a,
                                           const std::pair<std::string, CoreMLProfileCounter>& b) {
        return a.second.total_ms > b.second.total_ms;
    });
    std::fprintf(stderr, "\n[MNN CoreML backend profile]\n");
    std::fprintf(stderr, "%-34s %12s %9s %10s\n", "stage", "total_ms", "calls", "avg_ms");
    for (const auto& row : rows) {
        const auto& counter = row.second;
        const double avg = counter.calls > 0 ? counter.total_ms / counter.calls : 0.0;
        std::fprintf(stderr, "%-34s %12.2f %9d %10.2f\n",
                     row.first.c_str(), counter.total_ms, counter.calls, avg);
    }
}

bool coreMLProfileEnabled() {
    static const bool enabled = []() {
        const bool result = envFlagEnabled("MNN_COREML_PROFILE");
        if (result) {
            std::atexit(printCoreMLProfile);
        }
        return result;
    }();
    return enabled;
}

double elapsedMs(std::chrono::steady_clock::time_point start) {
    const auto elapsed = std::chrono::steady_clock::now() - start;
    return std::chrono::duration<double, std::milli>(elapsed).count();
}

uint64_t currentResidentBytes() {
    mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    const kern_return_t status = task_info(mach_task_self(),
                                           MACH_TASK_BASIC_INFO,
                                           reinterpret_cast<task_info_t>(&info),
                                           &count);
    if (status != KERN_SUCCESS) {
        return 0;
    }
    return static_cast<uint64_t>(info.resident_size);
}

void appendCoreMLTrace(const std::string& message) {
    const char* path = std::getenv("MNN_COREML_TRACE_FILE");
    if (path == nullptr || path[0] == '\0') {
        return;
    }
    std::ofstream out(path, std::ios::out | std::ios::app);
    if (!out) {
        return;
    }
    out << message << " resident=" << currentResidentBytes() << "\n";
}

void recordCoreMLProfile(const std::string& stage, double ms) {
    if (!coreMLProfileEnabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(coreMLProfileMutex());
    auto& counter = coreMLProfileCounters()[stage];
    counter.total_ms += ms;
    counter.calls += 1;
}

class ScopedCoreMLProfile {
public:
    explicit ScopedCoreMLProfile(std::string stage)
        : stage_(std::move(stage)), enabled_(coreMLProfileEnabled()), start_(std::chrono::steady_clock::now()) {}

    ~ScopedCoreMLProfile() {
        if (enabled_) {
            recordCoreMLProfile(stage_, elapsedMs(start_));
        }
    }

private:
    std::string stage_;
    bool enabled_ = false;
    std::chrono::steady_clock::time_point start_;
};

NSString* coreMLCacheDirectoryFromEnvironment() {
    const char* raw = std::getenv("MNN_COREML_CACHE_DIR");
    if (raw == nullptr || raw[0] == '\0') {
        return nil;
    }
    const std::string value(raw);
    const auto normalized = lowerAndDash(value);
    if (normalized == "0" || normalized == "false" || normalized == "off" || normalized == "none") {
        return nil;
    }
    NSString* path = nil;
    if (normalized == "1" || normalized == "true" || normalized == "default") {
        path = [NSTemporaryDirectory() stringByAppendingPathComponent:@"mnn-coreml-cache-v1"];
    } else {
        path = [NSString stringWithUTF8String:raw];
        path = [path stringByExpandingTildeInPath];
    }
    return [path stringByStandardizingPath];
}

std::string sha256Hex(const uint8_t* data, size_t size) {
    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_CTX ctx;
    CC_SHA256_Init(&ctx);
    while (size > 0) {
        const auto chunk = static_cast<CC_LONG>(std::min<size_t>(size, std::numeric_limits<CC_LONG>::max()));
        CC_SHA256_Update(&ctx, data, chunk);
        data += chunk;
        size -= chunk;
    }
    CC_SHA256_Final(digest, &ctx);

    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (unsigned char byte : digest) {
        stream << std::setw(2) << static_cast<int>(byte);
    }
    return stream.str();
}

MLComputeUnits computeUnitsFromEnvironment() {
    const char* raw = std::getenv("MNN_COREML_COMPUTE_UNITS");
    if (raw == nullptr || raw[0] == '\0') {
        return MLComputeUnitsAll;
    }

    const auto value = lowerAndDash(raw);

    if (value == "all" || value == "default" || value == "cpu-gpu-ne" || value == "cpu-gpu-ane") {
        return MLComputeUnitsAll;
    }
    if (value == "cpu" || value == "cpu-only" || value == "cpuonly") {
        return MLComputeUnitsCPUOnly;
    }
    if (value == "gpu" || value == "cpu-gpu" || value == "cpuandgpu") {
        return MLComputeUnitsCPUAndGPU;
    }
    if (value == "ne" || value == "ane" || value == "npu" || value == "cpu-ne" ||
        value == "cpu-ane" || value == "neural-engine" || value == "cpuandneuralengine") {
        if (@available(macOS 13.0, iOS 16.0, tvOS 16.0, watchOS 9.0, *)) {
            return MLComputeUnitsCPUAndNeuralEngine;
        }
        NSLog(@"MNN_COREML_COMPUTE_UNITS=%s requires macOS 13 / iOS 16; using MLComputeUnitsAll", raw);
        return MLComputeUnitsAll;
    }

    NSLog(@"Unknown MNN_COREML_COMPUTE_UNITS=%s; using MLComputeUnitsAll", raw);
    return MLComputeUnitsAll;
}
}  // namespace

@interface CoreMLExecutor ()
@property NSString* modelCacheKey;
@property BOOL compiledModelIsCached;
- (NSURL*)cacheURLForCurrentModel;
@end

@interface MultiArrayFeatureProvider : NSObject <MLFeatureProvider> {
    NSMutableDictionary* _inputs;
    NSSet* _featureNames;
}

- (instancetype)initWithInputs:(const std::vector<std::pair<const MNN::Tensor*, std::string>>*)inputs useImage:(bool)useImage
                 coreMlVersion:(int)coreMlVersion;
- (MLFeatureValue*)featureValueForName:(NSString*)featureName API_AVAILABLE(ios(11));
- (NSSet<NSString*>*)featureNames;

@property(nonatomic, readonly) int coreMlVersion;

@end

@implementation MultiArrayFeatureProvider

- (instancetype)initWithInputs:(const std::vector<std::pair<const MNN::Tensor*, std::string>>*)inputs useImage:(bool)useImage
                  coreMlVersion:(int)coreMlVersion {
    self = [super init];
    _inputs = [NSMutableDictionary dictionaryWithCapacity:inputs->size()];
    _coreMlVersion = coreMlVersion;
    _featureNames = nil;
    NSMutableArray* names = [[NSMutableArray alloc] init];
    for (auto& input : *inputs) {
        MLFeatureValue* value = nil;
        auto tensor = input.first;
        NSError* error = nil;
        NSString* name = [NSString stringWithCString:input.second.c_str() encoding:[NSString defaultCStringEncoding]];
        if (useImage) {
            CVPixelBufferRef pixelBuffer = NULL;
            OSType pixelFormat = kCVPixelFormatType_OneComponent8;
            size_t bytePerRow = tensor->width();
            CVReturn status = CVPixelBufferCreateWithBytes(nil, tensor->width(), tensor->height(), pixelFormat,
                                                           tensor->host<void>(), bytePerRow, nil, nil, nil, &pixelBuffer);
            if (status != kCVReturnSuccess) {
                NSLog(@"Failed to create CVPixelBufferRef for feature %@", name);
                return nil;
            }
            value = [MLFeatureValue featureValueWithPixelBuffer:pixelBuffer];
        } else {
            auto input_shape = input.first->shape();
            NSMutableArray* shape = [NSMutableArray arrayWithCapacity:input_shape.size()];
            NSMutableArray* strides = [NSMutableArray arrayWithCapacity:input_shape.size()];
            std::vector<int> stridesDim(input_shape.size());
            int curStride = 1;
            if (input_shape.size() >= 1) {
                for (int i=input_shape.size()-1; i>=0; --i) {
                    stridesDim[i] = curStride;
                    curStride *= input_shape[i];
                }
            }
            for (int i=0; i<input_shape.size(); ++i) {
                [shape addObject:@(input_shape[i])];
                [strides addObject:@(stridesDim[i])];
            }
            MLMultiArray* mlArray = [[MLMultiArray alloc] initWithDataPointer:tensor->host<float>()
                                                                        shape:shape
                                                                     dataType:MLMultiArrayDataTypeFloat32
                                                                      strides:strides
                                                                  deallocator:(^(void* bytes){})error:&error];
            if (error != nil) {
                NSLog(@"Failed to create MLMultiArray for feature %@ error: %@", name, [error localizedDescription]);
                return nil;
            }
            value= [MLFeatureValue featureValueWithMultiArray:mlArray];
        }
        [names addObject:name];
        [_inputs setValue:value forKey:(name)];
    }
    _featureNames = [NSSet setWithArray:names];
    return self;
}

- (NSSet<NSString*>*)featureNames {
    return _featureNames;
}

- (MLFeatureValue*)featureValueForName:(NSString*)featureName {
    return _inputs[featureName];
}
@end

@implementation CoreMLExecutor
- (bool)invokeWithInputs:(const std::vector<std::pair<const MNN::Tensor*, std::string>>&)inputs
                 outputs:(const std::vector<std::pair<const MNN::Tensor*, std::string>>&)outputs {
    ScopedCoreMLProfile totalProfile("invoke.total");
    appendCoreMLTrace("coreml.invoke.begin");
    if (_model == nil) {
        return NO;
    }

    @autoreleasepool{
        _outputArray = nil;
        _outputArray = [NSMutableArray arrayWithCapacity:0];
        NSError* error = nil;
        bool useImage = _precision == 2;
        MultiArrayFeatureProvider* inputFeature = nil;
        {
            ScopedCoreMLProfile profile("invoke.feature_provider");
            appendCoreMLTrace("coreml.invoke.feature_provider.begin");
            inputFeature = [[MultiArrayFeatureProvider alloc] initWithInputs:&inputs useImage:useImage coreMlVersion:[self coreMlVersion]];
            appendCoreMLTrace(inputFeature == nil ? "coreml.invoke.feature_provider.failed" : "coreml.invoke.feature_provider.end");
        }
        if (inputFeature == nil) {
            NSLog(@"inputFeature is not initialized.");
            return NO;
        }
        MLPredictionOptions* options = [[MLPredictionOptions alloc] init];
        // options.usesCPUOnly = true;
        id<MLFeatureProvider> _outputFeature = nil;
        {
            ScopedCoreMLProfile profile("invoke.prediction");
            appendCoreMLTrace("coreml.invoke.prediction.begin");
            _outputFeature = [_model predictionFromFeatures:inputFeature
                                                    options:options
                                                      error:&error];
            appendCoreMLTrace(_outputFeature == nil ? "coreml.invoke.prediction.failed" : "coreml.invoke.prediction.end");
        }
        if (error != nil) {
            NSLog(@"Error executing model: %@", [error localizedDescription]);
            return NO;
        }
        {
            ScopedCoreMLProfile profile("invoke.output_bind");
            appendCoreMLTrace("coreml.invoke.output_bind.begin");
            NSSet<NSString*>* outputFeatureNames = [_outputFeature featureNames];
            for (auto& output : outputs) {
                NSString* outputName = [NSString stringWithCString:output.second.c_str()
                                                          encoding:[NSString defaultCStringEncoding]];
                MLFeatureValue* outputValue = [_outputFeature featureValueForName:[outputFeatureNames member:outputName]];
                if ([outputValue type] == MLFeatureTypeImage) {
                    auto data = [outputValue imageBufferValue];
                    CVPixelBufferLockBaseAddress(data, kCVPixelBufferLock_ReadOnly);
                    auto pixelbuffer = (unsigned char*)CVPixelBufferGetBaseAddress(data);
                    auto width = CVPixelBufferGetWidth(data);
                    auto byte_per_row = CVPixelBufferGetBytesPerRow(data);
                    for (int row = 0; row < CVPixelBufferGetHeight(data); row++) {
                        memcpy(const_cast<MNN::Tensor*>(output.first)->buffer().host + row * width, pixelbuffer + row * byte_per_row, width);
                    }
                    CVPixelBufferUnlockBaseAddress(data, kCVPixelBufferLock_ReadOnly);
                } else {
                    auto* data = [outputValue multiArrayValue];
                    if (data.dataPointer == nullptr) {
                        return NO;
                    }
                    [_outputArray addObject:data];
                    const_cast<MNN::Tensor*>(output.first)->buffer().host = (unsigned char*)data.dataPointer;
                }
            }
            appendCoreMLTrace("coreml.invoke.output_bind.end");
        }
        inputFeature = nil;
    }
    appendCoreMLTrace("coreml.invoke.end");
    return YES;
}

- (bool)cleanup {
    NSError* error = nil;
    NSFileManager* fileManager = [NSFileManager defaultManager];
    if (_mlModelFilePath.length > 0 && [fileManager fileExistsAtPath:_mlModelFilePath]) {
        [fileManager removeItemAtPath:_mlModelFilePath error:&error];
        if (error != nil) {
            NSLog(@"Failed cleaning up model: %@", [error localizedDescription]);
            return NO;
        }
    }
    if (!_compiledModelIsCached && _compiledModelFilePath.length > 0 &&
        [fileManager fileExistsAtPath:_compiledModelFilePath]) {
        [fileManager removeItemAtPath:_compiledModelFilePath error:&error];
        if (error != nil) {
            NSLog(@"Failed cleaning up compiled model: %@", [error localizedDescription]);
            return NO;
        }
    }
    return YES;
}

- (NSURL*)saveModel:(CoreML__Specification__Model*)model {
    return [self saveModel:model forceWrite:NO];
}

- (NSURL*)saveModel:(CoreML__Specification__Model*)model forceWrite:(BOOL)forceWrite {
    ScopedCoreMLProfile totalProfile("build.save_model.total");
    appendCoreMLTrace("coreml.save_model.begin");
    _modelCacheKey = nil;
    _compiledModelIsCached = NO;
    NSURL* modelUrl = createTemporaryFile();
    NSString* modelPath = [modelUrl path];
    if (model->specificationversion == 3) {
        _coreMlVersion = 2;
    } else if (model->specificationversion == 4) {
        _coreMlVersion = 3;
    } else {
        NSLog(@"Only Core ML models with specification version 3 or 4 are supported");
        return nil;
    }
    size_t modelSize = 0;
    {
        ScopedCoreMLProfile profile("build.get_packed_size");
        modelSize = core_ml__specification__model__get_packed_size(model);
    }
    appendCoreMLTrace("coreml.save_model.packed_size=" + std::to_string(modelSize));
    std::unique_ptr<uint8_t[]> writeBuffer(new uint8_t[modelSize]);
    appendCoreMLTrace("coreml.save_model.write_buffer_allocated");
    {
        ScopedCoreMLProfile profile("build.pack_model");
        core_ml__specification__model__pack(model, writeBuffer.get());
    }
    appendCoreMLTrace("coreml.save_model.pack_done");
    {
        ScopedCoreMLProfile profile("build.hash_model");
        const auto hash = sha256Hex(writeBuffer.get(), modelSize);
        _modelCacheKey = [NSString stringWithFormat:@"mnn-coreml-v1-%s", hash.c_str()];
    }
    appendCoreMLTrace("coreml.save_model.hash_done");
    NSURL* cacheURL = [self cacheURLForCurrentModel];
    if (!forceWrite && cacheURL != nil && [[NSFileManager defaultManager] fileExistsAtPath:[cacheURL path]]) {
        recordCoreMLProfile("build.skip_source_write_cache_hit", 0.0);
        appendCoreMLTrace("coreml.save_model.cache_hit");
        return nil;
    }
    // TODO: Can we mmap this instead of actual writing it to phone ?
    {
        ScopedCoreMLProfile profile("build.write_model_file");
        std::ofstream file_stream([modelPath UTF8String], std::ios::out | std::ios::binary);
        const char* ptr = reinterpret_cast<const char*>(writeBuffer.get());
        file_stream.write(ptr, modelSize);
    }
    appendCoreMLTrace("coreml.save_model.write_done");
    return modelUrl;
}

- (MLModel*)loadCompiledModel:(NSURL*)compileUrl error:(NSError**)error {
    ScopedCoreMLProfile profile("build.load_mlmodel");
    appendCoreMLTrace("coreml.load_mlmodel.begin");
    MLModel* model = nil;
    if (@available(iOS 12.0, *)) {
        MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
        config.computeUnits = computeUnitsFromEnvironment();
        model = [MLModel modelWithContentsOfURL:compileUrl configuration:config error:error];
    } else {
        model = [MLModel modelWithContentsOfURL:compileUrl error:error];
    }
    appendCoreMLTrace(model == nil ? "coreml.load_mlmodel.failed" : "coreml.load_mlmodel.end");
    return model;
}

- (NSURL*)cacheURLForCurrentModel {
    ScopedCoreMLProfile profile("build.cache_lookup");
    NSString* cacheDirectory = coreMLCacheDirectoryFromEnvironment();
    if (cacheDirectory.length == 0 || _modelCacheKey.length == 0) {
        return nil;
    }
    NSError* error = nil;
    NSFileManager* fileManager = [NSFileManager defaultManager];
    if (![fileManager createDirectoryAtPath:cacheDirectory withIntermediateDirectories:YES attributes:nil error:&error]) {
        NSLog(@"Failed creating CoreML cache directory %@: %@", cacheDirectory, [error localizedDescription]);
        return nil;
    }
    NSString* cacheName = [_modelCacheKey stringByAppendingString:@".mlmodelc"];
    return [NSURL fileURLWithPath:[cacheDirectory stringByAppendingPathComponent:cacheName] isDirectory:YES];
}

- (void)copyCompiledModel:(NSURL*)compiledURL toCacheURL:(NSURL*)cacheURL {
    ScopedCoreMLProfile profile("build.cache_copy");
    if (compiledURL == nil || cacheURL == nil) {
        return;
    }
    NSFileManager* fileManager = [NSFileManager defaultManager];
    if ([fileManager fileExistsAtPath:[cacheURL path]]) {
        return;
    }
    NSError* error = nil;
    if (![fileManager copyItemAtURL:compiledURL toURL:cacheURL error:&error]) {
        if (![fileManager fileExistsAtPath:[cacheURL path]]) {
            NSLog(@"Failed caching compiled CoreML model at %@: %@", [cacheURL path], [error localizedDescription]);
        }
    }
}

- (bool)build:(NSURL*)modelUrl {
    ScopedCoreMLProfile totalProfile("build.total");
    appendCoreMLTrace("coreml.build.begin");
    NSError* error = nil;
    NSFileManager* fileManager = [NSFileManager defaultManager];
    NSURL* cacheUrl = [self cacheURLForCurrentModel];
    NSURL* compileUrl = nil;
    BOOL loadedFromCache = NO;

    if (cacheUrl != nil && [fileManager fileExistsAtPath:[cacheUrl path]]) {
        compileUrl = cacheUrl;
        loadedFromCache = YES;
        appendCoreMLTrace("coreml.build.cache_hit");
    }

    if (compileUrl == nil) {
        if (modelUrl == nil) {
            NSLog(@"CoreML cache miss but no source model file is available for compilation");
            return NO;
        }
        {
            ScopedCoreMLProfile profile("build.compile_model");
            appendCoreMLTrace("coreml.compile.begin");
            compileUrl = [MLModel compileModelAtURL:modelUrl error:&error];
            appendCoreMLTrace(compileUrl == nil ? "coreml.compile.failed" : "coreml.compile.end");
        }
        if (error != nil) {
            NSLog(@"Error compiling model %@", [error localizedDescription]);
            return NO;
        }
        [self copyCompiledModel:compileUrl toCacheURL:cacheUrl];
    }
    _mlModelFilePath = [modelUrl path];
    _compiledModelFilePath = [compileUrl path];
    _compiledModelIsCached = loadedFromCache;

    appendCoreMLTrace("coreml.build.load_begin");
    _model = [self loadCompiledModel:compileUrl error:&error];
    if (error != nil && loadedFromCache) {
        NSLog(@"Error loading cached CoreML model %@; recompiling. %@", [compileUrl path], [error localizedDescription]);
        [fileManager removeItemAtURL:compileUrl error:nil];
        if (modelUrl == nil) {
            NSLog(@"Cannot recompile cached CoreML model because the source model file was skipped");
            return NO;
        }
        error = nil;
        {
            ScopedCoreMLProfile profile("build.recompile_model");
            appendCoreMLTrace("coreml.recompile.begin");
            compileUrl = [MLModel compileModelAtURL:modelUrl error:&error];
            appendCoreMLTrace(compileUrl == nil ? "coreml.recompile.failed" : "coreml.recompile.end");
        }
        if (error != nil) {
            NSLog(@"Error compiling model %@", [error localizedDescription]);
            return NO;
        }
        [self copyCompiledModel:compileUrl toCacheURL:cacheUrl];
        _compiledModelFilePath = [compileUrl path];
        _compiledModelIsCached = NO;
        error = nil;
        appendCoreMLTrace("coreml.build.reload_begin");
        _model = [self loadCompiledModel:compileUrl error:&error];
    }
    if (error != NULL) {
        NSLog(@"Error Creating MLModel %@", [error localizedDescription]);
        return NO;
    }
    appendCoreMLTrace("coreml.build.end");
    return YES;
}
@end

@implementation RasterLayer
- (instancetype)initWithParameterDictionary:(NSDictionary<NSString *,id> *)parameters
                                      error:(NSError * _Nullable *)error {
    self = [super init];
    return self;
}
- (void) setRegionSampler
{
    samplers.resize(regions.size());
    for (int r = 0; r < regions.size(); r++) {
        const Region& region = regions[r];
        SamplerInfo& sampler = samplers[r];
        int sizeTotal = 1;
        for (int i=0; i<3; ++i) {
            sampler.size[i] = region.size[i];
            sampler.stride[i] = region.src.stride[i];
            sampler.extent[i] = region.dst.stride[i];
            sizeTotal *= region.size[i];
        }
        sampler.size[3] = sizeTotal;
        sampler.stride[3] = region.src.offset;
        sampler.extent[3] = region.dst.offset;
    }
}

- (std::pair<MTLSize, MTLSize>)computeBestGroupAndLocal:(SamplerInfo&) s {
    MTLSize t = MTLSizeMake(s.size[0], s.size[1], s.size[2]);
    auto local = [self computeBestGroup:t];
    #define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
    auto globalSize = MTLSizeMake(UP_DIV(t.width, local.width), UP_DIV(t.height, local.height), UP_DIV(t.depth, local.depth));
    #undef UP_DIV
    return std::make_pair(globalSize, local);
}

- (MTLSize)computeBestGroup:(MTLSize)t {
    if (pipeline.maxTotalThreadsPerThreadgroup > 64) {
        auto res = MTLSizeMake(8, 8, 8);
        int reduceNumber = 0;
        if (t.depth < 4) {
            res.depth = 1;
            reduceNumber++;
        }
        if (t.width < 4) {
            res.width = 1;
            reduceNumber++;
        }
        if (t.height < 4) {
            res.height = 1;
            reduceNumber++;
        }
        if (reduceNumber == 0) {
            return MTLSizeMake(4, 4, 4);
        }
        if (reduceNumber == 2) {
            if (res.width > 1) {
                res.width = 64;
            }
            if (res.height > 1) {
                res.height = 64;
            }
            if (res.depth > 1) {
                res.depth = 64;
            }
        }
        return res;
    }
    auto smallest_log2 = [](NSUInteger integer) -> NSUInteger {
        if (integer == 0)
            return 0;
        NSUInteger power = 0;
        while ((integer & 0b1) == 0) {
            integer = integer >> 1;
            power++;
        }
        return power;
    };
    auto pwarp = smallest_log2(pipeline.threadExecutionWidth);
    auto px = smallest_log2(t.width), sx = (NSUInteger)ceil(log2(t.width));
    auto py = smallest_log2(t.height), sy = (NSUInteger)ceil(log2(t.height));

    // accurately match on x
    if (px >= pwarp) {
        return {pipeline.threadExecutionWidth, 1, 1};
    }
    // accurately match on xy
    else if (px + py >= pwarp && sx < pwarp / 2) {
        NSUInteger x = pow(2, px);
        return {x, pipeline.threadExecutionWidth / x, 1};
    }
    // similarly match on x
    else if (sx >= pwarp) {
        return {pipeline.threadExecutionWidth, 1, 1};
    }
    // similarly match on xy
    else if (sx + sy >= pwarp) {
        NSUInteger x = pow(2, sx);
        return {x, pipeline.threadExecutionWidth / x, 1};
    }

    // on xyz (for most shaders do not protect gid.z, z axis must be accurately match)
    auto pz = smallest_log2(t.depth);
    auto sz = pz;
    if (px + py + pz >= pwarp) {
        NSUInteger x = pow(2, px), y = pow(2, py);
        return {x, y, pipeline.threadExecutionWidth / x / y};
    } else if (sx + sy + sz >= pwarp) {
        NSUInteger x = pow(2, sx), z = pow(2, MIN(sz, pwarp - sx));
        return {x, pipeline.threadExecutionWidth / x / z, z};
    } else {
        NSUInteger z = pow(2, sz);
        return {t.width, t.height, z};
    }
}

- (BOOL)setWeightData:(NSArray<NSData *> *)weights
                error:(NSError * _Nullable *)error {
    assert(weights.count > 1);
    outputShape.resize(weights[0].length / sizeof(int));
    memcpy(outputShape.data(), [weights[0] bytes], weights[0].length);
    regions.resize(weights.count - 1);
    for (int i = 1; i < weights.count; i++) {
        auto regionPtr = [weights[i] bytes];
        memcpy(&regions[i-1], regionPtr, weights[i].length);
    }
    [self setRegionSampler];
    return YES;
}

- (NSArray<NSArray<NSNumber *> *> *)outputShapesForInputShapes:(NSArray<NSArray<NSNumber *> *> *)inputShapes
                                                         error:(NSError * _Nullable *)error {
    NSMutableArray* shape = [[NSMutableArray alloc] initWithCapacity: outputShape.size()];
    for (int x : outputShape) {
        [shape addObject: [NSNumber numberWithInt:x]];
    }
    NSArray* outputShapes = @[ shape ];
    return outputShapes;
}

// execute on cpu
- (BOOL)evaluateOnCPUWithInputs:(NSArray<MLMultiArray *> *)inputs
                        outputs:(NSArray<MLMultiArray *> *)outputs
                          error:(NSError * _Nullable *)error {
    // NSLog(@"%@ -> %@", inputs[0].shape, outputs[0].shape);
    assert(inputs.count == regions.size());
    float* outputPtr = static_cast<float*>(outputs[0].dataPointer);
    for (int i = 0; i < inputs.count; i++) {
        const float* inputPtr = static_cast<const float*>(inputs[i].dataPointer);
        const auto& region = regions[i];
        for (int z = 0; z < region.size[0]; z++) {
            for (int y = 0; y < region.size[1]; y++) {
                for (int x = 0; x < region.size[2]; x++) {
                    outputPtr[region.dst.offset + z * region.dst.stride[0] + y * region.dst.stride[1] + x * region.dst.stride[2]] =
                        inputPtr[region.src.offset + z * region.src.stride[0] + y * region.src.stride[1] + x * region.src.stride[2]];
                }
            }
        }
    }
    return YES;
}

@end

@implementation DumpLayer
- (instancetype)initWithParameterDictionary:(NSDictionary<NSString *,id> *)parameters
                                      error:(NSError * _Nullable *)error {
    self = [super init];
    return self;
}

- (BOOL)setWeightData:(NSArray<NSData *> *)weights
                error:(NSError * _Nullable *)error {
    return YES;
}

- (NSArray<NSArray<NSNumber *> *> *)outputShapesForInputShapes:(NSArray<NSArray<NSNumber *> *> *)inputShapes
                                                         error:(NSError * _Nullable *)error {
    for (int i = 0; i < inputShapes.count; i++) {
        printf("### shape_%d : { ", i);
        for (int j = 0; j < inputShapes[i].count; j++) {
            printf("%d, ", inputShapes[i][j].intValue);
        }
        printf(" }\n");
    }
    return inputShapes;
}

- (BOOL)evaluateOnCPUWithInputs:(NSArray<MLMultiArray *> *)inputs
                        outputs:(NSArray<MLMultiArray *> *)outputs
                          error:(NSError * _Nullable *)error {
    assert(inputs.count == 1 && outputs.count == 1);
    assert(inputs[0].count == outputs[0].count);
    const float* inputPtr = static_cast<float*>(inputs[0].dataPointer);
    float* outputPtr = static_cast<float*>(outputs[0].dataPointer);
    printf(">>> "); for (int i = 0; i < 10; i++) printf("%f, ", inputPtr[i]); printf("\n");
    // memcpy(outputPtr, inputPtr, outputs[0].count);
    memcpy(outputPtr, inputPtr, outputs[0].count * sizeof(float));
    return YES;
}
@end

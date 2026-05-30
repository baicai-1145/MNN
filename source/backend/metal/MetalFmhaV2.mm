//
//  MetalFmhaV2.mm
//  MNN
//
//  Created by pymss on 2026/05/30.
//

#import "MetalFmhaV2.hpp"
#import "MNNMetalContext.h"
#import "MetalFmhaV2Shader.hpp"
#include "core/TensorUtils.hpp"

#if MNN_METAL_ENABLED
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

namespace MNN {

MetalFmhaV2::MetalFmhaV2(Backend* backend, const MNN::Op* op) : MetalExecution(backend) {
    auto param = op->main_as_FmhaV2Param();
    mHeads = param->heads();
    mAttention.reset(new AttentionBufExecution(backend, false));
    auto mtbn = static_cast<MetalBackend*>(backend);
    auto context = (__bridge MNNMetalContext*)mtbn->context();
    mSplitParam = [context newDeviceBuffer:sizeof(SplitParam) access:CPUWriteOnly];
}

bool MetalFmhaV2::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    *dst = new MetalFmhaV2(bn, op);
    return true;
}

void MetalFmhaV2::compileSplitKernel() {
    if (mSplitKernel != nil) {
        return;
    }
    auto mtbn = static_cast<MetalBackend*>(backend());
    auto rt = (MetalRuntime*)mtbn->runtime();
    std::string ftype = mtbn->useFp16InsteadFp32() ? "half" : "float";
    std::vector<std::string> keys = {"pymss_fmha_v2_split_qkv", ftype};
    auto pipeline = rt->findPipeline(keys);
    if (pipeline == nil) {
        MTLCompileOptions* option = [[MTLCompileOptions alloc] init];
        option.preprocessorMacros = @{
            @"ftype" : @(ftype.c_str()),
        };
        pipeline = mtbn->makeComputePipelineWithSourceOption(gFmhaV2SplitQKV, "split_qkv", option);
        rt->insertPipeline(keys, pipeline);
    }
    mSplitKernel = pipeline;
}

ErrorCode MetalFmhaV2::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    if (inputs.size() != 1 || outputs.size() != 1 || mHeads <= 0) {
        return NOT_SUPPORT;
    }
    auto input = inputs[0];
    if (input->dimensions() != 3) {
        return NOT_SUPPORT;
    }
    mBatch = input->length(0);
    mSeqLen = input->length(1);
    const int qkvDim = input->length(2);
    if (qkvDim <= 0 || qkvDim % (3 * mHeads) != 0) {
        return NOT_SUPPORT;
    }
    mHeadDim = qkvDim / (3 * mHeads);
    if (mHeadDim <= 0) {
        return NOT_SUPPORT;
    }

    std::vector<int> qkvShape = {mBatch, mSeqLen, mHeads, mHeadDim};
    mQ.reset(Tensor::createDevice<float>(qkvShape));
    mK.reset(Tensor::createDevice<float>(qkvShape));
    mV.reset(Tensor::createDevice<float>(qkvShape));
    TensorUtils::getDescribe(mQ.get())->dimensionFormat = TensorUtils::getDescribe(input)->dimensionFormat;
    TensorUtils::getDescribe(mK.get())->dimensionFormat = TensorUtils::getDescribe(input)->dimensionFormat;
    TensorUtils::getDescribe(mV.get())->dimensionFormat = TensorUtils::getDescribe(input)->dimensionFormat;

    backend()->onAcquireBuffer(mQ.get(), Backend::DYNAMIC);
    backend()->onAcquireBuffer(mK.get(), Backend::DYNAMIC);
    backend()->onAcquireBuffer(mV.get(), Backend::DYNAMIC);

    std::vector<Tensor*> attentionInputs = {mQ.get(), mK.get(), mV.get()};
    auto code = mAttention->onResize(attentionInputs, outputs);

    backend()->onReleaseBuffer(mQ.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mK.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mV.get(), Backend::DYNAMIC);
    return code;
}

void MetalFmhaV2::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, id<MTLComputeCommandEncoder> encoder) {
    compileSplitKernel();
    auto mtbn = static_cast<MetalBackend*>(backend());
    auto context = (__bridge MNNMetalContext*)mtbn->context();

    auto param = reinterpret_cast<SplitParam*>(mSplitParam.contents);
    param->batch = mBatch;
    param->seq_len = mSeqLen;
    param->heads = mHeads;
    param->head_dim = mHeadDim;

    [encoder setComputePipelineState:mSplitKernel];
    MetalBackend::setTensor(inputs[0], encoder, 0);
    MetalBackend::setTensor(mQ.get(), encoder, 1);
    MetalBackend::setTensor(mK.get(), encoder, 2);
    MetalBackend::setTensor(mV.get(), encoder, 3);
    [encoder setBuffer:mSplitParam offset:0 atIndex:4];
    const int total = mBatch * mSeqLen * mHeads * mHeadDim;
    auto gl = [context computeBestGroupAndLocal:mSplitKernel threads:MTLSizeMake(total, 1, 1)];
    [encoder dispatchThreadgroups:gl.first threadsPerThreadgroup:gl.second];

    std::vector<Tensor*> attentionInputs = {mQ.get(), mK.get(), mV.get()};
    mAttention->onEncode(attentionInputs, outputs, encoder);
}

class MetalFmhaV2Creator : public MetalBackend::Creator {
public:
    virtual Execution* onCreate(const std::vector<Tensor*>& inputs, const MNN::Op* op, Backend* backend, const std::vector<Tensor*>& outputs) const override {
        return new MetalFmhaV2(backend, op);
    }
};

REGISTER_METAL_OP_TRANSFORMER_CREATOR(MetalFmhaV2Creator, OpType_FmhaV2);

} // namespace MNN
#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
#endif /* MNN_METAL_ENABLED */

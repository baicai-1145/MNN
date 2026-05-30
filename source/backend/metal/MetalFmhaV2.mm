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
    mAttention.reset(new AttentionBufExecution(backend, false, true));
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

void MetalFmhaV2::compileSplitKernel(bool applyRotary, bool packedGate) {
    if (mSplitKernel != nil) {
        return;
    }
    auto mtbn = static_cast<MetalBackend*>(backend());
    auto rt = (MetalRuntime*)mtbn->runtime();
    std::string ftype = mtbn->useFp16InsteadFp32() ? "half" : "float";
    std::vector<std::string> keys = {"pymss_fmha_v2_split_qkv", ftype};
    const char* kernelName = "split_qkv";
    if (packedGate) {
        keys.emplace_back("qkvg");
        kernelName = "split_qkvg";
    }
    if (applyRotary) {
        keys.emplace_back("rotary");
        kernelName = packedGate ? "split_qkvg_rotary" : "split_qkv_rotary";
    }
    auto pipeline = rt->findPipeline(keys);
    if (pipeline == nil) {
        MTLCompileOptions* option = [[MTLCompileOptions alloc] init];
        option.preprocessorMacros = @{
            @"ftype" : @(ftype.c_str()),
        };
        pipeline = mtbn->makeComputePipelineWithSourceOption(gFmhaV2SplitQKV, kernelName, option);
        rt->insertPipeline(keys, pipeline);
    }
    mSplitKernel = pipeline;
}

ErrorCode MetalFmhaV2::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    if (inputs.empty() || inputs.size() > 4 || outputs.size() != 1 || mHeads <= 0) {
        return NOT_SUPPORT;
    }
    auto input = inputs[0];
    mApplyGate = false;
    mPackedGate = false;
    mGateInputIndex = -1;

    if ((inputs.size() == 3 || inputs.size() == 4) && input->dimensions() == 4) {
        mPackedInput = false;
        mApplyRotary = false;
        mBatch = input->length(0);
        mSeqLen = input->length(1);
        if (input->length(2) != mHeads) {
            return NOT_SUPPORT;
        }
        mHeadDim = input->length(3);
        if (inputs.size() == 4) {
            mApplyGate = true;
            mGateInputIndex = 3;
        }
        if (mApplyGate && (inputs[mGateInputIndex]->dimensions() != 3 ||
                           inputs[mGateInputIndex]->length(0) != mBatch ||
                           inputs[mGateInputIndex]->length(1) != mSeqLen ||
                           inputs[mGateInputIndex]->length(2) != mHeads)) {
            return NOT_SUPPORT;
        }
        std::vector<Tensor*> attentionInputs = {inputs[0], inputs[1], inputs[2]};
        if (mApplyGate) {
            attentionInputs.emplace_back(inputs[mGateInputIndex]);
        }
        return mAttention->onResize(attentionInputs, outputs);
    }

    mPackedInput = true;
    mApplyRotary = inputs.size() == 3 || inputs.size() == 4;
    const bool externalGate = inputs.size() == 2 || inputs.size() == 4;
    mApplyGate = externalGate;
    mGateInputIndex = inputs.size() == 2 ? 1 : (inputs.size() == 4 ? 3 : -1);
    if (input->dimensions() != 3) {
        return NOT_SUPPORT;
    }
    mBatch = input->length(0);
    mSeqLen = input->length(1);
    const int packedDim = input->length(2);
    int qkvDim = packedDim;
    if (!externalGate && packedDim > mHeads && (packedDim - mHeads) % (3 * mHeads) == 0 && packedDim % (3 * mHeads) != 0) {
        mPackedGate = true;
        mApplyGate = true;
        qkvDim = packedDim - mHeads;
    }
    if (qkvDim <= 0 || qkvDim % (3 * mHeads) != 0) {
        return NOT_SUPPORT;
    }
    mHeadDim = qkvDim / (3 * mHeads);
    if (mHeadDim <= 0) {
        return NOT_SUPPORT;
    }
    if (mApplyRotary) {
        if (inputs[1]->dimensions() != 4 || inputs[2]->dimensions() != 4) {
            return NOT_SUPPORT;
        }
        if (inputs[1]->length(1) != mSeqLen || inputs[2]->length(1) != mSeqLen ||
            inputs[1]->length(3) * 2 != mHeadDim || inputs[2]->length(3) * 2 != mHeadDim) {
            return NOT_SUPPORT;
        }
    }
    if (externalGate && (inputs[mGateInputIndex]->dimensions() != 3 ||
                         inputs[mGateInputIndex]->length(0) != mBatch ||
                         inputs[mGateInputIndex]->length(1) != mSeqLen ||
                         inputs[mGateInputIndex]->length(2) != mHeads)) {
        return NOT_SUPPORT;
    }

    std::vector<int> qkvShape = {mBatch, mSeqLen, mHeads, mHeadDim};
    mQ.reset(Tensor::createDevice<float>(qkvShape));
    mK.reset(Tensor::createDevice<float>(qkvShape));
    mV.reset(Tensor::createDevice<float>(qkvShape));
    TensorUtils::getDescribe(mQ.get())->dimensionFormat = TensorUtils::getDescribe(input)->dimensionFormat;
    TensorUtils::getDescribe(mK.get())->dimensionFormat = TensorUtils::getDescribe(input)->dimensionFormat;
    TensorUtils::getDescribe(mV.get())->dimensionFormat = TensorUtils::getDescribe(input)->dimensionFormat;
    if (mPackedGate) {
        mGate.reset(Tensor::createDevice<float>({mBatch, mSeqLen, mHeads}));
        TensorUtils::getDescribe(mGate.get())->dimensionFormat = TensorUtils::getDescribe(input)->dimensionFormat;
    } else {
        mGate.reset();
    }

    backend()->onAcquireBuffer(mQ.get(), Backend::DYNAMIC);
    backend()->onAcquireBuffer(mK.get(), Backend::DYNAMIC);
    backend()->onAcquireBuffer(mV.get(), Backend::DYNAMIC);
    if (mPackedGate) {
        backend()->onAcquireBuffer(mGate.get(), Backend::DYNAMIC);
    }

    std::vector<Tensor*> attentionInputs = {mQ.get(), mK.get(), mV.get()};
    if (mApplyGate) {
        attentionInputs.emplace_back(mPackedGate ? mGate.get() : inputs[mGateInputIndex]);
    }
    auto code = mAttention->onResize(attentionInputs, outputs);

    backend()->onReleaseBuffer(mQ.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mK.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mV.get(), Backend::DYNAMIC);
    if (mPackedGate) {
        backend()->onReleaseBuffer(mGate.get(), Backend::DYNAMIC);
    }
    return code;
}

void MetalFmhaV2::onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, id<MTLComputeCommandEncoder> encoder) {
    if (!mPackedInput) {
        std::vector<Tensor*> attentionInputs = {inputs[0], inputs[1], inputs[2]};
        if (mApplyGate) {
            attentionInputs.emplace_back(inputs[mGateInputIndex]);
        }
        mAttention->onEncode(attentionInputs, outputs, encoder);
        return;
    }
    compileSplitKernel(mApplyRotary, mPackedGate);
    auto mtbn = static_cast<MetalBackend*>(backend());
    auto context = (__bridge MNNMetalContext*)mtbn->context();

    auto param = reinterpret_cast<SplitParam*>(mSplitParam.contents);
    param->batch = mBatch;
    param->seq_len = mSeqLen;
    param->heads = mHeads;
    param->head_dim = mHeadDim;
    param->cos_batch = mApplyRotary ? inputs[1]->length(0) : 0;
    param->packed_dim = inputs[0]->length(2);

    [encoder setComputePipelineState:mSplitKernel];
    MetalBackend::setTensor(inputs[0], encoder, 0);
    MetalBackend::setTensor(mQ.get(), encoder, 1);
    MetalBackend::setTensor(mK.get(), encoder, 2);
    MetalBackend::setTensor(mV.get(), encoder, 3);
    if (mPackedGate && mApplyRotary) {
        MetalBackend::setTensor(inputs[1], encoder, 4);
        MetalBackend::setTensor(inputs[2], encoder, 5);
        MetalBackend::setTensor(mGate.get(), encoder, 6);
        [encoder setBuffer:mSplitParam offset:0 atIndex:7];
    } else if (mPackedGate) {
        MetalBackend::setTensor(mGate.get(), encoder, 4);
        [encoder setBuffer:mSplitParam offset:0 atIndex:5];
    } else if (mApplyRotary) {
        MetalBackend::setTensor(inputs[1], encoder, 4);
        MetalBackend::setTensor(inputs[2], encoder, 5);
        [encoder setBuffer:mSplitParam offset:0 atIndex:6];
    } else {
        [encoder setBuffer:mSplitParam offset:0 atIndex:4];
    }
    const int total = mBatch * mSeqLen * mHeads * mHeadDim;
    auto gl = [context computeBestGroupAndLocal:mSplitKernel threads:MTLSizeMake(total, 1, 1)];
    [encoder dispatchThreadgroups:gl.first threadsPerThreadgroup:gl.second];

    std::vector<Tensor*> attentionInputs = {mQ.get(), mK.get(), mV.get()};
    if (mApplyGate) {
        attentionInputs.emplace_back(mPackedGate ? mGate.get() : inputs[mGateInputIndex]);
    }
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

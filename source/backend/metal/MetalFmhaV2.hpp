//
//  MetalFmhaV2.hpp
//  MNN
//
//  Created by pymss on 2026/05/30.
//

#ifndef MetalFmhaV2_hpp
#define MetalFmhaV2_hpp

#import "MetalAttention.hpp"
#import "MetalBackend.hpp"
#include "MNN_generated.h"

#if MNN_METAL_ENABLED
#ifdef MNN_SUPPORT_TRANSFORMER_FUSE

namespace MNN {

class MetalFmhaV2 : public MetalExecution {
public:
    MetalFmhaV2(Backend* backend, const MNN::Op* op);
    virtual ~MetalFmhaV2() = default;
    virtual ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    virtual void onEncode(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, id<MTLComputeCommandEncoder> encoder) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;

private:
    struct SplitParam {
        int batch;
        int seq_len;
        int heads;
        int head_dim;
        int cos_batch;
        int packed_dim;
    };

    void compileSplitKernel(bool applyRotary, bool packedGate);

    int mHeads = 0;
    int mHeadDim = 0;
    int mBatch = 0;
    int mSeqLen = 0;
    bool mPackedInput = true;
    bool mApplyRotary = false;
    bool mApplyGate = false;
    bool mPackedGate = false;
    int mGateInputIndex = -1;
    std::shared_ptr<Tensor> mQ;
    std::shared_ptr<Tensor> mK;
    std::shared_ptr<Tensor> mV;
    std::shared_ptr<Tensor> mGate;
    std::shared_ptr<AttentionBufExecution> mAttention;
    id<MTLComputePipelineState> mSplitKernel = nil;
    id<MTLBuffer> mSplitParam = nil;
};

} // namespace MNN
#endif /* MNN_SUPPORT_TRANSFORMER_FUSE */
#endif /* MNN_METAL_ENABLED */
#endif /* MetalFmhaV2_hpp */

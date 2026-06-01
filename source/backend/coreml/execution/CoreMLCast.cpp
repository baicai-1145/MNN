//
//  CoreMLCast.cpp
//  MNN
//

#include "CoreMLCast.hpp"

namespace MNN {

CoreMLCast::CoreMLCast(Backend* backend, const Op* op, const std::vector<Tensor*>& inputs,
                       const std::vector<Tensor*>& outputs)
    : CoreMLCommonExecution(backend, op) {
    initLayer();
}

ErrorCode CoreMLCast::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    MNN_ASSERT(inputs.size() == 1 && outputs.size() == 1);
    if (inputs[0]->getType() != outputs[0]->getType()) {
        MNN_ERROR("[CoreML] Cast only supports same-type identity casts, src=(code:%d,bits:%d), dst=(code:%d,bits:%d)\n",
                  inputs[0]->getType().code, inputs[0]->getType().bits,
                  outputs[0]->getType().code, outputs[0]->getType().bits);
        return NOT_SUPPORT;
    }
    mLayer_->layer_case = CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_COPY;
    mLayer_->copy = mCoreMLBackend->create<CoreML__Specification__CopyLayerParams>();
    core_ml__specification__copy_layer_params__init(mLayer_->copy);
    setLayerInputsAndOutputs(mLayer_,
                             {mCoreMLBackend->getTensorName(inputs[0])},
                             {mCoreMLBackend->getTensorName(outputs[0])});
    mCoreMLBackend->addLayer(mLayer_);
    return NO_ERROR;
}

REGISTER_COREML_OP_CREATOR(CoreMLCast, OpType_Cast)

} // namespace MNN

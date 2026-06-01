//
//  CoreMLCast.hpp
//  MNN
//

#ifndef MNN_COREMLCAST_HPP
#define MNN_COREMLCAST_HPP

#include "CoreMLBackend.hpp"
#include "CoreMLCommonExecution.hpp"

namespace MNN {

class CoreMLCast : public CoreMLCommonExecution {
public:
    CoreMLCast(Backend* backend, const Op* op, const std::vector<Tensor*>& inputs,
               const std::vector<Tensor*>& outputs);
    ErrorCode onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) override;
    ~CoreMLCast() override = default;
};

} // namespace MNN

#endif // MNN_COREMLCAST_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "as_py_klif_backward_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
namespace {
constexpr uint32_t kTileLength = 2048;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  const gert::StorageShape* h_shape = context->GetInputShape(2);
  const gert::Shape& shape = h_shape->GetStorageShape();
  if (shape.GetDimNum() < 2) {
    return ge::GRAPH_FAILED;
  }
  uint64_t time_steps = static_cast<uint64_t>(shape.GetDim(0));
  uint64_t neuron_count = 1;
  for (size_t index = 1; index < shape.GetDimNum(); ++index) {
    neuron_count *= static_cast<uint64_t>(shape.GetDim(index));
  }
  if (time_steps == 0 || neuron_count == 0 || time_steps > UINT32_MAX ||
      neuron_count > UINT32_MAX || (neuron_count & 7U) != 0U) {
    return ge::GRAPH_FAILED;
  }
  const gert::Shape& k_shape = context->GetInputShape(7)->GetStorageShape();
  if (k_shape.GetDimNum() != 1 || k_shape.GetShapeSize() != 8) {
    return ge::GRAPH_FAILED;
  }
  const auto* attrs = context->GetAttrs();
  if (attrs == nullptr || attrs->GetFloat(0) == nullptr ||
      attrs->GetFloat(1) == nullptr || attrs->GetBool(2) == nullptr ||
      attrs->GetBool(3) == nullptr || attrs->GetFloat(4) == nullptr ||
      attrs->GetFloat(5) == nullptr || attrs->GetBool(6) == nullptr ||
      attrs->GetBool(7) == nullptr) {
    return ge::GRAPH_FAILED;
  }
  const float threshold = *attrs->GetFloat(0);
  const float reset = *attrs->GetFloat(1);
  const float surrogate_alpha = *attrs->GetFloat(4);
  const float tau = *attrs->GetFloat(5);
  if (!std::isfinite(threshold) || !std::isfinite(reset) ||
      !std::isfinite(surrogate_alpha) || surrogate_alpha <= 0.0f ||
      !std::isfinite(tau) || tau <= 1.0f) {
    return ge::GRAPH_FAILED;
  }
  const uint32_t tile_count = static_cast<uint32_t>(
      (neuron_count + kTileLength - 1) / kTileLength);
  AsPyKlifBackwardTilingData tiling;
  tiling.set_timeSteps(static_cast<uint32_t>(time_steps));
  tiling.set_neuronCount(static_cast<uint32_t>(neuron_count));
  tiling.set_tileLength(kTileLength);
  tiling.set_tileCount(tile_count);
  tiling.set_vThreshold(threshold);
  tiling.set_vReset(reset);
  tiling.set_surrogateAlpha(surrogate_alpha);
  tiling.set_reciprocalTau(1.0f / tau);
  tiling.set_hardReset(*attrs->GetBool(2) ? 1U : 0U);
  tiling.set_detachReset(*attrs->GetBool(3) ? 1U : 0U);
  tiling.set_decayInput(*attrs->GetBool(6) ? 1U : 0U);
  tiling.set_scaleReset(*attrs->GetBool(7) ? 1U : 0U);
  constexpr uint32_t kAscend910BAivCores = 20U;
  context->SetBlockDim(std::min(tile_count, kAscend910BAivCores));
  tiling.SaveToBuffer(
      context->GetRawTilingData()->GetData(),
      context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
  *context->GetOutputShape(0) = *context->GetInputShape(0);
  *context->GetOutputShape(1) = *context->GetInputShape(6);
  *context->GetOutputShape(2) = *context->GetInputShape(6);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
  const auto dtype = context->GetInputDataType(0);
  context->SetOutputDataType(0, dtype);
  context->SetOutputDataType(1, dtype);
  context->SetOutputDataType(2, dtype);
  return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class AsPyKlifBackward : public OpDef {
 public:
  explicit AsPyKlifBackward(const char* name) : OpDef(name) {
    for (const char* input : {"xSeq", "vPrevSeq", "hSeq", "spikeSeq",
                              "gradSpikeSeq", "gradVSeq", "gradVFinal", "k"}) {
      this->Input(input).ParamType(REQUIRED).DataType({ge::DT_FLOAT})
          .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    }
    this->Output("gradXSeq").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("gradVInit").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("gradKPartial").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Attr("vThreshold").Float();
    this->Attr("vReset").Float();
    this->Attr("hardReset").Bool();
    this->Attr("detachReset").Bool();
    this->Attr("surrogateAlpha").Float();
    this->Attr("tau").Float();
    this->Attr("decayInput").Bool();
    this->Attr("scaleReset").Bool();
    this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend910b");
  }
};

OP_ADD(AsPyKlifBackward);
}  // namespace ops

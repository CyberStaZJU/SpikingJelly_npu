#include <algorithm>
#include <cmath>
#include <cstdint>

#include "as_py_fed_snn_decay_lif_backward_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
namespace {
constexpr uint32_t kTileLength = 4096;
}  // namespace

static bool SameShape(const gert::Shape& left, const gert::Shape& right) {
  if (left.GetDimNum() != right.GetDimNum()) {
    return false;
  }
  for (size_t index = 0; index < left.GetDimNum(); ++index) {
    if (left.GetDim(index) != right.GetDim(index)) {
      return false;
    }
  }
  return true;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  const gert::StorageShape* h_shape = context->GetInputShape(0);
  const gert::StorageShape* grad_spike_shape = context->GetInputShape(1);
  const gert::Shape& shape = h_shape->GetStorageShape();
  const gert::Shape& grad_shape = grad_spike_shape->GetStorageShape();
  if (shape.GetDimNum() < 2 || !SameShape(shape, grad_shape)) {
    return ge::GRAPH_FAILED;
  }

  uint64_t time_steps = static_cast<uint64_t>(shape.GetDim(0));
  uint64_t neuron_count = 1;
  for (size_t index = 1; index < shape.GetDimNum(); ++index) {
    neuron_count *= static_cast<uint64_t>(shape.GetDim(index));
  }
  if (time_steps == 0 || neuron_count == 0 ||
      time_steps > UINT32_MAX || neuron_count > UINT32_MAX ||
      (neuron_count & 7U) != 0U) {
    return ge::GRAPH_FAILED;
  }

  const float membrane_decay = *context->GetAttrs()->GetFloat(0);
  if (!(membrane_decay >= 0.0f && membrane_decay <= 1.0f)) {
    return ge::GRAPH_FAILED;
  }
  const uint32_t tile_count =
      static_cast<uint32_t>((neuron_count + kTileLength - 1) / kTileLength);
  AsPyFedSNNDecayLifBackwardTilingData tiling;
  tiling.set_timeSteps(static_cast<uint32_t>(time_steps));
  tiling.set_neuronCount(static_cast<uint32_t>(neuron_count));
  tiling.set_tileLength(kTileLength);
  const float v_threshold = *context->GetAttrs()->GetFloat(1);
  const float surrogate_alpha = *context->GetAttrs()->GetFloat(2);
  if (!std::isfinite(v_threshold) || !std::isfinite(surrogate_alpha) ||
      surrogate_alpha <= 0.0f) {
    return ge::GRAPH_FAILED;
  }
  tiling.set_tileCount(tile_count);
  tiling.set_membraneDecay(membrane_decay);
  tiling.set_vThreshold(v_threshold);
  tiling.set_surrogateAlpha(surrogate_alpha);

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
  const gert::Shape* h_shape = context->GetInputShape(0);
  const gert::Shape* grad_spike_shape = context->GetInputShape(1);
  if (!optiling::SameShape(*h_shape, *grad_spike_shape)) {
    return ge::GRAPH_FAILED;
  }
  *context->GetOutputShape(0) = *h_shape;
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
  context->SetOutputDataType(0, context->GetInputDataType(0));
  return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class AsPyFedSNNDecayLifBackward : public OpDef {
 public:
  explicit AsPyFedSNNDecayLifBackward(const char* name) : OpDef(name) {
    this->Input("hSeq").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("gradSpikeSeq").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("gradCurrentSeq").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
        .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Attr("membraneDecay").Float();
    this->Attr("vThreshold").Float();
    this->Attr("surrogateAlpha").Float();
    this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
    this->AICore().SetTiling(optiling::TilingFunc);
    this->AICore().AddConfig("ascend910b");
  }
};

OP_ADD(AsPyFedSNNDecayLifBackward);
}  // namespace ops

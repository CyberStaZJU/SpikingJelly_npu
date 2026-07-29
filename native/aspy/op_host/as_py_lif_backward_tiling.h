#pragma once

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AsPyLifBackwardTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, timeSteps);
  TILING_DATA_FIELD_DEF(uint32_t, neuronCount);
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileCount);
  TILING_DATA_FIELD_DEF(float, vThreshold);
  TILING_DATA_FIELD_DEF(float, vReset);
  TILING_DATA_FIELD_DEF(float, surrogateAlpha);
  TILING_DATA_FIELD_DEF(float, reciprocalTau);
  TILING_DATA_FIELD_DEF(uint32_t, hardReset);
  TILING_DATA_FIELD_DEF(uint32_t, detachReset);
  TILING_DATA_FIELD_DEF(uint32_t, decayInput);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AsPyLifBackward, AsPyLifBackwardTilingData)
}  // namespace optiling

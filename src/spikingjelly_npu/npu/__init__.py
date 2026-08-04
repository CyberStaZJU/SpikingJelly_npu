"""Ascend runtime helpers with lazy ``torch_npu`` integration."""

from .amp import (
    BF16_MIXED_PRECISION_PROFILE,
    NPUAutocastState,
    autocast,
    get_npu_autocast_state,
    is_npu_bf16_autocast_active,
    npu_bf16_autocast,
)
from .graph import (
    GraphBucketRunner,
    GraphBucketSpec,
    GraphPreExecutionError,
    GraphRoute,
    StaticGraphRunner,
)
from .runtime import (
    BF16CapabilityStatus,
    NPUInfo,
    configure_npu,
    get_npu_info,
    is_npu_available,
)

__all__ = [
    "BF16CapabilityStatus",
    "BF16_MIXED_PRECISION_PROFILE",
    "NPUAutocastState",
    "NPUInfo",
    "GraphBucketRunner",
    "GraphBucketSpec",
    "GraphPreExecutionError",
    "GraphRoute",
    "StaticGraphRunner",
    "autocast",
    "configure_npu",
    "get_npu_autocast_state",
    "get_npu_info",
    "is_npu_available",
    "is_npu_bf16_autocast_active",
    "npu_bf16_autocast",
]

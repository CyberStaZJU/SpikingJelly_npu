"""Ascend runtime helpers with lazy ``torch_npu`` integration."""

from .amp import autocast
from .graph import (
    GraphBucketRunner,
    GraphBucketSpec,
    GraphPreExecutionError,
    GraphRoute,
    StaticGraphRunner,
)
from .runtime import NPUInfo, configure_npu, get_npu_info, is_npu_available

__all__ = [
    "NPUInfo",
    "GraphBucketRunner",
    "GraphBucketSpec",
    "GraphPreExecutionError",
    "GraphRoute",
    "StaticGraphRunner",
    "autocast",
    "configure_npu",
    "get_npu_info",
    "is_npu_available",
]

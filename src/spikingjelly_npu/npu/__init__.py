"""Ascend runtime helpers with lazy ``torch_npu`` integration."""

from .amp import autocast
from .graph import GraphBucketRunner, GraphBucketSpec, GraphRoute, StaticGraphRunner
from .runtime import NPUInfo, configure_npu, get_npu_info, is_npu_available

__all__ = [
    "NPUInfo",
    "GraphBucketRunner",
    "GraphBucketSpec",
    "GraphRoute",
    "StaticGraphRunner",
    "autocast",
    "configure_npu",
    "get_npu_info",
    "is_npu_available",
]

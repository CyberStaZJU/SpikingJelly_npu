"""PyTorch-native spiking neural-network building blocks for Ascend NPU."""

from . import activation_based, fedsnn, npu
from ._version import __version__
from .compat import (
    CompatibilityStatus,
    _enable_from_environment,
    enable_compat,
    get_compatibility_status,
    install_spikingjelly_alias,
)

_enable_from_environment()

__all__ = [
    "__version__",
    "CompatibilityStatus",
    "activation_based",
    "enable_compat",
    "fedsnn",
    "get_compatibility_status",
    "install_spikingjelly_alias",
    "npu",
]

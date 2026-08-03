"""Opt-in compatibility helpers for existing SpikingJelly/CUDA applications."""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from dataclasses import dataclass
from types import ModuleType

from . import activation_based

_COMPAT_ENV = "SPIKINGJELLY_NPU_COMPAT"


@dataclass(frozen=True)
class CompatibilityStatus:
    enabled: bool
    spikingjelly_alias: bool
    cuda_transfer: bool
    reason: str


_status = CompatibilityStatus(False, False, False, "compatibility mode is disabled")


def install_spikingjelly_alias(*, force: bool = False) -> ModuleType:
    """Expose the qualified subset under the ``spikingjelly`` import path.

    The alias is process-local and must be installed before the consumer imports
    SpikingJelly. ``force=True`` may shadow an installed but not-yet-imported real
    package; an already imported package is never partially replaced.
    """
    existing = sys.modules.get("spikingjelly")
    if existing is not None:
        if getattr(existing, "__spikingjelly_npu_alias__", False):
            return existing
        raise RuntimeError("spikingjelly is already imported; refusing a partial replacement")
    if importlib.util.find_spec("spikingjelly") is not None and not force:
        raise RuntimeError(
            "a real spikingjelly installation is available; pass force=True only if "
            "shadowing the documented compatibility subset is intentional"
        )
    package = ModuleType("spikingjelly")
    package.__path__ = []
    package.__package__ = "spikingjelly"
    package.__spikingjelly_npu_alias__ = True
    package.__version__ = "spikingjelly_npu-compat"
    package.activation_based = activation_based
    sys.modules["spikingjelly"] = package
    sys.modules["spikingjelly.activation_based"] = activation_based
    for name in ("base", "functional", "layer", "model", "neuron", "surrogate"):
        sys.modules[f"spikingjelly.activation_based.{name}"] = getattr(activation_based, name)
    sys.modules["spikingjelly.activation_based.model.spikformer"] = (
        activation_based.model.spikformer
    )
    return package


def enable_compat(
    *,
    spikingjelly: bool = True,
    cuda: bool = False,
    force_alias: bool = True,
) -> CompatibilityStatus:
    """Enable the qualified process-local migration helpers.

    ``cuda=True`` imports torch-npu's official ``transfer_to_npu`` compatibility
    module. That module performs process-global CUDA-to-NPU monkeypatching and is
    suitable only for a dedicated Ascend process. It is intentionally opt-in.
    """
    global _status
    existing_alias = sys.modules.get("spikingjelly")
    alias_enabled = _status.spikingjelly_alias and bool(
        getattr(existing_alias, "__spikingjelly_npu_alias__", False)
    )
    cuda_enabled = _status.cuda_transfer
    reasons: list[str] = []

    if spikingjelly and not alias_enabled:
        install_spikingjelly_alias(force=force_alias)
        alias_enabled = True
        reasons.append("qualified SpikingJelly activation_based alias enabled")
    if cuda and not cuda_enabled:
        importlib.import_module("torch_npu.contrib.transfer_to_npu")
        cuda_enabled = True
        reasons.append("torch-npu transfer_to_npu enabled")

    _status = CompatibilityStatus(
        enabled=alias_enabled or cuda_enabled,
        spikingjelly_alias=alias_enabled,
        cuda_transfer=cuda_enabled,
        reason="; ".join(reasons) if reasons else "compatibility mode was already enabled",
    )
    return _status


def get_compatibility_status() -> CompatibilityStatus:
    return _status


def _enable_from_environment() -> None:
    mode = os.environ.get(_COMPAT_ENV, "").strip().lower()
    if not mode or mode in {"0", "false", "off", "none"}:
        return
    if mode in {"1", "true", "spikingjelly", "alias"}:
        enable_compat(spikingjelly=True, cuda=False)
        return
    if mode in {"ascend", "cuda", "full"}:
        enable_compat(spikingjelly=True, cuda=True)
        return
    raise RuntimeError(
        f"unsupported {_COMPAT_ENV}={mode!r}; use off, spikingjelly, or ascend"
    )


__all__ = [
    "CompatibilityStatus",
    "enable_compat",
    "get_compatibility_status",
    "install_spikingjelly_alias",
]

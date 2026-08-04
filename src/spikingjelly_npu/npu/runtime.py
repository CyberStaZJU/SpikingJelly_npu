"""Lazy discovery and configuration of the Ascend PyTorch runtime."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch


class BF16CapabilityStatus(str, Enum):
    """Result of the runtime's NPU BF16 capability probe."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    QUERY_UNAVAILABLE = "query_unavailable"
    QUERY_FAILED = "query_failed"
    RUNTIME_UNAVAILABLE = "runtime_unavailable"


@dataclass(frozen=True)
class NPUInfo:
    available: bool
    device_count: int
    torch_version: str
    torch_npu_version: str | None
    bf16_supported: bool | None
    bf16_status: BF16CapabilityStatus
    bf16_reason: str | None
    graph_api_available: bool
    reason: str | None = None


def _import_torch_npu() -> Any:
    return importlib.import_module("torch_npu")


def is_npu_available() -> bool:
    try:
        _import_torch_npu()
        return bool(hasattr(torch, "npu") and torch.npu.is_available())
    except (ImportError, RuntimeError, OSError):
        return False


def _query_bf16_support() -> tuple[
    bool | None,
    BF16CapabilityStatus,
    str | None,
]:
    query = getattr(getattr(torch, "npu", None), "is_bf16_supported", None)
    if not callable(query):
        return (
            None,
            BF16CapabilityStatus.QUERY_UNAVAILABLE,
            "torch.npu.is_bf16_supported is unavailable",
        )
    try:
        result = query()
        if not isinstance(result, bool):
            raise TypeError(
                "torch.npu.is_bf16_supported returned "
                f"{type(result).__name__}, expected bool"
            )
    except (RuntimeError, TypeError, OSError) as error:
        return (
            None,
            BF16CapabilityStatus.QUERY_FAILED,
            f"{type(error).__name__}: {error}",
        )
    return (
        result,
        BF16CapabilityStatus.SUPPORTED
        if result
        else BF16CapabilityStatus.UNSUPPORTED,
        None,
    )


def get_npu_info() -> NPUInfo:
    try:
        torch_npu = _import_torch_npu()
    except (ImportError, RuntimeError, OSError) as error:
        reason = f"torch_npu import failed: {error}"
        return NPUInfo(
            available=False,
            device_count=0,
            torch_version=torch.__version__,
            torch_npu_version=None,
            bf16_supported=None,
            bf16_status=BF16CapabilityStatus.RUNTIME_UNAVAILABLE,
            bf16_reason=reason,
            graph_api_available=False,
            reason=reason,
        )

    available = bool(hasattr(torch, "npu") and torch.npu.is_available())
    count = int(torch.npu.device_count()) if available else 0
    if available:
        bf16_supported, bf16_status, bf16_reason = _query_bf16_support()
    else:
        bf16_supported = None
        bf16_status = BF16CapabilityStatus.RUNTIME_UNAVAILABLE
        bf16_reason = "torch_npu imported but torch.npu is unavailable"
    graph_api_available = bool(
        hasattr(torch, "npu") and hasattr(torch.npu, "make_graphed_callables")
    )
    return NPUInfo(
        available=available,
        device_count=count,
        torch_version=torch.__version__,
        torch_npu_version=getattr(torch_npu, "__version__", None),
        bf16_supported=bf16_supported,
        bf16_status=bf16_status,
        bf16_reason=bf16_reason,
        graph_api_available=graph_api_available,
        reason=None if available else "torch_npu imported but torch.npu is unavailable",
    )


def _require_bf16_capability() -> None:
    query = getattr(torch.npu, "is_bf16_supported", None)
    if not callable(query):
        raise RuntimeError(
            "NPU BF16 capability query is unavailable; cannot satisfy "
            "require_bf16=True"
        )
    try:
        result = query()
        if not isinstance(result, bool):
            raise TypeError(
                "torch.npu.is_bf16_supported returned "
                f"{type(result).__name__}, expected bool"
            )
    except (RuntimeError, TypeError, OSError) as error:
        raise RuntimeError(
            "NPU BF16 capability query failed; cannot satisfy require_bf16=True"
        ) from error
    if not result:
        raise RuntimeError("the Ascend runtime reports that NPU BF16 is unsupported")


def configure_npu(
    device: str | None = None,
    *,
    allow_internal_format: bool | None = False,
    jit_compile: bool | None = False,
    require_bf16: bool = False,
) -> torch.device:
    """Import torch-npu, select a device, and configure explicit runtime policy.

    ``jit_compile=False`` selects binary ACLNN/opapi operators in torch-npu 2.9,
    while ``allow_internal_format=False`` prevents legacy internal-format ACLop
    kernels that cannot enter NPUGraph capture. The latter may be write-only.

    ``require_bf16=True`` applies a fail-closed runtime capability prerequisite
    before any process-global NPU configuration is mutated. A supported result
    does not qualify arbitrary models, shapes, training trajectories, NPUGraph,
    or native AsPy BF16 arithmetic.
    """
    _import_torch_npu()
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch_npu is installed but no Ascend NPU is available")
    resolved = torch.device(device or "npu:0")
    if resolved.type != "npu":
        raise ValueError(f"expected an NPU device, got {resolved}")
    if require_bf16:
        _require_bf16_capability()
    if jit_compile is not None and hasattr(torch.npu, "set_compile_mode"):
        torch.npu.set_compile_mode(jit_compile=bool(jit_compile))
    if allow_internal_format is not None:
        config = getattr(torch.npu, "config", None)
        if config is not None:
            # torch-npu 2.9 exposes this as a write-only class-level setter, so
            # hasattr(config, "allow_internal_format") is always false.
            config.allow_internal_format = bool(allow_internal_format)
    torch.npu.set_device(resolved)
    return resolved

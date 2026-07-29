"""Lazy discovery and configuration of the Ascend PyTorch runtime."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class NPUInfo:
    available: bool
    device_count: int
    torch_version: str
    torch_npu_version: str | None
    bf16_supported: bool | None
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


def get_npu_info() -> NPUInfo:
    try:
        torch_npu = _import_torch_npu()
    except (ImportError, RuntimeError, OSError) as error:
        return NPUInfo(
            available=False,
            device_count=0,
            torch_version=torch.__version__,
            torch_npu_version=None,
            bf16_supported=None,
            graph_api_available=False,
            reason=f"torch_npu import failed: {error}",
        )

    available = bool(hasattr(torch, "npu") and torch.npu.is_available())
    count = int(torch.npu.device_count()) if available else 0
    bf16_supported = None
    if available and hasattr(torch.npu, "is_bf16_supported"):
        bf16_supported = bool(torch.npu.is_bf16_supported())
    graph_api_available = bool(
        hasattr(torch, "npu") and hasattr(torch.npu, "make_graphed_callables")
    )
    return NPUInfo(
        available=available,
        device_count=count,
        torch_version=torch.__version__,
        torch_npu_version=getattr(torch_npu, "__version__", None),
        bf16_supported=bf16_supported,
        graph_api_available=graph_api_available,
        reason=None if available else "torch_npu imported but torch.npu is unavailable",
    )


def configure_npu(
    device: str | None = None,
    *,
    allow_internal_format: bool | None = False,
    jit_compile: bool | None = False,
) -> torch.device:
    """Import torch-npu, select a device, and configure graph-friendly operators.

    ``jit_compile=False`` selects binary ACLNN/opapi operators in torch-npu 2.9,
    while ``allow_internal_format=False`` prevents legacy internal-format ACLop
    kernels that cannot enter NPUGraph capture. The latter may be write-only.
    """
    _import_torch_npu()
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch_npu is installed but no Ascend NPU is available")
    resolved = torch.device(device or "npu:0")
    if resolved.type != "npu":
        raise ValueError(f"expected an NPU device, got {resolved}")
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

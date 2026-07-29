"""Automatic mixed precision helpers for NPU and CPU fallback."""

from __future__ import annotations

from contextlib import nullcontext

import torch


def autocast(
    enabled: bool = True,
    *,
    dtype: torch.dtype | None = None,
    cache_enabled: bool = False,
):
    """Return an NPU autocast context suitable for NPUGraph capture.

    NPUGraph's graphed-callable API requires autocast caching to be disabled.
    On systems where PyTorch does not recognize ``device_type='npu'``, this
    helper returns a no-op context when disabled and propagates a clear error
    when enabled.
    """
    if not enabled:
        return nullcontext()
    kwargs = {"device_type": "npu", "enabled": True, "cache_enabled": cache_enabled}
    if dtype is not None:
        kwargs["dtype"] = dtype
    return torch.autocast(**kwargs)

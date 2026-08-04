"""Automatic mixed precision helpers for the Ascend NPU runtime."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator

import torch

BF16_MIXED_PRECISION_PROFILE = "ascend910b4_bf16_mixed_fp32_state_v1"


@dataclass(frozen=True)
class NPUAutocastState:
    """Package-managed autocast state visible to precision-sensitive modules.

    This state deliberately describes only contexts entered through this
    package. It does not infer or broaden support for arbitrary external
    ``torch.autocast`` contexts.
    """

    enabled: bool = False
    dtype: torch.dtype | None = None
    cache_enabled: bool = False
    profile: str | None = None


_DISABLED_STATE = NPUAutocastState()
_CURRENT_STATE: ContextVar[NPUAutocastState] = ContextVar(
    "spikingjelly_npu_npu_autocast_state",
    default=_DISABLED_STATE,
)


def get_npu_autocast_state() -> NPUAutocastState:
    """Return the current package-managed NPU autocast state.

    Querying this function never imports ``torch_npu`` and is safe on hosts
    without an NPU runtime.
    """

    return _CURRENT_STATE.get()


def is_npu_bf16_autocast_active() -> bool:
    """Whether the explicit NPU BF16 mixed-precision profile is active."""

    state = get_npu_autocast_state()
    return bool(
        state.enabled
        and state.dtype == torch.bfloat16
        and state.cache_enabled is False
        and state.profile == BF16_MIXED_PRECISION_PROFILE
    )


@contextmanager
def _autocast_context(
    *,
    enabled: bool,
    dtype: torch.dtype | None,
    cache_enabled: bool,
    profile: str | None,
) -> Iterator[None]:
    if not enabled:
        previous_state = get_npu_autocast_state()
        token = _CURRENT_STATE.set(_DISABLED_STATE)
        try:
            if previous_state.enabled:
                with torch.autocast(device_type="npu", enabled=False):
                    yield
            else:
                yield
        finally:
            _CURRENT_STATE.reset(token)
        return

    kwargs = {
        "device_type": "npu",
        "enabled": True,
        "cache_enabled": cache_enabled,
    }
    if dtype is not None:
        kwargs["dtype"] = dtype
    token = _CURRENT_STATE.set(
        NPUAutocastState(
            enabled=True,
            dtype=dtype,
            cache_enabled=cache_enabled,
            profile=profile,
        )
    )
    try:
        with torch.autocast(**kwargs):
            yield
    finally:
        _CURRENT_STATE.reset(token)


def autocast(
    enabled: bool = True,
    *,
    dtype: torch.dtype | None = None,
    cache_enabled: bool = False,
):
    """Return a generic package-managed NPU autocast context.

    NPUGraph's graphed-callable API requires autocast caching to be disabled.
    On systems where PyTorch does not recognize ``device_type='npu'``, this
    helper is a no-op when disabled and propagates the runtime error when
    enabled. Generic contexts do not activate the qualified BF16 model policy;
    use :func:`npu_bf16_autocast` for that explicit contract.
    """

    return _autocast_context(
        enabled=enabled,
        dtype=dtype,
        cache_enabled=cache_enabled,
        profile=None,
    )


def npu_bf16_autocast(*, enabled: bool = True):
    """Enter the explicit NPU BF16 mixed-precision profile.

    The profile fixes ``dtype=torch.bfloat16`` and ``cache_enabled=False``.
    It only establishes execution policy; callers should first use
    ``configure_npu(..., require_bf16=True)`` to verify device capability.
    Parameters, optimizer state, recurrent state, normalization, thresholds,
    surrogate math, and reductions remain FP32 unless a component documents a
    narrower qualified boundary.
    """

    return _autocast_context(
        enabled=enabled,
        dtype=torch.bfloat16,
        cache_enabled=False,
        profile=BF16_MIXED_PRECISION_PROFILE,
    )

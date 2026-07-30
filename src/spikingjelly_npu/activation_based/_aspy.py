"""Lazy routing helpers for the optional Ascend C AsPy backend.

This module is safe to import on machines without ``torch_npu``. The native
extension is considered only after cheap device, dtype, layout, and neuron
configuration checks have passed.
"""

from __future__ import annotations

import importlib
import math
import os
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import torch

from . import surrogate


@dataclass(frozen=True)
class AsPyRoute:
    """Observable result of the most recent AsPy routing decision."""

    requested_backend: str
    backend: str
    reason: str
    accelerated: bool


@dataclass(frozen=True)
class AsPyIFResult:
    """Transactional result returned by an AsPy IF implementation."""

    spike_seq: torch.Tensor
    v_final: torch.Tensor
    v_seq: torch.Tensor | None


class AsPyBackendError(RuntimeError):
    """Raised when strict AsPy execution cannot be provided safely."""


@dataclass(frozen=True)
class _AsPyIFRequest:
    x_seq: torch.Tensor
    v_init: torch.Tensor
    v_threshold: float
    v_reset: float | None
    detach_reset: bool
    surrogate_name: str
    surrogate_alpha: float
    store_v_seq: bool


@dataclass(frozen=True)
class _AsPyLIFRequest(_AsPyIFRequest):
    tau: float
    decay_input: bool


@dataclass(frozen=True)
class _AsPyPLIFRequest(_AsPyIFRequest):
    reciprocal_tau: torch.Tensor
    decay_input: bool


@dataclass(frozen=True)
class _AsPyFedSNNDecayLIFRequest:
    current_seq: torch.Tensor
    membrane_decay: float
    v_threshold: float
    surrogate_name: str
    surrogate_alpha: float


_EXTENSION_ENV = "SPIKINGJELLY_NPU_ASPY_EXTENSION"
_DEFAULT_EXTENSION = "spikingjelly_npu_aspy"


def eager_route(requested_backend: str, reason: str) -> AsPyRoute:
    return AsPyRoute(
        requested_backend=requested_backend,
        backend="torch",
        reason=reason,
        accelerated=False,
    )


def native_route(neuron_name: str) -> AsPyRoute:
    return AsPyRoute(
        requested_backend="aspy",
        backend="aspy",
        reason=f"Ascend C fused multi-step {neuron_name} kernel",
        accelerated=True,
    )


def _unsupported_reason(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    surrogate_function: surrogate.SurrogateFunctionBase,
) -> str | None:
    if x_seq.device.type != "npu":
        return f"AsPy requires an NPU tensor, got device={x_seq.device.type}"
    if x_seq.dtype != torch.float32:
        return f"AsPy currently requires torch.float32, got dtype={x_seq.dtype}"
    if x_seq.shape[0] == 0:
        return "AsPy requires at least one time step"
    if not x_seq.is_contiguous() or x_seq.storage_offset() != 0:
        return "AsPy currently requires contiguous input with storage offset zero"
    if v_init.device != x_seq.device or v_init.dtype != x_seq.dtype:
        return "AsPy initial voltage must match the input device and dtype"
    if not v_init.is_contiguous() or v_init.storage_offset() != 0:
        return "AsPy initial voltage must be contiguous with storage offset zero"
    if v_init.shape != x_seq.shape[1:]:
        return (
            "AsPy initial voltage shape must equal one time-step shape; "
            f"got v={tuple(v_init.shape)} and step={tuple(x_seq.shape[1:])}"
        )
    for name, tensor in (("input", x_seq), ("initial voltage", v_init)):
        format_reason = _require_npu_nd(tensor)
        if format_reason is not None:
            return f"AsPy {name} is not bridge-safe: {format_reason}"
    if not isinstance(surrogate_function, surrogate.ATan):
        return "AsPy currently supports only the ATan surrogate"
    if not surrogate_function.spiking:
        return "AsPy requires the surrogate to be in spiking mode"
    return None


def _unsupported_stateless_reason(
    current_seq: torch.Tensor,
    surrogate_function: surrogate.SurrogateFunctionBase,
) -> str | None:
    if current_seq.device.type != "npu":
        return f"AsPy requires an NPU tensor, got device={current_seq.device.type}"
    if current_seq.dtype != torch.float32:
        return f"AsPy currently requires torch.float32, got dtype={current_seq.dtype}"
    if current_seq.shape[0] == 0:
        return "AsPy requires at least one time step"
    if current_seq[0].numel() == 0:
        return "AsPy requires at least one neuron per time step"
    if not current_seq.is_contiguous() or current_seq.storage_offset() != 0:
        return "AsPy currently requires contiguous input with storage offset zero"
    if not isinstance(surrogate_function, surrogate.ATan):
        return "AsPy currently supports only the ATan surrogate"
    if not surrogate_function.spiking:
        return "AsPy requires the surrogate to be in spiking mode"
    return None


def _npu_format_value(tensor: torch.Tensor) -> tuple[int | None, str | None]:
    """Inspect physical Ascend storage without importing torch-npu at package load."""

    if tensor.device.type != "npu":
        return None, None
    try:
        torch_npu = importlib.import_module("torch_npu")
    except (ImportError, OSError) as error:
        return None, f"torch_npu format inspection is unavailable: {error}"
    get_format = getattr(torch_npu, "get_npu_format", None)
    if not callable(get_format):
        get_format = getattr(getattr(torch.ops, "npu", None), "get_npu_format", None)
    if not callable(get_format):
        return None, "torch_npu does not expose get_npu_format"
    try:
        return int(get_format(tensor)), None
    except (RuntimeError, TypeError, ValueError) as error:
        return None, f"could not inspect NPU storage format: {error}"


def _require_npu_nd(tensor: torch.Tensor) -> str | None:
    """Reject non-ND Ascend storage before the ND-only native bridge loads."""

    source_value, error = _npu_format_value(tensor)
    if error is not None:
        return error
    if source_value is not None and source_value != 2:
        return (
            "AsPy native bridge requires physical ACL_FORMAT_ND (2), "
            f"got format={source_value}"
        )
    return None


def _require_fedsnn_base_format(tensor: torch.Tensor) -> str | None:
    """Allow only formats that the FedSNN adapter can copy safely to native ND.

    Real packed convolutional BNTT produces rank-5 ``ACL_FORMAT_NCDHW`` (30).
    The adapter flattens and copies that documented base format to a fresh ND
    tensor before the native bridge. Internal formats remain pre-load rejects.
    """

    source_value, error = _npu_format_value(tensor)
    if error is not None:
        return error
    if source_value is None or source_value == 2:
        return None
    if tensor.ndim == 5 and source_value == 30:
        return None
    return (
        "AsPy FedSNN decay-LIF requires physical ACL_FORMAT_ND (2) or "
        "rank-5 ACL_FORMAT_NCDHW (30), "
        f"got format={source_value}"
    )


def _load_extension() -> tuple[ModuleType | None, str | None]:
    module_name = os.environ.get(_EXTENSION_ENV, _DEFAULT_EXTENSION)
    try:
        return importlib.import_module(module_name), None
    except (ImportError, OSError) as error:
        return None, f"AsPy extension {module_name!r} is unavailable: {error}"


def _normalize_result(value: Any, store_v_seq: bool) -> AsPyIFResult:
    if isinstance(value, AsPyIFResult):
        result = value
    elif isinstance(value, tuple) and len(value) == 3:
        result = AsPyIFResult(value[0], value[1], value[2])
    else:
        raise TypeError(
            "AsPy IF extension must return AsPyIFResult or "
            "(spike_seq, v_final, v_seq)"
        )
    if not isinstance(result.spike_seq, torch.Tensor):
        raise TypeError("AsPy spike_seq result must be a tensor")
    if not isinstance(result.v_final, torch.Tensor):
        raise TypeError("AsPy v_final result must be a tensor")
    if store_v_seq and not isinstance(result.v_seq, torch.Tensor):
        raise TypeError("AsPy v_seq result must be a tensor when store_v_seq=True")
    if not store_v_seq and result.v_seq is not None:
        raise TypeError("AsPy v_seq result must be None when store_v_seq=False")
    return result


def _validate_result(result: AsPyIFResult, request: _AsPyIFRequest) -> None:
    x_seq = request.x_seq
    if result.spike_seq.shape != x_seq.shape:
        raise ValueError(
            "AsPy spike_seq shape mismatch: "
            f"expected {tuple(x_seq.shape)}, got {tuple(result.spike_seq.shape)}"
        )
    if result.v_final.shape != request.v_init.shape:
        raise ValueError(
            "AsPy v_final shape mismatch: "
            f"expected {tuple(request.v_init.shape)}, got {tuple(result.v_final.shape)}"
        )
    if result.v_seq is not None and result.v_seq.shape != x_seq.shape:
        raise ValueError(
            "AsPy v_seq shape mismatch: "
            f"expected {tuple(x_seq.shape)}, got {tuple(result.v_seq.shape)}"
        )
    for name, tensor in (
        ("spike_seq", result.spike_seq),
        ("v_final", result.v_final),
        ("v_seq", result.v_seq),
    ):
        if tensor is None:
            continue
        if tensor.device != x_seq.device or tensor.dtype != x_seq.dtype:
            raise ValueError(
                f"AsPy {name} must match input device and dtype; "
                f"got device={tensor.device}, dtype={tensor.dtype}"
            )


def try_if_multi_step(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    *,
    v_threshold: float,
    v_reset: float | None,
    detach_reset: bool,
    surrogate_function: surrogate.SurrogateFunctionBase,
    store_v_seq: bool,
    strict: bool,
) -> tuple[AsPyIFResult | None, AsPyRoute]:
    """Run fused IF when qualified, otherwise return a pre-execution fallback.

    Once the native function is invoked, its errors and malformed results are
    propagated. This avoids silently replaying eager code after a kernel may
    already have executed.
    """

    reason = _unsupported_reason(x_seq, v_init, surrogate_function)
    if reason is not None:
        if strict:
            raise AsPyBackendError(reason)
        return None, eager_route("aspy", reason)

    extension, load_error = _load_extension()
    if extension is None:
        assert load_error is not None
        if strict:
            raise AsPyBackendError(load_error)
        return None, eager_route("aspy", load_error)

    implementation = getattr(extension, "if_multi_step", None)
    if not callable(implementation):
        reason = "AsPy extension does not provide callable if_multi_step"
        if strict:
            raise AsPyBackendError(reason)
        return None, eager_route("aspy", reason)

    request = _AsPyIFRequest(
        x_seq=x_seq,
        v_init=v_init,
        v_threshold=v_threshold,
        v_reset=v_reset,
        detach_reset=detach_reset,
        surrogate_name="atan",
        surrogate_alpha=float(surrogate_function.alpha),
        store_v_seq=store_v_seq,
    )
    raw_result = implementation(
        request.x_seq,
        request.v_init,
        request.v_threshold,
        request.v_reset,
        request.detach_reset,
        request.surrogate_name,
        request.surrogate_alpha,
        request.store_v_seq,
    )
    result = _normalize_result(raw_result, store_v_seq)
    _validate_result(result, request)
    return result, native_route("IF")


def try_lif_multi_step(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    *,
    v_threshold: float,
    v_reset: float | None,
    detach_reset: bool,
    surrogate_function: surrogate.SurrogateFunctionBase,
    store_v_seq: bool,
    tau: float,
    decay_input: bool,
    strict: bool,
) -> tuple[AsPyIFResult | None, AsPyRoute]:
    """Run fused fixed-tau LIF or return an observable pre-execution fallback."""

    reason = _unsupported_reason(x_seq, v_init, surrogate_function)
    if reason is None and (not isinstance(tau, float) or tau <= 1.0):
        reason = "AsPy LIF requires fixed float tau greater than 1"
    if reason is not None:
        if strict:
            raise AsPyBackendError(reason)
        return None, eager_route("aspy", reason)

    extension, load_error = _load_extension()
    if extension is None:
        assert load_error is not None
        if strict:
            raise AsPyBackendError(load_error)
        return None, eager_route("aspy", load_error)

    implementation = getattr(extension, "lif_multi_step", None)
    if not callable(implementation):
        reason = "AsPy extension does not provide callable lif_multi_step"
        if strict:
            raise AsPyBackendError(reason)
        return None, eager_route("aspy", reason)

    request = _AsPyLIFRequest(
        x_seq=x_seq,
        v_init=v_init,
        v_threshold=v_threshold,
        v_reset=v_reset,
        detach_reset=detach_reset,
        surrogate_name="atan",
        surrogate_alpha=float(surrogate_function.alpha),
        store_v_seq=store_v_seq,
        tau=tau,
        decay_input=decay_input,
    )
    raw_result = implementation(
        request.x_seq,
        request.v_init,
        request.v_threshold,
        request.v_reset,
        request.detach_reset,
        request.surrogate_name,
        request.surrogate_alpha,
        request.store_v_seq,
        request.tau,
        request.decay_input,
    )
    result = _normalize_result(raw_result, store_v_seq)
    _validate_result(result, request)
    return result, native_route("LIF")


def try_plif_multi_step(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    reciprocal_tau: torch.Tensor,
    *,
    v_threshold: float,
    v_reset: float | None,
    detach_reset: bool,
    surrogate_function: surrogate.SurrogateFunctionBase,
    store_v_seq: bool,
    decay_input: bool,
    strict: bool,
) -> tuple[AsPyIFResult | None, AsPyRoute]:
    """Run fused learnable-tau PLIF with a dynamic device scalar input."""

    reason = _unsupported_reason(x_seq, v_init, surrogate_function)
    if reason is None and (
        reciprocal_tau.device != x_seq.device
        or reciprocal_tau.dtype != torch.float32
        or reciprocal_tau.numel() != 1
        or not reciprocal_tau.is_contiguous()
        or reciprocal_tau.storage_offset() != 0
    ):
        reason = (
            "AsPy PLIF reciprocal_tau must be a contiguous FP32 scalar tensor "
            "on the input NPU with storage offset zero"
        )
    if reason is None:
        format_reason = _require_npu_nd(reciprocal_tau)
        if format_reason is not None:
            reason = f"AsPy PLIF reciprocal_tau is not bridge-safe: {format_reason}"
    if reason is not None:
        if strict:
            raise AsPyBackendError(reason)
        return None, eager_route("aspy", reason)

    extension, load_error = _load_extension()
    if extension is None:
        assert load_error is not None
        if strict:
            raise AsPyBackendError(load_error)
        return None, eager_route("aspy", load_error)

    implementation = getattr(extension, "plif_multi_step", None)
    if not callable(implementation):
        reason = "AsPy extension does not provide callable plif_multi_step"
        if strict:
            raise AsPyBackendError(reason)
        return None, eager_route("aspy", reason)

    request = _AsPyPLIFRequest(
        x_seq=x_seq,
        v_init=v_init,
        reciprocal_tau=reciprocal_tau,
        v_threshold=v_threshold,
        v_reset=v_reset,
        detach_reset=detach_reset,
        surrogate_name="atan",
        surrogate_alpha=float(surrogate_function.alpha),
        store_v_seq=store_v_seq,
        decay_input=decay_input,
    )
    raw_result = implementation(
        request.x_seq,
        request.v_init,
        request.reciprocal_tau,
        request.v_threshold,
        request.v_reset,
        request.detach_reset,
        request.surrogate_name,
        request.surrogate_alpha,
        request.store_v_seq,
        request.decay_input,
    )
    result = _normalize_result(raw_result, store_v_seq)
    _validate_result(result, request)
    return result, native_route("PLIF")


def try_fedsnn_decay_lif(
    current_seq: torch.Tensor,
    *,
    membrane_decay: float,
    v_threshold: float,
    surrogate_function: surrogate.SurrogateFunctionBase,
    strict: bool,
) -> tuple[torch.Tensor | None, AsPyRoute]:
    """Run the exact stateless FedSNN decay-LIF scan when qualified."""

    reason = _unsupported_stateless_reason(current_seq, surrogate_function)
    if reason is None and (
        not isinstance(membrane_decay, float)
        or not 0.0 <= membrane_decay <= 1.0
    ):
        reason = "AsPy FedSNN decay-LIF requires float membrane_decay in [0, 1]"
    if reason is None and not math.isfinite(v_threshold):
        reason = "AsPy FedSNN decay-LIF requires a finite v_threshold"
    if reason is None and (
        not math.isfinite(surrogate_function.alpha)
        or surrogate_function.alpha <= 0.0
    ):
        reason = "AsPy FedSNN decay-LIF requires finite positive ATan alpha"
    if reason is not None:
        if strict:
            raise AsPyBackendError(reason)
        return None, eager_route("aspy", reason)

    format_error = _require_fedsnn_base_format(current_seq)
    if format_error is not None:
        if strict:
            raise AsPyBackendError(format_error)
        return None, eager_route("aspy", format_error)

    extension, load_error = _load_extension()
    if extension is None:
        assert load_error is not None
        if strict:
            raise AsPyBackendError(load_error)
        return None, eager_route("aspy", load_error)

    implementation = getattr(extension, "fedsnn_decay_lif", None)
    supports_feature = getattr(extension, "supports_fedsnn_decay_lif", None)
    if not callable(implementation) or supports_feature is not True:
        reason = "AsPy extension does not provide FedSNN decay-LIF support"
        if strict:
            raise AsPyBackendError(reason)
        return None, eager_route("aspy", reason)

    request = _AsPyFedSNNDecayLIFRequest(
        current_seq=current_seq,
        membrane_decay=membrane_decay,
        v_threshold=v_threshold,
        surrogate_name="atan",
        surrogate_alpha=float(surrogate_function.alpha),
    )
    spike_seq = implementation(
        request.current_seq,
        request.membrane_decay,
        request.v_threshold,
        request.surrogate_name,
        request.surrogate_alpha,
    )
    if not isinstance(spike_seq, torch.Tensor):
        raise TypeError("AsPy FedSNN decay-LIF result must be a tensor")
    if spike_seq.shape != current_seq.shape:
        raise ValueError(
            "AsPy FedSNN decay-LIF spike_seq shape mismatch: "
            f"expected {tuple(current_seq.shape)}, got {tuple(spike_seq.shape)}"
        )
    if spike_seq.device != current_seq.device or spike_seq.dtype != current_seq.dtype:
        raise ValueError(
            "AsPy FedSNN decay-LIF spike_seq must match input device and dtype"
        )
    return spike_seq, native_route("FedSNN decay-LIF")


__all__ = ["AsPyBackendError", "AsPyIFResult", "AsPyRoute"]

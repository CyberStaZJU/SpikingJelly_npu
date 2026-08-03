"""Lazy routing helpers for the optional Ascend C AsPy backend.

This module is safe to import on machines without ``torch_npu``. The native
extension is considered only after cheap device, dtype, layout, and neuron
configuration checks have passed. Production decisions use the shared
:class:`spikingjelly_npu.routing.ProviderRoute` contract while preserving the
historical ``AsPyRoute`` name and compatibility properties.
"""

from __future__ import annotations

import importlib
import math
import os
from dataclasses import dataclass
from types import ModuleType
from typing import Any, NoReturn

import torch

from spikingjelly_npu.routing import (
    AsPyCapabilities,
    ProviderRoute,
    StrictProviderError,
    accelerated_route,
    probe_aspy_capabilities,
    strict_pre_execution_rejection,
    torch_route,
)

from . import surrogate

ASPY_ROUTE_SCHEMA_VERSION = 3


class AsPyRoute(ProviderRoute):
    """Backward-compatible name for an observable provider route.

    ``ASPY_ROUTE_SCHEMA_VERSION == 3`` defines the additive ``__dict__`` payload.
    The inherited ``requested_backend``, ``backend``, and ``training`` properties
    preserve the old API. ``__dict__`` is an additive serialization view: it keeps
    those historical keys while retaining every immutable :class:`ProviderRoute`
    field and identifying this compatibility schema explicitly.
    """

    __slots__ = ()

    @property
    def __dict__(self) -> dict[str, Any]:
        return {
            "requested_backend": self.requested_backend,
            "backend": self.backend,
            "training": self.training,
            **self.to_dict(),
            "route_schema_version": ASPY_ROUTE_SCHEMA_VERSION,
        }


@dataclass(frozen=True)
class AsPyIFResult:
    """Transactional result returned by an AsPy IF implementation."""

    spike_seq: torch.Tensor
    v_final: torch.Tensor
    v_seq: torch.Tensor | None


@dataclass(frozen=True)
class _AsPyPrecisionPlan:
    """Public/native dtype contract for one AsPy request."""

    public_dtype: torch.dtype
    native_dtype: torch.dtype
    dtype_conversion: str | None
    dtype_conversion_bytes: int


class AsPyBackendError(RuntimeError):
    """Raised when strict AsPy execution cannot be provided safely."""

    def __init__(self, route: ProviderRoute | str) -> None:
        if isinstance(route, str):
            route = _route(
                requested_provider="aspy",
                actual_provider=None,
                logical_operation="activation_based.unknown",
                reason_code="aspy.strict_rejection",
                reason=route,
                accelerated=False,
                strict=True,
                mode="eval",
                native_launch_attempted=False,
            )
        if route.actual_provider is not None or not route.strict:
            raise ValueError("AsPyBackendError requires a strict pre-execution route")
        self.route = route
        super().__init__(route.reason)


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
class _AsPyKLIFRequest(_AsPyIFRequest):
    k: torch.Tensor
    tau: float
    decay_input: bool
    scale_reset: bool


@dataclass(frozen=True)
class _AsPyFedSNNDecayLIFRequest:
    current_seq: torch.Tensor
    membrane_decay: float
    v_threshold: float
    surrogate_name: str
    surrogate_alpha: float


@dataclass(frozen=True)
class _LoadedAsPy:
    module: ModuleType | None
    capabilities: AsPyCapabilities


_EXTENSION_ENV = "SPIKINGJELLY_NPU_ASPY_EXTENSION"
_DEFAULT_EXTENSION = "spikingjelly_npu_aspy"

_LOGICAL_OPERATIONS = {
    "if": "activation_based.neuron.if.multi_step",
    "if_compact": "activation_based.neuron.if.multi_step",
    "lif": "activation_based.neuron.lif.multi_step",
    "lif_compact": "activation_based.neuron.lif.multi_step",
    "plif": "activation_based.neuron.plif.multi_step",
    "klif": "activation_based.neuron.klif.multi_step",
    "fedsnn_decay_lif": "fedsnn.decay_lif",
}

_NATIVE_REASONS = {
    "if": "Ascend C fused multi-step IF kernel",
    "if_compact": "Ascend C compact multi-step IF kernel without public voltage sequence",
    "lif": "Ascend C fused multi-step LIF kernel",
    "lif_compact": "Ascend C compact multi-step LIF kernel without public voltage sequence",
    "plif": "Ascend C fused multi-step PLIF kernel",
    "klif": "Ascend C fused multi-step KLIF kernel",
    "fedsnn_decay_lif": "Ascend C fused multi-step FedSNN decay-LIF kernel",
}


def _route(**values: Any) -> AsPyRoute:
    return AsPyRoute(**values)


def _mode(training: bool) -> str:
    return "train" if training else "eval"


def eager_route(
    requested_backend: str,
    reason: str,
    *,
    logical_operation: str = "activation_based.eager",
    reason_code: str = "torch.selected",
    strict: bool = False,
    training: bool = False,
    abi_version: int | None = None,
    schema_version: int | None = None,
    format_conversion: str | None = None,
    dtype_conversion: str | None = None,
    dtype_conversion_bytes: int | None = None,
) -> AsPyRoute:
    """Return an executed PyTorch route using the shared provider contract."""

    route = torch_route(
        logical_operation,
        requested_provider=requested_backend,
        reason_code=reason_code,
        reason=reason,
        strict=strict,
        mode=_mode(training),
        abi_version=abi_version,
        schema_version=schema_version,
        format_conversion=format_conversion,
        dtype_conversion=dtype_conversion,
        dtype_conversion_bytes=dtype_conversion_bytes,
    )
    return _route(**route.to_dict())


def native_route(
    capability: str,
    capabilities: AsPyCapabilities | None = None,
    *,
    requested_backend: str = "aspy",
    strict: bool = False,
    training: bool = False,
    format_conversion: str | None = None,
    dtype_conversion: str | None = None,
    dtype_conversion_bytes: int | None = None,
) -> AsPyRoute:
    """Return an executed native route after one implementation call succeeded."""

    if capability in _LOGICAL_OPERATIONS:
        logical_operation = _LOGICAL_OPERATIONS[capability]
        reason_code = f"aspy.{capability}.native"
        reason = _NATIVE_REASONS[capability]
        native_region = capability
    else:
        # Historical callers supplied display names such as ``"IF"``. Preserve
        # that helper API while production dispatch uses stable capability keys.
        logical_operation = "activation_based.neuron.multi_step"
        reason_code = "aspy.native"
        reason = f"Ascend C fused multi-step {capability} kernel"
        native_region = capability.lower()
    route = accelerated_route(
        logical_operation,
        requested_provider=requested_backend,
        actual_provider="aspy",
        reason_code=reason_code,
        reason=reason,
        strict=strict,
        mode=_mode(training),
        native_launch_attempted=True,
        abi_version=None if capabilities is None else capabilities.abi_version,
        schema_version=None if capabilities is None else capabilities.schema_version,
        native_region=native_region,
        format_conversion=format_conversion,
        dtype_conversion=dtype_conversion,
        dtype_conversion_bytes=dtype_conversion_bytes,
    )
    return _route(**route.to_dict())


def _strict_rejection(
    capability: str,
    *,
    reason_code: str,
    reason: str,
    training: bool,
    requested_backend: str = "aspy",
    capabilities: AsPyCapabilities | None = None,
    format_conversion: str | None = None,
    dtype_conversion: str | None = None,
    dtype_conversion_bytes: int | None = None,
) -> NoReturn:
    try:
        strict_pre_execution_rejection(
            _LOGICAL_OPERATIONS[capability],
            requested_provider=requested_backend,
            reason_code=reason_code,
            reason=reason,
            mode=_mode(training),
            abi_version=None if capabilities is None else capabilities.abi_version,
            schema_version=None if capabilities is None else capabilities.schema_version,
            native_region=capability,
            format_conversion=format_conversion,
            dtype_conversion=dtype_conversion,
            dtype_conversion_bytes=dtype_conversion_bytes,
        )
    except StrictProviderError as error:
        raise AsPyBackendError(_route(**error.route.to_dict())) from error


def _fallback_or_reject(
    capability: str,
    *,
    reason_code: str,
    reason: str,
    strict: bool,
    training: bool,
    requested_backend: str = "aspy",
    capabilities: AsPyCapabilities | None = None,
    format_conversion: str | None = None,
    dtype_conversion: str | None = None,
    dtype_conversion_bytes: int | None = None,
) -> tuple[None, AsPyRoute]:
    if strict:
        _strict_rejection(
            capability,
            reason_code=reason_code,
            reason=reason,
            training=training,
            requested_backend=requested_backend,
            capabilities=capabilities,
            format_conversion=format_conversion,
            dtype_conversion=dtype_conversion,
            dtype_conversion_bytes=dtype_conversion_bytes,
        )
    return None, eager_route(
        requested_backend,
        reason,
        logical_operation=_LOGICAL_OPERATIONS[capability],
        reason_code=reason_code,
        strict=False,
        training=training,
        abi_version=None if capabilities is None else capabilities.abi_version,
        schema_version=None if capabilities is None else capabilities.schema_version,
        format_conversion=format_conversion,
        dtype_conversion=dtype_conversion,
        dtype_conversion_bytes=dtype_conversion_bytes,
    )


def _unsupported_scalar_reason(
    surrogate_function: surrogate.SurrogateFunctionBase,
    *,
    v_threshold: float,
    v_reset: float | None,
) -> str | None:
    if not isinstance(surrogate_function, surrogate.ATan):
        return "AsPy currently supports only the ATan surrogate"
    if not surrogate_function.spiking:
        return "AsPy requires the surrogate to be in spiking mode"
    if not math.isfinite(v_threshold):
        return "AsPy requires a finite v_threshold"
    if v_reset is not None and not math.isfinite(v_reset):
        return "AsPy requires a finite v_reset"
    if not math.isfinite(surrogate_function.alpha) or surrogate_function.alpha <= 0.0:
        return "AsPy requires finite positive ATan alpha"
    return None


def _supported_public_dtype_reason(tensor: torch.Tensor) -> str | None:
    if tensor.dtype not in {torch.float32, torch.bfloat16}:
        return (
            "AsPy supports torch.float32 or torch.bfloat16 public tensors; "
            f"got dtype={tensor.dtype}"
        )
    return None


def _precision_plan(
    public_tensor: torch.Tensor,
    *fp32_native_inputs: torch.Tensor,
) -> _AsPyPrecisionPlan:
    if public_tensor.dtype == torch.float32:
        return _AsPyPrecisionPlan(torch.float32, torch.float32, None, 0)
    fp32_bytes = torch.empty((), dtype=torch.float32).element_size()
    conversion_bytes = sum(
        tensor.numel() * (tensor.element_size() + fp32_bytes)
        for tensor in (public_tensor, *fp32_native_inputs)
        if tensor.dtype != torch.float32
    )
    # This field estimates forward boundary traffic only. Training backward
    # cast traffic is measured separately by end-to-end profilers rather than
    # predicted by the immutable forward route record.
    # The public spike sequence crosses the same boundary in the opposite
    # direction after the FP32 native recurrence completes.
    conversion_bytes += public_tensor.numel() * (fp32_bytes + public_tensor.element_size())
    return _AsPyPrecisionPlan(
        public_dtype=torch.bfloat16,
        native_dtype=torch.float32,
        dtype_conversion="bf16-public-fp32-aspy-island",
        dtype_conversion_bytes=conversion_bytes,
    )


def _to_native_fp32(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.float32:
        return tensor
    return tensor.to(dtype=torch.float32).contiguous()


def _public_result(
    result: AsPyIFResult,
    plan: _AsPyPrecisionPlan,
) -> AsPyIFResult:
    if plan.public_dtype == plan.native_dtype:
        return result
    return AsPyIFResult(
        spike_seq=result.spike_seq.to(dtype=plan.public_dtype).contiguous(),
        v_final=result.v_final,
        v_seq=result.v_seq,
    )


def _prepare_stateful_native_inputs(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
) -> tuple[_AsPyPrecisionPlan, torch.Tensor, torch.Tensor, str | None]:
    plan = _precision_plan(x_seq, v_init)
    native_x_seq = _to_native_fp32(x_seq)
    native_v_init = _to_native_fp32(v_init)
    for name, tensor in (("input", native_x_seq), ("initial voltage", native_v_init)):
        format_reason = _require_npu_nd(tensor)
        if format_reason is not None:
            return (
                plan,
                native_x_seq,
                native_v_init,
                f"AsPy converted {name} is not bridge-safe: {format_reason}",
            )
    return plan, native_x_seq, native_v_init, None


def _unsupported_reason(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    surrogate_function: surrogate.SurrogateFunctionBase,
    *,
    v_threshold: float,
    v_reset: float | None,
) -> str | None:
    if x_seq.device.type != "npu":
        return f"AsPy requires an NPU tensor, got device={x_seq.device.type}"
    dtype_reason = _supported_public_dtype_reason(x_seq)
    if dtype_reason is not None:
        return dtype_reason
    if x_seq.shape[0] == 0:
        return "AsPy requires at least one time step"
    if not x_seq.is_contiguous() or x_seq.storage_offset() != 0:
        return "AsPy currently requires contiguous input with storage offset zero"
    if v_init.device != x_seq.device:
        return "AsPy initial voltage must match the input device"
    if v_init.dtype not in {x_seq.dtype, torch.float32}:
        return (
            "AsPy initial voltage must match the public input dtype or use the "
            "qualified FP32 state dtype"
        )
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
    return _unsupported_scalar_reason(
        surrogate_function,
        v_threshold=v_threshold,
        v_reset=v_reset,
    )


def _unsupported_stateless_reason(
    current_seq: torch.Tensor,
    surrogate_function: surrogate.SurrogateFunctionBase,
) -> str | None:
    if current_seq.device.type != "npu":
        return f"AsPy requires an NPU tensor, got device={current_seq.device.type}"
    dtype_reason = _supported_public_dtype_reason(current_seq)
    if dtype_reason is not None:
        return dtype_reason
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
        return f"AsPy native bridge requires physical ACL_FORMAT_ND (2), got format={source_value}"
    return None


def _require_fedsnn_base_format(tensor: torch.Tensor) -> str | None:
    """Allow only formats that the FedSNN adapter can copy safely to native ND."""

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


def _failed_capabilities(error: BaseException) -> AsPyCapabilities:
    def raise_error() -> object:
        raise error

    return probe_aspy_capabilities(loader=raise_error)


def _load_extension() -> tuple[ModuleType | None, AsPyCapabilities]:
    """Load and probe the public adapter without launching a native operator."""

    module_name = os.environ.get(_EXTENSION_ENV, _DEFAULT_EXTENSION)
    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        return None, _failed_capabilities(error)
    return module, _adapter_capabilities(module)


def _adapter_capabilities(module: object) -> AsPyCapabilities:
    """Probe native or public-adapter metadata without launching an operator."""

    capabilities = probe_aspy_capabilities(module)
    if capabilities.available:
        return capabilities
    native = getattr(module, "_native", None)
    if native is not None:
        # The loaded native object is authoritative. In particular, do not turn
        # malformed or partial declared metadata into an inferred adapter success.
        return probe_aspy_capabilities(native)
    if any(hasattr(module, name) for name in ("aspy_abi_version", "aspy_capabilities")):
        return capabilities
    adapter_groups = {
        "if": ("if_multi_step",),
        "lif": ("lif_multi_step",),
        "klif": ("klif_multi_step",),
        "plif": ("plif_multi_step",),
        "fedsnn_decay_lif": ("fedsnn_decay_lif",),
    }
    symbols = {
        name: (lambda: None)
        for name, required in adapter_groups.items()
        if all(callable(getattr(module, symbol, None)) for symbol in required)
    }
    return probe_aspy_capabilities(
        type(
            "_LegacyAdapterCapabilities",
            (),
            {f"{name}_forward": value for name, value in symbols.items()}
            | {f"{name}_backward": value for name, value in symbols.items()},
        )()
    )


def _normalize_loaded(value: object) -> _LoadedAsPy:
    """Normalize the production loader and historical two-tuple test seams."""

    if isinstance(value, _LoadedAsPy):
        return value
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError("_load_extension() must return (module, capabilities)")
    module, metadata = value
    if isinstance(metadata, AsPyCapabilities):
        return _LoadedAsPy(module, metadata)
    if module is not None:
        return _LoadedAsPy(module, _adapter_capabilities(module))
    if metadata is None:
        error: BaseException = ImportError("AsPy extension is absent")
    elif isinstance(metadata, BaseException):
        error = metadata
    else:
        # Historical tests returned a human-readable string for absence. Do not
        # infer unloadable state from wording; production passes structured data.
        error = ImportError(str(metadata))
    return _LoadedAsPy(None, _failed_capabilities(error))


def _load_for_capability(
    capability: str,
    *,
    strict: bool,
    training: bool,
    adapter_symbol: str,
    requested_backend: str = "aspy",
    dtype_conversion: str | None = None,
    dtype_conversion_bytes: int | None = None,
) -> tuple[ModuleType | None, AsPyCapabilities, AsPyRoute | None]:
    loaded = _normalize_loaded(_load_extension())
    capabilities = loaded.capabilities
    if loaded.module is None:
        _, route = _fallback_or_reject(
            capability,
            reason_code=capabilities.reason_code,
            reason=capabilities.reason,
            strict=strict,
            training=training,
            requested_backend=requested_backend,
            capabilities=capabilities,
            dtype_conversion=dtype_conversion,
            dtype_conversion_bytes=dtype_conversion_bytes,
        )
        return None, capabilities, route
    implementation = getattr(loaded.module, adapter_symbol, None)
    if not callable(implementation):
        _, route = _fallback_or_reject(
            capability,
            reason_code=f"aspy.{capability}.missing_adapter",
            reason=f"AsPy extension does not provide callable {adapter_symbol}",
            strict=strict,
            training=training,
            requested_backend=requested_backend,
            capabilities=capabilities,
            dtype_conversion=dtype_conversion,
            dtype_conversion_bytes=dtype_conversion_bytes,
        )
        return None, capabilities, route
    if not capabilities.supports(capability):
        _, route = _fallback_or_reject(
            capability,
            reason_code=f"aspy.{capability}.unsupported_bundle",
            reason=(
                f"AsPy bundle does not provide complete {capability} capability; "
                f"{capabilities.reason}"
            ),
            strict=strict,
            training=training,
            requested_backend=requested_backend,
            capabilities=capabilities,
            dtype_conversion=dtype_conversion,
            dtype_conversion_bytes=dtype_conversion_bytes,
        )
        return None, capabilities, route
    return loaded.module, capabilities, None


def _normalize_result(value: Any, store_v_seq: bool) -> AsPyIFResult:
    if isinstance(value, AsPyIFResult):
        result = value
    elif isinstance(value, tuple) and len(value) == 3:
        result = AsPyIFResult(value[0], value[1], value[2])
    else:
        raise TypeError("AsPy IF extension must return AsPyIFResult or (spike_seq, v_final, v_seq)")
    if not isinstance(result.spike_seq, torch.Tensor):
        raise TypeError("AsPy spike_seq result must be a tensor")
    if not isinstance(result.v_final, torch.Tensor):
        raise TypeError("AsPy v_final result must be a tensor")
    if store_v_seq and not isinstance(result.v_seq, torch.Tensor):
        raise TypeError("AsPy v_seq result must be a tensor when store_v_seq=True")
    if not store_v_seq and result.v_seq is not None:
        raise TypeError("AsPy v_seq result must be None when store_v_seq=False")
    return result


def _validate_native_result(result: AsPyIFResult, request: _AsPyIFRequest) -> None:
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
        if not tensor.is_contiguous() or tensor.storage_offset() != 0:
            raise ValueError(f"AsPy {name} must be contiguous with storage offset zero")
        format_reason = _require_npu_nd(tensor)
        if format_reason is not None:
            raise ValueError(f"AsPy {name} is not bridge-safe: {format_reason}")


def _executed_neuron_capability(
    capability: str,
    *,
    store_v_seq: bool,
    capabilities: AsPyCapabilities,
) -> str:
    compact = f"{capability}_compact"
    if not store_v_seq and capabilities.supports(compact):
        return compact
    return capability


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
    training: bool = False,
    requested_backend: str = "aspy",
) -> tuple[AsPyIFResult | None, AsPyRoute]:
    """Run fused IF when qualified, otherwise return a pre-execution fallback."""

    reason = _unsupported_reason(
        x_seq,
        v_init,
        surrogate_function,
        v_threshold=v_threshold,
        v_reset=v_reset,
    )
    if reason is not None:
        return _fallback_or_reject(
            "if",
            reason_code="aspy.if.unsupported_request",
            reason=reason,
            strict=strict,
            training=training,
            requested_backend=requested_backend,
        )

    plan, native_x_seq, native_v_init, conversion_error = _prepare_stateful_native_inputs(
        x_seq, v_init
    )
    if conversion_error is not None:
        return _fallback_or_reject(
            "if",
            reason_code="aspy.if.unsupported_converted_format",
            reason=conversion_error,
            strict=strict,
            training=training,
            requested_backend=requested_backend,
            dtype_conversion=plan.dtype_conversion,
            dtype_conversion_bytes=plan.dtype_conversion_bytes,
        )

    extension, capabilities, fallback = _load_for_capability(
        "if",
        strict=strict,
        training=training,
        adapter_symbol="if_multi_step",
        requested_backend=requested_backend,
        dtype_conversion=plan.dtype_conversion,
        dtype_conversion_bytes=plan.dtype_conversion_bytes,
    )
    if extension is None:
        assert fallback is not None
        return None, fallback

    request = _AsPyIFRequest(
        x_seq=native_x_seq,
        v_init=native_v_init,
        v_threshold=v_threshold,
        v_reset=v_reset,
        detach_reset=detach_reset,
        surrogate_name="atan",
        surrogate_alpha=float(surrogate_function.alpha),
        store_v_seq=store_v_seq,
    )
    implementation = extension.if_multi_step
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
    native_result = _normalize_result(raw_result, store_v_seq)
    _validate_native_result(native_result, request)
    result = _public_result(native_result, plan)
    return result, native_route(
        _executed_neuron_capability("if", store_v_seq=store_v_seq, capabilities=capabilities),
        capabilities,
        requested_backend=requested_backend,
        strict=strict,
        training=training,
        dtype_conversion=plan.dtype_conversion,
        dtype_conversion_bytes=plan.dtype_conversion_bytes,
    )


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
    training: bool = False,
    requested_backend: str = "aspy",
) -> tuple[AsPyIFResult | None, AsPyRoute]:
    """Run fused fixed-tau LIF or return an observable pre-execution fallback."""

    reason = _unsupported_reason(
        x_seq,
        v_init,
        surrogate_function,
        v_threshold=v_threshold,
        v_reset=v_reset,
    )
    if reason is None and (not isinstance(tau, float) or not math.isfinite(tau) or tau <= 1.0):
        reason = "AsPy LIF requires finite fixed float tau greater than 1"
    if reason is not None:
        return _fallback_or_reject(
            "lif",
            reason_code="aspy.lif.unsupported_request",
            reason=reason,
            strict=strict,
            training=training,
            requested_backend=requested_backend,
        )

    plan, native_x_seq, native_v_init, conversion_error = _prepare_stateful_native_inputs(
        x_seq, v_init
    )
    if conversion_error is not None:
        return _fallback_or_reject(
            "lif",
            reason_code="aspy.lif.unsupported_converted_format",
            reason=conversion_error,
            strict=strict,
            training=training,
            requested_backend=requested_backend,
            dtype_conversion=plan.dtype_conversion,
            dtype_conversion_bytes=plan.dtype_conversion_bytes,
        )

    extension, capabilities, fallback = _load_for_capability(
        "lif",
        strict=strict,
        training=training,
        adapter_symbol="lif_multi_step",
        requested_backend=requested_backend,
        dtype_conversion=plan.dtype_conversion,
        dtype_conversion_bytes=plan.dtype_conversion_bytes,
    )
    if extension is None:
        assert fallback is not None
        return None, fallback

    request = _AsPyLIFRequest(
        x_seq=native_x_seq,
        v_init=native_v_init,
        v_threshold=v_threshold,
        v_reset=v_reset,
        detach_reset=detach_reset,
        surrogate_name="atan",
        surrogate_alpha=float(surrogate_function.alpha),
        store_v_seq=store_v_seq,
        tau=tau,
        decay_input=decay_input,
    )
    implementation = extension.lif_multi_step
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
    native_result = _normalize_result(raw_result, store_v_seq)
    _validate_native_result(native_result, request)
    result = _public_result(native_result, plan)
    return result, native_route(
        _executed_neuron_capability("lif", store_v_seq=store_v_seq, capabilities=capabilities),
        capabilities,
        requested_backend=requested_backend,
        strict=strict,
        training=training,
        dtype_conversion=plan.dtype_conversion,
        dtype_conversion_bytes=plan.dtype_conversion_bytes,
    )


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
    training: bool = False,
    requested_backend: str = "aspy",
) -> tuple[AsPyIFResult | None, AsPyRoute]:
    """Run fused learnable-tau PLIF with a dynamic device scalar input."""

    reason = _unsupported_reason(
        x_seq,
        v_init,
        surrogate_function,
        v_threshold=v_threshold,
        v_reset=v_reset,
    )
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
        return _fallback_or_reject(
            "plif",
            reason_code="aspy.plif.unsupported_request",
            reason=reason,
            strict=strict,
            training=training,
            requested_backend=requested_backend,
        )

    plan, native_x_seq, native_v_init, conversion_error = _prepare_stateful_native_inputs(
        x_seq, v_init
    )
    if conversion_error is not None:
        return _fallback_or_reject(
            "plif",
            reason_code="aspy.plif.unsupported_converted_format",
            reason=conversion_error,
            strict=strict,
            training=training,
            requested_backend=requested_backend,
            dtype_conversion=plan.dtype_conversion,
            dtype_conversion_bytes=plan.dtype_conversion_bytes,
        )

    extension, capabilities, fallback = _load_for_capability(
        "plif",
        strict=strict,
        training=training,
        adapter_symbol="plif_multi_step",
        requested_backend=requested_backend,
        dtype_conversion=plan.dtype_conversion,
        dtype_conversion_bytes=plan.dtype_conversion_bytes,
    )
    if extension is None:
        assert fallback is not None
        return None, fallback

    request = _AsPyPLIFRequest(
        x_seq=native_x_seq,
        v_init=native_v_init,
        reciprocal_tau=reciprocal_tau,
        v_threshold=v_threshold,
        v_reset=v_reset,
        detach_reset=detach_reset,
        surrogate_name="atan",
        surrogate_alpha=float(surrogate_function.alpha),
        store_v_seq=store_v_seq,
        decay_input=decay_input,
    )
    implementation = extension.plif_multi_step
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
    native_result = _normalize_result(raw_result, store_v_seq)
    _validate_native_result(native_result, request)
    result = _public_result(native_result, plan)
    return result, native_route(
        "plif",
        capabilities,
        requested_backend=requested_backend,
        strict=strict,
        training=training,
        dtype_conversion=plan.dtype_conversion,
        dtype_conversion_bytes=plan.dtype_conversion_bytes,
    )


def try_klif_multi_step(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    k: torch.Tensor,
    *,
    v_threshold: float,
    v_reset: float | None,
    detach_reset: bool,
    surrogate_function: surrogate.SurrogateFunctionBase,
    store_v_seq: bool,
    tau: float,
    decay_input: bool,
    scale_reset: bool,
    strict: bool,
    training: bool = False,
    requested_backend: str = "aspy",
) -> tuple[AsPyIFResult | None, AsPyRoute]:
    """Run fused KLIF with a dynamic scalar ``k`` or pre-execution fallback."""

    reason = _unsupported_reason(
        x_seq,
        v_init,
        surrogate_function,
        v_threshold=v_threshold,
        v_reset=v_reset,
    )
    if reason is None and (not isinstance(tau, float) or not math.isfinite(tau) or tau <= 1.0):
        reason = "AsPy KLIF requires finite fixed float tau greater than 1"
    if reason is None and (
        k.device != x_seq.device
        or k.dtype != torch.float32
        or k.numel() != 1
        or not k.is_contiguous()
        or k.storage_offset() != 0
    ):
        reason = (
            "AsPy KLIF k must be a contiguous FP32 scalar tensor "
            "on the input NPU with storage offset zero"
        )
    if reason is None:
        format_reason = _require_npu_nd(k)
        if format_reason is not None:
            reason = f"AsPy KLIF k is not bridge-safe: {format_reason}"
    if reason is not None:
        return _fallback_or_reject(
            "klif",
            reason_code="aspy.klif.unsupported_request",
            reason=reason,
            strict=strict,
            training=training,
            requested_backend=requested_backend,
        )

    plan, native_x_seq, native_v_init, conversion_error = _prepare_stateful_native_inputs(
        x_seq, v_init
    )
    if conversion_error is not None:
        return _fallback_or_reject(
            "klif",
            reason_code="aspy.klif.unsupported_converted_format",
            reason=conversion_error,
            strict=strict,
            training=training,
            requested_backend=requested_backend,
            dtype_conversion=plan.dtype_conversion,
            dtype_conversion_bytes=plan.dtype_conversion_bytes,
        )

    extension, capabilities, fallback = _load_for_capability(
        "klif",
        strict=strict,
        training=training,
        adapter_symbol="klif_multi_step",
        requested_backend=requested_backend,
        dtype_conversion=plan.dtype_conversion,
        dtype_conversion_bytes=plan.dtype_conversion_bytes,
    )
    if extension is None:
        assert fallback is not None
        return None, fallback

    request = _AsPyKLIFRequest(
        x_seq=native_x_seq,
        v_init=native_v_init,
        k=k,
        v_threshold=v_threshold,
        v_reset=v_reset,
        detach_reset=detach_reset,
        surrogate_name="atan",
        surrogate_alpha=float(surrogate_function.alpha),
        store_v_seq=store_v_seq,
        tau=tau,
        decay_input=decay_input,
        scale_reset=scale_reset,
    )
    implementation = extension.klif_multi_step
    raw_result = implementation(
        request.x_seq,
        request.v_init,
        request.k,
        request.v_threshold,
        request.v_reset,
        request.detach_reset,
        request.surrogate_name,
        request.surrogate_alpha,
        request.store_v_seq,
        request.tau,
        request.decay_input,
        request.scale_reset,
    )
    native_result = _normalize_result(raw_result, store_v_seq)
    _validate_native_result(native_result, request)
    result = _public_result(native_result, plan)
    return result, native_route(
        "klif",
        capabilities,
        requested_backend=requested_backend,
        strict=strict,
        training=training,
        dtype_conversion=plan.dtype_conversion,
        dtype_conversion_bytes=plan.dtype_conversion_bytes,
    )


def try_fedsnn_decay_lif(
    current_seq: torch.Tensor,
    *,
    membrane_decay: float,
    v_threshold: float,
    surrogate_function: surrogate.SurrogateFunctionBase,
    strict: bool,
    training: bool = False,
    requested_backend: str = "aspy",
) -> tuple[torch.Tensor | None, AsPyRoute]:
    """Run the exact stateless FedSNN decay-LIF scan when qualified."""

    reason = _unsupported_stateless_reason(current_seq, surrogate_function)
    if reason is None and (
        not isinstance(membrane_decay, float) or not 0.0 <= membrane_decay <= 1.0
    ):
        reason = "AsPy FedSNN decay-LIF requires float membrane_decay in [0, 1]"
    if reason is None and not math.isfinite(v_threshold):
        reason = "AsPy FedSNN decay-LIF requires a finite v_threshold"
    if reason is None and (
        not math.isfinite(surrogate_function.alpha) or surrogate_function.alpha <= 0.0
    ):
        reason = "AsPy FedSNN decay-LIF requires finite positive ATan alpha"
    if reason is not None:
        return _fallback_or_reject(
            "fedsnn_decay_lif",
            reason_code="aspy.fedsnn_decay_lif.unsupported_request",
            reason=reason,
            strict=strict,
            training=training,
            requested_backend=requested_backend,
        )

    source_format = _npu_format_value(current_seq)[0]
    format_error = _require_fedsnn_base_format(current_seq)
    if format_error is not None:
        return _fallback_or_reject(
            "fedsnn_decay_lif",
            reason_code="aspy.fedsnn_decay_lif.unsupported_format",
            reason=format_error,
            strict=strict,
            training=training,
            requested_backend=requested_backend,
        )

    plan = _precision_plan(current_seq)
    native_current_seq = _to_native_fp32(current_seq)
    native_format_error = _require_fedsnn_base_format(native_current_seq)
    if native_format_error is not None:
        return _fallback_or_reject(
            "fedsnn_decay_lif",
            reason_code="aspy.fedsnn_decay_lif.unsupported_converted_format",
            reason=native_format_error,
            strict=strict,
            training=training,
            requested_backend=requested_backend,
            dtype_conversion=plan.dtype_conversion,
            dtype_conversion_bytes=plan.dtype_conversion_bytes,
        )

    extension, capabilities, fallback = _load_for_capability(
        "fedsnn_decay_lif",
        strict=strict,
        training=training,
        adapter_symbol="fedsnn_decay_lif",
        requested_backend=requested_backend,
        dtype_conversion=plan.dtype_conversion,
        dtype_conversion_bytes=plan.dtype_conversion_bytes,
    )
    if extension is None:
        assert fallback is not None
        return None, fallback

    supports_feature = getattr(extension, "supports_fedsnn_decay_lif", None)
    if supports_feature is not True:
        return _fallback_or_reject(
            "fedsnn_decay_lif",
            reason_code="aspy.fedsnn_decay_lif.missing_adapter_flag",
            reason="AsPy extension does not provide FedSNN decay-LIF support",
            strict=strict,
            training=training,
            requested_backend=requested_backend,
            capabilities=capabilities,
            dtype_conversion=plan.dtype_conversion,
            dtype_conversion_bytes=plan.dtype_conversion_bytes,
        )

    implementation = extension.fedsnn_decay_lif
    request = _AsPyFedSNNDecayLIFRequest(
        current_seq=native_current_seq,
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
    if spike_seq.shape != request.current_seq.shape:
        raise ValueError(
            "AsPy FedSNN decay-LIF spike_seq shape mismatch: "
            f"expected {tuple(request.current_seq.shape)}, got {tuple(spike_seq.shape)}"
        )
    if (
        spike_seq.device != request.current_seq.device
        or spike_seq.dtype != request.current_seq.dtype
    ):
        raise ValueError(
            "AsPy FedSNN decay-LIF native spike_seq must match native input device and dtype"
        )
    if not spike_seq.is_contiguous() or spike_seq.storage_offset() != 0:
        raise ValueError(
            "AsPy FedSNN decay-LIF native spike_seq must be contiguous with storage offset zero"
        )
    spike_format_error = _require_npu_nd(spike_seq)
    if spike_format_error is not None:
        raise ValueError(
            f"AsPy FedSNN decay-LIF native spike_seq is not bridge-safe: {spike_format_error}"
        )
    format_conversion = (
        "ncdhw-to-nd-copy" if current_seq.ndim == 5 and source_format == 30 else None
    )
    public_spike_seq = (
        spike_seq
        if plan.public_dtype == plan.native_dtype
        else spike_seq.to(dtype=plan.public_dtype).contiguous()
    )
    return public_spike_seq, native_route(
        "fedsnn_decay_lif",
        capabilities,
        requested_backend=requested_backend,
        strict=strict,
        training=training,
        format_conversion=format_conversion,
        dtype_conversion=plan.dtype_conversion,
        dtype_conversion_bytes=plan.dtype_conversion_bytes,
    )


__all__ = [
    "ASPY_ROUTE_SCHEMA_VERSION",
    "AsPyBackendError",
    "AsPyIFResult",
    "AsPyRoute",
]

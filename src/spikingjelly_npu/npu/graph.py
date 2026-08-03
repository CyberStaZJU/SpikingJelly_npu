"""Bounded exact-shape NPU graph routing with observable eager fallback."""

from __future__ import annotations

import copy
import enum
import struct
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.utils import _pytree


@dataclass(frozen=True)
class GraphRoute:
    backend: str
    reason: str
    captured: bool
    expected_batch_size: int | None


@dataclass(frozen=True)
class _TensorSignature:
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    layout: torch.layout
    requires_grad: bool
    stride: tuple[int, ...] | None
    storage_offset: int | None
    is_contiguous: bool | None
    contiguous_memory_formats: tuple[str, ...]


@dataclass(frozen=True)
class _StaticSignature:
    value: tuple[Any, ...]


@dataclass(frozen=True)
class _CallSignature:
    tree_spec: Any
    leaves: tuple[_TensorSignature | _StaticSignature, ...]
    tensor_storage_groups: tuple[int, ...]

    @property
    def tensor_count(self) -> int:
        return sum(isinstance(leaf, _TensorSignature) for leaf in self.leaves)

    @property
    def expected_batch_size(self) -> int | None:
        for leaf in self.leaves:
            if isinstance(leaf, _TensorSignature) and leaf.shape:
                return int(leaf.shape[0])
        return None


_TENSOR_SLOT = object()


@dataclass(frozen=True)
class _CallTemplate:
    tree_spec: Any
    leaves: tuple[Any, ...]
    tensor_positions: tuple[int, ...]

    def rebuild(self, tensors: tuple[torch.Tensor, ...]) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if len(tensors) != len(self.tensor_positions):
            raise RuntimeError("graphed call received the wrong number of tensor leaves")
        leaves = list(self.leaves)
        for position, tensor in zip(self.tensor_positions, tensors, strict=True):
            leaves[position] = tensor
        args, kwargs = _pytree.tree_unflatten(leaves, self.tree_spec)
        return tuple(args), kwargs


@dataclass(frozen=True, init=False)
class GraphBucketSpec:
    """An immutable exact call signature admitted by :class:`GraphBucketRunner`.

    ``args`` is the tuple of positional example arguments. ``kwargs`` is a mapping
    of example keyword arguments. Tensor values are discarded after their metadata
    is recorded, so mutating an example tensor cannot widen or alter the bucket.
    Use :meth:`from_call` when the model itself takes a single tuple argument.
    """

    _signature: _CallSignature
    name: str | None

    def __init__(
        self,
        args: tuple[Any, ...] | Any,
        kwargs: Mapping[str, Any] | None = None,
        *,
        name: str | None = None,
    ) -> None:
        if name is not None and (not isinstance(name, str) or not name):
            raise ValueError("bucket name must be a non-empty string or None")
        call_args = args if isinstance(args, tuple) else (args,)
        call_kwargs = {} if kwargs is None else dict(kwargs)
        signature, _, _ = _describe_call(call_args, call_kwargs)
        if signature.tensor_count == 0:
            raise ValueError("a graph bucket must contain at least one tensor leaf")
        object.__setattr__(self, "_signature", signature)
        object.__setattr__(self, "name", name)

    @classmethod
    def from_call(
        cls,
        *args: Any,
        bucket_name: str | None = None,
        **kwargs: Any,
    ) -> GraphBucketSpec:
        return cls(args, kwargs, name=bucket_name)

    @property
    def expected_batch_size(self) -> int | None:
        return self._signature.expected_batch_size

    @property
    def tensor_count(self) -> int:
        return self._signature.tensor_count


def _freeze_static_value(value: Any) -> tuple[Any, ...]:
    value_type = type(value)
    if value is None:
        return ("none",)
    if value is Ellipsis:
        return ("ellipsis",)
    if value is NotImplemented:
        return ("not-implemented",)
    if value_type is bool:
        return ("bool", value)
    if value_type is int:
        return ("int", value)
    if value_type is float:
        return ("float64", struct.pack("!d", value))
    if value_type is complex:
        return (
            "complex128",
            struct.pack("!d", value.real),
            struct.pack("!d", value.imag),
        )
    if value_type is str:
        return ("str", value)
    if value_type is bytes:
        return ("bytes", value)
    if isinstance(value, torch.device):
        return ("torch.device", value)
    if isinstance(value, torch.dtype):
        return ("torch.dtype", value)
    if isinstance(value, torch.layout):
        return ("torch.layout", value)
    if isinstance(value, torch.memory_format):
        return ("torch.memory_format", value)
    if isinstance(value, enum.Enum):
        return ("enum", type(value), value.name)
    if isinstance(value, range):
        return ("range", value.start, value.stop, value.step)
    if isinstance(value, slice):
        return (
            "slice",
            _freeze_static_value(value.start),
            _freeze_static_value(value.stop),
            _freeze_static_value(value.step),
        )
    if isinstance(value, frozenset):
        return (
            "frozenset",
            type(value),
            frozenset(_freeze_static_value(item) for item in value),
        )
    if isinstance(value, type):
        return ("type", value)
    try:
        hash(value)
    except (TypeError, ValueError):
        pass
    else:
        try:
            saved = copy.deepcopy(value)
        except Exception:
            saved = value
        return ("hashable", value_type, saved)
    raise TypeError(
        "graph bucket static leaves must be immutable scalar/configuration values; "
        f"got {type(value).__name__}"
    )


def _canonical_call_tree(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if any(not isinstance(key, str) for key in kwargs):
        raise TypeError("graph bucket keyword names must be strings")
    ordered_kwargs = {key: kwargs[key] for key in sorted(kwargs)}
    return tuple(args), ordered_kwargs


def _tensor_memory_signature(
    tensor: torch.Tensor,
) -> tuple[tuple[int, ...] | None, int | None, bool | None, tuple[str, ...]]:
    try:
        stride = tuple(int(value) for value in tensor.stride())
        storage_offset = int(tensor.storage_offset())
        is_contiguous = bool(tensor.is_contiguous())
    except (NotImplementedError, RuntimeError):
        return None, None, None, ()

    formats: list[str] = []
    for name, memory_format, expected_rank in (
        ("contiguous", torch.contiguous_format, None),
        ("channels_last", torch.channels_last, 4),
        ("channels_last_3d", torch.channels_last_3d, 5),
    ):
        if expected_rank is not None and tensor.dim() != expected_rank:
            continue
        try:
            if tensor.is_contiguous(memory_format=memory_format):
                formats.append(name)
        except (NotImplementedError, RuntimeError):
            continue
    return stride, storage_offset, is_contiguous, tuple(formats)


def _tensor_signature(tensor: torch.Tensor) -> _TensorSignature:
    stride, storage_offset, is_contiguous, memory_formats = _tensor_memory_signature(
        tensor
    )
    return _TensorSignature(
        shape=tuple(int(dimension) for dimension in tensor.shape),
        dtype=tensor.dtype,
        device=tensor.device,
        layout=tensor.layout,
        requires_grad=bool(tensor.requires_grad),
        stride=stride,
        storage_offset=storage_offset,
        is_contiguous=is_contiguous,
        contiguous_memory_formats=memory_formats,
    )


def _tensors_share_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    is_alias_of = getattr(torch._C, "_is_alias_of", None)
    if is_alias_of is not None:
        try:
            return bool(is_alias_of(left, right))
        except (NotImplementedError, RuntimeError):
            pass
    try:
        left_storage = left.untyped_storage()
        right_storage = right.untyped_storage()
    except (NotImplementedError, RuntimeError):
        return left is right
    left_identity = getattr(left_storage, "_cdata", None)
    right_identity = getattr(right_storage, "_cdata", None)
    if left_identity is not None and right_identity is not None:
        return bool(left_identity == right_identity)
    return left_storage is right_storage


def _tensor_storage_groups(tensors: tuple[torch.Tensor, ...]) -> tuple[int, ...]:
    representatives: list[torch.Tensor] = []
    groups: list[int] = []
    for tensor in tensors:
        group = next(
            (
                index
                for index, representative in enumerate(representatives)
                if _tensors_share_storage(tensor, representative)
            ),
            None,
        )
        if group is None:
            group = len(representatives)
            representatives.append(tensor)
        groups.append(group)
    return tuple(groups)


def _describe_call(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> tuple[_CallSignature, tuple[torch.Tensor, ...], _CallTemplate]:
    leaves, tree_spec = _pytree.tree_flatten(_canonical_call_tree(args, kwargs))
    signatures: list[_TensorSignature | _StaticSignature] = []
    tensors: list[torch.Tensor] = []
    template_leaves: list[Any] = []
    tensor_positions: list[int] = []
    for position, leaf in enumerate(leaves):
        if isinstance(leaf, torch.Tensor):
            signatures.append(_tensor_signature(leaf))
            tensors.append(leaf)
            template_leaves.append(_TENSOR_SLOT)
            tensor_positions.append(position)
        else:
            signatures.append(_StaticSignature(_freeze_static_value(leaf)))
            template_leaves.append(leaf)
    tensor_tuple = tuple(tensors)
    return (
        _CallSignature(
            tree_spec,
            tuple(signatures),
            _tensor_storage_groups(tensor_tuple),
        ),
        tensor_tuple,
        _CallTemplate(tree_spec, tuple(template_leaves), tuple(tensor_positions)),
    )


class _CaptureStateError(RuntimeError):
    """Capture cleanup failed, so eager execution is no longer known-safe."""


class _ForwardOnly(nn.Module):
    """Protect a canonical model from torch-npu's forward monkey patch."""

    def __init__(self, model: Any) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: torch.Tensor) -> Any:
        return self.model(batch)


class _PytreeForwardOnly(nn.Module):
    """Expose only tensor leaves to NPUGraph while rebuilding the exact call tree."""

    def __init__(self, model: Any, template: _CallTemplate) -> None:
        super().__init__()
        self.model = model
        self.template = template

    def forward(self, *tensor_leaves: torch.Tensor) -> Any:
        args, kwargs = self.template.rebuild(tuple(tensor_leaves))
        return self.model(*args, **kwargs)


@dataclass(frozen=True)
class _BufferSnapshot:
    name: str
    tensor: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class _GradientSnapshot:
    name: str
    parameter: nn.Parameter
    gradient: torch.Tensor | None
    value: torch.Tensor | None


@dataclass(frozen=True)
class _RuntimeTensorSnapshot:
    tensor: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class _RuntimeStaticSnapshot:
    value: Any


@dataclass(frozen=True)
class _RuntimeValueSnapshot:
    tree_spec: Any
    leaves: tuple[_RuntimeTensorSnapshot | _RuntimeStaticSnapshot, ...]


@dataclass(frozen=True)
class _RuntimeMappingSnapshot:
    module_name: str
    module: nn.Module
    attribute: str
    mapping: dict[str, Any]
    values: tuple[tuple[str, _RuntimeValueSnapshot], ...]


class _GraphRunnerSafety:
    model: Any
    assume_graph_safe: bool

    def _model_device_type(self) -> str | None:
        if isinstance(self.model, nn.Module):
            for parameter in self.model.parameters():
                return parameter.device.type
            for buffer in self.model.buffers():
                return buffer.device.type
        return None

    @property
    def device_type(self) -> str:
        return self._model_device_type() or "cpu"

    @property
    def enabled(self) -> bool:
        return self.device_type == "npu"

    def _execution_device_type(
        self, inputs: torch.Tensor | tuple[torch.Tensor, ...]
    ) -> str:
        if self._model_device_type() is not None:
            return self.device_type
        tensors = (inputs,) if isinstance(inputs, torch.Tensor) else inputs
        return tensors[0].device.type if tensors else "cpu"

    @property
    def backend(self) -> str:
        return "npugraph" if self.enabled else "eager"

    def _module_training_state(self) -> tuple[bool, ...]:
        if not isinstance(self.model, nn.Module):
            return ()
        return tuple(module.training for module in self.model.modules())

    def _training_capture_requested(self) -> bool:
        return any(self._module_training_state())

    @staticmethod
    def _deterministic_algorithms_enabled() -> bool:
        warn_only = getattr(
            torch, "is_deterministic_algorithms_warn_only_enabled", lambda: False
        )()
        return bool(torch.are_deterministic_algorithms_enabled() and not warn_only)

    def _deterministic_capture_state(self) -> bool | None:
        if not self._training_capture_requested():
            return None
        return self._deterministic_algorithms_enabled()

    @staticmethod
    def _tensor_data_ptr(tensor: torch.Tensor) -> int | None:
        try:
            return int(tensor.data_ptr())
        except (NotImplementedError, RuntimeError):
            return None

    @staticmethod
    def _tensor_version(tensor: torch.Tensor) -> int | None:
        try:
            return int(tensor._version)
        except (NotImplementedError, RuntimeError):
            return None

    def _module_structure_signature(
        self, *, include_versions: bool = True
    ) -> tuple[Any, ...]:
        if not isinstance(self.model, nn.Module):
            return ()
        modules = tuple(
            (name, id(module), type(module))
            for name, module in self.model.named_modules(remove_duplicate=False)
        )
        parameters = tuple(
            (
                name,
                id(parameter),
                self._tensor_data_ptr(parameter),
                self._tensor_version(parameter) if include_versions else None,
                _tensor_signature(parameter),
            )
            for name, parameter in self.model.named_parameters(remove_duplicate=False)
        )
        buffers = tuple(
            (
                name,
                id(buffer),
                self._tensor_data_ptr(buffer),
                self._tensor_version(buffer) if include_versions else None,
                _tensor_signature(buffer),
            )
            for name, buffer in self.model.named_buffers(remove_duplicate=False)
        )
        return modules, parameters, buffers

    def _has_module_hooks(self) -> bool:
        if not isinstance(self.model, nn.Module):
            return False
        return any(
            module._backward_hooks
            or module._backward_pre_hooks
            or module._forward_hooks
            or module._forward_pre_hooks
            for module in self.model.modules()
        )

    def _declares_graph_safe(self) -> bool:
        if self.assume_graph_safe:
            return True
        return isinstance(self.model, nn.Module) and bool(
            getattr(self.model, "_spikingjelly_npu_graph_safe", False)
        )

    def _rng_sensitive_training_reason(self) -> str | None:
        if not isinstance(self.model, nn.Module):
            return None
        for name, module in self.model.named_modules():
            if not module.training:
                continue
            qualified_name = name or "<root>"
            if bool(getattr(module, "_spikingjelly_npu_graph_rng_sensitive", False)):
                return f"{qualified_name} declares RNG-sensitive training"
            if "dropout" in type(module).__name__.lower():
                probability = getattr(module, "p", 0.5)
                if not isinstance(probability, int | float) or float(probability) > 0.0:
                    return f"{qualified_name} uses {type(module).__name__}"
        return None

    def _snapshot_training_modes(self) -> tuple[tuple[nn.Module, bool], ...]:
        if not isinstance(self.model, nn.Module):
            return ()
        return tuple((module, module.training) for module in self.model.modules())

    @staticmethod
    def _restore_training_modes(snapshot: tuple[tuple[nn.Module, bool], ...]) -> None:
        for module, training in snapshot:
            module.training = training

    def _snapshot_buffers(self) -> tuple[_BufferSnapshot, ...]:
        if not isinstance(self.model, nn.Module):
            return ()
        return tuple(
            _BufferSnapshot(name, buffer, buffer.detach().clone())
            for name, buffer in self.model.named_buffers(remove_duplicate=False)
        )

    def _restore_buffers(self, snapshot: tuple[_BufferSnapshot, ...]) -> None:
        if not isinstance(self.model, nn.Module):
            return
        current = dict(self.model.named_buffers(remove_duplicate=False))
        if current.keys() != {item.name for item in snapshot}:
            raise RuntimeError("model buffers changed during NPUGraph capture")
        with torch.no_grad():
            for item in snapshot:
                target = current[item.name]
                if target is not item.tensor:
                    raise RuntimeError(f"buffer {item.name!r} was replaced during capture")
                if (
                    target.shape != item.value.shape
                    or target.dtype != item.value.dtype
                    or target.device != item.value.device
                    or target.layout != item.value.layout
                ):
                    raise RuntimeError(f"buffer {item.name!r} changed during NPUGraph capture")
                target.copy_(item.value)

    def _snapshot_gradients(self) -> tuple[_GradientSnapshot, ...]:
        if not isinstance(self.model, nn.Module):
            return ()
        return tuple(
            _GradientSnapshot(
                name,
                parameter,
                parameter.grad,
                None if parameter.grad is None else parameter.grad.detach().clone(),
            )
            for name, parameter in self.model.named_parameters(remove_duplicate=False)
        )

    def _restore_gradients(self, snapshot: tuple[_GradientSnapshot, ...]) -> None:
        if not isinstance(self.model, nn.Module):
            return
        current = dict(self.model.named_parameters(remove_duplicate=False))
        if current.keys() != {item.name for item in snapshot}:
            raise RuntimeError("model parameters changed during NPUGraph capture")
        with torch.no_grad():
            for item in snapshot:
                parameter = current[item.name]
                if parameter is not item.parameter:
                    raise RuntimeError(f"parameter {item.name!r} was replaced during capture")
                if item.gradient is None:
                    parameter.grad = None
                    continue
                if item.value is None:
                    raise RuntimeError(f"gradient snapshot {item.name!r} is incomplete")
                current_gradient = parameter.grad
                if current_gradient is None:
                    parameter.grad = item.value.clone()
                    continue
                if (
                    current_gradient.shape != item.value.shape
                    or current_gradient.dtype != item.value.dtype
                    or current_gradient.device != item.value.device
                    or current_gradient.layout != item.value.layout
                ):
                    raise RuntimeError(f"gradient {item.name!r} changed during capture")
                current_gradient.copy_(item.value)

    @staticmethod
    def _snapshot_runtime_value(value: Any) -> _RuntimeValueSnapshot:
        leaves, tree_spec = _pytree.tree_flatten(value)
        snapshots: list[_RuntimeTensorSnapshot | _RuntimeStaticSnapshot] = []
        for leaf in leaves:
            if isinstance(leaf, torch.Tensor):
                snapshots.append(_RuntimeTensorSnapshot(leaf, leaf.detach().clone()))
            else:
                try:
                    snapshots.append(_RuntimeStaticSnapshot(copy.deepcopy(leaf)))
                except Exception as error:
                    raise RuntimeError(
                        "MemoryModule-style runtime state contains a value that cannot "
                        f"be safely snapshotted: {type(leaf).__name__}"
                    ) from error
        return _RuntimeValueSnapshot(tree_spec, tuple(snapshots))

    @staticmethod
    def _restore_runtime_value(snapshot: _RuntimeValueSnapshot) -> Any:
        leaves: list[Any] = []
        with torch.no_grad():
            for leaf in snapshot.leaves:
                if isinstance(leaf, _RuntimeTensorSnapshot):
                    if (
                        leaf.tensor.shape != leaf.value.shape
                        or leaf.tensor.dtype != leaf.value.dtype
                        or leaf.tensor.device != leaf.value.device
                        or leaf.tensor.layout != leaf.value.layout
                    ):
                        raise RuntimeError("runtime memory tensor changed during capture")
                    leaf.tensor.copy_(leaf.value)
                    leaves.append(leaf.tensor)
                else:
                    leaves.append(copy.deepcopy(leaf.value))
        return _pytree.tree_unflatten(leaves, snapshot.tree_spec)

    def _snapshot_runtime_memories(self) -> tuple[_RuntimeMappingSnapshot, ...]:
        if not isinstance(self.model, nn.Module):
            return ()
        snapshots: list[_RuntimeMappingSnapshot] = []
        for module_name, module in self.model.named_modules():
            has_memory_mapping = "_memories" in module.__dict__
            named_memories = getattr(module, "named_memories", None)
            if not has_memory_mapping and callable(named_memories):
                try:
                    if tuple(named_memories()):
                        raise RuntimeError(
                            f"module {module_name or '<root>'} exposes runtime memories "
                            "without a restorable _memories mapping"
                        )
                except TypeError:
                    pass
            for attribute in ("_memories", "_memories_rv"):
                if attribute not in module.__dict__:
                    continue
                mapping = module.__dict__[attribute]
                if not isinstance(mapping, dict):
                    raise RuntimeError(
                        f"module {module_name or '<root>'} has non-dict {attribute}; "
                        "runtime state cannot be safely snapshotted"
                    )
                values = tuple(
                    (name, self._snapshot_runtime_value(value))
                    for name, value in mapping.items()
                )
                snapshots.append(
                    _RuntimeMappingSnapshot(
                        module_name,
                        module,
                        attribute,
                        mapping,
                        values,
                    )
                )
        return tuple(snapshots)

    @staticmethod
    def _restore_runtime_memories(snapshot: tuple[_RuntimeMappingSnapshot, ...]) -> None:
        for item in snapshot:
            restored = {
                name: _GraphRunnerSafety._restore_runtime_value(value)
                for name, value in item.values
            }
            item.mapping.clear()
            item.mapping.update(restored)
            item.module.__dict__[item.attribute] = item.mapping

    @staticmethod
    def _capture_devices(tensors: tuple[torch.Tensor, ...]) -> tuple[torch.device, ...]:
        npu_devices: list[torch.device] = []
        for tensor in tensors:
            if tensor.device.type == "npu" and tensor.device not in npu_devices:
                npu_devices.append(tensor.device)
        if npu_devices:
            return tuple(npu_devices)
        # CPU-only tests install a fake ``torch.npu`` while preserving CPU tensors.
        # Real capture reaches this helper only after NPU routing has been selected.
        return (tensors[0].device,) if tensors else ()

    def _validate_capture_structure(self, snapshot: tuple[Any, ...]) -> None:
        if self._module_structure_signature(include_versions=False) != snapshot:
            raise RuntimeError("model structure changed during NPUGraph capture")

    def _make_graphed_callable(
        self,
        wrapper: nn.Module,
        sample_tensors: tuple[torch.Tensor, ...],
        num_warmup_iters: int,
    ) -> Any:
        if not hasattr(torch, "npu") or not hasattr(torch.npu, "make_graphed_callables"):
            raise RuntimeError("torch.npu.make_graphed_callables is unavailable")
        if self._has_module_hooks():
            raise RuntimeError("module hooks are incompatible with NPUGraph capture")

        structure_snapshot = self._module_structure_signature(include_versions=False)
        training_snapshot = self._snapshot_training_modes()
        buffer_snapshot = self._snapshot_buffers()
        gradient_snapshot = self._snapshot_gradients()
        memory_snapshot = self._snapshot_runtime_memories()
        cpu_rng_state = torch.random.get_rng_state()
        capture_devices = self._capture_devices(sample_tensors)
        npu_rng_states = tuple(
            (device, torch.npu.get_rng_state(device)) for device in capture_devices
        )
        cleanup_errors: list[tuple[str, Exception]] = []
        try:
            graphed = torch.npu.make_graphed_callables(
                wrapper,
                sample_tensors,
                num_warmup_iters=num_warmup_iters,
            )
        finally:
            cleanup_steps = (
                ("model buffers", lambda: self._restore_buffers(buffer_snapshot)),
                ("parameter gradients", lambda: self._restore_gradients(gradient_snapshot)),
                (
                    "runtime memories",
                    lambda: self._restore_runtime_memories(memory_snapshot),
                ),
                (
                    "module training state",
                    lambda: self._restore_training_modes(training_snapshot),
                ),
                ("CPU RNG", lambda: torch.random.set_rng_state(cpu_rng_state)),
            )
            for name, cleanup in cleanup_steps:
                try:
                    cleanup()
                except Exception as error:
                    cleanup_errors.append((name, error))
            for device, state in npu_rng_states:
                try:
                    torch.npu.set_rng_state(state, device)
                except Exception as error:
                    cleanup_errors.append((f"NPU RNG ({device})", error))
            try:
                self._validate_capture_structure(structure_snapshot)
            except Exception as error:
                cleanup_errors.append(("model structure", error))
            if cleanup_errors:
                failed = ", ".join(name for name, _ in cleanup_errors)
                raise _CaptureStateError(
                    f"failed to restore {failed} after NPUGraph capture"
                ) from cleanup_errors[0][1]
        return graphed


@dataclass
class _BucketCapture:
    graphed: Any | None = None
    capture_error: str | None = None
    capture_exception: Exception | None = None
    execution_state: tuple[Any, ...] | None = None


class GraphBucketRunner(_GraphRunnerSafety):
    """Capture an explicit bounded allowlist of exact PyTree call signatures."""

    def __init__(
        self,
        model: Any,
        buckets: Iterable[GraphBucketSpec] | GraphBucketSpec,
        *,
        max_buckets: int = 8,
        num_warmup_iters: int = 3,
        strict: bool = False,
        allow_training: bool = False,
        require_deterministic_training: bool = True,
        allow_unsafe_rng_training: bool = False,
        assume_graph_safe: bool = False,
    ) -> None:
        if isinstance(max_buckets, bool) or not isinstance(max_buckets, int):
            raise TypeError("max_buckets must be an integer")
        if max_buckets <= 0:
            raise ValueError("max_buckets must be positive")
        if num_warmup_iters < 0:
            raise ValueError("num_warmup_iters must be non-negative")
        normalized_buckets = (buckets,) if isinstance(buckets, GraphBucketSpec) else tuple(buckets)
        if len(normalized_buckets) > max_buckets:
            raise ValueError(
                f"received {len(normalized_buckets)} graph buckets, exceeding maximum "
                f"{max_buckets}"
            )
        if any(not isinstance(bucket, GraphBucketSpec) for bucket in normalized_buckets):
            raise TypeError("buckets must contain only GraphBucketSpec instances")
        signatures = [bucket._signature for bucket in normalized_buckets]
        if any(
            signature in signatures[:index]
            for index, signature in enumerate(signatures)
        ):
            raise ValueError("graph bucket signatures must be unique")

        self.model = model
        self.buckets = normalized_buckets
        self.max_buckets = max_buckets
        self.num_warmup_iters = int(num_warmup_iters)
        self.strict = bool(strict)
        self.allow_training = bool(allow_training)
        self.require_deterministic_training = bool(require_deterministic_training)
        self.allow_unsafe_rng_training = bool(allow_unsafe_rng_training)
        self.assume_graph_safe = bool(assume_graph_safe)
        # Each declared bucket owns at most one current capture or failed-attempt
        # record. Execution-state changes replace that slot rather than growing a
        # cache keyed by every train/eval combination.
        self._captures: dict[int, _BucketCapture] = {}
        self._capture_state_error: _CaptureStateError | None = None
        self._tracked_execution_state: tuple[Any, ...] | None = None
        self._last_capture_error: str | None = None
        self.last_route = GraphRoute("eager", "not called", False, None)

    @property
    def capture_error(self) -> str | None:
        return self._last_capture_error

    @property
    def capture_errors(self) -> tuple[tuple[int, tuple[bool, ...], str], ...]:
        errors: list[tuple[int, tuple[bool, ...], str]] = []
        for bucket_index, state in self._captures.items():
            if state.capture_error is None:
                continue
            training_state: tuple[bool, ...] = ()
            if state.execution_state is not None:
                training_state = state.execution_state[0]
            errors.append((bucket_index, training_state, state.capture_error))
        return tuple(errors)

    def reset_capture(self) -> None:
        self._captures.clear()
        self._tracked_execution_state = None
        self._last_capture_error = None

    def _execution_state(self) -> tuple[Any, ...]:
        return (
            self._module_training_state(),
            self._deterministic_capture_state(),
            self._module_structure_signature(),
        )

    def _bucket_label(self, index: int) -> str:
        name = self.buckets[index].name
        return f"{name!r}" if name is not None else str(index)

    def _route_fallback_or_raise(
        self,
        reason: str,
        expected_batch_size: int | None,
    ) -> None:
        self.last_route = GraphRoute("eager", reason, False, expected_batch_size)
        if self.strict:
            raise RuntimeError(reason)

    def _unknown_signature(self, detail: str | None = None) -> None:
        reason = "call signature is not in the exact graph bucket allowlist"
        if detail is not None:
            reason = f"{reason}: {detail}"
        self._route_fallback_or_raise(reason, None)

    def _capture(
        self,
        template: _CallTemplate,
        sample_tensors: tuple[torch.Tensor, ...],
    ) -> Any:
        wrapper = _PytreeForwardOnly(self.model, template)
        return self._make_graphed_callable(
            wrapper,
            sample_tensors,
            self.num_warmup_iters,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._capture_state_error is not None:
            raise self._capture_state_error
        try:
            call_signature, tensor_leaves, template = _describe_call(tuple(args), kwargs)
        except (TypeError, ValueError) as error:
            self._unknown_signature(str(error))
            return self.model(*args, **kwargs)

        bucket_index = next(
            (
                index
                for index, bucket in enumerate(self.buckets)
                if bucket._signature == call_signature
            ),
            None,
        )
        if bucket_index is None:
            self._unknown_signature()
            return self.model(*args, **kwargs)

        bucket = self.buckets[bucket_index]
        expected_batch_size = bucket.expected_batch_size
        fallback_reason: str | None = None
        if self._execution_device_type(tensor_leaves) != "npu":
            fallback_reason = "model is not on an NPU"
        elif not self._declares_graph_safe():
            fallback_reason = (
                "model does not declare graph-safe per-forward state; "
                "set assume_graph_safe=True only after qualification"
            )
        else:
            training_capture = self._training_capture_requested()
            if training_capture and not self.allow_training:
                fallback_reason = (
                    "training NPUGraph requires explicit allow_training=True "
                    "after parity qualification"
                )
            elif (
                training_capture
                and self.require_deterministic_training
                and not self._deterministic_algorithms_enabled()
            ):
                fallback_reason = (
                    "training NPUGraph requires "
                    "torch.use_deterministic_algorithms(True, warn_only=False); "
                    "set require_deterministic_training=False only after independent "
                    "parity qualification"
                )
            else:
                rng_reason = (
                    self._rng_sensitive_training_reason() if training_capture else None
                )
                if rng_reason is not None and not self.allow_unsafe_rng_training:
                    fallback_reason = (
                        f"RNG-sensitive training capture rejected: {rng_reason}; set "
                        "allow_unsafe_rng_training=True only for an independently "
                        "qualified unsafe path"
                    )
                elif self._has_module_hooks():
                    fallback_reason = "module hooks are incompatible with NPUGraph"
        if fallback_reason is not None:
            self._route_fallback_or_raise(fallback_reason, expected_batch_size)
            return self.model(*args, **kwargs)

        execution_state = self._execution_state()
        if self._tracked_execution_state != execution_state:
            self._captures.clear()
            self._last_capture_error = None
            self._tracked_execution_state = execution_state
        state = self._captures.get(bucket_index)
        if state is None:
            state = _BucketCapture(execution_state=execution_state)
        if state.capture_error is not None:
            self._last_capture_error = state.capture_error
            reason = (
                f"prior capture failed for bucket {self._bucket_label(bucket_index)}: "
                f"{state.capture_error}"
            )
            self.last_route = GraphRoute("eager", reason, False, expected_batch_size)
            if self.strict:
                raise RuntimeError(reason) from state.capture_exception
            return self.model(*args, **kwargs)

        if state.graphed is None:
            try:
                state.graphed = self._capture(template, tensor_leaves)
            except _CaptureStateError as error:
                self._capture_state_error = error
                raise
            except Exception as error:
                state.capture_error = f"{type(error).__name__}: {error}"
                state.capture_exception = error
                self._captures[bucket_index] = state
                self._last_capture_error = state.capture_error
                if self.strict:
                    raise
                warnings.warn(
                    "NPUGraph capture failed for bucket "
                    f"{self._bucket_label(bucket_index)}; using eager mode: "
                    f"{state.capture_error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.last_route = GraphRoute(
                    "eager",
                    f"capture failed for bucket {self._bucket_label(bucket_index)}: "
                    f"{state.capture_error}",
                    False,
                    expected_batch_size,
                )
                return self.model(*args, **kwargs)
            state.execution_state = self._execution_state()
            self._tracked_execution_state = state.execution_state
            self._captures[bucket_index] = state

        self._last_capture_error = None
        self.last_route = GraphRoute(
            "npugraph",
            f"exact graph bucket {self._bucket_label(bucket_index)} replay",
            True,
            expected_batch_size,
        )
        try:
            return state.graphed(*tensor_leaves)
        except Exception as error:
            poisoned = _CaptureStateError(
                "NPUGraph replay failed after launch; runner is poisoned and will not "
                "execute eager fallback"
            )
            self._capture_state_error = poisoned
            raise poisoned from error


class StaticGraphRunner(_GraphRunnerSafety):
    """Capture one fixed full-batch path and leave all other calls eager.

    This source-compatible facade preserves the original tensor-only call policy.
    Its first qualified full-batch call binds the one exact input signature; partial
    batches and diagnostic arguments remain eager. Training capture is disabled by
    default and deterministic algorithms are required by default when it is enabled.
    """

    def __init__(
        self,
        model: Any,
        batch_size: int,
        *,
        num_warmup_iters: int = 3,
        strict: bool = False,
        allow_training: bool = False,
        require_deterministic_training: bool = True,
        assume_graph_safe: bool = False,
        allow_unsafe_rng_training: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if num_warmup_iters < 0:
            raise ValueError("num_warmup_iters must be non-negative")
        self.model = model
        self.batch_size = int(batch_size)
        self.num_warmup_iters = int(num_warmup_iters)
        self.strict = bool(strict)
        self.allow_training = bool(allow_training)
        self.require_deterministic_training = bool(require_deterministic_training)
        self.assume_graph_safe = bool(assume_graph_safe)
        self.allow_unsafe_rng_training = bool(allow_unsafe_rng_training)
        self._graphed: nn.Module | None = None
        self._capture_error: str | None = None
        self._capture_state_error: _CaptureStateError | None = None
        self._captured_training_state: tuple[bool, ...] | None = None
        self._captured_deterministic_state: bool | None = None
        self._captured_structure_signature: tuple[Any, ...] | None = None
        self._capture_signature: tuple[Any, ...] | None = None
        self.last_route = GraphRoute("eager", "not called", False, self.batch_size)

    @property
    def capture_error(self) -> str | None:
        return self._capture_error

    def reset_capture(self) -> None:
        self._graphed = None
        self._capture_error = None
        self._captured_training_state = None
        self._captured_deterministic_state = None
        self._captured_structure_signature = None
        self._capture_signature = None

    @staticmethod
    def _input_signature(inputs: torch.Tensor) -> tuple[Any, ...]:
        # Preserve the legacy StaticGraphRunner signature contract. Exact stride,
        # storage-offset, memory-format, and alias checks are GraphBucketRunner-only.
        return (
            tuple(inputs.shape),
            inputs.dtype,
            inputs.device,
            inputs.requires_grad,
            inputs.layout,
        )

    def _capture(self, sample: torch.Tensor) -> nn.Module:
        wrapper = _ForwardOnly(self.model)
        graphed = self._make_graphed_callable(
            wrapper,
            (sample,),
            self.num_warmup_iters,
        )
        self._captured_training_state = self._module_training_state()
        self._captured_deterministic_state = self._deterministic_capture_state()
        self._captured_structure_signature = self._module_structure_signature()
        self._capture_signature = self._input_signature(sample)
        return graphed

    def __call__(self, inputs: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        if self._capture_state_error is not None:
            raise self._capture_state_error
        if args or kwargs:
            self.last_route = GraphRoute(
                "eager",
                "graph supports tensor-only ordinary forward",
                False,
                self.batch_size,
            )
            return self.model(inputs, *args, **kwargs)
        if int(inputs.shape[0]) != self.batch_size:
            self.last_route = GraphRoute(
                "eager",
                "batch shape does not match static capture bucket",
                False,
                self.batch_size,
            )
            return self.model(inputs)
        if self._execution_device_type(inputs) != "npu":
            self.last_route = GraphRoute(
                "eager", "model is not on an NPU", False, self.batch_size
            )
            return self.model(inputs)
        if not self._declares_graph_safe():
            self.last_route = GraphRoute(
                "eager",
                "model does not declare graph-safe per-forward state; "
                "set assume_graph_safe=True only after qualification",
                False,
                self.batch_size,
            )
            return self.model(inputs)
        training_capture = self._training_capture_requested()
        if training_capture and not self.allow_training:
            self.last_route = GraphRoute(
                "eager",
                "training NPUGraph requires explicit allow_training=True "
                "after parity qualification",
                False,
                self.batch_size,
            )
            return self.model(inputs)
        if (
            training_capture
            and self.require_deterministic_training
            and not self._deterministic_algorithms_enabled()
        ):
            self.last_route = GraphRoute(
                "eager",
                "training NPUGraph requires "
                "torch.use_deterministic_algorithms(True, warn_only=False); "
                "set require_deterministic_training=False only after independent parity "
                "qualification",
                False,
                self.batch_size,
            )
            return self.model(inputs)
        rng_reason = self._rng_sensitive_training_reason() if training_capture else None
        if rng_reason is not None and not self.allow_unsafe_rng_training:
            self.last_route = GraphRoute(
                "eager",
                f"RNG-sensitive training capture rejected: {rng_reason}; set "
                "allow_unsafe_rng_training=True only for an independently qualified "
                "unsafe path",
                False,
                self.batch_size,
            )
            return self.model(inputs)
        if self._capture_error is not None:
            self.last_route = GraphRoute(
                "eager",
                f"prior capture failed: {self._capture_error}",
                False,
                self.batch_size,
            )
            return self.model(inputs)
        if self._has_module_hooks():
            self.last_route = GraphRoute(
                "eager",
                "module hooks are incompatible with NPUGraph",
                False,
                self.batch_size,
            )
            return self.model(inputs)
        if self._graphed is not None and (
            self._captured_training_state != self._module_training_state()
            or self._captured_deterministic_state != self._deterministic_capture_state()
            or self._captured_structure_signature != self._module_structure_signature()
        ):
            self.reset_capture()
        if self._graphed is not None and self._capture_signature != self._input_signature(inputs):
            self.last_route = GraphRoute(
                "eager",
                "input signature does not match static capture",
                False,
                self.batch_size,
            )
            return self.model(inputs)
        if self._graphed is None:
            try:
                self._graphed = self._capture(inputs)
            except _CaptureStateError as error:
                self._capture_state_error = error
                raise
            except Exception as error:
                self._capture_error = f"{type(error).__name__}: {error}"
                if self.strict:
                    raise
                warnings.warn(
                    f"NPUGraph capture failed; using eager mode: {self._capture_error}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.last_route = GraphRoute(
                    "eager",
                    f"capture failed: {self._capture_error}",
                    False,
                    self.batch_size,
                )
                return self.model(inputs)
        self.last_route = GraphRoute(
            "npugraph", "static full-batch replay", True, self.batch_size
        )
        try:
            return self._graphed(inputs)
        except Exception as error:
            poisoned = _CaptureStateError(
                "NPUGraph replay failed after launch; runner is poisoned and will not "
                "execute eager fallback"
            )
            self._capture_state_error = poisoned
            raise poisoned from error


__all__ = [
    "GraphBucketRunner",
    "GraphBucketSpec",
    "GraphRoute",
    "StaticGraphRunner",
]

"""Bounded exact-shape NPU graph routing with observable eager fallback."""

from __future__ import annotations

import copy
import enum
import importlib
import struct
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


class GraphPreExecutionError(RuntimeError):
    """A strict graph request was rejected before capture or replay launched."""

    def __init__(self, route: GraphRoute) -> None:
        if route.backend != "eager" or route.captured:
            raise ValueError("GraphPreExecutionError requires a pre-execution eager route")
        self.route = route
        super().__init__(route.reason)


class _PhysicalFormatInspectionError(RuntimeError):
    """Ascend physical format could not be inspected before graph execution."""


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
    physical_device_format: int | None


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


def _exact_call_tree(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if any(not isinstance(key, str) for key in kwargs):
        raise TypeError("graph bucket keyword names must be strings")
    return tuple(args), dict(kwargs)


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


def _physical_device_format(tensor: torch.Tensor) -> int | None:
    """Return the Ascend physical format without changing CPU import safety.

    NPUGraph captures are physical-layout specific. Logical shape, stride, and
    PyTorch memory-format metadata do not distinguish base ND tensors from
    internal formats such as FRACTAL_NZ or NCDHW, so NPU signatures must include
    the runtime-reported format before bucket selection.
    """

    if tensor.device.type != "npu":
        return None
    try:
        torch_npu = importlib.import_module("torch_npu")
    except (ImportError, OSError) as error:
        raise _PhysicalFormatInspectionError(
            "cannot import torch-npu to inspect the Ascend physical format required "
            f"for exact NPUGraph matching: {error}"
        ) from error
    get_format = getattr(torch_npu, "get_npu_format", None)
    if not callable(get_format):
        npu_ops = getattr(torch.ops, "npu", None)
        get_format = None if npu_ops is None else getattr(npu_ops, "get_npu_format", None)
    if not callable(get_format):
        raise _PhysicalFormatInspectionError(
            "torch-npu does not expose get_npu_format for exact NPUGraph matching"
        )
    try:
        return int(get_format(tensor))
    except Exception as error:
        raise _PhysicalFormatInspectionError(
            "failed to inspect the Ascend physical format required for exact "
            f"NPUGraph matching: {error}"
        ) from error


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
        physical_device_format=_physical_device_format(tensor),
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
    leaves, tree_spec = _pytree.tree_flatten(_exact_call_tree(args, kwargs))
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
class _ParameterSnapshot:
    name: str
    parameter: nn.Parameter
    data_ptr: int | None
    version: int | None
    value: torch.Tensor


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

    def _finalize_module_structure_after_capture(
        self,
        signature: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        """Validate capture state without querying physical formats after launch.

        Parameter identity, storage, and version must remain exactly unchanged.
        Buffers are restored after warmup, so their versions may advance during
        restoration; only their identity and storage are revalidated before the
        restored version is recorded. Logical and physical tensor metadata remain
        the authoritative pre-launch snapshot.
        """

        if not isinstance(self.model, nn.Module):
            if signature:
                raise RuntimeError("non-module graph target gained module structure")
            return ()
        if not signature:
            raise RuntimeError("module graph target lost its structure signature")
        modules, parameters, buffers = signature
        current_modules = tuple(
            (name, id(module), type(module))
            for name, module in self.model.named_modules(remove_duplicate=False)
        )
        if current_modules != modules:
            raise RuntimeError("model modules changed during NPUGraph capture")

        def finalize_parameters(
            entries: tuple[Any, ...],
            current: tuple[tuple[str, torch.Tensor], ...],
        ) -> tuple[Any, ...]:
            if len(entries) != len(current):
                raise RuntimeError("model parameters changed during NPUGraph capture")
            finalized = []
            for saved, (current_name, tensor) in zip(entries, current, strict=True):
                name, identity, data_ptr, version, tensor_signature = saved
                current_version = self._tensor_version(tensor)
                if (
                    current_name != name
                    or id(tensor) != identity
                    or self._tensor_data_ptr(tensor) != data_ptr
                    or current_version != version
                ):
                    raise RuntimeError(
                        f"model parameter {name!r} changed during NPUGraph capture"
                    )
                finalized.append(saved)
            return tuple(finalized)

        def finalize_buffers(
            entries: tuple[Any, ...],
            current: tuple[tuple[str, torch.Tensor], ...],
        ) -> tuple[Any, ...]:
            if len(entries) != len(current):
                raise RuntimeError("model buffers changed during NPUGraph capture")
            finalized = []
            for saved, (current_name, tensor) in zip(entries, current, strict=True):
                name, identity, data_ptr, _version, tensor_signature = saved
                if (
                    current_name != name
                    or id(tensor) != identity
                    or self._tensor_data_ptr(tensor) != data_ptr
                ):
                    raise RuntimeError(
                        f"model buffer {name!r} changed during NPUGraph capture"
                    )
                finalized.append(
                    (
                        name,
                        identity,
                        data_ptr,
                        self._tensor_version(tensor),
                        tensor_signature,
                    )
                )
            return tuple(finalized)

        current_parameters = tuple(
            self.model.named_parameters(remove_duplicate=False)
        )
        current_buffers = tuple(self.model.named_buffers(remove_duplicate=False))
        return (
            modules,
            finalize_parameters(parameters, current_parameters),
            finalize_buffers(buffers, current_buffers),
        )

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

    @staticmethod
    def _parameter_value_snapshot(parameter: nn.Parameter) -> torch.Tensor:
        return parameter.detach().to(device="cpu", copy=True).contiguous()

    @staticmethod
    def _parameter_values_identical(
        parameter: nn.Parameter, snapshot: torch.Tensor
    ) -> bool:
        current = _GraphRunnerSafety._parameter_value_snapshot(parameter)
        if current.shape != snapshot.shape or current.dtype != snapshot.dtype:
            return False
        current_bytes = current.reshape(-1).view(torch.uint8)
        snapshot_bytes = snapshot.reshape(-1).view(torch.uint8)
        return bool(torch.equal(current_bytes, snapshot_bytes))

    def _snapshot_parameters(self) -> tuple[_ParameterSnapshot, ...]:
        if not isinstance(self.model, nn.Module):
            return ()
        return tuple(
            _ParameterSnapshot(
                name,
                parameter,
                self._tensor_data_ptr(parameter),
                self._tensor_version(parameter),
                self._parameter_value_snapshot(parameter),
            )
            for name, parameter in self.model.named_parameters(remove_duplicate=False)
        )

    def _restore_parameters(self, snapshot: tuple[_ParameterSnapshot, ...]) -> None:
        if not isinstance(self.model, nn.Module):
            if snapshot:
                raise RuntimeError("non-module graph target gained parameters")
            return
        current = tuple(self.model.named_parameters(remove_duplicate=False))
        if len(current) != len(snapshot):
            raise RuntimeError("model parameters changed during NPUGraph capture")
        changes: list[str] = []
        for item, (current_name, parameter) in zip(snapshot, current, strict=True):
            same_object = current_name == item.name and parameter is item.parameter
            same_storage = self._tensor_data_ptr(parameter) == item.data_ptr
            same_metadata = (
                parameter.shape == item.value.shape
                and parameter.dtype == item.value.dtype
                and parameter.layout == torch.strided
            )
            same_version = self._tensor_version(parameter) == item.version
            same_value = False
            if same_object and same_storage and same_metadata:
                same_value = self._parameter_values_identical(parameter, item.value)
                if not same_value:
                    with torch.no_grad():
                        parameter.copy_(item.value)
            if not (
                same_object
                and same_storage
                and same_metadata
                and same_version
                and same_value
            ):
                changes.append(item.name)
        if changes:
            joined = ", ".join(repr(name) for name in changes)
            raise RuntimeError(
                f"model parameters changed during NPUGraph capture: {joined}"
            )

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

    def _make_graphed_callable(
        self,
        wrapper: nn.Module,
        sample_tensors: tuple[torch.Tensor, ...],
        num_warmup_iters: int,
        *,
        structure_snapshot: tuple[Any, ...],
    ) -> tuple[Any, tuple[Any, ...]]:
        if not hasattr(torch, "npu") or not hasattr(torch.npu, "make_graphed_callables"):
            raise RuntimeError("torch.npu.make_graphed_callables is unavailable")
        if self._has_module_hooks():
            raise RuntimeError("module hooks are incompatible with NPUGraph capture")

        parameter_snapshot = self._snapshot_parameters()
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
        except Exception as error:
            capture_error: Exception | None = error
            graphed = None
        else:
            capture_error = None
        finally:
            cleanup_steps = (
                (
                    "model parameters",
                    lambda: self._restore_parameters(parameter_snapshot),
                ),
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
                finalized_structure = self._finalize_module_structure_after_capture(
                    structure_snapshot
                )
            except Exception as error:
                finalized_structure = ()
                cleanup_errors.append(("model structure", error))
        if cleanup_errors:
            failed = ", ".join(name for name, _ in cleanup_errors)
            raise _CaptureStateError(
                f"failed to restore {failed} after NPUGraph capture"
            ) from cleanup_errors[0][1]
        if capture_error is not None:
            raise _CaptureStateError(
                "NPUGraph capture failed after launch; runner is poisoned and will not "
                "execute eager fallback"
            ) from capture_error
        if graphed is None:
            raise _CaptureStateError(
                "NPUGraph capture returned no callable after launch; runner is poisoned"
            )
        return graphed, finalized_structure


@dataclass
class _BucketCapture:
    graphed: Any | None = None
    capture_error: str | None = None
    capture_exception: Exception | None = None
    execution_state: tuple[Any, ...] | None = None


@dataclass(frozen=True)
class _StaticCapturePreflight:
    input_signature: tuple[Any, ...]
    training_state: tuple[bool, ...]
    deterministic_state: bool | None
    structure_signature: tuple[Any, ...]


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
            raise GraphPreExecutionError(self.last_route)

    def _unknown_signature(self, detail: str | None = None) -> None:
        reason = "call signature is not in the exact graph bucket allowlist"
        if detail is not None:
            reason = f"{reason}: {detail}"
        self._route_fallback_or_raise(reason, None)

    def _capture(
        self,
        template: _CallTemplate,
        sample_tensors: tuple[torch.Tensor, ...],
        structure_snapshot: tuple[Any, ...],
    ) -> tuple[Any, tuple[Any, ...]]:
        wrapper = _PytreeForwardOnly(self.model, template)
        return self._make_graphed_callable(
            wrapper,
            sample_tensors,
            self.num_warmup_iters,
            structure_snapshot=structure_snapshot,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._capture_state_error is not None:
            raise self._capture_state_error
        try:
            call_signature, tensor_leaves, template = _describe_call(tuple(args), kwargs)
        except _PhysicalFormatInspectionError as error:
            self._route_fallback_or_raise(str(error), None)
            return self.model(*args, **kwargs)
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

        try:
            execution_state = self._execution_state()
        except _PhysicalFormatInspectionError as error:
            self._route_fallback_or_raise(str(error), expected_batch_size)
            return self.model(*args, **kwargs)
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
                raise GraphPreExecutionError(self.last_route) from state.capture_exception
            return self.model(*args, **kwargs)

        if state.graphed is None:
            try:
                state.graphed, finalized_structure = self._capture(
                    template,
                    tensor_leaves,
                    execution_state[2],
                )
            except _CaptureStateError as error:
                self._capture_state_error = error
                raise
            state.execution_state = (
                execution_state[0],
                execution_state[1],
                finalized_structure,
            )
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
    """Capture one fixed full-batch path with observable pre-execution routing.

    This source-compatible facade preserves the original tensor-only call policy.
    Its first qualified full-batch call binds the one exact input signature. In
    non-strict mode, partial batches, diagnostic arguments, and other known
    pre-capture rejections remain eager. In strict mode, every such rejection raises
    :class:`GraphPreExecutionError` before eager execution. Training capture is
    disabled by default and deterministic algorithms are required by default when
    it is enabled.
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
        self._pending_capture_preflight: _StaticCapturePreflight | None = None
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
        self._pending_capture_preflight = None

    @staticmethod
    def _input_signature(inputs: torch.Tensor) -> tuple[Any, ...]:
        # Preserve the legacy logical StaticGraphRunner signature contract while
        # preventing replay across different Ascend physical storage formats.
        return (
            tuple(inputs.shape),
            inputs.dtype,
            inputs.device,
            inputs.requires_grad,
            inputs.layout,
            _physical_device_format(inputs),
        )

    def _capture_preflight(self, sample: torch.Tensor) -> _StaticCapturePreflight:
        return _StaticCapturePreflight(
            input_signature=self._input_signature(sample),
            training_state=self._module_training_state(),
            deterministic_state=self._deterministic_capture_state(),
            structure_signature=self._module_structure_signature(),
        )

    def _capture(self, sample: torch.Tensor) -> nn.Module:
        # Every decision-capable signature and physical-format query happens before
        # ``make_graphed_callables``. Post-launch cleanup uses only object/storage/
        # version checks plus restoration snapshots; it never re-queries format.
        preflight = self._pending_capture_preflight
        if preflight is None:
            preflight = self._capture_preflight(sample)
        wrapper = _ForwardOnly(self.model)
        graphed, captured_structure_signature = self._make_graphed_callable(
            wrapper,
            (sample,),
            self.num_warmup_iters,
            structure_snapshot=preflight.structure_signature,
        )
        self._captured_training_state = preflight.training_state
        self._captured_deterministic_state = preflight.deterministic_state
        self._captured_structure_signature = captured_structure_signature
        self._capture_signature = preflight.input_signature
        return graphed

    def _route_fallback_or_raise(self, reason: str) -> None:
        self.last_route = GraphRoute("eager", reason, False, self.batch_size)
        if self.strict:
            raise GraphPreExecutionError(self.last_route)

    def __call__(self, inputs: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        if self._capture_state_error is not None:
            raise self._capture_state_error
        if args or kwargs:
            self._route_fallback_or_raise("graph supports tensor-only ordinary forward")
            return self.model(inputs, *args, **kwargs)
        if int(inputs.shape[0]) != self.batch_size:
            self._route_fallback_or_raise(
                "batch shape does not match static capture bucket"
            )
            return self.model(inputs)
        if self._execution_device_type(inputs) != "npu":
            self._route_fallback_or_raise("model is not on an NPU")
            return self.model(inputs)
        if not self._declares_graph_safe():
            self._route_fallback_or_raise(
                "model does not declare graph-safe per-forward state; "
                "set assume_graph_safe=True only after qualification"
            )
            return self.model(inputs)
        training_capture = self._training_capture_requested()
        if training_capture and not self.allow_training:
            self._route_fallback_or_raise(
                "training NPUGraph requires explicit allow_training=True "
                "after parity qualification"
            )
            return self.model(inputs)
        if (
            training_capture
            and self.require_deterministic_training
            and not self._deterministic_algorithms_enabled()
        ):
            self._route_fallback_or_raise(
                "training NPUGraph requires "
                "torch.use_deterministic_algorithms(True, warn_only=False); "
                "set require_deterministic_training=False only after independent parity "
                "qualification"
            )
            return self.model(inputs)
        rng_reason = self._rng_sensitive_training_reason() if training_capture else None
        if rng_reason is not None and not self.allow_unsafe_rng_training:
            self._route_fallback_or_raise(
                f"RNG-sensitive training capture rejected: {rng_reason}; set "
                "allow_unsafe_rng_training=True only for an independently qualified "
                "unsafe path"
            )
            return self.model(inputs)
        if self._has_module_hooks():
            self._route_fallback_or_raise(
                "module hooks are incompatible with NPUGraph"
            )
            return self.model(inputs)
        if self._capture_error is not None:
            self._route_fallback_or_raise(
                f"prior capture failed: {self._capture_error}"
            )
            return self.model(inputs)
        try:
            preflight = self._capture_preflight(inputs)
        except _PhysicalFormatInspectionError as error:
            self._route_fallback_or_raise(str(error))
            return self.model(inputs)
        execution_state_changed = self._graphed is not None and (
            self._captured_training_state != preflight.training_state
            or self._captured_deterministic_state != preflight.deterministic_state
            or self._captured_structure_signature != preflight.structure_signature
        )
        if execution_state_changed:
            self.reset_capture()
        elif (
            self._graphed is not None
            and self._capture_signature != preflight.input_signature
        ):
            self._route_fallback_or_raise(
                "input signature does not match static capture"
            )
            return self.model(inputs)
        if self._graphed is None:
            self._pending_capture_preflight = preflight
            try:
                self._graphed = self._capture(inputs)
            except _CaptureStateError as error:
                self._capture_state_error = error
                raise
            finally:
                self._pending_capture_preflight = None
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
    "GraphPreExecutionError",
    "GraphRoute",
    "StaticGraphRunner",
]

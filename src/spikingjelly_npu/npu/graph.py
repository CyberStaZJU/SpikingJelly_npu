"""Static-shape NPU graph routing for repeated FedSNN workloads."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class GraphRoute:
    backend: str
    reason: str
    captured: bool
    expected_batch_size: int


class _CaptureStateError(RuntimeError):
    """Capture cleanup failed, so eager execution is no longer known-safe."""


class _ForwardOnly(nn.Module):
    """Protect a canonical model from torch-npu's forward monkey patch."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, batch: torch.Tensor) -> Any:
        return self.model(batch)


class StaticGraphRunner:
    """Capture one fixed full-batch path and leave all other calls eager.

    This policy fits federated workloads: common full batches amortize graph
    capture while client-specific remainder batches and diagnostic keyword
    arguments retain ordinary eager semantics. Only models that declare
    graph-safe per-forward state are captured unless ``assume_graph_safe=True``
    is explicitly supplied. Training capture is disabled by default. When it is
    explicitly enabled, deterministic algorithms are required by default because
    small nondeterministic kernel differences can cross hard spike thresholds.
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
        self._graphed: nn.Module | None = None
        self._capture_error: str | None = None
        self._capture_state_error: _CaptureStateError | None = None
        self._captured_training_state: tuple[bool, ...] | None = None
        self._captured_deterministic_state: bool | None = None
        self._captured_structure_signature: tuple[Any, ...] | None = None
        self._capture_signature: tuple[Any, ...] | None = None
        self.last_route = GraphRoute("eager", "not called", False, self.batch_size)

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

    def _execution_device_type(self, inputs: torch.Tensor) -> str:
        if self._model_device_type() is None:
            return inputs.device.type
        return self.device_type

    @property
    def backend(self) -> str:
        return "npugraph" if self.enabled else "eager"

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
        return (
            tuple(inputs.shape),
            inputs.dtype,
            inputs.device,
            inputs.requires_grad,
            inputs.layout,
        )

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

    def _module_structure_signature(self) -> tuple[Any, ...]:
        if not isinstance(self.model, nn.Module):
            return ()
        parameters = tuple(
            (
                name,
                id(parameter),
                parameter.data_ptr(),
                tuple(parameter.shape),
                parameter.dtype,
                parameter.device,
                parameter.requires_grad,
                parameter.layout,
            )
            for name, parameter in self.model.named_parameters()
        )
        buffers = tuple(
            (
                name,
                id(buffer),
                buffer.data_ptr(),
                tuple(buffer.shape),
                buffer.dtype,
                buffer.device,
                buffer.requires_grad,
                buffer.layout,
            )
            for name, buffer in self.model.named_buffers()
        )
        return parameters, buffers

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

    def _snapshot_buffers(self) -> tuple[tuple[str, torch.Tensor], ...]:
        if not isinstance(self.model, nn.Module):
            return ()
        return tuple(
            (name, buffer.detach().clone())
            for name, buffer in self.model.named_buffers()
        )

    def _restore_buffers(self, snapshot: tuple[tuple[str, torch.Tensor], ...]) -> None:
        if not isinstance(self.model, nn.Module):
            return
        current = dict(self.model.named_buffers())
        if current.keys() != dict(snapshot).keys():
            raise RuntimeError("model buffers changed during NPUGraph capture")
        with torch.no_grad():
            for name, saved in snapshot:
                target = current[name]
                if (
                    target.shape != saved.shape
                    or target.dtype != saved.dtype
                    or target.device != saved.device
                ):
                    raise RuntimeError(f"buffer {name!r} changed during NPUGraph capture")
                target.copy_(saved)

    def _snapshot_gradients(self) -> tuple[tuple[str, torch.Tensor | None], ...]:
        if not isinstance(self.model, nn.Module):
            return ()
        return tuple(
            (
                name,
                None if parameter.grad is None else parameter.grad.detach().clone(),
            )
            for name, parameter in self.model.named_parameters()
        )

    def _restore_gradients(
        self, snapshot: tuple[tuple[str, torch.Tensor | None], ...]
    ) -> None:
        if not isinstance(self.model, nn.Module):
            return
        current = dict(self.model.named_parameters())
        if current.keys() != dict(snapshot).keys():
            raise RuntimeError("model parameters changed during NPUGraph capture")
        for name, saved in snapshot:
            parameter = current[name]
            if saved is None:
                parameter.grad = None
                continue
            if parameter.grad is None:
                parameter.grad = saved.clone()
                continue
            if (
                parameter.grad.shape != saved.shape
                or parameter.grad.dtype != saved.dtype
                or parameter.grad.device != saved.device
            ):
                raise RuntimeError(f"gradient {name!r} changed during NPUGraph capture")
            parameter.grad.copy_(saved)

    def _capture(self, sample: torch.Tensor) -> nn.Module:
        if not hasattr(torch, "npu") or not hasattr(torch.npu, "make_graphed_callables"):
            raise RuntimeError("torch.npu.make_graphed_callables is unavailable")
        if self._has_module_hooks():
            raise RuntimeError("module hooks are incompatible with NPUGraph capture")
        wrapper = _ForwardOnly(self.model)
        buffer_snapshot = self._snapshot_buffers()
        gradient_snapshot = self._snapshot_gradients()
        cpu_rng_state = torch.random.get_rng_state()
        npu_rng_state = torch.npu.get_rng_state(sample.device)
        cleanup_errors: list[tuple[str, Exception]] = []
        try:
            graphed = torch.npu.make_graphed_callables(
                wrapper,
                (sample,),
                num_warmup_iters=self.num_warmup_iters,
            )
        finally:
            try:
                self._restore_buffers(buffer_snapshot)
            except Exception as error:
                cleanup_errors.append(("model buffers", error))
            try:
                self._restore_gradients(gradient_snapshot)
            except Exception as error:
                cleanup_errors.append(("parameter gradients", error))
            try:
                torch.random.set_rng_state(cpu_rng_state)
            except Exception as error:
                cleanup_errors.append(("CPU RNG", error))
            try:
                torch.npu.set_rng_state(npu_rng_state, sample.device)
            except Exception as error:
                cleanup_errors.append(("NPU RNG", error))
            if cleanup_errors:
                failed = ", ".join(name for name, _ in cleanup_errors)
                raise _CaptureStateError(
                    f"failed to restore {failed} after NPUGraph capture"
                ) from cleanup_errors[0][1]
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
                "eager", "graph supports tensor-only ordinary forward", False, self.batch_size
            )
            return self.model(inputs, *args, **kwargs)
        if int(inputs.shape[0]) != self.batch_size:
            self.last_route = GraphRoute(
                "eager", "batch shape does not match static capture bucket", False, self.batch_size
            )
            return self.model(inputs)
        if self._execution_device_type(inputs) != "npu":
            self.last_route = GraphRoute("eager", "model is not on an NPU", False, self.batch_size)
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
        if self._capture_error is not None:
            self.last_route = GraphRoute(
                "eager", f"prior capture failed: {self._capture_error}", False, self.batch_size
            )
            return self.model(inputs)
        if self._has_module_hooks():
            self.last_route = GraphRoute(
                "eager", "module hooks are incompatible with NPUGraph", False, self.batch_size
            )
            return self.model(inputs)
        if self._graphed is not None and (
            self._captured_training_state != self._module_training_state()
            or self._captured_deterministic_state
            != self._deterministic_capture_state()
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
                    "eager", f"capture failed: {self._capture_error}", False, self.batch_size
                )
                return self.model(inputs)
        self.last_route = GraphRoute("npugraph", "static full-batch replay", True, self.batch_size)
        return self._graphed(inputs)


__all__ = ["GraphRoute", "StaticGraphRunner"]

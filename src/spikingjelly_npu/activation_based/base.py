"""State and step-mode primitives used by activation-based modules."""

from __future__ import annotations

import copy
from abc import abstractmethod
from typing import Any, Generator, Tuple

import torch
from torch import nn


def check_backend_library(backend: str) -> None:
    """Validate a backend name without importing accelerator packages eagerly."""
    if backend in {"torch", "npu", "aspy", "cupy"}:
        return
    raise NotImplementedError(
        f"backend={backend!r} is not available; spikingjelly_npu supports "
        "'torch', 'npu', qualified 'aspy', and 'cupy' as an AsPy preference alias"
    )


class StepModule:
    """Interface for modules supporting single-step and/or multi-step execution."""

    def supported_step_mode(self) -> Tuple[str, ...]:
        return ("s", "m")

    @property
    def step_mode(self) -> str:
        return self._step_mode

    @step_mode.setter
    def step_mode(self, value: str) -> None:
        if value not in self.supported_step_mode():
            raise ValueError(
                f"step_mode can only be {self.supported_step_mode()}, but got {value!r}"
            )
        self._step_mode = value


class SingleStepModule(StepModule):
    def supported_step_mode(self) -> Tuple[str, ...]:
        return ("s",)


class MultiStepModule(StepModule):
    def supported_step_mode(self) -> Tuple[str, ...]:
        return ("m",)


class MemoryModule(nn.Module, StepModule):
    """Base class for modules with resettable, non-persistent runtime state.

    Runtime memories intentionally are not parameters or state-dict buffers. Once a
    scalar memory has been materialized as a tensor, ``reset`` fills it in place. The
    stable address makes fixed-shape NPUGraph capture safe while preserving numerical
    reset semantics.
    """

    def __init__(self) -> None:
        super().__init__()
        self._memories: dict[str, Any] = {}
        self._memories_rv: dict[str, Any] = {}
        self._backend = "torch"
        self._step_mode = "s"

    @property
    def supported_backends(self) -> Tuple[str, ...]:
        return ("torch", "npu")

    @property
    def backend(self) -> str:
        return self._backend

    @backend.setter
    def backend(self, value: str) -> None:
        if value not in self.supported_backends:
            raise NotImplementedError(
                f"{value!r} is not a supported backend of {self._get_name()}"
            )
        check_backend_library(value)
        if value == "cupy":
            self.requested_backend = "cupy"
            self._backend = "aspy"
        else:
            self.requested_backend = value
            self._backend = value

    @abstractmethod
    def single_step_forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def multi_step_forward(self, x_seq: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        return torch.stack(
            [self.single_step_forward(x_seq[t], *args, **kwargs) for t in range(x_seq.shape[0])]
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.step_mode == "s":
            return self.single_step_forward(*args, **kwargs)
        if self.step_mode == "m":
            return self.multi_step_forward(*args, **kwargs)
        raise ValueError(self.step_mode)

    def register_memory(self, name: str, value: Any) -> None:
        if hasattr(self, name):
            raise ValueError(f"{name!r} is already an attribute")
        self._memories[name] = value
        self._memories_rv[name] = copy.deepcopy(value)

    def set_reset_value(self, name: str, value: Any) -> None:
        if name not in self._memories:
            raise KeyError(f"{name!r} is not a registered memory")
        self._memories_rv[name] = copy.deepcopy(value)

    def get_reset_value(self, name: str) -> Any:
        if name not in self._memories_rv:
            raise KeyError(f"{name!r} has no reset value")
        return self._memories_rv[name]

    def reset(self) -> None:
        for name, current in tuple(self._memories.items()):
            reset_value = self._memories_rv[name]
            if isinstance(current, torch.Tensor):
                if current.grad_fn is not None or current.requires_grad:
                    # Stateful training must sever the old autograd graph before the
                    # next independent sequence. A fresh tensor is the safe choice.
                    if isinstance(reset_value, torch.Tensor):
                        current = reset_value.detach().clone()
                    elif isinstance(reset_value, int | float | bool):
                        current = torch.full_like(current, reset_value, requires_grad=False)
                    else:
                        current = copy.deepcopy(reset_value)
                elif isinstance(reset_value, torch.Tensor):
                    if (
                        current.shape == reset_value.shape
                        and current.dtype == reset_value.dtype
                        and current.device == reset_value.device
                    ):
                        current.copy_(reset_value)
                    else:
                        current = reset_value.detach().clone()
                elif isinstance(reset_value, int | float | bool):
                    # Eval/no-grad state can be restored in place, retaining stable
                    # addresses for fixed-shape graph replay.
                    current.fill_(reset_value)
                else:
                    current = copy.deepcopy(reset_value)
                self._memories[name] = current
            else:
                self._memories[name] = copy.deepcopy(reset_value)

    def detach(self) -> None:
        for name, value in tuple(self._memories.items()):
            if isinstance(value, torch.Tensor):
                self._memories[name] = value.detach()

    def memories(self) -> Generator[Any, None, None]:
        yield from self._memories.values()

    def named_memories(self) -> Generator[tuple[str, Any], None, None]:
        yield from self._memories.items()

    def __getattr__(self, name: str) -> Any:
        memories = self.__dict__.get("_memories")
        if memories is not None and name in memories:
            return memories[name]
        return super().__getattr__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        memories = self.__dict__.get("_memories")
        if memories is not None and name in memories:
            memories[name] = value
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        memories = self.__dict__.get("_memories")
        if memories is not None and name in memories:
            del memories[name]
            del self._memories_rv[name]
        else:
            super().__delattr__(name)

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(self.__dict__.get("_memories", {})))

    def _apply(self, fn):
        for name, value in tuple(self._memories.items()):
            if isinstance(value, torch.Tensor):
                self._memories[name] = fn(value)
        return super()._apply(fn)

    def extra_repr(self) -> str:
        return f"step_mode={self.step_mode}, backend={self.backend}"


def named_memories(module: nn.Module, prefix: str = "") -> Generator[tuple[str, Any], None, None]:
    if isinstance(module, MemoryModule):
        for name, value in module.named_memories():
            yield (f"{prefix}.{name}" if prefix else name), value
    for child_name, child in module.named_children():
        child_prefix = f"{prefix}.{child_name}" if prefix else child_name
        yield from named_memories(child, child_prefix)


def memories(module: nn.Module) -> Generator[Any, None, None]:
    for _, value in named_memories(module):
        yield value

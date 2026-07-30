"""Network configuration and sequence-forward helpers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Union

import torch
from torch import Tensor, nn

from . import base


def _as_modules(module_or_modules: Any) -> tuple[Callable[[Tensor], Tensor], ...]:
    if isinstance(module_or_modules, list | tuple | nn.Sequential):
        return tuple(module_or_modules)
    return (module_or_modules,)


def _apply_modules(x: Tensor, module_or_modules: Any) -> Any:
    output = x
    for module in _as_modules(module_or_modules):
        output = module(output)
    return output


def multi_step_forward(x_seq: Tensor, single_step_module: Any) -> Tensor:
    """Apply one or more single-step modules over a ``[T, N, ...]`` sequence."""
    outputs = [_apply_modules(x_seq[t], single_step_module) for t in range(x_seq.shape[0])]
    return torch.stack(outputs)


def seq_to_ann_forward(x_seq: Tensor, stateless_module: Any) -> Any:
    """Flatten ``T`` and ``N``, execute stateless layers once, then restore them."""
    if x_seq.ndim < 2:
        raise ValueError(f"expected at least [T, N], got shape={tuple(x_seq.shape)}")
    time_steps, batch_size = x_seq.shape[:2]
    output = _apply_modules(x_seq.flatten(0, 1), stateless_module)
    if isinstance(output, tuple):
        return tuple(item.unflatten(0, (time_steps, batch_size)) for item in output)
    return output.unflatten(0, (time_steps, batch_size))


def reset_net(net: nn.Module) -> None:
    """Reset all resettable submodules, excluding ``net`` duplicate traversal."""
    for module in net.modules():
        if hasattr(module, "reset"):
            if not isinstance(module, base.MemoryModule):
                logging.debug("resetting non-MemoryModule %r", module)
            module.reset()


def detach_net(net: nn.Module) -> None:
    for module in net.modules():
        if hasattr(module, "detach"):
            if not isinstance(module, base.MemoryModule):
                logging.debug("detaching non-MemoryModule %r", module)
            module.detach()


def set_step_mode(net: nn.Module, step_mode: str) -> None:
    if step_mode not in {"s", "m"}:
        raise ValueError("step_mode must be 's' or 'm'")
    for module in net.modules():
        if hasattr(module, "step_mode"):
            module.step_mode = step_mode


def set_backend(
    net: nn.Module,
    backend: str,
    instance: Union[type[nn.Module], tuple[type[nn.Module], ...]] | None = None,
) -> None:
    for module in net.modules():
        if instance is not None and not isinstance(module, instance):
            continue
        if not hasattr(module, "backend"):
            continue
        supported = getattr(module, "supported_backends", ())
        if backend in supported:
            module.backend = backend
        else:
            logging.warning(
                "%s does not support backend=%r; keeping backend=%r",
                module.__class__.__name__,
                backend,
                getattr(module, "backend", None),
            )


__all__ = [
    "multi_step_forward",
    "seq_to_ann_forward",
    "reset_net",
    "detach_net",
    "set_step_mode",
    "set_backend",
]

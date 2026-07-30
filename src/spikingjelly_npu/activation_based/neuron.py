"""Integrate-and-fire neuron implementations for CPU and Ascend NPU."""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn

from . import _aspy, base, surrogate


class BaseNode(base.MemoryModule):
    """Base differentiable spiking neuron with SpikingJelly-compatible state."""

    def __init__(
        self,
        v_threshold: float = 1.0,
        v_reset: float | None = 0.0,
        surrogate_function: surrogate.SurrogateFunctionBase | None = None,
        detach_reset: bool = False,
        step_mode: str = "s",
        backend: str = "torch",
        store_v_seq: bool = False,
        backend_strict: bool = False,
    ) -> None:
        if not isinstance(v_threshold, float):
            raise TypeError("v_threshold must be a float")
        if v_reset is not None and not isinstance(v_reset, float):
            raise TypeError("v_reset must be a float or None")
        super().__init__()
        self.register_memory("v", 0.0 if v_reset is None else v_reset)
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.surrogate_function = (
            surrogate.Sigmoid() if surrogate_function is None else surrogate_function
        )
        self.detach_reset = bool(detach_reset)
        self.backend_strict = bool(backend_strict)
        self.last_backend_route = _aspy.eager_route(
            backend, "backend has not executed yet"
        )
        self.step_mode = step_mode
        self.backend = backend
        self.store_v_seq = store_v_seq

    @property
    def supported_backends(self) -> Tuple[str, ...]:
        return ("torch", "npu")

    @property
    def store_v_seq(self) -> bool:
        return self._store_v_seq

    @store_v_seq.setter
    def store_v_seq(self, value: bool) -> None:
        self._store_v_seq = bool(value)
        if value and not hasattr(self, "v_seq"):
            self.register_memory("v_seq", None)

    def v_float_to_tensor(self, x: torch.Tensor) -> None:
        if isinstance(self.v, int | float):
            self.v = torch.full_like(x, float(self.v), requires_grad=False)
        elif isinstance(self.v, torch.Tensor):
            if self.v.shape != x.shape:
                reset = 0.0 if self.v_reset is None else self.v_reset
                self.v = torch.full_like(x, reset, requires_grad=False)
            elif self.v.dtype != x.dtype or self.v.device != x.device:
                self.v = self.v.to(dtype=x.dtype, device=x.device)

    def neuronal_charge(self, x: torch.Tensor) -> None:
        raise NotImplementedError

    def neuronal_fire(self) -> torch.Tensor:
        return self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self, spike: torch.Tensor) -> None:
        spike_for_reset = spike.detach() if self.detach_reset else spike
        if self.v_reset is None:
            self.v = self.v - spike_for_reset * self.v_threshold
        else:
            self.v = spike_for_reset * self.v_reset + (1.0 - spike_for_reset) * self.v

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def _torch_multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        spikes = []
        voltages = [] if self.store_v_seq else None
        for t in range(x_seq.shape[0]):
            spikes.append(BaseNode.single_step_forward(self, x_seq[t]))
            if voltages is not None:
                voltages.append(self.v)
        output = torch.stack(spikes)
        if voltages is not None:
            self.v_seq = torch.stack(voltages)
        return output

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        if x_seq.ndim < 2:
            raise ValueError(f"expected [T, N, ...], got shape={tuple(x_seq.shape)}")
        self.last_backend_route = _aspy.eager_route(
            self.backend, f"backend={self.backend!r} uses the PyTorch implementation"
        )
        return self._torch_multi_step_forward(x_seq)

    def extra_repr(self) -> str:
        return (
            f"v_threshold={self.v_threshold}, v_reset={self.v_reset}, "
            f"detach_reset={self.detach_reset}, step_mode={self.step_mode}, "
            f"backend={self.backend}, backend_strict={self.backend_strict}, "
            f"store_v_seq={self.store_v_seq}"
        )


class IFNode(BaseNode):
    """Integrate-and-Fire neuron: ``v = v + x`` before firing/reset."""

    @property
    def supported_backends(self) -> Tuple[str, ...]:
        return ("torch", "npu", "aspy", "cupy")

    def neuronal_charge(self, x: torch.Tensor) -> None:
        self.v = self.v + x

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backend == "aspy":
            self.last_backend_route = _aspy.eager_route(
                "aspy", "AsPy acceleration currently supports only single-step fallback"
            )
        return super().single_step_forward(x)

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        if x_seq.ndim < 2:
            raise ValueError(f"expected [T, N, ...], got shape={tuple(x_seq.shape)}")
        if self.backend != "aspy":
            return super().multi_step_forward(x_seq)

        self.v_float_to_tensor(x_seq[0])
        result, route = _aspy.try_if_multi_step(
            x_seq,
            self.v,
            v_threshold=self.v_threshold,
            v_reset=self.v_reset,
            detach_reset=self.detach_reset,
            surrogate_function=self.surrogate_function,
            store_v_seq=self.store_v_seq,
            strict=self.backend_strict,
        )
        self.last_backend_route = route
        if result is None:
            return self._torch_multi_step_forward(x_seq)

        # Commit state only after the adapter has validated the full result.
        self.v = result.v_final
        if self.store_v_seq:
            self.v_seq = result.v_seq
        return result.spike_seq


class LIFNode(BaseNode):
    """Leaky Integrate-and-Fire neuron with configurable input decay."""

    def __init__(
        self,
        tau: float = 2.0,
        decay_input: bool = True,
        v_threshold: float = 1.0,
        v_reset: float | None = 0.0,
        surrogate_function: surrogate.SurrogateFunctionBase | None = None,
        detach_reset: bool = False,
        step_mode: str = "s",
        backend: str = "torch",
        store_v_seq: bool = False,
        backend_strict: bool = False,
    ) -> None:
        if not isinstance(tau, float) or tau <= 1.0:
            raise ValueError("tau must be a float greater than 1")
        super().__init__(
            v_threshold=v_threshold,
            v_reset=v_reset,
            surrogate_function=surrogate_function,
            detach_reset=detach_reset,
            step_mode=step_mode,
            backend=backend,
            store_v_seq=store_v_seq,
            backend_strict=backend_strict,
        )
        self.tau = tau
        self.decay_input = bool(decay_input)

    @property
    def supported_backends(self) -> Tuple[str, ...]:
        return ("torch", "npu", "aspy", "cupy")

    def neuronal_charge(self, x: torch.Tensor) -> None:
        reset = 0.0 if self.v_reset is None else self.v_reset
        if self.decay_input:
            self.v = self.v + (x - (self.v - reset)) / self.tau
        else:
            self.v = self.v - (self.v - reset) / self.tau + x

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backend == "aspy":
            self.last_backend_route = _aspy.eager_route(
                "aspy", "AsPy LIF acceleration supports multi-step only; using PyTorch single-step"
            )
        return super().single_step_forward(x)

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        if x_seq.ndim < 2:
            raise ValueError(f"expected [T, N, ...], got shape={tuple(x_seq.shape)}")
        if self.backend != "aspy":
            return super().multi_step_forward(x_seq)

        self.v_float_to_tensor(x_seq[0])
        result, route = _aspy.try_lif_multi_step(
            x_seq,
            self.v,
            v_threshold=self.v_threshold,
            v_reset=self.v_reset,
            detach_reset=self.detach_reset,
            surrogate_function=self.surrogate_function,
            store_v_seq=self.store_v_seq,
            tau=self.tau,
            decay_input=self.decay_input,
            strict=self.backend_strict,
        )
        self.last_backend_route = route
        if result is None:
            return self._torch_multi_step_forward(x_seq)

        self.v = result.v_final
        if self.store_v_seq:
            self.v_seq = result.v_seq
        return result.spike_seq

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, tau={self.tau}, decay_input={self.decay_input}"


class KLIFNode(LIFNode):
    """K-based LIF neuron compatible with SpikingJelly's public KLIF API.

    KLIF applies ``relu(k * h)`` after the ordinary LIF charge and before
    firing. ``k`` is a learnable scalar. When ``scale_reset=True``, the
    post-fire reset operates on voltage and threshold divided by ``k``.
    """

    def __init__(
        self,
        scale_reset: bool = False,
        tau: float = 2.0,
        decay_input: bool = True,
        v_threshold: float = 1.0,
        v_reset: float | None = 0.0,
        surrogate_function: surrogate.SurrogateFunctionBase | None = None,
        detach_reset: bool = False,
        step_mode: str = "s",
        backend: str = "torch",
        store_v_seq: bool = False,
        backend_strict: bool = False,
    ) -> None:
        super().__init__(
            tau=tau,
            decay_input=decay_input,
            v_threshold=v_threshold,
            v_reset=v_reset,
            surrogate_function=surrogate_function,
            detach_reset=detach_reset,
            step_mode=step_mode,
            backend=backend,
            store_v_seq=store_v_seq,
            backend_strict=backend_strict,
        )
        self.scale_reset = bool(scale_reset)
        self.k = nn.Parameter(torch.as_tensor(1.0))

    def neuronal_charge(self, x: torch.Tensor) -> None:
        super().neuronal_charge(x)
        self.v = torch.relu(self.k * self.v)

    def neuronal_reset(self, spike: torch.Tensor) -> None:
        spike_for_reset = spike.detach() if self.detach_reset else spike
        if self.scale_reset:
            voltage = self.v / self.k
            threshold = self.v_threshold / self.k
        else:
            voltage = self.v
            threshold = self.v_threshold
        if self.v_reset is None:
            self.v = voltage - spike_for_reset * threshold
        else:
            self.v = spike_for_reset * self.v_reset + (1.0 - spike_for_reset) * voltage

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backend == "aspy":
            reason = "AsPy KLIF acceleration supports multi-step only"
            if self.backend_strict:
                raise _aspy.AsPyBackendError(reason)
            self.last_backend_route = _aspy.eager_route(
                "aspy", f"{reason}; using PyTorch single-step"
            )
        return BaseNode.single_step_forward(self, x)

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        if x_seq.ndim < 2:
            raise ValueError(f"expected [T, N, ...], got shape={tuple(x_seq.shape)}")
        if self.backend != "aspy":
            return BaseNode.multi_step_forward(self, x_seq)

        self.v_float_to_tensor(x_seq[0])
        k = self.k.to(dtype=x_seq.dtype, device=x_seq.device)
        result, route = _aspy.try_klif_multi_step(
            x_seq,
            self.v,
            k,
            v_threshold=self.v_threshold,
            v_reset=self.v_reset,
            detach_reset=self.detach_reset,
            surrogate_function=self.surrogate_function,
            store_v_seq=self.store_v_seq,
            tau=self.tau,
            decay_input=self.decay_input,
            scale_reset=self.scale_reset,
            strict=self.backend_strict,
        )
        self.last_backend_route = route
        if result is None:
            return self._torch_multi_step_forward(x_seq)

        self.v = result.v_final
        if self.store_v_seq:
            self.v_seq = result.v_seq
        return result.spike_seq

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, scale_reset={self.scale_reset}"


class ParametricLIFNode(BaseNode):
    """LIF neuron with learnable reciprocal time constant ``sigmoid(w)``."""

    def __init__(
        self,
        init_tau: float = 2.0,
        decay_input: bool = True,
        v_threshold: float = 1.0,
        v_reset: float | None = 0.0,
        surrogate_function: surrogate.SurrogateFunctionBase | None = None,
        detach_reset: bool = False,
        step_mode: str = "s",
        backend: str = "torch",
        store_v_seq: bool = False,
        backend_strict: bool = False,
    ) -> None:
        if not isinstance(init_tau, float) or init_tau <= 1.0:
            raise ValueError("init_tau must be a float greater than 1")
        super().__init__(
            v_threshold=v_threshold,
            v_reset=v_reset,
            surrogate_function=surrogate_function,
            detach_reset=detach_reset,
            step_mode=step_mode,
            backend=backend,
            store_v_seq=store_v_seq,
            backend_strict=backend_strict,
        )
        self.decay_input = bool(decay_input)
        self.w = nn.Parameter(torch.as_tensor(-math.log(init_tau - 1.0)))

    @property
    def supported_backends(self) -> Tuple[str, ...]:
        return ("torch", "npu", "aspy", "cupy")

    def neuronal_charge(self, x: torch.Tensor) -> None:
        reciprocal_tau = self.w.sigmoid().to(x)
        reset = 0.0 if self.v_reset is None else self.v_reset
        if self.decay_input:
            self.v = self.v + (x - (self.v - reset)) * reciprocal_tau
        else:
            self.v = self.v - (self.v - reset) * reciprocal_tau + x

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backend == "aspy":
            self.last_backend_route = _aspy.eager_route(
                "aspy",
                "AsPy PLIF acceleration supports multi-step only; using PyTorch single-step",
            )
        return super().single_step_forward(x)

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        if x_seq.ndim < 2:
            raise ValueError(f"expected [T, N, ...], got shape={tuple(x_seq.shape)}")
        if self.backend != "aspy":
            return super().multi_step_forward(x_seq)

        self.v_float_to_tensor(x_seq[0])
        reciprocal_tau = self.w.sigmoid().to(dtype=x_seq.dtype, device=x_seq.device)
        result, route = _aspy.try_plif_multi_step(
            x_seq,
            self.v,
            reciprocal_tau,
            v_threshold=self.v_threshold,
            v_reset=self.v_reset,
            detach_reset=self.detach_reset,
            surrogate_function=self.surrogate_function,
            store_v_seq=self.store_v_seq,
            decay_input=self.decay_input,
            strict=self.backend_strict,
        )
        self.last_backend_route = route
        if result is None:
            return self._torch_multi_step_forward(x_seq)

        self.v = result.v_final
        if self.store_v_seq:
            self.v_seq = result.v_seq
        return result.spike_seq

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, decay_input={self.decay_input}"


__all__ = ["BaseNode", "IFNode", "LIFNode", "KLIFNode", "ParametricLIFNode"]

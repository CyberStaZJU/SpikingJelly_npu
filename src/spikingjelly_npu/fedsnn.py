"""Graph-safe, packed-time components for federated SNN workloads.

These modules use per-forward membrane state rather than persistent module memory.
That matches common FedSNN image classifiers, prevents state leakage across client
batches, and makes the complete fixed-shape forward safe for NPUGraph capture.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .activation_based import _aspy, layer, surrogate


class PoissonEncoder(nn.Module):
    """Encode normalized inputs into a ``[T, N, ...]`` Bernoulli spike tensor."""

    def __init__(self, time_steps: int) -> None:
        super().__init__()
        if time_steps <= 0:
            raise ValueError("time_steps must be positive")
        self.time_steps = int(time_steps)

    def forward(self, inputs: Tensor) -> Tensor:
        expanded = inputs.unsqueeze(0).expand(self.time_steps, *inputs.shape)
        return (torch.rand_like(expanded) <= expanded).to(inputs)


class MultiStepIF(nn.Module):
    """Graph-safe IF scan with zero membrane at the start of every forward."""

    def __init__(
        self,
        v_threshold: float = 1.0,
        v_reset: float | None = None,
        surrogate_function: surrogate.SurrogateFunctionBase | None = None,
        detach_reset: bool = True,
    ) -> None:
        super().__init__()
        self.v_threshold = float(v_threshold)
        self.v_reset = None if v_reset is None else float(v_reset)
        self.surrogate_function = (
            surrogate.ATan() if surrogate_function is None else surrogate_function
        )
        self.detach_reset = bool(detach_reset)

    def forward(self, current_seq: Tensor) -> Tensor:
        membrane = torch.zeros_like(current_seq[0])
        spikes = []
        for t in range(current_seq.shape[0]):
            membrane = membrane + current_seq[t]
            spike = self.surrogate_function(membrane - self.v_threshold)
            reset_spike = spike.detach() if self.detach_reset else spike
            if self.v_reset is None:
                membrane = membrane - reset_spike * self.v_threshold
            else:
                membrane = reset_spike * self.v_reset + (1.0 - reset_spike) * membrane
            spikes.append(spike)
        return torch.stack(spikes)


class FedSNNDecayLIF(nn.Module):
    """Stateless FedSNN decay-LIF scan with an exact optional AsPy route.

    Every forward starts from a zero membrane and returns only the spike
    sequence. The qualified native path is intentionally narrow: FP32,
    contiguous, storage-offset-zero NPU input, spiking ATan surrogate, soft
    reset, and detached reset. All other requests fall back before execution,
    unless ``backend_strict=True``.
    """

    def __init__(
        self,
        membrane_decay: float,
        v_threshold: float = 1.0,
        surrogate_function: surrogate.SurrogateFunctionBase | None = None,
        *,
        backend: str = "torch",
        backend_strict: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(membrane_decay, float) or not 0.0 <= membrane_decay <= 1.0:
            raise ValueError("membrane_decay must be a float in [0, 1]")
        if not math.isfinite(v_threshold):
            raise ValueError("v_threshold must be finite")
        if backend not in {"torch", "aspy"}:
            raise NotImplementedError("DecayLIF backend must be 'torch' or 'aspy'")
        self.membrane_decay = membrane_decay
        self.v_threshold = float(v_threshold)
        self.surrogate_function = (
            surrogate.ATan() if surrogate_function is None else surrogate_function
        )
        if isinstance(self.surrogate_function, surrogate.ATan) and (
            not math.isfinite(self.surrogate_function.alpha)
            or self.surrogate_function.alpha <= 0.0
        ):
            raise ValueError("ATan surrogate alpha must be finite and positive")
        self.backend = backend
        self.backend_strict = bool(backend_strict)
        self.last_backend_route = _aspy.eager_route(
            backend,
            "backend has not executed yet",
            logical_operation="fedsnn.decay_lif",
            reason_code="route.not_executed",
            training=self.training,
        )

    def _torch_forward(self, current_seq: Tensor) -> Tensor:
        membrane = torch.zeros_like(current_seq[0])
        spikes = []
        for t in range(current_seq.shape[0]):
            charged = membrane * self.membrane_decay
            charged = charged + current_seq[t]
            spike = self.surrogate_function(charged - self.v_threshold)
            membrane = charged - spike.detach() * self.v_threshold
            spikes.append(spike)
        return torch.stack(spikes)

    def forward(self, current_seq: Tensor) -> Tensor:
        if current_seq.ndim < 2:
            raise ValueError(
                f"expected [T, N, ...], got shape={tuple(current_seq.shape)}"
            )
        if current_seq.shape[0] == 0:
            raise ValueError("DecayLIF requires at least one time step")
        if current_seq[0].numel() == 0:
            raise ValueError("DecayLIF requires at least one neuron per time step")
        if self.backend != "aspy":
            self.last_backend_route = _aspy.eager_route(
                self.backend,
                f"backend={self.backend!r} uses the exact PyTorch implementation",
                logical_operation="fedsnn.decay_lif",
                reason_code="torch.explicit",
                training=self.training,
            )
            return self._torch_forward(current_seq)

        spike_seq, route = _aspy.try_fedsnn_decay_lif(
            current_seq,
            membrane_decay=self.membrane_decay,
            v_threshold=self.v_threshold,
            surrogate_function=self.surrogate_function,
            strict=self.backend_strict,
            training=self.training,
            requested_backend=self.backend,
        )
        self.last_backend_route = route
        if spike_seq is None:
            return self._torch_forward(current_seq)
        return spike_seq

    def extra_repr(self) -> str:
        return (
            f"membrane_decay={self.membrane_decay}, "
            f"v_threshold={self.v_threshold}, backend={self.backend}, "
            f"backend_strict={self.backend_strict}"
        )


DecayLIF = FedSNNDecayLIF


class MultiStepLIF(nn.Module):
    """Graph-safe LIF scan with zero membrane at the start of every forward."""

    def __init__(
        self,
        tau: float = 2.0,
        decay_input: bool = False,
        v_threshold: float = 1.0,
        v_reset: float | None = None,
        surrogate_function: surrogate.SurrogateFunctionBase | None = None,
        detach_reset: bool = True,
    ) -> None:
        super().__init__()
        if tau <= 1.0:
            raise ValueError("tau must be greater than 1")
        self.tau = float(tau)
        self.decay_input = bool(decay_input)
        self.v_threshold = float(v_threshold)
        self.v_reset = None if v_reset is None else float(v_reset)
        self.surrogate_function = (
            surrogate.ATan() if surrogate_function is None else surrogate_function
        )
        self.detach_reset = bool(detach_reset)

    def forward(self, current_seq: Tensor) -> Tensor:
        membrane = torch.zeros_like(current_seq[0])
        spikes = []
        reset = 0.0 if self.v_reset is None else self.v_reset
        for t in range(current_seq.shape[0]):
            if self.decay_input:
                membrane = membrane + (current_seq[t] - (membrane - reset)) / self.tau
            else:
                membrane = membrane - (membrane - reset) / self.tau + current_seq[t]
            spike = self.surrogate_function(membrane - self.v_threshold)
            reset_spike = spike.detach() if self.detach_reset else spike
            if self.v_reset is None:
                membrane = membrane - reset_spike * self.v_threshold
            else:
                membrane = reset_spike * self.v_reset + (1.0 - reset_spike) * membrane
            spikes.append(spike)
        return torch.stack(spikes)


class BNTT1d(nn.Module):
    """Independent BatchNorm1d modules for each simulation timestep."""

    def __init__(
        self,
        time_steps: int,
        num_features: int,
        *,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        if time_steps <= 0:
            raise ValueError("time_steps must be positive")
        self.time_steps = int(time_steps)
        self.layers = nn.ModuleList(
            [
                nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats)
                for _ in range(self.time_steps)
            ]
        )

    def forward(self, input_seq: Tensor) -> Tensor:
        if input_seq.shape[0] != self.time_steps:
            raise ValueError(
                f"expected T={self.time_steps}, got input shape={tuple(input_seq.shape)}"
            )
        return torch.stack([self.layers[t](input_seq[t]) for t in range(self.time_steps)])


class BNTT2d(nn.Module):
    """Independent BatchNorm2d modules for each simulation timestep."""

    def __init__(
        self,
        time_steps: int,
        num_features: int,
        *,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        if time_steps <= 0:
            raise ValueError("time_steps must be positive")
        self.time_steps = int(time_steps)
        self.layers = nn.ModuleList(
            [
                nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
                for _ in range(self.time_steps)
            ]
        )

    def forward(self, input_seq: Tensor) -> Tensor:
        if input_seq.shape[0] != self.time_steps:
            raise ValueError(
                f"expected T={self.time_steps}, got input shape={tuple(input_seq.shape)}"
            )
        return torch.stack([self.layers[t](input_seq[t]) for t in range(self.time_steps)])


class PackedBNTTConvNet(nn.Module):
    """Compact image SNN exercising Poisson, Conv, BNTT, LIF, pool, and readout.

    ``forward_current_seq`` is the accelerated shipped path. Stateless ANN
    operators consume ``[T, N, ...]`` tensors and flatten the first two axes,
    yielding larger, more efficient NPU kernels. ``forward_current_seq_stepwise``
    is an equivalent reference useful for parity tests and honest benchmarks.
    """

    _spikingjelly_npu_graph_safe = True

    def __init__(
        self,
        input_channels: int,
        classes: int,
        time_steps: int,
        *,
        channels: tuple[int, int] = (32, 64),
        hidden_features: int = 128,
        tau: float = 2.0,
        threshold: float = 1.0,
        surrogate_alpha: float = 2.0,
        pooled_size: int = 4,
    ) -> None:
        super().__init__()
        c1, c2 = channels
        self.time_steps = int(time_steps)
        self.encoder = PoissonEncoder(time_steps)
        self.conv1 = layer.Conv2d(input_channels, c1, 3, padding=1, bias=False, step_mode="m")
        self.bntt1 = BNTT2d(time_steps, c1)
        self.neuron1 = MultiStepLIF(
            tau=tau,
            decay_input=False,
            v_threshold=threshold,
            v_reset=None,
            surrogate_function=surrogate.ATan(alpha=surrogate_alpha),
            detach_reset=True,
        )
        self.pool1 = layer.AvgPool2d(2, 2, step_mode="m")
        self.conv2 = layer.Conv2d(c1, c2, 3, padding=1, bias=False, step_mode="m")
        self.bntt2 = BNTT2d(time_steps, c2)
        self.neuron2 = MultiStepLIF(
            tau=tau,
            decay_input=False,
            v_threshold=threshold,
            v_reset=None,
            surrogate_function=surrogate.ATan(alpha=surrogate_alpha),
            detach_reset=True,
        )
        self.pool2 = layer.AdaptiveAvgPool2d((pooled_size, pooled_size), step_mode="m")
        self.flatten = layer.Flatten(step_mode="m")
        self.fc1 = layer.Linear(
            c2 * pooled_size * pooled_size,
            hidden_features,
            bias=False,
            step_mode="m",
        )
        self.bntt3 = BNTT1d(time_steps, hidden_features)
        self.neuron3 = MultiStepLIF(
            tau=tau,
            decay_input=False,
            v_threshold=threshold,
            v_reset=None,
            surrogate_function=surrogate.ATan(alpha=surrogate_alpha),
            detach_reset=True,
        )
        self.readout = layer.Linear(hidden_features, classes, step_mode="m")

    def forward_current_seq(self, input_seq: Tensor) -> Tensor:
        if input_seq.shape[0] != self.time_steps:
            raise ValueError(
                f"expected T={self.time_steps}, got input shape={tuple(input_seq.shape)}"
            )
        spikes = self.neuron1(self.bntt1(self.conv1(input_seq)))
        spikes = self.neuron2(self.bntt2(self.conv2(self.pool1(spikes))))
        spikes = self.neuron3(self.bntt3(self.fc1(self.flatten(self.pool2(spikes)))))
        return self.readout(spikes).mean(0)

    def forward_current_seq_stepwise(self, input_seq: Tensor) -> Tensor:
        if input_seq.shape[0] != self.time_steps:
            raise ValueError(
                f"expected T={self.time_steps}, got input shape={tuple(input_seq.shape)}"
            )
        currents = torch.stack(
            [
                self.bntt1.layers[t](
                    torch.nn.functional.conv2d(
                        input_seq[t],
                        self.conv1.weight,
                        self.conv1.bias,
                        self.conv1.stride,
                        self.conv1.padding,
                        self.conv1.dilation,
                        self.conv1.groups,
                    )
                )
                for t in range(self.time_steps)
            ]
        )
        spikes = self.neuron1(currents)
        currents = torch.stack(
            [
                self.bntt2.layers[t](
                    torch.nn.functional.conv2d(
                        torch.nn.functional.avg_pool2d(spikes[t], 2, 2),
                        self.conv2.weight,
                        self.conv2.bias,
                        self.conv2.stride,
                        self.conv2.padding,
                        self.conv2.dilation,
                        self.conv2.groups,
                    )
                )
                for t in range(self.time_steps)
            ]
        )
        spikes = self.neuron2(currents)
        currents = []
        for t in range(self.time_steps):
            pooled = torch.nn.functional.adaptive_avg_pool2d(
                spikes[t], self.pool2.output_size
            )
            flattened = torch.flatten(pooled, 1)
            current = torch.nn.functional.linear(
                flattened, self.fc1.weight, self.fc1.bias
            )
            currents.append(self.bntt3.layers[t](current))
        spikes = self.neuron3(torch.stack(currents))
        logits = torch.stack(
            [torch.nn.functional.linear(spikes[t], self.readout.weight, self.readout.bias)
             for t in range(self.time_steps)]
        )
        return logits.mean(0)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.forward_current_seq(self.encoder(inputs))


class PackedBNTTMLP(nn.Module):
    """Small FedSNN canary that packs ANN work over ``T*N``.

    Input is an already encoded current sequence ``[T, N, in_features]``. The
    hidden linear layer runs once on ``T*N`` rather than once per timestep. The
    timestep scan remains explicit so spike dynamics and BNTT semantics remain
    transparent. Logits are the mean output current over time.
    """

    _spikingjelly_npu_graph_safe = True

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        time_steps: int,
        *,
        tau: float = 2.0,
        threshold: float = 1.0,
        surrogate_alpha: float = 2.0,
    ) -> None:
        super().__init__()
        self.time_steps = int(time_steps)
        self.fc1 = layer.Linear(in_features, hidden_features, step_mode="m")
        self.bntt1 = BNTT1d(time_steps, hidden_features)
        self.neuron1 = MultiStepLIF(
            tau=tau,
            decay_input=False,
            v_threshold=threshold,
            v_reset=None,
            surrogate_function=surrogate.ATan(alpha=surrogate_alpha),
            detach_reset=True,
        )
        self.fc2 = layer.Linear(hidden_features, out_features, step_mode="m")

    def forward(self, current_seq: Tensor, return_spikes: bool = False):
        if current_seq.shape[0] != self.time_steps:
            raise ValueError(
                f"expected T={self.time_steps}, got input shape={tuple(current_seq.shape)}"
            )
        hidden_current = self.fc1(current_seq)
        hidden_spikes = self.neuron1(self.bntt1(hidden_current))
        logits = self.fc2(hidden_spikes).mean(0)
        if return_spikes:
            return logits, hidden_spikes
        return logits


__all__ = [
    "PoissonEncoder",
    "MultiStepIF",
    "FedSNNDecayLIF",
    "DecayLIF",
    "MultiStepLIF",
    "BNTT1d",
    "BNTT2d",
    "PackedBNTTConvNet",
    "PackedBNTTMLP",
]

"""Step-aware wrappers around common ``torch.nn`` layers."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..npu.amp import is_npu_bf16_autocast_active
from . import base, functional, neuron


class _StepAwareMixin(base.StepModule):
    def _set_step_mode(self, step_mode: str) -> None:
        self.step_mode = step_mode

    def _forward_step_aware(self, x: Tensor, single_step) -> Tensor:
        if self.step_mode == "s":
            return single_step(x)
        return functional.seq_to_ann_forward(x, single_step)


class Linear(nn.Linear, _StepAwareMixin):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        step_mode: str = "s",
    ):
        super().__init__(in_features, out_features, bias)
        self._set_step_mode(step_mode)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_step_aware(x, super().forward)


class Conv1d(nn.Conv1d, _StepAwareMixin):
    def __init__(self, *args, step_mode: str = "s", **kwargs):
        super().__init__(*args, **kwargs)
        self._set_step_mode(step_mode)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_step_aware(x, super().forward)


class Conv2d(nn.Conv2d, _StepAwareMixin):
    def __init__(self, *args, step_mode: str = "s", **kwargs):
        super().__init__(*args, **kwargs)
        self._set_step_mode(step_mode)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_step_aware(x, super().forward)


class Conv3d(nn.Conv3d, _StepAwareMixin):
    def __init__(self, *args, step_mode: str = "s", **kwargs):
        super().__init__(*args, **kwargs)
        self._set_step_mode(step_mode)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_step_aware(x, super().forward)


def _fp32_island_forward(single_step, x: Tensor) -> Tensor:
    """Run a numerically sensitive operation in FP32 and restore public BF16."""

    if not is_npu_bf16_autocast_active():
        return single_step(x)
    public_dtype = x.dtype
    with torch.autocast(device_type="npu", enabled=False):
        output = single_step(x.float())
    if public_dtype == torch.bfloat16 and output.is_floating_point():
        return output.to(dtype=public_dtype)
    return output


class BatchNorm1d(nn.BatchNorm1d, _StepAwareMixin):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True, step_mode: str = "s"):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self._set_step_mode(step_mode)

    def forward(self, x: Tensor) -> Tensor:
        single_step = super().forward
        return self._forward_step_aware(
            x, lambda value: _fp32_island_forward(single_step, value)
        )


class BatchNorm2d(nn.BatchNorm2d, _StepAwareMixin):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True, step_mode: str = "s"):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self._set_step_mode(step_mode)

    def forward(self, x: Tensor) -> Tensor:
        single_step = super().forward
        return self._forward_step_aware(
            x, lambda value: _fp32_island_forward(single_step, value)
        )


class BatchNorm3d(nn.BatchNorm3d, _StepAwareMixin):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True, step_mode: str = "s"):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self._set_step_mode(step_mode)

    def forward(self, x: Tensor) -> Tensor:
        single_step = super().forward
        return self._forward_step_aware(
            x, lambda value: _fp32_island_forward(single_step, value)
        )


class MaxPool1d(nn.MaxPool1d, _StepAwareMixin):
    def __init__(self, *args, step_mode: str = "s", **kwargs):
        super().__init__(*args, **kwargs)
        self._set_step_mode(step_mode)

    def forward(self, x: Tensor):
        return self._forward_step_aware(x, super().forward)


class MaxPool2d(nn.MaxPool2d, _StepAwareMixin):
    def __init__(self, *args, step_mode: str = "s", **kwargs):
        super().__init__(*args, **kwargs)
        self._set_step_mode(step_mode)

    def forward(self, x: Tensor):
        return self._forward_step_aware(x, super().forward)


class MaxPool3d(nn.MaxPool3d, _StepAwareMixin):
    def __init__(self, *args, step_mode: str = "s", **kwargs):
        super().__init__(*args, **kwargs)
        self._set_step_mode(step_mode)

    def forward(self, x: Tensor):
        return self._forward_step_aware(x, super().forward)


class AvgPool1d(nn.AvgPool1d, _StepAwareMixin):
    def __init__(self, *args, step_mode: str = "s", **kwargs):
        super().__init__(*args, **kwargs)
        self._set_step_mode(step_mode)

    def forward(self, x: Tensor):
        single_step = super().forward
        return self._forward_step_aware(
            x, lambda value: _fp32_island_forward(single_step, value)
        )


class AvgPool2d(nn.AvgPool2d, _StepAwareMixin):
    def __init__(self, *args, step_mode: str = "s", **kwargs):
        super().__init__(*args, **kwargs)
        self._set_step_mode(step_mode)

    def forward(self, x: Tensor):
        single_step = super().forward
        return self._forward_step_aware(
            x, lambda value: _fp32_island_forward(single_step, value)
        )


class AvgPool3d(nn.AvgPool3d, _StepAwareMixin):
    def __init__(self, *args, step_mode: str = "s", **kwargs):
        super().__init__(*args, **kwargs)
        self._set_step_mode(step_mode)

    def forward(self, x: Tensor):
        single_step = super().forward
        return self._forward_step_aware(
            x, lambda value: _fp32_island_forward(single_step, value)
        )


class AdaptiveAvgPool1d(nn.AdaptiveAvgPool1d, _StepAwareMixin):
    def __init__(self, output_size, step_mode: str = "s"):
        super().__init__(output_size)
        self._set_step_mode(step_mode)

    def forward(self, x: Tensor):
        single_step = super().forward
        return self._forward_step_aware(
            x, lambda value: _fp32_island_forward(single_step, value)
        )


class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d, _StepAwareMixin):
    def __init__(self, output_size, step_mode: str = "s"):
        super().__init__(output_size)
        self._set_step_mode(step_mode)

    def forward(self, x: Tensor):
        single_step = super().forward
        return self._forward_step_aware(
            x, lambda value: _fp32_island_forward(single_step, value)
        )


class AdaptiveAvgPool3d(nn.AdaptiveAvgPool3d, _StepAwareMixin):
    def __init__(self, output_size, step_mode: str = "s"):
        super().__init__(output_size)
        self._set_step_mode(step_mode)

    def forward(self, x: Tensor):
        single_step = super().forward
        return self._forward_step_aware(
            x, lambda value: _fp32_island_forward(single_step, value)
        )


class Flatten(nn.Flatten, _StepAwareMixin):
    def __init__(self, start_dim: int = 1, end_dim: int = -1, step_mode: str = "s"):
        super().__init__(start_dim, end_dim)
        self._set_step_mode(step_mode)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_step_aware(x, super().forward)


class VotingLayer(nn.Module, base.StepModule):
    def __init__(self, voting_size: int = 10, step_mode: str = "s"):
        super().__init__()
        if voting_size <= 0:
            raise ValueError("voting_size must be positive")
        self.voting_size = int(voting_size)
        self.step_mode = step_mode

    def single_step_forward(self, x: Tensor) -> Tensor:
        def vote(value: Tensor) -> Tensor:
            return F.avg_pool1d(
                value.unsqueeze(1), self.voting_size, self.voting_size
            ).squeeze(1)

        return _fp32_island_forward(vote, x)

    def forward(self, x: Tensor) -> Tensor:
        if self.step_mode == "s":
            return self.single_step_forward(x)
        return functional.seq_to_ann_forward(x, self.single_step_forward)


_FP32_ISLAND_MODULES = (
    nn.modules.batchnorm._BatchNorm,
    nn.AvgPool1d,
    nn.AvgPool2d,
    nn.AvgPool3d,
    nn.AdaptiveAvgPool1d,
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveAvgPool3d,
)


def _module_forward_with_bf16_policy(module: nn.Module, x: Tensor) -> Tensor:
    if isinstance(module, _FP32_ISLAND_MODULES):
        return _fp32_island_forward(module, x)
    return module(x)


class SeqToANNContainer(nn.Sequential, base.MultiStepModule):
    def forward(self, x_seq: Tensor) -> Tensor:
        if not is_npu_bf16_autocast_active():
            return functional.seq_to_ann_forward(x_seq, tuple(self))
        if x_seq.ndim < 2:
            raise ValueError(f"expected at least [T, N], got shape={tuple(x_seq.shape)}")
        time_steps, batch_size = x_seq.shape[:2]
        output = x_seq.flatten(0, 1)
        for module in self:
            output = _module_forward_with_bf16_policy(module, output)
        return output.unflatten(0, (time_steps, batch_size))


class SpikingSelfAttention(nn.Module, base.MultiStepModule):
    """Spikformer's token-last, softmax-free spiking self-attention.

    Inputs and outputs use ``[T, N, C, L]``, where ``L`` is the token count.
    """

    def __init__(self, dim: int, num_heads: int = 8, backend: str = "torch") -> None:
        super().__init__()
        if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim must be a positive integer, but got {dim!r}")
        if isinstance(num_heads, bool) or not isinstance(num_heads, int) or num_heads <= 0:
            raise ValueError(
                f"num_heads must be a positive integer, but got {num_heads!r}"
            )
        self.dim = dim
        self.num_heads = num_heads
        if self.dim % self.num_heads != 0:
            raise ValueError(
                f"dim={self.dim} must be divisible by num_heads={self.num_heads}"
            )
        self.head_dim = self.dim // self.num_heads
        self.scale = 0.125

        self.qkv_conv_bn = SeqToANNContainer(
            nn.Conv1d(self.dim, self.dim * 3, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.dim * 3),
        )
        self.qkv_lif = neuron.LIFNode(
            tau=2.0,
            detach_reset=True,
            step_mode="m",
            backend=backend,
        )
        self.attn_lif = neuron.LIFNode(
            tau=2.0,
            v_threshold=0.5,
            detach_reset=True,
            step_mode="m",
            backend=backend,
        )
        self.proj_conv_bn = SeqToANNContainer(
            nn.Conv1d(self.dim, self.dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.dim),
        )
        self.proj_lif = neuron.LIFNode(
            tau=2.0,
            detach_reset=True,
            step_mode="m",
            backend=backend,
        )

    @property
    def backend(self) -> str:
        return self.qkv_lif.requested_backend

    @backend.setter
    def backend(self, value: str) -> None:
        base.check_backend_library(value)
        nodes = (self.qkv_lif, self.attn_lif, self.proj_lif)
        for node in nodes:
            if value not in node.supported_backends:
                raise NotImplementedError(
                    f"{value!r} is not a supported backend of {node._get_name()}"
                )
        for node in nodes:
            node.backend = value

    @staticmethod
    def _ssa_kernel_torch(qkv: Tensor, scale: float) -> Tensor:
        """Apply the exact token-last ``(V K^T) Q`` Spikformer kernel."""
        if qkv.ndim != 6 or qkv.shape[2] != 3:
            raise ValueError(
                "qkv must have shape [T, N, 3, num_heads, head_dim, L], "
                f"but got {tuple(qkv.shape)}"
            )
        q, k, v = qkv.unbind(dim=2)
        output = v @ k.transpose(-2, -1)
        output = output @ q
        return output * scale

    def forward(self, x_seq: Tensor) -> Tensor:
        if x_seq.ndim != 4:
            raise ValueError(
                "expected input with shape [T, N, C, L], "
                f"but got shape={tuple(x_seq.shape)}"
            )
        time_steps, batch_size, channels, tokens = x_seq.shape
        if min(time_steps, batch_size, channels, tokens) <= 0:
            raise ValueError(
                "all [T, N, C, L] dimensions must be positive, "
                f"but got shape={tuple(x_seq.shape)}"
            )
        if channels != self.dim:
            raise ValueError(
                f"expected C={self.dim}, but got C={channels} in shape={tuple(x_seq.shape)}"
            )

        qkv = self.qkv_lif(self.qkv_conv_bn(x_seq))
        expected_qkv_shape = (time_steps, batch_size, self.dim * 3, tokens)
        if tuple(qkv.shape) != expected_qkv_shape:
            raise ValueError(
                f"qkv projection must return shape={expected_qkv_shape}, "
                f"but got shape={tuple(qkv.shape)}"
            )
        qkv = qkv.reshape(
            time_steps,
            batch_size,
            3,
            self.num_heads,
            self.head_dim,
            tokens,
        )

        output = self._ssa_kernel_torch(qkv, self.scale)
        output = self.attn_lif(output).reshape(
            time_steps, batch_size, self.dim, tokens
        )
        output = self.proj_lif(self.proj_conv_bn(output))
        return output

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, num_heads={self.num_heads}, "
            f"backend={self.backend!r}"
        )


__all__ = [
    "Linear", "Conv1d", "Conv2d", "Conv3d",
    "BatchNorm1d", "BatchNorm2d", "BatchNorm3d",
    "MaxPool1d", "MaxPool2d", "MaxPool3d",
    "AvgPool1d", "AvgPool2d", "AvgPool3d",
    "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveAvgPool3d",
    "Flatten", "VotingLayer", "SeqToANNContainer", "SpikingSelfAttention",
]

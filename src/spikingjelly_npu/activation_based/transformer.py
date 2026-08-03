"""Eager spiking Transformer building blocks."""

from __future__ import annotations

from torch import Tensor, nn

from . import base, layer, neuron


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, but got {value!r}")
    return value


class SpikingSelfAttention(nn.Module, base.MultiStepModule):
    """Spikformer's token-last, softmax-free spiking self-attention.

    Inputs and outputs use ``[T, N, C, L]``, where ``L`` is the token count.
    """

    def __init__(self, dim: int, num_heads: int = 8, backend: str = "torch") -> None:
        super().__init__()
        self.dim = _positive_int("dim", dim)
        self.num_heads = _positive_int("num_heads", num_heads)
        if self.dim % self.num_heads != 0:
            raise ValueError(
                f"dim={self.dim} must be divisible by num_heads={self.num_heads}"
            )
        self.head_dim = self.dim // self.num_heads
        self.scale = 0.125

        self.qkv_conv_bn = layer.SeqToANNContainer(
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
        self.proj_conv_bn = layer.SeqToANNContainer(
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


__all__ = ["SpikingSelfAttention"]

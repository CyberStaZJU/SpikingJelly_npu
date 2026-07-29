"""Adapter for the optional out-of-tree AsPy native extension.

The module is imported only after the core package has qualified an Ascend NPU
request. A missing native extension therefore remains a pre-execution eager
fallback rather than a package import failure.
"""

from __future__ import annotations

import torch

from spikingjelly_npu._native import load_aspy_native

_native = load_aspy_native()
if_backward = _native.if_backward
if_forward = _native.if_forward
lif_backward = _native.lif_backward
lif_forward = _native.lif_forward
plif_backward = _native.plif_backward
plif_forward = _native.plif_forward


class _AsPyIF(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x_seq: torch.Tensor,
        v_init: torch.Tensor,
        v_threshold: float,
        v_reset: float,
        hard_reset: bool,
        detach_reset: bool,
        surrogate_alpha: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spike_seq, v_seq, v_final, h_seq = if_forward(
            x_seq,
            v_init,
            v_threshold,
            v_reset,
            hard_reset,
        )
        ctx.save_for_backward(h_seq, spike_seq)
        ctx.v_threshold = v_threshold
        ctx.v_reset = v_reset
        ctx.hard_reset = hard_reset
        ctx.detach_reset = detach_reset
        ctx.surrogate_alpha = surrogate_alpha
        return spike_seq, v_final, v_seq

    @staticmethod
    def backward(
        ctx,
        grad_spike_seq: torch.Tensor | None,
        grad_v_final: torch.Tensor | None,
        grad_v_seq: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        if torch.is_grad_enabled():
            raise RuntimeError("AsPy IF supports first-order gradients only")
        h_seq, spike_seq = ctx.saved_tensors
        if grad_spike_seq is None:
            grad_spike_seq = torch.zeros_like(spike_seq)
        else:
            grad_spike_seq = grad_spike_seq.contiguous()
        if grad_v_seq is None:
            grad_v_seq = torch.zeros_like(h_seq)
        else:
            grad_v_seq = grad_v_seq.contiguous()
        if grad_v_final is None:
            grad_v_final = torch.zeros_like(h_seq[0])
        else:
            grad_v_final = grad_v_final.contiguous()

        grad_x_seq, grad_v_init = if_backward(
            h_seq,
            spike_seq,
            grad_spike_seq,
            grad_v_seq,
            grad_v_final,
            ctx.v_threshold,
            ctx.v_reset,
            ctx.hard_reset,
            ctx.detach_reset,
            ctx.surrogate_alpha,
        )
        return grad_x_seq, grad_v_init, None, None, None, None, None


class _AsPyLIF(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x_seq: torch.Tensor,
        v_init: torch.Tensor,
        v_threshold: float,
        v_reset: float,
        hard_reset: bool,
        detach_reset: bool,
        surrogate_alpha: float,
        tau: float,
        decay_input: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spike_seq, v_seq, v_final, h_seq = lif_forward(
            x_seq,
            v_init,
            v_threshold,
            v_reset,
            hard_reset,
            tau,
            decay_input,
        )
        ctx.save_for_backward(h_seq, spike_seq)
        ctx.v_threshold = v_threshold
        ctx.v_reset = v_reset
        ctx.hard_reset = hard_reset
        ctx.detach_reset = detach_reset
        ctx.surrogate_alpha = surrogate_alpha
        ctx.tau = tau
        ctx.decay_input = decay_input
        return spike_seq, v_final, v_seq

    @staticmethod
    def backward(
        ctx,
        grad_spike_seq: torch.Tensor | None,
        grad_v_final: torch.Tensor | None,
        grad_v_seq: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        if torch.is_grad_enabled():
            raise RuntimeError("AsPy LIF supports first-order gradients only")
        h_seq, spike_seq = ctx.saved_tensors
        if grad_spike_seq is None:
            grad_spike_seq = torch.zeros_like(spike_seq)
        else:
            grad_spike_seq = grad_spike_seq.contiguous()
        if grad_v_seq is None:
            grad_v_seq = torch.zeros_like(h_seq)
        else:
            grad_v_seq = grad_v_seq.contiguous()
        if grad_v_final is None:
            grad_v_final = torch.zeros_like(h_seq[0])
        else:
            grad_v_final = grad_v_final.contiguous()

        grad_x_seq, grad_v_init = lif_backward(
            h_seq,
            spike_seq,
            grad_spike_seq,
            grad_v_seq,
            grad_v_final,
            ctx.v_threshold,
            ctx.v_reset,
            ctx.hard_reset,
            ctx.detach_reset,
            ctx.surrogate_alpha,
            ctx.tau,
            ctx.decay_input,
        )
        return grad_x_seq, grad_v_init, None, None, None, None, None, None, None


class _AsPyPLIF(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x_seq: torch.Tensor,
        v_init: torch.Tensor,
        reciprocal_tau: torch.Tensor,
        v_threshold: float,
        v_reset: float,
        hard_reset: bool,
        detach_reset: bool,
        surrogate_alpha: float,
        decay_input: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spike_seq, v_seq, v_final, h_seq, v_prev_seq = plif_forward(
            x_seq,
            v_init,
            reciprocal_tau,
            v_threshold,
            v_reset,
            hard_reset,
            decay_input,
        )
        ctx.save_for_backward(x_seq, v_prev_seq, h_seq, spike_seq, reciprocal_tau)
        ctx.v_threshold = v_threshold
        ctx.v_reset = v_reset
        ctx.hard_reset = hard_reset
        ctx.detach_reset = detach_reset
        ctx.surrogate_alpha = surrogate_alpha
        ctx.decay_input = decay_input
        return spike_seq, v_final, v_seq

    @staticmethod
    def backward(
        ctx,
        grad_spike_seq: torch.Tensor | None,
        grad_v_final: torch.Tensor | None,
        grad_v_seq: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        if torch.is_grad_enabled():
            raise RuntimeError("AsPy PLIF supports first-order gradients only")
        x_seq, v_prev_seq, h_seq, spike_seq, reciprocal_tau = ctx.saved_tensors
        if grad_spike_seq is None:
            grad_spike_seq = torch.zeros_like(spike_seq)
        else:
            grad_spike_seq = grad_spike_seq.contiguous()
        if grad_v_seq is None:
            grad_v_seq = torch.zeros_like(h_seq)
        else:
            grad_v_seq = grad_v_seq.contiguous()
        if grad_v_final is None:
            grad_v_final = torch.zeros_like(h_seq[0])
        else:
            grad_v_final = grad_v_final.contiguous()

        grad_x_seq, grad_v_init, grad_reciprocal_tau_partial = plif_backward(
            x_seq,
            v_prev_seq,
            h_seq,
            spike_seq,
            grad_spike_seq,
            grad_v_seq,
            grad_v_final,
            reciprocal_tau,
            ctx.v_threshold,
            ctx.v_reset,
            ctx.hard_reset,
            ctx.detach_reset,
            ctx.surrogate_alpha,
            ctx.decay_input,
        )
        grad_reciprocal_tau = grad_reciprocal_tau_partial.sum().reshape_as(
            reciprocal_tau
        )
        return (
            grad_x_seq,
            grad_v_init,
            grad_reciprocal_tau,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def if_multi_step(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    v_threshold: float,
    v_reset: float | None,
    detach_reset: bool,
    surrogate_name: str,
    surrogate_alpha: float,
    store_v_seq: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run the qualified fused IF forward/backward using the router contract."""

    if surrogate_name != "atan":
        raise ValueError(f"unsupported AsPy surrogate: {surrogate_name!r}")
    hard_reset = v_reset is not None
    reset = 0.0 if v_reset is None else v_reset
    time_steps = x_seq.shape[0]
    neuron_count = x_seq[0].numel()
    aligned_count = (neuron_count + 7) // 8 * 8
    padding = aligned_count - neuron_count
    x_native = x_seq.reshape(time_steps, neuron_count)
    v_native = v_init.reshape(neuron_count)
    if padding:
        x_native = torch.cat(
            (x_native, x_native.new_zeros((time_steps, padding))),
            dim=1,
        )
        v_native = torch.cat((v_native, v_native.new_zeros(padding)), dim=0)

    spike_native, v_final_native, v_seq_native = _AsPyIF.apply(
        x_native,
        v_native,
        v_threshold,
        reset,
        hard_reset,
        detach_reset,
        surrogate_alpha,
    )
    spike_seq = spike_native[:, :neuron_count].reshape_as(x_seq).contiguous()
    v_final = v_final_native[:neuron_count].reshape_as(v_init).contiguous()
    if not store_v_seq:
        return spike_seq, v_final, None
    v_seq = v_seq_native[:, :neuron_count].reshape_as(x_seq).contiguous()
    return spike_seq, v_final, v_seq


def lif_multi_step(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    v_threshold: float,
    v_reset: float | None,
    detach_reset: bool,
    surrogate_name: str,
    surrogate_alpha: float,
    store_v_seq: bool,
    tau: float,
    decay_input: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run the qualified fused fixed-tau LIF forward/backward."""

    if surrogate_name != "atan":
        raise ValueError(f"unsupported AsPy surrogate: {surrogate_name!r}")
    if not isinstance(tau, float) or tau <= 1.0:
        raise ValueError("tau must be a float greater than 1")
    hard_reset = v_reset is not None
    reset = 0.0 if v_reset is None else v_reset
    time_steps = x_seq.shape[0]
    neuron_count = x_seq[0].numel()
    aligned_count = (neuron_count + 7) // 8 * 8
    padding = aligned_count - neuron_count
    x_native = x_seq.reshape(time_steps, neuron_count)
    v_native = v_init.reshape(neuron_count)
    if padding:
        x_native = torch.cat(
            (x_native, x_native.new_zeros((time_steps, padding))),
            dim=1,
        )
        v_native = torch.cat((v_native, v_native.new_full((padding,), reset)), dim=0)

    spike_native, v_final_native, v_seq_native = _AsPyLIF.apply(
        x_native,
        v_native,
        v_threshold,
        reset,
        hard_reset,
        detach_reset,
        surrogate_alpha,
        tau,
        decay_input,
    )
    spike_seq = spike_native[:, :neuron_count].reshape_as(x_seq).contiguous()
    v_final = v_final_native[:neuron_count].reshape_as(v_init).contiguous()
    if not store_v_seq:
        return spike_seq, v_final, None
    v_seq = v_seq_native[:, :neuron_count].reshape_as(x_seq).contiguous()
    return spike_seq, v_final, v_seq


def plif_multi_step(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    reciprocal_tau: torch.Tensor,
    v_threshold: float,
    v_reset: float | None,
    detach_reset: bool,
    surrogate_name: str,
    surrogate_alpha: float,
    store_v_seq: bool,
    decay_input: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run fused PLIF using a dynamic FP32 NPU reciprocal-tau tensor."""

    if surrogate_name != "atan":
        raise ValueError(f"unsupported AsPy surrogate: {surrogate_name!r}")
    if reciprocal_tau.numel() != 1:
        raise ValueError("reciprocal_tau must contain exactly one value")
    hard_reset = v_reset is not None
    reset = 0.0 if v_reset is None else v_reset
    time_steps = x_seq.shape[0]
    neuron_count = x_seq[0].numel()
    aligned_count = (neuron_count + 7) // 8 * 8
    padding = aligned_count - neuron_count
    x_native = x_seq.reshape(time_steps, neuron_count)
    v_native = v_init.reshape(neuron_count)
    if padding:
        x_native = torch.cat(
            (x_native, x_native.new_zeros((time_steps, padding))),
            dim=1,
        )
        v_native = torch.cat((v_native, v_native.new_full((padding,), reset)), dim=0)

    spike_native, v_final_native, v_seq_native = _AsPyPLIF.apply(
        x_native,
        v_native,
        reciprocal_tau.reshape(1).contiguous(),
        v_threshold,
        reset,
        hard_reset,
        detach_reset,
        surrogate_alpha,
        decay_input,
    )
    spike_seq = spike_native[:, :neuron_count].reshape_as(x_seq).contiguous()
    v_final = v_final_native[:neuron_count].reshape_as(v_init).contiguous()
    if not store_v_seq:
        return spike_seq, v_final, None
    v_seq = v_seq_native[:, :neuron_count].reshape_as(x_seq).contiguous()
    return spike_seq, v_final, v_seq


__all__ = ["if_multi_step", "lif_multi_step", "plif_multi_step"]

"""Adapter for the optional out-of-tree AsPy native extension.

The module is imported only after the core package has qualified an Ascend NPU
request. A missing native extension therefore remains a pre-execution eager
fallback rather than a package import failure.
"""

from __future__ import annotations

import math

import torch

from spikingjelly_npu._native import load_aspy_native

_native = load_aspy_native()
if_backward = _native.if_backward
if_forward = _native.if_forward
lif_backward = _native.lif_backward
lif_forward = _native.lif_forward
plif_backward = _native.plif_backward
plif_forward = _native.plif_forward
klif_backward = getattr(_native, "klif_backward", None)
klif_forward = getattr(_native, "klif_forward", None)
supports_klif = callable(klif_forward) and callable(klif_backward)
fedsnn_decay_lif_backward = getattr(_native, "fedsnn_decay_lif_backward", None)
fedsnn_decay_lif_forward = getattr(_native, "fedsnn_decay_lif_forward", None)
supports_fedsnn_decay_lif = callable(fedsnn_decay_lif_forward) and callable(
    fedsnn_decay_lif_backward
)


def _npu_format(tensor: torch.Tensor) -> int | None:
    if tensor.device.type != "npu":
        return None
    import torch_npu

    get_format = getattr(torch_npu, "get_npu_format", None)
    if not callable(get_format):
        get_format = torch.ops.npu.get_npu_format
    return int(get_format(tensor))


def _require_nd(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """Fail before native execution unless a tensor is physical ND."""

    actual_format = _npu_format(tensor)
    if actual_format is not None and actual_format != 2:
        raise RuntimeError(
            f"{name} must use physical ACL_FORMAT_ND (2), got format={actual_format}"
        )
    return tensor


def _copy_to_nd(tensor: torch.Tensor) -> torch.Tensor:
    """Return physical ND without relying on npu_format_cast autograd support."""

    if tensor.device.type != "npu" or _npu_format(tensor) == 2:
        return tensor
    return tensor.clone()


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
        x_seq = _require_nd(x_seq, "x_seq")
        v_init = _require_nd(v_init, "v_init")
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

        h_seq = _require_nd(h_seq, "h_seq")
        spike_seq = _require_nd(spike_seq, "spike_seq")
        grad_spike_seq = _require_nd(grad_spike_seq, "grad_spike_seq")
        grad_v_seq = _require_nd(grad_v_seq, "grad_v_seq")
        grad_v_final = _require_nd(grad_v_final, "grad_v_final")
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
        x_seq = _require_nd(x_seq, "x_seq")
        v_init = _require_nd(v_init, "v_init")
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

        h_seq = _require_nd(h_seq, "h_seq")
        spike_seq = _require_nd(spike_seq, "spike_seq")
        grad_spike_seq = _require_nd(grad_spike_seq, "grad_spike_seq")
        grad_v_seq = _require_nd(grad_v_seq, "grad_v_seq")
        grad_v_final = _require_nd(grad_v_final, "grad_v_final")
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
        x_seq = _require_nd(x_seq, "x_seq")
        v_init = _require_nd(v_init, "v_init")
        reciprocal_tau = _require_nd(reciprocal_tau, "reciprocal_tau")
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

        x_seq = _require_nd(x_seq, "x_seq")
        v_prev_seq = _require_nd(v_prev_seq, "v_prev_seq")
        h_seq = _require_nd(h_seq, "h_seq")
        spike_seq = _require_nd(spike_seq, "spike_seq")
        grad_spike_seq = _require_nd(grad_spike_seq, "grad_spike_seq")
        grad_v_seq = _require_nd(grad_v_seq, "grad_v_seq")
        grad_v_final = _require_nd(grad_v_final, "grad_v_final")
        reciprocal_tau = _require_nd(reciprocal_tau, "reciprocal_tau")
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


class _AsPyKLIF(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x_seq: torch.Tensor,
        v_init: torch.Tensor,
        k: torch.Tensor,
        v_threshold: float,
        v_reset: float,
        hard_reset: bool,
        detach_reset: bool,
        surrogate_alpha: float,
        tau: float,
        decay_input: bool,
        scale_reset: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not supports_klif:
            raise RuntimeError("loaded AsPy native extension lacks KLIF symbols")
        x_seq = _require_nd(x_seq, "x_seq")
        v_init = _require_nd(v_init, "v_init")
        k = _require_nd(k, "k")
        spike_seq, v_seq, v_final, h_seq, v_prev_seq = klif_forward(
            x_seq,
            v_init,
            k,
            v_threshold,
            v_reset,
            hard_reset,
            tau,
            decay_input,
            scale_reset,
        )
        ctx.save_for_backward(x_seq, v_prev_seq, h_seq, spike_seq, k)
        ctx.v_threshold = v_threshold
        ctx.v_reset = v_reset
        ctx.hard_reset = hard_reset
        ctx.detach_reset = detach_reset
        ctx.surrogate_alpha = surrogate_alpha
        ctx.tau = tau
        ctx.decay_input = decay_input
        ctx.scale_reset = scale_reset
        return spike_seq, v_final, v_seq

    @staticmethod
    def backward(
        ctx,
        grad_spike_seq: torch.Tensor | None,
        grad_v_final: torch.Tensor | None,
        grad_v_seq: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        if torch.is_grad_enabled():
            raise RuntimeError("AsPy KLIF supports first-order gradients only")
        x_seq, v_prev_seq, h_seq, spike_seq, k = ctx.saved_tensors
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

        tensors = {
            "x_seq": x_seq,
            "v_prev_seq": v_prev_seq,
            "h_seq": h_seq,
            "spike_seq": spike_seq,
            "grad_spike_seq": grad_spike_seq,
            "grad_v_seq": grad_v_seq,
            "grad_v_final": grad_v_final,
            "k": k,
        }
        tensors = {name: _require_nd(value, name) for name, value in tensors.items()}
        grad_x_seq, grad_v_init, grad_k_partial = klif_backward(
            tensors["x_seq"],
            tensors["v_prev_seq"],
            tensors["h_seq"],
            tensors["spike_seq"],
            tensors["grad_spike_seq"],
            tensors["grad_v_seq"],
            tensors["grad_v_final"],
            tensors["k"],
            ctx.v_threshold,
            ctx.v_reset,
            ctx.hard_reset,
            ctx.detach_reset,
            ctx.surrogate_alpha,
            ctx.tau,
            ctx.decay_input,
            ctx.scale_reset,
        )
        # k is an eight-value repeated native buffer. Return the scalar partial
        # in one lane; expand(...).contiguous() then reduces it to public k.
        grad_k = torch.zeros_like(k)
        grad_k[0] = grad_k_partial.sum()
        return grad_x_seq, grad_v_init, grad_k, None, None, None, None, None, None, None, None


class _AsPyFedSNNDecayLIF(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        current_seq: torch.Tensor,
        membrane_decay: float,
        v_threshold: float,
        surrogate_alpha: float,
    ) -> torch.Tensor:
        if not supports_fedsnn_decay_lif:
            raise RuntimeError(
                "loaded AsPy native extension lacks FedSNN decay-LIF symbols"
            )
        current_seq = _require_nd(current_seq, "current_seq")
        spike_seq, h_seq = fedsnn_decay_lif_forward(
            current_seq,
            membrane_decay,
            v_threshold,
        )
        h_seq = _require_nd(h_seq, "h_seq")
        ctx.save_for_backward(h_seq)
        ctx.membrane_decay = membrane_decay
        ctx.v_threshold = v_threshold
        ctx.surrogate_alpha = surrogate_alpha
        return spike_seq

    @staticmethod
    def backward(
        ctx,
        grad_spike_seq: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, ...]:
        if torch.is_grad_enabled():
            raise RuntimeError("AsPy FedSNN decay-LIF supports first-order gradients only")
        (h_seq,) = ctx.saved_tensors
        if grad_spike_seq is None:
            grad_spike_seq = torch.zeros_like(h_seq)
        else:
            grad_spike_seq = _copy_to_nd(grad_spike_seq.contiguous())
        grad_spike_seq = _require_nd(grad_spike_seq, "grad_spike_seq")
        if not supports_fedsnn_decay_lif:
            raise RuntimeError(
                "loaded AsPy native extension lacks FedSNN decay-LIF symbols"
            )
        grad_current_seq = fedsnn_decay_lif_backward(
            h_seq,
            grad_spike_seq,
            ctx.membrane_decay,
            ctx.v_threshold,
            ctx.surrogate_alpha,
        )
        return grad_current_seq, None, None, None


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


def klif_multi_step(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    k: torch.Tensor,
    v_threshold: float,
    v_reset: float | None,
    detach_reset: bool,
    surrogate_name: str,
    surrogate_alpha: float,
    store_v_seq: bool,
    tau: float,
    decay_input: bool,
    scale_reset: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run fused KLIF with a dynamic scalar ``k`` tensor."""

    if surrogate_name != "atan":
        raise ValueError(f"unsupported AsPy surrogate: {surrogate_name!r}")
    if not supports_klif:
        raise RuntimeError("loaded AsPy native extension lacks KLIF symbols")
    if k.numel() != 1:
        raise ValueError("k must contain exactly one value")
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
            (x_native, x_native.new_zeros((time_steps, padding))), dim=1
        )
        v_native = torch.cat((v_native, v_native.new_full((padding,), reset)), dim=0)

    # AscendC vector DMA moves one 32-byte block for the dynamic scalar. Keep
    # the public tensor scalar while presenting eight readable FP32 values to
    # the native kernel; autograd reduces the repeated view back into k.
    k_native = k.reshape(1).expand(8).contiguous()
    spike_native, v_final_native, v_seq_native = _AsPyKLIF.apply(
        x_native,
        v_native,
        k_native,
        v_threshold,
        reset,
        hard_reset,
        detach_reset,
        surrogate_alpha,
        tau,
        decay_input,
        scale_reset,
    )
    spike_seq = spike_native[:, :neuron_count].reshape_as(x_seq).contiguous()
    v_final = v_final_native[:neuron_count].reshape_as(v_init).contiguous()
    if not store_v_seq:
        return spike_seq, v_final, None
    v_seq = v_seq_native[:, :neuron_count].reshape_as(x_seq).contiguous()
    return spike_seq, v_final, v_seq


def fedsnn_decay_lif(
    current_seq: torch.Tensor,
    membrane_decay: float,
    v_threshold: float,
    surrogate_name: str,
    surrogate_alpha: float,
) -> torch.Tensor:
    """Run the exact stateless FedSNN decay-LIF forward/backward."""

    if surrogate_name != "atan":
        raise ValueError(f"unsupported AsPy surrogate: {surrogate_name!r}")
    if not supports_fedsnn_decay_lif:
        raise RuntimeError("loaded AsPy native extension lacks FedSNN decay-LIF symbols")
    if not isinstance(membrane_decay, float) or not 0.0 <= membrane_decay <= 1.0:
        raise ValueError("membrane_decay must be a float in [0, 1]")
    if not math.isfinite(v_threshold):
        raise ValueError("v_threshold must be finite")
    if not math.isfinite(surrogate_alpha) or surrogate_alpha <= 0.0:
        raise ValueError("surrogate_alpha must be finite and positive")
    if current_seq.ndim < 2:
        raise ValueError("current_seq must be [T, N, ...]")
    time_steps = current_seq.shape[0]
    if time_steps == 0:
        raise ValueError("current_seq time dimension must be non-empty")
    neuron_count = current_seq.numel() // time_steps
    if neuron_count == 0:
        raise ValueError("current_seq flattened time-step size must be non-empty")
    aligned_count = (neuron_count + 7) // 8 * 8
    padding = aligned_count - neuron_count
    current_native = _copy_to_nd(current_seq.reshape(time_steps, neuron_count))
    current_native = _require_nd(current_native, "current_seq")
    if padding:
        current_native = torch.cat(
            (current_native, current_native.new_zeros((time_steps, padding))),
            dim=1,
        )
    spike_native = _AsPyFedSNNDecayLIF.apply(
        current_native,
        membrane_decay,
        v_threshold,
        surrogate_alpha,
    )
    return spike_native[:, :neuron_count].reshape_as(current_seq).contiguous()


__all__ = [
    "fedsnn_decay_lif",
    "if_multi_step",
    "klif_multi_step",
    "lif_multi_step",
    "plif_multi_step",
]

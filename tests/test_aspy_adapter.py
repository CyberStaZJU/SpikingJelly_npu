import importlib
import sys
from types import SimpleNamespace

import pytest
import torch


def install_fake_native(
    monkeypatch,
    *,
    forward,
    backward=None,
    lif_forward=None,
    lif_backward=None,
    plif_forward=None,
    plif_backward=None,
    fedsnn_decay_lif_forward=None,
    fedsnn_decay_lif_backward=None,
    include_fedsnn_symbols=True,
):
    symbols = {
        "if_forward": forward,
        "if_backward": (lambda *args: None) if backward is None else backward,
        "lif_forward": forward if lif_forward is None else lif_forward,
        "lif_backward": (lambda *args: None) if lif_backward is None else lif_backward,
        "plif_forward": forward if plif_forward is None else plif_forward,
        "plif_backward": (lambda *args: None) if plif_backward is None else plif_backward,
    }
    if include_fedsnn_symbols:
        symbols.update(
            fedsnn_decay_lif_forward=(
                forward
                if fedsnn_decay_lif_forward is None
                else fedsnn_decay_lif_forward
            ),
            fedsnn_decay_lif_backward=(
                (lambda *args: None)
                if fedsnn_decay_lif_backward is None
                else fedsnn_decay_lif_backward
            ),
        )
    native = SimpleNamespace(**symbols)
    monkeypatch.setitem(sys.modules, "_spikingjelly_npu_aspy", native)
    sys.modules.pop("spikingjelly_npu_aspy", None)
    return importlib.import_module("spikingjelly_npu_aspy")


def test_aspy_adapter_imports_older_native_without_fedsnn_symbols(monkeypatch):
    adapter = install_fake_native(
        monkeypatch,
        forward=lambda *args: None,
        include_fedsnn_symbols=False,
    )

    assert adapter.supports_fedsnn_decay_lif is False
    with pytest.raises(RuntimeError, match="lacks FedSNN decay-LIF symbols"):
        adapter.fedsnn_decay_lif(
            torch.zeros(2, 1, 1),
            0.75,
            1.0,
            "atan",
            2.0,
        )


def test_aspy_adapter_validates_every_generic_native_tensor(monkeypatch):
    checked = []

    def require_nd(tensor, name):
        checked.append(name)
        return tensor

    def forward(x_seq, v_init, *attributes):
        return (
            torch.ones_like(x_seq),
            torch.zeros_like(x_seq),
            torch.zeros_like(v_init),
            torch.full_like(x_seq, 0.5),
        )

    def backward(h_seq, spike_seq, *args):
        return torch.ones_like(h_seq), torch.ones_like(spike_seq[0])

    adapter = install_fake_native(monkeypatch, forward=forward, backward=backward)
    monkeypatch.setattr(adapter, "_require_nd", require_nd)
    x_seq = torch.zeros(2, 1, 8, requires_grad=True)
    spike_seq, v_final, v_seq = adapter.if_multi_step(
        x_seq,
        torch.zeros(1, 8),
        1.0,
        0.0,
        False,
        "atan",
        2.0,
        True,
    )
    (spike_seq.sum() + v_final.sum() + v_seq.sum()).backward()

    assert checked == [
        "x_seq",
        "v_init",
        "h_seq",
        "spike_seq",
        "grad_spike_seq",
        "grad_v_seq",
        "grad_v_final",
    ]


def test_aspy_adapter_matches_forward_router_contract(monkeypatch):
    calls = []

    def forward(x_seq, v_init, *attributes):
        calls.append((x_seq, v_init, attributes))
        return (
            torch.ones_like(x_seq),
            torch.full_like(x_seq, 0.25),
            torch.full_like(v_init, 0.25),
            torch.full_like(x_seq, 0.5),
        )

    adapter = install_fake_native(monkeypatch, forward=forward)
    x_seq = torch.zeros(3, 2, 4)
    v_init = torch.zeros(2, 4)

    spike_seq, v_final, v_seq = adapter.if_multi_step(
        x_seq,
        v_init,
        1.0,
        None,
        False,
        "atan",
        2.0,
        True,
    )

    torch.testing.assert_close(spike_seq, torch.ones_like(x_seq))
    torch.testing.assert_close(v_final, torch.full_like(v_init, 0.25))
    torch.testing.assert_close(v_seq, torch.full_like(x_seq, 0.25))
    assert calls[0][0].shape == (3, 8)
    assert calls[0][1].shape == (8,)
    assert calls[0][2] == (1.0, 0.0, False)


def test_aspy_adapter_pads_crops_and_dispatches_backward(monkeypatch):
    backward_calls = []

    def forward(x_seq, v_init, *attributes):
        return (
            torch.ones_like(x_seq),
            torch.zeros_like(x_seq),
            torch.zeros_like(v_init),
            torch.full_like(x_seq, 0.5),
        )

    def backward(*args):
        backward_calls.append(args)
        h_seq, _, grad_spike_seq, grad_v_seq, grad_v_final = args[:5]
        assert h_seq.shape == (2, 8)
        assert grad_spike_seq.shape == (2, 8)
        assert grad_v_seq.shape == (2, 8)
        assert grad_v_final.shape == (8,)
        return torch.full_like(h_seq, 2.0), torch.full_like(grad_v_final, 3.0)

    adapter = install_fake_native(
        monkeypatch,
        forward=forward,
        backward=backward,
    )
    x_seq = torch.zeros(2, 1, 1, requires_grad=True)
    v_init = torch.zeros(1, 1, requires_grad=True)

    spike_seq, v_final, v_seq = adapter.if_multi_step(
        x_seq,
        v_init,
        1.0,
        0.5,
        True,
        "atan",
        2.0,
        True,
    )
    (spike_seq.sum() + v_final.sum() + v_seq.sum()).backward()

    torch.testing.assert_close(x_seq.grad, torch.full_like(x_seq, 2.0))
    torch.testing.assert_close(v_init.grad, torch.full_like(v_init, 3.0))
    assert len(backward_calls) == 1


def test_aspy_adapter_rejects_higher_order_gradients(monkeypatch):
    def forward(x_seq, v_init, *attributes):
        return (
            torch.ones_like(x_seq),
            torch.zeros_like(x_seq),
            torch.zeros_like(v_init),
            torch.full_like(x_seq, 0.5),
        )

    def backward(h_seq, spike_seq, *args):
        return torch.ones_like(h_seq), torch.ones_like(spike_seq[0])

    adapter = install_fake_native(monkeypatch, forward=forward, backward=backward)
    x_seq = torch.zeros(2, 1, 1, requires_grad=True)
    spike_seq, _, _ = adapter.if_multi_step(
        x_seq,
        torch.zeros(1, 1),
        1.0,
        0.0,
        False,
        "atan",
        2.0,
        False,
    )

    try:
        torch.autograd.grad(spike_seq.sum(), x_seq, create_graph=True)
    except RuntimeError as error:
        assert "first-order gradients only" in str(error)
    else:
        raise AssertionError("AsPy higher-order gradient request should fail")


def test_aspy_lif_adapter_pads_with_reset_crops_and_dispatches_backward(monkeypatch):
    calls = []

    def lif_forward(x_seq, v_init, *attributes):
        calls.append((x_seq, v_init, attributes))
        return (
            torch.ones_like(x_seq),
            torch.full_like(x_seq, 0.25),
            torch.full_like(v_init, 0.25),
            torch.full_like(x_seq, 0.5),
        )

    def lif_backward(*args):
        h_seq, _, _, _, grad_v_final = args[:5]
        calls.append(args)
        return torch.full_like(h_seq, 2.0), torch.full_like(grad_v_final, 3.0)

    adapter = install_fake_native(
        monkeypatch,
        forward=lambda *args: None,
        lif_forward=lif_forward,
        lif_backward=lif_backward,
    )
    x_seq = torch.zeros(2, 1, 1, requires_grad=True)
    v_init = torch.zeros(1, 1, requires_grad=True)

    spike_seq, v_final, v_seq = adapter.lif_multi_step(
        x_seq,
        v_init,
        1.0,
        0.5,
        True,
        "atan",
        2.0,
        True,
        2.5,
        False,
    )
    (spike_seq.sum() + v_final.sum() + v_seq.sum()).backward()

    forward_call = calls[0]
    assert forward_call[0].shape == (2, 8)
    torch.testing.assert_close(forward_call[1][1:], torch.full((7,), 0.5))
    assert forward_call[2] == (1.0, 0.5, True, 2.5, False)
    torch.testing.assert_close(x_seq.grad, torch.full_like(x_seq, 2.0))
    torch.testing.assert_close(v_init.grad, torch.full_like(v_init, 3.0))


def test_aspy_lif_adapter_rejects_higher_order_gradients(monkeypatch):
    def lif_forward(x_seq, v_init, *attributes):
        return (
            torch.ones_like(x_seq),
            torch.zeros_like(x_seq),
            torch.zeros_like(v_init),
            torch.full_like(x_seq, 0.5),
        )

    def lif_backward(h_seq, spike_seq, *args):
        return torch.ones_like(h_seq), torch.ones_like(spike_seq[0])

    adapter = install_fake_native(
        monkeypatch,
        forward=lambda *args: None,
        lif_forward=lif_forward,
        lif_backward=lif_backward,
    )
    x_seq = torch.zeros(2, 1, 1, requires_grad=True)
    spike_seq, _, _ = adapter.lif_multi_step(
        x_seq,
        torch.zeros(1, 1),
        1.0,
        0.0,
        False,
        "atan",
        2.0,
        False,
        2.0,
        True,
    )

    try:
        torch.autograd.grad(spike_seq.sum(), x_seq, create_graph=True)
    except RuntimeError as error:
        assert "first-order gradients only" in str(error)
    else:
        raise AssertionError("AsPy LIF higher-order gradient request should fail")


def test_aspy_plif_adapter_dynamic_scalar_padding_gradient_and_cropping(monkeypatch):
    calls = []

    def plif_forward(x_seq, v_init, reciprocal_tau, *attributes):
        calls.append((x_seq, v_init, reciprocal_tau, attributes))
        return (
            torch.ones_like(x_seq),
            torch.full_like(x_seq, 0.25),
            torch.full_like(v_init, 0.25),
            torch.full_like(x_seq, 0.5),
            torch.full_like(x_seq, 0.125),
        )

    def plif_backward(*args):
        calls.append(args)
        x_seq, _, _, _, _, _, grad_v_final, reciprocal_tau = args[:8]
        return (
            torch.full_like(x_seq, 2.0),
            torch.full_like(grad_v_final, 3.0),
            torch.arange(1, grad_v_final.numel() + 1, dtype=grad_v_final.dtype).reshape_as(
                grad_v_final
            ),
        )

    adapter = install_fake_native(
        monkeypatch,
        forward=lambda *args: None,
        plif_forward=plif_forward,
        plif_backward=plif_backward,
    )
    x_seq = torch.zeros(2, 1, 1, requires_grad=True)
    v_init = torch.zeros(1, 1, requires_grad=True)
    reciprocal_tau = torch.tensor([0.4], requires_grad=True)

    spike_seq, v_final, v_seq = adapter.plif_multi_step(
        x_seq,
        v_init,
        reciprocal_tau,
        1.0,
        0.5,
        False,
        "atan",
        2.0,
        True,
        True,
    )
    (spike_seq.sum() + v_final.sum() + v_seq.sum()).backward()

    forward_call = calls[0]
    assert forward_call[0].shape == (2, 8)
    torch.testing.assert_close(forward_call[1][1:], torch.full((7,), 0.5))
    torch.testing.assert_close(forward_call[2], reciprocal_tau)
    assert forward_call[2].grad_fn is not None
    assert forward_call[3] == (1.0, 0.5, True, True)
    torch.testing.assert_close(x_seq.grad, torch.full_like(x_seq, 2.0))
    torch.testing.assert_close(v_init.grad, torch.full_like(v_init, 3.0))
    torch.testing.assert_close(reciprocal_tau.grad, torch.tensor([36.0]))


def test_aspy_fedsnn_decay_lif_adapter_pads_crops_and_dispatches_backward(monkeypatch):
    calls = []
    copied = []

    def fedsnn_forward(current_seq, *attributes):
        calls.append((current_seq, attributes))
        return torch.ones_like(current_seq), torch.full_like(current_seq, 0.5)

    def fedsnn_backward(h_seq, grad_spike_seq, *attributes):
        calls.append((h_seq, grad_spike_seq, attributes))
        return torch.full_like(h_seq, 2.0)

    adapter = install_fake_native(
        monkeypatch,
        forward=lambda *args: None,
        fedsnn_decay_lif_forward=fedsnn_forward,
        fedsnn_decay_lif_backward=fedsnn_backward,
    )
    monkeypatch.setattr(
        adapter,
        "_copy_to_nd",
        lambda tensor: copied.append(tensor) or tensor,
    )
    current_seq = torch.zeros(3, 1, 1, requires_grad=True)

    spike_seq = adapter.fedsnn_decay_lif(
        current_seq,
        0.75,
        1.0,
        "atan",
        2.5,
    )
    spike_seq.sum().backward()

    assert calls[0][0].shape == (3, 8)
    assert calls[0][1] == (0.75, 1.0)
    assert calls[1][0].shape == (3, 8)
    assert calls[1][1].shape == (3, 8)
    assert calls[1][2] == (0.75, 1.0, 2.5)
    assert len(copied) == 2
    assert copied[0].shape == (3, 1)
    assert copied[1].shape == (3, 8)
    torch.testing.assert_close(spike_seq, torch.ones_like(current_seq))
    torch.testing.assert_close(current_seq.grad, torch.full_like(current_seq, 2.0))


@pytest.mark.parametrize(
    ("threshold", "alpha", "match"),
    [
        (float("nan"), 2.0, "v_threshold must be finite"),
        (1.0, 0.0, "surrogate_alpha must be finite and positive"),
        (1.0, float("inf"), "surrogate_alpha must be finite and positive"),
    ],
)
def test_aspy_fedsnn_decay_lif_adapter_rejects_invalid_scalars(
    monkeypatch, threshold, alpha, match
):
    calls = []
    adapter = install_fake_native(
        monkeypatch,
        forward=lambda *args: None,
        fedsnn_decay_lif_forward=lambda *args: calls.append(args),
    )

    with pytest.raises(ValueError, match=match):
        adapter.fedsnn_decay_lif(
            torch.zeros(2, 1, 1),
            0.5,
            threshold,
            "atan",
            alpha,
        )
    assert calls == []


@pytest.mark.parametrize(
    ("shape", "match"),
    [
        ((0,), r"\[T, N"),
        ((0, 1), "time dimension must be non-empty"),
        ((2, 0), "flattened time-step size must be non-empty"),
    ],
)
def test_aspy_fedsnn_decay_lif_adapter_rejects_empty_or_rank_one_input(
    monkeypatch, shape, match
):
    calls = []
    adapter = install_fake_native(
        monkeypatch,
        forward=lambda *args: None,
        fedsnn_decay_lif_forward=lambda *args: calls.append(args),
    )

    with pytest.raises(ValueError, match=match):
        adapter.fedsnn_decay_lif(
            torch.zeros(shape),
            0.5,
            1.0,
            "atan",
            2.0,
        )
    assert calls == []


def test_aspy_fedsnn_decay_lif_adapter_rejects_higher_order_gradients(monkeypatch):
    def fedsnn_forward(current_seq, *attributes):
        return torch.ones_like(current_seq), torch.full_like(current_seq, 0.5)

    def fedsnn_backward(h_seq, *args):
        return torch.ones_like(h_seq)

    adapter = install_fake_native(
        monkeypatch,
        forward=lambda *args: None,
        fedsnn_decay_lif_forward=fedsnn_forward,
        fedsnn_decay_lif_backward=fedsnn_backward,
    )
    current_seq = torch.zeros(2, 1, 1, requires_grad=True)
    spike_seq = adapter.fedsnn_decay_lif(current_seq, 0.5, 1.0, "atan", 2.0)

    with pytest.raises(RuntimeError, match="first-order gradients only"):
        torch.autograd.grad(spike_seq.sum(), current_seq, create_graph=True)


def test_aspy_plif_adapter_rejects_higher_order_gradients(monkeypatch):
    def plif_forward(x_seq, v_init, reciprocal_tau, *attributes):
        return (
            torch.ones_like(x_seq),
            torch.zeros_like(x_seq),
            torch.zeros_like(v_init),
            torch.full_like(x_seq, 0.5),
            torch.zeros_like(x_seq),
        )

    def plif_backward(x_seq, *args):
        grad_v_final = args[5]
        return torch.ones_like(x_seq), torch.ones_like(grad_v_final), torch.ones_like(
            grad_v_final
        )

    adapter = install_fake_native(
        monkeypatch,
        forward=lambda *args: None,
        plif_forward=plif_forward,
        plif_backward=plif_backward,
    )
    x_seq = torch.zeros(2, 1, 1, requires_grad=True)
    reciprocal_tau = torch.tensor([0.4], requires_grad=True)
    spike_seq, _, _ = adapter.plif_multi_step(
        x_seq,
        torch.zeros(1, 1),
        reciprocal_tau,
        1.0,
        0.0,
        False,
        "atan",
        2.0,
        False,
        True,
    )

    with pytest.raises(RuntimeError, match="first-order gradients only"):
        torch.autograd.grad(spike_seq.sum(), (x_seq, reciprocal_tau), create_graph=True)

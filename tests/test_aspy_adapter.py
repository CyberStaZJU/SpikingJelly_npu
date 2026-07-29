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
):
    native = SimpleNamespace(
        if_forward=forward,
        if_backward=(lambda *args: None) if backward is None else backward,
        lif_forward=forward if lif_forward is None else lif_forward,
        lif_backward=(lambda *args: None) if lif_backward is None else lif_backward,
        plif_forward=forward if plif_forward is None else plif_forward,
        plif_backward=(lambda *args: None) if plif_backward is None else plif_backward,
    )
    monkeypatch.setitem(sys.modules, "_spikingjelly_npu_aspy", native)
    sys.modules.pop("spikingjelly_npu_aspy", None)
    return importlib.import_module("spikingjelly_npu_aspy")


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

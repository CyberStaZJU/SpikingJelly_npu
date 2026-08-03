import os
import subprocess
import sys

import pytest
import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence

from spikingjelly_npu.sequence import recurrent


def _clone_with_grad(tensor):
    return tensor.detach().clone().requires_grad_()


def _clone_state(state):
    if isinstance(state, tuple):
        return tuple(_clone_with_grad(value) for value in state)
    return _clone_with_grad(state)


def _make_state(module_class, kwargs, batch_size, *, batched=True):
    directions = 2 if kwargs.get("bidirectional", False) else 1
    leading = (kwargs.get("num_layers", 1) * directions,)
    if batched:
        leading += (batch_size,)
    hidden_size = kwargs["hidden_size"]
    if module_class is nn.LSTM:
        projected_size = kwargs.get("proj_size", 0) or hidden_size
        hidden = torch.randn(*leading, projected_size, dtype=torch.float64)
        cell = torch.randn(*leading, hidden_size, dtype=torch.float64)
        return hidden.requires_grad_(), cell.requires_grad_()
    return torch.randn(*leading, hidden_size, dtype=torch.float64).requires_grad_()


def _assert_nested_close(actual, expected):
    if isinstance(actual, tuple):
        assert isinstance(expected, tuple)
        assert len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected, strict=True):
            torch.testing.assert_close(actual_value, expected_value)
        return
    torch.testing.assert_close(actual, expected)


def _state_loss(state):
    if isinstance(state, tuple):
        return sum(value.square().sum() for value in state)
    return state.square().sum()


def _assert_state_grads_close(actual, expected):
    if isinstance(actual, tuple):
        for actual_value, expected_value in zip(actual, expected, strict=True):
            torch.testing.assert_close(actual_value.grad, expected_value.grad)
        return
    torch.testing.assert_close(actual.grad, expected.grad)


def _assert_parameter_grads_close(actual, expected):
    actual_parameters = dict(actual.named_parameters())
    expected_parameters = dict(expected.named_parameters())
    assert actual_parameters.keys() == expected_parameters.keys()
    for name in actual_parameters:
        assert actual_parameters[name].grad is not None
        assert expected_parameters[name].grad is not None
        torch.testing.assert_close(actual_parameters[name].grad, expected_parameters[name].grad)


RECURRENT_CASES = [
    pytest.param(
        recurrent.RNN,
        nn.RNN,
        {
            "input_size": 4,
            "hidden_size": 3,
            "num_layers": 2,
            "nonlinearity": "relu",
            "dropout": 0.0,
            "bidirectional": True,
            "batch_first": False,
        },
        id="rnn-sequence-first",
    ),
    pytest.param(
        recurrent.GRU,
        nn.GRU,
        {
            "input_size": 4,
            "hidden_size": 3,
            "num_layers": 2,
            "dropout": 0.0,
            "bidirectional": True,
            "batch_first": True,
        },
        id="gru-batch-first",
    ),
    pytest.param(
        recurrent.LSTM,
        nn.LSTM,
        {
            "input_size": 4,
            "hidden_size": 5,
            "num_layers": 2,
            "dropout": 0.0,
            "bidirectional": True,
            "batch_first": True,
            "proj_size": 3,
        },
        id="lstm-projected",
    ),
]


@pytest.mark.parametrize("training", [False, True], ids=["eval", "train"])
@pytest.mark.parametrize(("wrapper_class", "torch_class", "kwargs"), RECURRENT_CASES)
def test_dense_recurrent_forward_backward_and_state_dict_parity(
    wrapper_class, torch_class, kwargs, training
):
    torch.manual_seed(11)
    expected_module = torch_class(**kwargs).double().train(training)
    actual_module = wrapper_class(**kwargs).double().train(training)

    assert actual_module.__class__.__bases__ == (torch_class,)
    assert actual_module.state_dict().keys() == expected_module.state_dict().keys()
    actual_module.load_state_dict(expected_module.state_dict())
    expected_module.load_state_dict(actual_module.state_dict())

    batch_size, sequence_length = 2, 5
    shape = (
        (batch_size, sequence_length, kwargs["input_size"])
        if kwargs["batch_first"]
        else (sequence_length, batch_size, kwargs["input_size"])
    )
    expected_input = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
    actual_input = _clone_with_grad(expected_input)
    expected_state = _make_state(torch_class, kwargs, batch_size)
    actual_state = _clone_state(expected_state)

    expected_output, expected_next_state = expected_module(expected_input, expected_state)
    actual_output, actual_next_state = actual_module(actual_input, actual_state)

    torch.testing.assert_close(actual_output, expected_output)
    _assert_nested_close(actual_next_state, expected_next_state)

    expected_loss = expected_output.square().sum() + _state_loss(expected_next_state)
    actual_loss = actual_output.square().sum() + _state_loss(actual_next_state)
    expected_loss.backward()
    actual_loss.backward()

    torch.testing.assert_close(actual_input.grad, expected_input.grad)
    _assert_state_grads_close(actual_state, expected_state)
    _assert_parameter_grads_close(actual_module, expected_module)


@pytest.mark.parametrize(("wrapper_class", "torch_class", "base_kwargs"), RECURRENT_CASES)
def test_recurrent_unbatched_input_parity(wrapper_class, torch_class, base_kwargs):
    kwargs = {
        "input_size": base_kwargs["input_size"],
        "hidden_size": base_kwargs["hidden_size"],
        "num_layers": 1,
        "dropout": 0.0,
        "bidirectional": True,
        "batch_first": True,
    }
    if torch_class is nn.RNN:
        kwargs["nonlinearity"] = "tanh"

    torch.manual_seed(12)
    expected_module = torch_class(**kwargs).double()
    actual_module = wrapper_class(**kwargs).double()
    actual_module.load_state_dict(expected_module.state_dict())

    inputs = torch.randn(5, kwargs["input_size"], dtype=torch.float64)
    expected_state = _make_state(torch_class, kwargs, batch_size=1, batched=False)
    actual_state = tuple(value.detach().clone() for value in expected_state) if isinstance(
        expected_state, tuple
    ) else expected_state.detach().clone()

    expected_output, expected_next_state = expected_module(inputs, expected_state)
    actual_output, actual_next_state = actual_module(inputs.clone(), actual_state)

    torch.testing.assert_close(actual_output, expected_output)
    _assert_nested_close(actual_next_state, expected_next_state)


@pytest.mark.parametrize(("wrapper_class", "torch_class", "base_kwargs"), RECURRENT_CASES)
def test_recurrent_packed_sequence_parity(wrapper_class, torch_class, base_kwargs):
    kwargs = {
        "input_size": base_kwargs["input_size"],
        "hidden_size": base_kwargs["hidden_size"],
        "num_layers": 2,
        "dropout": 0.0,
        "bidirectional": True,
        "batch_first": True,
    }
    if torch_class is nn.RNN:
        kwargs["nonlinearity"] = "relu"

    torch.manual_seed(13)
    expected_module = torch_class(**kwargs).double()
    actual_module = wrapper_class(**kwargs).double()
    actual_module.load_state_dict(expected_module.state_dict())

    lengths = torch.tensor([5, 3, 4])
    inputs = torch.randn(3, 5, kwargs["input_size"], dtype=torch.float64)
    expected_packed = pack_padded_sequence(
        inputs,
        lengths,
        batch_first=True,
        enforce_sorted=False,
    )
    actual_packed = pack_padded_sequence(
        inputs.clone(),
        lengths,
        batch_first=True,
        enforce_sorted=False,
    )
    expected_state = _make_state(torch_class, kwargs, batch_size=3)
    actual_state = _clone_state(expected_state)

    expected_output, expected_next_state = expected_module(expected_packed, expected_state)
    actual_output, actual_next_state = actual_module(actual_packed, actual_state)

    assert isinstance(actual_output, PackedSequence)
    torch.testing.assert_close(actual_output.data, expected_output.data)
    torch.testing.assert_close(actual_output.batch_sizes, expected_output.batch_sizes)
    torch.testing.assert_close(actual_output.sorted_indices, expected_output.sorted_indices)
    torch.testing.assert_close(actual_output.unsorted_indices, expected_output.unsorted_indices)
    _assert_nested_close(actual_next_state, expected_next_state)


def test_recurrent_seeded_dropout_matches_torch_and_changes_with_seed():
    kwargs = {
        "input_size": 4,
        "hidden_size": 5,
        "num_layers": 2,
        "dropout": 0.4,
        "batch_first": True,
    }
    torch.manual_seed(14)
    expected_module = nn.GRU(**kwargs).train()
    actual_module = recurrent.GRU(**kwargs).train()
    actual_module.load_state_dict(expected_module.state_dict())
    inputs = torch.randn(3, 6, 4)

    torch.manual_seed(15)
    expected_output, expected_state = expected_module(inputs)
    torch.manual_seed(15)
    actual_output, actual_state = actual_module(inputs)

    torch.testing.assert_close(actual_output, expected_output)
    torch.testing.assert_close(actual_state, expected_state)

    torch.manual_seed(16)
    different_output, _ = actual_module(inputs)
    assert not torch.equal(actual_output, different_output)


def test_sequence_import_does_not_import_torch_npu():
    code = (
        "import sys; "
        "from spikingjelly_npu.sequence import GRU, Transformer; "
        "GRU(2, 3); Transformer(d_model=4, nhead=2, num_encoder_layers=1, "
        "num_decoder_layers=1); "
        "assert 'torch_npu' not in sys.modules; print('ok')"
    )
    env = os.environ.copy()
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.stdout.strip() == "ok"

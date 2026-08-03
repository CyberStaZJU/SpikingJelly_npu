import os
import subprocess
import sys
from typing import get_args

import pytest
import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence

from spikingjelly_npu.sequence import recurrent


def _clone_with_grad(tensor):
    return tensor.detach().clone().requires_grad_()


def _clone_state(state):
    if isinstance(state, tuple):
        return tuple(_clone_with_grad(value) for value in state)
    return _clone_with_grad(state)


def _float_state(state):
    if isinstance(state, tuple):
        return tuple(_clone_with_grad(value.float()) for value in state)
    return _clone_with_grad(state.float())


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


def _assert_parameters_close(actual, expected):
    actual_parameters = dict(actual.named_parameters())
    expected_parameters = dict(expected.named_parameters())
    assert actual_parameters.keys() == expected_parameters.keys()
    for name in actual_parameters:
        torch.testing.assert_close(actual_parameters[name], expected_parameters[name])


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

SCRIPT_CASES = [
    pytest.param(recurrent.RNN, nn.RNN, id="rnn"),
    pytest.param(recurrent.GRU, nn.GRU, id="gru"),
    pytest.param(recurrent.LSTM, nn.LSTM, id="lstm"),
]


class _DenseRNN(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        return self.module(inputs, state)


class _PackedRNN(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs: PackedSequence, state: Tensor) -> tuple[PackedSequence, Tensor]:
        return self.module(inputs, state)


class _DenseLSTM(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(
        self, inputs: Tensor, state: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        return self.module(inputs, state)


class _PackedLSTM(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(
        self, inputs: PackedSequence, state: tuple[Tensor, Tensor]
    ) -> tuple[PackedSequence, tuple[Tensor, Tensor]]:
        return self.module(inputs, state)


@pytest.mark.parametrize(("wrapper_class", "torch_class"), SCRIPT_CASES)
def test_recurrent_forward_redeclares_torch_overloads(wrapper_class, torch_class):
    assert wrapper_class.__bases__ == (torch_class,)
    assert "forward" in wrapper_class.__dict__
    assert wrapper_class.forward is not torch_class.forward

    overloads = torch._jit_internal._get_overloaded_methods(
        wrapper_class.forward,
        wrapper_class,
    )
    assert overloads is not None
    assert len(overloads) == 2

    annotations = [overload.__annotations__ for overload in overloads]
    dense_return = get_args(annotations[0]["return"])
    packed_return = get_args(annotations[1]["return"])
    assert annotations[0]["input"] is Tensor
    assert dense_return[0] is Tensor
    assert annotations[1]["input"] is PackedSequence
    assert packed_return[0] is PackedSequence
    assert annotations[0]["hx"] == annotations[1]["hx"]
    assert dense_return[1] == packed_return[1]


def _script_recurrent_container(module_class, module):
    container_class = _DenseLSTM if module_class is nn.LSTM else _DenseRNN
    packed_container_class = _PackedLSTM if module_class is nn.LSTM else _PackedRNN
    try:
        return (
            torch.jit.script(container_class(module)),
            torch.jit.script(packed_container_class(module)),
        )
    except (AssertionError, RuntimeError) as error:
        message = str(error)
        unsupported = (
            "Overloads are not usable when a module is redeclared",
            "try blocks aren't supported",
            "try blocks aren’t supported",
            "Unknown type name 'object'",
        )
        if not any(reason in message for reason in unsupported):
            raise
        pytest.skip(f"installed torch cannot script this recurrent subclass: {error}")


@pytest.mark.parametrize(("wrapper_class", "torch_class"), SCRIPT_CASES)
def test_recurrent_torchscript_dense_and_packed(wrapper_class, torch_class):
    kwargs = {
        "input_size": 3,
        "hidden_size": 4,
        "num_layers": 1,
        "dropout": 0.0,
        "batch_first": True,
    }
    torch.manual_seed(10)
    eager = wrapper_class(**kwargs).eval()
    scripted_dense, scripted_packed = _script_recurrent_container(torch_class, eager)

    inputs = torch.randn(3, 5, kwargs["input_size"])
    state = _make_state(torch_class, kwargs, batch_size=3)
    if isinstance(state, tuple):
        state = tuple(value.float() for value in state)
    else:
        state = state.float()
    expected_dense = eager(inputs, state)
    actual_dense = scripted_dense(inputs, state)
    _assert_nested_close(actual_dense, expected_dense)

    packed = pack_padded_sequence(
        inputs,
        torch.tensor([5, 3, 4]),
        batch_first=True,
        enforce_sorted=False,
    )
    expected_packed, expected_state = eager(packed, state)
    actual_packed, actual_state = scripted_packed(packed, state)

    assert type(actual_packed).__name__ == "PackedSequence"
    torch.testing.assert_close(actual_packed.data, expected_packed.data)
    torch.testing.assert_close(actual_packed.batch_sizes, expected_packed.batch_sizes)
    torch.testing.assert_close(actual_packed.sorted_indices, expected_packed.sorted_indices)
    torch.testing.assert_close(actual_packed.unsorted_indices, expected_packed.unsorted_indices)
    _assert_nested_close(actual_state, expected_state)


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
    actual_state = (
        tuple(value.detach().clone() for value in expected_state)
        if isinstance(expected_state, tuple)
        else expected_state.detach().clone()
    )

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


FORCED_FALLBACK_CASES = [
    pytest.param(
        recurrent.RNN,
        nn.RNN,
        {
            "input_size": 4,
            "hidden_size": 3,
            "num_layers": 2,
            "nonlinearity": "tanh",
            "bias": False,
            "batch_first": False,
            "dropout": 0.0,
            "bidirectional": True,
        },
        id="rnn-tanh-unbiased-bidirectional",
    ),
    pytest.param(
        recurrent.RNN,
        nn.RNN,
        {
            "input_size": 4,
            "hidden_size": 3,
            "num_layers": 1,
            "nonlinearity": "relu",
            "batch_first": True,
        },
        id="rnn-relu-batch-first",
    ),
    pytest.param(
        recurrent.GRU,
        nn.GRU,
        {
            "input_size": 4,
            "hidden_size": 3,
            "num_layers": 2,
            "batch_first": True,
            "dropout": 0.0,
            "bidirectional": True,
        },
        id="gru-bidirectional",
    ),
    pytest.param(
        recurrent.LSTM,
        nn.LSTM,
        {
            "input_size": 4,
            "hidden_size": 5,
            "num_layers": 2,
            "batch_first": True,
            "dropout": 0.0,
            "bidirectional": True,
        },
        id="lstm-ordinary-bidirectional",
    ),
    pytest.param(
        recurrent.LSTM,
        nn.LSTM,
        {
            "input_size": 4,
            "hidden_size": 5,
            "proj_size": 3,
            "num_layers": 2,
            "batch_first": False,
            "dropout": 0.0,
            "bidirectional": True,
        },
        id="lstm-projected-bidirectional",
    ),
]


def _force_fp32_fallback(monkeypatch):
    monkeypatch.setattr(recurrent, "_should_use_fp32_npu_fallback", lambda input: True)


def _forward_backward(module, inputs, state):
    output, next_state = module(inputs, state)
    output_tensor = output.data if isinstance(output, PackedSequence) else output
    loss = output_tensor.square().sum() + _state_loss(next_state)
    loss.backward()
    return output, next_state


@pytest.mark.parametrize(("wrapper_class", "torch_class", "kwargs"), FORCED_FALLBACK_CASES)
def test_forced_fp32_fallback_dense_forward_backward_and_update_parity(
    monkeypatch, wrapper_class, torch_class, kwargs
):
    _force_fp32_fallback(monkeypatch)
    torch.manual_seed(20)
    expected_module = torch_class(**kwargs)
    actual_module = wrapper_class(**kwargs)
    actual_module.load_state_dict(expected_module.state_dict())
    parameter_ids = {name: id(value) for name, value in actual_module.named_parameters()}

    batch_size, sequence_length = 3, 4
    shape = (
        (batch_size, sequence_length, kwargs["input_size"])
        if kwargs.get("batch_first", False)
        else (sequence_length, batch_size, kwargs["input_size"])
    )
    expected_input = torch.randn(*shape, requires_grad=True)
    actual_input = _clone_with_grad(expected_input)
    expected_state = _float_state(_make_state(torch_class, kwargs, batch_size))
    actual_state = _clone_state(expected_state)

    expected_output, expected_next_state = _forward_backward(
        expected_module, expected_input, expected_state
    )
    actual_output, actual_next_state = _forward_backward(actual_module, actual_input, actual_state)

    torch.testing.assert_close(actual_output, expected_output, rtol=2e-5, atol=2e-6)
    _assert_nested_close(actual_next_state, expected_next_state)
    torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=3e-5, atol=3e-6)
    _assert_state_grads_close(actual_state, expected_state)
    _assert_parameter_grads_close(actual_module, expected_module)
    expected_optimizer = torch.optim.SGD(expected_module.parameters(), lr=0.05, momentum=0.2)
    actual_optimizer = torch.optim.SGD(actual_module.parameters(), lr=0.05, momentum=0.2)
    expected_optimizer.step()
    actual_optimizer.step()
    _assert_parameters_close(actual_module, expected_module)
    assert parameter_ids == {name: id(value) for name, value in actual_module.named_parameters()}


@pytest.mark.parametrize(("wrapper_class", "torch_class"), SCRIPT_CASES)
def test_forced_fp32_fallback_seeded_inter_layer_dropout_parity(
    monkeypatch, wrapper_class, torch_class
):
    _force_fp32_fallback(monkeypatch)
    kwargs = {
        "input_size": 3,
        "hidden_size": 4,
        "num_layers": 2,
        "dropout": 0.35,
        "batch_first": True,
        "bidirectional": True,
    }
    torch.manual_seed(21)
    expected_module = torch_class(**kwargs).train()
    actual_module = wrapper_class(**kwargs).train()
    actual_module.load_state_dict(expected_module.state_dict())
    expected_input = torch.randn(3, 5, kwargs["input_size"], requires_grad=True)
    actual_input = _clone_with_grad(expected_input)
    expected_state = _float_state(_make_state(torch_class, kwargs, batch_size=3))
    actual_state = _clone_state(expected_state)

    torch.manual_seed(210)
    expected_output, expected_next_state = _forward_backward(
        expected_module, expected_input, expected_state
    )
    torch.manual_seed(210)
    actual_output, actual_next_state = _forward_backward(actual_module, actual_input, actual_state)

    torch.testing.assert_close(actual_output, expected_output, rtol=2e-5, atol=2e-6)
    _assert_nested_close(actual_next_state, expected_next_state)
    torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=3e-5, atol=3e-6)
    _assert_state_grads_close(actual_state, expected_state)
    _assert_parameter_grads_close(actual_module, expected_module)


@pytest.mark.parametrize(
    ("wrapper_class", "torch_class", "kwargs"),
    [
        pytest.param(
            recurrent.RNN,
            nn.RNN,
            {
                "input_size": 3,
                "hidden_size": 4,
                "nonlinearity": "relu",
                "bidirectional": True,
                "batch_first": True,
            },
            id="rnn-relu",
        ),
        pytest.param(
            recurrent.GRU,
            nn.GRU,
            {
                "input_size": 3,
                "hidden_size": 4,
                "bidirectional": True,
                "batch_first": True,
            },
            id="gru",
        ),
        pytest.param(
            recurrent.LSTM,
            nn.LSTM,
            {
                "input_size": 3,
                "hidden_size": 5,
                "proj_size": 3,
                "bidirectional": True,
                "batch_first": True,
            },
            id="projected-lstm",
        ),
    ],
)
def test_forced_fp32_fallback_unbatched_supplied_and_default_state(
    monkeypatch, wrapper_class, torch_class, kwargs
):
    _force_fp32_fallback(monkeypatch)
    torch.manual_seed(21)
    expected_module = torch_class(**kwargs)
    actual_module = wrapper_class(**kwargs)
    actual_module.load_state_dict(expected_module.state_dict())
    inputs = torch.randn(5, kwargs["input_size"])

    supplied_state = _float_state(_make_state(torch_class, kwargs, batch_size=1, batched=False))
    actual_supplied_state = _clone_state(supplied_state)
    expected_input = inputs.detach().clone().requires_grad_()
    actual_input = _clone_with_grad(expected_input)
    expected_supplied = expected_module(expected_input, supplied_state)
    actual_supplied = actual_module(actual_input, actual_supplied_state)
    _assert_nested_close(actual_supplied, expected_supplied)
    expected_supplied_loss = expected_supplied[0].square().sum() + _state_loss(expected_supplied[1])
    actual_supplied_loss = actual_supplied[0].square().sum() + _state_loss(actual_supplied[1])
    expected_supplied_loss.backward()
    actual_supplied_loss.backward()
    torch.testing.assert_close(actual_input.grad, expected_input.grad)
    _assert_state_grads_close(actual_supplied_state, supplied_state)
    _assert_parameter_grads_close(actual_module, expected_module)
    expected_module.zero_grad(set_to_none=True)
    actual_module.zero_grad(set_to_none=True)

    expected_default = expected_module(inputs)
    actual_default = actual_module(inputs.clone())
    _assert_nested_close(actual_default, expected_default)


@pytest.mark.parametrize(
    ("wrapper_class", "torch_class", "kwargs"),
    [
        pytest.param(
            recurrent.RNN,
            nn.RNN,
            {
                "input_size": 3,
                "hidden_size": 4,
                "num_layers": 2,
                "nonlinearity": "tanh",
                "bidirectional": True,
                "batch_first": True,
            },
            id="rnn",
        ),
        pytest.param(
            recurrent.GRU,
            nn.GRU,
            {
                "input_size": 3,
                "hidden_size": 4,
                "num_layers": 2,
                "bidirectional": True,
                "batch_first": True,
            },
            id="gru",
        ),
        pytest.param(
            recurrent.LSTM,
            nn.LSTM,
            {
                "input_size": 3,
                "hidden_size": 5,
                "proj_size": 3,
                "num_layers": 2,
                "bidirectional": True,
                "batch_first": True,
            },
            id="projected-lstm",
        ),
    ],
)
def test_forced_fp32_fallback_packed_unsorted_remainder_singleton_parity(
    monkeypatch, wrapper_class, torch_class, kwargs
):
    _force_fp32_fallback(monkeypatch)
    for case_index, lengths in enumerate((torch.tensor([5, 2, 4]), torch.tensor([1]))):
        torch.manual_seed(22 + case_index)
        expected_module = torch_class(**kwargs)
        actual_module = wrapper_class(**kwargs)
        actual_module.load_state_dict(expected_module.state_dict())
        batch_size = len(lengths)
        dense = torch.randn(batch_size, int(lengths.max()), kwargs["input_size"])
        expected_data = dense.detach().clone().requires_grad_()
        actual_data = _clone_with_grad(expected_data)
        expected_input = pack_padded_sequence(
            expected_data,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        actual_input = pack_padded_sequence(
            actual_data,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        expected_input.data.retain_grad()
        actual_input.data.retain_grad()
        expected_state = _float_state(_make_state(torch_class, kwargs, batch_size))
        actual_state = _clone_state(expected_state)

        expected_output, expected_next_state = expected_module(expected_input, expected_state)
        actual_output, actual_next_state = actual_module(actual_input, actual_state)
        torch.testing.assert_close(actual_output.data, expected_output.data, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(actual_output.batch_sizes, expected_output.batch_sizes)
        torch.testing.assert_close(actual_output.sorted_indices, expected_output.sorted_indices)
        torch.testing.assert_close(actual_output.unsorted_indices, expected_output.unsorted_indices)
        _assert_nested_close(actual_next_state, expected_next_state)
        if batch_size > 1:
            assert expected_input.sorted_indices is not None
            assert expected_input.unsorted_indices is not None
            sorted_state = (
                tuple(
                    value.index_select(1, expected_input.sorted_indices) for value in expected_state
                )
                if isinstance(expected_state, tuple)
                else expected_state.index_select(1, expected_input.sorted_indices)
            )
            sorted_expected_module = torch_class(**kwargs)
            sorted_expected_module.load_state_dict(expected_module.state_dict())
            sorted_hidden = sorted_expected_module(
                PackedSequence(
                    expected_input.data.detach(),
                    expected_input.batch_sizes,
                    None,
                    None,
                ),
                sorted_state,
            )[1]
            unsorted_hidden = (
                tuple(
                    value.index_select(1, expected_input.unsorted_indices)
                    for value in sorted_hidden
                )
                if isinstance(sorted_hidden, tuple)
                else sorted_hidden.index_select(1, expected_input.unsorted_indices)
            )
            _assert_nested_close(actual_next_state, unsorted_hidden)

        expected_loss = expected_output.data.square().sum() + _state_loss(expected_next_state)
        actual_loss = actual_output.data.square().sum() + _state_loss(actual_next_state)
        expected_loss.backward()
        actual_loss.backward()
        torch.testing.assert_close(actual_input.data.grad, expected_input.data.grad)
        torch.testing.assert_close(actual_data.grad, expected_data.grad)
        _assert_state_grads_close(actual_state, expected_state)
        _assert_parameter_grads_close(actual_module, expected_module)


@pytest.mark.parametrize(
    ("wrapper_class", "fused_name"),
    [
        pytest.param(recurrent.RNN, "rnn_tanh", id="rnn-tanh"),
        pytest.param(recurrent.RNN, "rnn_relu", id="rnn-relu"),
        pytest.param(recurrent.GRU, "gru", id="gru"),
        pytest.param(recurrent.LSTM, "lstm", id="lstm"),
    ],
)
def test_forced_fp32_fallback_bypasses_relevant_vf_fused_call(
    monkeypatch, wrapper_class, fused_name
):
    _force_fp32_fallback(monkeypatch)

    def fail_fused(*args, **kwargs):
        raise AssertionError(f"torch._VF.{fused_name} must be bypassed")

    monkeypatch.setattr(torch._VF, fused_name, fail_fused)
    kwargs = {"input_size": 3, "hidden_size": 4, "batch_first": True}
    if fused_name == "rnn_relu":
        kwargs["nonlinearity"] = "relu"
    module = wrapper_class(**kwargs)
    output, state = module(torch.randn(2, 5, 3))
    assert output.shape[:2] == (2, 5)
    assert state is not None


@pytest.mark.parametrize(("wrapper_class", "torch_class"), SCRIPT_CASES)
def test_normal_cpu_recurrent_delegates_immediately_to_super(
    monkeypatch, wrapper_class, torch_class
):
    calls = []
    original = torch_class.forward

    def observed_forward(self, input, hx=None):
        calls.append((self, input, hx))
        return original(self, input, hx)

    monkeypatch.setattr(torch_class, "forward", observed_forward)
    module = wrapper_class(3, 4, batch_first=True)
    inputs = torch.randn(2, 5, 3)
    expected = original(module, inputs)
    actual = module(inputs)

    assert len(calls) == 1
    assert calls[0][0] is module
    assert calls[0][1] is inputs
    _assert_nested_close(actual, expected)


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

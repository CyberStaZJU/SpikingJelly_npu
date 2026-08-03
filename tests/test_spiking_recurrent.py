import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from spikingjelly_npu.activation_based import surrogate
from spikingjelly_npu.activation_based.recurrent import (
    SpikingGRU,
    SpikingGRUCell,
    SpikingLSTM,
    SpikingLSTMCell,
    SpikingRNN,
    SpikingRNNCell,
)


class ShiftedSigmoid(nn.Module):
    def __init__(self, shift: float) -> None:
        super().__init__()
        self.shift = shift

    def forward(self, x):
        return torch.sigmoid(x + self.shift)


CELL_CASES = [
    (SpikingRNNCell, {}),
    (SpikingGRUCell, {}),
    (SpikingLSTMCell, {}),
]
SEQUENCE_CASES = [
    (SpikingRNN, {}),
    (SpikingGRU, {}),
    (SpikingLSTM, {}),
]


def _state_tensors(state):
    return state if isinstance(state, tuple) else (state,)


def _clone_state(state, *, requires_grad=False):
    values = tuple(
        value.detach().clone().requires_grad_(requires_grad) for value in _state_tensors(state)
    )
    return values if isinstance(state, tuple) else values[0]


def _assert_state_close(actual, expected, **kwargs):
    assert isinstance(actual, tuple) == isinstance(expected, tuple)
    for actual_value, expected_value in zip(
        _state_tensors(actual), _state_tensors(expected), strict=True
    ):
        torch.testing.assert_close(actual_value, expected_value, **kwargs)


def _zero_state(module, batch_size, *, dtype=torch.double):
    shape = (module.num_layers * module.num_directions, batch_size, module.hidden_size)
    hidden = torch.zeros(shape, dtype=dtype)
    if isinstance(module, SpikingLSTM):
        return hidden, hidden.clone()
    return hidden


def _cell_from_sequence(module, layer_index, direction):
    input_size = (
        module.input_size
        if layer_index == 0
        else module.hidden_size * module.num_directions
    )
    suffix = "_reverse" if direction else ""
    if isinstance(module, SpikingLSTM):
        cell = SpikingLSTMCell(
            input_size,
            module.hidden_size,
            bias=module.bias,
            surrogate_function1=copy.deepcopy(module.surrogate_function1),
            surrogate_function2=copy.deepcopy(module.surrogate_function2),
        )
    elif isinstance(module, SpikingGRU):
        cell = SpikingGRUCell(
            input_size,
            module.hidden_size,
            bias=module.bias,
            surrogate_function1=copy.deepcopy(module.surrogate_function1),
            surrogate_function2=copy.deepcopy(module.surrogate_function2),
        )
    else:
        cell = SpikingRNNCell(
            input_size,
            module.hidden_size,
            bias=module.bias,
            surrogate_function=copy.deepcopy(module.surrogate_function),
        )
    state = {
        "weight_ih": getattr(module, f"weight_ih_l{layer_index}{suffix}").detach(),
        "weight_hh": getattr(module, f"weight_hh_l{layer_index}{suffix}").detach(),
    }
    if module.bias:
        state.update(
            {
                "bias_ih": getattr(
                    module, f"bias_ih_l{layer_index}{suffix}"
                ).detach(),
                "bias_hh": getattr(
                    module, f"bias_hh_l{layer_index}{suffix}"
                ).detach(),
            }
        )
    cell.load_state_dict(state)
    return cell.to(device=module.weight_ih_l0.device, dtype=module.weight_ih_l0.dtype)


def _loop_reference(module, input, state):
    layer_input = input
    final_hidden = []
    final_cell = []
    reference_cells = []
    if isinstance(state, tuple):
        hidden_state, cell_state = state
    else:
        hidden_state = state

    for layer_index in range(module.num_layers):
        direction_outputs = []
        for direction in range(module.num_directions):
            state_index = layer_index * module.num_directions + direction
            cell = _cell_from_sequence(module, layer_index, direction)
            reference_cells.append(cell)
            if isinstance(module, SpikingLSTM):
                current = (hidden_state[state_index], cell_state[state_index])
            else:
                current = hidden_state[state_index]
            time_indices = (
                range(layer_input.shape[0])
                if direction == 0
                else range(layer_input.shape[0] - 1, -1, -1)
            )
            outputs = []
            for time_index in time_indices:
                current = cell(layer_input[time_index], current)
                outputs.append(current[0] if isinstance(current, tuple) else current)
            if direction:
                outputs.reverse()
            direction_outputs.append(torch.stack(outputs))
            if isinstance(current, tuple):
                final_hidden.append(current[0])
                final_cell.append(current[1])
            else:
                final_hidden.append(current)
        layer_input = (
            direction_outputs[0]
            if module.num_directions == 1
            else torch.cat(direction_outputs, dim=-1)
        )
    final_state = torch.stack(final_hidden)
    if isinstance(module, SpikingLSTM):
        final_state = final_state, torch.stack(final_cell)
    return layer_input, final_state, reference_cells


@pytest.mark.parametrize(("cell_type", "kwargs"), CELL_CASES)
def test_cell_subclasses_torch_cells_and_has_exact_parameter_keys(cell_type, kwargs):
    cell = cell_type(3, 4, **kwargs)
    expected_base = {
        SpikingRNNCell: nn.RNNCell,
        SpikingGRUCell: nn.GRUCell,
        SpikingLSTMCell: nn.LSTMCell,
    }[cell_type]
    assert isinstance(cell, expected_base)
    assert tuple(cell.state_dict()) == ("weight_ih", "weight_hh", "bias_ih", "bias_hh")
    if cell_type is SpikingRNNCell:
        assert not hasattr(cell, "nonlinearity")


@pytest.mark.parametrize(("cell_type", "kwargs"), CELL_CASES)
def test_default_surrogates_are_fresh_per_cell(cell_type, kwargs):
    first = cell_type(2, 3, **kwargs)
    second = cell_type(2, 3, **kwargs)
    if cell_type is SpikingRNNCell:
        assert isinstance(first.surrogate_function, surrogate.ATan)
        assert first.surrogate_function is not second.surrogate_function
    else:
        assert isinstance(first.surrogate_function1, surrogate.ATan)
        assert first.surrogate_function1 is not second.surrogate_function1
        assert first.surrogate_function2 is first.surrogate_function1


def test_spiking_rnn_cell_equation_forward_and_gradient():
    activation = ShiftedSigmoid(0.2)
    cell = SpikingRNNCell(2, 2, surrogate_function=activation).double()
    with torch.no_grad():
        cell.weight_ih.copy_(torch.tensor([[0.4, -0.2], [0.1, 0.3]], dtype=torch.double))
        cell.weight_hh.copy_(torch.tensor([[0.5, 0.2], [-0.4, 0.6]], dtype=torch.double))
        cell.bias_ih.copy_(torch.tensor([0.1, -0.3], dtype=torch.double))
        cell.bias_hh.copy_(torch.tensor([-0.2, 0.4], dtype=torch.double))
    input = torch.tensor([[0.6, -0.7]], dtype=torch.double, requires_grad=True)
    hidden = torch.tensor([[0.2, 0.5]], dtype=torch.double, requires_grad=True)
    reference_input = input.detach().clone().requires_grad_()
    reference_hidden = hidden.detach().clone().requires_grad_()

    actual = cell(input, hidden)
    expected = activation(
        F.linear(reference_input, cell.weight_ih, cell.bias_ih)
        + F.linear(reference_hidden, cell.weight_hh, cell.bias_hh)
    )
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(input.grad, reference_input.grad)
    torch.testing.assert_close(hidden.grad, reference_hidden.grad)


def test_spiking_gru_cell_gate_order_and_pytorch_efficient_equation():
    gate = ShiftedSigmoid(0.1)
    candidate_gate = ShiftedSigmoid(-0.35)
    cell = SpikingGRUCell(
        1,
        1,
        surrogate_function1=gate,
        surrogate_function2=candidate_gate,
    ).double()
    with torch.no_grad():
        cell.weight_ih.copy_(torch.tensor([[0.7], [-0.6], [0.8]], dtype=torch.double))
        cell.weight_hh.copy_(torch.tensor([[-0.2], [0.5], [0.9]], dtype=torch.double))
        cell.bias_ih.copy_(torch.tensor([0.05, 0.15, -0.25], dtype=torch.double))
        cell.bias_hh.copy_(torch.tensor([-0.1, 0.2, 0.3], dtype=torch.double))
    input = torch.tensor([[0.4]], dtype=torch.double, requires_grad=True)
    hidden = torch.tensor([[-0.3]], dtype=torch.double, requires_grad=True)
    reference_input = input.detach().clone().requires_grad_()
    reference_hidden = hidden.detach().clone().requires_grad_()

    input_r, input_z, input_n = F.linear(
        reference_input, cell.weight_ih, cell.bias_ih
    ).chunk(3, dim=-1)
    hidden_r, hidden_z, hidden_n = F.linear(
        reference_hidden, cell.weight_hh, cell.bias_hh
    ).chunk(3, dim=-1)
    reset = gate(input_r + hidden_r)
    update = gate(input_z + hidden_z)
    candidate = candidate_gate(input_n + reset * hidden_n)
    expected = (1.0 - update) * candidate + update * reference_hidden
    actual = cell(input, hidden)

    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(input.grad, reference_input.grad)
    torch.testing.assert_close(hidden.grad, reference_hidden.grad)


def test_spiking_lstm_cell_gate_order_clamp_no_tanh_and_clamp_gradient():
    gate = ShiftedSigmoid(0.15)
    candidate_gate = ShiftedSigmoid(-0.2)
    cell = SpikingLSTMCell(
        1,
        1,
        surrogate_function1=gate,
        surrogate_function2=candidate_gate,
    ).double()
    with torch.no_grad():
        cell.weight_ih.copy_(torch.tensor([[0.4], [-0.7], [0.9], [0.6]], dtype=torch.double))
        cell.weight_hh.copy_(torch.tensor([[0.5], [0.3], [-0.2], [0.8]], dtype=torch.double))
        cell.bias_ih.copy_(torch.tensor([0.1, -0.1, 0.2, 0.3], dtype=torch.double))
        cell.bias_hh.copy_(torch.tensor([-0.2, 0.4, 0.1, -0.3], dtype=torch.double))
    input = torch.tensor([[0.8]], dtype=torch.double, requires_grad=True)
    hidden = torch.tensor([[0.25]], dtype=torch.double, requires_grad=True)
    previous_cell = torch.tensor([[2.0]], dtype=torch.double, requires_grad=True)
    reference_input = input.detach().clone().requires_grad_()
    reference_hidden = hidden.detach().clone().requires_grad_()
    reference_cell = previous_cell.detach().clone().requires_grad_()

    gates = F.linear(reference_input, cell.weight_ih, cell.bias_ih) + F.linear(
        reference_hidden, cell.weight_hh, cell.bias_hh
    )
    input_gate, forget_gate, candidate, output_gate = gates.chunk(4, dim=-1)
    input_gate = gate(input_gate)
    forget_gate = gate(forget_gate)
    candidate = candidate_gate(candidate)
    output_gate = gate(output_gate)
    raw_cell = forget_gate * reference_cell + input_gate * candidate
    expected_cell = torch.clamp_max(raw_cell, 1.0)
    expected_hidden = output_gate * expected_cell

    actual_hidden, actual_cell = cell(input, (hidden, previous_cell))
    assert raw_cell.item() > 1.0
    torch.testing.assert_close(actual_cell, expected_cell)
    torch.testing.assert_close(actual_hidden, expected_hidden)
    assert not torch.allclose(actual_hidden, output_gate * torch.tanh(expected_cell))
    (actual_hidden + actual_cell).sum().backward()
    (expected_hidden + expected_cell).sum().backward()
    torch.testing.assert_close(input.grad, reference_input.grad)
    torch.testing.assert_close(hidden.grad, reference_hidden.grad)
    torch.testing.assert_close(previous_cell.grad, reference_cell.grad)
    assert torch.equal(previous_cell.grad, torch.zeros_like(previous_cell.grad))


@pytest.mark.parametrize(("cell_type", "kwargs"), CELL_CASES)
def test_cells_support_unbatched_input_and_default_zero_state(cell_type, kwargs):
    torch.manual_seed(1)
    cell = cell_type(3, 4, **kwargs).double()
    unbatched = torch.randn(3, dtype=torch.double)
    batched = unbatched.unsqueeze(0)
    actual = cell(unbatched)
    expected = cell(batched)
    if isinstance(actual, tuple):
        assert actual[0].shape == actual[1].shape == (4,)
        torch.testing.assert_close(actual[0], expected[0].squeeze(0))
        torch.testing.assert_close(actual[1], expected[1].squeeze(0))
    else:
        assert actual.shape == (4,)
        torch.testing.assert_close(actual, expected.squeeze(0))


@pytest.mark.parametrize(("module_type", "kwargs"), SEQUENCE_CASES)
def test_sequence_matches_explicit_cell_loop_forward_and_gradients(module_type, kwargs):
    torch.manual_seed(2)
    module = module_type(3, 4, num_layers=2, bidirectional=True, **kwargs).double()
    reference = copy.deepcopy(module)
    input = torch.randn(5, 3, 3, dtype=torch.double, requires_grad=True)
    reference_input = input.detach().clone().requires_grad_()
    state = _zero_state(module, input.shape[1])
    state = _clone_state(state, requires_grad=True)
    reference_state = _clone_state(state, requires_grad=True)

    actual_output, actual_state = module(input, state)
    expected_output, expected_state, reference_cells = _loop_reference(
        reference, reference_input, reference_state
    )
    torch.testing.assert_close(actual_output, expected_output)
    _assert_state_close(actual_state, expected_state)

    actual_loss = actual_output.square().mean() + sum(
        value.square().mean() for value in _state_tensors(actual_state)
    )
    expected_loss = expected_output.square().mean() + sum(
        value.square().mean() for value in _state_tensors(expected_state)
    )
    actual_loss.backward()
    expected_loss.backward()
    torch.testing.assert_close(input.grad, reference_input.grad)
    for actual_value, expected_value in zip(
        _state_tensors(state), _state_tensors(reference_state), strict=True
    ):
        torch.testing.assert_close(actual_value.grad, expected_value.grad)
    reference_parameters = [
        parameter for cell in reference_cells for parameter in cell.parameters()
    ]
    for actual_parameter, expected_parameter in zip(
        module.parameters(), reference_parameters, strict=True
    ):
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad)


@pytest.mark.parametrize(("module_type", "kwargs"), SEQUENCE_CASES)
def test_sequence_state_dict_uses_exact_flat_torch_names(module_type, kwargs):
    module = module_type(3, 4, num_layers=2, bidirectional=True, **kwargs)
    expected = []
    for layer_index in range(2):
        for suffix in ("", "_reverse"):
            expected.extend(
                [
                    f"weight_ih_l{layer_index}{suffix}",
                    f"weight_hh_l{layer_index}{suffix}",
                    f"bias_ih_l{layer_index}{suffix}",
                    f"bias_hh_l{layer_index}{suffix}",
                ]
            )
    assert tuple(module.state_dict()) == tuple(expected)
    assert not any("cells" in name or "surrogate" in name for name in module.state_dict())
    assert len(module.all_weights) == 4


@pytest.mark.parametrize(("module_type", "kwargs"), SEQUENCE_CASES)
def test_batch_first_matches_time_major(module_type, kwargs):
    torch.manual_seed(3)
    time_major = module_type(3, 4, num_layers=2, bidirectional=True, **kwargs).double()
    batch_first = module_type(
        3,
        4,
        num_layers=2,
        bidirectional=True,
        batch_first=True,
        **kwargs,
    ).double()
    batch_first.load_state_dict(time_major.state_dict())
    input = torch.randn(5, 2, 3, dtype=torch.double)
    state = _zero_state(time_major, input.shape[1])

    expected_output, expected_state = time_major(input, state)
    actual_output, actual_state = batch_first(input.transpose(0, 1), state)
    torch.testing.assert_close(actual_output.transpose(0, 1), expected_output)
    _assert_state_close(actual_state, expected_state)


@pytest.mark.parametrize(("module_type", "kwargs"), SEQUENCE_CASES)
def test_multilayer_bidirectional_state_order_is_layer_then_direction(module_type, kwargs):
    torch.manual_seed(4)
    module = module_type(2, 3, num_layers=2, bidirectional=True, **kwargs).double()
    input = torch.randn(4, 2, 2, dtype=torch.double)
    state = _zero_state(module, input.shape[1])
    if isinstance(state, tuple):
        state = (
            torch.arange(4 * 2 * 3, dtype=torch.double).reshape(4, 2, 3) / 20.0,
            torch.arange(4 * 2 * 3, dtype=torch.double).reshape(4, 2, 3) / 30.0,
        )
    else:
        state = torch.arange(4 * 2 * 3, dtype=torch.double).reshape(4, 2, 3) / 20.0

    actual_output, actual_state = module(input, state)
    expected_output, expected_state, _ = _loop_reference(module, input, state)
    torch.testing.assert_close(actual_output, expected_output)
    _assert_state_close(actual_state, expected_state)
    assert actual_output.shape == (4, 2, 6)
    for value in _state_tensors(actual_state):
        assert value.shape == (4, 2, 3)


@pytest.mark.parametrize(("module_type", "kwargs"), SEQUENCE_CASES)
@pytest.mark.parametrize("shape", [(5, 4, 3), (5, 3, 3), (1, 1, 3)])
def test_full_remainder_and_singleton_dense_batches(module_type, kwargs, shape):
    torch.manual_seed(5)
    module = module_type(3, 2, num_layers=2, **kwargs).double()
    input = torch.randn(*shape, dtype=torch.double, requires_grad=True)
    output, state = module(input)
    assert output.shape == (shape[0], shape[1], 2)
    for value in _state_tensors(state):
        assert value.shape == (2, shape[1], 2)
    (output.square().sum() + sum(value.sum() for value in _state_tensors(state))).backward()
    assert input.grad is not None and torch.isfinite(input.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )


@pytest.mark.parametrize(("module_type", "kwargs"), SEQUENCE_CASES)
def test_stateful_carry_matches_concatenated_sequence(module_type, kwargs):
    torch.manual_seed(6)
    stateful = module_type(3, 4, num_layers=2, stateful=True, **kwargs).double()
    whole = module_type(3, 4, num_layers=2, stateful=False, **kwargs).double()
    whole.load_state_dict(stateful.state_dict())
    first = torch.randn(3, 2, 3, dtype=torch.double)
    second = torch.randn(2, 2, 3, dtype=torch.double)

    first_output, _ = stateful(first)
    second_output, second_state = stateful(second)
    whole_output, whole_state = whole(torch.cat((first, second), dim=0))
    torch.testing.assert_close(torch.cat((first_output, second_output), dim=0), whole_output)
    _assert_state_close(second_state, whole_state)
    assert stateful.hx is not None
    assert "hx" not in stateful.state_dict()


@pytest.mark.parametrize(("module_type", "kwargs"), SEQUENCE_CASES)
def test_explicit_state_is_side_effect_free_for_stateful_module(module_type, kwargs):
    torch.manual_seed(7)
    module = module_type(3, 4, num_layers=2, stateful=True, **kwargs).double()
    module(torch.randn(2, 2, 3, dtype=torch.double))
    stored_before = _clone_state(module.hx)
    explicit = _zero_state(module, 2)
    input = torch.randn(3, 2, 3, dtype=torch.double)

    expected_output, expected_state, _ = _loop_reference(module, input, explicit)
    actual = module(input, explicit)
    torch.testing.assert_close(actual[0], expected_output)
    _assert_state_close(actual[1], expected_state)
    _assert_state_close(module.hx, stored_before)


@pytest.mark.parametrize(("module_type", "kwargs"), SEQUENCE_CASES)
def test_stateful_reset_detach_and_persistent_mismatch(module_type, kwargs):
    torch.manual_seed(8)
    module = module_type(3, 4, num_layers=2, stateful=True, **kwargs).double()
    input = torch.randn(2, 2, 3, dtype=torch.double, requires_grad=True)
    module(input)
    assert all(value.grad_fn is not None for value in _state_tensors(module.hx))

    module.detach()
    assert all(value.grad_fn is None for value in _state_tensors(module.hx))
    assert all(not value.requires_grad for value in _state_tensors(module.hx))
    with pytest.raises(RuntimeError, match=r"persistent recurrent state.*call reset\(\)"):
        module(torch.randn(2, 3, 3, dtype=torch.double))
    module.reset()
    assert module.hx is None
    output, state = module(torch.randn(2, 3, 3, dtype=torch.double))
    assert output.shape[:2] == (2, 3)
    assert all(value.shape[1] == 3 for value in _state_tensors(state))


@pytest.mark.parametrize(("module_type", "kwargs"), SEQUENCE_CASES)
def test_persistent_dtype_mismatch_instructs_reset(module_type, kwargs):
    module = module_type(2, 3, stateful=True, **kwargs).double()
    module(torch.randn(2, 2, 2, dtype=torch.double))
    if isinstance(module.hx, tuple):
        module.hx = tuple(value.float() for value in module.hx)
    else:
        module.hx = module.hx.float()
    with pytest.raises(RuntimeError, match=r"persistent recurrent state.*call reset\(\)"):
        module(torch.randn(2, 2, 2, dtype=torch.double))


def test_dropout_occurs_only_between_layers_and_is_seed_deterministic(monkeypatch):
    calls = []
    original_dropout = F.dropout

    def recording_dropout(input, p=0.5, training=True, inplace=False):
        calls.append((tuple(input.shape), p, training))
        return original_dropout(input, p=p, training=training, inplace=inplace)

    monkeypatch.setattr(F, "dropout", recording_dropout)
    torch.manual_seed(9)
    module = SpikingRNN(3, 4, num_layers=3, dropout=0.25).train()
    input = torch.randn(5, 2, 3)
    torch.manual_seed(10)
    first_output, first_state = module(input)
    assert calls == [((5, 2, 4), 0.25, True), ((5, 2, 4), 0.25, True)]

    calls.clear()
    torch.manual_seed(10)
    second_output, second_state = module(input)
    torch.testing.assert_close(second_output, first_output)
    _assert_state_close(second_state, first_state)
    assert len(calls) == 2

    calls.clear()
    module.eval()
    module(input)
    assert calls == []

    calls.clear()
    single_layer = SpikingRNN(3, 4, num_layers=1, dropout=0.0).train()
    single_layer(input)
    assert calls == []


@pytest.mark.parametrize(
    ("constructor", "match"),
    [
        (lambda: SpikingRNN(0, 2), "input_size"),
        (lambda: SpikingGRU(2, 0), "hidden_size"),
        (lambda: SpikingLSTM(2, 2, num_layers=0), "num_layers"),
        (lambda: SpikingRNN(2, 2, dropout=-0.1), "dropout"),
        (lambda: SpikingGRU(2, 2, dropout=1.1), "dropout"),
        (lambda: SpikingLSTM(2, 2, stateful=True, bidirectional=True), "unidirectional"),
    ],
)
def test_constructor_validation(constructor, match):
    with pytest.raises((TypeError, ValueError), match=match):
        constructor()


@pytest.mark.parametrize(("module_type", "kwargs"), SEQUENCE_CASES)
def test_dense_input_and_explicit_state_validation(module_type, kwargs):
    module = module_type(3, 4, num_layers=2, **kwargs).double()
    with pytest.raises(TypeError, match="dense torch.Tensor"):
        module([torch.randn(2, 3, dtype=torch.double)])
    with pytest.raises(ValueError, match="3D"):
        module(torch.randn(2, 3, dtype=torch.double))
    with pytest.raises(RuntimeError, match="input_size"):
        module(torch.randn(2, 3, 5, dtype=torch.double))
    bad_state = _zero_state(module, 3)
    if isinstance(bad_state, tuple):
        bad_state = bad_state[0][:-1], bad_state[1]
    else:
        bad_state = bad_state[:-1]
    with pytest.raises(RuntimeError, match="incompatible shape"):
        module(torch.randn(2, 3, 3, dtype=torch.double), bad_state)


@pytest.mark.parametrize(("module_type", "kwargs"), SEQUENCE_CASES)
def test_sequence_first_order_gradients_are_finite(module_type, kwargs):
    torch.manual_seed(11)
    module = module_type(3, 4, num_layers=2, bidirectional=True, **kwargs).double()
    input = torch.randn(4, 2, 3, dtype=torch.double, requires_grad=True)
    state = _clone_state(_zero_state(module, 2), requires_grad=True)
    output, final_state = module(input, state)
    loss = output.square().mean() + sum(
        value.square().mean() for value in _state_tensors(final_state)
    )
    loss.backward()
    tensors = [input, *module.parameters(), *_state_tensors(state)]
    assert all(tensor.grad is not None for tensor in tensors)
    assert all(torch.isfinite(tensor.grad).all() for tensor in tensors)

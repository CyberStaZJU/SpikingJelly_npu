"""Eager spiking recurrent cells and dense sequence modules.

The sequence modules intentionally own PyTorch-style top-level recurrent parameters.
They do not depend on an accelerator route and keep runtime carry state outside the
``state_dict``.
"""

from __future__ import annotations

import numbers
import warnings

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from . import base, surrogate


def _surrogate_or_default(value: nn.Module | None) -> nn.Module:
    return surrogate.ATan() if value is None else value


def _prepare_cell_input(
    input: Tensor,
    input_size: int,
    cell_name: str,
) -> tuple[Tensor, bool]:
    if input.dim() not in (1, 2):
        raise ValueError(
            f"{cell_name}: expected input to be 1D or 2D, got {input.dim()}D instead"
        )
    if input.shape[-1] != input_size:
        raise RuntimeError(
            f"{cell_name}: input has inconsistent input_size; expected {input_size}, "
            f"got {input.shape[-1]}"
        )
    is_batched = input.dim() == 2
    return (input if is_batched else input.unsqueeze(0)), is_batched


def _prepare_cell_state(
    value: Tensor | None,
    input: Tensor,
    hidden_size: int,
    is_batched: bool,
    cell_name: str,
    state_name: str,
) -> Tensor:
    expected_shape = (
        (input.shape[0], hidden_size) if is_batched else (hidden_size,)
    )
    if value is None:
        return input.new_zeros(input.shape[0], hidden_size)
    if not isinstance(value, Tensor):
        raise TypeError(f"{cell_name}: {state_name} must be a torch.Tensor")
    if tuple(value.shape) != expected_shape:
        raise RuntimeError(
            f"{cell_name}: {state_name} has shape {tuple(value.shape)}; "
            f"expected {expected_shape}"
        )
    if value.device != input.device or value.dtype != input.dtype:
        raise RuntimeError(
            f"{cell_name}: {state_name} must have the same device and dtype as input"
        )
    return value if is_batched else value.unsqueeze(0)


def _rnn_cell_forward(
    input: Tensor,
    hidden: Tensor,
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Tensor | None,
    bias_hh: Tensor | None,
    surrogate_function: nn.Module,
) -> Tensor:
    return surrogate_function(
        F.linear(input, weight_ih, bias_ih) + F.linear(hidden, weight_hh, bias_hh)
    )


def _gru_cell_forward(
    input: Tensor,
    hidden: Tensor,
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Tensor | None,
    bias_hh: Tensor | None,
    surrogate_function1: nn.Module,
    surrogate_function2: nn.Module,
) -> Tensor:
    input_r, input_z, input_n = F.linear(input, weight_ih, bias_ih).chunk(3, dim=-1)
    hidden_r, hidden_z, hidden_n = F.linear(
        hidden, weight_hh, bias_hh
    ).chunk(3, dim=-1)
    reset = surrogate_function1(input_r + hidden_r)
    update = surrogate_function1(input_z + hidden_z)
    candidate = surrogate_function2(input_n + reset * hidden_n)
    return (1.0 - update) * candidate + update * hidden


def _lstm_cell_forward(
    input: Tensor,
    state: tuple[Tensor, Tensor],
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Tensor | None,
    bias_hh: Tensor | None,
    surrogate_function1: nn.Module,
    surrogate_function2: nn.Module,
) -> tuple[Tensor, Tensor]:
    hidden, cell = state
    gates = F.linear(input, weight_ih, bias_ih) + F.linear(
        hidden, weight_hh, bias_hh
    )
    input_gate, forget_gate, candidate, output_gate = gates.chunk(4, dim=-1)
    input_gate = surrogate_function1(input_gate)
    forget_gate = surrogate_function1(forget_gate)
    candidate = surrogate_function2(candidate)
    output_gate = surrogate_function1(output_gate)
    cell = torch.clamp_max(forget_gate * cell + input_gate * candidate, 1.0)
    return output_gate * cell, cell


class SpikingRNNCell(nn.RNNCell):
    """A vanilla recurrent cell whose activation is a surrogate spike function."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        surrogate_function: nn.Module | None = None,
        device=None,
        dtype=None,
    ) -> None:
        nn.RNNCell.__init__(
            self,
            input_size,
            hidden_size,
            bias=bias,
            nonlinearity="tanh",
            device=device,
            dtype=dtype,
        )
        del self.nonlinearity
        self.surrogate_function = _surrogate_or_default(surrogate_function)

    def forward(self, input: Tensor, hx: Tensor | None = None) -> Tensor:
        input, is_batched = _prepare_cell_input(
            input, self.input_size, self.__class__.__name__
        )
        hidden = _prepare_cell_state(
            hx,
            input,
            self.hidden_size,
            is_batched,
            self.__class__.__name__,
            "hidden state",
        )
        hidden = _rnn_cell_forward(
            input,
            hidden,
            self.weight_ih,
            self.weight_hh,
            self.bias_ih,
            self.bias_hh,
            self.surrogate_function,
        )
        return hidden if is_batched else hidden.squeeze(0)


class SpikingGRUCell(nn.GRUCell):
    """A GRU cell with spiking reset, update, and candidate gates."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        surrogate_function1: nn.Module | None = None,
        surrogate_function2: nn.Module | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            input_size,
            hidden_size,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.surrogate_function1 = _surrogate_or_default(surrogate_function1)
        self.surrogate_function2 = (
            self.surrogate_function1
            if surrogate_function2 is None
            else surrogate_function2
        )

    def forward(self, input: Tensor, hx: Tensor | None = None) -> Tensor:
        input, is_batched = _prepare_cell_input(
            input, self.input_size, self.__class__.__name__
        )
        hidden = _prepare_cell_state(
            hx,
            input,
            self.hidden_size,
            is_batched,
            self.__class__.__name__,
            "hidden state",
        )
        hidden = _gru_cell_forward(
            input,
            hidden,
            self.weight_ih,
            self.weight_hh,
            self.bias_ih,
            self.bias_hh,
            self.surrogate_function1,
            self.surrogate_function2,
        )
        return hidden if is_batched else hidden.squeeze(0)


class SpikingLSTMCell(nn.LSTMCell):
    """An LSTM cell with binary gates, capped cell state, and no output tanh."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        surrogate_function1: nn.Module | None = None,
        surrogate_function2: nn.Module | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            input_size,
            hidden_size,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.surrogate_function1 = _surrogate_or_default(surrogate_function1)
        self.surrogate_function2 = (
            self.surrogate_function1
            if surrogate_function2 is None
            else surrogate_function2
        )

    def forward(
        self,
        input: Tensor,
        hx: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        input, is_batched = _prepare_cell_input(
            input, self.input_size, self.__class__.__name__
        )
        if hx is not None and (not isinstance(hx, tuple) or len(hx) != 2):
            raise TypeError("SpikingLSTMCell: hx must be a tuple of (hidden, cell)")
        hidden = _prepare_cell_state(
            None if hx is None else hx[0],
            input,
            self.hidden_size,
            is_batched,
            self.__class__.__name__,
            "hidden state",
        )
        cell = _prepare_cell_state(
            None if hx is None else hx[1],
            input,
            self.hidden_size,
            is_batched,
            self.__class__.__name__,
            "cell state",
        )
        hidden, cell = _lstm_cell_forward(
            input,
            (hidden, cell),
            self.weight_ih,
            self.weight_hh,
            self.bias_ih,
            self.bias_hh,
            self.surrogate_function1,
            self.surrogate_function2,
        )
        if is_batched:
            return hidden, cell
        return hidden.squeeze(0), cell.squeeze(0)


class _SpikingRecurrentBase(base.MemoryModule):
    _gate_count: int
    _is_lstm = False

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        stateful: bool = False,
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.step_mode = "m"
        self._validate_constructor(
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
            stateful,
        )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        self.stateful = stateful
        self.num_directions = 2 if bidirectional else 1
        self._flat_weights_names: list[str] = []
        self._all_weights: list[list[str]] = []

        factory_kwargs = {"device": device, "dtype": dtype}
        gate_size = self._gate_count * hidden_size
        for layer_index in range(num_layers):
            layer_input_size = (
                input_size
                if layer_index == 0
                else hidden_size * self.num_directions
            )
            for direction in range(self.num_directions):
                suffix = "_reverse" if direction else ""
                names = [
                    f"weight_ih_l{layer_index}{suffix}",
                    f"weight_hh_l{layer_index}{suffix}",
                ]
                parameters: list[nn.Parameter] = [
                    nn.Parameter(torch.empty(gate_size, layer_input_size, **factory_kwargs)),
                    nn.Parameter(torch.empty(gate_size, hidden_size, **factory_kwargs)),
                ]
                if bias:
                    names.extend(
                        [
                            f"bias_ih_l{layer_index}{suffix}",
                            f"bias_hh_l{layer_index}{suffix}",
                        ]
                    )
                    parameters.extend(
                        [
                            nn.Parameter(torch.empty(gate_size, **factory_kwargs)),
                            nn.Parameter(torch.empty(gate_size, **factory_kwargs)),
                        ]
                    )
                for name, parameter in zip(names, parameters, strict=True):
                    setattr(self, name, parameter)
                self._flat_weights_names.extend(names)
                self._all_weights.append(names)

        self.register_memory("hx", None)
        self.reset_parameters()

    @staticmethod
    def _validate_constructor(
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bias: bool,
        batch_first: bool,
        dropout: float,
        bidirectional: bool,
        stateful: bool,
    ) -> None:
        if not isinstance(input_size, int):
            raise TypeError(
                f"input_size should be of type int, got {type(input_size).__name__}"
            )
        if input_size <= 0:
            raise ValueError("input_size must be greater than zero")
        if not isinstance(hidden_size, int):
            raise TypeError(
                f"hidden_size should be of type int, got {type(hidden_size).__name__}"
            )
        if hidden_size <= 0:
            raise ValueError("hidden_size must be greater than zero")
        if not isinstance(num_layers, int):
            raise TypeError(
                f"num_layers should be of type int, got {type(num_layers).__name__}"
            )
        if num_layers <= 0:
            raise ValueError("num_layers must be greater than zero")
        for name, value in (
            ("bias", bias),
            ("batch_first", batch_first),
            ("bidirectional", bidirectional),
            ("stateful", stateful),
        ):
            if not isinstance(value, bool):
                raise TypeError(
                    f"{name} should be of type bool, got {type(value).__name__}"
                )
        if (
            not isinstance(dropout, numbers.Number)
            or isinstance(dropout, bool)
            or not 0.0 <= float(dropout) <= 1.0
        ):
            raise ValueError(
                "dropout should be a number in range [0, 1] representing the "
                "probability of an element being zeroed"
            )
        if dropout > 0.0 and num_layers == 1:
            warnings.warn(
                "dropout is applied after all but the last recurrent layer, so "
                f"non-zero dropout expects num_layers greater than 1, got {num_layers}",
                stacklevel=3,
            )
        if stateful and bidirectional:
            raise ValueError("stateful=True is supported only for unidirectional modules")

    def reset_parameters(self) -> None:
        stdv = self.hidden_size**-0.5
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -stdv, stdv)

    def reset(self) -> None:
        self.hx = None

    def detach(self) -> None:
        if isinstance(self.hx, tuple):
            self.hx = tuple(value.detach() for value in self.hx)
        elif isinstance(self.hx, Tensor):
            self.hx = self.hx.detach()

    def supported_step_mode(self) -> tuple[str, ...]:
        return ("m",)

    @property
    def supported_backends(self) -> tuple[str, ...]:
        return ("torch",)

    def _apply(self, fn):
        if isinstance(self.hx, tuple):
            self.hx = tuple(fn(value) for value in self.hx)
        elif isinstance(self.hx, Tensor):
            self.hx = fn(self.hx)
        return nn.Module._apply(self, fn)

    @property
    def all_weights(self) -> list[list[Tensor]]:
        return [[getattr(self, name) for name in names] for names in self._all_weights]

    def _expected_state_shape(self, batch_size: int) -> tuple[int, int, int]:
        return self.num_layers * self.num_directions, batch_size, self.hidden_size

    def _zero_state(self, input: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        shape = self._expected_state_shape(input.shape[1])
        hidden = input.new_zeros(shape)
        if self._is_lstm:
            return hidden, input.new_zeros(shape)
        return hidden

    def _validate_state_tensor(
        self,
        value: object,
        input: Tensor,
        name: str,
        *,
        persistent: bool,
    ) -> Tensor:
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        expected_shape = self._expected_state_shape(input.shape[1])
        problems = []
        if tuple(value.shape) != expected_shape:
            problems.append(f"shape {tuple(value.shape)} instead of {expected_shape}")
        if value.device != input.device:
            problems.append(f"device {value.device} instead of {input.device}")
        if value.dtype != input.dtype:
            problems.append(f"dtype {value.dtype} instead of {input.dtype}")
        if problems:
            detail = ", ".join(problems)
            if persistent:
                raise RuntimeError(
                    f"persistent recurrent state has incompatible {detail}; call reset() "
                    "before using a new batch shape, device, or dtype"
                )
            raise RuntimeError(f"{name} has incompatible {detail}")
        return value

    def _normalize_state(
        self,
        hx: object,
        input: Tensor,
        *,
        persistent: bool,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if self._is_lstm:
            if not isinstance(hx, tuple) or len(hx) != 2:
                raise TypeError("LSTM state must be a tuple of (hidden, cell)")
            return (
                self._validate_state_tensor(
                    hx[0], input, "hidden state", persistent=persistent
                ),
                self._validate_state_tensor(
                    hx[1], input, "cell state", persistent=persistent
                ),
            )
        return self._validate_state_tensor(
            hx, input, "hidden state", persistent=persistent
        )

    def _parameters_for(
        self, layer_index: int, direction: int
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        suffix = "_reverse" if direction else ""
        weight_ih = getattr(self, f"weight_ih_l{layer_index}{suffix}")
        weight_hh = getattr(self, f"weight_hh_l{layer_index}{suffix}")
        if self.bias:
            bias_ih = getattr(self, f"bias_ih_l{layer_index}{suffix}")
            bias_hh = getattr(self, f"bias_hh_l{layer_index}{suffix}")
        else:
            bias_ih = bias_hh = None
        return weight_ih, weight_hh, bias_ih, bias_hh

    def _cell_step(
        self,
        input: Tensor,
        state: Tensor | tuple[Tensor, Tensor],
        parameters: tuple[Tensor, Tensor, Tensor | None, Tensor | None],
    ) -> Tensor | tuple[Tensor, Tensor]:
        weight_ih, weight_hh, bias_ih, bias_hh = parameters
        if self._is_lstm:
            return _lstm_cell_forward(
                input,
                state,
                weight_ih,
                weight_hh,
                bias_ih,
                bias_hh,
                self.surrogate_function1,
                self.surrogate_function2,
            )
        if self._gate_count == 3:
            return _gru_cell_forward(
                input,
                state,
                weight_ih,
                weight_hh,
                bias_ih,
                bias_hh,
                self.surrogate_function1,
                self.surrogate_function2,
            )
        return _rnn_cell_forward(
            input,
            state,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            self.surrogate_function,
        )

    def _run_direction(
        self,
        input: Tensor,
        initial_state: Tensor | tuple[Tensor, Tensor],
        layer_index: int,
        direction: int,
    ) -> tuple[Tensor, Tensor | tuple[Tensor, Tensor]]:
        parameters = self._parameters_for(layer_index, direction)
        time_indices = (
            range(input.shape[0])
            if direction == 0
            else range(input.shape[0] - 1, -1, -1)
        )
        state = initial_state
        outputs = []
        for time_index in time_indices:
            state = self._cell_step(input[time_index], state, parameters)
            outputs.append(state[0] if self._is_lstm else state)
        if direction:
            outputs.reverse()
        return torch.stack(outputs), state

    def forward(self, input: Tensor, hx=None):
        if not isinstance(input, Tensor):
            raise TypeError(
                f"{self.__class__.__name__} accepts only a dense torch.Tensor input"
            )
        if input.dim() != 3:
            raise ValueError(
                f"{self.__class__.__name__}: expected input to be 3D, "
                f"got {input.dim()}D instead"
            )
        if input.shape[-1] != self.input_size:
            raise RuntimeError(
                f"input.size(-1) must be equal to input_size; expected "
                f"{self.input_size}, got {input.shape[-1]}"
            )
        sequence = input.transpose(0, 1) if self.batch_first else input
        if sequence.shape[0] == 0:
            raise RuntimeError("expected sequence length to be larger than 0")

        explicit_state = hx is not None
        if explicit_state:
            state = self._normalize_state(hx, sequence, persistent=False)
        elif self.stateful and self.hx is not None:
            state = self._normalize_state(self.hx, sequence, persistent=True)
        else:
            state = self._zero_state(sequence)

        if self._is_lstm:
            hidden_state, cell_state = state
            final_hidden: list[Tensor] = []
            final_cell: list[Tensor] = []
        else:
            hidden_state = state
            final_hidden = []

        layer_input = sequence
        for layer_index in range(self.num_layers):
            direction_outputs = []
            for direction in range(self.num_directions):
                state_index = layer_index * self.num_directions + direction
                if self._is_lstm:
                    initial = (
                        hidden_state[state_index],
                        cell_state[state_index],
                    )
                else:
                    initial = hidden_state[state_index]
                direction_output, direction_state = self._run_direction(
                    layer_input, initial, layer_index, direction
                )
                direction_outputs.append(direction_output)
                if self._is_lstm:
                    final_hidden.append(direction_state[0])
                    final_cell.append(direction_state[1])
                else:
                    final_hidden.append(direction_state)
            layer_output = (
                direction_outputs[0]
                if self.num_directions == 1
                else torch.cat(direction_outputs, dim=-1)
            )
            if (
                self.training
                and layer_index < self.num_layers - 1
                and self.dropout > 0.0
            ):
                layer_output = F.dropout(layer_output, p=self.dropout, training=True)
            layer_input = layer_output

        output_state: Tensor | tuple[Tensor, Tensor]
        if self._is_lstm:
            output_state = torch.stack(final_hidden), torch.stack(final_cell)
        else:
            output_state = torch.stack(final_hidden)
        if self.stateful and not explicit_state:
            self.hx = output_state
        output = layer_input.transpose(0, 1) if self.batch_first else layer_input
        return output, output_state

    def extra_repr(self) -> str:
        values = [str(self.input_size), str(self.hidden_size)]
        if self.num_layers != 1:
            values.append(f"num_layers={self.num_layers}")
        if not self.bias:
            values.append("bias=False")
        if self.batch_first:
            values.append("batch_first=True")
        if self.dropout:
            values.append(f"dropout={self.dropout}")
        if self.bidirectional:
            values.append("bidirectional=True")
        if self.stateful:
            values.append("stateful=True")
        return ", ".join(values)


class SpikingRNN(_SpikingRecurrentBase):
    """A dense multi-layer spiking vanilla RNN."""

    _gate_count = 1

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        stateful: bool = False,
        surrogate_function: nn.Module | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
            stateful,
            device=device,
            dtype=dtype,
        )
        self.surrogate_function = _surrogate_or_default(surrogate_function)


class SpikingGRU(_SpikingRecurrentBase):
    """A dense multi-layer spiking GRU."""

    _gate_count = 3

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        stateful: bool = False,
        surrogate_function1: nn.Module | None = None,
        surrogate_function2: nn.Module | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
            stateful,
            device=device,
            dtype=dtype,
        )
        self.surrogate_function1 = _surrogate_or_default(surrogate_function1)
        self.surrogate_function2 = (
            self.surrogate_function1
            if surrogate_function2 is None
            else surrogate_function2
        )


class SpikingLSTM(_SpikingRecurrentBase):
    """A dense multi-layer spiking LSTM without projection support."""

    _gate_count = 4
    _is_lstm = True

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        stateful: bool = False,
        surrogate_function1: nn.Module | None = None,
        surrogate_function2: nn.Module | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
            stateful,
            device=device,
            dtype=dtype,
        )
        self.surrogate_function1 = _surrogate_or_default(surrogate_function1)
        self.surrogate_function2 = (
            self.surrogate_function1
            if surrogate_function2 is None
            else surrogate_function2
        )


__all__ = [
    "SpikingRNNCell",
    "SpikingGRUCell",
    "SpikingLSTMCell",
    "SpikingRNN",
    "SpikingGRU",
    "SpikingLSTM",
]

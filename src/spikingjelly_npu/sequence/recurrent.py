"""Standard recurrent layers with an FP32-state NPU compatibility fallback.

The public classes remain direct :mod:`torch.nn` subclasses. CPU execution and
non-FP32 NPU storage delegate to the upstream fused implementation. FP32 NPU
inputs use a narrow eager decomposition because the CANN 8.5 fused recurrent
route is not available for that storage dtype. Under the explicit BF16 profile,
its affine operators may autocast to BF16 while gates and recurrent state are
promoted back to FP32.
"""

from typing import overload

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence

from ..npu.amp import is_npu_bf16_autocast_active


def _should_use_fp32_npu_fallback(input: object) -> bool:
    actual_input = input.data if isinstance(input, PackedSequence) else input
    return (
        isinstance(actual_input, Tensor)
        and actual_input.device.type == "npu"
        and actual_input.dtype == torch.float32
    )


def _prepare_input(module, input, hx, module_name: str, *, lstm: bool):
    module._update_flat_weights()
    num_directions = 2 if module.bidirectional else 1
    is_packed = isinstance(input, PackedSequence)

    if is_packed:
        data, batch_sizes, sorted_indices, unsorted_indices = input
        max_batch_size = int(batch_sizes[0])
        is_batched = True
    else:
        if input.dim() not in (2, 3):
            raise ValueError(
                f"{module_name}: Expected input to be 2D or 3D, got {input.dim()}D instead"
            )
        is_batched = input.dim() == 3
        input_batch_dim = 0 if module.batch_first else 1
        if not is_batched:
            input = input.unsqueeze(input_batch_dim)
        data = input
        batch_sizes = None
        sorted_indices = None
        unsorted_indices = None
        max_batch_size = input.size(0) if module.batch_first else input.size(1)

    if lstm:
        hidden_size = module.proj_size if module.proj_size > 0 else module.hidden_size
        if hx is None:
            hx = (
                data.new_zeros(
                    module.num_layers * num_directions,
                    max_batch_size,
                    hidden_size,
                ),
                data.new_zeros(
                    module.num_layers * num_directions,
                    max_batch_size,
                    module.hidden_size,
                ),
            )
        elif not is_packed:
            if is_batched:
                if hx[0].dim() != 3 or hx[1].dim() != 3:
                    raise RuntimeError(
                        "For batched 3-D input, hx and cx should also be 3-D but "
                        f"got ({hx[0].dim()}-D, {hx[1].dim()}-D) tensors"
                    )
            else:
                if hx[0].dim() != 2 or hx[1].dim() != 2:
                    raise RuntimeError(
                        "For unbatched 2-D input, hx and cx should also be 2-D but "
                        f"got ({hx[0].dim()}-D, {hx[1].dim()}-D) tensors"
                    )
                hx = (hx[0].unsqueeze(1), hx[1].unsqueeze(1))
        if is_packed:
            hx = module.permute_hidden(hx, sorted_indices)
    else:
        if hx is None:
            hx = data.new_zeros(
                module.num_layers * num_directions,
                max_batch_size,
                module.hidden_size,
            )
        elif not is_packed:
            if is_batched:
                if hx.dim() != 3:
                    raise RuntimeError(
                        f"For batched 3-D input, hx should also be 3-D but got {hx.dim()}-D tensor"
                    )
            else:
                if hx.dim() != 2:
                    raise RuntimeError(
                        "For unbatched 2-D input, hx should also be 2-D but got "
                        f"{hx.dim()}-D tensor"
                    )
                hx = hx.unsqueeze(1)
        if is_packed:
            hx = module.permute_hidden(hx, sorted_indices)

    module.check_forward_args(data, hx, batch_sizes)
    sequence_dim = 1 if module.batch_first else 0
    if batch_sizes is None and data.shape[sequence_dim] == 0:
        raise RuntimeError("Expected sequence length to be larger than 0 in RNN")
    if not is_packed and module.batch_first:
        data = data.transpose(0, 1)

    return (
        data,
        batch_sizes,
        sorted_indices,
        unsorted_indices,
        hx,
        is_packed,
        is_batched,
    )


def _parameters_for(module, layer: int, direction: int):
    suffix = "_reverse" if direction else ""
    weight_ih = getattr(module, f"weight_ih_l{layer}{suffix}")
    weight_hh = getattr(module, f"weight_hh_l{layer}{suffix}")
    if module.bias:
        bias_ih = getattr(module, f"bias_ih_l{layer}{suffix}")
        bias_hh = getattr(module, f"bias_hh_l{layer}{suffix}")
    else:
        bias_ih = None
        bias_hh = None
    weight_hr = getattr(module, f"weight_hr_l{layer}{suffix}") if module.proj_size > 0 else None
    return weight_ih, weight_hh, bias_ih, bias_hh, weight_hr


def _recurrent_step(
    kind: str,
    input_affine: Tensor,
    hidden: Tensor,
    cell: Tensor | None,
    weight_hh: Tensor,
    bias_hh: Tensor | None,
    weight_hr: Tensor | None,
) -> tuple[Tensor, Tensor | None]:
    hidden_affine = F.linear(hidden, weight_hh, bias_hh)
    if is_npu_bf16_autocast_active():
        input_affine = input_affine.float()
        hidden_affine = hidden_affine.float()
        hidden = hidden.float()
        if cell is not None:
            cell = cell.float()
    if kind == "rnn_tanh":
        return torch.tanh(input_affine + hidden_affine), None
    if kind == "rnn_relu":
        return torch.relu(input_affine + hidden_affine), None
    if kind == "gru":
        input_reset, input_update, input_new = input_affine.chunk(3, dim=-1)
        hidden_reset, hidden_update, hidden_new = hidden_affine.chunk(3, dim=-1)
        reset = torch.sigmoid(input_reset + hidden_reset)
        update = torch.sigmoid(input_update + hidden_update)
        candidate = torch.tanh(input_new + reset * hidden_new)
        return (1.0 - update) * candidate + update * hidden, None

    if cell is None:
        raise AssertionError("LSTM cell state must not be None")
    gates = input_affine + hidden_affine
    input_gate, forget_gate, candidate, output_gate = gates.chunk(4, dim=-1)
    next_cell = torch.sigmoid(forget_gate) * cell + torch.sigmoid(input_gate) * torch.tanh(
        candidate
    )
    next_hidden = torch.sigmoid(output_gate) * torch.tanh(next_cell)
    if weight_hr is not None:
        next_hidden = F.linear(next_hidden, weight_hr)
    return next_hidden, next_cell


def _run_dense_direction(
    kind: str,
    input_affine: Tensor,
    initial_hidden: Tensor,
    initial_cell: Tensor | None,
    weight_hh: Tensor,
    bias_hh: Tensor | None,
    weight_hr: Tensor | None,
    direction: int,
):
    time_indices = (
        range(input_affine.shape[0]) if direction == 0 else range(input_affine.shape[0] - 1, -1, -1)
    )
    hidden = initial_hidden
    cell = initial_cell
    outputs = []
    for time_index in time_indices:
        hidden, cell = _recurrent_step(
            kind,
            input_affine[time_index],
            hidden,
            cell,
            weight_hh,
            bias_hh,
            weight_hr,
        )
        outputs.append(hidden)
    if direction:
        outputs.reverse()
    return torch.stack(outputs), hidden, cell


def _run_packed_direction(
    kind: str,
    input_affine: Tensor,
    batch_sizes: Tensor,
    initial_hidden: Tensor,
    initial_cell: Tensor | None,
    weight_hh: Tensor,
    bias_hh: Tensor | None,
    weight_hr: Tensor | None,
    direction: int,
):
    sizes = [int(batch_sizes[index].item()) for index in range(batch_sizes.numel())]
    input_steps = input_affine.split(sizes)

    if direction == 0:
        hidden = initial_hidden
        cell = initial_cell
        outputs = []
        for input_step, batch_size in zip(input_steps, sizes, strict=True):
            active_cell = None if cell is None else cell[:batch_size]
            next_hidden, next_cell = _recurrent_step(
                kind,
                input_step,
                hidden[:batch_size],
                active_cell,
                weight_hh,
                bias_hh,
                weight_hr,
            )
            hidden = torch.cat((next_hidden, hidden[batch_size:]), dim=0)
            if cell is not None:
                if next_cell is None:
                    raise AssertionError("LSTM cell state must not be None")
                cell = torch.cat((next_cell, cell[batch_size:]), dim=0)
            outputs.append(next_hidden)
        return torch.cat(outputs, dim=0), hidden, cell

    active_hidden = initial_hidden[:0]
    active_cell = None if initial_cell is None else initial_cell[:0]
    previous_batch_size = 0
    reverse_outputs = []
    for input_step, batch_size in reversed(tuple(zip(input_steps, sizes, strict=True))):
        if batch_size > previous_batch_size:
            active_hidden = torch.cat(
                (active_hidden, initial_hidden[previous_batch_size:batch_size]),
                dim=0,
            )
            if initial_cell is not None:
                if active_cell is None:
                    raise AssertionError("LSTM cell state must not be None")
                active_cell = torch.cat(
                    (active_cell, initial_cell[previous_batch_size:batch_size]),
                    dim=0,
                )
        active_hidden, active_cell = _recurrent_step(
            kind,
            input_step,
            active_hidden,
            active_cell,
            weight_hh,
            bias_hh,
            weight_hr,
        )
        reverse_outputs.append(active_hidden)
        previous_batch_size = batch_size
    reverse_outputs.reverse()
    return torch.cat(reverse_outputs, dim=0), active_hidden, active_cell


def _run_layers(module, data: Tensor, batch_sizes: Tensor | None, hx, kind: str):
    num_directions = 2 if module.bidirectional else 1
    is_lstm = kind == "lstm"
    if is_lstm:
        hidden_state, cell_state = hx
        final_hidden = []
        final_cell = []
    else:
        hidden_state = hx
        cell_state = None
        final_hidden = []
        final_cell = None

    layer_input = data
    for layer in range(module.num_layers):
        direction_outputs = []
        for direction in range(num_directions):
            state_index = layer * num_directions + direction
            weight_ih, weight_hh, bias_ih, bias_hh, weight_hr = _parameters_for(
                module, layer, direction
            )
            input_affine = F.linear(layer_input, weight_ih, bias_ih)
            initial_cell = None if cell_state is None else cell_state[state_index]
            if batch_sizes is None:
                direction_output, direction_hidden, direction_cell = _run_dense_direction(
                    kind,
                    input_affine,
                    hidden_state[state_index],
                    initial_cell,
                    weight_hh,
                    bias_hh,
                    weight_hr,
                    direction,
                )
            else:
                direction_output, direction_hidden, direction_cell = _run_packed_direction(
                    kind,
                    input_affine,
                    batch_sizes,
                    hidden_state[state_index],
                    initial_cell,
                    weight_hh,
                    bias_hh,
                    weight_hr,
                    direction,
                )
            direction_outputs.append(direction_output)
            final_hidden.append(direction_hidden)
            if final_cell is not None:
                if direction_cell is None:
                    raise AssertionError("LSTM cell state must not be None")
                final_cell.append(direction_cell)

        layer_output = (
            direction_outputs[0] if num_directions == 1 else torch.cat(direction_outputs, dim=-1)
        )
        if module.training and layer < module.num_layers - 1 and module.dropout > 0.0:
            layer_output = F.dropout(layer_output, p=module.dropout, training=True)
        layer_input = layer_output

    if final_cell is not None:
        return layer_input, (torch.stack(final_hidden), torch.stack(final_cell))
    return layer_input, torch.stack(final_hidden)


def _finish_output(
    module,
    output: Tensor,
    hidden,
    batch_sizes: Tensor | None,
    sorted_indices: Tensor | None,
    unsorted_indices: Tensor | None,
    is_packed: bool,
    is_batched: bool,
):
    hidden = module.permute_hidden(hidden, unsorted_indices)
    if is_packed:
        return (
            PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices),
            hidden,
        )
    if not is_batched:
        output = output.squeeze(1)
        if isinstance(hidden, tuple):
            hidden = (hidden[0].squeeze(1), hidden[1].squeeze(1))
        else:
            hidden = hidden.squeeze(1)
    elif module.batch_first:
        output = output.transpose(0, 1)
    return output, hidden


def _fallback_forward(module, input, hx, module_name: str, kind: str):
    prepared = _prepare_input(
        module,
        input,
        hx,
        module_name,
        lstm=kind == "lstm",
    )
    (
        data,
        batch_sizes,
        sorted_indices,
        unsorted_indices,
        hidden,
        is_packed,
        is_batched,
    ) = prepared
    output, hidden = _run_layers(module, data, batch_sizes, hidden, kind)
    return _finish_output(
        module,
        output,
        hidden,
        batch_sizes,
        sorted_indices,
        unsorted_indices,
        is_packed,
        is_batched,
    )


class RNN(nn.RNN):
    """A direct :class:`torch.nn.RNN` subclass with FP32 NPU fallback."""

    @overload
    @torch._jit_internal._overload_method
    def forward(
        self,
        input: Tensor,
        hx: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        pass

    @overload
    @torch._jit_internal._overload_method
    def forward(
        self,
        input: PackedSequence,
        hx: Tensor | None = None,
    ) -> tuple[PackedSequence, Tensor]:
        pass

    def forward(self, input, hx=None):
        if not _should_use_fp32_npu_fallback(input):
            return super().forward(input, hx)
        kind = "rnn_tanh" if self.nonlinearity == "tanh" else "rnn_relu"
        return _fallback_forward(self, input, hx, "RNN", kind)


class GRU(nn.GRU):
    """A direct :class:`torch.nn.GRU` subclass with FP32 NPU fallback."""

    @overload
    @torch._jit_internal._overload_method
    def forward(
        self,
        input: Tensor,
        hx: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        pass

    @overload
    @torch._jit_internal._overload_method
    def forward(
        self,
        input: PackedSequence,
        hx: Tensor | None = None,
    ) -> tuple[PackedSequence, Tensor]:
        pass

    def forward(self, input, hx=None):
        if not _should_use_fp32_npu_fallback(input):
            return super().forward(input, hx)
        return _fallback_forward(self, input, hx, "GRU", "gru")


class LSTM(nn.LSTM):
    """A direct :class:`torch.nn.LSTM` subclass with FP32 NPU fallback."""

    @overload
    @torch._jit_internal._overload_method
    def forward(
        self,
        input: Tensor,
        hx: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        pass

    @overload
    @torch._jit_internal._overload_method
    def forward(
        self,
        input: PackedSequence,
        hx: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[PackedSequence, tuple[Tensor, Tensor]]:
        pass

    def forward(self, input, hx=None):
        if not _should_use_fp32_npu_fallback(input):
            return super().forward(input, hx)
        return _fallback_forward(self, input, hx, "LSTM", "lstm")


__all__ = ["RNN", "GRU", "LSTM"]

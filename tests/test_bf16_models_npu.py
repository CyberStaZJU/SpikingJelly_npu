import os

import pytest
import torch
from torch import nn

from spikingjelly_npu import sequence
from spikingjelly_npu.activation_based import functional, neuron
from spikingjelly_npu.activation_based.layer import SpikingSelfAttention
from spikingjelly_npu.activation_based.model import Spikformer
from spikingjelly_npu.activation_based.recurrent import (
    SpikingGRU,
    SpikingLSTM,
    SpikingRNN,
)
from spikingjelly_npu.npu import configure_npu, npu_bf16_autocast

pytestmark = pytest.mark.npu


@pytest.fixture(scope="module")
def bf16_device():
    device_index = os.environ.get("ASCEND_DEVICE_ID", os.environ.get("DEVICE_ID", "0"))
    return configure_npu(
        f"npu:{device_index}",
        require_bf16=True,
        allow_internal_format=False,
    )


def _state_tensors(state):
    if state is None:
        return ()
    return state if isinstance(state, tuple) else (state,)


def _assert_master_state(module, *, require_all_grads: bool):
    parameters = tuple(module.parameters())
    assert parameters
    assert all(parameter.dtype == torch.float32 for parameter in parameters)
    gradients = tuple(parameter.grad for parameter in parameters)
    if require_all_grads:
        assert all(gradient is not None for gradient in gradients)
    assert all(
        gradient is None or gradient.dtype == torch.float32 for gradient in gradients
    )


def _assert_optimizer_state_fp32(optimizer):
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                assert value.dtype == torch.float32


def _run_training_step(module, inputs, forward):
    module.train()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.01, momentum=0.2)
    optimizer.zero_grad(set_to_none=True)
    with npu_bf16_autocast():
        output, state = forward(module, inputs)
        loss = output.float().square().mean()
        for value in _state_tensors(state):
            loss = loss + 0.01 * value.float().square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    torch.npu.synchronize()
    assert all(value.grad is not None for value in inputs)
    assert all(value.grad.dtype == torch.float32 for value in inputs)
    assert all(torch.isfinite(value.grad).all() for value in inputs)
    _assert_master_state(module, require_all_grads=True)
    optimizer.step()
    _assert_optimizer_state_fp32(optimizer)
    return output, state


@pytest.mark.parametrize("batch_size", [4, 3, 1], ids=["full", "remainder", "singleton"])
@pytest.mark.parametrize("training", [False, True], ids=["eval", "train"])
@pytest.mark.parametrize("module_type", [sequence.RNN, sequence.GRU, sequence.LSTM])
def test_standard_recurrent_bf16_train_eval_batches(
    bf16_device, module_type, training, batch_size
):
    kwargs = dict(input_size=8, hidden_size=16, num_layers=2, batch_first=True)
    if module_type is sequence.RNN:
        kwargs["nonlinearity"] = "tanh"
    module = module_type(**kwargs).to(bf16_device)
    inputs = (torch.randn(batch_size, 5, 8, device=bf16_device, requires_grad=True),)

    def forward(model, values):
        return model(values[0])

    if training:
        output, state = _run_training_step(module, inputs, forward)
    else:
        module.eval()
        with torch.no_grad(), npu_bf16_autocast():
            output, state = forward(module, inputs)
        torch.npu.synchronize()
        _assert_master_state(module, require_all_grads=False)

    assert output.dtype == torch.float32
    assert output.shape == (batch_size, 5, 16)
    assert all(value.dtype == torch.float32 for value in _state_tensors(state))
    assert torch.isfinite(output).all()


TRANSFORMER_CASES = (
    pytest.param(
        lambda: sequence.MultiheadAttention(16, 4, dropout=0.0, batch_first=True),
        lambda batch, device: (
            torch.randn(batch, 5, 16, device=device, requires_grad=True),
            torch.randn(batch, 6, 16, device=device, requires_grad=True),
            torch.randn(batch, 6, 16, device=device, requires_grad=True),
        ),
        "mha",
        id="mha",
    ),
    pytest.param(
        lambda: sequence.TransformerEncoderLayer(
            16, 4, dim_feedforward=32, dropout=0.0, batch_first=True
        ),
        lambda batch, device: (
            torch.randn(batch, 5, 16, device=device, requires_grad=True),
        ),
        "encoder",
        id="encoder",
    ),
    pytest.param(
        lambda: sequence.TransformerDecoderLayer(
            16, 4, dim_feedforward=32, dropout=0.0, batch_first=True
        ),
        lambda batch, device: (
            torch.randn(batch, 5, 16, device=device, requires_grad=True),
            torch.randn(batch, 6, 16, device=device, requires_grad=True),
        ),
        "decoder",
        id="decoder",
    ),
    pytest.param(
        lambda: sequence.TransformerEncoder(
            nn.TransformerEncoderLayer(
                16, 4, dim_feedforward=32, dropout=0.0, batch_first=True
            ),
            num_layers=2,
            enable_nested_tensor=False,
        ),
        lambda batch, device: (
            torch.randn(batch, 5, 16, device=device, requires_grad=True),
        ),
        "encoder_stack",
        id="encoder-stack-upstream-layer",
    ),
    pytest.param(
        lambda: sequence.TransformerDecoder(
            nn.TransformerDecoderLayer(
                16, 4, dim_feedforward=32, dropout=0.0, batch_first=True
            ),
            num_layers=2,
        ),
        lambda batch, device: (
            torch.randn(batch, 5, 16, device=device, requires_grad=True),
            torch.randn(batch, 6, 16, device=device, requires_grad=True),
        ),
        "decoder_stack",
        id="decoder-stack-upstream-layer",
    ),
    pytest.param(
        lambda: sequence.Transformer(
            d_model=16,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            batch_first=True,
        ),
        lambda batch, device: (
            torch.randn(batch, 6, 16, device=device, requires_grad=True),
            torch.randn(batch, 5, 16, device=device, requires_grad=True),
        ),
        "transformer",
        id="transformer",
    ),
)


@pytest.mark.parametrize("batch_size", [4, 3, 1], ids=["full", "remainder", "singleton"])
@pytest.mark.parametrize("training", [False, True], ids=["eval", "train"])
@pytest.mark.parametrize(("factory", "input_factory", "kind"), TRANSFORMER_CASES)
def test_standard_transformer_bf16_train_eval_batches(
    bf16_device, factory, input_factory, kind, training, batch_size
):
    module = factory().to(bf16_device)
    inputs = input_factory(batch_size, bf16_device)

    def forward(model, values):
        result = model(*values)
        output = result[0] if kind == "mha" else result
        return output, None

    if training:
        output, _ = _run_training_step(module, inputs, forward)
    else:
        module.eval()
        with torch.no_grad(), npu_bf16_autocast():
            output, _ = forward(module, inputs)
        torch.npu.synchronize()
        _assert_master_state(module, require_all_grads=False)

    assert output.dtype in {torch.bfloat16, torch.float32}
    assert output.shape[:2] == (batch_size, 5)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("batch_size", [4, 3, 1], ids=["full", "remainder", "singleton"])
@pytest.mark.parametrize("training", [False, True], ids=["eval", "train"])
@pytest.mark.parametrize("module_type", [SpikingRNN, SpikingGRU, SpikingLSTM])
def test_spiking_recurrent_bf16_train_eval_batches(
    bf16_device, module_type, training, batch_size
):
    module = module_type(8, 16, num_layers=2, batch_first=True).to(bf16_device)
    inputs = (torch.randn(batch_size, 5, 8, device=bf16_device, requires_grad=True),)

    def forward(model, values):
        return model(values[0])

    if training:
        output, state = _run_training_step(module, inputs, forward)
    else:
        module.eval()
        with torch.no_grad(), npu_bf16_autocast():
            output, state = forward(module, inputs)
        torch.npu.synchronize()
        _assert_master_state(module, require_all_grads=False)

    assert output.dtype == torch.float32
    assert all(value.dtype == torch.float32 for value in _state_tensors(state))
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("batch_size", [4, 3, 1], ids=["full", "remainder", "singleton"])
@pytest.mark.parametrize("training", [False, True], ids=["eval", "train"])
def test_spiking_self_attention_bf16_train_eval_batches(
    bf16_device, training, batch_size
):
    module = SpikingSelfAttention(16, 4).to(bf16_device)
    inputs = (
        torch.randn(2, batch_size, 16, 8, device=bf16_device, requires_grad=True),
    )

    def forward(model, values):
        return model(values[0]), None

    if training:
        output, _ = _run_training_step(module, inputs, forward)
    else:
        module.eval()
        with torch.no_grad(), npu_bf16_autocast():
            output, _ = forward(module, inputs)
        torch.npu.synchronize()
        _assert_master_state(module, require_all_grads=False)

    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
    nodes = (module.qkv_lif, module.attn_lif, module.proj_lif)
    assert all(node.v.dtype == torch.float32 for node in nodes)
    functional.reset_net(module)


@pytest.mark.parametrize("batch_size", [4, 3, 1], ids=["full", "remainder", "singleton"])
@pytest.mark.parametrize("training", [False, True], ids=["eval", "train"])
def test_spikformer_bf16_train_eval_batches(bf16_device, training, batch_size):
    module = Spikformer(
        T=2,
        in_channels=3,
        img_size_h=32,
        img_size_w=32,
        patch_size=16,
        num_classes=5,
        embed_dims=32,
        num_heads=4,
        mlp_ratio=2.0,
        depths=1,
    ).to(bf16_device)
    inputs = (
        torch.randn(batch_size, 3, 32, 32, device=bf16_device, requires_grad=True),
    )

    def forward(model, values):
        return model(values[0]), None

    if training:
        output, _ = _run_training_step(module, inputs, forward)
    else:
        module.eval()
        with torch.no_grad(), npu_bf16_autocast():
            output, _ = forward(module, inputs)
        torch.npu.synchronize()
        _assert_master_state(module, require_all_grads=False)

    assert output.dtype == torch.float32
    assert output.shape == (2, batch_size, 5)
    assert torch.isfinite(output).all()
    nodes = [child for child in module.modules() if isinstance(child, neuron.BaseNode)]
    assert nodes and all(node.v.dtype == torch.float32 for node in nodes)
    functional.reset_net(module)

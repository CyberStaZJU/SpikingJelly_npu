import pytest
import torch
from torch import nn

from spikingjelly_npu.sequence import transformer


def _clone_with_grad(tensor):
    return tensor.detach().clone().requires_grad_()


def _causal_mask(size):
    return torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1)


def _assert_state_dict_round_trip(actual, expected):
    assert actual.state_dict().keys() == expected.state_dict().keys()
    actual.load_state_dict(expected.state_dict())
    expected.load_state_dict(actual.state_dict())


def _assert_parameter_grads_close(actual, expected):
    actual_parameters = dict(actual.named_parameters())
    expected_parameters = dict(expected.named_parameters())
    assert actual_parameters.keys() == expected_parameters.keys()
    for name in actual_parameters:
        assert actual_parameters[name].grad is not None
        assert expected_parameters[name].grad is not None
        torch.testing.assert_close(actual_parameters[name].grad, expected_parameters[name].grad)


def _run_tensor_module_parity(actual, expected, actual_inputs, expected_inputs, kwargs):
    expected_output = expected(*expected_inputs, **kwargs)
    actual_output = actual(*actual_inputs, **kwargs)
    torch.testing.assert_close(actual_output, expected_output)

    expected_output.square().sum().backward()
    actual_output.square().sum().backward()
    for actual_input, expected_input in zip(actual_inputs, expected_inputs, strict=True):
        torch.testing.assert_close(actual_input.grad, expected_input.grad)
    _assert_parameter_grads_close(actual, expected)


@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize("training", [False, True], ids=["eval", "train"])
def test_multihead_attention_masks_weights_and_gradients(batch_first, training):
    kwargs = {
        "embed_dim": 8,
        "num_heads": 2,
        "dropout": 0.0,
        "batch_first": batch_first,
    }
    torch.manual_seed(21)
    expected = nn.MultiheadAttention(**kwargs).double().train(training)
    actual = transformer.MultiheadAttention(**kwargs).double().train(training)
    assert actual.__class__.__bases__ == (nn.MultiheadAttention,)
    _assert_state_dict_round_trip(actual, expected)

    batch_size, target_length, source_length = 2, 3, 4
    query_shape = (
        (batch_size, target_length, 8) if batch_first else (target_length, batch_size, 8)
    )
    source_shape = (
        (batch_size, source_length, 8) if batch_first else (source_length, batch_size, 8)
    )
    expected_query = torch.randn(*query_shape, dtype=torch.float64, requires_grad=True)
    expected_key = torch.randn(*source_shape, dtype=torch.float64, requires_grad=True)
    expected_value = torch.randn(*source_shape, dtype=torch.float64, requires_grad=True)
    actual_query = _clone_with_grad(expected_query)
    actual_key = _clone_with_grad(expected_key)
    actual_value = _clone_with_grad(expected_value)
    attention_mask = torch.zeros(target_length, source_length, dtype=torch.bool)
    attention_mask[:, -1] = True
    key_padding_mask = torch.tensor([[False, False, False, True], [False] * 4])
    forward_kwargs = {
        "attn_mask": attention_mask,
        "key_padding_mask": key_padding_mask,
        "need_weights": True,
        "average_attn_weights": False,
    }

    expected_output, expected_weights = expected(
        expected_query,
        expected_key,
        expected_value,
        **forward_kwargs,
    )
    actual_output, actual_weights = actual(
        actual_query,
        actual_key,
        actual_value,
        **forward_kwargs,
    )

    torch.testing.assert_close(actual_output, expected_output)
    torch.testing.assert_close(actual_weights, expected_weights)

    expected_loss = expected_output.square().sum() + expected_weights.square().sum()
    actual_loss = actual_output.square().sum() + actual_weights.square().sum()
    expected_loss.backward()
    actual_loss.backward()

    torch.testing.assert_close(actual_query.grad, expected_query.grad)
    torch.testing.assert_close(actual_key.grad, expected_key.grad)
    torch.testing.assert_close(actual_value.grad, expected_value.grad)
    _assert_parameter_grads_close(actual, expected)


def test_multihead_attention_causal_hint_matches_torch():
    torch.manual_seed(22)
    expected = nn.MultiheadAttention(8, 2, dropout=0.0, batch_first=True).double()
    actual = transformer.MultiheadAttention(8, 2, dropout=0.0, batch_first=True).double()
    actual.load_state_dict(expected.state_dict())
    inputs = torch.randn(2, 4, 8, dtype=torch.float64)
    causal_mask = _causal_mask(4)

    expected_output, expected_weights = expected(
        inputs,
        inputs,
        inputs,
        attn_mask=causal_mask,
        need_weights=False,
        is_causal=True,
    )
    actual_output, actual_weights = actual(
        inputs,
        inputs,
        inputs,
        attn_mask=causal_mask,
        need_weights=False,
        is_causal=True,
    )

    torch.testing.assert_close(actual_output, expected_output)
    assert actual_weights is expected_weights is None


@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize("norm_first", [False, True])
def test_transformer_encoder_layer_mask_norm_and_gradients(batch_first, norm_first):
    kwargs = {
        "d_model": 8,
        "nhead": 2,
        "dim_feedforward": 16,
        "dropout": 0.0,
        "batch_first": batch_first,
        "norm_first": norm_first,
    }
    torch.manual_seed(23)
    expected = nn.TransformerEncoderLayer(**kwargs).double().train()
    actual = transformer.TransformerEncoderLayer(**kwargs).double().train()
    assert actual.__class__.__bases__ == (nn.TransformerEncoderLayer,)
    _assert_state_dict_round_trip(actual, expected)

    shape = (2, 4, 8) if batch_first else (4, 2, 8)
    expected_input = torch.randn(*shape, dtype=torch.float64, requires_grad=True)
    actual_input = _clone_with_grad(expected_input)
    mask = _causal_mask(4)
    padding_mask = torch.tensor([[False, False, False, True], [False] * 4])
    forward_kwargs = {
        "src_mask": mask,
        "src_key_padding_mask": padding_mask,
        "is_causal": True,
    }

    _run_tensor_module_parity(
        actual,
        expected,
        (actual_input,),
        (expected_input,),
        forward_kwargs,
    )


@pytest.mark.parametrize("norm_first", [False, True])
def test_transformer_decoder_layer_cross_attention_and_gradients(norm_first):
    kwargs = {
        "d_model": 8,
        "nhead": 2,
        "dim_feedforward": 16,
        "dropout": 0.0,
        "batch_first": True,
        "norm_first": norm_first,
    }
    torch.manual_seed(24)
    expected = nn.TransformerDecoderLayer(**kwargs).double().train()
    actual = transformer.TransformerDecoderLayer(**kwargs).double().train()
    assert actual.__class__.__bases__ == (nn.TransformerDecoderLayer,)
    _assert_state_dict_round_trip(actual, expected)

    expected_target = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    expected_memory = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
    actual_target = _clone_with_grad(expected_target)
    actual_memory = _clone_with_grad(expected_memory)
    target_mask = _causal_mask(3)
    memory_mask = torch.zeros(3, 4, dtype=torch.bool)
    memory_mask[:, -1] = True
    forward_kwargs = {
        "tgt_mask": target_mask,
        "memory_mask": memory_mask,
        "tgt_key_padding_mask": torch.tensor([[False, False, True], [False] * 3]),
        "memory_key_padding_mask": torch.tensor(
            [[False, False, False, True], [False] * 4]
        ),
        "tgt_is_causal": True,
        "memory_is_causal": False,
    }

    _run_tensor_module_parity(
        actual,
        expected,
        (actual_target, actual_memory),
        (expected_target, expected_memory),
        forward_kwargs,
    )


@pytest.mark.parametrize("training", [False, True], ids=["eval", "train"])
def test_transformer_encoder_stack_parity(training):
    torch.manual_seed(25)
    expected_layer = nn.TransformerEncoderLayer(
        8,
        2,
        dim_feedforward=16,
        dropout=0.0,
        batch_first=True,
        norm_first=True,
    ).double()
    actual_layer = transformer.TransformerEncoderLayer(
        8,
        2,
        dim_feedforward=16,
        dropout=0.0,
        batch_first=True,
        norm_first=True,
    ).double()
    expected = nn.TransformerEncoder(
        expected_layer,
        2,
        norm=nn.LayerNorm(8).double(),
        enable_nested_tensor=False,
    ).train(training)
    actual = transformer.TransformerEncoder(
        actual_layer,
        2,
        norm=nn.LayerNorm(8).double(),
        enable_nested_tensor=False,
    ).train(training)
    assert actual.__class__.__bases__ == (nn.TransformerEncoder,)
    _assert_state_dict_round_trip(actual, expected)

    inputs = torch.randn(2, 4, 8, dtype=torch.float64)
    mask = _causal_mask(4)
    padding_mask = torch.tensor([[False, False, False, True], [False] * 4])

    expected_output = expected(
        inputs,
        mask=mask,
        src_key_padding_mask=padding_mask,
        is_causal=True,
    )
    actual_output = actual(
        inputs,
        mask=mask,
        src_key_padding_mask=padding_mask,
        is_causal=True,
    )
    torch.testing.assert_close(actual_output, expected_output)


@pytest.mark.parametrize("training", [False, True], ids=["eval", "train"])
def test_transformer_decoder_stack_parity(training):
    torch.manual_seed(26)
    expected_layer = nn.TransformerDecoderLayer(
        8,
        2,
        dim_feedforward=16,
        dropout=0.0,
        batch_first=True,
    ).double()
    actual_layer = transformer.TransformerDecoderLayer(
        8,
        2,
        dim_feedforward=16,
        dropout=0.0,
        batch_first=True,
    ).double()
    expected = nn.TransformerDecoder(
        expected_layer,
        2,
        norm=nn.LayerNorm(8).double(),
    ).train(training)
    actual = transformer.TransformerDecoder(
        actual_layer,
        2,
        norm=nn.LayerNorm(8).double(),
    ).train(training)
    assert actual.__class__.__bases__ == (nn.TransformerDecoder,)
    _assert_state_dict_round_trip(actual, expected)

    target = torch.randn(2, 3, 8, dtype=torch.float64)
    memory = torch.randn(2, 4, 8, dtype=torch.float64)
    target_mask = _causal_mask(3)
    memory_mask = torch.zeros(3, 4, dtype=torch.bool)
    memory_mask[:, -1] = True
    forward_kwargs = {
        "tgt_mask": target_mask,
        "memory_mask": memory_mask,
        "tgt_key_padding_mask": torch.tensor([[False, False, True], [False] * 3]),
        "memory_key_padding_mask": torch.tensor(
            [[False, False, False, True], [False] * 4]
        ),
        "tgt_is_causal": True,
        "memory_is_causal": False,
    }

    expected_output = expected(target, memory, **forward_kwargs)
    actual_output = actual(target, memory, **forward_kwargs)
    torch.testing.assert_close(actual_output, expected_output)


@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize("norm_first", [False, True])
def test_transformer_end_to_end_masks_gradients_and_generated_mask(batch_first, norm_first):
    kwargs = {
        "d_model": 8,
        "nhead": 2,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "dim_feedforward": 16,
        "dropout": 0.0,
        "batch_first": batch_first,
        "norm_first": norm_first,
    }
    torch.manual_seed(27)
    expected = nn.Transformer(**kwargs).double().train()
    actual = transformer.Transformer(**kwargs).double().train()
    assert actual.__class__.__bases__ == (nn.Transformer,)
    _assert_state_dict_round_trip(actual, expected)

    source_shape = (2, 4, 8) if batch_first else (4, 2, 8)
    target_shape = (2, 3, 8) if batch_first else (3, 2, 8)
    expected_source = torch.randn(*source_shape, dtype=torch.float64, requires_grad=True)
    expected_target = torch.randn(*target_shape, dtype=torch.float64, requires_grad=True)
    actual_source = _clone_with_grad(expected_source)
    actual_target = _clone_with_grad(expected_target)
    source_mask = _causal_mask(4)
    target_mask = _causal_mask(3)
    memory_mask = torch.zeros(3, 4, dtype=torch.bool)
    memory_mask[:, -1] = True
    forward_kwargs = {
        "src_mask": source_mask,
        "tgt_mask": target_mask,
        "memory_mask": memory_mask,
        "src_key_padding_mask": torch.tensor(
            [[False, False, False, True], [False] * 4]
        ),
        "tgt_key_padding_mask": torch.tensor([[False, False, True], [False] * 3]),
        "memory_key_padding_mask": torch.tensor(
            [[False, False, False, True], [False] * 4]
        ),
        "src_is_causal": True,
        "tgt_is_causal": True,
        "memory_is_causal": False,
    }

    _run_tensor_module_parity(
        actual,
        expected,
        (actual_source, actual_target),
        (expected_source, expected_target),
        forward_kwargs,
    )

    expected_mask = nn.Transformer.generate_square_subsequent_mask(
        4,
        dtype=torch.float64,
    )
    actual_mask = transformer.Transformer.generate_square_subsequent_mask(
        4,
        dtype=torch.float64,
    )
    torch.testing.assert_close(actual_mask, expected_mask)


def test_transformer_seeded_dropout_matches_torch_and_changes_with_seed():
    kwargs = {
        "d_model": 8,
        "nhead": 2,
        "dim_feedforward": 16,
        "dropout": 0.3,
        "batch_first": True,
    }
    torch.manual_seed(28)
    expected = nn.TransformerEncoderLayer(**kwargs).train()
    actual = transformer.TransformerEncoderLayer(**kwargs).train()
    actual.load_state_dict(expected.state_dict())
    inputs = torch.randn(2, 4, 8)

    torch.manual_seed(29)
    expected_output = expected(inputs)
    torch.manual_seed(29)
    actual_output = actual(inputs)
    torch.testing.assert_close(actual_output, expected_output)

    torch.manual_seed(30)
    different_output = actual(inputs)
    assert not torch.equal(actual_output, different_output)

import pytest
import torch
from torch import nn

from spikingjelly_npu.activation_based import functional, neuron
from spikingjelly_npu.activation_based.model import (
    Spikformer,
    SpikformerBlock,
    SpikformerConv2dBNLIF,
    SpikformerMLP,
    SpikformerPatchStem,
    spikformer_s,
    spikformer_ti,
)


def _tiny_model(**kwargs):
    defaults = {
        "T": 2,
        "in_channels": 3,
        "img_size_h": 32,
        "img_size_w": 32,
        "patch_size": 16,
        "num_classes": 7,
        "embed_dims": 32,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "depths": 1,
    }
    defaults.update(kwargs)
    return Spikformer(**defaults)


def _voltage_keys(module):
    return [key for key in module.state_dict() if key.endswith(".v") or "v_seq" in key]


def test_legacy_models_path_reexports_canonical_objects():
    from spikingjelly_npu import models
    from spikingjelly_npu.activation_based.model import spikformer as canonical_module
    from spikingjelly_npu.models import spikformer as legacy_module

    assert models.Spikformer is Spikformer is canonical_module.Spikformer
    assert legacy_module.Spikformer is canonical_module.Spikformer
    assert models.spikformer_ti is spikformer_ti is canonical_module.spikformer_ti
    assert models.spikformer_s is spikformer_s is canonical_module.spikformer_s


def test_spikformer_topology_shapes_and_no_hidden_temporal_mean():
    model = _tiny_model().eval()

    assert isinstance(model.patch_embed, SpikformerPatchStem)
    assert len(model.patch_embed.stages) == 4
    expected_channels = ((3, 4), (4, 8), (8, 16), (16, 32))
    for stage, (in_channels, out_channels) in zip(
        model.patch_embed.stages, expected_channels, strict=True
    ):
        assert isinstance(stage, SpikformerConv2dBNLIF)
        conv = stage.conv_bn.block[0]
        pool = stage.conv_bn.block[2]
        assert (conv.in_channels, conv.out_channels) == (in_channels, out_channels)
        assert isinstance(stage.conv_bn.block[1], nn.BatchNorm2d)
        assert isinstance(pool, nn.MaxPool2d)
        assert pool.kernel_size == 3
        assert pool.stride == 2
        assert pool.padding == 1
        assert isinstance(stage.neuron, neuron.LIFNode)

    positional_conv = model.patch_embed.positional_encoding.conv_bn.block[0]
    assert positional_conv.in_channels == positional_conv.out_channels == 32
    assert len(model.patch_embed.positional_encoding.conv_bn.block) == 2
    assert isinstance(model.blocks[0], SpikformerBlock)
    assert isinstance(model.blocks[0].mlp, SpikformerMLP)
    assert isinstance(model.blocks[0].mlp.fc1[0], nn.Conv1d)
    assert isinstance(model.blocks[0].mlp.fc2[0], nn.Conv1d)
    assert not any(isinstance(module, nn.LayerNorm) for module in model.modules())

    x_seq = torch.randn(2, 2, 3, 32, 32)
    with torch.no_grad():
        stem_output = model.patch_embed(x_seq)
        features = model.forward_features(x_seq)
        logits = model(x_seq)
        expected_logits = model.head(features)

    assert stem_output.shape == (2, 2, 32, 2, 2)
    assert features.shape == (2, 2, 32)
    assert logits.shape == (2, 2, 7)
    torch.testing.assert_close(logits, expected_logits)
    assert logits.ndim == 3


def test_spikformer_4d_repeat_matches_explicit_5d_after_reset():
    torch.manual_seed(21)
    model = _tiny_model().eval()
    images = torch.randn(2, 3, 32, 32)
    explicit = images.unsqueeze(0).repeat(2, 1, 1, 1, 1)

    with torch.no_grad():
        output_4d = model(images)
        functional.reset_net(model)
        output_5d = model(explicit)

    torch.testing.assert_close(output_4d, output_5d)


def test_spikformer_4d_uses_configured_time_with_variable_spatial_size():
    model = _tiny_model(T=3).eval()

    with torch.no_grad():
        output = model(torch.randn(1, 3, 40, 48))

    assert output.shape == (3, 1, 7)


def test_spikformer_5d_accepts_variable_time_and_spatial_size():
    model = _tiny_model(T=2).eval()
    sequence = torch.randn(3, 1, 3, 40, 48)

    with torch.no_grad():
        features = model.forward_features(sequence)
        output = model(sequence)

    assert features.shape == (3, 1, 32)
    assert output.shape == (3, 1, 7)


def test_spikformer_manual_flow_matches_public_features_and_forward():
    torch.manual_seed(22)
    actual_model = _tiny_model().eval()
    reference_model = _tiny_model().eval()
    reference_model.load_state_dict(actual_model.state_dict())
    images = torch.randn(1, 3, 32, 32)

    with torch.no_grad():
        actual_features = actual_model.forward_features(images)
        functional.reset_net(actual_model)
        actual_logits = actual_model(images)

        sequence = images.unsqueeze(0).repeat(2, 1, 1, 1, 1)
        expected = reference_model.patch_embed(sequence)
        for block in reference_model.blocks:
            expected = block(expected)
        expected_features = expected.flatten(3).mean(dim=-1)
        expected_logits = reference_model.head(expected_features)

    torch.testing.assert_close(actual_features, expected_features)
    torch.testing.assert_close(actual_logits, expected_logits)


def test_spikformer_finite_gradients_cover_stem_attention_mlp_and_head():
    torch.manual_seed(23)
    model = _tiny_model().train()
    x = torch.randn(2, 2, 3, 32, 32, requires_grad=True)

    loss = model(x).square().mean()
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    parameters = {
        "stem": model.patch_embed.stages[0].conv_bn.block[0].weight,
        "ssa": model.blocks[0].attn.qkv_conv_bn[0].weight,
        "mlp": model.blocks[0].mlp.fc1[0].weight,
        "head": model.head.weight,
    }
    for name, parameter in parameters.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_spikformer_batch_norm_train_eval_buffers():
    torch.manual_seed(24)
    model = _tiny_model()
    batch_norms = [
        module
        for module in model.modules()
        if isinstance(module, nn.BatchNorm1d | nn.BatchNorm2d)
    ]
    initial_means = [module.running_mean.clone() for module in batch_norms]

    model.train()
    model(torch.randn(2, 2, 3, 32, 32))
    assert all(module.num_batches_tracked.item() == 1 for module in batch_norms)
    assert any(
        not torch.equal(module.running_mean, initial)
        for module, initial in zip(batch_norms, initial_means, strict=True)
    )

    running_means = [module.running_mean.clone() for module in batch_norms]
    running_vars = [module.running_var.clone() for module in batch_norms]
    tracked = [module.num_batches_tracked.clone() for module in batch_norms]
    functional.reset_net(model)
    model.eval()
    with torch.no_grad():
        model(torch.randn(2, 2, 3, 32, 32))

    for index, module in enumerate(batch_norms):
        torch.testing.assert_close(module.running_mean, running_means[index])
        torch.testing.assert_close(module.running_var, running_vars[index])
        torch.testing.assert_close(module.num_batches_tracked, tracked[index])


def test_spikformer_state_dict_round_trip_reset_and_no_voltage_keys():
    torch.manual_seed(25)
    source = _tiny_model().eval()
    target = _tiny_model().eval()
    target.load_state_dict(source.state_dict())
    assert _voltage_keys(source) == []

    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        source_output = source(x)
        target_output = target(x)
    torch.testing.assert_close(source_output, target_output)

    nodes = [module for module in source.modules() if isinstance(module, neuron.BaseNode)]
    assert nodes
    assert all(isinstance(node.v, torch.Tensor) for node in nodes)
    functional.reset_net(source)
    assert _voltage_keys(source) == []
    for node in nodes:
        torch.testing.assert_close(node.v, torch.zeros_like(node.v))


@pytest.mark.parametrize(
    ("factory", "embed_dims", "num_heads", "depths"),
    [
        (spikformer_ti, 256, 8, 4),
        (spikformer_s, 384, 12, 6),
    ],
)
def test_spikformer_factories_match_upstream_config_and_smoke(
    factory, embed_dims, num_heads, depths
):
    model = factory(
        T=1,
        in_channels=1,
        img_size_h=16,
        img_size_w=16,
        num_classes=5,
        backend="torch",
    ).eval()

    assert isinstance(model, Spikformer)
    assert model.T == 1
    assert model.in_channels == 1
    assert model.embed_dims == embed_dims
    assert model.num_heads == num_heads
    assert model.depths == depths
    assert model.num_classes == 5
    with torch.no_grad():
        output = model(torch.randn(1, 1, 16, 16))
    assert output.shape == (1, 1, 5)


def test_spikformer_backend_propagates_and_aspy_cpu_falls_back():
    model = _tiny_model(backend="aspy").eval()
    nodes = [module for module in model.modules() if isinstance(module, neuron.BaseNode)]
    assert nodes and all(node.backend == "aspy" for node in nodes)

    with torch.no_grad():
        model(torch.randn(1, 3, 32, 32))
    assert all(node.last_backend_route.requested_backend == "aspy" for node in nodes)
    assert all(node.last_backend_route.backend == "torch" for node in nodes)
    assert all("requires an NPU tensor" in node.last_backend_route.reason for node in nodes)

    model.backend = "torch"
    assert model.backend == "torch"
    assert all(node.backend == "torch" for node in nodes)


def test_spikformer_submodule_validation_and_positional_residual():
    with pytest.raises(ValueError, match="patch_size=16"):
        SpikformerPatchStem(patch_size=8)
    with pytest.raises(ValueError, match="divisible by 8"):
        SpikformerPatchStem(embed_dims=12)
    with pytest.raises(ValueError, match="positive finite"):
        SpikformerMLP(4, 8, 4, tau=float("nan"))

    block = SpikformerBlock(dim=8, num_heads=2)
    with pytest.raises(ValueError, match=r"\[T, N, C, H, W\]"):
        block(torch.randn(2, 1, 8, 4))
    with pytest.raises(ValueError, match="expected C=8"):
        block(torch.randn(2, 1, 4, 2, 2))

    stem = SpikformerPatchStem(
        img_size_h=32,
        img_size_w=32,
        in_channels=3,
        embed_dims=32,
    )
    with pytest.raises(ValueError, match="patch-stem input"):
        stem(torch.randn(1, 3, 32, 32))
    with pytest.raises(ValueError, match="expected patch-stem C=3"):
        stem(torch.randn(2, 1, 1, 32, 32))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"T": 0}, "T must be a positive integer"),
        ({"in_channels": 0}, "in_channels must be a positive integer"),
        ({"img_size_h": 0}, "img_size_h must be a positive integer"),
        ({"img_size_w": 0}, "img_size_w must be a positive integer"),
        ({"num_classes": 0}, "num_classes must be a positive integer"),
        ({"depths": 0}, "depths must be a positive integer"),
        ({"patch_size": 8}, "patch_size=16"),
        ({"embed_dims": 30}, "divisible by 8"),
        ({"embed_dims": 24, "num_heads": 5}, "divisible by num_heads"),
        ({"mlp_ratio": 0.0}, "mlp_ratio must be a positive"),
        ({"tau": 1.0}, "tau must be greater than 1"),
    ],
)
def test_spikformer_constructor_validation(overrides, message):
    with pytest.raises(ValueError, match=message):
        _tiny_model(**overrides)


def test_spikformer_input_validation():
    model = _tiny_model()
    with pytest.raises(ValueError, match="expected 4D image input"):
        model(torch.randn(3, 32, 32))
    with pytest.raises(ValueError, match="expected 4D input C=3"):
        model(torch.randn(1, 1, 32, 32))
    with pytest.raises(ValueError, match="expected 5D input C=3"):
        model(torch.randn(3, 1, 1, 32, 32))

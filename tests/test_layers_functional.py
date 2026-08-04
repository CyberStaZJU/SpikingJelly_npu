import torch
from torch import nn

from spikingjelly_npu.activation_based import functional, layer, neuron
from spikingjelly_npu.npu.amp import npu_bf16_autocast


def test_linear_multi_step_matches_flattened_reference_and_state_dict():
    torch.manual_seed(1)
    module = layer.Linear(5, 3, step_mode="m")
    x = torch.randn(4, 2, 5)
    actual = module(x)
    expected = nn.functional.linear(x, module.weight, module.bias)
    torch.testing.assert_close(actual, expected)
    reference = nn.Linear(5, 3)
    reference.load_state_dict(module.state_dict())
    torch.testing.assert_close(reference(x), actual)


def test_conv_bn_pool_flatten_multi_step_shapes():
    torch.manual_seed(2)
    x = torch.randn(3, 2, 1, 8, 8)
    conv = layer.Conv2d(1, 4, 3, padding=1, step_mode="m")
    bn = layer.BatchNorm2d(4, step_mode="m")
    pool = layer.AvgPool2d(2, step_mode="m")
    flatten = layer.Flatten(step_mode="m")
    output = flatten(pool(bn(conv(x))))
    assert output.shape == (3, 2, 4 * 4 * 4)


def test_seq_to_ann_forward_matches_step_loop_for_stateless_module():
    torch.manual_seed(3)
    x = torch.randn(5, 3, 7)
    linear = nn.Linear(7, 4)
    packed = functional.seq_to_ann_forward(x, linear)
    looped = functional.multi_step_forward(x, linear)
    torch.testing.assert_close(packed, looped)


def test_seq_to_ann_container_keeps_batch_norm_in_fp32_bf16_profile(monkeypatch):
    calls = []

    class FakeAutocast:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(torch, "autocast", lambda **kwargs: FakeAutocast(**kwargs))
    container = layer.SeqToANNContainer(nn.Identity(), nn.BatchNorm1d(3)).eval()
    x = torch.randn(2, 4, 3, dtype=torch.bfloat16)

    with npu_bf16_autocast():
        output = container(x)

    assert output.dtype == torch.bfloat16
    assert calls == [
        {
            "device_type": "npu",
            "enabled": True,
            "cache_enabled": False,
            "dtype": torch.bfloat16,
        },
        {"device_type": "npu", "enabled": False},
    ]


def test_direct_batch_norm_wrappers_use_fp32_island_and_restore_public_bf16(
    monkeypatch,
):
    original_autocast = torch.autocast
    disabled_calls = []

    class CPUAutocast:
        def __init__(self, **kwargs):
            if not kwargs.get("enabled", True):
                disabled_calls.append(kwargs)
                self.context = original_autocast(device_type="cpu", enabled=False)
            else:
                self.context = original_autocast(
                    device_type="cpu", dtype=torch.bfloat16, cache_enabled=False
                )

        def __enter__(self):
            return self.context.__enter__()

        def __exit__(self, exc_type, exc, traceback):
            return self.context.__exit__(exc_type, exc, traceback)

    monkeypatch.setattr(torch, "autocast", lambda **kwargs: CPUAutocast(**kwargs))
    cases = (
        (layer.BatchNorm1d(3).eval(), torch.randn(4, 3, dtype=torch.bfloat16)),
        (layer.BatchNorm2d(3).eval(), torch.randn(4, 3, 5, 5, dtype=torch.bfloat16)),
        (layer.BatchNorm3d(3).eval(), torch.randn(2, 3, 4, 5, 5, dtype=torch.bfloat16)),
        (
            layer.BatchNorm1d(3, step_mode="m").eval(),
            torch.randn(2, 4, 3, dtype=torch.bfloat16),
        ),
    )

    with npu_bf16_autocast():
        outputs = [module(value) for module, value in cases]

    assert all(output.dtype == torch.bfloat16 for output in outputs)
    assert all(module.running_mean.dtype == torch.float32 for module, _ in cases)
    assert len(disabled_calls) == len(cases)


def test_average_pool_wrappers_reduce_in_fp32_and_restore_public_bf16(monkeypatch):
    original_autocast = torch.autocast
    disabled_calls = []

    class CPUAutocast:
        def __init__(self, **kwargs):
            if not kwargs.get("enabled", True):
                disabled_calls.append(kwargs)
                self.context = original_autocast(device_type="cpu", enabled=False)
            else:
                self.context = original_autocast(
                    device_type="cpu", dtype=torch.bfloat16, cache_enabled=False
                )

        def __enter__(self):
            return self.context.__enter__()

        def __exit__(self, exc_type, exc, traceback):
            return self.context.__exit__(exc_type, exc, traceback)

    monkeypatch.setattr(torch, "autocast", lambda **kwargs: CPUAutocast(**kwargs))
    x = torch.randn(2, 3, 4, 8, 8, dtype=torch.bfloat16)
    pool = layer.AvgPool2d(2, step_mode="m")
    adaptive = layer.AdaptiveAvgPool2d((1, 1), step_mode="m")

    with npu_bf16_autocast():
        pooled = pool(x)
        reduced = adaptive(pooled)

    assert pooled.dtype == torch.bfloat16
    assert reduced.dtype == torch.bfloat16
    assert len(disabled_calls) == 2


def test_seq_to_ann_container_and_voting_layer_reduce_in_fp32(monkeypatch):
    original_autocast = torch.autocast
    disabled_calls = []

    class CPUAutocast:
        def __init__(self, **kwargs):
            if not kwargs.get("enabled", True):
                disabled_calls.append(kwargs)
                self.context = original_autocast(device_type="cpu", enabled=False)
            else:
                self.context = original_autocast(
                    device_type="cpu", dtype=torch.bfloat16, cache_enabled=False
                )

        def __enter__(self):
            return self.context.__enter__()

        def __exit__(self, exc_type, exc, traceback):
            return self.context.__exit__(exc_type, exc, traceback)

    monkeypatch.setattr(torch, "autocast", lambda **kwargs: CPUAutocast(**kwargs))
    container = layer.SeqToANNContainer(
        nn.AvgPool2d(2), nn.AdaptiveAvgPool2d((1, 1))
    )
    voting = layer.VotingLayer(voting_size=2)

    with npu_bf16_autocast():
        pooled = container(torch.randn(2, 3, 1, 4, 4, dtype=torch.bfloat16))
        voted = voting(torch.randn(3, 6, dtype=torch.bfloat16))

    assert pooled.dtype == torch.bfloat16
    assert voted.dtype == torch.bfloat16
    assert len(disabled_calls) == 3


def test_network_configuration_helpers():
    network = nn.Sequential(
        layer.Linear(2, 2),
        neuron.IFNode(),
    )
    functional.set_step_mode(network, "m")
    assert network[0].step_mode == "m"
    assert network[1].step_mode == "m"
    functional.set_backend(network, "npu")
    assert network[1].backend == "npu"

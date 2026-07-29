import torch
from torch import nn

from spikingjelly_npu.activation_based import functional, layer, neuron


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

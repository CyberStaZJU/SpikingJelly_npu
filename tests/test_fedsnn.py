import torch

from spikingjelly_npu.fedsnn import MultiStepLIF, PackedBNTTConvNet, PoissonEncoder


def test_poisson_encoder_shape_values_and_seed_repeatability():
    encoder = PoissonEncoder(4)
    inputs = torch.tensor([[0.0, 0.5, 1.0]])
    torch.manual_seed(9)
    first = encoder(inputs)
    torch.manual_seed(9)
    second = encoder(inputs)
    assert first.shape == (4, 1, 3)
    assert torch.equal(first, second)
    assert torch.equal(first[..., 0], torch.zeros_like(first[..., 0]))
    assert torch.equal(first[..., 2], torch.ones_like(first[..., 2]))


def test_multistep_lif_matches_manual_soft_reset():
    module = MultiStepLIF(tau=2.0, decay_input=False, v_threshold=1.0, v_reset=None)
    currents = torch.tensor([[[0.8]], [[0.8]], [[0.0]]])
    spikes = module(currents)
    torch.testing.assert_close(spikes, torch.tensor([[[0.0]], [[1.0]], [[0.0]]]))


def test_packed_convnet_matches_stepwise_reference_in_eval_and_gradients():
    torch.manual_seed(11)
    packed = PackedBNTTConvNet(
        input_channels=1,
        classes=3,
        time_steps=3,
        channels=(4, 6),
        hidden_features=8,
        pooled_size=2,
    ).eval()
    stepwise = PackedBNTTConvNet(
        input_channels=1,
        classes=3,
        time_steps=3,
        channels=(4, 6),
        hidden_features=8,
        pooled_size=2,
    ).eval()
    stepwise.load_state_dict(packed.state_dict())
    input_seq_a = torch.rand(3, 2, 1, 8, 8, requires_grad=True)
    input_seq_b = input_seq_a.detach().clone().requires_grad_(True)
    output_a = packed.forward_current_seq(input_seq_a)
    output_b = stepwise.forward_current_seq_stepwise(input_seq_b)
    torch.testing.assert_close(output_a, output_b, rtol=1e-5, atol=1e-6)
    output_a.sum().backward()
    output_b.sum().backward()
    torch.testing.assert_close(input_seq_a.grad, input_seq_b.grad, rtol=2e-5, atol=2e-6)
    for (_, parameter_a), (_, parameter_b) in zip(
        packed.named_parameters(), stepwise.named_parameters(), strict=True
    ):
        torch.testing.assert_close(parameter_a.grad, parameter_b.grad, rtol=2e-5, atol=2e-6)


def test_packed_convnet_input_forward_and_backward():
    torch.manual_seed(12)
    model = PackedBNTTConvNet(1, 5, 2, channels=(4, 4), hidden_features=8, pooled_size=2)
    inputs = torch.rand(3, 1, 8, 8)
    logits = model(inputs)
    assert logits.shape == (3, 5)
    logits.square().mean().backward()
    assert model.conv1.weight.grad is not None

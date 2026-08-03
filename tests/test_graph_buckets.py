import warnings

import pytest
import torch
from torch import nn

from spikingjelly_npu.activation_based import base
from spikingjelly_npu.npu.graph import (
    GraphBucketRunner,
    GraphBucketSpec,
    StaticGraphRunner,
    _CaptureStateError,
)


class FakeNPU:
    def __init__(self, capture=None):
        self.capture = capture
        self.capture_calls = []
        self.rng_state = torch.tensor([37], dtype=torch.uint8)
        self.restored_rng_states = []

    def get_rng_state(self, device):
        return self.rng_state.clone()

    def set_rng_state(self, state, device):
        self.restored_rng_states.append((state.clone(), device))

    def make_graphed_callables(self, wrapper, sample_args, num_warmup_iters):
        self.capture_calls.append((wrapper, sample_args, num_warmup_iters))
        if self.capture is not None:
            return self.capture(wrapper, sample_args, num_warmup_iters)
        return wrapper


class RecurrentMaskModel(nn.Module):
    _spikingjelly_npu_graph_safe = True

    def forward(self, inputs, state, *, mask, scale=1.0):
        hidden, cell = state
        return (inputs + hidden - cell) * mask.to(inputs.dtype) * scale


class LinearModel(nn.Module):
    _spikingjelly_npu_graph_safe = True

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)
        self.calls = 0

    def forward(self, inputs):
        self.calls += 1
        return self.linear(inputs)


def enable_fake_npu(monkeypatch, runner, fake_npu):
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    monkeypatch.setattr(runner, "_execution_device_type", lambda tensors: "npu")


def assert_same(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_nested_tuple_state_kwargs_mask_and_five_changed_replays(monkeypatch):
    model = RecurrentMaskModel().eval()
    sample = torch.zeros(2, 3)
    state = (torch.ones(2, 3), torch.full((2, 3), 0.25))
    mask = torch.tensor([[True, False, True], [False, True, True]])
    bucket = GraphBucketSpec(
        (sample, state),
        {"scale": 2.0, "mask": mask},
        name="recurrent-mask",
    )
    runner = GraphBucketRunner(model, [bucket])
    fake_npu = FakeNPU()
    enable_fake_npu(monkeypatch, runner, fake_npu)

    for replay in range(5):
        inputs = torch.full((2, 3), float(replay + 1))
        hidden = torch.full((2, 3), float(replay) / 10.0)
        cell = torch.full((2, 3), float(replay) / 20.0)
        replay_mask = torch.tensor(
            [[replay % 2 == 0, True, False], [True, replay % 2 == 1, True]]
        )
        expected = model(inputs, (hidden, cell), mask=replay_mask, scale=2.0)
        actual = runner(
            inputs,
            (hidden, cell),
            scale=2.0,
            mask=replay_mask,
        )
        assert_same(actual, expected)
        assert runner.last_route.backend == "npugraph"
        assert runner.last_route.captured
        assert "recurrent-mask" in runner.last_route.reason

    assert len(fake_npu.capture_calls) == 1
    _, sample_args, _ = fake_npu.capture_calls[0]
    assert len(sample_args) == 4


def test_keyword_order_is_canonical_but_static_values_are_exact(monkeypatch):
    model = RecurrentMaskModel().eval()
    sample = torch.zeros(2, 3)
    state = (torch.zeros(2, 3), torch.zeros(2, 3))
    mask = torch.ones(2, 3, dtype=torch.bool)
    runner = GraphBucketRunner(
        model,
        [GraphBucketSpec((sample, state), {"mask": mask, "scale": 3.0})],
    )
    fake_npu = FakeNPU()
    enable_fake_npu(monkeypatch, runner, fake_npu)

    runner(sample, state, scale=3.0, mask=mask)
    assert runner.last_route.backend == "npugraph"

    eager = runner(sample, state, mask=mask, scale=4.0)
    assert_same(eager, model(sample, state, mask=mask, scale=4.0))
    assert runner.last_route.backend == "eager"
    assert "allowlist" in runner.last_route.reason
    assert len(fake_npu.capture_calls) == 1


def test_unknown_bucket_uses_observable_eager_fallback(monkeypatch):
    model = LinearModel().eval()
    runner = GraphBucketRunner(model, [GraphBucketSpec((torch.zeros(2, 3),))])
    fake_npu = FakeNPU()
    enable_fake_npu(monkeypatch, runner, fake_npu)

    inputs = torch.randn(3, 3)
    output = runner(inputs)

    assert_same(output, model.linear(inputs))
    assert runner.last_route.backend == "eager"
    assert "allowlist" in runner.last_route.reason
    assert fake_npu.capture_calls == []


def test_unknown_bucket_strict_raises_before_capture_or_eager(monkeypatch):
    model = LinearModel().eval()
    runner = GraphBucketRunner(
        model,
        [GraphBucketSpec((torch.zeros(2, 3),))],
        strict=True,
    )
    fake_npu = FakeNPU()
    enable_fake_npu(monkeypatch, runner, fake_npu)

    with pytest.raises(RuntimeError, match="allowlist"):
        runner(torch.randn(3, 3))

    assert model.calls == 0
    assert fake_npu.capture_calls == []


def test_max_bucket_validation_and_duplicate_rejection():
    specs = [GraphBucketSpec((torch.zeros(size, 3),)) for size in range(1, 10)]

    GraphBucketRunner(nn.Identity(), specs[:8])
    with pytest.raises(ValueError, match="exceeding maximum 8"):
        GraphBucketRunner(nn.Identity(), specs)
    with pytest.raises(ValueError, match="max_buckets must be positive"):
        GraphBucketRunner(nn.Identity(), specs[:1], max_buckets=0)
    with pytest.raises(ValueError, match="must be unique"):
        GraphBucketRunner(nn.Identity(), [specs[0], specs[0]])


def test_train_eval_identity_is_separate(monkeypatch):
    model = LinearModel().eval()
    sample = torch.zeros(2, 3)
    runner = GraphBucketRunner(
        model,
        [GraphBucketSpec((sample,))],
        allow_training=True,
        require_deterministic_training=False,
    )
    fake_npu = FakeNPU()
    enable_fake_npu(monkeypatch, runner, fake_npu)

    runner(sample)
    model.train()
    runner(sample)
    model.eval()
    runner(sample)

    assert len(fake_npu.capture_calls) == 2
    assert runner.last_route.backend == "npugraph"


def test_exact_signature_checks_dtype_layout_and_requires_grad(monkeypatch):
    model = nn.Identity().eval()
    model._spikingjelly_npu_graph_safe = True
    sample = torch.zeros(2, 3, dtype=torch.float32)
    runner = GraphBucketRunner(model, [GraphBucketSpec((sample,))])
    fake_npu = FakeNPU()
    enable_fake_npu(monkeypatch, runner, fake_npu)

    runner(sample)
    assert runner.last_route.backend == "npugraph"

    for mismatch in (
        torch.zeros(2, 3, dtype=torch.float64),
        torch.zeros(2, 3, dtype=torch.float32, requires_grad=True),
        torch.sparse_coo_tensor(
            torch.tensor([[0], [1]]),
            torch.tensor([1.0]),
            size=(2, 3),
            check_invariants=True,
        ),
    ):
        assert runner(mismatch) is mismatch
        assert runner.last_route.backend == "eager"
        assert "allowlist" in runner.last_route.reason

    assert len(fake_npu.capture_calls) == 1


def test_parameter_replacement_invalidates_all_bucket_captures(monkeypatch):
    model = LinearModel().eval()
    runner = GraphBucketRunner(
        model,
        [
            GraphBucketSpec((torch.zeros(2, 3),)),
            GraphBucketSpec((torch.zeros(4, 3),)),
        ],
    )
    fake_npu = FakeNPU()
    enable_fake_npu(monkeypatch, runner, fake_npu)

    runner(torch.randn(2, 3))
    runner(torch.randn(4, 3))
    assert len(fake_npu.capture_calls) == 2

    model.linear.weight = nn.Parameter(model.linear.weight.detach().clone())
    runner(torch.randn(2, 3))
    runner(torch.randn(4, 3))

    assert len(fake_npu.capture_calls) == 4


def test_hooks_force_eager_without_capture(monkeypatch):
    model = LinearModel().eval()
    runner = GraphBucketRunner(model, [GraphBucketSpec((torch.zeros(2, 3),))])
    fake_npu = FakeNPU()
    enable_fake_npu(monkeypatch, runner, fake_npu)
    handle = model.linear.register_forward_hook(lambda module, args, output: output)
    try:
        output = runner(torch.randn(2, 3))
    finally:
        handle.remove()

    assert output.shape == (2, 2)
    assert runner.last_route.backend == "eager"
    assert "hooks" in runner.last_route.reason
    assert fake_npu.capture_calls == []


def test_per_bucket_capture_failure_isolation(monkeypatch):
    model = LinearModel().eval()
    runner = GraphBucketRunner(
        model,
        [
            GraphBucketSpec((torch.zeros(2, 3),), name="bad"),
            GraphBucketSpec((torch.zeros(4, 3),), name="good"),
        ],
    )

    def capture(wrapper, sample_args, num_warmup_iters):
        if sample_args[0].shape[0] == 2:
            raise RuntimeError("unsupported exact shape")
        return wrapper

    fake_npu = FakeNPU(capture)
    enable_fake_npu(monkeypatch, runner, fake_npu)

    with pytest.warns(RuntimeWarning, match="unsupported exact shape"):
        failed_output = runner(torch.randn(2, 3))
    assert failed_output.shape == (2, 2)
    assert runner.last_route.backend == "eager"

    good_output = runner(torch.randn(4, 3))
    assert good_output.shape == (4, 2)
    assert runner.last_route.backend == "npugraph"

    with warnings.catch_warnings(record=True) as seen:
        runner(torch.randn(2, 3))
    assert seen == []
    assert "prior capture failed" in runner.last_route.reason
    assert len(fake_npu.capture_calls) == 2
    assert len(runner.capture_errors) == 1


def test_fatal_cleanup_failure_poisons_entire_runner(monkeypatch):
    model = LinearModel().eval()
    runner = GraphBucketRunner(
        model,
        [
            GraphBucketSpec((torch.zeros(2, 3),)),
            GraphBucketSpec((torch.zeros(4, 3),)),
        ],
    )
    fake_npu = FakeNPU()
    enable_fake_npu(monkeypatch, runner, fake_npu)
    monkeypatch.setattr(
        runner,
        "_restore_buffers",
        lambda snapshot: (_ for _ in ()).throw(RuntimeError("buffer cleanup failed")),
    )

    with pytest.raises(_CaptureStateError, match="model buffers"):
        runner(torch.randn(2, 3))
    calls_after_failure = model.calls

    for inputs in (torch.randn(4, 3), torch.randn(7, 3)):
        with pytest.raises(_CaptureStateError, match="model buffers"):
            runner(inputs)

    assert model.calls == calls_after_failure
    assert len(fake_npu.capture_calls) == 1


def test_capture_restores_memorymodule_runtime_memories(monkeypatch):
    class MemoryModel(base.MemoryModule):
        _spikingjelly_npu_graph_safe = True

        def __init__(self):
            super().__init__()
            self.register_memory("state", 0.0)

        def single_step_forward(self, inputs):
            return inputs

    model = MemoryModel().eval()
    sample = torch.zeros(2, 3)
    runner = GraphBucketRunner(model, [GraphBucketSpec((sample,))])

    def capture(wrapper, sample_args, num_warmup_iters):
        wrapper.model.state = torch.ones_like(sample_args[0])
        return lambda *tensor_args: tensor_args[0]

    fake_npu = FakeNPU(capture)
    enable_fake_npu(monkeypatch, runner, fake_npu)

    runner(sample)

    assert model.state == 0.0
    assert runner.last_route.backend == "npugraph"


def test_cleanup_order_restores_all_categories_after_early_failures(monkeypatch):
    model = LinearModel().eval()
    sample = torch.zeros(2, 3)
    runner = GraphBucketRunner(model, [GraphBucketSpec((sample,))])
    fake_npu = FakeNPU()
    enable_fake_npu(monkeypatch, runner, fake_npu)
    calls = []

    def fail_buffers(snapshot):
        calls.append("buffers")
        raise RuntimeError("buffer restore failed")

    def fail_gradients(snapshot):
        calls.append("gradients")
        raise RuntimeError("gradient restore failed")

    def fail_memories(snapshot):
        calls.append("memories")
        raise RuntimeError("memory restore failed")

    def fail_training(snapshot):
        calls.append("training")
        raise RuntimeError("training restore failed")

    def fail_cpu_rng(state):
        calls.append("cpu_rng")
        raise RuntimeError("CPU RNG restore failed")

    def fail_structure(snapshot):
        calls.append("structure")
        raise RuntimeError("structure restore failed")

    monkeypatch.setattr(runner, "_restore_buffers", fail_buffers)
    monkeypatch.setattr(runner, "_restore_gradients", fail_gradients)
    monkeypatch.setattr(runner, "_restore_runtime_memories", fail_memories)
    monkeypatch.setattr(runner, "_restore_training_modes", fail_training)
    monkeypatch.setattr(torch.random, "set_rng_state", fail_cpu_rng)
    monkeypatch.setattr(runner, "_validate_capture_structure", fail_structure)
    fake_npu.set_rng_state = lambda state, device: (
        calls.append("npu_rng"),
        (_ for _ in ()).throw(RuntimeError("NPU RNG restore failed")),
    )[1]

    with pytest.raises(_CaptureStateError) as error_info:
        runner(sample)

    assert calls == [
        "buffers",
        "gradients",
        "memories",
        "training",
        "cpu_rng",
        "npu_rng",
        "structure",
    ]
    message = str(error_info.value)
    assert "model buffers" in message
    assert "parameter gradients" in message
    assert "runtime memories" in message
    assert "module training state" in message
    assert "CPU RNG" in message
    assert "NPU RNG" in message
    assert "model structure" in message


def test_capture_restores_training_state_changed_by_warmup(monkeypatch):
    class ModeMutatingModel(nn.Module):
        _spikingjelly_npu_graph_safe = True

        def __init__(self):
            super().__init__()
            self.child = nn.Identity()

        def forward(self, inputs):
            return inputs

    model = ModeMutatingModel().eval()
    sample = torch.zeros(2, 3)
    runner = GraphBucketRunner(model, [GraphBucketSpec((sample,))])

    def capture(wrapper, sample_args, num_warmup_iters):
        wrapper.model.child.train()
        return lambda *tensor_args: tensor_args[0]

    fake_npu = FakeNPU(capture)
    enable_fake_npu(monkeypatch, runner, fake_npu)
    before = tuple(module.training for module in model.modules())

    runner(sample)

    after = tuple(module.training for module in model.modules())
    assert after == before


def test_dropout_training_rejected_unless_explicitly_unsafe(monkeypatch):
    model = nn.Sequential(nn.Linear(3, 3), nn.Dropout(p=0.5)).train()
    model._spikingjelly_npu_graph_safe = True
    sample = torch.zeros(2, 3)
    runner = GraphBucketRunner(
        model,
        [GraphBucketSpec((sample,))],
        allow_training=True,
        require_deterministic_training=False,
    )
    fake_npu = FakeNPU()
    enable_fake_npu(monkeypatch, runner, fake_npu)

    runner(sample)

    assert runner.last_route.backend == "eager"
    assert "RNG-sensitive" in runner.last_route.reason
    assert "allow_unsafe_rng_training=True" in runner.last_route.reason
    assert fake_npu.capture_calls == []

    unsafe_runner = GraphBucketRunner(
        model,
        [GraphBucketSpec((sample,))],
        allow_training=True,
        require_deterministic_training=False,
        allow_unsafe_rng_training=True,
    )
    unsafe_fake_npu = FakeNPU()
    enable_fake_npu(monkeypatch, unsafe_runner, unsafe_fake_npu)
    unsafe_runner(sample)
    assert unsafe_runner.last_route.backend == "npugraph"
    assert len(unsafe_fake_npu.capture_calls) == 1


def test_static_graph_runner_facade_compatibility(monkeypatch):
    model = LinearModel().eval()
    runner = StaticGraphRunner(model, batch_size=2)
    fake_npu = FakeNPU()
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    monkeypatch.setattr(runner, "_execution_device_type", lambda inputs: "npu")

    first = runner(torch.randn(2, 3))
    second = runner(torch.randn(2, 3))
    partial = runner(torch.randn(1, 3))

    assert first.shape == second.shape == (2, 2)
    assert partial.shape == (1, 2)
    assert len(fake_npu.capture_calls) == 1
    assert runner.last_route.backend == "eager"
    assert "batch shape" in runner.last_route.reason

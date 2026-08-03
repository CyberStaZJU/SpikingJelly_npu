import warnings

import pytest
import torch
from torch import nn

from spikingjelly_npu.activation_based import base
from spikingjelly_npu.npu import (
    GraphBucketRunner as ExportedGraphBucketRunner,
)
from spikingjelly_npu.npu import (
    GraphBucketSpec as ExportedGraphBucketSpec,
)
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


def test_train_eval_change_replaces_one_bounded_bucket_capture(monkeypatch):
    model = LinearModel().eval()
    sample = torch.zeros(2, 3)
    runner = GraphBucketRunner(
        model,
        [GraphBucketSpec((sample,))],
        max_buckets=1,
        allow_training=True,
        require_deterministic_training=False,
    )
    fake_npu = FakeNPU()
    enable_fake_npu(monkeypatch, runner, fake_npu)

    for _ in range(3):
        runner(sample)
        assert len(runner._captures) == 1
        model.train()
        runner(sample)
        assert len(runner._captures) == 1
        model.eval()

    assert len(fake_npu.capture_calls) == 6
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


def test_exact_signature_checks_stride_storage_offset_and_memory_format(monkeypatch):
    class SingleInputModel(nn.Module):
        _spikingjelly_npu_graph_safe = True

        def forward(self, inputs):
            return inputs.clone()

    base = torch.arange(48, dtype=torch.float32).reshape(4, 12)
    sample = base[:, 1:7]
    runner = GraphBucketRunner(SingleInputModel().eval(), [GraphBucketSpec((sample,))])
    fake_npu = FakeNPU()
    enable_fake_npu(monkeypatch, runner, fake_npu)

    matching = torch.arange(48, dtype=torch.float32).reshape(4, 12)[:, 1:7]
    runner(matching)
    assert runner.last_route.backend == "npugraph"

    mismatches = (
        matching.clone(),
        torch.arange(52, dtype=torch.float32).reshape(4, 13)[:, 1:7],
        torch.zeros(4, 6).t().contiguous().t(),
    )
    for mismatch in mismatches:
        output = runner(mismatch)
        assert_same(output, mismatch)
        assert runner.last_route.backend == "eager"
        assert "allowlist" in runner.last_route.reason

    contiguous_4d = torch.zeros(2, 3, 4, 5)
    channels_last = contiguous_4d.contiguous(memory_format=torch.channels_last)
    format_runner = GraphBucketRunner(
        SingleInputModel().eval(),
        [GraphBucketSpec((channels_last,))],
    )
    format_fake_npu = FakeNPU()
    enable_fake_npu(monkeypatch, format_runner, format_fake_npu)
    output = format_runner(contiguous_4d)
    assert_same(output, contiguous_4d)
    assert format_runner.last_route.backend == "eager"
    assert format_fake_npu.capture_calls == []


def test_exact_signature_supports_empty_storage_groups(monkeypatch):
    model = RecurrentMaskModel().eval()
    first = torch.empty(0, 3)
    second = torch.empty(0, 3)
    mask = torch.empty(0, 3, dtype=torch.bool)
    runner = GraphBucketRunner(
        model,
        [GraphBucketSpec((first, (second, second)), {"mask": mask})],
    )
    fake_npu = FakeNPU()
    enable_fake_npu(monkeypatch, runner, fake_npu)

    replay_first = torch.empty(0, 3)
    replay_second = torch.empty(0, 3)
    replay_mask = torch.empty(0, 3, dtype=torch.bool)
    output = runner(
        replay_first,
        (replay_second, replay_second),
        mask=replay_mask,
    )

    assert output.shape == (0, 3)
    assert runner.last_route.backend == "npugraph"
    assert len(fake_npu.capture_calls) == 1


def test_exact_signature_encodes_alias_and_view_relationships(monkeypatch):
    class AliasModel(nn.Module):
        _spikingjelly_npu_graph_safe = True

        def forward(self, inputs, state, *, mask):
            return inputs + state + mask

    storage = torch.arange(64, dtype=torch.float32).reshape(4, 16)
    inputs = storage[:, :4]
    state = storage[:, 4:8]
    mask = storage[:, 8:12]
    runner = GraphBucketRunner(
        AliasModel().eval(),
        [GraphBucketSpec((inputs, state), {"mask": mask})],
    )
    fake_npu = FakeNPU()
    enable_fake_npu(monkeypatch, runner, fake_npu)

    replay_storage = torch.arange(64, dtype=torch.float32).reshape(4, 16)
    runner(
        replay_storage[:, :4],
        replay_storage[:, 4:8],
        mask=replay_storage[:, 8:12],
    )
    assert runner.last_route.backend == "npugraph"

    independent_inputs = replay_storage[:, :4].clone()
    independent_state = replay_storage[:, 4:8].clone()
    independent_mask = replay_storage[:, 8:12].clone()
    eager = runner(independent_inputs, independent_state, mask=independent_mask)
    assert_same(eager, independent_inputs + independent_state + independent_mask)
    assert runner.last_route.backend == "eager"
    assert "allowlist" in runner.last_route.reason

    partially_aliased_storage = torch.arange(64, dtype=torch.float32).reshape(4, 16)
    eager = runner(
        partially_aliased_storage[:, :4],
        partially_aliased_storage[:, 4:8],
        mask=partially_aliased_storage[:, 8:12].clone(),
    )
    assert eager.shape == (4, 4)
    assert runner.last_route.backend == "eager"
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
    assert len(fake_npu.capture_calls) == 3
    assert set(runner._captures) == {0}

    runner(torch.randn(4, 3))
    assert len(fake_npu.capture_calls) == 4
    assert set(runner._captures) == {0, 1}


def test_parameter_and_buffer_version_changes_recapture_current_bucket(monkeypatch):
    class VersionedModel(nn.Module):
        _spikingjelly_npu_graph_safe = True

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(3))
            self.register_buffer("bias", torch.zeros(3))

        def forward(self, inputs):
            return inputs * self.weight + self.bias

    model = VersionedModel().eval()
    sample = torch.zeros(2, 3)
    runner = GraphBucketRunner(
        model,
        [GraphBucketSpec((sample,))],
        max_buckets=1,
    )
    fake_npu = FakeNPU()
    enable_fake_npu(monkeypatch, runner, fake_npu)

    runner(sample)
    with torch.no_grad():
        model.weight.add_(1)
    runner(sample)
    model.bias.add_(2)
    runner(sample)

    assert len(fake_npu.capture_calls) == 3
    assert len(runner._captures) == 1


def test_parameter_version_change_invalidates_every_bucket_immediately(monkeypatch):
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

    runner(torch.zeros(2, 3))
    runner(torch.zeros(4, 3))
    assert set(runner._captures) == {0, 1}

    with torch.no_grad():
        model.linear.weight.add_(1)
    runner(torch.zeros(2, 3))

    assert len(fake_npu.capture_calls) == 3
    assert set(runner._captures) == {0}


def test_total_capture_slots_stay_bounded_across_subtree_training_modes(monkeypatch):
    class MixedModeModel(nn.Module):
        _spikingjelly_npu_graph_safe = True

        def __init__(self):
            super().__init__()
            self.left = nn.Identity()
            self.right = nn.Identity()

        def forward(self, inputs):
            return self.right(self.left(inputs))

    model = MixedModeModel().eval()
    runner = GraphBucketRunner(
        model,
        [
            GraphBucketSpec((torch.zeros(2, 3),)),
            GraphBucketSpec((torch.zeros(4, 3),)),
        ],
        max_buckets=2,
        allow_training=True,
        require_deterministic_training=False,
    )
    fake_npu = FakeNPU()
    enable_fake_npu(monkeypatch, runner, fake_npu)

    for left_training, right_training in (
        (False, False),
        (True, False),
        (False, True),
        (True, True),
        (False, False),
    ):
        model.left.train(left_training)
        model.right.train(right_training)
        runner(torch.zeros(2, 3))
        runner(torch.zeros(4, 3))
        assert len(runner._captures) <= runner.max_buckets == 2

    assert set(runner._captures) == {0, 1}


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


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("cpu", "model is not on an NPU"),
        ("graph_unsafe", "graph-safe"),
        ("training_disabled", "allow_training=True"),
        ("nondeterministic", "use_deterministic_algorithms"),
        ("rng_sensitive", "RNG-sensitive"),
    ],
)
def test_strict_known_fallbacks_raise_before_capture_or_eager(
    monkeypatch, case, reason
):
    class StrictModel(LinearModel):
        def __init__(self):
            super().__init__()
            self.dropout = nn.Identity()

        def forward(self, inputs):
            self.calls += 1
            return self.dropout(self.linear(inputs))

    model = StrictModel().eval()
    if case == "graph_unsafe":
        model._spikingjelly_npu_graph_safe = False
    if case in {"training_disabled", "nondeterministic", "rng_sensitive"}:
        model.train()
    if case == "rng_sensitive":
        model.dropout = nn.Dropout(p=0.5)
    runner = GraphBucketRunner(
        model,
        [GraphBucketSpec((torch.zeros(2, 3),))],
        strict=True,
        allow_training=case in {"nondeterministic", "rng_sensitive"},
        require_deterministic_training=case == "nondeterministic",
    )
    fake_npu = FakeNPU()
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    if case != "cpu":
        monkeypatch.setattr(runner, "_execution_device_type", lambda tensors: "npu")
    if case == "nondeterministic":
        monkeypatch.setattr(runner, "_deterministic_algorithms_enabled", lambda: False)

    with pytest.raises(RuntimeError, match=reason):
        runner(torch.zeros(2, 3))

    assert model.calls == 0
    assert fake_npu.capture_calls == []


def test_strict_hooks_raise_before_capture_or_eager(monkeypatch):
    model = LinearModel().eval()
    runner = GraphBucketRunner(
        model,
        [GraphBucketSpec((torch.zeros(2, 3),))],
        strict=True,
    )
    fake_npu = FakeNPU()
    enable_fake_npu(monkeypatch, runner, fake_npu)
    handle = model.linear.register_forward_hook(lambda module, args, output: output)
    try:
        with pytest.raises(RuntimeError, match="hooks"):
            runner(torch.zeros(2, 3))
    finally:
        handle.remove()

    assert model.calls == 0
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


def test_replay_launch_failure_never_runs_eager_and_poisons_runner(monkeypatch):
    model = LinearModel().eval()
    sample = torch.zeros(2, 3)

    def capture(wrapper, sample_args, num_warmup_iters):
        def fail_replay(*tensor_args):
            raise RuntimeError("launch failed")

        return fail_replay

    runner = GraphBucketRunner(model, [GraphBucketSpec((sample,))])
    fake_npu = FakeNPU(capture)
    enable_fake_npu(monkeypatch, runner, fake_npu)

    with pytest.raises(_CaptureStateError, match="replay failed") as error_info:
        runner(sample)
    assert "launch failed" in str(error_info.value.__cause__)
    calls_after_failure = model.calls

    for later in (sample, torch.zeros(3, 3)):
        with pytest.raises(_CaptureStateError, match="replay failed"):
            runner(later)

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


def test_graph_bucket_public_exports():
    assert ExportedGraphBucketRunner is GraphBucketRunner
    assert ExportedGraphBucketSpec is GraphBucketSpec


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

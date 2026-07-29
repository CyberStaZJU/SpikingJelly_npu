import warnings

import torch
from torch import nn

from spikingjelly_npu.npu.graph import StaticGraphRunner, _CaptureStateError


class DiagnosticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)
        self.calls = []

    def forward(self, inputs, scale=1.0):
        self.calls.append((inputs.shape[0], scale))
        return self.linear(inputs) * scale


def test_cpu_runner_stays_eager_and_reports_route():
    model = DiagnosticModel()
    runner = StaticGraphRunner(
        model,
        batch_size=4,
        allow_training=True,
        require_deterministic_training=False,
        assume_graph_safe=True,
    )
    output = runner(torch.randn(4, 3))
    assert output.shape == (4, 2)
    assert runner.backend == "eager"
    assert runner.last_route.reason == "model is not on an NPU"


def test_partial_batch_and_kwargs_stay_eager():
    model = DiagnosticModel()
    runner = StaticGraphRunner(
        model,
        batch_size=4,
        allow_training=True,
        require_deterministic_training=False,
        assume_graph_safe=True,
    )
    runner(torch.randn(2, 3))
    assert "batch shape" in runner.last_route.reason
    runner(torch.randn(4, 3), scale=2.0)
    assert "tensor-only" in runner.last_route.reason
    assert model.calls[-1] == (4, 2.0)


def test_parameterless_module_uses_input_device_for_routing(monkeypatch):
    model = nn.Identity().eval()
    runner = StaticGraphRunner(model, batch_size=4, assume_graph_safe=True)
    fake_inputs = torch.randn(4, 3)
    monkeypatch.setattr(runner, "_execution_device_type", lambda inputs: "npu")

    def capture(sample):
        runner._captured_training_state = runner._module_training_state()
        runner._captured_deterministic_state = runner._deterministic_capture_state()
        runner._captured_structure_signature = runner._module_structure_signature()
        runner._capture_signature = runner._input_signature(sample)
        return model

    monkeypatch.setattr(runner, "_capture", capture)
    output = runner(fake_inputs)

    assert output is fake_inputs
    assert runner.last_route.backend == "npugraph"


def test_unqualified_model_stays_eager_on_npu(monkeypatch):
    model = DiagnosticModel().eval()
    runner = StaticGraphRunner(model, batch_size=4)
    monkeypatch.setattr(type(runner), "device_type", property(lambda self: "npu"))
    capture_calls = []
    monkeypatch.setattr(runner, "_capture", lambda sample: capture_calls.append(sample))

    output = runner(torch.randn(4, 3))

    assert output.shape == (4, 2)
    assert runner.last_route.backend == "eager"
    assert "graph-safe" in runner.last_route.reason
    assert capture_calls == []


def test_graph_safe_child_does_not_qualify_unmarked_wrapper(monkeypatch):
    child = DiagnosticModel().eval()
    child._spikingjelly_npu_graph_safe = True
    model = nn.Sequential(child).eval()
    runner = StaticGraphRunner(model, batch_size=4)
    monkeypatch.setattr(type(runner), "device_type", property(lambda self: "npu"))
    capture_calls = []
    monkeypatch.setattr(runner, "_capture", lambda sample: capture_calls.append(sample))

    output = runner(torch.randn(4, 3))

    assert output.shape == (4, 2)
    assert runner.last_route.backend == "eager"
    assert "graph-safe" in runner.last_route.reason
    assert capture_calls == []


def test_aspy_module_can_attempt_capture(monkeypatch):
    model = nn.Sequential(DiagnosticModel()).eval()
    model[0].backend = "aspy"
    runner = StaticGraphRunner(model, batch_size=4, assume_graph_safe=True)
    monkeypatch.setattr(type(runner), "device_type", property(lambda self: "npu"))
    captured_samples = []

    def capture(sample):
        captured_samples.append(sample)
        runner._captured_training_state = runner._module_training_state()
        runner._captured_deterministic_state = runner._deterministic_capture_state()
        runner._captured_structure_signature = runner._module_structure_signature()
        runner._capture_signature = runner._input_signature(sample)
        return lambda inputs: model(inputs)

    monkeypatch.setattr(runner, "_capture", capture)
    output = runner(torch.randn(4, 3))

    assert output.shape == (4, 2)
    assert runner.last_route.backend == "npugraph"
    assert runner.last_route.captured
    assert len(captured_samples) == 1


def test_training_capture_requires_explicit_opt_in(monkeypatch):
    model = DiagnosticModel().train()
    runner = StaticGraphRunner(model, batch_size=4, assume_graph_safe=True)
    monkeypatch.setattr(type(runner), "device_type", property(lambda self: "npu"))
    capture_calls = []
    monkeypatch.setattr(runner, "_capture", lambda sample: capture_calls.append(sample))

    output = runner(torch.randn(4, 3))

    assert output.shape == (4, 2)
    assert runner.last_route.backend == "eager"
    assert "allow_training=True" in runner.last_route.reason
    assert capture_calls == []


def test_training_child_under_eval_wrapper_still_requires_opt_in(monkeypatch):
    model = nn.Sequential(DiagnosticModel().train()).eval()
    model[0].train()
    runner = StaticGraphRunner(model, batch_size=4, assume_graph_safe=True)
    monkeypatch.setattr(type(runner), "device_type", property(lambda self: "npu"))
    capture_calls = []
    monkeypatch.setattr(runner, "_capture", lambda sample: capture_calls.append(sample))

    output = runner(torch.randn(4, 3))

    assert output.shape == (4, 2)
    assert runner.last_route.backend == "eager"
    assert "allow_training=True" in runner.last_route.reason
    assert capture_calls == []


def test_warn_only_deterministic_algorithms_are_not_qualified(monkeypatch):
    monkeypatch.setattr(
        torch, "are_deterministic_algorithms_enabled", lambda: True
    )
    monkeypatch.setattr(
        torch, "is_deterministic_algorithms_warn_only_enabled", lambda: True
    )

    assert not StaticGraphRunner._deterministic_algorithms_enabled()


def test_training_capture_requires_deterministic_algorithms(monkeypatch):
    model = DiagnosticModel().train()
    runner = StaticGraphRunner(
        model,
        batch_size=4,
        allow_training=True,
        assume_graph_safe=True,
    )
    monkeypatch.setattr(type(runner), "device_type", property(lambda self: "npu"))
    monkeypatch.setattr(
        runner, "_deterministic_algorithms_enabled", lambda: False
    )
    capture_calls = []
    monkeypatch.setattr(runner, "_capture", lambda sample: capture_calls.append(sample))

    output = runner(torch.randn(4, 3))

    assert output.shape == (4, 2)
    assert runner.last_route.backend == "eager"
    assert "use_deterministic_algorithms(True, warn_only=False)" in runner.last_route.reason
    assert capture_calls == []


def test_deterministic_training_capture_can_run(monkeypatch):
    model = DiagnosticModel().train()
    runner = StaticGraphRunner(
        model,
        batch_size=4,
        allow_training=True,
        assume_graph_safe=True,
    )
    monkeypatch.setattr(type(runner), "device_type", property(lambda self: "npu"))
    monkeypatch.setattr(
        runner, "_deterministic_algorithms_enabled", lambda: True
    )
    captured_samples = []

    def capture(sample):
        captured_samples.append(sample)
        runner._captured_training_state = runner._module_training_state()
        runner._captured_deterministic_state = True
        runner._captured_structure_signature = runner._module_structure_signature()
        runner._capture_signature = runner._input_signature(sample)
        return lambda inputs: model(inputs)

    monkeypatch.setattr(runner, "_capture", capture)

    output = runner(torch.randn(4, 3))

    assert output.shape == (4, 2)
    assert runner.last_route.backend == "npugraph"
    assert len(captured_samples) == 1


def test_deterministic_mode_change_recaptures_training_graph(monkeypatch):
    model = DiagnosticModel().train()
    runner = StaticGraphRunner(
        model,
        batch_size=4,
        allow_training=True,
        require_deterministic_training=False,
        assume_graph_safe=True,
    )
    monkeypatch.setattr(type(runner), "device_type", property(lambda self: "npu"))
    deterministic_state = [False]
    monkeypatch.setattr(
        runner,
        "_deterministic_algorithms_enabled",
        lambda: deterministic_state[0],
    )
    capture_states = []

    def capture(sample):
        capture_states.append(deterministic_state[0])
        runner._captured_training_state = runner._module_training_state()
        runner._captured_deterministic_state = deterministic_state[0]
        runner._captured_structure_signature = runner._module_structure_signature()
        runner._capture_signature = runner._input_signature(sample)
        return lambda inputs: model(inputs)

    monkeypatch.setattr(runner, "_capture", capture)
    runner(torch.randn(4, 3))
    deterministic_state[0] = True
    runner(torch.randn(4, 3))

    assert capture_states == [False, True]
    assert runner.last_route.backend == "npugraph"


def test_deterministic_mode_change_does_not_recapture_eval_graph(monkeypatch):
    model = DiagnosticModel().eval()
    runner = StaticGraphRunner(model, batch_size=4, assume_graph_safe=True)
    monkeypatch.setattr(type(runner), "device_type", property(lambda self: "npu"))
    deterministic_state = [False]
    monkeypatch.setattr(
        runner,
        "_deterministic_algorithms_enabled",
        lambda: deterministic_state[0],
    )
    capture_calls = []

    def capture(sample):
        capture_calls.append(sample)
        runner._captured_training_state = runner._module_training_state()
        runner._captured_deterministic_state = runner._deterministic_capture_state()
        runner._captured_structure_signature = runner._module_structure_signature()
        runner._capture_signature = runner._input_signature(sample)
        return lambda inputs: model(inputs)

    monkeypatch.setattr(runner, "_capture", capture)
    runner(torch.randn(4, 3))
    deterministic_state[0] = True
    runner(torch.randn(4, 3))

    assert len(capture_calls) == 1
    assert runner.last_route.backend == "npugraph"


def test_mismatched_capture_signature_stays_eager(monkeypatch):
    model = DiagnosticModel()
    runner = StaticGraphRunner(
        model,
        batch_size=4,
        allow_training=True,
        require_deterministic_training=False,
        assume_graph_safe=True,
    )
    monkeypatch.setattr(type(runner), "device_type", property(lambda self: "npu"))
    runner._graphed = lambda inputs: model(inputs)
    runner._captured_training_state = runner._module_training_state()
    runner._captured_deterministic_state = runner._deterministic_capture_state()
    runner._captured_structure_signature = runner._module_structure_signature()
    runner._capture_signature = runner._input_signature(torch.randn(4, 3))

    inputs = torch.randn(4, 3, requires_grad=True)
    output = runner(inputs)
    assert output.shape == (4, 2)
    assert runner.last_route.backend == "eager"
    assert "signature" in runner.last_route.reason


def test_module_hooks_force_eager_without_capture(monkeypatch):
    model = DiagnosticModel()
    runner = StaticGraphRunner(
        model,
        batch_size=4,
        allow_training=True,
        require_deterministic_training=False,
        assume_graph_safe=True,
    )
    monkeypatch.setattr(type(runner), "device_type", property(lambda self: "npu"))
    capture_calls = []
    monkeypatch.setattr(runner, "_capture", lambda sample: capture_calls.append(sample))
    handle = model.linear.register_forward_hook(lambda module, args, output: output)
    try:
        output = runner(torch.randn(4, 3))
    finally:
        handle.remove()

    assert output.shape == (4, 2)
    assert runner.last_route.backend == "eager"
    assert "hooks" in runner.last_route.reason
    assert capture_calls == []
    assert model.calls == [(4, 1.0)]


def test_parameter_replacement_invalidates_existing_capture(monkeypatch):
    model = DiagnosticModel()
    runner = StaticGraphRunner(
        model,
        batch_size=4,
        allow_training=True,
        require_deterministic_training=False,
        assume_graph_safe=True,
    )
    monkeypatch.setattr(type(runner), "device_type", property(lambda self: "npu"))
    runner._graphed = lambda inputs: model(inputs)
    runner._captured_training_state = runner._module_training_state()
    runner._captured_deterministic_state = runner._deterministic_capture_state()
    runner._captured_structure_signature = runner._module_structure_signature()
    runner._capture_signature = runner._input_signature(torch.randn(4, 3))
    model.linear.weight = nn.Parameter(model.linear.weight.detach().clone())
    captured_samples = []

    def recapture(sample):
        captured_samples.append(sample)
        return lambda inputs: model(inputs)

    monkeypatch.setattr(runner, "_capture", recapture)
    output = runner(torch.randn(4, 3))

    assert output.shape == (4, 2)
    assert len(captured_samples) == 1
    assert runner.last_route.backend == "npugraph"


def test_capture_preserves_mixed_module_training_modes(monkeypatch):
    class MixedModeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3, 3)
            self.frozen_bn = nn.BatchNorm1d(3).eval()

        def forward(self, inputs):
            return self.frozen_bn(self.linear(inputs))

    model = MixedModeModel().train()
    model.frozen_bn.eval()
    runner = StaticGraphRunner(
        model,
        batch_size=4,
        allow_training=True,
        require_deterministic_training=False,
        assume_graph_safe=True,
    )
    fake_npu_state = torch.tensor([5], dtype=torch.uint8)
    fake_npu = type("FakeNPU", (), {})()
    fake_npu.get_rng_state = lambda _device: fake_npu_state.clone()
    fake_npu.set_rng_state = lambda _state, _device: None
    fake_npu.make_graphed_callables = (
        lambda wrapper, sample_args, num_warmup_iters: wrapper
    )
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)

    before = tuple(module.training for module in model.modules())
    runner._capture(torch.zeros(4, 3))
    after = tuple(module.training for module in model.modules())
    assert before == after
    assert runner._captured_training_state == before


def test_capture_restores_buffers_and_rng(monkeypatch):
    class StatefulModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("counter", torch.zeros(()))

        def forward(self, inputs):
            self.counter.add_(1)
            return inputs + torch.rand_like(inputs)

    model = StatefulModel()
    runner = StaticGraphRunner(
        model,
        batch_size=4,
        allow_training=True,
        require_deterministic_training=False,
        assume_graph_safe=True,
    )
    fake_npu_state = torch.tensor([7], dtype=torch.uint8)
    restored_npu_states = []
    fake_npu = type("FakeNPU", (), {})()
    fake_npu.get_rng_state = lambda _device: fake_npu_state.clone()
    fake_npu.set_rng_state = lambda state, device: restored_npu_states.append(
        (state.clone(), device)
    )
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)

    def fake_make_graphed_callables(wrapper, sample_args, num_warmup_iters):
        for _ in range(num_warmup_iters + 1):
            wrapper(*sample_args)
            torch.rand(1)
        return wrapper

    fake_npu.make_graphed_callables = fake_make_graphed_callables
    sample = torch.zeros(4, 3)
    cpu_rng_state = torch.random.get_rng_state().clone()
    runner._capture(sample)

    torch.testing.assert_close(model.counter, torch.zeros(()))
    assert torch.equal(torch.random.get_rng_state(), cpu_rng_state)
    assert len(restored_npu_states) == 1
    assert torch.equal(restored_npu_states[0][0], fake_npu_state)
    assert restored_npu_states[0][1] == sample.device


def test_capture_restores_existing_and_missing_parameter_gradients(monkeypatch):
    model = DiagnosticModel()
    model.linear.weight.grad = torch.full_like(model.linear.weight, 3.0)
    assert model.linear.bias.grad is None
    runner = StaticGraphRunner(
        model,
        batch_size=4,
        allow_training=True,
        require_deterministic_training=False,
        assume_graph_safe=True,
    )
    fake_npu = type("FakeNPU", (), {})()
    fake_npu.get_rng_state = lambda _device: torch.tensor([9], dtype=torch.uint8)
    fake_npu.set_rng_state = lambda _state, _device: None

    def fake_make_graphed_callables(wrapper, sample_args, num_warmup_iters):
        model.linear.weight.grad.zero_()
        model.linear.bias.grad = torch.ones_like(model.linear.bias)
        return wrapper

    fake_npu.make_graphed_callables = fake_make_graphed_callables
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)

    runner._capture(torch.zeros(4, 3))

    torch.testing.assert_close(
        model.linear.weight.grad, torch.full_like(model.linear.weight, 3.0)
    )
    assert model.linear.bias.grad is None


def test_capture_failure_restores_buffers_and_rng(monkeypatch):
    class StatefulModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("counter", torch.zeros(()))

        def forward(self, inputs):
            self.counter.add_(1)
            return inputs

    model = StatefulModel()
    runner = StaticGraphRunner(
        model,
        batch_size=4,
        allow_training=True,
        require_deterministic_training=False,
        assume_graph_safe=True,
    )
    fake_npu_state = torch.tensor([11], dtype=torch.uint8)
    restored_npu_states = []
    fake_npu = type("FakeNPU", (), {})()
    fake_npu.get_rng_state = lambda _device: fake_npu_state.clone()
    fake_npu.set_rng_state = lambda state, device: restored_npu_states.append(
        (state.clone(), device)
    )
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)

    def fail_after_side_effect(wrapper, sample_args, num_warmup_iters):
        wrapper(*sample_args)
        torch.rand(1)
        raise RuntimeError("capture failed")

    fake_npu.make_graphed_callables = fail_after_side_effect
    sample = torch.zeros(4, 3)
    cpu_rng_state = torch.random.get_rng_state().clone()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            runner._capture(sample)
        except RuntimeError:
            pass

    torch.testing.assert_close(model.counter, torch.zeros(()))
    assert torch.equal(torch.random.get_rng_state(), cpu_rng_state)
    assert len(restored_npu_states) == 1
    assert torch.equal(restored_npu_states[0][0], fake_npu_state)


def test_rng_is_restored_even_if_buffer_restoration_fails(monkeypatch):
    model = DiagnosticModel()
    runner = StaticGraphRunner(
        model,
        batch_size=4,
        allow_training=True,
        require_deterministic_training=False,
        assume_graph_safe=True,
    )
    fake_npu_state = torch.tensor([13], dtype=torch.uint8)
    restored_npu_states = []
    fake_npu = type("FakeNPU", (), {})()
    fake_npu.get_rng_state = lambda _device: fake_npu_state.clone()
    fake_npu.set_rng_state = lambda state, device: restored_npu_states.append(
        (state.clone(), device)
    )
    fake_npu.make_graphed_callables = (
        lambda wrapper, sample_args, num_warmup_iters: wrapper
    )
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    monkeypatch.setattr(
        runner,
        "_restore_buffers",
        lambda _snapshot: (_ for _ in ()).throw(RuntimeError("buffer mismatch")),
    )
    sample = torch.zeros(4, 3)
    cpu_rng_state = torch.random.get_rng_state().clone()

    try:
        runner._capture(sample)
    except RuntimeError as error:
        assert "restore model buffers" in str(error)
        assert "buffer mismatch" in str(error.__cause__)
    else:
        raise AssertionError("expected buffer restoration to fail")

    assert torch.equal(torch.random.get_rng_state(), cpu_rng_state)
    assert len(restored_npu_states) == 1
    assert torch.equal(restored_npu_states[0][0], fake_npu_state)


def test_buffer_restoration_failure_never_falls_back_eager(monkeypatch):
    model = DiagnosticModel()
    runner = StaticGraphRunner(
        model,
        batch_size=4,
        strict=False,
        allow_training=True,
        require_deterministic_training=False,
        assume_graph_safe=True,
    )
    monkeypatch.setattr(type(runner), "device_type", property(lambda self: "npu"))
    monkeypatch.setattr(
        runner,
        "_capture",
        lambda _sample: (_ for _ in ()).throw(_CaptureStateError("unsafe cleanup")),
    )

    try:
        runner(torch.randn(4, 3))
    except RuntimeError as error:
        assert "unsafe cleanup" in str(error)
    else:
        raise AssertionError("unsafe capture cleanup must remain fatal")

    for unsafe_call in (
        lambda: runner(torch.randn(2, 3)),
        lambda: runner(torch.randn(4, 3), scale=2.0),
        lambda: runner(torch.randn(4, 3)),
    ):
        try:
            unsafe_call()
        except _CaptureStateError as error:
            assert "unsafe cleanup" in str(error)
        else:
            raise AssertionError("unsafe runner must never execute eagerly later")
    assert model.calls == []


def test_cpu_rng_restoration_failure_never_falls_back_eager(monkeypatch):
    model = DiagnosticModel()
    runner = StaticGraphRunner(
        model,
        batch_size=4,
        strict=False,
        allow_training=True,
        require_deterministic_training=False,
        assume_graph_safe=True,
    )
    restored_npu_states = []
    fake_npu = type("FakeNPU", (), {})()
    fake_npu.get_rng_state = lambda _device: torch.tensor([17], dtype=torch.uint8)
    fake_npu.set_rng_state = lambda state, device: restored_npu_states.append(
        (state.clone(), device)
    )
    fake_npu.make_graphed_callables = (
        lambda wrapper, sample_args, num_warmup_iters: wrapper
    )
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    monkeypatch.setattr(type(runner), "device_type", property(lambda self: "npu"))
    monkeypatch.setattr(
        torch.random,
        "set_rng_state",
        lambda _state: (_ for _ in ()).throw(RuntimeError("CPU RNG restore failed")),
    )

    try:
        runner(torch.randn(4, 3))
    except _CaptureStateError as error:
        assert "CPU RNG" in str(error)
        assert "CPU RNG restore failed" in str(error.__cause__)
    else:
        raise AssertionError("CPU RNG restoration failure must remain fatal")

    assert len(restored_npu_states) == 1
    assert model.calls == []


def test_npu_rng_restoration_failure_never_falls_back_eager(monkeypatch):
    model = DiagnosticModel()
    runner = StaticGraphRunner(
        model,
        batch_size=4,
        strict=False,
        allow_training=True,
        require_deterministic_training=False,
        assume_graph_safe=True,
    )
    fake_npu = type("FakeNPU", (), {})()
    fake_npu.get_rng_state = lambda _device: torch.tensor([19], dtype=torch.uint8)
    fake_npu.set_rng_state = lambda _state, _device: (_ for _ in ()).throw(
        RuntimeError("NPU RNG restore failed")
    )
    fake_npu.make_graphed_callables = (
        lambda wrapper, sample_args, num_warmup_iters: wrapper
    )
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    monkeypatch.setattr(type(runner), "device_type", property(lambda self: "npu"))
    sample = torch.randn(4, 3)
    cpu_rng_state = torch.random.get_rng_state().clone()

    try:
        runner(sample)
    except _CaptureStateError as error:
        assert "NPU RNG" in str(error)
        assert "NPU RNG restore failed" in str(error.__cause__)
    else:
        raise AssertionError("NPU RNG restoration failure must remain fatal")

    assert torch.equal(torch.random.get_rng_state(), cpu_rng_state)
    assert model.calls == []


def test_multiple_cleanup_failures_are_aggregated(monkeypatch):
    model = DiagnosticModel()
    runner = StaticGraphRunner(
        model,
        batch_size=4,
        allow_training=True,
        require_deterministic_training=False,
        assume_graph_safe=True,
    )
    fake_npu = type("FakeNPU", (), {})()
    fake_npu.get_rng_state = lambda _device: torch.tensor([23], dtype=torch.uint8)
    fake_npu.set_rng_state = lambda _state, _device: (_ for _ in ()).throw(
        RuntimeError("NPU RNG restore failed")
    )
    fake_npu.make_graphed_callables = (
        lambda wrapper, sample_args, num_warmup_iters: wrapper
    )
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    monkeypatch.setattr(
        runner,
        "_restore_buffers",
        lambda _snapshot: (_ for _ in ()).throw(RuntimeError("buffer restore failed")),
    )

    try:
        runner._capture(torch.randn(4, 3))
    except _CaptureStateError as error:
        assert "model buffers, NPU RNG" in str(error)
        assert "buffer restore failed" in str(error.__cause__)
    else:
        raise AssertionError("all cleanup failures must be reported as fatal")


def test_capture_failure_falls_back_once(monkeypatch):
    model = DiagnosticModel()
    runner = StaticGraphRunner(
        model,
        batch_size=4,
        allow_training=True,
        require_deterministic_training=False,
        assume_graph_safe=True,
    )
    monkeypatch.setattr(type(runner), "device_type", property(lambda self: "npu"))

    def fail(_sample):
        raise RuntimeError("unsupported op")

    monkeypatch.setattr(runner, "_capture", fail)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        runner(torch.randn(4, 3))
        runner(torch.randn(4, 3))
    assert len(seen) == 1
    assert "unsupported op" in runner.capture_error
    assert "prior capture failed" in runner.last_route.reason

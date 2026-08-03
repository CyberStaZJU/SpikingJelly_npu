import os

import pytest
import torch

from spikingjelly_npu.activation_based import functional, neuron, surrogate
from spikingjelly_npu.fedsnn import DecayLIF
from spikingjelly_npu.npu import StaticGraphRunner, configure_npu, is_npu_available

pytestmark = [pytest.mark.npu, pytest.mark.aspy]


def require_native_aspy():
    if os.environ.get("SPIKINGJELLY_NPU_ASPY_EXPECT_NATIVE") != "1":
        pytest.skip("optional AsPy native extension was not requested")
    if not is_npu_available():
        pytest.skip("Ascend NPU unavailable")
    index = int(os.environ.get("ASCEND_DEVICE_ID", "0"))
    return configure_npu(f"npu:{index}")


def npu_format(tensor: torch.Tensor) -> int:
    torch_npu = pytest.importorskip("torch_npu")
    get_format = getattr(torch_npu, "get_npu_format", None)
    if not callable(get_format):
        get_format = torch.ops.npu.get_npu_format
    return int(get_format(tensor))


def make_fractal_nz(tensor: torch.Tensor) -> torch.Tensor:
    """Construct a physical format-29 tensor without changing test policy."""

    torch_npu = pytest.importorskip("torch_npu")
    torch.npu.config.allow_internal_format = True
    try:
        internal = torch_npu.npu_format_cast(tensor, int(torch_npu.Format.FRACTAL_NZ))
        torch.npu.synchronize(tensor.device)
    finally:
        torch.npu.config.allow_internal_format = False
    assert int(torch.ops.npu.get_npu_format(internal)) == int(torch_npu.Format.FRACTAL_NZ)
    return internal


@pytest.mark.parametrize("shape", [(4, 2, 5), (3, 1, 4097)])
def test_aspy_fedsnn_decay_lif_exact_forward_backward(shape):
    device = require_native_aspy()
    torch.manual_seed(20260901)
    current_reference = torch.rand(shape, dtype=torch.float32, device=device, requires_grad=True)
    current_accelerated = current_reference.detach().clone().requires_grad_(True)
    kwargs = {
        "membrane_decay": 0.75,
        "v_threshold": 0.7,
        "surrogate_function": surrogate.ATan(alpha=2.5),
    }
    reference = DecayLIF(backend="torch", **kwargs).to(device)
    accelerated = DecayLIF(backend="aspy", backend_strict=True, **kwargs).to(device)

    expected = reference(current_reference)
    actual = accelerated(current_accelerated)
    weight = torch.linspace(
        0.25, 1.25, expected.numel(), dtype=expected.dtype, device=device
    ).reshape_as(expected)
    (expected * weight).sum().backward()
    (actual * weight).sum().backward()
    torch.npu.synchronize(device)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        current_accelerated.grad,
        current_reference.grad,
        rtol=5e-5,
        atol=3e-5,
    )
    assert accelerated.last_backend_route.backend == "aspy"
    assert accelerated.last_backend_route.accelerated
    assert accelerated.state_dict() == {}


@pytest.mark.parametrize("shape", [(4, 2, 5), (3, 1, 4097)])
def test_aspy_fedsnn_decay_lif_bf16_public_fp32_island(shape):
    device = require_native_aspy()
    torch.manual_seed(20260921)
    current_bf16 = torch.rand(shape, dtype=torch.bfloat16, device=device)
    current_reference = current_bf16.float().detach().requires_grad_(True)
    current_accelerated = current_bf16.detach().requires_grad_(True)
    kwargs = {
        "membrane_decay": 0.75,
        "v_threshold": 0.7,
        "surrogate_function": surrogate.ATan(alpha=2.5),
    }
    reference = DecayLIF(backend="torch", **kwargs).to(device)
    accelerated = DecayLIF(backend="aspy", backend_strict=True, **kwargs).to(device)

    expected_fp32 = reference(current_reference)
    actual_bf16 = accelerated(current_accelerated)
    expected_bf16 = expected_fp32.to(torch.bfloat16)
    weight = torch.linspace(
        0.25, 1.25, expected_fp32.numel(), dtype=torch.float32, device=device
    ).reshape_as(expected_fp32)
    (expected_fp32 * weight).sum().backward()
    (actual_bf16.float() * weight).sum().backward()
    torch.npu.synchronize(device)

    torch.testing.assert_close(actual_bf16, expected_bf16, rtol=0, atol=0)
    torch.testing.assert_close(
        current_accelerated.grad.float(),
        current_reference.grad,
        rtol=5e-3,
        atol=5e-3,
    )
    route = accelerated.last_backend_route
    assert route.backend == "aspy"
    assert route.dtype_conversion == "bf16-public-fp32-aspy-island"
    assert route.dtype_conversion_bytes == 2 * current_bf16.numel() * (2 + 4)


def test_aspy_fedsnn_decay_lif_real_ncdhw_copy_preserves_forward_and_gradient(
    monkeypatch,
):
    device = require_native_aspy()
    torch_npu = pytest.importorskip("torch_npu")
    import spikingjelly_npu_aspy as adapter

    torch.manual_seed(20260905)
    time_steps = 4
    batch_size = 2
    channels = 8
    encoded = torch.rand(
        time_steps,
        batch_size,
        3,
        8,
        8,
        dtype=torch.float32,
        device=device,
    )
    convolution = torch.nn.Conv2d(3, channels, 3, padding=1, bias=False).to(device)
    bntt = torch.nn.ModuleList(
        torch.nn.BatchNorm2d(channels).to(device).eval() for _ in range(time_steps)
    )
    packed = convolution(encoded.flatten(0, 1)).reshape(
        time_steps,
        batch_size,
        channels,
        8,
        8,
    )
    ncdhw = torch.stack([bntt[step](packed[step]) for step in range(time_steps)])
    torch.npu.synchronize(device)
    ncdhw_format = int(getattr(torch_npu.Format, "NCDHW", 30))
    assert npu_format(ncdhw) == ncdhw_format, (
        "the qualified packed Conv/BNTT producer must materialize physical "
        f"ACL_FORMAT_NCDHW ({ncdhw_format})"
    )
    ncdhw.retain_grad()
    reference_current = ncdhw.detach().clone().requires_grad_(True)
    native_formats = []
    original_require_nd = adapter._require_nd

    def record_require_nd(tensor, name):
        if name == "current_seq":
            native_formats.append(npu_format(tensor))
        return original_require_nd(tensor, name)

    monkeypatch.setattr(adapter, "_require_nd", record_require_nd)
    kwargs = {
        "membrane_decay": 0.75,
        "v_threshold": 0.7,
        "surrogate_function": surrogate.ATan(alpha=2.5),
    }
    expected = DecayLIF(backend="torch", **kwargs).to(device)(reference_current)
    accelerated = DecayLIF(backend="aspy", backend_strict=True, **kwargs).to(device)
    actual = accelerated(ncdhw)
    weight = torch.linspace(
        0.25, 1.25, actual.numel(), dtype=actual.dtype, device=device
    ).reshape_as(actual)
    (expected * weight).sum().backward()
    (actual * weight).sum().backward()
    torch.npu.synchronize(device)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        ncdhw.grad,
        reference_current.grad,
        rtol=5e-5,
        atol=3e-5,
    )
    assert convolution.weight.grad is not None
    assert native_formats and all(value == 2 for value in native_formats)
    assert accelerated.last_backend_route.backend == "aspy"
    assert accelerated.last_backend_route.format_conversion == "ncdhw-to-nd-copy"


def test_aspy_fedsnn_decay_lif_bf16_real_ncdhw_copy_preserves_forward_and_gradient(
    monkeypatch,
):
    device = require_native_aspy()
    torch_npu = pytest.importorskip("torch_npu")
    import spikingjelly_npu_aspy as adapter

    torch.manual_seed(20260926)
    time_steps = 4
    batch_size = 2
    channels = 8
    encoded = torch.rand(
        time_steps,
        batch_size,
        3,
        8,
        8,
        dtype=torch.float32,
        device=device,
    )
    convolution = torch.nn.Conv2d(3, channels, 3, padding=1, bias=False).to(device)
    bntt = torch.nn.ModuleList(
        torch.nn.BatchNorm2d(channels).to(device).eval() for _ in range(time_steps)
    )
    with torch.autocast(device_type="npu", dtype=torch.bfloat16, cache_enabled=False):
        packed = convolution(encoded.flatten(0, 1)).reshape(
            time_steps,
            batch_size,
            channels,
            8,
            8,
        )
        ncdhw_bf16 = torch.stack([bntt[step](packed[step]) for step in range(time_steps)])
    torch.npu.synchronize(device)
    ncdhw_format = int(getattr(torch_npu.Format, "NCDHW", 30))
    assert ncdhw_bf16.dtype == torch.bfloat16
    assert npu_format(ncdhw_bf16) == ncdhw_format
    ncdhw_bf16.retain_grad()
    reference_current = ncdhw_bf16.float().detach().clone().requires_grad_(True)
    native_formats = []
    original_require_nd = adapter._require_nd

    def record_require_nd(tensor, name):
        if name == "current_seq":
            native_formats.append((tensor.dtype, npu_format(tensor)))
        return original_require_nd(tensor, name)

    monkeypatch.setattr(adapter, "_require_nd", record_require_nd)
    kwargs = {
        "membrane_decay": 0.75,
        "v_threshold": 0.7,
        "surrogate_function": surrogate.ATan(alpha=2.5),
    }
    expected_fp32 = DecayLIF(backend="torch", **kwargs).to(device)(reference_current)
    accelerated = DecayLIF(backend="aspy", backend_strict=True, **kwargs).to(device)
    actual_bf16 = accelerated(ncdhw_bf16)
    weight = torch.linspace(
        0.25, 1.25, expected_fp32.numel(), dtype=torch.float32, device=device
    ).reshape_as(expected_fp32)
    (expected_fp32 * weight).sum().backward()
    (actual_bf16.float() * weight).sum().backward()
    torch.npu.synchronize(device)

    torch.testing.assert_close(actual_bf16, expected_fp32.to(torch.bfloat16), rtol=0, atol=0)
    torch.testing.assert_close(
        ncdhw_bf16.grad.float(),
        reference_current.grad,
        rtol=5e-3,
        atol=5e-3,
    )
    assert convolution.weight.grad is not None
    assert native_formats and all(
        dtype == torch.float32 and physical_format == 2 for dtype, physical_format in native_formats
    )
    route = accelerated.last_backend_route
    assert route.backend == "aspy"
    assert route.format_conversion == "ncdhw-to-nd-copy"
    assert route.dtype_conversion == "bf16-public-fp32-aspy-island"
    assert route.dtype_conversion_bytes == 2 * ncdhw_bf16.numel() * (2 + 4)


def test_aspy_fedsnn_decay_lif_internal_format_falls_back_before_native(
    monkeypatch,
):
    device = require_native_aspy()
    import spikingjelly_npu.activation_based._aspy as aspy_router

    current = torch.rand(4, 64, dtype=torch.float32, device=device)
    internal = make_fractal_nz(current)
    assert internal.is_contiguous()
    assert internal.storage_offset() == 0
    load_calls = []
    monkeypatch.setattr(
        aspy_router,
        "_load_extension",
        lambda: load_calls.append(True) or (None, "must not load"),
    )
    module = DecayLIF(
        membrane_decay=0.75,
        v_threshold=0.7,
        surrogate_function=surrogate.ATan(alpha=2.5),
        backend="aspy",
    ).to(device)

    actual = module(internal)
    expected = module._torch_forward(internal)
    torch.npu.synchronize(device)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert load_calls == []
    assert module.last_backend_route.backend == "torch"
    assert "ACL_FORMAT_ND (2)" in module.last_backend_route.reason
    assert "ACL_FORMAT_NCDHW (30)" in module.last_backend_route.reason
    assert "format=29" in module.last_backend_route.reason

    strict = DecayLIF(
        membrane_decay=0.75,
        v_threshold=0.7,
        surrogate_function=surrogate.ATan(alpha=2.5),
        backend="aspy",
        backend_strict=True,
    ).to(device)
    with pytest.raises(RuntimeError, match="ACL_FORMAT_ND"):
        strict(internal)
    assert load_calls == []


def test_aspy_fedsnn_decay_lif_bf16_internal_format_rejects_before_native(
    monkeypatch,
):
    device = require_native_aspy()
    import spikingjelly_npu.activation_based._aspy as aspy_router

    current = torch.rand(4, 64, dtype=torch.bfloat16, device=device)
    internal = make_fractal_nz(current)
    assert internal.dtype == torch.bfloat16
    load_calls = []
    monkeypatch.setattr(
        aspy_router,
        "_load_extension",
        lambda: load_calls.append(True) or (None, "must not load"),
    )
    non_strict = DecayLIF(
        membrane_decay=0.75,
        v_threshold=0.7,
        surrogate_function=surrogate.ATan(alpha=2.5),
        backend="aspy",
    ).to(device)

    actual = non_strict(internal)
    expected = non_strict._torch_forward(internal)
    torch.npu.synchronize(device)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == torch.bfloat16
    assert load_calls == []
    assert non_strict.last_backend_route.backend == "torch"
    assert non_strict.last_backend_route.dtype_conversion is None
    assert "format=29" in non_strict.last_backend_route.reason

    strict = DecayLIF(
        membrane_decay=0.75,
        v_threshold=0.7,
        surrogate_function=surrogate.ATan(alpha=2.5),
        backend="aspy",
        backend_strict=True,
    ).to(device)
    with pytest.raises(RuntimeError, match="ACL_FORMAT_ND"):
        strict(internal)
    assert load_calls == []


def test_aspy_native_bridge_directly_rejects_bf16():
    device = require_native_aspy()
    from spikingjelly_npu._native import load_aspy_native

    native = load_aspy_native()
    sequence = torch.rand(4, 64, dtype=torch.bfloat16, device=device)
    voltage = torch.zeros(64, dtype=torch.bfloat16, device=device)

    with pytest.raises(RuntimeError, match="FP32"):
        native.if_forward(sequence, voltage, 1.0, 0.0, True)
    with pytest.raises(RuntimeError, match="FP32"):
        native.fedsnn_decay_lif_forward(sequence, 0.75, 0.7)


def test_aspy_native_bridge_directly_rejects_internal_format():
    device = require_native_aspy()
    from spikingjelly_npu._native import load_aspy_native

    current = torch.rand(4, 64, dtype=torch.float32, device=device)
    internal = make_fractal_nz(current)

    with pytest.raises(RuntimeError, match="ACL_FORMAT_ND"):
        load_aspy_native().fedsnn_decay_lif_forward(internal, 0.75, 0.7)


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (
            lambda native, sequence: native.fedsnn_decay_lif_forward(
                torch.rand(
                    4,
                    7,
                    dtype=sequence.dtype,
                    device=sequence.device,
                ),
                0.75,
                0.7,
            ),
            "multiple of 8",
        ),
        (
            lambda native, sequence: native.fedsnn_decay_lif_forward(sequence[0], 0.75, 0.7),
            r"\[T, N",
        ),
        (
            lambda native, sequence: native.fedsnn_decay_lif_forward(sequence[:0], 0.75, 0.7),
            "time dimension must be non-empty",
        ),
        (
            lambda native, sequence: native.fedsnn_decay_lif_forward(
                torch.empty(
                    4,
                    0,
                    dtype=sequence.dtype,
                    device=sequence.device,
                ),
                0.75,
                0.7,
            ),
            "flattened time-step size must be non-empty",
        ),
        (
            lambda native, sequence: native.fedsnn_decay_lif_forward(sequence, -0.1, 0.7),
            "membrane_decay",
        ),
        (
            lambda native, sequence: native.fedsnn_decay_lif_forward(sequence, 0.75, float("nan")),
            "v_threshold",
        ),
        (
            lambda native, sequence: native.fedsnn_decay_lif_backward(
                sequence, sequence, 0.75, 0.7, 0.0
            ),
            "surrogate_alpha",
        ),
    ],
)
def test_aspy_native_bridge_direct_validation(call, match):
    device = require_native_aspy()
    from spikingjelly_npu._native import load_aspy_native

    native = load_aspy_native()
    sequence = torch.rand(4, 64, dtype=torch.float32, device=device)
    with pytest.raises(RuntimeError, match=match):
        call(native, sequence)


def test_aspy_native_bridge_rejects_internal_format_for_all_generic_exports():
    device = require_native_aspy()
    from spikingjelly_npu._native import load_aspy_native

    native = load_aspy_native()
    sequence = torch.rand(4, 64, dtype=torch.float32, device=device)
    voltage = torch.rand(64, dtype=torch.float32, device=device)
    scalar = torch.tensor([0.4], dtype=torch.float32, device=device)
    internal = make_fractal_nz(sequence)

    forward_calls = [
        lambda: native.if_forward(internal, voltage, 1.0, 0.0, True),
        lambda: native.if_forward_compact(internal, voltage, 1.0, 0.0, True),
        lambda: native.lif_forward(internal, voltage, 1.0, 0.0, True, 2.0, False),
        lambda: native.lif_forward_compact(internal, voltage, 1.0, 0.0, True, 2.0, False),
        lambda: native.plif_forward(internal, voltage, scalar, 1.0, 0.0, True, False),
    ]
    backward_calls = [
        lambda: native.if_backward(
            sequence,
            sequence,
            internal,
            sequence,
            voltage,
            1.0,
            0.0,
            True,
            False,
            2.0,
        ),
        lambda: native.if_backward_compact(
            sequence,
            sequence,
            internal,
            voltage,
            1.0,
            0.0,
            True,
            False,
            2.0,
        ),
        lambda: native.lif_backward(
            sequence,
            sequence,
            internal,
            sequence,
            voltage,
            1.0,
            0.0,
            True,
            False,
            2.0,
            2.0,
            False,
        ),
        lambda: native.lif_backward_compact(
            sequence,
            sequence,
            internal,
            voltage,
            1.0,
            0.0,
            True,
            False,
            2.0,
            2.0,
            False,
        ),
        lambda: native.plif_backward(
            sequence,
            sequence,
            sequence,
            sequence,
            internal,
            sequence,
            voltage,
            scalar,
            1.0,
            0.0,
            True,
            False,
            2.0,
            False,
        ),
    ]
    for call in (*forward_calls, *backward_calls):
        with pytest.raises(RuntimeError, match="ACL_FORMAT_ND"):
            call()


def test_aspy_fedsnn_decay_lif_unsupported_npu_input_falls_back_before_native(
    monkeypatch,
):
    device = require_native_aspy()
    import spikingjelly_npu.activation_based._aspy as aspy_router

    load_calls = []
    monkeypatch.setattr(
        aspy_router,
        "_load_extension",
        lambda: load_calls.append(True) or (None, "must not load"),
    )
    module = DecayLIF(
        membrane_decay=0.75,
        v_threshold=0.7,
        surrogate_function=surrogate.ATan(alpha=2.5),
        backend="aspy",
    ).to(device)
    current = torch.rand(4, 2, 5, dtype=torch.float16, device=device)

    actual = module(current)
    expected = module._torch_forward(current)
    torch.npu.synchronize(device)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert load_calls == []
    assert module.last_backend_route.backend == "torch"
    assert "torch.float32 or torch.bfloat16" in module.last_backend_route.reason


@pytest.mark.parametrize("shape", [(3, 2, 4), (5, 3, 5), (4, 1, 4097), (4, 2, 4096)])
@pytest.mark.parametrize("v_reset", [0.0, None])
def test_aspy_if_forward_exact_parity(shape, v_reset):
    device = require_native_aspy()
    torch.manual_seed(20260729)
    x_seq = torch.rand(shape, dtype=torch.float32, device=device)
    kwargs = {
        "v_threshold": 1.0,
        "v_reset": v_reset,
        "surrogate_function": surrogate.ATan(alpha=2.0),
        "detach_reset": False,
        "step_mode": "m",
        "store_v_seq": True,
    }
    reference = neuron.IFNode(backend="torch", **kwargs).to(device)
    accelerated = neuron.IFNode(
        backend="aspy",
        backend_strict=True,
        **kwargs,
    ).to(device)

    expected = reference(x_seq)
    actual = accelerated(x_seq)
    torch.npu.synchronize(device)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(accelerated.v, reference.v, rtol=0, atol=0)
    torch.testing.assert_close(accelerated.v_seq, reference.v_seq, rtol=0, atol=0)
    assert accelerated.last_backend_route.backend == "aspy"
    assert accelerated.last_backend_route.accelerated


@pytest.mark.parametrize("shape", [(4, 2, 5), (3, 1, 4097)])
def test_aspy_if_bf16_public_fp32_state_and_gradients(shape):
    device = require_native_aspy()
    torch.manual_seed(20260922)
    x_bf16 = torch.rand(shape, dtype=torch.bfloat16, device=device)
    x_reference = x_bf16.float().detach().requires_grad_(True)
    x_accelerated = x_bf16.detach().requires_grad_(True)
    kwargs = {
        "v_threshold": 0.7,
        "v_reset": None,
        "surrogate_function": surrogate.ATan(alpha=2.0),
        "detach_reset": False,
        "step_mode": "m",
        "store_v_seq": True,
    }
    reference = neuron.IFNode(backend="torch", **kwargs).to(device)
    accelerated = neuron.IFNode(backend="aspy", backend_strict=True, **kwargs).to(device)

    expected_fp32 = reference(x_reference)
    actual_bf16 = accelerated(x_accelerated)
    weight = torch.linspace(
        0.5, 1.5, expected_fp32.numel(), dtype=torch.float32, device=device
    ).reshape_as(expected_fp32)
    (
        (expected_fp32 * weight).sum()
        + reference.v.square().mean()
        + reference.v_seq.square().mean()
    ).backward()
    (
        (actual_bf16.float() * weight).sum()
        + accelerated.v.square().mean()
        + accelerated.v_seq.square().mean()
    ).backward()
    torch.npu.synchronize(device)

    torch.testing.assert_close(actual_bf16, expected_fp32.to(torch.bfloat16), rtol=0, atol=0)
    torch.testing.assert_close(accelerated.v, reference.v, rtol=0, atol=0)
    torch.testing.assert_close(accelerated.v_seq, reference.v_seq, rtol=0, atol=0)
    torch.testing.assert_close(x_accelerated.grad.float(), x_reference.grad, rtol=5e-3, atol=5e-3)
    assert actual_bf16.dtype == torch.bfloat16
    assert accelerated.v.dtype == torch.float32
    assert accelerated.v_seq.dtype == torch.float32
    route = accelerated.last_backend_route
    assert route.backend == "aspy"
    assert route.dtype_conversion == "bf16-public-fp32-aspy-island"
    assert route.dtype_conversion_bytes == 2 * x_bf16.numel() * (2 + 4)


@pytest.mark.parametrize("shape", [(3, 2, 4), (5, 2, 4097)])
@pytest.mark.parametrize("v_reset", [0.25, None])
@pytest.mark.parametrize("detach_reset", [False, True])
@pytest.mark.parametrize("store_v_seq", [False, True])
def test_aspy_if_backward_matches_eager(
    shape,
    v_reset,
    detach_reset,
    store_v_seq,
):
    device = require_native_aspy()
    torch.manual_seed(20260730)
    x_reference = torch.rand(shape, dtype=torch.float32, device=device, requires_grad=True)
    x_accelerated = x_reference.detach().clone().requires_grad_(True)
    kwargs = {
        "v_threshold": 0.7,
        "v_reset": v_reset,
        "surrogate_function": surrogate.ATan(alpha=2.5),
        "detach_reset": detach_reset,
        "step_mode": "m",
        "store_v_seq": store_v_seq,
    }
    reference = neuron.IFNode(backend="torch", **kwargs).to(device)
    accelerated = neuron.IFNode(
        backend="aspy",
        backend_strict=True,
        **kwargs,
    ).to(device)

    expected = reference(x_reference)
    actual = accelerated(x_accelerated)
    spike_weight = torch.linspace(
        0.25,
        1.25,
        expected.numel(),
        dtype=expected.dtype,
        device=device,
    ).reshape_as(expected)
    final_weight = torch.linspace(
        0.5,
        1.5,
        reference.v.numel(),
        dtype=reference.v.dtype,
        device=device,
    ).reshape_as(reference.v)
    reference_loss = (expected * spike_weight).sum() + (reference.v * final_weight).sum()
    accelerated_loss = (actual * spike_weight).sum() + (accelerated.v * final_weight).sum()
    if store_v_seq:
        voltage_weight = torch.linspace(
            0.1,
            0.9,
            reference.v_seq.numel(),
            dtype=reference.v_seq.dtype,
            device=device,
        ).reshape_as(reference.v_seq)
        reference_loss = reference_loss + (reference.v_seq * voltage_weight).sum()
        accelerated_loss = accelerated_loss + (accelerated.v_seq * voltage_weight).sum()

    reference_loss.backward()
    accelerated_loss.backward()
    torch.npu.synchronize(device)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(accelerated.v, reference.v, rtol=0, atol=0)
    if store_v_seq:
        torch.testing.assert_close(accelerated.v_seq, reference.v_seq, rtol=0, atol=0)
    torch.testing.assert_close(x_accelerated.grad, x_reference.grad, rtol=2e-5, atol=2e-6)
    assert accelerated.last_backend_route.backend == "aspy"


@pytest.mark.parametrize("v_reset", [0.0, None])
def test_aspy_stateful_sequence_gradient_chains_through_initial_voltage(v_reset):
    device = require_native_aspy()
    torch.manual_seed(20260731)
    first_reference = torch.rand(3, 2, 8, device=device, requires_grad=True)
    second_reference = torch.rand(2, 2, 8, device=device, requires_grad=True)
    first_accelerated = first_reference.detach().clone().requires_grad_(True)
    second_accelerated = second_reference.detach().clone().requires_grad_(True)
    kwargs = {
        "v_reset": v_reset,
        "surrogate_function": surrogate.ATan(alpha=2.0),
        "detach_reset": False,
        "step_mode": "m",
        "store_v_seq": False,
    }
    reference = neuron.IFNode(backend="torch", **kwargs).to(device)
    accelerated = neuron.IFNode(
        backend="aspy",
        backend_strict=True,
        **kwargs,
    ).to(device)

    reference(first_reference)
    accelerated(first_accelerated)
    reference_output = reference(second_reference)
    accelerated_output = accelerated(second_accelerated)
    reference_loss = reference_output.square().sum() + reference.v.square().sum()
    accelerated_loss = accelerated_output.square().sum() + accelerated.v.square().sum()
    reference_loss.backward()
    accelerated_loss.backward()
    torch.npu.synchronize(device)

    torch.testing.assert_close(
        first_accelerated.grad,
        first_reference.grad,
        rtol=2e-5,
        atol=2e-6,
    )
    torch.testing.assert_close(
        second_accelerated.grad,
        second_reference.grad,
        rtol=2e-5,
        atol=2e-6,
    )


def test_aspy_if_forward_without_voltage_sequence():
    device = require_native_aspy()
    x_seq = torch.full((3, 2, 4), 0.6, dtype=torch.float32, device=device)
    node = neuron.IFNode(
        backend="aspy",
        backend_strict=True,
        step_mode="m",
        store_v_seq=False,
        surrogate_function=surrogate.ATan(),
    ).to(device)

    output = node(x_seq)
    torch.npu.synchronize(device)

    assert output.shape == x_seq.shape
    assert not hasattr(node, "v_seq")
    assert node.last_backend_route.backend == "aspy"


class _BatchFirstAsPy(torch.nn.Module):
    _spikingjelly_npu_graph_safe = True

    def __init__(self):
        super().__init__()
        self.node = neuron.IFNode(
            backend="aspy",
            backend_strict=True,
            step_mode="m",
            surrogate_function=surrogate.ATan(),
        )

    def forward(self, batch_first):
        functional.reset_net(self)
        return self.node(batch_first.transpose(0, 1).contiguous())


def test_aspy_npugraph_capture_matches_native_eager_forward_and_gradient():
    device = require_native_aspy()
    eager_model = _BatchFirstAsPy().to(device).eval()
    graph_model = _BatchFirstAsPy().to(device).eval()
    runner = StaticGraphRunner(graph_model, batch_size=4, strict=True)
    inputs = torch.rand(4, 3, 8, dtype=torch.float32, device=device)
    eager_inputs = inputs.detach().clone().requires_grad_(True)
    graph_inputs = inputs.detach().clone().requires_grad_(True)

    eager_output = eager_model(eager_inputs)
    eager_output.square().sum().backward()
    graph_output = runner(graph_inputs)
    graph_output.square().sum().backward()
    torch.npu.synchronize(device)

    assert graph_output.shape == (3, 4, 8)
    assert runner.last_route.backend == "npugraph"
    assert runner.last_route.captured
    assert runner.capture_error is None
    assert graph_model.node.last_backend_route.backend == "aspy"
    torch.testing.assert_close(graph_output, eager_output, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(graph_inputs.grad, eager_inputs.grad, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("shape", [(4, 2, 5), (3, 1, 4097)])
@pytest.mark.parametrize("v_reset", [0.25, None])
@pytest.mark.parametrize("detach_reset", [False, True])
@pytest.mark.parametrize("decay_input", [False, True])
@pytest.mark.parametrize("store_v_seq", [False, True])
def test_aspy_lif_forward_backward_matches_eager(
    shape,
    v_reset,
    detach_reset,
    decay_input,
    store_v_seq,
):
    device = require_native_aspy()
    torch.manual_seed(20260801)
    x_reference = torch.rand(shape, dtype=torch.float32, device=device, requires_grad=True)
    x_accelerated = x_reference.detach().clone().requires_grad_(True)
    kwargs = {
        "tau": 2.5,
        "decay_input": decay_input,
        "v_threshold": 0.7,
        "v_reset": v_reset,
        "surrogate_function": surrogate.ATan(alpha=2.5),
        "detach_reset": detach_reset,
        "step_mode": "m",
        "store_v_seq": store_v_seq,
    }
    reference = neuron.LIFNode(backend="torch", **kwargs).to(device)
    accelerated = neuron.LIFNode(
        backend="aspy",
        backend_strict=True,
        **kwargs,
    ).to(device)

    expected = reference(x_reference)
    actual = accelerated(x_accelerated)
    spike_weight = torch.linspace(
        0.25, 1.25, expected.numel(), dtype=expected.dtype, device=device
    ).reshape_as(expected)
    final_weight = torch.linspace(
        0.5, 1.5, reference.v.numel(), dtype=reference.v.dtype, device=device
    ).reshape_as(reference.v)
    reference_loss = (expected * spike_weight).sum() + (reference.v * final_weight).sum()
    accelerated_loss = (actual * spike_weight).sum() + (accelerated.v * final_weight).sum()
    if store_v_seq:
        voltage_weight = torch.linspace(
            0.1,
            0.9,
            reference.v_seq.numel(),
            dtype=reference.v_seq.dtype,
            device=device,
        ).reshape_as(reference.v_seq)
        reference_loss = reference_loss + (reference.v_seq * voltage_weight).sum()
        accelerated_loss = accelerated_loss + (accelerated.v_seq * voltage_weight).sum()

    reference_loss.backward()
    accelerated_loss.backward()
    torch.npu.synchronize(device)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(accelerated.v, reference.v, rtol=1e-6, atol=1e-7)
    if store_v_seq:
        torch.testing.assert_close(accelerated.v_seq, reference.v_seq, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(x_accelerated.grad, x_reference.grad, rtol=3e-5, atol=3e-6)
    assert accelerated.last_backend_route.backend == "aspy"
    assert accelerated.last_backend_route.accelerated
    assert "LIF" in accelerated.last_backend_route.reason


def test_aspy_lif_bf16_remainder_public_fp32_state_and_gradients():
    device = require_native_aspy()
    torch.manual_seed(20260923)
    shape = (5, 3, 7)
    x_bf16 = torch.rand(shape, dtype=torch.bfloat16, device=device)
    x_reference = x_bf16.float().detach().requires_grad_(True)
    x_accelerated = x_bf16.detach().requires_grad_(True)
    kwargs = {
        "tau": 2.5,
        "decay_input": True,
        "v_threshold": 0.7,
        "v_reset": 0.25,
        "surrogate_function": surrogate.ATan(alpha=2.0),
        "detach_reset": True,
        "step_mode": "m",
        "store_v_seq": True,
    }
    reference = neuron.LIFNode(backend="torch", **kwargs).to(device)
    accelerated = neuron.LIFNode(backend="aspy", backend_strict=True, **kwargs).to(device)

    expected_fp32 = reference(x_reference)
    actual_bf16 = accelerated(x_accelerated)
    loss_reference = expected_fp32.float().mean() + reference.v.square().mean()
    loss_accelerated = actual_bf16.float().mean() + accelerated.v.square().mean()
    loss_reference.backward()
    loss_accelerated.backward()
    torch.npu.synchronize(device)

    torch.testing.assert_close(actual_bf16, expected_fp32.to(torch.bfloat16), rtol=0, atol=0)
    torch.testing.assert_close(accelerated.v, reference.v, rtol=5e-5, atol=3e-5)
    torch.testing.assert_close(x_accelerated.grad.float(), x_reference.grad, rtol=5e-3, atol=5e-3)
    assert accelerated.v.dtype == torch.float32
    assert accelerated.v_seq.dtype == torch.float32
    assert accelerated.last_backend_route.backend == "aspy"


@pytest.mark.parametrize("decay_input", [False, True])
@pytest.mark.parametrize("v_reset", [0.0, None])
def test_aspy_lif_stateful_sequence_gradient_chains_through_initial_voltage(
    decay_input,
    v_reset,
):
    device = require_native_aspy()
    torch.manual_seed(20260802)
    first_reference = torch.rand(3, 2, 9, device=device, requires_grad=True)
    second_reference = torch.rand(2, 2, 9, device=device, requires_grad=True)
    first_accelerated = first_reference.detach().clone().requires_grad_(True)
    second_accelerated = second_reference.detach().clone().requires_grad_(True)
    kwargs = {
        "tau": 3.0,
        "decay_input": decay_input,
        "v_reset": v_reset,
        "surrogate_function": surrogate.ATan(alpha=2.0),
        "detach_reset": False,
        "step_mode": "m",
        "store_v_seq": False,
    }
    reference = neuron.LIFNode(backend="torch", **kwargs).to(device)
    accelerated = neuron.LIFNode(
        backend="aspy",
        backend_strict=True,
        **kwargs,
    ).to(device)

    reference(first_reference)
    accelerated(first_accelerated)
    reference_output = reference(second_reference)
    accelerated_output = accelerated(second_accelerated)
    reference_loss = reference_output.square().sum() + reference.v.square().sum()
    accelerated_loss = accelerated_output.square().sum() + accelerated.v.square().sum()
    reference_loss.backward()
    accelerated_loss.backward()
    torch.npu.synchronize(device)

    torch.testing.assert_close(first_accelerated.grad, first_reference.grad, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(
        second_accelerated.grad,
        second_reference.grad,
        rtol=3e-5,
        atol=3e-6,
    )
    assert accelerated.last_backend_route.backend == "aspy"


class _BatchFirstAsPyLIF(torch.nn.Module):
    _spikingjelly_npu_graph_safe = True

    def __init__(self):
        super().__init__()
        self.node = neuron.LIFNode(
            tau=2.5,
            decay_input=True,
            v_reset=None,
            backend="aspy",
            backend_strict=True,
            step_mode="m",
            surrogate_function=surrogate.ATan(alpha=2.0),
        )

    def forward(self, batch_first):
        functional.reset_net(self)
        return self.node(batch_first.transpose(0, 1).contiguous())


def test_aspy_lif_npugraph_capture_and_five_variable_replays():
    device = require_native_aspy()
    eager_model = _BatchFirstAsPyLIF().to(device).eval()
    graph_model = _BatchFirstAsPyLIF().to(device).eval()
    runner = StaticGraphRunner(graph_model, batch_size=4, strict=True)

    for replay in range(5):
        torch.manual_seed(20260810 + replay)
        inputs = torch.rand(4, 3, 17, dtype=torch.float32, device=device)
        eager_inputs = inputs.detach().clone().requires_grad_(True)
        graph_inputs = inputs.detach().clone().requires_grad_(True)

        eager_output = eager_model(eager_inputs)
        eager_output.square().sum().backward()
        graph_output = runner(graph_inputs)
        graph_output.square().sum().backward()
        torch.npu.synchronize(device)

        torch.testing.assert_close(graph_output, eager_output, rtol=3e-5, atol=3e-6)
        torch.testing.assert_close(
            graph_inputs.grad,
            eager_inputs.grad,
            rtol=3e-5,
            atol=3e-6,
        )
        assert runner.last_route.backend == "npugraph"
        assert runner.last_route.captured
        assert runner.capture_error is None
        assert graph_model.node.last_backend_route.backend == "aspy"


@pytest.mark.parametrize("v_reset", [0.25, None])
@pytest.mark.parametrize("detach_reset", [False, True])
@pytest.mark.parametrize("decay_input", [False, True])
def test_aspy_plif_eight_combinations_forward_backward_and_w_grad(
    v_reset, detach_reset, decay_input
):
    device = require_native_aspy()
    torch.manual_seed(20260820)
    shape = (5, 3, 7)
    x_reference = torch.rand(shape, dtype=torch.float32, device=device, requires_grad=True)
    x_accelerated = x_reference.detach().clone().requires_grad_(True)
    kwargs = {
        "init_tau": 2.5,
        "decay_input": decay_input,
        "v_threshold": 0.7,
        "v_reset": v_reset,
        "surrogate_function": surrogate.ATan(alpha=2.5),
        "detach_reset": detach_reset,
        "step_mode": "m",
        "store_v_seq": True,
    }
    reference = neuron.ParametricLIFNode(backend="torch", **kwargs).to(device)
    accelerated = neuron.ParametricLIFNode(backend="aspy", backend_strict=True, **kwargs).to(device)
    accelerated.load_state_dict(reference.state_dict())

    expected = reference(x_reference)
    actual = accelerated(x_accelerated)
    spike_weight = torch.linspace(
        0.25, 1.25, expected.numel(), dtype=expected.dtype, device=device
    ).reshape_as(expected)
    voltage_weight = torch.linspace(
        0.1, 0.9, reference.v_seq.numel(), dtype=reference.v_seq.dtype, device=device
    ).reshape_as(reference.v_seq)
    final_weight = torch.linspace(
        0.5, 1.5, reference.v.numel(), dtype=reference.v.dtype, device=device
    ).reshape_as(reference.v)
    reference_loss = (
        (expected * spike_weight).sum()
        + (reference.v_seq * voltage_weight).sum()
        + (reference.v * final_weight).sum()
    )
    accelerated_loss = (
        (actual * spike_weight).sum()
        + (accelerated.v_seq * voltage_weight).sum()
        + (accelerated.v * final_weight).sum()
    )
    reference_loss.backward()
    accelerated_loss.backward()
    torch.npu.synchronize(device)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(accelerated.v_seq, reference.v_seq, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(accelerated.v, reference.v, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(x_accelerated.grad, x_reference.grad, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(accelerated.w.grad, reference.w.grad, rtol=3e-5, atol=3e-6)
    assert accelerated.last_backend_route.backend == "aspy"
    assert accelerated.last_backend_route.reason == "Ascend C fused multi-step PLIF kernel"


def test_aspy_plif_bf16_public_fp32_state_and_master_gradient():
    device = require_native_aspy()
    torch.manual_seed(20260924)
    shape = (4, 1, 9)
    x_bf16 = torch.rand(shape, dtype=torch.bfloat16, device=device)
    x_reference = x_bf16.float().detach().requires_grad_(True)
    x_accelerated = x_bf16.detach().requires_grad_(True)
    kwargs = {
        "init_tau": 2.5,
        "decay_input": True,
        "v_threshold": 0.7,
        "v_reset": None,
        "surrogate_function": surrogate.ATan(alpha=2.0),
        "detach_reset": False,
        "step_mode": "m",
        "store_v_seq": True,
    }
    reference = neuron.ParametricLIFNode(backend="torch", **kwargs).to(device)
    accelerated = neuron.ParametricLIFNode(backend="aspy", backend_strict=True, **kwargs).to(device)
    accelerated.load_state_dict(reference.state_dict())

    expected_fp32 = reference(x_reference)
    actual_bf16 = accelerated(x_accelerated)
    (expected_fp32.float().mean() + reference.v.square().mean()).backward()
    (actual_bf16.float().mean() + accelerated.v.square().mean()).backward()
    torch.npu.synchronize(device)

    torch.testing.assert_close(actual_bf16, expected_fp32.to(torch.bfloat16), rtol=0, atol=0)
    torch.testing.assert_close(accelerated.v, reference.v, rtol=5e-5, atol=3e-5)
    torch.testing.assert_close(x_accelerated.grad.float(), x_reference.grad, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(accelerated.w.grad, reference.w.grad, rtol=5e-3, atol=5e-3)
    assert accelerated.w.dtype == torch.float32
    assert accelerated.w.grad.dtype == torch.float32
    assert accelerated.v.dtype == torch.float32
    assert accelerated.v_seq.dtype == torch.float32
    assert accelerated.last_backend_route.backend == "aspy"


@pytest.mark.parametrize(
    ("node_factory", "parameter_name"),
    [
        (
            lambda: neuron.ParametricLIFNode(
                init_tau=2.5,
                decay_input=True,
                v_threshold=0.7,
                v_reset=None,
                surrogate_function=surrogate.ATan(alpha=2.0),
                detach_reset=False,
                step_mode="m",
                store_v_seq=False,
            ),
            "w",
        ),
        (
            lambda: neuron.KLIFNode(
                scale_reset=True,
                tau=2.5,
                decay_input=False,
                v_threshold=0.7,
                v_reset=None,
                surrogate_function=surrogate.ATan(alpha=2.0),
                detach_reset=False,
                step_mode="m",
                store_v_seq=False,
            ),
            "k",
        ),
    ],
)
def test_aspy_bf16_stateful_two_forward_optimizer_update(node_factory, parameter_name):
    device = require_native_aspy()
    torch.manual_seed(20260927)
    first_bf16 = torch.rand(3, 2, 9, dtype=torch.bfloat16, device=device)
    second_bf16 = torch.rand(2, 2, 9, dtype=torch.bfloat16, device=device)
    first_reference = first_bf16.float().detach().requires_grad_(True)
    second_reference = second_bf16.float().detach().requires_grad_(True)
    first_accelerated = first_bf16.detach().requires_grad_(True)
    second_accelerated = second_bf16.detach().requires_grad_(True)
    reference = node_factory().to(device)
    accelerated = node_factory().to(device)
    accelerated.backend = "aspy"
    accelerated.backend_strict = True
    accelerated.load_state_dict(reference.state_dict())
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05, momentum=0.2)
    accelerated_optimizer = torch.optim.SGD(accelerated.parameters(), lr=0.05, momentum=0.2)

    reference(first_reference)
    accelerated(first_accelerated)
    expected = reference(second_reference)
    actual = accelerated(second_accelerated)
    reference_loss = expected.float().square().mean() + reference.v.square().mean()
    accelerated_loss = actual.float().square().mean() + accelerated.v.square().mean()
    reference_loss.backward()
    accelerated_loss.backward()
    reference_optimizer.step()
    accelerated_optimizer.step()
    torch.npu.synchronize(device)

    torch.testing.assert_close(actual, expected.to(torch.bfloat16), rtol=0, atol=0)
    torch.testing.assert_close(reference.v, accelerated.v, rtol=5e-5, atol=3e-5)
    torch.testing.assert_close(
        first_accelerated.grad.float(), first_reference.grad, rtol=5e-3, atol=5e-3
    )
    torch.testing.assert_close(
        second_accelerated.grad.float(), second_reference.grad, rtol=5e-3, atol=5e-3
    )
    reference_parameter = getattr(reference, parameter_name)
    accelerated_parameter = getattr(accelerated, parameter_name)
    torch.testing.assert_close(
        accelerated_parameter,
        reference_parameter,
        rtol=5e-3,
        atol=5e-3,
    )
    assert accelerated_parameter.dtype == torch.float32
    assert accelerated_parameter.grad is not None
    assert accelerated_parameter.grad.dtype == torch.float32
    momentum = accelerated_optimizer.state[accelerated_parameter]["momentum_buffer"]
    assert momentum.dtype == torch.float32
    assert accelerated.v.dtype == torch.float32
    assert actual.dtype == torch.bfloat16
    route = accelerated.last_backend_route
    assert route.backend == "aspy"
    assert route.dtype_conversion == "bf16-public-fp32-aspy-island"


@pytest.mark.parametrize("decay_input", [False, True])
def test_aspy_plif_cross_tile_state_chain_and_gradients(decay_input):
    device = require_native_aspy()
    torch.manual_seed(20260821)
    first_reference = torch.rand(3, 1, 4097, device=device, requires_grad=True)
    second_reference = torch.rand(2, 1, 4097, device=device, requires_grad=True)
    first_accelerated = first_reference.detach().clone().requires_grad_(True)
    second_accelerated = second_reference.detach().clone().requires_grad_(True)
    kwargs = {
        "init_tau": 3.0,
        "decay_input": decay_input,
        "v_reset": None,
        "surrogate_function": surrogate.ATan(alpha=2.0),
        "detach_reset": False,
        "step_mode": "m",
        "store_v_seq": False,
    }
    reference = neuron.ParametricLIFNode(backend="torch", **kwargs).to(device)
    accelerated = neuron.ParametricLIFNode(backend="aspy", backend_strict=True, **kwargs).to(device)
    accelerated.load_state_dict(reference.state_dict())

    reference(first_reference)
    accelerated(first_accelerated)
    expected = reference(second_reference)
    actual = accelerated(second_accelerated)
    reference_loss = expected.square().sum() + reference.v.square().mean()
    accelerated_loss = actual.square().sum() + accelerated.v.square().mean()
    reference_loss.backward()
    accelerated_loss.backward()
    torch.npu.synchronize(device)

    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(first_accelerated.grad, first_reference.grad, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(second_accelerated.grad, second_reference.grad, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(accelerated.w.grad, reference.w.grad, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("shape", [(5, 3, 7), (3, 1, 4097), (2, 1, 1)])
@pytest.mark.parametrize("scale_reset", [False, True])
@pytest.mark.parametrize("decay_input", [False, True])
@pytest.mark.parametrize("v_reset", [None, 0.0])
@pytest.mark.parametrize("detach_reset", [False, True])
def test_aspy_klif_full_remainder_singleton_forward_state_and_gradients(
    shape, scale_reset, decay_input, v_reset, detach_reset
):
    device = require_native_aspy()
    torch.manual_seed(20260910)
    x_reference = torch.rand(shape, dtype=torch.float32, device=device, requires_grad=True)
    x_accelerated = x_reference.detach().clone().requires_grad_(True)
    kwargs = {
        "scale_reset": scale_reset,
        "tau": 2.5,
        "decay_input": decay_input,
        "v_threshold": 0.7,
        "v_reset": v_reset,
        "surrogate_function": surrogate.ATan(alpha=2.5),
        "detach_reset": detach_reset,
        "step_mode": "m",
        "store_v_seq": True,
    }
    reference = neuron.KLIFNode(backend="torch", **kwargs).to(device)
    accelerated = neuron.KLIFNode(backend="aspy", backend_strict=True, **kwargs).to(device)
    initial_reference = (
        torch.linspace(0.05, 0.35, x_reference[0].numel(), dtype=x_reference.dtype, device=device)
        .reshape_as(x_reference[0])
        .requires_grad_(True)
    )
    initial_accelerated = initial_reference.detach().clone().requires_grad_(True)
    reference.v = initial_reference
    accelerated.v = initial_accelerated
    with torch.no_grad():
        reference.k.fill_(1.25)
        accelerated.k.copy_(reference.k)

    expected = reference(x_reference)
    actual = accelerated(x_accelerated)
    spike_weight = torch.linspace(
        0.25, 1.25, expected.numel(), dtype=expected.dtype, device=device
    ).reshape_as(expected)
    voltage_weight = torch.linspace(
        0.1, 0.9, reference.v_seq.numel(), dtype=reference.v_seq.dtype, device=device
    ).reshape_as(reference.v_seq)
    final_weight = torch.linspace(
        0.5, 1.5, reference.v.numel(), dtype=reference.v.dtype, device=device
    ).reshape_as(reference.v)
    reference_loss = (
        (expected * spike_weight).sum()
        + (reference.v_seq * voltage_weight).sum()
        + (reference.v * final_weight).sum()
    )
    accelerated_loss = (
        (actual * spike_weight).sum()
        + (accelerated.v_seq * voltage_weight).sum()
        + (accelerated.v * final_weight).sum()
    )
    reference_loss.backward()
    accelerated_loss.backward()
    torch.npu.synchronize(device)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(accelerated.v_seq, reference.v_seq, rtol=5e-5, atol=3e-5)
    torch.testing.assert_close(accelerated.v, reference.v, rtol=5e-5, atol=3e-5)
    torch.testing.assert_close(x_accelerated.grad, x_reference.grad, rtol=5e-5, atol=3e-5)
    torch.testing.assert_close(
        initial_accelerated.grad, initial_reference.grad, rtol=5e-5, atol=3e-5
    )
    torch.testing.assert_close(accelerated.k.grad, reference.k.grad, rtol=5e-5, atol=3e-5)
    assert accelerated.last_backend_route.backend == "aspy"
    assert accelerated.last_backend_route.reason == "Ascend C fused multi-step KLIF kernel"


def test_aspy_klif_bf16_singleton_public_fp32_state_and_master_gradient():
    device = require_native_aspy()
    torch.manual_seed(20260925)
    shape = (4, 1, 7)
    x_bf16 = torch.rand(shape, dtype=torch.bfloat16, device=device)
    x_reference = x_bf16.float().detach().requires_grad_(True)
    x_accelerated = x_bf16.detach().requires_grad_(True)
    kwargs = {
        "scale_reset": True,
        "tau": 2.5,
        "decay_input": False,
        "v_threshold": 0.7,
        "v_reset": None,
        "surrogate_function": surrogate.ATan(alpha=2.0),
        "detach_reset": False,
        "step_mode": "m",
        "store_v_seq": True,
    }
    reference = neuron.KLIFNode(backend="torch", **kwargs).to(device)
    accelerated = neuron.KLIFNode(backend="aspy", backend_strict=True, **kwargs).to(device)
    accelerated.load_state_dict(reference.state_dict())

    expected_fp32 = reference(x_reference)
    actual_bf16 = accelerated(x_accelerated)
    (expected_fp32.float().mean() + reference.v.square().mean()).backward()
    (actual_bf16.float().mean() + accelerated.v.square().mean()).backward()
    torch.npu.synchronize(device)

    torch.testing.assert_close(actual_bf16, expected_fp32.to(torch.bfloat16), rtol=0, atol=0)
    torch.testing.assert_close(accelerated.v, reference.v, rtol=5e-5, atol=3e-5)
    torch.testing.assert_close(x_accelerated.grad.float(), x_reference.grad, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(accelerated.k.grad, reference.k.grad, rtol=5e-3, atol=5e-3)
    assert accelerated.k.dtype == torch.float32
    assert accelerated.k.grad.dtype == torch.float32
    assert accelerated.v.dtype == torch.float32
    assert accelerated.v_seq.dtype == torch.float32
    assert accelerated.last_backend_route.backend == "aspy"


class _BatchFirstAsPyPLIF(torch.nn.Module):
    _spikingjelly_npu_graph_safe = True

    def __init__(self):
        super().__init__()
        self.node = neuron.ParametricLIFNode(
            init_tau=2.5,
            decay_input=True,
            v_threshold=0.35,
            v_reset=None,
            backend="aspy",
            backend_strict=True,
            step_mode="m",
            surrogate_function=surrogate.ATan(alpha=2.0),
        )

    def forward(self, batch_first):
        functional.reset_net(self)
        return self.node(batch_first.transpose(0, 1).contiguous())


def test_aspy_plif_npugraph_dynamic_w_five_replays_forward_backward():
    device = require_native_aspy()
    torch.use_deterministic_algorithms(True, warn_only=False)
    eager_model = _BatchFirstAsPyPLIF().to(device).train()
    graph_model = _BatchFirstAsPyPLIF().to(device).train()
    graph_model.load_state_dict(eager_model.state_dict())
    runner = StaticGraphRunner(
        graph_model,
        batch_size=4,
        strict=True,
        allow_training=True,
        assume_graph_safe=True,
    )
    previous_output = None

    for replay in range(5):
        with torch.no_grad():
            new_w = torch.tensor(-0.8 + replay * 0.3, dtype=torch.float32, device=device)
            eager_model.node.w.copy_(new_w)
            graph_model.node.w.copy_(new_w)
        torch.manual_seed(20260830 + replay)
        inputs = torch.rand(4, 3, 17, dtype=torch.float32, device=device)
        eager_inputs = inputs.detach().clone().requires_grad_(True)
        graph_inputs = inputs.detach().clone().requires_grad_(True)
        eager_model.node.w.grad = None
        graph_model.node.w.grad = None

        eager_output = eager_model(eager_inputs)
        eager_loss = eager_output.square().sum()
        eager_loss.backward()
        graph_output = runner(graph_inputs)
        graph_loss = graph_output.square().sum()
        graph_loss.backward()
        torch.npu.synchronize(device)

        torch.testing.assert_close(graph_output, eager_output, rtol=3e-5, atol=3e-6)
        torch.testing.assert_close(graph_inputs.grad, eager_inputs.grad, rtol=3e-5, atol=3e-6)
        torch.testing.assert_close(
            graph_model.node.w.grad,
            eager_model.node.w.grad,
            rtol=3e-5,
            atol=3e-6,
        )
        if previous_output is not None:
            assert not torch.equal(graph_output, previous_output)
        previous_output = graph_output.detach().clone()
        assert runner.last_route.backend == "npugraph"
        assert runner.last_route.captured
        assert graph_model.node.last_backend_route.backend == "aspy"

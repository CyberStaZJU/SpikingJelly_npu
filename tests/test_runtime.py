import pytest
import torch

from spikingjelly_npu.npu import runtime
from spikingjelly_npu.npu.amp import (
    BF16_MIXED_PRECISION_PROFILE,
    NPUAutocastState,
    autocast,
    get_npu_autocast_state,
    is_npu_bf16_autocast_active,
    npu_bf16_autocast,
)
from spikingjelly_npu.npu.runtime import (
    BF16CapabilityStatus,
    configure_npu,
    get_npu_info,
    is_npu_available,
)


def test_runtime_probe_is_safe_without_torch_npu():
    info = get_npu_info()
    assert isinstance(info.available, bool)
    assert isinstance(info.device_count, int)
    assert is_npu_available() == info.available


def test_npu_bf16_autocast_sets_explicit_context_local_profile(monkeypatch):
    calls = []

    class RecordingAutocast:
        def __init__(self, **kwargs):
            calls.append(("created", kwargs))

        def __enter__(self):
            calls.append(("entered", get_npu_autocast_state()))

        def __exit__(self, exc_type, exc, traceback):
            calls.append(("exited", get_npu_autocast_state()))

    monkeypatch.setattr(torch, "autocast", lambda **kwargs: RecordingAutocast(**kwargs))

    assert get_npu_autocast_state() == NPUAutocastState()
    assert not is_npu_bf16_autocast_active()
    with npu_bf16_autocast():
        state = get_npu_autocast_state()
        assert state == NPUAutocastState(
            enabled=True,
            dtype=torch.bfloat16,
            cache_enabled=False,
            profile=BF16_MIXED_PRECISION_PROFILE,
        )
        assert is_npu_bf16_autocast_active()
    assert get_npu_autocast_state() == NPUAutocastState()
    assert not is_npu_bf16_autocast_active()
    assert calls == [
        (
            "created",
            {
                "device_type": "npu",
                "enabled": True,
                "cache_enabled": False,
                "dtype": torch.bfloat16,
            },
        ),
        (
            "entered",
            NPUAutocastState(
                enabled=True,
                dtype=torch.bfloat16,
                cache_enabled=False,
                profile=BF16_MIXED_PRECISION_PROFILE,
            ),
        ),
        (
            "exited",
            NPUAutocastState(
                enabled=True,
                dtype=torch.bfloat16,
                cache_enabled=False,
                profile=BF16_MIXED_PRECISION_PROFILE,
            ),
        ),
    ]


def test_npu_bf16_autocast_restores_nested_state_after_exception(monkeypatch):
    class FakeAutocast:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(torch, "autocast", lambda **_kwargs: FakeAutocast())

    with autocast(dtype=torch.float16, cache_enabled=True):
        outer = get_npu_autocast_state()
        assert outer == NPUAutocastState(
            enabled=True,
            dtype=torch.float16,
            cache_enabled=True,
            profile=None,
        )
        assert not is_npu_bf16_autocast_active()
        with pytest.raises(RuntimeError, match="boom"):
            with npu_bf16_autocast():
                assert is_npu_bf16_autocast_active()
                raise RuntimeError("boom")
        assert get_npu_autocast_state() == outer
    assert get_npu_autocast_state() == NPUAutocastState()


def test_npu_bf16_autocast_restores_state_when_torch_context_entry_fails(monkeypatch):
    class FailingAutocast:
        def __enter__(self):
            raise RuntimeError("autocast entry failed")

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(torch, "autocast", lambda **_kwargs: FailingAutocast())

    with pytest.raises(RuntimeError, match="autocast entry failed"):
        with npu_bf16_autocast():
            pytest.fail("body must not execute")
    assert get_npu_autocast_state() == NPUAutocastState()
    assert not is_npu_bf16_autocast_active()


def test_npu_bf16_autocast_nested_disabled_restores_outer_state(monkeypatch):
    class FakeAutocast:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(torch, "autocast", lambda **_kwargs: FakeAutocast())

    with npu_bf16_autocast():
        assert is_npu_bf16_autocast_active()
        with npu_bf16_autocast(enabled=False):
            assert get_npu_autocast_state() == NPUAutocastState()
            assert not is_npu_bf16_autocast_active()
        assert is_npu_bf16_autocast_active()
    assert not is_npu_bf16_autocast_active()


def test_disabled_autocast_disables_torch_and_package_profile(monkeypatch):
    calls = []

    class FakeAutocast:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(torch, "autocast", lambda **kwargs: FakeAutocast(**kwargs))

    with npu_bf16_autocast(enabled=False):
        assert get_npu_autocast_state() == NPUAutocastState()
        assert not is_npu_bf16_autocast_active()

    assert calls == []

    with npu_bf16_autocast():
        with npu_bf16_autocast(enabled=False):
            assert not is_npu_bf16_autocast_active()

    assert calls == [
        {
            "device_type": "npu",
            "enabled": True,
            "cache_enabled": False,
            "dtype": torch.bfloat16,
        },
        {"device_type": "npu", "enabled": False},
    ]


def test_generic_npu_bf16_autocast_does_not_activate_qualified_profile(monkeypatch):
    class FakeAutocast:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(torch, "autocast", lambda **_kwargs: FakeAutocast())

    with autocast(dtype=torch.bfloat16):
        state = get_npu_autocast_state()
        assert state.dtype == torch.bfloat16
        assert state.profile is None
        assert not is_npu_bf16_autocast_active()


def test_configure_npu_selects_graph_friendly_compile_mode(monkeypatch):
    calls = []
    fake_device = type("FakeDevice", (), {"type": "npu"})()

    class WriteOnlyConfig:
        def __setattr__(self, name, value):
            assert name == "allow_internal_format"
            calls.append(("internal_format", value))

    fake_config = WriteOnlyConfig()
    fake_npu = type("FakeNPU", (), {})()
    fake_npu.is_available = lambda: True
    fake_npu.set_compile_mode = lambda *, jit_compile: calls.append(("compile", jit_compile))
    fake_npu.set_device = lambda device: calls.append(("device", device))
    fake_npu.config = fake_config
    monkeypatch.setattr(runtime, "_import_torch_npu", lambda: object())
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    monkeypatch.setattr(torch, "device", lambda _value: fake_device)

    device = configure_npu("npu:3")

    assert device is fake_device
    assert calls == [
        ("compile", False),
        ("internal_format", False),
        ("device", device),
    ]


def test_configure_npu_allows_explicit_compile_mode_override(monkeypatch):
    calls = []
    fake_device = type("FakeDevice", (), {"type": "npu"})()
    fake_npu = type("FakeNPU", (), {})()
    fake_npu.is_available = lambda: True
    fake_npu.set_compile_mode = lambda *, jit_compile: calls.append(jit_compile)
    fake_npu.set_device = lambda _device: None
    monkeypatch.setattr(runtime, "_import_torch_npu", lambda: object())
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    monkeypatch.setattr(torch, "device", lambda _value: fake_device)

    configure_npu("npu:1", allow_internal_format=None, jit_compile=True)

    assert calls == [True]


def test_configure_npu_requires_reported_bf16_support_before_mutation(monkeypatch):
    calls = []
    fake_device = type("FakeDevice", (), {"type": "npu"})()
    fake_npu = type("FakeNPU", (), {})()
    fake_npu.is_available = lambda: True
    fake_npu.set_device = lambda device: calls.append(("device", device))
    fake_npu.is_bf16_supported = lambda: calls.append(("bf16", None)) or True
    monkeypatch.setattr(runtime, "_import_torch_npu", lambda: object())
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    monkeypatch.setattr(torch, "device", lambda _value: fake_device)

    device = configure_npu(
        "npu:2",
        allow_internal_format=None,
        jit_compile=None,
        require_bf16=True,
    )

    assert device is fake_device
    assert calls == [("bf16", None), ("device", fake_device)]


@pytest.mark.parametrize(
    ("capability", "message"),
    [
        (False, "NPU BF16 is unsupported"),
        (RuntimeError("query failed"), "capability query failed"),
        ("yes", "capability query failed"),
    ],
)
def test_configure_npu_rejects_unverified_bf16_support(
    monkeypatch, capability, message
):
    fake_device = type("FakeDevice", (), {"type": "npu"})()
    fake_npu = type("FakeNPU", (), {})()
    mutations = []
    fake_npu.is_available = lambda: True
    fake_npu.set_compile_mode = lambda **_kwargs: mutations.append("compile")
    fake_npu.set_device = lambda _device: mutations.append("device")

    class WriteOnlyConfig:
        def __setattr__(self, name, value):
            mutations.append((name, value))

    fake_npu.config = WriteOnlyConfig()

    def query_bf16_support():
        if isinstance(capability, Exception):
            raise capability
        return capability

    fake_npu.is_bf16_supported = query_bf16_support
    monkeypatch.setattr(runtime, "_import_torch_npu", lambda: object())
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    monkeypatch.setattr(torch, "device", lambda _value: fake_device)

    with pytest.raises(RuntimeError, match=message):
        configure_npu("npu:2", require_bf16=True)
    assert mutations == []


def test_configure_npu_rejects_missing_bf16_capability_query(monkeypatch):
    fake_device = type("FakeDevice", (), {"type": "npu"})()
    fake_npu = type("FakeNPU", (), {})()
    fake_npu.is_available = lambda: True
    fake_npu.set_device = lambda _device: None
    monkeypatch.setattr(runtime, "_import_torch_npu", lambda: object())
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    monkeypatch.setattr(torch, "device", lambda _value: fake_device)

    with pytest.raises(RuntimeError, match="capability query is unavailable"):
        configure_npu(
            "npu:2",
            allow_internal_format=None,
            jit_compile=None,
            require_bf16=True,
        )


def test_get_npu_info_treats_bf16_query_failure_as_unknown(monkeypatch):
    fake_npu = type("FakeNPU", (), {})()
    fake_npu.is_available = lambda: True
    fake_npu.device_count = lambda: 1
    fake_npu.is_bf16_supported = lambda: (_ for _ in ()).throw(
        RuntimeError("query failed")
    )
    fake_torch_npu = type("FakeTorchNPU", (), {"__version__": "test"})()
    monkeypatch.setattr(runtime, "_import_torch_npu", lambda: fake_torch_npu)
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)

    info = get_npu_info()

    assert info.available is True
    assert info.device_count == 1
    assert info.bf16_supported is None
    assert info.bf16_status is BF16CapabilityStatus.QUERY_FAILED
    assert info.bf16_reason == "RuntimeError: query failed"


@pytest.mark.parametrize(
    ("capability", "expected_supported", "expected_status"),
    [
        (True, True, BF16CapabilityStatus.SUPPORTED),
        (False, False, BF16CapabilityStatus.UNSUPPORTED),
        ("yes", None, BF16CapabilityStatus.QUERY_FAILED),
    ],
)
def test_get_npu_info_reports_explicit_bf16_status(
    monkeypatch, capability, expected_supported, expected_status
):
    fake_npu = type("FakeNPU", (), {})()
    fake_npu.is_available = lambda: True
    fake_npu.device_count = lambda: 1
    fake_npu.is_bf16_supported = lambda: capability
    fake_torch_npu = type("FakeTorchNPU", (), {"__version__": "test"})()
    monkeypatch.setattr(runtime, "_import_torch_npu", lambda: fake_torch_npu)
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)

    info = get_npu_info()

    assert info.bf16_supported is expected_supported
    assert info.bf16_status is expected_status


def test_get_npu_info_reports_unavailable_bf16_query(monkeypatch):
    fake_npu = type("FakeNPU", (), {})()
    fake_npu.is_available = lambda: True
    fake_npu.device_count = lambda: 1
    fake_torch_npu = type("FakeTorchNPU", (), {"__version__": "test"})()
    monkeypatch.setattr(runtime, "_import_torch_npu", lambda: fake_torch_npu)
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)

    info = get_npu_info()

    assert info.bf16_supported is None
    assert info.bf16_status is BF16CapabilityStatus.QUERY_UNAVAILABLE


def test_configure_npu_does_not_query_bf16_by_default(monkeypatch):
    fake_device = type("FakeDevice", (), {"type": "npu"})()
    fake_npu = type("FakeNPU", (), {})()
    fake_npu.is_available = lambda: True
    fake_npu.set_device = lambda _device: None
    fake_npu.is_bf16_supported = lambda: pytest.fail("unexpected BF16 query")
    monkeypatch.setattr(runtime, "_import_torch_npu", lambda: object())
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    monkeypatch.setattr(torch, "device", lambda _value: fake_device)

    configure_npu("npu:1", allow_internal_format=None, jit_compile=None)

import pytest
import torch

from spikingjelly_npu.npu import runtime
from spikingjelly_npu.npu.runtime import configure_npu, get_npu_info, is_npu_available


def test_runtime_probe_is_safe_without_torch_npu():
    info = get_npu_info()
    assert isinstance(info.available, bool)
    assert isinstance(info.device_count, int)
    assert is_npu_available() == info.available


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


def test_configure_npu_preserves_conv_hf32_by_default(monkeypatch):
    fake_device = type("FakeDevice", (), {"type": "npu"})()

    class FailOnWriteConv:
        allow_hf32 = True

        def __setattr__(self, name, value):
            raise AssertionError(f"unexpected Conv policy write: {name}={value!r}")

    fake_npu = type("FakeNPU", (), {})()
    fake_npu.is_available = lambda: True
    fake_npu.set_device = lambda _device: None
    fake_npu.conv = FailOnWriteConv()
    monkeypatch.setattr(runtime, "_import_torch_npu", lambda: object())
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    monkeypatch.setattr(torch, "device", lambda _value: fake_device)

    configure_npu("npu:1", allow_internal_format=None, jit_compile=None)

    assert fake_npu.conv.allow_hf32 is True


def test_configure_npu_applies_explicit_conv_hf32_policy_after_device_selection(
    monkeypatch,
):
    calls = []
    fake_device = type("FakeDevice", (), {"type": "npu"})()

    class RecordingConv:
        allow_hf32 = True

        def __setattr__(self, name, value):
            assert name == "allow_hf32"
            calls.append(("conv_hf32", value))
            object.__setattr__(self, name, value)

    fake_npu = type("FakeNPU", (), {})()
    fake_npu.is_available = lambda: True
    fake_npu.set_device = lambda device: calls.append(("device", device))
    fake_npu.conv = RecordingConv()
    monkeypatch.setattr(runtime, "_import_torch_npu", lambda: object())
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    monkeypatch.setattr(torch, "device", lambda _value: fake_device)

    configure_npu(
        "npu:1",
        allow_internal_format=None,
        jit_compile=None,
        allow_conv_hf32=False,
    )

    assert calls == [("device", fake_device), ("conv_hf32", False)]


def test_configure_npu_rejects_missing_explicit_conv_hf32_control(monkeypatch):
    fake_device = type("FakeDevice", (), {"type": "npu"})()
    fake_npu = type("FakeNPU", (), {})()
    fake_npu.is_available = lambda: True
    fake_npu.set_device = lambda _device: None
    monkeypatch.setattr(runtime, "_import_torch_npu", lambda: object())
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    monkeypatch.setattr(torch, "device", lambda _value: fake_device)

    with pytest.raises(RuntimeError, match=r"torch\.npu\.conv\.allow_hf32 is unavailable"):
        configure_npu(
            "npu:1",
            allow_internal_format=None,
            jit_compile=None,
            allow_conv_hf32=False,
        )


def test_configure_npu_rejects_conv_hf32_policy_that_does_not_stick(monkeypatch):
    fake_device = type("FakeDevice", (), {"type": "npu"})()

    class IgnoringConv:
        allow_hf32 = True

        def __setattr__(self, name, value):
            assert name == "allow_hf32"

    fake_npu = type("FakeNPU", (), {})()
    fake_npu.is_available = lambda: True
    fake_npu.set_device = lambda _device: None
    fake_npu.conv = IgnoringConv()
    monkeypatch.setattr(runtime, "_import_torch_npu", lambda: object())
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    monkeypatch.setattr(torch, "device", lambda _value: fake_device)

    with pytest.raises(RuntimeError, match="did not retain the requested Conv HF32 policy"):
        configure_npu(
            "npu:1",
            allow_internal_format=None,
            jit_compile=None,
            allow_conv_hf32=False,
        )

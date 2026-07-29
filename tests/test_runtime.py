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
    fake_npu.set_compile_mode = lambda *, jit_compile: calls.append(
        ("compile", jit_compile)
    )
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

from pathlib import Path
from types import SimpleNamespace

import pytest

from spikingjelly_npu import _native


def test_native_loader_reports_missing_bundle(monkeypatch, tmp_path):
    monkeypatch.setattr(_native, "_bundle_root", lambda: tmp_path)
    monkeypatch.setattr(
        _native.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError(name)),
    )
    with pytest.raises(ImportError, match="AsPy native extension is unavailable"):
        _native.load_aspy_native()


def test_native_loader_uses_relocatable_bundle(monkeypatch, tmp_path):
    python_dir = tmp_path / "python"
    library_dir = tmp_path / "lib"
    python_dir.mkdir()
    library_dir.mkdir()
    extension = python_dir / (
        "_spikingjelly_npu_aspy" + _native.importlib.machinery.EXTENSION_SUFFIXES[0]
    )
    extension.write_bytes(b"extension")
    (library_dir / "libcust_opapi.so").write_bytes(b"library")
    loaded = SimpleNamespace(name="native")
    module_calls = []
    cdll_calls = []

    def import_module(name):
        module_calls.append(name)
        if name == "_spikingjelly_npu_aspy":
            raise ImportError(name)
        return SimpleNamespace()

    class Loader:
        def exec_module(self, module):
            module.loaded = loaded

    monkeypatch.setattr(_native, "_bundle_root", lambda: tmp_path)
    monkeypatch.setattr(_native.importlib, "import_module", import_module)
    monkeypatch.setattr(_native.ctypes, "CDLL", lambda path, mode: cdll_calls.append(Path(path)))
    monkeypatch.setattr(
        _native.importlib.util,
        "spec_from_file_location",
        lambda name, path: SimpleNamespace(loader=Loader()),
    )
    monkeypatch.setattr(
        _native.importlib.util,
        "module_from_spec",
        lambda spec: SimpleNamespace(),
    )

    result = _native.load_aspy_native()
    assert result.loaded is loaded
    assert module_calls == ["_spikingjelly_npu_aspy", "torch_npu"]
    assert cdll_calls == [library_dir / "libcust_opapi.so"]

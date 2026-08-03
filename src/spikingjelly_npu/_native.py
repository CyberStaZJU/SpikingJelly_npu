"""Relocatable loader for the optional AsPy native release bundle."""

from __future__ import annotations

import ctypes
import importlib
import importlib.machinery
import importlib.util
import os
from pathlib import Path
from types import ModuleType

_EXTENSION_NAME = "_spikingjelly_npu_aspy"
_BUNDLE_ENV = "SPIKINGJELLY_NPU_ASPY_BUNDLE"


def _bundle_root() -> Path:
    configured = os.environ.get(_BUNDLE_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    marker = Path(__file__).with_name("_aspy_bundle_path.txt")
    if marker.is_file():
        value = marker.read_text(encoding="utf-8").strip()
        if value:
            return Path(value).expanduser().resolve()
    return Path(__file__).resolve().parent / "_native"


def _extension_candidates(root: Path) -> list[Path]:
    python_dir = root / "python"
    candidates: list[Path] = []
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        candidates.extend(sorted(python_dir.glob(f"{_EXTENSION_NAME}*{suffix}")))
    return list(dict.fromkeys(candidates))


def _direct_extension_present() -> bool:
    try:
        return importlib.util.find_spec(_EXTENSION_NAME) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def load_aspy_native() -> ModuleType:
    """Load the source-build extension or a relocatable installed bundle.

    This function is called only after the AsPy router has qualified an NPU
    request. It may therefore import ``torch_npu`` and load CANN-linked shared
    libraries without changing ordinary package-import safety.
    """
    direct_present = _direct_extension_present()
    try:
        return importlib.import_module(_EXTENSION_NAME)
    except ImportError as error:
        direct_error: ImportError | OSError = error
        direct_was_absent = not direct_present
    except OSError as error:
        direct_error = error
        direct_was_absent = False
    except Exception as error:
        raise OSError(
            "AsPy native extension discovery failed while loading a present or "
            f"misconfigured runtime: {error}"
        ) from error

    root = _bundle_root()
    library = root / "lib" / "libcust_opapi.so"
    candidates = _extension_candidates(root)
    if not library.is_file() or len(candidates) != 1:
        if direct_was_absent:
            raise ImportError(
                "AsPy native extension is unavailable. Expected either a source-build "
                f"module on sys.path or one release bundle under {root}. "
                f"Direct import error: {direct_error}"
            ) from direct_error
        raise OSError(
            "AsPy native extension is present on sys.path but could not be loaded, "
            f"and no complete relocatable bundle exists under {root}. "
            f"Direct load error: {direct_error}"
        ) from direct_error

    # Importing torch-npu first makes the torch/c10/NPU shared libraries
    # available before the standalone extension is loaded.
    importlib.import_module("torch_npu")
    ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)

    extension_path = candidates[0]
    spec = importlib.util.spec_from_file_location(_EXTENSION_NAME, extension_path)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"could not create an extension spec for {extension_path}"
        ) from direct_error
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


__all__ = ["load_aspy_native"]

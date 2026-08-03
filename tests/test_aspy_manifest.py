import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_aspy.py"
MANIFEST = ROOT / "native" / "aspy" / "operator_manifest.json"


def _build_module():
    spec = importlib.util.spec_from_file_location("build_aspy_manifest", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_operator_manifest_matches_all_current_native_sources_and_symbols():
    build_aspy = _build_module()
    manifest = build_aspy.load_manifest(MANIFEST)
    build_aspy.validate_manifest(manifest, root=ROOT)

    assert manifest.capabilities == (
        "fedsnn_decay_lif",
        "if",
        "klif",
        "lif",
        "plif",
    )
    assert manifest.symbols == (
        "fedsnn_decay_lif_backward",
        "fedsnn_decay_lif_forward",
        "if_backward",
        "if_forward",
        "klif_backward",
        "klif_forward",
        "lif_backward",
        "lif_forward",
        "plif_backward",
        "plif_forward",
    )
    assert len(manifest.operators) == 10


def test_manifest_script_output_is_deterministic_and_machine_readable():
    symbols = subprocess.run(
        [sys.executable, str(SCRIPT), "--symbols"],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    ).stdout.splitlines()
    capabilities_output = subprocess.run(
        [sys.executable, str(SCRIPT), "--capabilities"],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    ).stdout
    capabilities = json.loads(capabilities_output)

    assert symbols == sorted(symbols)
    assert capabilities["schema_version"] == 1
    assert capabilities["aspy_abi_version"] == 1
    assert capabilities["symbols"] == symbols
    assert sorted(capabilities["capabilities"]) == [
        "fedsnn_decay_lif",
        "if",
        "klif",
        "lif",
        "plif",
    ]

import copy
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

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


def _copy_manifest_native_tree(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    shutil.copytree(ROOT / "native", root / "native")
    return root


def test_manifest_drives_msopgen_project_and_bridge_staging(tmp_path):
    build_aspy = _build_module()
    manifest = build_aspy.load_manifest(MANIFEST)
    project = tmp_path / "project"
    bridge = tmp_path / "bridge"

    commands = build_aspy.msopgen_commands(
        manifest,
        msopgen="/tool/msopgen",
        project=project,
        root=ROOT,
    )
    build_aspy.stage_project(manifest, root=ROOT, project=project)
    build_aspy.stage_bridge(manifest, root=ROOT, destination=bridge)

    assert [command[command.index("-op") + 1] for command in commands] == [
        operator.op for operator in manifest.operators
    ]
    assert [command[command.index("-i") + 1] for command in commands] == [
        str(ROOT / operator.definition) for operator in manifest.operators
    ]
    assert "-m" not in commands[0]
    assert all(command[command.index("-m") + 1] == "1" for command in commands[1:])

    assert {path.name for path in (project / "op_host").iterdir()} == {
        Path(source).name
        for operator in manifest.operators
        for source in operator.host_sources
    }
    assert {path.name for path in (project / "op_kernel").iterdir()} == {
        "CMakeLists.txt",
        *(Path(operator.kernel_source).name for operator in manifest.operators),
    }
    generated_cmake = (project / "op_kernel" / "CMakeLists.txt").read_text()
    for operator in manifest.operators:
        assert operator.op in generated_cmake
        assert Path(operator.kernel_source).name in generated_cmake

    header = (bridge / "aspy_capabilities.generated.h").read_text()
    expected_payload = json.dumps(
        manifest.capability_payload(), sort_keys=True, separators=(",", ":")
    )
    assert f"= {manifest.aspy_abi_version};" in header
    assert expected_payload in header


def test_build_shell_consumes_manifest_helper_without_operator_inventory():
    shell = (ROOT / "scripts" / "build_aspy.sh").read_text()

    assert "--msopgen-plan" in shell
    assert "--stage-project" in shell
    assert "--stage-bridge" in shell
    assert "--capabilities" in shell
    for operator in json.loads(MANIFEST.read_text())["capabilities"]:
        for entry in operator["operators"]:
            assert entry["op"] not in shell
            assert Path(entry["kernel_source"]).name not in shell
            assert f'"{entry["symbol"]}"' not in shell


@pytest.mark.parametrize(
    ("relative", "old", "new", "match"),
    [
        (
            "native/aspy/definition/aspy_if_forward.json",
            '"op": "AsPyIfForward"',
            '"op": "AsPyIfForwardDrift"',
            "definition .* operators differ",
        ),
        (
            "native/aspy/op_host/as_py_if_forward.cpp",
            "OP_ADD(AsPyIfForward);",
            "OP_ADD(AsPyIfForwardDrift);",
            "host OP_ADD differs",
        ),
        (
            "native/aspy/op_host/as_py_if_forward_tiling.h",
            "AsPyIfForward,",
            "AsPyIfForwardDrift,",
            "tiling registration differs",
        ),
        (
            "native/aspy/op_kernel/as_py_if_forward.cpp",
            "void as_py_if_forward(",
            "void as_py_if_forward_drift(",
            "kernel entry differs",
        ),
        (
            "native/aspy/op_kernel/CMakeLists.txt",
            "add_kernel_compile(AsPyIfForward ",
            "add_kernel_compile(AsPyIfForwardDrift ",
            "kernel CMake entries differ",
        ),
        (
            "native/aspy/bridge/aspy_bridge.cpp",
            'module.def("if_forward",',
            'module.def("if_forward_drift",',
            "bridge symbols differ",
        ),
        (
            "native/aspy/bridge/aspy_bridge.cpp",
            "aclnnAsPyIfForward(",
            "aclnnAsPyIfForwardDrift(",
            "bridge ACLNN calls differ",
        ),
    ],
)
def test_manifest_validation_rejects_each_native_contract_drift(
    tmp_path, relative, old, new, match
):
    build_aspy = _build_module()
    root = _copy_manifest_native_tree(tmp_path)
    target = root / relative
    text = target.read_text()
    assert old in text
    target.write_text(text.replace(old, new, 1))
    manifest = build_aspy.load_manifest(root / "native/aspy/operator_manifest.json")

    with pytest.raises(build_aspy.ManifestError, match=match):
        build_aspy.validate_manifest(manifest, root=root)


def test_manifest_mutation_changes_every_generated_build_contract(tmp_path):
    build_aspy = _build_module()
    payload = json.loads(MANIFEST.read_text())
    mutated = copy.deepcopy(payload)
    operator = mutated["capabilities"][1]["operators"][0]
    operator["op"] = "AsPyIfForwardV2"
    operator["symbol"] = "if_forward_v2"
    operator["kernel_source"] = "native/aspy/op_kernel/as_py_if_forward_v2.cpp"
    manifest_path = tmp_path / "operator_manifest.json"
    manifest_path.write_text(json.dumps(mutated))
    manifest = build_aspy.load_manifest(manifest_path)

    assert "AsPyIfForwardV2" in {entry.op for entry in manifest.operators}
    assert "if_forward_v2" in manifest.symbols
    commands = build_aspy.msopgen_commands(
        manifest,
        msopgen="msopgen",
        project=tmp_path / "project",
        root=ROOT,
    )
    assert "AsPyIfForwardV2" in commands[2]
    assert "as_py_if_forward_v2.cpp" in build_aspy._render_kernel_cmake(manifest)
    assert "if_forward_v2" in json.dumps(manifest.capability_payload())

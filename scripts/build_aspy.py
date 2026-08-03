#!/usr/bin/env python3
"""Validate the AsPy operator manifest and materialize deterministic build inputs."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "native" / "aspy" / "operator_manifest.json"
ASPY_ABI_VERSION = 1
ASPY_CAPABILITY_SCHEMA_VERSION = 1

_PYBIND_SYMBOL = re.compile(r'module\.def\(\s*"([a-z0-9_]+)"')
_CMAKE_KERNEL = re.compile(
    r"add_kernel_compile\(\s*([A-Za-z0-9_]+)\s+"
    r"\$\{CMAKE_CURRENT_SOURCE_DIR\}/([A-Za-z0-9_./-]+)\s*\)"
)
_HOST_CLASS = re.compile(r"class\s+([A-Za-z0-9_]+)\s*:\s*public\s+OpDef")
_HOST_OP_ADD = re.compile(r"OP_ADD\(\s*([A-Za-z0-9_]+)\s*\)")
_TILING_REGISTRATION = re.compile(
    r"REGISTER_TILING_DATA_CLASS\(\s*([A-Za-z0-9_]+)\s*,",
    re.DOTALL,
)
_KERNEL_ENTRY = re.compile(
    r'extern\s+"C"\s+__global__\s+__aicore__\s+void\s+([a-z0-9_]+)\s*\('
)
_ACLNN_INCLUDE = re.compile(r'#include\s+"aclnn_([a-z0-9_]+)\.h"')
_ACLNN_CALL = re.compile(
    r"\baclnn(AsPy[A-Za-z0-9]+?)(?:GetWorkspaceSize)?\s*\("
)
_GENERATED_CAPABILITY_INCLUDE = '#include "aspy_capabilities.generated.h"'
_CAPABILITY_ABI_REFERENCE = "kSpikingJellyNpuAsPyAbiVersion"
_CAPABILITY_JSON_REFERENCE = "kSpikingJellyNpuAsPyCapabilitiesJson"


class ManifestError(ValueError):
    """Raised when operator manifest data does not match tracked native sources."""


@dataclass(frozen=True, slots=True)
class Operator:
    capability: str
    definition: str
    direction: str
    op: str
    symbol: str
    host_sources: tuple[str, ...]
    kernel_source: str

    @property
    def host_implementation(self) -> str:
        return next(source for source in self.host_sources if source.endswith(".cpp"))

    @property
    def host_tiling_header(self) -> str:
        return next(source for source in self.host_sources if source.endswith(".h"))

    @property
    def kernel_entry(self) -> str:
        return Path(self.kernel_source).stem


@dataclass(frozen=True, slots=True)
class OperatorManifest:
    schema_version: int
    aspy_abi_version: int
    bridge_source: str
    kernel_cmake: str
    capability_api_symbols: tuple[str, ...]
    operators: tuple[Operator, ...]

    @property
    def capabilities(self) -> tuple[str, ...]:
        return tuple(sorted({operator.capability for operator in self.operators}))

    @property
    def definitions(self) -> tuple[str, ...]:
        return tuple(sorted({operator.definition for operator in self.operators}))

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(sorted(operator.symbol for operator in self.operators))

    @property
    def sources(self) -> tuple[str, ...]:
        paths = {self.bridge_source, self.kernel_cmake, *self.definitions}
        for operator in self.operators:
            paths.update(operator.host_sources)
            paths.add(operator.kernel_source)
        return tuple(sorted(paths))

    def capability_payload(self, *, include_abi: bool = False) -> dict[str, Any]:
        groups = {
            capability: sorted(
                operator.symbol
                for operator in self.operators
                if operator.capability == capability
            )
            for capability in self.capabilities
        }
        payload = {
            "schema_version": self.schema_version,
            "capabilities": groups,
            "symbols": list(self.symbols),
        }
        if include_abi:
            payload["aspy_abi_version"] = self.aspy_abi_version
        return payload


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ManifestError(f"{field} must be an object")
    return value


def _sequence(value: object, field: str) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ManifestError(f"{field} must be an array")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ManifestError(f"{field} must be a non-empty, trimmed string")
    return value


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ManifestError(f"{field} must be a positive integer")
    return value


def _repository_path(value: object, field: str) -> str:
    relative = _string(value, field)
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
        raise ManifestError(f"{field} must be a normalized repository-relative path")
    return relative


def load_manifest(path: Path = DEFAULT_MANIFEST) -> OperatorManifest:
    """Parse a manifest into an immutable normalized representation."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"could not read {path}: {error}") from error
    root = _mapping(payload, "manifest")
    schema_version = _positive_int(root.get("schema_version"), "schema_version")
    aspy_abi_version = _positive_int(
        root.get("aspy_abi_version"), "aspy_abi_version"
    )
    bridge_source = _repository_path(root.get("bridge_source"), "bridge_source")
    kernel_cmake = _repository_path(root.get("kernel_cmake"), "kernel_cmake")
    capability_api_symbols = tuple(
        _string(value, "capability_api_symbols item")
        for value in _sequence(
            root.get("capability_api_symbols"), "capability_api_symbols"
        )
    )
    capabilities = _sequence(root.get("capabilities"), "capabilities")
    operators = []
    capability_names = set()
    for capability_index, raw_capability in enumerate(capabilities):
        capability = _mapping(raw_capability, f"capabilities[{capability_index}]")
        name = _string(capability.get("name"), "capability name")
        if name in capability_names:
            raise ManifestError(f"duplicate capability name: {name}")
        capability_names.add(name)
        definition = _repository_path(
            capability.get("definition"), f"capability {name} definition"
        )
        raw_operators = _sequence(
            capability.get("operators"), f"capability {name} operators"
        )
        directions = set()
        for operator_index, raw_operator in enumerate(raw_operators):
            entry = _mapping(
                raw_operator, f"capability {name} operators[{operator_index}]"
            )
            direction = _string(entry.get("direction"), "operator direction")
            if direction not in {"forward", "backward"}:
                raise ManifestError(
                    f"capability {name} has unsupported direction {direction!r}"
                )
            if direction in directions:
                raise ManifestError(
                    f"capability {name} contains duplicate {direction} operator"
                )
            directions.add(direction)
            host_sources = tuple(
                _repository_path(value, f"operator {name}/{direction} host source")
                for value in _sequence(
                    entry.get("host_sources"),
                    f"operator {name}/{direction} host_sources",
                )
            )
            if len(host_sources) != 2:
                raise ManifestError(
                    f"operator {name}/{direction} must name implementation and tiling header"
                )
            suffixes = sorted(Path(source).suffix for source in host_sources)
            if suffixes != [".cpp", ".h"]:
                raise ManifestError(
                    f"operator {name}/{direction} host_sources must contain one .cpp and one .h"
                )
            operators.append(
                Operator(
                    capability=name,
                    definition=definition,
                    direction=direction,
                    op=_string(entry.get("op"), "operator op"),
                    symbol=_string(entry.get("symbol"), "operator symbol"),
                    host_sources=host_sources,
                    kernel_source=_repository_path(
                        entry.get("kernel_source"), "operator kernel_source"
                    ),
                )
            )
        if directions != {"forward", "backward"}:
            raise ManifestError(
                f"capability {name} must contain exactly forward and backward operators"
            )
    if not operators:
        raise ManifestError("manifest contains no operators")
    duplicate_api_symbols = _duplicates(capability_api_symbols)
    if duplicate_api_symbols:
        raise ManifestError(
            "duplicate capability API symbols: " + ", ".join(duplicate_api_symbols)
        )
    return OperatorManifest(
        schema_version=schema_version,
        aspy_abi_version=aspy_abi_version,
        bridge_source=bridge_source,
        kernel_cmake=kernel_cmake,
        capability_api_symbols=capability_api_symbols,
        operators=tuple(operators),
    )


def _duplicates(values: Iterable[str]) -> tuple[str, ...]:
    seen = set()
    duplicates = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return tuple(sorted(duplicates))


def _read_text(path: Path, description: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise ManifestError(f"could not read {description} {path}: {error}") from error


def _definition_ops(root: Path, definitions: Iterable[str]) -> dict[str, set[str]]:
    result = {}
    for relative in definitions:
        path = root / relative
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ManifestError(
                f"could not read native definition {relative}: {error}"
            ) from error
        entries = _sequence(payload, relative)
        result[relative] = {
            _string(
                _mapping(raw_entry, f"{relative}[{index}]").get("op"),
                f"{relative}[{index}].op",
            )
            for index, raw_entry in enumerate(entries)
        }
    return result


def _bridge_symbols(text: str) -> set[str]:
    return set(_PYBIND_SYMBOL.findall(text))


def _cmake_kernels(path: Path) -> set[tuple[str, str]]:
    return {(op, source) for op, source in _CMAKE_KERNEL.findall(_read_text(path, "CMake"))}


def _host_contract(root: Path, operator: Operator) -> tuple[set[str], set[str], set[str]]:
    implementation = _read_text(root / operator.host_implementation, "host implementation")
    tiling_header = _read_text(root / operator.host_tiling_header, "tiling header")
    return (
        set(_HOST_CLASS.findall(implementation)),
        set(_HOST_OP_ADD.findall(implementation)),
        set(_TILING_REGISTRATION.findall(tiling_header)),
    )


def _kernel_contract(root: Path, operator: Operator) -> set[str]:
    text = _read_text(root / operator.kernel_source, "kernel source")
    return set(_KERNEL_ENTRY.findall(text))


def _bridge_native_contract(text: str) -> tuple[set[str], set[str]]:
    includes = set(_ACLNN_INCLUDE.findall(text))
    calls = set(_ACLNN_CALL.findall(text))
    return includes, calls


def _tracked_native_files(root: Path) -> set[str]:
    native = root / "native" / "aspy"
    return {
        path.relative_to(root).as_posix()
        for directory in (
            native,
            native / "definition",
            native / "op_host",
            native / "op_kernel",
        )
        for path in directory.iterdir()
        if path.is_file()
        and (
            path.suffix in {".json", ".cpp", ".h"}
            or path.name == "CMakeLists.txt"
        )
    }


def validate_manifest(
    manifest: OperatorManifest,
    *,
    root: Path = ROOT,
    require_complete_native_tree: bool = True,
) -> None:
    """Validate definition, host, kernel, CMake, bridge, and tree consistency."""

    errors = []
    if manifest.schema_version != ASPY_CAPABILITY_SCHEMA_VERSION:
        errors.append(f"unsupported manifest schema_version={manifest.schema_version}")
    if manifest.aspy_abi_version != ASPY_ABI_VERSION:
        errors.append(f"unsupported aspy_abi_version={manifest.aspy_abi_version}")
    if manifest.capability_api_symbols != ("aspy_abi_version", "aspy_capabilities"):
        errors.append(
            "capability_api_symbols must be ['aspy_abi_version', 'aspy_capabilities']"
        )

    duplicate_ops = _duplicates(operator.op for operator in manifest.operators)
    duplicate_symbols = _duplicates(operator.symbol for operator in manifest.operators)
    duplicate_sources = _duplicates(
        source
        for operator in manifest.operators
        for source in (*operator.host_sources, operator.kernel_source)
    )
    if duplicate_ops:
        errors.append(f"duplicate operator names: {', '.join(duplicate_ops)}")
    if duplicate_symbols:
        errors.append(f"duplicate bridge symbols: {', '.join(duplicate_symbols)}")
    if duplicate_sources:
        errors.append(
            f"native source reused by multiple operators: {', '.join(duplicate_sources)}"
        )

    for relative in manifest.sources:
        if not (root / relative).is_file():
            errors.append(f"missing manifest source: {relative}")
    if errors:
        raise ManifestError("\n".join(errors))

    definitions = _definition_ops(root, manifest.definitions)
    for relative, defined_ops in definitions.items():
        expected = {
            operator.op
            for operator in manifest.operators
            if operator.definition == relative
        }
        if defined_ops != expected:
            errors.append(
                f"definition {relative} operators differ: "
                f"missing={sorted(expected - defined_ops)}, "
                f"extra={sorted(defined_ops - expected)}"
            )
    all_defined_ops = {
        op for defined_ops in definitions.values() for op in defined_ops
    }
    expected_ops = {operator.op for operator in manifest.operators}
    if all_defined_ops != expected_ops:
        errors.append(
            "combined definition operators differ: "
            f"missing={sorted(expected_ops - all_defined_ops)}, "
            f"extra={sorted(all_defined_ops - expected_ops)}"
        )

    for operator in manifest.operators:
        host_classes, host_registrations, tiling_registrations = _host_contract(
            root, operator
        )
        expected_op = {operator.op}
        if host_classes != expected_op:
            errors.append(
                f"host class differs for {operator.op}: "
                f"expected={sorted(expected_op)}, actual={sorted(host_classes)}"
            )
        if host_registrations != expected_op:
            errors.append(
                f"host OP_ADD differs for {operator.op}: "
                f"expected={sorted(expected_op)}, actual={sorted(host_registrations)}"
            )
        if tiling_registrations != expected_op:
            errors.append(
                f"tiling registration differs for {operator.op}: "
                f"expected={sorted(expected_op)}, actual={sorted(tiling_registrations)}"
            )
        kernel_entries = _kernel_contract(root, operator)
        expected_entry = {operator.kernel_entry}
        if kernel_entries != expected_entry:
            errors.append(
                f"kernel entry differs for {operator.op}: "
                f"expected={sorted(expected_entry)}, actual={sorted(kernel_entries)}"
            )

    cmake_kernels = _cmake_kernels(root / manifest.kernel_cmake)
    expected_cmake = {
        (operator.op, Path(operator.kernel_source).name)
        for operator in manifest.operators
    }
    if cmake_kernels != expected_cmake:
        errors.append(
            "kernel CMake entries differ: "
            f"missing={sorted(expected_cmake - cmake_kernels)}, "
            f"extra={sorted(cmake_kernels - expected_cmake)}"
        )

    bridge_text = _read_text(root / manifest.bridge_source, "bridge source")
    bridge_symbols = _bridge_symbols(bridge_text)
    expected_bridge_symbols = set(manifest.symbols) | set(
        manifest.capability_api_symbols
    )
    if bridge_symbols != expected_bridge_symbols:
        errors.append(
            "bridge symbols differ: "
            f"missing={sorted(expected_bridge_symbols - bridge_symbols)}, "
            f"extra={sorted(bridge_symbols - expected_bridge_symbols)}"
        )
    if _GENERATED_CAPABILITY_INCLUDE not in bridge_text:
        errors.append("bridge must include aspy_capabilities.generated.h")
    if _CAPABILITY_ABI_REFERENCE not in bridge_text:
        errors.append("bridge ABI export must reference generated manifest metadata")
    if _CAPABILITY_JSON_REFERENCE not in bridge_text:
        errors.append("bridge capability export must reference generated manifest metadata")

    bridge_includes, bridge_calls = _bridge_native_contract(bridge_text)
    expected_includes = {operator.kernel_entry for operator in manifest.operators}
    expected_calls = {operator.op for operator in manifest.operators}
    if bridge_includes != expected_includes:
        errors.append(
            "bridge ACLNN includes differ: "
            f"missing={sorted(expected_includes - bridge_includes)}, "
            f"extra={sorted(bridge_includes - expected_includes)}"
        )
    if bridge_calls != expected_calls:
        errors.append(
            "bridge ACLNN calls differ: "
            f"missing={sorted(expected_calls - bridge_calls)}, "
            f"extra={sorted(bridge_calls - expected_calls)}"
        )

    if require_complete_native_tree:
        expected_native = {
            relative
            for relative in manifest.sources
            if relative.startswith(
                (
                    "native/aspy/definition/",
                    "native/aspy/op_host/",
                    "native/aspy/op_kernel/",
                )
            )
        }
        expected_native.add("native/aspy/operator_manifest.json")
        actual_native = _tracked_native_files(root)
        if actual_native != expected_native:
            errors.append(
                "native source coverage differs: "
                f"unlisted={sorted(actual_native - expected_native)}, "
                f"missing={sorted(expected_native - actual_native)}"
            )

    if errors:
        raise ManifestError("\n".join(errors))


def msopgen_commands(
    manifest: OperatorManifest,
    *,
    msopgen: str,
    project: Path,
    soc: str = "ascend910b",
    root: Path = ROOT,
) -> tuple[tuple[str, ...], ...]:
    """Return deterministic msopgen invocations derived only from the manifest."""

    commands = []
    for index, operator in enumerate(manifest.operators):
        command = [
            msopgen,
            "gen",
            "-i",
            str(root / operator.definition),
            "-f",
            "aclnn",
            "-c",
            f"ai_core-{soc}",
            "-out",
            str(project),
        ]
        if index:
            command.extend(("-m", "1"))
        command.extend(("-op", operator.op, "-lan", "cpp"))
        commands.append(tuple(command))
    return tuple(commands)


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    destination.chmod(0o644)


def stage_project(
    manifest: OperatorManifest,
    *,
    root: Path,
    project: Path,
) -> None:
    """Stage reviewed host/kernel sources and generated kernel registration."""

    for operator in manifest.operators:
        for relative in operator.host_sources:
            _copy_file(root / relative, project / "op_host" / Path(relative).name)
        _copy_file(
            root / operator.kernel_source,
            project / "op_kernel" / Path(operator.kernel_source).name,
        )
    cmake_text = _render_kernel_cmake(manifest)
    cmake_path = project / "op_kernel" / "CMakeLists.txt"
    cmake_path.parent.mkdir(parents=True, exist_ok=True)
    cmake_path.write_text(cmake_text, encoding="utf-8")
    cmake_path.chmod(0o644)


def _render_kernel_cmake(manifest: OperatorManifest) -> str:
    lines = [
        "# Generated from native/aspy/operator_manifest.json by scripts/build_aspy.py.",
        'if ("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")',
        "    add_ops_compile_options(ALL OPTIONS -g -O0 --cce-ignore-always-inline=true)",
        "endif()",
        "",
    ]
    lines.extend(
        "add_kernel_compile("
        f"{operator.op} ${{CMAKE_CURRENT_SOURCE_DIR}}/{Path(operator.kernel_source).name})"
        for operator in manifest.operators
    )
    lines.extend(
        (
            "",
            "if (ENABLE_TEST AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/testcases)",
            "    add_subdirectory(testcases)",
            "endif()",
            "",
        )
    )
    return "\n".join(lines)


def stage_bridge(
    manifest: OperatorManifest,
    *,
    root: Path,
    destination: Path,
) -> None:
    """Stage the bridge and generated ABI/capability metadata header."""

    destination.mkdir(parents=True, exist_ok=True)
    _copy_file(root / manifest.bridge_source, destination / Path(manifest.bridge_source).name)
    setup_source = root / "native" / "aspy" / "bridge" / "setup.py"
    _copy_file(setup_source, destination / setup_source.name)
    capability_json = json.dumps(
        manifest.capability_payload(), sort_keys=True, separators=(",", ":")
    )
    header = (
        "#pragma once\n\n"
        "// Generated from native/aspy/operator_manifest.json.\n"
        f"inline constexpr int {_CAPABILITY_ABI_REFERENCE} = "
        f"{manifest.aspy_abi_version};\n"
        f"inline constexpr char {_CAPABILITY_JSON_REFERENCE}[] = R\"ASPY("
        f"{capability_json})ASPY\";\n"
    )
    header_path = destination / "aspy_capabilities.generated.h"
    header_path.write_text(header, encoding="utf-8")
    header_path.chmod(0o644)


def _emit_lines(values: Iterable[str]) -> str:
    return "".join(f"{value}\n" for value in values)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--sources", action="store_true", help="emit source paths")
    output.add_argument("--symbols", action="store_true", help="emit operator symbols")
    output.add_argument("--definitions", action="store_true", help="emit definition paths")
    output.add_argument("--capabilities", action="store_true", help="emit capability JSON")
    output.add_argument("--msopgen-plan", action="store_true", help="emit msopgen plan JSON")
    output.add_argument("--stage-project", type=Path, help="stage host/kernel build inputs")
    output.add_argument("--stage-bridge", type=Path, help="stage bridge build inputs")
    parser.add_argument("--msopgen", default="msopgen")
    parser.add_argument("--project", type=Path)
    parser.add_argument("--soc", default="ascend910b")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        validate_manifest(manifest)
    except ManifestError as error:
        print(f"AsPy operator manifest validation failed: {error}", file=sys.stderr)
        return 1

    if args.sources:
        sys.stdout.write(_emit_lines(manifest.sources))
    elif args.symbols:
        sys.stdout.write(_emit_lines(manifest.symbols))
    elif args.definitions:
        sys.stdout.write(_emit_lines(manifest.definitions))
    elif args.capabilities:
        print(
            json.dumps(
                manifest.capability_payload(include_abi=True),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    elif args.msopgen_plan:
        if args.project is None:
            print("--msopgen-plan requires --project", file=sys.stderr)
            return 2
        print(
            json.dumps(
                msopgen_commands(
                    manifest,
                    msopgen=args.msopgen,
                    project=args.project,
                    soc=args.soc,
                    root=ROOT,
                ),
                separators=(",", ":"),
            )
        )
    elif args.stage_project is not None:
        stage_project(manifest, root=ROOT, project=args.stage_project)
    elif args.stage_bridge is not None:
        stage_bridge(manifest, root=ROOT, destination=args.stage_bridge)
    else:
        print(f"validated {len(manifest.operators)} AsPy operators")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

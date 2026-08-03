#!/usr/bin/env python3
"""Validate the deterministic AsPy operator manifest and emit build inputs."""

from __future__ import annotations

import argparse
import json
import re
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
_CAPABILITY_RAW_STRING = re.compile(
    r'"aspy_capabilities",\s*\[\]\(\) \{\s*return std::string\(\s*R"\((\{.*?\})\)"\);',
    re.DOTALL,
)
_CMAKE_KERNEL = re.compile(
    r"add_kernel_compile\(\s*([A-Za-z0-9_]+)\s+"
    r"\$\{CMAKE_CURRENT_SOURCE_DIR\}/([A-Za-z0-9_./-]+)\s*\)"
)


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


def _definition_ops(root: Path, definitions: Iterable[str]) -> set[str]:
    result = set()
    for relative in definitions:
        path = root / relative
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ManifestError(f"could not read native definition {relative}: {error}") from error
        entries = _sequence(payload, relative)
        for index, raw_entry in enumerate(entries):
            entry = _mapping(raw_entry, f"{relative}[{index}]")
            result.add(_string(entry.get("op"), f"{relative}[{index}].op"))
    return result


def _bridge_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise ManifestError(f"could not read bridge source {path}: {error}") from error


def _bridge_symbols(text: str) -> set[str]:
    return set(_PYBIND_SYMBOL.findall(text))


def _bridge_capability_payload(text: str) -> object:
    match = _CAPABILITY_RAW_STRING.search(text)
    if match is None:
        raise ManifestError("bridge does not contain a literal aspy_capabilities payload")
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError as error:
        raise ManifestError(f"bridge AsPy capability payload is invalid JSON: {error}") from error


def _cmake_kernels(path: Path) -> set[tuple[str, str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ManifestError(f"could not read kernel CMake file {path}: {error}") from error
    return {(op, source) for op, source in _CMAKE_KERNEL.findall(text)}


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
    """Validate paths, definitions, bridge symbols, CMake entries, and coverage."""

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
        errors.append(f"native source reused by multiple operators: {', '.join(duplicate_sources)}")

    for relative in manifest.sources:
        if not (root / relative).is_file():
            errors.append(f"missing manifest source: {relative}")

    if errors:
        raise ManifestError("\n".join(errors))

    defined_ops = _definition_ops(root, manifest.definitions)
    expected_ops = {operator.op for operator in manifest.operators}
    if defined_ops != expected_ops:
        errors.append(
            "definition operators differ: "
            f"missing={sorted(expected_ops - defined_ops)}, "
            f"extra={sorted(defined_ops - expected_ops)}"
        )

    bridge_text = _bridge_text(root / manifest.bridge_source)
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
    try:
        bridge_capabilities = _bridge_capability_payload(bridge_text)
    except ManifestError as error:
        errors.append(str(error))
    else:
        if bridge_capabilities != manifest.capability_payload():
            errors.append("bridge capability payload differs from operator manifest")

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
                f"missing={sorted(actual_native - expected_native)}, "
                f"stale={sorted(expected_native - actual_native)}"
            )

    if errors:
        raise ManifestError("\n".join(errors))


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
    else:
        print(f"validated {len(manifest.operators)} AsPy operators")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

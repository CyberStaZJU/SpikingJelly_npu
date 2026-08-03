"""Serializable provider routes and lazy AsPy capability discovery.

This module contains no PyTorch or accelerator imports. Importing it is safe on
machines without an Ascend runtime; the optional AsPy extension is considered
only when :func:`probe_aspy_capabilities` or :func:`get_aspy_capabilities` is
called explicitly.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, NoReturn

REQUESTED_PROVIDERS = ("torch", "vendor", "aspy", "auto", "npu", "cupy")
ACTUAL_PROVIDERS = ("torch", "vendor", "aspy")
ROUTE_MODES = ("train", "eval")

ASPY_ABI_VERSION = 1
ASPY_CAPABILITY_SCHEMA_VERSION = 1

ASPY_REASON_DECLARED = "aspy.bundle.declared"
ASPY_REASON_LEGACY = "aspy.bundle.legacy"
ASPY_REASON_ABSENT = "aspy.bundle.absent"
ASPY_REASON_LOAD_ERROR = "aspy.bundle.load_error"
ASPY_REASON_MALFORMED = "aspy.bundle.malformed"
ASPY_REASON_UNSUPPORTED_ABI = "aspy.bundle.unsupported_abi"
ASPY_REASON_UNSUPPORTED_SCHEMA = "aspy.bundle.unsupported_schema"

_REASON_CODE = re.compile(r"^[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*$")
_CAPABILITY_SOURCES = frozenset({"declared", "legacy", "absent", "invalid"})
_LEGACY_CAPABILITY_SYMBOLS = {
    "fedsnn_decay_lif": (
        "fedsnn_decay_lif_forward",
        "fedsnn_decay_lif_backward",
    ),
    "if": ("if_forward", "if_backward"),
    "if_compact": ("if_forward_compact", "if_backward_compact"),
    "klif": ("klif_forward", "klif_backward"),
    "lif": ("lif_forward", "lif_backward"),
    "lif_compact": ("lif_forward_compact", "lif_backward_compact"),
    "plif": ("plif_forward", "plif_backward"),
}


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if not value or value.strip() != value:
        raise ValueError(f"{field} must be a non-empty, trimmed string")
    return value


def _validated_optional_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _nonempty_string(value, field)


def _optional_version(value: object, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer or None")
    if value < 0:
        raise ValueError(f"{field} must be non-negative")
    return value


def _optional_nonnegative_int(value: object, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer or None")
    if value < 0:
        raise ValueError(f"{field} must be non-negative")
    return value


def validate_provider(provider: str, *, actual: bool = False) -> str:
    """Validate and return a provider name.

    ``auto`` is a request policy, never an actual execution provider.
    """

    provider = _nonempty_string(provider, "provider")
    allowed = ACTUAL_PROVIDERS if actual else REQUESTED_PROVIDERS
    if provider not in allowed:
        kind = "actual" if actual else "requested"
        choices = ", ".join(repr(item) for item in allowed)
        raise ValueError(f"unsupported {kind} provider {provider!r}; expected one of {choices}")
    return provider


def validate_route_mode(mode: str) -> str:
    """Validate and return the stable ``train``/``eval`` route mode."""

    mode = _nonempty_string(mode, "mode")
    if mode not in ROUTE_MODES:
        raise ValueError(f"unsupported route mode {mode!r}; expected 'train' or 'eval'")
    return mode


@dataclass(frozen=True, slots=True)
class ProviderRoute:
    """An immutable, JSON-serializable provider decision.

    ``actual_provider`` is ``None`` only for a strict rejection that happened
    before any provider executed. Provider-specific metadata remains optional
    so the same contract can describe eager, vendor, and AsPy routes.
    """

    requested_provider: str
    actual_provider: str | None
    logical_operation: str
    reason_code: str
    reason: str
    accelerated: bool
    strict: bool
    mode: str
    native_launch_attempted: bool
    abi_version: int | None = None
    schema_version: int | None = None
    bucket: str | None = None
    native_region: str | None = None
    format_conversion: str | None = None
    dtype_conversion: str | None = None
    dtype_conversion_bytes: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "requested_provider",
            validate_provider(self.requested_provider),
        )
        if self.actual_provider is not None:
            object.__setattr__(
                self,
                "actual_provider",
                validate_provider(self.actual_provider, actual=True),
            )
        object.__setattr__(
            self,
            "logical_operation",
            _nonempty_string(self.logical_operation, "logical_operation"),
        )
        reason_code = _nonempty_string(self.reason_code, "reason_code")
        if _REASON_CODE.fullmatch(reason_code) is None:
            raise ValueError(
                "reason_code must contain only letters, digits, dots, underscores, or hyphens"
            )
        object.__setattr__(self, "reason_code", reason_code)
        object.__setattr__(self, "reason", _nonempty_string(self.reason, "reason"))
        object.__setattr__(self, "mode", validate_route_mode(self.mode))
        if type(self.accelerated) is not bool:
            raise TypeError("accelerated must be a bool")
        if type(self.strict) is not bool:
            raise TypeError("strict must be a bool")
        if type(self.native_launch_attempted) is not bool:
            raise TypeError("native_launch_attempted must be a bool")
        object.__setattr__(self, "abi_version", _optional_version(self.abi_version, "abi_version"))
        object.__setattr__(
            self,
            "schema_version",
            _optional_version(self.schema_version, "schema_version"),
        )
        object.__setattr__(self, "bucket", _validated_optional_string(self.bucket, "bucket"))
        object.__setattr__(
            self,
            "native_region",
            _validated_optional_string(self.native_region, "native_region"),
        )
        object.__setattr__(
            self,
            "format_conversion",
            _validated_optional_string(self.format_conversion, "format_conversion"),
        )
        object.__setattr__(
            self,
            "dtype_conversion",
            _validated_optional_string(self.dtype_conversion, "dtype_conversion"),
        )
        object.__setattr__(
            self,
            "dtype_conversion_bytes",
            _optional_nonnegative_int(
                self.dtype_conversion_bytes,
                "dtype_conversion_bytes",
            ),
        )
        if self.actual_provider is None:
            if not self.strict:
                raise ValueError("actual_provider=None is reserved for strict rejection")
            if self.requested_provider not in {"vendor", "aspy", "auto", "npu", "cupy"}:
                raise ValueError(
                    "a strict provider rejection requires an accelerator-capable request"
                )
            if self.accelerated or self.native_launch_attempted:
                raise ValueError(
                    "a pre-execution strict rejection cannot be accelerated or launch native code"
                )
        elif self.actual_provider == "torch":
            if self.accelerated:
                raise ValueError("actual_provider='torch' cannot be accelerated")
            if self.native_launch_attempted:
                raise ValueError(
                    "a PyTorch route cannot follow a native launch; native failures must propagate"
                )
        else:
            allowed_actual = {
                "vendor": {"vendor", "auto", "npu"},
                "aspy": {"aspy", "auto", "npu", "cupy"},
            }
            if self.requested_provider not in allowed_actual[self.actual_provider]:
                raise ValueError(
                    f"requested_provider={self.requested_provider!r} cannot execute "
                    f"actual_provider={self.actual_provider!r}"
                )
            if not self.accelerated:
                raise ValueError("actual_provider='vendor' or 'aspy' requires accelerated=True")
            if not self.native_launch_attempted:
                raise ValueError("an accelerated route requires native_launch_attempted=True")

    @property
    def requested_backend(self) -> str:
        """Compatibility spelling for consumers that currently say backend."""

        return self.requested_provider

    @property
    def backend(self) -> str | None:
        """Compatibility spelling for the actual provider."""

        return self.actual_provider

    @property
    def training(self) -> bool:
        """Whether the route describes training rather than evaluation."""

        return self.mode == "train"

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible route metadata."""

        return asdict(self)


class StrictProviderError(RuntimeError):
    """Raised when strict routing rejects a request before provider execution."""

    def __init__(self, route: ProviderRoute) -> None:
        if route.actual_provider is not None or not route.strict:
            raise ValueError("StrictProviderError requires a strict pre-execution route")
        self.route = route
        super().__init__(f"{route.reason_code}: {route.reason}")


def torch_route(
    logical_operation: str,
    *,
    requested_provider: str = "torch",
    reason_code: str,
    reason: str,
    strict: bool = False,
    mode: str = "eval",
    native_launch_attempted: bool = False,
    abi_version: int | None = None,
    schema_version: int | None = None,
    bucket: str | None = None,
    native_region: str | None = None,
    format_conversion: str | None = None,
    dtype_conversion: str | None = None,
    dtype_conversion_bytes: int | None = None,
) -> ProviderRoute:
    """Build an executed PyTorch route, including observable fallback reasons."""

    return ProviderRoute(
        requested_provider=requested_provider,
        actual_provider="torch",
        logical_operation=logical_operation,
        reason_code=reason_code,
        reason=reason,
        accelerated=False,
        strict=strict,
        mode=mode,
        native_launch_attempted=native_launch_attempted,
        abi_version=abi_version,
        schema_version=schema_version,
        bucket=bucket,
        native_region=native_region,
        format_conversion=format_conversion,
        dtype_conversion=dtype_conversion,
        dtype_conversion_bytes=dtype_conversion_bytes,
    )


def accelerated_route(
    logical_operation: str,
    *,
    requested_provider: str,
    actual_provider: str,
    reason_code: str,
    reason: str,
    strict: bool = False,
    mode: str = "eval",
    native_launch_attempted: bool = True,
    abi_version: int | None = None,
    schema_version: int | None = None,
    bucket: str | None = None,
    native_region: str | None = None,
    format_conversion: str | None = None,
    dtype_conversion: str | None = None,
    dtype_conversion_bytes: int | None = None,
) -> ProviderRoute:
    """Build an executed vendor or AsPy accelerated route."""

    actual_provider = validate_provider(actual_provider, actual=True)
    if actual_provider == "torch":
        raise ValueError("accelerated_route requires actual_provider='vendor' or 'aspy'")
    return ProviderRoute(
        requested_provider=requested_provider,
        actual_provider=actual_provider,
        logical_operation=logical_operation,
        reason_code=reason_code,
        reason=reason,
        accelerated=True,
        strict=strict,
        mode=mode,
        native_launch_attempted=native_launch_attempted,
        abi_version=abi_version,
        schema_version=schema_version,
        bucket=bucket,
        native_region=native_region,
        format_conversion=format_conversion,
        dtype_conversion=dtype_conversion,
        dtype_conversion_bytes=dtype_conversion_bytes,
    )


def strict_pre_execution_rejection(
    logical_operation: str,
    *,
    requested_provider: str,
    reason_code: str,
    reason: str,
    mode: str = "eval",
    abi_version: int | None = None,
    schema_version: int | None = None,
    bucket: str | None = None,
    native_region: str | None = None,
    format_conversion: str | None = None,
    dtype_conversion: str | None = None,
    dtype_conversion_bytes: int | None = None,
) -> NoReturn:
    """Raise a structured strict rejection before any native launch."""

    raise StrictProviderError(
        ProviderRoute(
            requested_provider=requested_provider,
            actual_provider=None,
            logical_operation=logical_operation,
            reason_code=reason_code,
            reason=reason,
            accelerated=False,
            strict=True,
            mode=mode,
            native_launch_attempted=False,
            abi_version=abi_version,
            schema_version=schema_version,
            bucket=bucket,
            native_region=native_region,
            format_conversion=format_conversion,
            dtype_conversion=dtype_conversion,
            dtype_conversion_bytes=dtype_conversion_bytes,
        )
    )


reject_strict_pre_execution = strict_pre_execution_rejection


@dataclass(frozen=True, slots=True)
class AsPyCapabilityGroup:
    """One immutable AsPy capability and its required callable symbols."""

    name: str
    symbols: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty_string(self.name, "capability name"))
        normalized = tuple(self.symbols)
        if not normalized:
            raise ValueError("capability symbols must not be empty")
        for symbol in normalized:
            _nonempty_string(symbol, "capability symbol")
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"capability {self.name!r} contains duplicate symbols")
        object.__setattr__(self, "symbols", normalized)


@dataclass(frozen=True, slots=True)
class AsPyCapabilities:
    """Immutable result of a versioned or legacy AsPy bundle probe."""

    bundle_present: bool
    available: bool
    abi_version: int | None
    schema_version: int | None
    groups: tuple[AsPyCapabilityGroup, ...]
    symbols: tuple[str, ...]
    source: str
    reason_code: str
    reason: str

    def __post_init__(self) -> None:
        if type(self.bundle_present) is not bool or type(self.available) is not bool:
            raise TypeError("bundle_present and available must be bool values")
        object.__setattr__(self, "abi_version", _optional_version(self.abi_version, "abi_version"))
        object.__setattr__(
            self,
            "schema_version",
            _optional_version(self.schema_version, "schema_version"),
        )
        groups = tuple(self.groups)
        symbols = tuple(self.symbols)
        if any(not isinstance(group, AsPyCapabilityGroup) for group in groups):
            raise TypeError("groups must contain AsPyCapabilityGroup values")
        if len({group.name for group in groups}) != len(groups):
            raise ValueError("capability group names must be unique")
        for symbol in symbols:
            _nonempty_string(symbol, "symbol")
        if len(set(symbols)) != len(symbols):
            raise ValueError("symbols must be unique")
        source = _nonempty_string(self.source, "source")
        if source not in _CAPABILITY_SOURCES:
            raise ValueError(f"unsupported capability source {source!r}")
        object.__setattr__(self, "source", source)
        reason_code = _nonempty_string(self.reason_code, "reason_code")
        if _REASON_CODE.fullmatch(reason_code) is None:
            raise ValueError(f"invalid capability reason code {reason_code!r}")
        object.__setattr__(self, "reason_code", reason_code)
        object.__setattr__(self, "reason", _nonempty_string(self.reason, "reason"))
        if self.available:
            if not self.bundle_present:
                raise ValueError("an available AsPy bundle must be present")
            if not groups:
                raise ValueError("an available AsPy bundle must expose a capability group")
            if source not in {"declared", "legacy"}:
                raise ValueError("an available AsPy bundle must be declared or legacy")
        elif groups:
            raise ValueError("an unavailable AsPy bundle cannot expose capability groups")
        if not self.bundle_present and source != "absent":
            raise ValueError("bundle_present=False requires source='absent'")
        if source == "absent" and (self.bundle_present or self.available):
            raise ValueError("an absent AsPy bundle cannot be present or available")
        if self.reason_code == ASPY_REASON_ABSENT and source != "absent":
            raise ValueError("ASPY_REASON_ABSENT requires source='absent'")
        if self.reason_code == ASPY_REASON_LOAD_ERROR and source != "invalid":
            raise ValueError("ASPY_REASON_LOAD_ERROR requires source='invalid'")
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "symbols", symbols)

    @property
    def capabilities(self) -> tuple[str, ...]:
        """Stable capability names in deterministic order."""

        return tuple(group.name for group in self.groups)

    @property
    def legacy(self) -> bool:
        return self.source == "legacy"

    def supports(self, capability: str) -> bool:
        """Return whether the complete capability group is available."""

        return capability in self.capabilities

    def has_symbol(self, symbol: str) -> bool:
        """Return whether the probe validated a callable operator symbol."""

        return symbol in self.symbols

    def to_dict(self) -> dict[str, Any]:
        """Return immutable probe data as a JSON-compatible dictionary."""

        return {
            "bundle_present": self.bundle_present,
            "available": self.available,
            "abi_version": self.abi_version,
            "schema_version": self.schema_version,
            "capabilities": list(self.capabilities),
            "groups": {group.name: list(group.symbols) for group in self.groups},
            "symbols": list(self.symbols),
            "source": self.source,
            "reason_code": self.reason_code,
            "reason": self.reason,
        }


def _capability_result(
    *,
    bundle_present: bool,
    available: bool,
    abi_version: int | None,
    schema_version: int | None,
    groups: Sequence[AsPyCapabilityGroup] = (),
    symbols: Sequence[str] = (),
    source: str,
    reason_code: str,
    reason: str,
) -> AsPyCapabilities:
    return AsPyCapabilities(
        bundle_present=bundle_present,
        available=available,
        abi_version=abi_version,
        schema_version=schema_version,
        groups=tuple(sorted(groups, key=lambda group: group.name)),
        symbols=tuple(sorted(symbols)),
        source=source,
        reason_code=reason_code,
        reason=reason,
    )


def _safe_attribute(module: object, name: str) -> object | None:
    try:
        return getattr(module, name, None)
    except Exception:
        return None


def _callable_symbol(module: object, name: str) -> bool:
    return callable(_safe_attribute(module, name))


def _legacy_capabilities(module: object) -> AsPyCapabilities:
    groups = []
    symbols = set()
    for name, required_symbols in _LEGACY_CAPABILITY_SYMBOLS.items():
        callable_symbols = tuple(
            symbol for symbol in required_symbols if _callable_symbol(module, symbol)
        )
        symbols.update(callable_symbols)
        if len(callable_symbols) == len(required_symbols):
            groups.append(AsPyCapabilityGroup(name, required_symbols))
    available = bool(groups)
    if available:
        reason = (
            "Loaded an unversioned AsPy bundle and inferred complete capabilities "
            "from callable forward/backward symbols."
        )
    else:
        reason = (
            "Loaded an unversioned AsPy bundle, but no complete known forward/backward "
            "capability pair was present."
        )
    return _capability_result(
        bundle_present=True,
        available=available,
        abi_version=None,
        schema_version=None,
        groups=groups,
        symbols=symbols,
        source="legacy",
        reason_code=ASPY_REASON_LEGACY,
        reason=reason,
    )


def _decode_capability_payload(value: object) -> Mapping[str, object]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, Mapping):
        raise TypeError("aspy_capabilities() must return a mapping or JSON object")
    return value


def _string_sequence(value: object, field: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field} must be a sequence of strings")
    result = tuple(_nonempty_string(item, field) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must not contain duplicates")
    return result


def _declared_groups(
    payload: Mapping[str, object],
) -> tuple[tuple[AsPyCapabilityGroup, ...], tuple[str, ...]]:
    raw_groups = payload.get("capabilities")
    raw_symbols = payload.get("symbols")
    groups = []

    if isinstance(raw_groups, Mapping):
        for raw_name, raw_group_symbols in raw_groups.items():
            name = _nonempty_string(raw_name, "capability name")
            group_symbols = _string_sequence(raw_group_symbols, f"capability {name!r} symbols")
            groups.append(AsPyCapabilityGroup(name, group_symbols))
    else:
        names = _string_sequence(raw_groups, "capabilities")
        declared_symbols = _string_sequence(raw_symbols, "symbols")
        declared_set = set(declared_symbols)
        for name in names:
            required = _LEGACY_CAPABILITY_SYMBOLS.get(name)
            if required is None:
                raise ValueError(f"capability {name!r} must declare its symbol group explicitly")
            if not set(required).issubset(declared_set):
                raise ValueError(f"capability {name!r} is missing a required symbol")
            groups.append(AsPyCapabilityGroup(name, required))

    if not groups:
        raise ValueError("aspy_capabilities() declared no capability groups")
    group_names = [group.name for group in groups]
    if len(set(group_names)) != len(group_names):
        raise ValueError("aspy_capabilities() contains duplicate capability names")

    grouped_symbols = {symbol for group in groups for symbol in group.symbols}
    if raw_symbols is None:
        symbols = tuple(sorted(grouped_symbols))
    else:
        symbols = _string_sequence(raw_symbols, "symbols")
        if set(symbols) != grouped_symbols:
            raise ValueError("declared symbols must exactly match capability group symbols")

    for group in groups:
        expected = _LEGACY_CAPABILITY_SYMBOLS.get(group.name)
        if expected is not None and set(group.symbols) != set(expected):
            raise ValueError(
                f"ABI {ASPY_ABI_VERSION} capability {group.name!r} has an invalid symbol set"
            )
    return tuple(groups), tuple(symbols)


def _malformed_capabilities(
    reason: str,
    *,
    abi_version: int | None = None,
    schema_version: int | None = None,
    reason_code: str = ASPY_REASON_MALFORMED,
) -> AsPyCapabilities:
    return _capability_result(
        bundle_present=True,
        available=False,
        abi_version=abi_version,
        schema_version=schema_version,
        source="invalid",
        reason_code=reason_code,
        reason=reason,
    )


def _probe_loaded_aspy(module: object) -> AsPyCapabilities:
    abi_api = _safe_attribute(module, "aspy_abi_version")
    capabilities_api = _safe_attribute(module, "aspy_capabilities")
    if abi_api is None and capabilities_api is None:
        return _legacy_capabilities(module)
    if not callable(abi_api) or not callable(capabilities_api):
        return _malformed_capabilities(
            "AsPy bundle must expose callable aspy_abi_version() and aspy_capabilities() together."
        )

    try:
        abi_version = abi_api()
    except Exception as error:
        return _malformed_capabilities(f"aspy_abi_version() failed: {error}")
    if isinstance(abi_version, bool) or not isinstance(abi_version, int):
        return _malformed_capabilities("aspy_abi_version() must return an integer")
    if abi_version != ASPY_ABI_VERSION:
        return _malformed_capabilities(
            f"Unsupported AsPy ABI version {abi_version}; expected {ASPY_ABI_VERSION}.",
            abi_version=abi_version,
            reason_code=ASPY_REASON_UNSUPPORTED_ABI,
        )

    try:
        payload = _decode_capability_payload(capabilities_api())
    except (TypeError, ValueError, UnicodeDecodeError) as error:
        return _malformed_capabilities(
            f"Malformed aspy_capabilities() payload: {error}",
            abi_version=abi_version,
        )
    except Exception as error:
        return _malformed_capabilities(
            f"aspy_capabilities() failed: {error}",
            abi_version=abi_version,
        )

    schema_version = payload.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        return _malformed_capabilities(
            "aspy_capabilities() schema_version must be an integer",
            abi_version=abi_version,
        )
    if schema_version != ASPY_CAPABILITY_SCHEMA_VERSION:
        return _malformed_capabilities(
            "Unsupported AsPy capability schema version "
            f"{schema_version}; expected {ASPY_CAPABILITY_SCHEMA_VERSION}.",
            abi_version=abi_version,
            schema_version=schema_version,
            reason_code=ASPY_REASON_UNSUPPORTED_SCHEMA,
        )

    try:
        groups, symbols = _declared_groups(payload)
    except (TypeError, ValueError) as error:
        return _malformed_capabilities(
            f"Malformed AsPy capability groups: {error}",
            abi_version=abi_version,
            schema_version=schema_version,
        )

    missing = tuple(symbol for symbol in symbols if not _callable_symbol(module, symbol))
    if missing:
        return _malformed_capabilities(
            f"AsPy capability metadata names non-callable symbols: {', '.join(missing)}",
            abi_version=abi_version,
            schema_version=schema_version,
        )

    return _capability_result(
        bundle_present=True,
        available=True,
        abi_version=abi_version,
        schema_version=schema_version,
        groups=groups,
        symbols=symbols,
        source="declared",
        reason_code=ASPY_REASON_DECLARED,
        reason="Loaded versioned AsPy capability metadata and validated all declared symbols.",
    )


def _default_aspy_loader() -> object:
    from ._native import load_aspy_native

    return load_aspy_native()


def probe_aspy_capabilities(
    module: object | None = None,
    *,
    loader: Callable[[], object] | None = None,
) -> AsPyCapabilities:
    """Probe one AsPy module without launching a native operator.

    Passing ``module`` keeps tests and embedding code independent of import
    state. With neither argument, native loading remains deferred until this
    function is called.
    """

    if module is not None and loader is not None:
        raise ValueError("pass either module or loader, not both")
    if module is None:
        selected_loader = _default_aspy_loader if loader is None else loader
        try:
            module = selected_loader()
        except ImportError as error:
            return _capability_result(
                bundle_present=False,
                available=False,
                abi_version=None,
                schema_version=None,
                source="absent",
                reason_code=ASPY_REASON_ABSENT,
                reason=f"AsPy bundle is absent: {error}",
            )
        except OSError as error:
            return _capability_result(
                bundle_present=True,
                available=False,
                abi_version=None,
                schema_version=None,
                source="invalid",
                reason_code=ASPY_REASON_LOAD_ERROR,
                reason=f"AsPy bundle is present but unloadable: {error}",
            )
        except Exception as error:
            return _capability_result(
                bundle_present=True,
                available=False,
                abi_version=None,
                schema_version=None,
                source="invalid",
                reason_code=ASPY_REASON_LOAD_ERROR,
                reason=f"AsPy bundle loading failed: {error}",
            )
    return _probe_loaded_aspy(module)


@lru_cache(maxsize=1)
def get_aspy_capabilities() -> AsPyCapabilities:
    """Return the process-cached lazy AsPy capability probe."""

    return probe_aspy_capabilities()


def clear_aspy_capability_cache() -> None:
    """Clear the explicit capability cache, primarily for runtime reconfiguration."""

    get_aspy_capabilities.cache_clear()


__all__ = [
    "ACTUAL_PROVIDERS",
    "ASPY_ABI_VERSION",
    "ASPY_CAPABILITY_SCHEMA_VERSION",
    "ASPY_REASON_ABSENT",
    "ASPY_REASON_DECLARED",
    "ASPY_REASON_LEGACY",
    "ASPY_REASON_LOAD_ERROR",
    "ASPY_REASON_MALFORMED",
    "ASPY_REASON_UNSUPPORTED_ABI",
    "ASPY_REASON_UNSUPPORTED_SCHEMA",
    "AsPyCapabilities",
    "AsPyCapabilityGroup",
    "ProviderRoute",
    "REQUESTED_PROVIDERS",
    "ROUTE_MODES",
    "StrictProviderError",
    "accelerated_route",
    "clear_aspy_capability_cache",
    "get_aspy_capabilities",
    "probe_aspy_capabilities",
    "reject_strict_pre_execution",
    "strict_pre_execution_rejection",
    "torch_route",
    "validate_provider",
    "validate_route_mode",
]

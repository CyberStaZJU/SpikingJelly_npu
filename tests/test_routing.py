import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from spikingjelly_npu.routing import (
    ASPY_REASON_ABSENT,
    ASPY_REASON_DECLARED,
    ASPY_REASON_LEGACY,
    ASPY_REASON_LOAD_ERROR,
    ASPY_REASON_MALFORMED,
    ASPY_REASON_UNSUPPORTED_ABI,
    ASPY_REASON_UNSUPPORTED_SCHEMA,
    ProviderRoute,
    StrictProviderError,
    accelerated_route,
    probe_aspy_capabilities,
    strict_pre_execution_rejection,
    torch_route,
    validate_provider,
)


def _callable():
    return None


def test_provider_validation_and_route_serialization():
    for provider in ("torch", "vendor", "aspy", "auto"):
        assert validate_provider(provider) == provider
    with pytest.raises(ValueError, match="unsupported requested provider"):
        validate_provider("cuda")
    with pytest.raises(ValueError, match="unsupported actual provider"):
        validate_provider("auto", actual=True)

    route = accelerated_route(
        "sequence.attention",
        requested_provider="auto",
        actual_provider="aspy",
        reason_code="aspy.selected",
        reason="qualified fused sequence region",
        strict=True,
        mode="train",
        abi_version=1,
        schema_version=1,
        bucket="t16-b8",
        native_region="qkv-to-output",
        format_conversion="ncdhw-to-nd-copy",
    )
    assert route.requested_backend == "auto"
    assert route.backend == "aspy"
    assert route.training
    assert route.native_launch_attempted
    assert route.accelerated
    assert json.loads(json.dumps(route.to_dict())) == route.to_dict()
    with pytest.raises(AttributeError):
        route.__dict__["reason"] = "changed"

    with pytest.raises((AttributeError, TypeError)):
        route.reason = "changed"
    with pytest.raises(ValueError, match="reason_code"):
        ProviderRoute(
            requested_provider="torch",
            actual_provider="torch",
            logical_operation="sequence.test",
            reason_code="contains spaces",
            reason="invalid",
            accelerated=False,
            strict=False,
            mode="eval",
            native_launch_attempted=False,
        )


def test_torch_route_and_structured_strict_pre_execution_rejection():
    fallback = torch_route(
        "sequence.rnn",
        requested_provider="aspy",
        reason_code="aspy.unsupported_shape",
        reason="shape is outside the qualified bucket",
        strict=False,
        mode="eval",
    )
    assert fallback.actual_provider == "torch"
    assert not fallback.accelerated
    assert not fallback.native_launch_attempted

    with pytest.raises(StrictProviderError) as captured:
        strict_pre_execution_rejection(
            "sequence.rnn",
            requested_provider="aspy",
            reason_code="aspy.unsupported_shape",
            reason="shape is outside the qualified bucket",
            mode="train",
            bucket="t32-b4",
        )
    rejection = captured.value.route
    assert rejection.actual_provider is None
    assert rejection.strict
    assert rejection.training
    assert not rejection.native_launch_attempted
    assert rejection.bucket == "t32-b4"
    assert "aspy.unsupported_shape" in str(captured.value)


def test_routing_import_does_not_load_accelerator_modules():
    code = (
        "import sys; import spikingjelly_npu.routing; "
        "assert 'spikingjelly_npu_aspy' not in sys.modules; "
        "assert '_spikingjelly_npu_aspy' not in sys.modules; "
        "assert 'torch_npu' not in sys.modules; print('ok')"
    )
    env = os.environ.copy()
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.stdout.strip() == "ok"


def test_capability_probe_does_not_launch_native_operator():
    launches = []
    module = SimpleNamespace(
        if_forward=lambda *args: launches.append(args),
        if_backward=lambda *args: launches.append(args),
    )

    capabilities = probe_aspy_capabilities(module)

    assert capabilities.supports("if")
    assert launches == []


def test_old_bundle_capabilities_are_inferred_from_complete_symbol_pairs():
    module = SimpleNamespace(
        if_forward=_callable,
        if_backward=_callable,
        lif_forward=_callable,
        lif_backward=_callable,
        klif_forward=_callable,
    )
    capabilities = probe_aspy_capabilities(module)

    assert capabilities.bundle_present
    assert capabilities.available
    assert capabilities.legacy
    assert capabilities.reason_code == ASPY_REASON_LEGACY
    assert capabilities.capabilities == ("if", "lif")
    assert capabilities.symbols == (
        "if_backward",
        "if_forward",
        "klif_forward",
        "lif_backward",
        "lif_forward",
    )
    assert not capabilities.supports("klif")
    assert json.loads(json.dumps(capabilities.to_dict())) == capabilities.to_dict()
    with pytest.raises(AttributeError):
        capabilities.groups.append("mutated")


def test_versioned_capability_metadata_is_validated():
    symbols = {
        "if_forward": _callable,
        "if_backward": _callable,
        "lif_forward": _callable,
        "lif_backward": _callable,
    }
    module = SimpleNamespace(
        aspy_abi_version=lambda: 1,
        aspy_capabilities=lambda: json.dumps(
            {
                "schema_version": 1,
                "capabilities": {
                    "if": ["if_backward", "if_forward"],
                    "lif": ["lif_backward", "lif_forward"],
                },
                "symbols": [
                    "if_backward",
                    "if_forward",
                    "lif_backward",
                    "lif_forward",
                ],
            }
        ),
        **symbols,
    )
    capabilities = probe_aspy_capabilities(module)

    assert capabilities.available
    assert not capabilities.legacy
    assert capabilities.reason_code == ASPY_REASON_DECLARED
    assert capabilities.abi_version == 1
    assert capabilities.schema_version == 1
    assert capabilities.capabilities == ("if", "lif")


@pytest.mark.parametrize(
    "module",
    [
        SimpleNamespace(
            aspy_abi_version=lambda: "1",
            aspy_capabilities=lambda: {},
        ),
        SimpleNamespace(
            aspy_abi_version=lambda: 1,
            aspy_capabilities=lambda: "not-json",
        ),
        SimpleNamespace(
            aspy_abi_version=lambda: 1,
            aspy_capabilities=lambda: {
                "schema_version": 1,
                "capabilities": {"if": ["if_forward", "if_backward"]},
                "symbols": ["if_forward", "if_backward"],
            },
            if_forward=_callable,
        ),
    ],
)
def test_malformed_capability_data_fails_closed(module):
    capabilities = probe_aspy_capabilities(module)
    assert capabilities.bundle_present
    assert not capabilities.available
    assert capabilities.source == "invalid"
    assert capabilities.reason_code == ASPY_REASON_MALFORMED
    assert capabilities.groups == ()


@pytest.mark.parametrize(
    ("module", "reason_code"),
    [
        (
            SimpleNamespace(
                aspy_abi_version=lambda: 0,
                aspy_capabilities=lambda: {},
            ),
            ASPY_REASON_UNSUPPORTED_ABI,
        ),
        (
            SimpleNamespace(
                aspy_abi_version=lambda: 1,
                aspy_capabilities=lambda: {"schema_version": 2},
            ),
            ASPY_REASON_UNSUPPORTED_SCHEMA,
        ),
    ],
)
def test_old_or_unknown_versioned_bundles_have_stable_reason_codes(
    module, reason_code
):
    capabilities = probe_aspy_capabilities(module)
    assert capabilities.bundle_present
    assert not capabilities.available
    assert capabilities.reason_code == reason_code


def test_absent_bundle_probe_has_stable_reason_code():
    capabilities = probe_aspy_capabilities(
        loader=lambda: (_ for _ in ()).throw(ImportError("missing bundle"))
    )
    assert not capabilities.bundle_present
    assert not capabilities.available
    assert capabilities.reason_code == ASPY_REASON_ABSENT


def test_bundle_load_failure_has_stable_reason_code():
    capabilities = probe_aspy_capabilities(
        loader=lambda: (_ for _ in ()).throw(RuntimeError("broken loader"))
    )
    assert not capabilities.bundle_present
    assert not capabilities.available
    assert capabilities.reason_code == ASPY_REASON_LOAD_ERROR

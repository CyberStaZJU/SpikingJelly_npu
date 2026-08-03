import importlib
import sys

import pytest

from spikingjelly_npu.compat import (
    enable_compat,
    get_compatibility_status,
    install_spikingjelly_alias,
)


def _remove_alias(monkeypatch):
    for name in list(sys.modules):
        if name == "spikingjelly" or name.startswith("spikingjelly."):
            monkeypatch.delitem(sys.modules, name, raising=False)


def test_process_local_spikingjelly_alias(monkeypatch):
    _remove_alias(monkeypatch)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    package = install_spikingjelly_alias(force=True)
    from spikingjelly.activation_based import neuron

    assert package.activation_based.neuron is neuron
    assert package.models.__name__ == "spikingjelly_npu.models"
    assert package.sequence.__name__ == "spikingjelly_npu.sequence"
    assert package.__spikingjelly_npu_alias__
    assert neuron.IFNode.__module__.startswith("spikingjelly_npu")
    _remove_alias(monkeypatch)


def test_canonical_sequence_and_activation_aliases_share_objects(monkeypatch):
    _remove_alias(monkeypatch)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    enable_compat(spikingjelly=True, cuda=False, force_alias=True)

    from spikingjelly.activation_based.layer import SpikingSelfAttention
    from spikingjelly.activation_based.model import Spikformer, spikformer_s, spikformer_ti
    from spikingjelly.activation_based.model.spikformer import (
        Spikformer as CanonicalSpikformer,
    )
    from spikingjelly.activation_based.recurrent import SpikingGRU
    from spikingjelly.activation_based.transformer import (
        SpikingSelfAttention as TransformerAttention,
    )
    from spikingjelly.models.spikformer import Spikformer as ModelsSpikformer
    from spikingjelly.sequence.recurrent import LSTM
    from spikingjelly.sequence.transformer import TransformerEncoderLayer

    from spikingjelly_npu.activation_based.layer import (
        SpikingSelfAttention as NpuSpikingSelfAttention,
    )
    from spikingjelly_npu.activation_based.model import (
        Spikformer as NpuSpikformer,
    )
    from spikingjelly_npu.activation_based.model import (
        spikformer_s as npu_spikformer_s,
    )
    from spikingjelly_npu.activation_based.model import (
        spikformer_ti as npu_spikformer_ti,
    )
    from spikingjelly_npu.activation_based.recurrent import SpikingGRU as NpuSpikingGRU
    from spikingjelly_npu.models import Spikformer as LegacySpikformer
    from spikingjelly_npu.sequence.recurrent import LSTM as NpuLSTM
    from spikingjelly_npu.sequence.transformer import (
        TransformerEncoderLayer as NpuTransformerEncoderLayer,
    )

    assert SpikingSelfAttention is TransformerAttention is NpuSpikingSelfAttention
    assert (
        Spikformer
        is CanonicalSpikformer
        is ModelsSpikformer
        is NpuSpikformer
        is LegacySpikformer
    )
    assert SpikingGRU is NpuSpikingGRU
    assert LSTM is NpuLSTM
    assert TransformerEncoderLayer is NpuTransformerEncoderLayer
    assert spikformer_ti is npu_spikformer_ti
    assert spikformer_s is npu_spikformer_s
    _remove_alias(monkeypatch)


def test_alias_imported_before_canonical_modules_still_reuses_objects(monkeypatch):
    _remove_alias(monkeypatch)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    install_spikingjelly_alias(force=True)

    alias_recurrent = importlib.import_module("spikingjelly.activation_based.recurrent")
    alias_transformer = importlib.import_module("spikingjelly.activation_based.transformer")
    alias_sequence = importlib.import_module("spikingjelly.sequence.recurrent")
    canonical_recurrent = importlib.import_module(
        "spikingjelly_npu.activation_based.recurrent"
    )
    canonical_transformer = importlib.import_module(
        "spikingjelly_npu.activation_based.transformer"
    )
    canonical_sequence = importlib.import_module("spikingjelly_npu.sequence.recurrent")

    assert alias_recurrent is canonical_recurrent
    assert alias_transformer is canonical_transformer
    assert alias_sequence is canonical_sequence
    assert alias_recurrent.SpikingLSTM is canonical_recurrent.SpikingLSTM
    assert alias_sequence.GRU is canonical_sequence.GRU
    _remove_alias(monkeypatch)


def test_alias_is_idempotent_but_refuses_partial_replacement(monkeypatch):
    _remove_alias(monkeypatch)
    package = install_spikingjelly_alias(force=True)
    assert install_spikingjelly_alias(force=True) is package
    _remove_alias(monkeypatch)
    monkeypatch.setitem(sys.modules, "spikingjelly", importlib)
    with pytest.raises(RuntimeError, match="already imported"):
        install_spikingjelly_alias(force=True)
    _remove_alias(monkeypatch)


def test_enable_compat_alias_only_does_not_import_torch_npu(monkeypatch):
    _remove_alias(monkeypatch)
    monkeypatch.delitem(sys.modules, "torch_npu", raising=False)
    status = enable_compat(spikingjelly=True, cuda=False, force_alias=True)
    from spikingjelly.activation_based.layer import SpikingSelfAttention
    from spikingjelly.activation_based.model.spikformer import spikformer_ti

    assert status.enabled and status.spikingjelly_alias
    assert not status.cuda_transfer
    assert SpikingSelfAttention.__module__ == "spikingjelly_npu.activation_based.layer"
    assert spikformer_ti.__module__ == "spikingjelly_npu.activation_based.model.spikformer"
    assert "torch_npu" not in sys.modules
    assert get_compatibility_status().spikingjelly_alias
    _remove_alias(monkeypatch)

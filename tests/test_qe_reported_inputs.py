"""Inputs come from what QE and the deck report, never from a guess (CPU, seconds).

The V_NL spin-orbit mode is QE's ``<spinorbit>`` from the schema that
authenticates the WFN; FR pseudos on an nspinor=2 WFN with no record refuse
by name.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from ffi import _services

_services.ensure_on_path()

from symmetry_maps import read_qe_symmetry_receipt  # noqa: E402

_HS = Path(__file__).resolve().parent / "hsuite"


def test_schema_spinorbit_is_read():
    assert read_qe_symmetry_receipt(_HS / "fixture/data-file-schema.xml").spinorbit is True
    assert read_qe_symmetry_receipt(_HS / "fixture_na/data-file-schema.xml").spinorbit is False


def test_soc_mode_read_or_refused(monkeypatch):
    import psp.vnl_ops as vnl_ops
    monkeypatch.setattr(vnl_ops, "pseudo_has_j_channels", lambda p: True)
    monkeypatch.setattr(vnl_ops, "pseudo_soc_strength_ry", lambda p: 0.1)
    pseudos, quiet = {"Pb": None}, (lambda *a, **k: None)

    def mode(spinorbit, nspinor):
        wfn = SimpleNamespace(spinorbit=spinorbit, qe_symmetry_diagnostic="none bound")
        return vnl_ops.resolve_soc_mode(pseudos, wfn, nspinor=nspinor, print_fn=quiet)

    assert mode(True, 2) is True
    assert mode(False, 2) is False
    assert mode(None, 1) is False
    with pytest.raises(ValueError, match="no QE <spinorbit> record"):
        mode(None, 2)


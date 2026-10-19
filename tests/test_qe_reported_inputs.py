"""Inputs come from what QE and the deck report, never from a guess (CPU, seconds).

The V_NL spin-orbit mode is QE's ``<spinorbit>`` from the schema that
authenticates the WFN; FR pseudos on an nspinor=2 WFN with no record refuse
by name. ``kmeans_cli`` opens the deck's ``wfn_file`` (relative to the deck),
not ``./WFN.h5``. Its runtime start-up is stubbed; the resolver is the
production one.
"""
import importlib
import sys
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


def test_kmeans_opens_deck_wfn_file(monkeypatch, tmp_path):
    import runtime
    monkeypatch.setattr(runtime, "initialize_communicator_stack",
                        lambda **kw: SimpleNamespace(process_index=0))
    monkeypatch.delitem(sys.modules, "centroid.kmeans_cli", raising=False)
    kmeans_cli = importlib.import_module("centroid.kmeans_cli")
    monkeypatch.delitem(sys.modules, "centroid.kmeans_cli")
    args = SimpleNamespace(input=None, prune_window="v_x_vc", fit_window=None)
    with pytest.warns(RuntimeWarning):
        assert kmeans_cli._resolve_deck(args) == (None, "WFN.h5")
    for wfn_file, want in (("nscf/W.h5", tmp_path / "nscf/W.h5"),
                           ("/abs/W.h5", Path("/abs/W.h5"))):
        deck = tmp_path / "kmeans.in"
        deck.write_text(f"[cohsex]\nsys_dim = 3\nncond = 7\nwfn_file = {wfn_file}\n")
        args.input = str(deck)
        assert kmeans_cli._resolve_deck(args) == (7, str(want))

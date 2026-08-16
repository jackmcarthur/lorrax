"""The deck-vs-WFN refusals that fire at the WFN open.

Three conditions the deck cannot check by itself and the run must not
discover late: ``nband`` past the file's band count (which both readers
zero-pad in silence), ``nval`` past the occupied edge, and a collinear
``nspin = 2`` file (whose only guard used to be the eqp writer, at the
END of the run).

``gw.gw_jax`` cannot be IMPORTED on a login node — it builds the
communicator stack and enforces the FFI gate at module scope — so this
suite compiles the one function out of the source text and calls it with
stand-ins, the same way ``tests/test_ff_compute_mode.py`` reads the
driver's entry refusal out of the source rather than running it.
"""
from __future__ import annotations

import ast
import pathlib
import types

import pytest

_GW_JAX = (pathlib.Path(__file__).resolve().parent.parent
           / "src" / "gw" / "gw_jax.py")
_SOURCE = _GW_JAX.read_text()
_FN = "_refuse_deck_wfn_mismatch"


def _load_refusal():
    """Compile ``_refuse_deck_wfn_mismatch`` alone, with no gw_jax import."""
    tree = ast.parse(_SOURCE)
    fn = next((n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == _FN), None)
    assert fn is not None, f"{_FN} is gone from gw_jax.py"
    ns: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]),
                 str(_GW_JAX), "exec"), ns)
    return ns[_FN]


def _config(*, nband=40, nval=4):
    return types.SimpleNamespace(
        nband=nband, nval=nval,
        paths=types.SimpleNamespace(wfn_file="WFN.h5"))


def _wfn(*, nbands=60, nelec=8, nspin=1):
    return types.SimpleNamespace(nbands=nbands, nelec=nelec, nspin=nspin)


def test_a_deck_the_wfn_can_serve_passes():
    """RED TWIN: the guard sits on the driver's fast path."""
    assert _load_refusal()(_config(), _wfn()) is None


def test_nband_past_the_files_band_count_refuses_naming_both():
    with pytest.raises(ValueError) as exc:
        _load_refusal()(_config(nband=100), _wfn(nbands=60))
    msg = str(exc.value)
    assert "nband = 100" in msg and "60 bands" in msg and "WFN.h5" in msg


def test_nband_equal_to_the_files_band_count_is_legal():
    """The boundary is <=: a deck may sum every band in the file."""
    assert _load_refusal()(_config(nband=60), _wfn(nbands=60)) is None


def test_nval_past_the_occupied_edge_refuses():
    with pytest.raises(ValueError) as exc:
        _load_refusal()(_config(nval=12), _wfn(nelec=8))
    msg = str(exc.value)
    assert "nval = 12" in msg and "8 occupied" in msg


def test_nval_equal_to_n_occ_is_legal():
    """b1 = n_occ - nval = 0 is the whole occupied manifold, not an error."""
    assert _load_refusal()(_config(nval=8), _wfn(nelec=8)) is None


def test_collinear_spin_file_refuses_by_name():
    with pytest.raises(NotImplementedError) as exc:
        _load_refusal()(_config(), _wfn(nspin=2))
    assert "nspin = 2" in str(exc.value)


def test_the_refusal_runs_at_the_wfn_open_not_later():
    """WHERE it sits is the value of it.

    Everything after this line is the ISDF fit and beyond, so the call
    must be the statement that follows the loader construction — not
    merely somewhere before the fit.
    """
    open_at = _SOURCE.index("wfn = WfnLoader(config.paths.wfn_file")
    # Searched FROM the open: the def line above it is the same text.
    call_at = _SOURCE.index(f"{_FN}(config, wfn)", open_at)
    between = _SOURCE[open_at:call_at]
    assert between.count("\n") == 1, (
        "the refusal must be the NEXT statement after the WFN open; "
        f"something moved in between:\n{between}")


def test_the_eqp_writers_own_nspin_guard_is_still_there():
    """``eqp_bgw`` is also a standalone CLI that opens WFN.h5 itself.

    Its guard is not made redundant by the driver-side one — it is the
    only one on that entry point.
    """
    eqp = (_GW_JAX.parent / "eqp_bgw.py").read_text()
    assert "LORRAX runs at nspin=1" in eqp

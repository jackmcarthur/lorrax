"""The Σ stage gates on the self-consistent path.

``gw_jax`` runs ``check_finite(Σ_x)``, ``check_finite(V_H)``,
``check_sign(Σ_x diagonal, negative)`` and ``check_in_range(Σ_x
diagonal, −200…0 eV)`` inside ``if qp_solver is not
QPSolver.SELF_CONSISTENT:``.  The SC loop — the only path that rebuilds
Σ_x in a rotated band basis, and it does so 2·max_iter + 1 times — was
therefore the one path with none of them.  ``sc_iteration``'s
``_check_sigma_stage`` is where they now run, once per iteration.

Every gate is tripped here.  A refusal nobody has fired is untested code
sitting exactly where silent-wrong lives, and both of its arms matter:
``LORRAX_SANITY=strict`` (CI / regression) must raise, the default warn
level must emit the greppable ``*** LORRAX SANITY FAILURE`` line and
keep going.

Requires jax (``sc_iteration`` imports it at module scope), so this runs
in the container, not on a login node.
"""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

jax = pytest.importorskip("jax")

from common import sanity                                     # noqa: E402
from common.units import RYD_TO_EV                             # noqa: E402
from gw import sc_iteration                                    # noqa: E402
from gw.sigma_dispatch import SigmaResult                      # noqa: E402


NK, NB = 2, 4


def _hermitian_with_diag(diag_ev):
    """(nk, nb, nb) complex Σ with the requested real diagonal, in Ry."""
    rng = np.random.default_rng(0)
    off = 0.01 * (rng.normal(size=(NK, NB, NB))
                  + 1j * rng.normal(size=(NK, NB, NB)))
    A = 0.5 * (off + np.conj(np.swapaxes(off, -1, -2)))
    idx = np.arange(NB)
    A[:, idx, idx] = np.asarray(diag_ev, dtype=np.float64) / RYD_TO_EV
    return A.astype(np.complex128)


def _sigma(*, sig_x_diag_ev=-20.0, v_h=None, sig_x=None):
    """A SigmaResult carrying only the fields the stage gates read."""
    if sig_x is None:
        sig_x = _hermitian_with_diag(np.full((NK, NB), sig_x_diag_ev))
    if v_h is None:
        v_h = _hermitian_with_diag(np.full((NK, NB), 12.0))
    return SigmaResult(v_h_kij_ry=v_h, sigma_x_kij_ry=sig_x,
                       sigma_xc_kij_ry=sig_x)


def _run(sig, monkeypatch, *, level="1"):
    monkeypatch.setenv("LORRAX_SANITY", level)
    lines = []
    sc_iteration._check_sigma_stage(sig, print_fn=lines.append)
    return lines


def _failures(lines):
    return [ln for ln in lines if "LORRAX SANITY FAILURE" in ln]


# ---------------------------------------------------------------------------
# The healthy deck must be silent, or nobody will believe a failure.
# ---------------------------------------------------------------------------

def test_healthy_sigma_passes_every_gate(monkeypatch):
    assert _failures(_run(_sigma(), monkeypatch)) == []


def test_healthy_sigma_does_not_raise_under_strict(monkeypatch):
    assert _failures(_run(_sigma(), monkeypatch, level="strict")) == []


def test_gates_are_skipped_when_sanity_is_off(monkeypatch):
    """The escape hatch still works on this path."""
    bad = _sigma(sig_x=_hermitian_with_diag(np.full((NK, NB), +30.0)))
    assert _failures(_run(bad, monkeypatch, level="off")) == []


# ---------------------------------------------------------------------------
# One trip per gate.
# ---------------------------------------------------------------------------

def test_nan_in_sigma_x_is_refused(monkeypatch):
    sig_x = _hermitian_with_diag(np.full((NK, NB), -20.0))
    sig_x[1, 2, 3] = np.nan
    bad = _sigma(sig_x=sig_x)

    assert any("Σ_x contains" in ln for ln in _failures(_run(bad, monkeypatch)))
    with pytest.raises(sanity.SanityError, match="Σ_x contains"):
        _run(bad, monkeypatch, level="strict")


def test_nan_in_v_h_is_refused(monkeypatch):
    v_h = _hermitian_with_diag(np.full((NK, NB), 12.0))
    v_h[0, 0, 0] = np.inf
    bad = _sigma(v_h=v_h)

    assert any("V_H contains" in ln for ln in _failures(_run(bad, monkeypatch)))
    with pytest.raises(sanity.SanityError, match="V_H contains"):
        _run(bad, monkeypatch, level="strict")


def test_positive_sigma_x_diagonal_is_refused(monkeypatch):
    """The sign slip the gate exists for: Σ_x[i,i] is negative definite."""
    diag = np.full((NK, NB), -20.0)
    diag[0, 2] = +3.5
    bad = _sigma(sig_x=_hermitian_with_diag(diag))

    fails = _failures(_run(bad, monkeypatch))
    assert any("are not negative" in ln for ln in fails)
    with pytest.raises(sanity.SanityError, match="not negative"):
        _run(bad, monkeypatch, level="strict")


def test_absurd_sigma_x_diagonal_is_refused(monkeypatch):
    """Right sign, wrong units — caught by the bracket, not by the sign."""
    diag = np.full((NK, NB), -20.0)
    diag[1, 1] = -5000.0
    bad = _sigma(sig_x=_hermitian_with_diag(diag))

    fails = _failures(_run(bad, monkeypatch))
    assert fails and all("not negative" not in ln for ln in fails)
    assert any("outside the physical window" in ln for ln in fails)
    with pytest.raises(sanity.SanityError, match="outside the physical window"):
        _run(bad, monkeypatch, level="strict")


# ---------------------------------------------------------------------------
# The gates are wired into the iteration, not merely defined.
# ---------------------------------------------------------------------------

def test_gw_iteration_map_calls_the_stage_gate(monkeypatch):
    """Guards against the checks being unreachable from the loop.

    ``gw_iteration_map`` cannot be driven in-process without a deck, so
    the wiring is asserted on the source: the call must be there, and it
    must be after the ``compute_sigma_xc`` whose result it checks.
    """
    src = (pathlib.Path(sc_iteration.__file__)).read_text()
    body = src[src.index("def gw_iteration_map("):
               src.index("def _scissor_E_qp_for_outofrange(")]
    assert "_check_sigma_stage(sigma_result" in body
    assert body.index("sigma_result = compute_sigma_xc(") < body.index(
        "_check_sigma_stage(sigma_result")


def test_sc_mpa_maps_receive_the_authenticated_zeta_receipt():
    """The one-shot MPA provenance gate applies equally to every SC map."""
    src = pathlib.Path(sc_iteration.__file__).read_text()
    map_body = src[src.index("def gw_iteration_map("):
                   src.index("def _scissor_E_qp_for_outofrange(")]
    for spelling in (
        "wfn_fingerprint_binding=inputs.wfn_fingerprint_binding",
        "charge_zeta_identity=inputs.charge_zeta_identity",
    ):
        assert spelling in map_body

    driver_body = src[src.index("def run_sc_driver("):
                      src.index("def final_qp_eigenstates(")]
    for spelling in (
        "wfn_fingerprint_binding=wfn_fingerprint_binding",
        "charge_zeta_identity=charge_zeta_identity",
    ):
        assert spelling in driver_body

    gw_jax = pathlib.Path(sc_iteration.__file__).with_name("gw_jax.py")
    driver = gw_jax.read_text()
    assert "wfn_fingerprint_binding=isdf.wfn_fingerprint_binding" in driver
    assert "charge_zeta_identity=isdf.charge_zeta_identity" in driver

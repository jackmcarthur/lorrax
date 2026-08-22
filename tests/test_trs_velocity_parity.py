"""Time-reversal parity of the head velocity — the convention, pinned.

WHAT IS BEING PINNED.  ``v_i(-k) = -conj(v_i(k))``, eq. (2) of the
``src/gw/qsgw_head.py`` module docstring, and the three things about it a
future reader will otherwise get wrong:

1. **It applies to the QSGW velocity, not only the bare one.**  Every term
   of ``v^Q = v^DFT + d_k Sigma - i[A, Sigma]`` carries the SAME odd
   parity (``Sigma`` is even, differentiation flips it, the Berry
   connection is even and the commutator's leading ``-i`` flips it back),
   so a sign error in the correction shows up here and is not absorbed.
   The tree had only ever recorded the bare term's parity.
2. **The verdict statistic is the band TRACE, and that is load-bearing.**
   The elementwise form is gauge-dependent in exactly two ways — an
   unrelated little-group row between ``k`` and ``-k``, and the Kramers
   partner ambiguity of a spinor deck — and the trace survives both.  The
   spinor cell below constructs the second case and shows the elementwise
   diagnostic failing while the trace is exact.
3. **It is an identity only where time reversal is MEASURED to hold.**  On
   a ferromagnet the asymmetry is anomalous-velocity physics, and the gate
   must take no verdict.  Passing ``trs_measured=None`` (no measurement)
   is likewise no verdict — an unmeasured system is not a TRS system.

The trace's own blindness is pinned too (a traceless parity error), so a
null on it can never be quoted as coverage it does not have.
"""

from __future__ import annotations

import numpy as np
import pytest

from gw.qsgw_head import (
    report_trs_velocity_parity,
    trs_velocity_parity_residual,
)

_KGRID = (3, 1, 1)          # neg map 0->0, 1<->2; only k=0 is self-negative


def _hermitian(seed: int, nb: int = 2) -> np.ndarray:
    rng = np.random.RandomState(seed)
    a = rng.randn(nb, nb) + 1j * rng.randn(nb, nb)
    return 0.5 * (a + a.conj().T)


def _parity_clean_velocity(nb: int = 2) -> np.ndarray:
    """``(3, 3, nb, nb)`` obeying ``v(-k) = -conj(v(k))`` exactly.

    k=0 is its own negative, so the relation forces ``v(0)`` to be purely
    imaginary; a Hermitian purely-imaginary matrix has a zero diagonal, so
    ``tr v(0) = 0`` and the trace scale comes from the k=1/k=2 pair.  That
    is not a special case — it is what the identity says about a TRIM.
    """
    v = np.zeros((3, 3, nb, nb), dtype=np.complex128)
    for comp in range(3):
        anti = np.triu(np.ones((nb, nb)), 1)
        v[comp, 0] = 1j * (anti - anti.T) * (comp + 1)
        one = _hermitian(10 * comp + 1, nb)
        v[comp, 1] = one
        v[comp, 2] = -np.conj(one)
    return v


def test_a_parity_clean_velocity_passes_at_the_roundoff_floor():
    m = trs_velocity_parity_residual(
        _parity_clean_velocity(), kgrid=_KGRID, trs_measured=True)
    assert m["trace_rel"] < 1.0e-13, m
    assert m["elementwise_rel"] < 1.0e-13, m
    assert m["verdict"] == 1.0
    assert report_trs_velocity_parity(
        "clean", m, trs_measured=True, print_fn=lambda *a: None) is True


def test_an_inverted_sign_reads_two_and_REFUSES():
    """The failure a wrong sign in ``d_k Sigma`` or ``-i[A, Sigma]`` makes.

    Flipping the parity turns ``|v(-k) + conj(v(k))|`` into ``2|v|``, so
    the residual is 2.0 — the same ``rel 2.000`` signature the symmetry
    register measured for ``dipole_cart`` without its TRS sign.  That is
    strictly more than the whole signal, which is why the refusal bar sits
    at 1.0 and cannot be reached by a gauge or band-window artefact.
    """
    from common.sanity import SanityError

    v = _parity_clean_velocity()
    v[:, 2] = -v[:, 2]                          # the sign error
    m = trs_velocity_parity_residual(v, kgrid=_KGRID, trs_measured=True)
    assert m["trace_rel"] > 1.5, m
    assert m["verdict"] == 0.0
    with pytest.raises(SanityError, match="parity is inverted"):
        report_trs_velocity_parity(
            "flipped", m, trs_measured=True, print_fn=lambda *a: None)


def test_the_named_override_lets_an_operator_proceed_and_leaves_a_trace(
        monkeypatch):
    v = _parity_clean_velocity()
    v[:, 2] = -v[:, 2]
    m = trs_velocity_parity_residual(v, kgrid=_KGRID, trs_measured=True)
    monkeypatch.setenv("LORRAX_ALLOW_TRS_VELOCITY_PARITY_BREAK", "1")
    lines: list[str] = []
    assert report_trs_velocity_parity(
        "flipped", m, trs_measured=True, print_fn=lines.append) is False
    assert any("LORRAX SANITY FAILURE" in ln for ln in lines), lines


# ---------------------------------------------------------------------------
# RED TWIN — magnetic (TRS broken), and unmeasured
# ---------------------------------------------------------------------------

def test_a_magnetic_deck_takes_no_verdict_from_this_statistic():
    """``v(-k) = -conj(v(k))`` is not an identity without Theta.

    A ferromagnet is EXPECTED to violate it — that asymmetry is the
    anomalous-velocity physics — so the same array that refuses above must
    pass here on the strength of the measured density alone.
    """
    v = _parity_clean_velocity()
    v[:, 2] = -v[:, 2]
    m = trs_velocity_parity_residual(v, kgrid=_KGRID, trs_measured=False)
    assert np.isnan(m["verdict"])
    assert m["trace_rel"] > 1.5              # the number is still reported
    lines: list[str] = []
    assert report_trs_velocity_parity(
        "FM", m, trs_measured=False, print_fn=lines.append) is True
    assert any("TIME REVERSAL IS BROKEN" in ln for ln in lines), lines


def test_an_unmeasured_deck_is_not_a_trs_deck():
    """``None`` means nobody measured, which is not the same as True."""
    v = _parity_clean_velocity()
    v[:, 2] = -v[:, 2]
    m = trs_velocity_parity_residual(v, kgrid=_KGRID, trs_measured=None)
    assert np.isnan(m["verdict"])
    lines: list[str] = []
    assert report_trs_velocity_parity(
        "unknown", m, trs_measured=None, print_fn=lines.append) is True
    assert any("NOT MEASURED" in ln for ln in lines), lines


# ---------------------------------------------------------------------------
# RED TWIN — spinor / Kramers, and the trace's own blindness
# ---------------------------------------------------------------------------

def _kramers_rotation() -> np.ndarray:
    """A unitary mixing one Kramers doublet — the gauge freedom at -k."""
    theta = 0.7
    return np.asarray([[np.cos(theta), -np.sin(theta) * 1j],
                       [-np.sin(theta) * 1j, np.cos(theta)]],
                      dtype=np.complex128)


def test_the_kramers_gauge_breaks_the_elementwise_form_and_not_the_trace():
    """WHY THE VERDICT IS THE TRACE, constructed rather than asserted.

    With ``Theta^2 = -1`` the partner of band n at -k is its Kramers
    partner, and inside a degenerate doublet which one carries the label
    is gauge-arbitrary: the stored ``v(-k)`` is
    ``-U conj(v(k)) U^dagger`` for some doublet-block unitary U.  The
    elementwise statistic sees that as a violation.  The trace does not,
    because a trace is invariant under any unitary conjugation — which is
    the whole reason it is the verdict and the elementwise number is only
    a diagnostic.
    """
    U = _kramers_rotation()
    v = _parity_clean_velocity()
    for comp in range(3):
        v[comp, 2] = -U @ np.conj(v[comp, 1]) @ U.conj().T
    m = trs_velocity_parity_residual(v, kgrid=_KGRID, trs_measured=True)
    assert m["trace_rel"] < 1.0e-13, m          # the verdict survives
    assert m["elementwise_rel"] > 1.0e-2, m     # the diagnostic does not
    assert m["verdict"] == 1.0


def test_the_trace_is_blind_to_a_traceless_parity_error():
    """The stated SENSITIVITY LIMIT, pinned so no null is over-quoted.

    A parity error whose band matrix is traceless — a sign flip confined
    to the strictly off-diagonal transition sector is the realistic form —
    moves ``elementwise_rel`` and leaves ``trace_rel`` at the floor.  This
    cell exists so that "the parity gate is green" is never read as "the
    transition sector is verified"; the elementwise number is what carries
    that, on a deck whose gauge is known pair-coherent.
    """
    v = _parity_clean_velocity()
    traceless = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    for comp in range(3):
        v[comp, 2] = v[comp, 2] + traceless
    m = trs_velocity_parity_residual(v, kgrid=_KGRID, trs_measured=True)
    assert m["trace_rel"] < 1.0e-13, m
    assert m["elementwise_rel"] > 1.0e-2, m


def test_the_shape_contract_refuses_rather_than_broadcasting():
    with pytest.raises(ValueError, match=r"\(3, nk, nb, nb\)"):
        trs_velocity_parity_residual(
            np.zeros((3, 5, 2, 2), dtype=np.complex128),
            kgrid=_KGRID, trs_measured=True)

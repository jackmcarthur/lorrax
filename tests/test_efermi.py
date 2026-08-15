"""Step-occupation Fermi level: gapped, metallic, degenerate, k-weighted.

RUNS IN THE CONTAINER, NOT ON A LOGIN NODE.  ``gw.efermi`` is a jax.jit
kernel now, so this imports it normally instead of by path and runs with
the container gates: ``/scratch2/08271/jackmc/dsc_demo/sc_gates.sbatch``
runs it alongside ``test_scissor_weights`` / ``test_sc_band_window`` /
``test_layering``.

The degenerate refusal and the occupation dtype are checked with E on a
device as well as on the host — that is the path the SC loop takes, and
the refusal is decoded from a kernel flag rather than raised where it is
found.
"""
import numpy as np
import pytest

from gw.efermi import (fermi_level_step, mp1_occupations,
                       occupied_band_count, solve_mp1_occupations,
                       step_occupations)


def _roundtrip(E, w, n):
    return occupied_band_count(step_occupations(E, fermi_level_step(E, w, n)), w)


def test_gapped_uniform_lands_midgap():
    """VBM=1, CBM=2 at every k ⇒ E_F is the gap midpoint and the count is exact."""
    E = np.array([[0.0, 1.0, 2.0, 3.0]] * 4)
    w = np.full(4, 0.25)
    assert fermi_level_step(E, w, 2.0) == pytest.approx(1.5)
    assert _roundtrip(E, w, 2.0) == pytest.approx(2.0)


def test_gapped_with_nonuniform_ibz_weights():
    """The IBZ case: weights differ per k and the count must still be exact.

    A fixed-band-index cut cannot express this in general; a weighted
    cumulative can.
    """
    E = np.array([[0.0, 1.0, 2.0, 3.0],
                  [0.1, 1.1, 2.1, 3.1],
                  [0.2, 0.9, 2.2, 3.2]])
    w = np.array([0.0625, 0.125, 0.125])
    w = w / w.sum()
    assert _roundtrip(E, w, 2.0) == pytest.approx(2.0)


def test_metal_shortfall_is_bounded_by_the_largest_weight():
    """A step function realises only partial sums of the k-weights.

    The realised count is therefore AT MOST the target and short of it by
    less than ``max(w)``.  This is the documented, inherent limit of step
    occupations — not a defect — and it is the reason the fractional
    treatment is wanted.
    """
    rng = np.random.default_rng(0)
    E = np.sort(rng.standard_normal((10, 600)), axis=1)
    w = rng.random(10)
    w = w / w.sum()
    got = _roundtrip(E, w, 208.0)
    assert got <= 208.0 + 1e-12
    assert 208.0 - got < w.max() + 1e-12


def test_degenerate_manifold_is_refused_not_approximated():
    """E_F inside a degenerate set has no step representation — refuse.

    ``E < E_F`` takes every state at that energy or none, so any returned
    number would surface only as a wrong electron count much later.
    """
    import jax.numpy as jnp

    E = np.array([[0.0, 1.0, 1.0, 1.0, 5.0]] * 2)
    w = np.array([0.5, 0.5])
    with pytest.raises(ValueError, match="degenerate manifold"):
        fermi_level_step(E, w, 3.0)
    # And with E on a device: the condition is decoded from the kernel's
    # returned 3-vector there, and that is the path the SC loop takes.
    with pytest.raises(ValueError, match="degenerate manifold"):
        fermi_level_step(jnp.asarray(E), w, 3.0)


def test_agrees_with_sc_iteration_midgap_on_a_real_gap():
    """No second convention: on the case they share, the two agree exactly.

    ``sc_iteration._diagonalize_and_get_efermi`` uses ``0.5·(vbm+cbm)``
    with a fixed band cut.  That is right when a genuine gap separates
    band ``nocc-1`` from ``nocc`` at every k, and there this must match it
    to the last bit.  (They legitimately differ when bands overlap across
    k — the fixed cut is not the general answer, which is the reason this
    module exists.)
    """
    rng = np.random.default_rng(0)
    nk, nb, nocc = 6, 20, 8
    E = np.sort(rng.standard_normal((nk, nb)), axis=1)
    E[:, nocc:] += 10.0                      # force a real gap at the cut
    w = np.full(nk, 1.0 / nk)
    vbm = E[:, :nocc].max()
    cbm = E[:, nocc:].min()
    assert fermi_level_step(E, w, float(nocc)) == pytest.approx(
        0.5 * (vbm + cbm), abs=1e-12)
    assert _roundtrip(E, w, float(nocc)) == pytest.approx(float(nocc))


def test_kweights_shape_mismatch_refuses():
    """Full-BZ weights with IBZ energies silently rescales the count."""
    E = np.zeros((4, 3))
    with pytest.raises(ValueError, match="kweights must be"):
        fermi_level_step(E, np.full(16, 1 / 16), 1.0)


def test_occupations_are_float64_for_the_fractional_successor():
    """The dtype is the contract ρ contracts against; keep it float64.

    ``rho_from_wfns`` takes these as a WEIGHT, so the finite-temperature
    successor swaps in a Fermi–Dirac factor with no consumer changing.
    The device operand must keep the dtype: it is the one that reaches ρ.
    """
    import jax.numpy as jnp

    E = np.array([[0.0, 1.0, 2.0]])
    occ = step_occupations(E, 1.5)
    assert occ.dtype == np.float64
    assert occ.tolist() == [[1.0, 1.0, 0.0]]
    occ_dev = step_occupations(jnp.asarray(E), 1.5)
    assert occ_dev.dtype == np.float64
    assert occ_dev.tolist() == [[1.0, 1.0, 0.0]]


def test_mp1_matches_the_berkeleygw_fe_reference_state():
    """Pin the formula and the factor-of-two width to the BGW 4.0 artifact."""
    ryd_to_ev = 13.605693122994
    occ = mp1_occupations(
        np.array([[18.547842 / ryd_to_ev]]),
        18.526851685673 / ryd_to_ev,
        0.27211385 / ryd_to_ev,
    )
    # sigma.out prints this occupation to six decimals.
    assert float(occ[0, 0]) == pytest.approx(0.467387, abs=5.0e-7)

    # Red twin: treating BGW occ_broadening as QE's denominator is a
    # different convention and must not accidentally reproduce the fixture.
    wrong = mp1_occupations(
        np.array([[18.547842 / ryd_to_ev]]),
        18.526851685673 / ryd_to_ev,
        0.5 * 0.27211385 / ryd_to_ev,
    )
    assert abs(float(wrong[0, 0]) - 0.467387) > 1.0e-2


def test_mp1_fixed_electron_spinor_and_nonuniform_ibz_weights():
    """One electron per spinor state, with the IBZ star weights in the root."""
    E = np.array([
        [-1.0, -0.20, 0.45, 1.7],
        [-0.8,  0.05, 0.60, 1.9],
        [-0.7,  0.25, 0.85, 2.1],
    ])
    w = np.array([0.125, 0.375, 0.500])
    target = 2.25
    mu, occ = solve_mp1_occupations(
        E, w, target, 0.08, state_capacity=1.0)
    got = float(np.einsum("k,kn->", w, np.asarray(occ)))
    assert got == pytest.approx(target, abs=2.0e-13)
    assert np.isfinite(float(mu))

    # A restricted scalar state has twice the capacity.  The same physical
    # electron target therefore has a lower chemical potential; silently
    # applying this factor to the spinor arm would be observable here.
    mu_scalar, occ_scalar = solve_mp1_occupations(
        E, w, target, 0.08, state_capacity=2.0)
    got_scalar = 2.0 * float(np.einsum("k,kn->", w, np.asarray(occ_scalar)))
    assert got_scalar == pytest.approx(target, abs=2.0e-13)
    assert float(mu_scalar) < float(mu)


def test_mp1_solver_refuses_ambiguous_weights_and_width():
    E = np.zeros((2, 4))
    with pytest.raises(ValueError, match="sum to 1"):
        solve_mp1_occupations(
            E, np.array([0.25, 0.25]), 2.0, 0.1, state_capacity=1.0)
    with pytest.raises(ValueError, match="broadening_ry"):
        solve_mp1_occupations(
            E, np.array([0.5, 0.5]), 2.0, 0.0, state_capacity=1.0)

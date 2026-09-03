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


# ---------------------------------------------------------------------------
# OccupationState — the frozen cross-agent contract (metal_mpa_plan W3)
# ---------------------------------------------------------------------------

from gw.efermi import (OccupationState, assert_fixed_n,
                       assert_wfn_occupation_consistency)


def _metal_grid():
    """A dense symmetric one-k 'metal': mu lands at 0 by MP1 symmetry."""
    E = np.linspace(-0.1, 0.1, 201)[None, :]
    w = np.array([1.0])
    return E, w


def test_occupation_state_mp1_owns_the_fixed_n_invariant():
    E, w = _metal_grid()
    target = 100.5
    st = OccupationState.solve_mp1(E, w, target, 0.02, state_capacity=1.0)
    assert st.smearing_family == "mp1"
    assert st.smearing_width_ry == pytest.approx(0.02)
    assert st.n_electrons == pytest.approx(target)
    # The invariant the constructor asserted, re-checked here explicitly.
    realized = assert_fixed_n(st, w, state_capacity=1.0)
    assert realized == pytest.approx(target, abs=1e-10)
    # A mismatched target refuses by name (direct construction + assert).
    bad = OccupationState(
        f_kn=st.f_kn, mu_ry=st.mu_ry, smearing_family="mp1",
        smearing_width_ry=0.02, n_electrons=target + 1.0)
    with pytest.raises(ValueError, match="fixed-N invariant"):
        assert_fixed_n(bad, w, state_capacity=1.0)


def test_occupation_state_keeps_mp1_overshoot_unclipped():
    """MP1 exceeds 1 just below mu (max ~1.025 at x=-1); it must survive.

    Clipping would silently change every ρ/Σ contraction on a metal; the
    contract says NEVER clipped, so the overshoot is the discriminating
    observable.
    """
    E, w = _metal_grid()
    st = OccupationState.solve_mp1(E, w, 100.5, 0.02, state_capacity=1.0)
    f = np.asarray(st.f_kn)
    assert float(f.max()) > 1.0
    assert float(f.min()) < 0.0  # the mirrored negative lobe above mu


def test_occupation_state_hash_binds_to_the_table():
    E, w = _metal_grid()
    a = OccupationState.solve_mp1(E, w, 100.5, 0.02, state_capacity=1.0)
    b = OccupationState.solve_mp1(E, w, 100.5, 0.02, state_capacity=1.0)
    c = OccupationState.solve_mp1(E, w, 101.5, 0.02, state_capacity=1.0)
    assert a.occ_hash == b.occ_hash
    assert a.occ_hash != c.occ_hash
    assert len(a.occ_hash) == 16
    with pytest.raises(ValueError, match="occ_hash"):
        OccupationState(
            f_kn=a.f_kn, mu_ry=a.mu_ry, smearing_family="mp1",
            smearing_width_ry=0.02, n_electrons=100.5,
            occ_hash="0" * 16)


def test_occupation_state_hash_ignores_only_trailing_exact_zero_columns():
    base = np.asarray([[1.0, 0.4], [1.0, 0.6]])
    padded = np.pad(base, ((0, 0), (0, 7)))
    a = OccupationState(
        f_kn=base, mu_ry=0.1, smearing_family="mp1",
        smearing_width_ry=0.02, n_electrons=2.0)
    b = OccupationState(
        f_kn=padded, mu_ry=0.1, smearing_family="mp1",
        smearing_width_ry=0.02, n_electrons=2.0)
    assert a.occ_hash == b.occ_hash

    changed = padded.copy()
    changed[0, -1] = np.nextafter(0.0, 1.0)
    c = OccupationState(
        f_kn=changed, mu_ry=0.1, smearing_family="mp1",
        smearing_width_ry=0.02, n_electrons=2.0)
    assert c.occ_hash != a.occ_hash


def test_occupation_state_step_is_insulating_only():
    # Gapped: works, family "fixed", zero width, capacity-weighted target.
    E = np.array([[0.0, 1.0, 2.0, 3.0]] * 4)
    w = np.full(4, 0.25)
    st = OccupationState.step(E, w, 2.0, state_capacity=2.0)
    assert st.smearing_family == "fixed"
    assert st.smearing_width_ry == 0.0
    assert st.n_electrons == pytest.approx(4.0)
    assert st.mu_ry == pytest.approx(1.5)
    # Metallic partial fill: refused by name, pointing at solve_mp1.
    rng = np.random.default_rng(1)
    Em = np.sort(rng.standard_normal((10, 60)), axis=1)
    wm = rng.random(10)
    wm = wm / wm.sum()
    with pytest.raises(ValueError, match="solve_mp1"):
        OccupationState.step(Em, wm, 20.8)


def test_occupation_state_family_and_width_are_coupled():
    E, w = _metal_grid()
    st = OccupationState.solve_mp1(E, w, 100.5, 0.02, state_capacity=1.0)
    with pytest.raises(ValueError, match="width"):
        OccupationState(
            f_kn=st.f_kn, mu_ry=st.mu_ry, smearing_family="fixed",
            smearing_width_ry=0.02, n_electrons=100.5)
    with pytest.raises(ValueError, match="width"):
        OccupationState(
            f_kn=st.f_kn, mu_ry=st.mu_ry, smearing_family="mp1",
            smearing_width_ry=0.0, n_electrons=100.5)


def test_wfn_occupation_consistency_discriminates_width():
    """The degauss-vs-degauss/2 trap must fire, and the matched case pass.

    A WFN whose stored occupations were made at width w agrees with our
    solve at w to round-off; the same WFN checked at 2w deviates at the
    Fermi surface by ~1e-1 — far above the 1e-6 gate.
    """
    E, w = _metal_grid()
    width = 0.02
    st = OccupationState.solve_mp1(E, w, 100.5, width, state_capacity=1.0)
    stored = np.asarray(mp1_occupations(E, st.mu_ry, width))
    dev = assert_wfn_occupation_consistency(
        st, stored, w, state_capacity=1.0, num_electrons=100.5)
    assert dev <= 1e-12
    stored_wrong = np.asarray(mp1_occupations(E, st.mu_ry, 2.0 * width))
    with pytest.raises(ValueError, match="degauss/2"):
        assert_wfn_occupation_consistency(
            st, stored_wrong, w, state_capacity=1.0, num_electrons=100.5)

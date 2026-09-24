"""Identity classification, local multiplets and shared SC pad contracts."""
import numpy as np
import jax.numpy as jnp
import pytest

from common.units import RYD_TO_EV
from gw.band_partition import BandPartition, apply_band_partition, build_omega_band_partition
from gw.sc_state_identity import assign_qp_identity
from gw.scissor import sc_state_pad_ev


def _partition(energy_ev, reference_ev, *, previous=None, indices=None, log=None):
    return build_omega_band_partition(
        np.asarray(energy_ev) / RYD_TO_EV,
        np.asarray(reference_ev) / RYD_TO_EV,
        band_offset=0, omega_min_abs_ev=-2.0, omega_max_abs_ev=5.0,
        previous_partition=previous, mu_ev=0.0,
        current_indices_kn=indices, print_fn=(log.append if log is not None else lambda _: None))


@pytest.mark.parametrize('energy,pad', [(0.0, 0.5), (5.0, 1.0), (20.0, 2.5)])
def test_pad_sizes(energy, pad):
    assert sc_state_pad_ev(energy) == pytest.approx(pad)
    assert sc_state_pad_ev(-energy) == pytest.approx(pad)


def test_hysteresis_enter_retain_escape():
    reference = np.array([[0.0, 4.9], [0.0, 5.1]])
    outside = _partition(reference, reference)
    assert not np.asarray(outside.protected_mask)[:, 1].any()
    entered_e = np.array([[0.0, 4.9], [0.0, 5.0]])
    entered = _partition(entered_e, reference, previous=outside)
    assert np.asarray(entered.protected_mask)[:, 1].all()
    retained_e = np.array([[0.0, 5.9], [0.0, 6.0]])
    retained = _partition(retained_e, reference, previous=entered)
    assert np.asarray(retained.protected_mask)[:, 1].all()
    assert not np.asarray(retained.in_range_mask)[:, 1].any()
    log = []
    escaped_e = np.array([[0.0, 5.9], [0.0, 6.2]])
    escaped = _partition(escaped_e, reference, previous=retained, log=log)
    assert not np.asarray(escaped.protected_mask)[:, 1].any()
    assert any('escape: band=2, k=1' in line and 'pad=1.120000' in line for line in log)
    h = jnp.asarray([[[1., .2], [.2, 7.]], [[1., .2], [.2, 8.]]])
    kept = apply_band_partition(h, protected_mask=retained.protected_mask,
                               in_range_mask=retained.in_range_mask,
                               scissor_E_qp_kn=jnp.zeros((2, 2)))
    np.testing.assert_array_equal(kept, h)


def test_reference_multiplets_close_per_k_without_transitive_union():
    # Labels 1/2 share a multiplet only at k=0; 2/3 only at k=1.
    reference = np.array([[0., 0., 8.], [0., 8., 8.]])
    part = _partition(reference, reference)
    np.testing.assert_array_equal(part.protected_mask,
                                  [[True, True, False], [True, False, False]])


def test_crossing_scissored_doublet_preserves_protected_identity_block():
    # A spectator plus protected singlet below the window edge; the tail
    # doublet crosses the singlet only at k=1. Carry coordinates stay DFT.
    reference = np.array([[0., 4., 7., 7.], [0., 4., 7., 7.]])
    identity_e = np.array([[0., 4., 7., 7.], [0., 4., 3., 3.]])
    order = np.argsort(identity_e, axis=1, kind='stable')
    u0 = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    u = np.stack([u0[k][:, order[k]] for k in range(2)])
    sorted_e = np.take_along_axis(identity_e, order, axis=1)
    indices, aligned_e, _, _ = assign_qp_identity(
        u0, reference, u, sorted_e, np.ones((2, 4), bool), degeneracy_tol_ev=1e-5)
    log = []
    part = _partition(aligned_e, reference, indices=indices, log=log)
    np.testing.assert_array_equal(part.protected_mask,
                                  [[True, True, False, False]] * 2)
    assert any('k=1:' in line and 'sorted columns=[1, 4]' in line for line in log)
    h = np.broadcast_to(np.diag([1., 5., 9., 9.]), (2, 4, 4)).copy()
    h[:, 0, 1] = h[:, 1, 0] = .25
    h[:, 1, 2] = h[:, 2, 1] = .75
    result = np.asarray(apply_band_partition(
        jnp.asarray(h), protected_mask=part.protected_mask,
        in_range_mask=part.in_range_mask,
        scissor_E_qp_kn=jnp.asarray([[1., 5., 10., 10.]] * 2)))
    np.testing.assert_array_equal(result[:, 0, 1], [.25, .25])
    np.testing.assert_array_equal(result[:, 1, 2], [0., 0.])
    # Red twin: substituting sorted-column masks on the DFT carry drops
    # the physical protected coupling at precisely the crossing k.
    wrong_mask = np.array([[True, True, False, False],
                           [True, False, False, True]])
    wrong = np.asarray(apply_band_partition(
        jnp.asarray(h), protected_mask=jnp.asarray(wrong_mask),
        in_range_mask=jnp.asarray(wrong_mask),
        scissor_E_qp_kn=jnp.asarray([[1., 5., 10., 10.]] * 2)))
    assert wrong[0, 0, 1] == pytest.approx(.25)
    assert wrong[1, 0, 1] == 0.0
    assert not np.array_equal(wrong, result)
    # In sorted coordinates the same retained coupling has moved to column 4.
    sorted_result = u.conj().transpose(0, 2, 1) @ result @ u
    assert sorted_result[1, 0, 3] == pytest.approx(.25)
    np.testing.assert_array_equal(sorted_result[1, 0, 1:3], [0., 0.])


def test_map0_grid_is_the_requested_grid_and_grows_only_on_escape():
    # The requested grid is the evaluation grid at map 0 for every state
    # (SC iteration 1 equals the one-shot; states outside the requested
    # window keep the one-shot treatment).  The pad widens the quadrature
    # support and the hysteresis bounds; the sampled grid grows only when
    # a RETAINED state escapes, to E +/- pad(E), keeping every old sample.
    from types import SimpleNamespace
    from gw.gw_config import LorraxConfig, QPSolver
    from gw.scissor import extend_sc_omega_grid_ev, sc_padded_window_ev
    sigma = SimpleNamespace(omega_min_ev=-2., omega_max_ev=5.,
                            omega_step_ev=.25, parsed_omega_patches_ev=lambda: [])
    original = -2. + .25 * np.arange(29)
    for solver in (QPSolver.SELF_CONSISTENT, QPSolver.ONE_SHOT_DFT):
        cfg = SimpleNamespace(sigma=sigma, qp_solver=solver)
        np.testing.assert_array_equal(LorraxConfig.omega_grid_ev.fget(cfg), original)
    # a hand-built shim without qp_solver (the crossing-cost-law tests)
    np.testing.assert_array_equal(
        LorraxConfig.omega_grid_ev.fget(SimpleNamespace(sigma=sigma)), original)
    # the session's grown support overrides the requested grid for SC only
    grown = SimpleNamespace(sigma=sigma, qp_solver=QPSolver.SELF_CONSISTENT,
                            sc_omega_grid_ev=tuple(original) + (5.25, 5.5))
    assert LorraxConfig.omega_grid_ev.fget(grown)[-1] == 5.5
    lo, hi = sc_padded_window_ev(-2., 5.)
    assert lo == pytest.approx(-2.5 / .9) and hi == pytest.approx(5.5 / .9)
    assert hi - sc_state_pad_ev(hi) == pytest.approx(5.)
    # no retained escape: unchanged
    energies = np.array([[-1., 0.5, 5.3, 9.]])
    retained = np.array([True, True, False, False])
    np.testing.assert_array_equal(
        extend_sc_omega_grid_ev(original, energies, retained, .25), original)
    # a retained state at 5.3 eV escaped the top: grow to E + pad(E), keep
    # every old sample, do not grow the bottom
    retained[2] = True
    extended = extend_sc_omega_grid_ev(original, energies, retained, .25)
    np.testing.assert_array_equal(extended[:29], original)
    assert extended[-1] >= 5.3 + sc_state_pad_ev(5.3)
    assert extended[0] == -2. and np.allclose(np.diff(extended), .25)


def test_identity_priority_preserves_established_readout_assignment():
    rng = np.random.default_rng(74)
    reference = np.eye(5)[None]
    current = np.linalg.qr(rng.normal(size=(5, 5)))[0][None]
    energy = np.arange(5.)[None]
    trusted = np.array([True, True, False, False, False])
    old, _, _, _ = assign_qp_identity(reference, energy, current, energy,
                                     trusted, degeneracy_tol_ev=1e-4)
    extended, _, _, _ = assign_qp_identity(
        reference, energy, current, energy, np.ones(5, bool),
        priority_mask=trusted, degeneracy_tol_ev=1e-4)
    np.testing.assert_array_equal(extended[:, trusted], old[:, trusted])
    np.testing.assert_array_equal(np.sort(extended, axis=1), [np.arange(5)])


def test_promoted_multiplet_grows_sampled_support_and_reuses_it():
    from gw.scissor import extend_sc_omega_grid_ev

    reference = np.array([[0., 4., 4.], [0., 4., 8.]])
    energy = np.array([[0., 4., 7.], [0., 4., 8.]])
    part = _partition(energy, reference)
    required = np.asarray(part.protected_mask | part.in_range_mask)
    assert required[0, 2] and not required[1, 2]
    original = np.arange(-3., 6.25 + .125, .25)
    grown = extend_sc_omega_grid_ev(original, energy, required, .25)
    assert grown[-1] == pytest.approx(8.25)
    assert grown[0] == original[0]
    np.testing.assert_array_equal(grown[:original.size], original)
    # Without growth the retained 7 eV member would interpolate to 6.25.
    assert np.interp(7., original, original) != pytest.approx(7.)
    assert np.interp(7., grown, grown) == pytest.approx(7.)
    repeated = extend_sc_omega_grid_ev(grown, energy, required, .25)
    np.testing.assert_array_equal(repeated, grown)


def test_sc_support_growth_low_edge_and_unrequired_outlier():
    from gw.scissor import extend_sc_omega_grid_ev

    original = np.arange(-3., 6.25 + .125, .25)
    energy = np.array([[-4., 0., 50.]])
    grown = extend_sc_omega_grid_ev(original, energy, [[True, True, False]], .25)
    assert grown[0] == pytest.approx(-5.)
    assert grown[-1] == original[-1]
    np.testing.assert_array_equal(grown[-original.size:], original)


def test_sc_support_unchanged_for_covered_energies():
    from gw.scissor import extend_sc_omega_grid_ev

    original = np.arange(-3., 6.25 + .125, .25)
    grown = extend_sc_omega_grid_ev(original, [[-3., 6.25]], [[True, True]], .25)
    np.testing.assert_array_equal(grown, original)


def test_sc_support_retains_patched_grid_hole_refusal():
    from gw.scissor import extend_sc_omega_grid_ev

    patched = np.r_[np.arange(-3., -.9, .25), np.arange(2., 6.3, .25)]
    with pytest.raises(ValueError, match="omega_grid_hole"):
        extend_sc_omega_grid_ev(patched, [[0.]], [[True]], .25)


def test_scissor_fit_keeps_a_crossed_protected_sample_paired():
    from gw.scissor import fit_scissor

    dft = np.array([[0., 4., 7., 7.]])
    qp = np.array([[.5, 4.5, 3., 3.]])
    keep = np.array([[True, True, False, False]])
    valence = np.zeros_like(keep)
    fit = fit_scissor(dft, qp, valence, keep, k_weights=np.ones(1))
    assert fit.alpha_c == pytest.approx(1.)
    assert fit.beta_c_ev == pytest.approx(.5)
    # Red twin: independently sorting QP energies substitutes the tail
    # doublet's 3 eV value for the protected singlet's 4.5 eV sample.
    wrong = fit_scissor(dft, np.sort(qp, axis=1), valence, keep,
                        k_weights=np.ones(1))
    assert wrong.alpha_c == pytest.approx(.625)
    assert wrong.alpha_c != fit.alpha_c
    permutation = [2, 0, 3, 1]
    shuffled = fit_scissor(dft[:, permutation], qp[:, permutation],
                          valence[:, permutation], keep[:, permutation],
                          k_weights=np.ones(1))
    assert shuffled.alpha_c == fit.alpha_c
    assert shuffled.beta_c_ev == fit.beta_c_ev


def test_hysteresis_retained_state_remains_a_scissor_fit_sample():
    from types import SimpleNamespace
    from gw.sc_iteration import _apply_scissor_partition_policy

    reference = np.array([[-1., 4., 8.]])
    initial = _partition(reference, reference)
    energy = np.array([[-1., 6., 10.]])
    retained = _partition(energy, reference, previous=initial)
    assert retained.protected_mask[0, 1]
    assert not retained.in_range_mask[0, 1]
    kstar = SimpleNamespace(irr_idx=np.array([0]), select=lambda a: a)
    h = jnp.asarray(np.diag(energy[0] / RYD_TO_EV)[None])
    result, fit, promoted = _apply_scissor_partition_policy(
        h, reference / RYD_TO_EV, np.array([[True, False, False]]),
        retained, kstar, efermi_dft_ry=0., n_occ=1,
        candidate_efermi_fn=lambda _: 0., print_fn=lambda _: None)
    assert fit.n_fit_c == 1
    assert fit.alpha_c == pytest.approx(1.)
    assert fit.beta_c_ev == pytest.approx(2.)
    # The retained 6 eV diagonal stays measured; its +2 eV fit corrects
    # the excluded DFT 8 eV state to 10 eV instead of leaving it at DFT.
    np.testing.assert_allclose(np.diagonal(result, axis1=1, axis2=2) * RYD_TO_EV,
                               [[-1., 6., 10.]])


def test_frontier_manifold_is_promoted_instead_of_refused():
    """A crossing band outside the energy window is protected by identity."""
    from types import SimpleNamespace
    from gw.sc_iteration import _apply_scissor_partition_policy
    from gw.scissor import ScissorBandClasses

    reference = np.array([[-1., 4., 8.]])
    energy = np.array([[-1., 6., 10.]])
    partition = _partition(energy, reference)
    assert not np.asarray(partition.protected_mask)[0, 1]
    assert not np.asarray(partition.in_range_mask)[0, 1]
    classes = ScissorBandClasses(valence_stop=1, conduction_start=2)
    kstar = SimpleNamespace(irr_idx=np.array([0]), select=lambda a: a)
    h = jnp.asarray(np.diag(energy[0] / RYD_TO_EV)[None])
    result, _fit, promoted = _apply_scissor_partition_policy(
        h, reference / RYD_TO_EV, np.array([[True, False, False]]),
        partition, kstar, efermi_dft_ry=0., n_occ=1,
        candidate_efermi_fn=lambda _: 0., band_classes=classes,
        print_fn=lambda _: None)
    assert np.asarray(promoted.protected_mask)[0, 1]
    # The promoted crossing band keeps its full candidate diagonal.
    assert np.diag(np.asarray(result)[0])[1] == pytest.approx(
        energy[0, 1] / RYD_TO_EV)


def test_frozen_partition_never_promotes_a_later_map():
    """Owner ruling 2026-09-19: the protected set is decided once, on map 0.

    Same crossing fixture as the promotion test, with the promotion switched
    off the way ``gw_iteration_map`` switches it off for every map after the
    first: the identity set must not grow, so the crossing band keeps the
    scissor law for the whole loop.
    """
    from types import SimpleNamespace
    from gw.sc_iteration import _apply_scissor_partition_policy
    from gw.scissor import ScissorBandClasses

    reference = np.array([[-1., 4., 8.]])
    energy = np.array([[-1., 6., 10.]])
    partition = _partition(energy, reference)
    classes = ScissorBandClasses(valence_stop=1, conduction_start=2)
    kstar = SimpleNamespace(irr_idx=np.array([0]), select=lambda a: a)
    h = jnp.asarray(np.diag(energy[0] / RYD_TO_EV)[None])
    _result, _fit, promoted = _apply_scissor_partition_policy(
        h, reference / RYD_TO_EV, np.array([[True, False, False]]),
        partition, kstar, efermi_dft_ry=0., n_occ=1,
        candidate_efermi_fn=lambda _: 0., band_classes=classes,
        allow_frontier_promotion=False, print_fn=lambda _: None)
    np.testing.assert_array_equal(
        np.asarray(promoted.protected_mask),
        np.asarray(partition.protected_mask))


def test_index_fallback_frontier_is_identity_space_and_crossing_only():
    """Without an occupation table the fallback frontier must still name a band.

    ``retained_kn`` inside the policy is a ``(k, DFT identity)`` mask, so a
    fallback that names sorted columns promotes whatever identity happens to
    sit in that column on that map -- measured as the Fe 4x4x4 promotion
    flipping 21 -> 22 -> 21 between calls and moving a band 3-5 eV against a
    1e-4 eV cutoff.  Two statements pin the replacement: a far LUMO across a
    vacuum gap is never promoted, and a near-Fermi LUMO that the window
    scissored is promoted by identity.
    """
    from types import SimpleNamespace
    from gw.band_partition import BandPartition
    from gw.sc_iteration import _apply_scissor_partition_policy

    kstar = SimpleNamespace(irr_idx=np.array([0]), select=lambda a: a)
    retained = BandPartition(
        protected_mask=jnp.asarray([[True, False, False]]),
        in_range_mask=jnp.asarray([[True, False, False]]))

    def promoted_for(reference_ev, fermi_ev):
        energy = np.asarray(reference_ev, dtype=np.float64)
        h = jnp.asarray(np.diag(energy[0] / RYD_TO_EV)[None])
        _result, _fit, promoted = _apply_scissor_partition_policy(
            h, energy / RYD_TO_EV, np.asarray([[True, False, False]]),
            retained, kstar, efermi_dft_ry=fermi_ev / RYD_TO_EV, n_occ=2,
            candidate_efermi_fn=lambda _: 0., print_fn=lambda _: None)
        return np.asarray(promoted.protected_mask)

    # 20 eV above the Fermi level: no crossing, nothing to protect.
    vacuum = promoted_for(np.array([[-1.0, 20.0, 30.0]]), 0.5)
    assert vacuum[0, 1] is np.bool_(False)
    # 0.1 eV above the Fermi level, scissored by the window: promoted, and the
    # identity that is promoted is the LUMO's, not a column index.
    crossing = promoted_for(np.array([[-1.0, 0.6, 30.0]]), 0.5)
    assert crossing[0, 1] is np.bool_(True)
    assert crossing[0, 2] is np.bool_(False)


def test_fermi_classes_follow_per_k_state_identities():
    from gw.scissor import ScissorBandClasses

    classes = ScissorBandClasses(
        valence_stop=1, conduction_start=2,
        current_indices_kn=np.array([[0, 1, 2], [2, 0, 1]]))
    valence, crossing = classes.masks((2, 3))
    np.testing.assert_array_equal(valence, [[True, False, False], [False, True, False]])
    np.testing.assert_array_equal(crossing, [[False, True, False], [False, False, True]])
    np.testing.assert_array_equal(~(valence | crossing),
                                  [[False, False, True], [True, False, False]])
    # The sorted-only class mask puts k=1's crossing label on the wrong state.
    _, wrong_crossing = ScissorBandClasses(1, 2).masks((2, 3))
    assert not np.array_equal(crossing[1], wrong_crossing[1])
    with pytest.raises(ValueError, match="same active k/band shape"):
        classes.masks((1, 3))

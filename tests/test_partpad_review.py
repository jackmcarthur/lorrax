"""Failure scenarios from the PARTPAD landing review, 2026-09-06."""
from dataclasses import replace
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from common.units import RYD_TO_EV
from gw.band_partition import BandPartition, build_omega_band_partition
from gw.sc_state_identity import assign_qp_identity
from gw.scissor import extend_sc_omega_grid_ev


def test_priority_labels_cannot_share_a_reference_block_with_untrusted_labels():
    # The nonmonotone reference grouping used to report trusted E=3 instead
    # of E=1 by averaging it with the untrusted E=5 column.
    with pytest.raises(ValueError, match='cuts a multiplet'):
        assign_qp_identity(np.eye(4)[None], [[0., 1., .95, 2.]],
                           np.eye(4)[None], [[0., 1., 5., 6.]],
                           np.ones(4, bool), degeneracy_tol_ev=1e-4,
                           priority_mask=[False, True, False, False])


def test_all_retained_preserves_supplied_frozen_scissor():
    from gw.sc_iteration import _scissor_E_qp_for_outofrange
    fit = object()
    energy = jnp.array([[0., 1.]])
    result, returned = _scissor_E_qp_for_outofrange(
        jnp.eye(2)[None], energy, [[True, False]], [True, True], None,
        scissor_fit=fit, fermi_displacement_ry=0.)
    assert returned is fit
    np.testing.assert_array_equal(result, energy)


def test_sorted_frontier_checks_the_assigned_identity():
    from gw.sc_iteration import _apply_scissor_partition_policy
    from gw.scissor import fit_scissor
    e = np.array([[-1., 1., 3., 4.]])
    fit = fit_scissor(e, e + .1, [[True, False, False, False]],
                     [[True, True, False, False]], k_weights=np.ones(1))
    with pytest.raises(ValueError, match='Fermi anchor'):
        _apply_scissor_partition_policy(
            jnp.asarray(np.diag(e[0] / RYD_TO_EV)[None]), e / RYD_TO_EV,
            [[True, False, False, False]],
            BandPartition(jnp.array([[True, True, False, False]]),
                          jnp.array([[True, True, False, False]])), None,
            efermi_dft_ry=0., n_occ=1, candidate_efermi_fn=lambda _: 0.,
            scissor_fit=fit, current_indices_kn=np.array([[0, 3, 1, 2]]),
            print_fn=lambda _: None)


def test_growth_uses_four_sample_blocks_and_keeps_old_samples():
    grid = np.arange(-2., 5.01, .25)
    grown = extend_sc_omega_grid_ev(grid, [[-2.1, 5.3]], [[True, True]], .25)
    low = np.flatnonzero(grown == grid[0])[0]
    high = grown.size - grid.size - low
    assert low % 4 == high % 4 == 0
    np.testing.assert_array_equal(grown[low:low + grid.size], grid)
    np.testing.assert_array_equal(
        extend_sc_omega_grid_ev(grown, [[0., 1.]], [[True, True]], .25), grown)


def test_invalid_session_grid_refuses_before_sigma():
    from gw.gw_config import LorraxConfig, QPSolver
    cfg = SimpleNamespace(qp_solver=QPSolver.SELF_CONSISTENT,
                          sc_omega_grid_ev=(0., 1., .5))
    with pytest.raises(ValueError, match='ascending finite'):
        LorraxConfig.omega_grid_ev.fget(cfg)


def test_real_device_overlaps_preserve_complex_assignment():
    from gw.qsgw_density import band_rotation_weights
    rng = np.random.default_rng(19)
    u = np.linalg.qr(rng.normal(size=(5, 5)) +
                     1j * rng.normal(size=(5, 5)))[0][None]
    energy = np.arange(5.)[None]
    weights = np.asarray(band_rotation_weights(jnp.asarray(u)))
    assert weights.nbytes * 2 == u.nbytes
    expected = assign_qp_identity(np.eye(5)[None], energy, u, energy,
                                  np.ones(5, bool), degeneracy_tol_ev=1e-4)
    actual = assign_qp_identity(None, energy, None, energy, np.ones(5, bool),
                                degeneracy_tol_ev=1e-4, overlap_weights=weights)
    for left, right in zip(actual, expected):
        np.testing.assert_allclose(left, right, atol=1e-14)


def test_storage_only_bands_never_enter_sigma_branch_geometry():
    from gw.mpa.sigma import _branches
    from gw.wavefunction_bundle import BandSlices
    def branches(npad):
        slices = BandSlices.from_band_edges(0, 0, 1, 3, npad,
                                            b4_logical=3)
        wfns = SimpleNamespace(slices=slices,
            enk=jnp.asarray([[-1., 1., 3.] + [0.] * (npad - 3)]),
            occ=jnp.asarray([[1., 0., 0.] + [0.] * (npad - 3)]))
        return _branches(wfns, np.array([-.5, .5]), 0.)
    for p1, p4 in zip(branches(3), branches(4)):
        a = np.asarray(p1.E_A)[np.asarray(p1.base_mask_A)]
        b = np.asarray(p4.E_A)[np.asarray(p4.base_mask_A)]
        np.testing.assert_array_equal(a, b)


def test_coverage_uses_grown_result_grid():
    from gw.production_report import GWProductionReport
    lines = []
    report = SimpleNamespace(heading=lines.append, emit=lines.append)
    config = SimpleNamespace(compute_mode=SimpleNamespace(is_dynamic=True),
        sigma=SimpleNamespace(omega_min_ev=-10., omega_max_ev=10.,
                              omega_step_ev=.1, fermi_reference='midgap',
                              band_extrapolation=False, band_extrapolation_estimator='none',
                              band_extrapolation_bracket_scheme='none'))
    result = SimpleNamespace(omega_grid_ev=np.linspace(-13.2, 11.4, 247),
                             efermi_dft_ev=0.)
    GWProductionReport.sigma_coverage(report, config=config,
        band_slices=SimpleNamespace(b0=0, b1=0, b2=1, b3=2),
        enk_dft_ry=np.array([[-1., 10.5]]) / RYD_TO_EV, sigma_result=result)
    assert any('[-13.20000, +11.40000]' in line and '247 points' in line for line in lines)
    assert 'Coverage status : COMPLETE' in lines


def test_trial_growth_does_not_change_future_sample_shapes():
    grid = np.arange(-2., 5.01, .25)
    trial = extend_sc_omega_grid_ev(grid, [[0., 6.2]], [[True, True]], .25,
                                   role='trial')
    np.testing.assert_array_equal(trial, grid)
    accepted = extend_sc_omega_grid_ev(trial, [[0., 4.9]], [[True, True]], .25)
    np.testing.assert_array_equal(accepted, grid)


def test_trial_classification_does_not_change_accepted_gram(monkeypatch):
    from mixing import acceleration
    from gw.sc_iteration import _refresh_rcrop_metric
    mask = np.ones((1, 2), bool)
    accepted = BandPartition(jnp.ones((1, 2), bool), jnp.ones((1, 2), bool))
    trial = BandPartition(jnp.array([[True, False]]), jnp.array([[True, False]]))
    weights_seen = []
    original = acceleration._solve_crop_alpha_stacked
    def spy(f):
        weights_seen.append(np.asarray(f).copy())
        return original(f)
    monkeypatch.setattr(acceleration, '_solve_crop_alpha_stacked', spy)
    calls = [0]
    def residual(x):
        role = 'trial' if calls[0] % 2 else 'accepted'
        calls[0] += 1
        _refresh_rcrop_metric(mask, trial if role == 'trial' else accepted, 2, role=role)
        return .5 * (jnp.array([[[1., 2.], [3., 4.]]]) - x)
    result = acceleration.rcrop_nojit(residual, jnp.zeros((1, 2, 2), dtype=jnp.complex128),
                                     m=2, maxit=1, tol=0., metric=mask)
    assert weights_seen[0][..., 1, 1].any()
    np.testing.assert_array_equal(mask, True)
    plain = acceleration.rcrop_nojit(
        lambda x: .5 * (jnp.array([[[1., 2.], [3., 4.]]]) - x),
        jnp.zeros((1, 2, 2), dtype=jnp.complex128), m=2, maxit=1, tol=0.)
    np.testing.assert_array_equal(result.x, plain.x)


def test_partition_transition_cannot_report_convergence(monkeypatch):
    from test_sc_state_identity import _identity_call_fixture
    e = np.array([[0., 1., 3.]])
    sc, inputs, state, _ = _identity_call_fixture(monkeypatch, e, np.eye(3)[None])
    state = replace(state, partition=replace(state.partition, changed=True))
    verdict, _ = sc._sc_identity_for_call(inputs, state, e, e, {}, cutoff_ev=.01)
    assert verdict.max_abs_ev == 0.
    assert not verdict.converged


def test_vbm_origin_exposes_conduction_escape_hidden_by_midgap():
    from gw.efermi import resolve_sigma_efermi_ry
    wfn = SimpleNamespace(efermi=1. / RYD_TO_EV, vbm=0.)
    mu, _ = resolve_sigma_efermi_ry('vbm', occupation_state=None, wfn=wfn)
    grid = np.arange(-2., 5.01, .25)
    actual = extend_sc_omega_grid_ev(grid, [[5.4 - mu * RYD_TO_EV]], [[True]], .25)
    wrong = extend_sc_omega_grid_ev(grid, [[5.4 - wfn.efermi * RYD_TO_EV]], [[True]], .25)
    assert actual[-1] >= 5.4 and wrong[-1] == 5.


def test_partition_summary_caps_different_k_rows():
    prot = jnp.tile(jnp.array([[True, False]]), (144, 1))
    partition = BandPartition(prot, prot)
    line = partition.summary(np.tile([[0., 8.]], (144, 1)),
        np.tile([[1, 0]], (144, 1)), band_offset=0, mu_ev=0.)
    assert 'k=[0, 1, 2, 3, 4] +139 more' in line
    assert '\n' not in line
    assert len(line) < 200


def test_constructor_emits_multiplet_and_support_diagnostics():
    log = []
    initial = BandPartition(jnp.array([[True, True]]), jnp.array([[True, True]]))
    build_omega_band_partition(np.array([[0., 5.5]]) / RYD_TO_EV,
        np.array([[0., 4.]]) / RYD_TO_EV, band_offset=0,
        omega_min_abs_ev=-2., omega_max_abs_ev=5., mu_ev=0.,
        previous_partition=initial, print_fn=log.append)
    assert any('no boundary splits a multiplet' in line for line in log)
    assert any('outside the requested window' in line for line in log)


def test_both_external_growth_directions_are_reserved(monkeypatch):
    from test_sigma_box_plan import _plan, _branch
    from gw.sigma_box_plan import _box_contains, make_sigma_box_spec
    for negative in (False, True):
        session = {'external_support_ev': (-20., 20.)}
        _, geometry = _plan(monkeypatch, _branch(negative=negative),
                             fixed_rule_session=session)
        # Both session directions are carried regardless of the branch's
        # causal flag; state/pole padding still expands both box edges.
        for entry in session['rules'].values():
            assert entry['fit']['rule_box'][0] < entry['initial_box'][0]
            assert entry['fit']['rule_box'][1] > entry['initial_box'][1]
        assert geometry['sc_fixed_quadrature']


@pytest.mark.parametrize('restart', [False, True])
def test_restart_active_window_edge_is_checked_before_loop(restart):
    import inspect
    from gw.sc_iteration import run_sc_driver
    from common.band_degeneracy import BandWindowDegeneracyError
    # All expensive inputs deliberately absent: the same edge refusal must
    # run before either fresh/restart path enters the SC machinery.
    kw = {name: None for name, p in inspect.signature(run_sc_driver).parameters.items()
          if p.default is inspect.Parameter.empty}
    from gw.wavefunction_bundle import BandSlices
    kw.update(config=SimpleNamespace(restart=restart),
              wfn=SimpleNamespace(energies=np.array([[[-1., 1., 1., 3.]]]) / RYD_TO_EV),
              band_slices=BandSlices.from_band_edges(0, 0, 0, 2, 2))
    with pytest.raises(BandWindowDegeneracyError, match='nval/ncond'):
        run_sc_driver(**kw)


def test_eqp2_passes_sc_exact_multiplet_tolerance(monkeypatch):
    from common.collectives import single_device_mesh
    from gw import sc_iteration as sc
    from gw.sigma_dispatch import SigmaResult
    from gw.wavefunction_bundle import BandSlices
    from test_fixed_sigma_evsc import _config
    energy = np.array([[-1., 1., 1.0005]]) / RYD_TO_EV
    nb = 3
    z = jnp.zeros((1, nb, nb), dtype=jnp.complex128)
    result = SigmaResult(v_h_kij_ry=z, sigma_x_kij_ry=z, sigma_xc_kij_ry=z,
        sigma_c_omega_kij_ry=jnp.zeros((2, 1, nb, nb), dtype=jnp.complex128),
        omega_grid_ev=np.array([-2., 1.0002]),
        omega_grid_ry=np.array([-2., 1.0002]) / RYD_TO_EV, efermi_dft_ev=0.)
    class Checked(Exception):
        pass
    original = sc.build_omega_band_partition
    def check(*args, **kwargs):
        assert kwargs['degeneracy_tol_ev'] == 1e-4
        partition = original(*args, **kwargs)
        np.testing.assert_array_equal(partition.protected_mask, [[True, True, False]])
        raise Checked
    monkeypatch.setattr(sc, 'build_omega_band_partition', check)
    with pytest.raises(Checked):
        sc.run_fixed_sigma_evsc(result, z, energy, config=_config(),
            meta=SimpleNamespace(nelec=1),
            band_slices=BandSlices.from_band_edges(0, 0, 1, 3, 3),
            wfn=SimpleNamespace(energies=energy[None]),
            mesh_xy=single_device_mesh(), print_fn=lambda _: None)


def test_sign_preserving_pad_contains_the_reserved_support():
    from gw.sigma_box_plan import make_sigma_box_spec, _sc_padded_box_spec, _box_contains
    # The original zero-side edge is far from zero; the reserved external
    # frequencies approach it. Clipping at half the ORIGINAL edge loses
    # most of that reservation even though the result remains sign definite.
    args = dict(name='reserved-negative', states=np.array([1.]),
                pole_stats=[(2., 2., .1, .1)], pole_sign=1., eta_ry=.1)
    original = make_sigma_box_spec(frequencies=np.array([0., 1.]), **args)
    original['sc_support_frequencies'] = np.array([0., 2.8])
    prospective = make_sigma_box_spec(frequencies=np.array([0., 2.8]), **args)
    padded = _sc_padded_box_spec(original, .1)
    assert _box_contains(padded['box'], prospective['box'])
    assert padded['box'][1] < 0.


@pytest.mark.parametrize('pole_sign', [-1., 1.])
def test_prospective_crossing_does_not_reclassify_a_sign_definite_tail(pole_sign):
    from gw.sigma_box_plan import make_sigma_box_spec, _sc_padded_box_spec, _box_contains
    args = dict(name='future-tail', states=np.array([1.]),
                pole_stats=[(2., 2., .1, .1)], pole_sign=pole_sign, eta_ry=.1)
    frequencies = np.array([0., 1.]) if pole_sign == 1. else np.array([-1., 0.])
    # For either causal sign, move only the outer sampled endpoint far
    # enough to cross a currently sign-definite denominator support.
    original = make_sigma_box_spec(frequencies=frequencies, **args)
    assert original['kind'].startswith('sign_definite')
    original['sc_support_frequencies'] = (np.array([0., 4.]) if pole_sign == 1.
                                          else np.array([-4., 0.]))
    padded = _sc_padded_box_spec(original, .1)
    assert padded['kind'] == original['kind']
    assert _box_contains(padded['box'], original['box'])


def test_new_product_name_reuses_an_existing_certificate(monkeypatch):
    from test_sigma_box_plan import _branch, _plan
    session = {}
    _plan(monkeypatch, fixed_rule_session=session)
    renamed = _branch(tag='new product family')
    _, geometry = _plan(monkeypatch, renamed, fixed_rule_session=session)
    assert geometry['sc_fixed_rebuilds_this_iteration'] == 0


def test_relative_window_does_not_borrow_a_crossing_certificate(monkeypatch):
    from gw import sigma_box_plan as boxes
    from test_sigma_box_plan import _fake_rule
    monkeypatch.setattr(boxes, 'build_uniform_rule', _fake_rule)
    args = dict(states=np.array([1.]), pole_stats=[(.5, .5, .1, .1)],
                pole_sign=1., eta_ry=.1)
    crossing = boxes.make_sigma_box_spec(
        name='crossing', frequencies=np.array([0., 2.]), **args)
    relative = boxes.make_sigma_box_spec(
        name='new-relative', frequencies=np.array([0., .2]), **args)
    session = {}
    kwargs = dict(eps=1e-4, reduction_seconds=20., cache_dir=None, session=session)
    boxes._fit_fixed_sc_rules([crossing], .1, **kwargs)
    assert boxes._box_contains(session['rules']['crossing']['fit']['rule_box'],
                               relative['box'])
    fits, _, _ = boxes._fit_fixed_sc_rules([relative], .1, **kwargs)
    assert fits[0]['relative']
    assert session['rebuild_count'] == 1
    assert fits[0]['cache_status'] == 'rebuild:sc-fixed'


def test_initial_sc_zero_side_reservation_keeps_the_causal_sign():
    from gw.sigma_box_plan import make_sigma_box_spec, _sc_padded_box_spec
    for sign in (-1., 1.):
        spec = make_sigma_box_spec(name='future pole channel',
            frequencies=np.array([0., .1]), states=np.array([sign * 2.]),
            pole_stats=[(3., 3., .1, .1)], pole_sign=sign, eta_ry=.1)
        spec['sc_support_frequencies'] = np.array([-.2, .2])
        narrow = _sc_padded_box_spec(spec, .1)
        reserved = _sc_padded_box_spec(spec, .1, reserve_zero_side=True)
        assert reserved['kind'] == narrow['kind']
        if sign > 0:
            assert reserved['box'][1] == -.1
        else:
            assert reserved['box'][0] == .1


@pytest.mark.parametrize('positive_frequencies', [(.2, .5), (0.,)])
def test_outward_reservation_keeps_the_physical_half_inner_edge(
        monkeypatch, positive_frequencies):
    from gw import sigma_box_plan as boxes
    from gw.ppm_windows import _SigmaBranch
    from test_sigma_box_plan import _branch, _summaries, _fake_rule
    seen = {}
    original = boxes._sc_padded_box_spec
    def capture(spec, eta, **kwargs):
        seen[spec['branch'].tag] = spec['sc_support_frequencies']
        return original(spec, eta, **kwargs)
    monkeypatch.setattr(boxes, '_sc_padded_box_spec', capture)
    monkeypatch.setattr(boxes, 'build_uniform_rule', _fake_rule)
    positive = _branch('positive')._replace(
        omega_abs=np.array(positive_frequencies),
        omega_idx=np.arange(2, 2 + len(positive_frequencies)))
    negative = _SigmaBranch(tag='negative', E_A=positive.E_A,
        base_mask_A=positive.base_mask_A, space='cond', neg_omega_half=True,
        omega_abs=np.array([.5, .2]), omega_idx=np.array([0, 1]))
    boxes.plan_sigma_windows(_summaries(), [negative, positive],
        np.array([-.5, -.2, *positive_frequencies]), .1, eps=1e-4,
        reduction_seconds=20., cache_dir=None, print_fn=lambda _: None,
        fixed_rule_session={'external_support_ev': np.array([-.8, .9]) * RYD_TO_EV})
    np.testing.assert_allclose(seen['negative'], [-.8, -.2])
    np.testing.assert_allclose(seen['positive'], [min(positive_frequencies), .9])


def test_near_boundary_cache_hit_reaudits_the_same_nodes(tmp_path):
    from gw.sigma_box_plan import _rule_cache_lookup, _rule_cache_store
    from minimax import build_uniform_rule, box_samples, rule_roundoff_amplification
    box = (1., 2., .1, .2)
    rule = build_uniform_rule(box, 1e-3, time_budget=1.)
    cloud = box_samples(*box, per_unit=8., n_im=48)
    noise = rule_roundoff_amplification(rule.times, rule.weights, cloud, np.abs(cloud))
    assert _rule_cache_store(str(tmp_path), rule, noise) is None
    request = (1., 2., .1, .2 + 5e-12)
    hit, warnings = _rule_cache_lookup(str(tmp_path), request, 1e-3, True,
                                      noise_amplification_cap=100.)
    assert not warnings and hit is not None
    audited, name = hit
    assert name.endswith(':boundary-audit')
    np.testing.assert_array_equal(audited.times, rule.times)
    np.testing.assert_array_equal(audited.weights, rule.weights)
    assert audited.box[3] >= request[3]
    assert audited.sup_error <= 1e-3
    # A stored success stamp is not enough when the expanded box is audited.
    path = next(tmp_path.glob('*.npz'))
    with np.load(path) as data:
        payload = {key: data[key] for key in data.files}
    payload['weights'] = np.zeros_like(payload['weights'])
    np.savez(path, **payload)
    hit, _ = _rule_cache_lookup(str(tmp_path), request, 1e-3, True,
                               noise_amplification_cap=100.)
    assert hit is None

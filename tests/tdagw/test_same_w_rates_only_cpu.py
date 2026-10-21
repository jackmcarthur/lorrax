"""Tiny CPU diagnostics of the same-W rate replay, not a material run."""
from dataclasses import replace

import jax
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

from tests.test_tdagw_full_microscopic_cpu import _native_port_fixture, _reciprocal_banks


@pytest.mark.parametrize('density_case', ['coherent', 'tiny_coherence', 'projector', 'empty', 'filled'])
def test_native_rates_only_matches_default_active_rates_and_excludes_deep_spectator(
        tmp_path, monkeypatch, density_case):
    from common.units import RYD_TO_EV
    from file_io.shared_pole_store import write_general_shared_pole_model
    from ffi import fft
    from gw import ppm_sigma as ps
    import gw.mpa.sigma as native
    from tdagw.collision import branch_operators

    case = _native_port_fixture(monkeypatch)
    mesh, rng, nk, nmu = (case[k] for k in ('mesh', 'rng', 'nk', 'nmu'))
    banks = _reciprocal_banks(rng, nk, nmu)
    width = 4*nmu
    left = np.stack([np.concatenate(list(r), axis=1) for _, r in banks])[:, case['perm'], None]
    right = np.broadcast_to(np.tile(np.eye(nmu), (1, 4)), (nk, nmu, width))[:, case['perm'], None]
    poles = np.stack([np.repeat(p, nmu) for p, _ in banks])
    put = lambda a: jax.device_put(np.ascontiguousarray(a, complex),
                                   NamedSharding(mesh, P(None, 'x', None, 'y')))
    identity = dict(iteration_id='rates', hamiltonian='current', energies='current',
                    occupations='coherent', wavefunctions='all', centroids='all')
    path = str(tmp_path/'same_w.h5')
    header = write_general_shared_pole_model(path, put(left), put(right),
        jax.device_put(poles, NamedSharding(mesh, P())), np.full(nk, width),
        q_span=(0, nk), meta=case['meta'], tables=case['tables'],
        recipe=dict(accuracy='test'), receipts=dict(identity=identity), kmax=width)
    source = ps.shared_pole_window_source(dict(path=path, identity=identity,
        digest=header['digest'], K=header['K'], representation='ordered_complex'),
        case['meta'], efermi_ry=0., linalg='distributed')
    rho = np.broadcast_to(np.array([[.64, .13+.08j], [.13-.08j, .31]]), (nk, 2, 2)).copy()
    if density_case == 'tiny_coherence':
        v = np.array([1., 2e-7j])
        rho[:] = np.outer(v, v.conj())/np.vdot(v, v).real
        assert 0. < abs(rho[0, 0, 1]) < 1e-5
    elif density_case != 'coherent':
        diagonal = dict(projector=[1., 0.], empty=[0., 0.], filled=[1., 1.])[density_case]
        rho = np.broadcast_to(np.diag(diagonal), (nk, 2, 2)).astype(complex).copy()
    else:
        h = case['energy'][:, 1:, None]*np.eye(2)[None]
        assert np.linalg.norm(h@rho-rho@h) > .1  # Right phases cannot move through rho.
    energy = np.asarray(case['wfns'].enk).copy()
    energy[:, 0] = -20.  # Deep frozen occupied line must not widen rate-only boxes.
    window = dict(b_lo=1, U=np.broadcast_to(np.eye(2), (nk, 2, 2)).copy(),
        e=case['energy'][:, 1:], e_frozen=energy,
        f=np.diagonal(rho, axis1=-2, axis2=-1).real, rho=rho)
    config = case['fixture']._config(np.array([-2., 2.]))
    config.sigma.regularization_ev = .14*RYD_TO_EV
    config.sigma.quadrature_eps = 2e-7
    plan = ps.window_sigma_plan(config, window_only=True)
    observed = []
    original = native.ordered_shared_pole_sigma
    def observe(*args, **kwargs):
        observed.append(dict(branches=kwargs['sigma_branches'], brackets=kwargs['band_brackets']))
        return original(*args, **kwargs)
    monkeypatch.setattr(native, 'ordered_shared_pole_sigma', observe)
    with fft.xla_reference():
        default = ps.window_sigma_c(case['wfns'], source, window, plan,
            meta=case['meta'], mesh_xy=mesh, branches=True, pairing='vertex',
            print_fn=lambda *a, **k: None)
        rates = ps.window_sigma_c(case['wfns'], source, window, plan,
            meta=case['meta'], mesh_xy=mesh, branches=True, pairing='vertex', internal='active',
            print_fn=lambda *a, **k: None)
    assert len(observed) == 2
    assert len(observed[0]['brackets']) > 1
    assert observed[1]['brackets'] == ((1, 5),)
    for branch in observed[1]['branches']:
        np.testing.assert_array_equal(np.asarray(branch.band_weight)[:, 0], 0.)
        np.testing.assert_array_equal(np.asarray(branch.base_mask_A)[:, 0], False)
        live_energy = np.asarray(branch.E_A)[np.asarray(branch.base_mask_A)]
        if live_energy.size:
            assert live_energy.min() > -1.
        weight = np.asarray(branch.band_weight)
        expected_h = np.any(abs(rho) != 0., axis=-2)
        expected_p = np.any(abs(np.eye(2)[None]-rho) != 0., axis=-2)
        np.testing.assert_array_equal(weight[:, 1:3], expected_h if branch.space == 'val' else 0.)
        np.testing.assert_array_equal(weight[:, 3:5], expected_p if branch.space == 'cond' else 0.)
        np.testing.assert_array_equal(np.asarray(branch.base_mask_A), weight != 0.)
    for space in ('cond', 'val'):
        np.testing.assert_allclose(rates['active_branch_pair'][space],
            default['active_branch_pair'][space], rtol=3e-5, atol=6e-8)
        np.testing.assert_array_equal(rates['branch_pair'][space], rates['active_branch_pair'][space])
    gd, dd, _ = branch_operators(default['active_branch_pair']['cond'], default['active_branch_pair']['val'])
    gr, dr, _ = branch_operators(rates['active_branch_pair']['cond'], rates['active_branch_pair']['val'])
    np.testing.assert_allclose(gr, gd, rtol=3e-5, atol=6e-8)
    np.testing.assert_allclose(dr, dd, rtol=3e-5, atol=6e-8)
    assert default['native_receipt']['virtual_intermediates'] == 'all'
    assert rates['native_receipt']['virtual_intermediates'] == 'active'
    assert rates['full_rho']['carrier'] == 'rho_and_complement'
    assert rates['full_rho']['live'] == dict(val=int(np.sum(expected_h)), cond=int(np.sum(expected_p)))
    absent = dict(empty='val', filled='cond').get(density_case)
    if absent is not None:
        np.testing.assert_array_equal(rates['active_branch_pair'][absent], 0.)
        assert not any(w['name'].startswith(f'ordered:{absent}:')
                       for w in rates['native_receipt']['windows'])
        present = 'val' if absent == 'cond' else 'cond'
        assert np.linalg.norm(rates['active_branch_pair'][present]) > 1e-4


def test_actual_finite_width_head_has_zero_projector_collision_and_boundary():
    from gw.head_correction import HeadSigmaDiag
    from tdagw.collision import branch_operators, collision

    energy = np.array([[-.8, -.2, .6], [-.7, -.1, .9]])
    rho = np.broadcast_to(np.diag([1., 1., 0.]), (2, 3, 3)).astype(complex)
    eye = np.eye(3)[None]
    ef = .071
    head = HeadSigmaDiag.build(omega_grid_ry=[-2., 2.], enk_ry=energy,
        occupations=np.ones_like(energy), efermi_ry=ef,
        poles_ry=[.63-.037j, 1.17-.061j], residues_ry=[.027, .043],
        cell_volume=2.3, nk_tot=2)
    rp = np.zeros((2, 2, 3, 3), complex)
    rh = np.zeros_like(rp)
    for slot in range(2):
        for a in range(3):
            shell = (np.repeat((energy-ef)[:, a:a+1], 3, axis=1)
                     if slot == 0 else energy-ef)
            hole = head.on_shell(shell)[0].conj()
            particle = replace(head, occupations=np.zeros_like(energy)).on_shell(shell)[0]
            rp[slot, :, a] = (eye-rho)[:, a]*particle
            rh[slot, :, a] = rho[:, a]*hole
    advanced_hole = rh[::-1].conj().swapaxes(-1, -2)
    with jax.enable_x64(True):
        gamma, injection, mh = map(np.asarray, branch_operators(rp, advanced_hole))
        remainder = np.asarray(collision(gamma, injection, rho))
    assert np.linalg.norm(gamma) > 1e-4 and np.linalg.norm(injection) > 1e-4
    np.testing.assert_allclose(remainder, 0., atol=2e-17)
    np.testing.assert_allclose(injection[:, 2:, 2:], 0., atol=2e-17)
    np.testing.assert_allclose((gamma-injection)[:, :2, :2], 0., atol=2e-17)
    np.testing.assert_allclose(mh@rho-rho@mh, 0., atol=2e-17)

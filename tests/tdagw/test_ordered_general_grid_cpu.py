"""Tiny native full-matrix grid oracle; no positive closure is constructed."""
import jax
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

from tests._full_microscopic_oracle import density_vertices, retarded_reference_pairs
from tests.test_tdagw_full_microscopic_cpu import _native_port_fixture, _reciprocal_banks


def test_coherent_general_grid_is_one_native_sweep_with_full_common_energy_matrices(tmp_path, monkeypatch):
    from common.units import RYD_TO_EV
    from ffi import fft
    from file_io.shared_pole_store import write_general_shared_pole_model
    from gw import ppm_sigma as ps
    from gw.head_correction import HeadSigmaDiag
    import gw.mpa.sigma as native

    case = _native_port_fixture(monkeypatch)
    mesh, rng, nk, nmu = (case[k] for k in ('mesh', 'rng', 'nk', 'nmu'))
    banks = _reciprocal_banks(rng, nk, nmu)
    width = 4*nmu
    left = np.stack([np.concatenate(list(r), axis=1) for _, r in banks])[:, case['perm'], None]
    right = np.broadcast_to(np.tile(np.eye(nmu), (1, 4)), (nk, nmu, width))[:, case['perm'], None]
    poles = np.stack([np.repeat(p, nmu) for p, _ in banks])
    put = lambda a: jax.device_put(np.ascontiguousarray(a, complex),
                                  NamedSharding(mesh, P(None, 'x', None, 'y')))
    identity = dict(iteration_id='grid', hamiltonian='current', energies='current',
                    occupations='coherent', wavefunctions='all', centroids='all')
    path = str(tmp_path/'general_grid.h5')
    header = write_general_shared_pole_model(path, put(left), put(right),
        jax.device_put(poles, NamedSharding(mesh, P())), np.full(nk, width),
        q_span=(0, nk), meta=case['meta'], tables=case['tables'],
        recipe=dict(accuracy='test'), receipts=dict(identity=identity), kmax=width)
    ef = .071
    source = ps.shared_pole_window_source(dict(path=path, identity=identity,
        digest=header['digest'], K=header['K'], representation='ordered_complex'),
        case['meta'], efermi_ry=ef, linalg='distributed')
    rotation, density = [], []
    for _ in range(nk):
        u, _ = np.linalg.qr(rng.normal(size=(2, 2))+1j*rng.normal(size=(2, 2)))
        v, _ = np.linalg.qr(rng.normal(size=(2, 2))+1j*rng.normal(size=(2, 2)))
        rotation.append(u)
        density.append(v@np.diag([.19, .83])@v.conj().T)
    rotation, density = np.asarray(rotation), np.asarray(density)
    energy = case['energy'].copy()
    energy[:, 1:] += np.asarray([.03, -.02])[None]
    head = HeadSigmaDiag.build(omega_grid_ry=[-2., 2.], enk_ry=energy+3.,
        occupations=np.zeros_like(energy), efermi_ry=-1.,
        poles_ry=[.63-.037j, 1.17-.061j], residues_ry=[.027, .043],
        cell_volume=2.3, nk_tot=nk)
    config = case['fixture']._config(np.array([-2., 2.]))
    config.sigma.regularization_ev = .14*RYD_TO_EV
    config.sigma.quadrature_eps = 2e-7
    plan = ps.window_sigma_plan(config, head=head, window_only=True)
    padded = np.asarray(case['wfns'].enk).copy()
    padded[:, :3] = energy
    window = dict(b_lo=1, U=rotation, e=energy[:, 1:], e_frozen=padded,
        f=np.diagonal(density, axis1=-2, axis2=-1).real, rho=density)
    frequencies = np.asarray([-.35, 0., .72])
    observed = []
    original = native.ordered_shared_pole_sigma
    def observe(*args, **kwargs):
        observed.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(native, 'ordered_shared_pole_sigma', observe)
    with fft.xla_reference():
        got = ps.window_sigma_c(case['wfns'], source, window, plan,
            meta=case['meta'], mesh_xy=mesh, omega_ry=frequencies,
            branches=True, pairing='vertex', internal='active', print_fn=lambda *a, **k: None)
    assert len(observed) == 1 and observed[0]['request'] is None
    np.testing.assert_array_equal(observed[0]['omega_grid_ry'], frequencies)
    np.testing.assert_array_equal(window['e'], energy[:, 1:])
    assert got['pair'] is None and got['request'] is None and got['branch_pair'] is None
    assert got['native_receipt']['output'] == 'grid_only_full_matrices'
    assert got['native_receipt']['branch_convention'] == 'cond_and_val_both_retarded'

    rotated = case['psi'].copy()
    rotated[:, 1:] = np.einsum('kmsu,kmi->kisu', case['psi'][:, 1:], rotation)
    expected = {space: np.zeros((3, nk, 2, 2), complex) for space in ('cond', 'val')}
    for q, (p, residue) in enumerate(banks):
        partner = (np.arange(nk)-q) % nk
        vertex = density_vertices(rotated, partner, [1, 2])[:, :, 1:]
        for i, omega in enumerate(frequencies):
            # All matrix entries use the SAME frequency; no column-shell
            # surrogate. Literal m,n,port sums retain coherent internal rho.
            rp, rh = retarded_reference_pairs(vertex, energy[partner, 1:]-ef,
                density[partner], np.full((nk, 2), omega), p, residue, .14)
            expected['cond'][i] += rp[1]/nk
            expected['val'][i] += rh[1]/nk
    pref = np.asarray(head.residues_ry)/(head.cell_volume*nk)
    for i, omega in enumerate(frequencies):
        delta = omega-(energy[:, 1:]-ef)
        part = np.sum(pref[:, None, None]/(delta[None]-head.poles_ry[:, None, None]), axis=0)
        hole = np.sum(pref.conj()[:, None, None]/(delta[None]+head.poles_ry.conj()[:, None, None]), axis=0)
        expected['cond'][i] += (np.eye(2)[None]-density)*part[:, None, :]
        expected['val'][i] += density*hole[:, None, :]
    for space in expected:
        np.testing.assert_allclose(got['retarded_omega'][space], expected[space], rtol=1e-5, atol=3e-8)
        np.testing.assert_array_equal(got['active_retarded_omega'][space], got['retarded_omega'][space])
        rate = 1j*(got['retarded_omega'][space]-got['retarded_omega'][space].conj().swapaxes(-1, -2))
        np.testing.assert_allclose(rate, rate.conj().swapaxes(-1, -2), atol=1e-16)
    total = expected['cond']+expected['val']
    np.testing.assert_allclose(got['omega'], rotation[None]@total@rotation.conj().swapaxes(-1, -2)[None],
                               rtol=1e-5, atol=3e-8)


@pytest.mark.parametrize('shell_request,omega', [(None, None), (object(), np.array([.1]))])
def test_ordered_executor_refuses_ambiguous_frequency_route_before_model_io(shell_request, omega):
    from gw.mpa.sigma import ordered_shared_pole_sigma
    with pytest.raises(ValueError, match='supply one request or omega grid'):
        ordered_shared_pole_sigma(None, None, None, None, request=shell_request, omega_grid_ry=omega,
            sigma_branches=(), regularization_width_ry=.1, quadrature_eps=1e-7, projection_bands=(0, 2))

"""Independent dense checks of the micro-reducible ladder-head contract.

CPU only, no native FFI. A small transition Hamiltonian is inverted directly;
the production head solver, mini-BZ reducer and scalar-pole Sigma consumer
must reproduce it. This does not certify a material's band convergence.
"""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from bse.head_resolvent import HeadOperator, head_prefactor, solve_head_tensor
from bse.bse_ring_comm import make_bse_shardings

jax.config.update("jax_enable_x64", True)


def _system(ladder=True, ring=True):
    rng = np.random.default_rng(142)
    delta = np.array([[.6, .4], [1.1, .9]]).reshape(-1)
    d = .15 * rng.normal(size=(4, 3))
    rho = .15 * rng.normal(size=(4, 2))
    v = rho @ np.diag([2., .7]) @ rho.T if ring else np.zeros((4, 4))
    r = rng.normal(size=(4, 4))
    direct = .012 * r @ r.T if ladder else np.zeros((4, 4))
    a, b = np.diag(delta) + v - direct, v - .5 * direct
    h = np.block([[a, b], [-b, -a]])
    return delta, d, h


def _xi(h, d, z, pref):
    seed = np.concatenate([d, -d])
    x = np.linalg.solve(z * np.eye(len(h)) - h, seed)
    return pref * d.T @ (x[:len(d)] + x[len(d):])


def test_head_engine_matches_dense_free_rpa_and_ladder():
    """The [d;-d] seed, X+Y readout and spin/volume prefactor match dense inverses."""
    from common.chi_from_dipole import compute_S_omega_cv
    mesh = Mesh(np.array(jax.devices('cpu')[:1]).reshape(1, 1), ('x', 'y'))
    sh = make_bse_shardings(mesh)
    z = np.array([0j, .4j, .3 + .6j, -.3 - .6j])
    pref = head_prefactor(60., 1, 1, 2)
    for ladder, ring in ((False, False), (False, True), (True, True)):
        delta, d, h = _system(ladder, ring)

        @jax.jit
        def matvec(x):
            rows = x.transpose(1, 0, 2, 3, 4).reshape(x.shape[1], 8)
            return (rows @ jnp.asarray(h.T)).reshape(x.shape[1], 2, 2, 2, 1).transpose(1, 0, 2, 3, 4)

        op = HeadOperator(matvec, jnp.asarray(np.diag(h).reshape(2, 1, 2, 2, 1)),
                          sh, lambda data: (), ('dense-head', ladder, ring))
        result = solve_head_tensor(op, {}, d.T.reshape(3, 2, 2, 1), z,
                                   pref=pref, max_iter=8, tol=1e-12)
        expected = np.stack([_xi(h, d, point, pref) for point in z])
        np.testing.assert_allclose(result.xi, expected, rtol=2e-11, atol=1e-13)
        assert np.max(result.resids) < 1e-11
        np.testing.assert_allclose(result.xi[3], result.xi[2].T, rtol=2e-11, atol=1e-13)
        if not ring:
            velocity = (d * delta[:, None]).T.reshape(3, 1, 2, 2)
            independent = compute_S_omega_cv(jnp.asarray(velocity),
                jnp.asarray(delta.reshape(1, 2, 2)), 60., 1, 1, 2, jnp.asarray(z))
            np.testing.assert_allclose(result.xi, independent, rtol=2e-11, atol=1e-13)


def _dense_macro_w(h, d, z, pref, q, v):
    """Restore the omitted macroscopic channel by a full matrix inverse, not scalar Dyson."""
    f = np.sqrt(pref) * (q @ d.T)
    seed = np.concatenate([f, -f], axis=1)
    readout = np.concatenate([f, f], axis=1)
    full = h[None] + v[:, None, None] * seed[:, :, None] * readout[:, None, :]
    x = np.linalg.solve(z * np.eye(len(h))[None] - full, seed[:, :, None])[..., 0]
    response = np.einsum('qi,qi->q', readout, x)
    return v + v * v * response


def test_micro_head_restores_macro_v_once_in_bulk_and_slab(monkeypatch):
    """An anisotropic cell average equals dense restoration; a second fold differs."""
    from gw import vcoul as adapter
    from gw.qsgw_head import head_samples_from_s
    from vcoul import CoulombGeometry, get_kernel
    from vcoul.minibz import _sample_q0_minibz_qpoints, slab_minibz_photon_cubature
    original = adapter.compute_q0_averages_screened

    def small_draw(wfn, meta, tensors, **kwargs):
        return original(wfn, meta, tensors, nsamples=256, qmc_reps=1, **kwargs)

    monkeypatch.setattr(adapter, 'compute_q0_averages_screened', small_draw)
    _, d, h = _system()
    z = np.array([0j, .4j, .3 + .6j])
    pref = head_prefactor(60., 1, 1, 2)
    xi = np.stack([_xi(h, d, point, pref) for point in z])
    assert np.linalg.eigvalsh(xi[0]).max() < 0
    config = SimpleNamespace(head=SimpleNamespace(vhead=None, whead_0freq=None,
        whead_imfreq=None, head_minibz_average=False))
    for dim, bvec, grid in ((3, [[.5, .07, .03], [0, .8, .04], [0, 0, .6]], (4, 3, 5)),
                            (2, [[.5, .07, 0], [0, .8, 0], [0, 0, .2]], (6, 6, 1))):
        geometry = CoulombGeometry(np.asarray(bvec), 60.)
        wfn = SimpleNamespace(blat=1., bvec=geometry.bvec, cell_volume=60.)
        meta = SimpleNamespace(sys_dim=dim, nkx=grid[0], nky=grid[1], nkz=grid[2])
        rows = head_samples_from_s(xi, z, wfn=wfn, meta=meta, config=config,
                                   response_kind='micro_reducible', source_prefix='dense_ladder')
        if dim == 3:
            q = np.asarray(_sample_q0_minibz_qpoints(geometry, grid, nsamples=256,
                           method='sobol', qmc_reps=1, analytic_sphere=False, is_2d=False)[0])
            v = 8 * np.pi / np.sum(q*q, axis=1)
            weight = np.ones(len(q))
        else:
            chunk = slab_minibz_photon_cubature(get_kernel(2), geometry, grid).chunks[-1]
            n = int(chunk.physical_count)
            q, v = np.asarray(chunk.q_cart[:n]), np.asarray(chunk.D_raw[:n, 0, 0])
            weight = np.asarray(chunk.sample_weight[:n])
        for point, tensor, sample in zip(z, xi, rows):
            dense = _dense_macro_w(h, d, point, pref, q, v)
            expected = np.sum(weight*dense)/np.sum(weight)
            np.testing.assert_allclose(sample.wcoul0, expected, rtol=2e-11, atol=1e-10)
            assert sample.response_kind.value == 'micro_reducible'
            assert sample.source.startswith('dense_ladder')
            # Reapplying macro-v to an already macro-reducible response is observably wrong.
            scalar = np.einsum('qi,ij,qj->q', q, tensor, q)
            twice = v/(1-v*scalar/(1-v*scalar))
            assert abs(np.sum(weight*twice)/np.sum(weight) - expected) > 1e-4
        assert (rows[0].wcoul0 - rows[0].vc0).real < 0


def test_scalar_head_fit_preserves_static_moment_and_sigma_slope():
    """A known two-pole head checks held-out frequencies and the actual Sigma consumer."""
    from gw.head_correction import HeadSample, compute_complex_pole_head_sigma_diag
    from gw.mpa.model import fit_head_samples
    from gw.qsgw_head import head_moment_m1, head_quadrature_rule
    poles, residues = np.array([.7, 1.8]), np.array([.08, .15])
    exact = lambda z: np.sum(2*poles*residues/(z*z-poles*poles))
    z = np.array([0j, .2+.25j, .8+.5j, 2.+.7j])
    xi, _ = head_quadrature_rule()
    all_z = np.r_[z, 1j*xi]
    samples = tuple(HeadSample(vc0=4., wcoul0=4.+exact(point), source='dense',
        omega=point) for point in all_z)
    expected_m1 = np.sum(residues/poles**2)
    moment = head_moment_m1(samples, all_z, metal=False)
    np.testing.assert_allclose(moment, expected_m1, rtol=1e-8)
    head = fit_head_samples(samples[:len(z)], z, 2, model='bse_resolvent_micro_loewner',
                            solve='loewner', moment_m1=moment)
    p, b = head['Omega_p'], head['B_p']
    for point in (0j, .12j, .45+.4j, 1.3+.6j):
        np.testing.assert_allclose(np.sum(2*p*b/(point*point-p*p)), exact(point), rtol=2e-8, atol=1e-10)
    np.testing.assert_allclose(np.sum(b/p**2), expected_m1, rtol=1e-8)
    # The independent high-frequency coefficient is not M1.
    np.testing.assert_allclose(np.sum(2*p*b), np.sum(2*poles*residues), rtol=2e-8)
    energy, occ, step = np.array([[-.1, .2]]), np.array([[1., 0.]]), 1e-4
    grid = np.r_[energy[0, 0] + np.array([-step, 0, step]),
                 energy[0, 1] + np.array([-step, 0, step])]
    sigma = compute_complex_pole_head_sigma_diag(omega_grid_ry=grid, enk_ry=energy,
        efermi_ry=0., occupations=occ, poles_ry=p, residues_ry=b, cell_volume=60., nk_tot=5)
    on_shell = np.array([sigma[1, 0, 0], sigma[4, 0, 1]])
    slope = np.array([(sigma[2, 0, 0]-sigma[0, 0, 0]),
                      (sigma[5, 0, 1]-sigma[3, 0, 1])])/(2*step)
    np.testing.assert_allclose(on_shell, (.5-occ[0])*exact(0j)/(60*5), rtol=2e-8)
    np.testing.assert_allclose(slope, -expected_m1/(60*5), rtol=1e-7)

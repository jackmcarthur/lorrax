"""Actual general store/FFT -> same-sweep virtual-shell tail, CPU4 only."""
import runtime  # noqa: F401
import jax
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from tests.test_tdagw_full_microscopic_cpu import _native_port_fixture, _reciprocal_banks
from tests._full_microscopic_oracle import (
    branch_coefficients, density_vertices, retarded_reference_pairs)


def test_native_same_sweep_tail_keeps_complex_columns_and_active_rates(tmp_path, monkeypatch):
    from common.collectives import gather_to_host
    from common.units import RYD_TO_EV
    from file_io.shared_pole_store import write_general_shared_pole_model
    from gw.shared_pole_recipe import CapacityLedger
    from gw.band_extrapolation import BandLadder, fit_band_extrapolation_spectral
    from gw import ppm_sigma as ps
    from tdagw.collision import branch_operators
    from ffi import fft

    case = _native_port_fixture(monkeypatch)
    mesh, fixture, rng = case['mesh'], case['fixture'], case['rng']
    nk, nb, ns, nmu, active = 3, 8, 2, 4, slice(1, 3)
    monkeypatch.setattr(fixture, 'NBF', nb)
    monkeypatch.setattr(fixture, 'NB_PAD', nb)
    psi = .23*(rng.normal(size=(nk, nb, ns, nmu))+1j*rng.normal(size=(nk, nb, ns, nmu)))
    energy = np.array([[-.91, -.24, .53, 1.1, 1.5, 1.9, 2.3, 2.7],
                       [-.82, -.17, .68, 1.2, 1.6, 2.0, 2.4, 2.8],
                       [-1.03, -.31, .71, 1.3, 1.7, 2.1, 2.5, 2.9]])
    occ = np.broadcast_to([1., 1., 0., 0., 0., 0., 0., 0.], (nk, nb)).copy()
    wfns, meta = fixture._gw_bundle(mesh, psi[..., case['perm']], energy, occ,
                                    case['wfns'].green_parent.plan)
    meta.mu_basis = case['meta'].mu_basis
    meta.shared_pole_capacity = CapacityLedger(meta, mesh_xy=mesh, device_budget_bytes=2**30)
    meta.shared_pole_capacity.live_stages = ()
    banks = _reciprocal_banks(rng, nk, nmu)
    width = 4*nmu
    left = np.stack([np.concatenate(list(residue), axis=1) for _, residue in banks])[:, case['perm'], None]
    right = np.broadcast_to(np.tile(np.eye(nmu), (1, 4)), (nk, nmu, width))[:, case['perm'], None]
    poles = np.stack([np.repeat(p, nmu) for p, _ in banks])
    put = lambda a: jax.device_put(np.ascontiguousarray(a, complex),
                                  NamedSharding(mesh, P(None, 'x', None, 'y')))
    identity = dict(iteration_id='tail-oracle', hamiltonian='current-H', energies='current-E',
                    occupations='coherent-rho', wavefunctions='full-psi', centroids='full-mu')
    path = str(tmp_path/'ordered.h5')
    header = write_general_shared_pole_model(path, put(left), put(right),
        jax.device_put(poles, NamedSharding(mesh, P())), np.full(nk, width),
        q_span=(0, nk), meta=meta, tables=case['tables'], recipe=dict(accuracy='test'),
        receipts=dict(identity=identity), kmax=width)
    handle = dict(path=path, identity=identity, digest=header['digest'], K=header['K'],
                  representation='ordered_complex')
    source = ps.shared_pole_window_source(handle, meta, efermi_ry=0., linalg='distributed')
    rotations, density = [], []
    for _ in range(nk):
        u, _ = np.linalg.qr(rng.normal(size=(2, 2))+1j*rng.normal(size=(2, 2)))
        r, _ = np.linalg.qr(rng.normal(size=(2, 2))+1j*rng.normal(size=(2, 2)))
        rotations.append(u)
        density.append(r@np.diag([.19, .83])@r.conj().T)
    rotations, density = np.asarray(rotations), np.asarray(density)
    # A native equilibrium fit with a known denominator-shell shape. Only
    # its shape/ladder is reused, not its amplitudes or per-band ratios.
    ladder = BandLadder(e_dft_ev=energy.T*RYD_TO_EV,
        e_weyl_ev=np.arange(3.2, 5.6, .3)*RYD_TO_EV, w_k=np.full(nk, 1/nk),
        e0_ev=0., n0=0., c_ev=1., r2=1., estar_ev=10., n_dft=8, n_target=16,
        fit_window=(4, 8))
    e = energy[:, active]*RYD_TO_EV
    def moment(lo, hi):
        nodes, weights = ladder._terms(lo, hi)
        return np.sum(weights/(nodes-e[..., None]+2.)**3, axis=-1)
    base = np.full(e.shape, .2+.4j)
    fit = fit_band_extrapolation_spectral([4, 6, 8],
        np.stack([base, base+moment(4, 6), base+moment(4, 8)]), ladder, e_state_ev=e)
    assert fit.n_failed == 0 and abs(fit.omega_ev-2.) < 1e-12
    reference = dict(fit=fit, input_frame='accepted-input', ladder_shift_ev=0.)
    config = fixture._config(np.array([-3., 4.]))
    config.sigma.regularization_ev = .14*RYD_TO_EV
    config.sigma.quadrature_eps = 2e-7
    plan = ps.window_sigma_plan(config, window_only=True)
    plan['spectral_tail'] = reference
    observed = {}
    import gw.mpa.sigma as native
    original = native.ordered_shared_pole_sigma
    def observe(*args, **kwargs):
        observed['calls'] = observed.get('calls', 0)+1
        result, receipt = original(*args, **kwargs)
        observed['matrix_counts'] = kwargs['band_matrix_counts']
        observed['labels'] = kwargs['band_counts']
        observed['points'] = {space: {i: np.asarray(gather_to_host(
            result.branches[space].point(i).pair))[..., :2, :2]
            for i in kwargs['band_matrix_counts']} for space in ('cond', 'val')}
        return result, receipt
    monkeypatch.setattr(native, 'ordered_shared_pole_sigma', observe)
    window = dict(b_lo=1, U=rotations, e=energy[:, active],
        f=np.diagonal(density, axis1=-2, axis2=-1).real, rho=density)
    with fft.xla_reference():
        got = ps.window_sigma_c(wfns, source, window, plan, meta=meta,
            mesh_xy=mesh, branches=True, pairing='vertex', print_fn=lambda *a, **kw: None)
    assert observed['matrix_counts'] == (0, 2, 4)
    assert observed['labels'] == (2, 3, 4, 6, 8)
    assert observed['calls'] == 1
    rotated = psi.copy()
    rotated[:, active] = np.einsum('kmsu,kmi->kisu', psi[:, active], rotations)
    rho = np.zeros((nk, nb, nb), complex)
    rho[:, 0, 0], rho[:, active, active] = 1., density
    totals = {cut: {name: np.zeros((2, nk, 2, 2), complex) for name in ('cond', 'val')}
              for cut in ('active', 4, 8)}
    for q, (p, residue) in enumerate(banks):
        partner = (np.arange(nk)-q) % nk
        vertex = density_vertices(rotated, partner, [1, 2])
        for cut in totals:
            if cut == 'active':
                v, en, r = vertex[:, :, active], energy[partner, active], rho[partner][:, active, active]
            else:
                v, en, r = vertex[:, :, :cut], energy[partner, :cut], rho[partner, :cut, :cut]
            rp, rh = retarded_reference_pairs(v, en, r, energy[:, active], p, residue, .14)
            totals[cut]['cond'] += rp/nk
            totals[cut]['val'] += rh/nk
    ratio = moment(8, 16)/moment(4, 8)
    complex_tail_size = 0.
    for name in ('cond', 'val'):
        for i, cut in ((0, 'active'), (2, 4), (4, 8)):
            np.testing.assert_allclose(observed['points'][name][i], totals[cut][name], rtol=1e-5, atol=3e-8)
        expected = totals[8][name].copy()
        for slot in range(2):
            for k in range(nk):
                for a in range(2):
                    for b in range(2):
                        expected[slot, k, a, b] += (totals[8][name]-totals[4][name])[slot, k, a, b]*ratio[k, a if slot == 0 else b]
        np.testing.assert_allclose(got['retarded_pair'][name], expected, rtol=1e-5, atol=3e-8)
        complex_tail_size = max(complex_tail_size, float(np.max(abs((expected-totals[8][name]).imag))))
    assert complex_tail_size > 1e-6
    expected_rates = branch_coefficients(totals['active']['cond'], totals['active']['val'])
    gamma, injection, _ = branch_operators(got['active_branch_pair']['cond'], got['active_branch_pair']['val'])
    np.testing.assert_allclose(gamma, expected_rates['gamma'], rtol=1e-5, atol=3e-8)
    np.testing.assert_allclose(injection, expected_rates['injection'], rtol=1e-5, atol=3e-8)
    assert got['native_receipt']['spectral_tail']['cond']['counts'] == [4, 6, 8]

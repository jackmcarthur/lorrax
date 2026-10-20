"""Frozen analytic head on the actual general-store/native Sigma CPU route."""
import numpy as np

from tests._full_microscopic_oracle import (
    advanced_hole_compatibility_pair, branch_coefficients, density_vertices,
    retarded_reference_pairs,
)
from tests.test_tdagw_full_microscopic_cpu import _native_port_fixture, _reciprocal_banks


def test_native_general_head_uses_current_columns_once_in_active_and_full(tmp_path, monkeypatch):
    """Independent pole sums include body, coherent head and all virtual bands."""
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.units import RYD_TO_EV
    from ffi import fft as F
    from file_io.shared_pole_store import write_general_shared_pole_model
    from gw import ppm_sigma as ps
    from gw.head_correction import HeadSigmaDiag
    from tdagw.collision import branch_operators

    case = _native_port_fixture(monkeypatch)
    rng, mesh, nk, nb, nmu, psi, energy, perm, wfns, meta, tables, fixture = (
        case[key] for key in ('rng', 'mesh', 'nk', 'nb', 'nmu', 'psi', 'energy',
                             'perm', 'wfns', 'meta', 'tables', 'fixture'))
    banks = _reciprocal_banks(rng, nk, nmu)
    width = len(banks[0][0])*nmu
    left = np.stack([np.concatenate(list(r), axis=1) for _, r in banks])[:, perm, None, :]
    right = np.broadcast_to(np.tile(np.eye(nmu), (1, 4)), (nk, nmu, width))[:, perm, None, :]
    poles = np.stack([np.repeat(p, nmu) for p, _ in banks])
    put = lambda a: jax.device_put(np.ascontiguousarray(a, dtype=complex),
                                  NamedSharding(mesh, P(None, 'x', None, 'y')))
    identity = dict(iteration_id='head', hamiltonian='H', energies='E',
                    occupations='rho', wavefunctions='psi', centroids='mu')
    path = str(tmp_path/'general_head.h5')
    header = write_general_shared_pole_model(path, put(left), put(right),
        jax.device_put(poles, NamedSharding(mesh, P())), np.full(nk, width),
        q_span=(0, nk), meta=meta, tables=tables, recipe=dict(accuracy='test'),
        receipts=dict(identity=identity), kmax=width)
    source = ps.shared_pole_window_source(dict(path=path, identity=identity,
        digest=header['digest'], K=header['K'], representation='ordered_complex'),
        meta, efermi_ry=.071, linalg='distributed')
    rotation, rho_a = [], []
    for _ in range(nk):
        u, _ = np.linalg.qr(rng.normal(size=(2, 2))+1j*rng.normal(size=(2, 2)))
        v, _ = np.linalg.qr(rng.normal(size=(2, 2))+1j*rng.normal(size=(2, 2)))
        rotation.append(u)
        rho_a.append(v@np.diag([.19, .83])@v.conj().T)
    rotation, rho_a = np.asarray(rotation), np.asarray(rho_a)
    # Current electronic energies differ from both the source WFN and frozen
    # head tables. Only the source head poles/residues remain frozen.
    current = energy + np.array([.031, -.047, .083])[None]
    head = HeadSigmaDiag.build(omega_grid_ry=np.array([-2., 2.]),
        enk_ry=energy+3., occupations=np.zeros_like(energy), efermi_ry=-1.2,
        poles_ry=np.array([.63-.037j, 1.17-.061j]),
        residues_ry=np.array([.027, .043]), cell_volume=2.3, nk_tot=nk)
    config = fixture._config(np.array([-2., 2.]))
    config.sigma.regularization_ev = .14*RYD_TO_EV
    config.sigma.quadrature_eps = 2e-7
    plan = ps.window_sigma_plan(config, head=head, window_only=True)
    padded_current = np.asarray(wfns.enk).copy()
    padded_current[:, :nb] = current
    window = dict(b_lo=1, U=rotation, e=current[:, 1:], e_frozen=padded_current,
        f=np.diagonal(rho_a, axis1=-2, axis2=-1).real, rho=rho_a)
    with F.xla_reference():
        got = ps.window_sigma_c(wfns, source, window, plan, meta=meta,
            mesh_xy=mesh, branches=True, pairing='vertex', print_fn=lambda *a, **k: None)

    rotated = psi.copy()
    rotated[:, 1:] = np.einsum('kmsu,kmi->kisu', psi[:, 1:], rotation)
    rho = np.zeros((nk, nb, nb), complex)
    rho[:, 0, 0], rho[:, 1:, 1:] = 1., rho_a
    expected = {count: {space: np.zeros((2, nk, 2, 2), complex)
                       for space in ('cond', 'val')} for count in ('active', 'full')}
    for q, (p, r) in enumerate(banks):
        partner = np.array([(k-q) % nk for k in range(nk)])
        vertex = density_vertices(rotated, partner, [1, 2])
        for count, bands in (('active', slice(1, 3)), ('full', slice(None))):
            rp, rh = retarded_reference_pairs(vertex[:, :, bands], current[partner, bands],
                rho[partner, bands, :][:, :, bands], current[:, 1:], p, r, .14)
            expected[count]['cond'] += rp/nk
            expected[count]['val'] += rh/nk

    # Literal scalar denominators, independent of HeadSigmaDiag.on_shell.
    # Retarded hole poles are -conj(p); its reference residues are conj(R).
    hp = {space: np.zeros((2, nk, 2, 2), complex) for space in ('cond', 'val')}
    pref = head.residues_ry/(head.cell_volume*nk)
    for slot in range(2):
        for a in range(2):
            external = current[:, a+1:a+2] if slot == 0 else current[:, 1:]
            delta = external-current[:, 1:]
            hp['cond'][slot, :, a] = (np.eye(2)[None]-rho_a)[:, a]*np.sum(
                pref[:, None, None]/(delta[None]-head.poles_ry[:, None, None]), axis=0)
            hp['val'][slot, :, a] = rho_a[:, a]*np.sum(
                pref.conj()[:, None, None]/(delta[None]+head.poles_ry.conj()[:, None, None]), axis=0)
    assert np.linalg.norm(hp['val'][0]-hp['val'][1]) > 1e-4
    for count, output in (('active', 'active_branch_pair'), ('full', 'branch_pair')):
        for space in ('cond', 'val'):
            expected[count][space] += hp[space]
        pair = expected[count]
        np.testing.assert_allclose(got[output]['cond'], pair['cond'], rtol=1e-5, atol=3e-8)
        np.testing.assert_allclose(got[output]['val'], advanced_hole_compatibility_pair(pair['val']),
                                   rtol=1e-5, atol=3e-8)
        for actual, target in zip(branch_operators(got[output]['cond'], got[output]['val']),
                                  (branch_coefficients(pair['cond'], pair['val'])[key]
                                   for key in ('gamma', 'injection', 'm_h'))):
            np.testing.assert_allclose(actual, target, rtol=1e-5, atol=3e-8)
        np.testing.assert_array_equal(got[output]['static'], 0.)
    np.testing.assert_allclose(got['pair'], expected['full']['cond']+expected['full']['val'],
                               rtol=1e-5, atol=3e-8)
    # Body/head assembly adds no HF or static-W counterterm.
    np.testing.assert_array_equal(got['static'], 0.)
    assert 'static_limit' not in got
    receipt = got['native_receipt']
    np.testing.assert_array_equal(receipt['head_pole_imaginary_parts_ry'], head.poles_ry.imag)
    assert receipt['bulk_evaluation_eta_ry'] == .14
    assert receipt['head_width_convention'] == 'frozen_source_poles_no_added_bulk_eta'

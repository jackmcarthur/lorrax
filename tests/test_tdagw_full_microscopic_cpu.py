"""Independent full-spatial-port, full-coherence, general-pole CPU gates."""
import numpy as np
import pytest

from tests._full_microscopic_oracle import (
    advanced_hole_compatibility_pair, branch_coefficients, dagger,
    density_vertices, ordered_pr, ordered_pr_time, retarded_reference_pairs,
)
from tdagw import keldysh_rational as kr
from tdagw import snapshot_collision as sc
from tdagw import snapshot_screening as ss


def _case(seed=723):
    rng = np.random.default_rng(seed)
    nk, nb, ns, nmu = 3, 3, 2, 3
    psi = .23 * (rng.normal(size=(nk, nb, ns, nmu))
                 + 1j * rng.normal(size=(nk, nb, ns, nmu)))
    energy = np.array([[-.91, -.24, .53], [-.82, -.17, .68], [-1.03, -.31, .71]])
    rho = np.zeros((nk, nb, nb), complex)
    for k in range(nk):
        u, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
        rho[k, 0, 0] = 1.
        rho[k, 1:, 1:] = u @ np.diag([.19, .83]) @ u.conj().T
    partners = np.array([[(k-q) % nk for k in range(nk)] for q in range(nk)])
    return rng, psi, energy, rho, partners


def _reciprocal_banks(rng, nq, nmu):
    positive_poles = np.array([.41+.017j, .93-.061j])
    positive = .07 * (rng.normal(size=(nq, 2, nmu, nmu))
                      + 1j * rng.normal(size=(nq, 2, nmu, nmu)))
    banks = []
    for q in range(nq):
        qm = (-q) % nq
        poles = np.concatenate((positive_poles, -positive_poles.conjugate()))
        residues = np.concatenate((positive[q], -positive[qm].conjugate()))
        banks.append((poles, residues))
    return banks


def test_full_port_pr_literal_indices_and_coherent_endpoint_order():
    _, psi, energy, rho, partners = _case()
    sites = np.array([-.37+.23j, .19+.31j, .87+.46j])
    gamma, spin = .043, 1.7
    for q, partner in enumerate(partners):
        exact = ordered_pr(psi, energy, rho, partner, sites, gamma, spin)
        pgpl = ss.polarization_samples(psi, energy, rho, np.eye(psi.shape[-1]),
            partner, sites, adiabatic_ry=gamma, spin_weight=spin, band_block=2)
        np.testing.assert_allclose(pgpl[1]-pgpl[0], exact, rtol=3e-13, atol=3e-15)
        pinched = ordered_pr(psi, energy, rho*np.eye(3), partner, sites, gamma, spin)
        assert np.linalg.norm(exact-pinched) > 1e-3
        # q/-q crossing conjugates ports but DOES NOT transpose them.
        reverse = ordered_pr(psi, energy, rho, partners[(-q) % 3],
                             -sites.conjugate(), gamma, spin)
        np.testing.assert_allclose(reverse, exact.conjugate(), rtol=3e-13, atol=3e-15)


def test_full_density_fft_stream_matches_literal_transitions_and_right_endpoint():
    import jax
    from jax.sharding import Mesh
    from ffi import fft as F
    from gw.greens_function_kernel import density_left_endpoint
    from gw.w_isdf import full_density_response_stream

    jax.config.update('jax_enable_x64', True)
    mesh = Mesh(np.asarray(jax.devices('cpu')[:1]).reshape(1, 1), ('x', 'y'))
    _, psi, energy, rho, partners = _case()
    nk, nb, ns, nmu = psi.shape
    mun = np.transpose(psi, (0, 2, 3, 1))
    # Native psi_nmu is the ket face, not the conjugated Green right face:
    # build_G owns that conjugation. Preserve this independently below.
    nmu_face = psi
    endpoint = density_left_endpoint(mesh, mun.shape)(mun, rho)
    expected_endpoint = np.einsum('ksma,kab->ksmb', mun, rho)
    np.testing.assert_allclose(endpoint, expected_endpoint, rtol=2e-13, atol=2e-15)
    times = np.array([.11+.17j, .28-.23j, .04+.51j])
    with F.xla_reference():
        stream = full_density_response_stream(mesh, (3, 1, 1),
            (nk, nb, nmu, ns), q_ids=[0, 1, 2], n_outputs=len(times))
        got = np.asarray(stream(times, np.eye(len(times), dtype=complex),
                                mun, nmu_face, endpoint, energy)) / np.sqrt(nk)
    expected = np.stack([ordered_pr_time(psi, energy, rho, p, times) for p in partners])
    np.testing.assert_allclose(got, expected, rtol=3e-12, atol=3e-14)


def test_coherent_projected_pr_retains_nonzero_one_over_z_moment():
    _, psi, energy, rho, partners = _case()
    # In a finite band projection the spatial density operators need not
    # commute. A forced even/z^2/passive-moment ansatz would erase this term.
    leading = ordered_pr_time(psi, energy, rho, partners[1], [0.])[0]
    assert np.linalg.norm(leading) > 1e-3
    z = 1.3e8+2.1e8j
    high = ordered_pr(psi, energy, rho, partners[1], [z], .043)[0]
    np.testing.assert_allclose(z*high, leading, rtol=3e-7, atol=3e-10)


def test_native_full_density_response_rules_and_fft_match_direct_resolvent():
    import jax
    from jax.sharding import Mesh
    from ffi import fft as F
    from gw.greens_function_kernel import density_left_endpoint
    from gw.response_bank import full_density_sample_rules
    from gw.w_isdf import full_density_response_stream

    jax.config.update('jax_enable_x64', True)
    mesh = Mesh(np.asarray(jax.devices('cpu')[:1]).reshape(1, 1), ('x', 'y'))
    _, psi, energy, rho, partners = _case()
    nk, nb, ns, nmu = psi.shape
    mun = np.transpose(psi, (0, 2, 3, 1))
    endpoint = density_left_endpoint(mesh, mun.shape)(mun, rho)
    sites = np.array([-.27+.63j, .42+.81j])
    gamma = .043
    rules = full_density_sample_rules(energy, sites, gamma_ry=gamma, tolerance=1e-8)
    got = np.zeros((nk, len(sites), nmu, nmu), complex)
    with F.xla_reference():
        for rule in rules:
            stream = full_density_response_stream(mesh, (3, 1, 1),
                (nk, nb, nmu, ns), q_ids=[0, 1, 2], n_outputs=len(rule['members']),
                rule_reference_ry=rule['reference_ry'])
            value = np.asarray(stream(rule['times'], rule['weights'], mun,
                                       psi, endpoint, energy))/np.sqrt(nk)
            got[:, rule['members']] += value
    expected = np.stack([ordered_pr(psi, energy, rho, p, sites, gamma) for p in partners])
    np.testing.assert_allclose(got, expected, rtol=5e-7, atol=5e-9)


def test_native_coherent_moments_keep_full_one_over_z_and_one_over_z_squared():
    import jax
    from jax.sharding import Mesh
    from types import SimpleNamespace
    from ffi import fft as F
    from gw.greens_function_kernel import density_left_endpoint
    from gw.response_bank import full_density_bare_moments

    jax.config.update('jax_enable_x64', True)
    mesh = Mesh(np.asarray(jax.devices('cpu')[:1]).reshape(1, 1), ('x', 'y'))
    _, psi, energy, rho, partners = _case()
    mun = np.transpose(psi, (0, 2, 3, 1))
    endpoint = density_left_endpoint(mesh, mun.shape)(mun, rho)
    with F.xla_reference():
        m0, m1 = full_density_bare_moments((mun, psi, endpoint, energy),
            SimpleNamespace(nkx=3, nky=1, nkz=1), mesh_xy=mesh, q_ids=[0, 1, 2])
    for q, partner in enumerate(partners):
        np.testing.assert_allclose(np.asarray(m0)[q]/np.sqrt(3),
            ordered_pr_time(psi, energy, rho, partner, [0.])[0], rtol=3e-12, atol=3e-14)
        h = .001
        vals = ordered_pr_time(psi, energy, rho, partner, [h, -h, h/2, -h/2])
        derivative = (4*(-(vals[2]-vals[3])/h)-(-(vals[0]-vals[1])/(2*h)))/3
        np.testing.assert_allclose(np.asarray(m1)[q]/np.sqrt(3), derivative,
                                   rtol=3e-10, atol=3e-12)


def test_native_general_sigma_scalar_quadrature_preserves_both_retarded_branches():
    from types import SimpleNamespace
    from gw.mpa.sigma import ordered_shared_pole_plan

    _, _, energy, _, _ = _case()
    poles = np.array([[.41+.017j, .93-.061j, -.41+.017j, -.93-.061j]])
    omega, eta = np.array([-.35, .42]), .14
    branches = [SimpleNamespace(space=space, E_A=energy,
        base_mask_A=np.ones(energy.shape, bool), band_weight=None) for space in ['cond', 'val']]
    rows, receipt = ordered_shared_pole_plan(poles, [4], branches, omega,
                                            eta_ry=eta, eps=2e-7)
    got = {space: np.zeros((2, *energy.shape, 4), complex) for space in ['cond', 'val']}
    for row in rows:
        window = row.window
        t, weight = np.asarray(window.nodes.t), np.asarray(window.nodes.alpha)
        for pi, bounds in zip(row.pole_indices, row.bounds):
            assert pi == 0
            for ip, p in enumerate(poles[0]):
                if not ((p.real > bounds[0] or (bounds[0] == 0 and p.real == 0))
                        and p.real <= bounds[1]):
                    continue
                for k, n in zip(*np.nonzero(window.mask_A)):
                    for iw in row.omega_idx:
                        value = np.sum(weight*np.exp(1j*(omega[iw]-energy[k, n]-p)*t))
                        # Native spatial contraction has a separate minus sign;
                        # its window prefactor=-1 cancels it, without another PV.
                        got[row.space][iw, k, n, ip] += value
    exact = 1/(omega[:, None, None, None]-energy[None, :, :, None]-poles[None, :, None, :]+1j*eta)
    for space, sign in [('cond', 1), ('val', -1)]:
        target = exact*((sign*poles.real >= 0)[None, :, None, :])
        np.testing.assert_allclose(got[space], target, rtol=3e-6, atol=3e-6)
    assert receipt['evaluation_margin_ry'] == pytest.approx(eta-.017)


def test_general_complex_reference_sigma_matches_literal_both_retarded_columns():
    rng, psi, energy, rho, partners = _case()
    active = np.array([1, 2])
    banks = _reciprocal_banks(rng, 3, 3)
    eta = .14
    for q, (poles, residues) in enumerate(banks):
        partner = partners[q]
        vertices = density_vertices(psi, partner, active)
        ei, ri, ext = energy[partner], rho[partner], energy[:, active]
        rp, rh = retarded_reference_pairs(vertices, ei, ri, ext, poles, residues, eta)
        # The general spatial residues are not restricted to bb^H or real poles.
        assert np.max(abs(residues-dagger(residues))) > .01
        assert poles.imag.max() > 0
        ret = kr.pole_residue_realization(poles, residues)
        less_residue = np.where((poles.real < 0)[:, None, None], -residues, 0.)
        less = kr.pole_residue_realization(poles, less_residue)
        got = sc.q_sigma_operators(vertices, ei, ri, ext,
                                   dict(retarded=ret, lesser=less), eta=eta)
        expected = branch_coefficients(rp, rh)
        for name in expected:
            np.testing.assert_allclose(got[name], expected[name], rtol=3e-12, atol=3e-14)
        np.testing.assert_allclose(got['ain'], 1j*rh[1], rtol=3e-12, atol=3e-14)
        np.testing.assert_allclose(got['aout'], 1j*rp[1], rtol=3e-12, atol=3e-14)
        val = advanced_hole_compatibility_pair(rh)
        np.testing.assert_array_equal(dagger(val[0]), rh[1])
        # Neither arithmetic shell averaging nor a retarded row in val[0]
        # is equivalent to the required right-column one-sided operator.
        assert np.linalg.norm(rh[0]-rh[1]) > 1e-4


def test_complete_q_active_collision_number_conservation_with_complex_signed_residues():
    rng, psi, energy, rho, partners = _case()
    active = np.array([1, 2])
    gamma = np.zeros((3, 2, 2), complex)
    injection = np.zeros_like(gamma)
    full_mh = np.zeros_like(gamma)
    active_mh = np.zeros_like(gamma)
    for q, (poles, residues) in enumerate(_reciprocal_banks(rng, 3, 3)):
        partner = partners[q]
        vertices = density_vertices(psi, partner, active)
        rp, rh = retarded_reference_pairs(vertices[:, :, active],
            energy[partner][:, active], rho[partner][:, active][:, :, active],
            energy[:, active], poles, residues, .14)
        row = branch_coefficients(rp, rh)
        gamma += row['gamma']/3
        injection += row['injection']/3
        active_mh += row['m_h']/3
        fp, fh = retarded_reference_pairs(vertices, energy[partner], rho[partner],
            energy[:, active], poles, residues, .14)
        full_mh += branch_coefficients(fp, fh)['m_h']/3
    r = rho[:, active][:, :, active]
    collision = injection-.5*(gamma@r+r@gamma)
    np.testing.assert_allclose(np.trace(collision, axis1=-2, axis2=-1).sum()/3,
                               0., atol=3e-14)
    assert np.linalg.norm(full_mh-active_mh) > 1e-3
    assert np.linalg.norm(collision) > 1e-3  # not zero by subtracting a residual


def test_reference_sigma_oracle_refuses_nonconvergent_laplace_integral():
    _, psi, energy, rho, partners = _case()
    vertex = density_vertices(psi, partners[1], [1, 2])
    with pytest.raises(ValueError, match='Laplace'):
        retarded_reference_pairs(vertex, energy[partners[1]], rho[partners[1]],
            energy[:, 1:], [.4+.15j], np.ones((1, 3, 3)), .14)


def test_roundoff_zero_reference_branch_keeps_original_half_weight_policy():
    rng, psi, energy, rho, partners = _case()
    partner = partners[1]
    vertex = density_vertices(psi, partner, [1, 2])
    poles = np.array([1e-16-.04j])
    residue = .07*(rng.normal(size=(1, 3, 3))+1j*rng.normal(size=(1, 3, 3)))
    rp, rh = retarded_reference_pairs(vertex, energy[partner], rho[partner],
                                      energy[:, 1:], poles, residue, .14)
    branches = kr.retarded_reference_branches(
        kr.pole_residue_realization(poles, residue), evaluation_height=.14)
    got = sc.q_sigma_operators(vertex, energy[partner], rho[partner], energy[:, 1:],
        dict(retarded=branches['correlation'], lesser=branches['lesser']), eta=.14)
    for name, value in branch_coefficients(rp, rh).items():
        np.testing.assert_allclose(got[name], value, rtol=3e-12, atol=3e-14)
    assert np.linalg.norm(rh[1]) > 1e-4
    assert np.linalg.norm(rp[1]) > 1e-4


def _native_port_fixture(monkeypatch):
    """Proper packed centroid, identity symmetry, capacity and full-k faces."""
    import warnings
    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from common.centroid_basis import PackedCentroidBasis
    from gw.centroid_k_unfold import build_centroid_k_unfold_plan
    from gw.shared_pole_screening import _shared_pole_tables
    from gw.shared_pole_recipe import CapacityLedger
    from common.units import RYD_TO_EV
    from symmetry_maps import SymMaps
    from tests import test_tdagw_dynsig_sigma as fixture

    jax.config.update('jax_enable_x64', True)
    devs = np.asarray(jax.devices('cpu'))
    mesh = Mesh(devs[:4].reshape(2, 2) if len(devs) >= 4 else devs[:1].reshape(1, 1), ('x', 'y'))
    for name, value in dict(KGRID=(3, 1, 1), NBF=3, NB_PAD=4, NS=2,
                            NMU=4, NOCC=2, B_LO=1, NA=2).items():
        monkeypatch.setattr(fixture, name, value)
    monkeypatch.setattr(fixture, 'RYD', RYD_TO_EV)
    rng = np.random.default_rng(20261021)
    nk, nb, ns, nmu = 3, 3, 2, 4
    psi = .23*(rng.normal(size=(nk, nb, ns, nmu))+1j*rng.normal(size=(nk, nb, ns, nmu)))
    energy = np.array([[-.91, -.24, .53], [-.82, -.17, .68], [-1.03, -.31, .71]])
    occ = np.broadcast_to([1., 1., 0.], (nk, nb)).copy()
    kf = np.array([[0., 0., 0.], [1/3, 0., 0.], [2/3, 0., 0.]])
    cidx = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.int32)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        sym = SymMaps(fixture._WfnStub(kf))
    kplan = build_centroid_k_unfold_plan(sym, cidx, fixture.FFT_GRID, mesh,
                                        nspinor=ns, parent_k_frac=kf)
    basis = PackedCentroidBasis(mesh, kplan.layout, cidx, nmu)
    perm = np.asarray(kplan.layout.axis.pack_host(np.arange(nmu)[None], axis=1))[0]
    psi_packed = psi[..., perm]
    wfns, meta = fixture._gw_bundle(mesh, psi_packed, energy, occ, kplan)
    meta.mu_basis = basis
    meta.shared_pole_capacity = CapacityLedger(meta, mesh_xy=mesh, device_budget_bytes=2**30)
    meta.shared_pole_capacity.live_stages = ()
    tables = _shared_pole_tables(meta, sym, cidx)
    return dict(rng=rng, mesh=mesh, nk=nk, nb=nb, ns=ns, nmu=nmu, psi=psi,
                energy=energy, occ=occ, perm=perm, wfns=wfns, meta=meta,
                tables=tables, sym=sym, fixture=fixture)


@pytest.mark.parametrize('density_case', ['coherent', 'projector', 'passive_projector'])
def test_native_general_store_fft_projection_and_active_rates_against_literal_sum(tmp_path, monkeypatch, density_case):
    """Actual store -> native synthesis/FFT -> compact4-index output; no FFT mocks."""
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from file_io.shared_pole_store import write_general_shared_pole_model
    from gw import ppm_sigma as ps
    from common.units import RYD_TO_EV
    from ffi import fft as F

    case = _native_port_fixture(monkeypatch)
    rng, mesh, nk, nb, ns, nmu, psi, energy, occ, perm, wfns, meta, tables, fixture = (
        case[key] for key in ['rng', 'mesh', 'nk', 'nb', 'ns', 'nmu', 'psi',
            'energy', 'occ', 'perm', 'wfns', 'meta', 'tables', 'fixture'])
    banks = _reciprocal_banks(rng, nk, nmu)
    if density_case == 'passive_projector':
        psd = []
        for _ in range(nk):
            q = .15*(rng.normal(size=(2, nmu, nmu))+1j*rng.normal(size=(2, nmu, nmu)))
            psd.append(q@dagger(q))
        banks = [(np.array([.41, .93, -.41, -.93], complex),
                  np.concatenate((psd[q], -np.asarray(psd)[(-q) % nk].conjugate())))
                 for q in range(nk)]
    width = 4*nmu
    left = np.stack([np.concatenate([r for r in residues], axis=1)
                     for _, residues in banks])[:, perm, None, :]
    right = np.broadcast_to(np.tile(np.eye(nmu), (1, 4)), (nk, nmu, width))[:, perm, None, :]
    poles = np.stack([np.repeat(p, nmu) for p, _ in banks])
    put = lambda a: jax.device_put(np.ascontiguousarray(a, dtype=complex), NamedSharding(mesh, P(None, 'x', None, 'y')))
    identity = dict(iteration_id='tiny-1', hamiltonian='tiny-H', energies='tiny-E',
                    occupations='tiny-rho', wavefunctions='tiny-psi', centroids='tiny-mu')
    path = str(tmp_path/'general.h5')
    header = write_general_shared_pole_model(path, put(left), put(right),
        jax.device_put(poles, NamedSharding(mesh, P())), np.full(nk, width),
        q_span=(0, nk), meta=meta, tables=tables, recipe=dict(accuracy='test'),
        receipts=dict(identity=identity), kmax=width)
    handle = dict(path=path, identity=identity, digest=header['digest'], K=header['K'],
                  representation='ordered_complex')
    source = ps.shared_pole_window_source(handle, meta, efermi_ry=0., linalg='distributed')
    rotation, rho_a = [], []
    for _ in range(nk):
        u, _ = np.linalg.qr(rng.normal(size=(2, 2))+1j*rng.normal(size=(2, 2)))
        v, _ = np.linalg.qr(rng.normal(size=(2, 2))+1j*rng.normal(size=(2, 2)))
        rotation.append(u)
        rho_a.append(v@np.diag([.19, .83])@v.conj().T)
    rotation, rho_a = np.asarray(rotation), np.asarray(rho_a)
    if density_case != 'coherent':
        rotation = np.broadcast_to(np.eye(2), (nk, 2, 2)).copy()
        rho_a = np.broadcast_to(np.diag([1., 0.]), (nk, 2, 2)).astype(complex).copy()
    config = fixture._config(np.array([-2., 2.]))
    config.sigma.regularization_ev = .14*RYD_TO_EV
    config.sigma.quadrature_eps = 2e-7
    plan = ps.window_sigma_plan(config, window_only=True)
    window = dict(b_lo=1, U=rotation, e=energy[:, 1:],
                  f=np.diagonal(rho_a, axis1=-2, axis2=-1).real, rho=rho_a)
    with F.xla_reference():
        got = ps.window_sigma_c(wfns, source, window, plan, meta=meta,
            mesh_xy=mesh, branches=True, pairing='vertex', print_fn=lambda *a, **k: None)
    rotated = psi.copy()
    rotated[:, 1:] = np.einsum('kmsu,kmi->kisu', psi[:, 1:], rotation)
    rho = np.zeros((nk, nb, nb), complex)
    rho[:, 0, 0], rho[:, 1:, 1:] = 1., rho_a
    totals = {name: np.zeros((2, nk, 2, 2), complex) for name in ['cond', 'val']}
    active_totals = {name: np.zeros_like(value) for name, value in totals.items()}
    for q, (p, r) in enumerate(banks):
        partner = np.array([(k-q) % nk for k in range(nk)])
        vertex = density_vertices(rotated, partner, [1, 2])
        rp, rh = retarded_reference_pairs(vertex, energy[partner], rho[partner], energy[:, 1:], p, r, .14)
        ap, ah = retarded_reference_pairs(vertex[:, :, 1:], energy[partner, 1:],
            rho[partner, 1:, 1:], energy[:, 1:], p, r, .14)
        for name, full, active in [('cond', rp, ap), ('val', rh, ah)]:
            totals[name] += full/nk
            active_totals[name] += active/nk
    for name in totals:
        np.testing.assert_allclose(got['retarded_pair'][name], totals[name], rtol=1e-5, atol=3e-8)
    active = branch_coefficients(active_totals['cond'], active_totals['val'])
    full = branch_coefficients(totals['cond'], totals['val'])
    from tdagw.collision import branch_operators
    ga, da, _ = branch_operators(got['active_branch_pair']['cond'], got['active_branch_pair']['val'])
    _, _, mh = branch_operators(got['branch_pair']['cond'], got['branch_pair']['val'])
    for actual, expected in [(ga, active['gamma']), (da, active['injection']), (mh, full['m_h'])]:
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=3e-8)
    fresh = da-.5*(ga@rho_a+rho_a@ga)
    np.testing.assert_allclose(np.trace(fresh, axis1=-2, axis2=-1).mean(), 0., atol=2e-8)


def test_full_native_screening_builder_matches_literal_full_port_dyson(tmp_path, monkeypatch):
    """Full-rho FFT bank + general pencil + real store, against direct Dyson."""
    import h5py
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.units import RYD_TO_EV
    from types import SimpleNamespace
    from ffi import fft as F
    from gw.shared_pole_recipe import bind_shared_pole_census, resolve_shared_pole_recipe
    from gw.shared_pole_screening import screen_general_shared_poles

    case = _native_port_fixture(monkeypatch)
    rng, mesh, nk, nb, nmu, psi, energy, wfns, meta, sym, fixture = (
        case[key] for key in ['rng', 'mesh', 'nk', 'nb', 'nmu', 'psi', 'energy',
                              'wfns', 'meta', 'sym', 'fixture'])
    config = fixture._config(np.array([-2., 2.]))
    # CPU emulated meshes have no MPI ScaLAPACK process grid. Tiny local
    # dense solves are the supported owner route; all faces/FFTs stay sharded.
    config.backend = SimpleNamespace(linalg='local')
    config.bispinor = False
    config.sigma.w_model = 'shared_pole'
    config.sigma.w_accuracy = 'relaxed'
    config.sigma.w_support_sites_ev = '1,3,5,7,9,11,13,15,18,21,24 | 2,4,8,16 | 1.9'
    config.sigma.regularization_ev = .14*RYD_TO_EV
    bind_shared_pole_census(wfns, meta, occupation_state=None,
        trs_allowed=False, state_capacity=1., kweights=np.full(nk, 1/nk))
    meta.shared_pole_recipe = resolve_shared_pole_recipe(config, wfns, meta,
        mesh_xy=mesh, print_fn=lambda *a, **k: None, support_reads_ev=energy[:, 1:]*RYD_TO_EV)
    meta.shared_pole_capacity.live_stages = ()
    rho_a = []
    for _ in range(nk):
        u, _ = np.linalg.qr(rng.normal(size=(2, 2))+1j*rng.normal(size=(2, 2)))
        rho_a.append(u@np.diag([.19, .83])@u.conj().T)
    rho_a = np.asarray(rho_a)
    rho = np.zeros((nk, nb, nb), complex)
    rho[:, 0, 0], rho[:, 1:, 1:] = 1., rho_a
    bare = np.broadcast_to(.15*np.eye(nmu), (nk, nmu, nmu)).astype(complex)
    V = jax.device_put(bare, NamedSharding(mesh, P(None, 'x', 'y')))
    snapshot = dict(active_slice=slice(1, 3), U_active=np.broadcast_to(np.eye(2), (nk, 2, 2)),
        E_active_ry=energy[:, 1:], rho_active=rho_a, gamma_ry=.043, eta_ry=.14,
        fit_tolerance=.02, host_workspace_bytes=128*1024**2)
    with F.xla_reference():
        built = screen_general_shared_poles(wfns, V, meta, config, mesh_xy=mesh,
            sym=sym, snapshot=snapshot, run_dir=str(tmp_path), label='builder',
            print_fn=lambda *a, **k: None)
    with h5py.File(built['shared_pole']['path']) as f:
        left, right = np.asarray(f['left'])[:, :, 0], np.asarray(f['right'])[:, :, 0]
        poles, counts = np.asarray(f['poles_ry']), np.asarray(f['K'])
    assert left.shape[1] == right.shape[1] == nmu
    sites = np.array([-.28+.21j, .39+.26j, .73+.31j])
    for q in range(nk):
        partner = np.array([(k-q) % nk for k in range(nk)])
        pr = ordered_pr(psi, energy, rho, partner, sites, .043)
        n = int(counts[q])
        # Independent full-matrix infinity coefficients, not only the
        # retained-direction diagnostic or fitted/held support samples.
        c0 = ordered_pr_time(psi, energy, rho, partner, [0.])[0]
        h = .001
        vals = ordered_pr_time(psi, energy, rho, partner, [h, -h, h/2, -h/2])
        c1 = (4*(-(vals[2]-vals[3])/h)-(-(vals[0]-vals[1])/(2*h)))/3
        exact0 = bare[q]@c0@bare[q]
        exact1 = (bare[q]@(c1-1j*.043*c0)@bare[q]
                  +bare[q]@c0@bare[q]@c0@bare[q])
        model0 = left[q, :, :n]@right[q, :, :n].conj().T
        model1 = (left[q, :, :n]*poles[q, :n])@right[q, :, :n].conj().T
        assert np.linalg.norm(exact0) > .001
        np.testing.assert_allclose(model0, exact0, rtol=5e-8, atol=2e-11)
        np.testing.assert_allclose(model1, exact1, rtol=5e-8, atol=2e-11)
        for iz, z in enumerate(sites):
            exact = np.linalg.solve(np.eye(nmu)-bare[q]@pr[iz], bare[q])-bare[q]
            got = (left[q, :, :n]/(z-poles[q, :n]))@right[q, :, :n].conj().T
            assert np.linalg.norm(got-exact)/np.linalg.norm(exact) < 5e-8
    assert built['receipt']['input_density_coherence_norm'] > .01
    assert built['receipt']['n_mu'] == nmu
    assert built['receipt']['n_q'] == nk

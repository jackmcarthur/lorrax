"""Independent complex endpoint, Dyson and projected-self-energy identities."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from gw import contour_reference as cd, response_bank as rb
from gw.plane_wave_lehmann import OrderedLehmannPair
from gw.plane_wave_screening import SphereScreening
from runtime.padding import padded_axis


@pytest.fixture
def mesh():
    return Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))


def face(value, mesh):
    return jax.device_put(np.asarray(value, np.complex128), NamedSharding(mesh, P(None, "x", "y")))


@pytest.mark.parametrize("broadcast", ["none", "map", "interaction"])
def test_complex_endpoint_projection_and_lift_literal_sum(mesh, broadcast):
    rng = np.random.default_rng(7493720)
    pair = rng.normal(size=(2, 3, 6)) + 1j * rng.normal(size=(2, 3, 6))
    zeta = rng.normal(size=(2, 6, 4)) + 1j * rng.normal(size=(2, 6, 4))
    interaction = rng.normal(size=(2, 4, 4)) + 1j * rng.normal(size=(2, 4, 4))
    if broadcast == "map": zeta = zeta[:1]
    if broadcast == "interaction": interaction = interaction[:1]
    mapped = cd.project_density_endpoints(face(pair, mesh), face(zeta, mesh), mesh=mesh)
    expected = np.zeros((2, 3, 4), complex)
    lifted = np.zeros((2, 6, 6), complex)
    for b in range(2):
        a = zeta[0 if len(zeta) == 1 else b]
        w = interaction[0 if len(interaction) == 1 else b]
        for t in range(3):
            for g in range(4):
                expected[b, t, g] = sum(pair[b, t, mu] * a[mu, g] for mu in range(6))
        for mu in range(6):
            for nu in range(6):
                lifted[b, mu, nu] = sum(a[mu, g].conj() * w[g, h] * a[nu, h]
                                            for g in range(4) for h in range(4)) / 13.
    got = cd.lift_interaction_endpoints(face(interaction, mesh), face(zeta, mesh),
                                       mesh=mesh, prefactor=1./13.)
    np.testing.assert_allclose(mapped, expected, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(got, lifted, rtol=3e-13, atol=3e-13)
    assert mapped.sharding == got.sharding == NamedSharding(mesh, P(None, "x", "y"))
    wrong = np.einsum("bmg,bgh,bnh->bmn", zeta, interaction, zeta.conj()) / 13.
    assert np.linalg.norm(lifted - wrong) > 1e-2
    # Mapping a conjugated actual density is different from conjugating its map.
    reverse = cd.project_density_endpoints(face(pair.conj(), mesh), face(zeta, mesh), mesh=mesh)
    assert np.linalg.norm(np.asarray(reverse) - expected.conj()) > 1e-2


def literal_chi(density, de, z):
    value = np.zeros((len(z), 1, density.shape[-1], density.shape[-1]), complex)
    slope = np.zeros_like(value)
    for t, gap in enumerate(de):
        for i, site in enumerate(z):
            a = np.outer(density[0, t], density[0, t].conj())
            b = a.conj()
            value[i, 0] += a/(gap+site) - b/(-gap+site)
            slope[i, 0] += -a/(2*site*(gap+site)**2) + b/(2*site*(-gap+site)**2)
    return value, slope


def test_asymmetric_complex_map_full_dyson_slope_and_sigma_equivalence(mesh):
    rng = np.random.default_rng(9121776)
    nk, volume, M, G, T = 2, 13., 6, 4, 8
    zeta = (rng.normal(size=(1, M, G)) + 1j*rng.normal(size=(1, M, G))) * .2
    zeta[:, 1] = 0.; zeta[..., 3] = 0.
    density = (rng.normal(size=(1, T, M)) + 1j*rng.normal(size=(1, T, M))) * .2
    density[..., 1] = 0.
    de = -np.linspace(.7, 2.2, T)
    sites = np.asarray([.7+.25j, .25j])
    literal, literal_ds = literal_chi(density, de, sites)
    raw = OrderedLehmannPair.from_density_face(face(density, mesh), de, np.ones(T),
        np.ones(T, bool), mesh=mesh, physical_prefactor=1./np.sqrt(nk),
        endpoint_valid=np.asarray([True, False, True, True, True, True]),
        normalization="raw centroid 1/sqrtNk", panel_bytes=4096)
    chi, dchi = raw.evaluate_same_k_pair(sites, with_derivative=True)
    np.testing.assert_allclose(chi, literal/np.sqrt(nk), rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(dchi, literal_ds/np.sqrt(nk), rtol=2e-12, atol=2e-13)
    projected = []
    for reverse in (False, True):
        data = cd.project_density_endpoints(face(density.conj() if reverse else density, mesh),
                                            face(zeta, mesh), mesh=mesh)
        bank = OrderedLehmannPair.from_density_face(data, -de if reverse else de,
            -np.ones(T) if reverse else np.ones(T), np.ones(T, bool), mesh=mesh,
            physical_prefactor=2./(nk*volume), endpoint_valid=np.asarray([True, True, True, False]),
            normalization="fitted density spin/(Omega Nk)", panel_bytes=4096)
        projected.append(bank.evaluate(sites, with_derivative=True))
    chi_g = projected[0][0] + projected[1][0]
    ds_g = projected[0][1] + projected[1][1]
    expected_g = np.stack([zeta[0].T @ c[0] @ zeta[0].conj() * 2./(nk*volume) for c in literal])[:, None]
    np.testing.assert_allclose(chi_g, expected_g, rtol=2e-12, atol=2e-13)
    v = np.diag([0., 2., 3., 0.]).astype(complex)
    V = zeta[0].conj() @ v @ zeta[0].T / volume
    w, u = np.linalg.eigh(V)
    H = (u * np.sqrt(np.maximum(w, 0.))) @ u.conj().T
    meta = SimpleNamespace(nk_tot=nk, nspin=1, nspinor_wfnfile=1)
    from gw.w_isdf import _w_solve_pref_scalar
    # The P1 literal fixture uses the canonical native program; distributed
    # LU is separately priced on a genuine one-process-per-cell P4 mesh.
    dyson, _, _, _ = rb._response_programs(mesh, M, "off", "auto",
                                          _w_solve_pref_scalar(meta), True, None)
    placed_root = dyson.place(face(H[None], mesh))
    solver = SphereScreening.__new__(SphereScreening)
    solver.mesh=mesh; solver.linalg="local"; solver.batched_route="auto"
    solver.axis=padded_axis(G, mesh, name="endpoint algebra plant")
    solver._V=face(v[None], mesh)
    pair = rng.normal(size=(1, 2, M)) + 1j*rng.normal(size=(1, 2, M)); pair[..., 1]=0.
    for i in range(len(sites)):
        wc_mu, dw_mu = dyson.pair("face")(placed_root, chi[i], dchi[i])
        Wg, dw_g = solver.solve_pair(chi_g[i], ds_g[i])
        wc_g = Wg-face(v[None], mesh)
        lift = cd.lift_interaction_endpoints(wc_g, face(zeta, mesh), mesh=mesh, prefactor=1./volume)
        ds_lift = cd.lift_interaction_endpoints(dw_g, face(zeta, mesh), mesh=mesh, prefactor=1./volume)
        expected_W = np.linalg.solve(np.eye(M)-V@(literal[i, 0]*2./nk), V)
        expected_ds = expected_W @ (literal_ds[i, 0]*2./nk) @ expected_W
        np.testing.assert_allclose(wc_mu[0], expected_W-V, rtol=2e-11, atol=3e-13)
        np.testing.assert_allclose(dw_mu[0], expected_ds, rtol=2e-11, atol=3e-13)
        np.testing.assert_allclose(lift, wc_mu, rtol=2e-11, atol=3e-13)
        np.testing.assert_allclose(ds_lift, dw_mu, rtol=2e-11, atol=3e-13)
        for actual_pair in (pair, pair.conj()):
            transformed = cd.project_density_endpoints(face(actual_pair, mesh), face(zeta, mesh), mesh=mesh)
            _, mu_sigma = cd.project_interaction_diagonal(-wc_mu, face(actual_pair, mesh),
                mesh=mesh, prefactor=1./nk, scalar_replication_bound_bytes=32)
            _, g_sigma = cd.project_interaction_diagonal(-wc_g, transformed,
                mesh=mesh, prefactor=1./(nk*volume), scalar_replication_bound_bytes=32)
            expected_sigma = [-np.vdot(p, (expected_W-V)@p)/nk for p in actual_pair[0]]
            np.testing.assert_allclose(mu_sigma[0], expected_sigma, rtol=2e-11, atol=3e-13)
            np.testing.assert_allclose(g_sigma, mu_sigma, rtol=2e-11, atol=3e-13)
    assert np.linalg.norm(V@(literal[0, 0]*2./nk)-(literal[0, 0]*2./nk)@V) > 1e-5
    assert np.linalg.norm(np.asarray(projected[1][0])-np.asarray(projected[0][0]).conj()) > 1e-5


@pytest.mark.parametrize("prefactor", [0., -1., np.nan, np.inf])
def test_endpoint_lift_requires_explicit_finite_physical_prefactor(mesh, prefactor):
    with pytest.raises(ValueError):
        cd.lift_interaction_endpoints(face(np.eye(4)[None], mesh), face(np.ones((1,6,4)), mesh),
                                     mesh=mesh, prefactor=prefactor)


def test_endpoint_map_rejects_wrong_axes_dtype_and_layout(mesh):
    pair=face(np.ones((1,2,6)), mesh); zeta=face(np.ones((1,6,4)), mesh)
    with pytest.raises(ValueError): cd.project_density_endpoints(pair, zeta[..., :3, :], mesh=mesh)
    with pytest.raises(ValueError): cd.project_density_endpoints(pair.real, zeta, mesh=mesh)
    with pytest.raises(ValueError): cd.project_density_endpoints(pair, jax.device_put(np.ones((1,6,4), complex)), mesh=mesh)
    with pytest.raises(ValueError): cd.lift_interaction_endpoints(face(np.ones((1,4,3)),mesh), zeta, mesh=mesh,prefactor=1.)

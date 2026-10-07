"""Independent cached-face orientation, normalization and donation controls."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from gw.efermi import OccupationState
from gw.lehmann_response import centroid_response_denominator, centroid_pair_density
from gw.plane_wave_lehmann import OrderedLehmannPair
from gw.wavefunction_bundle import BandSlices, Wavefunctions
from gw.w_isdf import compute_chi0_direct_fractional


@pytest.fixture
def mesh():
    return Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))


def face(value, mesh):
    return jax.device_put(np.asarray(value, np.complex128),
                          NamedSharding(mesh, P(None, "x", "y")))


def literal(density, de, df, valid, z, scale, permutation=None):
    """Literal directed pairs, with an independently written reverse term."""
    values = np.zeros((len(z), 1, density.shape[-1], density.shape[-1]), complex)
    slopes = np.zeros_like(values)
    for pair in np.flatnonzero(valid):
        m = density[0, pair]
        for j, site in enumerate(z):
            outer = np.outer(m, m.conj())
            denominator = de[pair] + site
            values[j, 0] += df[pair] * outer / denominator
            slopes[j, 0] -= df[pair] * outer / (2 * site * denominator ** 2)
            if permutation is not None:
                reverse = m[permutation].conj()
                outer = np.outer(reverse, reverse.conj())
                denominator = -de[pair] + site
                values[j, 0] -= df[pair] * outer / denominator
                slopes[j, 0] += df[pair] * outer / (2 * site * denominator ** 2)
    return values * scale, slopes * scale


def plant(mesh, *, donate=False, **changes):
    rng = np.random.default_rng(710079)
    endpoint = np.asarray([True, False, True, True, False, True])
    valid = np.asarray([True, True, True, True, False])
    de = np.asarray([-.8, -2.1, 1.4, .7, np.nan])
    df = np.asarray([.3, 1., -.6, .04, np.inf])
    density = rng.normal(size=(1, 5, 6)) + 1j * rng.normal(size=(1, 5, 6))
    density[:, ~valid] = 0; density[..., ~endpoint] = 0
    arg = dict(mesh=mesh, physical_prefactor=1 / np.sqrt(512.),
               endpoint_valid=endpoint, normalization="raw centroid;1/sqrt(Nk)",
               panel_bytes=4096, donate_density=donate)
    arg.update(changes)
    value = face(density, mesh)
    bank = OrderedLehmannPair.from_density_face(value, de, df, valid, **arg)
    return bank, value, density, de, df, valid, endpoint


@pytest.mark.parametrize("paired", [False, True])
def test_complex_centroid_value_and_slope_match_literal_sum(mesh, paired):
    bank, raw, density, de, df, valid, endpoints = plant(mesh)
    z = np.asarray([.7 + .25j, -.3 + .5j, .8j])
    permutation = np.arange(6) if paired else None
    expected, slope = literal(density, de, df, valid, z, 1 / np.sqrt(512.), permutation)
    evaluate = bank.evaluate_same_k_pair if paired else bank.evaluate
    got, derivative = evaluate(z, with_derivative=True)
    np.testing.assert_allclose(got, expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(derivative, slope, rtol=2e-12, atol=2e-12)
    assert np.linalg.norm(expected - expected.swapaxes(-1, -2)) > 1e-5
    assert np.max(np.abs(np.asarray(got)[..., ~endpoints, :])) == 0.
    assert np.max(np.abs(np.asarray(got)[..., :, ~endpoints])) == 0.
    np.testing.assert_array_equal(np.asarray(raw), density)
    assert bank.receipt["physical_pair_count"] == 4
    assert bank.receipt["physical_endpoint_count"] == 4
    assert bank.receipt["scale"] == 1 / np.sqrt(512.)


def test_donation_invalidates_only_raw_input_preserving_both_response_faces(mesh):
    plain, _, density, de, df, valid, _ = plant(mesh)
    donated, raw, _, _, _, _, _ = plant(mesh, donate=True)
    assert raw.is_deleted()
    np.testing.assert_array_equal(np.asarray(donated.left), density.swapaxes(1, 2))
    np.testing.assert_array_equal(np.asarray(donated.right), density.conj())
    z = np.asarray([.7 + .25j, .8j])
    expected, slope = literal(density, de, df, valid, z, 1 / np.sqrt(512.), np.arange(6))
    got, derivative = donated.evaluate_same_k_pair(z, with_derivative=True)
    np.testing.assert_allclose(got, expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(derivative, slope, rtol=2e-12, atol=2e-12)
    ordinary, ordinary_ds = plain.evaluate_same_k_pair(z, with_derivative=True)
    np.testing.assert_allclose(got, ordinary, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(derivative, ordinary_ds, rtol=2e-12, atol=2e-12)
    assert donated.receipt["raw_density_donated"] is True


@pytest.mark.parametrize("nspinor", [1, 2])
def test_factory_matches_physical_ordered_centroid_scanner_and_explicit_transpose(mesh, nspinor):
    rng = np.random.default_rng(710089)
    energy = np.asarray([[-2., -.4, .9, 1e100], [-1.8, -.3, 1.1, -1e100]])
    valid = np.asarray([[True, True, True, False]] * 2)
    occupation = np.asarray([[.91, .42, .08, 0.], [.88, .37, .06, 0.]])
    endpoints = np.asarray([True, False, True, True, False, True])
    psi = rng.normal(size=(2, 4, nspinor, 6)) + 1j * rng.normal(size=(2, 4, nspinor, 6))
    psi[~valid] = 0; psi[..., ~endpoints] = 0
    wavefunctions = Wavefunctions(enk=jnp.asarray(energy), occ=jnp.asarray(occupation),
        slices=BandSlices.from_band_edges(0, 0, 2, 3, 4, b4_logical=4), valid_kn=jnp.asarray(valid))
    wavefunctions.psi_nmu = jax.device_put(psi, NamedSharding(mesh, P(None, "x", None, "y")))
    wavefunctions.psi_mun = jax.device_put(psi.transpose(0, 2, 3, 1), NamedSharding(mesh, P(None, None, "x", "y")))
    capacity = 2 if nspinor == 1 else 1
    state = OccupationState(jnp.asarray(occupation), .1, "fd", .02, capacity * occupation.mean(axis=0).sum())
    meta = SimpleNamespace(nk_tot=2, nspin=1, nspinor=nspinor, nspinor_wfnfile=nspinor,
        n_rmu=6, cell_volume=13., b_id_4_chi_user=4)
    density, de, df = [], [], []
    for k in range(2):
        for a in range(3):
            for b in range(a + 1, 3):
                density.append(sum(psi[k, a, spin].conj() * psi[k, b, spin] for spin in range(nspinor)))
                de.append(energy[k, a] - energy[k, b])
                df.append(occupation[k, a] - occupation[k, b])
    density = np.asarray(density)[None]; de = np.asarray(de); df = np.asarray(df)
    bank = OrderedLehmannPair.from_density_face(face(density, mesh), de, df,
        np.ones(len(de), bool), mesh=mesh, physical_prefactor=1 / centroid_response_denominator(2),
        endpoint_valid=endpoints, normalization="raw centroid;1/sqrt(Nk)", panel_bytes=4096)
    z = np.asarray([.7 + .25j, .8j])
    got, slope = bank.evaluate_same_k_pair(z, with_derivative=True)
    expected, expected_ds = literal(density, de, df, np.ones(len(de), bool), z,
                                    1 / np.sqrt(2.), np.arange(6))
    np.testing.assert_allclose(got, expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(slope, expected_ds, rtol=2e-12, atol=2e-12)
    scanned, scanned_ds = compute_chi0_direct_fractional(wavefunctions, z, meta, mesh,
        occupation_state=state, kminq_rows=np.asarray([[0, 1]]), nb_logical=4,
        ordered=True, with_derivative=True, pair_tile=2)
    np.testing.assert_allclose(got, scanned, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(slope, scanned_ds, rtol=2e-12, atol=2e-12)
    transpose, transpose_ds = compute_chi0_direct_fractional(wavefunctions, z, meta, mesh,
        occupation_state=state, kminq_rows=np.asarray([[0, 1]]), nb_logical=4,
        ordered=False, with_derivative=True, pair_tile=2)
    np.testing.assert_allclose(transpose, expected.swapaxes(-1, -2), rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(transpose_ds, expected_ds.swapaxes(-1, -2), rtol=2e-12, atol=2e-12)
    assert np.linalg.norm(expected - expected.swapaxes(-1, -2)) > 1e-5


def test_pw_density_factory_preserves_directed_and_negated_paired_owner(mesh):
    rng = np.random.default_rng(710091)
    vertices = rng.normal(size=(2, 2, 2, 4)) + 1j * rng.normal(size=(2, 2, 2, 4))
    vertices[..., 3] = 0
    de = np.asarray([[[-2., -1.], [-2.2, -.7]], [[-1.4, -.5], [-1.2, -.3]]])
    df = np.asarray([[[.8, .3], [.9, .4]], [[.5, .2], [.7, .1]]])
    valid = np.ones(de.shape, bool)
    raw = jax.device_put(vertices, NamedSharding(mesh, P(("x", "y"), None, None, None)))
    ordered = OrderedLehmannPair(raw, de, df, valid, mesh=mesh, cell_volume=13.,
        n_k=512, physical_g_count=3, panel_bytes=4096)
    cached = OrderedLehmannPair.from_density_face(face(vertices.reshape(1, -1, 4), mesh),
        de.ravel(), df.ravel(), valid.ravel(), mesh=mesh, physical_prefactor=2 / (13 * 512),
        endpoint_valid=np.asarray([True, True, True, False]),
        normalization="PW;spin/(Omega*Nk)", panel_bytes=4096)
    z = np.asarray([.7 + .25j, .8j]); gvecs = np.asarray([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
    for method_a, method_b in ((ordered.evaluate, cached.evaluate),
            (lambda z, **k: ordered.evaluate_gamma_pair(z, gvecs=gvecs, **k),
             lambda z, **k: cached.evaluate_same_k_pair(z, endpoint_negation=np.asarray([0, 2, 1, 3]), **k))):
        a, ad = method_a(z, with_derivative=True); b, bd = method_b(z, with_derivative=True)
        np.testing.assert_allclose(a, b, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(ad, bd, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("changes", [dict(endpoint_valid=np.ones(6, int)),
    dict(endpoint_valid=np.ones(5, bool)), dict(endpoint_valid=np.zeros(6, bool)),
    dict(physical_prefactor=0.), dict(physical_prefactor=-1.), dict(physical_prefactor=np.inf),
    dict(physical_prefactor=np.nan), dict(normalization=" "), dict(donate_density=1),
    dict(panel_bytes=True)])
def test_factory_refuses_ambiguous_normalization_endpoint_and_resource_controls(mesh, changes):
    with pytest.raises((ValueError, TypeError)):
        plant(mesh, **changes)


@pytest.mark.parametrize("where", ["pair", "endpoint", "nonfinite"])
def test_factory_refuses_nonzero_ghost_even_tiny_and_physical_nonfinite(mesh, where):
    _, _, density, de, df, valid, endpoints = plant(mesh)
    if where == "pair": density[0, 4, 0] = 1e-30
    elif where == "endpoint": density[0, 0, 1] = 1e-30
    else: density[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        OrderedLehmannPair.from_density_face(face(density, mesh), de, df, valid,
            mesh=mesh, physical_prefactor=1., endpoint_valid=endpoints,
            normalization="declared", panel_bytes=4096)


@pytest.mark.parametrize("permutation", [np.asarray([1, 0, 2, 3, 4, 5]),
    np.asarray([2, 0, 1, 3, 4, 5]), np.arange(6, dtype=float), np.arange(5)])
def test_same_k_reverse_refuses_invalid_endpoint_involution(mesh, permutation):
    bank, *_ = plant(mesh)
    with pytest.raises(ValueError):
        bank.evaluate_same_k_pair([.7 + .25j], endpoint_negation=permutation)


def test_interleaved_centroids_cannot_be_misread_as_prefix_pw_sphere(mesh):
    bank, *_ = plant(mesh)
    with pytest.raises(ValueError, match="interleaved"):
        bank.evaluate_gamma_pair([.7 + .25j], gvecs=np.asarray([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0]]))


@pytest.mark.parametrize("nk", [True, 0, -1, 1.2])
def test_centroid_denominator_refuses_nonphysical_census(nk):
    with pytest.raises((ValueError, TypeError)):
        centroid_response_denominator(nk)


def test_centroid_denominator_keeps_original_sqrt_arithmetic():
    for nk in (1, 2, 512):
        np.testing.assert_array_equal(centroid_response_denominator(nk),
                                      jnp.sqrt(jnp.asarray(nk, jnp.float64)))


@pytest.mark.parametrize("nspinor", [1, 2])
def test_centroid_pair_density_keeps_literal_complex_spin_trace(nspinor):
    rng = np.random.default_rng(710099)
    left = rng.normal(size=(2, nspinor, 3, 2)) + 1j * rng.normal(size=(2, nspinor, 3, 2))
    right = rng.normal(size=(2, nspinor, 3, 4)) + 1j * rng.normal(size=(2, nspinor, 3, 4))
    expected = np.empty((2, 3, 2, 4), complex)
    for k in range(2):
        for mu in range(3):
            for a in range(2):
                for b in range(4):
                    expected[k, mu, a, b] = sum(left[k, spin, mu, a] * right[k, spin, mu, b].conj() for spin in range(nspinor))
    got = centroid_pair_density(left, right)
    np.testing.assert_allclose(got, expected, rtol=2e-13, atol=2e-13)
    assert np.linalg.norm(expected - expected.conj()) > 1e-4

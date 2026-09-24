"""Finite-temperature Matsubara chi0 against an independent k-space Lehmann sum."""

import harness
from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax.config.update("jax_enable_x64", True)

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

from gw import shared_pole_recipe, w_isdf  # noqa: E402
from gw.wavefunction_bundle import (  # noqa: E402
    BandSlices,
    PSI_MUN_SPEC,
    PSI_NMU_SPEC,
    Wavefunctions,
)


def _mesh_xy():
    devices = np.asarray(jax.devices("cpu"), dtype=object)
    if devices.size >= 4:
        devices = devices[:4].reshape(2, 2)
    else:
        devices = devices[:1].reshape(1, 1)
    return Mesh(devices, ("x", "y"))


def _put(a, mesh, spec):
    return jax.device_put(jnp.asarray(a), NamedSharding(mesh, spec))


def _stand_ins(monkeypatch):
    import common.fft_helpers as fft_helpers
    import distrib_la

    def gemm_plan(mesh, **kwargs):
        gemm = lambda a, b: jnp.einsum("qmk,qkn->qmn", a, b)
        gemm.mesh = mesh
        return harness.with_active_range(gemm)

    def flat_k_fftn(mesh, kgrid, spec, *, norm="ortho", out_spec=None):
        fft3 = fft_helpers.make_sharded_fftn_3d(mesh, spec, spec, axes=(0, 1, 2), norm=norm)
        return lambda x: fft3(jnp.reshape(x, tuple(kgrid) + x.shape[1:])).reshape(x.shape)

    monkeypatch.setattr(distrib_la, "gemm_plan", gemm_plan)
    monkeypatch.setattr(fft_helpers, "make_flat_k_fftn", flat_k_fftn)
    w_isdf._chi_minimax_kernel_cache.clear()


def _fixture(mesh, *, metal):
    rng = np.random.default_rng(20260917)
    nk, nb, ns, nmu = 3, 4, 2, 4
    psi = rng.normal(size=(nk, nb, ns, nmu)) + 1j * rng.normal(size=(nk, nb, ns, nmu))
    enk = np.array([[-1.3, -0.4, 0.2, 1.1], [-1.1, -0.2, 0.5, 1.4], [-1.4, -0.1, 0.7, 1.2]])
    if metal:
        mu, kT = 0.1, 0.15
    else:
        mu, kT = 0.05, 0.02
    f = np.exp(-np.logaddexp(0.0, (enk - mu) / kT))
    wfns = Wavefunctions(
        psi_mun=_put(psi.transpose(0, 2, 3, 1), mesh, PSI_MUN_SPEC),
        psi_nmu=_put(psi, mesh, PSI_NMU_SPEC),
        enk=_put(enk, mesh, P(None, None)), occ=_put(f, mesh, P(None, None)),
        slices=BandSlices.from_band_edges(0, 0, 2, nb, nb), layout="face")
    state = SimpleNamespace(f_kn=f, mu_ry=mu, smearing_family="fd", smearing_width_ry=kT)
    return psi, enk, f, 1.0 / kT, wfns, state


def _lehmann(psi, enk, f, beta, z):
    """Incumbent orientation: sum (f_a(k) - f_b(k-q)) / (z + e_a(k) - e_b(k-q)) M M^H / sqrt(nk)."""
    nk, nb, _, nmu = psi.shape
    out = np.zeros((nk, nmu, nmu), np.complex128)
    for q in range(nk):
        for k in range(nk):
            kmq = (k - q) % nk
            for a in range(nb):
                for b in range(nb):
                    de = enk[k, a] - enk[kmq, b]
                    if z == 0 and abs(de) < 1e-14:
                        w = -beta * f[k, a] * (1.0 - f[k, a])
                    else:
                        w = (f[k, a] - f[kmq, b]) / (z + de)
                    M = np.einsum("sm,sm->m", psi[k, a], np.conj(psi[kmq, b]))
                    out[q] += w * np.outer(M, np.conj(M))
    return out / np.sqrt(float(nk))


@pytest.mark.parametrize("metal", [True, False])
def test_matsubara_chi0_matches_the_lehmann_sum(monkeypatch, metal):
    _stand_ins(monkeypatch)
    mesh = _mesh_xy()
    psi, enk, f, beta, wfns, state = _fixture(mesh, metal=metal)
    meta = SimpleNamespace(nkx=3, nky=1, nkz=1, nk_tot=3)
    indices = (0, 1, 3)
    got = w_isdf.compute_chi0_matsubara(
        wfns, meta, mesh, occupation_state=state, nu_indices=indices, rel_tol=1e-10)
    for n, value in zip(indices, got):
        want = _lehmann(psi, enk, f, beta, 1j * 2 * np.pi * n / beta)
        np.testing.assert_allclose(np.asarray(jax.device_get(value)), want,
                                   rtol=1e-8, atol=1e-8 * np.abs(want).max())
    # One index returns one array, identical to that index of a multi-index sweep.
    single = w_isdf.compute_chi0_matsubara(
        wfns, meta, mesh, occupation_state=state, nu_indices=(1,), rel_tol=1e-10)
    want = _lehmann(psi, enk, f, beta, 1j * 2 * np.pi / beta)
    np.testing.assert_allclose(np.asarray(jax.device_get(single)), want,
                               rtol=1e-8, atol=1e-8 * np.abs(want).max())


def test_matsubara_chi0_refuses_what_it_cannot_represent(monkeypatch):
    _stand_ins(monkeypatch)
    mesh = _mesh_xy()
    _, _, f, _, wfns, state = _fixture(mesh, metal=True)
    meta = SimpleNamespace(nkx=3, nky=1, nkz=1, nk_tot=3)
    calls = {
        "chi0_matsubara_needs_fermi_dirac": dict(occupation_state=SimpleNamespace(
            f_kn=f, mu_ry=state.mu_ry, smearing_family="mp1", smearing_width_ry=state.smearing_width_ry)),
        "chi0_matsubara_occupations": dict(occupation_state=SimpleNamespace(
            f_kn=f + 1e-7, mu_ry=state.mu_ry, smearing_family="fd", smearing_width_ry=state.smearing_width_ry)),
        "chi0_matsubara_vertex": dict(occupation_state=state, vertex="current"),
        "chi0_matsubara_occupation_extent": dict(occupation_state=SimpleNamespace(
            f_kn=f[:, :3], mu_ry=state.mu_ry, smearing_family="fd", smearing_width_ry=state.smearing_width_ry)),
    }
    for gate, kwargs in calls.items():
        with pytest.raises(ValueError, match=gate):
            w_isdf.compute_chi0_matsubara(wfns, meta, mesh, nu_indices=(0,), rel_tol=1e-8, **kwargs)


def test_matsubara_indices_follow_the_tier_count():
    recipe = shared_pole_recipe.shared_real_pole_v1_r3b
    beta, width = 100.0, 11.0
    n_top = int(np.ceil(width * beta / (2 * np.pi)))
    got = shared_pole_recipe.matsubara_indices(beta, width, "production")
    count = shared_pole_recipe.imaginary_sample_count(float(n_top), "production", recipe)
    assert got[0] == 0 and got[1] == 1 and got[-1] == n_top
    assert np.all(np.diff(got) > 0) and 1 < got.size - 1 <= count
    relaxed = shared_pole_recipe.matsubara_indices(beta, width, "relaxed")
    assert relaxed.tolist() == [0, 1, n_top]
    # The shared-pole imaginary ladder keeps its historical count expression.
    kappa = 37.5
    legacy = max(recipe["imaginary_min_count"], round(
        np.log(16 * kappa**2) * np.log(4 / recipe["imaginary_count_epsilon"]) / (2 * np.pi**2)))
    assert shared_pole_recipe.imaginary_sample_count(kappa, "production", recipe) == legacy
    with pytest.raises(ValueError, match="matsubara_indices"):
        shared_pole_recipe.matsubara_indices(0.0, width, "production")

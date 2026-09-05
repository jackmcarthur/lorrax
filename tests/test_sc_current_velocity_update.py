"""The Sigma-velocity of the per-channel current carriers in the SC loop.

``psi_S^(a) += (alpha/4) sigma^a sum_m psi_L,m (v_Delta)^a_mn`` on each
channel carrier (``wavefunction_bundle.add_covariant_current_velocity``),
the interband dipole route for ``v_Delta`` (``qsgw_head.
interband_dipole_velocity_correction``), the per-channel rotation, and the
deck key.  Single device; the kernels are the production ones.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.bispinor_init import HALFALPHA
from common.gamma_matrices import paulis
from gw.wavefunction_bundle import (
    BandSlices, LorentzCarriers, add_covariant_current_velocity,
    build_wavefunctions, build_wavefunctions_face, rotate_lorentz_carriers,
    rotate_wavefunctions,
)


def _mesh():
    return Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("x", "y"))


def _crand(rng, *shape):
    return (rng.standard_normal(shape)
            + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def _bundles(mesh, rng, *, nk=2, nb=8, ns=4, n_rmu=6, edges=(0, 2, 3, 6)):
    psi = _crand(rng, nk, nb, ns, n_rmu)
    rep4 = NamedSharding(mesh, P(None, None, None, None))
    psi_rmu_Y = jax.device_put(jnp.asarray(psi), rep4)
    psi_rmuT_X = jax.device_put(
        jnp.asarray(np.conj(psi).transpose(0, 3, 1, 2)), rep4)
    enk = jax.device_put(
        jnp.asarray(np.sort(rng.standard_normal((nk, nb)), axis=1)),
        NamedSharding(mesh, P(None, None)))
    b0, b1, b2, b3 = edges
    slices = BandSlices.from_band_edges(b0, b1, b2, b3, nb + b0)
    with mesh:
        legacy = build_wavefunctions(
            psi_rmu_Y, psi_rmuT_X, enk_full=enk, slices=slices, mesh_xy=mesh)
        face = build_wavefunctions_face(
            psi_rmu_Y, psi_rmuT_X, enk_full=enk, slices=slices, mesh_xy=mesh)
    return psi, legacy, face


def _expected(psi, v, a, nb_m):
    """small(n) += (alpha/4) sigma^a sum_{m<nb_m} psi_L,m v_mn for n < nb_m."""
    out = psi.copy()
    rot = np.einsum("kmsu,kmn->knsu", psi[:, :nb_m, :2, :], v[a - 1])
    out[:, :nb_m, 2:, :] += 0.5 * HALFALPHA * np.einsum(
        "st,kntu->knsu", np.asarray(paulis[a - 1]), rot)
    return out


@pytest.mark.parametrize("layout", ["legacy", "face"])
def test_add_covariant_current_velocity_matches_explicit_sum(layout):
    if layout == "face" and jax.default_backend() not in ("gpu", "cuda"):
        pytest.skip("the face layout's planned GEMM has a GPU kernel only")
    mesh = _mesh()
    rng = np.random.default_rng(905)
    nk, nb, n_rmu, nb_m = 2, 8, 6, 5
    psi, legacy, face = _bundles(mesh, rng, nk=nk, nb=nb, n_rmu=n_rmu)
    v = _crand(rng, 3, nk, nb_m, nb_m)
    v = v + np.conj(np.swapaxes(v, -1, -2))          # Hermitian like D Delta H
    v_dev = jax.device_put(jnp.asarray(v),
                           NamedSharding(mesh, P(None, None, None, None)))
    for wfns in ((legacy,) if layout == "legacy" else (face,)):
        # Three distinct channel carriers, as the velocity balance builds them.
        carriers = LorentzCarriers(tuple(
            type(wfns)(**{f.name: getattr(wfns, f.name)
                          for f in wfns.__dataclass_fields__.values()})
            for _ in range(3)))
        out = add_covariant_current_velocity(carriers, v_dev, mesh_xy=mesh)
        assert isinstance(out, LorentzCarriers) and not out.one_carrier
        for a in (1, 2, 3):
            ch = out.channel(a)
            want = _expected(psi, v, a, nb_m)          # (nk, n, s, mu)
            if wfns.layout == "legacy":
                got = {
                    "psi_xn": np.asarray(ch.psi_xn).transpose(0, 3, 1, 2),
                    "psi_xr": np.asarray(ch.psi_xr),
                    "psi_yr": np.asarray(ch.psi_yr),
                    "psi_yn": np.asarray(ch.psi_yn).transpose(0, 3, 1, 2),
                }
            else:
                got = {
                    "psi_nmu": np.asarray(ch.psi_nmu),
                    "psi_mun": np.asarray(ch.psi_mun).transpose(0, 3, 1, 2),
                }
            for name, arr in got.items():
                np.testing.assert_allclose(
                    arr, want, rtol=0.0, atol=1e-13, err_msg=f"{name} a={a}")
            # Large components and the bands beyond the manifold are untouched.
            np.testing.assert_array_equal(want[:, :, :2, :], psi[:, :, :2, :])
            np.testing.assert_array_equal(want[:, nb_m:], psi[:, nb_m:])
        # Different channels differ (sigma^a differ): no channel leaked.
        assert np.max(np.abs(np.asarray(out.channel(1).enk)
                             - np.asarray(out.channel(2).enk))) == 0.0
        c1 = np.asarray(out.channel(1).psi_yr if wfns.layout == "legacy"
                        else out.channel(1).psi_nmu)
        c2 = np.asarray(out.channel(2).psi_yr if wfns.layout == "legacy"
                        else out.channel(2).psi_nmu)
        assert np.max(np.abs(c1 - c2)) > 1e-6
    with pytest.raises(ValueError, match="per_channel_carriers"):
        add_covariant_current_velocity(
            LorentzCarriers.shared(legacy), v_dev, mesh_xy=mesh)


def test_lorentz_carriers_rotate_per_distinct_channel_and_refuse_bare_rotation():
    mesh = _mesh()
    rng = np.random.default_rng(906)
    nk, nb = 2, 8
    _, legacy, _ = _bundles(mesh, rng, nk=nk, nb=nb)
    na = legacy.slices.sigma.stop - (legacy.slices.sigma.start or 0)
    A = _crand(rng, nk, na, na)
    E, U = np.linalg.eigh(A + np.conj(np.swapaxes(A, -1, -2)))
    U = jax.device_put(jnp.asarray(U), NamedSharding(mesh, P(None, None, None)))
    E = jax.device_put(jnp.asarray(E), NamedSharding(mesh, P(None, None)))
    shared = LorentzCarriers.shared(legacy)
    with pytest.raises(TypeError, match="rotate_lorentz_carriers"):
        rotate_wavefunctions(shared, U, enk_active_new=E, efermi=0.0,
                             mesh_xy=mesh)
    out = rotate_lorentz_carriers(shared, U, enk_active_new=E, efermi=0.0,
                                  mesh_xy=mesh)
    assert out.one_carrier                   # one rotation, still shared
    ref = rotate_wavefunctions(legacy, U, enk_active_new=E, efermi=0.0,
                               mesh_xy=mesh)
    np.testing.assert_array_equal(np.asarray(out.channel(3).psi_yr),
                                  np.asarray(ref.psi_yr))


def test_interband_dipole_velocity_is_the_scissor_renormalised_velocity():
    """Delta H = diag(delta): -i[r, Delta]_mn = v_mn (delta_n - delta_m)/(e_n - e_m)
    on resolved pairs, zero on degenerate ones; a rotated Delta gives the
    explicit commutator."""
    from gw.qsgw_head import interband_dipole_velocity_correction
    mesh = _mesh()
    rng = np.random.default_rng(907)
    nk, nb = 2, 6
    e = np.sort(rng.standard_normal((nk, nb)), axis=1)
    e[:, 3] = e[:, 2]                                   # one degenerate pair
    v = _crand(rng, 3, nk, nb, nb)
    v = v + np.conj(np.swapaxes(v, -1, -2))
    delta_diag = rng.standard_normal((nk, nb)) * 0.1
    d = np.zeros((nk, nb, nb), complex)
    for k in range(nk):
        d[k] = np.diag(delta_diag[k])
    put = lambda a, spec: jax.device_put(jnp.asarray(a), NamedSharding(mesh, spec))
    got = np.asarray(interband_dipole_velocity_correction(
        put(d, P(None, None, None)), put(v, P(None, None, None, None)),
        put(e, P(None, None)), mesh=mesh, degeneracy_tol_ry=1e-6))
    de = e[:, :, None] - e[:, None, :]
    ratio = np.where(np.abs(de) > 1e-6,
                     (delta_diag[:, None, :] - delta_diag[:, :, None])
                     / np.where(np.abs(de) > 1e-6, -de, 1.0), 0.0)
    want = v * ratio[None]
    np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-13)
    assert np.all(got[:, :, 2, 3] == 0.0) and np.all(got[:, :, 3, 2] == 0.0)
    # Rotated Delta: the explicit interband commutator.
    A = _crand(rng, nk, nb, nb)
    d_rot = A + np.conj(np.swapaxes(A, -1, -2))
    got = np.asarray(interband_dipole_velocity_correction(
        put(d_rot, P(None, None, None)), put(v, P(None, None, None, None)),
        put(e, P(None, None)), mesh=mesh, degeneracy_tol_ry=1e-6))
    r = np.where(np.abs(de) > 1e-6, -1j * v / np.where(np.abs(de) > 1e-6, de, 1.0)[None], 0.0)
    want = -1j * (np.einsum("akml,kln->akmn", r, d_rot)
                  - np.einsum("kml,akln->akmn", d_rot, r))
    np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-12)


def test_deck_key_parses_and_refuses(tmp_path):
    from gw.gw_config import LorraxConfig
    base = ("[cohsex]\nnval = 2\nncond = 2\nnband = 10\n"
            "memory_per_device_gb = 4.0\n")
    quiet = dict(print_fn=lambda *a, **k: None)
    deck = tmp_path / "a.in"
    deck.write_text(base)
    assert LorraxConfig.from_input_file(
        str(deck), **quiet).sc.current_velocity_update == "auto"
    deck.write_text(base + "sc_current_velocity_update = Interband\n")
    assert LorraxConfig.from_input_file(
        str(deck), **quiet).sc.current_velocity_update == "interband"
    deck.write_text(base + "sc_current_velocity_update = links\n")
    with pytest.raises(ValueError, match="sc_current_velocity_update"):
        LorraxConfig.from_input_file(str(deck), **quiet)

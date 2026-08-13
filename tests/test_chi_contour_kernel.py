"""Complex-contour chi keeps the production FFT contraction and sharding."""

from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
jax.config.update("jax_enable_x64", True)

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

from gw import w_isdf  # noqa: E402
from gw.wavefunction_bundle import (  # noqa: E402
    BandSlices,
    PSI_XN_SPEC,
    PSI_XR_SPEC,
    PSI_YN_SPEC,
    PSI_YR_SPEC,
    Wavefunctions,
)


def _mesh_xy():
    devices = np.asarray(jax.devices("cpu"), dtype=object)
    if devices.size >= 4:
        devices = devices[:4].reshape(2, 2)
    elif devices.size >= 2:
        devices = devices[:2].reshape(1, 2)
    else:
        devices = devices[:1].reshape(1, 1)
    return Mesh(devices, ("x", "y"))


def _put(a, mesh, spec):
    return jax.device_put(jnp.asarray(a), NamedSharding(mesh, spec))


def _emulated_flat_k_fftn(mesh, kgrid, spec, *, norm="ortho",
                          out_spec=None):
    from common.fft_helpers import make_sharded_fftn_3d

    assert out_spec is None or out_spec == spec
    fft3 = make_sharded_fftn_3d(
        mesh, spec, spec, axes=(0, 1, 2), norm=norm)

    def flat(x):
        return fft3(jnp.reshape(x, tuple(kgrid) + x.shape[1:])).reshape(x.shape)

    return flat


def _toy(mesh):
    rng = np.random.default_rng(20260811)
    nk, nv, nc, ns, nmu = 2, 2, 2, 2, 4
    psi = (rng.normal(size=(nk, nv + nc, ns, nmu))
           + 1j * rng.normal(size=(nk, nv + nc, ns, nmu)))
    enk = np.array([
        [-1.20, -0.50, 0.70, 1.50],
        [-1.10, -0.40, 0.90, 1.70],
    ])
    slices = BandSlices.from_band_edges(0, 0, nv, nv + nc, nv + nc)
    wfns = Wavefunctions(
        psi_xn=_put(psi.transpose(0, 2, 3, 1), mesh, PSI_XN_SPEC),
        psi_xr=_put(psi, mesh, PSI_XR_SPEC),
        psi_yr=_put(psi, mesh, PSI_YR_SPEC),
        psi_yn=_put(psi.transpose(0, 2, 3, 1), mesh, PSI_YN_SPEC),
        enk=_put(enk, mesh, P(None, None)),
        occ=_put(np.zeros_like(enk), mesh, P(None, None)),
        slices=slices,
    )
    return psi, enk, slices, wfns


def _direct_node_sum(psi, enk, slices, tau, alpha_rows, *, wrong_time=False):
    nk, _, _, nmu = psi.shape
    psi_v, psi_c = psi[:, slices.val], psi[:, slices.cond]
    eps_v, eps_c = enk[:, slices.val], enk[:, slices.cond]
    vmax, cmin = np.max(eps_v), np.min(eps_c)
    out = np.zeros((alpha_rows.shape[0], nk, nmu, nmu), np.complex128)
    tau_c = np.conj(tau) if wrong_time else tau
    for q in range(nk):
        for k in range(nk):
            kmq = (k - q) % nk
            for c in range(psi_c.shape[1]):
                dc = eps_c[kmq, c] - cmin
                for v in range(psi_v.shape[1]):
                    dv = vmax - eps_v[k, v]
                    M = np.einsum(
                        "sm,sm->m", np.conj(psi_c[kmq, c]), psi_v[k, v])
                    node_value = np.exp(-tau * dv - tau_c * dc)
                    for row in range(alpha_rows.shape[0]):
                        out[row, q] += np.dot(
                            alpha_rows[row], node_value) * np.outer(
                                M, np.conj(M))
    return out / np.sqrt(float(nk))


def test_complex_contour_matches_direct_k_minus_q_sum(monkeypatch):
    import common.fft_helpers as fft_helpers

    monkeypatch.setattr(
        fft_helpers, "make_flat_k_fftn", _emulated_flat_k_fftn)
    mesh = _mesh_xy()
    psi, enk, slices, wfns = _toy(mesh)
    tau = np.array([0.21 + 0.16j, 0.44 - 0.09j, 0.73 + 0.12j])
    signs = np.array([+1, -1, +1])
    z = np.array([0.25 + 0.08j, 0.70 + 0.12j])
    weights = np.array([
        [0.31 + 0.07j, 0.22 - 0.05j, 0.18 + 0.03j],
        [0.27 - 0.02j, 0.00 + 0.00j, 0.24 + 0.06j],
    ])
    got = w_isdf.compute_chi0_contour(
        wfns, tau, weights, signs, z,
        SimpleNamespace(nkx=2, nky=1, nkz=1), mesh)
    got = np.stack([np.asarray(jax.device_get(x)) for x in got])
    E_gap = np.min(enk[:, slices.cond]) - np.max(enk[:, slices.val])
    alpha = w_isdf._chi0_contour_alpha_rows(
        tau, weights, signs, z, E_gap)
    want = _direct_node_sum(psi, enk, slices, tau, alpha)
    np.testing.assert_allclose(got, want, rtol=2e-13, atol=2e-13)

    tau_real = np.real(tau).astype(np.complex128)
    np.testing.assert_array_equal(
        _direct_node_sum(psi, enk, slices, tau_real, alpha),
        _direct_node_sum(psi, enk, slices, tau_real, alpha, wrong_time=True))
    assert np.max(np.abs(
        want - _direct_node_sum(
            psi, enk, slices, tau, alpha, wrong_time=True))) > 1e-3


def test_contour_rows_encode_both_resolvent_signs():
    tau0 = 0.37 + 0.11j
    tau = np.array([tau0, tau0])
    weights = np.array([
        [0.42 - 0.09j, 0.31 + 0.04j],
        [0.42 - 0.09j, 0.42 - 0.09j],
    ])
    signs = np.array([+1, -1])
    z = np.array([0.63 + 0.08j, 0.0])
    got = w_isdf._chi0_contour_alpha_rows(tau, weights, signs, z, 1.20)
    np.testing.assert_allclose(
        got[0, 0], -weights[0, 0] * np.exp(-tau0 * (1.20 - z[0])))
    np.testing.assert_allclose(
        got[0, 1], -weights[0, 1] * np.exp(-tau0 * (1.20 + z[0])))
    np.testing.assert_allclose(
        np.sum(got[1]), -2.0 * weights[1, 0] * np.exp(-tau0 * 1.20))

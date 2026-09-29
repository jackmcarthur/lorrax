"""The face Green route builds G and its conjugate-face partner from one panel exchange.

build_G_parents on a face GEMM (backend not 'local') with complex phases and an
antiunitary row returns (G, partner) from panel_matmul's stacked weights.  Both must
equal the two separate local builds of the Green equation, G = psi diag(w) psi^dagger
and partner = conj(psi) diag(w) conj(psi)^dagger, on an emulated 2x2 mesh.
"""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from lxkit.testing import require_devices

from common.collectives import device_put_process_local
from gw.greens_function_kernel import build_G_parents


def test_face_pair_matches_two_local_builds():
    platform = jax.default_backend()
    require_devices(4, platform)
    mesh = Mesh(np.asarray(jax.devices(platform)[:4]).reshape(2, 2), ("x", "y"))
    rng = np.random.default_rng(5)
    nk, nb, mu, ns = 2, 8, 4, 2
    cpx = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)
    psi_mun = cpx(nk, ns, mu, nb)          # (k, s, mu_X, n_Y)
    psi_nmu = cpx(nk, nb, ns, mu)          # (k, n_X, s, mu_Y)
    phases = cpx(nk, nb)
    lo, hi = np.asarray([1, 0], np.int32), np.asarray([7, 8], np.int32)
    plan = SimpleNamespace(sym_idx=np.array([0, 1]), n_sym_spatial=1, mesh_xy=mesh, n_full=nk)
    gemm = SimpleNamespace(backend="cublasmp", mesh=mesh)
    put = lambda x, spec: device_put_process_local(x, NamedSharding(mesh, spec))
    pg = build_G_parents(put(psi_mun, P(None, None, "x", "y")), put(psi_nmu, P(None, "x", None, "y")),
                         phases=put(phases, P()), gemm=gemm, k_unfold_plan=plan,
                         band_range=(put(lo, P()), put(hi, P())))
    live = (np.arange(nb)[None] >= lo[:, None]) & (np.arange(nb)[None] < hi[:, None])
    w = np.where(live, phases, 0)
    # merge_spin_centroid's centroid-major order: (k, mu*s, n) and (k, n, mu*s)
    A = np.swapaxes(psi_mun, 1, 2).reshape(nk, mu * ns, nb)
    B = np.conj(np.swapaxes(psi_nmu, 2, 3)).reshape(nk, nb, mu * ns)
    want_g = (A * w[:, None, :]) @ B
    want_t = (np.conj(A) * w[:, None, :]) @ np.conj(B)
    got_g = np.asarray(jax.device_get(pg.G)).reshape(nk, mu * ns, mu * ns)
    got_t = np.asarray(jax.device_get(pg.transpose)).reshape(nk, mu * ns, mu * ns)
    np.testing.assert_allclose(got_g, want_g, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got_t, want_t, rtol=1e-12, atol=1e-12)
    assert not np.allclose(got_t, np.conj(got_g))   # complex phases: a genuine partner

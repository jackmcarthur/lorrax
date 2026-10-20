"""BSE refit ζ (bse.vq_interp._refit_kernels): C = XᴴX and ζ = B(BᴴZ) on the face, B the
producer's charge factor (isdf.core._factor_c_q_face), equal to the whole-tile dense solve on the
logical block, pad rows zero, C and B tiled 1/P per device, no whole n x n in the GEMM programs.
n_μ = 33 does not tile the 2x2 mesh (carrier 36); 36 is the V8 carrier with 3 inert centroids."""
import re

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

NK, NB, NS, NLOG, R, RCOND = 1, 8, 1, 33, 10, 1e-8


@pytest.mark.parametrize("n_mu", [33, 36])
def test_refit_zeta_on_the_face_is_the_dense_solve(n_mu, monkeypatch):
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 host devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    from bse import vq_interp as vq
    from common import gpu_utils
    from isdf import core
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))
    rooms = {"BSE refit GEMM": 1 << 10}                     # the GEMMs on the face SUMMA (1 KiB)
    monkeypatch.setattr(gpu_utils, "shared_device_room", lambda stage: rooms.get(stage, 1 << 34))
    vq._REFIT_KERNELS.clear()
    g = np.random.default_rng(5)
    cz = lambda *s: g.standard_normal(s) + 1j * g.standard_normal(s)
    psi_m, psi = cz(NK, NB, NS, n_mu), cz(NK, NB, NS, n_mu)
    psi_m[..., NLOG:] = psi[..., NLOG:] = 0                 # the carrier's pad centroids
    Z = cz(n_mu, R)
    Z[NLOG:] = 0
    x, gram, apply = vq._refit_kernels(NK, NB, NS, n_mu, mesh)
    Xf = x(jnp.asarray(psi_m), jnp.asarray(psi))
    C = gram(Xf)
    B = core._factor_c_q_face(C, mesh, NLOG, zeta_rcond=RCOND)
    zeta = np.asarray(apply(B, jnp.asarray(Z)))[0, :, :R]
    X = np.einsum("kmsu,knsu->kmnu", psi_m.conj(), psi).reshape(-1, n_mu)[:, :NLOG]
    ref, _ = core.solve_zeta_charge_dense(jnp.asarray(X.conj().T @ X), jnp.asarray(Z[:NLOG]),
                                          charge_zeta_solve="rank_truncate", zeta_rcond=RCOND)
    ref = np.asarray(ref)
    assert np.max(np.abs(zeta[:NLOG] - ref)) <= 1e-9 * np.max(np.abs(ref))
    assert not np.any(zeta[NLOG:])
    n = C.shape[-1]
    assert n == 36
    for a in (C, B):                                        # 1/P of the global bytes per device
        assert {s.data.nbytes for s in a.addressable_shards} == {a.nbytes // 4}
    whole = re.compile(rf"c128\[1,{n},{n}\]")
    for fn, args in ((gram, (Xf,)), (apply, (B, jnp.asarray(Z)))):
        text = fn.lower(*args).compile().as_text()
        bad = [l for l in text.splitlines() if whole.search(l.replace(" ", ""))]
        assert not bad, bad[:4]
    vq._REFIT_KERNELS.clear()

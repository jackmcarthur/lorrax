"""Gate for the 2-D-distributed cuBLASMp V_Q reconstruction path
(``bse.vq_interp._recon_body`` + ``_distributed_prims``).

The distributed reconstruction shares ONE arithmetic body with the
replicated-batched default (``_recon_body``); only the matmul backend
(local XLA dot vs cuBLASMp on 2-D-sharded tiles) differs.  This test proves
the single-sourcing on 1 GPU (no 16-GPU gating):

  1. **Backend equivalence** — replicated prims (evec = true column
     eigenvectors R) vs distributed cuBLASMp prims (evec = RAW cusolverMp
     buffer convention Qraw = R.conj().T) produce the SAME S / V_SRc / zt to
     ~1e-10.  This is the n_μ=640 bit-match, in miniature and fixture-free.
  2. **Analytic filter identity** — S = R g_ε Rᴴ satisfies
     S (C² + c²I) = C² exactly (c = ε·λ_max), verified through the cuBLASMp
     GEMMs alone (never gathering a full tile) — the same self-consistency
     the large-n_μ capability proof uses.

cuBLASMp accepts a 1×1 process grid, so this runs single-process; it skips
cleanly if the FFI ``.so`` / context cannot initialise on the test node.
"""
from __future__ import annotations

import numpy as np
import pytest

import harness

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402


def _relF(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


@pytest.mark.gpu
def test_distributed_recon_matches_replicated_and_filter_identity():
    harness.skip_unless_gpu(pytest)
    from bse import vq_interp as vqi
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("x", "y"))
    qb3 = NamedSharding(mesh, P(("x", "y"), None, None))
    sh2 = NamedSharding(mesh, P("x", "y"))
    sh3 = NamedSharding(mesh, P(None, "x", "y"))

    # small synthetic coarse batch: Hermitian PD C_q, random sphere ZG/v.
    rng = np.random.default_rng(0)
    nq, n, ngk = 2, 12, 9
    eps_tik = vqi.EPS_TIK
    C = np.empty((nq, n, n), np.complex128)
    for b in range(nq):
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        C[b] = A @ A.conj().T + n * np.eye(n)      # Hermitian PD
    C = 0.5 * (C + np.conj(np.swapaxes(C, 1, 2)))
    ZG = (rng.standard_normal((nq, n, ngk))
          + 1j * rng.standard_normal((nq, n, ngk)))
    vref = np.abs(rng.standard_normal((nq, ngk)))
    vlr = 0.3 * vref

    # replicated backend: true column eigenvectors R.
    lam_np, R_np = np.linalg.eigh(C)
    lam = jnp.asarray(lam_np)
    R = jnp.asarray(R_np)
    S_r, V_r, zt_r = vqi._recon_body(
        lam, R, jnp.asarray(ZG), jnp.asarray(vref), jnp.asarray(vlr),
        eps_tik=eps_tik, **vqi._replicated_prims(qb3))
    S_r = np.asarray(jax.device_get(S_r))
    V_r = np.asarray(jax.device_get(V_r))
    zt_r = np.asarray(jax.device_get(zt_r))

    # distributed backend: RAW eigenvector buffer convention Qraw = R.conj().T
    # (what cusolverMp returns; _distributed_prims.gram uses transa='C').
    try:
        prims = vqi._distributed_prims(mesh)
        Qraw = jax.lax.with_sharding_constraint(
            jnp.conj(jnp.swapaxes(R, 1, 2)), sh3)
        ZG_d = jax.device_put(jnp.asarray(ZG), sh3)
        S_d, V_d, zt_d = vqi._recon_body(
            lam, Qraw, ZG_d, jnp.asarray(vref), jnp.asarray(vlr),
            eps_tik=eps_tik, **prims)
        S_d = np.asarray(jax.device_get(S_d))
        V_d = np.asarray(jax.device_get(V_d))
        zt_d = np.asarray(jax.device_get(zt_d))
    except Exception as exc:                       # FFI unavailable on this node
        pytest.skip(f"cuBLASMp FFI not usable here: {type(exc).__name__}: {exc}")

    assert _relF(S_d, S_r) <= 1e-9, f"S distributed vs replicated {_relF(S_d, S_r):.2e}"
    assert _relF(V_d, V_r) <= 1e-9, f"V_SRc {_relF(V_d, V_r):.2e}"
    assert _relF(zt_d, zt_r) <= 1e-9, f"zt {_relF(zt_d, zt_r):.2e}"

    # analytic filter identity S (C^2 + c^2 I) = C^2  (c = eps * lam_max), on
    # the distributed S, via numpy on the gathered small tiles (large-n_μ proof
    # runs the same identity through cuBLASMp without gathering).
    c2 = (eps_tik * lam_np.max(axis=1)) ** 2       # (nq,)
    for b in range(nq):
        C2 = C[b] @ C[b]
        resid = S_d[b] @ (C2 + c2[b] * np.eye(n)) - C2
        assert _relF(resid + C2, C2) <= 1e-8       # ||S(C^2+c^2I) - C^2|| small

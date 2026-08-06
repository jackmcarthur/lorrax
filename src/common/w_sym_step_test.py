"""Step-by-step verification of the low_mem symmetric W-solve."""
from __future__ import annotations
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

_DIST = "_LORRAX_JAX_DISTRIBUTED_DONE"
def _init():
    if os.environ.get(_DIST): return
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        init_kwargs = {"local_device_ids": [0]} if cvd and "," not in cvd else {}
        try: jax.distributed.initialize(**init_kwargs)
        except Exception: pass
    os.environ[_DIST] = "1"
_init()

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import multihost_utils
from ffi.cusolvermp import (batched_distributed_cholesky,
                             batched_distributed_potrs,
                             cholesky_handle_to_natural_L)


def _log(s):
    if jax.process_index() == 0: print(s, flush=True)


def main():
    nq, n = 1, 64
    Px, Py = 2, 2
    mesh = Mesh(np.asarray(jax.devices()).reshape(Px, Py), axis_names=("x","y"))
    sh = NamedSharding(mesh, P(None, "x", "y"))
    dtype = jnp.complex128

    @jax.jit
    def make_inputs():
        k = jax.random.key(0)
        kr, ki, kc1, kc2 = jax.random.split(k, 4)
        Vr = jax.random.normal(kr, (nq, n, n), dtype=jnp.float64)
        Vi = jax.random.normal(ki, (nq, n, n), dtype=jnp.float64)
        Vbase = (Vr + 1j*Vi).astype(dtype)
        V = Vbase @ jnp.conj(jnp.swapaxes(Vbase, -1, -2)) + (n*2.0) * jnp.eye(n, dtype=dtype)[None]
        V = 0.5 * (V + jnp.conj(jnp.swapaxes(V, -1, -2)))
        Cr = jax.random.normal(kc1, (nq, n, n), dtype=jnp.float64)
        Ci = jax.random.normal(kc2, (nq, n, n), dtype=jnp.float64)
        Cbase = (Cr + 1j*Ci).astype(dtype)
        Cpd = Cbase @ jnp.conj(jnp.swapaxes(Cbase, -1, -2))
        Cpd = 0.5 * (Cpd + jnp.conj(jnp.swapaxes(Cpd, -1, -2)))
        chi = -0.01 * Cpd
        return (jax.lax.with_sharding_constraint(V, sh),
                jax.lax.with_sharding_constraint(chi, sh))

    V_q, chi_q = make_inputs()
    V_np = np.asarray(multihost_utils.process_allgather(V_q))
    C_np = np.asarray(multihost_utils.process_allgather(chi_q))

    # Numpy reference per q.
    W_ref = np.empty((nq, n, n), dtype=np.complex128)
    for q in range(nq):
        W_ref[q] = np.linalg.solve(np.eye(n) - V_np[q] @ C_np[q], V_np[q])

    # JAX distributed pipeline step-by-step.
    X_handle = batched_distributed_cholesky(V_q, mesh=mesh)
    X = cholesky_handle_to_natural_L(X_handle)
    X_np = np.asarray(multihost_utils.process_allgather(X))

    # Check X X† = V on gathered.
    if jax.process_index() == 0:
        for q in range(nq):
            recon = X_np[q] @ X_np[q].conj().T
            err = np.max(np.abs(recon - V_np[q])) / max(np.max(np.abs(V_np[q])), 1.0)
            _log(f"q={q} step-1 |X X^† - V|/|V| = {err:.3e}")

    X_dagger = jnp.conj(jnp.swapaxes(X, -1, -2))
    Xd_np = np.asarray(multihost_utils.process_allgather(X_dagger))

    T = X_dagger @ chi_q @ X
    T_np = np.asarray(multihost_utils.process_allgather(T))
    if jax.process_index() == 0:
        for q in range(nq):
            T_ref = Xd_np[q] @ C_np[q] @ X_np[q]
            err = np.max(np.abs(T_np[q] - T_ref)) / max(np.max(np.abs(T_ref)), 1.0)
            _log(f"q={q} step-2 |T_jax - T_np|/|T_np| = {err:.3e}")
            is_herm = np.max(np.abs(T_np[q] - T_np[q].conj().T))
            _log(f"q={q} step-2 |T - T^†| = {is_herm:.3e}")

    I_q = jnp.broadcast_to(jnp.eye(n, dtype=dtype)[None,:,:], (nq, n, n))
    H = jax.lax.with_sharding_constraint(I_q - T, sh)
    H_np = np.asarray(multihost_utils.process_allgather(H))

    L_H_handle = batched_distributed_cholesky(H, mesh=mesh)
    L_H = cholesky_handle_to_natural_L(L_H_handle)
    LH_np = np.asarray(multihost_utils.process_allgather(L_H))
    if jax.process_index() == 0:
        for q in range(nq):
            recon = LH_np[q] @ LH_np[q].conj().T
            err = np.max(np.abs(recon - H_np[q])) / max(np.max(np.abs(H_np[q])), 1.0)
            _log(f"q={q} step-3 |L_H L_H^† - H|/|H| = {err:.3e}")

    # Cross-check via JAX native solve BEFORE potrs (which donates B).
    Y_jaxsolve = jnp.linalg.solve(H, X_dagger)
    Y_js_np = np.asarray(multihost_utils.process_allgather(Y_jaxsolve))
    if jax.process_index() == 0:
        for q in range(nq):
            Y_ref = np.linalg.solve(H_np[q], Xd_np[q])
            err = np.max(np.abs(Y_js_np[q] - Y_ref)) / max(np.max(np.abs(Y_ref)), 1.0)
            _log(f"q={q} step-4b |Y_jaxsolve - H^-1 X^†_np|/|.| = {err:.3e}")

    # Y = H^-1 X^†.  potrs donates its B input.  Rebuild X_dagger first.
    X_dagger_fresh = jnp.conj(jnp.swapaxes(X, -1, -2))
    X_dagger_fresh = jax.lax.with_sharding_constraint(X_dagger_fresh, sh)
    Y = batched_distributed_potrs(L_H_handle, X_dagger_fresh, mesh=mesh)
    Y_np = np.asarray(multihost_utils.process_allgather(Y))
    if jax.process_index() == 0:
        for q in range(nq):
            Y_ref = np.linalg.solve(H_np[q], Xd_np[q])
            err = np.max(np.abs(Y_np[q] - Y_ref)) / max(np.max(np.abs(Y_ref)), 1.0)
            _log(f"q={q} step-4 |Y_jax - H^-1 X^†_np|/|.| = {err:.3e}")

    W = X @ Y
    W_np = np.asarray(multihost_utils.process_allgather(W))
    if jax.process_index() == 0:
        for q in range(nq):
            err = np.max(np.abs(W_np[q] - W_ref[q])) / max(np.max(np.abs(W_ref[q])), 1.0)
            _log(f"q={q} step-5 |W_jax - W_direct|/|.| = {err:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

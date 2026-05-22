"""Verify cholesky_handle_to_natural_L recovers L such that L L^† = A."""
from __future__ import annotations
import os, sys
from runtime import set_default_env, init_jax_distributed, fallback_to_cpu_if_no_gpu_backend
set_default_env()
init_jax_distributed()
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import multihost_utils
from ffi.cusolvermp import (batched_distributed_cholesky,
                             cholesky_handle_to_natural_L)


def _log(s):
    if jax.process_index() == 0: print(s, flush=True)


def main():
    nq, n = 2, 64
    Px, Py = 2, 2
    mesh = Mesh(np.asarray(jax.devices()).reshape(Px, Py), axis_names=("x","y"))
    sh = NamedSharding(mesh, P(None, "x", "y"))
    dtype = jnp.complex128

    @jax.jit
    def make_A():
        k = jax.random.key(0)
        kr, ki = jax.random.split(k)
        Vr = jax.random.normal(kr, (nq, n, n), dtype=jnp.float64)
        Vi = jax.random.normal(ki, (nq, n, n), dtype=jnp.float64)
        V = (Vr + 1j * Vi).astype(dtype)
        A = V @ jnp.conj(jnp.swapaxes(V, -1, -2)) + (n * 2.0) * jnp.eye(n, dtype=dtype)[None]
        A = 0.5 * (A + jnp.conj(jnp.swapaxes(A, -1, -2)))
        return jax.lax.with_sharding_constraint(A, sh)

    A = make_A()
    A_np = np.asarray(multihost_utils.process_allgather(A))

    L_handle = batched_distributed_cholesky(A, mesh=mesh)
    X = cholesky_handle_to_natural_L(L_handle)
    jax.block_until_ready(X)
    X_np = np.asarray(multihost_utils.process_allgather(X))

    if jax.process_index() == 0:
        # Check L L^† == A
        for q in range(nq):
            Xq = X_np[q]
            Aq = A_np[q]
            recon = Xq @ Xq.conj().T
            err = np.max(np.abs(recon - Aq)) / max(np.max(np.abs(Aq)), 1.0)
            upper = np.triu(Xq, k=1)
            print(f"q={q}: |X X^† - A|/|A| = {err:.3e}, "
                  f"max |X upper| = {np.max(np.abs(upper)):.3e}")
        # Reference numpy cholesky
        L_ref = np.linalg.cholesky(A_np[0])
        err_ref = np.max(np.abs(X_np[0] - L_ref)) / max(np.max(np.abs(L_ref)), 1.0)
        print(f"q=0: |X - L_np|/|L_np| = {err_ref:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

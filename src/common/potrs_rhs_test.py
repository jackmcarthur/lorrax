"""Isolate whether batched_distributed_potrs is correct on its own.

Given a matrix A = L L^dagger, solve A X = B for multiple RHS B.
Compares potrs output against numpy's direct solve.
"""
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
                             batched_distributed_potrs)


def _log(s):
    if jax.process_index() == 0: print(s, flush=True)


def main():
    nq, n = 1, 64
    import argparse
    _ap = argparse.ArgumentParser(allow_abbrev=False)
    _ap.add_argument("--nrhs", type=int, default=0, help="0 = n")
    _args, _ = _ap.parse_known_args()
    nrhs = _args.nrhs or n
    Px, Py = 2, 2
    mesh = Mesh(np.asarray(jax.devices()).reshape(Px, Py), axis_names=("x","y"))
    sh = NamedSharding(mesh, P(None, "x", "y"))
    dtype = jnp.complex128

    @jax.jit
    def make():
        k = jax.random.key(1)
        kr, ki, br, bi = jax.random.split(k, 4)
        Mr = jax.random.normal(kr, (nq, n, n), dtype=jnp.float64)
        Mi = jax.random.normal(ki, (nq, n, n), dtype=jnp.float64)
        M = (Mr + 1j*Mi).astype(dtype)
        A = M @ jnp.conj(jnp.swapaxes(M, -1, -2)) + (n*2.0)*jnp.eye(n, dtype=dtype)[None]
        A = 0.5 * (A + jnp.conj(jnp.swapaxes(A, -1, -2)))
        # Build B: Hermitian (same as X_dagger in w_isdf — comes from conj-swap
        # of a lower-tri matrix, so B is upper-tri. But potrs should work
        # on any B; test with generic complex for now).
        Br = jax.random.normal(br, (nq, n, nrhs), dtype=jnp.float64)
        Bi = jax.random.normal(bi, (nq, n, nrhs), dtype=jnp.float64)
        B = (Br + 1j*Bi).astype(dtype)
        return (jax.lax.with_sharding_constraint(A, sh),
                jax.lax.with_sharding_constraint(B, sh))

    A, B = make()
    A_np = np.asarray(multihost_utils.process_allgather(A))
    B_np = np.asarray(multihost_utils.process_allgather(B))

    L_handle = batched_distributed_cholesky(A, mesh=mesh)
    X = batched_distributed_potrs(L_handle, B, mesh=mesh)
    X_np = np.asarray(multihost_utils.process_allgather(X))

    if jax.process_index() == 0:
        for q in range(nq):
            X_ref = np.linalg.solve(A_np[q], B_np[q])
            err = np.max(np.abs(X_np[q] - X_ref)) / max(np.max(np.abs(X_ref)), 1.0)
            _log(f"q={q} |X_potrs - A^-1 B|/|.| = {err:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

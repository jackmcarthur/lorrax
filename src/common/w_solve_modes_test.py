"""Compare solve_w in high_mem and low_mem modes on random V, chi0.

Synthetic inputs: V SPD, chi0 Hermitian negative-semidefinite (like static
response).  Both modes should produce the same W (up to ~1e-13 roundoff).

Usage::
    lxalloc
    LORRAX_MPI_TYPE=pmix lxrun python3 -u -m common.w_solve_modes_test
"""
from __future__ import annotations
import os
import sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

_DIST = "_LORRAX_JAX_DISTRIBUTED_DONE"
def _init():
    if os.environ.get(_DIST):
        return
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        one_gpu = cvd and "," not in cvd
        init_kwargs = {"local_device_ids": [0]} if one_gpu else {}
        try:
            jax.distributed.initialize(**init_kwargs)
        except Exception:
            pass
    os.environ[_DIST] = "1"
_init()

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import multihost_utils
from gw.w_isdf import solve_w
from types import SimpleNamespace


def _log(s):
    if jax.process_index() == 0:
        print(s, flush=True)


def main():
    nq = 2
    n  = 64
    Px, Py = 2, 2
    if jax.process_count() != Px * Py:
        _log(f"world={jax.process_count()} != {Px*Py}")
        return 2

    mesh = Mesh(np.asarray(jax.devices()).reshape(Px, Py),
                axis_names=("x", "y"))
    sh = NamedSharding(mesh, P(None, "x", "y"))
    dtype = jnp.complex128

    @jax.jit
    def make():
        k = jax.random.key(42)
        kv, kc = jax.random.split(k)
        Vr = jax.random.normal(kv, (nq, n, n), dtype=jnp.float64)
        Vi = jax.random.normal(jax.random.key(43), (nq, n, n), dtype=jnp.float64)
        V = (Vr + 1j * Vi).astype(dtype)
        V = V @ jnp.conj(jnp.swapaxes(V, -1, -2)) + (n * 2.0) * jnp.eye(n, dtype=dtype)[None]
        V = 0.5 * (V + jnp.conj(jnp.swapaxes(V, -1, -2)))  # force Hermitian
        Cr = jax.random.normal(kc, (nq, n, n), dtype=jnp.float64)
        Ci = jax.random.normal(jax.random.key(44), (nq, n, n), dtype=jnp.float64)
        C = (Cr + 1j * Ci).astype(dtype)
        C = C @ jnp.conj(jnp.swapaxes(C, -1, -2))
        C = 0.5 * (C + jnp.conj(jnp.swapaxes(C, -1, -2)))
        chi0 = -0.01 * C   # negative semi-def, small scale
        return (jax.lax.with_sharding_constraint(V, sh),
                jax.lax.with_sharding_constraint(chi0, sh))

    V_q, chi0_q = make()
    jax.block_until_ready(V_q); jax.block_until_ready(chi0_q)

    meta = SimpleNamespace(nk_tot=nq, nspin=1, nspinor=1)

    W_hi = solve_w(V_q, chi0_q, meta, mesh, memory_mode="high_mem")
    jax.block_until_ready(W_hi)
    W_lo = solve_w(V_q, chi0_q, meta, mesh, memory_mode="low_mem")
    jax.block_until_ready(W_lo)

    W_hi_full = np.asarray(multihost_utils.process_allgather(W_hi))
    W_lo_full = np.asarray(multihost_utils.process_allgather(W_lo))

    if jax.process_index() == 0:
        diff = np.max(np.abs(W_hi_full - W_lo_full))
        rel  = diff / max(np.max(np.abs(W_hi_full)), 1.0)
        print(f"max |W_hi| = {np.max(np.abs(W_hi_full)):.3e}")
        print(f"max |W_lo| = {np.max(np.abs(W_lo_full)):.3e}")
        print(f"max |W_hi - W_lo| = {diff:.3e}")
        print(f"relative diff = {rel:.3e}")
        tol = 1e-10
        print("PASS" if rel < tol else "FAIL")
        return 0 if rel < tol else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

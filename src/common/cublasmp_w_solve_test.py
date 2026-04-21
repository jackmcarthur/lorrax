"""Correctness test for the fused cuBLASMp + cuSOLVERMp W-solve.

Compares against numpy's direct solve of (I - V chi) W = V.
"""
from __future__ import annotations
import os, sys, argparse, time
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
from ffi.cublasmp import batched_fused_w_solve


def _log(s):
    if jax.process_index() == 0: print(s, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nbatch", type=int, default=2)
    ap.add_argument("-n", type=int, default=64)
    ap.add_argument("--mesh", type=str, default="2x2")
    ap.add_argument("--pref_re", type=float, default=0.67)
    ap.add_argument("--pref_im", type=float, default=0.0)
    args = ap.parse_args()

    Px, Py = map(int, args.mesh.lower().split("x"))
    if Px * Py != jax.process_count():
        _log(f"world={jax.process_count()} != {Px*Py}"); return 2

    mesh = Mesh(np.asarray(jax.devices()).reshape(Px, Py),
                axis_names=("x", "y"))
    sh = NamedSharding(mesh, P(None, "x", "y"))
    dtype = jnp.complex128
    nq, n = args.nbatch, args.n
    pref = complex(args.pref_re, args.pref_im)

    @jax.jit
    def make():
        kv, ki, kc1, kc2 = jax.random.split(jax.random.key(42), 4)
        Vr = jax.random.normal(kv, (nq, n, n), dtype=jnp.float64)
        Vi = jax.random.normal(ki, (nq, n, n), dtype=jnp.float64)
        Vbase = (Vr + 1j * Vi).astype(dtype)
        V = Vbase @ jnp.conj(jnp.swapaxes(Vbase, -1, -2)) + (n * 2.0) * jnp.eye(n, dtype=dtype)[None]
        V = 0.5 * (V + jnp.conj(jnp.swapaxes(V, -1, -2)))
        Cr = jax.random.normal(kc1, (nq, n, n), dtype=jnp.float64)
        Ci = jax.random.normal(kc2, (nq, n, n), dtype=jnp.float64)
        Cbase = (Cr + 1j * Ci).astype(dtype)
        Cpd = Cbase @ jnp.conj(jnp.swapaxes(Cbase, -1, -2))
        Cpd = 0.5 * (Cpd + jnp.conj(jnp.swapaxes(Cpd, -1, -2)))
        chi = -0.01 * Cpd     # Hermitian negative-SD (like static response)
        return (jax.lax.with_sharding_constraint(V, sh),
                jax.lax.with_sharding_constraint(chi, sh))

    V, chi = make()
    jax.block_until_ready(V); jax.block_until_ready(chi)

    # Snapshot inputs before solve (V is donated).
    V_np   = np.asarray(multihost_utils.process_allgather(V))
    chi_np = np.asarray(multihost_utils.process_allgather(chi))

    _log(f"=== fused w_solve: nbatch={nq} n={n} mesh={Px}x{Py} pref={pref} ===")
    t0 = time.perf_counter()
    W = batched_fused_w_solve(V, chi, pref, mesh=mesh)
    jax.block_until_ready(W)
    dt = time.perf_counter() - t0
    _log(f"  wall: {dt*1000:.1f} ms")

    W_np = np.asarray(multihost_utils.process_allgather(W))

    if jax.process_index() == 0:
        # Reference: W_ref = (I - V pref*chi)^{-1} V
        W_ref = np.empty_like(W_np)
        for q in range(nq):
            A = np.eye(n, dtype=np.complex128) - V_np[q] @ (pref * chi_np[q])
            W_ref[q] = np.linalg.solve(A, V_np[q])
        err = np.max(np.abs(W_np - W_ref)) / max(np.max(np.abs(W_ref)), 1.0)
        _log(f"  max |W_ffi - W_direct|/|W| = {err:.3e}")
        tol = 1e-10
        _log(f"  {'PASS' if err < tol else 'FAIL'} at tol {tol:.1e}")
        return 0 if err < tol else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

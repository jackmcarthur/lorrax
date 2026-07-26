"""Correctness test for the isdf_fitting memory_mode abstraction.

Exercises factor_c_q + solve_zeta in both ``high_mem``
(legacy JAX replicate-L vmap trsm) and ``low_mem`` (batched cuSOLVERMp
potrf+potrs on per-X-row sub-comm) modes on the same random HPD input.

Checks:
  1. Each mode returns ``zeta`` satisfying ``C @ zeta ≈ Z``.
  2. The two modes agree to machine precision.

Usage (4 GPUs)::

    lxalloc
    lxrun python3 -u -m common.isdf_zeta_mode_test --nq 8 -n 128 --mrhs 256 --mesh 2x2
"""
from __future__ import annotations

import argparse
import os
import sys
import time

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
from isdf import factor_c_q, solve_zeta


def _log(s):
    if jax.process_index() == 0:
        print(s, flush=True)


def build_hpd_batch(nq, n, seed, mesh):
    """Random (nq, N, N) Hermitian PD, sharded P(None, 'x', 'y')."""
    sh = NamedSharding(mesh, P(None, 'x', 'y'))

    @jax.jit
    def _b():
        k_r, k_i = jax.random.split(jax.random.key(seed), 2)
        a = jax.random.normal(k_r, (nq, n, n), dtype=jnp.float64)
        b = jax.random.normal(k_i, (nq, n, n), dtype=jnp.float64)
        z = (a + 1j * b).astype(jnp.complex128)
        H = 0.5 * (z + jnp.conj(jnp.swapaxes(z, -1, -2)))
        H = H + n * jnp.eye(n, dtype=jnp.complex128)[None, :, :]
        return jax.lax.with_sharding_constraint(H, sh)
    return _b()


def build_rhs_batch(nq, n, mrhs, seed, mesh):
    """Random (nq, N, Mrhs), sharded P(None, 'x', 'y')."""
    sh = NamedSharding(mesh, P(None, 'x', 'y'))

    @jax.jit
    def _b():
        k_r, k_i = jax.random.split(jax.random.key(seed), 2)
        a = jax.random.normal(k_r, (nq, n, mrhs), dtype=jnp.float64)
        b = jax.random.normal(k_i, (nq, n, mrhs), dtype=jnp.float64)
        return jax.lax.with_sharding_constraint(
            (a + 1j * b).astype(jnp.complex128), sh)
    return _b()


def _parse_mesh(s):
    px, py = s.lower().split("x")
    return int(px), int(py)


def _run_mode(mode, A, B, mesh):
    """Factor + solve once; return gathered zeta."""
    t0 = time.perf_counter()
    factor = factor_c_q(A, mesh, memory_mode=mode)
    factor.block_until_ready()
    dt_f = time.perf_counter() - t0

    # Pre-reshard B to what this mode's solve wants (matches
    # fit_zeta_chunked_to_h5's convention at line ~1350).
    if factor.mode == "low_mem":
        z_col_shard = NamedSharding(mesh, P('x', None, 'y'))
    else:
        z_col_shard = NamedSharding(mesh, P(None, None, ('x', 'y')))
    B_col = jax.lax.with_sharding_constraint(B, z_col_shard)
    B_col.block_until_ready()

    t0 = time.perf_counter()
    zeta = solve_zeta(factor, B_col, mesh, q_chunk_size=1024)
    zeta.block_until_ready()
    dt_s = time.perf_counter() - t0

    zeta_full = np.asarray(
        multihost_utils.process_allgather(zeta, tiled=False))
    return zeta_full, dt_f, dt_s, factor.mode


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--nq", type=int, default=8)
    ap.add_argument("-n", type=int, default=128)
    ap.add_argument("--mrhs", type=int, default=0, help="rhs cols; default n")
    ap.add_argument("--mesh", type=str, default="2x2")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    Px, Py = _parse_mesh(args.mesh)
    if Px * Py != jax.process_count():
        _log(f"world={jax.process_count()} != Px*Py={Px*Py}")
        return 2
    mesh = Mesh(np.asarray(jax.devices()).reshape(Px, Py),
                axis_names=("x", "y"))
    mrhs = args.mrhs or args.n

    _log(f"=== isdf zeta-mode test: nq={args.nq} n={args.n} "
         f"mrhs={mrhs} mesh={Px}x{Py} ===")

    A = build_hpd_batch(args.nq, args.n, args.seed, mesh)
    B = build_rhs_batch(args.nq, args.n, mrhs, args.seed + 1, mesh)
    jax.block_until_ready(A); jax.block_until_ready(B)
    multihost_utils.sync_global_devices("inputs_built")

    A_full = np.asarray(multihost_utils.process_allgather(A, tiled=False))
    B_full = np.asarray(multihost_utils.process_allgather(B, tiled=False))

    results = {}
    for mode in ("high_mem", "low_mem"):
        zeta_full, dt_f, dt_s, resolved_mode = _run_mode(mode, A, B, mesh)
        _log(f"  {mode:<9s} (resolved={resolved_mode}):  "
             f"chol {dt_f*1000:7.1f} ms   solve {dt_s*1000:7.1f} ms")

        # Residual: C @ zeta == B ?
        res_max = 0.0
        for q in range(args.nq):
            res = (np.linalg.norm(A_full[q] @ zeta_full[q] - B_full[q])
                   / max(np.linalg.norm(B_full[q]), 1.0))
            res_max = max(res_max, res)
        _log(f"    max |A z - B|/|B| = {res_max:.3e}")
        results[mode] = (zeta_full, res_max)

    if jax.process_index() == 0:
        zeta_h, res_h = results["high_mem"]
        zeta_l, res_l = results["low_mem"]
        tol = 1e-10
        cross = np.max(np.abs(zeta_h - zeta_l))
        rel_cross = cross / max(np.max(np.abs(zeta_h)), 1.0)
        _log(f"  max |zeta_high - zeta_low| = {cross:.3e}  "
             f"(rel {rel_cross:.3e})")
        ok = (res_h < tol and res_l < tol and rel_cross < 1e-9)
        _log(f"  {'PASS' if ok else 'FAIL'} at tol {tol:.0e}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

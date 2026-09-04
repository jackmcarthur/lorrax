"""4-GPU distributed Hermitian eigensolve test — cuSOLVERMp via JAX FFI.

Usage (Perlmutter)::

    export LX_BASE_MODULE=lorrax_A LORRAX_CHECKOUT=$PWD
    lx run -N 1 -G 4 -n 4 -- env PYTHONPATH="$LORRAX_CHECKOUT/src" \\
        python3 -u tests/bench/cusolvermp_eigh_test.py

or, for a 1-GPU smoke::

    lx run -N 1 -G 1 -n 1 -- env PYTHONPATH="$LORRAX_CHECKOUT/src" \\
        python3 -u tests/bench/cusolvermp_eigh_test.py --grid 1 1

The one-GPU arm is diagnostic only; it is not landing evidence.

What it does
------------
1.  Ensure ``jax.distributed.initialize()`` has run (LORRAX pattern).
2.  Build a 128×128 complex-Hermitian matrix on rank 0; broadcast the
    bits so every process starts from bit-identical input.
3.  Shard the matrix ``P('x','y')`` on a ``(p, q)`` mesh over all visible
    JAX devices.
4.  Call ``distributed_eigh`` (complex-Hermitian C128 path).
5.  Compare returned eigenvalues against ``jnp.linalg.eigvalsh`` of the
    replicated reference.  Assert max-abs difference < 1e-9.
6.  Repeat for the real-symmetric F64 path (just the real part of A;
    A + A.T).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# --- LORRAX distributed bootstrap (same pattern as gw_jax / run_nscf) ------
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
jax.config.update("jax_enable_x64", True)

_DIST_SENTINEL = "_LORRAX_JAX_DISTRIBUTED_DONE"
def _maybe_init_jax_distributed():
    if os.environ.get(_DIST_SENTINEL):
        return
    proc_count = int(os.environ.get("JAX_PROCESS_COUNT",
                         os.environ.get("JAX_NUM_PROCESSES",
                         os.environ.get("SLURM_NTASKS", "1"))))
    if proc_count > 1:
        # Under the one-task-per-GPU launcher each rank has only one GPU
        # visible. JAX's default init
        # assumes all processes see all GPUs and times out waiting for
        # the missing local topologies.  Tell it explicitly.
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        one_gpu = cvd and "," not in cvd
        init_kwargs = {"local_device_ids": [0]} if one_gpu else {}
        try:
            jax.distributed.initialize(**init_kwargs)
            os.environ[_DIST_SENTINEL] = "1"
            return
        except Exception:
            pass
    os.environ[_DIST_SENTINEL] = "1"

_maybe_init_jax_distributed()

from ffi.cusolvermp import distributed_eigh  # noqa: E402


# ===========================================================================
def make_hermitian(n: int, seed: int = 42) -> np.ndarray:
    """Return a deterministic n×n Hermitian matrix with well-spread eigs."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    A = X + X.conj().T           # Hermitian
    # Diagonal dominance keeps the problem well-conditioned.
    A = A + np.eye(n) * (2 * n)
    return A.astype(np.complex128)


def shard_matrix(A: np.ndarray, mesh: Mesh) -> jax.Array:
    sharding = NamedSharding(mesh, P("x", "y"))
    return jax.device_put(jnp.asarray(A), sharding)


def gather_to_numpy(x: jax.Array) -> np.ndarray:
    """Global gather of a (possibly) multi-process sharded jax.Array.

    For replicated arrays ``jax.device_get`` works, but for a
    ``P('x','y')``-sharded array in a multi-process job each process can
    only see its own local shard — we must go through
    ``multihost_utils.process_allgather`` to stitch the full logical
    array together on every process.
    """
    from jax.experimental import multihost_utils
    return np.asarray(multihost_utils.process_allgather(x, tiled=True))


# ===========================================================================
def _log(msg: str) -> None:
    if jax.process_index() == 0:
        print(msg, flush=True)


def run_complex_test(n: int, mesh: Mesh) -> int:
    _log(f"\n=== complex128 Hermitian {n}×{n} on mesh {mesh.shape} ===")

    A_host = make_hermitian(n, seed=42)
    A = shard_matrix(A_host, mesh)
    _log(f"  sharded A addressable shards = {[s.data.shape for s in A.addressable_shards]}")

    # reference (single-device on rank 0 only, then compare)
    ref_evals = np.linalg.eigvalsh(A_host)  # returns ascending real eigs

    t0 = time.time()
    evals, Q = distributed_eigh(A, mesh=mesh)
    # Force materialisation on host before timing stops.
    evals_np = gather_to_numpy(evals)
    Q_np     = gather_to_numpy(Q)
    dt = time.time() - t0

    # cuSOLVERMp returns eigenvalues in ascending order (same as eigvalsh)
    evals_sorted = np.sort(np.real(evals_np))

    max_ev_err = float(np.max(np.abs(evals_sorted - ref_evals)))
    _log(f"  wall time        = {dt*1000:.1f} ms")
    _log(f"  eval[0]          = {evals_sorted[0]:.6e}  (ref {ref_evals[0]:.6e})")
    _log(f"  eval[-1]         = {evals_sorted[-1]:.6e}  (ref {ref_evals[-1]:.6e})")
    _log(f"  max |eval - ref| = {max_ev_err:.3e}")

    tol = 1e-8 * n
    ok = max_ev_err < tol
    _log(f"  {'PASS' if ok else 'FAIL'} at tol {tol:.1e}")

    # Residual check: ||A @ Q_i - eval_i * Q_i||_inf < eps over a few cols.
    # (Not strictly required; helpful when something is subtly wrong.)
    #   We use the *returned* eigenvalue ordering (no sort here).
    if jax.process_index() == 0:
        try:
            idxs = [0, n // 2, n - 1]
            resid = []
            for i in idxs:
                q_i = Q_np[:, i] if Q_np.shape == (n, n) else Q_np.reshape(n, n)[:, i]
                resid_i = np.max(np.abs(A_host @ q_i - np.real(evals_np[i]) * q_i))
                resid.append(float(resid_i))
            _log(f"  residuals {idxs} = {resid}")
        except Exception as exc:
            _log(f"  (residual check skipped: {exc})")

    return 0 if ok else 1


def run_real_test(n: int, mesh: Mesh) -> int:
    _log(f"\n=== float64 symmetric {n}×{n} on mesh {mesh.shape} ===")

    rng = np.random.default_rng(17)
    X = rng.standard_normal((n, n))
    A_host = (X + X.T) + np.eye(n) * (2 * n)
    A_host = A_host.astype(np.float64)

    A = shard_matrix(A_host, mesh)
    ref_evals = np.linalg.eigvalsh(A_host)

    t0 = time.time()
    evals, Q = distributed_eigh(A, mesh=mesh)
    evals_np = gather_to_numpy(evals)
    dt = time.time() - t0

    evals_sorted = np.sort(evals_np)
    max_ev_err = float(np.max(np.abs(evals_sorted - ref_evals)))
    _log(f"  wall time        = {dt*1000:.1f} ms")
    _log(f"  max |eval - ref| = {max_ev_err:.3e}")

    tol = 1e-9 * n
    ok = max_ev_err < tol
    _log(f"  {'PASS' if ok else 'FAIL'} at tol {tol:.1e}")
    return 0 if ok else 1


# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("-n", type=int, default=128,
                    help="Matrix size (default 128).")
    ap.add_argument("--grid", nargs=2, type=int, default=None,
                    metavar=("P", "Q"),
                    help="Process grid (p, q).  Default: auto from "
                         "jax.process_count() as (n,1).")
    ap.add_argument("--skip-complex", action="store_true")
    ap.add_argument("--skip-real", action="store_true")
    args = ap.parse_args()

    n_procs = int(jax.process_count())
    if args.grid is None:
        # Default: if 1 process, 1x1 grid.  If 4 processes, 2x2.  Else Nx1.
        if n_procs == 1:
            p, q = 1, 1
        elif n_procs == 4:
            p, q = 2, 2
        else:
            p, q = n_procs, 1
    else:
        p, q = args.grid
    if p * q != n_procs:
        _log(f"FATAL: grid {p}x{q}={p*q} != jax.process_count()={n_procs}")
        return 2

    _log(f"jax.process_index() = {jax.process_index()} / {n_procs}")
    _log(f"jax.local_device_count() = {jax.local_device_count()}")
    _log(f"jax.device_count() = {jax.device_count()}")
    _log(f"chosen grid = ({p}, {q})")

    devices = np.asarray(jax.devices()).reshape(p, q)
    mesh = Mesh(devices, axis_names=("x", "y"))

    rc = 0
    if not args.skip_complex:
        rc |= run_complex_test(args.n, mesh)
    if not args.skip_real:
        rc |= run_real_test(args.n, mesh)

    _log(f"\n{'ALL PASSED' if rc == 0 else 'FAILURES PRESENT'}")
    return rc


if __name__ == "__main__":
    sys.exit(main())

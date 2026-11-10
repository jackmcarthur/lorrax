"""P4 check of the padded cuSOLVERMp LU (``_cusolvermp.lu_layout``).

A 2x2 solve whose tile has no block in [128, 256] (n 16396: tile 8198 = 2*4099, padded
to 8200), against the one-tile-per-rank solve the padding replaces, and the native
getrf+getrs workspace both ways at that tile and at the 60k W Dyson tile (10002 =
2*3*1667, whose one-tile getrs asked 83.5 GB/rank on 6x6).  Workspace figures are
the size query the handler sizes its plan with (``lrx_lu_workspace_bytes``, the
default 2 GiB getrs room); nothing is allocated for them.

Usage (one node, P4)::

    srun -N 1 -n 4 --gpus-per-node=4 src/ffi/cpp/select_gpu.sh \\
        python -u services/distrib_la/bench/lu_pad_check.py -n 16396
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# The one-tile reference's workspace lives outside XLA's pool (cudaMallocAsync).
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")

import cusolvermp_solve_lu_test as base       # noqa: E402  (distributed init, builders)
import jax                                     # noqa: E402
import jax.numpy as jnp                        # noqa: E402
import numpy as np                             # noqa: E402
from jax.sharding import Mesh                  # noqa: E402

from distrib_la import _cusolvermp as cm       # noqa: E402
from distrib_la.workspace import _vendor_query  # noqa: E402

GB = 1e9


def workspace(ctx, n, nrhs, mb, nb, mb_b, nb_b):
    return _vendor_query(ctx, "solve_lu", (n, nrhs, mb, nb, mb_b, nb_b, 0), np.complex128)[0]


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("-n", type=int, default=16396)
    ap.add_argument("--ref-limit-gb", type=float, default=50.0,
                    help="skip the one-tile reference solve above this workspace")
    args = ap.parse_args()
    n, p = args.n, 2
    mesh = Mesh(np.asarray(jax.devices()).reshape(p, p), axis_names=("x", "y"))
    ctx = cm.get_or_init_context(mesh, col_major=False)
    base._log(f"=== lu_pad_check n={n} on 2x2, distrib_la from {cm.__file__} ===")

    one = {}
    for tile in (n // p, 10002):
        side, mb, _ = cm.lu_layout(p * tile, p, p)
        cols, nb_b = cm.rhs_layout(p * tile, p, mb != side // p)
        one[tile] = workspace(ctx, p * tile, p * tile, tile, tile, tile, tile)
        padded = workspace(ctx, side, cols, mb, mb, mb, nb_b)
        base._log(f"  tile {tile}: one tile per rank {one[tile] / GB:.2f} GB/rank -> padded to "
                  f"{side // p} block {mb}: {padded / GB:.2f} GB/rank")

    def solve():
        A = base.build_general_batch(1, n, jnp.complex128, 0, mesh)
        B = base.build_rhs_batch(1, n, n, jnp.complex128, 1, mesh)
        jax.block_until_ready((A, B))
        t = time.perf_counter()
        X = jax.block_until_ready(cm.batched_distributed_solve_lu(A, B, mesh=mesh))
        return X, time.perf_counter() - t

    X, wall = solve()
    A = base.build_general_batch(1, n, jnp.complex128, 0, mesh)
    B = base.build_rhs_batch(1, n, n, jnp.complex128, 1, mesh)
    res = float(jax.jit(lambda a, x, b: jnp.linalg.norm(a @ x - b) / jnp.linalg.norm(b))(A, X, B))
    base._log(f"  padded solve n={n}: {wall:.2f} s (first call, compile included), "
              f"|AX - B|/|B| = {res:.2e}")
    del A, B

    if one[n // p] / GB > args.ref_limit_gb:
        base._log(f"  one-tile reference skipped: {one[n // p] / GB:.1f} GB/rank > {args.ref_limit_gb}")
        return 0 if res < 1e-12 else 1
    cm.lu_layout = lambda n, Px, Py: (n, n // Px, n // Py)
    cm.rhs_layout = lambda nrhs, Py, relabeled: (nrhs, nrhs // Py)
    X_ref, wall_ref = solve()
    diff = float(jax.jit(lambda x, r: jnp.linalg.norm(x - r) / jnp.linalg.norm(r))(X, X_ref))
    base._log(f"  one-tile reference: {wall_ref:.2f} s; |X - X_ref|/|X_ref| = {diff:.2e}")
    ok = res < 1e-12 and diff < 1e-12
    base._log(f"  {'PASS' if ok else 'FAIL'} at 1e-12")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

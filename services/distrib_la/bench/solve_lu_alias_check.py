"""The distributed LU solve on one platform: its residual, and that it writes only what it declares.

    lx run -N 1 -G 4 -n 4 -- python services/distrib_la/bench/solve_lu_alias_check.py         # cuSOLVERMp
    lx run -N 1 -G 4 -n 4 -- python services/distrib_la/bench/solve_lu_alias_check.py --cpu   # ScaLAPACK

Per dtype (F64, C128) on the run's 2x2 mesh:

1. ``batched_distributed_solve_lu``: max_q |A X - B| / |B| below 1e-10.
2. The FFI target, called the way a fused program can reach it: a jit with nothing donated
   hands it the caller's A and B already in its local layout, so no transpose stands between
   the caller's buffers and the handler, and returns A and B as read after the solve
   (``optimization_barrier`` orders the reads after it).  Both reads, and the caller's arrays,
   must equal their copies bit for bit: a write into A or B that the call does not declare as
   an aliased output fails here.  Its X must equal the public solve's bit for bit.

Rank 0 prints one line per dtype and ``ALL PASS`` or ``FAIL``; the exit code follows.
"""
from __future__ import annotations

import argparse
import sys
from functools import partial


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--cpu", action="store_true", help="CPU mesh: the ScaLAPACK handler")
    ap.add_argument("--nq", type=int, default=3)
    ap.add_argument("-n", type=int, default=256)
    args = ap.parse_args()

    import runtime
    stack = runtime.initialize_communicator_stack(platform="cpu" if args.cpu else "gpu")
    import jax
    import numpy as np
    from jax import shard_map
    from jax.experimental.multihost_utils import process_allgather
    from jax.sharding import NamedSharding, PartitionSpec as P

    mesh = stack.mesh
    px, py = int(mesh.shape["x"]), int(mesh.shape["y"])
    nq, n = args.nq, args.n
    if args.cpu:
        from distrib_la import _scalapack as be
        be.ensure_registered(mesh)
        attrs = dict(nq=nq, n=n, nrhs=n, g=n // max(px, py), nb_b=n // py if py > 1 else n,
                     ctx_key=int(be.context_key(mesh)))
    else:
        from distrib_la import _cusolvermp as be
        attrs = dict(nq=nq, n=n, nrhs=n, mb_a=n // px, nb_a=n // py, mb_b=n // px, nb_b=n // py,
                     ctx_key=int(be.context_key(mesh, col_major=False)))

    face = NamedSharding(mesh, P(None, "x", "y"))
    local = NamedSharding(mesh, P(None, "y", "x"))   # shards = the handler's transposed blocks

    def put(h, sharding):
        return jax.make_array_from_callback(h.shape, sharding, lambda idx: h[idx])

    def gather(a):
        return np.asarray(process_allgather(a, tiled=True))

    rng = np.random.default_rng(7)
    failed = False
    for dtype in (np.float64, np.complex128):
        def draw():
            z = rng.standard_normal((nq, n, n))
            return (z + 1j * rng.standard_normal((nq, n, n)) if dtype == np.complex128 else z).astype(dtype)
        A = draw() + 2.0 * n * np.eye(n, dtype=dtype)
        B = draw()
        X = gather(be.batched_distributed_solve_lu(put(A, face), put(B, face), mesh=mesh))
        res = max(np.linalg.norm(A[q] @ X[q] - B[q]) / np.linalg.norm(B[q]) for q in range(nq))

        AT, BT = np.swapaxes(A, 1, 2).copy(), np.swapaxes(B, 1, 2).copy()
        a_in, b_in = put(AT, local), put(BT, local)
        block = jax.ShapeDtypeStruct((nq, n // py, n // px), dtype)

        @jax.jit
        @partial(shard_map, mesh=mesh, in_specs=(P(None, "y", "x"),) * 2,
                 out_specs=(P(None, "y", "x"),) * 3, check_vma=False)
        def raw(a, b):
            x, _ = jax.ffi.ffi_call(be._SOLVE_LU_TARGET, (block, block),
                                    input_output_aliases={0: 1, 1: 0})(a, b, **attrs)
            return jax.lax.optimization_barrier((x, a, b))

        XT, a_read, b_read = (gather(v) for v in raw(a_in, b_in))
        checks = {
            "A read after": np.array_equal(a_read, AT), "B read after": np.array_equal(b_read, BT),
            "caller A": np.array_equal(gather(a_in), AT), "caller B": np.array_equal(gather(b_in), BT),
            "X == public": np.array_equal(np.swapaxes(XT, 1, 2), X), "residual": res < 1e-10,
        }
        ok = all(checks.values())
        failed |= not ok
        runtime.rank0_print(
            f"[solve_lu_alias] {be.__name__} {np.dtype(dtype).name} nq={nq} n={n} mesh={px}x{py}: "
            f"residual {res:.3e}; max|A read - A| {np.abs(a_read - AT).max():.3e}; "
            + ", ".join(f"{k} {'ok' if v else 'FAIL'}" for k, v in checks.items())
            + ("  PASS" if ok else "  FAIL"))
    runtime.rank0_print("[solve_lu_alias] " + ("FAIL" if failed else "ALL PASS"))
    return 1 if failed else 0


if __name__ == "__main__":
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"))
    import runtime
    runtime.run_main_and_finalize(main)

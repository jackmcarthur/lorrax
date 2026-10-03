"""Eigh-stack route bench: the service's route for a face stack against the forced whole mesh.

For each (B, n) a stack of B random Hermitian complex128 matrices at
P(None,'x','y') is solved through ``distrib_la.plan('eigh', backend=
'distributed', budget_bytes=ROOM).batched`` (the service decides: route (c)
in rounds, or the whole mesh) and again with the private whole-mesh override
(``_route='scan'``). Per stack: the decision, wall time of each (one warm-up
call excluded), probe residual/orthogonality (``eigh_errors``), the largest
eigenvalue difference between the two routes relative to max|w|, NaN and
ascending-order checks, and the unchecked route-(c) wall (the check's cost).
``.bin`` (row-major complex128 n x n) or ``.npy`` matrices on the command line
enter as stacks of B copies (FACEGRAM's failing inputs). Exit 1 on any FAIL.

    lx run --jid JID -N 1 -G 4 -n 4 -- bash -c 'source config/perlmutter/gpu_env.sh; \\
        python3 -u services/distrib_la/bench/eigh_stack_route_bench.py [G_fail.bin ...]'
"""
import os
import sys
import time

import numpy as np

from runtime import initialize_communicator_stack, run_main_and_finalize
RUNTIME = initialize_communicator_stack()
ROOM = int(float(os.environ.get("EIGHSTACK_ROOM_GB", "40")) * 1e9)
CASES = tuple(tuple(int(v) for v in c.split("x")) for c in
              os.environ.get("EIGHSTACK_CASES", "13x3328,6x9152,5x12000,5x18304").split(","))


def main():
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    import distrib_la
    from distrib_la._batch_reshard import reshard_rounds_call
    from distrib_la._result_check import ACCEPT, eigh_errors

    mesh = RUNTIME.mesh
    px, py = int(mesh.shape["x"]), int(mesh.shape["y"])
    face = NamedSharding(mesh, P(None, "x", "y"))
    rep = NamedSharding(mesh, P())
    lead = jax.process_index() == 0
    failed = []
    errors = jax.jit(eigh_errors, out_shardings=rep)

    def hermitian(b, n, seed):
        key = jax.random.key(seed)
        a = (jax.random.normal(key, (b, n, n)) + 1j * jax.random.normal(jax.random.fold_in(key, 1), (b, n, n)))
        return (a + jnp.conj(jnp.swapaxes(a, -1, -2))) / 2

    def timed(fn, a):
        jax.block_until_ready(fn(a))
        t = time.perf_counter()
        out = jax.block_until_ready(fn(a))
        return out, time.perf_counter() - t

    def case(name, a):
        b, n = int(a.shape[0]), int(a.shape[-1])
        p = distrib_la.plan("eigh", mesh, n=n, backend="distributed", budget_bytes=ROOM)
        route = p.stack_route(a.shape, a.dtype)
        if lead:
            for line in distrib_la.new_stack_routes():
                print("distrib_la " + line, flush=True)
        (w, v), wall = timed(jax.jit(p.batched), a)
        (ws, vs), wall_scan = timed(jax.jit(lambda x: p.batched(x, _route="scan")), a)
        r, o = (float(x) for x in errors(a, w, v))
        rs, os_ = (float(x) for x in errors(a, ws, vs))
        w_h, ws_h = np.asarray(jax.device_get(w)), np.asarray(jax.device_get(ws))
        scale = max(float(np.max(np.abs(ws_h))), 1e-300)
        gap = float(np.max(np.abs(w_h - ws_h)) / scale)
        finite = bool(np.all(np.isfinite(w_h)))
        ascending = bool(np.all(np.diff(w_h, axis=-1) >= -1e-12 * scale))
        unchecked = ""
        if route.route == "batch_reshard":
            _, wall_raw = timed(jax.jit(lambda x: reshard_rounds_call("eigh", mesh, x, rounds=route.rounds)), a)
            unchecked = f" unchecked={wall_raw:.3f}s check={(wall - wall_raw) / b * 1e3:+.1f}ms/eigh"
        ok = finite and ascending and max(r, o) <= ACCEPT and gap <= 1e-10
        if not ok:
            failed.append(name)
        if lead:
            print(f"eigh_stack P{px * py} {name:22s} B={b:3d} n={n:6d} route={route.route} "
                  f"per_rank={route.per_rank} rounds={route.rounds} "
                  f"compiled={0 if route.program_bytes is None else route.program_bytes / 1e9:.2f}GB room={ROOM / 1e9:.0f}GB | "
                  f"service {wall:.3f}s ({wall / b:.3f}s/eigh){unchecked} | whole-mesh {wall_scan:.3f}s "
                  f"({wall_scan / b:.3f}s/eigh) speedup {wall_scan / wall:.2f}x | residual {r:.1e} orth {o:.1e} "
                  f"(mesh {rs:.1e}/{os_:.1e}) eig_gap {gap:.1e} finite={finite} ascending={ascending} "
                  f"{'PASS' if ok else 'FAIL'}", flush=True)
        del w, v, ws, vs

    for seed, (b, n) in enumerate(CASES):
        a = jax.jit(hermitian, static_argnums=(0, 1, 2), out_shardings=face)(b, n, seed)
        case(f"random {b}x{n}", a)
        del a
    for path in sys.argv[1:]:
        if path.endswith(".npy"):
            whole = np.load(path)
        else:
            raw = np.fromfile(path, dtype=np.complex128)
            n = int(round(np.sqrt(raw.size)))
            whole = raw.reshape(n, n)
        n = int(whole.shape[-1])
        if n % px or n % py:
            continue
        b = px * py + 1
        stack = np.broadcast_to(whole, (b, n, n))
        a = jax.make_array_from_callback(stack.shape, face, lambda i: stack[i])
        case(f"real {os.path.basename(path)[:14]}", a)
        del a
    # Decision only: a room that holds no whole matrix keeps the whole mesh.
    small = distrib_la.plan("eigh", mesh, n=18304, backend="distributed", budget_bytes=int(2e9))
    decided = small.stack_route((5, 18304, 18304), np.complex128)
    if decided.route == "batch_reshard":
        failed.append("small-room decision")
    if lead:
        print(f"eigh_stack decision only: room 2 GB, 5x18304 -> {decided.route}", flush=True)
        print("eigh_stack describe: " + small.describe(), flush=True)
        print("eigh_stack " + ("FAILED: " + ", ".join(failed) if failed else "PASS"), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    run_main_and_finalize(main)

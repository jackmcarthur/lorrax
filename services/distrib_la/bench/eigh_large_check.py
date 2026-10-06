"""P4 check: the checked distributed eigh's first attempt at production sides.

Sides (global n, carried on a mesh multiple with exact zero rows as callers
pad): 7908 (CrI3 6x6 TT at 2634 current points, n/p = 2*3*659), 7907 and
7919 (prime n, carried at 7908 and 7920), 7910, 7902 (2634*3) and 7894
(n/p = 3947 prime). Families, built on the devices from fixed seeds:
  low-rank  B^H B / n, B (n/4 x n): a 3/4-wide cluster at rounding level
  half      B^H B / n, B (n/2 x n)
  graded    B^H D^2 B / n, B (n/2 x n), D = 10^(-6 j/(n/2)): a graded tail
            into the null space, the shape of a response spectrum
Per case:
  before  the bare library at the unpadded block (_block_size), unshifted,
          deflated: the chain's first attempt before this change (reported)
  first   the plan's first attempt and its check (phase "first"), which
          must pass, and its warm wall
At n = 7908: the compiled first-phase and retry-phase programs (temp +
output per rank) and the workspace distrib_la prices for the solve.
``--bench``: random Hermitian at the sizes of the block table in
KNOWN_LORRAX_ISSUES, unpadded default block against solve_layout (median of
two warm calls). Exit 1 if any first attempt fails.

    lx run --pool POOL -N 1 -G 4 -n 4 -- bash -c 'source config/perlmutter/gpu_env.sh; \\
        python3 -u services/distrib_la/bench/eigh_large_check.py [--bench]'
"""
import sys
import time

import numpy as np

from runtime import initialize_communicator_stack, run_main_and_finalize
RUNTIME = initialize_communicator_stack()
SIDES = ((7908, 7908), (7907, 7908), (7919, 7920), (7910, 7910), (7902, 7902), (7894, 7894))
BENCH = (778, 2104, 4052, 16144, 16832, 4000, 16800)


def main():
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    import distrib_la
    from distrib_la import _cusolvermp
    from distrib_la._result_check import accept, deflate_zero_rows, eigh_errors, eigh_layout
    from distrib_la.plan import _eigh_columns

    mesh = RUNTIME.mesh
    p = int(mesh.shape["x"])
    face = NamedSharding(mesh, P("x", "y"))
    rep = NamedSharding(mesh, P())
    lead = jax.process_index() == 0
    failed = []
    say = (lambda *a: print(*a, flush=True)) if lead else (lambda *a: None)

    def gaussian(seed, shape):
        key = jax.random.key(seed)
        return jax.random.normal(key, shape) + 1j * jax.random.normal(jax.random.fold_in(key, 1), shape)

    def build(family, n, carrier, seed):
        def make():
            if family == "random":
                b = gaussian(seed, (n, n))
                a = (b + b.conj().T) / 2
            else:
                rank = n // 4 if family == "low-rank" else n // 2
                b = gaussian(seed, (rank, n))
                if family == "graded":
                    b = b * (10.0 ** (-6.0 * jnp.arange(rank) / rank))[:, None]
                a = b.conj().T @ b / n
            return jnp.pad(a, ((0, carrier - n), (0, carrier - n)))
        return jax.jit(make, out_shardings=face)()

    errors = jax.jit(eigh_errors, out_shardings=rep)
    pin = eigh_layout(mesh, 2)

    def bare(n):
        block = _cusolvermp._block_size(n, p)
        solve = lambda x: _eigh_columns("cusolvermp", *_cusolvermp.distributed_eigh(
            x, mesh=mesh, side=n, block=block))
        return jax.jit(deflate_zero_rows(solve, constrain=pin)), block

    def first_phase(plan):
        safe, _ = plan._entry("one")
        return jax.jit(lambda x: safe(x, mesh=mesh, _phase="first"))

    def timed(fn, *args):
        jax.block_until_ready(fn(*args))
        started = time.perf_counter()
        out = jax.block_until_ready(fn(*args))
        return out, time.perf_counter() - started

    for n, carrier in SIDES:
        plan = distrib_la.plan("eigh", mesh, n=carrier, backend="distributed")
        side, block = _cusolvermp.solve_layout(carrier, p)
        first = first_phase(plan)
        old, old_block = bare(carrier)
        for family in ("low-rank", "half", "graded"):
            a = build(family, n, carrier, seed=n)
            started = time.perf_counter()
            r0, o0 = (float(x) for x in errors(a, *old(a)))
            old_wall = time.perf_counter() - started
            (w, v, status), wall = timed(first, a)
            table = np.asarray(jax.device_get(status[1].addressable_data(0)))[0]
            r, o = (float(x) for x in errors(a, w, v))
            ok = r <= accept(carrier) and o <= accept(carrier) and not bool(
                np.asarray(jax.device_get(status[0].addressable_data(0))))
            if not ok:
                failed.append(f"{family} n={n}")
            say(f"eigh_large P{p * p} {family:8s} n={n} carrier={carrier} | before block={old_block:3d} "
                f"residual={r0:.1e} orth={o0:.1e}{' WRONG' if max(r0, o0) > accept(carrier) else ''} "
                f"wall={old_wall:.1f}s(cold) | first side={side} block={block} residual={r:.1e} "
                f"orth={o:.1e} check={table[0]:.1e}/{table[1]:.1e} wall={wall:.2f}s {'PASS' if ok else 'FAIL'}")
            del a, w, v
        if carrier == 7908 and n == 7908:
            shape = jax.ShapeDtypeStruct((carrier, carrier), jnp.complex128, sharding=face)
            safe, _ = plan._entry("one")
            rows = []
            for phase in ("first", "retry"):
                stats = jax.jit(lambda x, ph=phase: safe(x, mesh=mesh, _phase=ph)).lower(
                    shape).compile().memory_analysis()
                rows.append(f"{phase} temp+out {(stats.temp_size_in_bytes + stats.output_size_in_bytes) / 1e9:.2f} GB")
            priced = distrib_la.workspace_bytes_per_rank(plan, "eigh", ((carrier, carrier),), np.complex128)
            say(f"eigh_large P{p * p} bytes n={carrier}: {'; '.join(rows)}; priced workspace "
                f"{priced / 1e9:.2f} GB/rank (solve side {side}, block {block}, retry block "
                f"{_cusolvermp.retry_block(carrier, p)})")

    if "--bench" in sys.argv[1:]:
        for n in BENCH:
            a = build("random", n, n, seed=1)
            old, old_block = bare(n)
            plain = jax.jit(lambda x: _cusolvermp.distributed_eigh(x, mesh=mesh))
            side, block = _cusolvermp.solve_layout(n, p)
            walls = []
            for fn in (old, plain):
                jax.block_until_ready(fn(a))
                times = []
                for _ in range(2):
                    started = time.perf_counter()
                    jax.block_until_ready(fn(a))
                    times.append(time.perf_counter() - started)
                walls.append(float(np.median(times)))
            w_old = np.asarray(jax.device_get(old(a)[0].addressable_data(0)))
            w_new = np.asarray(jax.device_get(plain(a)[0].addressable_data(0)))
            gap = float(np.max(np.abs(w_old - w_new)) / np.max(np.abs(w_old)))
            say(f"eigh_bench P{p * p} n={n:6d} n/p={n // p:5d} unpadded block {old_block:3d}: {walls[0]:.3f} s | "
                f"solve_layout side {side} block {block}: {walls[1]:.3f} s | eigenvalue gap {gap:.1e}")
            del a
    say("eigh_large " + ("FAILED: " + ", ".join(failed) if failed else "PASS"))
    return 1 if failed else 0


if __name__ == "__main__":
    run_main_and_finalize(main)

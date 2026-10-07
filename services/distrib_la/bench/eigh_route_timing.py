"""Eigh route timing at production sides: local, whole mesh, route (c) at b < P and b >= P.

For each side n (complex128, random Hermitian) and this mesh:
  local      one rank's own ``jnp.linalg.eigh`` of one n x n matrix (cuSolverDn), the kernel
             route (c) runs on each rank; every rank runs it at once, rank 0's wall is printed
  mesh       the whole-mesh cuSOLVERMp solve of one n x n tile at P('x','y') through the
             plan's checked call (the production first attempt and its check)
  mesh*b     the same for a face stack of b matrices one after another (``_route='scan'``)
  (c) b      route (c) for a face stack of b matrices, the service decision at ROOM
             (``budget_bytes``), checked; plus the bare ``reshard_rounds_call`` (unchecked)
Per route: cold (the first call: compile, library plan, run) and warm (the smaller of two
further calls) seconds, seconds per matrix, and rank 0's device peak after the case. The
same env as the production legs (``config/perlmutter/gpu_env.sh``; the runtime sets the
NCCL transport); no other setting.

    EIGHROUTE_SIDES=3328,8736,12288,12416,13824,18432 EIGHROUTE_BATCHES=3,64 EIGHROUTE_ROOM_GB=42 \\
    lx run --pool POOL -N 16 -G 4 -n 64 -- bash -c 'source config/perlmutter/gpu_env.sh; \\
        python3 -u services/distrib_la/bench/eigh_route_timing.py'
"""
import os
import time

import numpy as np

from runtime import initialize_communicator_stack, run_main_and_finalize
RUNTIME = initialize_communicator_stack()
ROOM = int(float(os.environ.get("EIGHROUTE_ROOM_GB", "42")) * 1e9)
SIDES = tuple(int(v) for v in os.environ.get("EIGHROUTE_SIDES", "3328,8736,12288,12416,13824,18432").split(","))
BATCHES = tuple(int(v) for v in os.environ.get("EIGHROUTE_BATCHES", "3,64").split(","))
ROUTES = tuple(os.environ.get("EIGHROUTE_ROUTES", "local,mesh,meshb,c,craw").split(","))
REPEATS = max(2, int(os.environ.get("EIGHROUTE_REPEATS", "3")))


def main():
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    import distrib_la
    from distrib_la._batch_reshard import reshard_rounds_call

    mesh = RUNTIME.mesh
    px, py = int(mesh.shape["x"]), int(mesh.shape["y"])
    ranks = px * py
    face = NamedSharding(mesh, P(None, "x", "y"))
    tile = NamedSharding(mesh, P("x", "y"))
    lead = jax.process_index() == 0
    device = jax.local_devices()[0]
    say = (lambda *a: print(*a, flush=True)) if lead else (lambda *a: None)
    say(f"eigh_route P{ranks} mesh {px}x{py} room {ROOM / 1e9:.0f} GB sides {SIDES} batches {BATCHES} "
        f"env MPICH_GPU_SUPPORT_ENABLED={os.environ.get('MPICH_GPU_SUPPORT_ENABLED')} "
        f"NCCL_NET_PLUGIN={os.environ.get('NCCL_NET_PLUGIN', '')} "
        f"FI_CXI_RDZV_THRESHOLD={os.environ.get('FI_CXI_RDZV_THRESHOLD', '')} "
        f"SLURM_NETWORK={os.environ.get('SLURM_NETWORK', '')}")

    def hermitian(shape, seed, sharding):
        def make():
            key = jax.random.key(seed)
            a = (jax.random.normal(key, shape) + 1j * jax.random.normal(jax.random.fold_in(key, 1), shape))
            return (a + jnp.conj(jnp.swapaxes(a, -1, -2))) / 2
        return jax.jit(make, out_shardings=sharding)()

    def timed(fn, *a):
        walls = []
        for _ in range(REPEATS):
            t = time.perf_counter()
            out = jax.block_until_ready(fn(*a))
            walls.append(time.perf_counter() - t)
            del out
        return walls[0], min(walls[1:])

    def peak():
        stats = device.memory_stats() or {}
        return stats.get("peak_bytes_in_use", 0) / 1e9

    def report(name, n, b, cold, warm, extra=""):
        say(f"eigh_route P{ranks} n={n:6d} {name:12s} b={b:3d} cold {cold:8.3f} s warm {warm:8.3f} s "
            f"({warm / b:7.3f} s/matrix) peak {peak():6.2f} GB{extra}")

    for seed, n in enumerate(SIDES):
        if n % px or n % py:
            say(f"eigh_route n={n} skipped: not a mesh multiple")
            continue
        if "local" in ROUTES:
            # Each rank: its own whole matrix on its own device (the kernel of route (c)).
            def one(n, seed):
                key = jax.random.key(seed)
                a = jax.random.normal(key, (n, n)) + 1j * jax.random.normal(jax.random.fold_in(key, 1), (n, n))
                return (a + jnp.conj(a.T)) / 2
            with jax.default_device(device):
                a1 = jax.jit(one, static_argnums=(0, 1))(n, seed)
                cold, warm = timed(jax.jit(jnp.linalg.eigh), a1)
            report("local", n, 1, cold, warm)
            del a1
        if "mesh" in ROUTES:
            a = hermitian((n, n), seed, tile)
            p = distrib_la.plan("eigh", mesh, n=n, backend="distributed")
            cold, warm = timed(p, a)
            report("mesh", n, 1, cold, warm)
            del a
        for b in BATCHES:
            if b < 1:
                continue
            a = hermitian((b, n, n), seed, face)
            if "meshb" in ROUTES and b <= 8:
                p = distrib_la.plan("eigh", mesh, n=n, backend="distributed")
                cold, warm = timed(jax.jit(lambda x: p.batched(x, _route="scan")), a)
                report("mesh*b", n, b, cold, warm)
            if "c" in ROUTES:
                p = distrib_la.plan("eigh", mesh, n=n, backend="distributed", budget_bytes=ROOM)
                route = p.stack_route(a.shape, a.dtype)
                for line in distrib_la.new_stack_routes():
                    say("distrib_la " + line)
                cold, warm = timed(jax.jit(p.batched), a)
                report("(c) checked", n, b, cold, warm,
                       f" route={route.route} per_rank={route.per_rank} rounds={route.rounds}")
            if "craw" in ROUTES:
                rounds = -(-b // ranks)
                cold, warm = timed(jax.jit(lambda x: reshard_rounds_call("eigh", mesh, x, rounds=rounds)), a)
                report("(c) raw", n, b, cold, warm, f" rounds={rounds}")
            del a
    say("eigh_route DONE")
    return 0


if __name__ == "__main__":
    run_main_and_finalize(main)

"""Calibrate ``gw.comm_model`` on a machine: T = alpha0 + alpha_peer*n_peers + V/beta.

Measure (one rank per GPU, square mesh, every rank):
    lx run -N 4 -G 4 -n 16 -- python3 tools/comm_model_bench.py > bench.log
Fit (no JAX; any host):
    python3 tools/comm_model_bench.py --fit bench.log [more.log ...]
then store the printed row in ``gw.comm_model.MACHINES`` with the machine,
date and pool.  Collectives are emitted as LORRAX emits them: shard_map over
('x','y'), ``lax.all_to_all(tiled=True)`` / ``all_gather(tiled=True)``, n
steps of one ``lax.scan`` in one jit.  Only the peer count varies (all mesh
ranks, then one mesh axis); nothing is topology-aware.  Env: SIZES_MB
(default 1,16,128,1024 per rank).
"""
import json
import os
import sys
import time

MB = 2**20
ITEM = 16  # complex128, the ISDF carrier


def bench():
    from runtime import initialize_communicator_stack
    rt = initialize_communicator_stack(print_fn=print)
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.shard_map import shard_map

    mesh = rt.mesh
    px, py = (int(mesh.shape[a]) for a in ("x", "y"))
    n = px * py
    say = print if jax.process_index() == 0 else (lambda *a, **k: None)
    emit = lambda **kw: say("BENCH " + json.dumps(dict(kw, P=n, px=px)), flush=True)
    sizes = [int(float(s) * MB) for s in
             os.environ.get("SIZES_MB", "1,16,128,1024").split(",")]

    def make(kind, nbytes, n_iter):
        if kind == "ag_xy":
            m = max(1, nbytes // ITEM // n)
            def step(x, i):
                y = jax.lax.all_gather(x, ("x", "y"), axis=0, tiled=True)
                return jax.lax.dynamic_slice_in_dim(y, ((i + 1) % n) * m, m), None
            buf = n * m * ITEM                       # the all_gather output
        elif kind in ("a2a_xy", "a2a_x"):
            m = max(1, nbytes // ITEM // n) * n
            ax = ("x", "y") if kind == "a2a_xy" else "x"
            def step(x, i):
                return jax.lax.all_to_all(x, ax, 0, 0, tiled=True), None
            buf = m * ITEM
        else:                                        # scan step, no collective
            m = max(1, nbytes // ITEM)
            def step(x, i):
                return x * (1 + 1e-9), None
            buf = m * ITEM

        def body(x):
            return jax.lax.scan(step, x, jnp.arange(n_iter, dtype=jnp.int32),
                                unroll=1)[0]
        f = jax.jit(shard_map(body, mesh=mesh, in_specs=P(("x", "y")),
                              out_specs=P(("x", "y")), check_vma=False))
        sh = NamedSharding(mesh, P(("x", "y")))
        x = jax.jit(lambda: jnp.ones((m * n,), jnp.complex128), out_shardings=sh)()
        return f, x, buf

    def timed(f, x, reps=3):
        y = f(f(x)).block_until_ready()             # compile, communicators
        ts = []
        for _ in range(reps):
            t = time.perf_counter()
            y = f(y).block_until_ready()
            ts.append(time.perf_counter() - t)
        return ts

    tiny = jax.jit(lambda v: v + 1)
    v = jax.device_put(jnp.zeros((8,)))
    for _ in range(20):
        v = tiny(v).block_until_ready()
    t = time.perf_counter()
    for _ in range(200):
        v = tiny(v).block_until_ready()
    emit(kind="dispatch_sync_noop", t_s=(time.perf_counter() - t) / 200)
    for kind in ("noop", "a2a_xy", "a2a_x", "ag_xy"):   # per-step costs, 1 KB
        for n_iter in (8, 64):
            f, x, b = make(kind, 1024, n_iter)
            emit(kind="scanstep_" + kind, bytes=b, n_iter=n_iter, t_s=timed(f, x))
    for kind in ("a2a_xy", "a2a_x", "ag_xy"):
        for nbytes in sizes:
            n_iter = int(min(64, max(4, 2048 * MB // nbytes)))
            f, x, b = make(kind, nbytes, n_iter)
            ts = timed(f, x)
            emit(kind=kind, bytes=b, n_iter=n_iter, t_s=ts,
                 per_call_s=min(ts) / n_iter)
    say("DONE", flush=True)


def fit(logs):
    import numpy as np
    rows = [json.loads(l.split("BENCH ", 1)[1]) for f in logs
            for l in open(f) if "BENCH {" in l]
    by = {}
    for r in rows:
        by.setdefault(r["kind"], []).append(r)
    n = rows[0]["P"]
    peers = {"a2a_xy": n - 1, "a2a_x": rows[0]["px"] - 1, "ag_xy": n - 1}

    def per_step(kind):                              # dispatch cancels
        d = {r["n_iter"]: min(r["t_s"]) for r in by[kind]}
        return (d[64] - d[8]) / 56

    noop = per_step("scanstep_noop")
    pts = []
    for k in ("a2a_xy", "a2a_x"):
        pts.append((k + "@1KB", peers[k], 1024, per_step("scanstep_" + k) - noop))
        pts += [(k, peers[k], r["bytes"], r["per_call_s"]) for r in by.get(k, [])]
    A = np.array([[1.0, p, v] for _, p, v, _ in pts])
    T = np.array([t for *_, t in pts])
    (a0, ap, ib), *_ = np.linalg.lstsq(A / T[:, None], np.ones_like(T), rcond=None)
    print(f"P{n}: alpha0_s={a0:.3g}, alpha_peer_s={ap:.3g}, beta_Bps={1 / ib:.3g}, "
          f"t_dispatch_s={by['dispatch_sync_noop'][0]['t_s']:.2g}, "
          f"t_scan_step_s={noop:.2g}")
    pts += [(k, peers[k], r["bytes"], r["per_call_s"]) for k in ("ag_xy",)
            for r in by.get(k, [])]
    for k, p, v, t in pts:
        m = a0 + ap * p + v * ib
        print(f"  {k:<11}{p:>5} peers {v / MB:>10.3f} MB  meas {t * 1e3:9.3f} ms"
              f"  model/meas {m / t:5.2f}")


if __name__ == "__main__":
    fit(sys.argv[2:]) if sys.argv[1:2] == ["--fit"] else bench()

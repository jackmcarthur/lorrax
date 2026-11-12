"""Shape sweep of distrib_la: residual and device memory against the price, per op, mesh and size class.

The size classes are the tiles (n/p per mesh axis) that broke a layout before: a tile of 200 (one
block), 3954 = 2*3*659 (no eigh block in [128, 256]), 8191 / 8198 (either side of the LU relabel
threshold; 8198 = 2*4099 has no LU block either), 10002 = 2*3*1667 and 12000 (large tiles).  Per case
(one JSON line, rank 0), through the public API with the leg's library named:

    eigh      plan('eigh', backend=<library>)(A)                       (the checked chain)
    solve_lu  plan('solve_lu', backend=<library>, budget_bytes=<room>).batched(A, B), nrhs = q and n,
              per --lu-rooms (none: the 2 GiB getrs room; cuSOLVERMp keeps one tile per rank where
              it fits the room, ``_cusolvermp.lu_one_tile``)
    cholesky  plan('cholesky', backend=<library>).batched(A)
    polar     plan_polar_factor(backend=<library>)(A)                  (its eigh runs at 2n)
    gemm      matmul(A, B, backend='distributed', batched_route='auto') (panel_matmul)
    flat      project_out on a (1, 4096, P*2048) column face, 256 directions

it records the residual (``scale_distrib_la``'s distributed checks; polar: ||L^H L - I||, the
Hermitian part of L^H A; gemm: A(BX) against (AB)X; flat: the norms against the dense projection),
the per-rank peak (the XLA arena's high water plus device bytes outside it, sampled) and the price:
the provider's native workspace (``workspace_bytes_per_rank``; ``lu_workspace_query`` with the
default 2 GiB getrs room and ``lu_workspace_bytes``, both at the layout the plan's room runs;
cholesky and the XLA-only ops price none) and, for eigh, ``Plan.mesh_bytes`` (compiled program plus
that workspace).  A mesh whose geometry the op refuses records the refusal (``refused``) instead.

    srun -N1 -n4 --gpus-per-node=4 src/ffi/cpp/select_gpu.sh python -u \\
        services/distrib_la/bench/shape_sweep.py --mesh 2x2 --out sweep.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scale_distrib_la as S  # noqa: E402  (runtime bootstrap, memory sampler, distributed checks)

jax, jnp, np, D = S.jax, S.jnp, S.np, S.D
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

CLASSES = {"block": 200, "prime": 3954, "lu_below": 8191, "lu_above": 8198,
           "large_prime": 10002, "large": 12000}


def _max(row, key, field):
    vals = (row.get(key) or {}).get(field)
    return None if not vals else max(vals)


def priced(op, mesh, n, dname, plan=None, nrhs=None):
    """The service's own figures for this case (bytes per rank)."""
    out = {}
    dt = np.dtype(dname)
    try:
        if op == "eigh" and plan is not None and plan.backend == "cusolvermp":
            out["priced_native"] = int(D.workspace_bytes_per_rank(plan, "eigh", ((n, n),), dt))
            mb = plan.mesh_bytes("eigh", 1, n, dt, phase="first")
            out["priced_program_plus_native"] = None if mb is None else int(mb)
        elif op == "solve_lu" and plan is not None and plan.backend == "cusolvermp":
            from distrib_la._cusolvermp import lu_layout
            kw = {} if plan.budget_bytes is None else {"room": plan.budget_bytes}  # a room: the LU room API
            px, py = int(mesh.shape["x"]), int(mesh.shape["y"])
            side, mb, _ = lu_layout(n, px, py, **kw)
            out["nb"] = mb
            out["lu_layout"] = "one tile" if mb == side // px else f"relabeled {mb}" + (
                f" (pad to {side})" if side != n else "")
            out["priced_native"] = int(D.lu_workspace_query(mesh, n, nrhs, dt, 2 << 30, **kw))
            out["priced_closed_form"] = int(D.lu_workspace_bytes(mesh, n, dt, **kw))
        elif op == "polar" and plan is not None:
            out["priced_native"] = int(D.workspace_bytes_per_rank(plan.eigh_plan, "eigh",
                                                                  ((2 * n, 2 * n),), dt))
        elif op in ("cholesky", "gemm", "flat"):
            out["priced_native"] = (int(D.matmul_workspace_bytes_per_rank(
                mesh, ((n, n), (n, n)), dt, backend="distributed", batched_route="auto"))
                if op == "gemm" else 0)
            if op == "cholesky":
                out["priced_note"] = "no native workspace price for cuSOLVERMp potrf"
    except Exception as exc:                                     # noqa: BLE001
        out["priced_error"] = f"{type(exc).__name__}: {' '.join(str(exc).split())[:300]}"
    return out


def _measured(row, key="mem_cold"):
    return dict(measured_native=_max(row, key, "outside_max"), measured_peak=_max(row, key, "device_peak"),
                measured_arena=_max(row, key, "arena_in_use_max"))


def _room_make_plan(room, base):
    """``scale_distrib_la.make_plan`` with the solve plan's room (``budget_bytes``) for solve_lu."""
    def make_plan(op, mesh, n, force_library=False):
        if op != "solve_lu":
            return base(op, mesh, n, force_library)
        lib, plat = S.leg_library(op, mesh)
        p = D.plan(op, mesh, backend=lib, n=n, batched_route="auto", budget_bytes=room)
        return p, dict(requested=lib, backend=p.backend, platform=plat, batched_route=p.batched_route,
                       describe=p.describe(), ledger=D.new_stack_routes())
    return make_plan


def run_library_case(op, n, dname, mesh, args, nrhs=None, room=None):
    base = S.make_plan
    if room is not None:
        S.make_plan = _room_make_plan(room, base)               # run_case reads it as a module global
    try:
        row = S.run_case(op, n, dname, mesh, args, nrhs=nrhs)
    finally:
        S.make_plan = base
    if op == "solve_lu":
        row["lu_room"] = room
    if "error" in row and any(s in row["error"] for s in ("square", "divisible", "refus", "GATE")):
        row["refused"] = row["error"]
    plan = None
    if "error" not in row:
        try:
            plan = D.plan(op, mesh, backend=row["backend"], n=n, batched_route="auto", budget_bytes=room)
        except Exception:                                        # noqa: BLE001
            plan = None
    row.update(priced(op, mesh, n, dname, plan, nrhs))
    row.update(_measured(row))
    return row


def run_polar(n, dname, mesh, args):
    row = dict(op="polar", n=n, dtype=dname, mesh=f"{mesh.shape['x']}x{mesh.shape['y']}")
    tile = NamedSharding(mesh, P("x", "y"))
    try:
        lib, _ = S.leg_library("eigh", mesh)
        pp = D.plan_polar_factor(mesh, n=n, backend=lib, batched_route="auto")
        row["backend"] = pp.backend
        A = S.general(mesh, (n, n), jnp.dtype(dname), 17, 1 / np.sqrt(n))
        (L, s), w, m = S.timed(pp, lambda: (A,), args.sample_s)
        row["cold_s"], row["mem_cold"] = w, S.gather(m)
        G = S.mm(L, L, mesh, transa="C")
        G = jax.jit(lambda g: g - jnp.eye(n, dtype=g.dtype), out_shardings=tile)(G)
        row["orth"] = S.fro(G) / np.sqrt(n)
        H = S.mm(L, A, mesh, transa="C")
        Hh = S.mm(H, jax.jit(lambda h: jnp.eye(n, dtype=h.dtype), out_shardings=tile)(H), mesh, transa="C")
        row["resid"] = S.fro(jax.jit(lambda a, b: a - b, out_shardings=tile)(H, Hh)) / S.fro(H)
        row.update(priced("polar", mesh, n, dname, pp))
    except Exception as exc:                                     # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {' '.join(str(exc).split())[:800]}"
    row.update(_measured(row))
    return row


def run_gemm(n, dname, mesh, args):
    row = dict(op="gemm", n=n, dtype=dname, mesh=f"{mesh.shape['x']}x{mesh.shape['y']}")
    try:
        dt = jnp.dtype(dname)
        q = int(mesh.shape["y"])
        A = S.general(mesh, (n, n), dt, 19, 1 / np.sqrt(n))
        B = S.general(mesh, (n, n), dt, 23, 1 / np.sqrt(n))
        X = S.general(mesh, (n, q), dt, 29, 1.0)
        C, w, m = S.timed(lambda a, b: S.mm(a, b, mesh), lambda: (A, B), args.sample_s)
        row["cold_s"], row["mem_cold"] = w, S.gather(m)
        lhs = S.mm(C, X, mesh)
        rhs = S.mm(A, S.mm(B, X, mesh), mesh)
        row["resid"] = S.fro(lhs - rhs) / S.fro(rhs)
        row.update(priced("gemm", mesh, n, dname))
    except Exception as exc:                                     # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {' '.join(str(exc).split())[:800]}"
    row.update(_measured(row))
    return row


def run_flat(dname, mesh, args, rows=4096, per_rank=2048, t=256):
    P_ = int(mesh.size)
    row = dict(op="flat", n=P_ * per_rank, rows=rows, dtype=dname,
               mesh=f"{mesh.shape['x']}x{mesh.shape['y']}")
    try:
        dt = jnp.dtype(dname)
        spec = P(None, None, ("x", "y"))
        Y0 = jax.jit(lambda k: S._normal(k, (1, rows, P_ * per_rank), dt),
                     out_shardings=NamedSharding(mesh, spec))(jax.random.PRNGKey(31))
        q, _ = jnp.linalg.qr(S._normal(jax.random.PRNGKey(37), (rows, t), dt))
        Q = jax.device_put(q[None], NamedSharding(mesh, P()))
        ref = jax.jit(lambda y, qq: jnp.sum(jnp.abs(y - qq @ (jnp.conj(jnp.swapaxes(qq, 1, 2)) @ y)) ** 2,
                                            axis=1), out_shardings=NamedSharding(mesh, P(None, ("x", "y"))))(Y0, Q)
        run = jax.jit(lambda y, qq: D.project_out(y, qq, mesh=mesh, panel_bytes=64 << 20, spec=spec, chunk=128),
                      donate_argnums=(0,))
        temp = run.lower(Y0, Q).compile().memory_analysis().temp_size_in_bytes
        (Y, d), w, m = S.timed(run, lambda: (jnp.copy(Y0), Q), args.sample_s)
        row["cold_s"], row["mem_cold"] = w, S.gather(m)
        row["resid"] = float(jnp.max(jnp.abs(d - ref)) / jnp.max(jnp.abs(ref)))
        row["compiled_temp"] = int(temp)
        row["block_bytes"] = rows * per_rank * dt.itemsize
        row.update(priced("flat", mesh, row["n"], dname))
    except Exception as exc:                                     # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {' '.join(str(exc).split())[:800]}"
    row.update(_measured(row))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="2x2")
    ap.add_argument("--classes", nargs="*", default=list(CLASSES))
    ap.add_argument("--ns", nargs="*", type=int, default=[],
                    help="extra operand sizes n, run after the classes as class 'n<n>' (tile n // px)")
    ap.add_argument("--dtypes", nargs="+", default=["float64", "complex128"])
    ap.add_argument("--ops", nargs="+", default=["eigh", "lu", "chol", "polar", "gemm", "flat"])
    ap.add_argument("--polar-classes", nargs="+", default=["block", "prime"])
    ap.add_argument("--nrhs", nargs="+", default=["min", "n"])
    ap.add_argument("--lu-rooms", nargs="+", default=["none"],
                    help="solve_lu plan rooms in GB per rank (budget_bytes); 'none' = no room")
    ap.add_argument("--steady-max", type=float, default=0.0)
    ap.add_argument("--case-timeout", type=float, default=1800.0)
    ap.add_argument("--sample-s", type=float, default=0.05)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    args.force_library, args.eigh_nb = True, None

    px, py = (int(v) for v in args.mesh.lower().split("x"))
    mesh = Mesh(np.asarray(jax.devices()).reshape(px, py), ("x", "y"))

    def emit(row):
        row["class"] = row.get("class")
        S.say("[sweep]", json.dumps(row, default=str))
        if S.R0 and args.out:
            with open(args.out, "a") as fh:
                fh.write(json.dumps(row, default=str) + "\n")

    emit(dict(check="env", mesh=args.mesh, jax=jax.__version__, processes=jax.process_count(),
              kind=str(jax.devices()[0].device_kind), job=os.environ.get("SLURM_JOB_ID", ""),
              distrib_la=D.__file__, arena_limit=int((jax.local_devices()[0].memory_stats() or {})
                                                     .get("bytes_limit", 0)),
              device_total=S.XM.device_total_bytes(0)))
    lcm = px * py // np.gcd(px, py)
    cases = []
    for cls in args.classes:
        tile = CLASSES[cls]
        cases.append((cls, tile, max(px, py) * tile if px == py else lcm * max(1, (max(px, py) * tile) // lcm)))
    cases += [(f"n{n}", n // px, n) for n in args.ns]               # explicit sizes (e.g. at a threshold)
    for cls, tile, n in cases:
        for dname in args.dtypes:
            for o in args.ops:
                t0 = time.time()
                if o == "eigh":
                    rows = [run_library_case("eigh", n, dname, mesh, args)]
                elif o == "chol":
                    rows = [run_library_case("cholesky", n, dname, mesh, args)]
                elif o == "lu":
                    rows = [run_library_case("solve_lu", n, dname, mesh, args,
                                             nrhs=py if s == "min" else n if s == "n" else int(s),
                                             room=None if g == "none" else int(float(g) * 1e9))
                            for g in args.lu_rooms for s in args.nrhs]
                elif o == "polar":
                    rows = [run_polar(n, dname, mesh, args)] if cls in args.polar_classes else []
                elif o == "gemm":
                    rows = [run_gemm(n, dname, mesh, args)]
                elif o == "flat":
                    rows = [run_flat(dname, mesh, args)] if cls == cases[0][0] else []
                for r in rows:
                    r["class"], r["tile"], r["case_wall"] = cls, tile, time.time() - t0
                    for k in ("ledger_after", "describe", "ledger"):
                        r.pop(k, None)
                    emit(r)
    S.say("[sweep] done")


if __name__ == "__main__":
    sys.exit(main())

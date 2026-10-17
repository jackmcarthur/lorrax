"""Capacity-scale check of distrib_la's DISTRIBUTED route: matrices beyond one device.

Every case goes through the public API (``distrib_la.plan``), never a backend module:

    eigh      plan('eigh', mesh, backend='distributed', n=n, batched_route='auto')(A)
    solve_lu  plan('solve_lu', mesh, backend='distributed', n=n, batched_route='auto').batched(A, B)
    cholesky  plan('cholesky', mesh, backend=<leg library>, n=n, batched_route='auto')   (no
              'distributed' spelling for cholesky: slate on ROCM, cusolvermp on CUDA)

and REFUSES to time a case whose plan did not resolve the leg's distributed library (no
one_device, no native, no host): the resolved backend, the plan's describe() line and the
route-ledger lines (distrib_la.new_stack_routes) are recorded per case.

    srun -N1 -n4 ... python services/distrib_la/bench/scale_distrib_la.py --mesh 2x2 \
        --sizes 20480 30720 --dtypes float64 complex128 --ops eigh lu chol --nrhs min 64 n \
        --out scale.jsonl

Per case (JSON line, rank 0):
  wall     cold (first call: compile + library context + run) and, when the cold wall is under
           --steady-max seconds, one steady call (fresh operands for the donating ops)
  checks   eigh ||AZ - Z diag(w)||_F/||A||_F and ||Z^H Z - I||_F/sqrt(n);
           lu ||AX - B||_F/(||A||_F ||X||_F); cholesky ||L L^H - A||_F/||A||_F
           -- all distributed (distrib_la.matmul, face SUMMA): nothing is gathered
  memory   per rank: the XLA arena's in-use high water (ROCm: sampled bytes_in_use + the
           cumulative peak_bytes_in_use; CUDA: the mempool's used high water, reset per call)
           plus device bytes OUTSIDE the arena (runtime.xla_memory.outside_pool_bytes,
           sampled every --sample-s), and their sampled sum: the true per-rank device peak
  host     per-rank process CPU seconds during the call and ru_maxrss
  layout   the library tile (nb) and process grid the plan runs at (read from the backend's
           layout function: this file is a bench, allowed past the public API)

``--routing`` adds the plan-level routing checks (no library work): the size route at the
one-device bound, the room rules, and what the DEFAULT batched route does with a matrix
beyond the bound.  ``--default-route-lu N`` runs that default-route solve once at n=N.
"""
from __future__ import annotations

import argparse
import faulthandler
import gc
import json
import os
import resource
import sys
import threading
import time

os.environ.setdefault("JAX_ENABLE_X64", "1")
faulthandler.enable()

_BENCH = os.path.dirname(os.path.abspath(__file__))

import runtime  # noqa: E402  (LORRAX runtime: env, jax.distributed, MPI before collectives)

runtime.bootstrap()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.experimental import multihost_utils as mhu  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

import distrib_la as D  # noqa: E402
from runtime import xla_memory as XM  # noqa: E402

R0 = jax.process_index() == 0


def say(*a):
    if R0:
        print(*a, flush=True)


# --------------------------------------------------------------------------- memory
class Sampler:
    """Per-rank device memory during one call: arena in-use and outside-arena bytes."""

    def __init__(self, period: float):
        self.period = period
        self.dev = jax.local_devices()[0]
        self.cuda = "NVIDIA" in str(self.dev.device_kind).upper()

    def __enter__(self):
        self.stop = threading.Event()
        self.max_in, self.max_out, self.max_sum, self.n = 0, 0, 0, 0
        if self.cuda:
            XM.pool_high_water(reset=True)
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
        return self

    def _sample(self):
        st = self.dev.memory_stats() or {}
        inside = int(st.get("bytes_in_use", 0))
        out = XM.outside_pool_bytes(0)
        out = 0 if out is None else int(out)
        self.max_in = max(self.max_in, inside)
        self.max_out = max(self.max_out, out)
        self.max_sum = max(self.max_sum, inside + out)
        self.n += 1

    def _run(self):
        while not self.stop.is_set():
            try:
                self._sample()
            except Exception:                                   # noqa: BLE001
                pass
            self.stop.wait(self.period)

    def __exit__(self, *exc):
        self.stop.set()
        self.t.join()
        try:
            self._sample()
        except Exception:                                       # noqa: BLE001
            pass
        st = self.dev.memory_stats() or {}
        hw = XM.pool_high_water() if self.cuda else None
        self.result = dict(
            arena_in_use_max=self.max_in,
            arena_peak_cumulative=int(st.get("peak_bytes_in_use", 0)),
            arena_limit=int(st.get("bytes_limit", 0)),
            pool_high_water=hw,
            outside_max=self.max_out,
            # True per-rank peak: CUDA's mempool high water covers every async user of the pool
            # (XLA and the library's cudaMallocAsync); ROCm's BFC arena holds only XLA's buffers.
            device_peak=(max(self.max_sum, (hw or 0) + self.max_out) if self.cuda
                         else self.max_sum),
            samples=self.n)
        return False


def gather(d: dict) -> dict:
    """Per-rank values of ``d`` (ints/floats) -> lists over ranks."""
    keys = sorted(k for k, v in d.items() if isinstance(v, (int, float)) and v is not None)
    vec = np.array([float(d[k]) for k in keys], np.float64)
    allv = np.asarray(mhu.process_allgather(vec))
    return {k: allv[:, i].tolist() for i, k in enumerate(keys)}


def timed(fn, make, sample_s):
    """Run fn(*make()) once; (out, wall, per-rank metrics). Operands built outside the wall."""
    ops = make()
    jax.block_until_ready(ops)
    mhu.sync_global_devices("pre")
    cpu0 = time.process_time()
    with Sampler(sample_s) as s:
        t0 = time.perf_counter()
        out = fn(*ops)
        jax.block_until_ready(out)
        wall = time.perf_counter() - t0
    cpu = time.process_time() - cpu0
    del ops
    m = dict(s.result, cpu_s=cpu, rss_max_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             wall_local=wall)
    return out, wall, m


# --------------------------------------------------------------------------- operands
def _normal(key, shape, dt):
    k1, k2 = jax.random.split(key)
    g = jax.random.normal(k1, shape, jnp.float64)
    if jnp.dtype(dt).kind == "c":
        g = g + 1j * jax.random.normal(k2, shape, jnp.float64)
    return g.astype(dt)


def herm(mesh, n, dt, seed, shift=0.0):
    """(G + G^H)/sqrt(2n) (+ shift I): spectrum in about [-2.83, 2.83] (complex), [-2, 2] (real)."""
    tile = NamedSharding(mesh, P("x", "y"))

    g = jax.jit(lambda k: _normal(k, (n, n), dt), out_shardings=tile)(jax.random.PRNGKey(seed))

    def f(g):
        a = (g + jnp.conj(g.T)) * (1.0 / np.sqrt(2.0 * n))
        if shift:
            a = a + shift * jnp.eye(n, dtype=dt)
        return a.astype(dt)
    return jax.jit(f, out_shardings=tile, donate_argnums=0)(g)


def general(mesh, shape, dt, seed, scale):
    spec = P("x", "y") if len(shape) == 2 else P(None, "x", "y")

    def f(k):
        return (_normal(k, shape, dt) * scale).astype(dt)
    return jax.jit(f, out_shardings=NamedSharding(mesh, spec))(jax.random.PRNGKey(seed))


_fro = jax.jit(lambda x: jnp.sqrt(jnp.sum(jnp.real(x * jnp.conj(x)))))


def fro(x):
    return float(_fro(x))


# --------------------------------------------------------------------------- layout
def layout(op, backend, n, px, py):
    """(nb, grid, note) the backend runs this op at (bench-only read of private layout fns)."""
    grid = f"{px}x{py}"
    try:
        if backend == "slate":
            from distrib_la import _slate as S
            if op == "eigh":
                side, nb = S.eigh_layout(n, px)
                return nb, grid, f"heev side {side}, two-stage (he2hb devices, hb2st rank-0 host)"
            if op == "solve_lu":
                return S.lu_layout(n, px, py), grid, "getrf PartialPiv + getrs, Target::Devices"
            if op == "cholesky":
                return n // max(px, py), grid, "potrf one tile per rank (fromDevices, in place)"
        if backend == "cusolvermp":
            from distrib_la import _cusolvermp as C
            if op == "eigh":
                side, nb = C.solve_layout(n, px)
                return nb, grid, f"syevd side {side}, mb=nb block-cyclic by relabeling"
            return n // px, grid, "one tile per rank (mb = n/Px, nb = n/Py)"
    except Exception as exc:                                     # noqa: BLE001
        return None, grid, f"layout unavailable: {exc}"
    return None, grid, ""


# --------------------------------------------------------------------------- cases
def leg_library(op, mesh):
    from distrib_la import resolve as _r
    plat = _r.mesh_platform(mesh)
    if op == "cholesky":
        return {"ROCM": "slate", "CUDA": "cusolvermp"}.get(plat), plat
    return _r._DISTRIBUTED_DEFAULT.get((op, plat)), plat


def make_plan(op, mesh, n, force_library=False):
    lib, plat = leg_library(op, mesh)
    requested = "distributed" if op in ("eigh", "solve_lu") and not force_library else lib
    p = D.plan(op, mesh, backend=requested, n=n, batched_route="auto")
    ledger = D.new_stack_routes()
    if p.backend != lib:
        raise RuntimeError(f"ROUTING: {op} n={n} {requested!r} resolved {p.backend!r}, "
                           f"not the {plat} distributed library {lib!r}; refusing to time it")
    return p, dict(requested=requested, backend=p.backend, platform=plat,
                   batched_route=p.batched_route, describe=p.describe(), ledger=ledger)


def lower_of(handle):
    if hasattr(handle, "to_jax_lower"):
        return handle.to_jax_lower()                             # SlateLowerL: (n, n) P('x','y')
    from distrib_la._cusolvermp import cholesky_handle_to_natural_L
    return cholesky_handle_to_natural_L(handle)[0]               # (1, n, n) -> (n, n)


def mm(a, b, mesh, **kw):
    return D.matmul(a, b, mesh=mesh, backend="distributed", batched_route="auto", **kw)


_SAMPLE = 256


def _cols(n, k=_SAMPLE):
    """k column indices spread over the whole index range (the spectrum, for eigh)."""
    return np.unique(np.linspace(0, n - 1, min(n, k)).astype(np.int64))


def sampled_eigh_check(A, Z, w, n):
    """Residual and orthogonality on k sampled eigenpairs (when the full check does not
    fit beside the operands): ||A Z_S - Z_S W_S||_F / (||A||_F sqrt(k/n)), ||Z^H Z_S - I_S||_F / sqrt(k)."""
    idx = jnp.asarray(_cols(n))
    k = int(idx.shape[0])
    f = jax.jit(lambda a, z, lam: (
        jnp.sqrt(jnp.sum(jnp.abs(a @ z[:, idx] - z[:, idx] * lam[idx][None, :]) ** 2)),
        jnp.sqrt(jnp.sum(jnp.abs(jnp.conj(z.T) @ z[:, idx]
                                 - jnp.eye(n, dtype=z.dtype)[:, idx]) ** 2))))
    r, o = f(A, Z, w)
    return float(r) / (fro(A) * np.sqrt(k / n)), float(o) / np.sqrt(k), k


def sampled_solve_check(A, X, B, n):
    """||A X_S - B_S||_F / (||A||_F ||X_S||_F) on k sampled RHS columns."""
    idx = jnp.asarray(_cols(int(X.shape[1])))
    f = jax.jit(lambda a, x, b: (jnp.sqrt(jnp.sum(jnp.abs(a @ x[:, idx] - b[:, idx]) ** 2)),
                                 jnp.sqrt(jnp.sum(jnp.abs(x[:, idx]) ** 2))))
    r, xs = f(A, X, B)
    return float(r) / (fro(A) * float(xs)), int(idx.shape[0])


def sampled_chol_check(L, A, n):
    """||L L^H[:, S] - A[:, S]||_F / (||A||_F sqrt(k/n)) on k sampled columns."""
    idx = jnp.asarray(_cols(n))
    k = int(idx.shape[0])
    f = jax.jit(lambda l, a: jnp.sqrt(jnp.sum(jnp.abs(l @ jnp.conj(l[idx, :].T) - a[:, idx]) ** 2)))
    return float(f(L, A)) / (fro(A) * np.sqrt(k / n)), k


def run_case(op, n, dname, mesh, args, nrhs=None):
    dt = jnp.dtype(dname)
    px, py = int(mesh.shape["x"]), int(mesh.shape["y"])
    tile = NamedSharding(mesh, P("x", "y"))
    row = dict(op=op, n=n, dtype=dname, mesh=f"{px}x{py}", nodes=int(os.environ.get(
        "SLURM_JOB_NUM_NODES", "1")), ranks=jax.process_count())
    if nrhs is not None:
        row["nrhs"] = nrhs
    faulthandler.dump_traceback_later(args.case_timeout, exit=True)
    try:
        p, info = make_plan(op, mesh, n, args.force_library)
        row.update(info)
        row["nb"], row["grid"], row["layout"] = layout(op, p.backend, n, px, py)
        if op == "solve_lu" and p.backend == "cusolvermp":
            from distrib_la._cusolvermp import lu_blocks
            row["nb"] = lu_blocks(n, px, py)[0]
        walls, metrics = [], []
        if op == "eigh":
            A = herm(mesh, n, dt, 5)
            make = lambda: (A,)                                  # noqa: E731
            fn = p
            kw = {}
            if args.eigh_nb is not None:
                # Plan(A, **kw) forwards keywords to the backend's entry
                # (distrib_la._slate.distributed_eigh(block_size=...)).
                kw["block_size"] = args.eigh_nb
                row["nb"] = args.eigh_nb
            if kw:
                fn = lambda a: p(a, **kw)                        # noqa: E731
        elif op == "solve_lu":
            make = lambda: (general(mesh, (1, n, n), dt, 7, 1 / np.sqrt(n)),  # noqa: E731
                            general(mesh, (1, n, nrhs), dt, 11, 1.0))
            fn = p.batched
        else:
            make = lambda: (herm(mesh, n, dt, 13, shift=4.0),)   # noqa: E731
            fn = p.batched if p.backend == "cusolvermp" else p
            if p.backend == "cusolvermp":
                make = lambda: (herm(mesh, n, dt, 13, shift=4.0)[None],)  # noqa: E731
        out, w, m = timed(fn, make, args.sample_s)
        walls.append(w)
        metrics.append(m)
        if w < args.steady_max:
            del out
            out, w, m = timed(fn, make, args.sample_s)
            walls.append(w)
            metrics.append(m)
        row["cold_s"] = walls[0]
        row["steady_s"] = walls[1] if len(walls) > 1 else None
        row["mem_cold"] = gather(metrics[0])
        if len(metrics) > 1:
            row["mem_steady"] = gather(metrics[1])
        # ---- checks (distributed; operands regenerated where the op donated them)
        t0 = time.perf_counter()
        if op == "eigh":
            w_, Z = out
            del out
            try:
                AZ = mm(A, Z, mesh)
                Rm = jax.jit(lambda az, z, lam: az - z * lam[None, :], donate_argnums=0,
                             out_shardings=tile)(AZ, Z, w_)
                row["resid"] = fro(Rm) / fro(A)
                del Rm, AZ
                G = mm(Z, Z, mesh, transa="C")
                G = jax.jit(lambda g: g - jnp.eye(n, dtype=g.dtype), donate_argnums=0,
                            out_shardings=tile)(G)
                row["orth"] = fro(G) / np.sqrt(n)
                row["check"] = "full"
                del G
            except Exception as exc:                             # noqa: BLE001
                AZ = Rm = G = None
                row["check_full_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            # The fallback runs after the except block: the live traceback pins the
            # failed check's operands (jit argument frames), which is the memory the
            # sampled check needs.
            if "check_full_error" in row:
                gc.collect()
                row["resid"], row["orth"], k = sampled_eigh_check(A, Z, w_, n)
                row["check"] = f"sampled {k} eigenpairs"
            wn = np.asarray(jax.device_get(w_))
            row["w_range"] = [float(wn.min()), float(wn.max())]
            row["w_sorted"] = bool(np.all(np.diff(wn) >= 0))
            del Z, w_, A
        elif op == "solve_lu":
            X = out
            del out
            A, B = make()
            sq = jax.jit(lambda t: t[0], out_shardings=tile)
            A2, B2, X2 = sq(A), sq(B), sq(X)
            del A, B, X
            try:
                AX = mm(A2, X2, mesh)
                Rm = jax.jit(lambda ax, b: ax - b, donate_argnums=0, out_shardings=tile)(AX, B2)
                row["resid"] = fro(Rm) / (fro(A2) * fro(X2))
                row["check"] = "full"
                del Rm, AX
            except Exception as exc:                             # noqa: BLE001
                AX = Rm = None
                row["check_full_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            if "check_full_error" in row:
                gc.collect()
                row["resid"], k = sampled_solve_check(A2, X2, B2, n)
                row["check"] = f"sampled {k} columns"
            row["finite"] = bool(np.isfinite(row["resid"]))
            del A2, B2, X2
        else:
            L = lower_of(out)
            del out
            try:
                LLH = mm(L, L, mesh, transb="C")
                A = make()[0]
                if A.ndim == 3:
                    A = jax.jit(lambda t: t[0], out_shardings=tile)(A)
                Rm = jax.jit(lambda x, a: x - a, donate_argnums=0, out_shardings=tile)(LLH, A)
                row["resid"] = fro(Rm) / fro(A)
                row["check"] = "full"
                del Rm, LLH, A, L
            except Exception as exc:                             # noqa: BLE001
                LLH = Rm = A = None
                row["check_full_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            if "check_full_error" in row:
                gc.collect()
                A = make()[0]
                if A.ndim == 3:
                    A = jax.jit(lambda t: t[0], out_shardings=tile)(A)
                row["resid"], k = sampled_chol_check(L, A, n)
                row["check"] = f"sampled {k} columns"
                del A, L
        row["check_s"] = time.perf_counter() - t0
    except Exception as exc:                                     # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {' '.join(str(exc).split())[:1500]}"
    finally:
        faulthandler.cancel_dump_traceback_later()
    row["ledger_after"] = D.new_stack_routes()
    return row


# --------------------------------------------------------------------------- routing checks
def routing_checks(mesh, sizes):
    """Plan-level facts (no library work): what each spelling resolves to, per n and room."""
    from distrib_la import resolve as _r
    plat = _r.mesh_platform(mesh)
    rows = []
    for op in ("eigh", "solve_lu"):
        bound = _r.one_device_max_n(op, plat)
        probe = sorted({*(sizes or []), *([bound, bound + 2 * int(mesh.shape["x"]) *
                                            int(mesh.shape["y"])] if bound else [])})
        for n in probe:
            for room in (None, 0, int(2e9), int(200e9)):
                r = dict(check="route", op=op, n=n, room=room, platform=plat, bound=bound)
                try:
                    for route in (None, "auto"):
                        kw = {} if route is None else dict(batched_route=route)
                        p = D.plan(op, mesh, backend="distributed", n=n, budget_bytes=room, **kw)
                        tag = "default" if route is None else route
                        r[f"backend[{tag}]"] = p.backend
                        r[f"batched_route[{tag}]"] = p.batched_route
                        r[f"size_route[{tag}]"] = p.size_route
                    r["ledger"] = D.new_stack_routes()
                except Exception as exc:                         # noqa: BLE001
                    r["error"] = f"{type(exc).__name__}: {' '.join(str(exc).split())[:400]}"
                rows.append(r)
    return rows


def default_route_lu(mesh, n, dname, sample_s):
    """What plan('solve_lu','distributed', n) with the DEFAULT batched route actually runs."""
    dt = jnp.dtype(dname)
    tile = NamedSharding(mesh, P("x", "y"))
    q = int(mesh.shape["y"])
    row = dict(check="default_route_lu", n=n, dtype=dname)
    try:
        p = D.plan("solve_lu", mesh, backend="distributed", n=n)
        row.update(backend=p.backend, batched_route=p.batched_route, describe=p.describe())
        make = lambda: (general(mesh, (1, n, n), dt, 7, 1 / np.sqrt(n)),   # noqa: E731
                        general(mesh, (1, n, q), dt, 11, 1.0))
        X, w, m = timed(p.batched, make, sample_s)
        row["wall_s"] = w
        row["mem"] = gather(m)
        A, B = make()
        sq = jax.jit(lambda t: t[0], out_shardings=tile)
        A2, B2, X2 = sq(A), sq(B), sq(X)
        AX = mm(A2, X2, mesh)
        row["resid"] = fro(AX - B2) / (fro(A2) * fro(X2))
    except Exception as exc:                                     # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {' '.join(str(exc).split())[:800]}"
    return row


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="2x2")
    ap.add_argument("--sizes", type=int, nargs="*", default=[])
    ap.add_argument("--dtypes", nargs="+", default=["float64", "complex128"])
    ap.add_argument("--ops", nargs="+", default=["eigh", "lu", "chol"])
    ap.add_argument("--skip", nargs="*", default=[],
                    help="cases not to run, op:dtype:n with * wildcards (e.g. eigh:complex128:20480 "
                         "eigh:*:30720) -- one process can then cover a grid minus the cases a "
                         "baseline aborts on")
    ap.add_argument("--nrhs", nargs="+", default=["min", "64", "n"],
                    help="'min' = the mesh's q (SLATE/cuSOLVERMp need q | nrhs), an int, or 'n'")
    ap.add_argument("--steady-max", type=float, default=120.0)
    ap.add_argument("--case-timeout", type=float, default=2400.0)
    ap.add_argument("--sample-s", type=float, default=0.25)
    ap.add_argument("--routing", action="store_true")
    ap.add_argument("--eigh-nb", type=int, default=None,
                    help="eigh tile (block_size) override, a divisor of n/p")
    ap.add_argument("--force-library", action="store_true",
                    help="request the leg library by name (below the one-device bound too)")
    ap.add_argument("--default-route-lu", type=int, default=0)
    ap.add_argument("--default-route-dtype", default="complex128")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    px, py = (int(v) for v in args.mesh.lower().split("x"))
    mesh = Mesh(np.asarray(jax.devices()).reshape(px, py), ("x", "y"))

    def emit(row):
        say("[scale]", json.dumps(row, default=str))
        if R0 and args.out:
            with open(args.out, "a") as fh:
                fh.write(json.dumps(row, default=str) + "\n")

    emit(dict(check="env", jax=jax.__version__, devices=jax.device_count(),
              processes=jax.process_count(), kind=str(jax.devices()[0].device_kind),
              job=os.environ.get("SLURM_JOB_ID", ""), nodes=os.environ.get("SLURM_JOB_NUM_NODES"),
              ffi=os.environ.get("LORRAX_FFI_ROCM_SO") or os.environ.get("LORRAX_FFI_SO", ""),
              arena_limit=int((jax.local_devices()[0].memory_stats() or {}).get("bytes_limit", 0)),
              device_total=XM.device_total_bytes(0),
              outside_at_start=XM.outside_pool_bytes(0)))
    if args.routing:
        for r in routing_checks(mesh, args.sizes):
            emit(r)
    if args.default_route_lu:
        emit(default_route_lu(mesh, args.default_route_lu, args.default_route_dtype, args.sample_s))
    opname = {"eigh": "eigh", "lu": "solve_lu", "chol": "cholesky"}
    for n in sorted(args.sizes):
        for dname in args.dtypes:
            for o in args.ops:
                op = opname[o]
                if any(all(pat in ("*", val) for pat, val in zip(sk.split(":"), (o, dname, str(n))))
                       for sk in args.skip):
                    continue
                if op != "solve_lu":
                    emit(run_case(op, n, dname, mesh, args))
                    continue
                for spec in args.nrhs:
                    nrhs = py if spec == "min" else n if spec == "n" else int(spec)
                    emit(run_case(op, n, dname, mesh, args, nrhs=nrhs))
    say("[scale] done")


if __name__ == "__main__":
    sys.exit(main())

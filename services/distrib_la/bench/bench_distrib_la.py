"""Microbench: op x backend x shape, at 1x1 and at a real 2x2.

RECORD, DO NOT OPTIMIZE.  Step 2's job is to make the numbers exist and be
comparable; step 6 changes them.  Nothing here is a test and nothing here
has a threshold — a bench cell that fails a run is a slow test wearing a
different name, and the charter is explicit that the default suite stays
seconds-fast and that perf lands as recorded baseline files.

    lx run -N 1 -G 4 -n 4 python3 \\
        services/distrib_la/bench/bench_distrib_la.py --mesh 2x2 --tag gpu2x2
    lx run --cpu -N 1 -n 4 python3 \\
        services/distrib_la/bench/bench_distrib_la.py --mesh 2x2 --tag cpu2x2
    lx run -N 1 -G 1 -n 1 python3 \\
        services/distrib_la/bench/bench_distrib_la.py --mesh 1x1 --tag gpu1x1

One JSON row per (op, backend, shape, mesh, nodes, seconds, jobid) into
``baselines/``, the claims-style shape the charter names.  Regression
detection is DIFFING BASELINE FILES ACROSS BRANCHES, not asserting a
number in a test: a threshold that has to hold on a shared machine either
gets loosened until it means nothing or fails on somebody else's
contention.

THIS SCRIPT OVERWRITES ITS TAG'S FILE, AND SOME ROWS IN THERE ARE NOT
ITS OWN.  ``cpu2x2.json`` / ``gpu2x2.json`` also carry the step-6 eigh
investigation's single-matrix sweep (``shape [1, n]``, ``n`` up to 4096,
tagged ``"leg": "step6.cross_size"`` / ``"step6.slate4096"``), produced by
a DIFFERENT harness — ``svc_distrib_la_perf/perf_eigh.py``.  Re-running
this script drops them.  Merge, do not clobber: the row key across the
two sources is ``(op, backend, shape, route)``, because the step-6 rows
record the replicated and the sharded native input layouts at the same
shape and they are different measurements.

WHAT IS TIMED, precisely.  The FIRST call of every (backend, shape) is a
compile and is discarded; then ``warmup`` calls settle the executable
cache, then ``reps`` timed calls, each ``block_until_ready``-ed, and the
MEDIAN is recorded along with min and max.  Median rather than mean
because a shared GPU produces occasional long tails that a mean folds
into the number and hides.  ``compile_seconds`` is recorded separately,
because "how long does the first call take" is a real question with a
different answer and conflating them is how a plan-caching regression
goes unnoticed.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

_BENCH = os.path.dirname(os.path.abspath(__file__))
_SERVICES = os.path.dirname(os.path.dirname(_BENCH))
for _svc in ("lxkit", "distrib_la"):
    _src = os.path.join(_SERVICES, _svc, "src")
    if os.path.isdir(_src) and _src not in sys.path:
        sys.path.insert(0, _src)

if __name__ == "__main__":
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        from lxkit.gate import platform_from_env
        _plat = platform_from_env()
        if _plat == "CUDA":
            try:
                from distrib_la.loader import get_lib as _get_lib
                _get_lib(_plat).lrx_slate_init_mpi()
            except Exception as _exc:                          # noqa: BLE001
                print(f"slate_init_mpi skipped: {_exc}", flush=True)
        import jax
        if _plat == "CUDA":
            jax.distributed.initialize(local_device_ids=[0])
        else:
            jax.distributed.initialize()

import distrib_la as D                                        # noqa: E402

#: (op, backend) pairs worth a row.  ``off`` is included on purpose: the
#: replicated jnp path is the thing every distributed backend has to beat,
#: and a baseline table without it cannot answer "was this worth it".
_ROWS = [
    ("eigh", "off"), ("eigh", "distributed"), ("eigh", "scalapack"),
    ("eigh", "cusolvermp"), ("eigh", "slate"),
    ("cholesky", "native2d"), ("cholesky", "cusolvermp"),
    ("cholesky", "slate"),
    ("solve_lu", "distributed"), ("solve_lu", "scalapack"),
    ("solve_lu", "cusolvermp"),
]

#: (nq, n) shapes.  Small enough that a full sweep is a couple of minutes
#: on a shared node; large enough that dispatch overhead is not the whole
#: measurement.  Step 6 owns the production sizes.
_SHAPES = [(2, 64), (2, 256), (8, 256), (2, 1024)]


def _rng_mat(rng, shape, dtype):
    a = rng.standard_normal(shape)
    if np.dtype(dtype).kind == "c":
        a = a + 1j * rng.standard_normal(shape)
    return a.astype(dtype)


def _hpd(rng, nq, n, dtype):
    z = _rng_mat(rng, (nq, n, n), dtype)
    return (0.5 * (z + np.conj(np.swapaxes(z, -1, -2)))
            + (n + 4) * np.eye(n)[None]).astype(dtype)


def _put(a, mesh, spec):
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    return jax.device_put(np.asarray(a), NamedSharding(mesh, P(*spec)))


def _time_one(fn, args, warmup, reps, build=None):
    """(compile_seconds, [timed seconds])  — first call is the compile.

    REFUSES to time anything that is not an array, and that refusal is the
    difference between a baseline and a fiction.  MEASURED (gpu2x2,
    2026-08-07): ``plan.batched`` on cuSOLVERMp cholesky returns a
    ``CusolverMpBatchedLowerL`` HANDLE, ``jax.block_until_ready`` on a
    handle waits for nothing, and the row came back at **0.032 ms** —
    flat across n = 64 and n = 1024, which is the tell.  A number that
    does not move with the problem size is not a measurement of the
    problem.

    ``build`` REBUILDS the operands for every call, and the donating ops
    need it.  Also measured (cpu2x2 / gpu2x2, 2026-08-07) and recorded in
    those baseline files as ERROR rows rather than quietly dropped::

        solve_lu distributed [2, 64] ->
          ValueError: INVALID_ARGUMENT: Invalid buffer passed: buffer has
          been deleted or donated.

    ``solve_lu`` donates BOTH A and B (C6: donation is declared per OP),
    so the second call of a warmup loop was handed a deleted buffer.  Not
    a library defect — the bench was reusing operands the library had been
    told it could consume.  Rebuilding them puts the host->device staging
    inside the timed interval for those ops, which is why the row says so:
    ``operands='fresh per call (donated)'``.
    """
    import jax
    args = build() if build else args
    out = fn(*args)
    if not hasattr(out, "shape") and not isinstance(out, (tuple, list)):
        raise TypeError(
            f"refusing to time a {type(out).__name__}: block_until_ready "
            f"cannot wait on it, so the elapsed time would be dispatch "
            f"overhead wearing the name of a factorization")
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*(build() if build else args)))
    compile_s = time.perf_counter() - t0
    for _ in range(warmup):
        jax.block_until_ready(fn(*(build() if build else args)))
    out = []
    for _ in range(reps):
        a = build() if build else args
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*a))
        out.append(time.perf_counter() - t0)
    return compile_s, out


def run(mesh, *, dtype="complex128", warmup=2, reps=5, only=""):
    import jax
    rows = []
    px, py = int(mesh.shape["x"]), int(mesh.shape["y"])
    for op, backend in _ROWS:
        if only and only not in f"{op}/{backend}":
            continue
        for nq, n in _SHAPES:
            tag = f"{op}/{backend} nq={nq} n={n}"
            try:
                resolved = D.resolve_backend(op, backend, mesh, n=n)
            except (ValueError, RuntimeError) as exc:
                # Not a failure: the guard ladder said this combination
                # does not apply on this mesh.  RECORDED, because "no row"
                # and "row refused" are different facts and a baseline
                # file that omits the second is unreadable later.
                rows.append(dict(op=op, backend=backend, shape=[nq, n],
                                 mesh=f"{px}x{py}", refused=" ".join(
                                     str(exc).split())[:240]))
                continue
            rng = np.random.default_rng(20260807)
            A_np = _hpd(rng, nq, n, dtype)
            B_np = _rng_mat(rng, (nq, n, max(8, n // 4)), dtype)
            A = _put(A_np, mesh, (None, "x", "y"))
            build = None
            if op == "solve_lu":
                # DONATES A and B (C6), so every call needs its own.
                def build(_A=A_np, _B=B_np):
                    return (_put(_A, mesh, (None, "x", "y")),
                            _put(_B, mesh, (None, "x", "y")))
                fn, args = D.plan(op, mesh, backend=backend, n=n).batched, build()
            elif op == "cholesky" and resolved in ("slate", "cusolvermp"):
                # MEASURED (cpu1x1 baseline, 2026-08-07): plan.batched does
                # not give these a timeable array -- their factor is a
                # library HANDLE.  SLATE's single-tile potrf is declared
                # ``one_handle`` in plan._IMPL and plan.batched REFUSES it
                # by name; cuSOLVERMp's stacked potrf returns a handle of
                # its own.  Timing the refusal would have left the two
                # distributed cholesky backends with no row at all, which
                # is the shape of a baseline table that quietly measures
                # only the easy half.  factor()+solve() is the route those
                # backends actually have, so it is the route timed.
                # factor() DONATES A and solve() DONATES B, so this
                # route needs fresh operands per call for the same reason
                # solve_lu does.  MEASURED (gpu2x2, 2026-08-07): without
                # it, cholesky/cusolvermp came back as four
                # XlaRuntimeError rows.  On the CPU 2x2 the slate rows
                # happened to survive, which is exactly why the fix is
                # keyed on the ROUTE and not on which one failed today.
                def build(_A=A_np, _B=B_np):
                    return (_put(_A, mesh, (None, "x", "y")),
                            _put(_B, mesh, (None, "x", "y")))

                def fn(a, b, _op=op, _bk=backend, _n=n):
                    return D.solve(
                        D.factor(_op, a, mesh, backend=_bk, n=_n), b)
                args = build()
            else:
                fn, args = D.plan(op, mesh, backend=backend, n=n).batched, (A,)
            try:
                compile_s, ts = _time_one(fn, args, warmup, reps, build)
            except Exception as exc:                           # noqa: BLE001
                rows.append(dict(op=op, backend=backend, shape=[nq, n],
                                 mesh=f"{px}x{py}", resolved=resolved,
                                 error=f"{type(exc).__name__}: "
                                       f"{' '.join(str(exc).split())[:240]}"))
                if jax.process_index() == 0:
                    print(f"ERROR {tag}: {type(exc).__name__}", flush=True)
                continue
            route = ("factor+solve"
                     if op == "cholesky" and resolved in ("slate",
                                                          "cusolvermp")
                     else "plan.batched")
            row = dict(op=op, backend=backend, resolved=resolved,
                       route=route,
                       operands=("fresh per call (donated)" if build
                                 else "reused"),
                       shape=[nq, n], dtype=dtype, mesh=f"{px}x{py}",
                       seconds=statistics.median(ts),
                       seconds_min=min(ts), seconds_max=max(ts),
                       compile_seconds=compile_s, reps=reps)
            rows.append(row)
            if jax.process_index() == 0:
                print(f"{tag:44s} -> {resolved:11s} "
                      f"median {row['seconds'] * 1e3:9.3f} ms  "
                      f"(min {min(ts) * 1e3:.3f} max {max(ts) * 1e3:.3f}, "
                      f"compile {compile_s:.2f} s)", flush=True)
    return rows


def main():
    import jax
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--tag", required=True,
                    help="baseline file name stem, e.g. cpu2x2")
    ap.add_argument("--only", default="")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--dtype", default="complex128")
    args = ap.parse_args()

    from jax.sharding import Mesh
    px, py = (int(v) for v in args.mesh.lower().split("x"))
    mesh = Mesh(np.asarray(jax.devices()).reshape(px, py), ("x", "y"))
    rows = run(mesh, dtype=args.dtype, warmup=args.warmup, reps=args.reps,
               only=args.only)

    if jax.process_index() != 0:
        return 0
    doc = dict(
        tag=args.tag,
        jobid=os.environ.get("SLURM_JOB_ID", os.environ.get("SLURM_JOBID", "")),
        nodes=int(os.environ.get("SLURM_JOB_NUM_NODES", "1")),
        processes=jax.process_count(),
        devices=jax.device_count(),
        backend=jax.default_backend(),
        mesh=args.mesh,
        jax_version=jax.__version__,
        machine=os.environ.get("NERSC_HOST", os.environ.get("LX_MACHINE", "")),
        ffi_so=os.environ.get("LORRAX_FFI_SO", ""),
        ffi_host_so=os.environ.get("LORRAX_FFI_HOST_SO", ""),
        recorded=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        rows=rows,
    )
    out = os.path.join(_BENCH, "baselines", f"{args.tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
        fh.write("\n")
    timed = sum(1 for r in rows if "seconds" in r)
    print(f"wrote {out}: {len(rows)} rows, {timed} timed "
          f"({os.path.getsize(out)} bytes)", flush=True)
    # Judge by artifacts: a baseline file with no timed rows is not a
    # measurement, and saying so here is cheaper than discovering it in a
    # diff three steps later.
    return 0 if timed else 1


if __name__ == "__main__":
    sys.exit(main())

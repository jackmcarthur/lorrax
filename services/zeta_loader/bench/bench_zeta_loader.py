"""Microbench: the two ζ read plans, at 1x1 and at a real 2x2.

RECORD, DO NOT OPTIMIZE.  Step 2's job is to make the numbers exist and be
comparable; step 6 changes them.  Nothing here is a test and nothing here has
a threshold — a bench cell that fails a run is a slow test wearing a
different name, and the charter is explicit that the default suite stays
seconds-fast and that perf lands as recorded baseline files.

    lx run --cpu -N 1 -n 4 python3 \\
        services/zeta_loader/bench/bench_zeta_loader.py \\
        --mesh 2x2 --tag cpu2x2 --tmpdir $SCRATCH/svc_zeta/bench
    lx run --cpu -N 1 -n 1 python3 \\
        services/zeta_loader/bench/bench_zeta_loader.py \\
        --mesh 1x1 --tag cpu1x1 --tmpdir $SCRATCH/svc_zeta/bench

One JSON row per (op, shape, mesh, nodes, seconds, MB/s, jobid) into
``baselines/``, the claims-style shape the charter names.  Regression
detection is DIFFING BASELINE FILES ACROSS BRANCHES, not asserting a number
in a test: a threshold that has to hold on a shared filesystem either gets
loosened until it means nothing or fails on somebody else's contention — and
this bench is I/O bound on a Lustre scratch shared with the whole machine,
which is the worst possible place for a threshold.

THE ONE NUMBER THIS EXISTS TO PRODUCE  (survey §7.2 G7)

``read_zeta_G_slab`` is the hot V_q read and has NO claims-style baseline.
``slab_io``'s calibration band is the reference:

    dd                          725 MB/s
    serial h5py                 967 MB/s
    SlabIO, SAME HANDLE         953-961 MB/s
    SlabIO, open-close per leg  410-580 MB/s

``ZetaLoader`` holds ONE SlabIO handle open for the loader's lifetime
precisely to stay in the first band, and NOTHING MEASURES THAT IT STILL
DOES.  So the driver records both routes explicitly — ``held-open`` (the
loader's design: open once, read N times) and ``open-close`` (a fresh loader
per read) — because the gap between them IS the claim.  A run where the two
are equal means the amortisation stopped working, and that is invisible in a
single number.

``read_zeta_G_local`` rides along: it is serial h5py through the loader's own
handle, so its band is the 967 MB/s row, and having it next to the collective
number is what makes "is the collective path paying for itself" answerable.

WHY THE FIXTURE IS WRITTEN BY THIS SCRIPT AND NOT SHIPPED.  A ζ big enough to
measure is hundreds of MB; committing one would be a binary in the repo, and
pointing at a production ζ would make the numbers depend on somebody else's
run.  Rank 0 writes a synthetic file of the requested shape into ``--tmpdir``
(a SHARED path) and every rank reads it, which is the same build-and-barrier
the L-c suite uses.

NO BASELINE JSON IS INVENTED HERE.  ``baselines/`` ships empty: every number
in it must come from the machine it claims to describe, and this dev box has
no phdf5 FFI at all (every SlabIO read refuses at open).  A WSL-authored
"baseline" would be a fiction with a filename.
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
_REPO = os.path.dirname(_SERVICES)
for _p in (os.path.join(_SERVICES, "lxkit", "src"),
           os.path.join(_SERVICES, "zeta_loader", "src"),
           os.path.join(_REPO, "src"),
           os.path.join(_SERVICES, "zeta_loader", "tests")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

if __name__ == "__main__":
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        # select_gpu.sh already narrowed CUDA_VISIBLE_DEVICES to one GPU
        # per rank; a bare initialize() narrows AGAIN by local process id
        # and rank k dies on ordinal k of a 1-device view.  Same cure as
        # the multiproc CLI and distrib_la's (local_device_ids=[0]).
        from lxkit.gate import platform_from_env
        import jax
        if platform_from_env() == "CUDA":
            jax.distributed.initialize(local_device_ids=[0])
        else:
            jax.distributed.initialize()

import zeta_synth as Z                                         # noqa: E402

#: ``(n_q, n_rmu, ngkmax)``.  Chosen so the largest is a few hundred MB — big
#: enough that the transport rather than the dispatch is being measured, small
#: enough that a full sweep is a couple of minutes on a shared scratch.  Step
#: 6 owns the production sizes (CrI3 6x6 30Ry bispinor is n_rmu 300 / ngkmax
#: in the thousands).
_SHAPES = [(8, 64, 512), (8, 256, 1024), (32, 256, 1024), (8, 512, 2048)]

_BYTES_PER = 16                                    # complex128


def _barrier(tag):
    import jax
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils
        multihost_utils.sync_global_devices(tag)


def _build(path, n_q, n_rmu, ngkmax):
    """Rank 0 writes; everybody waits.  Reuses an existing file of the same
    shape, because rebuilding a 400 MB fixture per invocation is most of the
    wall clock and none of the measurement."""
    import jax
    if jax.process_index() == 0 and not os.path.exists(path):
        Z.build_gflat(path, n_q=n_q, n_rmu=n_rmu, ngkmax=ngkmax)
    _barrier(f"bench_fixture:{os.path.basename(path)}")
    return path


def _time(fn, warmup, reps):
    """``(compile_seconds, [timed seconds])`` — the first call is discarded.

    The FIRST call of every (route, shape) pays the phdf5 context setup and,
    on the collective route, an XLA compile; it is recorded separately rather
    than folded in, because "how long does the first read take" is a real
    question with a different answer and conflating them is how a handle-
    reuse regression goes unnoticed.  MEDIAN of the rest, because a shared
    Lustre produces occasional long tails that a mean folds into the number
    and hides.
    """
    import jax
    t0 = time.perf_counter()
    jax.block_until_ready(fn())
    compile_s = time.perf_counter() - t0
    for _ in range(warmup):
        jax.block_until_ready(fn())
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        out.append(time.perf_counter() - t0)
    return compile_s, out


def run(mesh, tmpdir, *, warmup=1, reps=5, only=""):
    import jax
    from zeta_loader import ZetaLoader

    px, py = int(mesh.shape["x"]), int(mesh.shape["y"])
    world = px * py
    p0 = print if jax.process_index() == 0 else (lambda *a, **k: None)
    rows = []
    for n_q, n_rmu, ngkmax in _SHAPES:
        # μ is sharded over ('x','y'), so the requested extent must divide
        # the mesh.  Round UP, which is what production does (v_q_g_flat
        # reads at n_rmu_padded) and what makes the pad path part of the
        # measurement rather than an untimed special case.
        mu_count = -(-n_rmu // world) * world
        nbytes = n_q * mu_count * ngkmax * _BYTES_PER
        path = os.path.join(tmpdir, f"bench_zeta_{n_q}_{n_rmu}_{ngkmax}.h5")
        _build(path, n_q, n_rmu, ngkmax)

        for op in ("read_zeta_G_slab.held_open",
                   "read_zeta_G_slab.open_close",
                   "read_zeta_G_local"):
            if only and only not in op:
                continue
            tag = f"{op} nq={n_q} mu={n_rmu}->{mu_count} ngk={ngkmax}"
            try:
                if op == "read_zeta_G_slab.held_open":
                    zl = ZetaLoader(path, mesh=mesh)
                    try:
                        compile_s, ts = _time(
                            lambda: zl.read_zeta_G_slab(
                                q_offset=0, q_count=n_q, mu_offset=0,
                                mu_count=mu_count),
                            warmup, reps)
                    finally:
                        zl.close()
                    read_bytes = nbytes
                elif op == "read_zeta_G_slab.open_close":
                    # A FRESH loader per call: the 410-580 MB/s band, and the
                    # thing the held-open design exists to avoid.  Timing it
                    # is what turns "the design amortises the ctx" into a
                    # number instead of a comment.
                    def _fresh():
                        with ZetaLoader(path, mesh=mesh) as z:
                            return z.read_zeta_G_slab(
                                q_offset=0, q_count=n_q, mu_offset=0,
                                mu_count=mu_count)
                    compile_s, ts = _time(_fresh, warmup, reps)
                    read_bytes = nbytes
                else:
                    zl = ZetaLoader(path, mesh=mesh)
                    try:
                        compile_s, ts = _time(
                            lambda: zl.read_zeta_G_local(slice(0, n_q)),
                            warmup, reps)
                    finally:
                        zl.close()
                    # The LOCAL plan reads the on-disk extent, not the pad.
                    read_bytes = n_q * n_rmu * ngkmax * _BYTES_PER
            except Exception as exc:                           # noqa: BLE001
                rows.append(dict(
                    op=op, shape=[n_q, n_rmu, ngkmax], mu_count=mu_count,
                    mesh=f"{px}x{py}",
                    error=f"{type(exc).__name__}: "
                          f"{' '.join(str(exc).split())[:300]}"))
                p0(f"ERROR {tag}: {type(exc).__name__}", flush=True)
                continue
            median = statistics.median(ts)
            row = dict(op=op, shape=[n_q, n_rmu, ngkmax], mu_count=mu_count,
                       dtype="complex128", mesh=f"{px}x{py}",
                       bytes=read_bytes,
                       seconds=median, seconds_min=min(ts),
                       seconds_max=max(ts), compile_seconds=compile_s,
                       mb_per_s=(read_bytes / 1e6) / median if median else None,
                       reps=reps)
            rows.append(row)
            p0(f"{tag:56s} -> median {median * 1e3:9.3f} ms  "
               f"{row['mb_per_s']:7.1f} MB/s  (first call {compile_s:.3f} s)",
               flush=True)
    return rows


def main():
    import jax
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--tag", required=True,
                    help="baseline file name stem, e.g. cpu2x2")
    ap.add_argument("--tmpdir", required=True,
                    help="SHARED-FS scratch for the fixtures ($SCRATCH, not "
                         "/tmp): rank 0 writes them, every rank reads them")
    ap.add_argument("--only", default="")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    from jax.sharding import Mesh
    px, py = (int(v) for v in args.mesh.lower().split("x"))
    mesh = Mesh(np.asarray(jax.devices()).reshape(px, py), ("x", "y"))
    if jax.process_index() == 0:
        os.makedirs(args.tmpdir, exist_ok=True)
    _barrier("bench_tmpdir_ready")
    rows = run(mesh, args.tmpdir, warmup=args.warmup, reps=args.reps,
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
        # The reference band this table is read against (slab_io's own
        # calibration).  Carried IN the artifact so a baseline file is
        # interpretable without the docstring that produced it.
        reference_band_mb_s=dict(dd=725, serial_h5py=967,
                                 slabio_same_handle=[953, 961],
                                 slabio_open_close=[410, 580]),
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

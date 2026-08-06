#!/usr/bin/env python3
"""SlabIO phdf5_ffi ADVERSARIAL SCALING benchmark.

Differences from ``slabio_sweep.py`` (the harness that produced the original
optimistic table), all of them deliberate:

1. **Write and read are separate PHASES, hence separate processes.**  The old
   harness read the file it had just written in the same process, so every
   "read" number was served from the node's page cache.  Run ``--phase write``
   on one node set and ``--phase read`` on a DISJOINT node set and the read is
   genuinely cold: no client on the reading nodes has ever seen those pages.
   (``lctl`` is absent and ``drop_caches`` needs root on Perlmutter computes,
   so disjoint node sets are the only honest cold-read instrument available.)

2. **File size is decoupled from memory** by ``slabs``: the dataset is created
   at ``(slabs * rows, cols)`` and written by ``slabs`` successive
   ``write_slab`` calls at increasing offset.  That is exactly the production
   ``V_qmunu`` (nq, N_mu, N_mu) pattern, and it lets a 4-node job produce a
   file far larger than its aggregate HBM.

3. **Per-config geometry.**  Each config may override rows/cols/slabs, so
   per-rank tile size, total file size, and rank count move independently --
   they are three axes, not one.

Every row carries geometry, payload bytes, per-rank tile bytes, the striping
actually requested, and the phase's cold/warm status.
"""
import argparse
import ctypes
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.environ.get(
    "LORRAX_SRC", "/global/u2/j/jackm/software/lorrax_P/src"))

ap = argparse.ArgumentParser()
ap.add_argument("--platform", default="gpu", choices=("gpu", "cpu"))
ap.add_argument("--dir", required=True)
ap.add_argument("--phase", required=True, choices=("write", "read", "both"))
ap.add_argument("--tag", required=True,
                help="path namespace; read phase must use the write phase tag")
ap.add_argument("--rows", type=int, default=32768, help="rows per slab")
ap.add_argument("--cols", type=int, default=4096)
ap.add_argument("--slabs", type=int, default=1)
ap.add_argument("--reps", type=int, default=1)
ap.add_argument("--verify", action="store_true")
ap.add_argument("--configs", required=True, help="JSON list or @file")
ap.add_argument("--out", default="")
args = ap.parse_args()

CONFIGS = (json.load(open(args.configs[1:])) if args.configs.startswith("@")
           else json.loads(args.configs))

from runtime import initialize_communicator_stack          # noqa: E402
RUNTIME = initialize_communicator_stack(platform=args.platform)

import numpy as np                                          # noqa: E402
import jax                                                  # noqa: E402
import jax.numpy as jnp                                     # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P  # noqa: E402
from jax.experimental.multihost_utils import sync_global_devices  # noqa: E402

mesh = RUNTIME.mesh
PID = jax.process_index()
NPROC = jax.process_count()


def p0(*a):
    if PID == 0:
        print(*a, flush=True)


# --- world-size assertion: never trust a launcher ---------------------------
from ffi.common.ffi_loader import phdf5_init_mpi            # noqa: E402
phdf5_init_mpi("CUDA" if args.platform == "gpu" else "cpu")
_mpi = None
for _so in ("libmpi.so.12", "libmpi_gnu_123.so.12", "libmpi.so"):
    try:
        _mpi = ctypes.CDLL(_so)
        break
    except OSError:
        continue
_sz = ctypes.c_int(-1)
_mpi.MPI_Comm_size(ctypes.c_int(0x44000000), ctypes.byref(_sz))
if _sz.value != NPROC:
    raise SystemExit(f"FATAL SINGLETON-MPI: MPI_Comm_size={_sz.value} "
                     f"!= process_count={NPROC}")

NODELIST = os.environ.get("SLURM_JOB_NODELIST", "?")
NNODES = int(os.environ.get("SLURM_NNODES", "0"))
p0(f"[bench] phase={args.phase} tag={args.tag} ranks={NPROC} "
   f"nodes={NNODES} mesh={dict(mesh.shape)} MPI_Comm_size={_sz.value}")
p0(f"[bench] nodelist={NODELIST}")

from file_io.slab_io import SlabIO                          # noqa: E402

SPEC = P("x", "y")
SH = NamedSharding(mesh, SPEC)


def make_tile(nr, nc):
    @jax.jit
    def _gen():
        i = jax.lax.broadcasted_iota(jnp.int32, (nr, nc), 0)
        j = jax.lax.broadcasted_iota(jnp.int32, (nr, nc), 1)
        v = jnp.sin((i.astype(jnp.float64) * nc + j.astype(jnp.float64)) * 1e-4)
        return jax.lax.with_sharding_constraint(
            (v + 1j * (v * 0.5)).astype(jnp.complex128), SH)
    return jax.block_until_ready(_gen())


def stripe_of(path):
    """Read back the layout the file ACTUALLY got.  Never assume the hint."""
    try:
        out = subprocess.run(["lfs", "getstripe", "-c", "-S", path],
                             capture_output=True, text=True, timeout=30)
        vals = [ln.strip() for ln in out.stdout.split() if ln.strip().isdigit()]
        if len(vals) >= 2:
            return f"{vals[0]}x{int(vals[1]) // 2**20}M"
    except Exception:
        pass
    return "?"


_TUNING = [k for k in list(os.environ)
           if k.startswith("LORRAX_PHDF5_") or k.startswith("MPICH_MPIIO_")]
_BASE = {k: os.environ[k] for k in _TUNING}

rows_out = []
for ci, cfg in enumerate(CONFIGS):
    name = cfg["name"]
    envd = cfg.get("env", {})
    NR = int(cfg.get("rows", args.rows))
    NC = int(cfg.get("cols", args.cols))
    NS = int(cfg.get("slabs", args.slabs))
    reps = int(cfg.get("reps", args.reps))

    tile_bytes = NR * NC * 16
    total_bytes = tile_bytes * NS
    per_rank_tile = tile_bytes / NPROC

    # Reset tuning env to launch baseline, then apply this config.  The C++
    # context re-reads its knobs at open_file, and ffi.io caches one context
    # per PATH, so a fresh path genuinely re-reads them.
    for k in list(os.environ):
        if (k.startswith("LORRAX_PHDF5_") or k.startswith("MPICH_MPIIO_")) \
                and k not in _BASE:
            del os.environ[k]
    os.environ.update(_BASE)
    for k, v in envd.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)

    A = make_tile(NR, NC) if args.phase in ("write", "both") else None

    best_w = best_r = 0.0
    err = float("nan")
    layout = "?"
    for rep in range(reps):
        path = os.path.join(args.dir, f"{args.tag}_c{ci:02d}_r{rep}.h5")

        if args.phase in ("write", "both"):
            sync_global_devices(f"w-{ci}-{rep}")
            t0 = time.perf_counter()
            with SlabIO(path, mode="w", mesh=mesh) as io:
                io.create_dataset("A", shape=(NR * NS, NC),
                                  dtype=jnp.complex128)
                for s in range(NS):
                    io.write_slab("A", A, offset=(s * NR, 0))
            sync_global_devices(f"wd-{ci}-{rep}")
            tw = time.perf_counter() - t0
            best_w = max(best_w, total_bytes / 2**30 / tw)

        if args.phase in ("read", "both"):
            if not os.path.exists(path):
                p0(f"[row] {name:38s} MISSING {path}")
                continue
            sync_global_devices(f"r-{ci}-{rep}")
            t0 = time.perf_counter()
            with SlabIO(path, mode="r", mesh=mesh) as io:
                for s in range(NS):
                    B = io.read_slab("A", shape=(NR, NC),
                                     dtype=jnp.complex128,
                                     offset=(s * NR, 0), mesh=mesh,
                                     partition_spec=SPEC)
                    B = jax.block_until_ready(B)
            sync_global_devices(f"rd-{ci}-{rep}")
            tr = time.perf_counter() - t0
            best_r = max(best_r, total_bytes / 2**30 / tr)
            if args.verify and rep == reps - 1:
                ref = make_tile(NR, NC)
                err = float(jax.device_get(jnp.max(jnp.abs(B - ref))))

        if PID == 0 and layout == "?":
            layout = stripe_of(path)

    rows_out.append(dict(
        name=name, env=envd, rows=NR, cols=NC, slabs=NS,
        total_gib=total_bytes / 2**30,
        per_rank_tile_mib=per_rank_tile / 2**20,
        ranks=NPROC, nodes=NNODES, layout=layout,
        write=best_w, read=best_r, err=err, phase=args.phase))
    p0(f"[row] {name:38s} {total_bytes/2**30:8.2f}GiB "
       f"tile/rank={per_rank_tile/2**20:9.2f}MiB {layout:>9s} "
       f"W {best_w:7.3f}  R {best_r:7.3f} GiB/s")

p0("")
hdr = (f"{'CONFIG':38s} {'GiB':>8s} {'tile/rk MiB':>12s} {'layout':>9s} "
       f"{'W GiB/s':>9s} {'R GiB/s':>9s}")
p0(hdr)
for r in rows_out:
    p0(f"{r['name']:38s} {r['total_gib']:8.2f} {r['per_rank_tile_mib']:12.2f} "
       f"{r['layout']:>9s} {r['write']:9.3f} {r['read']:9.3f}")

if args.out and PID == 0:
    json.dump(dict(phase=args.phase, tag=args.tag, ranks=NPROC,
                   nodes=NNODES, nodelist=NODELIST,
                   mesh=dict(mesh.shape), rows=rows_out),
              open(args.out, "w"), indent=1)
    p0(f"[bench] wrote {args.out}")
sync_global_devices("bench-done")

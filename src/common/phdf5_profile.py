"""Capture an xprof trace of :meth:`PhdfWfnReader.coeffs_gspace`.

Drives a small synthetic read loop — a handful of band chunks on the
nosym MoS2 3×3 WFN or the sym Si 4×4×4 WFN — and writes a JAX
profiler bundle per section.  Parse the trace on the login node with
the recipe in ``reports/ppm_sigma_profiling_2026-04-05/XPROF_TRACE_GUIDE.md``.

Usage (4-GPU)::

    lxalloc
    export SLURM_JOBID=<from lxalloc>
    export ISDF_JAX_PROFILE_DIR=$PWD/jax_profiles
    LORRAX_NGPU=4 LORRAX_MPI_TYPE=pmix \\
        lxrun python3 -u -m common.phdf5_profile \\
            --wfn /path/to/WFN.h5 --band-chunk-size 20 --n-chunks 4
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

import argparse
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)

_DIST = "_LORRAX_JAX_DISTRIBUTED_DONE"
def _init():
    if os.environ.get(_DIST):
        return
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        try:
            jax.distributed.initialize()
        except Exception:
            pass
    os.environ[_DIST] = "1"
_init()

from jax.experimental import multihost_utils
from jax.sharding import Mesh

from common import jax_profile
from common.phdf5_wfn_reader import PhdfWfnReader


def log(msg: str) -> None:
    if jax.process_index() == 0:
        print(msg, flush=True)


def sync(tag: str) -> None:
    try:
        multihost_utils.sync_global_devices(tag)
    except Exception:
        pass


def build_mesh(world: int) -> Mesh:
    if world == 4:
        p, q = 2, 2
    elif world == 1:
        p, q = 1, 1
    else:
        p, q = world, 1
    return Mesh(np.asarray(jax.devices()).reshape(p, q), axis_names=("x", "y"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--wfn",
        default=("/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/"
                 "02_mos2_3x3_nosym/qe/nscf/WFN.h5"))
    ap.add_argument("--band-chunk-size", type=int, default=20)
    ap.add_argument("--n-chunks", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--timed", type=int, default=4)
    ap.add_argument("--kchunk", type=int, default=0,
                    help="if >0, split full-k into chunks of this size.")
    args = ap.parse_args()

    world = jax.process_count()
    mesh = build_mesh(world)

    reader = PhdfWfnReader(args.wfn, mesh=mesh)
    try:
        nb_total = args.band_chunk_size * args.n_chunks
        if nb_total > reader.nbands:
            log(f"ERROR: want {nb_total} bands but wfn has only {reader.nbands}")
            return 1
        band_ranges = [
            (i * args.band_chunk_size, (i + 1) * args.band_chunk_size)
            for i in range(args.n_chunks)
        ]
        nk = reader.nk_full
        if args.kchunk > 0 and args.kchunk < nk:
            k_id_chunks = [
                np.arange(k0, min(k0 + args.kchunk, nk), dtype=np.int32)
                for k0 in range(0, nk, args.kchunk)
            ]
        else:
            k_id_chunks = [None]

        log(f"world={world}  wfn={args.wfn}")
        log(f"band_chunk_size={args.band_chunk_size}  n_chunks={args.n_chunks}  "
            f"n_kchunks={len(k_id_chunks)}  ntran={reader.ntran}")

        def one_pass():
            for br in band_ranges:
                for k_ids in k_id_chunks:
                    out = reader.coeffs_gspace(br, k_ids=k_ids)
            jax.block_until_ready(out)

        # Warmup (compiles shape specializations).
        for _ in range(args.warmup):
            one_pass()
        sync("warmup_done")

        # Time without profiler first.
        wall = []
        for _ in range(args.timed):
            t0 = time.perf_counter()
            one_pass()
            wall.append(time.perf_counter() - t0)
        log(f"  mean wall over {args.timed} iters = "
            f"{np.mean(wall) * 1e3:.1f} ms   "
            f"min = {np.min(wall) * 1e3:.1f} ms")

        # Capture one traced iter.
        sync("trace_start")
        with jax_profile.trace_section("coeffs_gspace"):
            one_pass()
        sync("trace_end")
        log("  trace dumped (if ISDF_JAX_PROFILE_DIR is set).")
    finally:
        reader.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

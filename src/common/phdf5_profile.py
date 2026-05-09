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
from file_io import WFNReader
from common.symmetry_maps import SymMaps
from common.load_wfns import load_centroids_band_chunked
import types


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
    ap.add_argument("--centroids", action="store_true",
                    help="instead of reader-only, profile the full "
                         "``load_centroids_band_chunked`` loop (read + FFT + "
                         "centroid gather) for both legacy and phdf5 paths. "
                         "Warms each path once before tracing.")
    ap.add_argument("--n-centroids", type=int, default=16)
    ap.add_argument("--drop-cache", action="store_true",
                    help="``posix_fadvise(POSIX_FADV_DONTNEED)`` the WFN "
                         "file before every timed iter so both paths hit "
                         "disk instead of the OS page cache — necessary for "
                         "a fair legacy-vs-phdf5 I/O comparison on small "
                         "WFN files that would otherwise be RAM-resident.")
    ap.add_argument("--memory-per-device-gb", type=float, default=0.0,
                    help="Force ``meta.memory_per_device_gb`` to this "
                         "value so the k-chunker sizes against a known "
                         "budget; a post-run ``peak_bytes_in_use`` readout "
                         "then tells us whether the 9x peak-copies "
                         "heuristic lines up with what XLA actually "
                         "allocates.")
    args = ap.parse_args()

    world = jax.process_count()
    mesh = build_mesh(world)

    if args.centroids:
        return _centroids_mode(args, world, mesh)

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


def _drop_wfn_cache(path: str) -> None:
    """Hint the kernel to drop the WFN file's cached pages.  Best-
    effort — works only if nothing else has the file mmapped; between
    load_wfns calls in this driver that's satisfied.  On MPI jobs only
    rank 0 issues the call; the kernel's page cache is node-global so
    the other ranks benefit without extra syscalls."""
    if jax.process_index() != 0:
        return
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError:
        pass


def _centroids_mode(args, world: int, mesh: Mesh) -> int:
    """Warm + trace ``load_centroids_band_chunked`` for both the legacy
    and phdf5 paths.  Each path is called three times: once to trigger
    compile, twice to get a timed mean, then once more inside a
    ``trace_section`` to dump a profiler bundle for the steady-state
    iter.  This gives a fair overlap comparison — cold compile stays
    out of the trace and only the read/FFT/gather loop is measured.
    """
    wfn = WFNReader(args.wfn)
    sym = SymMaps(wfn)
    fft_grid = tuple(int(x) for x in wfn.fft_grid)
    meta = types.SimpleNamespace(
        fft_grid=fft_grid, nspinor=int(wfn.nspinor),
        nspinor_wfnfile=int(wfn.nspinor),
        nk_tot=int(sym.nk_tot),
        kgrid=tuple(int(x) for x in wfn.kgrid),
        memory_per_device_gb=args.memory_per_device_gb)
    nb_total = args.band_chunk_size * args.n_chunks
    if nb_total > wfn.nbands:
        log(f"ERROR: want {nb_total} bands but wfn has only {wfn.nbands}")
        return 1
    rng = np.random.default_rng(42)
    n_rtot = int(np.prod(fft_grid))
    flat = rng.choice(n_rtot, size=args.n_centroids, replace=False)
    nx, ny, nz = fft_grid
    centroids_np = np.stack(
        [flat // (ny * nz), (flat // nz) % ny, flat % nz], axis=1
    ).astype(np.int32)
    centroids = jnp.asarray(centroids_np)

    base_args = dict(
        wfn=wfn, sym=sym, meta=meta,
        centroid_indices=centroids,
        bispinor=False, mesh_xy=mesh,
        band_range=(0, nb_total),
        band_chunk_size=args.band_chunk_size,
    )

    def run(use_phdf5: bool, drop_cache: bool = False) -> float:
        if drop_cache:
            _drop_wfn_cache(args.wfn)
        sync(f"run_{use_phdf5}_start")
        t0 = time.perf_counter()
        out = load_centroids_band_chunked(**base_args, use_phdf5=use_phdf5)
        jax.block_until_ready(out)
        dt = time.perf_counter() - t0
        sync(f"run_{use_phdf5}_end")
        return dt

    log(f"world={world}  wfn={args.wfn}")
    log(f"band_chunk_size={args.band_chunk_size}  n_chunks={args.n_chunks}  "
        f"nk_tot={sym.nk_tot}  n_centroids={args.n_centroids}")

    cache_tag = " (drop-cache)" if args.drop_cache else " (cached)"
    for path_name, flag in (("legacy", False), ("phdf5", True)):
        # Warm once (compile), then time.
        run(flag)
        walls = [run(flag, drop_cache=args.drop_cache)
                 for _ in range(args.timed)]
        log(f"  {path_name:<6}{cache_tag} wall: "
            f"mean {np.mean(walls)*1e3:7.1f} ms  "
            f"min {np.min(walls)*1e3:7.1f} ms")
        with jax_profile.trace_section(f"centroids_{path_name}"):
            run(flag, drop_cache=args.drop_cache)
        # Report per-device peak high-water mark for this path.
        peak = jax.local_devices()[0].memory_stats().get(
            "peak_bytes_in_use", 0)
        log(f"  {path_name:<6} local-device peak_bytes_in_use: "
            f"{peak/1e9:.2f} GB"
            + (f"   budget={args.memory_per_device_gb:.2f} GB"
               if args.memory_per_device_gb > 0 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

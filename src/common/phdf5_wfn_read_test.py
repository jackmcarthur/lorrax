"""Parallel WFN.h5 read via phdf5 FFI + safe-corner scatter into FFT box.

End-to-end correctness + timing harness: compares the existing
``common.load_wfns.read_Gvecs_to_devices`` pipeline against a phdf5-FFI
parallel-read path that fans the wavefunction coefficients directly into
device-local band shards without a rank-0 round trip.

Two paths, both produce a real-space band-sharded wavefunction array of
shape ``(nk, nb_padded, nspinor, nx, ny, nz)`` with
``P(None, ('x','y'), None, None, None, None)``:

  (B) baseline
      WFNReader(path)  →  read_Gvecs_to_devices  →  jitted iFFT
      WFNReader eagerly slurps ``wfns/coeffs[:]`` into host RAM on every
      rank, then read_Gvecs_to_devices does per-k host-side scatter-into-
      G-space-FFT-box + make_array_from_process_local_data.

  (F) FFI parallel read
      SlabIO(mode='r', use_ffi_io=True)  →
          per-k collective H5Dread of (nb, ns, ngkmax, 2) sharded along
          the band axis over the combined ('x','y') mesh axes, no h5py
          slurp, no host copy of the whole file.
      jitted post-read kernel: mask trailing rows to zero + remap their
      gvec indices to the safe corner ``fft_grid // 2`` (guaranteed unused
      for files with ``ecutrho >= 4*ecutwfc``).
      jitted scatter into the (nb_padded, ns, nx, ny, nz) FFT box.
      same iFFT.

v1 limits: nosym files only (``ntran == 1``) so we can skip symmetry
unfolding on the (F) path.  Planned v2 ports the U_spinor + Umklapp + τ
phase from ``SymMaps.get_cnk_fullzone_batch`` into the jitted kernel.

Correctness check: bitwise-equal real-space outputs.
Timing: per-stage walls, iters-mean summary, peak host RSS before/after.

Usage (4-GPU on Perlmutter — requires the phdf5 FFI + PMIx):

    lxalloc
    export SLURM_JOBID=<from lxalloc>
    LORRAX_NGPU=4 LORRAX_MPI_TYPE=pmix \
        src/ffi/common/cpp/run_shifter.sh env \
        XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async \
        HDF5_USE_FILE_LOCKING=FALSE \
        python3 -u -m common.phdf5_wfn_read_test \
            --wfn /pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/qe/nscf/WFN.h5
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

import argparse
import resource
import sys
import time
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

# Multi-process bootstrap (SLURM-aware, same pattern as phdf5_write_test).
_DIST_SENTINEL = "_LORRAX_JAX_DISTRIBUTED_DONE"
def _maybe_init_jax_distributed():
    if os.environ.get(_DIST_SENTINEL):
        return
    proc_count = int(os.environ.get("JAX_PROCESS_COUNT",
                         os.environ.get("JAX_NUM_PROCESSES",
                         os.environ.get("SLURM_NTASKS", "1"))))
    if proc_count > 1:
        try:
            jax.distributed.initialize()
        except Exception:
            pass
    os.environ[_DIST_SENTINEL] = "1"
_maybe_init_jax_distributed()

from jax.experimental import multihost_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.wfnreader import WFNReader
from common.symmetry_maps import SymMaps
from common.load_wfns import read_Gvecs_to_devices
from common.fft_helpers import make_jittable_local_ifftn_3d
from file_io.slab_io import SlabIO


# =========================================================================
# helpers
# =========================================================================
def _log(msg: str) -> None:
    if jax.process_index() == 0:
        print(msg, flush=True)


def _rss_gb() -> float:
    """Peak host RSS this process has seen, in GB.  ru_maxrss is KiB on Linux."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def _sync(tag: str) -> None:
    try:
        multihost_utils.sync_global_devices(tag)
    except Exception:
        pass


def _time(fn, tag: str, *args, wait_array: bool = True, **kw):
    """Sync-time a callable.  Returns (dt_seconds, result).

    ``wait_array`` controls whether we call ``jax.block_until_ready`` on
    the return value — enable for stages whose output is a jax.Array,
    disable for things like file-handle construction that return opaque
    Python objects.
    """
    _sync(f"{tag}_start")
    t0 = time.perf_counter()
    out = fn(*args, **kw)
    if wait_array and isinstance(out, jax.Array):
        jax.block_until_ready(out)
    _sync(f"{tag}_end")
    return time.perf_counter() - t0, out


# =========================================================================
# (F) FFI parallel-read path
# =========================================================================
def _fft_safe_corner(fft_grid: tuple[int, int, int]) -> np.ndarray:
    """The FFT-box slot guaranteed unused by any WFN G-vector when
    ecutrho >= 4*ecutwfc.  See reports/... (TBD).
    """
    nx, ny, nz = (int(x) for x in fft_grid)
    return np.asarray([nx // 2, ny // 2, nz // 2], dtype=np.int32)


def _assert_safe_corner_ok(wfn: WFNReader) -> None:
    """Fail fast if ecutrho < 4*ecutwfc OR any WFN G-vec wraps to the midpoint."""
    ecutrho = float(wfn.ecutrho)
    ecutwfc = float(wfn.ecutwfc)
    if ecutrho + 1e-8 < 4.0 * ecutwfc:
        raise RuntimeError(
            f"safe-corner scatter requires ecutrho >= 4*ecutwfc; "
            f"got ecutrho={ecutrho}, ecutwfc={ecutwfc}")
    # Belt + suspenders: empirical check on the actual gvecs in the file.
    fft = np.asarray(wfn.fft_grid, dtype=np.int64)
    wrapped = wfn.gvecs % fft[None, :]
    mid = fft // 2
    hits = np.all(wrapped == mid[None, :], axis=1)
    if np.any(hits):
        raise RuntimeError(
            f"FFT midpoint {tuple(int(x) for x in mid)} is hit by "
            f"{int(hits.sum())} WFN G-vectors — can't use as safe corner")


def _build_gvecs_slabs(wfn: WFNReader, ngkmax: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute per-k fixed-width gvec slabs and mask-bounds on host.

    Returns
    -------
    gvecs_slabs: (nk, ngkmax, 3) int32 — padded with safe-corner rows where
        the slab extends past ng[k] (or where we had to backshift the slab
        for the last k to avoid overflowing ngktot).
    slab_offsets: (nk,) int64 — the slab_start used for the FFI read
        (kpt_starts[k] except for the last k where it's ngktot-ngkmax).
    valid_starts: (nk,) int32 — start of valid rows within each slab.
    valid_ends:   (nk,) int32 — end   of valid rows within each slab (exclusive).
    """
    nk = int(wfn.nkpts)
    ngk = np.asarray(wfn.ngk, dtype=np.int64)
    kpt_starts = np.asarray(wfn.kpt_starts, dtype=np.int64)
    ngktot = int(kpt_starts[-1] + ngk[-1]) if nk > 0 else 0
    gvecs_all = np.asarray(wfn.gvecs, dtype=np.int32)  # (ngktot, 3)

    slab_offsets = np.zeros(nk, dtype=np.int64)
    valid_starts = np.zeros(nk, dtype=np.int32)
    valid_ends   = np.zeros(nk, dtype=np.int32)
    for k in range(nk):
        # If reading ngkmax rows from kpt_starts[k] would overflow the
        # dataset, backshift the slab so we read the final ngkmax rows
        # and track where the valid data lives within that slab.
        if kpt_starts[k] + ngkmax <= ngktot:
            slab_start = int(kpt_starts[k])
            valid_start = 0
            valid_end = int(ngk[k])
        else:
            slab_start = int(ngktot - ngkmax)
            valid_start = int(kpt_starts[k] - slab_start)
            valid_end = valid_start + int(ngk[k])
        slab_offsets[k] = slab_start
        valid_starts[k] = valid_start
        valid_ends[k] = valid_end

    # Host-build gvecs_slabs with safe-corner padding at the invalid rows
    # so the jitted scatter can unconditionally dispatch — cheap host op
    # since gvecs is small (MB).
    corner = _fft_safe_corner(tuple(int(x) for x in wfn.fft_grid))
    gvecs_slabs = np.broadcast_to(corner[None, None, :],
                                  (nk, ngkmax, 3)).copy()
    for k in range(nk):
        s = int(slab_offsets[k])
        gvecs_slabs[k] = gvecs_all[s:s + ngkmax]
        vs = int(valid_starts[k])
        ve = int(valid_ends[k])
        # Zero-out any padding rows at the slab ends that lie outside
        # [vs, ve) — those will match the safe corner after the remap.
        if vs > 0:
            gvecs_slabs[k, :vs] = corner
        if ve < ngkmax:
            gvecs_slabs[k, ve:] = corner

    return gvecs_slabs, slab_offsets, valid_starts, valid_ends


def _make_post_read_and_scatter(mesh: Mesh, ngkmax: int, nb: int, nspinor: int,
                                fft_grid: tuple[int, int, int]):
    """Jit a per-rank kernel: f64-real → complex + mask-to-zero + scatter
    into the G-space FFT box shard.  One compile, per-k dispatch.

    Inputs (per rank):
      slab_local: (nb_shard, ns, ngkmax, 2) f64
      gvec_slab : (ngkmax, 3) int32   [replicated]
      valid_start, valid_end: scalar int32 jax arrays

    Output (per rank):
      psi_G_local: (nb_shard, ns, nx, ny, nz) c128
    """
    nx, ny, nz = (int(x) for x in fft_grid)
    p = int(mesh.shape["x"])
    q = int(mesh.shape["y"])
    bands_per_shard = (nb + p * q - 1) // (p * q)

    band_spec = P(("x", "y"), None, None, None)     # slab_local
    rep_spec  = P(None, None)                       # gvec_slab
    sc_spec   = P()                                 # scalars
    out_spec  = P(("x", "y"), None, None, None, None)  # psi_G_local

    def _per_rank(slab_local, gvec_slab, valid_start, valid_end):
        # slab_local is (bands_per_shard, ns, ngkmax, 2)
        idx = jnp.arange(ngkmax, dtype=jnp.int32)
        mask = (idx >= valid_start) & (idx < valid_end)          # (ngkmax,)
        # complex cnk = (re + i*im) * mask
        cnk = (slab_local[..., 0] + 1j * slab_local[..., 1]) * mask
        # cnk: (bands_per_shard, ns, ngkmax) c128

        # gvec remap: rows outside [valid_start, valid_end) go to safe corner.
        corner = jnp.asarray(_fft_safe_corner(fft_grid), dtype=jnp.int32)
        gvec = jnp.where(mask[:, None], gvec_slab, corner[None, :])

        # Scatter into the per-rank FFT box shard.  Band axis is local; FFT
        # axes are replicated within the shard, so no cross-rank collectives.
        psi_G = jnp.zeros((bands_per_shard, nspinor, nx, ny, nz),
                          dtype=jnp.complex128)
        # Scatter: psi[b, s, gx, gy, gz] = cnk[b, s, g].  Advanced
        # indexing on trailing axes with 1-D index arrays.
        psi_G = psi_G.at[:, :, gvec[:, 0], gvec[:, 1], gvec[:, 2]].set(cnk)
        return psi_G

    sm = shard_map(
        _per_rank, mesh=mesh,
        in_specs=(band_spec, rep_spec, sc_spec, sc_spec),
        out_specs=out_spec,
        check_rep=False,
    )
    return jax.jit(sm)


def run_ffi_path(
    wfn_path: str, wfn: WFNReader, band_range: tuple[int, int],
    mesh: Mesh, ngkmax: int,
):
    """(F) path: phdf5 parallel read → static-shape scatter → iFFT.

    Returns
    -------
    psi_r_global: (nk, nb_padded, nspinor, nx, ny, nz) c128, sharded
    stage_times: dict — per-stage wall times (open, read+scatter, fft)
    """
    nk = int(wfn.nkpts)
    nspinor = int(wfn.nspinor)
    fft_grid = tuple(int(x) for x in wfn.fft_grid)
    nx, ny, nz = fft_grid
    b_lo, b_hi = band_range
    nb = b_hi - b_lo

    p = int(mesh.shape["x"])
    q = int(mesh.shape["y"])
    world = p * q
    bands_per_shard = (nb + world - 1) // world
    nb_padded = bands_per_shard * world

    # Must divide evenly across mesh for the band-axis shard_map.
    if nb % world != 0:
        raise RuntimeError(
            f"band count {nb} not divisible by world={world}; pick "
            f"a band_range multiple of world for this test.")

    # Host prep: per-k gvec slabs + offset tables + mask bounds
    # (tiny, <1 MB even for 1k k-points).
    gvecs_slabs_np, slab_offsets_np, vs_np, ve_np = _build_gvecs_slabs(wfn, ngkmax)

    # Replicated per-k gvec stack on device (once): (nk, ngkmax, 3).
    rep_sharding_3d = NamedSharding(mesh, P(None, None, None))
    gvec_stack_dev = jax.device_put(jnp.asarray(gvecs_slabs_np), rep_sharding_3d)
    vs_dev = jax.device_put(jnp.asarray(vs_np))
    ve_dev = jax.device_put(jnp.asarray(ve_np))

    # Jitted post-read-and-scatter kernel (one compile, per-k dispatch).
    post = _make_post_read_and_scatter(mesh, ngkmax, nb_padded, nspinor, fft_grid)

    # iFFT.
    ifftn = make_jittable_local_ifftn_3d(
        mesh,
        P(None, ("x", "y"), None, None, None, None),
        P(None, ("x", "y"), None, None, None, None),
    )

    stage_times = {"open": 0.0, "read_scatter": 0.0, "fft": 0.0}

    # ---- stage: open ----
    def _open():
        return SlabIO(wfn_path, mode="r", mesh=mesh, use_ffi_io=True)
    t_open, io = _time(_open, "wfn_ffi_open", wait_array=False)
    stage_times["open"] = t_open

    try:
        # ---- stage: read + scatter (one compile, cached dispatch per k) ----
        t_rs0 = time.perf_counter()
        _sync("ffi_read_loop_start")

        # Collect per-k G-space FFT boxes; stack into the global
        # (nk, nb_pad, ns, nx, ny, nz) array at the end.  Stack on axis 0
        # is free: the stacked axis is replicated, band sharding on axis
        # 0 of each input becomes axis 1 of the stack.
        psi_G_list = []
        for k in range(nk):
            slab = io.read_slab(
                "wfns/coeffs",
                shape=(nb, nspinor, ngkmax, 2),
                dtype=np.float64,
                offset=(b_lo, 0, int(slab_offsets_np[k]), 0),
                partition_spec=P(("x", "y"), None, None, None),
            )
            psi_G_k = post(slab, gvec_stack_dev[k], vs_dev[k], ve_dev[k])
            psi_G_list.append(psi_G_k)
        psi_G_global = jnp.stack(psi_G_list, axis=0)
        out_sharding = NamedSharding(
            mesh, P(None, ("x", "y"), None, None, None, None))
        psi_G_global = jax.lax.with_sharding_constraint(psi_G_global, out_sharding)
        jax.block_until_ready(psi_G_global)
        _sync("ffi_read_loop_end")
        stage_times["read_scatter"] = time.perf_counter() - t_rs0

        # ---- stage: iFFT to real space ----
        t_fft, psi_r = _time(lambda psi_G: ifftn(psi_G), "ffi_fft", psi_G_global)
        stage_times["fft"] = t_fft
    finally:
        io.close()
    return psi_r, stage_times


# =========================================================================
# (B) baseline path
# =========================================================================
def run_baseline_path(
    wfn_path: str, band_range: tuple[int, int], mesh: Mesh,
):
    """(B) path: WFNReader slurp + read_Gvecs_to_devices + iFFT.

    Returns (psi_r, stage_times) matching run_ffi_path.
    """
    stage_times = {"open": 0.0, "read_scatter": 0.0, "fft": 0.0}

    # stage: open (WFNReader also slurps coeffs here).
    def _open():
        wfn = WFNReader(wfn_path)
        sym = SymMaps(wfn)
        return wfn, sym
    t_open, (wfn, sym) = _time(_open, "base_open", wait_array=False)
    stage_times["open"] = t_open

    nspinor = int(wfn.nspinor)
    fft_grid = tuple(int(x) for x in wfn.fft_grid)

    # stage: read + scatter — the FFT box in G-space.
    from common.meta import Meta
    # Minimal Meta-like struct: only the fields read_Gvecs_to_devices reads.
    # We hand-build one instead of full Meta.from_system to avoid requiring
    # nval/ncond/nband — this test only probes the read pipeline.
    import types
    meta = types.SimpleNamespace(
        fft_grid=fft_grid,
        nspinor=nspinor,
        nspinor_wfnfile=nspinor,
        nk_tot=int(sym.nk_tot),
    )

    def _read():
        psi_G, _ = read_Gvecs_to_devices(
            wfn, sym, band_range, meta, bispinor=False, mesh_xy=mesh)
        return psi_G
    t_rs, psi_G = _time(_read, "base_read")
    stage_times["read_scatter"] = t_rs

    # stage: iFFT to real space.
    ifftn = make_jittable_local_ifftn_3d(
        mesh,
        P(None, ("x", "y"), None, None, None, None),
        P(None, ("x", "y"), None, None, None, None),
    )
    t_fft, psi_r = _time(lambda: ifftn(psi_G), "base_fft")
    stage_times["fft"] = t_fft

    return psi_r, stage_times


# =========================================================================
# comparison
# =========================================================================
def compare_sharded(psi_B: jax.Array, psi_F: jax.Array) -> float:
    """Max |B - F| reduced across ranks.  Both inputs must have the same
    sharding — returned value is global."""
    diff = jnp.abs(psi_B - psi_F).max()
    diff_host = float(jax.device_get(diff))
    # All-reduce the scalar across processes (they all see the same value
    # for a replicated scalar, but be defensive).
    all_diffs = np.asarray(
        multihost_utils.process_allgather(jnp.asarray(diff_host), tiled=False))
    return float(np.max(all_diffs))


# =========================================================================
# main
# =========================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wfn",
                    default="/pscratch/sd/j/jackm/lorrax_sandbox/runs/"
                            "MoS2/02_mos2_3x3_nosym/qe/nscf/WFN.h5",
                    help="WFN.h5 path (nosym file for v1).")
    ap.add_argument("--band-range", type=int, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="Band range [LO, HI).  Default: all bands, trimmed "
                         "down to a multiple of world size.")
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    world = jax.process_count()
    if world == 4:
        p, q = 2, 2
    elif world == 1:
        p, q = 1, 1
    else:
        p, q = world, 1

    _log(f"world={world}, mesh=({p},{q}), wfn={args.wfn}")
    _log(f"rss_pre_any_read = {_rss_gb():.3f} GB")

    devices = np.asarray(jax.devices()).reshape(p, q)
    mesh = Mesh(devices, axis_names=("x", "y"))

    # Peek at the file to get nbands + ntran + ngkmax for band-range defaults.
    # This is a small per-rank read via h5py (just metadata datasets, not
    # coeffs).  We keep it OUT of timed stages.
    import h5py
    with h5py.File(args.wfn, "r") as f:
        nbands = int(f["mf_header/kpoints/mnband"][()])
        ntran = int(f["mf_header/symmetry/ntran"][()])
        ngk = np.asarray(f["mf_header/kpoints/ngk"][:], dtype=np.int64)
        ngkmax = int(ngk.max())
    if ntran != 1:
        _log(f"ERROR: v1 of this test only supports nosym files (ntran=1); "
             f"got ntran={ntran}.  Port symmetry unfolding for v2.")
        return 1

    if args.band_range is None:
        nb_trim = (nbands // world) * world
        band_range = (0, nb_trim)
    else:
        band_range = tuple(args.band_range)
    nb = band_range[1] - band_range[0]
    _log(f"band_range = {band_range}  (nb = {nb})")
    _log(f"ngkmax     = {ngkmax}")

    # Safe-corner sanity check (reads fft_grid, ecutrho, ecutwfc).
    wfn_peek = WFNReader(args.wfn)
    _assert_safe_corner_ok(wfn_peek)
    _log(f"safe corner  = {tuple(int(x) for x in _fft_safe_corner(tuple(int(x) for x in wfn_peek.fft_grid)))}")
    del wfn_peek  # don't keep the slurp around

    # ============================================================
    # Warmup — trigger all compile caches before timed iters so the
    # timings reflect steady-state dispatch, not XLA compile time.
    # ============================================================
    _log("\n--- warmup ---")
    _log(f"rss_pre_baseline_warmup = {_rss_gb():.3f} GB")
    psi_B_warm, tB_warm = run_baseline_path(args.wfn, band_range, mesh)
    _log(f"  baseline warmup: open={tB_warm['open']*1e3:7.1f} ms  "
         f"read+scatter={tB_warm['read_scatter']*1e3:7.1f} ms  "
         f"fft={tB_warm['fft']*1e3:7.1f} ms")
    _log(f"rss_post_baseline_warmup = {_rss_gb():.3f} GB")

    psi_F_warm, tF_warm = run_ffi_path(args.wfn, WFNReader(args.wfn),
                                       band_range, mesh, ngkmax)
    _log(f"  FFI      warmup: open={tF_warm['open']*1e3:7.1f} ms  "
         f"read+scatter={tF_warm['read_scatter']*1e3:7.1f} ms  "
         f"fft={tF_warm['fft']*1e3:7.1f} ms")
    _log(f"rss_post_ffi_warmup = {_rss_gb():.3f} GB")

    # ---- correctness ----
    diff = compare_sharded(psi_B_warm, psi_F_warm)
    _log(f"\nmax |psi_B - psi_F| = {diff:.3e}   "
         f"{'PASS (bit-identical)' if diff == 0.0 else ('PASS (tol)' if diff < 1e-12 else 'FAIL')}")
    del psi_B_warm, psi_F_warm

    # ============================================================
    # Timed iters.
    # ============================================================
    _log(f"\n--- timed iters ({args.iters}) ---")
    B_times = {"open": [], "read_scatter": [], "fft": []}
    F_times = {"open": [], "read_scatter": [], "fft": []}

    for it in range(args.iters):
        _log(f"\n[iter {it}]")
        _, tB = run_baseline_path(args.wfn, band_range, mesh)
        for k in B_times:
            B_times[k].append(tB[k])
        _log(f"  (B) open={tB['open']*1e3:7.1f}  "
             f"r+s={tB['read_scatter']*1e3:7.1f}  "
             f"fft={tB['fft']*1e3:7.1f}   (ms)")

        _, tF = run_ffi_path(args.wfn, WFNReader(args.wfn), band_range, mesh, ngkmax)
        for k in F_times:
            F_times[k].append(tF[k])
        _log(f"  (F) open={tF['open']*1e3:7.1f}  "
             f"r+s={tF['read_scatter']*1e3:7.1f}  "
             f"fft={tF['fft']*1e3:7.1f}   (ms)")

    def _mean_ms(ts): return 1e3 * float(np.mean(ts))

    _log("\n=== summary (mean over iters) ===")
    _log(f"{'stage':>14} {'(B) base':>12} {'(F) ffi':>12} {'speedup':>10}")
    total_B = total_F = 0.0
    for k in ("open", "read_scatter", "fft"):
        mB = _mean_ms(B_times[k])
        mF = _mean_ms(F_times[k])
        total_B += mB
        total_F += mF
        sp = f"{mB/mF:6.2f}x" if mF > 0 else "  inf "
        _log(f"{k:>14} {mB:>9.1f} ms {mF:>9.1f} ms {sp:>10}")
    _log(f"{'TOTAL':>14} {total_B:>9.1f} ms {total_F:>9.1f} ms "
         f"{total_B/total_F:>6.2f}x")

    _log(f"\nrss_final = {_rss_gb():.3f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

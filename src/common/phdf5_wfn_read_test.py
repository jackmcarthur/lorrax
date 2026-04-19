"""End-to-end test for :class:`PhdfWfnReader` on a nosym WFN file.

Compares the production phdf5 reader against the legacy baseline
(``WFNReader`` + ``read_Gvecs_to_devices``) at each stage of the
"open → read coefficients → scatter into G-space FFT box → iFFT to
real space" pipeline:

* **Correctness**: the two real-space outputs must be bitwise-equal
  (identity symmetry means no unfolding ambiguity).
* **Timing**: per-stage wall time, averaged over a few iterations
  after warmup.
* **Memory**: peak RSS before and after each read, to make visible
  the host-RAM cost of the baseline's full-coeffs slurp.

Test file defaults to the MoS2 3×3 nosym WFN.h5 in the sandbox;
override with ``--wfn <path>``.

Usage (4-GPU on Perlmutter)::

    lxalloc
    export SLURM_JOBID=<from lxalloc>
    LORRAX_NGPU=4 LORRAX_MPI_TYPE=pmix \
        lxrun python3 -u -m common.phdf5_wfn_read_test

Add ``LORRAX_PHDF5_TIME=1`` to see per-phase timing from the C++
handler printed by rank 0.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

import argparse
import resource
import sys
import time
import types

import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)

_DIST_SENTINEL = "_LORRAX_JAX_DISTRIBUTED_DONE"
def _maybe_init_jax_distributed():
    if os.environ.get(_DIST_SENTINEL):
        return
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        try:
            jax.distributed.initialize()
        except Exception:
            pass
    os.environ[_DIST_SENTINEL] = "1"
_maybe_init_jax_distributed()

from jax.experimental import multihost_utils
from jax.sharding import Mesh, PartitionSpec as P

from common.wfnreader import WFNReader
from common.symmetry_maps import SymMaps
from common.load_wfns import read_Gvecs_to_devices
from common.fft_helpers import make_jittable_local_ifftn_3d
from common.phdf5_wfn_reader import PhdfWfnReader


# =============================================================================
#  Reporting / timing helpers
# =============================================================================
def _log(msg: str) -> None:
    if jax.process_index() == 0:
        print(msg, flush=True)


def _peak_rss_gb() -> float:
    # ru_maxrss is KiB on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def _sync_global(tag: str) -> None:
    try:
        multihost_utils.sync_global_devices(tag)
    except Exception:
        pass


def _time_stage(fn, tag: str, *args, wait_array: bool = True, **kw):
    _sync_global(f"{tag}_start")
    t0 = time.perf_counter()
    out = fn(*args, **kw)
    if wait_array and isinstance(out, jax.Array):
        jax.block_until_ready(out)
    _sync_global(f"{tag}_end")
    return time.perf_counter() - t0, out


def _build_ifftn(mesh: Mesh):
    """3-D inverse FFT over the three spatial axes, identical layout
    in and out (so sharding is preserved)."""
    spec = P(None, ("x", "y"), None, None, None, None)
    return make_jittable_local_ifftn_3d(mesh, spec, spec)


# =============================================================================
#  Baseline path: legacy WFNReader + read_Gvecs_to_devices + iFFT
# =============================================================================
def run_baseline_path(
    wfn_path: str, band_range: tuple[int, int], mesh: Mesh,
) -> tuple[jax.Array, dict]:
    times = {"open": 0.0, "read_scatter": 0.0, "fft": 0.0}

    def _open():
        wfn = WFNReader(wfn_path)
        sym = SymMaps(wfn)
        return wfn, sym
    t_open, (wfn, sym) = _time_stage(_open, "base_open", wait_array=False)
    times["open"] = t_open

    nspinor = int(wfn.nspinor)
    fft_grid = tuple(int(x) for x in wfn.fft_grid)
    meta_stub = types.SimpleNamespace(
        fft_grid=fft_grid, nspinor=nspinor,
        nspinor_wfnfile=nspinor, nk_tot=int(sym.nk_tot))

    def _read():
        psi_G, _ = read_Gvecs_to_devices(
            wfn, sym, band_range, meta_stub, bispinor=False, mesh_xy=mesh)
        return psi_G
    t_rs, psi_G = _time_stage(_read, "base_read")
    times["read_scatter"] = t_rs

    ifftn = _build_ifftn(mesh)
    t_fft, psi_r = _time_stage(lambda: ifftn(psi_G), "base_fft")
    times["fft"] = t_fft
    return psi_r, times


# =============================================================================
#  Reader path: PhdfWfnReader + iFFT
# =============================================================================
def run_reader_path(
    wfn_path: str, band_range: tuple[int, int], mesh: Mesh,
) -> tuple[jax.Array, dict]:
    times = {"open": 0.0, "read_scatter": 0.0, "fft": 0.0}

    t_open, reader = _time_stage(
        lambda: PhdfWfnReader(wfn_path, mesh=mesh),
        "reader_open", wait_array=False)
    times["open"] = t_open

    try:
        t_rs, psi_G = _time_stage(
            lambda: reader.load_band_chunk_gspace(band_range),
            "reader_read")
        times["read_scatter"] = t_rs

        ifftn = _build_ifftn(mesh)
        t_fft, psi_r = _time_stage(lambda: ifftn(psi_G), "reader_fft")
        times["fft"] = t_fft
    finally:
        reader.close()
    return psi_r, times


# =============================================================================
#  Correctness comparison — one scalar max-abs diff reduced across ranks
# =============================================================================
def compare_sharded(psi_B: jax.Array, psi_F: jax.Array) -> float:
    local_max = float(jax.device_get(jnp.abs(psi_B - psi_F).max()))
    gathered = np.asarray(multihost_utils.process_allgather(
        jnp.asarray(local_max), tiled=False))
    return float(np.max(gathered))


# =============================================================================
#  Main
# =============================================================================
def _build_mesh(world_size: int) -> tuple[Mesh, int, int]:
    if world_size == 4:
        p, q = 2, 2
    elif world_size == 1:
        p, q = 1, 1
    else:
        p, q = world_size, 1
    devices = np.asarray(jax.devices()).reshape(p, q)
    return Mesh(devices, axis_names=("x", "y")), p, q


def _peek_band_defaults(wfn_path: str, world_size: int) -> tuple[int, int, int]:
    """Quick h5py peek for nbands + ntran + ngkmax.  Used only to pick
    a default band range divisible by world; kept out of the timed
    stages."""
    import h5py
    with h5py.File(wfn_path, "r") as f:
        nbands = int(f["mf_header/kpoints/mnband"][()])
        ntran = int(f["mf_header/symmetry/ntran"][()])
        ngkmax = int(np.asarray(f["mf_header/kpoints/ngk"][:]).max())
    nb_trim = (nbands // world_size) * world_size
    return nb_trim, ntran, ngkmax


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--wfn",
        default=("/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/"
                 "02_mos2_3x3_nosym/qe/nscf/WFN.h5"))
    ap.add_argument("--band-range", type=int, nargs=2, default=None,
                    metavar=("LO", "HI"))
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    world_size = jax.process_count()
    mesh, p, q = _build_mesh(world_size)

    nb_default, ntran, ngkmax = _peek_band_defaults(args.wfn, world_size)
    if ntran != 1:
        _log(f"ERROR: v1 test supports nosym files only; got ntran={ntran}")
        return 1
    band_range = tuple(args.band_range) if args.band_range else (0, nb_default)
    nb = band_range[1] - band_range[0]

    _log(f"world={world_size}, mesh=({p},{q})  wfn={args.wfn}")
    _log(f"band_range={band_range}  (nb={nb})  ngkmax={ngkmax}")
    _log(f"rss_pre_reads = {_peak_rss_gb():.3f} GB")

    # -------- warmup (trigger jit compiles, warm MPI-IO / page cache) --------
    _log("\n--- warmup ---")
    psi_B_warm, tB = run_baseline_path(args.wfn, band_range, mesh)
    _log(f"  baseline : open={tB['open']*1e3:7.1f}  "
         f"r+s={tB['read_scatter']*1e3:7.1f}  fft={tB['fft']*1e3:7.1f}  (ms)")
    _log(f"rss_post_baseline_warmup = {_peak_rss_gb():.3f} GB")

    psi_F_warm, tF = run_reader_path(args.wfn, band_range, mesh)
    _log(f"  phdf5    : open={tF['open']*1e3:7.1f}  "
         f"r+s={tF['read_scatter']*1e3:7.1f}  fft={tF['fft']*1e3:7.1f}  (ms)")
    _log(f"rss_post_phdf5_warmup    = {_peak_rss_gb():.3f} GB")

    max_diff = compare_sharded(psi_B_warm, psi_F_warm)
    verdict = (
        "PASS (bit-identical)" if max_diff == 0.0
        else ("PASS (tol)" if max_diff < 1e-12 else "FAIL"))
    _log(f"\nmax |psi_B - psi_F| = {max_diff:.3e}   {verdict}")
    del psi_B_warm, psi_F_warm
    if max_diff > 1e-12:
        return 1

    # -------- timed iters --------
    _log(f"\n--- timed iters ({args.iters}) ---")
    baseline_times = {k: [] for k in ("open", "read_scatter", "fft")}
    reader_times = {k: [] for k in ("open", "read_scatter", "fft")}
    for it in range(args.iters):
        _, tB = run_baseline_path(args.wfn, band_range, mesh)
        _, tF = run_reader_path(args.wfn, band_range, mesh)
        for k in baseline_times:
            baseline_times[k].append(tB[k])
            reader_times[k].append(tF[k])
        _log(f"  [iter {it}]  "
             f"(B) open={tB['open']*1e3:7.1f}  r+s={tB['read_scatter']*1e3:7.1f}  "
             f"fft={tB['fft']*1e3:7.1f}   "
             f"(F) open={tF['open']*1e3:7.1f}  r+s={tF['read_scatter']*1e3:7.1f}  "
             f"fft={tF['fft']*1e3:7.1f}")

    def _mean_ms(seq): return 1e3 * float(np.mean(seq))

    _log("\n=== summary (mean ms, 3 iters) ===")
    _log(f"{'stage':>14} {'(B) base':>12} {'(F) phdf5':>12} {'ratio':>10}")
    total_B = total_F = 0.0
    for stage in ("open", "read_scatter", "fft"):
        mB = _mean_ms(baseline_times[stage])
        mF = _mean_ms(reader_times[stage])
        total_B += mB
        total_F += mF
        _log(f"{stage:>14} {mB:>9.1f} ms {mF:>9.1f} ms {mB/mF:>8.2f}x")
    _log(f"{'TOTAL':>14} {total_B:>9.1f} ms {total_F:>9.1f} ms "
         f"{total_B/total_F:>8.2f}x")
    _log(f"\nrss_final = {_peak_rss_gb():.3f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

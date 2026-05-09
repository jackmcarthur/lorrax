"""Integration test for the ``use_phdf5`` switch on the two LORRAX
consumers that drive band-chunked wavefunction loads:

  * :func:`common.load_wfns.load_centroids_band_chunked`
  * :func:`common.load_wfns.get_psi_rchunk`

For a small centroid set / r-chunk, run each function twice — once
via the legacy path (``use_phdf5=False``, default) and once via the
phdf5 path (``use_phdf5=True``) — and assert bit-identical output.

Usage (4-GPU)::

    lxalloc
    export SLURM_JOBID=<from lxalloc>
    LORRAX_NGPU=4 LORRAX_MPI_TYPE=pmix \
        lxrun python3 -u -m common.phdf5_plumbing_test
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

import argparse
import sys
import time
import types

import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)

_DIST = "_LORRAX_JAX_DISTRIBUTED_DONE"
def _init_once():
    if os.environ.get(_DIST):
        return
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        try:
            jax.distributed.initialize()
        except Exception:
            pass
    os.environ[_DIST] = "1"
_init_once()

from jax.experimental import multihost_utils
from jax.sharding import Mesh

from file_io import WFNReader
from common.symmetry_maps import SymMaps
from common.load_wfns import (
    load_centroids_band_chunked, get_psi_rchunk, _close_phdf5_readers)
from common import jax_profile


# =============================================================================
#  Helpers
# =============================================================================
def log(msg: str) -> None:
    if jax.process_index() == 0:
        print(msg, flush=True)


def time_call(fn, *args, **kwargs) -> tuple[float, object]:
    try:
        multihost_utils.sync_global_devices("time_call_start")
    except Exception:
        pass
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    jax.block_until_ready(out)
    try:
        multihost_utils.sync_global_devices("time_call_end")
    except Exception:
        pass
    return time.perf_counter() - t0, out


def global_max_diff(a: jax.Array, b: jax.Array) -> float:
    local = float(jax.device_get(jnp.abs(a - b).max()))
    all_ = np.asarray(
        multihost_utils.process_allgather(jnp.asarray(local), tiled=False))
    return float(np.max(all_))


def build_mesh(world: int) -> Mesh:
    if world == 4:
        p, q = 2, 2
    elif world == 1:
        p, q = 1, 1
    else:
        p, q = world, 1
    return Mesh(np.asarray(jax.devices()).reshape(p, q), axis_names=("x", "y"))


def meta_stub(wfn, sym, *, bispinor: bool = False) -> types.SimpleNamespace:
    """Minimal Meta-like object with the fields both consumers read."""
    fft_grid = tuple(int(x) for x in wfn.fft_grid)
    nspinor_wfnfile = int(wfn.nspinor)
    return types.SimpleNamespace(
        fft_grid=fft_grid,
        nspinor=4 if bispinor else nspinor_wfnfile,
        nspinor_wfnfile=nspinor_wfnfile,
        nk_tot=int(sym.nk_tot),
        kgrid=tuple(int(x) for x in wfn.kgrid),
        b_id_4_user=int(wfn.nbands),
    )


def pick_centroids(wfn, n: int, seed: int = 42) -> np.ndarray:
    """Pick n distinct (ix, iy, iz) cells from the FFT box."""
    fft_grid = np.asarray(wfn.fft_grid, dtype=np.int32)
    rng = np.random.default_rng(seed)
    n_rtot = int(np.prod(fft_grid))
    flat = rng.choice(n_rtot, size=n, replace=False)
    nx, ny, nz = fft_grid
    iz = flat % nz
    iy = (flat // nz) % ny
    ix = flat // (ny * nz)
    return np.stack([ix, iy, iz], axis=1).astype(np.int32)


# =============================================================================
#  Comparisons
# =============================================================================
def run_centroids(wfn_path: str, mesh: Mesh, n_centroids: int, band_range,
                  k_chunk_size: int | None = None, *, bispinor: bool = False):
    wfn = WFNReader(wfn_path)
    sym = SymMaps(wfn)
    meta = meta_stub(wfn, sym, bispinor=bispinor)
    centroids = jnp.asarray(pick_centroids(wfn, n_centroids))

    args = dict(
        wfn=wfn, sym=sym, meta=meta,
        centroid_indices=centroids,
        bispinor=bispinor, mesh_xy=mesh,
        band_range=band_range,
        band_chunk_size=band_range[1] - band_range[0],
        k_chunk_size=k_chunk_size,
    )

    with jax_profile.trace_section("centroids_legacy"):
        t_legacy, (rmu_B, rmuT_B) = time_call(
            load_centroids_band_chunked, **args)
    with jax_profile.trace_section("centroids_phdf5"):
        t_phdf5, (rmu_F, rmuT_F) = time_call(
            load_centroids_band_chunked, **args, use_phdf5=True)

    diff_rmu = global_max_diff(rmu_B, rmu_F)
    diff_rmuT = global_max_diff(rmuT_B, rmuT_F)
    tag = f"kchunk={k_chunk_size}" if k_chunk_size else "all-k"
    log(f"centroids [{tag}]  psi_rmu  max|B-F| = {diff_rmu:.3e}   "
        f"psi_rmuT max|B-F| = {diff_rmuT:.3e}")
    log(f"           legacy {t_legacy*1e3:7.1f} ms    "
        f"phdf5 {t_phdf5*1e3:7.1f} ms   "
        f"ratio {t_legacy/t_phdf5:.2f}x")
    return max(diff_rmu, diff_rmuT)


def run_rchunk(wfn_path: str, mesh: Mesh, r_range, band_range,
               k_chunk_size: int = 0, *, bispinor: bool = False):
    wfn = WFNReader(wfn_path)
    sym = SymMaps(wfn)
    meta = meta_stub(wfn, sym, bispinor=bispinor)

    args = dict(
        wfn=wfn, sym=sym, meta=meta, mesh_xy=mesh,
        band_range=band_range,
        r_start=r_range[0], r_end=r_range[1],
        bispinor=bispinor,
        band_chunk_size=band_range[1] - band_range[0],
        k_chunk_size=k_chunk_size,
    )

    t_legacy, psi_B = time_call(get_psi_rchunk, **args)
    t_phdf5, psi_F = time_call(get_psi_rchunk, **args, use_phdf5=True)

    diff = global_max_diff(psi_B, psi_F)
    tag = f"kchunk={k_chunk_size}" if k_chunk_size else "all-k"
    log(f"rchunk    [{tag}]  max|B-F| = {diff:.3e}")
    log(f"           legacy {t_legacy*1e3:7.1f} ms    "
        f"phdf5 {t_phdf5*1e3:7.1f} ms   "
        f"ratio {t_legacy/t_phdf5:.2f}x")
    return diff


# =============================================================================
#  Main
# =============================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--wfn",
        default=("/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/"
                 "02_mos2_3x3_nosym/qe/nscf/WFN.h5"))
    ap.add_argument("--n-centroids", type=int, default=16)
    ap.add_argument("--n-rcells", type=int, default=128,
                    help="length of r-chunk (contiguous flattened-xyz range).")
    ap.add_argument("--skip-kchunk", action="store_true",
                    help="skip the k-chunked-read variant of the test.")
    ap.add_argument("--bispinor", action="store_true",
                    help="compare the 4-component small-spinor expansion.")
    args = ap.parse_args()

    world = jax.process_count()
    mesh = build_mesh(world)

    # Pick a small band range divisible by world.
    import h5py
    with h5py.File(args.wfn, "r") as f:
        nbands = int(f["mf_header/kpoints/mnband"][()])
    nb = (nbands // world) * world
    band_range = (0, min(nb, 80 if world > 1 else nb))

    wfn_probe = WFNReader(args.wfn)
    sym_probe = SymMaps(wfn_probe)
    nk_tot = int(sym_probe.nk_tot)
    # Force k-chunking: half the unfolded k-set, rounded up so a small nk
    # still produces at least two chunks.
    k_chunk_size = max(1, (nk_tot + 1) // 2)

    log(f"world={world}  wfn={args.wfn}  band_range={band_range}  "
        f"nk_tot={nk_tot}  k_chunk_size={k_chunk_size}  "
        f"bispinor={args.bispinor}")

    tol = 1e-12

    log("\n--- all-k read ---")
    d_centroids = run_centroids(
        args.wfn, mesh, args.n_centroids, band_range,
        bispinor=args.bispinor)
    d_rchunk = run_rchunk(
        args.wfn, mesh, (0, args.n_rcells), band_range,
        bispinor=args.bispinor)
    worst = max(d_centroids, d_rchunk)

    if not args.skip_kchunk and nk_tot > 1:
        log("\n--- k-chunked read ---")
        d_centroids_k = run_centroids(
            args.wfn, mesh, args.n_centroids, band_range,
            k_chunk_size=k_chunk_size, bispinor=args.bispinor)
        d_rchunk_k = run_rchunk(
            args.wfn, mesh, (0, args.n_rcells), band_range,
            k_chunk_size=k_chunk_size, bispinor=args.bispinor)
        worst = max(worst, d_centroids_k, d_rchunk_k)

    log("\nPASS" if worst <= tol else "\nFAIL")
    return 0 if worst <= tol else 1


if __name__ == "__main__":
    sys.exit(main())

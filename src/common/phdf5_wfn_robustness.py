"""Robustness driver for the PHDF5 WFN reader's k-union path.

The normal PHDF5 plumbing tests exercise all-k and contiguous k-chunk
loads.  This driver targets the more awkward edge cases in
``PhdfWfnReader.coeffs_gspace``:

* arbitrary full-BZ k order,
* duplicated k IDs,
* symmetry-related full-BZ k IDs that map to one IBZ file slab,
* band chunks that cross the physical ``mnband`` and require zero pad.

It compares the PHDF5 output against the canonical ``WFNReader`` +
``read_Gvecs_to_devices`` path, loading one requested k at a time for the
reference so non-contiguous and duplicated k-order is represented exactly.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import types

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import Mesh

jax.config.update("jax_enable_x64", True)

_DIST = "_LORRAX_JAX_DISTRIBUTED_DONE"


def _init_distributed_once() -> None:
    if os.environ.get(_DIST):
        return
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        try:
            jax.distributed.initialize()
        except Exception:
            pass
    os.environ[_DIST] = "1"


_init_distributed_once()

from common.load_wfns import read_Gvecs_to_devices
from common.phdf5_wfn_reader import PhdfWfnReader
from common.symmetry_maps import SymMaps
from file_io import WFNReader


def _log(msg: str) -> None:
    if jax.process_index() == 0:
        print(msg, flush=True)


def _sync(tag: str) -> None:
    try:
        multihost_utils.sync_global_devices(tag)
    except Exception:
        pass


def _time_call(tag: str, fn):
    _sync(f"{tag}_start")
    t0 = time.perf_counter()
    out = fn()
    jax.block_until_ready(out)
    _sync(f"{tag}_end")
    return time.perf_counter() - t0, out


def _build_mesh(world: int) -> Mesh:
    if world == 4:
        p, q = 2, 2
    elif world == 1:
        p, q = 1, 1
    else:
        p, q = world, 1
    return Mesh(np.asarray(jax.devices()).reshape(p, q), axis_names=("x", "y"))


def _meta_stub(wfn, sym, nk_tot: int, *, bispinor: bool = False) -> types.SimpleNamespace:
    nspinor_wfnfile = int(wfn.nspinor)
    return types.SimpleNamespace(
        fft_grid=tuple(int(x) for x in wfn.fft_grid),
        nspinor=4 if bispinor else nspinor_wfnfile,
        nspinor_wfnfile=nspinor_wfnfile,
        nk_tot=int(nk_tot),
        kgrid=tuple(int(x) for x in wfn.kgrid),
    )


def _global_max_diff(a: jax.Array, b: jax.Array) -> float:
    local = float(jax.device_get(jnp.abs(a - b).max()))
    gathered = np.asarray(
        multihost_utils.process_allgather(jnp.asarray(local), tiled=False))
    return float(np.max(gathered))


def _legacy_for_k_ids(
    wfn,
    sym,
    band_range: tuple[int, int],
    mesh: Mesh,
    k_ids: np.ndarray,
    *,
    bispinor: bool,
) -> jax.Array:
    """Reference path preserving arbitrary requested k order."""
    pieces = []
    for k in k_ids:
        meta = _meta_stub(wfn, sym, nk_tot=1, bispinor=bispinor)
        psi_k, _ = read_Gvecs_to_devices(
            wfn, sym, band_range, meta, bispinor=bispinor, mesh_xy=mesh,
            k_range=(int(k), int(k) + 1))
        pieces.append(psi_k)
    return jnp.concatenate(pieces, axis=0) if len(pieces) > 1 else pieces[0]


def _case_table(sym: SymMaps, nk_tot: int) -> list[tuple[str, np.ndarray]]:
    all_k = np.arange(nk_tot, dtype=np.int32)
    cases: list[tuple[str, np.ndarray]] = [
        ("all_k", all_k),
        ("reverse_all_k", all_k[::-1]),
    ]

    discontig = np.array(
        [nk_tot - 1, 0, max(0, nk_tot // 2), 1 % nk_tot, nk_tot - 1],
        dtype=np.int32)
    cases.append(("discontiguous_with_duplicate", discontig))

    groups: dict[int, list[int]] = {}
    for k, ibz in enumerate(np.asarray(sym.irk_to_k_map, dtype=np.int32)):
        groups.setdefault(int(ibz), []).append(k)
    related = next((ks for ks in groups.values() if len(ks) >= 2), None)
    if related is not None:
        arr = np.asarray([related[-1], related[0], related[-1]], dtype=np.int32)
        cases.append(("sym_related_same_file_slab", arr))
    return cases


def _band_ranges(nbands: int, world: int, requested) -> list[tuple[str, tuple[int, int]]]:
    if requested is not None:
        lo, hi = (int(requested[0]), int(requested[1]))
        return [("requested", (lo, hi))]

    head_hi = min(max(world, 2 * world), (nbands // world) * world)
    ranges = [("head", (0, head_hi))]

    if nbands >= 4 * world:
        mid_lo = world
        ranges.append(("mid", (mid_lo, mid_lo + 2 * world)))

    # Cross the file's physical nbands and verify that pad-past-file
    # remains zero while preserving requested k order.
    tail_lo = max(0, nbands - world)
    ranges.append(("pad_past_file", (tail_lo, tail_lo + 2 * world)))
    return ranges


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wfn", required=True)
    ap.add_argument("--band-range", type=int, nargs=2, default=None,
                    metavar=("LO", "HI"))
    ap.add_argument("--bispinor", action="store_true",
                    help="compare the 4-component small-spinor expansion.")
    ap.add_argument("--tol", type=float, default=1e-12)
    args = ap.parse_args()

    world = jax.process_count()
    mesh = _build_mesh(world)
    wfn = WFNReader(args.wfn)
    sym = SymMaps(wfn)
    reader = PhdfWfnReader(args.wfn, mesh=mesh)

    try:
        _log(f"world={world}  wfn={args.wfn}")
        _log(f"nbands={wfn.nbands}  nk_full={sym.nk_tot}  "
             f"ntran={wfn.ntran}  bispinor={args.bispinor}")
        worst = 0.0
        failures = 0
        for br_name, band_range in _band_ranges(
            int(wfn.nbands), world, args.band_range):
            nb = band_range[1] - band_range[0]
            if nb <= 0 or nb % world:
                _log(f"SKIP band_range[{br_name}]={band_range}: "
                     f"nb={nb} not positive/divisible by world={world}")
                continue
            _log(f"\n--- band_range[{br_name}]={band_range} ---")
            for case_name, k_ids in _case_table(sym, int(sym.nk_tot)):
                t_ref, ref = _time_call(
                    f"legacy_{br_name}_{case_name}",
                    lambda br=band_range, ks=k_ids: _legacy_for_k_ids(
                        wfn, sym, br, mesh, ks, bispinor=args.bispinor))
                t_phdf5, got = _time_call(
                    f"phdf5_{br_name}_{case_name}",
                    lambda br=band_range, ks=k_ids: reader.coeffs_gspace(
                        br, k_ids=ks, bispinor=args.bispinor))
                diff = _global_max_diff(ref, got)
                worst = max(worst, diff)
                status = "PASS" if diff <= args.tol else "FAIL"
                if status == "FAIL":
                    failures += 1
                _log(
                    f"{status:4s} {case_name:<32s} nk={len(k_ids):2d}  "
                    f"max|ref-phdf5|={diff:.3e}  "
                    f"legacy={t_ref*1e3:7.1f} ms  "
                    f"phdf5={t_phdf5*1e3:7.1f} ms")
                del ref, got
        _log(f"\nworst max|ref-phdf5| = {worst:.3e}")
        _log("PASS" if failures == 0 else "FAIL")
        return 0 if failures == 0 else 1
    finally:
        reader.close()


if __name__ == "__main__":
    sys.exit(main())

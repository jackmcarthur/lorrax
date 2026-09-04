"""Real 4-rank CUDA gate: the face carrier built from a real deck, restart
write + read, and per-rank psi bytes ~2*S/P by buffer inspection.

Guide: reports/gwjax_low_mem_bands_audit_2026-08-22/report.md, census rows
1/10/11 and the VERIFY list ("a real 4-rank CUDA gate that builds the face
carrier from the k6_c50 deck, writes + reads restart, and proves per-rank
psi bytes ~2S/P via buffer inspection").

This calls the PRODUCTION ``gw.gw_init.prepare_isdf_and_wavefunctions`` —
the same function a real ``low_mem_bands = true`` deck runs — rather than
reimplementing its bootstrap, so what is certified is the code path an
operator actually gets, not a paraphrase of it.  It stops there
deliberately: everything past the ISDF/wavefunction stage (chi0/W/Sigma)
is OUT OF SCOPE for this carrier task and is refused by name by the
``.xn()``/``.xr()``/``.yr()``/``.yn()`` accessors on an unported consumer,
which is a different (and already emulated-mesh-gated) claim.

Checks, strongest first:

  1. THE BUNDLE IS FACE-LAYOUT.  ``prepare_isdf_and_wavefunctions`` under
     ``low_mem_bands = true`` returns ``wf_bundle.layout == "face"`` with
     psi_nmu/psi_mun populated and all four legacy fields None.
  2. PER-RANK PSI BYTES ~ 2*S/P.  Every addressable shard of psi_nmu and
     of psi_mun is ``S/P`` bytes (``S = 16*nk*nspinor*nb*nmu``); summed,
     one rank's total ψ residency is ``2*S/P`` -- the report's headline
     claim, measured off the ACTUAL jax.Array buffers this run produced,
     not a formula.
  3. RESTART ROUND-TRIP, BIT-IDENTICAL.  Reading ``tensors_filename`` back
     with ``load_restart_state_from_h5(..., low_mem_bands=True)`` recovers
     psi_nmu/psi_mun equal to the in-memory bundle on EVERY addressable
     shard -- two direct SlabIO hyperslab reads (report's "request both
     face specs" branch), no y-only full-band replica staged in between.
     (The ABSENCE of a reshard on this path is a source-level fact, not a
     runtime one: ``load_restart_state_from_h5``'s ``low_mem_bands=True``
     branch returns the two SlabIO reads directly with no
     ``with_sharding_constraint``/transpose in between -- contrast the
     legacy branch's single documented y->x reshard.)

Run:
    lx run -G 4 -n 4 env PYTHONPATH=... python3 \
        tests/multi_device/face_carrier_restart_gate.py \
        /path/to/attempt/dir/cohsex.in
"""
import os
import sys
import time

from runtime import initialize_communicator_stack, finalize_process

RUNTIME = initialize_communicator_stack()

import jax                                                     # noqa: E402
import numpy as np                                             # noqa: E402

import common.timing as timing                                 # noqa: E402
from common import Meta                                        # noqa: E402
from common.collectives import barrier                         # noqa: E402
from file_io import load_centroids                              # noqa: E402
from wfn_loader import WfnLoader                                # noqa: E402
import symmetry_maps                                            # noqa: E402
from gw.gw_config import LorraxConfig                            # noqa: E402
from gw.gw_init import prepare_isdf_and_wavefunctions            # noqa: E402
from gw.wavefunction_bundle import BandSlices                    # noqa: E402
from file_io import load_restart_state_from_h5                   # noqa: E402

_C128 = 16


def _addressable_bytes(arr):
    return sum(int(np.asarray(s.data).nbytes) for s in arr.addressable_shards)


def _shard_map(arr):
    return {str(sh.index): np.asarray(sh.data) for sh in arr.addressable_shards}


def _worst_shard_diff(a, b):
    sa, sb = _shard_map(a), _shard_map(b)
    if set(sa) != set(sb):
        return float("inf")
    worst = 0.0
    for k in sa:
        worst = max(worst, float(np.abs(sa[k] - sb[k]).max()))
    return worst


def main():
    rank, world = jax.process_index(), jax.process_count()
    p0 = print if rank == 0 else (lambda *a, **k: None)

    if len(sys.argv) != 2:
        p0("usage: face_carrier_restart_gate.py /path/to/cohsex.in")
        return 2
    input_file = sys.argv[1]
    input_dir = os.path.dirname(os.path.abspath(input_file))

    def print0(*a, **k):
        if rank == 0:
            k.setdefault("flush", True)
            print(*a, **k)

    timing.reset()
    mesh_xy = RUNTIME.mesh
    px, py = (int(s) for s in mesh_xy.devices.shape)
    p0(f"[fcr] world={world} mesh=({px},{py}) input={input_file}")

    cfg = LorraxConfig.from_input_file(input_file, print_fn=print0)
    if not cfg.memory.low_mem_bands:
        p0("[fcr] FAIL: deck did not set low_mem_bands = true")
        return 1

    wfn = WfnLoader(cfg.paths.wfn_file, mesh=mesh_xy)
    sym = symmetry_maps.SymMaps(wfn)
    _, centroid_indices, n_rmu = load_centroids(
        cfg.paths.centroids_file, wfn.fft_grid)
    tmp_dir = os.path.join(input_dir, "tmp")
    if rank == 0:
        os.makedirs(tmp_dir, exist_ok=True)
    barrier("fcr_tmp_dir")
    tensors_filename = os.path.join(tmp_dir, f"isdf_tensors_{n_rmu}.h5")

    meta = Meta.from_system(wfn, sym, cfg.nval, cfg.ncond, cfg.nband,
                            n_rmu, cfg.bispinor,
                            nband_chi=cfg.bands.chi,
                            nband_sigma=cfg.bands.sigma)
    meta.rank = rank
    meta.n_proc = world
    meta.sys_dim = cfg.sys_dim
    meta.bispinor = cfg.bispinor
    band_slices = BandSlices.from_band_edges(
        *meta.band_edges, b4_chi=meta.b_id_4_chi, b4_sigma=meta.b_id_4_sigma)

    p0(f"[fcr] nk={meta.nk_tot} ns={meta.nspinor} nb_full={band_slices.nb_full} "
       f"n_rmu={n_rmu} n_rmu_padded={getattr(meta, 'n_rmu_padded', n_rmu)}")

    t0 = time.time()
    isdf = prepare_isdf_and_wavefunctions(
        cfg=cfg, wfn=wfn, sym=sym, meta=meta,
        centroid_indices=centroid_indices, band_slices=band_slices,
        mesh_xy=mesh_xy, tmp_dir=tmp_dir, tensors_filename=tensors_filename,
        print0=print0,
    )
    p0(f"[fcr] prepare_isdf_and_wavefunctions: {time.time() - t0:.1f} s")
    wfns = isdf.wf_bundle

    # ---- 1. face layout ---------------------------------------------------
    ok1 = (wfns.layout == "face" and wfns.psi_nmu is not None
           and wfns.psi_mun is not None and wfns.psi_xn is None
           and wfns.psi_xr is None and wfns.psi_yr is None
           and wfns.psi_yn is None)
    p0(f"[fcr] 1. bundle layout={wfns.layout!r}  {'PASS' if ok1 else 'FAIL'}")

    # ---- 2. per-rank psi bytes ~ 2*S/P ------------------------------------
    nk = int(meta.nk_tot)
    ns = int(meta.nspinor)
    nmu_pad = int(getattr(meta, "n_rmu_padded", None) or n_rmu)
    nb_full = int(band_slices.nb_full)
    s_bytes = _C128 * nk * ns * nmu_pad * nb_full
    p_total = px * py
    want_per_shard = s_bytes / p_total

    nmu_local = _addressable_bytes(wfns.psi_nmu)
    mun_local = _addressable_bytes(wfns.psi_mun)
    got_total = nmu_local + mun_local
    want_total = 2 * want_per_shard
    rel = abs(got_total - want_total) / max(want_total, 1.0)
    ok2 = rel < 0.05
    p0(f"[fcr] 2. per-rank psi bytes: nmu={nmu_local/1e6:.3f} MB "
       f"mun={mun_local/1e6:.3f} MB  total={got_total/1e6:.3f} MB  "
       f"want~2S/P={want_total/1e6:.3f} MB  rel_err={rel:.4f}  "
       f"{'PASS' if ok2 else 'FAIL'}")

    # ---- 3. restart round-trip, bit-identical -----------------------------
    barrier("fcr_pre_reload")
    with timing.section("fcr.restart_reload"):
        rs = load_restart_state_from_h5(
            tensors_filename, mesh_xy, band_slices=band_slices,
            n_rmu_logical=int(meta.n_rmu), low_mem_bands=True)

    worst_nmu = _worst_shard_diff(rs.psi_nmu, wfns.psi_nmu)
    worst_mun = _worst_shard_diff(rs.psi_mun, wfns.psi_mun)
    ok3 = worst_nmu == 0.0 and worst_mun == 0.0
    p0(f"[fcr] 3. restart round-trip  worst|nmu diff|={worst_nmu:.3e}  "
       f"worst|mun diff|={worst_mun:.3e}  {'PASS' if ok3 else 'FAIL'}")

    ok = ok1 and ok2 and ok3
    p0(f"[fcr] VERDICT {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    import traceback
    rc = 1
    try:
        rc = main()
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        rc = 1
    finalize_process(rc)

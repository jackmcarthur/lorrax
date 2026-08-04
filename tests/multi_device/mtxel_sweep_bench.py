"""Before/after for the ⟨mk|V_H|nk⟩ sweep: the k-partitioned plan vs the k-scan.

ONE ARM PER LAUNCH (``MTX_ARM=before|after``).  ``VmHWM`` is a high-water
mark and never decreases, so running both arms in one process would report
the second arm's peak as ``max(A, B)`` and the whole memory claim would be
unfalsifiable.  Correctness is gated separately, both arms in one process,
by ``mtxel_sweep_gate.py``.

  before — ``collectives.gather_k_blocks`` over ``compute_local_V_k``.  The
           production route today: k partitioned round-robin over ranks,
           each k's FULL-BAND FFT BOX built on one rank (wall W1), result
           gathered to an ``(nk, nb, nb)`` array IDENTICAL on every rank
           (wall W2), and — wall W3 — no ability to use more than ``nk``
           ranks at all, so its wall is one full-band k however large P is.
  after  — ``mtxel_sweep.sweep_matrix_elements``.  One k at a time, that
           k's bands sharded over EVERY process, output sharded
           ``P(None,'x','y')``.  Its efficiency is ``nb/P`` and does not
           depend on ``nk``; it keeps scaling to ``P = nb``.

``nk`` is therefore NOT a variable in the after arm's efficiency and this
benchmark does not treat it as one.  What the sweep pays instead is a per-k
reshard collective, which the k-partitioned arm does not need — so the two
arms differ in COMMUNICATION vs SCALABILITY, not in how many ranks are busy.

THE COMPARISON IS DELIBERATELY UNFAIR TO THE NEW PLAN.  Both arms start
from the same RESIDENT sphere ψ, so the ``before`` arm builds its FFT box
from device memory.  In production it loads that box from disk through
``load_kpoint_fftbox_local`` — hundreds of ms of h5py per k that this
benchmark does not charge it.  Any speedup measured here is therefore a
lower bound on the real one.

Reported per rank, because a plan that merely RELOCATED a peak looks clean
on rank 0 alone.

Env: MTX_ARM, MTX_NB, MTX_NS, MTX_NK, MTX_GRID, MTX_NGK, MTX_NPAD, MTX_REPS.
"""
import os
import sys
import time

from runtime import initialize_communicator_stack, finalize_process

RUNTIME = initialize_communicator_stack()

import numpy as np                                            # noqa: E402
import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P    # noqa: E402

from common.collectives import (process_count, process_rank,   # noqa: E402
                                resolve_mesh, gather_k_blocks)
from common.mtxel_sweep import (SweepGeometry,                 # noqa: E402
                                local_potential_operator,
                                sweep_matrix_elements)
from psp.get_DFT_mtxels import compute_local_V_k              # noqa: E402
from common.jax_compile_cache import _STATE as CC_STATE       # noqa: E402

ARM = os.environ.get("MTX_ARM", "after").strip().lower()
NB = int(os.environ.get("MTX_NB", "128"))
NS = int(os.environ.get("MTX_NS", "2"))
NK = int(os.environ.get("MTX_NK", "16"))
GRID = tuple(int(v) for v in os.environ.get("MTX_GRID", "20,20,64").split(","))
NGK = int(os.environ.get("MTX_NGK", "1200"))
NPAD = int(os.environ.get("MTX_NPAD", "8"))
REPS = int(os.environ.get("MTX_REPS", "3"))


def vmhwm_gib():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM:"):
            return float(line.split()[1]) / (1024.0 * 1024.0)
    return float("nan")


def main():
    rank, world = process_rank(), process_count()
    p0 = print if rank == 0 else (lambda *a, **k: None)
    mesh = resolve_mesh()
    px, py = (int(s) for s in mesh.devices.shape)
    nx, ny, nz = GRID
    ngkmax = NGK + NPAD
    volume = 137.035

    box_gib = NB * NS * nx * ny * nz * 16 / 2**30
    tile_gib = NK * NB * NB * 16 / 2**30
    sphere_gib = NK * NB * NS * ngkmax * 16 / 2**30
    p0(f"[bench] arm={ARM} world={world} mesh=({px},{py}) nk={NK} nb={NB} "
       f"ns={NS} grid={GRID} ngk={NGK} pad={NPAD} reps={REPS}")
    p0(f"[bench] one full-band FFT box {box_gib:.3f} GiB (W1) | "
       f"replicated (nk,nb,nb) {tile_gib:.3f} GiB (W2) | "
       f"resident sphere psi {sphere_gib:.3f} GiB global, "
       f"{sphere_gib / world:.3f} GiB/rank sharded")

    rng = np.random.default_rng(20260804)
    gv = np.zeros((NK, ngkmax, 3), dtype=np.int32)
    bidx = np.full((NK, nx, ny, nz), ngkmax, dtype=np.int32)
    for ik in range(NK):
        cells = rng.choice(nx * ny * nz, size=NGK, replace=False)
        gv[ik, :NGK, 0] = cells // (ny * nz)
        gv[ik, :NGK, 1] = (cells // nz) % ny
        gv[ik, :NGK, 2] = cells % nz
        bidx[ik, gv[ik, :NGK, 0], gv[ik, :NGK, 1], gv[ik, :NGK, 2]] = \
            np.arange(NGK, dtype=np.int32)
    gmask = np.zeros((NK, ngkmax), dtype=np.float64)
    gmask[:, :NGK] = 1.0

    psi = (rng.standard_normal((NK, NB, NS, ngkmax))
           + 1j * rng.standard_normal((NK, NB, NS, ngkmax))
           ).astype(np.complex128)
    psi[..., NGK:] = 0.0
    V_r = jnp.asarray(rng.standard_normal(GRID))
    kvecs = rng.standard_normal((NK, 3)) * 0.25

    # BOTH arms hold the same host-side psi, so the VmHWM baseline is
    # identical and the difference between the arms is purely each plan's
    # DEVICE-side cost.  The host array stands in for the on-disk WFN that
    # ``load_kpoint_fftbox_local`` reads in production.
    hwm_base = vmhwm_gib()
    p0(f"[bench] VmHWM after inputs resident: {hwm_base:.3f} GiB")

    gv_j = jnp.asarray(gv)
    gm_j = jnp.asarray(gmask)

    if ARM == "before":
        # ---- the production route: k-partitioned, box per k, gather -----
        # per_k must return a rank-LOCAL array: sweep_local_k ends in a host
        # readback, and a globally sharded block raises "Fetching value for
        # jax.Array that spans non-addressable devices" (measured, job
        # 7888812).  That is not a harness detail -- it is the shape of the
        # plan.  Each rank owns whole k, loads that k's FULL-BAND box by
        # itself (wall W1), and the result is gathered replicated (wall W2).
        def _vh_block(ik):
            box = np.zeros((NB, NS, nx, ny, nz), dtype=np.complex128)
            box[:, :, gv[ik, :NGK, 0], gv[ik, :NGK, 1],
                gv[ik, :NGK, 2]] = psi[ik, :, :, :NGK]
            return compute_local_V_k(jnp.asarray(box), gv_j[ik], V_r, volume,
                                     g_mask=gm_j[ik])

        def run():
            return gather_k_blocks(NK, _vh_block, item_shape=(NB, NB),
                                   label="vh_bench",
                                   print_fn=lambda *a, **k: None)
        first = run()                       # warm-up / compile
        _ = np.asarray(first)[0, 0]
        c0 = CC_STATE.compiles
        t = []
        for _ in range(REPS):
            t0 = time.time()
            out = run()
            _ = np.asarray(out)[0, 0]
            t.append(time.time() - t0)
        compiles = CC_STATE.compiles - c0
        hwm = vmhwm_gib()
        shape_note = f"replicated {tuple(np.shape(out))} on EVERY rank"

    else:
        # ---- the k-scan --------------------------------------------------
        # NOT jax.device_put(array, NamedSharding) IN MULTI-PROCESS.  That
        # path goes through multihost_utils.assert_equal, which runs a
        # process_allgather(tiled=True) of the WHOLE array to check every
        # process passed the same value -- an O(P x array) collective before
        # any work starts.  Measured: 13.0 GiB peak at nb=64/P=16 (against
        # 1.5 GiB for the whole k-partitioned arm) and >900 s at nb=128,
        # which is what made the first bench look as though the SWEEP had a
        # memory problem (jobs 7888820/7888821).  make_array_from_callback
        # builds each process's own shards locally, no collective.
        geom = SweepGeometry(mesh=mesh, fft_grid=GRID, ngkmax=ngkmax,
                             nb=NB, ns=NS, nk=NK, cell_volume=volume)
        # Pad the band axis BEFORE constructing the sharded array: an
        # indivisible band extent is refused at CONSTRUCTION
        # (IndivisibleError, job 7888869), so padding inside the sweep would
        # be too late.  geom.nb is the mesh-padded extent, geom.nb_logical
        # what the caller has; round_up is the same helper both use.
        psi_pad = psi
        if geom.nb != NB:
            psi_pad = np.pad(psi, ((0, 0), (0, geom.nb - NB), (0, 0), (0, 0)))
            p0(f"[bench] band pad {NB} -> {geom.nb} for a "
               f"{px}x{py} mesh (p_prod={geom.p_prod})")
        sharding = NamedSharding(mesh, P(None, ('x', 'y'), None, None))
        psi_j = jax.make_array_from_callback(
            psi_pad.shape, sharding, lambda idx: psi_pad[idx])
        op = local_potential_operator(geom, V_r)
        kw = dict(geom=geom, gvecs=gv, gmask=gmask, box_index=bidx,
                  kvecs=kvecs)

        def run():
            return sweep_matrix_elements(psi_j, operator=op, **kw)
        first = run()
        first.block_until_ready()
        c0 = CC_STATE.compiles
        t = []
        for _ in range(REPS):
            t0 = time.time()
            out = run()
            out.block_until_ready()
            t.append(time.time() - t0)
        compiles = CC_STATE.compiles - c0
        hwm = vmhwm_gib()
        shape_note = (f"sharded {out.sharding.spec}, this rank holds "
                      f"{tuple(out.addressable_shards[0].data.shape)}")

    best, med = min(t), float(np.median(t))
    print(f"[bench] rank={rank:03d} arm={ARM} best={best:.3f}s "
          f"median={med:.3f}s VmHWM={hwm:.3f} GiB "
          f"(delta over inputs {hwm - hwm_base:+.3f}) compiles={compiles}",
          flush=True)
    p0(f"[bench] arm={ARM} result: {shape_note}")
    p0(f"[bench] arm={ARM} SUMMARY best={best:.3f}s median={med:.3f}s "
       f"peak={hwm:.3f} GiB")
    return 0


if __name__ == "__main__":
    # finalize_process() ends the process with os._exit and DOES NOT RETURN.
    # A bare `finally: finalize_process()` therefore swallows any exception
    # main() raised AND exits 0 — the job reports rc=0 on every rank with no
    # traceback and no result, which is exactly how the first run of this
    # bench (job 7888809) looked.  Print the traceback ourselves and hand the
    # real code to finalize_process.
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

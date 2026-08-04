"""Certify the unified k-scan against the per-k local plan it replaces.

Same k, same V(r), same ψ — the ONLY difference is the work split and the
control flow, so any disagreement is the sweep.  Owner tolerance is 1e-12
relative; the sharding deliberately reassociates the G sum, so bit-identity
is neither expected nor demanded (numerical-tolerance ruling).

Four questions, in the order ``docs/dev/matrix_element_sweep_handoff.md``
§6 asks them:

  1. KINETIC — no FFT, so a failure isolates the scan, the reshard and the
     einsum, never the transform.  vs ``compute_kinetic_k``.
  2. LOCAL POTENTIAL — the FFT round trip.  vs ``compute_local_V_k``.
  3. SCAN vs PYTHON LOOP — identical arithmetic and shardings, different
     control flow.  A disagreement here is ``lax.scan`` itself.
  4. ONE COMPILE for the whole k sweep, not one per k.  This is the point
     of D10 and the single biggest design constraint on the sweep.

Every comparison is PER SHARD: each rank checks its own block against the
matching slice of the local plan.  A global ``device_get`` would gather the
very tile this design exists to remove, and a rank-0-only check passes a
plan that puts the right answer on the wrong rank.

Env: MTX_NB, MTX_NS, MTX_NK, MTX_GRID, MTX_NGK, MTX_NPAD.
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
                                resolve_mesh)
from common.mtxel_sweep import (SweepGeometry, kinetic_operator,  # noqa: E402
                                local_potential_operator,
                                sweep_matrix_elements)
from common.jax_compile_cache import _STATE as CC_STATE   # noqa: E402
from psp.get_DFT_mtxels import (compute_kinetic_k,            # noqa: E402
                                compute_local_V_k)

NB = int(os.environ.get("MTX_NB", "32"))
NS = int(os.environ.get("MTX_NS", "2"))
NK = int(os.environ.get("MTX_NK", "5"))
GRID = tuple(int(v) for v in os.environ.get("MTX_GRID", "12,12,20").split(","))
NGK = int(os.environ.get("MTX_NGK", "97"))
NPAD = int(os.environ.get("MTX_NPAD", "3"))
RTOL = 1e-12


def vmhwm_gib():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM:"):
            return float(line.split()[1]) / (1024.0 * 1024.0)
    return float("nan")


def shard_vs_reference(H_sharded, H_ref_all, label, p0):
    """Per-shard max relative delta against the (nk, nb, nb) reference."""
    worst = 0.0
    for sh in H_sharded.addressable_shards:
        blk = np.asarray(sh.data)
        idx = sh.index
        ref = H_ref_all[idx]
        scale = max(float(np.abs(ref).max()), 1e-300)
        worst = max(worst, float(np.abs(blk - ref).max()) / scale)
    p0(f"[mtxel] {label:<34} max rel delta = {worst:.3e}   "
       f"{'PASS' if worst <= RTOL else 'FAIL'}")
    return worst


def main():
    rank, world = process_rank(), process_count()
    p0 = print if rank == 0 else (lambda *a, **k: None)
    mesh = resolve_mesh()
    px, py = (int(s) for s in mesh.devices.shape)
    nx, ny, nz = GRID
    ngkmax = NGK + NPAD
    volume = 137.035

    p0(f"[mtxel] world={world} mesh=({px},{py}) nk={NK} nb={NB} ns={NS} "
       f"grid={GRID} ngk={NGK} pad={NPAD} "
       f"(nb%px={NB % px}, nb%py={NB % py})")

    # ---- deterministic inputs, identical on every rank -------------------
    rng = np.random.default_rng(20260804)
    gv = np.zeros((NK, ngkmax, 3), dtype=np.int32)
    bidx = np.full((NK, nx, ny, nz), ngkmax, dtype=np.int32)
    for ik in range(NK):
        cells = rng.choice(nx * ny * nz, size=NGK, replace=False)
        for i, c in enumerate(cells):
            gv[ik, i] = [c // (ny * nz), (c // nz) % ny, c % nz]
            bidx[ik, gv[ik, i, 0], gv[ik, i, 1], gv[ik, i, 2]] = i
    gmask = np.zeros((NK, ngkmax), dtype=np.float64)
    gmask[:, :NGK] = 1.0

    psi = (rng.standard_normal((NK, NB, NS, ngkmax))
           + 1j * rng.standard_normal((NK, NB, NS, ngkmax))
           ).astype(np.complex128)
    psi[..., NGK:] = 0.0
    V_r = rng.standard_normal(GRID)
    kvecs = rng.standard_normal((NK, 3)) * 0.25
    bdot = np.eye(3) * 1.31 + 0.07

    geom = SweepGeometry(mesh=mesh, fft_grid=GRID, ngkmax=ngkmax,
                         nb=NB, ns=NS, nk=NK, cell_volume=volume)

    psi_j = jax.device_put(jnp.asarray(psi),
                           NamedSharding(mesh, P(None, ('x', 'y'), None, None)))

    # ---- reference: the local per-k plan, on the FULL FFT box (wall W1) ---
    T_ref = np.zeros((NK, NB, NB), dtype=np.complex128)
    V_ref = np.zeros((NK, NB, NB), dtype=np.complex128)
    hwm_before = vmhwm_gib()
    for ik in range(NK):
        box = np.zeros((NB, NS, nx, ny, nz), dtype=np.complex128)
        for i in range(NGK):
            box[:, :, gv[ik, i, 0], gv[ik, i, 1], gv[ik, i, 2]] = psi[ik, :, :, i]
        box_j = jnp.asarray(box)
        T_ref[ik] = np.asarray(compute_kinetic_k(
            box_j, jnp.asarray(gv[ik]), jnp.asarray(kvecs[ik]), bdot,
            g_mask=jnp.asarray(gmask[ik])))
        V_ref[ik] = np.asarray(compute_local_V_k(
            box_j, jnp.asarray(gv[ik]), jnp.asarray(V_r), volume,
            g_mask=jnp.asarray(gmask[ik])))
    hwm_local = vmhwm_gib()
    p0(f"[mtxel] local plan VmHWM {hwm_before:.3f} -> {hwm_local:.3f} GiB "
       f"(the per-k full-band box is wall W1)")

    # ---- 1. kinetic, scanned ---------------------------------------------
    op_T = kinetic_operator(geom, bdot)
    kw = dict(geom=geom, gvecs=gv, gmask=gmask, box_index=bidx, kvecs=kvecs)
    H_T = sweep_matrix_elements(psi_j, operator=op_T, **kw)
    H_T.block_until_ready()
    d_T = shard_vs_reference(H_T, T_ref, "kinetic  scan vs local", p0)

    # ---- 2. local potential, scanned -------------------------------------
    op_V = local_potential_operator(geom, V_r)
    c_before = CC_STATE.compiles
    t0 = time.time()
    H_V = sweep_matrix_elements(psi_j, operator=op_V, **kw)
    H_V.block_until_ready()
    t_sweep = time.time() - t0
    c_scan = CC_STATE.compiles - c_before
    hwm_dist = vmhwm_gib()
    d_V = shard_vs_reference(H_V, V_ref, "local V  scan vs local", p0)

    # ---- 3. scan vs Python loop ------------------------------------------
    # Same arithmetic, same shardings, different control flow.  The compile
    # counts are the D10 claim made measurable: the scan body lowers ONCE
    # for the whole k range, the Python loop lowers per k.
    c_before = CC_STATE.compiles
    H_V_loop = sweep_matrix_elements(psi_j, operator=op_V, use_scan=False, **kw)
    H_V_loop.block_until_ready()
    c_loop = CC_STATE.compiles - c_before
    p0(f"[mtxel] compiles: scan={c_scan}  python-loop={c_loop}  "
       f"(nk={NK}; the loop is expected to grow with nk, the scan is not)")
    d_loop = 0.0
    for a, b in zip(H_V.addressable_shards, H_V_loop.addressable_shards):
        xa, xb = np.asarray(a.data), np.asarray(b.data)
        scale = max(float(np.abs(xb).max()), 1e-300)
        d_loop = max(d_loop, float(np.abs(xa - xb).max()) / scale)
    p0(f"[mtxel] {'local V  scan vs python loop':<34} max rel delta = "
       f"{d_loop:.3e}   {'PASS' if d_loop <= RTOL else 'FAIL'}")

    # ---- 4. sharding and residency ---------------------------------------
    spec = H_V.sharding.spec
    shp = [tuple(s.data.shape) for s in H_V.addressable_shards]
    p0(f"[mtxel] H spec={spec} shard shapes={shp[:2]}{'...' if len(shp) > 2 else ''}")
    p0(f"[mtxel] rank={rank:03d} VmHWM after local {hwm_local:.3f} GiB, "
       f"after sweep {hwm_dist:.3f} GiB   sweep wall {t_sweep:.3f}s",
       flush=True)
    # Every rank reports: a plan that merely RELOCATED a peak looks clean
    # on rank 0 alone.
    if rank != 0:
        print(f"[mtxel] rank={rank:03d} VmHWM after local {hwm_local:.3f} GiB, "
              f"after sweep {hwm_dist:.3f} GiB", flush=True)

    ok = max(d_T, d_V, d_loop) <= RTOL
    p0(f"[mtxel] VERDICT {'PASS' if ok else 'FAIL'} "
       f"(tol {RTOL:.0e} relative, per shard)")
    return 0 if ok else 1


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    finally:
        finalize_process()
    sys.exit(rc)

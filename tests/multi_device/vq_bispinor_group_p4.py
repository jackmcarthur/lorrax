"""P=4 gate for the grouped bispinor V_q build (``gw.v_q_bispinor``).

Four processes, one GPU each, 2x2 mesh.  The synthetic G-flat ζ files of
``tests/test_compute_V_q_bispinor_g_flat.py`` (charge + three transverse,
identity group, every q a parent) go through
``compute_V_q_bispinor_g_flat_to_h5``, which contracts CC alone and the six
TT tiles as one group (``v_q_g_flat._compute_V_q_g_flat_tiles``).  Every
tile must match the per-q einsum ``Σ_G conj(ζ_L) v_q ζ_R`` at 1e-10, and
each ζ file must be read once per q-tile: the charge file once, each ζ_T
once (it was three times, once per TT tile that uses it).  Red twin: the
reference built with T1 and T2 swapped must miss.
Run: ``lx run -N 1 -G 4 -n 4 python3 -u tests/multi_device/vq_bispinor_group_p4.py``.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, _ROOT)

from runtime import initialize_communicator_stack, run_main_and_finalize  # noqa: E402

RUNTIME = initialize_communicator_stack(platform="gpu")

import h5py  # noqa: E402
import jax  # noqa: E402
from jax.experimental import multihost_utils  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

TAG = "[vq-bispinor-group-p4]"


def main():
    from tests.test_compute_V_q_bispinor_g_flat import (
        _build_g_flat_zeta, _identity_symmetry, _ref_tile_V)
    from gw.v_q_bispinor import (UNIQUE_TILES, compute_V_q_bispinor_g_flat_to_h5,
                                 tile_dataset_name)
    from zeta_loader import ZetaLoader

    if jax.process_count() != 4 or jax.device_count() != 4:
        raise SystemExit(f"{TAG} FAIL: needs 4 processes x 1 GPU")
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ("x", "y"))
    tmp = Path(os.environ["SCRATCH"]) / ("vq_bispinor_group_p4_" + os.environ.get(
        "SLURM_JOB_ID", "0") + "_" + os.environ.get("SLURM_STEP_ID", "0"))
    fft_grid, kgrid = (4, 4, 6), (2, 2, 1)
    bvec, cutoff, cell_volume, sys_dim = np.diag([0.7, 0.7, 0.3]), 6.0, 100.0, 2
    n_rmu_C, n_rmu_T = 4, 3
    files = [("C", "zeta_q.h5", n_rmu_C, 0xC0DE), ("T1", "zeta_q_mu1.h5", n_rmu_T, 0xD15C),
             ("T2", "zeta_q_mu2.h5", n_rmu_T, 0xD15D), ("T3", "zeta_q_mu3.h5", n_rmu_T, 0xD15E)]
    disks = {}
    if jax.process_index() == 0:
        tmp.mkdir(parents=True, exist_ok=True)
    multihost_utils.sync_global_devices("vq_bispinor_group_p4 dir")
    for label, name, n_rmu, seed in files:
        # Every rank builds the same host arrays; rank 0 writes the file.
        if jax.process_index() == 0:
            out = _build_g_flat_zeta(tmp, name, fft_grid=fft_grid, kgrid=kgrid,
                                     bvec=bvec, cutoff=cutoff, n_rmu=n_rmu,
                                     sys_dim=sys_dim, seed=seed)
            disks[label] = out[1:]
    multihost_utils.sync_global_devices("vq_bispinor_group_p4 files")
    try:
        loaders = [ZetaLoader(str(tmp / name), mesh=mesh) for _, name, _, _ in files]
        reads = {id(ld): 0 for ld in loaders}
        for ld in loaders:
            orig = ld.read_zeta_G_slab

            def counted(*a, _orig=orig, _id=id(ld), **k):
                reads[_id] += 1
                return _orig(*a, **k)
            ld.read_zeta_G_slab = counted
        out_path = tmp / "v_q_bispinor.h5"
        with mesh:
            compute_V_q_bispinor_g_flat_to_h5(
                zeta_C_loader=loaders[0], zeta_T_loaders=tuple(loaders[1:]),
                sym=_identity_symmetry(kgrid), use_ibz=True,
                centroid_C_idx=loaders[0].r_mu_fft_idx,
                centroid_T_idx=loaders[1].r_mu_fft_idx,
                output_h5_path=str(out_path), mesh_xy=mesh, kgrid=kgrid,
                fft_grid=fft_grid, bvec=bvec, cell_volume=cell_volume,
                sys_dim=sys_dim, n_rmu_C=n_rmu_C, n_rmu_T=n_rmu_T,
                bare_coulomb_cutoff_ry=cutoff, verbose=False)
        n_reads = [reads[id(ld)] for ld in loaders]
        for ld in loaders:
            ld.close()
        multihost_utils.sync_global_devices("vq_bispinor_group_p4 written")
        if jax.process_index() == 0:
            z = {0: disks["C"][0], 1: disks["T1"][0], 2: disks["T2"][0], 3: disks["T3"][0]}
            qf, gv, nk = disks["C"][1:]
            worst = 0.0
            with h5py.File(out_path, "r") as f:
                for mu_L, nu_L in UNIQUE_TILES:
                    got = np.asarray(f[tile_dataset_name(mu_L, nu_L)][...])
                    ref = _ref_tile_V(z[mu_L], z[nu_L], qf, gv, nk, mu_L=mu_L, nu_L=nu_L,
                                      bvec=bvec, cell_volume=cell_volume,
                                      sys_dim=sys_dim, cutoff=cutoff)
                    e = float(np.max(np.abs(got - ref)) / np.max(np.abs(ref)))
                    worst = max(worst, e)
                    print(f"{TAG} tile ({mu_L},{nu_L}) rel={e:.2e}", flush=True)
                got = np.asarray(f[tile_dataset_name(1, 2)][...])
                red = _ref_tile_V(z[2], z[1], qf, gv, nk, mu_L=1, nu_L=2, bvec=bvec,
                                  cell_volume=cell_volume, sys_dim=sys_dim, cutoff=cutoff)
                e_red = float(np.max(np.abs(got - red)) / np.max(np.abs(red)))
            print(f"{TAG} reads per ζ file (C, T1, T2, T3): {n_reads}", flush=True)
            print(f"{TAG} red twin (T1<->T2 in the (1,2) reference): rel={e_red:.2e}",
                  flush=True)
            if not worst <= 1e-10:
                raise SystemExit(f"{TAG} FAIL worst={worst:.3e}")
            if n_reads != [1, 1, 1, 1]:
                raise SystemExit(f"{TAG} FAIL reads {n_reads}, want one per ζ per q-tile")
            if not e_red > 1e-3:
                raise SystemExit(f"{TAG} FAIL red twin did not fire ({e_red:.3e})")
            print(f"{TAG} PASS worst={worst:.2e}", flush=True)
    finally:
        multihost_utils.sync_global_devices("vq_bispinor_group_p4 done")
        if jax.process_index() == 0:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    run_main_and_finalize(main)

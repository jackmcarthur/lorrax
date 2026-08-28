"""Validation harness for the bse_k_grid coarse→fine general-init feature.

Solver A (physics + generality): a Q=0 stack-matvec block-Lanczos that consumes
ONLY the ``load_bse_data_from_restart_sharded`` bundle — so whatever grid the
GENERAL init hands it (coarse or bse_k_grid-densified fine) is what it solves.

Usage (under srun+shifter, 1 GPU):
  python -m tools.bse_kgrid_validate --coarse-restart R3.h5 --coarse-input i3.in \
     --fine NX,NY,NZ [--native-restart R12.h5 --native-input i12.in] \
     --n-val 4 --n-cond 4 --n-eig 6 --out out.npz
"""
from __future__ import annotations
import argparse, os, time
import numpy as np

from runtime import set_default_env
set_default_env()
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from runtime import init_jax_distributed, fallback_to_cpu_if_no_gpu_backend
init_jax_distributed(); fallback_to_cpu_if_no_gpu_backend()

from bse.bse_io import load_bse_data_from_restart_sharded
from bse.bse_ring_comm import make_bse_shardings
from bse.bse_stack_matvec import build_bse_stack_matvec
from common.collectives import single_device_mesh
from solvers.lanczos import block_lanczos_eig_jit

RY2EV = 13.6056980659


def solve_lowest(data, mesh_xy, n_eig, block_size, max_iter, kernel="bse"):
    """Lowest n_eig TDA excitons (Ry) from a loader bundle via the stack matvec."""
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    nk = nkx * nky * nkz
    nc_pad, nv_pad = int(data["n_cond_pad"]), int(data["n_val_pad"])
    n_flat = nc_pad * nv_pad * nk
    sh = make_bse_shardings(mesh_xy)
    matvec = build_bse_stack_matvec(mesh_xy, nkx, nky, nkz, kernel=kernel)
    W_R = jnp.fft.ifftn(data["W_q"], axes=(2, 3, 4), norm="ortho")

    @jax.jit
    def _solve(psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R, V, M_X, M_Y):
        def mvb(Vb):
            X = Vb.reshape(block_size, nc_pad, nv_pad, nk)
            X = jax.lax.with_sharding_constraint(X, sh.X)
            HX = matvec(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y,
                        eps_c, eps_v, W_R, V, M_X, M_Y)
            return HX.reshape(block_size, -1)
        evs, _ = block_lanczos_eig_jit(mvb, n_flat, n_eig=n_eig,
                                       block_size=block_size, max_iter=max_iter,
                                       n_reorth=max_iter)
        return evs[:n_eig].real
    evs = _solve(data["psi_c_X"], data["psi_c_Y"], data["psi_v_X"],
                 data["psi_v_Y"], data["eps_c"], data["eps_v"], W_R,
                 data["V_q0"], data["M_X"], data["M_Y"])
    return np.asarray(jax.device_get(evs))


def bundle_arrays(data):
    """Device-get the array fields for a byte-identity comparison."""
    keys = ["psi_c_X", "psi_c_Y", "psi_v_X", "psi_v_Y", "M_X", "M_Y",
            "eps_c", "eps_v", "W_q", "V_q0"]
    return {k: np.asarray(jax.device_get(data[k])) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coarse-restart", required=True)
    ap.add_argument("--coarse-input", required=True)
    ap.add_argument("--fine", required=True, help="NX,NY,NZ")
    ap.add_argument("--native-restart", default=None)
    ap.add_argument("--native-input", default=None)
    ap.add_argument("--n-val", type=int, default=4)
    ap.add_argument("--n-cond", type=int, default=4)
    ap.add_argument("--n-eig", type=int, default=6)
    ap.add_argument("--block-size", type=int, default=8)
    ap.add_argument("--max-iter", type=int, default=40)
    ap.add_argument("--out", default="bse_kgrid_validate.npz")
    args = ap.parse_args()
    fine = tuple(int(s) for s in args.fine.replace(",", " ").split())
    mesh_xy = single_device_mesh()
    kw = dict(n_val=args.n_val, n_cond=args.n_cond, mesh_xy=mesh_xy)
    res = {}

    def _load(restart, inp, bse_k_grid):
        return load_bse_data_from_restart_sharded(
            restart, input_file=inp, inject_head=True, bse_k_grid=bse_k_grid, **kw)

    # 1) coarse (no flag) ----------------------------------------------------
    print("\n=== [1] COARSE (no bse_k_grid) ===", flush=True)
    d_coarse = _load(args.coarse_restart, args.coarse_input, None)
    cg = (int(d_coarse["nkx"]), int(d_coarse["nky"]), int(d_coarse["nkz"]))
    print(f"coarse grid = {cg}, nk = {cg[0]*cg[1]*cg[2]}")
    ev_coarse = solve_lowest(d_coarse, mesh_xy, args.n_eig, args.block_size, args.max_iter)
    print("coarse excitons (eV):", np.round(ev_coarse * RY2EV, 5))
    res["ev_coarse"] = ev_coarse; res["coarse_grid"] = np.array(cg)

    # 2) on-grid identity: bse_k_grid == coarse must be byte-identical --------
    print("\n=== [2] ON-GRID IDENTITY (bse_k_grid == coarse) ===", flush=True)
    d_ident = _load(args.coarse_restart, args.coarse_input, cg)
    a0, a1 = bundle_arrays(d_coarse), bundle_arrays(d_ident)
    maxdiff = {k: float(np.max(np.abs(a0[k] - a1[k]))) for k in a0}
    ident = all(np.array_equal(a0[k], a1[k]) for k in a0)
    print("byte-identical bundle:", ident)
    for k, v in maxdiff.items():
        print(f"   max|Δ {k}| = {v:.3e}")
    ev_ident = solve_lowest(d_ident, mesh_xy, args.n_eig, args.block_size, args.max_iter)
    id_ev_diff = float(np.max(np.abs(ev_ident - ev_coarse)))
    print("identity solve max|Δeig| (Ry):", f"{id_ev_diff:.3e}")
    res["identity_byte_equal"] = ident
    res["identity_ev_maxdiff_ry"] = id_ev_diff

    # 3) interpolated fine ---------------------------------------------------
    print(f"\n=== [3] INTERPOLATED FINE (bse_k_grid = {fine}) ===", flush=True)
    d_fine = _load(args.coarse_restart, args.coarse_input, fine)
    fg = (int(d_fine["nkx"]), int(d_fine["nky"]), int(d_fine["nkz"]))
    print(f"fine grid = {fg}, nk = {fg[0]*fg[1]*fg[2]}")
    assert fg == fine
    ev_fine = solve_lowest(d_fine, mesh_xy, args.n_eig, args.block_size, args.max_iter)
    print("interp-fine excitons (eV):", np.round(ev_fine * RY2EV, 5))
    res["ev_interp_fine"] = ev_fine; res["fine_grid"] = np.array(fg)

    # 3b) generality: SECOND solver (RPA-kernel stack matvec = a different
    #     kernel path) on the SAME fine bundle proves the interpolation is in
    #     the init, not one solver.  (A fuller second driver, bse_feast CLI, is
    #     run separately.)
    ev_fine_rpa = solve_lowest(d_fine, mesh_xy, args.n_eig, args.block_size,
                               args.max_iter, kernel="rpa")
    print("interp-fine RPA (D+V, no W) excitons (eV):",
          np.round(ev_fine_rpa * RY2EV, 5))
    res["ev_interp_fine_rpa"] = ev_fine_rpa

    print("\nΔ(interp-fine − coarse) lowest (meV):",
          round(float((ev_fine[0] - ev_coarse[0]) * RY2EV * 1e3), 2))

    # 4) native fine (if provided) ------------------------------------------
    if args.native_restart and args.native_input:
        print(f"\n=== [4] NATIVE FINE ({args.native_restart}) ===", flush=True)
        d_nat = _load(args.native_restart, args.native_input, None)
        ng = (int(d_nat["nkx"]), int(d_nat["nky"]), int(d_nat["nkz"]))
        print(f"native grid = {ng}, nk = {ng[0]*ng[1]*ng[2]}")
        ev_nat = solve_lowest(d_nat, mesh_xy, args.n_eig, args.block_size, args.max_iter)
        print("native-fine excitons (eV):", np.round(ev_nat * RY2EV, 5))
        res["ev_native_fine"] = ev_nat; res["native_grid"] = np.array(ng)
        if ng == fg:
            interp_err = (ev_fine - ev_nat) * RY2EV * 1e3
            print("INTERPOLATION ERROR (interp−native) lowest states (meV):",
                  np.round(interp_err, 2))
            res["interp_error_mev"] = interp_err

    np.savez(args.out, **res)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

"""bse/test_davidson_bse.py — Block Davidson on the BSE problem.

Loads the same Si 4×4×4, 8×8 BSE setup that ``solve_bse_sharded`` uses,
runs the refactored shape-agnostic Davidson against it, and compares
the lowest 20 eigenvalues + per-state |A^S[c,v,k]|² densities to the
BGW reference at
    /pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/
        00_bgw_bse_8x8/eigenvectors.h5

This is a smoke + numerical sanity test, NOT a unit test (no
asserts). It prints residual norms per iteration so the
profiling-stack tooling can pick up the wall-time picture.

Usage (from the cohsex_bse.in run dir, under lxrun):
    LORRAX_NGPU=4 lxrun python3 -u -m bse.test_davidson_bse \\
        -i cohsex_bse.in --eqp <bgw_eqp.dat> --n-occ 4 \\
        --n-val 8 --n-cond 8 --n-eig 20 --max-iter 60 \\
        --bgw-h5 /pscratch/.../00_bgw_bse_8x8/eigenvectors.h5
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from functools import partial
from typing import Optional, Tuple

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update("jax_enable_x64", True)

from common.fft_helpers import make_sharded_ifftn_3d
from solvers.davidson import davidson, warmup_davidson_jit

from .bse_io import _find_restart_file, load_bse_data_from_restart_sharded
from .bse_ring_comm import create_mesh_2d, make_bse_shardings
from .bse_simple import build_bse_simple_matvec
from .bse_davidson_helpers import bse_diagonal_precond, init_bse_subspace


RYD2EV = 13.6056980659


# ═══════════════════════════════════════════════════════════════════════
#  Data load + matvec build
# ═══════════════════════════════════════════════════════════════════════

def _load_data_and_matvec(
    input_file: str,
    n_val: int,
    n_cond: int,
    n_occ: Optional[int],
    eqp_file: Optional[str],
    include_W: bool,
):
    restart = _find_restart_file(input_file)
    mesh_xy = create_mesh_2d()
    sh = make_bse_shardings(mesh_xy)
    grid_x, grid_y = mesh_xy.devices.shape
    print(f"Mesh: ({grid_x}, {grid_y}) → {grid_x*grid_y} devices")

    data = load_bse_data_from_restart_sharded(
        restart, n_val=n_val, n_cond=n_cond, mesh_xy=mesh_xy,
        input_file=input_file, n_occ=n_occ,
    )

    # Optional EQP override (same flow as bse_jax._preview_lanczos).
    # Use data["n_val"] / data["n_cond"] (capped at availability) rather
    # than the user-requested n_val / n_cond — otherwise eps_v can be
    # wider than psi_v_X (when n_occ < n_val) and the matvec einsum fails.
    if eqp_file is not None:
        from .bse_io import apply_eqp_corrections, _pad_axis_to_multiple
        n_val_eff = int(data["n_val"])
        n_cond_eff = int(data["n_cond"])
        with h5py.File(restart, "r") as f:
            enk_full_np = np.asarray(f["enk_full"][:])
        enk_full_np = apply_eqp_corrections(
            enk_full_np, eqp_file, input_file=input_file)
        mean_enk = np.mean(enk_full_np, axis=0)
        n_occ_eff = n_occ if n_occ is not None else int((mean_enk < 0.0).sum())
        val_idx = np.arange(n_occ_eff - n_val_eff, n_occ_eff)
        cond_idx = np.arange(n_occ_eff, n_occ_eff + n_cond_eff)
        eps_v_new = jnp.asarray(enk_full_np[:, val_idx])
        eps_c_new = jnp.asarray(enk_full_np[:, cond_idx])
        eps_v_new, _ = _pad_axis_to_multiple(eps_v_new, axis=1, multiple=grid_y)
        eps_c_new, _ = _pad_axis_to_multiple(eps_c_new, axis=1, multiple=grid_x)
        data["eps_v"] = eps_v_new
        data["eps_c"] = eps_c_new

    nc_pad = int(data["n_cond_pad"])
    nv_pad = int(data["n_val_pad"])
    nk = int(data["nkx"]) * int(data["nky"]) * int(data["nkz"])
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])

    # ── Build matvec ─────────────────────────────────────────────────
    matvec_simple = build_bse_simple_matvec(
        mesh_xy, nkx, nky, nkz, include_W=include_W)

    # Precompute W_R outside the per-call jit. Use the same sharded
    # 3D-FFT helper as solve_bse_sharded.
    if include_W:
        _W_ifftn = make_sharded_ifftn_3d(
            mesh_xy, sh.W.spec, sh.W.spec, axes=(2, 3, 4), norm='ortho')
        W_R = jax.jit(_W_ifftn, in_shardings=sh.W, out_shardings=sh.W)(data["W_q"])
        # Force materialise so we don't keep W_q alive.
        jax.block_until_ready(W_R)
    else:
        W_R = data["W_q"]

    # Bind (psi_*, eps_*, W_R, V_q0) to a single-argument apply_H(X).
    # Davidson hands back X with shape (m, nc, nv, nk) and sharding
    # P(None, "x", "y", None); matvec_simple already wraps that with
    # with_sharding_constraint internally and returns HX with the same
    # spec. Pure jit on a closure preserves XLA partitioning.
    psi_c_X = data["psi_c_X"]
    psi_c_Y = data["psi_c_Y"]
    psi_v_X = data["psi_v_X"]
    psi_v_Y = data["psi_v_Y"]
    eps_c = data["eps_c"]
    eps_v = data["eps_v"]
    V_q0 = data["V_q0"]

    def apply_H(X):
        return matvec_simple(
            X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y,
            eps_c, eps_v, W_R, V_q0,
        )

    return mesh_xy, sh, data, apply_H


# ═══════════════════════════════════════════════════════════════════════
#  BGW reference loader + density cosine similarity
# ═══════════════════════════════════════════════════════════════════════

def _load_bgw_eigvecs(bgw_h5: str, n_eig: int, n_val: int, n_cond: int, nk: int):
    """Read BGW eigenvectors.h5 and return (evals_eV, |A|² density (n, nc, nv, nk)).

    Density is gauge-invariant per state so safe for cosine sim.
    Valence axis is REVERSED on read: BGW iv=1 is highest valence, our
    internal v=0 is lowest. We flip on the way in so densities are in
    LORRAX convention.
    """
    with h5py.File(bgw_h5, "r") as f:
        eigvals = np.asarray(f["exciton_data/eigenvalues"][:n_eig])  # eV
        # BGW shape (nq=1, N, nk, nc, nv, ns, 2)
        evecs = np.asarray(f["exciton_data/eigenvectors"][0, :n_eig])
        # Combine real/imag → complex.
        evecs_c = evecs[..., 0] + 1j * evecs[..., 1]
        # Drop ns=1 axis: (n_eig, nk, nc, nv, 1) → (n_eig, nk, nc, nv).
        evecs_c = evecs_c[..., 0]
        # Flip valence axis (BGW vs LORRAX convention).
        evecs_c = evecs_c[..., ::-1]
        # Reorder to LORRAX convention (n, c, v, k).
        evecs_c = np.transpose(evecs_c, (0, 2, 3, 1))   # (n, nc, nv, nk)
    density = np.abs(evecs_c) ** 2   # |A^S[c,v,k]|² — gauge-invariant
    return eigvals, density


def _cosine_similarity_density(
    lorrax_density: np.ndarray, bgw_density: np.ndarray
) -> np.ndarray:
    """Per-state best-match cosine similarity between two (n, ...) density sets.

    For each LORRAX state, find the BGW state with maximal cosine
    similarity (densities flattened over c, v, k).
    """
    L = lorrax_density.reshape(lorrax_density.shape[0], -1)
    B = bgw_density.reshape(bgw_density.shape[0], -1)
    L_norm = np.linalg.norm(L, axis=1, keepdims=True)
    B_norm = np.linalg.norm(B, axis=1, keepdims=True)
    L = L / np.maximum(L_norm, 1e-30)
    B = B / np.maximum(B_norm, 1e-30)
    cos = L @ B.T
    return cos.max(axis=1)


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="BSE Block Davidson smoke + comparison")
    parser.add_argument("-i", "--input", required=True,
                        help="cohsex.in (used to find isdf_tensors_*.h5 + WFN.h5)")
    parser.add_argument("--n-val", type=int, default=8)
    parser.add_argument("--n-cond", type=int, default=8)
    parser.add_argument("--n-occ", type=int, default=4)
    parser.add_argument("--n-eig", type=int, default=20)
    parser.add_argument("--n-random-init", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=60)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--eqp", type=str, default=None,
                        help="BGW eqp1.dat for QP corrections")
    parser.add_argument(
        "--bgw-h5",
        type=str,
        default="/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/00_bgw_bse_8x8/eigenvectors.h5",
        help="BGW eigenvectors.h5 reference for comparison")
    parser.add_argument("--no-W", action="store_true",
                        help="Disable W (RPA-like) — debug only")
    args = parser.parse_args()

    print("─" * 70)
    print("BSE Block Davidson smoke test")
    print("─" * 70)

    include_W = not args.no_W

    t0 = time.perf_counter()
    mesh_xy, sh, data, apply_H = _load_data_and_matvec(
        args.input, args.n_val, args.n_cond, args.n_occ, args.eqp, include_W,
    )
    nc_pad = int(data["n_cond_pad"])
    nv_pad = int(data["n_val_pad"])
    nk = int(data["nkx"]) * int(data["nky"]) * int(data["nkz"])
    print(f"Loaded BSE: nc_pad={nc_pad}, nv_pad={nv_pad}, nk={nk} "
          f"(dim={nc_pad*nv_pad*nk})  {time.perf_counter()-t0:.2f}s")

    # ── initial subspace (sharded) ───────────────────────────────────
    V0 = init_bse_subspace(
        data["eps_c"], data["eps_v"], n_eig=args.n_eig,
        n_random=args.n_random_init,
        mesh=mesh_xy, sharding=sh.X, seed=42,
    )
    print(f"V0 shape={V0.shape} dtype={V0.dtype} sharding={V0.sharding}")

    # ── preconditioner (sharded) ─────────────────────────────────────
    # Drop the batch axis from sh.X (P(None,"x","y",None) → P("x","y",None)).
    delta_E_sh = NamedSharding(mesh_xy, P("x", "y", None))
    precond_fn = bse_diagonal_precond(
        data["eps_c"], data["eps_v"], epsilon_shift=1e-3,
        sharding=delta_E_sh,
    )

    # ── warmup the Ritz-projection JIT ───────────────────────────────
    t1 = time.perf_counter()
    warmup_davidson_jit(
        args.n_eig, (nc_pad, nv_pad, nk),
        m_max=4 * args.n_eig, dtype=jnp.complex128, sharding=sh.X,
    )
    # Warm up apply_H at the two batch sizes Davidson uses (n_eig and
    # 2·n_eig … expansion). Cheap because matvec is already jit'd.
    HV0 = apply_H(V0)
    jax.block_until_ready(HV0)
    print(f"JIT warmup: {time.perf_counter()-t1:.2f}s")

    # ── Davidson run ────────────────────────────────────────────────
    print("\n── Davidson ──")
    t_dav = time.perf_counter()
    eigvals, eigvecs = davidson(
        apply_H,
        n_eig=args.n_eig,
        precond_fn=precond_fn,
        X0=V0,
        m_max=4 * args.n_eig,
        max_iter=args.max_iter,
        tol=args.tol,
        verbose=True,
    )
    jax.block_until_ready(eigvecs)
    dav_dt = time.perf_counter() - t_dav
    eigvals_np = np.asarray(eigvals)
    print(f"\nDavidson total: {dav_dt:.2f}s")
    print(f"eigvals (Ry): {np.array2string(eigvals_np, precision=6)}")
    print(f"eigvals (eV): {np.array2string(eigvals_np*RYD2EV, precision=6)}")

    # ── BGW comparison ──────────────────────────────────────────────
    if not os.path.exists(args.bgw_h5):
        print(f"BGW reference not found: {args.bgw_h5}; skipping comparison")
        return

    bgw_evals_eV, bgw_density = _load_bgw_eigvecs(
        args.bgw_h5, args.n_eig, args.n_val, args.n_cond, nk)
    print("\n── BGW comparison ──")
    print(f"BGW lowest {args.n_eig} (eV): {np.array2string(bgw_evals_eV, precision=6)}")
    eig_diff_meV = (eigvals_np*RYD2EV - bgw_evals_eV) * 1000.0
    print(f"Eigenvalue diff (meV): {np.array2string(eig_diff_meV, precision=2)}")
    print(f"  mean={np.mean(np.abs(eig_diff_meV)):.2f} meV, "
          f"max={np.max(np.abs(eig_diff_meV)):.2f} meV")

    # Density similarity (drop padded bands for a fair comparison).
    n_val = args.n_val
    n_cond = args.n_cond
    # eigvecs is sharded P(None,"x","y",None) across all processes.
    # Materialise replicated, then host-fetch.
    rep_full = NamedSharding(mesh_xy, P())
    eigvecs_rep = jax.jit(lambda x: x, out_shardings=rep_full)(eigvecs)
    jax.block_until_ready(eigvecs_rep)
    from solvers.davidson import _to_host as _dav_to_host
    eigvecs_np = _dav_to_host(eigvecs_rep)[:, :n_cond, :n_val, :]
    lorrax_density = np.abs(eigvecs_np) ** 2
    cos_sim = _cosine_similarity_density(lorrax_density, bgw_density)
    print(f"\nPer-state |A|² cosine similarity (Davidson vs BGW), best match per state:")
    for i, c in enumerate(cos_sim):
        print(f"  S={i:3d}  cos_sim = {c:.4f}")
    print(f"  mean cos_sim = {cos_sim.mean():.4f}, "
          f"min = {cos_sim.min():.4f}, max = {cos_sim.max():.4f}")

    print("\n── done ──")


if __name__ == "__main__":
    main()

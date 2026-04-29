"""Full diagonalization of the LORRAX BSE Hamiltonian.

Mirrors BGW's ``absorption.x diagonalization`` mode: builds the dense
H_BSE matrix by batched matvec, diagonalizes via ``jnp.linalg.eigh``,
projects the requested polarisation of the dipole onto each
eigenvector, and writes a BGW-format ``eigenvalues_b{1,2,3}.dat``.

Use this when you want a fair LORRAX-vs-BGW eigenvector-route
comparison. Lanczos at finite Krylov mixes near-degenerate
subspaces and redistributes per-state oscillator strengths even
when eigenvalues are well-converged — only full diag avoids that.

Cost: BSE_dim × matvec/batch matvec calls + one eigh. For Si 8×8
(BSE_dim = 4096): ~2-3 min on 4 GPUs, ~256 MB of HBM for H.
"""
from __future__ import annotations
import argparse
import os
import time

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from .bse_io import (
    _find_restart_file,
    apply_eqp_corrections,
    load_bse_data_from_restart_sharded,
    resolve_n_occ,
)
from .bse_lanczos import solve_bse_sharded  # for matvec build
from .bse_ring_comm import create_mesh_2d, make_bse_shardings
from .bse_simple import build_bse_simple_matvec
from .absorption_common import (
    load_dipole_h5,
    slice_dipole_to_bse_window,
    write_eigenvalues_dat,
)
from common.fft_helpers import make_sharded_ifftn_3d


def build_dense_bse_via_matvec(matvec_fn, shape, batch=64):
    """Probe the matvec with batched unit vectors → dense H.

    matvec_fn : callable taking ``(m, *trailing)`` → ``(m, *trailing)``.
    shape     : trailing shape of one BSE state vector, e.g. ``(nc, nv, nk)``.

    Returns
    -------
    H : (N, N) ndarray, complex128.   N = prod(shape).
    """
    N = int(np.prod(shape))
    H = np.zeros((N, N), dtype=np.complex128)
    flat = np.arange(N)
    n_batches = (N + batch - 1) // batch
    print(f"[full_diag] building dense H of size {N}×{N} in {n_batches} batches of {batch}...", flush=True)
    t0 = time.time()
    for ib in range(n_batches):
        i0 = ib * batch
        i1 = min(N, i0 + batch)
        m = i1 - i0
        X = np.zeros((m,) + tuple(shape), dtype=np.complex128)
        for j, i in enumerate(range(i0, i1)):
            idx = np.unravel_index(i, shape)
            X[(j,) + idx] = 1.0
        HX = np.asarray(jax.device_get(matvec_fn(jnp.asarray(X))))
        H[:, i0:i1] = HX.reshape(m, N).T
        if (ib + 1) % 8 == 0 or ib == n_batches - 1:
            print(f"  batch {ib+1}/{n_batches}  ({(ib+1)*batch}/{N})  "
                  f"{time.time()-t0:.1f} s", flush=True)
    print(f"[full_diag] dense H built in {time.time()-t0:.1f} s", flush=True)
    return 0.5 * (H + H.conj().T)   # enforce hermiticity (TDA → real eigvals)


def _apply_eqp_inplace(data, eqp_file, restart_file, input_file):
    """Replace eps_v / eps_c with QP-corrected values (matches bse_jax)."""
    with h5py.File(restart_file, "r") as f:
        enk = np.asarray(f["enk_full"][:])
    enk = apply_eqp_corrections(enk, eqp_file, input_file=input_file)
    n_occ = resolve_n_occ(enk, n_occ=None, input_file=input_file)
    n_val = int(data["n_val"])
    n_cond = int(data["n_cond"])
    val_idx = np.arange(n_occ - n_val, n_occ)
    cond_idx = np.arange(n_occ, n_occ + n_cond)
    eps_v = jnp.asarray(enk[:, val_idx])
    eps_c = jnp.asarray(enk[:, cond_idx])
    nv_pad = int(data["n_val_pad"])
    nc_pad = int(data["n_cond_pad"])
    if eps_v.shape[1] < nv_pad:
        eps_v = jnp.pad(eps_v, ((0, 0), (0, nv_pad - eps_v.shape[1])))
    if eps_c.shape[1] < nc_pad:
        eps_c = jnp.pad(eps_c, ((0, 0), (0, nc_pad - eps_c.shape[1])))
    data["eps_v"] = eps_v
    data["eps_c"] = eps_c


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("-i", "--input", required=True,
                   help="cohsex.in (used to find restart + WFN.h5 / n_occ)")
    p.add_argument("--n-val", type=int, required=True)
    p.add_argument("--n-cond", type=int, required=True)
    p.add_argument("--n-occ", type=int, default=None,
                   help="Override n_occ (else read from WFN.h5 ifmax)")
    p.add_argument("--eqp", default=None,
                   help="BGW eqp.dat for QP corrections (recommended)")
    p.add_argument("--dipole", default="dipole.h5",
                   help="dipole.h5 for oscillator strengths "
                        "(use dipole_p_only.h5 from --skip-vnl)")
    p.add_argument("--V-cell", type=float, required=True,
                   help="Unit-cell volume in bohr³")
    p.add_argument("--n-eig-write", type=int, default=None,
                   help="How many eigenvalues to write (default: all)")
    p.add_argument("--out-prefix", default="eigenvalues_full",
                   help="Output prefix; writes <prefix>_b{1,2,3}.dat")
    p.add_argument("--batch", type=int, default=64,
                   help="Matvec batch size (default 64)")
    p.add_argument("--matvec-kind", choices=("simple", "ring"), default="simple")
    args = p.parse_args(argv)

    restart_file = _find_restart_file(args.input)
    mesh_xy = create_mesh_2d()
    grid_x, grid_y = mesh_xy.devices.shape
    sh = make_bse_shardings(mesh_xy)

    print(f"[full_diag] loading BSE data from {restart_file}", flush=True)
    data = load_bse_data_from_restart_sharded(
        restart_file, n_val=args.n_val, n_cond=args.n_cond, mesh_xy=mesh_xy,
        input_file=args.input, n_occ=args.n_occ,
    )
    if args.eqp is not None:
        _apply_eqp_inplace(data, args.eqp, restart_file, args.input)
        print(f"[full_diag] applied EQP from {args.eqp}", flush=True)

    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    nk = nkx * nky * nkz
    nc_pad = int(data["n_cond_pad"])
    nv_pad = int(data["n_val_pad"])
    N = nc_pad * nv_pad * nk
    print(f"[full_diag] BSE_dim = {nc_pad}c × {nv_pad}v × {nk}k = {N}", flush=True)

    # Precompute W_R = ifft(W_q) once.
    _W_local_ifftn = make_sharded_ifftn_3d(
        mesh_xy, sh.W.spec, sh.W.spec, axes=(2, 3, 4), norm="ortho")
    W_R = _W_local_ifftn(data["W_q"])

    matvec_ring = build_bse_simple_matvec(
        mesh_xy, nkx, nky, nkz, include_W=True)

    psi_c_X, psi_c_Y = data["psi_c_X"], data["psi_c_Y"]
    psi_v_X, psi_v_Y = data["psi_v_X"], data["psi_v_Y"]
    eps_c, eps_v = data["eps_c"], data["eps_v"]
    V_q0 = data["V_q0"]

    @jax.jit
    def matvec_jit(X):
        X = jax.lax.with_sharding_constraint(X, sh.X)
        return matvec_ring(X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y,
                           eps_c, eps_v, W_R, V_q0)

    H = build_dense_bse_via_matvec(matvec_jit, (nc_pad, nv_pad, nk), batch=args.batch)
    max_imag = float(np.abs(H.imag).max()) if np.iscomplexobj(H) else 0.0
    print(f"[full_diag] H built; max|imag| = {max_imag:.2e}", flush=True)

    print(f"[full_diag] diagonalising {N}x{N}...", flush=True)
    t0 = time.time()
    eigvals_Ry, eigvecs = np.linalg.eigh(H)
    print(f"[full_diag] eigh done in {time.time()-t0:.1f} s, "
          f"lowest 5 (eV): {eigvals_Ry[:5]*13.6056980659}", flush=True)

    n_write = args.n_eig_write or N
    n_write = min(n_write, N)
    eigvals_eV = eigvals_Ry[:n_write] * 13.6056980659
    eigvecs = eigvecs[:, :n_write]   # (N, n_write)

    # Dipole projections — slice dipole.h5 to (nc, nv) window matching BSE basis.
    dipole_cart, deltaE, _ = load_dipole_h5(args.dipole)
    n_occ = args.n_occ if args.n_occ is not None else resolve_n_occ(
        np.asarray(jax.device_get(eps_v)), input_file=args.input)
    d_alpha, _ = slice_dipole_to_bse_window(
        dipole_cart, deltaE, n_occ=n_occ, n_val=nv_pad, n_cond=nc_pad)
    # d_alpha: (3, nk, nc, nv).  Reshape to BSE block layout (3, nc, nv, nk).
    d_block = np.transpose(d_alpha, (0, 2, 3, 1))  # (3, nc, nv, nk)
    d_flat = d_block.reshape(3, N)
    # Project onto each eigenvector: d_S_α = Σ_i conj(eigvec[i, S]) · d_α[i]
    proj = np.einsum("iS,ai->Sa", eigvecs.conj(), d_flat, optimize=True)  # (n_write, 3)

    n_spinor = 2  # SOC default for Si runs; override via attr if needed
    V_super = args.V_cell * nk
    for a, suffix in enumerate(("b1", "b2", "b3")):
        out = f"{args.out_prefix}_{suffix}.dat"
        write_eigenvalues_dat(
            out, eigvals_eV, proj[:, a],
            n_spin=1, n_spinor=n_spinor, vol_supercell=V_super,
        )
        print(f"[full_diag] wrote {out} ({n_write} eigvals)", flush=True)


if __name__ == "__main__":
    main()

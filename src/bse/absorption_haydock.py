"""ε₂(ω) via Haydock continued fraction — eigenvector-free route.

Mirrors BGW's ``BSE/haydock.f90 → iterate.f90 → absh.f90`` pathway:

  - seed Lanczos at the dipole vector ``|d^α⟩``, one per polarisation
    α ∈ {x, y, z};  3-block recursion shares the BSE matvec
  - iterate ``n_iter`` steps without reorthogonalisation, accumulating
    ``α_n`` and ``β_n``  (``β_n`` already factored out of the next vector)
  - at evaluation time, build ε₂(ω) from the backward continued fraction:

        g(z) = 1 / (z − α_1 − β_1²/(z − α_2 − β_2²/(z − α_3 − …)))
        ε_2^α(ω) = (16π² / (V_cell · N_k · n_spin · n_spinor))
                   × ‖d^α‖² · (−Im g(ω+iη) / π)

The Haydock trick: never form a single eigenvector; spectrum lives in
the (α_n, β_n) recurrence.

Convention on ``n_spin`` vs ``n_spinor``: BGW's ``n_spin`` is 2 only for
collinear spin-polarised calcs.  For our spinor calcs ``n_spin = 1`` and
``n_spinor = 2``.
"""
from __future__ import annotations

# Ensure x64 + jax.distributed bootstrap before any jax-collective code.
from runtime import init_jax_distributed
init_jax_distributed()

import argparse
from functools import partial
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.fft_helpers import make_sharded_ifftn_3d

from .absorption_common import (
    RYD2EV,
    build_dipole_vector_bse,
    kramers_kronig_eps1,
    load_dipole_h5,
    slice_dipole_to_bse_window,
    write_absorption_dat,
    write_absorption_h5,
)
from .bse_io import _find_restart_file, load_bse_data_from_restart_sharded
from .bse_ring_comm import build_bse_ring_matvec, create_mesh_2d, make_bse_shardings


def haydock_recursion_block(matvec_block, d_block, n_iter):
    """Run a 3-block Haydock recurrence (no reorthogonalisation).

    Parameters
    ----------
    matvec_block : (block, n_cond, n_val, nk) → (block, n_cond, n_val, nk)
        Applies the BSE Hamiltonian to a block of trial vectors.
    d_block : jax.Array, shape (block, n_cond, n_val, nk)
        Seed dipole vectors |d^α⟩ for α = 1..block.
    n_iter : int
        Number of recurrence steps.

    Returns
    -------
    alphas : (block, n_iter) float — α_n = ⟨s_n|H|s_n⟩.
    betas  : (block, n_iter) float — β_n = ‖r_n‖ for r_n = (H−α_n)s_n − β_{n−1} s_{n−1}.
    norms  : (block,) float        — ‖d^α‖.
    """
    flat_shape = (d_block.shape[0], -1)
    d_flat = d_block.reshape(flat_shape)
    norms = jnp.linalg.norm(d_flat, axis=1)                                  # (block,)
    bcast = (-1,) + (1,) * (d_block.ndim - 1)
    s_curr = d_block / norms.reshape(bcast)
    s_prev = jnp.zeros_like(s_curr)

    def body(carry, _):
        s_curr, s_prev, b_curr = carry
        Hs = matvec_block(s_curr)
        a = jnp.einsum("a...,a...->a", s_curr.conj(), Hs).real               # (block,)
        a_b = a.reshape(bcast)
        b_b = b_curr.reshape(bcast)
        r = Hs - a_b * s_curr - b_b * s_prev
        b_next = jnp.linalg.norm(r.reshape(flat_shape), axis=1)              # (block,)
        s_next = r / b_next.reshape(bcast)
        return (s_next, s_curr, b_next), (a, b_next)

    init_b = jnp.zeros(d_block.shape[0], dtype=jnp.float64)
    (_, _, _), (alphas_T, betas_T) = lax.scan(
        body, (s_curr, s_prev, init_b), None, length=n_iter)
    # scan stacks along axis 0 → transpose to (block, n_iter)
    return alphas_T.T, betas_T.T, norms


def absorption_from_haydock(
    omegas_Ry, eta_Ry, alphas, betas, norms,
    V_cell, n_k, n_spin, n_spinor,
):
    """Build ε₂(ω) from the Haydock (α_n, β_n) coefficients.

    Three independent continued fractions, one per polarisation.
    """
    alphas = np.asarray(alphas)                                              # (3, n_iter)
    betas = np.asarray(betas)
    norms = np.asarray(norms)
    z = omegas_Ry + 1j * eta_Ry
    pref = 16.0 * np.pi ** 2 / (V_cell * n_k * n_spin * n_spinor)
    n_pol, n_iter = alphas.shape
    eps2 = np.zeros((omegas_Ry.size, 3), dtype=np.float64)
    for a in range(n_pol):
        cf = np.zeros_like(z)
        for n in range(n_iter - 1, 0, -1):
            cf = (betas[a, n - 1] ** 2) / (z - alphas[a, n] - cf)
        g = 1.0 / (z - alphas[a, 0] - cf)
        eps2[:, a] = -pref * (norms[a] ** 2) * g.imag / np.pi
    return eps2


def jdos_from_dipole(d_alpha, de_cv, omegas_Ry, V_cell, n_k, eta_Ry, n_spin, n_spinor):
    """Independent-particle ε₂^0(ω) for the 4th column of absorption_*.dat."""
    pref = 16.0 * np.pi ** 2 / (V_cell * n_k * n_spin * n_spinor)
    de_flat = np.asarray(de_cv).flatten()
    f_alpha = (np.abs(np.asarray(d_alpha)) ** 2).astype(np.float64)
    jdos = np.zeros((omegas_Ry.size, 3), dtype=np.float64)
    for a in range(3):
        weights = f_alpha[a].flatten()
        delta = omegas_Ry[:, None] - de_flat[None, :]
        L = (eta_Ry / np.pi) / (delta * delta + eta_Ry * eta_Ry)
        jdos[:, a] = pref * (L @ weights)
    return jdos


def run_haydock(
    *,
    input_file: str,
    n_val: int,
    n_cond: int,
    n_occ: int,
    eqp_file: str | None,
    dipole_file: str,
    V_cell: float,
    n_iter: int,
    eta_eV: float,
    n_spin: int,
    n_spinor: int,
    omega_min_eV: float,
    omega_max_eV: float,
    n_omega: int,
    out_prefix: str,
    no_eps1: bool,
    matvec_kind: str = "ring",
):
    """End-to-end Haydock absorption: load BSE data, run 3-block recursion,
    evaluate continued fraction, write outputs."""
    restart_file = _find_restart_file(input_file)
    n_devices = jax.device_count()
    if n_devices < 2:
        raise RuntimeError(
            "Haydock route currently requires the sharded matvec (≥2 devices).")
    mesh_xy = create_mesh_2d()
    sh = make_bse_shardings(mesh_xy)
    grid_x, grid_y = mesh_xy.devices.shape

    print(f"[haydock] loading BSE data from {restart_file}")
    data = load_bse_data_from_restart_sharded(
        restart_file, n_val=n_val, n_cond=n_cond, mesh_xy=mesh_xy,
        input_file=input_file, n_occ=n_occ,
    )
    # Resolve n_occ for the downstream dipole slice — same source as the
    # loader used (WFN.h5 ifmax → largest-gap fallback).
    with h5py.File(restart_file, "r") as f:
        enk_full_np = np.asarray(f["enk_full"][:])
    from .bse_io import resolve_n_occ
    n_occ = resolve_n_occ(enk_full_np, n_occ=n_occ, input_file=input_file)
    if eqp_file is not None:
        from .bse_io import _pad_axis_to_multiple, apply_eqp_corrections
        enk_full_np = apply_eqp_corrections(enk_full_np, eqp_file, input_file=input_file)
        val_idx = np.arange(n_occ - n_val, n_occ)
        cond_idx = np.arange(n_occ, n_occ + n_cond)
        data["eps_v"] = jnp.asarray(enk_full_np[:, val_idx])
        data["eps_c"] = jnp.asarray(enk_full_np[:, cond_idx])
        data["eps_v"], _ = _pad_axis_to_multiple(data["eps_v"], axis=1, multiple=grid_y)
        data["eps_c"], _ = _pad_axis_to_multiple(data["eps_c"], axis=1, multiple=grid_x)

    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    nk = nkx * nky * nkz
    nc_pad = int(data["n_cond_pad"])
    nv_pad = int(data["n_val_pad"])
    print(f"[haydock] mesh {grid_x}x{grid_y}, problem "
          f"{nc_pad} cond × {nv_pad} val × {nk} k = {nc_pad * nv_pad * nk}")

    # Build the dipole seed vectors |d^α⟩ in BSE basis with proper padding.
    dipole_cart, deltaE, _ = load_dipole_h5(dipole_file)
    d_alpha, de_cv = slice_dipole_to_bse_window(dipole_cart, deltaE, n_occ, n_val, n_cond)
    print(f"[haydock] dipole sliced → (3, nk={nk}, nc={n_cond}, nv={n_val})")
    d_block_np = build_dipole_vector_bse(d_alpha, n_cond_pad=nc_pad, n_val_pad=nv_pad)
    d_block = jnp.asarray(d_block_np)                                         # (3, nc_pad, nv_pad, nk)

    # Build the BSE matvec (ring or simple), wrap to accept the (3, ...) block shape.
    if matvec_kind == "simple":
        from .bse_simple import build_bse_simple_matvec
        matvec_ring = build_bse_simple_matvec(mesh_xy, nkx, nky, nkz, include_W=True)
    else:
        matvec_ring = build_bse_ring_matvec(
            mesh_xy, nkx, nky, nkz, include_W=True,
            low_mem=(matvec_kind == "ring"))

    _W_local_ifftn = make_sharded_ifftn_3d(
        mesh_xy, sh.W.spec, sh.W.spec, axes=(2, 3, 4), norm="ortho")
    rep = NamedSharding(mesh_xy, P())

    @partial(
        jax.jit,
        in_shardings=(
            sh.X, sh.psi_x, sh.psi_y, sh.psi_x, sh.psi_y,
            sh.eps, sh.eps, sh.W, sh.V,
        ),
        out_shardings=(rep, rep, rep),
        donate_argnums=(7,),  # W_q
    )
    def _full_run(d_seed, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_q, V_q0):
        W_R = _W_local_ifftn(W_q)

        def matvec_block(X_block):
            X_block = jax.lax.with_sharding_constraint(X_block, sh.X)
            return matvec_ring(
                X_block, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y,
                eps_c, eps_v, W_R, V_q0,
            )

        return haydock_recursion_block(matvec_block, d_seed, n_iter)

    print(f"[haydock] running 3-block recurrence × {n_iter} iters …")
    alphas, betas, norms = _full_run(
        d_block,
        data["psi_c_X"], data["psi_c_Y"], data["psi_v_X"], data["psi_v_Y"],
        data["eps_c"], data["eps_v"], data["W_q"], data["V_q0"],
    )
    alphas = np.asarray(alphas)
    betas = np.asarray(betas)
    norms = np.asarray(norms)
    print(f"[haydock] α_1 = {alphas[:, 0]}  (Ry)")
    print(f"[haydock] β_1 = {betas[:, 0]}  (Ry)")
    print(f"[haydock] ‖d‖ = {norms}")

    omegas_eV = np.linspace(omega_min_eV, omega_max_eV, n_omega)
    omegas_Ry = omegas_eV / RYD2EV
    eta_Ry = eta_eV / RYD2EV

    eps2 = absorption_from_haydock(
        omegas_Ry, eta_Ry, alphas, betas, norms,
        V_cell, nk, n_spin, n_spinor,
    )
    jdos = jdos_from_dipole(
        d_alpha, de_cv, omegas_Ry, V_cell, nk, eta_Ry, n_spin, n_spinor,
    )
    if no_eps1:
        eps1 = np.ones_like(eps2)
    else:
        eps1 = np.stack(
            [kramers_kronig_eps1(omegas_Ry, eps2[:, a]) for a in range(3)], axis=1)

    suffixes = ["b1", "b2", "b3"]
    out_prefix_p = Path(out_prefix)
    for a, sfx in enumerate(suffixes):
        out_dat = out_prefix_p.with_name(f"absorption_haydock_{sfx}_eh.dat")
        header = [
            "BSE absorption — Haydock continued-fraction route",
            f"polarization: {sfx}",
            f"V_cell = {V_cell:.6f} bohr^3,  N_k = {nk}",
            f"eta = {eta_eV:.4f} eV,  n_spin = {n_spin},  n_spinor = {n_spinor}",
            f"n_iter = {n_iter},  ‖d‖ = {norms[a]:.6e}",
        ]
        write_absorption_dat(out_dat, omegas_eV, eps2[:, a], eps1[:, a], jdos[:, a], header)
        print(f"[haydock] wrote {out_dat}")

    out_h5 = out_prefix_p.with_name(f"{out_prefix_p.name}.h5")
    write_absorption_h5(
        out_h5,
        omegas_eV=omegas_eV,
        eps2_3pol=eps2, eps1_3pol=eps1, jdos_3pol=jdos,
        haydock_alphas=alphas, haydock_betas=betas, haydock_norms=norms,
        metadata={
            "route": "haydock",
            "V_cell_bohr3": V_cell,
            "N_k": nk,
            "eta_eV": eta_eV,
            "n_spin": n_spin,
            "n_spinor": n_spinor,
            "n_iter": n_iter,
        },
    )
    print(f"[haydock] wrote {out_h5}")


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Haydock-route BSE absorption (continued fraction; no eigenvectors)",
    )
    p.add_argument("-i", "--input", required=True,
                   help="cohsex.in (BSE input; restart bundle resolved from it)")
    p.add_argument("--n-val", type=int, required=True)
    p.add_argument("--n-cond", type=int, required=True)
    p.add_argument("--n-occ", type=int, required=True)
    p.add_argument("--eqp", default=None,
                   help="BGW eqp1.dat for QP corrections (optional)")
    p.add_argument("--dipole", default="dipole.h5",
                   help="dipole.h5 from psp.get_dipole_mtxels (default: dipole.h5)")
    p.add_argument("--V-cell", type=float, required=True,
                   help="Unit-cell volume (bohr^3)")
    p.add_argument("--n-iter", type=int, default=200,
                   help="Haydock iterations per polarisation (default 200)")
    p.add_argument("--n-spin", type=int, default=1,
                   help="Spin multiplicity (2 only for collinear spin-polarised)")
    p.add_argument("--n-spinor", type=int, default=2,
                   help="Spinor components (2 for spinor calcs, default 2)")
    p.add_argument("--eta-eV", type=float, default=0.1)
    p.add_argument("--omega-min-eV", type=float, default=0.0)
    p.add_argument("--omega-max-eV", type=float, default=15.0)
    p.add_argument("--n-omega", type=int, default=1500)
    p.add_argument("--out-prefix", default="absorption_haydock")
    p.add_argument("--no-eps1", action="store_true",
                   help="Skip Kramers-Kronig eps1 (faster)")
    p.add_argument("--matvec-kind", choices=("ring", "gather", "simple"), default="ring")
    args = p.parse_args(argv)

    run_haydock(
        input_file=args.input,
        n_val=args.n_val, n_cond=args.n_cond, n_occ=args.n_occ,
        eqp_file=args.eqp, dipole_file=args.dipole,
        V_cell=args.V_cell, n_iter=args.n_iter,
        eta_eV=args.eta_eV, n_spin=args.n_spin, n_spinor=args.n_spinor,
        omega_min_eV=args.omega_min_eV, omega_max_eV=args.omega_max_eV,
        n_omega=args.n_omega,
        out_prefix=args.out_prefix, no_eps1=args.no_eps1,
        matvec_kind=args.matvec_kind,
    )


if __name__ == "__main__":
    main()

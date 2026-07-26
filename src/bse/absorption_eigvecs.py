"""ε₂(ω) from explicit BSE eigenvectors A^S — diagonalisation route.

Mirrors BGW's ``BSE/diag.f90 → absp.f90`` pathway:

  d^α_{cvk}     = ⟨c|v̂_α|v⟩ / (E_c − E_v)              (r-form, Ry-bohr)
  ⟨0|r^α|S⟩    = Σ_{cvk} A^S_{cvk} · d^α_{cvk}
  f_S^α        = |⟨0|r^α|S⟩|²
  ε_2^α(ω)     = (16π² / (V_cell · N_k · n_spin · n_spinor))
                 × Σ_S f_S^α · L(ω − E_S, η)

Note on ``n_spin`` vs ``n_spinor``:
  In BGW (and our codebase) ``n_spin`` is 2 only for collinear spin-polarised
  calculations.  For spinor calculations (``bispinor = true``, our normal
  case) ``n_spin = 1`` and ``n_spinor = 2``.  The eigenvectors.h5 header
  carries ``spin_kernel == 3`` for spinor; this module infers ``n_spinor``
  from that.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from .absorption_common import (
    RYD2EV,
    kramers_kronig_eps1,
    load_dipole_h5,
    load_eigenvectors_h5,
    lorentzian_broaden,
    slice_dipole_to_bse_window,
    write_absorption_dat,
    write_absorption_h5,
    write_eigenvalues_dat,
)


def compute_dipole_projections(A, d_alpha):
    """⟨0|r̂_α|S⟩ = Σ_{cvk} A^S_{cvk} · d^α_{cvk}

    A       : (N, nk, nc, nv) complex
    d_alpha : (3, nk, nc, nv) complex

    Returns: (N, 3) complex — per-state, per-polarisation projection.
    """
    return np.einsum("Nkcv,akcv->Na", A, d_alpha, optimize=True).astype(np.complex128)


def compute_oscillator_strengths(A, d_alpha):
    """|⟨0|r̂_α|S⟩|².  Convenience wrapper around ``compute_dipole_projections``."""
    return (np.abs(compute_dipole_projections(A, d_alpha)) ** 2).astype(np.float64)


def compute_jdos_oscillators(d_alpha):
    """Non-interacting JDOS oscillator weights at each (cvk) transition.

    f_{cvk}^α = |d^α_{cvk}|².  Used with ΔE_cv as the transition-energy
    list to build the independent-particle JDOS column.
    """
    return (np.abs(d_alpha) ** 2).astype(np.float64)


def compute_eps2(
    eigvals_Ry, A, d_alpha, omegas_Ry, V_cell, eta_Ry, n_spin, n_spinor, n_k,
):
    """Returns eps2 (n_omega, 3), oscillator strengths (N, 3), and complex
    dipole projections ⟨0|r̂_α|S⟩ (N, 3) for the eigenvalues_b*.dat writer."""
    proj = compute_dipole_projections(A, d_alpha)              # (N, 3) complex
    f_Sa = (np.abs(proj) ** 2).astype(np.float64)              # (N, 3) real
    pref = 16.0 * np.pi ** 2 / (V_cell * n_k * n_spin * n_spinor)
    eps2 = np.zeros((omegas_Ry.size, 3), dtype=np.float64)
    for a in range(3):
        eps2[:, a] = pref * lorentzian_broaden(omegas_Ry, eigvals_Ry, f_Sa[:, a], eta_Ry)
    return eps2, f_Sa, proj


def compute_jdos(d_alpha, de_cv, omegas_Ry, V_cell, eta_Ry, n_spin, n_spinor, n_k):
    """Independent-particle ε₂^0(ω) (the "JDOS" column in BGW absorption_*.dat)."""
    f_alpha = compute_jdos_oscillators(d_alpha)                # (3, nk, nc, nv)
    pref = 16.0 * np.pi ** 2 / (V_cell * n_k * n_spin * n_spinor)
    de_flat = de_cv.flatten()
    jdos = np.zeros((omegas_Ry.size, 3), dtype=np.float64)
    for a in range(3):
        weights = f_alpha[a].flatten()
        jdos[:, a] = pref * lorentzian_broaden(omegas_Ry, de_flat, weights, eta_Ry)
    return jdos


def main(argv=None):
    p = argparse.ArgumentParser(allow_abbrev=False,
        description="BSE absorption from BGW-format eigenvectors.h5 + dipole.h5",
    )
    p.add_argument("--eigenvectors", default="eigenvectors.h5",
                   help="BGW-format eigenvectors.h5 (default: eigenvectors.h5)")
    p.add_argument("--dipole", default="dipole.h5",
                   help="dipole.h5 from psp.get_dipole_mtxels (default: dipole.h5)")
    p.add_argument("--n-occ", type=int, required=True,
                   help="Number of occupied bands (Fermi-band index)")
    p.add_argument("--V-cell", type=float, required=True,
                   help="Unit-cell volume (bohr^3)")
    p.add_argument("--n-spin", type=int, default=1,
                   help="Spin multiplicity (2 only for collinear spin-polarised; "
                        "1 for spinor / unpolarised; default 1)")
    p.add_argument("--n-spinor", type=int, default=None,
                   help="Spinor components (2 for spinor calcs; default: "
                        "infer from spin_kernel in eigenvectors.h5)")
    p.add_argument("--eta-eV", type=float, default=0.1,
                   help="Lorentzian half-width η in eV (default 0.1)")
    p.add_argument("--omega-min-eV", type=float, default=0.0)
    p.add_argument("--omega-max-eV", type=float, default=15.0)
    p.add_argument("--n-omega", type=int, default=1500)
    p.add_argument("--out-prefix", default="absorption_eigvecs",
                   help="Output filename prefix")
    p.add_argument("--no-eps1", action="store_true",
                   help="Skip Kramers-Kronig eps1 (faster; sets eps1=1)")
    args = p.parse_args(argv)

    eigvals_Ry, A, params = load_eigenvectors_h5(args.eigenvectors)
    dipole_cart, deltaE, attrs = load_dipole_h5(args.dipole)

    nc, nv = params["nc"], params["nv"]
    n_k = params["nk"]
    n_spinor = args.n_spinor if args.n_spinor is not None else params["nspinor"]
    print(f"[absorption_eigvecs] {eigvals_Ry.size} eigenvectors, "
          f"BSE window: nc={nc}, nv={nv}, nk={n_k}, "
          f"n_spin={args.n_spin}, n_spinor={n_spinor} "
          f"(spin_kernel={params['spin_kernel']})")

    d_alpha, de_cv = slice_dipole_to_bse_window(
        dipole_cart, deltaE, args.n_occ, nv, nc)
    print(f"[absorption_eigvecs] dipole sliced → (3, nk={d_alpha.shape[1]}, "
          f"nc={nc}, nv={nv})")

    omegas_eV = np.linspace(args.omega_min_eV, args.omega_max_eV, args.n_omega)
    omegas_Ry = omegas_eV / RYD2EV
    eta_Ry = args.eta_eV / RYD2EV

    eps2, f_Sa, proj = compute_eps2(
        eigvals_Ry, A, d_alpha, omegas_Ry,
        args.V_cell, eta_Ry, args.n_spin, n_spinor, n_k,
    )
    jdos = compute_jdos(
        d_alpha, de_cv, omegas_Ry,
        args.V_cell, eta_Ry, args.n_spin, n_spinor, n_k,
    )

    if args.no_eps1:
        eps1 = np.ones_like(eps2)
    else:
        eps1 = np.stack(
            [kramers_kronig_eps1(omegas_Ry, eps2[:, a]) for a in range(3)], axis=1)

    suffixes = ["b1", "b2", "b3"]
    out_prefix = Path(args.out_prefix)
    eigvals_eV = eigvals_Ry * RYD2EV
    vol_supercell = args.V_cell * n_k
    for a, sfx in enumerate(suffixes):
        out_abs = out_prefix.with_name(f"absorption_{sfx}_eh.dat")
        header = [
            "BSE absorption — eigenvector route",
            f"polarization: {sfx}",
            f"V_cell = {args.V_cell:.6f} bohr^3,  N_k = {n_k}",
            f"eta = {args.eta_eV:.4f} eV,  n_spin = {args.n_spin},  n_spinor = {n_spinor}",
            f"n_eigenvectors = {eigvals_Ry.size}",
        ]
        write_absorption_dat(out_abs, omegas_eV, eps2[:, a], eps1[:, a], jdos[:, a], header)
        print(f"[absorption_eigvecs] wrote {out_abs}")

        out_eig = out_prefix.with_name(f"eigenvalues_{sfx}.dat")
        write_eigenvalues_dat(
            out_eig, eigvals_eV, proj[:, a],
            n_spin=args.n_spin, n_spinor=n_spinor, vol_supercell=vol_supercell,
        )
        print(f"[absorption_eigvecs] wrote {out_eig}")

    out_h5 = out_prefix.with_name(f"{out_prefix.name}.h5")
    write_absorption_h5(
        out_h5,
        omegas_eV=omegas_eV,
        eps2_3pol=eps2, eps1_3pol=eps1, jdos_3pol=jdos,
        eigenvalues_Ry=eigvals_Ry,
        oscillator_strengths=f_Sa,
        metadata={
            "route": "eigenvector",
            "V_cell_bohr3": args.V_cell,
            "N_k": n_k,
            "eta_eV": args.eta_eV,
            "n_spin": args.n_spin,
            "n_spinor": n_spinor,
            "n_eigvecs": int(eigvals_Ry.size),
        },
    )
    print(f"[absorption_eigvecs] wrote {out_h5}")


if __name__ == "__main__":
    main()

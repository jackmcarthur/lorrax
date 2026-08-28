"""psp/kpm_dos.py — KPM density of states for the DFT Hamiltonian.

Computes the electronic DOS from the KS Hamiltonian using the Kernel
Polynomial Method.  Follows the same potential-setup pipeline as run_nscf,
then replaces Davidson with Chebyshev moment accumulation from solvers/.

Usage:
    lxrun python3 -u -m psp.kpm_dos --save qe/nscf/silicon.save --nk 4 4 4

The structure mirrors bse_kpm.py as closely as possible — the only
differences are the Hamiltonian construction (DFT vs BSE) and k-point
handling (explicit loop here vs ring communication in BSE).
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import argparse
import time

import numpy as np
import jax
import jax.numpy as jnp

from file_io import CrystalData
from psp.pseudos import load_pseudopotentials
from psp.ionic_gspace import build_ionic_and_core
from psp.dft_operators import build_G_cart, compute_V_H_and_V_xc, build_V_scf
from psp.dft_operators import HamiltonianK, setup_H_k_from_kvec
from psp.h_dft import make_apply_H
from psp.gvec_utils import build_master_gvec_list, compute_ngkmax
import psp.vnl_ops as vnl_ops

from solvers.chebyshev import (
    make_chebyshev_recurrence,
    chebyshev_moments,
    jackson_coefficients,
    reconstruct_dos,
)
import common.timing as timing

RY_TO_EV = 13.6056980659


# ═══════════════════════════════════════════════════════════════════════
#  DFT-specific callables for the Chebyshev solver
# ═══════════════════════════════════════════════════════════════════════

def make_dft_h_tilde(apply_H, e_center, half_width):
    """Return a @jax.jit'd rescaled DFT matvec: H_tilde x = (Hx - center*x) / half_width.

    Wraps the Davidson-style (batch, nspinor, dim) matvec to operate on
    a single vector (nspinor, dim), matching the Chebyshev solver interface.
    """
    inv_hw = jnp.asarray(1.0 / half_width, dtype=jnp.float64)
    center_jnp = jnp.asarray(e_center, dtype=jnp.float64)

    @jax.jit
    def apply_h_tilde(x):
        hx = apply_H(x[None, ...])[0]       # (nspinor, dim)
        return (hx - center_jnp * x) * inv_hw

    return apply_h_tilde


def make_dft_random_vector(nspinor, ngkmax, mask):
    """Return a callable (key,) -> x_random for the DFT Hilbert space.

    Generates real Rademacher ±1 vectors (cast to complex) with padding
    entries zeroed out.  Rademacher gives <r|r> = N exactly, matching the
    stochastic trace normalization convention in solvers.chebyshev.
    """
    mask_cplx = mask[None, :].astype(jnp.complex128)   # (1, ngkmax)

    def fn(key):
        x = (2.0 * jax.random.bernoulli(key, shape=(nspinor, ngkmax)).astype(jnp.float64)
             - 1.0)
        return x.astype(jnp.complex128) * mask_cplx

    return fn


# ═══════════════════════════════════════════════════════════════════════
#  Spectral bounds from the Hamiltonian diagonal
# ═══════════════════════════════════════════════════════════════════════

def estimate_spectral_bounds_from_diag(h_diags, buffer=0.10):
    """Estimate [E_min, E_max] from the diagonal approximations across k-points.

    Parameters
    ----------
    h_diags : list of (ngkmax,) arrays
        Hamiltonian diagonal at each k-point (padding entries are 1e10).
    buffer : float
        Fractional buffer added to both sides.

    Returns
    -------
    e_center, half_width : float
        Rescaling parameters in Ry.
    """
    e_min = min(float(jnp.min(jnp.where(h < 1e9, h, jnp.inf))) for h in h_diags)
    e_max = max(float(jnp.max(jnp.where(h < 1e9, h, -jnp.inf))) for h in h_diags)
    bandwidth = e_max - e_min
    e_min_buf = e_min - buffer * bandwidth
    e_max_buf = e_max + buffer * bandwidth
    e_center = 0.5 * (e_min_buf + e_max_buf)
    half_width = 0.5 * (e_max_buf - e_min_buf)
    return e_center, half_width


# ═══════════════════════════════════════════════════════════════════════
#  Main driver
# ═══════════════════════════════════════════════════════════════════════

def run_kpm_dos(
    crystal: CrystalData,
    pseudos: dict,
    kgrid: tuple[int, int, int],
    *,
    n_moments: int = 300,
    n_random: int = 3,
    seed: int = 0,
    buffer: float = 0.10,
    nosym: bool = False,
    soc: bool | None = None,
    plot_file: str = "dft_dos_kpm.pdf",
    verbose: bool = True,
) -> dict:
    """Compute and plot the DFT density of states via KPM.

    Parameters
    ----------
    crystal : CrystalData
    pseudos : dict
    kgrid : (nk1, nk2, nk3)
    n_moments : int
        Number of Chebyshev moments M.
    n_random : int
        Stochastic trace vectors per k-point.
    seed : int
        Random seed.
    buffer : float
        Fractional buffer on spectral bounds.
    nosym : bool
        If True, use full (unreduced) k-grid.
    soc : bool or None
        j-RESOLVED (True) vs j-AVERAGED (False) V_NL from an FR pseudo.
        ``None`` = not declared: ``crystal.spinorbit`` (QE <spinorbit>,
        always present on the .save route) resolves it authoritatively.
    plot_file : str
        Output plot path.
    verbose : bool

    Returns
    -------
    dict with keys: mu_damped, E_grid_eV, rho, e_center_ry, half_width_ry
    """
    timing.reset()
    truncation_2d = crystal.assume_isolated == "2D"
    fft_grid = crystal.fft_grid
    nspinor = crystal.nspinor
    ecutwfc = float(crystal.ecutwfc)

    # ── k-independent potentials (same as run_nscf) ────────────────
    with timing.section("kpm_dos.potentials"):
        V_loc, rho_core, rho_core_G = build_ionic_and_core(
            crystal, pseudos, fft_grid, truncation_2d=truncation_2d)
        rho_val = jnp.asarray(crystal.load_charge_density()[0], dtype=jnp.float64)
        nx, ny, nz = int(fft_grid[0]), int(fft_grid[1]), int(fft_grid[2])
        G_cart = build_G_cart(nx, ny, nz,
                              float(crystal.blat) * np.asarray(crystal.bvec, dtype=float))
        V_H, V_xc = compute_V_H_and_V_xc(
            rho_val, rho_core, rho_core_G, G_cart,
            jnp.asarray(crystal.bdot, dtype=jnp.float64),
            jnp.asarray(crystal.bvec, dtype=jnp.float64), crystal.blat,
            truncation_2d=truncation_2d)
        V_scf = build_V_scf(V_loc, V_H, V_xc)
        vnl_setup = vnl_ops.build_vnl_setup(
            crystal, pseudos=pseudos, nspinor=nspinor, soc=soc,
            q_max=float(np.sqrt(ecutwfc)) * 1.01)

    # ── k-point grid ───────────────────────────────────────────────
    kpoints, weights = crystal.build_kgrid(
        nk=kgrid, nosym=nosym, noinv=nosym, no_t_rev=nosym)
    kpoints = np.asarray(kpoints, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    nk = len(kpoints)

    G_master, _ = build_master_gvec_list(crystal)
    bdot = np.asarray(crystal.bdot, dtype=float)
    ngkmax = compute_ngkmax(kpoints, bdot, ecutwfc, fft_grid)
    grid_label = "full" if nosym else "IBZ"

    # ── Build all H_k and estimate spectral bounds ─────────────────
    H_ks = []
    for ik in range(nk):
        H_k = setup_H_k_from_kvec(kpoints[ik], V_scf, vnl_setup, crystal, None,
                                    V_loc_r=V_loc, ngkmax=ngkmax)
        H_ks.append(H_k)

    nG_min = min(H_k.nG for H_k in H_ks)
    nG_max = max(H_k.nG for H_k in H_ks)
    e_center, half_width = estimate_spectral_bounds_from_diag(
        [H_k.h_diag for H_k in H_ks], buffer=buffer)
    e_min_eV = (e_center - half_width) * RY_TO_EV
    e_max_eV = (e_center + half_width) * RY_TO_EV
    resolution_eV = 2 * half_width * RY_TO_EV / n_moments

    if verbose:
        print("=" * 60)
        print("KPM DENSITY OF STATES — H_DFT")
        print("=" * 60)
        print(f"  ecutwfc        {ecutwfc:.1f} Ry  ({ecutwfc*RY_TO_EV:.1f} eV)")
        print(f"  FFT grid       {nx} x {ny} x {nz}")
        print(f"  k-grid         {'x'.join(str(k) for k in kgrid)}  "
              f"({nk} {grid_label})")
        print(f"  nspinor        {nspinor}")
        print(f"  ngkmax         {ngkmax}  (nG range: {nG_min}–{nG_max})")
        print(f"  Chebyshev M    {n_moments}")
        print(f"  Random vectors {n_random} per k-point")
        print(f"  Total matvecs  {nk * n_random * n_moments}")
        print(f"  Spectrum       [{e_min_eV:.1f}, {e_max_eV:.1f}] eV  "
              f"(bandwidth {2*half_width*RY_TO_EV:.1f} eV)")
        print(f"  Resolution     ~{resolution_eV:.2f} eV/moment")
        print("-" * 60)

    # ── JIT warmup: compile the recurrence at the first k-point ────
    with timing.section("kpm_dos.jit_warmup"):
        apply_H_0 = make_apply_H(H_ks[0])
        h_tilde_0 = make_dft_h_tilde(apply_H_0, e_center, half_width)
        recurrence_warmup = make_chebyshev_recurrence(h_tilde_0, n_moments)
        x_dummy = jnp.zeros((nspinor, ngkmax), dtype=jnp.complex128)
        _ = recurrence_warmup(x_dummy)

    # ── per-k: Chebyshev moments ──────────────────────────────────
    mu_total = np.zeros(n_moments + 1)

    with timing.section("kpm_dos.moments"):
        for ik in range(nk):
            tk = time.perf_counter()
            H_k = H_ks[ik]
            dim_k = nspinor * H_k.nG    # physical dimension (no padding)
            apply_H = make_apply_H(H_k)
            apply_h_tilde = make_dft_h_tilde(apply_H, e_center, half_width)
            random_vector_fn = make_dft_random_vector(nspinor, ngkmax, H_k.mask)

            mu_k = chebyshev_moments(
                apply_h_tilde, dim_k, n_moments, n_random,
                seed=seed + ik,
                make_random_vector=random_vector_fn,
                verbose=False,
            )

            mu_total += weights[ik] * mu_k

            if verbose and (ik < 3 or ik == nk - 1 or (ik + 1) % 16 == 0):
                print(f"  k={ik:3d}/{nk}: {time.perf_counter()-tk:.2f}s  "
                      f"nG={H_k.nG}  mu_0={mu_k[0]:.4f}")

    # ── Jackson damping + DOS reconstruction ───────────────────────
    with timing.section("kpm_dos.reconstruct"):
        sigma = jackson_coefficients(n_moments)
        mu_damped = mu_total * sigma

        e_center_eV = e_center * RY_TO_EV
        half_width_eV = half_width * RY_TO_EV
        E_grid = np.linspace(e_center_eV - 0.95 * half_width_eV,
                             e_center_eV + 0.95 * half_width_eV, 2000)
        rho = reconstruct_dos(mu_damped, E_grid, e_center_eV, half_width_eV)
        rho = np.maximum(rho, 0.0)

    # ── Plot ───────────────────────────────────────────────────────
    _plot_dos(E_grid, rho, n_moments, n_random, nk, plot_file)

    if verbose:
        print("-" * 60)
        timing.report(print_fn=print, title="--- KPM DOS Timing ---")
        print(f"  → {plot_file}")

    return {
        "mu_raw": mu_total,
        "mu_damped": mu_damped,
        "E_grid_eV": E_grid,
        "rho": rho,
        "e_center_ry": e_center,
        "half_width_ry": half_width,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════

def _plot_dos(E_grid, rho, n_moments, n_random, nk, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.5))

    ax.fill_between(E_grid, rho, alpha=0.25, color="steelblue")
    ax.plot(E_grid, rho, color="steelblue", linewidth=0.8)

    ax.set_xlabel("Energy (eV)", fontsize=12)
    ax.set_ylabel("DOS (states / eV)", fontsize=12)
    ax.set_xlim(E_grid[0], E_grid[-1])
    ax.set_ylim(bottom=0)

    ax.tick_params(axis="both", labelsize=10, direction="in",
                   top=True, right=True)
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=True, right=True)

    ax.annotate(f"KPM  M={n_moments}  R={n_random}  nk={nk}",
                xy=(0.97, 0.93), xycoords="axes fraction",
                ha="right", va="top", fontsize=8, color="0.4")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(allow_abbrev=False, description="KPM density of states for H_DFT")
    parser.add_argument("--save", required=True, help="QE .save directory")
    parser.add_argument("--pseudo_dir", default=None)
    parser.add_argument("--nk", type=int, nargs=3, default=[4, 4, 4])
    parser.add_argument("--nosym", action="store_true")
    parser.add_argument("--soc", choices=("auto", "true", "false"),
                        default="auto",
                        help="V_NL from an FR pseudo: j-resolved (true) / "
                             "j-averaged average_pp (false) / read QE's "
                             "<spinorbit> (auto).  Same contract as "
                             "gw.kin_ion_io --soc.")
    parser.add_argument("--n-moments", type=int, default=300)
    parser.add_argument("--n-random", type=int, default=3)
    parser.add_argument("--buffer", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", default="dft_dos_kpm.pdf")
    args = parser.parse_args()

    crystal = CrystalData.from_qe_save(args.save)
    pseudos = load_pseudopotentials(args.pseudo_dir or args.save)

    run_kpm_dos(
        crystal, pseudos,
        kgrid=tuple(args.nk),
        n_moments=args.n_moments,
        n_random=args.n_random,
        buffer=args.buffer,
        seed=args.seed,
        nosym=args.nosym,
        soc={"auto": None, "true": True, "false": False}[args.soc],
        plot_file=args.plot,
    )


if __name__ == "__main__":
    raise SystemExit(main())

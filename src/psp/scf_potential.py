"""psp/scf_potential.py — Build the DFT self-consistent potential V_scf.

Two public entry points, one helper:

- ``build_dft_potentials(mf, pseudos, rho_val, *, truncation_2d, verbose)``:
    Assemble (V_scf, V_loc, vnl_setup) from a duck-typed mean-field
    information object ``mf`` (either ``CrystalData`` or ``WFNReader`` —
    both expose ``bvec``, ``bdot``, ``blat``, ``fft_grid``, ``nspinor``,
    ``ecutwfc``, ``atom_types``, ``atom_positions``, ``cell_volume``)
    plus an explicit real-space valence density.

    V_scf = V_loc + V_H[ρ_val + ρ_core] + V_xc[ρ_val + ρ_core]

- ``build_rho_val_from_wfn(wfn, sym, meta, n_occ)``:
    Build ρ_val(r) on the FFT grid by streaming over the **full BZ** and
    accumulating Σ_k Σ_{v<n_occ} |ψ_{v,k}(r)|².  No symmetrisation is
    required because the full-BZ sum is already invariant under the
    crystal point group.  Cheaper and more robust than the IBZ +
    symmetrise approach (which ``tests/bench/charge_density.py`` does;
    ``_symmetrise_density`` there is known broken — see psp/dev_status.md).

Lifted from ``psp/run_nscf._build_potentials`` so both the NSCF driver and
the Sternheimer driver can share the pipeline without CrystalData
coupling in the Sternheimer path.
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from psp.ionic_gspace import build_ionic_and_core
from psp.dft_operators import build_G_cart, compute_V_H_and_V_xc, build_V_scf
import psp.vnl_ops as vnl_ops


# ═══════════════════════════════════════════════════════════════════════
#  V_scf builder (lift-out of run_nscf._build_potentials)
# ═══════════════════════════════════════════════════════════════════════

def build_dft_potentials(
    mf,
    pseudos: dict,
    rho_val: jax.Array,
    *,
    truncation_2d: bool,
    verbose: bool = True,
) -> tuple[jax.Array, jax.Array, vnl_ops.VNLSetup]:
    """Build (V_scf, V_loc, vnl_setup) on the FFT grid.

    Parameters
    ----------
    mf : WFNReader or CrystalData (duck-typed)
        Source of structural data: ``fft_grid``, ``nspinor``, ``bvec``,
        ``bdot``, ``blat``, ``ecutwfc``, ``atom_types``, ``atom_positions``,
        ``cell_volume``.
    pseudos : dict
        ``{symbol: Pseudopotential}`` from ``psp.pseudos.load_pseudopotentials``.
    rho_val : (nx, ny, nz) float64
        Real-space valence charge density in e/bohr³.
    truncation_2d : bool
        Apply 2D Coulomb truncation in V_H (slab geometries).
    verbose : bool
        Print timing.

    Returns
    -------
    V_scf : (nx, ny, nz) float64
        Combined local potential V_loc + V_H + V_xc.
    V_loc : (nx, ny, nz) float64
        Ionic local potential alone (kept separately for h_diag).
    vnl_setup : vnl_ops.VNLSetup
        Nonlocal-projector setup (radial tables + channel data).
    """
    fft_grid = mf.fft_grid
    nspinor = int(mf.nspinor)
    t0 = time.perf_counter()

    V_loc, rho_core, rho_core_G = build_ionic_and_core(
        mf, pseudos, fft_grid, truncation_2d=truncation_2d)
    rho_val = jnp.asarray(rho_val, dtype=jnp.float64)
    nx, ny, nz = int(fft_grid[0]), int(fft_grid[1]), int(fft_grid[2])
    G_cart = build_G_cart(nx, ny, nz,
                          float(mf.blat) * np.asarray(mf.bvec, dtype=float))
    V_H, V_xc = compute_V_H_and_V_xc(
        rho_val, rho_core, rho_core_G, G_cart,
        jnp.asarray(mf.bdot, dtype=jnp.float64),
        jnp.asarray(mf.bvec, dtype=jnp.float64), mf.blat,
        truncation_2d=truncation_2d)
    V_scf = build_V_scf(V_loc, V_H, V_xc)
    vnl_setup = vnl_ops.build_vnl_setup(
        mf, pseudos=pseudos, nspinor=nspinor,
        q_max=float(np.sqrt(float(mf.ecutwfc))) * 1.01)

    if verbose:
        print(f"  Potentials: {time.perf_counter()-t0:.2f}s")

    return V_scf, V_loc, vnl_setup


# ═══════════════════════════════════════════════════════════════════════
#  ρ_val from WFN.h5 (full-BZ sum, no symmetrisation)
# ═══════════════════════════════════════════════════════════════════════

def build_rho_val_from_wfn(wfn, sym, meta, n_occ: int, *, verbose: bool = True) -> jax.Array:
    """Build ρ_val(r) from a WFN.h5 via full-BZ sum over occupied bands.

        ρ_val(r) = (1/N_k) · spin_factor · Σ_{k∈BZ} Σ_{v<n_occ} Σ_s |ψ_{v,k,s}(r)|²

    spin_factor = 2 for nspinor=1 (scalar), 1 for nspinor=2 (FR).  The 1/N_k
    factor comes from the uniform full-BZ weight (k-weights all 1/N_k once
    symmetry-related k are unfolded).

    Unlike the IBZ path in ``tests/bench/charge_density.py::build_density_from_ibz``,
    the full-BZ sum is exactly invariant under the crystal point group without
    an explicit ρ(G) star-averaging pass, so the broken ``_symmetrise_density``
    helper is sidestepped.

    Parameters
    ----------
    wfn : file_io.WFNReader
    sym : common.symmetry_maps.SymMaps
    meta : common.Meta
    n_occ : int
        Number of occupied bands per k (insulator).  For FR/nspinor=2 this
        is ``wfn.nelec``; for scalar/nspinor=1 this is ``wfn.nelec // 2``.
    verbose : bool
        Print density integral (should equal total electron count within
        FFT-grid discretisation error).

    Returns
    -------
    rho_r : (nx, ny, nz) float64 — ρ_val in e/bohr³.
    """
    from common.wfn_transforms import load_kpoint_fftbox

    nx, ny, nz = meta.fft_grid
    N_grid = nx * ny * nz
    vol = float(wfn.cell_volume)
    nk_full = int(sym.nk_tot)
    nspinor = int(meta.nspinor)
    spin_factor = 2 if nspinor == 1 else 1

    t0 = time.perf_counter()
    rho_r = jnp.zeros((nx, ny, nz), dtype=jnp.float64)
    for ik in range(nk_full):
        # load_kpoint_fftbox takes an UNFOLDED index and applies symmetry
        # rotation + fractional-translation phase + spinor rotation internally.
        psi_box = load_kpoint_fftbox(wfn, sym, meta, ik, n_occ)
        # Ortho IFFT to real space: <ψ|ψ> = Σ_j|ψ(r_j)|² = 1 per band.
        psi_r = jnp.fft.ifftn(psi_box, axes=(-3, -2, -1), norm='ortho')
        rho_r = rho_r + jnp.sum(jnp.abs(psi_r) ** 2, axis=(0, 1))

    # Average over full BZ (uniform weights) and convert |ψ|² normalisation to
    # continuous ρ in e/bohr³ via the N_grid/vol factor (cf. charge_density.py).
    rho_r = rho_r * (spin_factor / nk_full) * (N_grid / vol)

    if verbose:
        integral = float(jnp.sum(rho_r)) * vol / N_grid
        dt = time.perf_counter() - t0
        print(f"  ρ_val integral: {integral:.4f} e  "
              f"(expected {spin_factor * n_occ}, {dt:.2f}s)")

    return rho_r


__all__ = ["build_dft_potentials", "build_rho_val_from_wfn"]

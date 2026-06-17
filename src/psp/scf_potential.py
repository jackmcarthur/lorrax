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
    symmetrise approach (which ``psp/archive/charge_density.py`` does;
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
    m_vec: tuple | None = None,
    verbose: bool = True,
) -> tuple[jax.Array, jax.Array, vnl_ops.VNLSetup, jax.Array | None]:
    """Build (V_scf, V_loc, vnl_setup, B_vec) on the FFT grid.

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
    m_vec : (m_x, m_y, m_z) of (nx,ny,nz) float64, or None
        Real-space magnetization (from ``build_magnetization_from_wfn``).  When
        given, V_xc is the spin-polarized noncollinear potential: its charge
        part V̄=½(V↑+V↓) goes into V_scf and the xc field B=½(V↑−V↓)·m̂ is
        returned as ``B_vec`` for ``(B·σ)|ψ⟩`` in the Hamiltonian.  When None
        (non-magnetic), the spin-unpolarized V_xc is used and B_vec is None —
        bit-identical to the scalar path.
    verbose : bool
        Print timing.

    Returns
    -------
    V_scf : (nx, ny, nz) float64 — V_loc + V_H + V_xc (charge part).
    V_loc : (nx, ny, nz) float64 — ionic local potential alone (for h_diag).
    vnl_setup : vnl_ops.VNLSetup — nonlocal-projector setup.
    B_vec : (3, nx, ny, nz) float64 Ry xc field, or None if non-magnetic.
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

    B_vec = None
    if m_vec is not None:
        # Noncollinear V_xc.  Local spin frame n↑/↓=½(n±segni|m|), V̄=½(V↑+V↓) into
        # V_scf, B=segni·½(V↑−V↓)·m̂ the xc field.  segni=sign(m·û) (û=net-moment
        # axis) gives the SIGNED magnetization, which is smooth — using |m| directly
        # puts gradient kinks where m flips sign (core polarization), and the FFT
        # gradient rings there → corrupts the GGA σ → wrong B on those states (QE's
        # segni, compute_rho.f90).  ponytail: û from the net moment assumes a
        # dominant collinear axis — exact for FM/collinear (CrI3); revisit for AFM.
        from psp.xc import compute_V_xc_spin
        m_x, m_y, m_z = (jnp.asarray(c, dtype=jnp.float64) for c in m_vec)
        amag = jnp.sqrt(m_x ** 2 + m_y ** 2 + m_z ** 2)
        M_net = jnp.array([m_x.sum(), m_y.sum(), m_z.sum()])
        u_ax = M_net / (jnp.linalg.norm(M_net) + 1e-30)
        segni = jnp.where(amag > 1e-12,
                          jnp.sign(m_x * u_ax[0] + m_y * u_ax[1] + m_z * u_ax[2]), 1.0)
        n_tot = rho_val + rho_core
        n_up, n_dn = (n_tot + segni * amag) / 2, (n_tot - segni * amag) / 2
        # analytic-core gradient (precise-core trick, as the scalar path uses)
        core_grid = jnp.real(jnp.fft.ifftn(rho_core_G))
        rho_G_up = jnp.fft.fftn(n_up - core_grid / 2) + rho_core_G / 2
        rho_G_dn = jnp.fft.fftn(n_dn - core_grid / 2) + rho_core_G / 2
        V_up, V_dn = compute_V_xc_spin(n_up, n_dn, rho_G_up, rho_G_dn, G_cart)
        V_xc = 0.5 * (V_up + V_dn)
        Bmag = 0.5 * segni * (V_up - V_dn)
        inv = jnp.where(amag > 1e-12, 1.0 / (amag + 1e-30), 0.0)
        B_vec = jnp.stack([Bmag * m_x * inv, Bmag * m_y * inv, Bmag * m_z * inv], axis=0)

    V_scf = build_V_scf(V_loc, V_H, V_xc)
    vnl_setup = vnl_ops.build_vnl_setup(
        mf, pseudos=pseudos, nspinor=nspinor,
        q_max=float(np.sqrt(float(mf.ecutwfc))) * 1.01)

    if verbose:
        print(f"  Potentials: {time.perf_counter()-t0:.2f}s")

    return V_scf, V_loc, vnl_setup, B_vec


# ═══════════════════════════════════════════════════════════════════════
#  ρ_val from WFN.h5 (full-BZ sum, no symmetrisation)
# ═══════════════════════════════════════════════════════════════════════

def build_rho_val_from_wfn(wfn, sym, meta, n_occ: int, *, verbose: bool = True) -> jax.Array:
    """Build ρ_val(r) from a WFN.h5 via full-BZ sum over occupied bands.

        ρ_val(r) = (1/N_k) · spin_factor · Σ_{k∈BZ} Σ_{v<n_occ} Σ_s |ψ_{v,k,s}(r)|²

    spin_factor = 2 for nspinor=1 (scalar), 1 for nspinor=2 (FR).  The 1/N_k
    factor comes from the uniform full-BZ weight (k-weights all 1/N_k once
    symmetry-related k are unfolded).

    Unlike the IBZ path in ``psp/archive/charge_density.build_density_from_ibz``,
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


def build_magnetization_from_wfn(wfn, sym, meta, n_occ: int, *, verbose: bool = True):
    """Build m(r)=(m_x,m_y,m_z) from WFN.h5 via full-BZ Pauli sandwiches.

        m_α(r) = (1/N_k) Σ_{k∈BZ} Σ_{v<n_occ} ψ_v†(r) σ_α ψ_v(r)

    Mirror of ``build_rho_val_from_wfn`` (same load/IFFT/normalisation;
    spin_factor=1) with |ψ|² replaced by the three Pauli sandwiches.  Only
    defined for nspinor=2.  Returns three (nx,ny,nz) float64 arrays in e/bohr³
    (same units as ρ_val), matching QE charge-density.hdf5 m_x/m_y/m_z.
    """
    from common.load_wfns import load_kpoint_fftbox
    assert int(meta.nspinor) == 2, "magnetization needs nspinor=2 spinors"

    nx, ny, nz = meta.fft_grid
    N_grid = nx * ny * nz
    vol = float(wfn.cell_volume)
    nk_full = int(sym.nk_tot)

    t0 = time.perf_counter()
    mx = jnp.zeros((nx, ny, nz), dtype=jnp.float64)
    my = jnp.zeros_like(mx)
    mz = jnp.zeros_like(mx)
    for ik in range(nk_full):
        psi_box = load_kpoint_fftbox(wfn, sym, meta, ik, n_occ)
        psi_r = jnp.fft.ifftn(psi_box, axes=(-3, -2, -1), norm='ortho')
        up = psi_r[:, 0]
        dn = psi_r[:, 1]
        ud = jnp.sum(jnp.conj(up) * dn, axis=0)         # Σ_v conj(↑)·↓
        mx = mx + 2.0 * jnp.real(ud)                    # σ_x: 2 Re
        my = my + 2.0 * jnp.imag(ud)                    # σ_y: 2 Im
        mz = mz + jnp.sum(jnp.abs(up) ** 2 - jnp.abs(dn) ** 2, axis=0)  # σ_z

    f = (1.0 / nk_full) * (N_grid / vol)
    mx, my, mz = mx * f, my * f, mz * f

    if verbose:
        amag = jnp.sqrt(mx ** 2 + my ** 2 + mz ** 2)
        print(f"  |m| integral: {float(jnp.sum(amag)) * vol / N_grid:.4f} μ_B  "
              f"net m_z: {float(jnp.sum(mz)) * vol / N_grid:.4f}  "
              f"({time.perf_counter() - t0:.2f}s)")

    return mx, my, mz


__all__ = ["build_dft_potentials", "build_rho_val_from_wfn",
           "build_magnetization_from_wfn"]

"""
psp/dft_operators.py — Plane-wave DFT Hamiltonian: build, apply, differentiate.

Canonical module for all DFT Hamiltonian operations.  Other modules
(kin_ion_io_chunked, get_dipole_mtxels, davidson) should call these
functions rather than reimplementing operator construction.

Public API (Hamiltonian construction + application):
  HamiltonianK         — per-k data: T_diag, V_scf, G-indices, VNL Z/E, h_diag, mask
  setup_H_k            — build HamiltonianK from SymMaps path (GW pipeline)
  setup_H_k_from_kvec  — build HamiltonianK from k-vector (standalone/Davidson)
  apply_H_k            — fused JIT H|ψ⟩, psi_box donated; 2 ms on A100 for Si
  build_matrix_k       — full ⟨m|H|n⟩ matrix

Public API (per-component builders):
  build_T_diag          — |k+G|² + G-indices from SymMaps
  build_T_diag_from_kvec — same from explicit k-vector + ecutwfc (no SymMaps)
  build_V_scf           — combine V_loc + V_H + V_xc into one array
  compute_V_H_and_V_xc  — @jax.jit: V_H (Poisson) + V_xc (PBE GGA) in 1.2 ms cached
  build_h_diag          — preconditioner diagonal: T + V_loc(G=0) + V_NL_diag
  build_vnl_kdata       — dense VNL projectors (Z, E) from vnl_ops

Public API (velocity / dipole, autodiff through V_NL):
  vnl_matrix_at_k       — V_NL as pure function of k (jax.jacfwd-able)
  velocity_matrix_k     — dH/dk = 2(k+G) + dV_NL/dk
  compute_dipole_all    — batch velocity matrix elements for all k-points

V_scf = V_loc + V_H + V_xc is a single (nx,ny,nz) real-space potential.
The caller builds it from charge_density.py (V_xc, V_H) and
build_projectors_qe.py (V_loc), then passes it to setup_H_k*.

Normalization: ⟨m|O|n⟩ = Σ_{s,G} conj(ψ_m[s,G]) · (O ψ)_n[s,G].
No volume prefactors.  Ortho-FFT convention for local potentials:
(V ψ)_G = FFT_ortho(V(r) · IFFT_ortho(ψ_box))_G.

ngkmax padding: setup_H_k_from_kvec accepts ngkmax to pad all arrays
to uniform size.  Combined with warmup_jit() from davidson.py, this
ensures one JIT compilation serves all k-points.  Mask field on
HamiltonianK zeros padding in apply_H_k and build_matrix_k.
"""
from __future__ import annotations

import os
import functools
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import jax
import jax.numpy as jnp

import common.timing as timing


# ═══════════════════════════════════════════════════════════════════════
#  Data structures
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class HamiltonianK:
    """Everything needed to apply H at one k-point.

    Built by ``setup_H_k``, consumed by ``apply_H_k`` and ``build_matrix_k``.
    """
    # Kinetic: T|ψ⟩_G = T_diag[G] · ψ_G
    T_diag: jax.Array               # (nG,) float64 — |k+G|² in Ry

    # Local self-consistent potential: V_scf = V_loc + V_H + V_xc
    V_scf: jax.Array                # (nx, ny, nz) float64 — Ry

    # G-vector map: FFT-box indices for the valid G-vectors at this k
    Gx: jax.Array                   # (nG,) int32
    Gy: jax.Array                   # (nG,) int32
    Gz: jax.Array                   # (nG,) int32

    # Nonlocal: Kleinman–Bylander projectors (dense, all channels concatenated)
    vnl_Z: jax.Array                # (total_R, nG) complex128
    vnl_E: jax.Array                # (nspinor, nspinor, total_R, total_R) complex128

    # Preconditioner diagonal (for Davidson): h_diag = T + v_of_0 + V_NL_diag
    h_diag: jax.Array               # (nG_padded,) float64 — QE's g_psi convention

    # G-vector mask: True for valid, False for padding
    mask: jax.Array                  # (nG_padded,) bool

    nG: int                          # actual (unpadded) count
    fft_grid: tuple[int, int, int]


# ═══════════════════════════════════════════════════════════════════════
#  Poisson solver
# ═══════════════════════════════════════════════════════════════════════

def poisson_potential_from_rhoG(
    rho_G: jnp.ndarray,
    bdot: jnp.ndarray,
    bvec: jnp.ndarray | None = None,
    blat: float | None = None,
    truncation_2d: bool = True,
) -> jnp.ndarray:
    """Solve Poisson equation: V(G) = 8π ρ(G) / |G|² (G≠0), V(G=0) = 0.

    If truncation_2d=True, applies Ismail-Beigi 2D slab truncation:
        v_2D(G) = v_3D(G) × (1 − e^{−|G_xy|·L_z/2} · cos(G_z·L_z/2))
    """
    nx2, ny2, nz2 = rho_G.shape
    fx = jnp.fft.fftfreq(nx2) * nx2
    fy = jnp.fft.fftfreq(ny2) * ny2
    fz = jnp.fft.fftfreq(nz2) * nz2

    M = jnp.asarray(bdot, dtype=jnp.float64)
    ix, iy, iz = fx[:, None, None], fy[None, :, None], fz[None, None, :]
    G2 = (M[0, 0] * ix * ix + M[1, 1] * iy * iy + M[2, 2] * iz * iz
          + 2 * M[0, 1] * ix * iy + 2 * M[0, 2] * ix * iz + 2 * M[1, 2] * iy * iz)
    zero_mask = ((jnp.arange(nx2)[:, None, None] == 0)
                 & (jnp.arange(ny2)[None, :, None] == 0)
                 & (jnp.arange(nz2)[None, None, :] == 0))
    G2_safe = jnp.where(zero_mask, 1.0, G2)

    V_G = 8.0 * jnp.pi * rho_G / G2_safe

    if truncation_2d and bvec is not None and blat is not None:
        B = jnp.asarray(bvec, dtype=jnp.float64) * float(blat)
        Gx_c = ix * B[0, 0] + iy * B[1, 0] + iz * B[2, 0]
        Gy_c = ix * B[0, 1] + iy * B[1, 1] + iz * B[2, 1]
        Gz_c = ix * B[0, 2] + iy * B[1, 2] + iz * B[2, 2]
        zc = jnp.pi / B[2, 2]
        kxy = jnp.sqrt(Gx_c ** 2 + Gy_c ** 2)
        f2d = 1.0 - jnp.exp(-zc * kxy) * jnp.cos(Gz_c * zc)
        V_G = V_G * jnp.where(zero_mask, 0.0, f2d)

    V_G = V_G.at[0, 0, 0].set(0.0)
    return jnp.real(jnp.fft.ifftn(V_G, norm='ortho'))


def generate_gvectors_k(kpoint_idx, sym, wfn, meta):
    """G-vectors for one k-point via SymMaps (GW path).

    Returns (Gk_crys, kpoint_crys): (nG, 3) int and (3,) float.
    """
    kpoint_crys = jnp.asarray(sym.unfolded_kpts[kpoint_idx], dtype=jnp.float64)
    Gk_crys = sym.get_gvecs_kfull(wfn, kpoint_idx)
    return Gk_crys, kpoint_crys


# ═══════════════════════════════════════════════════════════════════════
#  G-vector utilities
# ═══════════════════════════════════════════════════════════════════════

def build_G_cart(nx: int, ny: int, nz: int, B: np.ndarray) -> jax.Array:
    """Cartesian G-vector grid from FFT frequencies and lattice matrix B.

    B = blat * bvec (rows = reciprocal lattice vectors in bohr⁻¹).
    Returns (nx, ny, nz, 3) float64.
    """
    gx = np.fft.fftfreq(nx, d=1.0 / nx).astype(int)
    gy = np.fft.fftfreq(ny, d=1.0 / ny).astype(int)
    gz = np.fft.fftfreq(nz, d=1.0 / nz).astype(int)
    Gx, Gy, Gz = np.meshgrid(gx, gy, gz, indexing='ij')
    G_crys = np.stack([Gx, Gy, Gz], axis=-1).astype(float)
    return jnp.asarray(np.einsum('...i,ij->...j', G_crys, B), dtype=jnp.float64)


# ═══════════════════════════════════════════════════════════════════════
#  Per-component builders
# ═══════════════════════════════════════════════════════════════════════
#
# Each function constructs one piece of the Hamiltonian.
# They are called by setup_H_k and also available individually.

def compute_ngkmax(kpoints, bdot, ecutwfc, fft_grid):
    """Maximum number of G-vectors across all k-points.

    Pure integer counting — no large arrays stored.  Use this once at
    startup so every ``setup_H_k_from_kvec`` call can pad to the same
    ``ngkmax``, giving one JIT compilation for all k-points.

    Parameters
    ----------
    kpoints : (nk, 3) — k-points in crystal coordinates
    bdot : (3,3) — reciprocal metric in bohr⁻²
    ecutwfc : float — PW cutoff (Ry)
    fft_grid : (nx, ny, nz)
    """
    kpoints = np.asarray(kpoints, dtype=np.float64)
    bdot = np.asarray(bdot, dtype=np.float64)
    nx, ny, nz = int(fft_grid[0]), int(fft_grid[1]), int(fft_grid[2])

    gx = np.fft.fftfreq(nx, d=1.0 / nx).astype(int)
    gy = np.fft.fftfreq(ny, d=1.0 / ny).astype(int)
    gz = np.fft.fftfreq(nz, d=1.0 / nz).astype(int)
    Gx, Gy, Gz = np.meshgrid(gx, gy, gz, indexing="ij")
    G_all = np.stack([Gx.ravel(), Gy.ravel(), Gz.ravel()], axis=-1).astype(np.float64)

    ngk_max = 0
    for ik in range(len(kpoints)):
        KG = G_all + kpoints[ik][None, :]
        KG_sq = np.einsum("gi,ij,gj->g", KG, bdot, KG)
        ngk_max = max(ngk_max, int(np.sum(KG_sq <= ecutwfc)))
    return ngk_max


def build_T_diag(
    k_idx: int,
    wfn,
    sym,
    meta,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Kinetic diagonal |k+G|² and G-vector FFT-box indices (SymMaps path).

    Returns (T_diag, Gx, Gy, Gz) where T_diag is (nG,) float64 in Ry
    and Gx, Gy, Gz are (nG,) int32 FFT-box indices.
    """
    kvec = np.asarray(sym.unfolded_kpts[k_idx], dtype=float)

    Gk_crys, _ = generate_gvectors_k(k_idx, sym, wfn, meta)
    return _T_diag_from_G(np.asarray(Gk_crys, dtype=int), kvec, wfn.bdot)


def build_T_diag_from_kvec(
    kvec: np.ndarray,
    crystal,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Kinetic diagonal |k+G|² and G-vector indices (standalone, no SymMaps).

    Generates the PW sphere {G : |k+G|² ≤ ecutwfc} from the cutoff and
    reciprocal metric.  Use this for self-consistent DFT (Davidson) where
    SymMaps / WFN.h5 are not available.

    Parameters
    ----------
    kvec : (3,) — k-point in crystal coordinates
    crystal : CrystalData or WFNReader — needs bdot, ecutwfc, fft_grid
    """
    kvec = np.asarray(kvec, dtype=float)
    bdot = np.asarray(crystal.bdot, dtype=float)
    nx, ny, nz = crystal.fft_grid

    gx = np.fft.fftfreq(nx, d=1.0 / nx).astype(int)
    gy = np.fft.fftfreq(ny, d=1.0 / ny).astype(int)
    gz = np.fft.fftfreq(nz, d=1.0 / nz).astype(int)
    Gx, Gy, Gz = np.meshgrid(gx, gy, gz, indexing="ij")
    G_all = np.stack([Gx.ravel(), Gy.ravel(), Gz.ravel()], axis=-1)

    KG = G_all.astype(float) + kvec[None, :]
    KG_sq = np.einsum("gi,ij,gj->g", KG, bdot, KG)
    mask = KG_sq <= crystal.ecutwfc
    Gk = G_all[mask]

    # Sort by |k+G|² (matches QE's convention at Gamma; convenient for
    # any k-point even though QE's ordering isn't strictly sorted for k≠0)
    order = np.argsort(KG_sq[mask])
    Gk = Gk[order]

    return _T_diag_from_G(Gk, kvec, bdot)


def _T_diag_from_G(
    Gk_int: np.ndarray,
    kvec: np.ndarray,
    bdot: np.ndarray,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Common core: G-vectors + k-vector + metric → (T_diag, Gx, Gy, Gz)."""
    Gx = jnp.asarray(Gk_int[:, 0], dtype=jnp.int32)
    Gy = jnp.asarray(Gk_int[:, 1], dtype=jnp.int32)
    Gz = jnp.asarray(Gk_int[:, 2], dtype=jnp.int32)

    bdot = np.asarray(bdot, dtype=float)
    K_crys = np.asarray(Gk_int, dtype=float) + np.asarray(kvec, dtype=float)[None, :]
    T_diag = jnp.asarray(
        np.einsum("gi,ij,gj->g", K_crys, bdot, K_crys),
        dtype=jnp.float64,
    )
    return T_diag, Gx, Gy, Gz


@functools.partial(jax.jit, static_argnames=("truncation_2d", "blat"))
def compute_V_H_and_V_xc(
    rho_val: jax.Array,
    rho_core: jax.Array,
    rhog_core: jax.Array,
    G_cart: jax.Array,
    bdot: jax.Array,
    bvec: jax.Array,
    blat: float,
    *,
    truncation_2d: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Compute V_H and V_xc in a single JIT.

    Compiles in ~1.5s on A100, cached calls ~2 ms.
    The autodiff uses grad(sum(ρ ε_xc)) which traces ONE scalar backward
    pass over the full grid (no vmap unrolling).

    Returns (V_H_r, V_xc_r) both (nx, ny, nz) in Ry.
    """
    from jax_xc_local.pbe import pbe_xc

    # ── V_H via Poisson ──
    rho_G_ortho = jnp.fft.fftn(rho_val, norm='ortho')
    V_H_r = jnp.real(poisson_potential_from_rhoG(
        rho_G_ortho, bdot, bvec, blat, truncation_2d=truncation_2d))

    # ── V_xc (PBE GGA) ──
    rho_total = rho_val + rho_core
    rho_safe = jnp.maximum(rho_total, 1e-10)

    # Gradient |∇ρ|² with precise G-space core charge
    rho_core_gridded = jnp.real(jnp.fft.ifftn(rhog_core))
    rho_G_total = jnp.fft.fftn(rho_total - rho_core_gridded) + rhog_core

    grad_rho_sq = jnp.zeros_like(rho_total)
    for i in range(3):
        drho = jnp.real(jnp.fft.ifftn(1j * G_cart[..., i] * rho_G_total))
        grad_rho_sq = grad_rho_sq + drho ** 2
    sigma = jnp.maximum(grad_rho_sq, 0.0)

    # PBE functional derivatives: grad(sum(ρ ε_xc)) w.r.t. ρ and σ arrays
    def E_xc_lda(rho):
        return jnp.sum(rho * pbe_xc(rho, jnp.zeros_like(rho)))
    def E_xc_full(rho, sig):
        return jnp.sum(rho * pbe_xc(rho, sig))

    df_drho_lda = jax.grad(E_xc_lda)(rho_safe)
    df_drho_full = jax.grad(E_xc_full, argnums=0)(rho_safe, sigma)
    df_dsigma = jax.grad(E_xc_full, argnums=1)(rho_safe, sigma)

    gga_mask = (rho_total > 1e-6) & (grad_rho_sq > 1e-10)
    df_drho = df_drho_lda + jnp.where(gga_mask, df_drho_full - df_drho_lda, 0.0)
    df_dsigma = jnp.where(gga_mask, df_dsigma, 0.0)

    # GGA divergence: -2 ∇·[df/dσ ∇ρ]
    gga_corr = jnp.zeros_like(rho_total)
    for i in range(3):
        drho_ri = jnp.real(jnp.fft.ifftn(1j * G_cart[..., i] * rho_G_total))
        h_i_G = jnp.fft.fftn(df_dsigma * drho_ri)
        gga_corr = gga_corr + jnp.real(
            jnp.fft.ifftn(1j * G_cart[..., i] * h_i_G))

    V_xc_r = df_drho - 2.0 * gga_corr
    return V_H_r, V_xc_r


def build_V_scf(
    V_loc_r: jax.Array,
    V_H_r: jax.Array | None = None,
    V_xc_r: jax.Array | None = None,
) -> jax.Array:
    """Combine the local potentials into V_scf = V_loc + V_H + V_xc.

    Each argument is (nx, ny, nz) in Ry on the FFT grid, or None to omit.

    Components
    ----------
    V_loc : ionic pseudopotential (local part), from build_local_ionic_potential_on_G_total
    V_H   : Hartree potential from valence density, from poisson_potential_from_rhoG
    V_xc  : exchange-correlation from ρ_val + ρ_core, from charge_density.build_V_xc
    """
    V_scf = jnp.asarray(V_loc_r, dtype=jnp.float64)
    if V_H_r is not None:
        V_scf = V_scf + jnp.asarray(V_H_r, dtype=jnp.float64)
    if V_xc_r is not None:
        V_scf = V_scf + jnp.asarray(V_xc_r, dtype=jnp.float64)
    return V_scf


def build_vnl_kdata(
    k_idx: int,
    vnl_setup,
    wfn,
    sym,
    meta,
    nspinor: int | None = None,
):
    """Dense VNL projectors (Z, E) for one k-point via vnl_ops.

    Returns (vnl_Z, vnl_E) where:
      vnl_Z : (total_R, nG) — all channels × atoms × betas concatenated
      vnl_E : (nspinor, nspinor, total_R, total_R) — block-diagonal D matrix
    """
    import psp.vnl_ops as vnl_ops

    if nspinor is None:
        nspinor = int(meta.nspinor)

    kdata = vnl_ops.build_vnl_kdata(k_idx, vnl_setup, wfn, sym, meta)
    return kdata.Z, kdata.E_super


@jax.jit
def gather_psi_G(psi_box, Gx, Gy, Gz, mask=None):
    """Gather sparse plane-wave coefficients from the FFT box.

    Parameters
    ----------
    psi_box : (nb, nspinor, nx, ny, nz) complex128
    Gx, Gy, Gz : (nG,) int32 FFT-box indices
    mask : (nG,) bool or float, optional

    Returns
    -------
    psi_G : (nb, nspinor, nG) complex128
    """
    psi_G = psi_box[:, :, Gx, Gy, Gz]
    if mask is None:
        return psi_G
    mask_f = jnp.asarray(mask, dtype=psi_G.dtype)[None, None, :]
    return psi_G * mask_f


def gather_psi_G_from_crys(psi_box, Gk_crys, mask=None):
    """Convenience wrapper around ``gather_psi_G`` for an integer G-list."""
    G_int = jnp.asarray(Gk_crys, dtype=jnp.int32)
    return gather_psi_G(psi_box, G_int[:, 0], G_int[:, 1], G_int[:, 2], mask)


def vnl_matrix_from_kdata(psi_box, Gk_crys, kdata, mask=None):
    """Convenience: <m|V_NL|n> from FFT-box states and prebuilt VNL k-data."""
    import psp.vnl_ops as vnl_ops

    psi_G = gather_psi_G_from_crys(psi_box, Gk_crys, mask)
    return vnl_ops.vnl_matrix(psi_G, kdata.Z, kdata.E_super)


def vnl_velocity_from_kdata(psi_box, Gk_crys, kdata, mask=None):
    """Convenience: dV_NL/dK_cart from FFT-box states and prebuilt VNL k-data."""
    import psp.vnl_ops as vnl_ops

    if kdata.dZ is None:
        raise ValueError("kdata.dZ is required for vnl_velocity_from_kdata")
    psi_G = gather_psi_G_from_crys(psi_box, Gk_crys, mask)
    return vnl_ops.vnl_velocity_matrix(psi_G, kdata.Z, kdata.dZ, kdata.E_super)


def build_h_diag(
    T_diag: jax.Array,
    V_loc_r: jax.Array,
    vnl_Z: jax.Array,
    vnl_E: jax.Array,
) -> jax.Array:
    """Hamiltonian diagonal for the Davidson preconditioner (QE convention).

    h_diag(G) = |k+G|² + V_loc(G=0) + V_NL_diag(G)

    - V_loc(G=0) = mean(V_loc_r): the G=0 Fourier component of the local
      ionic potential.  This is the `v_of_0` in QE's `setlocal.f90`.
      V_H and V_xc are NOT included (V_H(G=0)=0 by convention;
      V_xc(G=0) is small and not included in QE's h_diag).

    - V_NL_diag(G) = Σ_R |Z(R,G)|² E(R,R): the diagonal of the KB
      nonlocal potential in the plane-wave basis.

    Parameters
    ----------
    T_diag : (nG,) — |k+G|²
    V_loc_r : (nx, ny, nz) — ionic local potential (NOT V_scf)
    vnl_Z : (total_R, nG) — KB projectors
    vnl_E : (nspinor, nspinor, total_R, total_R) — D-matrix
    """
    v_of_0 = jnp.mean(V_loc_r)

    # V_NL diagonal: sum over spinor channels and projectors
    # ⟨G,s|V_NL|G,s⟩ = Σ_{R,Q} conj(Z[R,G]) E[s,s,R,Q] Z[Q,G]
    # For the diagonal, sum over s:
    nspinor = vnl_E.shape[0]
    vnl_diag = jnp.zeros(T_diag.shape[0], dtype=jnp.float64)
    for s in range(nspinor):
        # E_ss[R,Q] = vnl_E[s,s,R,Q]
        E_ss = vnl_E[s, s]  # (total_R, total_R)
        # Σ_R,Q conj(Z[R,G]) E_ss[R,Q] Z[Q,G]
        EZ = E_ss @ vnl_Z  # (total_R, nG)
        vnl_diag = vnl_diag + jnp.real(
            jnp.sum(jnp.conj(vnl_Z) * EZ, axis=0)
        )

    return T_diag + v_of_0 + vnl_diag


# ═══════════════════════════════════════════════════════════════════════
#  Setup: build HamiltonianK for one k-point
# ═══════════════════════════════════════════════════════════════════════

def setup_H_k(
    k_idx: int,
    V_scf: jax.Array,
    vnl_setup,
    wfn,
    sym,
    meta,
    V_loc_r: jax.Array | None = None,
    ngkmax: int | None = None,
) -> HamiltonianK:
    """Assemble all per-k Hamiltonian data (SymMaps path).

    Parameters
    ----------
    k_idx : index into sym.unfolded_kpts (full BZ)
    V_scf : (nx, ny, nz) — V_loc + V_H + V_xc, from build_V_scf
    vnl_setup : from vnl_ops.build_vnl_setup (k-independent, built once)
    wfn, sym, meta : standard LORRAX objects
    V_loc_r : (nx, ny, nz) — ionic local potential alone, for h_diag.
        If None, h_diag falls back to T_diag only.
    ngkmax : int, optional — pad all arrays to this size for uniform JIT.
        Compute once via ``compute_ngkmax`` and pass to every k-point.
    """
    T_diag, Gx, Gy, Gz = build_T_diag(k_idx, wfn, sym, meta)
    nG_actual = int(Gx.shape[0])

    # Pad G-vectors BEFORE building VNL so _table_interp sees uniform shapes
    if ngkmax is not None and ngkmax > nG_actual:
        pad = ngkmax - nG_actual
        T_diag = jnp.pad(T_diag, (0, pad), constant_values=1e10)
        Gx = jnp.pad(Gx, (0, pad), constant_values=0)
        Gy = jnp.pad(Gy, (0, pad), constant_values=0)
        Gz = jnp.pad(Gz, (0, pad), constant_values=0)
        mask = jnp.concatenate([jnp.ones(nG_actual, dtype=jnp.bool_),
                                jnp.zeros(pad, dtype=jnp.bool_)])
    else:
        mask = jnp.ones(nG_actual, dtype=jnp.bool_)

    # Build VNL at padded size — one JIT trace for all k-points
    Gk_int = np.stack([np.asarray(Gx), np.asarray(Gy), np.asarray(Gz)], axis=-1)
    kvec = np.asarray(sym.unfolded_kpts[k_idx], dtype=float)
    import psp.vnl_ops as vnl_ops
    kdata = vnl_ops.build_vnl_kdata_from_kvec(kvec, Gk_int, vnl_setup)

    h_diag = (build_h_diag(T_diag, V_loc_r, kdata.Z, kdata.E_super)
              if V_loc_r is not None else T_diag)

    return HamiltonianK(
        T_diag=T_diag,
        V_scf=V_scf,
        Gx=Gx, Gy=Gy, Gz=Gz,
        vnl_Z=kdata.Z,
        vnl_E=kdata.E_super,
        h_diag=h_diag,
        mask=mask,
        nG=nG_actual,
        fft_grid=tuple(int(x) for x in meta.fft_grid),
    )


def setup_H_k_from_kvec(
    kvec: np.ndarray,
    V_scf: jax.Array,
    vnl_setup,
    crystal,
    meta,
    V_loc_r: jax.Array | None = None,
    ngkmax: int | None = None,
) -> HamiltonianK:
    """Assemble per-k Hamiltonian data (standalone, no SymMaps / WFN.h5).

    Parameters
    ----------
    kvec : (3,) — k-point in crystal coordinates
    V_scf : (nx, ny, nz) — combined local potential
    vnl_setup : from vnl_ops.build_vnl_setup
    crystal : CrystalData or any object with bdot, ecutwfc, fft_grid
    meta : Meta object (for nspinor)
    V_loc_r : (nx, ny, nz) — ionic local potential alone, for h_diag.
    ngkmax : int, optional — pad all arrays to this size for uniform JIT.
        If None, no padding (arrays have natural nG length).
    """
    import psp.vnl_ops as vnl_ops

    T_diag, Gx, Gy, Gz = build_T_diag_from_kvec(kvec, crystal)
    nG_actual = int(Gx.shape[0])
    Gk_int = np.stack([np.asarray(Gx), np.asarray(Gy), np.asarray(Gz)], axis=-1)
    kdata = vnl_ops.build_vnl_kdata_from_kvec(kvec, Gk_int, vnl_setup)
    h_diag = (build_h_diag(T_diag, V_loc_r, kdata.Z, kdata.E_super)
              if V_loc_r is not None else T_diag)

    mask = jnp.ones(nG_actual, dtype=jnp.bool_)

    # Pad to ngkmax if requested (uniform shapes for JIT reuse)
    if ngkmax is not None and ngkmax > nG_actual:
        pad = ngkmax - nG_actual
        T_diag = jnp.pad(T_diag, (0, pad), constant_values=1e10)
        h_diag = jnp.pad(h_diag, (0, pad), constant_values=1e10)
        Gx = jnp.pad(Gx, (0, pad), constant_values=0)
        Gy = jnp.pad(Gy, (0, pad), constant_values=0)
        Gz = jnp.pad(Gz, (0, pad), constant_values=0)
        vnl_Z = jnp.pad(kdata.Z, ((0, 0), (0, pad)), constant_values=0.0)
        vnl_E = kdata.E_super
        mask = jnp.concatenate([mask, jnp.zeros(pad, dtype=jnp.bool_)])
    else:
        vnl_Z = kdata.Z
        vnl_E = kdata.E_super

    return HamiltonianK(
        T_diag=T_diag,
        V_scf=V_scf,
        Gx=Gx, Gy=Gy, Gz=Gz,
        vnl_Z=vnl_Z,
        vnl_E=vnl_E,
        h_diag=h_diag,
        mask=mask,
        nG=nG_actual,
        fft_grid=tuple(int(x) for x in crystal.fft_grid),
    )


# ═══════════════════════════════════════════════════════════════════════
#  Core fused JIT kernels
# ═══════════════════════════════════════════════════════════════════════

@functools.partial(jax.jit, donate_argnums=(0,))
def apply_H_k(psi_box, T_diag, V_scf, Gx, Gy, Gz, vnl_Z, vnl_E, mask):
    """H|ψ⟩ = (T + V_scf + V_NL)|ψ⟩.  Single fused JIT dispatch.

    The input psi_box is **donated** (its buffer is reused by XLA).
    After this call the caller must not read psi_box.

    Padding G-vectors (where mask=False) are zeroed in the output.

    Parameters
    ----------
    psi_box  : (nvec, nspinor, nx, ny, nz) complex128
    T_diag   : (nG_padded,) float64
    V_scf    : (nx, ny, nz) float64
    Gx,Gy,Gz : (nG_padded,) int32
    vnl_Z    : (total_R, nG_padded) complex128
    vnl_E    : (nspinor, nspinor, total_R, total_R) complex128
    mask     : (nG_padded,) bool — True for valid G-vectors

    Returns
    -------
    H_psi : (nvec, nspinor, nG_padded) complex128 — sparse-G (padding zeroed)
    """
    mask_f = mask[None, None, :].astype(psi_box.dtype)
    psi_G = psi_box[:, :, Gx, Gy, Gz] * mask_f

    # ── T: kinetic energy (diagonal in G-space) ──────────────────────
    H_G = T_diag[None, None, :] * psi_G

    # ── V_scf: self-consistent local potential (real-space multiply) ─
    psi_r = jnp.fft.ifftn(psi_box, axes=(-3, -2, -1), norm='ortho')
    H_G = H_G + jnp.fft.fftn(
        psi_r * V_scf, axes=(-3, -2, -1), norm='ortho'
    )[:, :, Gx, Gy, Gz] * mask_f

    # ── V_NL: Kleinman–Bylander (project → D → unproject) ───────────
    P = jnp.einsum('RG,vsG->Rsv', jnp.conj(vnl_Z), psi_G, optimize=True)
    D = jnp.einsum('stRQ,Qtv->Rsv', vnl_E, P, optimize=True)
    H_G = H_G + jnp.einsum('RG,Rsv->vsG', vnl_Z, D, optimize=True) * mask_f

    return H_G


def apply(psi_box: jax.Array, H_k: HamiltonianK) -> jax.Array:
    """Convenience: H|ψ⟩ from a HamiltonianK dataclass."""
    return apply_H_k(
        psi_box, H_k.T_diag, H_k.V_scf,
        H_k.Gx, H_k.Gy, H_k.Gz,
        H_k.vnl_Z, H_k.vnl_E, H_k.mask,
    )


@jax.jit
def build_matrix_k(psi_box, T_diag, V_scf, Gx, Gy, Gz, vnl_Z, vnl_E, mask):
    """Full ⟨m|H|n⟩ matrix at one k-point.  Returns (nb, nb) complex128."""
    import psp.vnl_ops as vnl_ops

    mask_f = mask[None, None, :].astype(psi_box.dtype)
    psi_G = psi_box[:, :, Gx, Gy, Gz] * mask_f

    # T
    H_mn = jnp.einsum(
        'msG,nsG->mn', jnp.conj(psi_G),
        T_diag[None, None, :] * psi_G, optimize=True,
    )

    # V_scf
    psi_r = jnp.fft.ifftn(psi_box, axes=(-3, -2, -1), norm='ortho')
    Vpsi_G = jnp.fft.fftn(
        psi_r * V_scf, axes=(-3, -2, -1), norm='ortho'
    )[:, :, Gx, Gy, Gz] * mask_f
    H_mn = H_mn + jnp.einsum(
        'msG,nsG->mn', jnp.conj(psi_G), Vpsi_G, optimize=True,
    )

    # V_NL
    H_mn = H_mn + vnl_ops.vnl_matrix(psi_G, vnl_Z, vnl_E)

    return H_mn


def matrix(psi_box: jax.Array, H_k: HamiltonianK) -> jax.Array:
    """Convenience: ⟨m|H|n⟩ from a HamiltonianK dataclass."""
    return build_matrix_k(
        psi_box, H_k.T_diag, H_k.V_scf,
        H_k.Gx, H_k.Gy, H_k.Gz,
        H_k.vnl_Z, H_k.vnl_E, H_k.mask,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Autodiff-compatible V_NL: differentiate through k for velocity
# ═══════════════════════════════════════════════════════════════════════
#
# The velocity operator v = dH/dk = 2(k+G) + dV_NL/dk.
# The kinetic part is trivial; the V_NL part requires differentiating
# the KB projectors Z(k) through the k-dependence of |k+G|.
#
# Three evaluation paths, all giving the same result:
#   vnl_velocity_autodiff  — full jacfwd (safest, verifiable)
#   vnl_velocity_from_dZ   — precomputed Z and dZ (fast for repeated use)
#   vnl_matrix_at_k        — k-traceable V_NL for custom autodiff chains

from psp.solid_harmonics import solid_harmonics_jax as _solid_harmonics_jax
from psp.radial_jax import (
    RadialTable,
    differentiate_uniform_table,
    interp_uniform_jax,
)


# --- VNL channel data (k-independent, for autodiff path) -----------------

@dataclass
class VNLChannelData:
    """Pre-extracted data for one (species, l) VNL channel.

    All fields are plain arrays suitable for JIT / autodiff.
    """
    tau: jax.Array              # (natoms, 3) crystal positions
    prefactor: float            # 4π / √Ω
    l: int
    nbeta: int
    q0: float
    dq: float
    reduced_tables: tuple[np.ndarray, ...]    # G_l(q) = F_l(q) / q^l
    reduced_dtables: tuple[np.ndarray, ...]   # d/dq of reduced_tables
    E: jax.Array                # (nspinor, nspinor, R, R) D-matrix


def _build_reduced_tables(tab: RadialTable, l: int) -> tuple[np.ndarray, np.ndarray]:
    """Reduced radial table G_l(q) = F_l(q)/q^l and its q-derivative."""
    F_vals = np.asarray(tab.values, dtype=np.float64)
    q_grid = tab.q0 + tab.dq * np.arange(F_vals.size, dtype=np.float64)
    if l == 0:
        G_vals = F_vals.copy()
    else:
        G_vals = np.empty_like(F_vals)
        G_vals[1:] = F_vals[1:] / q_grid[1:] ** l
        G_vals[0] = F_vals[1] / q_grid[1] ** l
    return G_vals, differentiate_uniform_table(G_vals, tab.dq)


def extract_vnl_channel_data(
    plan: dict,
    nspinor: int = 2,
) -> list[VNLChannelData]:
    """Extract all VNL channels from a projector plan into autodiff-ready form."""
    channels = []
    for _key, sp in plan.items():
        tau = np.asarray(sp['atoms']['tau'], dtype=np.float64)
        if tau.size == 0:
            continue
        if tau.ndim == 1:
            tau = tau.reshape(1, 3)
        pref = float(sp['prefactor'])
        radial_tables = sp['radial_tables']

        for l_key, info in sp['l_channels'].items():
            l = int(l_key)
            E_np = info['E']
            if E_np is None:
                continue
            beta_ids = info['beta_ids']
            if not beta_ids:
                continue

            q0 = None
            dq = None
            red_tables = []
            red_dtables = []
            for bid in beta_ids:
                tab = radial_tables[(l, int(bid))]
                if not isinstance(tab, RadialTable):
                    raise TypeError("Expected RadialTable in projector plan")
                q0 = float(tab.q0) if q0 is None else q0
                dq = float(tab.dq) if dq is None else dq
                G_vals, Gp_vals = _build_reduced_tables(tab, l)
                red_tables.append(G_vals)
                red_dtables.append(Gp_vals)

            E_j = jnp.asarray(E_np, dtype=jnp.complex128)[:nspinor, :nspinor]
            channels.append(VNLChannelData(
                tau=jnp.asarray(tau, dtype=jnp.float64),
                prefactor=pref,
                l=l,
                nbeta=len(beta_ids),
                q0=0.0 if q0 is None else q0,
                dq=1.0 if dq is None else dq,
                reduced_tables=tuple(red_tables),
                reduced_dtables=tuple(red_dtables),
                E=E_j,
            ))
    return channels


# --- KB projector construction (JAX-traceable through k) ------------------

def _build_Z_channel_jax(K_crys, K_cart, ch):
    """KB projector Z for one channel.  Pure JAX, k-traceable.

    Uses solid-harmonic factorisation:
        Z = pref · i^l · [F_l(q)/q^l] · S_lm(K) · exp(-2πi K·τ)
    """
    nG = K_crys.shape[0]
    radial_times_S = _radial_times_solid_harm(K_cart, ch, ch.prefactor)
    phase = jnp.exp(-2j * jnp.pi * (K_crys @ ch.tau.T)).T
    Z_atoms = phase[:, None, None, :] * radial_times_S[None, ...]
    R = ch.nbeta * (2 * ch.l + 1)
    return Z_atoms.reshape(ch.tau.shape[0], R, nG)


def _radial_times_solid_harm(K_cart, ch, pref):
    return _radial_times_solid_harm_impl(
        K_cart,
        tuple(ch.reduced_tables),
        tuple(ch.reduced_dtables),
        ch.q0,
        ch.dq,
        pref,
        ch.l,
        ch.nbeta,
    )


@functools.partial(jax.custom_jvp, nondiff_argnums=(1, 2, 3, 4, 5, 6, 7))
def _radial_times_solid_harm_impl(
    K_cart, tables, dtables, q0, dq, pref, l, nbeta,
):
    _EPS2 = 1e-60
    q = jnp.sqrt(jnp.sum(K_cart ** 2, axis=1) + _EPS2)
    G_bG = jnp.stack(
        [interp_uniform_jax(q, q0, dq, jnp.asarray(tables[ib], dtype=jnp.float64)) for ib in range(nbeta)],
    )
    S = _solid_harmonics_jax(l, K_cart)
    return pref * (1j) ** l * G_bG[:, None, :] * S[None, :, :]


@_radial_times_solid_harm_impl.defjvp
def _radial_times_solid_harm_jvp(
    tables, dtables, q0, dq, pref, l, nbeta,
    primals, tangents,
):
    """Stable JVP: G'_l from precomputed derivative table, no autodiff
    through sqrt(K²) which would give NaN at K=0."""
    (K_cart,) = primals
    (dK_cart,) = tangents
    _EPS2 = 1e-60

    K_sq = jnp.sum(K_cart ** 2, axis=1)
    q = jnp.sqrt(K_sq + _EPS2)
    G_list, Gp_list = [], []
    for ib in range(nbeta):
        G_list.append(interp_uniform_jax(q, q0, dq, jnp.asarray(tables[ib], dtype=jnp.float64)))
        Gp_list.append(interp_uniform_jax(q, q0, dq, jnp.asarray(dtables[ib], dtype=jnp.float64)))
    G_bG = jnp.stack(G_list)
    Gp_bG = jnp.stack(Gp_list)
    S = _solid_harmonics_jax(l, K_cart)

    primal_out = pref * (1j) ** l * G_bG[:, None, :] * S[None, :, :]

    dq = jnp.sum(K_cart * dK_cart, axis=1) / q
    dG = Gp_bG * dq[None, :]
    _, dS = jax.jvp(lambda K: _solid_harmonics_jax(l, K), (K_cart,), (dK_cart,))

    tangent_out = pref * (1j) ** l * (
        dG[:, None, :] * S[None, :, :] + G_bG[:, None, :] * dS[None, :, :]
    )
    return primal_out, tangent_out


# --- V_NL matrix elements (k-traceable for autodiff) ---------------------

def vnl_matrix_at_k(k_crys, psi_G, G_int, B, channels):
    """V_NL matrix elements as a pure function of k.

    Fully JAX-traceable — jax.jacfwd w.r.t. k_crys gives dV_NL/dk.
    Returns (nb, nb) complex128.
    """
    K_crys = G_int.astype(jnp.float64) + k_crys[None, :]
    K_cart = K_crys @ B
    nb = psi_G.shape[0]
    V_NL = jnp.zeros((nb, nb), dtype=jnp.complex128)

    for ch in channels:
        Z = _build_Z_channel_jax(K_crys, K_cart, ch)
        proj = jnp.einsum('aqG,vtG->aqtv', jnp.conj(Z), psi_G, optimize=True)
        d = jnp.einsum('strq,aqtv->arsv', ch.E, proj, optimize=True)
        vnl_G = jnp.einsum('arG,arsv->vsG', Z, d, optimize=True)
        V_NL = V_NL + jnp.einsum(
            'msG,nsG->mn', jnp.conj(psi_G), vnl_G, optimize=True,
        )
    return V_NL


def build_Z_and_dZ(k_crys, G_int, B, channels):
    """Precompute Z and dZ/dK_cart for all channels.

    Returns list of (Z, dZ, E) per channel where:
      Z  : (natoms, R, nG) complex128
      dZ : (3, natoms, R, nG) complex128
      E  : (nspinor, nspinor, R, R) complex128
    """
    _EPS2 = 1e-60
    K_crys = G_int.astype(jnp.float64) + k_crys[None, :]
    K_cart = K_crys @ B
    Binv = jnp.linalg.inv(B)
    q = jnp.sqrt(jnp.sum(K_cart ** 2, axis=1) + _EPS2)

    result = []
    for ch in channels:
        l, pref = ch.l, ch.prefactor
        msize = 2 * l + 1

        G_bG = jnp.stack([
            interp_uniform_jax(q, ch.q0, ch.dq, jnp.asarray(ch.reduced_tables[ib], dtype=jnp.float64))
            for ib in range(ch.nbeta)
        ])
        Gp_bG = jnp.stack([
            interp_uniform_jax(q, ch.q0, ch.dq, jnp.asarray(ch.reduced_dtables[ib], dtype=jnp.float64))
            for ib in range(ch.nbeta)
        ])
        S = _solid_harmonics_jax(l, K_cart)
        dS = jnp.stack([
            jax.jvp(lambda K: _solid_harmonics_jax(l, K), (K_cart,),
                     (jnp.zeros_like(K_cart).at[:, j].set(1.0),))[1]
            for j in range(3)
        ])

        phase = jnp.exp(-2j * jnp.pi * (K_crys @ ch.tau.T)).T
        tau_cart_eff = ch.tau @ Binv.T
        dphase = -2j * jnp.pi * tau_cart_eff[:, :, None] * phase[:, None, :]

        c_il = pref * (1j) ** l
        radS = G_bG[:, None, :] * S[None, :, :]
        Z_atoms = c_il * phase[:, None, None, :] * radS[None, ...]

        K_over_q = K_cart / q[:, None]
        drad = Gp_bG[:, None, :] * K_over_q.T[None, :, :]
        term1 = drad[:, :, None, :] * S[None, None, :, :]
        term2 = G_bG[:, None, None, :] * dS[None, :, :, :]
        dZ_core = c_il * (term1 + term2)
        dZ_full = (phase[:, None, None, None, :] * dZ_core[None, ...]
                   + c_il * radS[None, :, None, :, :] * dphase[:, None, :, None, :])

        R = ch.nbeta * msize
        nG = K_cart.shape[0]
        result.append((
            Z_atoms.reshape(ch.tau.shape[0], R, nG),
            dZ_full.transpose(2, 0, 1, 3, 4).reshape(3, ch.tau.shape[0], R, nG),
            ch.E,
        ))
    return result


def vnl_velocity_from_dZ(psi_G, Z_dZ_E):
    """V_NL velocity from precomputed Z and dZ.  Returns (3, nb, nb)."""
    nb = psi_G.shape[0]
    v = jnp.zeros((3, nb, nb), dtype=jnp.complex128)
    for Z, dZ, E in Z_dZ_E:
        proj = jnp.einsum('aqG,vtG->aqtv', jnp.conj(Z), psi_G, optimize=True)
        d = jnp.einsum('strq,aqtv->arsv', E, proj, optimize=True)
        # dZ† E Z ψ
        for j in range(3):
            dproj = jnp.einsum('aqG,vtG->aqtv', jnp.conj(dZ[j]), psi_G, optimize=True)
            dd = jnp.einsum('strq,aqtv->arsv', E, dproj, optimize=True)
            vnl_dZ_G = jnp.einsum('arG,arsv->vsG', Z, dd, optimize=True)
            vnl_Z_dG = jnp.einsum('arG,arsv->vsG', dZ[j], d, optimize=True)
            v_j_G = vnl_dZ_G + vnl_Z_dG
            v = v.at[j].add(jnp.einsum(
                'msG,nsG->mn', jnp.conj(psi_G), v_j_G, optimize=True,
            ))
    return v


def vnl_velocity_autodiff(k_crys, psi_G, G_int, B, channels):
    """V_NL velocity via jacfwd.  Returns (3, nb, nb)."""
    def f(k):
        return vnl_matrix_at_k(k, psi_G, G_int, B, channels)
    return jax.jacfwd(f)(k_crys) @ jnp.linalg.inv(B)


# ═══════════════════════════════════════════════════════════════════════
#  Velocity / dipole matrix elements
# ═══════════════════════════════════════════════════════════════════════

@jax.jit
def momentum_matrix_k(psi_G, G_int, k_crys, B):
    """Kinetic part of velocity: p_i = 2(k+G)_i.  Returns (3, nb, nb)."""
    K_cart = (G_int.astype(jnp.float64) + k_crys[None, :]) @ B
    return 2.0 * jnp.einsum(
        'msG,Gi,nsG->imn', jnp.conj(psi_G), K_cart, psi_G, optimize=True,
    )


def velocity_matrix_k(psi_G, G_int, k_crys, B, channels, *, Z_dZ_E=None):
    """Full velocity: v = dH/dk = 2(k+G) + dV_NL/dk.  Returns (3, nb, nb)."""
    p = momentum_matrix_k(psi_G, G_int, k_crys, B)
    if Z_dZ_E is None:
        Z_dZ_E = build_Z_and_dZ(k_crys, G_int, B, channels)
    return p + vnl_velocity_from_dZ(psi_G, Z_dZ_E)


# ═══════════════════════════════════════════════════════════════════════
#  Batch helpers
# ═══════════════════════════════════════════════════════════════════════

def compute_dipole_all(wfn, sym, meta, vnl_plan, B, nb=None):
    """Velocity matrix elements for all k-points.

    Returns (dipole, deltaE) where:
      dipole : (3, nk, nb, nb) complex128
      deltaE : (nk, nb, nb) float64
    """

    from common.load_wfns import load_kpoint_fftbox

    if nb is None:
        nb = int(meta.b_id_4)
    nk = sym.nk_tot
    nspinor = int(meta.nspinor)

    channels = extract_vnl_channel_data(vnl_plan, nspinor=nspinor)
    B_j = jnp.asarray(B, dtype=jnp.float64)

    dipole = np.zeros((3, nk, nb, nb), dtype=np.complex128)
    deltaE = np.zeros((nk, nb, nb), dtype=np.float64)
    energies = np.asarray(wfn.energies)

    for ik in range(nk):
        with timing.section(f"dipole_k{ik}"):
            wfn_k = load_kpoint_fftbox(wfn, sym, meta, ik, nb)
            Gk_crys, _ = generate_gvectors_k(ik, sym, wfn, meta)
            psi_G = wfn_k[:, :,
                          np.asarray(Gk_crys)[:, 0],
                          np.asarray(Gk_crys)[:, 1],
                          np.asarray(Gk_crys)[:, 2]]
            G_int = jnp.asarray(np.asarray(Gk_crys, dtype=int), dtype=jnp.int32)
            k_j = jnp.asarray(sym.unfolded_kpts[ik], dtype=jnp.float64)

            Z_dZ_E = build_Z_and_dZ(k_j, G_int, B_j, channels)
            dipole[:, ik] = np.asarray(
                velocity_matrix_k(psi_G, G_int, k_j, B_j, channels, Z_dZ_E=Z_dZ_E)
            )

            try:
                k_red = int(sym.irk_to_k_map[ik])
            except Exception:
                k_red = int(ik)
            e_b = np.asarray(
                energies[0, k_red, :nb] if energies.ndim >= 3 else energies[:nb],
                dtype=float,
            )
            deltaE[ik] = e_b[:, None] - e_b[None, :]
            del wfn_k

    return dipole, deltaE

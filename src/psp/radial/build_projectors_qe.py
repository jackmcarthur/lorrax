"""
Spin–orbit (fully relativistic) projector algebra, mirroring the QE appendix
and Dal Corso & Mosca Conte (PRB 87, 115112 (2013)) real-harmonics setup.

This module builds, for each ℓ and j=ℓ±1/2:
  - U^ℓ: (2ℓ+1)x(2ℓ+1) complex←real spherical-harmonics rotation
  - α^{σ,ℓ,j}: length-(2j+1) Clebsch–Gordan vectors (σ∈{↑,↓})
  - U^{σ,ℓ,j}: (2j+1)x(2ℓ+1) row-pickers from U^ℓ according to Eqs. (3)-(6)
  - f^{σσ'}_{ℓ j m m'}: (2,2,2ℓ+1,2ℓ+1) spin blocks from α and U^{σ,ℓ,j}
  - E^{I,σσ'}_{ℓ m m'}: (2,2,2ℓ+1,2ℓ+1) assembled from channel strengths per (ℓ,j)

All arrays are dense with deterministic axes; indices m and m' are ordered as
[-ℓ..+ℓ] mapped to [0..2ℓ] via m_to_idx. Spin index order is (0→↑, 1→↓).

These builders let you reconstruct the VKB-style nonlocal operator blocks
  V_NL^{σ,σ'} = Σ_I Σ_{ℓ,j} Σ_{m,m'} E^{I,σ,σ'}_{ℓ m m'} |β^I_{ℓ j} Y'_{ℓ m}><β^I_{ℓ j} Y'_{ℓ m'}|
with Y' the real spherical harmonics used in QE radial-channel separation.
"""

from __future__ import annotations

import os
import numpy as np
from typing import Dict, Tuple, List, Sequence, Mapping

import jax.numpy as jnp

from .radial_jax import (
    RadialTable,
    make_local_sr_table,
    make_projector_table,
    make_uniform_q_grid,
    radial_weights,
)
# The QE radial quadrature schemes have ONE owner: psp.radial_tables.
# (No cycle: radial_tables imports psp.species + psp.radial.radial_jax only.)
from psp.radial_tables import (
    qe_beta_radial_scheme as _qe_beta_radial_scheme,
    qe_vloc_radial_scheme as _qe_vloc_radial_scheme,
)

# Self-contained: provide local implementations of helpers previously imported


# --------------------------
# Indexing helpers
# --------------------------

def m_to_idx(m: int, l: int) -> int:
    """Row index for complex harmonic with magnetic quantum number m (−ℓ..ℓ)."""
    return int(m) + int(l)


def mj_grid(j: float) -> np.ndarray:
    """Return array of m_j from -j..+j (step 1) as float values.

    j may be half-integer; we represent values exactly as floats such that
    2*j is an integer. The length is 2j+1.
    """
    two_j = int(round(2 * j))
    # Values from -j to +j inclusive with step 1
    return (np.arange(two_j + 1, dtype=float) - j)


# --------------------------
# U^ℓ: complex <- real spherical-harmonics rotation
# --------------------------

def U_complex_from_real(l: int) -> np.ndarray:
    """Build QE-style unitary U^ℓ such that
    Y_{ℓ,m} = Σ_{m'=-ℓ}^{ℓ} U^ℓ[m,m'] Y'_{ℓ,m'}.

    Conventions follow the appendix definitions:
      Y'_{ℓ,|m|,cos} = (Y_{ℓ,|m|} + (-1)^{|m|} Y_{ℓ,-|m|}) / √2  (stored at column m'=+|m|)
      Y'_{ℓ,|m|,sin} = -i (Y_{ℓ,|m|} - (-1)^{|m|} Y_{ℓ,-|m|}) / √2 (stored at column m'=-|m|)
      Y'_{ℓ,0} ≡ Y_{ℓ,0}

    We store columns in order m'=-ℓ..+ℓ; for |m|>0 we assign the sin
    real harmonic to column m'=-|m| and the cos to column m'=+|m|.
    Rows are m=-ℓ..+ℓ.
    """
    n = 2 * l + 1
    U = np.zeros((n, n), dtype=np.complex128)

    # Column ordering follows QE/uspp real-harmonics convention:
    #   0 -> Y'_{ℓ,0}
    #   1 -> cos component for m=+1, 2 -> sin component for m=+1,
    #   3 -> cos for m=+2, 4 -> sin for m=+2, etc.

    U[m_to_idx(0, l), 0] = 1.0

    for m in range(1, l + 1):
        s = (-1) ** m
        col_cos = 2 * m - 1
        col_sin = 2 * m

        row_p = m_to_idx(+m, l)
        row_m = m_to_idx(-m, l)

        U[row_p, col_cos] = 1.0 / np.sqrt(2)
        U[row_p, col_sin] = 1j / np.sqrt(2)
        U[row_m, col_cos] = s / np.sqrt(2)
        U[row_m, col_sin] = -s * 1j / np.sqrt(2)

    return U


def _ylmr2_qe(lmax: int, vectors: np.ndarray) -> np.ndarray:
    """Replicate QE's ylmr2 real-spherical-harmonic generator."""
    eps = 1e-9
    fpi = 4.0 * np.pi
    g = np.asarray(vectors, dtype=np.float64)
    ng = g.shape[0]
    lmax2 = (lmax + 1) ** 2
    ylm = np.zeros((ng, lmax2), dtype=np.float64)

    for ig in range(ng):
        gx, gy, gz = g[ig]
        gmod = np.linalg.norm(g[ig])
        cost = gz / gmod if gmod >= eps else 0.0
        sent = np.sqrt(max(0.0, 1.0 - cost * cost))
        arr = ylm[ig]
        arr[0] = 1.0
        if lmax >= 1:
            arr[1] = cost
            arr[3] = -sent / np.sqrt(2.0)
        for l in range(2, lmax + 1):
            for m in range(0, l - 1):
                lm = l * l + 2 * m
                lm1 = (l - 1) * (l - 1) + 2 * m
                lm2 = (l - 2) * (l - 2) + 2 * m
                denom = np.sqrt(l * l - m * m)
                term1 = cost * (2 * l - 1) * arr[lm1] / denom
                term2 = np.sqrt((l - 1) * (l - 1) - m * m) * arr[lm2] / denom
                arr[lm] = term1 - term2
            lm1 = l * l + 2 * (l - 1)
            lm = l * l + 2 * l
            lm2 = (l - 1) * (l - 1) + 2 * (l - 1)
            arr[lm1] = cost * np.sqrt(2 * l - 1.0) * arr[lm2]
            arr[lm] = -(np.sqrt(2 * l - 1.0) / np.sqrt(2 * l)) * sent * arr[lm2]

        if gmod < eps:
            phi = 0.0
        elif abs(gx) > eps:
            phi = np.arctan2(gy, gx)
        else:
            phi = np.sign(gy) * (np.pi / 2.0)

        lm = 0
        arr[0] /= np.sqrt(fpi)
        for l in range(1, lmax + 1):
            c = np.sqrt((2 * l + 1) / fpi)
            lm = l * l
            arr[lm] *= c
            for m in range(1, l + 1):
                cos_idx = l * l + 2 * m - 1
                sin_idx = l * l + 2 * m
                val = c * np.sqrt(2.0) * arr[sin_idx]
                arr[cos_idx] = val * np.cos(m * phi)
                arr[sin_idx] = val * np.sin(m * phi)

    return ylm


def qe_real_sph_harmonics(l: int, vectors: np.ndarray) -> np.ndarray:
    """Return QE-normalised real spherical harmonics for degree ℓ."""
    ylm_all = _ylmr2_qe(l, vectors)
    start = l * l
    end = (l + 1) * (l + 1)
    return ylm_all[:, start:end].T


# Removed: qe_complex_sph_harmonics (unused in production)


# --------------------------
# Real-harmonics with angular gradients (Cartesian)
# --------------------------

def _angles_from_cart_vectors(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = np.asarray(vectors, dtype=float)
    q = np.linalg.norm(g, axis=1)
    q_safe = np.where(q > 0, q, 1.0)
    x = g[:, 0] / q_safe
    y = g[:, 1] / q_safe
    z = g[:, 2] / q_safe
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)
    return q, theta, phi


def _spherical_frame(theta: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    st = np.sin(theta); ct = np.cos(theta)
    cp = np.cos(phi); sp = np.sin(phi)
    e_r = np.stack([st * cp, st * sp, ct], axis=1)
    e_t = np.stack([ct * cp, ct * sp, -st], axis=1)
    e_p = np.stack([-sp, cp, np.zeros_like(st)], axis=1)
    return e_r, e_t, e_p


def _complex_sph_and_angular_partials(l: int, theta: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Y_complex: (2l+1, nG) with m ordered [-l..+l]
    nG = theta.shape[0]
    ms = np.arange(-l, l + 1, dtype=int)
    try:
        from scipy.special import sph_harm as _Y
        Y = np.stack([_Y(m, l, phi, theta) for m in ms], axis=0)
        # Angular partials: use small-step central differences for ∂θ and ∂φ
        h = 1e-6
        Y_t1 = np.stack([_Y(m, l, phi, theta + h) for m in ms], axis=0)
        Y_t0 = np.stack([_Y(m, l, phi, theta - h) for m in ms], axis=0)
        dth = (Y_t1 - Y_t0) / (2 * h)
        Y_p1 = np.stack([_Y(m, l, phi + h, theta) for m in ms], axis=0)
        Y_p0 = np.stack([_Y(m, l, phi - h, theta) for m in ms], axis=0)
        dph = (Y_p1 - Y_p0) / (2 * h)
        return Y, dth, dph
    except Exception:
        # Fallback supports l<=1 explicitly
        Y = np.zeros((2 * l + 1, nG), dtype=complex)
        dth = np.zeros_like(Y)
        dph = np.zeros_like(Y)
        if l == 0:
            Y[0, :] = 0.5 / np.sqrt(np.pi)
            dth[0, :] = 0.0
            dph[0, :] = 0.0
        elif l == 1:
            # m=-1,0,1
            c = np.sqrt(3 / (8 * np.pi))
            Y[0, :] = c * np.exp(-1j * phi) * np.sin(theta)
            Y[1, :] = np.sqrt(3 / (4 * np.pi)) * np.cos(theta)
            Y[2, :] = -c * np.exp(1j * phi) * np.sin(theta)
            # Finite differences for partials
            h = 1e-6
            th1 = theta + h; th0 = theta - h
            ph1 = phi + h; ph0 = phi - h
            Y_t1 = np.stack([
                c * np.exp(-1j * phi) * np.sin(th1),
                np.sqrt(3 / (4 * np.pi)) * np.cos(th1),
                -c * np.exp(1j * phi) * np.sin(th1),
            ], axis=0)
            Y_t0 = np.stack([
                c * np.exp(-1j * phi) * np.sin(th0),
                np.sqrt(3 / (4 * np.pi)) * np.cos(th0),
                -c * np.exp(1j * phi) * np.sin(th0),
            ], axis=0)
            dth[:, :] = (Y_t1 - Y_t0) / (2 * h)
            Y_p1 = np.stack([
                c * np.exp(-1j * ph1) * np.sin(theta),
                np.sqrt(3 / (4 * np.pi)) * np.cos(theta) * 1.0,
                -c * np.exp(1j * ph1) * np.sin(theta),
            ], axis=0)
            Y_p0 = np.stack([
                c * np.exp(-1j * ph0) * np.sin(theta),
                np.sqrt(3 / (4 * np.pi)) * np.cos(theta) * 1.0,
                -c * np.exp(1j * ph0) * np.sin(theta),
            ], axis=0)
            dph[:, :] = (Y_p1 - Y_p0) / (2 * h)
        else:
            raise RuntimeError("Angular gradient fallback supports l<=1 only")
        return Y, dth, dph


def qe_real_sph_harmonics_with_grad(l: int, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (Y_real, gradY_cart) with shapes (2l+1, nG) and (3, 2l+1, nG).

    Analytic QE-style: build Y′ via ylmr2-like recurrences (same as qe_real_sph_harmonics)
    and propagate ∂θ; take ∂φ analytically; map to Cartesian via eθ ∂θ + eφ (∂φ/sinθ).
    """
    g = np.asarray(vectors, dtype=float)
    nG = g.shape[0]
    if nG == 0:
        return np.zeros((2 * l + 1, 0), dtype=float), np.zeros((3, 2 * l + 1, 0), dtype=float)
    q, theta, phi = _angles_from_cart_vectors(g)
    st = np.sin(theta); ct = np.cos(theta)
    st_safe = np.maximum(st, 1e-12)
    e_r, e_t, e_p = _spherical_frame(theta, phi)

    lmax2 = (l + 1) * (l + 1)
    ylm = np.zeros((nG, lmax2), dtype=np.float64)
    dth = np.zeros_like(ylm)
    ylm[:, 0] = 1.0; dth[:, 0] = 0.0
    if l >= 1:
        ylm[:, 1] = ct; dth[:, 1] = -st
        ylm[:, 3] = -st/np.sqrt(2.0); dth[:, 3] = -ct/np.sqrt(2.0)
    for L in range(2, l + 1):
        for m in range(0, L - 1):
            lm = L*L + 2*m
            lm1 = (L-1)*(L-1) + 2*m
            lm2 = (L-2)*(L-2) + 2*m
            denom = np.sqrt(L*L - m*m)
            a = (2*L - 1)/denom
            b = np.sqrt((L-1)*(L-1) - m*m)/denom
            ylm[:, lm] = a*(ct*ylm[:, lm1]) - b*ylm[:, lm2]
            dth[:, lm] = a*((-st)*ylm[:, lm1] + ct*dth[:, lm1]) - b*dth[:, lm2]
        lm1 = L*L + 2*(L-1)
        lm = L*L + 2*L
        lm2 = (L-1)*(L-1) + 2*(L-1)
        r1 = np.sqrt(2*L - 1.0)
        r2 = r1/np.sqrt(2.0*L)
        ylm[:, lm1] = ct*r1*ylm[:, lm2]
        dth[:, lm1] = (-st)*r1*ylm[:, lm2] + ct*r1*dth[:, lm2]
        ylm[:, lm] = -(r2)*st*ylm[:, lm2]
        dth[:, lm] = -(r2)*(ct*ylm[:, lm2] + st*dth[:, lm2])

    fpi = 4.0*np.pi
    ylm[:, 0] /= np.sqrt(fpi); dth[:, 0] /= np.sqrt(fpi)
    for L in range(1, l + 1):
        c = np.sqrt((2*L + 1)/fpi)
        lm0 = L*L
        ylm[:, lm0] *= c; dth[:, lm0] *= c
        for m in range(1, L + 1):
            cos_idx = L*L + 2*m - 1
            sin_idx = L*L + 2*m
            val = c*np.sqrt(2.0)*ylm[:, sin_idx]
            dval = c*np.sqrt(2.0)*dth[:, sin_idx]
            cm = np.cos(m*phi); sm = np.sin(m*phi)
            ylm[:, cos_idx] = val*cm
            ylm[:, sin_idx] = val*sm
            dth[:, cos_idx] = dval*cm
            dth[:, sin_idx] = dval*sm

    dph = np.zeros_like(ylm)
    for L in range(1, l + 1):
        for m in range(1, L + 1):
            cos_idx = L*L + 2*m - 1
            sin_idx = L*L + 2*m
            dph[:, cos_idx] = -m*ylm[:, sin_idx]
            dph[:, sin_idx] = m*ylm[:, cos_idx]

    start = l*l; end = (l+1)*(l+1)
    Y_block = ylm[:, start:end].T
    dth_block = dth[:, start:end].T
    dph_block = dph[:, start:end].T
    gt_x = e_t[:, 0][None, :]*dth_block + e_p[:, 0][None, :]*(dph_block/st_safe[None, :])
    gt_y = e_t[:, 1][None, :]*dth_block + e_p[:, 1][None, :]*(dph_block/st_safe[None, :])
    gt_z = e_t[:, 2][None, :]*dth_block + e_p[:, 2][None, :]*(dph_block/st_safe[None, :])
    grad_cart = np.stack([gt_x, gt_y, gt_z], axis=0)
    return Y_block, grad_cart


def _spherical_hankel_transform_l(l: int,
                                  r: np.ndarray,
                                  beta_r: np.ndarray,
                                  q: np.ndarray,
                                  rab: np.ndarray | None = None) -> np.ndarray:
    q_grid = np.asarray(q, dtype=np.float64)
    weights = radial_weights(np.asarray(r, dtype=float), rab, scheme='rab')
    return make_projector_table(
        int(l),
        np.asarray(r, dtype=float),
        np.asarray(beta_r, dtype=float),
        q_grid,
        weights,
    ).values


def spherical_hankel_transform_l_np(l: int,
                                    r: np.ndarray,
                                    beta_r: np.ndarray,
                                    q: np.ndarray,
                                    rab: np.ndarray | None = None) -> np.ndarray:
    """Public alias for the numpy spherical Hankel transform used by QE paths."""
    return _spherical_hankel_transform_l(l, r, beta_r, q, rab)


def _fourier_transform_vloc_qe(
    r: np.ndarray,
    rab: np.ndarray | None,
    vloc_r: np.ndarray,
    q_grid: np.ndarray,
    z_valence: float,
    cell_volume: float,
) -> np.ndarray:
    """QE-style Fourier transform of the local pseudopotential (Ry).

    Matches upflib/vloc_mod by tabulating the short-range part with an
    erf-subtracted integrand and re-adding the analytic tail in G space,
    integrating with QE's own radial scheme (r <= 10 bohr, odd-point
    Simpson — see :func:`_qe_vloc_radial_scheme`).
    """
    r = np.asarray(r, dtype=float)
    msh, weights = _qe_vloc_radial_scheme(r, rab)
    return make_local_sr_table(
        r[:msh],
        np.asarray(vloc_r, dtype=float)[:msh],
        float(z_valence),
        np.asarray(q_grid, dtype=float),
        weights,
    ).values


def _form_factors_qe(pseudo, K_norm: np.ndarray, cell_volume: float) -> Dict[Tuple[int, int], np.ndarray]:
    ppmesh = getattr(pseudo, 'pp_mesh', None)
    ppnl = getattr(pseudo, 'pp_nonlocal', None)
    if ppmesh is None or ppnl is None:
        return {}
    r = np.asarray(ppmesh.pp_r.value, dtype=float)
    try:
        rab = np.asarray(ppmesh.pp_rab.value, dtype=float)
    except Exception:
        rab = None
    betas = list(getattr(ppnl, 'pp_beta', []))
    if not betas:
        return {}
    K_norm = np.asarray(K_norm, dtype=float)
    qmax = float(np.max(K_norm)) if K_norm.size > 0 else 0.0
    q_points = max(1024, 2 * len(r))
    q_grid = make_uniform_q_grid(qmax, q_points)
    kkb, weights = _qe_beta_radial_scheme(r, rab, _kkbeta_of(betas, len(r)))
    result: Dict[Tuple[int, int], np.ndarray] = {}
    for idx, beta in enumerate(betas, start=1):
        l = int(getattr(beta, 'lll', getattr(beta, 'angular_momentum', 0)))
        raw_beta = np.asarray(beta.value, dtype=float)
        beta_r = np.zeros_like(r)
        if r[0] == 0.0 and len(r) > 1:
            beta_r[1:] = raw_beta[1:] / r[1:]
            beta_r[0] = beta_r[1]
        else:
            beta_r = raw_beta / r
        tab = make_projector_table(l, r[:kkb], beta_r[:kkb], q_grid, weights)
        result[(l, idx)] = tab(K_norm)
    return result


def _kkbeta_of(betas, n_r: int) -> int:
    """QE's ``upf%kkbeta`` from raw UPF beta objects: max cutoff_radius_index
    over the betas, full mesh when the UPF carries none.  Twin of the
    ``SpeciesData.kkbeta`` extraction in ``psp.species.extract_species``."""
    kkb = 0
    for beta in betas:
        kb = getattr(beta, 'cutoff_radius_index', None)
        kkb = max(kkb, int(kb) if kb else n_r)
    return min(kkb, n_r) if betas else n_r


# Removed: build_species_rows_qe (not used by production pipeline)


# --------------------------
# Local ionic potential helper
# --------------------------

def build_local_ionic_potential_on_G_total(
    assignments: Sequence[Mapping[str, object]],
    species_groups: Sequence[tuple[object, np.ndarray]],
    fft_grid: Tuple[int, int, int],
    bdot: np.ndarray,
    cell_volume: float,
    bvec: np.ndarray | None = None,
    blat: float | None = None,
    truncation_2d: bool = False,
) -> jax.Array:
    """Simplified builder: return total V_loc(r) on the provided FFT grid (Ry).

    - Builds SR in G with QE's SR integrand (adds Z e2 erf(r)/r) and alpha‑Z at G=0.
    - Adds LR Coulomb tail from ρ_ion(G) with Gaussian damping; G=0 set to 0.
    - Maps with phases exp(-2π i τ·G) on the FFT grid and IFFT (orthonormal) to real.
    
    If truncation_2d=True, applies 2D slab truncation factor to the LR part:
        cutoff_2D = 1 - exp(-|G_xy| * lz) * cos(G_z * lz)
    where lz = π / b_z and b_z is the z-component of the reciprocal lattice.
    This matches QE's Coul_cut_2D module for assume_isolated='2D'.
    """
    nx, ny, nz = (int(fft_grid[0]), int(fft_grid[1]), int(fft_grid[2]))

    fx = np.fft.fftfreq(nx) * nx
    fy = np.fft.fftfreq(ny) * ny
    fz = np.fft.fftfreq(nz) * nz
    ix = fx[:, None, None]
    iy = fy[None, :, None]
    iz = fz[None, None, :]

    M = np.asarray(bdot, dtype=float)
    G2 = (
        M[0, 0] * ix * ix + M[1, 1] * iy * iy + M[2, 2] * iz * iz
        + 2.0 * M[0, 1] * ix * iy + 2.0 * M[0, 2] * ix * iz + 2.0 * M[1, 2] * iy * iz
    )
    q_flat = np.sqrt(np.maximum(0.0, G2)).ravel()
    sqrtN = np.sqrt(float(nx * ny * nz))
    inv_omega = 1.0 / float(cell_volume)

    # ρ_ion(G) on grid
    rho_ion_G = np.zeros((nx, ny, nz), dtype=np.complex128)
    for entry in assignments:
        pseudo = entry.get("pseudo")
        tau = np.asarray(entry.get("position"), dtype=float).reshape(3)
        Zval = float(getattr(getattr(pseudo, 'pp_header', None), 'z_valence', 0.0)) if pseudo is not None else 0.0
        rho_ion_G -= Zval * np.exp(-2j * np.pi * (tau[0] * ix + tau[1] * iy + tau[2] * iz))

    # SR from UPF
    Vloc_G_sr = np.zeros((nx, ny, nz), dtype=np.complex128)
    for pseudo, positions in species_groups:
        if pseudo is None:
            continue
        positions = np.asarray(positions, dtype=float)
        if positions.size == 0:
            continue
        ppmesh = getattr(pseudo, 'pp_mesh', None)
        vloc = getattr(getattr(pseudo, 'pp_local', None), 'value', None)
        if ppmesh is None or vloc is None:
            continue
        r = np.asarray(ppmesh.pp_r.value, dtype=float)
        try:
            rab = np.asarray(ppmesh.pp_rab.value, dtype=float)
        except Exception:
            rab = None
        v_r = np.asarray(vloc, dtype=float)
        Zval = float(getattr(pseudo.pp_header, 'z_valence', 0.0))

        # 8x-per-mesh-point q resolution: the table is consumed through
        # LINEAR interpolation, whose error is O(dq²); at 2*len(r) the
        # residual against QE's dq=0.01 cubic table was a measured
        # ~0.005 meV band-to-band spread on the KIH diagonal, at 8x it
        # is ~0.0004 meV.  One-time cost, host-side numpy.
        q_points = max(8192, 8 * len(r))
        q_grid = make_uniform_q_grid(float(np.max(q_flat)), q_points)
        V_q_sr = _fourier_transform_vloc_qe(r, rab, v_r, q_grid, Zval, cell_volume)
        tab = RadialTable(q0=float(q_grid[0]), dq=float(q_grid[1] - q_grid[0]), values=V_q_sr)
        V_sr_on_flat = (sqrtN * (4.0 * np.pi) * inv_omega) * tab(q_flat)
        V_sr_on_grid = V_sr_on_flat.reshape((nx, ny, nz))

        # Alpha‑Z at G=0
        # For 2D (modified_coulomb), QE uses tab_vloc(0) = tab_vloc(1), meaning the G=0
        # value is set to the q→0 limit of the main formula (NOT the alpha_Z integral).
        # This effectively means using the interpolated value at q→0.
        # For 3D, we use the full alpha_Z integral.
        try:
            e2 = 2.0
            if truncation_2d:
                # 2D: V_sr at G=0 already has the q→0 limit from interpolation (spl(0))
                # We don't need to override it - the spline already handles this.
                # Just ensure it's properly scaled.
                pass  # V_sr_on_grid[0,0,0] is already set by spl(0) * prefactor
            else:
                # 3D: alpha_Z integral with QE's exact radial scheme
                # (upflib/vloc_mod.f90 init_tab_vloc G=0 branch: simpson
                # over msh points of r*(r*vloc + Z*e2)).  See
                # _qe_vloc_radial_scheme for why bare-rab rectangle
                # weights put a -0.08 meV constant on every KIH diagonal.
                if len(r) > 1:
                    msh, w_qe = _qe_vloc_radial_scheme(r, rab)
                    integrand = r[:msh] * (r[:msh] * v_r[:msh] + Zval * e2)
                    alphaZ = (4.0 * np.pi) * np.sum(integrand * w_qe) * inv_omega
                else:
                    alphaZ = 0.0
                V_sr_on_grid[0, 0, 0] = sqrtN * alphaZ
        except Exception:
            pass

        # Species phases
        S = np.zeros((nx, ny, nz), dtype=np.complex128)
        for tau in positions:
            tau = np.asarray(tau, dtype=float).reshape(3)
            S += np.exp(-2j * np.pi * (tau[0] * ix + tau[1] * iy + tau[2] * iz))
        Vloc_G_sr += V_sr_on_grid * S

    # LR with Gaussian damping
    e2 = 2.0
    zero_mask = (ix == 0) & (iy == 0) & (iz == 0)
    G2_safe = np.where(zero_mask, 1.0, G2)
    Vloc_G_lr = (sqrtN * 4.0 * np.pi * e2) * inv_omega * (rho_ion_G / G2_safe) * np.exp(-0.25 * G2)
    
    # Apply 2D truncation if requested (matches QE's cutoff_lr_Vloc)
    if truncation_2d and bvec is not None and blat is not None:
        # Compute G vectors in Cartesian coordinates
        B = np.asarray(bvec, dtype=float) * float(blat)  # (3,3) reciprocal lattice in bohr^-1
        # G_cart = G_crys @ B^T
        Gx_cart = ix * B[0, 0] + iy * B[1, 0] + iz * B[2, 0]
        Gy_cart = ix * B[0, 1] + iy * B[1, 1] + iz * B[2, 1]
        Gz_cart = ix * B[0, 2] + iy * B[1, 2] + iz * B[2, 2]
        
        # 2D cutoff: lz = π / b_z, cutoff = 1 - exp(-|G_xy| * lz) * cos(G_z * lz)
        lz = np.pi / B[2, 2]
        Gxy = np.sqrt(Gx_cart * Gx_cart + Gy_cart * Gy_cart)
        cutoff_2d = 1.0 - np.exp(-Gxy * lz) * np.cos(Gz_cart * lz)
        cutoff_2d = np.where(zero_mask, 0.0, cutoff_2d)
        
        Vloc_G_lr = Vloc_G_lr * cutoff_2d
    
    Vloc_G_lr[0, 0, 0] = 0.0

    # Total to real
    V_tot = Vloc_G_sr + Vloc_G_lr
    V_r = jnp.real(jnp.fft.ifftn(jnp.asarray(V_tot), norm='ortho'))
    return jnp.asarray(V_r, dtype=jnp.float64)

# --------------------------
# α^{σ,ℓ,j} (Clebsch–Gordan) and U^{σ,ℓ,j}
# --------------------------

def alpha_sigma_lj(l: int, j: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (alpha_up, alpha_dn) vectors of length (2j+1).

    For j = ℓ+1/2 with m = m_j - 1/2:
      α^↑ = sqrt((ℓ+m+1)/(2ℓ+1));  α^↓ = sqrt((ℓ-m)/(2ℓ+1))
    For j = ℓ-1/2 with m = m_j + 1/2:
      α^↑ = sqrt((ℓ-m+1)/(2ℓ+1)); α^↓ = -sqrt((ℓ+m)/(2ℓ+1))
    """
    mj = mj_grid(j)  # (2j+1,)
    denom = float(2 * l + 1)
    if abs(j - (l + 0.5)) < 1e-12:
        m = mj - 0.5
        alpha_up = np.sqrt(np.maximum(0.0, (l + m + 1.0)) / denom)
        alpha_dn = np.sqrt(np.maximum(0.0, (l - m)) / denom)
    elif abs(j - (l - 0.5)) < 1e-12:
        m = mj + 0.5
        alpha_up = np.sqrt(np.maximum(0.0, (l - m + 1.0)) / denom)
        alpha_dn = -np.sqrt(np.maximum(0.0, (l + m)) / denom)
    else:
        raise ValueError("j must be l±1/2")
    return alpha_up, alpha_dn


def U_sigma_lj(l: int, j: float, U_l: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build (U^{↑,ℓ,j}, U^{↓,ℓ,j}) with shape ((2j+1), (2ℓ+1)).

    For j=ℓ+1/2 (m=m_j-1/2):
      U^↑ row r ← U^ℓ[m,:]     if m∈[-ℓ..ℓ] and integer
      U^↓ row r ← U^ℓ[m+1,:]   if m+1∈[-ℓ..ℓ]
    For j=ℓ-1/2 (m=m_j+1/2):
      U^↑ row r ← U^ℓ[m-1,:]   if m-1∈[-ℓ..ℓ]
      U^↓ row r ← U^ℓ[m,:]     if m∈[-ℓ..ℓ]

    Out-of-range rows are zeroed.
    """
    nrow = int(round(2 * j)) + 1
    ncol = 2 * l + 1
    U_up = np.zeros((nrow, ncol), dtype=np.complex128)
    U_dn = np.zeros((nrow, ncol), dtype=np.complex128)
    mj = mj_grid(j)

    if abs(j - (l + 0.5)) < 1e-12:
        m = mj - 0.5
        m_up = m
        m_dn = m + 1.0
    elif abs(j - (l - 0.5)) < 1e-12:
        m = mj + 0.5
        m_up = m - 1.0
        m_dn = m
    else:
        raise ValueError("j must be l±1/2")

    # Vectorized gather for rows that map to valid integer m in [-l, l]
    def fill_rows(target: np.ndarray, m_vals: np.ndarray) -> None:
        valid = (m_vals >= -l) & (m_vals <= l) & (np.isclose(m_vals, np.round(m_vals)))
        if not np.any(valid):
            return
        row_idx = np.nonzero(valid)[0]
        src_idx = (m_vals[valid].astype(int) + l)
        target[row_idx, :] = U_l[src_idx, :]

    fill_rows(U_up, m_up)
    fill_rows(U_dn, m_dn)

    return U_up, U_dn


# --------------------------
# f^{σσ'}_{ℓ j m m'} tensor
# --------------------------

def f_blocks_lj(l: int, j: float, U_l: np.ndarray) -> np.ndarray:
    """Compute f^{σσ'}_{ℓ j m m'} as (2,2,2ℓ+1,2ℓ+1) with spin leading dims.

    f^{σσ'}_{m m'} = Σ_{m_j} α^{σ}_{m_j} U^{σ}_{m_j,m} · (α^{σ'}_{m_j} U^{σ'}_{m_j,m'})^*
    with σ=0→↑, 1→↓, m and m' ordered as -ℓ..+ℓ.
    """
    alpha_up, alpha_dn = alpha_sigma_lj(l, j)
    U_up, U_dn = U_sigma_lj(l, j, U_l)  # (2j+1, 2ℓ+1)

    A_up = alpha_up[:, None] * U_up
    A_dn = alpha_dn[:, None] * U_dn

    f_uu = np.einsum("am,an->mn", A_up, np.conjugate(A_up), optimize=True)
    f_ud = np.einsum("am,an->mn", A_up, np.conjugate(A_dn), optimize=True)
    f_du = np.einsum("am,an->mn", A_dn, np.conjugate(A_up), optimize=True)
    f_dd = np.einsum("am,an->mn", A_dn, np.conjugate(A_dn), optimize=True)

    f = np.stack([[f_uu, f_ud], [f_du, f_dd]], axis=0)
    return f  # (2,2,2ℓ+1,2ℓ+1)


# --------------------------
# Assemble E^{I,σσ'}_{ℓ m m'} from channel strengths per (ℓ,j)
# --------------------------

def E_spin_blocks_for_atom_l(l: int, channel_strengths: Dict[float, complex]) -> np.ndarray:
    """Return E^{σσ'}_{ℓ m m'} for a single atom and fixed ℓ.

    channel_strengths: dict keyed by j (float) with scalar complex weight per (ℓ,j).
    For USPP with multiple projectors per (ℓ,j), pass the sum over (τ,τ') already
    combined for the desired contraction; or extend this function to broadcast.

    Output shape: (2,2,2ℓ+1,2ℓ+1)
    """
    U_l = U_complex_from_real(l)
    E = np.zeros((2, 2, 2 * l + 1, 2 * l + 1), dtype=np.complex128)
    for j, w in channel_strengths.items():
        f = f_blocks_lj(l, float(j), U_l)
        E += complex(w) * f
    return E


def E_all_l_for_atom(l_to_channels: Dict[int, Dict[float, complex]]) -> Dict[int, np.ndarray]:
    """Return dict l -> E^{σσ'}_{ℓ m m'} for all ℓ present in l_to_channels."""
    out: Dict[int, np.ndarray] = {}
    for l, chans in l_to_channels.items():
        out[int(l)] = E_spin_blocks_for_atom_l(int(l), chans)
    return out


def _real_m_index_pairs(l: int):
    """Yield (m_label, row_index) matching QE's real-harmonic ordering."""
    yield 0, 0
    for m in range(1, l + 1):
        yield m, 2 * m - 1  # cosine component
        yield -m, 2 * m     # sine component


def real_harmonic_slot(l: int, m_label: int) -> int:
    """Return QE slot index (0..2ℓ) for real harmonic with magnetic number m."""
    if m_label == 0:
        return 0
    if m_label > 0:
        return 2 * m_label - 1
    return -2 * m_label


# Removed: per_type_beta_m_slots (unused)


# ---------------------------------------------------------------------------
# j-RESOLVED vs j-AVERAGED — WHICH PROJECTORS GET BUILT
# ---------------------------------------------------------------------------
#
# A fully-relativistic (FR) UPF carries a SEPARATE radial β for each
# j = ℓ ± 1/2, with a SEPARATE D coefficient.  Two DIFFERENT operators can be
# built from it, and they are not small perturbations of each other:
#
#   j-RESOLVED (``soc=True``)  —  V_NL = Σ_{ℓ j} D_{ℓj} |β_{ℓj}⟩ f^{σσ'}_{ℓj} ⟨β_{ℓj}|
#       Carries spin-orbit.  This is what QE builds when ``lspinorb=.true.``.
#
#   j-AVERAGED (``soc=False``) —  the two j channels of each (ℓ, radial index)
#       are collapsed into ONE scalar-relativistic channel, spin-diagonal, and
#       V_NL becomes  V^scalar ⊗ 1_{2×2}.  Carries NO spin-orbit.  This is what
#       QE builds when ``noncolin=.true., lspinorb=.false.`` — see
#       ``PW/src/average_pp.f90``, which this function reproduces exactly.
#
# ``noncolin`` (⇒ nspinor=2) DOES NOT IMPLY ``lspinorb``.  A QE run with
# ``noncolin=.true., lspinorb=.false.`` writes 2-component spinor wavefunctions
# whose eigenvalues carry NO spin-orbit splitting, because ``average_pp`` ran
# first.  Feeding those wavefunctions a j-resolved V_NL produces an operator
# that splits degeneracies the wavefunctions' own Hamiltonian did not have.
#
# MEASURED, si_cohsex_debug fixture (WFN.h5 ≡ QE run 06_si_4x4x4_nosoc,
# ``<spinorbit>false</spinorbit>``, ``el`` matches to max|Δ| = 0.0 eV over all
# 496 (k,band)):  at Γ the Γ25' manifold is 6-fold degenerate in ``el`` to
# 1e-9 eV AND in QE's own ``kih.dat`` to 0.0000 meV, while LORRAX's
# ``kin_ion`` splits it 2+4 by 47.91 meV — silicon's Δ₀.  That is the
# j-resolved operator acting on j-averaged wavefunctions.
#
# THE ALGEBRA OF THE AVERAGE.  QE's ``average_pp`` collapses the pair
# (β_-, β_+) ≡ (β_{ℓ,ℓ-1/2}, β_{ℓ,ℓ+1/2}) into a single β_avg:
#
#     D_avg  = [ (ℓ+1)·D_+ + ℓ·D_- ] / (2ℓ+1)
#     β_avg  = [ (ℓ+1)·√(D_+/D_avg)·β_+ + ℓ·√(D_-/D_avg)·β_- ] / (2ℓ+1)
#
# so the operator it builds is  |β_avg⟩ D_avg ⟨β_avg| ⊗ 1_spin.  We do NOT
# rebuild the radial tables to form β_avg: keeping BOTH β_± in the basis and
# replacing the (2×2) D sub-block by the RANK-1 matrix
#
#     D'_{rc} = D_avg · u_r u_c ,   u_- = ℓ·√(D_-/D_avg)/(2ℓ+1)
#                                   u_+ = (ℓ+1)·√(D_+/D_avg)/(2ℓ+1)
#
# reproduces  |β_avg⟩ D_avg ⟨β_avg|  IDENTICALLY, because β_avg is by
# construction the linear combination u_-β_- + u_+β_+ scaled by 1.  Nothing
# downstream changes: same radial tables, same projector rows, same
# contraction.  ONLY the E block differs — which is the point: the choice of
# projectors is upstream of the contraction and does not touch it.
#
# ℓ=0 has a single j=1/2 channel and ``f_blocks_lj(0, 0.5)`` is ALREADY the
# spin identity, so the two paths agree there exactly (verified numerically:
# max|E^{↑↓}| = 0 and max|E^{↑↑}−E^{↓↓}| = 0 for ℓ=0 either way).

def pseudo_has_j_channels(pseudo) -> bool:
    """True when this UPF actually resolves j = ℓ±1/2 (fully relativistic).

    Reads the file, not a caller's belief.  Positive only when a β with ℓ>0
    carries an explicit ``jjj`` — that is the ONLY thing that makes a
    j-resolved operator constructible.  ``has_so`` / ``relativistic="full"``
    in PP_HEADER are cross-checked but not trusted alone: the ℓ>0 ``jjj``
    values are what ``build_E_blocks_full`` consumes, so they are what decides.

    Returns FALSE for: a scalar-relativistic UPF, a non-relativistic UPF, and
    an FR UPF with only ℓ=0 projectors (where j resolution is vacuous — the
    j=1/2 block is already the spin identity).
    """
    ppnl = getattr(pseudo, 'pp_nonlocal', None)
    if ppnl is None:
        return False
    for beta in getattr(ppnl, 'pp_beta', []) or []:
        l = int(getattr(beta, 'lll', getattr(beta, 'angular_momentum', 0)))
        if l > 0 and getattr(beta, 'jjj', None) is not None:
            return True
    return False


def pseudo_soc_strength_ry(pseudo) -> float:
    """max over ℓ>0 of |D(ℓ, j=ℓ+1/2) − D(ℓ, j=ℓ−1/2)| in Ry.

    The spin-orbit strength the FR pseudopotential would inject if built
    j-resolved.  Zero means the two j channels are degenerate and the
    j-resolved and j-averaged operators coincide, so the choice cannot matter;
    non-zero is the amount at stake.  Used only for reporting.
    """
    ppnl = getattr(pseudo, 'pp_nonlocal', None)
    if ppnl is None:
        return 0.0
    betas = list(getattr(ppnl, 'pp_beta', []) or [])
    if not betas:
        return 0.0
    try:
        nproj = int(getattr(pseudo.pp_header, 'number_of_proj', len(betas)))
        D = np.asarray(ppnl.pp_dij.value, dtype=np.float64).reshape(nproj, nproj)
    except Exception:
        return 0.0
    groups: Dict[tuple, List[int]] = {}
    for i, beta in enumerate(betas):
        l = int(getattr(beta, 'lll', getattr(beta, 'angular_momentum', 0)))
        j = getattr(beta, 'jjj', None)
        if l > 0 and j is not None:
            groups.setdefault((l, float(j)), []).append(i)
    worst = 0.0
    for (l, j), idx in groups.items():
        partner = groups.get((l, 2 * l - j))  # j ↔ 2ℓ − j maps ℓ±1/2 to ℓ∓1/2
        if partner is None or len(partner) != len(idx):
            continue
        for a, b in zip(idx, partner):
            worst = max(worst, abs(float(D[a, a]) - float(D[b, b])))
    return worst


def _spin_identity(msize: int) -> np.ndarray:
    """δ_{σσ'} ⊗ 1_{msize}, shaped (2, 2, msize, msize)."""
    ident = np.zeros((2, 2, msize, msize), dtype=np.complex128)
    ident[0, 0] = np.eye(msize)
    ident[1, 1] = np.eye(msize)
    return ident


def _pair_j_channels(l: int, entries: List[tuple[int, object]]) -> List[tuple]:
    """Pair the β of an ℓ>0 shell into (idx_minus, idx_plus) j = ℓ∓1/2 pairs.

    Mirrors ``average_pp.f90``'s consecutive-index walk, including its two
    ``errore('average_pp','wrong beta functions')`` refusals: an FR UPF must
    present the two j channels of each radial index adjacently.  Raises rather
    than guessing — a UPF that does not pair is one whose scalar-relativistic
    reduction we cannot define, and silently picking one is the failure mode
    this whole module exists to remove.
    """
    pairs: List[tuple] = []
    i = 0
    n = len(entries)
    while i < n:
        gi, bi = entries[i]
        ji = float(getattr(bi, 'jjj', l + 0.5))
        if i + 1 >= n:
            raise RuntimeError(
                f"j-average: ℓ={l} has an ODD number of β functions "
                f"({n}); a fully-relativistic UPF must supply both "
                f"j = ℓ±1/2 for every radial index.")
        gj, bj = entries[i + 1]
        jj = float(getattr(bj, 'jjj', l + 0.5))
        if abs(ji - (l - 0.5)) < 1e-7 and abs(jj - (l + 0.5)) < 1e-7:
            pairs.append((gi, gj))
        elif abs(ji - (l + 0.5)) < 1e-7 and abs(jj - (l - 0.5)) < 1e-7:
            pairs.append((gj, gi))
        else:
            raise RuntimeError(
                f"j-average: β pair ({i}, {i+1}) of ℓ={l} has "
                f"j = ({ji}, {jj}); expected {{{l-0.5}, {l+0.5}}}.  "
                f"This is average_pp.f90's 'wrong beta functions'.")
        i += 2
    return pairs


def build_E_blocks_full(pseudo, *, soc: bool = True) -> Dict[int, np.ndarray]:
    """Assemble full E^{σσ'} blocks per ℓ retaining all radial projectors.

    Parameters
    ----------
    pseudo : parsed UPF object.
    soc : bool, default True
        ``True``  — j-RESOLVED projectors: the operator carries spin-orbit.
                    Use for wavefunctions from ``lspinorb=.true.``.
        ``False`` — j-AVERAGED projectors: scalar-relativistic V_NL ⊗ 1_spin,
                    reproducing QE ``average_pp``.  Use for wavefunctions from
                    ``noncolin=.true., lspinorb=.false.``.

        The default preserves the historical behaviour EXACTLY for FR UPFs.
        It is a default, not a detection: the caller
        (``vnl_ops.build_vnl_setup``) is responsible for establishing which
        one the wavefunctions actually want and for announcing when it cannot.
    """
    ppnl = getattr(pseudo, 'pp_nonlocal', None)
    if ppnl is None:
        return {}
    betas = list(getattr(ppnl, 'pp_beta', []))
    if not betas:
        return {}

    try:
        D_mat = np.asarray(ppnl.pp_dij.value, dtype=np.complex128)
        nproj = int(getattr(pseudo.pp_header, 'number_of_proj', len(betas)))
        D_mat = D_mat.reshape(nproj, nproj)
    except Exception as exc:  # pragma: no cover - propagate meaningful error
        raise RuntimeError("Pseudopotential missing PP_DIJ matrix") from exc

    has_j = pseudo_has_j_channels(pseudo)

    per_l: Dict[int, List[tuple[int, object]]] = {}
    for idx, beta in enumerate(betas, start=1):
        l = int(getattr(beta, 'lll', getattr(beta, 'angular_momentum', 0)))
        per_l.setdefault(l, []).append((idx - 1, beta))

    U_cache: Dict[int, np.ndarray] = {}
    result: Dict[int, np.ndarray] = {}

    for l, entries in per_l.items():
        msize = 2 * l + 1
        n_beta_l = len(entries)
        size = n_beta_l * msize
        E = np.zeros((2, 2, size, size), dtype=np.complex128)

        order_map = {global_idx: pos for pos, (global_idx, _) in enumerate(entries)}

        # ── NO j information in the UPF (scalar-relativistic pseudo) ──
        # The historical code fell through to ``getattr(beta,'jjj', l+0.5)``,
        # i.e. it built a PURE j=ℓ+1/2 channel: spin-mixing, m-dependent, and
        # carrying only (2j+1)/(2(2ℓ+1)) of the channel's trace for ℓ>0.  The
        # correct operator for a pseudo with no j resolution is the spin
        # identity, and that is what is built here regardless of ``soc`` —
        # there are no j channels to resolve.
        if not has_j:
            ident = _spin_identity(msize)
            for g_r in (g for g, _ in entries):
                pos_r = order_map[g_r]
                slice_r = slice(pos_r * msize, (pos_r + 1) * msize)
                for g_c in (g for g, _ in entries):
                    pos_c = order_map[g_c]
                    slice_c = slice(pos_c * msize, (pos_c + 1) * msize)
                    E[:, :, slice_r, slice_c] += D_mat[g_r, g_c] * ident
            result[l] = E
            continue

        # ── j-AVERAGED: QE average_pp, expressed as a rank-1 D sub-block ──
        if not soc:
            ident = _spin_identity(msize)
            if l == 0:
                # single j=1/2 channel; f_blocks_lj(0, 0.5) IS the identity,
                # so nothing to average — keep D as it stands.
                for g_r in (g for g, _ in entries):
                    pos_r = order_map[g_r]
                    slice_r = slice(pos_r * msize, (pos_r + 1) * msize)
                    for g_c in (g for g, _ in entries):
                        pos_c = order_map[g_c]
                        slice_c = slice(pos_c * msize, (pos_c + 1) * msize)
                        E[:, :, slice_r, slice_c] += D_mat[g_r, g_c] * ident
            else:
                for g_minus, g_plus in _pair_j_channels(l, entries):
                    d_m = complex(D_mat[g_minus, g_minus]).real
                    d_p = complex(D_mat[g_plus, g_plus]).real
                    d_avg = ((l + 1.0) * d_p + l * d_m) / (2.0 * l + 1.0)
                    if d_avg == 0.0:
                        raise RuntimeError(
                            f"j-average: ℓ={l} channel has D_avg = 0; "
                            f"average_pp's √(D/D_avg) is undefined.")
                    # QE's √(D_j / D_avg) is well-defined only when both j
                    # channels share the sign of their weighted mean (true
                    # for e.g. the negative-D ℓ=2 channels of ONCV FR Si/Mo).
                    # It is FALSE for the first ℓ=1 shell of the PseudoDojo
                    # FR 5d pseudos (Au/Hf/Hg/Ir/Os/Re: D_jj of opposite
                    # sign, Au −0.4798/+2.5893 Ry) — np.sqrt(negative)
                    # returned silent NaN here until 2026-08-28.  QE's own
                    # average_pp.f90:71-80 carries the same sqrt, so such a
                    # pseudo is not j-averageable in QE either: refuse.
                    if d_m / d_avg < 0.0 or d_p / d_avg < 0.0:
                        raise ValueError(
                            f"j-average: ℓ={l} channel pair has opposite-sign "
                            f"D_jj (D_j- = {d_m:.6f}, D_j+ = {d_p:.6f} Ry, "
                            f"D_avg = {d_avg:.6f}); average_pp's √(D/D_avg) "
                            f"is imaginary, so this fully-relativistic "
                            f"pseudopotential cannot be j-averaged (QE's "
                            f"average_pp would NaN identically).  got: "
                            f"soc=False with a non-averageable FR pseudo; "
                            f"want: soc=True with an nspinor=2 WFN, or a "
                            f"scalar-relativistic (nc-sr) pseudopotential "
                            f"for this element.")
                    u = {g_minus: l * np.sqrt(d_m / d_avg) / (2.0 * l + 1.0),
                         g_plus: (l + 1.0) * np.sqrt(d_p / d_avg) / (2.0 * l + 1.0)}
                    for g_r in (g_minus, g_plus):
                        pos_r = order_map[g_r]
                        slice_r = slice(pos_r * msize, (pos_r + 1) * msize)
                        for g_c in (g_minus, g_plus):
                            pos_c = order_map[g_c]
                            slice_c = slice(pos_c * msize, (pos_c + 1) * msize)
                            E[:, :, slice_r, slice_c] += (
                                d_avg * u[g_r] * u[g_c]) * ident
            result[l] = E
            continue

        # ── j-RESOLVED (historical default, bit-identical) ──
        j_groups: Dict[float, List[int]] = {}
        for global_idx, beta in entries:
            j_val = float(getattr(beta, 'jjj', l + 0.5))
            j_groups.setdefault(j_val, []).append(global_idx)

        U_l = U_cache.setdefault(l, U_complex_from_real(l))

        for j_val, global_indices in j_groups.items():
            f = f_blocks_lj(l, j_val, U_l)
            idxs = np.asarray(global_indices, dtype=int)
            D_sub = D_mat[np.ix_(idxs, idxs)]
            for r_local, g_r in enumerate(global_indices):
                pos_r = order_map[g_r]
                slice_r = slice(pos_r * msize, (pos_r + 1) * msize)
                for c_local, g_c in enumerate(global_indices):
                    pos_c = order_map[g_c]
                    slice_c = slice(pos_c * msize, (pos_c + 1) * msize)
                    E[:, :, slice_r, slice_c] += D_sub[r_local, c_local] * f

        result[l] = E

    return result


def compute_type_projectors_real(
    pseudo,
    atom_indices: List[int],
    atom_tau: List[np.ndarray],
    K_crys_total: np.ndarray,
    K_cart_total: np.ndarray,
    cell_volume: float,
    Y_real_by_l_pre: Dict[int, np.ndarray] | None = None,
    F_splines: Dict[Tuple[int, int], RadialTable] | None = None,
    K_norm_in: np.ndarray | None = None,
):
    """Compute QE real-harmonic projector rows with structure factors.

    Returns (rows, meta) where rows has shape (nrows, nG) and meta is a list of
    dictionaries describing each row with keys: atom, beta_index, l, j, m,
    row_index.
    """

    K_cart = np.asarray(K_cart_total, dtype=float)
    K_crys = np.asarray(K_crys_total, dtype=float)
    nG = K_cart.shape[0]
    K_norm = np.sqrt(np.sum(K_cart ** 2, axis=1))

    betas = list(getattr(getattr(pseudo, 'pp_nonlocal', None), 'pp_beta', []))
    if not betas:
        return np.zeros((0, nG), dtype=np.complex128), []

    lmax = max(int(getattr(beta, 'lll', getattr(beta, 'angular_momentum', 0))) for beta in betas)
    if Y_real_by_l_pre is not None:
        Y_real_by_l = Y_real_by_l_pre
    else:
        Y_real_by_l = {l: qe_real_sph_harmonics(l, K_cart) for l in range(lmax + 1)}
    if F_splines is not None and K_norm_in is not None:
        Fdict: Dict[Tuple[int, int], np.ndarray] = {}
        for idx, beta in enumerate(betas, start=1):
            l = int(getattr(beta, 'lll', getattr(beta, 'angular_momentum', 0)))
            spl = F_splines.get((l, idx))
            if spl is None:
                continue
            Fdict[(l, idx)] = spl(K_norm_in)
    else:
        # Local computation of radial form factors if no precomputed splines provided
        Fdict = _form_factors_qe(pseudo, K_norm, float(cell_volume))
    prefactor = (4.0 * np.pi) / np.sqrt(float(cell_volume))

    rows: List[np.ndarray] = []
    meta: List[dict] = []

    for beta_idx, beta in enumerate(betas, start=1):
        l = int(getattr(beta, 'lll', getattr(beta, 'angular_momentum', 0)))
        j_val = float(getattr(beta, 'jjj', l + 0.5))
        F_on_K = Fdict.get((l, beta_idx))
        if F_on_K is None:
            continue
        radial = prefactor * (1j) ** l * F_on_K
        Y_real = Y_real_by_l[l]

        for m_label, row_idx in _real_m_index_pairs(l):
            angular = Y_real[row_idx, :]
            base_row = radial * angular
            for atom_id, tau in zip(atom_indices, atom_tau):
                phase = np.exp(-2j * np.pi * (K_crys @ np.asarray(tau)))
                row = base_row * phase
                rows.append(row)
                meta.append({
                    'atom': atom_id,
                    'beta_index': beta_idx,
                    'l': l,
                    'j': j_val,
                    'm': m_label,
                    'row_index': len(rows) - 1,
                })

    if rows:
        projector_array = np.vstack(rows).astype(np.complex128, copy=False)
    else:
        projector_array = np.zeros((0, nG), dtype=np.complex128)

    return projector_array, meta


def precompute_projector_tables(
    pseudo,
    cell_volume: float,
    q_max: float,
    q_points: int | None = None,
) -> Dict[Tuple[int, int], RadialTable]:
    """Precompute uniformly tabulated projector form factors F_l(q).

    Returns dict keyed by ``(l, beta_index)`` mapping to ``RadialTable``.
    The function name is kept for compatibility with older callers.
    """
    ppmesh = getattr(pseudo, 'pp_mesh', None)
    ppnl = getattr(pseudo, 'pp_nonlocal', None)
    if ppmesh is None or ppnl is None:
        return {}
    r = np.asarray(ppmesh.pp_r.value, dtype=float)
    try:
        rab = np.asarray(ppmesh.pp_rab.value, dtype=float)
    except Exception:
        rab = None
    betas = list(getattr(ppnl, 'pp_beta', []))
    if not betas:
        return {}
    if q_points is None:
        q_points = max(2048, 2 * len(r))
    q_grid = make_uniform_q_grid(float(q_max), q_points)
    kkb, weights = _qe_beta_radial_scheme(r, rab, _kkbeta_of(betas, len(r)))
    tables: Dict[Tuple[int, int], RadialTable] = {}
    for idx, beta in enumerate(betas, start=1):
        l = int(getattr(beta, 'lll', getattr(beta, 'angular_momentum', 0)))
        raw_beta = np.asarray(beta.value, dtype=float)
        if r[0] == 0.0 and len(r) > 1:
            beta_r = np.empty_like(r)
            beta_r[1:] = raw_beta[1:] / r[1:]
            beta_r[0] = beta_r[1]
        else:
            beta_r = raw_beta / r
        tables[(l, idx)] = make_projector_table(l, r[:kkb], beta_r[:kkb],
                                                q_grid, weights)
    return tables


def precompute_projector_splines(
    pseudo,
    cell_volume: float,
    q_max: float,
    q_points: int | None = None,
) -> Dict[Tuple[int, int], RadialTable]:
    """Compatibility wrapper returning radial tables, not SciPy splines."""
    return precompute_projector_tables(pseudo, cell_volume, q_max, q_points)


__all__ = [
    "m_to_idx",
    "mj_grid",
    "U_complex_from_real",
    "alpha_sigma_lj",
    "U_sigma_lj",
    "f_blocks_lj",
    "E_spin_blocks_for_atom_l",
    "E_all_l_for_atom",
    "qe_real_sph_harmonics",
    "qe_real_sph_harmonics_with_grad",
    "real_harmonic_slot",
    "build_E_blocks_full",
    "pseudo_has_j_channels",
    "pseudo_soc_strength_ry",
    "compute_type_projectors_real",
    "precompute_projector_tables",
    "precompute_projector_splines",
    "build_local_ionic_potential_on_G_total",
]

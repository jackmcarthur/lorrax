"""psp/xc.py — Exchange-correlation potentials via autodiff.

Computes V_xc on the real-space grid for any functional that maps
density quantities → energy per electron.  The functional is a callable:

  LDA:       eps_xc(rho)             → ε_xc  (energy/electron, Ry)
  GGA:       eps_xc(rho, sigma)      → ε_xc
  meta-GGA:  eps_xc(rho, sigma, tau) → ε_xc

V_xc is obtained by autodiff of E_xc = Σ ρ · ε_xc w.r.t. each input,
plus the GGA divergence correction for gradient-dependent terms.
All functionals go through one code path.

Usage
-----
    from psp.xc import compute_V_xc, pbe_functional
    V_xc = compute_V_xc(rho_total, rho_G_total, G_cart, pbe_functional)
"""
from __future__ import annotations

from enum import Enum
from typing import Callable

import jax
import jax.numpy as jnp


# ═══════════════════════════════════════════════════════════════════════
#  Functional registry
# ═══════════════════════════════════════════════════════════════════════

class XCLevel(Enum):
    """What input quantities the functional depends on."""
    LDA = "lda"           # ε(ρ)
    GGA = "gga"           # ε(ρ, σ)         σ = |∇ρ|²
    MGGA = "mgga"         # ε(ρ, σ, τ)      τ = ½Σ|∇ψ_i|²


def pbe_functional():
    """PBE GGA functional.  Returns (eps_xc_fn, XCLevel.GGA).

    eps_xc_fn(rho, sigma) → energy per electron in Ry.
    """
    from jax_xc_local.pbe import pbe_xc
    return pbe_xc, XCLevel.GGA


# ═══════════════════════════════════════════════════════════════════════
#  Compute input quantities from density
# ═══════════════════════════════════════════════════════════════════════

def _compute_sigma(rho_G_total, G_cart):
    """σ = |∇ρ|² via G-space derivatives."""
    sigma = jnp.zeros(rho_G_total.shape, dtype=jnp.float64)
    for i in range(3):
        drho = jnp.real(jnp.fft.ifftn(1j * G_cart[..., i] * rho_G_total))
        sigma = sigma + drho ** 2
    return jnp.maximum(sigma, 0.0)


def _compute_grad_components(rho_G_total, G_cart):
    """∂ρ/∂r_i for each Cartesian direction.  Returns list of 3 arrays."""
    return [jnp.real(jnp.fft.ifftn(1j * G_cart[..., i] * rho_G_total))
            for i in range(3)]


# ═══════════════════════════════════════════════════════════════════════
#  V_xc via autodiff — one function for all levels
# ═══════════════════════════════════════════════════════════════════════

def compute_V_xc(
    rho_total: jax.Array,
    rho_G_total: jax.Array,
    G_cart: jax.Array,
    xc_fn: Callable,
    level: XCLevel = XCLevel.GGA,
) -> jax.Array:
    """Compute V_xc(r) on the FFT grid via autodiff.

    Parameters
    ----------
    rho_total : (nx, ny, nz) total electron density (valence + core)
    rho_G_total : (nx, ny, nz) complex — G-space density (with precise core)
    G_cart : (nx, ny, nz, 3) Cartesian G-vectors
    xc_fn : callable matching the level:
        LDA:  xc_fn(rho) → eps_xc
        GGA:  xc_fn(rho, sigma) → eps_xc
        MGGA: xc_fn(rho, sigma, tau) → eps_xc
    level : XCLevel enum

    Returns
    -------
    V_xc : (nx, ny, nz) in Ry
    """
    rho = jnp.maximum(rho_total, 1e-10)

    if level == XCLevel.LDA:
        return _vxc_lda(rho, xc_fn)
    elif level == XCLevel.GGA:
        sigma = _compute_sigma(rho_G_total, G_cart)
        return _vxc_gga(rho, rho_total, sigma, rho_G_total, G_cart, xc_fn)
    elif level == XCLevel.MGGA:
        sigma = _compute_sigma(rho_G_total, G_cart)
        # tau placeholder — needs wavefunctions, not yet wired
        tau = jnp.zeros_like(rho)
        return _vxc_mgga(rho, rho_total, sigma, tau, rho_G_total, G_cart, xc_fn)
    else:
        raise ValueError(f"Unknown XC level: {level}")


# ═══════════════════════════════════════════════════════════════════════
#  Level-specific V_xc implementations
# ═══════════════════════════════════════════════════════════════════════

def _vxc_lda(rho, xc_fn):
    """V_xc = d(ρ·ε)/dρ for LDA."""
    def E_xc(r):
        return jnp.sum(r * xc_fn(r))
    return jax.grad(E_xc)(rho)


def _vxc_gga(rho, rho_raw, sigma, rho_G, G_cart, xc_fn):
    """V_xc = d(ρ·ε)/dρ − 2∇·(d(ρ·ε)/dσ · ∇ρ) for GGA."""
    def E_xc(r, s):
        return jnp.sum(r * xc_fn(r, s))

    # LDA part (σ=0 baseline for masking)
    def E_lda(r):
        return jnp.sum(r * xc_fn(r, jnp.zeros_like(r)))

    df_drho_lda = jax.grad(E_lda)(rho)
    df_drho_full = jax.grad(E_xc, argnums=0)(rho, sigma)
    df_dsigma = jax.grad(E_xc, argnums=1)(rho, sigma)

    # Mask: fall back to LDA where density/gradient is negligible
    mask = (rho_raw > 1e-6) & (sigma > 1e-10)
    df_drho = df_drho_lda + jnp.where(mask, df_drho_full - df_drho_lda, 0.0)
    df_dsigma = jnp.where(mask, df_dsigma, 0.0)

    # GGA divergence: −2 ∇·(df/dσ · ∇ρ)
    div = jnp.zeros_like(rho)
    for i in range(3):
        drho_i = jnp.real(jnp.fft.ifftn(1j * G_cart[..., i] * rho_G))
        h_G = jnp.fft.fftn(df_dsigma * drho_i)
        div = div + jnp.real(jnp.fft.ifftn(1j * G_cart[..., i] * h_G))

    return df_drho - 2.0 * div


def _vxc_mgga(rho, rho_raw, sigma, tau, rho_G, G_cart, xc_fn):
    """V_xc for meta-GGA: adds dE/dτ term (placeholder)."""
    def E_xc(r, s, t):
        return jnp.sum(r * xc_fn(r, s, t))

    df_drho = jax.grad(E_xc, argnums=0)(rho, sigma, tau)
    df_dsigma = jax.grad(E_xc, argnums=1)(rho, sigma, tau)
    df_dtau = jax.grad(E_xc, argnums=2)(rho, sigma, tau)

    # GGA divergence (same as GGA)
    div = jnp.zeros_like(rho)
    for i in range(3):
        drho_i = jnp.real(jnp.fft.ifftn(1j * G_cart[..., i] * rho_G))
        h_G = jnp.fft.fftn(df_dsigma * drho_i)
        div = div + jnp.real(jnp.fft.ifftn(1j * G_cart[..., i] * h_G))

    # meta-GGA: V_xc += dE/dτ (applied to KE density, needs −½∇² on ψ)
    # For now this is the potential part; the τ-dependent Hamiltonian
    # contribution (non-multiplicative) would need to be wired separately.
    return df_drho - 2.0 * div + df_dtau


# ═══════════════════════════════════════════════════════════════════════
#  Spin-polarized PBE — for the xc magnetic field B_xc on magnetic systems
# ═══════════════════════════════════════════════════════════════════════
#
#  Magnetic / noncollinear systems need a SPIN-dependent V_xc.  At each grid
#  point the local spin frame gives n↑=(n+|m|)/2, n↓=(n−|m|)/2; the
#  spin-resolved potentials V↑,V↓ assemble into V_xc = V̄ I + B (m̂·σ) with
#  V̄=(V↑+V↓)/2, B=(V↑−V↓)/2 (the xc magnetic field).  Spin-unpolarized V_xc
#  (everything above) is the |m|=0 limit, exact for non-magnetic systems.
#
#  References: spin-scaled PBE exchange E_x[n↑,n↓]=½(E_x[2n↑]+E_x[2n↓])
#  (Perdew-Burke-Ernzerhof, PRL 77, 3865 (1996), Eq.10); PW92 spin-
#  interpolated correlation (Perdew-Wang, PRB 45, 13244 (1992), Table I) +
#  the φ(ζ)-rescaled PBE gradient term (PBE96, Eq.7,8).


def _pbe_x_spin(n_up, n_dn, sigma_uu, sigma_dd):
    """Spin-polarized PBE exchange ε_x per electron (Hartree).

    Exact spin-scaling E_x[n↑,n↓]=½(E_x[2n↑]+E_x[2n↓]): reuse the unpolarized
    ``pbe_x`` per channel with DOUBLED density and DOUBLED-magnitude gradient
    (cross gradient σ_ud does not enter — exchange is spin-diagonal).
    """
    from jax_xc_local.pbe import pbe_x
    n = jnp.maximum(n_up + n_dn, 1e-30)
    ex_up = pbe_x(2.0 * n_up, 4.0 * sigma_uu)
    ex_dn = pbe_x(2.0 * n_dn, 4.0 * sigma_dd)
    return (n_up * ex_up + n_dn * ex_dn) / n


def _pbe_c_spin(n_up, n_dn, sigma_total):
    """Spin-polarized PBE correlation ε_c per electron (Hartree).

    PW92 spin interpolation ε_c(rs,ζ) + the φ(ζ)-rescaled PBE gradient term H.
    Reduces to ``jax_xc_local.pbe.pbe_c(n, σ)`` at n↑=n↓ (ζ=0).
    σ_total = |∇(n↑+n↓)|².
    """
    n = jnp.maximum(n_up + n_dn, 1e-30)
    zeta = jnp.clip((n_up - n_dn) / n, -1.0 + 1e-12, 1.0 - 1e-12)
    rs = (3.0 / (4.0 * jnp.pi * n)) ** (1.0 / 3.0)
    rs_h = jnp.sqrt(rs)

    def G(A, a1, b1, b2, b3, b4):                # PW92 Eq.10 (returns < 0)
        return -2.0 * A * (1.0 + a1 * rs) * jnp.log(
            1.0 + 1.0 / (2.0 * A * (b1 * rs_h + b2 * rs
                                    + b3 * rs * rs_h + b4 * rs * rs)))
    # PW92 Table I parametrisations (== QE Modules/funct pw_spin)
    Gu = G(0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294)   # ζ=0 (unpol)
    Gp = G(0.015545, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517)  # ζ=1 (ferro)
    Ga = G(0.016887, 0.11125, 10.357, 3.6231, 0.88026, 0.49671)  # −α_c (stiffness)

    c = 0.5198420997897464                       # 2^(4/3) − 2
    fz0 = 1.709920934161365                      # f''(0) = 8/(9c)
    z4 = zeta ** 4
    f = ((1.0 + zeta) ** (4.0 / 3.0) + (1.0 - zeta) ** (4.0 / 3.0) - 2.0) / c
    ec = Gu * (1.0 - f * z4) + Gp * (f * z4) - Ga * (f * (1.0 - z4) / fz0)

    phi = 0.5 * ((1.0 + zeta) ** (2.0 / 3.0) + (1.0 - zeta) ** (2.0 / 3.0))
    phi3 = phi ** 3
    gamma = 0.03109069086965489503               # (1−ln2)/π²
    beta = 0.06672455060314922
    kf = (3.0 * jnp.pi ** 2 * n) ** (1.0 / 3.0)
    ks = jnp.sqrt(4.0 * kf / jnp.pi)
    t2 = sigma_total / ((2.0 * phi * ks * n) ** 2 + 1e-60)
    A_pbe = (beta / gamma) / (jnp.exp(-ec / (gamma * phi3)) - 1.0 + 1e-60)
    At2 = A_pbe * t2
    H = gamma * phi3 * jnp.log(
        1.0 + beta / gamma * t2 * (1.0 + At2) / (1.0 + At2 + At2 ** 2))
    return ec + H


def pbe_xc_spin(rho_up, rho_dn, sigma_uu, sigma_ud, sigma_dd):
    """Spin-polarized PBE ε_xc per electron, in Ry (×2 Hartree).

    σ_uu=|∇n↑|², σ_ud=∇n↑·∇n↓, σ_dd=|∇n↓|²; the correlation uses the total
    σ_total = σ_uu + 2σ_ud + σ_dd = |∇(n↑+n↓)|².  Reduces EXACTLY to
    ``pbe_xc(n, σ)`` at rho_up==rho_dn — the non-magnetic bit-identity gate.
    """
    sigma_total = sigma_uu + 2.0 * sigma_ud + sigma_dd
    return 2.0 * (_pbe_x_spin(rho_up, rho_dn, sigma_uu, sigma_dd)
                  + _pbe_c_spin(rho_up, rho_dn, sigma_total))


def _compute_sigma_spin(rho_G_up, rho_G_dn, G_cart):
    """(σ_uu, σ_ud, σ_dd) = (|∇n↑|², ∇n↑·∇n↓, |∇n↓|²) via G-space derivatives."""
    suu = jnp.zeros(rho_G_up.shape, dtype=jnp.float64)
    sud = jnp.zeros_like(suu)
    sdd = jnp.zeros_like(suu)
    for i in range(3):
        du = jnp.real(jnp.fft.ifftn(1j * G_cart[..., i] * rho_G_up))
        dd = jnp.real(jnp.fft.ifftn(1j * G_cart[..., i] * rho_G_dn))
        suu = suu + du * du
        sud = sud + du * dd
        sdd = sdd + dd * dd
    return jnp.maximum(suu, 0.0), sud, jnp.maximum(sdd, 0.0)


def compute_V_xc_spin(rho_up, rho_dn, rho_G_up, rho_G_dn, G_cart):
    """Spin-resolved GGA V_xc: returns (V_up, V_dn) on the FFT grid in Ry.

    V_s = ∂(nε)/∂n_s − ∇·[∂(nε)/∂(∇n_s)], with the GGA flux
        ∂(nε)/∂(∇n↑) = 2(∂nε/∂σ_uu)∇n↑ + (∂nε/∂σ_ud)∇n↓     (↑↔↓ for n↓).
    Spin generalisation of ``_vxc_gga`` (same LDA-fallback mask); reduces to
    ``compute_V_xc`` at n↑=n↓.

    Parameters
    ----------
    rho_up, rho_dn   : (nx,ny,nz) spin densities (valence + half the core)
    rho_G_up, rho_G_dn : (nx,ny,nz) complex — per-spin G-space density (precise core)
    G_cart           : (nx,ny,nz,3) Cartesian G-vectors
    """
    nu = jnp.maximum(rho_up, 1e-10)
    nd = jnp.maximum(rho_dn, 1e-10)
    suu, sud, sdd = _compute_sigma_spin(rho_G_up, rho_G_dn, G_cart)

    def E(u, d, auu, aud, add):
        return jnp.sum((u + d) * pbe_xc_spin(u, d, auu, aud, add))

    def E_lda(u, d):
        z = jnp.zeros_like(u)
        return jnp.sum((u + d) * pbe_xc_spin(u, d, z, z, z))

    df_du_lda = jax.grad(E_lda, 0)(nu, nd)
    df_dd_lda = jax.grad(E_lda, 1)(nu, nd)
    df_du = jax.grad(E, 0)(nu, nd, suu, sud, sdd)
    df_dd = jax.grad(E, 1)(nu, nd, suu, sud, sdd)
    df_duu = jax.grad(E, 2)(nu, nd, suu, sud, sdd)
    df_dud = jax.grad(E, 3)(nu, nd, suu, sud, sdd)
    df_ddd = jax.grad(E, 4)(nu, nd, suu, sud, sdd)

    # LDA fallback where density / gradient negligible (mirror _vxc_gga)
    rho_raw = rho_up + rho_dn
    sigma_tot = suu + 2.0 * sud + sdd
    mask = (rho_raw > 1e-6) & (sigma_tot > 1e-10)
    df_du = df_du_lda + jnp.where(mask, df_du - df_du_lda, 0.0)
    df_dd = df_dd_lda + jnp.where(mask, df_dd - df_dd_lda, 0.0)
    df_duu = jnp.where(mask, df_duu, 0.0)
    df_dud = jnp.where(mask, df_dud, 0.0)
    df_ddd = jnp.where(mask, df_ddd, 0.0)

    gnu = [jnp.real(jnp.fft.ifftn(1j * G_cart[..., i] * rho_G_up)) for i in range(3)]
    gnd = [jnp.real(jnp.fft.ifftn(1j * G_cart[..., i] * rho_G_dn)) for i in range(3)]

    def div(fields):
        out = jnp.zeros_like(nu)
        for i in range(3):
            hG = jnp.fft.fftn(fields[i])
            out = out + jnp.real(jnp.fft.ifftn(1j * G_cart[..., i] * hG))
        return out

    V_up = df_du - div([2.0 * df_duu * gnu[i] + df_dud * gnd[i] for i in range(3)])
    V_dn = df_dd - div([2.0 * df_ddd * gnd[i] + df_dud * gnu[i] for i in range(3)])
    return V_up, V_dn

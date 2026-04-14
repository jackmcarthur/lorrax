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

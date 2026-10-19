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

from common.fft_helpers import local_fftn3, local_ifftn3  # 3-D fields: all three axes


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
    # The upstream public factories take a callable spatial density.  We
    # already have rho and sigma from the periodic FFT grid, so use its
    # generated scalar kernels and vectorize over grid points here.
    from jax_xc.impl import gga_x_pbe, gga_c_pbe
    from jax_xc.utils import get_p

    exchange = get_p("gga_x_pbe", False)
    correlation = get_p("gga_c_pbe", False)

    def scalar_eps(rho, sigma):
        return 2.0 * (gga_x_pbe.unpol(exchange, rho, sigma)
                      + gga_c_pbe.unpol(correlation, rho, sigma))

    def eps_xc(rho, sigma):
        rho, sigma = jnp.broadcast_arrays(rho, sigma)
        return jax.vmap(scalar_eps)(rho.reshape(-1), sigma.reshape(-1)).reshape(rho.shape)

    return eps_xc, XCLevel.GGA


def pbe_functional_polarized():
    """Spin-polarized PBE.  Returns ``eps_xc(rho_up, rho_dn, s_uu, s_ud, s_dd)``
    in Ry per electron, ``s_ab = ∇ρ_a·∇ρ_b`` (libxc's σ ordering)."""
    from jax_xc.impl import gga_x_pbe, gga_c_pbe
    from jax_xc.utils import get_p

    exchange = get_p("gga_x_pbe", True)
    correlation = get_p("gga_c_pbe", True)

    def scalar_eps(ru, rd, suu, sud, sdd):
        r, sig = (ru, rd), (suu, sud, sdd)
        return 2.0 * (gga_x_pbe.pol(exchange, r, sig)
                      + gga_c_pbe.pol(correlation, r, sig))

    def eps_xc(ru, rd, suu, sud, sdd):
        args = jnp.broadcast_arrays(ru, rd, suu, sud, sdd)
        flat = [a.reshape(-1) for a in args]
        return jax.vmap(scalar_eps)(*flat).reshape(args[0].shape)

    return eps_xc


# ═══════════════════════════════════════════════════════════════════════
#  Compute input quantities from density
# ═══════════════════════════════════════════════════════════════════════

def _compute_sigma(rho_G_total, G_cart):
    """σ = |∇ρ|² via G-space derivatives."""
    sigma = jnp.zeros(rho_G_total.shape, dtype=jnp.float64)
    for i in range(3):
        drho = jnp.real(local_ifftn3(1j * G_cart[..., i] * rho_G_total))
        sigma = sigma + drho ** 2
    return jnp.maximum(sigma, 0.0)


def _compute_grad_components(rho_G_total, G_cart):
    """∂ρ/∂r_i for each Cartesian direction.  Returns list of 3 arrays."""
    return [jnp.real(local_ifftn3(1j * G_cart[..., i] * rho_G_total))
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
        drho_i = jnp.real(local_ifftn3(1j * G_cart[..., i] * rho_G))
        h_G = local_fftn3(df_dsigma * drho_i)
        div = div + jnp.real(local_ifftn3(1j * G_cart[..., i] * h_G))

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
        drho_i = jnp.real(local_ifftn3(1j * G_cart[..., i] * rho_G))
        h_G = local_fftn3(df_dsigma * drho_i)
        div = div + jnp.real(local_ifftn3(1j * G_cart[..., i] * h_G))

    # meta-GGA: V_xc += dE/dτ (applied to KE density, needs −½∇² on ψ)
    # For now this is the potential part; the τ-dependent Hamiltonian
    # contribution (non-multiplicative) would need to be wired separately.
    return df_drho - 2.0 * div + df_dtau


# ═══════════════════════════════════════════════════════════════════════
#  Noncollinear magnetic V_xc: v δ_αβ + B·σ_αβ (QE's general branch)
# ═══════════════════════════════════════════════════════════════════════

def compute_V_xc_noncollinear(rho_total, rho_G_total, mag, G_cart, xc_fn,
                              ecutrho):
    """V_xc^{αβ}(r) = v(r) δ_αβ + B(r)·σ_αβ from ρ and m for a spin-polarized GGA.

    The local-frame construction QE uses with ``lsign = .false.`` (the general
    branch; ``skills/build_inputs`` magnetic recipe): at each r the
    functional is evaluated on

        ρ_↑,↓ = (ρ ± |m|)/2,        ∇ρ_↑,↓ = (∇ρ ± ∇|m|)/2,

    with ρ = ρ_val + ρ_core (the core enters ρ only) and ∇ taken on the FFT
    grid, ∇ρ from ``rho_G_total`` (analytic core) and ∇|m| from the FFT of
    the |m| field.  With f = ρ ε_xc and the GGA divergence as in
    :func:`_vxc_gga`,

        v_σ = ∂f/∂ρ_σ − ∇·(2 ∂f/∂σ_σσ ∇ρ_σ + ∂f/∂σ_↑↓ ∇ρ_σ̄),
        v = (v_↑ + v_↓)/2,     B = (v_↑ − v_↓)/2 · m/|m|,

    i.e. B_i = δE_xc/δm_i, exact because E depends on m only through the
    field |m|.  Gradient terms are dropped where ρ ≤ 1e-6 or |∇ρ|² ≤ 1e-10
    (the scalar route's QE thresholds); B = 0 where |m| ≤ 1e-20.

    Every gradient and divergence keeps only the density sphere
    |G|² ≤ ``ecutrho`` (QE's ``fft_gradient_g2r``/``fft_graddot`` act on the
    ngm G-vectors).  It matters here: |m| has a cusp where m changes sign
    (37 % of the bcc Fe grid has m_z < 0), so its box FFT carries weight
    outside the sphere that QE never sees.

    Parameters: ``rho_total`` (nx,ny,nz), ``rho_G_total`` its complex FFT,
    ``mag`` (3,nx,ny,nz) m in the same density units, ``G_cart``
    (nx,ny,nz,3), ``xc_fn`` from :func:`pbe_functional_polarized`.
    Returns ``(v, B)`` in Ry, shapes (nx,ny,nz) and (3,nx,ny,nz).
    """
    rho = jnp.maximum(rho_total, 1e-10)
    sphere = jnp.sum(G_cart ** 2, axis=-1) <= ecutrho
    amag = jnp.sqrt(jnp.sum(mag ** 2, axis=0))
    amag_G = jnp.where(sphere, local_fftn3(amag), 0.0)
    grad_rho = _compute_grad_components(jnp.where(sphere, rho_G_total, 0.0),
                                        G_cart)
    grad_amag = _compute_grad_components(amag_G, G_cart)
    ru = 0.5 * (rho + amag)
    rd = jnp.maximum(0.5 * (rho - amag), 1e-12)
    gu = [0.5 * (a + b) for a, b in zip(grad_rho, grad_amag)]
    gd = [0.5 * (a - b) for a, b in zip(grad_rho, grad_amag)]
    suu = sum(a * a for a in gu)
    sud = sum(a * b for a, b in zip(gu, gd))
    sdd = sum(b * b for b in gd)

    def energy(ru_, rd_, suu_, sud_, sdd_):
        return jnp.sum((ru_ + rd_) * xc_fn(ru_, rd_, suu_, sud_, sdd_))

    zero = jnp.zeros_like(rho)
    lda_u, lda_d = jax.grad(energy, argnums=(0, 1))(ru, rd, zero, zero, zero)
    f_u, f_d, f_uu, f_ud, f_dd = jax.grad(energy, argnums=(0, 1, 2, 3, 4))(
        ru, rd, suu, sud, sdd)
    sigma = sum(a * a for a in grad_rho)
    active = (rho_total > 1e-6) & (sigma > 1e-10)
    f_u = jnp.where(active, f_u, lda_u)
    f_d = jnp.where(active, f_d, lda_d)
    f_uu, f_ud, f_dd = (jnp.where(active, x, 0.0) for x in (f_uu, f_ud, f_dd))

    def divergence(field):
        out = jnp.zeros_like(rho)
        for i in range(3):
            h_G = jnp.where(sphere, local_fftn3(field[i]), 0.0)
            out = out + jnp.real(local_ifftn3(1j * G_cart[..., i] * h_G))
        return out

    v_u = f_u - divergence([2.0 * f_uu * a + f_ud * b for a, b in zip(gu, gd)])
    v_d = f_d - divergence([2.0 * f_dd * b + f_ud * a for a, b in zip(gu, gd)])
    unit = jnp.where(amag > 1e-20, mag / jnp.maximum(amag, 1e-300), 0.0)
    return 0.5 * (v_u + v_d), 0.5 * (v_u - v_d) * unit

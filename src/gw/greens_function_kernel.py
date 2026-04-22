"""Unified Green's function builder for ISDF-basis GW.

  G_μν(k) = Σ_{ij} ψ_i(μ) W_ij ψ*_j(ν)

Convention: psi_xn is direct (μ side). psi_yr is conjugated (ν side).
This matches the tested COHSEX convention throughout gw_jax.py.
"""
import jax.numpy as jnp


def build_G(psi_xn, psi_yr, *, Gij=None, phases=None):
    """Build G_μν(k).  Returns (nk, s, μ_X, s, μ_Y) flat-k.

    psi_xn:  (nk, s, μ_X, nb) — direct (μ side)
    psi_yr:  (nk, nb, s, μ_Y) — conjugated internally (ν side)
    Gij:     (nk, nb, nb) or None — band-space matrix. None → identity.
    phases:  (nk, nb) complex or None — per-band weights. None → ones.
    """
    if Gij is not None and phases is not None:
        p = phases.astype(jnp.complex128)
        return jnp.einsum('ksxi,kij,kjty->ksxty',
            psi_xn, p[:, :, None] * Gij * p[:, None, :],
            jnp.conj(psi_yr), optimize=True)
    if Gij is not None:
        return jnp.einsum('ksxi,kij,kjty->ksxty',
            psi_xn, Gij, jnp.conj(psi_yr), optimize=True)
    if phases is not None:
        return jnp.einsum('ksxn,kn,knty->ksxty',
            psi_xn, phases.astype(jnp.complex128), jnp.conj(psi_yr), optimize=True)
    return jnp.einsum('ksxn,knty->ksxty',
        psi_xn, jnp.conj(psi_yr), optimize=True)


def build_G_tau(psi_xn, psi_yr, enk, t, *, e_ref=0.0, mask=None):
    """G(t)_k(μ, ν) = Σ_n ψ_n(μ) · exp(-t · (e_n - e_ref)) · ψ_n*(ν).

    Unified time-evolution G builder shared by χ₀ (imaginary-time) and
    Σ (real-time).  The imaginary/real-time split is entirely in the
    caller's choice of ``t``:

        real    t → imaginary-time evolution  (χ₀ minimax quadrature).
        pure-i  t → real-time    evolution   (Σ_c ppm/minimax quadrature).

    psi_xn:  (nk, s, μ_X, nb)  direct (μ side), c128
    psi_yr:  (nk, nb, s, μ_Y)  conjugated internally (ν side), c128
    enk:     (nk, nb)           band energies, f64
    t:       scalar              complex evolution time
    e_ref:   scalar              energy reference subtracted from enk before phase
    mask:    (nk, nb) bool or None   zeros bands outside the chosen window (sigma's mask_A)

    Returns (nk, s, μ_X, s, μ_Y) flat-k.  Thin wrapper around ``build_G``
    with phases = exp(-t·(enk - e_ref)) (optionally gated by mask).
    """
    phases = jnp.exp(-t * (enk - e_ref))
    if mask is not None:
        phases = jnp.where(mask, phases,
                           jnp.asarray(0.0 + 0.0j, dtype=jnp.complex128))
    return build_G(psi_xn, psi_yr, phases=phases)

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

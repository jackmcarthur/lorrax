"""Band-basis projection for ISDF self-energy.

  Σ_mn(k) = Σ_{s,μ} ψ*_m(s,μ) Σ(s,μ,s',μ') ψ_n(s',μ')

Two variants:
  project:     Σ(nk,s,μ,s,μ) → Σ(nk,m,n)
  project_ri:  same but returns (2,nk,m,n) with [Re,Im] channels
               for windowed frequency integration in ppm_sigma
"""
import jax.numpy as jnp


def project(psi_xr, psi_yn, sigma_k):
    """Σ(nk, s, μ, s, μ) → Σ(nk, m, n) in band basis."""
    left = jnp.einsum('kmsx,ksxty->kmty', jnp.conj(psi_xr), sigma_k, optimize=True)
    return jnp.einsum('kmty,ktyn->kmn', left, psi_yn, optimize=True)


def project_ri(psi_xr, psi_yn, sigma_k):
    """Σ(nk, s, μ, s, μ) → (2, nk, m, n) with [Re, Im] channels."""
    sigma_ri = jnp.stack((jnp.real(sigma_k), jnp.imag(sigma_k)), axis=0)
    left = jnp.einsum('kmsx,cksxty->ckmty', jnp.conj(psi_xr), sigma_ri, optimize=True)
    return jnp.einsum('ckmty,ktyn->ckmn', left, psi_yn, optimize=True).astype(jnp.complex128)

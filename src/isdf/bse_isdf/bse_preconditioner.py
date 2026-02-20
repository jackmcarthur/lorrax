"""Diagonal preconditioner utilities for shifted BSE linear solves.

This module builds a diagonal approximation for (z I - H_BSE), where the
Hamiltonian in the spinor default is H_BSE = D + V - W. The diagonal is
approximated by the sum of:
  1) The single-particle energy differences (E_c(k) - E_v(k))
  2) The diagonal (c,v,k) elements of the direct term V at q=0
  3) The diagonal (c,v,k) elements of the screened term W at q=0

These diagonal terms can be assembled once and applied via Hadamard products
to trial vectors.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp


class BSEPreconditionerTerms(NamedTuple):
    """Diagonal contributions for a BSE preconditioner."""
    delta_E: jax.Array  # (nc, nv, nk)
    V_diag: jax.Array   # (nc, nv, nk)
    W_diag: jax.Array   # (nc, nv, nk)


def energy_diff_cv_k(eps_c: jax.Array, eps_v: jax.Array) -> jax.Array:
    """Compute energy differences delta_E(c,v,k) = eps_c(k) - eps_v(k).

    Args:
        eps_c: (nk, nc) conduction energies
        eps_v: (nk, nv) valence energies

    Returns:
        delta_E: (nc, nv, nk) array
    """
    # eps_c: (nk, nc) -> (nc, 1, nk)
    # eps_v: (nk, nv) -> (1, nv, nk)
    return eps_c.T[:, None, :] - eps_v.T[None, :, :]


def _pair_amplitude(psi_c: jax.Array, psi_v: jax.Array) -> jax.Array:
    """Spin-traced pair amplitude M(k,c,v,mu) = sum_s psi*_c(k,c,s,mu) psi_v(k,v,s,mu)."""
    return jnp.einsum("kcsm,kvsm->kcvm", jnp.conj(psi_c), psi_v)


def compute_v_diagonal(
    psi_c: jax.Array,
    psi_v: jax.Array,
    V_q0: jax.Array,
    nk: Optional[int] = None,
) -> jax.Array:
    """Diagonal V term at q=0 for each (c,v,k).

    V_diag(c,v,k) = (1/Nk) sum_{mu,nu} M*_cv(k,mu) V_q=0(mu,nu) M_cv(k,nu)

    Args:
        psi_c: (nk, nc, nspinor, n_rmu) conduction wfns
        psi_v: (nk, nv, nspinor, n_rmu) valence wfns
        V_q0: (n_rmu, n_rmu) Coulomb at q=0
        nk: total number of k-points (defaults to psi_c.shape[0])

    Returns:
        V_diag: (nc, nv, nk)
    """
    if nk is None:
        nk = psi_c.shape[0]
    M = _pair_amplitude(psi_c, psi_v)  # (nk, nc, nv, n_rmu)
    V_diag = jnp.einsum("kcvM,MN,kcvN->cvk", jnp.conj(M), V_q0, M)
    return V_diag / jnp.asarray(nk, dtype=V_diag.dtype)


def compute_w_diagonal(
    psi_c: jax.Array,
    psi_v: jax.Array,
    W_q0: jax.Array,
    nk: Optional[int] = None,
) -> jax.Array:
    """Diagonal W term at q=0 for each (c,v,k).

    W_diag(c,v,k) = (1/Nk) sum_{mu,nu} rho_c(k,c,mu) W_q=0(mu,nu) rho_v(k,v,nu)
    where rho_c = sum_s |psi_c|^2 and rho_v = sum_s |psi_v|^2.

    Args:
        psi_c: (nk, nc, nspinor, n_rmu) conduction wfns
        psi_v: (nk, nv, nspinor, n_rmu) valence wfns
        W_q0: (n_rmu, n_rmu) screened interaction at q=0
        nk: total number of k-points (defaults to psi_c.shape[0])

    Returns:
        W_diag: (nc, nv, nk)
    """
    if nk is None:
        nk = psi_c.shape[0]
    rho_c = jnp.einsum("kcsm,kcsm->kcm", jnp.conj(psi_c), psi_c)
    rho_v = jnp.einsum("kvsm,kvsm->kvm", jnp.conj(psi_v), psi_v)
    W_diag = jnp.einsum("kcm,mn,kvm->cvk", rho_c, W_q0, rho_v)
    return W_diag / jnp.asarray(nk, dtype=W_diag.dtype)


def extract_w_q0(W_q: jax.Array) -> jax.Array:
    """Extract W(q=0) from W_q(mu,nu,qx,qy,qz)."""
    return W_q[:, :, 0, 0, 0]


def build_preconditioner_terms(
    eps_c: jax.Array,
    eps_v: jax.Array,
    psi_c: jax.Array,
    psi_v: jax.Array,
    V_q0: jax.Array,
    W_q0: jax.Array,
    nk: Optional[int] = None,
) -> BSEPreconditionerTerms:
    """Compute diagonal D, V, and W terms for a BSE preconditioner.

    Returns:
        BSEPreconditionerTerms with arrays shaped (nc, nv, nk).
    """
    if nk is None:
        nk = psi_c.shape[0]
    delta_E = energy_diff_cv_k(eps_c, eps_v)
    V_diag = compute_v_diagonal(psi_c, psi_v, V_q0, nk=nk)
    W_diag = compute_w_diagonal(psi_c, psi_v, W_q0, nk=nk)
    return BSEPreconditionerTerms(delta_E=delta_E, V_diag=V_diag, W_diag=W_diag)


def build_shifted_preconditioner(
    eps_c: jax.Array,
    eps_v: jax.Array,
    psi_c: jax.Array,
    psi_v: jax.Array,
    V_q0: jax.Array,
    W_q0: jax.Array,
    z: jax.Array,
    nk: Optional[int] = None,
    return_inverse: bool = True,
) -> jax.Array:
    """Build a diagonal preconditioner for (z I - H_BSE) Q = X.

    The diagonal approximation is: diag(H_BSE) = delta_E + V_diag - W_diag.
    The shifted diagonal is: diag(z I - H_BSE) = z - diag(H_BSE).

    Args:
        z: shift (scalar or array broadcastable to (nc, nv, nk))
        return_inverse: if True, returns 1 / (z - diag(H_BSE));
            otherwise returns (z - diag(H_BSE)) directly.
    """
    terms = build_preconditioner_terms(
        eps_c, eps_v, psi_c, psi_v, V_q0, W_q0, nk=nk
    )
    diag_h = terms.delta_E + terms.V_diag - terms.W_diag
    shifted = jnp.asarray(z) - diag_h
    if return_inverse:
        return jnp.reciprocal(shifted)
    return shifted

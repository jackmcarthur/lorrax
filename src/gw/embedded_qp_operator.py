"""Finite-rank QSGW Hamiltonian embedded in a full wavefunction carrier.

For an orthonormal stored-WFN basis ``Psi_W`` this module applies

``H_emb = Psi_W H_W Psi_W^dagger + Q_W H_tail Q_W``,

where ``Q_W = 1 - Psi_W Psi_W^dagger``.  ``H_W`` may contain a QSGW block
and retained DFT exterior bands.  The caller-owned ``H_tail`` acts only in
the unresolved complement, so it need not reproduce the stored states and
does not require constructing ``V_xc[rho]``.  In particular, ``H_tail`` may
be the existing kinetic/local/nonlocal Hamiltonian apply with whatever
explicit no-``V_xc`` local terms are actually available.

The routines are placement-neutral: they never gather or ``device_put``.
The caller remains responsible for placing every full-G carrier over all
processors and for supplying the existing Hamiltonian matvec appropriate to
that carrier.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from solvers.projectors import expand_subspace, subspace_coefficients


__all__ = [
    "apply_embedded_qp_hamiltonian",
    "embedded_qp_contract_residuals",
    "validate_embedded_qp_contract",
]


def _check_shapes(x, basis_w, h_w):
    x_shape = tuple(np.shape(x))
    basis_shape = tuple(np.shape(basis_w))
    h_shape = tuple(np.shape(h_w))
    if len(x_shape) != 3:
        raise ValueError(
            "x must have shape (nvec,nspinor,nG), got "
            f"{x_shape}.")
    if len(basis_shape) != 3 or basis_shape[0] < 1:
        raise ValueError(
            "basis_w must have shape (nband,nspinor,nG) with nband >= 1, "
            f"got {basis_shape}.")
    if basis_shape[1:] != x_shape[1:]:
        raise ValueError(
            "x and basis_w must share the complete spinor/G carrier, got "
            f"{x_shape} and {basis_shape}.")
    if h_shape != (basis_shape[0], basis_shape[0]):
        raise ValueError(
            "h_w must be square on the stored-band axis: expected "
            f"{(basis_shape[0], basis_shape[0])}, got {h_shape}.")


def apply_embedded_qp_hamiltonian(
    x: jax.Array,
    basis_w: jax.Array,
    h_w: jax.Array,
    apply_tail: Callable[[jax.Array], jax.Array],
) -> jax.Array:
    r"""Apply ``Psi_W H_W Psi_W^dagger + Q_W H_tail Q_W``.

    Parameters
    ----------
    x : (nvec, nspinor, nG) complex
        Input kets on the complete G-sphere carrier.
    basis_w : (nband, nspinor, nG) complex
        Orthonormal stored-WFN basis at the same k point.
    h_w : (nband, nband) complex
        Hermitian retained-band Hamiltonian.  A QSGW rotation/eigenvalue
        bundle can reconstruct its corrected block; stored exterior bands
        may remain diagonal at their DFT energies.
    apply_tail : callable
        Existing full-G Hamiltonian apply.  It is called exactly once, on
        ``Q_W x``; this module owns no FFT or pseudopotential implementation.

    Returns
    -------
    (nvec, nspinor, nG) complex
        Embedded Hamiltonian action.  Its stored/complement cross blocks are
        exactly zero by construction.

    Notes
    -----
    This is the operator needed in a plane-wave Sternheimer equation.  The
    double projection is load-bearing: merely dropping ``V_xc`` from a DFT
    Hamiltonian leaves ``P_W H_tail Q_W`` leakage and therefore destroys the
    exact retained QSGW eigenpairs.
    """
    _check_shapes(x, basis_w, h_w)
    coefficients = subspace_coefficients(basis_w, x)
    p_x = expand_subspace(basis_w, coefficients)
    q_x = x - p_x
    tail_q = apply_tail(q_x)
    if tuple(np.shape(tail_q)) != tuple(np.shape(x)):
        raise ValueError(
            "apply_tail must preserve the ket carrier: expected "
            f"{tuple(np.shape(x))}, got {tuple(np.shape(tail_q))}.")

    retained_coefficients = jnp.einsum(
        "mn,bn->bm", h_w, coefficients, optimize=True)
    retained = expand_subspace(basis_w, retained_coefficients)
    q_tail_q = tail_q - expand_subspace(
        basis_w, subspace_coefficients(basis_w, tail_q))
    return retained + q_tail_q


def embedded_qp_contract_residuals(basis_w, h_w):
    """Return orthonormality and Hermiticity max-residual scalars."""
    basis_shape = tuple(np.shape(basis_w))
    h_shape = tuple(np.shape(h_w))
    if len(basis_shape) != 3 or basis_shape[0] < 1:
        raise ValueError(
            "basis_w must have shape (nband,nspinor,nG) with nband >= 1, "
            f"got {basis_shape}.")
    if h_shape != (basis_shape[0], basis_shape[0]):
        raise ValueError(
            "h_w must be square on the stored-band axis: expected "
            f"{(basis_shape[0], basis_shape[0])}, got {h_shape}.")
    overlap = jnp.einsum(
        "msG,nsG->mn", jnp.conj(basis_w), basis_w, optimize=True)
    identity = jnp.eye(basis_shape[0], dtype=overlap.dtype)
    orthonormality = jnp.max(jnp.abs(overlap - identity))
    hermiticity = jnp.max(jnp.abs(h_w - jnp.conj(jnp.swapaxes(h_w, -1, -2))))
    return orthonormality, hermiticity


def validate_embedded_qp_contract(
    basis_w,
    h_w,
    *,
    tolerance: float = 1.0e-10,
):
    """Refuse a non-orthonormal basis or non-Hermitian retained block.

    Only the two scalar residuals are transferred to the host; neither the
    stored WFN basis nor its overlap matrix is gathered.
    """
    tol = float(tolerance)
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError(
            f"tolerance must be finite and non-negative, got {tolerance!r}.")
    orthonormality, hermiticity = embedded_qp_contract_residuals(basis_w, h_w)
    orth_value = float(jax.device_get(orthonormality))
    herm_value = float(jax.device_get(hermiticity))
    if orth_value > tol:
        raise ValueError(
            "basis_w is not orthonormal within the embedded-QP tolerance: "
            f"max residual {orth_value:.3e} > {tol:.3e}.")
    if herm_value > tol:
        raise ValueError(
            "h_w is not Hermitian within the embedded-QP tolerance: "
            f"max residual {herm_value:.3e} > {tol:.3e}.")
    return orth_value, herm_value

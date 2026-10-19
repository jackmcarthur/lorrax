"""solvers/sternheimer_precond.py — Teter–Payne–Allan (TPA) preconditioner for
Sternheimer / DFPT linear solves.

The standard DFPT preconditioner — cf. Teter, Payne & Allan, PRB **40** (1989)
12255 and Payne, Teter, Allan, Arias, Joannopoulos, Rev. Mod. Phys. **64**
(1992) 1045, also used in QE's ``ph.x`` ``cgsolve_all.f90``. The motivation:
the Sternheimer operator ``A = Q_{k-q} (H_{k-q} − ε_{v,k}) Q_{k-q}`` is
diagonally dominated by the kinetic energy at large ``|k+G|``, so a smooth
rescaling of the kinetic diagonal serves as a cheap approximate inverse. The
TPA rational form is specifically designed to:

    * approach 0 at large ``x = T_G / K̄²_v`` (high-frequency modes damped),
    * approach 1 at ``x → 0`` (low-frequency modes passed through unaltered),
    * be smooth through ``x ≈ 1`` without sign flips (unlike the
      ``1 / (h_diag − ε)`` Davidson preconditioner in psp/dft_precond.py,
      which can switch sign near the shift).

Formula::

    P^{-1}_{v, G} = TPA(x_{v,G}),   x_{v, G} = T_G / K̄²_v
    TPA(x) = (27 + 18x + 12x² + 8x³) / (27 + 18x + 12x² + 8x³ + 16x⁴)

where ``T_G = |k − q + G|²`` is the kinetic diagonal of ``H_{k-q}`` and
``K̄²_v = ⟨ψ_{v,k} | T_k | ψ_{v,k}⟩`` is the per-band characteristic kinetic
energy (evaluated at the *source* k, not the shifted ``k−q``; this is the
right scale for the Sternheimer orbital ``δψ_v``, which inherits its envelope
from ``ψ_{v,k}``).

Notes
-----
The TPA polynomial is numerically symmetric (``TPA(x) + TPA(1/x) = 1``) and
passes through ``TPA(1) = 65/81 ≈ 0.802``. No absolute-value or sign guard
is needed because both numerator and denominator are strictly positive for
``x > 0``.
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp


# ═══════════════════════════════════════════════════════════════════════
#  TPA rational polynomial
# ═══════════════════════════════════════════════════════════════════════

@jax.jit
def _tpa(x: jax.Array) -> jax.Array:
    """TPA rational form.  ``x`` ≥ 0, any JAX-compatible shape."""
    num = 27.0 + x * (18.0 + x * (12.0 + x * 8.0))
    den = num + 16.0 * x ** 4
    return num / den


# ═══════════════════════════════════════════════════════════════════════
#  Per-band kinetic expectation value K̄²_v
# ═══════════════════════════════════════════════════════════════════════

@jax.jit
def compute_per_band_kinetic(U_val: jax.Array, T_diag: jax.Array) -> jax.Array:
    """Return ``K̄²_v = ⟨ψ_v | T | ψ_v⟩ = Σ_{s,G} T_G · |ψ_{v,s,G}|²``.

    Parameters
    ----------
    U_val  : (nv, nspinor, nG) complex — orthonormal occupied block at k.
    T_diag : (nG,) float — kinetic diagonal at k (``|k+G|²`` in Ry units,
             available as ``HamiltonianK.T_diag``).

    Returns
    -------
    K_bar_sq : (nv,) float — one number per band.
    """
    # |ψ|² summed over spinor + G, weighted by T_G.
    psi_abs_sq = jnp.sum(jnp.abs(U_val) ** 2, axis=1)   # (nv, nG)
    return jnp.einsum('nG,G->n', psi_abs_sq, T_diag)


# ═══════════════════════════════════════════════════════════════════════
#  Factory: preconditioner closure
# ═══════════════════════════════════════════════════════════════════════

def tpa_preconditioner_diag(
    T_diag_kminq: jax.Array,
    K_bar_sq: jax.Array,
) -> jax.Array:
    """Return the TPA diagonal weights as an array  ``(nv, 1, nG_p)``.

    Equivalent to the factory below but skips the Python-callable wrapper
    — for plugging directly into a JIT-stable Sternheimer primitive where
    the preconditioner is an elementwise multiply by a precomputed array.
    """
    K_safe = jnp.where(K_bar_sq > 0.0, K_bar_sq, 1.0)
    x = T_diag_kminq[None, :] / K_safe[:, None]
    return _tpa(x)[:, None, :]


def make_tpa_preconditioner(
    T_diag_kminq: jax.Array,
    K_bar_sq: jax.Array,
) -> Callable[[jax.Array], jax.Array]:
    """Return a batched TPA preconditioner callable.

    Parameters
    ----------
    T_diag_kminq : (nG_p,) float
        Kinetic diagonal of ``H_{k-q}``: ``|k − q + G|²`` for G on the
        k−q sphere.  Take this straight from ``setup_H_k_from_kvec(kvec_kminq,
        ...).T_diag`` — no extra work needed.
    K_bar_sq : (nv,) float
        Per-band characteristic kinetic energy at the *source* k, from
        :func:`compute_per_band_kinetic`.  Built once per k before the
        outer solve and passed in verbatim.

    Returns
    -------
    precond : callable (nv, nspinor, nG_p) → (nv, nspinor, nG_p)
        Diagonal rescaling, one TPA weight per (v, G).
    """
    # Broadcast x[v, G] = T_diag_kminq[G] / K_bar_sq[v].
    # Guard K̄² = 0 only on the off chance an all-zero band (shouldn't happen
    # for real occupied bands but keeps the JIT trace shape-stable).
    K_safe = jnp.where(K_bar_sq > 0.0, K_bar_sq, 1.0)
    x = T_diag_kminq[None, :] / K_safe[:, None]          # (nv, nG_p)
    weights = _tpa(x)[:, None, :]                         # (nv, 1, nG_p)

    @jax.jit
    def precond(R: jax.Array) -> jax.Array:
        """(nv, nspinor, nG_p) → same; diagonal multiply."""
        return R * weights
    return precond


__all__ = [
    "compute_per_band_kinetic",
    "make_tpa_preconditioner",
    "tpa_preconditioner_diag",
]

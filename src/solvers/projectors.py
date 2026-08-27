"""solvers/projectors.py — Jitted subspace projectors used by the Sternheimer solver.

Two shared contractions and four factories returning callables of shape
``(batch, nspinor, nG) → (batch, nspinor, nG)``.  Closure-captured
subspace bases ``U`` are static JAX arrays, so the compiled kernel is
reusable across Sternheimer iterations on the same (k, k−q) pair.

    P_val      = U_val · U_val^†              (occupied subspace at some k)
    P_precond  = U_extra · U_extra^†          (Schur low-energy T-block)
    P_rest     = 1 − P_val − P_precond        (everything else = R-block)
    Q_kminq    = 1 − P_val^{k-q}              (standard off-occupied projector)

Orthonormality assumption: each ``U`` block is expected to be
orthonormal (``U^† U = 1``), which is true for WFN.h5 eigenstates at
well-converged QE calculations and for Ritz vectors extracted by
Davidson.  If a downstream caller ever passes a non-orthonormal block
we'd want to factor through a QR, but that is out of scope here and
would show up as a correctness violation at the ``U^† b ≈ 0`` sanity
check (``reports/*/report.md``).

Notation follows psp/sternheimer_guidelines.md §1 (Cancès et al. 2023)
and uses the convention ``p = k − q`` everywhere (matches
``SymMaps.kq_map[ik_full, iq_red]``).
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp


# ═══════════════════════════════════════════════════════════════════════
#  Core contraction — single shared JIT'd kernel
# ═══════════════════════════════════════════════════════════════════════

@jax.jit
def subspace_coefficients(U: jax.Array, x: jax.Array) -> jax.Array:
    """Return ``c[b,m] = <U_m|x_b>`` for an orthonormal basis block."""
    return jnp.einsum('msG,bsG->bm', jnp.conj(U), x, optimize=True)


@jax.jit
def expand_subspace(U: jax.Array, coefficients: jax.Array) -> jax.Array:
    """Return ``sum_m coefficients[b,m] U[m]`` on the ket carrier."""
    return jnp.einsum(
        'bm,msG->bsG', coefficients, U, optimize=True)


@jax.jit
def _apply_P_U(U: jax.Array, x: jax.Array) -> jax.Array:
    """Return  P_U x = U · (U^† x).

    U : (m, nspinor, nG)    basis block, orthonormal
    x : (batch, nspinor, nG)
    """
    return expand_subspace(U, subspace_coefficients(U, x))


# ═══════════════════════════════════════════════════════════════════════
#  Factories
# ═══════════════════════════════════════════════════════════════════════

def make_P_val(U_val: jax.Array) -> Callable[[jax.Array], jax.Array]:
    """P_val = U_val · U_val^†.  Projects onto occupied subspace.

    U_val : (nv, nspinor, nG) — orthonormal occupied block at some k.
    """
    def P_val(x: jax.Array) -> jax.Array:
        return _apply_P_U(U_val, x)
    return P_val


def make_P_precond(U_extra: jax.Array) -> Callable[[jax.Array], jax.Array]:
    """P_precond = U_extra · U_extra^†.  Schur T-block low-energy conduction
    subspace (a handful of Ritz vectors above the gap).  Stage-2c Schur split
    uses this to peel off the ill-conditioned directions before MINRES.

    U_extra : (M, nspinor, nG) — orthonormal.
    """
    def P_precond(x: jax.Array) -> jax.Array:
        return _apply_P_U(U_extra, x)
    return P_precond


def make_P_rest(
    U_val: jax.Array,
    U_extra: jax.Array | None = None,
) -> Callable[[jax.Array], jax.Array]:
    """P_rest = 1 − P_val − P_precond.  The "R" block we iterate in.

    When ``U_extra is None`` (pre-Schur path) this reduces to
    ``P_rest = 1 − P_val = Q_val``; a convenience alias for Stage 1
    where we iterate in the whole conduction subspace at once.
    """
    if U_extra is None:
        def P_rest_novq(x: jax.Array) -> jax.Array:
            return x - _apply_P_U(U_val, x)
        return P_rest_novq

    def P_rest(x: jax.Array) -> jax.Array:
        return x - _apply_P_U(U_val, x) - _apply_P_U(U_extra, x)
    return P_rest


def make_Q_kminq(U_val_kminq: jax.Array) -> Callable[[jax.Array], jax.Array]:
    """Q_{k−q} = 1 − P_val^{k−q}.  Standard off-occupied projector at p = k−q.

    Math-notation alias for ``make_P_rest(U_val_kminq, U_extra=None)``.  The
    guide's Sternheimer equation reads  Q_{k−q} (H_{k−q} − ε_{v,k}) Q_{k−q}
    |δψ_{v,k}⟩ = −Q_{k−q} |b_{v,k}⟩; keeping this alias lets the driver
    code spell the operator 1:1 with the math.

    U_val_kminq : (nv, nspinor, nG_p) — orthonormal occupied block at p=k−q.
    """
    def Q_kminq(x: jax.Array) -> jax.Array:
        return x - _apply_P_U(U_val_kminq, x)
    return Q_kminq


__all__ = [
    "subspace_coefficients",
    "expand_subspace",
    "make_P_val",
    "make_P_precond",
    "make_P_rest",
    "make_Q_kminq",
]

"""Charge ISDF conditioning, the one seam: C_q⁺ by rank truncation.

``factor`` turns the logical, Hermitian-PSD C_q stack ``(nq, n, n)`` into B
with ``B Bᴴ = C⁺`` (keep ``λ > rcond·λ_max``, the cut closed over degenerate
multiplets and certified against the Gram ceiling); ``apply`` is
``ζ = C⁺Z = B(BᴴZ)``.  Every charge producer calls these two -- the μ-batch
fit (``isdf.zeta_mubatch``), the dense refit (``solve_zeta_charge_dense``) and
the per-q factor stacks of ``isdf.core`` -- so a different conditioning
procedure replaces this file and nothing else.  Callers own the logical
extent (``runtime.padding.solve_at_logical``); nothing here sees a pad.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from common import rank_criterion


def apply(B, Z):
    """ζ = C⁺Z = B(BᴴZ) for the factor ``B`` of :func:`factor`."""
    return B @ (jnp.conj(jnp.swapaxes(B, -1, -2)) @ Z)


def factor(C_log, *, rcond: float, rank_log: bool, n_log: int):
    """B with B Bᴴ = C⁺ for a logical C_q stack; see the module docstring."""
    from isdf.core import _certify_the_cut, _close_the_cut
    # WHY THIS FEATURE EXISTS: the charge CCT near-singularizes when
    # n_μ over-completes the pair-density rank (κ~1e13); plain
    # Cholesky then amplifies ULP/mesh/nband roundoff into O(1) V_q
    # errors that GN-PPM magnifies to tens of eV.  Rank-truncation
    # DROPS eigenvalues < zeta_rcond·λ_max (the near-null
    # directions) → a conditioned, mesh-invariant ζ = C⁺Z.
    lam, V = jnp.linalg.eigh(C_log)      # Hermitian-SPD, λ ascending
    lam_max = lam[..., -1:]              # (nqb,1) largest λ per q
    keep = lam > (rcond * lam_max)       # near-null cut
    # …and the near-null cut is not allowed to stop mid-multiplet.  THIS
    # IS THE SEAM the 6×6×6 saga's §6 conjectured about
    # (tests/known_failures/2026-08-10-ibz-cascade-vs-full-bz-sigma-\
    # 6x6x6.md): C_q commutes with the point group when the centroid set
    # is orbit-closed and the band window degeneracy-closed, so a
    # symmetry maps each C_q eigenspace onto itself and mixes a degenerate
    # block's members freely.  Cut between blocks and ζ's retained span is
    # invariant, so C_{Sq} = P C_q P† survives the truncation; cut THROUGH
    # a block and the span is a round-off-chosen slice that differs
    # between q and Sq, and the k-star identity fails for W and Σ_x alike.
    # That deck turned out to be covariant by luck — 0 of 16 q-stars
    # carried a non-constant n_keep, MEASURED after the fact, enforced by
    # nothing.  This is the enforcement.
    keep = _close_the_cut(lam, keep, where="zeta rank_truncate")
    # …and a cut that lands in a gap can still be a cut nobody has
    # certified.  THE GATE: when the criterion BINDS, the achieved
    # amplification must not exceed the ceiling any measurement supports
    # for a PSD overlap Gram (1e8 — R19's rcond ladder and the Si 4×4×4
    # 1776-centroid run, both in ``common/rank_criterion``).  Until
    # 2026-08-22 the ``rank_log`` block below announced exactly these
    # numbers and gated on neither.
    _certify_the_cut(lam, keep, where="zeta rank_truncate",
                     kappa_certified=rank_criterion.KAPPA_CERTIFIED_GRAM,
                     rcond=rcond)
    # B = V·diag(1/√λ_kept) ⇒ B Bᴴ = Σ_{keep} vᵢvᵢᴴ/λᵢ = C⁺.
    # Double-``where`` keeps rsqrt off the dropped (tiny/≤0) modes.
    inv_sqrt = jnp.where(
        keep, jax.lax.rsqrt(jnp.where(keep, lam, 1.0)), 0.0)
    # OBSERVABILITY: the retained-mode count IS the conditioning
    # signal for this route — it is what tells you whether n_μ has
    # over-completed the pair-density rank (κ blow-up) and by how
    # much.  It lives inside the jit, so print it from there.
    # ``n_keep`` per q + the spectral span λ_max/λ_min(kept).
    # Mandatory conditioning receipt; there is no silence knob.
    #
    # THE CRITERION, stated: ``keep`` above is NOT a search for a
    # gap in λ — a real ISDF charge spectrum is smooth and has
    # none.  It is a CAP on how much C⁺ may amplify round-off:
    # κ_eff = λ_max/λ_min(kept) ≤ 1/zeta_rcond by construction.
    # ``common/rank_criterion`` carries the derivation, the three
    # standard alternatives (discrepancy principle / L-curve /
    # GCV) and the measurement that refutes each of them here.
    #
    # The three extra fields below are the ones a run needs in
    # order to be auditable without a sweep:
    #   kappa/q     achieved amplification — the invariant
    #   ldrop_hi/q  the LARGEST discarded λ, i.e. the top of the
    #               discarded band (paired with lam_min_kept it
    #               gives the whole cut, and shows there is no
    #               plateau at the cut — there never is)
    #   margin/q    fractional rank inflation from loosening
    #               rcond by 1e-4.  §R19 measured +41 % of rank
    #               costing 5000 eV, so a LARGE margin means the
    #               basis is over-complete and rcond must NOT be
    #               loosened on this run.
    if rank_log:
        lam_keep_min = jnp.min(
            jnp.where(keep, lam, jnp.inf), axis=-1)
        n_keep = jnp.sum(keep, axis=-1)
        lam_drop_hi = jnp.max(
            jnp.where(keep, -jnp.inf, lam), axis=-1)
        n_loose = jnp.sum(lam > (rcond * 1e-4 * lam_max), axis=-1)
        margin = (n_loose - n_keep) / jnp.maximum(n_keep, 1)
        jax.debug.print(
            "[zeta rank_truncate] n_log={n} rcond={rc:.1e} "
            "n_keep/q={k} lam_max/q={mx} lam_min_kept/q={mn} "
            "kappa/q={kp} ldrop_hi/q={dh} lam_min/q={lo} "
            "margin/q={mg}",
            n=n_log, rc=rcond,
            k=n_keep,
            mx=lam_max[..., 0], mn=lam_keep_min,
            kp=lam_max[..., 0] / lam_keep_min,
            dh=lam_drop_hi, lo=jnp.min(lam, axis=-1),
            mg=margin,
            ordered=False)
    return V * inv_sqrt[..., None, :].astype(V.dtype)

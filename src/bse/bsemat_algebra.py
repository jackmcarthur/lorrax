"""BerkeleyGW head/wing/body of the static W from the stored ISDF tensors.

The physics is docs/theory/bgw-bsemat-export.md. Per q, with
``Z[G, μ] = ζ_qμ(G)/√Ω``, the restart stores ``V = Zᴴ v Z`` and
``W = Zᴴ W_PW Z`` (E2). BerkeleyGW splits W at its head slot G₁ (the shortest
q + G, BerkeleyGW's own tie rule). From V, W, the one column
``z = ζ_q(·, G₁)/√Ω`` and ``v₁ = v(q + G₁)``:

    V a = z*,   s = aᴴ(W − V)a,   r = aᴴ(W − V)                        (E3)
    W₁₁ = v₁ + v₁² s                                                    (E4)
    y   = v₁ r − v₁² s z      (row G₁ of W_PW off the head, times Z)    (E5)
    H = W₁₁ zᴴz,  Y = zᴴy + yᴴz,  B = W − H − Y                          (E6)

and the kernel parts of a (k, k') pair follow from the centroid pair
amplitudes C (E7). Every function here is plain array algebra on jnp/np
arrays. Callers pass sharded faces and a face eigensolver, so nothing here
chooses a backend.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

#: The V-span cut of the solve V a = z* (relative to λ_max). E3 needs only
#: some solution, so the cut is a conditioning choice, not a truncation of W.
RCOND = 1.0e-12
#: Refuse a split whose solve misses z* by more than this (relative).
SOLVE_RESIDUAL_TOL = 1.0e-6
#: v₁ z V⁺ z* ≤ 1 is V_b = V − v₁ zᴴz ≽ 0, i.e. z is V's head row. A wrong
#: (unfolded, conjugated, other-slot) z breaks it by O(1).
HEAD_ROW_TOL = 1.0e-6
#: s = aᴴ(W − V)a is invariant under the centroid action that carries the
#: parent q to a star member. A key's s against its parent's beyond this
#: (relative) means its W tile is not the parent's W under that action.
ROTATION_S_TOL = 1.0e-8


class HeadSplit(NamedTuple):
    """E4–E5 at one q (host scalars, μ-vectors as the caller's arrays).

    ``W11`` the plane-wave head W(G₁, G₁) (Ry·bohr³ as V carries v);
    ``y`` (μ,) the off-head row of E5; ``z`` (μ,) the head column used;
    ``figures`` the gates' measured values.
    """
    W11: float
    y: object
    z: object
    figures: dict


def head_solve(lam, U, z, *, xp=np, rcond=RCOND):
    """E3: ``a = V⁺ z*`` from V's eigenpairs, and the kept rank.

    ``lam`` (μ,) ascending (real part used) and ``U`` (μ, μ) columns of the
    Hermitian V; ``z`` (μ,) the head column. The span cut keeps
    λ > rcond · λ_max, the one rule every caller uses. The eigenpairs may come
    from V itself or from a parent tile V_q of which V = M V_q M† (``M`` a
    centroid action): the spectrum is the same, so is the cut.
    """
    lam = xp.asarray(lam).real
    keep = lam > rcond * xp.max(lam)
    inv = xp.where(keep, 1.0 / xp.where(keep, lam, 1.0), 0.0)
    a = U @ (inv * (xp.conj(U).T @ xp.conj(z)))             # V⁺ z*   (E3)
    return a, xp.sum(keep)


def head_row_values(V, W, z, a, v1, *, xp=np):
    """Array-valued E3 residual, E4 and E5, safe inside a sharded program."""
    zc = xp.conj(z)
    resid = xp.linalg.norm(V @ a - zc) / xp.linalg.norm(zc)
    D = W - V
    r = xp.conj(a) @ D                                       # aᴴ(W − V), row
    s = xp.real(r @ a)
    W11 = v1 + v1 * v1 * s                                  # (E4)
    y = v1 * r - (v1 * v1 * s) * z                           # (E5)
    head_row = v1 * xp.real(z @ a)                           # v₁ z V⁺ z*
    return W11, y, resid, head_row, s


def head_rows(V, W, z, a, v1, *, rank, xp=np):
    """E3–E5 and scalar receipts from a solution ``a`` of ``V a = z*``.

    A rotated parent solution serves its star member because the equations
    do not see its ker V part. Shapes and units are those of head_split.
    """
    W11, y, resid, head_row, s = head_row_values(V, W, z, a, v1, xp=xp)
    figures = {"solve_residual": float(resid), "head_row": float(head_row),
               "eps_inv_head": float(W11) / float(v1), "rank": int(rank), "s": float(s)}
    return HeadSplit(float(W11), y, z, figures)


def head_split(V, W, z, v1, *, eigh, xp=np, rcond=RCOND):
    """E3–E5: the plane-wave head W₁₁ and the off-head row y of W at one q.

    Parameters
    ----------
    V, W : (μ, μ) complex Hermitian
        The stored bare and static screened tiles at this q (Ry; tiles
        carry 1/Ω), on any layout ``eigh`` and ``xp`` accept.
    z : (μ,) complex
        ``ζ_q(μ, G₁)/√Ω``, the head column (replicated).
    v1 : float
        ``v(q + G₁)`` in Ry·bohr³, the value V carries in that slot.
    eigh : callable
        ``eigh(V) -> (λ ascending, U columns)``; a face plan in production.
    rcond : float
        V-span cut of the solve, relative to λ_max.

    Returns
    -------
    HeadSplit
    """
    lam, U = eigh(V)
    a, rank = head_solve(lam, U, z, xp=xp, rcond=rcond)
    return head_rows(V, W, z, a, v1, rank=rank, xp=xp)


def check_split(split: HeadSplit, *, where: str):
    """Refuse a split whose gates fail (rule id, got, want, why, fix)."""
    f = split.figures
    if not f["solve_residual"] <= SOLVE_RESIDUAL_TOL:
        raise ValueError(
            f"GATE bsemat_head_solve: {where}: |V a - z*|/|z*| = "
            f"{f['solve_residual']:.3e}; want <= {SOLVE_RESIDUAL_TOL:g}; why: z* "
            "must lie in V's range for E3; fix: check the ζ column and the "
            "q row it was read for.")
    if not f["head_row"] <= 1.0 + HEAD_ROW_TOL:
        raise ValueError(
            f"GATE bsemat_head_row: {where}: v1 z V+ z* = {f['head_row']:.6f}; "
            f"want <= 1; why: V - v1 z^H z must be positive, so this z is not "
            "the head row of V; fix: the ζ column (slot, conjugation or "
            "unfold phase) or v1 (point vs cell-averaged head) is wrong.")
    if not 0.0 < f["eps_inv_head"] <= 1.0 + HEAD_ROW_TOL:
        raise ValueError(
            f"GATE bsemat_head_screening: {where}: W11/v1 = "
            f"{f['eps_inv_head']:.6f}; want in (0, 1]; why: a static "
            "screened head lies between 0 and the bare v; fix: W is not the "
            "static W0 of this V.")


def check_rotation(split: HeadSplit, s_parent: float, *, where: str):
    """Refuse a key whose s differs from its parent q's (rule id, got, want, why, fix)."""
    s = float(split.figures["s"])
    rel = abs(s - float(s_parent)) / max(abs(float(s_parent)), 1e-300)
    if not rel <= ROTATION_S_TOL:
        raise ValueError(
            f"GATE bsemat_head_rotation: {where}: |s - s_parent|/|s_parent| = "
            f"{rel:.3e}; want <= {ROTATION_S_TOL:g}; why: s is invariant under "
            "the centroid action that carries the parent q to this star "
            "member, so this W tile is not the parent's W under the action "
            "that carried a; fix: the restart's W and V q-star tables differ, "
            "or W was unfolded by another rule.")
    return rel


def pair_amplitudes(psi_l, psi_r, xp=np):
    """C[k, n, n', μ] = Σ_s conj(ψ_l[k, n, s, μ]) ψ_r[k, n', s, μ] (E7).

    ``psi_l`` (nk, nl, ns, μ) at k, ``psi_r`` (nk, nr, ns, μ) at k′ of the same pair row.
    """
    return xp.einsum("knsm,kpsm->knpm", xp.conj(psi_l), psi_r)


def direct_parts(Ccc, Cvv, Wq, z, y, W11, *, omega, lam_q, xp=np):
    """E7: BerkeleyGW head, wing and body of a batch of (k, k′) pairs.

    Parameters
    ----------
    Ccc : (b, nc, nc′, μ) — pair amplitudes of the conduction (n2) legs.
    Cvv : (b, nv, nv′, μ) — pair amplitudes of the valence (n1) legs.
    Wq  : (b, μ, μ) — W(k − k′) of each pair.
    z, y : (b, μ) — head column and off-head row of each pair's q (E5).
    W11 : (b,) — plane-wave head of each pair's q (E4).
    omega : float — cell volume, bohr³.
    lam_q : (b,) — |q| on an untruncated 3D deck, 1 with truncation (E1b).

    Returns
    -------
    head, wing, body : (b, nc, nc′, nv, nv′) complex, in BerkeleyGW's units
        (head dimensionless; wing and body bohr², i.e. W/8π).
    """
    t = xp.einsum("bcdm,bmn->bcdn", Ccc, Wq)
    full = xp.einsum("bcdn,bwun->bcdwu", t, xp.conj(Cvv))
    m_c = xp.einsum("bcdm,bm->bcd", Ccc, xp.conj(z))
    m_v = xp.einsum("bwum,bm->bwu", Cvv, xp.conj(z))
    n_c = xp.einsum("bcdm,bm->bcd", Ccc, xp.conj(y))
    n_v = xp.einsum("bwum,bm->bwu", Cvv, xp.conj(y))
    mm = xp.einsum("bcd,bwu->bcdwu", m_c, xp.conj(m_v))
    wing = (xp.einsum("bcd,bwu->bcdwu", m_c, xp.conj(n_v))
            + xp.einsum("bcd,bwu->bcdwu", n_c, xp.conj(m_v)))
    body = full - W11[:, None, None, None, None] * mm - wing
    f = omega / (8.0 * np.pi)
    return omega * mm, f * lam_q[:, None, None, None, None] * wing, f * body


def exchange_part(Ccv_k, Ccv_kp, Vbar, *, omega, xp=np):
    """E7: BerkeleyGW exchange ``(Ω/8π) Σ C_cv(k) V̄_Γ C*_c′v′(k′)`` for all (k, k′).

    ``Ccv_k`` (nk, nc, nv, μ) with C_cv = Σ_s conj(ψ_c)ψ_v (LORRAX's
    exchange amplitude); ``Ccv_kp`` the same at the k′ rows; ``Vbar`` the
    head-less Γ tile. Returns (nk′, nk, nc′, nc, nv′, nv)-ordered blocks
    ``X[k′, k, c′, c, v′, v]``. Equal to BerkeleyGW's literal
    Σ M*_vc v̄ M_v′c′ when V̄_Γ is real (ζ at Γ is real in r).
    """
    t = xp.einsum("kcvm,mn->kcvn", Ccv_k, Vbar)
    X = xp.einsum("kcvn,pdun->pkdcuv", t, xp.conj(Ccv_kp))
    return (omega / (8.0 * np.pi)) * X


def add_q0_body_update(W0, A, D, xp=np):
    """The q → 0 hook: ``W(Γ) + Aᴴ D A``, the body term BerkeleyGW's full-inverse eps0mat carries.

    BerkeleyGW's q0 body is the body block of the inverted ε at q0, so it holds
    the O(q⁰) wing-mediated part ε_b0 ε_00⁻¹ ε_0b that the head-less Γ tile of
    W lacks (docs/theory/bgw-bsemat-export.md#q0). In μ space that part is a
    rank-n update ``Aᴴ D A``: rows ``A`` (n, μ) and a (n, n) ``D``, supplied by the
    q → 0 dipole wing vector of the eps export. With no update the writer keeps
    BerkeleyGW's semiconductor convention for the wings (zero) and the head-less body.
    Wings at q = 0 stay zero either way, as BerkeleyGW stores them.
    """
    A = xp.asarray(A)
    return W0 + xp.conj(A).T @ xp.asarray(D) @ A


def q0_dipole_body_rows(W0, L, R, S_eff, S_direct, q0_cart, omega, xp=np):
    """(A, D) of BerkeleyGW's 3-D q0 body term from the restart's stored q → 0 wings.

    ``head_wing_L`` L (3, μ), ``head_wing_R`` R (μ, 3) = Lᴴ, ``S_cart_head`` S_eff and
    ``S_cart_head_direct`` S_d follow the eps export's convention
    (docs/theory/bgw_epsmat_export.md E9 on the eps branch). Block inversion
    (E13) projected onto μ (Z† ⋯ Z, Z† v_b Z = V, W = V + Vχ^μV) gives
    BerkeleyGW's q0 body as W(Γ) + c (W R q₀)(q₀ᵀ L W), c = v(q₀)/(Ω ε_M),
    v(q₀) = 8π/|q₀|², ε_M = 1 − v(q₀) q₀ᵀ S_eff q₀ (E11). With q̂ = q₀/|q₀| that is
    ``Aᴴ D A`` for ``add_q0_body_update``: A = (W R q̂)† (1, μ), D = 8π/(Ω ε_M),
    O(1) and independent of |q₀|. ``q0_cart`` is BerkeleyGW's q0 direction (1/bohr).

    Returns ``A, D, fig`` with ``fig['fold_gate']`` the relative mismatch of
    Ω q̂ᵀ(S_eff − S_d)q̂ and q̂ᵀ L W R q̂ on THIS W: it certifies L's μ order and
    normalisation where the term consumes them.
    """
    q = xp.asarray(q0_cart, dtype=xp.float64)
    qh = q / xp.linalg.norm(q)
    a = W0 @ (R @ qh)
    fold_W = qh @ (L @ a)
    fold_S = omega * (qh @ ((S_eff - S_direct) @ qh))
    eps_M = 1.0 - 8.0 * np.pi * xp.real(qh @ (S_eff @ qh))
    A = xp.conj(a)[None, :]
    D = xp.reshape(8.0 * np.pi / (omega * eps_M), (1, 1)).astype(A.dtype)
    fig = {"eps_M": eps_M, "fold_W": fold_W, "fold_S": fold_S,
           "fold_gate": xp.abs(fold_S - fold_W) / xp.abs(fold_W),
           "R_vs_LH": xp.linalg.norm(R - xp.conj(L).T) / xp.linalg.norm(L)}
    return A, D, fig

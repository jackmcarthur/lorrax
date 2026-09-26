"""Project two-point objects between ISDF ζ bases of very different size.

Problem (owner directive 2026-07-29).  A production GW run fits a LARGE
ISDF basis over a wide band window (``μ_L`` ~ 10k centroids); a BSE /
post-processing consumer only needs a few valence and conduction bands
and wants the same physics in a SMALL basis (``μ_S`` ~ 500).  The object
to move is the screened interaction

    W_L[q, μ_L, ν_L]   →   W_S[q, μ_S, ν_S]

Both are two-point functions of the *same* field

    W_q(r, r') ≈ Σ_{μν} ζ_{q,μ}(r) · W[q,μ,ν] · conj(ζ_{q,ν}(r'))

so the change of basis is a **congruence with a rectangular transfer
matrix** ``T[q, μ_S, μ_L]``::

    W_S = T · W_L · T†                                            (★)

which is structurally the band projection this repo already owns —
``common.contract_bands.contract_bands_block_reshard`` — with
``ψ_left/ψ_right`` carrying ``T``/``T†`` and its ``k`` batch axis
carrying ``q``.  This module is that primitive's first non-Σ consumer;
it adds only (a) the operand construction and (b) the least-squares solve
for ``T``.  The caller builds the two matrices that solve consumes;
``gw.downfold`` builds them from pair densities.  Everything the primitive
encodes (two-stage psum_scatter so no ``(μ,μ)`` tile is ever gathered,
large payload on the node-local mesh axis, f64-split de-promotion, the
vendor-BLAS GEMM dial, the impl=mpi warm-up contract, actionable
divisibility refusals) is inherited unchanged — see
``docs/dev/staged_reshard_primitive.md``.

The transfer (READ THIS BEFORE CHANGING THE ALGEBRA)
----------------------------------------------------
Given the small basis's Gram ``G_S[q, μ_S, ν_S]`` and the cross-overlap
``O[q, μ_S, μ_L]`` in one metric, the least-squares transfer is

    T = G_S^{-1} · O                                                  (‡)

It is invariant under ``metric → c · metric`` (both ``G_S`` and ``O``
scale by ``c``), so a grid normalization or a sphere truncation cannot
leak an arbitrary scale into ``W_S``; a bare congruence with ``O`` would
scale as ``c²``.  ``T`` is also the *minimizer* of
``‖W_L(r,r') − W_S(r,r')‖_F`` over the small basis, and it is EXACT
whenever the field is representable there (see
:func:`least_squares_transfer`).

``G_S`` is ``(n_q, μ_S, μ_S)`` — bounded by construction (μ_S is small:
that is the entire premise) and the ONLY replicated object here.

Doctrine 1 (LORRAX scaling target: thousands of low-memory ranks; no
``N_μ²`` tile on any single rank) — how each object obeys it
------------------------------------------------------------------
=========================  =========================  ==================
object                     global shape / spec        per-rank bytes
=========================  =========================  ==================
``W_L``                    (n_q, μ_L, μ_L)            ∝ μ_L²/P   ↓ with P
                           ``P(None,'x','y')``
``T_left``                 (n_q, μ_S, 1, μ_L)         ∝ μ_S·μ_L/p_x ↓
                           ``P(None,None,None,'x')``
``T_right``                (n_q, 1, μ_L, μ_S)         ∝ μ_S·μ_L/p_y ↓
                           ``P(None,None,'y',None)``
``G_S``                    (n_q, μ_S, μ_S) replicated  bounded, ∝ μ_S²
``W_S``                    (n_q, μ_S, μ_S)            ∝ μ_S²/P
                           ``P(None,'x','y')``
=========================  =========================  ==================

Every entry except ``G_S`` FALLS with P, and ``G_S`` is the small basis's
own Gram — it can never reach ``μ_L²``.

Refusals (all raise BEFORE any collective, with the fix named)
--------------------------------------------------------------
* non-2-D mesh, or minor axis ≠ ``axes[1]`` (inherited from the
  primitive: the large payload must reduce-scatter over consecutive-rank
  groups);
* a Gram that is not positive definite (``rcond=None``) — reported with
  the failing q and the suggestion to shrink the small basis or check
  for duplicate centroids.  No silent ridge, no silent pinv.
"""
from __future__ import annotations

import os
from typing import Callable

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common import rank_criterion
from common import spectral_closure
from common.collectives import warm_mesh_cliques
from common.contract_bands import contract_bands_block_reshard

__all__ = [
    "least_squares_transfer",
    "project_w_between_zeta_bases",
    "transfer_operands_from_dense",
]


def _check_mesh(mesh_xy: Mesh, axes: tuple[str, str]) -> tuple[int, int]:
    ax_x, ax_y = axes
    names = tuple(mesh_xy.axis_names)
    if len(names) != 2:
        raise ValueError(
            f"zeta_projection needs a 2-D mesh, got axes {names}.  The "
            f"congruence tiles μ_S on one axis and ν_S on the other; a 1-D "
            f"mesh would force a full (μ_S, μ_S) row on every rank.")
    if ax_x not in names or ax_y not in names:
        raise ValueError(f"mesh axes {names} do not contain axes={axes!r}")
    if names[-1] != ax_y:
        raise ValueError(
            f"zeta_projection: the mesh's minor axis is {names[-1]!r} but "
            f"the large reduce-scatter payload must ride {ax_y!r} (only the "
            f"LAST mesh axis has consecutive-rank replica groups on the "
            f"standard process-ordered layout).  Build the mesh with "
            f"{ax_y!r} minor, or pass axes=(major, minor).  This is the "
            f"same refusal contract_bands_block_reshard enforces.")
    return int(mesh_xy.shape[ax_x]), int(mesh_xy.shape[ax_y])


# ---------------------------------------------------------------------------
# (‡) the least-squares transfer
# ---------------------------------------------------------------------------

def least_squares_transfer(
    gram_S, O_sharded, mesh_xy: Mesh, mu_axis: str, *,
    rcond: float | None = None, print_fn=print,
) -> jax.Array:
    """``T = G_S^{-1} O`` (Cholesky) or ``T = G_S^+ O`` (rank-truncated).

    ``rcond=None`` — Cholesky back-substitution, valid when the small ζ
    set is linearly independent on the G-sphere.  Done INSIDE a shard_map
    with ``G_S`` replicated and ``O``'s μ_L axis sharded, so the solve is
    rank-local by construction and the partitioner is never asked to plan
    a collective for it.  **Refuses** (does not ridge, does not pinv)
    when the Cholesky fails, and names ``rcond`` as the fix.

    ``rcond=<float>`` — Hermitian eigendecomposition of ``G_S``, keeping
    the eigenvalues above ``rcond · λ_max`` per q and zeroing the rest:
    the Moore–Penrose pseudo-inverse truncated at a stated rank.  The
    retained rank is ANNOUNCED per q, because it is physics: it is the
    number of small-basis ζ's that are actually independent on the
    sphere, and if it is far below μ_S the small basis is not the size
    the caller thinks it is.

    **Why the truncated route is not optional on real data.**  Measured
    on the production MoS2 12×12 80 Ry ζ (job 7879532): the 606-centroid
    basis has ``cond(G_S) = 5.7e18`` with eigenvalues spanning
    ``-4.5e-05 … 6.9e+11`` — numerically singular, the negative end being
    round-off on a positive-semidefinite matrix.  A real ISDF basis is
    strongly linearly dependent on its own sphere; the ISDF FIT already
    fights exactly this with ``charge_zeta_solve=rank_truncate`` and the
    RCOND dials.  The synthetic study could never have shown it
    (cond ≈ 20 there), which is why it had to be run on real ζ.

    ``T`` is exact on the retained subspace, so the representability
    property of (‡) survives truncation with "representable" read as
    "representable in the retained span".
    """
    from common.shard_map import shard_map

    if rcond is None:
        chol = jnp.linalg.cholesky(gram_S)
        bad = jnp.isnan(
            jnp.real(jnp.diagonal(chol, axis1=-2, axis2=-1))).any()
        if bool(jax.device_get(bad)):
            diag = jnp.real(jnp.diagonal(gram_S, axis1=-2, axis2=-1))
            raise ValueError(
                f"zeta_projection: the SMALL-basis Gram G_S = <ζ^S|ζ^S> is "
                f"not positive definite (Cholesky produced NaN) — the small "
                f"ζ set is linearly dependent on the G-sphere.  For a REAL "
                f"ISDF basis this is the NORMAL case, not a bug: measured "
                f"cond(G_S)=5.7e18 on the production 606-centroid MoS2 "
                f"12×12 basis (job 7879532).  FIX: pass rcond (e.g. "
                f"rcond=1e-10) to use the rank-truncated pseudo-inverse — the "
                f"same route the ISDF fit itself takes "
                f"(charge_zeta_solve=rank_truncate).  Other causes worth "
                f"excluding first: duplicated centroids, or a ζ file whose "
                f"trailing μ rows are the zero pad (pass the LOGICAL "
                f"centroid count).  Gram diagonal min/max = "
                f"{float(jnp.min(diag)):.3e}/{float(jnp.max(diag)):.3e}.  "
                f"No ridge is applied and no pseudo-inverse is substituted "
                f"silently: that would be indistinguishable from a correct "
                f"projection downstream.")

        def _body(L_loc, O_loc):
            y = jax.lax.linalg.triangular_solve(
                L_loc, O_loc, left_side=True, lower=True,
                transpose_a=False, conjugate_a=False)
            return jax.lax.linalg.triangular_solve(
                L_loc, y, left_side=True, lower=True,
                transpose_a=True, conjugate_a=True)

        return shard_map(
            _body, mesh=mesh_xy,
            in_specs=(P(None, None, None), P(None, None, mu_axis)),
            out_specs=P(None, None, mu_axis), check_vma=False)(chol, O_sharded)

    if not (0.0 < float(rcond) < 1.0):
        raise ValueError(
            f"rcond must be in (0, 1) — it is a RELATIVE threshold on "
            f"λ/λ_max, got {rcond!r}.")
    w, V = jnp.linalg.eigh(gram_S)              # ascending, replicated
    lam_max = jnp.max(w, axis=-1, keepdims=True)
    keep = w > (float(rcond) * lam_max)
    # CLOSURE.  The cut above is a cap on amplification; it is not allowed to
    # stop halfway through a degenerate block.  ``G_S`` commutes with every
    # symmetry the centroid set is closed under, so a symmetry maps each of
    # its eigenspaces onto itself and mixes a degenerate block's members
    # freely: retain the block whole and the retained span is invariant,
    # retain part of it and the span is a round-off-chosen slice that differs
    # between q and Sq.  ``common/spectral_closure`` moves the mask off any
    # block it straddles by DROPPING that block whole (owner ruling
    # 2026-08-10) — pure jnp, so it stays inside the same replicated
    # computation and the verdict is identical on every process.
    #
    # ``off`` skips it entirely; ``strict`` computes the same verdict and
    # refuses on host below rather than snapping, so the flag means the same
    # thing here as at every other seam.
    # The caller reads the dial and passes it — ``spectral_closure`` is L2 and
    # must be a function of its arguments (tests/test_layering.py).
    _sc_mode = spectral_closure.resolve_mode(
        os.environ.get(spectral_closure.MODE_ENV))
    _keep_closed, _n_pre, _n_post = spectral_closure.close_keep_mask(
        w, keep, rtol=spectral_closure.DEFAULT_RTOL)
    if _sc_mode == "snap":
        keep = _keep_closed
    # THE CRITERION: ``keep`` is a CAP on how much G_S⁺ may amplify round-off
    # (κ_eff = λ_max/λ_min(kept) ≤ 1/rcond), NOT a search for a gap — these
    # ζ-overlap spectra are smooth and have none.  ``common/rank_criterion``
    # carries the derivation and the measurements that refute the standard
    # alternatives (discrepancy principle / L-curve / GCV) for this code.
    #
    # Diagnostic, on every run and in ONE device_get: retained rank, the
    # retained block's λ range (hence κ_eff), the discarded λ range, and the
    # margin — the fractional rank inflation from loosening rcond by 1e-4.
    # §R19 measured +41 % of retained rank costing 5000 eV, so a large margin
    # says the basis is over-complete and rcond must not be loosened here.
    _stats = jax.device_get((
        jnp.sum(keep, axis=-1),                                  # ranks
        lam_max[..., 0],                                         # λ_max
        jnp.min(jnp.where(keep, w, jnp.inf), axis=-1),           # λ_min kept
        jnp.max(jnp.where(keep, -jnp.inf, w), axis=-1),          # λ top dropped
        jnp.min(w, axis=-1),                                     # λ_min
        jnp.sum(w > (float(rcond) * 1e-4 * lam_max), axis=-1),   # loose rank
        _n_pre,                                                  # rank at the
        _n_post,                                                 # cap / closed
    ))
    ranks = np.asarray(_stats[0])
    lmax_h, lminkeep_h, ldrop_h, lmin_h, ranks_loose = (
        np.asarray(x) for x in _stats[1:6])
    sc_pre, sc_post = np.asarray(_stats[6]), np.asarray(_stats[7])
    n_mu_s = int(gram_S.shape[-1])
    # Computed on EVERY process (gram_S is replicated, so the verdict is the
    # same everywhere) — a refusal raised only on rank 0 would hang the rest.
    with np.errstate(divide='ignore', invalid='ignore'):
        kappa = lmax_h / lminkeep_h
        margin = (ranks_loose - ranks) / np.maximum(ranks, 1)
    if jax.process_index() == 0:
        print_fn(
            f"[zeta_projection] rank-truncated transfer at rcond={rcond:g}: "
            f"retained rank per q min/median/max = {ranks.min()}/"
            f"{int(np.median(ranks))}/{ranks.max()} of μ_S={n_mu_s} "
            f"({100.0 * ranks.mean() / n_mu_s:.1f}% of the nominal basis). "
            f"The discarded directions are ζ's that are linearly dependent "
            f"on the rest ON THE SPHERE — W_S is the projection onto the "
            f"retained span, and a rank far below μ_S means the small "
            f"basis is smaller than it looks.")
        print_fn(
            f"[zeta_projection]   retained λ {np.max(lmax_h):.6e} .. "
            f"{np.min(lminkeep_h):.6e} -> kappa_eff max {np.nanmax(kappa):.3e} "
            f"(cap 1/rcond = {1.0/float(rcond):.3e}); discarded λ "
            f"{np.min(lmin_h):.6e} .. {np.max(ldrop_h):.6e}; grid alignment "
            f"discarded 0 (this route never rounds the rank to the mesh); "
            f"margin (rcond x 1e-4) max +{100.0*np.max(margin):.1f}% "
            f"— R19 anchor: +41% cost 5000 eV.")
    # ── The closure verdict, per q ────────────────────────────────────────
    # ``sc_pre`` is the rank the amplification cap chose, ``sc_post`` the rank
    # that closes whatever degenerate block that cut landed in.  They differ
    # only on q whose cut sliced a block, and under the default direction the
    # difference is NEGATIVE — the straddled block is dropped.  Counting with
    # ``!=`` rather than ``>`` is what keeps that a firing instead of silence.
    _n_fired = int(np.count_nonzero(sc_post != sc_pre))
    if _sc_mode != "off" and jax.process_index() == 0:
        if _n_fired:
            print_fn(
                f"*** [spectral-closure] zeta_projection transfer: the rank "
                f"cut lands INSIDE a degenerate block of G_S on {_n_fired} of "
                f"{len(sc_pre)} q.  Retained rank per q would be "
                f"{int(sc_pre.min())}..{int(sc_pre.max())}; closing the block "
                f"takes it to {int(sc_post.min())}..{int(sc_post.max())} "
                f"(max -{int(np.max(sc_pre - sc_post))} directions on one q).  "
                f"A cut through a degenerate block retains a symmetry-"
                f"ARBITRARY slice of an eigenspace, so W_S would project onto "
                f"a different subspace at q than at Sq. "
                + (f"THE STRADDLED BLOCK WAS DROPPED "
                   f"({spectral_closure.DEFAULT_DIRECTION}). ***"
                   if _sc_mode == "snap" else f"NOT closed (strict). ***"))
        else:
            print_fn(
                f"[zeta_projection]   spectral closure: the cut falls in a gap "
                f"on all {len(sc_pre)} q at rtol "
                f"{spectral_closure.DEFAULT_RTOL:.1e} (noise floor eps/rcond = "
                f"{spectral_closure.degeneracy_noise_rtol(rcond):.2e}) — no "
                f"degenerate block is cut.")
    if _sc_mode == "strict" and _n_fired:
        raise spectral_closure.SpectralClusterError(
            f"zeta_projection: the rcond={rcond:g} rank cut falls inside a "
            f"degenerate block of G_S on {_n_fired} of {len(sc_pre)} q, so the "
            f"retained span is not point-group invariant.  Fix: "
            f"LORRAX_SPECTRAL_CLOSURE=snap drops each straddled block whole "
            f"(retained rank {int(sc_pre.max())} -> {int(sc_post.min())} at "
            f"worst), or =off to cut through deliberately.")
    # The cap check.  Under the default direction the closure DROPS the
    # straddled block, which raises λ_min(kept) and so can only LOWER
    # κ_eff — the cap cannot be violated by the guard, and this expression
    # is identically 1.0, i.e. the historical bare ``1 + 1e-9`` check.  It
    # is written in the general form rather than deleted because it is what
    # keeps the assertion honest if a site ever opts into ``keep_block``:
    # admitting m links each within ``rtol`` of the last can lower
    # λ_min(kept) by at most (1+rtol)^m, and that — nothing looser — is the
    # slack such a site would be entitled to.
    _slack = float(np.max(
        (1.0 + spectral_closure.DEFAULT_RTOL) ** np.maximum(sc_post - sc_pre, 0)))
    if np.nanmax(kappa) > (1.0 / float(rcond)) * _slack * (1.0 + 1e-9):
        raise ValueError(
            f"zeta_projection: achieved amplification "
            f"{np.nanmax(kappa):.3e} exceeds the cap "
            f"{1.0/float(rcond):.3e} the rcond={rcond:g} truncation was "
            f"supposed to enforce (closure slack {_slack:.6f}) — the retained "
            f"set is not the one the criterion selected.")
    # THE GATE (docs/dev/rank_truncation_policy.md §2).  The cap check above
    # asks whether the code did what it was told — κ_eff ≤ 1/rcond — and
    # that is NECESSARY AND NOT SUFFICIENT: both registered ζ catastrophes
    # satisfied it exactly.  This asks whether the regime is one anyone has
    # certified, and it only applies where the cut BOUND.  Worst q, not the
    # mean: a gate reported on an average cannot fire.
    _bound_q = ranks < n_mu_s
    if bool(np.any(_bound_q)):
        _w_dropped = np.max(np.where(_bound_q, n_mu_s - ranks, 0))
        rank_criterion.certify_numbers(
            kappa_eff=float(np.nanmax(np.where(_bound_q, kappa, -np.inf))),
            n_dropped=int(_w_dropped), n_total=n_mu_s,
            # This route reduces over q before anything reaches host and does
            # not carry the per-q trace, so the weight finding is not
            # available here.  0.0 is the honest "not measured" value for a
            # ceiling test that only fires upward; the kappa finding is the
            # one this site can make.
            discarded_weight=0.0,
            kappa_certified=rank_criterion.KAPPA_CERTIFIED_GRAM,
            quantity="eigenvalues of G_S",
            site="zeta_projection rank-truncated transfer",
            mode=os.environ.get(rank_criterion.POLICY_MODE_ENV),
            cause=(f"the small ζ basis is over-complete on the G-sphere: "
                   f"μ_S={n_mu_s} with a retained rank as low as "
                   f"{int(ranks.min())}, and the retained block runs down to "
                   f"the rcond={rcond:g} cut."),
            fix=("lower μ_S (the small basis is smaller than it looks), or "
                 "raise rcond back onto the certified plateau."),
            log=print_fn)
    if int(ranks.min()) == 0:
        raise ValueError(
            f"zeta_projection: rcond={rcond:g} retained ZERO directions at "
            f"some q — either the threshold is above λ_max (lower rcond), or "
            f"the spectral-closure guard dropped a degenerate block that "
            f"reached λ_max, which leaves nothing above the cut.  The "
            f"closure verdict above says which: it is the second when the "
            f"guard fired on that q.  A block spanning the whole retained "
            f"range means G_S is flat to "
            f"{spectral_closure.DEFAULT_RTOL:.1e} there, which is a "
            f"statement about the small basis and not about rcond.")
    w_inv = jnp.where(keep, 1.0 / jnp.where(keep, w, 1.0), 0.0)

    def _body(V_loc, wi_loc, O_loc):
        # T = V diag(w_inv) V^H O, all rank-local (V, w_inv replicated)
        y = jnp.einsum('qkm,qkn->qmn', jnp.conj(V_loc), O_loc, optimize=True)
        y = y * wi_loc[:, :, None].astype(y.dtype)
        return jnp.einsum('qmk,qkn->qmn', V_loc, y, optimize=True)

    return shard_map(
        _body, mesh=mesh_xy,
        in_specs=(P(None, None, None), P(None, None),
                  P(None, None, mu_axis)),
        out_specs=P(None, None, mu_axis), check_vma=False)(V, w_inv, O_sharded)


# ---------------------------------------------------------------------------
# operand construction for the primitive
# ---------------------------------------------------------------------------

def transfer_operands_from_dense(T_x, T_y, mesh_xy: Mesh,
                                 axes: tuple[str, str] = ("x", "y")):
    """``(T_x, T_y) -> (psi_left, psi_right)`` in the primitive's layout.

    The conjugate / transpose are applied ONCE here rather than on every
    ``project`` call: they are rank-local (the sharded μ_L axis only
    changes position, never mesh axis) but they do allocate, and the
    congruence is meant to be called repeatedly (per ω, per iteration).

    ``psi_left  = conj(T_x)[:, :, None, :]``  → the primitive conjugates
    it again internally (ψ†Oψ semantics), recovering ``T``.
    ``psi_right = conj(T_y)ᵀ`` → ``T†``.
    """
    ax_x, ax_y = axes
    psi_left = jax.lax.with_sharding_constraint(
        jnp.conj(T_x)[:, :, None, :],
        NamedSharding(mesh_xy, P(None, None, None, ax_x)))
    psi_right = jax.lax.with_sharding_constraint(
        jnp.conj(jnp.swapaxes(T_y, 1, 2))[:, None, :, :],
        NamedSharding(mesh_xy, P(None, None, ax_y, None)))
    return psi_left, psi_right


# ---------------------------------------------------------------------------
# (★) the congruence itself — contract_bands_block_reshard, unmodified
# ---------------------------------------------------------------------------

def project_w_between_zeta_bases(
    mesh_xy: Mesh, *, axes: tuple[str, str] = ("x", "y"),
) -> Callable:
    """Build ``project(W_L, psi_left, psi_right) -> W_S``.

    ``W_S = T W_L T†`` executed by
    ``contract_bands_block_reshard(extra="none", channels="none")`` with
    the identification

        primitive        here
        ---------        ----
        k  (batch)   →   q   (momentum transfer)
        μ, ν         →   μ_L, ν_L   (the LARGE basis, contracted)
        m, n         →   μ_S, ν_S   (the SMALL basis, the output tile)
        ψ_left       →   conj(T) on 'x'
        ψ_right      →   T†        on 'y'
        s, s'        →   size-1 (ζ's are spin-independent, manual §5.2)

    W_L enters as the primitive's ``O`` at spec
    ``P(None, None, 'x', None, 'y')`` — the size-1 spinor axes are free
    reshapes of the production ``P(None,'x','y')`` W, so a caller holding
    W from ``gw.w_isdf.solve_w`` passes it with no reshard and no copy.

    Returns W_S at ``(n_q, μ_S, ν_S)``, ``P(None, 'x', 'y')`` — the same
    layout as W_L, so the projected object is a drop-in for any consumer
    that already eats a sharded W.
    """
    ax_x, ax_y = axes
    _check_mesh(mesh_xy, axes)
    warm_mesh_cliques(mesh_xy)
    inner = contract_bands_block_reshard(
        mesh_xy, channels="none", extra="none", axes=axes)
    o_sh = NamedSharding(mesh_xy, P(None, None, ax_x, None, ax_y))

    def _project(W_L, psi_left, psi_right):
        if W_L.ndim != 3:
            raise ValueError(
                f"W_L must be (n_q, μ_L, ν_L); got {tuple(W_L.shape)}.  If "
                f"you hold a single-q W, add the leading axis — the q batch "
                f"is the primitive's k axis and must be present.")
        O = jax.lax.with_sharding_constraint(
            W_L[:, None, :, None, :], o_sh)
        return inner(psi_left, O, psi_right)

    return _project

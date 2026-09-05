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
it adds only (a) the operand construction and (b) the build of ``T``
itself from the two ζ sets.  Everything the primitive encodes (two-stage
psum_scatter so no ``(μ,μ)`` tile is ever gathered, large payload on the
node-local mesh axis, f64-split de-promotion, the vendor-BLAS GEMM dial,
the impl=mpi warm-up contract, actionable divisibility refusals) is
inherited unchanged — see ``docs/dev/staged_reshard_primitive.md``.

The metric convention (READ THIS BEFORE CHANGING THE ALGEBRA)
-------------------------------------------------------------
The discretized inner product of two ζ's in this codebase is fixed by
what ``gw.v_q_g_flat`` already does to build ``V_q`` — the module docstring
there states the contract::

    V_q[μ, ν] = Σ_G conj(ζ̃_{q,μ}(G)) · v(q+G) · ζ̃_{q,ν}(G)

over the per-q G-sphere, with ζ̃ the on-disk G-flat ζ:
``ζ̃_{q,μ}(G) = FFT[e^{-iq·r} ζ_{q,μ}(r)]`` at ``norm="backward"``
(``common.wfn_transforms.accumulate_rchunk_to_gflat``).  The ζ overlap is
therefore *the same contraction with the Coulomb weight set to one*::

    O[q, μ_S, μ_L] = Σ_G conj(ζ̃^S_{q,μ_S}(G)) · ζ̃^L_{q,μ_L}(G)      (†)

Three consequences that are easy to get wrong:

* **The e^{iq·r} phase cancels.**  Manual §5.1: ζ_{q,μ}(r) is a phase
  ``e^{+iq·r}`` times a cell-periodic ``z_{q,μ}(r)``.  Both bases are
  evaluated at the SAME q, so ``conj(ζ^S) ζ^L = conj(z^S) z^L`` — the
  G-flat (cell-periodic) buffers are exactly the right operands, no
  phase bookkeeping.
* **This is the sphere-truncated, N_r-scaled r-grid overlap.**  At
  ``norm="backward"`` Parseval gives ``Σ_G conj(z̃^S) z̃^L = N_r ·
  Σ_r conj(z^S(r)) z^L(r)`` when the sum runs over the full G box; the
  production sphere truncates it exactly the way ``V_q`` is truncated.
  We use the same metric ``V_q`` uses, not a different one.
* **Which is why the transfer is NOT the raw overlap.**  The
  least-squares transfer

        T = G_S^{-1} · O ,      G_S[q] = Σ_G conj(ζ̃^S) ζ̃^S           (‡)

  is invariant under ``metric → c · metric`` (both ``G_S`` and ``O``
  scale by ``c``), so the N_r factor and the sphere truncation cannot
  leak an arbitrary scale into ``W_S``.  A bare congruence with ``O``
  would scale as ``c²``.  ``T`` is also the *minimizer* of
  ``‖W_L(r,r') − W_S(r,r')‖_F`` over the small basis, and it is EXACT
  whenever the field is representable there (see
  :func:`least_squares_transfer` and the round-trip gate).  ``mode="raw"``
  exposes the bare-overlap congruence for callers who want it, and says
  so out loud.

``G_S`` is ``(n_q, μ_S, μ_S)`` — bounded by construction (μ_S is small:
that is the entire premise) and the ONLY replicated object here; it is
announced with its byte size at build time.

Doctrine 1 (LORRAX scaling target: thousands of low-memory ranks; no
``N_μ²`` tile on any single rank) — how each object obeys it
------------------------------------------------------------------
=========================  =========================  ==================
object                     global shape / spec        per-rank bytes
=========================  =========================  ==================
``W_L``                    (n_q, μ_L, μ_L)            ∝ μ_L²/P   ↓ with P
                           ``P(None,'x','y')``
``ζ̃^L``                    (n_q, μ_L, n_G)            ∝ μ_L·n_G/P ↓
                           ``P(None,None,('x','y'))``
``ζ̃^S``                    (n_q, μ_S, n_G)            ∝ μ_S·n_G/P ↓
                           ``P(None,None,('x','y'))``
``T_left``                 (n_q, μ_S, 1, μ_L)         ∝ μ_S·μ_L/p_x ↓
                           ``P(None,None,None,'x')``
``T_right``                (n_q, 1, μ_L, μ_S)         ∝ μ_S·μ_L/p_y ↓
                           ``P(None,None,'y',None)``
``G_S``                    (n_q, μ_S, μ_S) replicated  bounded, ∝ μ_S²
``W_S``                    (n_q, μ_S, μ_S)            ∝ μ_S²/P
                           ``P(None,'x','y')``
=========================  =========================  ==================

Every entry except ``G_S`` FALLS with P, and ``G_S`` is the small basis's
own Gram — it can never reach ``μ_L²``.  The overlap build (†) contracts
over **G, sharded flat over the whole mesh** — the r/G axis is the
parallel axis, exactly as required — and its rank-local partial is
bounded independently of P by chunking the μ_S axis
(``mu_s_chunk``), so no ``(μ_S, μ_L)``-full transient is forced either.

Two overlap plans (pick by what layout your ζ can be supplied in)
-----------------------------------------------------------------
* :func:`zeta_overlap_block_reshard` — ζ sharded on **G only**; both
  shardings of ``O`` come out of one pass.  Convenient (one ζ layout)
  but its rank-local partial is μ_L-FULL, so its reduce-scatter payload
  is **P-independent** and the stage is FLAT in P (measured 6.9 / 6.8 /
  8.2 / 6.8 s at P = 4 / 16 / 64 / 144).
* :func:`zeta_overlap_single_axis` — ζ sharded on **μ_L and G**; the
  rank-local partial IS the output shard, so it needs ONE ``psum`` per
  output, no μ_S chunking, and a payload that FALLS with the mesh.
  Measured 5.26 / 2.43 / 2.22 / 1.26 s at the same four P — faster at
  every P and the only one that scales.  Costs one extra ζ layout
  (μ on 'x'/G on 'y' for ``O_x``, μ on 'y'/G on 'x' for ``O_y``).
  **Use this one unless you cannot produce the second layout.**

The two are gated against each other at production scale (they compute
the same matrix through structurally different collectives); measured
agreement 3.8e-16 … 1.0e-15 at P = 4 … 144.

Refusals (all raise BEFORE any collective, with the fix named)
--------------------------------------------------------------
* non-2-D mesh, or minor axis ≠ ``axes[1]`` (inherited from the
  primitive: the large payload must reduce-scatter over consecutive-rank
  groups);
* ``μ_L % p_x`` / ``μ_L % p_y`` / ``μ_S % p_x`` / ``μ_S % p_y`` ≠ 0;
* ``n_G % (p_x·p_y)`` ≠ 0 for the ζ operands;
* a Gram that is not positive definite (``mode="ls"``) — reported with
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
    "assert_zeta_bases_compatible",
    "zeta_overlap_block_reshard",
    "zeta_overlap_single_axis",
    "zeta_gram_single_axis",
    "zeta_gram_replicated",
    "least_squares_transfer",
    "build_zeta_transfer",
    "project_w_between_zeta_bases",
    "transfer_operands_from_dense",
]


_MU_DIV_MSG = (
    "zeta_projection: {what}={n} must be a multiple of BOTH mesh axes "
    "(p_{ax_x}={px}, p_{ax_y}={py}) — the ζ/W tiles are block-sharded on "
    "both.  Pad {what} independently per axis at the caller (round up to "
    "lcm(p_x, p_y) is NOT required; round each axis's own extent) and "
    "strip the zero pad downstream.  Padding ζ rows with zeros is exactly "
    "the existing ISDF n_rmu_padded contract (common.meta.Meta), and a "
    "zero ζ row contributes zero to the overlap, so the pad is inert.")


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


def _require_div(n: int, mesh_xy: Mesh, what: str, axes) -> None:
    from runtime.padding import authenticate_padded_axis
    ax_x, ax_y = axes
    authenticate_padded_axis(
        n, n, mesh_xy, name=f"zeta projection {what} carrier",
        specs=((P(ax_x, None), 0), (P(None, ax_y), 1)))


# ---------------------------------------------------------------------------
# Two-ζ-file compatibility: the per-q sphere must be THE SAME sphere
# ---------------------------------------------------------------------------

def assert_zeta_bases_compatible(loader_S, loader_L, *, print_fn=print) -> dict:
    """Refuse two ζ files whose G axes are not the same G's, in the same order.

    The overlap (†) is ``Σ_G conj(ζ̃^S(G)) ζ̃^L(G)`` — a sum over a shared
    index.  On the G-flat format that index is a **per-q WFN.h5-style
    sphere** whose positions vary with q
    (``isdf_header/gvec_components``, shape ``(n_q, 3, ngkmax)``;
    ``zeta_loader`` says so explicitly and refuses to narrow it
    with a single shared ``sphere_idx``).  Two files can therefore agree
    on ``ngkmax``, ``ngk``, ``FFTgrid`` and ``n_q`` and STILL be summing
    different G's at the same array position — in which case the overlap
    is silently meaningless.

    This is not hypothetical.  Measured on the MoS2 12×12 80 Ry decks:
    the μ=2406, μ=606 and two of the μ=276 ζ files share
    ``gvec_components`` bit-for-bit, while a third μ=276 file
    (``mos2_80ry_12x12/tmp/zeta_q.h5``) has identical ngkmax=8603,
    identical ngk, identical FFTgrid=(36,36,135) and identical n_q=144
    but a DIFFERENT ``gvec_components``.  Nothing in the shapes would
    have caught it.

    Checks, in order (each refusal names the fix): ζ layout, completeness
    flag, FFT grid, n_q on disk and q layout, ngkmax, per-q ngk, and
    finally ``gvec_components`` byte-for-byte.

    Returns a dict of the agreed geometry for the caller's log.
    """
    import numpy as np

    def _fail(what, a, b, fix):
        raise ValueError(
            f"zeta_projection: the two ζ files disagree on {what} — "
            f"small={a!r} vs large={b!r}.  {fix}")

    for lo, tag in ((loader_S, "small"), (loader_L, "large")):
        if getattr(lo, "zeta_layout", None) != "G_flat":
            raise ValueError(
                f"zeta_projection: the {tag} ζ file is layout "
                f"{getattr(lo, 'zeta_layout', None)!r}; the overlap consumes "
                f"the G-flat format (the r-space layout would need an FFT "
                f"per read).  Refit with the G-flat writer.")
        if not bool(getattr(lo, "zeta_is_done", True)):
            raise ValueError(
                f"zeta_projection: the {tag} ζ file's zeta_is_done flag is "
                f"FALSE — it was left behind by a job that died mid-write "
                f"and its trailing chunks are undefined.  Re-run the fit; "
                f"do NOT project from it.")

    a, b = np.asarray(loader_S.fft_grid), np.asarray(loader_L.fft_grid)
    if not np.array_equal(a, b):
        _fail("FFTgrid", a.tolist(), b.tolist(),
              "The two bases must be fitted on the same real-space grid; "
              "refit one of them on the other's grid.")
    if int(loader_S.n_q_on_disk) != int(loader_L.n_q_on_disk):
        _fail("n_q on disk", int(loader_S.n_q_on_disk),
              int(loader_L.n_q_on_disk),
              "The overlap is per-q and indexes both files with the same q; "
              "an IBZ-only file cannot be paired with a full-BZ one without "
              "unfolding first.")
    if loader_S.q_layout != loader_L.q_layout:
        _fail("q layout", loader_S.q_layout, loader_L.q_layout,
              "Unfold the IBZ file (or fold the full-BZ one) before "
              "projecting.")
    if int(loader_S.n_G_sph_disk) != int(loader_L.n_G_sph_disk):
        _fail("ngkmax", int(loader_S.n_G_sph_disk),
              int(loader_L.n_G_sph_disk),
              "Different sphere sizes mean different cutoffs; refit at a "
              "common zeta_cutoff_ry.")

    ngk_s = getattr(loader_S, "ngk_per_q", None)
    ngk_l = getattr(loader_L, "ngk_per_q", None)
    if ngk_s is not None and ngk_l is not None:
        if not np.array_equal(np.asarray(ngk_s), np.asarray(ngk_l)):
            _fail("per-q ngk", "…", "…",
                  "The per-q logical sphere sizes differ; the two fits did "
                  "not see the same q-sphere.")

    gv_s = getattr(loader_S, "gvec_components", None)
    gv_l = getattr(loader_L, "gvec_components", None)
    if gv_s is None or gv_l is None:
        raise ValueError(
            "zeta_projection: a G-flat ζ file is missing "
            "isdf_header/gvec_components, so the per-q sphere it was "
            "written against cannot be verified.  Refusing rather than "
            "assuming the two files index the same G's — see this "
            "function's docstring for the measured counterexample.")
    gv_s = np.asarray(gv_s, dtype=np.int32)
    gv_l = np.asarray(gv_l, dtype=np.int32)
    if gv_s.shape != gv_l.shape or not np.array_equal(gv_s, gv_l):
        n_bad = (int(np.count_nonzero(np.any(gv_s != gv_l, axis=(1, 2))))
                 if gv_s.shape == gv_l.shape else -1)
        raise ValueError(
            f"zeta_projection: the two ζ files carry DIFFERENT per-q "
            f"G-spheres (isdf_header/gvec_components differs"
            + (f" at {n_bad} of {gv_s.shape[0]} q's" if n_bad >= 0
               else f"; shapes {gv_s.shape} vs {gv_l.shape}")
            + ").  Every other header field can match — ngkmax, ngk, "
            "FFTgrid, n_q all did in the case that motivated this check — "
            "and the overlap would still be summing conj(ζ^S) at one G "
            "against ζ^L at a different G, position by position, giving a "
            "plausible-looking but meaningless transfer.  Fix: use two ζ "
            "files written by the same sphere convention (fits on the same "
            "deck normally are), or scatter both onto a shared sphere via "
            "gvec_components first — that scatter is NOT implemented here "
            "or in file_io.zeta_loader.")

    geom = {"n_q_on_disk": int(loader_L.n_q_on_disk),
            "ngkmax": int(loader_L.n_G_sph_disk),
            "fft_grid": tuple(int(x) for x in np.asarray(loader_L.fft_grid)),
            "mu_S": int(loader_S.n_rmu), "mu_L": int(loader_L.n_rmu),
            "q_layout": loader_L.q_layout}
    print_fn(f"[zeta_projection] ζ pair verified compatible: "
             f"μ_S={geom['mu_S']} μ_L={geom['mu_L']} n_q={geom['n_q_on_disk']} "
             f"({geom['q_layout']}) ngkmax={geom['ngkmax']} "
             f"FFT={geom['fft_grid']}; gvec_components identical "
             f"byte-for-byte.")
    return geom


# ---------------------------------------------------------------------------
# (†) the overlap / transfer numerator — G is the parallel axis
# ---------------------------------------------------------------------------

def zeta_overlap_block_reshard(
    mesh_xy: Mesh,
    *,
    axes: tuple[str, str] = ("x", "y"),
    mu_s_chunk: int | None = None,
    target_partial_mb: float = 256.0,
) -> Callable:
    """Build ``overlap(zeta_S, zeta_L) -> (O_x, O_y)``.

    Computes (†) with the **G axis flat-sharded over the whole mesh** —
    the contraction axis is the parallel axis, so nothing of size
    ``μ_L × n_G`` or ``μ_L²`` is ever rank-resident whole.

    Operands (global shape → spec)::

        zeta_S  (n_q, μ_S, n_G)   P(None, None, ('x','y'))
        zeta_L  (n_q, μ_L, n_G)   P(None, None, ('x','y'))
        O_x     (n_q, μ_S, μ_L)   P(None, None, 'x')
        O_y     (n_q, μ_S, μ_L)   P(None, None, 'y')

    Two shardings of the same matrix are produced in ONE pass because the
    congruence needs ``T`` on both mesh axes (ψ_left wants μ_L on 'x',
    ψ_right wants μ_L on 'y').  Deriving the second by resharding the
    first would hand the partitioner an all-to-all it plans itself
    (QUALITY_PATTERNS §4); both chains are therefore written
    structurally, inside the shard_map, from the SAME rank-local partial:

        loc   = Σ_{G_local} conj(ζ^S_chunk) ζ^L          rank-local
        O_x   = psum(psum_scatter(loc, 'x', dim=μ_L), 'y')
        O_y   = psum(psum_scatter(loc, 'y', dim=μ_L), 'x')

    Each chain completes the G reduction (scatter-axis group first, then
    the orthogonal group) while tiling μ_L, so the sum is over ALL ranks
    and the result lands already sharded.  ``loc`` is the only object of
    size ``(n_q, chunk, μ_L)``; ``mu_s_chunk`` bounds it independently of
    P (default: whatever keeps it under ``target_partial_mb``).
    """
    from common.shard_map import shard_map

    ax_x, ax_y = axes
    _check_mesh(mesh_xy, axes)

    def _overlap(zeta_S, zeta_L):
        if zeta_S.ndim != 3 or zeta_L.ndim != 3:
            raise ValueError(
                f"zeta operands must be rank 3 (n_q, μ, n_G); got "
                f"{tuple(zeta_S.shape)} / {tuple(zeta_L.shape)}")
        n_q, mu_s, n_g = (int(d) for d in zeta_S.shape)
        n_q_l, mu_l, n_g_l = (int(d) for d in zeta_L.shape)
        if n_q_l != n_q or n_g_l != n_g:
            raise ValueError(
                f"ζ_S {tuple(zeta_S.shape)} and ζ_L {tuple(zeta_L.shape)} "
                f"must agree on (n_q, n_G) — the overlap is per-q and over "
                f"the SAME per-q G-sphere.  Two ζ files built on different "
                f"FFT grids / cutoffs cannot be overlapped; refit one of "
                f"them on the other's grid.")
        _require_div(mu_l, mesh_xy, "mu_L", axes)
        from runtime.padding import authenticate_padded_axis
        authenticate_padded_axis(
            n_g, n_g, mesh_xy, name="zeta overlap G carrier",
            spec=P(None, None, (ax_x, ax_y)), axis=2)

        chunk = mu_s_chunk
        if chunk is None:
            per_row = n_q * mu_l * 16.0
            chunk = max(1, int(target_partial_mb * 1e6 / max(per_row, 1.0)))
        chunk = int(min(max(chunk, 1), mu_s))
        bounds = [(c, min(c + chunk, mu_s)) for c in range(0, mu_s, chunk)]

        def _body(zS, zL):
            xs, ys = [], []
            for c0, c1 in bounds:
                loc = jnp.einsum('qmg,qng->qmn',
                                 jnp.conj(zS[:, c0:c1, :]), zL,
                                 optimize=True)
                part_x = jax.lax.psum_scatter(
                    loc, ax_x, scatter_dimension=2, tiled=True)
                xs.append(jax.lax.psum(part_x, ax_y))
                part_y = jax.lax.psum_scatter(
                    loc, ax_y, scatter_dimension=2, tiled=True)
                ys.append(jax.lax.psum(part_y, ax_x))
            if len(xs) == 1:
                return xs[0], ys[0]
            return (jnp.concatenate(xs, axis=1),
                    jnp.concatenate(ys, axis=1))

        return shard_map(
            _body, mesh=mesh_xy,
            in_specs=(P(None, None, (ax_x, ax_y)),
                      P(None, None, (ax_x, ax_y))),
            out_specs=(P(None, None, ax_x), P(None, None, ax_y)),
            check_vma=False)(zeta_S, zeta_L)

    warm_mesh_cliques(mesh_xy)
    return _overlap


def zeta_overlap_single_axis(
    mesh_xy: Mesh, *, mu_axis: str, g_axis: str,
) -> Callable:
    """One-collective overlap with μ_L AND G both sharded — the scaling plan.

    ``zeta_overlap_block_reshard`` (above) shards ONLY G and therefore
    forms a rank-local partial that is μ_L-FULL: its reduce-scatter input
    is ``n_q·chunk·μ_L``, **independent of P**.  Measured consequence
    (jobs 7879488/7879491): the overlap stage is FLAT in P — 7.0 s at
    P=4 and 7.0 s at P=16 — while its arithmetic falls as 1/P.

    This plan removes that.  ζ_L is sharded on BOTH axes, μ_L over
    ``mu_axis`` and G over ``g_axis``; ζ_S is sharded on G over
    ``g_axis`` only (μ_S replicated — μ_S is small by premise).  Then::

        loc[q, μ_S, μ_L_local] = Σ_{G_local} conj(ζ^S) ζ^L    rank-local
        O                       = psum(loc, g_axis)

    is the whole thing: ONE all-reduce, whose payload
    ``n_q·μ_S·μ_L/p_mu`` FALLS with the mesh, and whose rank-local
    partial IS the output shard — so there is no transient larger than
    the result and no μ_S chunking is needed at all.  ``psum_scatter`` is
    unnecessary here precisely because μ_L is already tiled on entry.

    The cost is that the caller must supply ζ in two layouts, one per
    output sharding (μ on 'x'/G on 'y' for ``O_x``, μ on 'y'/G on 'x' for
    ``O_y``) — one reshard, or one re-read, or (as in the synthetic
    driver, where ζ is analytic) one regeneration.  That is the trade:
    a P-independent collective, or a one-off operand reshard.

    Operands (global shape → spec)::

        zeta_S  (n_q, μ_S, n_G)   P(None, None, g_axis)
        zeta_L  (n_q, μ_L, n_G)   P(None, mu_axis, g_axis)
        O       (n_q, μ_S, μ_L)   P(None, None, mu_axis)

    Per-rank ζ_S is ``n_q·μ_S·n_G/p_g`` — it falls with ``p_g``, not with
    P.  Stated rather than hidden: it is bounded by the SMALL basis
    (65 MB at n_q=16, μ_S=504, n_G=4032, p_g=8) and can never approach
    μ_L², but it is the one operand here that does not scale with the
    full mesh.
    """
    from common.shard_map import shard_map

    names = tuple(mesh_xy.axis_names)
    if mu_axis not in names or g_axis not in names or mu_axis == g_axis:
        raise ValueError(
            f"zeta_overlap_single_axis: mu_axis={mu_axis!r} and "
            f"g_axis={g_axis!r} must be two DISTINCT axes of the mesh "
            f"{names} — μ_L and G are tiled on different mesh axes so the "
            f"rank-local partial is already the output shard.")
    def _overlap(zeta_S, zeta_L):
        n_q, mu_s, n_g = (int(d) for d in zeta_S.shape)
        n_q_l, mu_l, n_g_l = (int(d) for d in zeta_L.shape)
        if n_q_l != n_q or n_g_l != n_g:
            raise ValueError(
                f"ζ_S {tuple(zeta_S.shape)} and ζ_L {tuple(zeta_L.shape)} "
                f"must agree on (n_q, n_G) — the overlap is per-q and over "
                f"the SAME per-q G-sphere.")
        from runtime.padding import authenticate_padded_axis
        authenticate_padded_axis(
            mu_l, mu_l, mesh_xy,
            name="single-axis zeta large-centroid carrier",
            spec=P(None, mu_axis, g_axis), axis=1)
        authenticate_padded_axis(
            n_g, n_g, mesh_xy, name="single-axis zeta G carrier",
            spec=P(None, mu_axis, g_axis), axis=2)

        def _body(zS, zL):
            loc = jnp.einsum('qmg,qng->qmn', jnp.conj(zS), zL, optimize=True)
            return jax.lax.psum(loc, g_axis)

        return shard_map(
            _body, mesh=mesh_xy,
            in_specs=(P(None, None, g_axis), P(None, mu_axis, g_axis)),
            out_specs=P(None, None, mu_axis), check_vma=False)(zeta_S, zeta_L)

    warm_mesh_cliques(mesh_xy)
    return _overlap


def zeta_gram_single_axis(mesh_xy: Mesh, *, g_axis: str) -> Callable:
    """``gram(zeta_S) -> G_S`` for the :func:`zeta_overlap_single_axis` layout.

    ζ_S is G-sharded over ``g_axis`` and replicated over the other axis, so
    one ``psum`` over ``g_axis`` completes the G sum and leaves ``G_S``
    replicated everywhere — the same bounded (n_q, μ_S, μ_S) object the
    one-pass plan produces, by a cheaper route.
    """
    from common.shard_map import shard_map

    names = tuple(mesh_xy.axis_names)
    if g_axis not in names:
        raise ValueError(f"g_axis={g_axis!r} not in mesh axes {names}")
    def _gram(zeta_S):
        n_q, mu_s, n_g = (int(d) for d in zeta_S.shape)
        from runtime.padding import authenticate_padded_axis
        authenticate_padded_axis(
            n_g, n_g, mesh_xy, name="single-axis zeta Gram G carrier",
            spec=P(None, None, g_axis), axis=2)

        def _body(zS):
            loc = jnp.einsum('qmg,qng->qmn', jnp.conj(zS), zS, optimize=True)
            return jax.lax.psum(loc, g_axis)

        out_spec = P(None, None, None)
        return shard_map(
            _body, mesh=mesh_xy, in_specs=(P(None, None, g_axis),),
            out_specs=out_spec, check_vma=False)(zeta_S)

    warm_mesh_cliques(mesh_xy)
    return _gram


def zeta_gram_replicated(
    mesh_xy: Mesh, *, axes: tuple[str, str] = ("x", "y"),
    mu_s_chunk: int | None = None, target_partial_mb: float = 256.0,
) -> Callable:
    """Build ``gram(zeta_S) -> G_S`` of shape ``(n_q, μ_S, μ_S)``, REPLICATED.

    The *small* basis's own Gram (‡).  Replicated deliberately: it is the
    one object here whose extent is set by μ_S, the dimension the whole
    exercise makes small, and the subsequent solve ``G_S T = O`` is then
    embarrassingly parallel over the μ_L columns (no collective).  Size is
    announced by the caller; at μ_S=504, n_q=16 it is 65 MB.  If a future
    caller wants μ_S in the thousands this is the line to revisit — it is
    the only place in the module where an object does not fall with P.
    """
    from common.shard_map import shard_map

    ax_x, ax_y = axes
    p_x, p_y = _check_mesh(mesh_xy, axes)

    def _gram(zeta_S):
        n_q, mu_s, n_g = (int(d) for d in zeta_S.shape)
        from runtime.padding import authenticate_padded_axis
        authenticate_padded_axis(
            n_g, n_g, mesh_xy, name="replicated zeta Gram G carrier",
            spec=P(None, None, (ax_x, ax_y)), axis=2)
        chunk = mu_s_chunk
        if chunk is None:
            per_row = n_q * mu_s * 16.0
            chunk = max(1, int(target_partial_mb * 1e6 / max(per_row, 1.0)))
        chunk = int(min(max(chunk, 1), mu_s))
        bounds = [(c, min(c + chunk, mu_s)) for c in range(0, mu_s, chunk)]

        def _body(zS):
            out = []
            for c0, c1 in bounds:
                loc = jnp.einsum('qmg,qng->qmn',
                                 jnp.conj(zS[:, c0:c1, :]), zS, optimize=True)
                out.append(jax.lax.psum(loc, (ax_x, ax_y)))
            return out[0] if len(out) == 1 else jnp.concatenate(out, axis=1)

        return shard_map(
            _body, mesh=mesh_xy,
            in_specs=(P(None, None, (ax_x, ax_y)),),
            out_specs=P(None, None, None), check_vma=False)(zeta_S)

    warm_mesh_cliques(mesh_xy)
    return _gram


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
                f"rcond=1e-10, mode='ls_trunc' at the build_zeta_transfer "
                f"level) to use the rank-truncated pseudo-inverse — the "
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


def build_zeta_transfer(
    zeta_S, zeta_L, mesh_xy: Mesh, *,
    axes: tuple[str, str] = ("x", "y"),
    mode: str = "ls",
    rcond: float | None = None,
    mu_s_chunk: int | None = None,
    print_fn=print,
    announce: bool = True,
):
    """ζ_S, ζ_L → the congruence operands ``(psi_left, psi_right)``.

    ``mode="ls"`` (default) returns the least-squares transfer
    ``T = G_S^{-1} O`` of (‡): the minimizer of the r-space error, exact
    on fields representable in the small basis, and invariant to the
    metric's overall normalization.  ``mode="raw"`` returns the bare
    overlap ``O`` of (†) — the Galerkin numerator — and announces that
    the result then carries the metric's scale squared.  Nothing is
    chosen silently.

    Returns ``(psi_left, psi_right, info)`` where ``info`` carries the
    Gram bytes and the transfer's global shape for the caller's log.
    """
    if mode not in ("ls", "ls_trunc", "raw"):
        raise ValueError(
            f"zeta_projection mode must be 'ls' (least-squares transfer via "
            f"Cholesky), 'ls_trunc' (rank-truncated pseudo-inverse — "
            f"REQUIRED on real ISDF bases, whose Gram is numerically "
            f"singular; pass rcond) or 'raw' (bare overlap congruence), "
            f"got {mode!r}")
    if mode == "ls_trunc" and rcond is None:
        raise ValueError(
            "mode='ls_trunc' needs an explicit rcond (relative eigenvalue "
            "threshold, e.g. 1e-10).  It is not defaulted: the retained "
            "rank is a physics choice and must be stated by the caller, "
            "the same way the ISDF fit states its RCOND.")
    if mode == "ls" and rcond is not None:
        raise ValueError(
            "rcond is only meaningful with mode='ls_trunc'; mode='ls' is "
            "the exact Cholesky solve.")
    ax_x, ax_y = axes
    p_x, p_y = _check_mesh(mesh_xy, axes)
    n_q, mu_s, n_g = (int(d) for d in zeta_S.shape)
    mu_l = int(zeta_L.shape[1])
    _require_div(mu_s, mesh_xy, "mu_S", axes)
    _require_div(mu_l, mesh_xy, "mu_L", axes)

    overlap = zeta_overlap_block_reshard(
        mesh_xy, axes=axes, mu_s_chunk=mu_s_chunk)
    O_x, O_y = jax.jit(overlap)(zeta_S, zeta_L)

    gram_bytes = n_q * mu_s * mu_s * 16
    if mode == "raw":
        if announce:
            print_fn(
                f"[zeta_projection] mode=raw: congruence uses the BARE "
                f"overlap O = <ζ_S|ζ_L> (†).  W_S then carries the ζ "
                f"metric's normalization SQUARED (N_r² at norm='backward'); "
                f"mode='ls' is the metric-invariant least-squares transfer.")
        T_x, T_y = O_x, O_y
    else:
        gram = zeta_gram_replicated(mesh_xy, axes=axes, mu_s_chunk=mu_s_chunk)
        G_S = jax.jit(gram)(zeta_S)
        if announce:
            print_fn(
                f"[zeta_projection] mode=ls: T = G_S^-1·O.  G_S is the ONLY "
                f"replicated object ({n_q}×{mu_s}×{mu_s} c128 = "
                f"{gram_bytes / 1e6:.1f} MB/rank, bounded by the SMALL "
                f"basis); T is (n_q={n_q}, μ_S={mu_s}, μ_L={mu_l}) sharded "
                f"on μ_L, {n_q * mu_s * mu_l * 16 / 1e6 / p_x:.1f} MB/rank "
                f"on '{ax_x}' + {n_q * mu_s * mu_l * 16 / 1e6 / p_y:.1f} "
                f"MB/rank on '{ax_y}'.")
        T_x = least_squares_transfer(G_S, O_x, mesh_xy, ax_x,
                                     rcond=rcond, print_fn=print_fn)
        T_y = least_squares_transfer(G_S, O_y, mesh_xy, ax_y,
                                     rcond=rcond, print_fn=lambda *a: None)

    psi_left, psi_right = jax.jit(
        lambda a, b: transfer_operands_from_dense(a, b, mesh_xy, axes)
    )(T_x, T_y)
    info = {
        "n_q": n_q, "mu_S": mu_s, "mu_L": mu_l, "n_G": n_g,
        "gram_bytes": gram_bytes, "mode": mode,
        "transfer_bytes_global": n_q * mu_s * mu_l * 16,
    }
    return psi_left, psi_right, info


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

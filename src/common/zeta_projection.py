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

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.contract_bands import (
    contract_bands_block_reshard,
    ensure_grouped_collectives_ready,
)

__all__ = [
    "ensure_world_clique_ready",
    "zeta_overlap_block_reshard",
    "zeta_overlap_single_axis",
    "zeta_gram_single_axis",
    "zeta_gram_replicated",
    "least_squares_transfer",
    "build_zeta_transfer",
    "project_w_between_zeta_bases",
    "transfer_operands_from_dense",
]


# ---------------------------------------------------------------------------
# impl=mpi world-clique warm-up — stronger than the primitive's helper
# ---------------------------------------------------------------------------

_world_clique_warmed = False


def ensure_world_clique_ready(mesh_xy: Mesh, *,
                              axes: tuple[str, str] = ("x", "y"),
                              print_fn=print) -> bool:
    """Warm the impl=mpi WORLD clique — and ANNOUNCE that it is not enough.

    Background.  ``common.contract_bands.ensure_grouped_collectives_ready``
    encodes the memo's world-collective-first contract (RESHARD_OVERHEAD_MEMO
    Sec. 4.2, job 7878883): under
    ``JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi`` a GROUPED clique can only be
    created after a WORLD-clique collective has first-touched the runtime.
    It implements that with ``common.collectives.barrier`` →
    ``multihost_utils.sync_global_devices``.

    **Measured (job 7879485, P=4, 5 variants, one fresh process each): under
    impl=mpi NO warm-up makes a standalone driver's grouped psum_scatter
    work.**  ``none``, ``sync_global_devices``, a ``lax.psum`` over both
    mesh axes on the caller's own mesh, a world all-gather on the caller's
    own mesh, and psum+sgd together ALL die with

        UNKNOWN: MPI: Communicator requested from a thread that is not the
        one MPI was initialized from.

    (jaxlib's ``MpiCollectives::CreateCommunicators`` gates on
    ``MPI_Is_thread_main``; XLA:CPU executes the collective from its thread
    pool.)  The same probe run shows **gloo with the ib0 pin passing the
    identical chain with no warm-up at all** — consistent with
    RESHARD_OVERHEAD_MEMO Sec. 4.3, which measured this exact staged
    reduce-scatter chain on gloo/ib0 at P=64 (job 7878883 step 1c, rc=0).
    Production GW *does* run grouped collectives under impl=mpi (job
    7879010, 8/8 passes rc=0), so the contract is satisfiable — but by
    something in the full gw_jax process that a standalone driver does not
    reproduce, and that was not chased down here.

    This helper therefore (a) still issues the world-clique collective the
    memo's probe used — cheap, harmless, and the right thing if the
    upstream contract is ever the whole story again — and (b) ANNOUNCES the
    known-bad configuration on rank 0 rather than letting the caller
    discover it as an ``UNKNOWN`` error mid-run.  It is NOT a refusal:
    production runs this configuration successfully, so refusing would
    break a working path.

    A no-op off impl=mpi and in single-process runs.  MUST be called
    synchronously on every rank (it is a collective).  The factories below
    call it, so consumers inherit it.

    Returns True when a warm-up collective was actually executed.
    """
    global _world_clique_warmed
    if _world_clique_warmed:
        return False
    impl = os.environ.get(
        "JAX_CPU_COLLECTIVES_IMPLEMENTATION", "").strip().lower()
    if impl != "mpi":
        return False
    from common.collectives import process_count, device_put_process_local
    if process_count() <= 1:
        return False
    from jax.experimental.shard_map import shard_map
    import numpy as np

    if jax.process_index() == 0:
        print_fn(
            "*** zeta_projection: JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi in a "
            "standalone driver.  The grouped psum_scatter this module issues "
            "is MEASURED to fail here with 'MPI: Communicator requested from "
            "a thread that is not the one MPI was initialized from' under "
            "EVERY known warm-up (job 7879485: none / sync_global_devices / "
            "world psum / world all-gather / both).  Use "
            "JAX_CPU_COLLECTIVES_IMPLEMENTATION=gloo with the ib0 pin "
            "(runtime.pin_gloo_interface) — measured green for this exact "
            "chain in the same probe and at P=64 in job 7878883.  Warming "
            "the world clique anyway and continuing. ***")
    ax = tuple(axes)
    one = device_put_process_local(
        np.ones(1, dtype=np.float64), NamedSharding(mesh_xy, P(None)))
    fn = jax.jit(shard_map(
        lambda a: jax.lax.psum(a, ax), mesh=mesh_xy,
        in_specs=(P(None),), out_specs=P(None), check_rep=False))
    total = float(jax.device_get(fn(one))[0])
    _world_clique_warmed = True
    if total != float(mesh_xy.size) and jax.process_index() == 0:
        print_fn(f"*** zeta_projection world-clique warm-up returned {total} "
                 f"but the mesh has {mesh_xy.size} devices — the all-reduce "
                 f"did not span the mesh. ***")
    return True


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


def _require_div(n: int, px: int, py: int, what: str, axes) -> None:
    if n % px or n % py:
        raise ValueError(_MU_DIV_MSG.format(
            what=what, n=n, px=px, py=py, ax_x=axes[0], ax_y=axes[1]))


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
    from jax.experimental.shard_map import shard_map

    ax_x, ax_y = axes
    p_x, p_y = _check_mesh(mesh_xy, axes)

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
        _require_div(mu_l, p_x, p_y, "mu_L", axes)
        if n_g % (p_x * p_y):
            raise ValueError(
                f"zeta_projection: n_G={n_g} must be a multiple of the mesh "
                f"size p_{ax_x}·p_{ax_y}={p_x * p_y} — G is the contraction "
                f"axis and is flat-sharded over the whole mesh.  Pad the "
                f"G-sphere with zero rows (ngkmax padding is already the "
                f"on-disk ζ contract: file_io.zeta_loader zero-fills past "
                f"ngk[q]) or drop to a mesh whose size divides n_G.")

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
            check_rep=False)(zeta_S, zeta_L)

    ensure_world_clique_ready(mesh_xy, axes=axes)
    ensure_grouped_collectives_ready()
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
    from jax.experimental.shard_map import shard_map

    names = tuple(mesh_xy.axis_names)
    if mu_axis not in names or g_axis not in names or mu_axis == g_axis:
        raise ValueError(
            f"zeta_overlap_single_axis: mu_axis={mu_axis!r} and "
            f"g_axis={g_axis!r} must be two DISTINCT axes of the mesh "
            f"{names} — μ_L and G are tiled on different mesh axes so the "
            f"rank-local partial is already the output shard.")
    p_mu = int(mesh_xy.shape[mu_axis])
    p_g = int(mesh_xy.shape[g_axis])

    def _overlap(zeta_S, zeta_L):
        n_q, mu_s, n_g = (int(d) for d in zeta_S.shape)
        n_q_l, mu_l, n_g_l = (int(d) for d in zeta_L.shape)
        if n_q_l != n_q or n_g_l != n_g:
            raise ValueError(
                f"ζ_S {tuple(zeta_S.shape)} and ζ_L {tuple(zeta_L.shape)} "
                f"must agree on (n_q, n_G) — the overlap is per-q and over "
                f"the SAME per-q G-sphere.")
        if mu_l % p_mu:
            raise ValueError(
                f"zeta_overlap_single_axis: mu_L={mu_l} must be a multiple "
                f"of p_{mu_axis}={p_mu} (ζ_L is block-sharded on that axis). "
                f"Pad the large centroid set with zero ζ rows — inert in "
                f"the overlap — and slice the pad off T afterwards.")
        if n_g % p_g:
            raise ValueError(
                f"zeta_overlap_single_axis: n_G={n_g} must be a multiple of "
                f"p_{g_axis}={p_g} (G is the sharded contraction axis). Pad "
                f"the G-sphere with zero rows; that is already the on-disk "
                f"ngkmax contract (file_io.zeta_loader zero-fills past "
                f"ngk[q]).")

        def _body(zS, zL):
            loc = jnp.einsum('qmg,qng->qmn', jnp.conj(zS), zL, optimize=True)
            return jax.lax.psum(loc, g_axis)

        return shard_map(
            _body, mesh=mesh_xy,
            in_specs=(P(None, None, g_axis), P(None, mu_axis, g_axis)),
            out_specs=P(None, None, mu_axis), check_rep=False)(zeta_S, zeta_L)

    ensure_world_clique_ready(mesh_xy)
    ensure_grouped_collectives_ready()
    return _overlap


def zeta_gram_single_axis(mesh_xy: Mesh, *, g_axis: str) -> Callable:
    """``gram(zeta_S) -> G_S`` for the :func:`zeta_overlap_single_axis` layout.

    ζ_S is G-sharded over ``g_axis`` and replicated over the other axis, so
    one ``psum`` over ``g_axis`` completes the G sum and leaves ``G_S``
    replicated everywhere — the same bounded (n_q, μ_S, μ_S) object the
    one-pass plan produces, by a cheaper route.
    """
    from jax.experimental.shard_map import shard_map

    names = tuple(mesh_xy.axis_names)
    if g_axis not in names:
        raise ValueError(f"g_axis={g_axis!r} not in mesh axes {names}")
    p_g = int(mesh_xy.shape[g_axis])
    other = [a for a in names if a != g_axis]

    def _gram(zeta_S):
        n_q, mu_s, n_g = (int(d) for d in zeta_S.shape)
        if n_g % p_g:
            raise ValueError(
                f"zeta_gram_single_axis: n_G={n_g} must be a multiple of "
                f"p_{g_axis}={p_g}.")

        def _body(zS):
            loc = jnp.einsum('qmg,qng->qmn', jnp.conj(zS), zS, optimize=True)
            return jax.lax.psum(loc, g_axis)

        out_spec = P(None, None, None)
        return shard_map(
            _body, mesh=mesh_xy, in_specs=(P(None, None, g_axis),),
            out_specs=out_spec, check_rep=False)(zeta_S)

    ensure_world_clique_ready(mesh_xy)
    ensure_grouped_collectives_ready()
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
    from jax.experimental.shard_map import shard_map

    ax_x, ax_y = axes
    p_x, p_y = _check_mesh(mesh_xy, axes)

    def _gram(zeta_S):
        n_q, mu_s, n_g = (int(d) for d in zeta_S.shape)
        if n_g % (p_x * p_y):
            raise ValueError(
                f"zeta_projection.gram: n_G={n_g} must be a multiple of the "
                f"mesh size {p_x * p_y} (G is the sharded contraction axis).")
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
            out_specs=P(None, None, None), check_rep=False)(zeta_S)

    ensure_world_clique_ready(mesh_xy, axes=axes)
    ensure_grouped_collectives_ready()
    return _gram


# ---------------------------------------------------------------------------
# (‡) the least-squares transfer
# ---------------------------------------------------------------------------

def least_squares_transfer(
    gram_S, O_sharded, mesh_xy: Mesh, mu_axis: str,
) -> jax.Array:
    """``T = G_S^{-1} · O`` for one sharding of ``O``.

    ``G_S`` is Hermitian positive definite whenever the small ζ set is
    linearly independent, so the solve is a Cholesky back-substitution.
    It is done INSIDE a shard_map with ``G_S`` replicated and ``O``'s μ_L
    axis sharded, which makes the solve rank-local by construction — the
    partitioner is never asked to plan a collective for it.

    Refuses (does not ridge, does not pinv) when the Cholesky fails: a
    non-PD small-basis Gram means duplicated / linearly dependent
    centroids in the SMALL set, and silently regularizing it would hide
    a basis bug behind a plausible-looking W_S.
    """
    from jax.experimental.shard_map import shard_map

    chol = jnp.linalg.cholesky(gram_S)          # (n_q, μ_S, μ_S), replicated
    bad = jnp.isnan(jnp.real(jnp.diagonal(chol, axis1=-2, axis2=-1))).any()
    if bool(jax.device_get(bad)):
        diag = jnp.real(jnp.diagonal(gram_S, axis1=-2, axis2=-1))
        raise ValueError(
            f"zeta_projection: the SMALL-basis Gram G_S = <ζ^S|ζ^S> is not "
            f"positive definite (Cholesky produced NaN).  That means the "
            f"small ζ set is linearly dependent on the G-sphere — typically "
            f"duplicated centroids, a small basis larger than the sphere "
            f"rank, or a ζ file whose trailing μ rows are the zero pad "
            f"(strip the pad, or pass mu_S = the LOGICAL centroid count).  "
            f"Gram diagonal min/max = "
            f"{float(jnp.min(diag)):.3e}/{float(jnp.max(diag)):.3e}.  No "
            f"ridge is applied and no pseudo-inverse is substituted: a "
            f"silent regularization here would be indistinguishable from a "
            f"correct projection downstream.")

    def _body(L_loc, O_loc):
        # L_loc replicated (n_q, μ_S, μ_S); O_loc (n_q, μ_S, μ_L_local).
        y = jax.lax.linalg.triangular_solve(
            L_loc, O_loc, left_side=True, lower=True,
            transpose_a=False, conjugate_a=False)
        return jax.lax.linalg.triangular_solve(
            L_loc, y, left_side=True, lower=True,
            transpose_a=True, conjugate_a=True)

    return shard_map(
        _body, mesh=mesh_xy,
        in_specs=(P(None, None, None), P(None, None, mu_axis)),
        out_specs=P(None, None, mu_axis), check_rep=False)(chol, O_sharded)


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
    if mode not in ("ls", "raw"):
        raise ValueError(
            f"zeta_projection mode must be 'ls' (least-squares transfer, "
            f"G_S^-1·O) or 'raw' (bare overlap congruence), got {mode!r}")
    ax_x, ax_y = axes
    p_x, p_y = _check_mesh(mesh_xy, axes)
    n_q, mu_s, n_g = (int(d) for d in zeta_S.shape)
    mu_l = int(zeta_L.shape[1])
    _require_div(mu_s, p_x, p_y, "mu_S", axes)
    _require_div(mu_l, p_x, p_y, "mu_L", axes)

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
        T_x = least_squares_transfer(G_S, O_x, mesh_xy, ax_x)
        T_y = least_squares_transfer(G_S, O_y, mesh_xy, ax_y)

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
    ensure_world_clique_ready(mesh_xy, axes=axes)
    inner = contract_bands_block_reshard(
        mesh_xy, channels="none", extra="none", axes=axes,
        divisibility_hint=(
            "Here m = n = mu_S (the SMALL ζ basis) and the contracted axis "
            "is mu_L: pad the small centroid set to a multiple of p_x and "
            "of p_y (zero ζ rows are inert in the overlap) and slice the "
            "pad off W_S afterwards."))
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

"""Streamed real-space Galerkin projection over a two-dimensional mesh.

This module owns the named distributed pattern shared by operator-basis
fits: for each bounded real-space chunk, sum every contracted band chunk
into ``Q`` before folding ``Q Q^H`` into the projected Gram matrix.  The
ordering is load-bearing.  Folding once per band chunk would omit all
cross-band terms while still producing a plausible Hermitian matrix.

Wavefunction loading and G-to-r transforms remain owned by
``common.wfn_transforms``.  The caller resolves policy such as environment
overrides and the device-pool limit, then passes explicit values here.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.shard_map import shard_map
from runtime.padding import round_up, spec_divisor


__all__ = [
    "GalerkinStreamPlan",
    "build_streamed_projected_gram",
    "galerkin_q_ledger",
    "plan_galerkin_stream",
]


_accum_G_cache: dict = {}
_fold_G_cache: dict = {}


def _make_accum_kernel(rank_, bc_size, nspinor_, mesh_, rep_,
                       psi_layout_, sharding_q_):
    """One compiled Galerkin Q-accumulation kernel per static config.

    THE psi INPUT SHARDING IS PINNED to Q's r layout (in_shardings); the
    iterator's canonical staged reshard puts it there before this kernel.
    The historical kernel reshaped (nk, bc, ns, r_'y') to
    (nk·bc, ns·r) — merging the replicated spinor axis with the sharded r
    axis, which no NamedSharding can express — and then asked for
    ('x','y')-sharded output columns.  The legacy SPMD partitioner cannot
    synthesize that band-to-r exchange: at P16 it fell back to fully
    replicating the 54.3-GiB c128[81,1,2,1406256] slab per 40-GB GPU and
    OOMed (Perlmutter JID 57271407, "Involuntary full rematerialization",
    125.60 GiB live).  With psi already r-sharded in Q's own layout and
    the spinor axis kept SEPARATE, the contraction is purely
    device-local: UH_bc is replicated, psi's contracted (nk·bc) rows are
    all present locally, and each device writes only its own r block of
    Q.  Module-level so the P>1 transition twin
    (``tests/test_htransform_q_accum.py``) drives the production kernel,
    not a re-implementation.
    """
    key = (id(mesh_), rank_, bc_size, nspinor_)
    fn = _accum_G_cache.get(key)
    if fn is not None:
        return fn

    # Only the accumulator can alias the result.  ``psi_bc`` is a streamed
    # read-only source and cannot back an output with Q's different shape;
    # advertising it as donated merely asks XLA for an impossible alias and
    # emits one warning per band chunk.
    @partial(jax.jit, donate_argnums=(3,),
             in_shardings=(rep_, rep_, psi_layout_, sharding_q_),
             out_shardings=sharding_q_)
    def _accum(UH_bc, inv_s, psi_bc, Q_in):
        # UH_bc: (rank, nk·bc) replicated; inv_s: (rank,1) replicated
        # psi_bc: (nk, bc, ns, r) sharded P(None,None,None,r_entry)
        nkv, bcv, nsv, rcv = psi_bc.shape
        # Merge only the two REPLICATED leading axes; the sharded r
        # axis and the spinor axis are never combined.
        psi_flat = psi_bc.reshape(nkv * bcv, nsv, rcv)
        Q = inv_s[:, :, None] * jnp.einsum(
            'ak,ksr->asr', UH_bc, psi_flat, optimize=True)
        return Q_in + Q

    _accum_G_cache[key] = _accum
    return _accum


def _make_fold_G_kernel(rank_, mesh_, sharding_q_, grid_xy_):
    """Add one already-bounded, r-sharded ``Q_chunk Q_chunk†`` to G.

    The caller owns the zeta-style outer-r loop and therefore never hands this
    executable Q over all ``r_tot``.  Each device forms the Gram contribution
    from its unique local-r shard; the established two-stage
    ``psum_scatter`` sums those shards while distributing matrix rows and
    columns onto ``P('x','y')``.
    """
    key = (id(mesh_), int(rank_), tuple(sharding_q_.spec),
           tuple(grid_xy_.spec))
    fn = _fold_G_cache.get(key)
    if fn is not None:
        return fn

    p_x = int(mesh_.shape['x'])
    p_y = int(mesh_.shape['y'])
    if rank_ % p_x or rank_ % p_y:
        raise ValueError(
            "_make_fold_G_kernel: the carried Galerkin rank must divide "
            f"both mesh axes; rank={rank_}, mesh={p_x}x{p_y}")

    @partial(
        shard_map,
        mesh=mesh_,
        in_specs=(sharding_q_.spec, P('x', 'y')),
        out_specs=P('x', 'y'),
        check_vma=False,
    )
    def _fold_local(Q_local, G_local):
        partial = jnp.einsum(
            'asr,bsr->ab', Q_local, jnp.conj(Q_local),
            optimize=True)
        partial = jax.lax.psum_scatter(
            partial, 'x', scatter_dimension=0, tiled=True)
        partial = jax.lax.psum_scatter(
            partial, 'y', scatter_dimension=1, tiled=True)
        return G_local + partial

    fn = jax.jit(
        _fold_local,
        donate_argnums=(1,),
        in_shardings=(sharding_q_, grid_xy_),
        out_shardings=grid_xy_,
    )
    _fold_G_cache[key] = fn
    return fn


@dataclass(frozen=True)
class GalerkinStreamPlan:
    """The mesh-aligned outer-r schedule selected after rank is known."""

    r_chunk_ranges: tuple[tuple[int, int], ...]
    max_r_logical: int
    max_r_carrier: int
    q_tile_local_bytes: int


def plan_galerkin_stream(*, rank: int, nspinor: int, n_rtot: int,
                         r_mesh_divisor: int,
                         q_tile_budget: int) -> GalerkinStreamPlan:
    """Choose the incumbent Q-budget-bounded, mesh-aligned r schedule."""
    q_bytes_per_local_r = (
        rank * nspinor * np.dtype(np.complex128).itemsize)
    if q_bytes_per_local_r > q_tile_budget:
        raise ValueError(
            "streaming_galerkin_solve: one local r column of Q needs "
            f"{q_bytes_per_local_r / 1024**3:.6f} GiB/device, exceeding "
            f"LORRAX_GALERKIN_CHUNK_GIB={q_tile_budget / 1024**3:.6f}. "
            "Increase that budget or reduce the retained rank.")
    r_local_cap = q_tile_budget // q_bytes_per_local_r
    r_chunk = min(n_rtot, r_local_cap * r_mesh_divisor)
    if r_chunk < n_rtot:
        r_chunk = max(
            r_mesh_divisor,
            (r_chunk // r_mesh_divisor) * r_mesh_divisor,
        )
    r_chunk_ranges = tuple(
        (r0, min(r0 + r_chunk, n_rtot))
        for r0 in range(0, n_rtot, r_chunk)
    )
    max_r_logical = max(r1 - r0 for r0, r1 in r_chunk_ranges)
    max_r_carrier = round_up(max_r_logical, r_mesh_divisor)
    q_tile_local_bytes = (
        rank * nspinor * (max_r_carrier // r_mesh_divisor)
        * np.dtype(np.complex128).itemsize
    )
    return GalerkinStreamPlan(
        r_chunk_ranges=r_chunk_ranges,
        max_r_logical=max_r_logical,
        max_r_carrier=max_r_carrier,
        q_tile_local_bytes=q_tile_local_bytes,
    )


def galerkin_q_ledger(*, rank: int, nk: int, nspinor: int, n_rtot: int,
                      band_chunk: int, m_states: int, mu_pad: int,
                      psi_win_elems: int, p_total: int, q_shards: int,
                      y_shards: int) -> dict:
    """Per-device byte ledger for one streamed-Galerkin r chunk.

    ``n_rtot`` is the chunk carrier, never the full FFT grid.  The Q charge
    includes both the donated accumulation overlap and one conservative
    Q-sized conjugate/layout workspace for the subsequent Gram fold.  Thus
    the compiler may choose that workspace without violating the ledger, but
    no full-``r_tot`` Q exists to copy.

    Keys are printable labels; ``TOTAL`` is their sum.
    """
    C16 = 16.0
    led = {
        # The bounded Q accumulator, donated input/output overlap, and one
        # Q-sized fold workspace:
        # ``_accum`` donates Q_in and writes Q_in + delta into it, but the
        # GEMM's delta-Q result buffer coexists with the accumulator, so
        # the loop's floor is 2x Q per device.
        "Q r-chunk (x2 accumulation + x1 fold workspace)":
            3.0 * rank * nspinor * n_rtot * C16 / q_shards,
        # One streamed psi band chunk in Q's r layout (the _accum input).
        "psi chunk (r-layout)":
            1.0 * nk * band_chunk * nspinor * n_rtot * C16 / q_shards,
        # The source shard remains live during the two staged all-to-alls.
        # It is band-sharded over the same full mesh product, so source and
        # destination are equal-volume shards; no y-replicated carrier exists.
        "psi chunk (band-layout, transition overlap)":
            1.0 * nk * band_chunk * nspinor * n_rtot * C16 / q_shards,
        # Persistent Gram state, replicated on every device: coeffs
        # (m_states, rank), UH (rank, m_states), UH_kb (its eager-reshape
        # copy), the per-chunk UH_bc block and inv_s.
        "Gram state (replicated)":
            (3.0 * m_states * rank + 1.0 * rank * nk * band_chunk
             + 1.0 * rank) * C16,
        # Vh, mu-sharded on 'y', live from step 2 until B_at_mu.
        "Vh (mu on 'y')":
            1.0 * rank * nspinor * mu_pad * C16 / y_shards,
        # ``_fold_local`` forms the full rank×rank partial on every device
        # before the two psum_scatter operations distribute its rows/columns.
        "fold partial (replicated)": 1.0 * rank * rank * C16,
        # The G face, P('x','y').
        "G face": 1.0 * rank * rank * C16 / p_total,
        # The resident psi(G-flat) window, band-sharded over all P.
        "psi(G-flat) window": 1.0 * psi_win_elems * C16 / p_total,
    }
    led["TOTAL"] = sum(led.values())
    return led


def _refuse_unfit_galerkin_mesh(
        ledger: dict, *, rank: int, nk: int, nspinor: int, n_rtot: int,
        band_chunk: int, m_states: int, mu_pad: int, psi_win_elems: int,
        mesh_xy: Mesh, q_spec, device_pool_limit: float | None,
        log_fn) -> None:
    """Refuse a non-fitting live set before any Q compilation.

    ``device_pool_limit`` is resolved by the caller through the owned GPU
    memory reader.  Passing it explicitly keeps machine policy out of this
    numerical owner while preserving the incumbent gate and diagnostics.
    """
    limit = device_pool_limit
    if limit is None or limit <= 0:
        log_fn("  [galerkin-mem] device pool size unreadable (CPU backend "
               "or platform allocator) — ledger printed above, the "
               "non-fitting-mesh refusal DID NOT RUN")
        return
    total = float(ledger["TOTAL"])
    if total <= float(limit):
        return
    fitting = None
    # Search ALL supported square meshes: a larger current mesh can use more
    # memory when another live object replicates, so the truthful remedy is
    # the minimum-P fitting geometry, not merely the first larger one.
    for s in range(1, 65):
        n_r_carrier_s = round_up(n_rtot, s * s)
        led_s = galerkin_q_ledger(
            rank=rank, nk=nk, nspinor=nspinor, n_rtot=n_r_carrier_s,
            band_chunk=band_chunk, m_states=m_states, mu_pad=mu_pad,
            psi_win_elems=psi_win_elems, p_total=s * s,
            q_shards=s * s, y_shards=s)
        if led_s["TOTAL"] <= float(limit):
            fitting = (s, float(led_s["TOTAL"]))
            break
    detail = "; ".join(
        f"{name} {b / 1024**3:.2f} GiB" for name, b in ledger.items()
        if name != "TOTAL")
    if fitting is not None:
        s, tot_s = fitting
        remedy = (f"the smallest square mesh that fits at this pool size is "
                  f"{s}x{s} (P={s * s}: projected "
                  f"{tot_s / 1024**3:.2f} GiB/device)")
    else:
        remedy = ("no square mesh up to 64x64 fits this r chunk; lower "
                  "LORRAX_GALERKIN_CHUNK_GIB or narrow the model rank")
    raise ValueError(
        f"streaming_galerkin_solve: the Galerkin live set does not fit "
        f"this mesh.  Projected {total / 1024**3:.2f} GiB/device against a "
        f"{limit / 1024**3:.2f} GiB pool on the "
        f"{mesh_xy.shape['x']}x{mesh_xy.shape['y']} mesh "
        f"(Q sharded {q_spec}, {spec_divisor(mesh_xy, q_spec, axis=2)}-way on "
        f"r).  Ledger: {detail}.  {remedy}.  This is refused BEFORE "
        f"compilation.  Lower LORRAX_GALERKIN_CHUNK_GIB to use more, smaller "
        f"r chunks without changing the fit.")


def build_streamed_projected_gram(
        *, wfn, meta, mesh_xy: Mesh, band_range: tuple[int, int],
        UH, inv_s, gram_init, band_chunk_size: int, mu_pad: int,
        q_tile_budget: int, device_pool_limit: float | None,
        bispinor: bool = False, log_fn=None, progress_fn=None):
    """Build ``G = Q Q^H`` with r outermost and contracted bands innermost.

    ``band_chunk_size`` is the already-resolved, mesh-aligned carrier width;
    ``q_tile_budget`` and ``device_pool_limit`` are caller-resolved policy.
    ``gram_init`` carries the caller's exact-null padding block and must be
    sharded on ``P('x', 'y')``.  ``progress_fn`` is an optional presentation
    callback; it does not participate in the numerical schedule.
    """
    if log_fn is None:
        log_fn = lambda *a, **kw: None

    from common.wfn_transforms import iter_psi_rchunk_bandwise

    b_start, b_end = band_range
    nb = b_end - b_start
    nk = int(meta.nk_tot)
    nspinor = int(meta.nspinor)
    n_rtot = int(meta.n_rtot)
    rank = int(UH.shape[0])
    m_states = nk * nb
    _bc = int(band_chunk_size)

    rep = NamedSharding(mesh_xy, P())
    grid_xy = NamedSharding(mesh_xy, P('x', 'y'))
    q_spec = P(None, None, ('y', 'x'))
    r_mesh_divisor = spec_divisor(mesh_xy, q_spec, axis=2)

    # Q[α, x] = Σ_{k,n} inv_s[α] U^H[α,(k,n)] ψ[(k,n), x].  The contraction
    # runs over the pair index (k, n) and is free in x = (spinor, r).  Thus
    # splitting r and summing G over the pieces is exact, while forming one
    # Gram per band chunk drops every cross-band term of
    # ``(Σ_bc Q_bc)(Σ_bc' Q_bc')^H``.  That wrong route stays Hermitian and
    # lets Cholesky succeed; its measured symptom on MoS2 12x12 / n_mu=640 /
    # nb=40 was a 1742.48-meV on-grid energy drift, against 0.63 meV for the
    # correct single-Q fold.  Bands are therefore innermost and every band
    # chunk is summed into Q before the fold below.
    #
    # The caller has already aligned ``_bc`` to the product-band mesh.  The
    # loader pads a narrower width to that same carrier anyway, so pushing
    # partial carriers through separate FFTs only adds iterations and compile
    # shapes; matching zero columns in UH below make the alignment inert.
    band_chunk_ranges = [
        (b_start + i * _bc, min(b_start + (i + 1) * _bc, b_end))
        for i in range((nb + _bc - 1) // _bc)
    ]

    # r is the free Q column.  Runtime padding adds exact-zero terminal
    # columns and product-shards them over the full mesh.
    sharding_q = NamedSharding(mesh_xy, q_spec)
    _q_r_entry = sharding_q.spec[2]
    q_shards = r_mesh_divisor
    psi_r_layout = NamedSharding(
        mesh_xy, P(None, None, None, _q_r_entry))

    plan = plan_galerkin_stream(
        rank=rank, nspinor=nspinor, n_rtot=n_rtot,
        r_mesh_divisor=r_mesh_divisor, q_tile_budget=q_tile_budget)
    log_fn(
        f"  Galerkin r plan: {len(plan.r_chunk_ranges)} chunk(s), "
        f"logical width <= {plan.max_r_logical}, "
        f"carrier <= {plan.max_r_carrier}; "
        f"Q <= {plan.q_tile_local_bytes / 1024**3:.2f} GiB/device "
        f"(budget {q_tile_budget / 1024**3:.2f} GiB/device)")

    _p_y_shards = int(mesh_xy.shape['y'])
    ledger = galerkin_q_ledger(
        rank=rank, nk=nk, nspinor=nspinor,
        n_rtot=plan.max_r_carrier, band_chunk=_bc,
        m_states=m_states, mu_pad=mu_pad, psi_win_elems=0,
        p_total=int(mesh_xy.size), q_shards=q_shards,
        y_shards=_p_y_shards)
    log_fn(f"  Galerkin Q ledger (per device): Q sharded "
           f"{sharding_q.spec} ({q_shards}-way on r), "
           + ", ".join(f"{name} {b / 1024**3:.2f} GiB"
                       for name, b in ledger.items()))
    _refuse_unfit_galerkin_mesh(
        ledger, rank=rank, nk=nk, nspinor=nspinor,
        n_rtot=plan.max_r_logical, band_chunk=_bc,
        m_states=m_states, mu_pad=mu_pad, psi_win_elems=0,
        mesh_xy=mesh_xy, q_spec=sharding_q.spec,
        device_pool_limit=device_pool_limit, log_fn=log_fn)

    fold_G = _make_fold_G_kernel(rank, mesh_xy, sharding_q, grid_xy)
    G = gram_init
    t0 = time.time()
    band_eval_count = 0
    UH_kb = UH.reshape(rank, nk, nb)
    from common.progress import LoopProgress
    progress_steps = len(plan.r_chunk_ranges) * len(band_chunk_ranges)
    progress = LoopProgress(
        progress_steps, progress_fn or (lambda *_: None),
        title="Galerkin real-space accumulation",
        item_name="(r, band) chunk",
        max_updates=min(progress_steps, 12),
        enabled=progress_fn is not None).start()

    @partial(jax.jit, static_argnums=(0,), out_shardings=sharding_q)
    def _alloc_Q(r_extent):
        return jnp.zeros(
            (rank, nspinor, r_extent), dtype=jnp.complex128)

    for r_idx, (r0, r1) in enumerate(plan.r_chunk_ranges):
        r_carrier = round_up(r1 - r0, r_mesh_divisor)
        Q = _alloc_Q(r_carrier)
        for bc_range, psi_bc_r in iter_psi_rchunk_bandwise(
                wfn, None, meta, mesh_xy, band_range, r0, r1, bispinor,
                band_chunk_size=_bc,
                band_chunk_ranges=band_chunk_ranges,
                band_pad_to=_bc,
                product_r_spec=psi_r_layout.spec):
            if int(psi_bc_r.shape[-1]) != r_carrier:
                raise ValueError(
                    "Galerkin iterator r carrier disagrees with Q chunk: "
                    f"psi={int(psi_bc_r.shape[-1])}, Q={r_carrier}, "
                    f"r=[{r0},{r1})")
            bc = bc_range[1] - bc_range[0]
            bc_lo = bc_range[0] - b_start
            bc_hi = bc_range[1] - b_start
            w = int(psi_bc_r.shape[1])
            UH_bc_kb = UH_kb[:, :, bc_lo:bc_hi]
            if w > bc:
                UH_bc_kb = jnp.pad(
                    UH_bc_kb, ((0, 0), (0, 0), (0, w - bc)))
            UH_bc = UH_bc_kb.reshape(rank, nk * w)

            accum = _make_accum_kernel(
                rank, w, nspinor, mesh_xy,
                rep, psi_r_layout, sharding_q)
            Q = accum(UH_bc, inv_s, psi_bc_r, Q)
            del psi_bc_r
            band_eval_count += 1
            progress.step()

        G = fold_G(Q, G)
        jax.block_until_ready(G)
        del Q
        log_fn(
            f"  G r-chunk {r_idx + 1}/{len(plan.r_chunk_ranges)}: "
            f"[{r0},{r1}) -> carrier {r_carrier}")

    progress.finish()
    log_fn(
        f"  G accumulation: {len(plan.r_chunk_ranges)} r chunk(s) x "
        f"{len(band_chunk_ranges)} band chunk(s) "
        f"({band_eval_count} streamed evaluations), {time.time()-t0:.2f}s")
    return G

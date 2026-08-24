import os
import math
import argparse
import numpy as np

# THE startup call (runtime module docstring): env defaults, fail-fast
# hook, jax.distributed, CPU fallback, the run's clique-warmed ('x','y')
# mesh, compile cache, rank-0 report.  MUST run before this module's own
# `import jax`; idempotent, so importing htransform as a LIBRARY from an
# already-started driver (bse.exciton_bands does) returns the same stack.
from runtime import initialize_communicator_stack
RUNTIME = initialize_communicator_stack()

import jax
import jax.numpy as jnp
from jax.scipy import linalg as jsp_linalg
from jax.scipy.special import erf
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from common.shard_map import shard_map
from functools import partial

from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                                 # owner's workspace fix -- see _services.py)

_services.ensure_on_path()

from wfn_loader import WfnLoader                                    # noqa: E402
from common import Meta
from common import rank_criterion
from common import spectral_closure
from common import timing
from common.units import RYD_TO_EV
from common.wfn_layout import band_sphere_spec
from runtime.padding import round_up, spec_divisor
from common.wfn_transforms import (
    get_enk_bandrange, load_centroids_band_chunked, iter_psi_rchunk_bandwise,
    load_psi_gflat_padded,
)
# ``eigh_backend`` + ``use_low_mem_eigh`` are ONE axis with ONE resolver;
# this driver reads a raw params dict rather than a LorraxConfig, which is
# exactly the case that function exists for.
from gw.gw_config import (
    distrib_la_batched_route_choices,
    eigh_backend_choices,
    resolve_distrib_la_batched_route,
    resolve_eigh_backend,
)
from common.fft_helpers import make_flat_k_ifftn
# Q's free r axis is zero-padded through ``runtime.padding`` and split over
# the full mesh product.  ``common.staged_reshard`` owns the exact
# product-band → product-r exchange used to put streamed wavefunctions there.
from common.collectives import gather_to_host
from common.sharding_fit import fit_sharding as _fit
from common.sharding_fit import padded_extent as _pad_to
from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                                 # owner's workspace fix -- see _services.py)

_services.ensure_on_path()

import symmetry_maps                                            # noqa: E402


def _build_mesh_xy() -> Mesh:
    """The run's ('x','y') mesh — the one the module-top startup call built.

    History, shortest form: this was seven lines hand-rolling a mesh (one of
    five dialects, 2026-07-30 audit), then ``resolve_mesh``, then
    ``prepare_mesh`` once measurement answered the warm-up question — without
    the clique warm-up this driver's first collective fires from an XLA
    parallel-executor worker thread and jaxlib refuses communicator creation
    (MPI_Is_thread_main false; P=16 job 7884867 failed on all 16 ranks at the
    first ``load_centroids`` collective).  Since 2026-08-01 the mesh and its
    warm-up come from ``initialize_communicator_stack()`` at the top of this
    module, so this helper only hands back ``RUNTIME.mesh`` for the library
    callers (``initialize_wfns`` with ``mesh_xy=None``) that resolve the mesh
    late.  A second ``prepare_mesh()`` here would be a second Mesh object —
    a second set of communicators and a second copy of every shape-keyed jit
    cache.
    """
    return RUNTIME.mesh


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

# Per-device ceiling on one streamed Galerkin Q r-chunk.
#
# The outer-r / inner-band loop mirrors zeta fitting: all band chunks are
# summed into Q for ONE r chunk before its Gram contribution is formed and Q
# is discarded.  This ceiling bounds that Q shard on each device.  Splitting
# r changes only the Gram reduction order; splitting bands before the outer
# product would drop cross terms and is forbidden.
# Override with LORRAX_GALERKIN_CHUNK_GIB (GiB, float).
#
# Resolved INSIDE the consuming function, not at module scope: the old
# module-level ``float(os.environ.get(...))`` meant a malformed export
# crashed ``import bandstructure.htransform`` itself — a bare
# ``ValueError: could not convert string to float`` from the import
# storm, naming neither the variable nor the fix (the import-time-crash
# class, P1 audit).  ``resolve_galerkin_chunk_bytes`` refuses BY NAME,
# from the call that actually consumes the budget.


def resolve_galerkin_chunk_bytes() -> int:
    """Per-device ``LORRAX_GALERKIN_CHUNK_GIB`` Q-tile budget in bytes.

    Blank/unset → the default; garbage REFUSES naming the variable
    (``gw_config.env_float`` refuse mode); non-positive values refuse too
    — a zero-byte accumulation budget is never what anyone meant.
    """
    from gw.gw_config import env_float
    gib = env_float("LORRAX_GALERKIN_CHUNK_GIB", 6.0, refuse=True)
    if gib <= 0.0:
        raise ValueError(
            f"LORRAX_GALERKIN_CHUNK_GIB={gib!r} must be > 0 (GiB budget "
            f"for one Galerkin Q r-chunk; unset/blank = 6).")
    return int(gib * 1024 ** 3)


def resolve_extra_rank_pad() -> int:
    """``LORRAX_EXTRA_RANK_PAD`` (default 0) — TEST-ONLY pad-invariance knob.

    The one env resolver for the rank axis's extra null directions (see
    the block comment at the use site); blank/unset → 0, negative or
    non-integer REFUSES naming the variable.  Never set in production.
    """
    raw = os.environ.get("LORRAX_EXTRA_RANK_PAD")
    if raw is None or not raw.strip():
        return 0
    try:
        extra = int(raw)
    except ValueError:
        raise ValueError(
            f"LORRAX_EXTRA_RANK_PAD={raw!r} is not an integer.  Accepted: "
            f"a non-negative int, or unset/blank for 0 (no extra pad).") \
            from None
    if extra < 0:
        raise ValueError(
            f"LORRAX_EXTRA_RANK_PAD must be >= 0; got {extra}")
    return extra


def resolve_fh_ortho_tol(log_fn=None) -> float:
    """``LORRAX_FH_ORTHO_TOL`` (default 1e-6) — the build_fH_R gate cap.

    Routed through ``gw_config.env_float`` in refuse mode: blank/unset →
    the default (the old inline ``float(get(...) or 0.0)`` made a BLANK
    export silently DISABLE the orthonormality gate — the failure it
    guards is a wrong number, not a crash, so silent-off is the worst
    possible reading of a typo); garbage REFUSES naming the variable; a
    NON-DEFAULT value is announced, and ``0`` (gate off) is announced as
    exactly that.
    """
    from gw.gw_config import env_float
    tol = env_float("LORRAX_FH_ORTHO_TOL", 1e-6, refuse=True)
    if tol != 1e-6 and log_fn is not None:
        log_fn(f"  [gate] LORRAX_FH_ORTHO_TOL={tol:.3e} overrides the "
               f"default 1e-6"
               + ("  ** THE ORTHONORMALITY GATE IS DISABLED — only ever "
                  "to reproduce a known-bad run **" if tol == 0.0 else ""))
    return tol


def resolve_rank_policy_mode() -> str:
    """``LORRAX_RANK_POLICY`` (default ``refuse``) — the truncation gate's authority.

    The name and the grammar live once, in ``common/rank_criterion``, which is
    L2 and must be a function of its arguments (``tests/test_layering.py``);
    this driver does the lookup and hands the answer over.  Same shape as
    :func:`resolve_fh_ortho_tol` two functions up, and it exists for the same
    reason: an L1 library dial reaches the environment through EXACTLY ONE
    named resolver, so a reader can find every env-dependent decision this
    module makes by grepping for ``resolve``.

    A mis-spelled mode REFUSES naming the variable — a gate disarmed by a
    typo reads clean in the log, which is the worst possible reading of one.

    WHY THE NAME IS SPELLED OUT HERE.  Everywhere else in the tree the lookup
    is ``os.environ.get(rank_criterion.POLICY_MODE_ENV)`` — one source of
    truth for the spelling.  This module is under the EXACT-SET env ratchet in
    ``tests/test_layering.py``, whose AST scan reads the argument literally
    and records a constant as ``<dynamic>``; pinning that would make the
    ratchet stop naming the variable, which is the degenerate-value failure
    ``TASTE.md`` #18 is about.  So the literal is written for the scanner and
    checked against the constant immediately below, which makes a rename a
    loud refusal at exactly one line instead of a silent second spelling.
    """
    if rank_criterion.POLICY_MODE_ENV != "LORRAX_RANK_POLICY":
        raise RuntimeError(
            f"resolve_rank_policy_mode: common/rank_criterion renamed its "
            f"policy dial to {rank_criterion.POLICY_MODE_ENV!r}, but this "
            f"resolver still reads 'LORRAX_RANK_POLICY' — the literal exists "
            f"only so the L1 env ratchet can see the name.  Update both.")
    return rank_criterion.resolve_policy_mode(
        os.environ.get("LORRAX_RANK_POLICY"))


def resolve_galerkin_rank_multiplier(value) -> float:
    """Validate the opt-in cross-k Galerkin model-order multiplier.

    ``0`` preserves the historical numerical-rank solve.  A positive value
    means ``ceil(value * N_band)`` shared alpha directions, capped by the
    numerical rank.  Values below one are refused: fewer alpha directions
    than bands at one k cannot give the row-orthonormal coefficient block
    required by :func:`build_fH_R`.

    This is deliberately separate from ``rtol``.  The latter protects a
    pseudo-inverse from numerical noise; this value is an explicit physical
    model-order approximation that exploits redundancy between k points.
    """
    try:
        multiplier = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"htransform_rank_multiplier={value!r} is not a finite number; "
            "use 0 for the exact numerical-rank path or a value >= 1.") \
            from None
    if not np.isfinite(multiplier) or multiplier < 0.0:
        raise ValueError(
            f"htransform_rank_multiplier={value!r} must be finite and >= 0.")
    if 0.0 < multiplier < 1.0:
        raise ValueError(
            f"htransform_rank_multiplier={multiplier:g} would retain fewer "
            "directions than bands at one k.  Use 0 for the exact path or a "
            "value >= 1.")
    return multiplier


def validate_centroid_subset_idx(selection, n_parent: int) -> np.ndarray:
    """Validate the ordered parent-to-child centroid row selection."""
    idx = np.asarray(selection)
    if idx.ndim != 1 or idx.dtype.kind not in "iu":
        raise ValueError(
            "htransform centroid subset must be a one-dimensional integer "
            f"row list; got shape={idx.shape}, dtype={idx.dtype}.")
    idx = idx.astype(np.int64, copy=False)
    if idx.size == 0:
        raise ValueError("htransform centroid subset may not be empty.")
    if int(idx.min()) < 0 or int(idx.max()) >= int(n_parent):
        raise ValueError(
            "htransform centroid subset escapes its parent table: "
            f"min/max={int(idx.min())}/{int(idx.max())}, parent rows="
            f"{int(n_parent)}.")
    if np.unique(idx).size != idx.size:
        raise ValueError(
            "htransform centroid subset contains duplicate parent rows; the "
            "downfold basis must be a strict ordered subset.")
    return idx


def _lowdin_orthonormalize_band_rows(ctilde: jax.Array):
    """Per-k polar/Löwdin row orthonormalization for a reduced shared span."""
    gram = jnp.einsum('kna,kma->knm', ctilde, jnp.conj(ctilde),
                      optimize=True)
    gram = 0.5 * (gram + jnp.swapaxes(gram, -1, -2).conj())
    evals, evecs = jnp.linalg.eigh(gram)
    safe_evals = jnp.maximum(evals, jnp.finfo(evals.dtype).tiny)
    invsqrt = jnp.einsum(
        'kni,ki,kmi->knm', evecs, 1.0 / jnp.sqrt(safe_evals),
        jnp.conj(evecs), optimize=True)
    out = jnp.einsum('knm,kma->kna', invsqrt, ctilde, optimize=True)
    eye = jnp.eye(ctilde.shape[1], dtype=ctilde.dtype)[None]
    gram_out = jnp.einsum('kna,kma->knm', out, jnp.conj(out),
                          optimize=True)
    before = jnp.max(jnp.abs(gram - eye))
    after = jnp.max(jnp.abs(gram_out - eye))
    rel_move = jnp.max(
        jnp.linalg.norm(out - ctilde, axis=(-2, -1)) /
        jnp.maximum(jnp.linalg.norm(ctilde, axis=(-2, -1)),
                    jnp.finfo(ctilde.real.dtype).tiny))
    return out, jnp.min(evals), jnp.max(evals), before, after, rel_move


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


def _refuse_unfit_galerkin_mesh(ledger: dict, *, rank: int, nk: int,
                                nspinor: int, n_rtot: int, band_chunk: int,
                                m_states: int, mu_pad: int,
                                psi_win_elems: int, mesh_xy: Mesh,
                                q_spec, log_fn) -> None:
    """REFUSE, before any Q compilation, a mesh whose Galerkin live set
    cannot fit the device pool — naming the smallest square mesh that can.

    ``n_rtot`` is the largest logical r-chunk extent, not the full FFT grid.
    Each candidate uses the same canonical padding carrier as the live path.

    The pool size comes from the allocator's own ``bytes_limit`` (the
    owned reader in ``common.gpu_utils``); when it is unreadable (CPU
    backend, platform allocator) the ledger is printed and the gate is
    announced as OFF — a gate that silently checked nothing is the worse
    failure (TASTE 2026-08-15).  The candidate-mesh scan holds this run's
    carried ``rank`` fixed; the lcm mesh-alignment pad varies by a few
    rows across meshes and is noise at refusal scale.
    """
    from common.gpu_utils import _get_jax_gpu_memory_bytes
    limit, _, _ = _get_jax_gpu_memory_bytes()
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


def streaming_galerkin_solve(wfn, sym, meta, centroid_indices, mesh_xy: Mesh,
                             band_range: tuple[int, int],
                             rtol: float = 1e-8, log_fn=None,
                             band_chunk_size: int = 64,
                             bispinor: bool = False,
                             return_full_proj: bool = False,
                             eigh_backend: str = "auto",
                             rank_multiplier: float = 0.0):
    """Galerkin projection using gw_jax shared loaders.

    Single ('x','y') mesh throughout. ψ at centroids comes from
    ``load_centroids_band_chunked``; ψ at full r is streamed via
    ``iter_psi_rchunk_bandwise`` (band+r chunked). G is built
    sharded ``P('x','y')`` and Cholesky-factored without changing its gauge.

    MEMORY AND SHARDING (2026-08-24).  The accumulator is
    ``Q_chunk[rank, ns, r_chunk_carrier]``: r chunks are outermost, and
    every band chunk is summed into Q_chunk before its Gram contribution.
    Q over ``r_tot`` is never materialized.  The free r axis is zero-padded
    to the mesh product and sharded over ('y','x'); each streamed ψ chunk goes
    directly from
    product-band to product-r sharding through the two volume-preserving
    exchanges owned by ``common.staged_reshard`` before the pinned
    contraction.  The y-replicated carrier and the merged-axis layout that
    forced the SPMD
    partitioner's full-replication fallback at P16 (JID 57271407) is
    gone.  ``galerkin_q_ledger`` prices the largest Q chunk (accumulation
    overlap plus fold workspace), both equal-volume ψ transition shards,
    replicated Gram state, Vh and G per device, then refuses a non-fitting
    chunk before compilation.

    Args:
        wfn, sym, meta: standard gw_jax handles.
        centroid_indices: (n_μ, 3) FFT-grid coordinates.
        mesh_xy: 2D ``('x','y')`` device mesh.
        band_range: ``(b_start, b_end)`` — bands included in the α basis.
        rtol: σ truncation tolerance (relative to s.max()); σ come from the
            Gram-eigh of A Aᴴ (see step 2), the same criterion as the old
            replicated SVD up to sqrt-of-eigenvalue round-off.
        eigh_backend: distrib_la backend for the (nk·nb)² Gram eigh — the
            same plan family (and deck key) as the fH_q eigh downstream;
            ``auto`` = native replicated eigh, ``distributed`` = ScaLAPACK
            pzheevd, one tile over the mesh.
        rank_multiplier: 0 (default) carries the full numerical rank.  A
            value >= 1 targets ``ceil(rank_multiplier * nb)`` shared alpha
            directions, capped by the numerical rank, then Löwdin-
            orthonormalizes each k's band rows.  This is an explicit
            model-order approximation, not another spelling of ``rtol``.
        band_chunk_size: throughput ceiling for bands per FFT chunk inside
            the loader.  The r-outer planner jointly bounds Q and the ψ
            transition layouts after the retained rank is known.
        bispinor: passed through to ``load_centroids_band_chunked``.
        return_full_proj: also return the full-r α-basis projector
            ``W_proj = L⁻¹ diag(1/s) U^H`` (rank, nk·nb), replicated.  For
            any streamed ψ chunk (nk, bc, ns, r_c) restricted to the SAME
            band window, ``B_full[α, s·r_c] = W_proj_bc @ ψ_flat`` evaluates
            the α-basis on the full r-grid, so
            ``ψ_{n,q}(r) = Σ_α c_{n,q}[α] B_full[α](r)`` reconstructs ψ at
            ANY q off the grid — the per-Q ζ-refit consumer
            (``bse.vq_interp.refit_vq``).

    Returns ``(S, ctilde, B_at_mu)`` (legacy contract), plus ``W_proj``
    appended when ``return_full_proj``.  ``B_at_mu`` is μ-sharded on 'y'
    (fitted — replicated only when n_μ divides no mesh axis); ``S``,
    ``ctilde`` and ``W_proj`` are replicated and N_μ-free.
    """
    import time
    if log_fn is None:
        log_fn = lambda *a, **kw: None

    b_start, b_end = band_range
    nb = b_end - b_start
    nk = meta.nk_tot
    nspinor = meta.nspinor
    n_mu = int(centroid_indices.shape[0])
    n_rtot = int(meta.n_rtot)
    q_spec = P(None, None, ('y', 'x'))
    r_mesh_divisor = spec_divisor(mesh_xy, q_spec, axis=2)
    n_r_carrier = round_up(n_rtot, r_mesh_divisor)
    rank_multiplier = resolve_galerkin_rank_multiplier(rank_multiplier)
    if rank_multiplier > 0.0 and return_full_proj:
        raise ValueError(
            "streaming_galerkin_solve: htransform_rank_multiplier cannot be "
            "combined with return_full_proj/refit.  The reduced route applies "
            "a k-dependent Löwdin map to ctilde, while W_proj is one global "
            "full-r projector; pretending they share one alpha gauge would "
            "give the refit wrong wavefunctions.  Use vq-mode=interp/ongrid "
            "or htransform_rank_multiplier=0 for refit.")

    rep = NamedSharding(mesh_xy, P())               # fully replicated
    grid_xy = NamedSharding(mesh_xy, P('x', 'y'))   # (rank, rank) face

    # Bound the full-grid FFT source before the retained rank is known.  The
    # later r planner bounds Q and its sliced transition layouts, but
    # ``iter_psi_rchunk_bandwise`` still performs a full-grid FFT for each
    # band chunk before taking the requested r slab.  Keep that source below
    # the same streaming budget; the mesh-aligned carrier is the minimum
    # useful width on a product-band mesh.
    stream_budget = resolve_galerkin_chunk_bytes()
    bytes_per_band = nk * nspinor * n_r_carrier * 16
    bc_cap = max(1, stream_budget // max(1, bytes_per_band))
    band_chunk_size = max(1, min(int(band_chunk_size), bc_cap, nb))
    # A product-band chunk smaller than the mesh still occupies one band on
    # every device after the loader's canonical band pad.  Carry that actual
    # width through the banner, range construction, FFT cache and ledger.
    _p_band = spec_divisor(mesh_xy, band_sphere_spec(), axis=1)
    _bc = round_up(band_chunk_size, _p_band)
    _n_band_chunks = (nb + _bc - 1) // _bc

    _rank_policy = ("numerical (exact-span default)" if rank_multiplier == 0.0
                    else f"target {rank_multiplier:g}*nb="
                         f"{math.ceil(rank_multiplier * nb)}")
    log_fn(
        f"  Streaming Galerkin: nk={nk}, nb={nb}, nr={n_rtot}"
        + (f" -> {n_r_carrier} zero-padded carrier" if n_r_carrier != n_rtot
           else "") + ", "
        f"n_mu={n_mu}, mesh=({mesh_xy.shape['x']}x{mesh_xy.shape['y']}), "
        f"band_chunk={_bc}"
        + (f" (planner {band_chunk_size}, mesh-aligned)"
           if _bc != band_chunk_size else "")
        + f", rank={_rank_policy}"
    )
    log_fn(
        f"  FFT source chunk <= "
        f"{bytes_per_band * _bc / mesh_xy.size / 1024**3:.2f} GiB/device "
        f"({_bc} mesh-aligned bands; streaming budget "
        f"{stream_budget / 1024**3:.2f} GiB)")
    # Q over r_tot is never materialized.  The retained rank is not known
    # until the centroid Gram-eigh, so the exact r-chunk and live-set ledger
    # are printed there.
    _rank_cap = min(nk * nb, nspinor * n_mu)
    log_fn(f"  Q rank ceiling={_rank_cap}; the zeta-style outer-r planner "
           f"will bound each Q shard after rank selection")
    if _n_band_chunks > 1:
        log_fn(f"  (band axis split into "
               f"{_n_band_chunks} chunks; G is "
               f"accumulated r-outer / band-inner so the split stays exact)")

    # ── 1. Load ψ(G-flat) ONCE for the whole band window ──
    # One capped+padded load (band-sharded over ('x','y')) serves centroid
    # sampling.  It must NOT be dynamically sliced along that sharded band
    # axis for the later r sweep: GSPMD gathers the full window to perform
    # that slice (15.36 GiB/device on this CrI3 deck).  Step 3 therefore uses
    # the iterator's bounded per-chunk loader and this one-shot window is
    # released as soon as its centroid consumer has been dispatched.
    t0 = time.time()
    psi_G_win = load_psi_gflat_padded(
        wfn, band_range, mesh_xy=mesh_xy, bispinor=bispinor)
    if psi_G_win is None:
        raise ValueError(
            f"streaming_galerkin_solve: band window {band_range} lies "
            f"entirely past the file's band extent ({int(wfn.nbands)})")
    log_fn(f"  ψ(G-flat) window load: {time.time()-t0:.2f}s "
           f"(shape {tuple(psi_G_win.shape)}, centroid sample only)")

    # ── 1b. ψ at centroids (band-sharded internally on 'y') ──
    psi_rmu_Y, _ = load_centroids_band_chunked(
        wfn, sym, meta, centroid_indices, bispinor, mesh_xy,
        band_range=band_range, band_chunk_size=band_chunk_size,
        psi_G_flat=psi_G_win,
    )
    del psi_G_win
    # psi_rmu_Y: (nk, nb, ns, n_μ), sharded P(None, None, None, 'y')
    log_fn(f"  load_centroids_band_chunked: {time.time()-t0:.2f}s")
    # Captured for the Q memory ledger below — psi_rmu_Y is deleted before
    # the ledger runs.
    mu_pad = int(psi_rmu_Y.shape[3])

    # ── 2. Gram-eigh of A Aᴴ, A = ψ@centroids reshaped to (nk·nb, ns·n_μ) ──
    # Until 2026-08-01 this step gathered A REPLICATED and ran a dense SVD on
    # every rank — the last N_μ-scaling replicated core in the chain (A itself
    # nk·nb·ns·N_μ·16 B/rank plus the gesdd workspace, and Vh/B_at_mu
    # re-replicated downstream).  A Aᴴ is (nk·nb, nk·nb) — N_μ-FREE — and
    # eigh(A Aᴴ) gives the same left factor: λ = σ², U = eigenvectors, so
    # σ = sqrt(λ) and Vᴴ = diag(1/σ) Uᴴ A is formed μ-SHARDED where consumed
    # (see ``_vh_sharded`` below).  The Gram product contracts (s, μ) locally
    # on each μ-shard + one (nk·nb)² all-reduce; ψ@centroids is never
    # replicated and no O(N_μ) object is.
    #
    # NUMERICS.  (a) Gauge, not bits: eigenvectors of A Aᴴ match the SVD's U
    # only up to per-σ phases (rotations inside degenerate σ groups), so
    # ctilde/B/fH transform covariantly and every physical output (energies,
    # ψ reconstruction) is invariant analytically, equal to the SVD path to
    # ~κ·ε in floats — the same "different valid gauge" class as the
    # distributed ζ tier.  (b) Squaring halves the precision of SMALL σ:
    # sqrt(λ) resolves σ only down to ~sqrt(ε)·σ_max ≈ 1.5e-8·σ_max, i.e.
    # exactly the rtol=1e-8 cut.  That is safe HERE because the retained
    # spectrum is bounded away from the cut (measured job 7883150:
    # σ_min/σ_max = 2.25e-5, 4.5 decades above it — see the truncation block
    # below); a deck whose retained block hugged the cut would fail the
    # ``rank_report`` violations gate, not silently drift.
    #
    # The eigh goes through the distrib_la plan family: ``eigh_backend`` from
    # the deck (auto = native batched eigh, replicated — (nk·nb)² is small;
    # distributed = ScaLAPACK pzheevd, one tile over the mesh, for band
    # windows where even (nk·nb)² replicated is unaffordable).
    t1 = time.time()
    m_states = nk * nb

    @partial(jax.jit, out_shardings=rep)
    def _gram(psi):
        # psi: (nk, nb, ns, μ_pad) at P(None, None, None, 'y').  dot_general
        # over BOTH (s, μ) axes — no ns·μ reshape ever touches the sharded
        # axis.  Zero μ-pad columns add exact zeros to the sum.
        M = jnp.einsum('aism,bjsm->aibj', psi, jnp.conj(psi), optimize=True)
        M = M.reshape(m_states, m_states)
        return 0.5 * (M + M.conj().T)

    M = _gram(psi_rmu_Y)

    from ffi import _services
    _services.ensure_on_path()
    from distrib_la import plan as linalg_plan
    eigh_plan = linalg_plan("eigh", mesh_xy, backend=eigh_backend, n=m_states)
    if eigh_plan.is_native:
        @partial(jax.jit, out_shardings=(rep, rep))
        def _eigh_native(M_):
            w_, V_ = jnp.linalg.eigh(M_)
            return w_, V_
        w, V = _eigh_native(M)
    else:
        log_fn("  Gram eigh: " + eigh_plan.describe())
        w, V = eigh_plan(M)          # λ replicated, V one tile P('x','y')
        V = jax.device_put(V, rep)   # (nk·nb)² — N_μ-free, small
    del M

    # eigh is ascending; the SVD contract everywhere below is descending σ.
    @partial(jax.jit, out_shardings=(rep, rep))
    def _sv_from_eigs(w, V):
        s = jnp.sqrt(jnp.clip(w[::-1], 0.0, None))
        return s, V[:, ::-1]

    s, U = _sv_from_eigs(w, V)
    del w, V
    s_host = np.asarray(s)

    # ── The truncation criterion ──────────────────────────────────────────
    # ``rank_numerical`` is the ONLY place numerical admissibility is decided,
    # and the decision is a cap on how much the pseudo-inverse below (``inv_s``
    # = 1/σ) may amplify round-off: keep σ > σ_max/κ_cap with κ_cap = 1/rtol.
    # It is NOT a search for a gap — these are ISDF/Galerkin overlap spectra
    # and they are smooth by construction, with no knee to find.  See
    # ``common/rank_criterion`` for the criterion, for the three standard
    # alternatives (discrepancy principle / L-curve / GCV) and the measured
    # reason each is refuted here, and for the §R19 table in which retaining
    # 41 % MORE rank moved a QP gap from 3.13 eV to −5049 eV.
    #
    # THE STRUCTURAL CEILING, and why this route needs it more than any other.
    # ``A`` is (nk·nb, nspinor·n_μ) but the spectrum above comes from an eigh
    # of ``A Aᴴ`` at the LARGER of the two dimensions (see the NUMERICS note),
    # so the null space of a tall rank-deficient A arrives as round-off-sized
    # POSITIVE eigenvalues and a relative threshold COUNTS THEM.  Measured on
    # Na bands 1–24: A is (12288, 2032) and rtol=1e-8 selected rank 2034 —
    # two directions more than the matrix algebraically has — after which the
    # capacity line printed the self-contradiction ``rank=2034`` beside
    # ``nspinor·n_μ = 2032``.  ``min(rows, cols)`` is the ceiling; the report
    # below refuses an UNCLAMPED overshoot so a future caller that drops this
    # argument is not silently believed.
    _rank_ceiling = min(nk * nb, nspinor * n_mu)
    rank_numerical = rank_criterion.select_rank(
        s_host, rtol, ceiling=_rank_ceiling)

    # ── …and the criterion is not allowed to stop mid-multiplet ───────────
    # ``common/spectral_closure``, the sibling of the band-window guard.  A
    # cut through a degenerate σ block keeps a symmetry-ARBITRARY slice of an
    # eigenspace, and this function's own NUMERICS note (a) says so from the
    # other side: "eigenvectors of A Aᴴ match the SVD's U only up to per-σ
    # phases (rotations inside degenerate σ groups)".  That rotation freedom
    # is harmless while a degenerate group is retained WHOLE — every physical
    # output is invariant under it — and is exactly what breaks covariance
    # when the group is split.  The repair drops the straddled group whole
    # (owner ruling 2026-08-10), so ``rank_phys`` comes DOWN and the κ_cap
    # this criterion just enforced holds with room to spare; the padding
    # below then aligns whatever survives.  See that module's TWO-RULE FAMILY
    # on why the BAND-window guard rounds the other way and refuses instead.
    rank_numerical, _sc_numerical = spectral_closure.resolve_spectral_cut(
        s_host, rank_numerical,
        where="htransform psi@centroids Gram-eigh sigma numerical rank",
        rcond=rtol, log=log_fn)
    if not _sc_numerical["fired"]:
        log_fn(spectral_closure.describe_clean(
            _sc_numerical,
            where="htransform psi@centroids sigma numerical rank"))

    # ── Optional physical model-order cut (NOT the numerical rtol) ────────
    # Periodic wavefunctions sampled at different k can be strongly redundant.
    # The exact-span default above nevertheless retains up to nk*nb left-
    # singular directions, which makes every later fH(q) solve cubic in nk*nb.
    # An explicit multiplier requests a shared basis sized by bands PER k.
    # This is deliberately opt-in and separately logged/gated: changing rtol
    # to reach the same integer would falsely call a physical approximation
    # "round-off", obscure its convergence axis, and trip the condition-number
    # service's meaning.
    rank_phys = rank_numerical
    _sc_model = None
    if rank_multiplier > 0.0:
        requested = int(math.ceil(rank_multiplier * nb))
        proposed = min(rank_numerical, requested)
        rank_phys, _sc_model = spectral_closure.resolve_spectral_cut(
            s_host, proposed,
            where="htransform reduced cross-k Galerkin model rank",
            rcond=rtol, log=log_fn)
        if not _sc_model["fired"]:
            log_fn(spectral_closure.describe_clean(
                _sc_model,
                where="htransform reduced cross-k Galerkin model rank"))
        if rank_phys < nb:
            raise ValueError(
                "streaming_galerkin_solve: the reduced cross-k Galerkin "
                f"cut retained rank {rank_phys} for {nb} bands per k after "
                "spectral closure.  Per-k row orthonormality is impossible. "
                "Increase htransform_rank_multiplier or use 0 for the "
                "exact numerical-rank path.")
        log_fn(
            f"  [reduced-galerkin] EXPLICIT MODEL ORDER: requested "
            f"ceil({rank_multiplier:g}*{nb})={requested}; numerical rank "
            f"{rank_numerical} of {nk*nb}; retaining {rank_phys} shared "
            f"cross-k directions ({rank_phys/nb:.2f} per band at one k, "
            f"{rank_phys/(nk*nb):.1%} of the exact stacked-state span).  "
            "This is an observable-convergence approximation, not rtol.")

    # ── Mesh alignment: PAD, never round the rank down ────────────────────
    # G, its Cholesky factor, ctilde, B and fH all live on a (rank, rank)
    # face sharded P('x','y'), so the carried extent has to divide BOTH mesh
    # axes; an unaligned extent makes the first ``device_put`` onto that face
    # raise "global size of its dimension 0 should be divisible by 4 …".
    #
    # This used to round the retained rank DOWN to a multiple of
    # lcm(px, py) — at P=64 on an 8×8 mesh, up to 7 physically-selected
    # directions discarded FOR A DEVICE-GRID REASON, which makes the answer a
    # function of the machine it ran on.  The rationale ("the dropped ones
    # sit at the rtol threshold, the same decision rtol already makes") does
    # not survive contact with the measured spectrum.
    #
    # MEASURED, job 7883150 (MoS2 4×4, n_μ=785, nval=26/ncond=16, rtol 1e-8):
    # the SVD is (672, 1570), rank 672 = FULL row rank, and the retained block
    # runs σ_max = 9.857614e-01 down to σ_min = 2.22115e-05, i.e.
    #     σ_min/σ_max = 2.2532e-05,
    # which is FOUR AND A HALF DECADES ABOVE the cut at σ_max·rtol.  κ_eff =
    # 4.44e4 against a cap of 1e8.  The directions a round-down removes are
    # therefore not threshold noise at all — they are ordinary members of a
    # smooth spectrum, comfortably inside the retained block.
    #
    # DO NOT re-derive that ratio from an older log line.  Until 2026-07-31
    # this function printed
    #     f"σ_min/{rtol:.0e}={float(s_host[rank-1]):.3e}"
    # whose LABEL says the value is divided by rtol and whose ARGUMENT is the
    # raw σ.  Reading its "σ_min/1e-08=2.221e-05" as σ_min/σ_max ≈ 2e-13
    # understates the true ratio by eight decades, and that misreading was
    # carried into a campaign brief as evidence that this deck sits "exactly
    # at the interpolation capacity limit".  It does not: the capacity limit
    # on this deck is at nb ≈ 98 (nk·nb = 1568 vs nspinor·n_μ = 1570), where
    # the same sweep measures rank 1562 < 1568, ctilde orthogonality 6.31e-04
    # and an on-grid energy error of 0.242 meV.  The rank report below prints
    # both ends of the retained block with correct labels; use it.
    #
    # The fix is to pad the face instead.  The α-basis is EXTENDED by
    # ``n_pad`` exactly-null directions:
    #   coeffs[:, rank_phys:] = 0,  inv_s[rank_phys:] = 0,  UH[rank_phys:] = 0
    # ⇒ Q's pad rows are exactly zero ⇒ G's pad rows/cols are exactly zero.
    # An identity block is placed on the pad diagonal of G so the Cholesky is
    # non-singular; for a block-diagonal SPD matrix ``potrf`` returns exactly
    # blockdiag(chol(G_phys), I) — every off-diagonal update is an exact 0/1
    # division — so ctilde, B and W_proj all acquire exactly-zero pad
    # columns/rows and every downstream contraction over α is bit-unchanged.
    # fH = Σ_n f(ε_n) c_n c_nᴴ likewise gains an exactly-zero block, i.e.
    # ``n_pad`` extra exact-zero eigenvalues on top of the (rank − nb) exact
    # zeros it already carries; band selection is ascending-index and the
    # f-transform makes f(ε) ≤ 0, so the extra zeros sort ABOVE every selected
    # band and no selection moves.
    #
    # ``LORRAX_EXTRA_RANK_PAD`` adds further null directions on top of the
    # mesh round-up.  It is the pad-extent-invariance knob for this axis (the
    # same role ``LORRAX_EXTRA_MU_PAD`` plays for μ): any result that moves
    # under it depends on the pad extent and is a defect.  Never set it in
    # production.
    align = math.lcm(int(mesh_xy.shape['x']), int(mesh_xy.shape['y']))
    rank = round_up(rank_phys, align)
    _extra = resolve_extra_rank_pad()
    if _extra:
        rank = round_up(rank + _extra, align)
    n_pad = rank - rank_phys

    # Slice to the criterion's rank and null-extend to the carried extent in
    # ONE jit with an explicit replicated out_sharding: done op-by-op,
    # ``jnp.pad`` on a multi-process mesh would leave these operands with an
    # inferred sharding instead of the ``rep`` the eigh produced.
    @partial(jax.jit, static_argnames=('r_phys', 'n_pad'),
             out_shardings=(rep, rep, rep))
    def _trim_and_pad(U, s, r_phys, n_pad):
        U = U[:, :r_phys]
        s = s[:r_phys]
        coeffs = U * s[None, :]              # (nk·nb, r_phys)
        inv_s = (1.0 / s)[:, None]           # (r_phys, 1)
        UH = U.conj().T                      # (r_phys, nk·nb)
        if n_pad:
            coeffs = jnp.pad(coeffs, ((0, 0), (0, n_pad)))
            inv_s = jnp.pad(inv_s, ((0, n_pad), (0, 0)))
            UH = jnp.pad(UH, ((0, n_pad), (0, 0)))
        return coeffs, inv_s, UH

    # ── Truncation diagnostic — printed on EVERY run ──────────────────────
    # retained rank, σ range of the retained block, σ range discarded, and
    # how much of the discard was the physics criterion (all of it, now) vs
    # the mesh alignment (must be zero).  ``violations()`` is the assertion:
    # it refuses a run whose achieved amplification exceeds the cap, whose
    # retained set depends on the device grid, or whose σ_max is zero/NaN
    # (the documented `nband`-is-an-absolute-index trap, in which the SVD of
    # an all-zero ψ window returns rank 0 and everything downstream is
    # meaningless).
    # ``n_dropped_closure`` is what keeps check 2 meaningful after the
    # closure guard started lowering the rank: the block members it dropped
    # are a deficit against the criterion, and without this attribution
    # ``violations()`` would read them as a device-grid round-down and refuse
    # the run.  They are not — the mesh had no part in choosing them — and
    # anything left over after this subtraction is still a real violation.
    _numerical_report = rank_criterion.rank_report(
        s_host, rtol,
        label=f"htransform ψ@centroids numerical rank ({nk*nb}, "
              f"{nspinor*n_mu})",
        quantity="singular values", rank_used=rank_numerical,
        n_rows=nk * nb, n_cols=nspinor * n_mu,
        n_dropped_closure=max(
            0, _sc_numerical["n_keep"] -
            _sc_numerical["n_keep_closed"]),
        # ``rank_used`` here is a SELECTION, so the ceiling applies (the
        # padded report below is a carried EXTENT and declares none).
        rank_ceiling=_rank_ceiling,
        kappa_certified=rank_criterion.KAPPA_CERTIFIED_GRAM)
    _bad = _numerical_report.violations()
    if _bad:
        raise ValueError(
            "streaming_galerkin_solve: the ψ-at-centroids numerical rank is "
            "not self-consistent — " + "  ".join(_bad))
    # THE GATE (docs/dev/rank_truncation_policy.md §2).  ``violations()``
    # above asks whether the code did what it was told; this asks whether
    # what it was told is a regime anyone certified.  Both registered
    # catastrophes satisfied the first and failed the second.
    rank_criterion.certify(
        _numerical_report,
        site="htransform ψ@centroids Gram-eigh σ",
        mode=resolve_rank_policy_mode(),
        cause=(f"the Galerkin basis is over-complete for this window: "
               f"{nk * nb} stacked states against nspinor·n_μ = "
               f"{nspinor * n_mu} interpolation columns, and the retained "
               f"block runs all the way down to the rtol cut."),
        fix=("reduce the band window, raise n_μ, or declare an explicit "
             "reduced model order with htransform_rank_multiplier (>= 1; "
             "ceil(m·nb) shared cross-k directions, N_k-independent by "
             "construction)."),
        log=log_fn)

    if rank_multiplier == 0.0:
        # The numerical report above owns the full left-Gram spectrum, its
        # structural ceiling and any symmetry-closure drop.  This second
        # report describes only the physical block actually selected plus
        # the exact-null carrier pad.  Feeding it the left-Gram null tail
        # would mislabel above-rtol round-off as a device-grid round-down;
        # feeding it the pre-closure ceiling would hide closure-sized pads.
        _trunc = rank_criterion.rank_report(
            s_host[:rank_phys], rtol,
            label=f"htransform ψ@centroids Gram-eigh σ ({nk*nb}, "
                  f"{nspinor*n_mu})",
            quantity="singular values", rank_used=rank,
            n_rows=nk * nb, n_cols=nspinor * n_mu)
        _closure_for_report = _sc_numerical
    else:
        # The model-order tail is not a GRID drop and must never be fed to
        # rank_report as one.  Report the retained operator's conditioning;
        # the explicit reduced-galerkin line above owns the physical tail.
        _trunc = rank_criterion.rank_report(
            s_host[:rank_phys], rtol,
            label=f"htransform retained reduced Galerkin block ({rank_phys} "
                  f"of numerical {rank_numerical})",
            quantity="singular values", rank_used=rank,
            n_rows=rank_phys, n_cols=rank_phys)
        _closure_for_report = _sc_model
    log_fn(f"  Gram-eigh σ of ({nk*nb}, {nspinor*n_mu}): rank={rank_phys}"
           + (f" (+{n_pad} null pad → carried extent {rank}, "
              f"mesh-aligned to {align})" if n_pad else "")
           + f" ({time.time()-t1:.2f}s)")
    log_fn(_trunc.describe())
    if _closure_for_report is not None and _closure_for_report["fired"]:
        # The two guards stay orthogonal and the reader is told which took
        # what.  ``rank_report`` now carries the closure deficit in its own
        # column (see ``n_dropped_closure``), so this line names the block
        # rather than re-deriving the arithmetic.
        _closure_drop = max(
            0, _closure_for_report["n_keep"] -
            _closure_for_report["n_keep_closed"])
        log_fn(f"  [rank]   of the directions not carried, "
               f"{_closure_drop} were "
               f"DROPPED BY DEGENERACY CLOSURE — the members of a block of "
               f"{len(_closure_for_report['members'])} that the cut at "
               f"{_closure_for_report['n_keep']} straddled (relative span "
               f"{_closure_for_report['span_rel']:.3e}).  They are "
               f"real directions with sigma > 0, discarded so the retained "
               f"span is a representation of the point group; the other "
               f"legal cut was {_closure_for_report['n_keep_kept']}, and the ruling of "
               f"2026-08-10 takes the lower.  The {n_pad} null pad above is "
               f"mesh alignment and unrelated.")
    _bad = _trunc.violations()
    if _bad:
        raise ValueError(
            "streaming_galerkin_solve: the ψ-at-centroids truncation is not "
            "self-consistent — " + "  ".join(_bad))
    if rank_numerical < nk * nb:
        log_fn(f"  [warn] ψ-at-centroids is NUMERICALLY RANK-DEFICIENT: "
               f"{nk*nb} states vs numerical rank {rank_numerical} "
               f"(structural ceiling min(nk·nb, nspinor·n_μ) = "
               f"{_rank_ceiling}).  The "
               f"Galerkin basis cannot span the band window — fH energy "
               f"recovery will degrade.  Capacity rule: nk·nb < rank(ψ_μ) "
               f"≤ nspinor·n_μ, i.e. nb < {rank_numerical/nk:.2f} here.  (The pad "
               f"directions are exactly null and add NO capacity — the "
               f"capacity bound is on numerical rank, never on the carried "
               f"extent.  The rank quoted here is CLAMPED to the ceiling: a "
               f"Gram route counts null-space round-off as positive "
               f"eigenvalues, which is how this line once printed "
               f"rank=2034 beside nspinor·n_μ=2032.)")

    # Null-extend to the carried extent.  The zeros here are what make every
    # pad row of Q — hence of G, ctilde, B and W_proj — exactly zero.
    coeffs, inv_s, UH = _trim_and_pad(U, s, rank_phys, n_pad)
    del U, s

    # Vᴴ = diag(1/σ) Uᴴ A, formed μ-SHARDED on 'y' against the sharded ψ tile
    # — the replicated SVD Vh this replaces was (rank, ns·N_μ) on every rank.
    # The contraction runs over (k, n) only, so each device builds its own μ
    # columns with no communication; the σ-pad rows (inv_s = 0) and the
    # loader μ-pad columns come out exactly zero.  Kept at the padded μ
    # extent until ``_b_at_mu`` trims to the true centroid count.
    vh_shard = NamedSharding(mesh_xy, P(None, None, 'y'))

    @partial(jax.jit, out_shardings=vh_shard)
    def _vh_sharded(inv_s, UH, psi):
        UH_kb_ = UH.reshape(rank, nk, nb)
        Vh = jnp.einsum('akn,knsm->asm', UH_kb_, psi, optimize=True)
        return inv_s[:, :, None] * Vh        # (rank, ns, μ_pad), μ on 'y'

    Vh_sh = _vh_sharded(inv_s, UH, psi_rmu_Y)
    del psi_rmu_Y

    # ── 3. G = Q Q^H with Q = inv_s · U^H ψ, streamed r-OUTER / band-INNER ──
    #
    # Q[α, x] = Σ_{k,n} inv_s[α] U^H[α,(k,n)] ψ[(k,n), x]  — the contraction
    # runs over the PAIR index (k, n) and is FREE in x = (spinor, r).  So:
    #
    #   * splitting x (the r axis) and summing G over the pieces is EXACT —
    #     G = Σ_rc Q_rc Q_rc^H, because each Q_rc is a disjoint column block;
    #   * splitting the CONTRACTED (k, n) axis and summing G over the pieces
    #     is NOT — Σ_bc Q_bc Q_bc^H drops every bc ≠ bc' cross term of
    #     (Σ_bc Q_bc)(Σ_bc' Q_bc')^H.
    #
    # The band chunks must therefore be summed INTO Q before the outer
    # product, which is what this loop does.  Getting that backwards is a
    # silent wrong answer, not a crash: G stays Hermitian positive-definite,
    # Cholesky succeeds, and the only symptom is that ctilde comes out
    # non-orthonormal — ``ctilde[0] orthogonality error`` jumps to ~0.5 —
    # so fH = Σ_n f(ε_n) c_n c_n^H no longer has eigenvalues f(ε_n) and the
    # recovered energies drift by ~1.7 eV.  Measured on MoS2 12x12 30 Ry /
    # n_μ=640 / nb=40, on-grid max|Δε_c| vs the stored grid:
    #   one band chunk (nb ≤ band_chunk_size)  0.63 meV
    #   two band chunks, G summed per chunk    1742.48 meV
    # It stayed hidden while ``band_chunk_size`` defaulted to a fixed 64 ≥ nb
    # (one chunk, no cross terms to lose) and surfaced when the chunk was
    # sized to the ψ box instead.
    # Align band_chunk to the mesh band-shard count (p_band = mesh.size): the
    # loader pads nb to round_up(band_chunk, p_band) regardless, so e.g. 7 real
    # bands on a 16-way mesh still COSTS 16 -- and pushes 9 zero-pad bands
    # through every local FFT and the UH@psi GEMM (~2.3x waste) while doubling
    # the iteration count (and the per-chunk XLA retrace storm).  Rounding up to
    # a p_band multiple is free (same padded memory) and both removes the waste
    # and halves the iterations/compiles.  Pure chunking change -> identical G.
    band_chunk_ranges = [
        (b_start + i * _bc, min(b_start + (i + 1) * _bc, b_end))
        for i in range((nb + _bc - 1) // _bc)
    ]
    # r is a FREE column index of Q.  The canonical runtime-padding backend
    # adds exact-zero columns until the carrier divides the full mesh product;
    # these contribute exactly zero to Q and to G = Q Q^H.  Consequently Q
    # always uses both mesh axes instead of sharding_fit's replication fallback
    # on FFT grids such as CrI3's 1,406,250 cells.  The ('y','x') order matches
    # the canonical staged exchange: r is split by y first and x second.  No
    # observable sees Q's internal column order.
    sharding_q = NamedSharding(mesh_xy, q_spec)
    _q_r_entry = sharding_q.spec[2]
    q_shards = r_mesh_divisor
    # The staged-reshard target: the SAME r entry as Q, so the iterator's
    # output, accumulation input, and Q layout cannot drift apart.
    psi_r_layout = NamedSharding(mesh_xy, P(None, None, None, _q_r_entry))

    # ── The zeta-style r plan + Q memory ledger ───────────────────────────
    # The budget is per-device Q storage.  Convert it to a count of LOCAL r
    # columns, then to the global logical chunk owned collectively by the
    # product-r mesh.  Every non-final chunk is mesh-aligned; the canonical
    # iterator zero-pads only the final carrier.
    q_tile_budget = stream_budget
    q_bytes_per_local_r = rank * nspinor * np.dtype(np.complex128).itemsize
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
    r_chunk_ranges = [
        (r0, min(r0 + r_chunk, n_rtot))
        for r0 in range(0, n_rtot, r_chunk)
    ]
    max_r_logical = max(r1 - r0 for r0, r1 in r_chunk_ranges)
    max_r_carrier = round_up(max_r_logical, r_mesh_divisor)
    q_tile_local_bytes = (
        rank * nspinor * (max_r_carrier // r_mesh_divisor)
        * np.dtype(np.complex128).itemsize
    )
    log_fn(
        f"  Galerkin r plan: {len(r_chunk_ranges)} chunk(s), "
        f"logical width <= {max_r_logical}, carrier <= {max_r_carrier}; "
        f"Q <= {q_tile_local_bytes / 1024**3:.2f} GiB/device "
        f"(budget {q_tile_budget / 1024**3:.2f} GiB/device)")

    # Price the largest r chunk before compilation: Q accumulation + fold
    # workspace, both ψ transition layouts, Gram state, Vh and G.
    _p_y_shards = int(mesh_xy.shape['y'])
    _ledger = galerkin_q_ledger(
        rank=rank, nk=nk, nspinor=nspinor, n_rtot=max_r_carrier,
        band_chunk=_bc,
        m_states=m_states, mu_pad=mu_pad,
        psi_win_elems=0,
        p_total=int(mesh_xy.size), q_shards=q_shards, y_shards=_p_y_shards)
    log_fn(f"  Galerkin Q ledger (per device): Q sharded "
           f"{sharding_q.spec} ({q_shards}-way on r), "
           + ", ".join(f"{name} {b / 1024**3:.2f} GiB"
                       for name, b in _ledger.items()))
    _refuse_unfit_galerkin_mesh(
        _ledger, rank=rank, nk=nk, nspinor=nspinor,
        n_rtot=max_r_logical,
        band_chunk=_bc, m_states=m_states, mu_pad=mu_pad,
        psi_win_elems=0, mesh_xy=mesh_xy,
        q_spec=sharding_q.spec, log_fn=log_fn)
    grid_xy_G = grid_xy

    fold_G = _make_fold_G_kernel(
        rank, mesh_xy, sharding_q, grid_xy_G)

    # Allocate G sharded directly.  ``jax.device_put(jnp.zeros(...), sharding)``
    # on a multi-process mesh hands JAX an UNCOMMITTED fully-addressable array,
    # which takes ``_device_put_sharding_impl``'s ``multihost_utils.assert_equal``
    # branch — a real ``process_allgather`` of ``P · rank² · 16`` B (178 MB/rank
    # per process at rank 4716, so 11 GB/rank at P=64) to assert that a block of
    # zeros is the same everywhere.  Same fix, same reason, as ``Q`` below.
    #
    # PAD BLOCK.  ``inv_s`` is zero on the pad rows, so Q's pad rows — and
    # therefore Q Qᴴ's pad rows and columns — are EXACTLY zero and G would be
    # singular there.  Seed the pad diagonal with 1 so G is block-diagonal
    # ``[[G_phys, 0], [0, I]]``.  ``potrf`` on that returns exactly
    # ``blockdiag(chol(G_phys), I)``: for j ≥ rank_phys every A[j,k<j] is an
    # exact 0 and L[j,k] = 0/L[k,k] = 0, so L[j,j] = sqrt(1 − 0) = 1.  The
    # physical block is therefore factored bit-identically to the unpadded
    # run, and ctilde/B/W_proj acquire exactly-zero pad columns/rows.
    # ``n_pad == 0`` keeps the historical zeros allocation untouched.
    def _alloc_G():
        Z = jnp.zeros((rank, rank), dtype=jnp.complex128)
        if n_pad:
            d = jnp.concatenate([jnp.zeros(rank - n_pad, dtype=jnp.float64),
                                 jnp.ones(n_pad, dtype=jnp.float64)])
            Z = Z + jnp.diag(d).astype(jnp.complex128)
        return Z

    G = jax.jit(_alloc_G, out_shardings=grid_xy)()

    t2 = time.time()
    band_eval_count = 0
    UH_kb = UH.reshape(rank, nk, nb)  # for band-chunk slicing

    @partial(jax.jit, static_argnums=(0,), out_shardings=sharding_q)
    def _alloc_Q(r_extent):
        return jnp.zeros(
            (rank, nspinor, r_extent), dtype=jnp.complex128)

    for r_idx, (r0, r1) in enumerate(r_chunk_ranges):
        r_carrier = round_up(r1 - r0, r_mesh_divisor)
        # Allocate only Q for this r chunk, already product-r sharded.  All
        # band chunks are summed into it before the outer product, preserving
        # every cross-band term exactly.
        Q = _alloc_Q(r_carrier)
        for bc_range, psi_bc_r in iter_psi_rchunk_bandwise(
                wfn, sym, meta, mesh_xy, band_range, r0, r1, bispinor,
                band_chunk_size=band_chunk_size,
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
            # The iterator zero-pads every band chunk to the one mesh-aligned
            # width w.  Matching zero columns in UH preserve the contraction.
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

        G = fold_G(Q, G)
        jax.block_until_ready(G)
        del Q
        log_fn(
            f"  G r-chunk {r_idx + 1}/{len(r_chunk_ranges)}: "
            f"[{r0},{r1}) -> carrier {r_carrier}")

    log_fn(
        f"  G accumulation: {len(r_chunk_ranges)} r chunk(s) x "
        f"{len(band_chunk_ranges)} band chunk(s) "
        f"({band_eval_count} streamed evaluations), {time.time()-t2:.2f}s")

    # ── 4. Cholesky on G ──
    # FFI seam: gw_jax's factor_c_q (which has dual 1×1-dense /
    # 2D-blocked paths) is the natural drop-in for distributed scaling, but
    # its 1e-14·trace ridge is calibrated for ISDF's huge-trace C_q matrices
    # and biases htransform's small-trace G by ~1e-5. Use raw Cholesky for
    # numerical parity with the legacy pipeline; swap to factor_c_q
    # (or its FFI variant) when n_μ scales to where rank-deficiency dominates.
    t3 = time.time()
    L = jnp.linalg.cholesky(G)
    log_fn(f"  Cholesky of G: {time.time()-t3:.2f}s")

    # ── 5. ctilde = coeffs · L + ortho diagnostic + B = L⁻¹V^H ──
    # B is the rank-α basis evaluated at the centroids:
    #   ψ_nk(r_μ, s) = Σ_α ctilde[k,n,α] · B[α, s, μ]
    # Math: with A = ψ at centroids = U s V^H, the Cholesky-orthogonalised
    # interpolation vectors at r_μ are exactly L^{-1} V^H (in the (ns, n_μ)
    # column index of V^H). This is what downstream Fourier-upscaling +
    # reconstruction needs to recover ψ at any new k at the centroids.
    @jax.jit
    def _finalize(coeffs, L):
        coeffs = coeffs @ L
        ctilde = coeffs.reshape(nk, nb, rank)
        CtC = ctilde[0] @ ctilde[0].conj().T
        ortho_err = jnp.max(jnp.abs(CtC - jnp.eye(nb, dtype=ctilde.dtype)))
        return ctilde, ortho_err

    ctilde, ortho_err = _finalize(coeffs, L)
    ctilde = jax.device_put(ctilde, rep)
    if rank_multiplier > 0.0:
        # A truncated GLOBAL SVD is a least-squares shared span, but its band
        # rows at one k are not exactly orthonormal.  fH's coarse-grid energy
        # identity requires that local invariant.  The polar factor is the
        # closest row-isometry and changes only this explicit approximate
        # route; the historical numerical-rank coefficients never pass here.
        _lowdin = jax.jit(
            _lowdin_orthonormalize_band_rows,
            out_shardings=(rep, rep, rep, rep, rep, rep))
        (ctilde, _lowdin_lmin, _lowdin_lmax, _ortho_before,
         _ortho_after, _lowdin_move) = _lowdin(ctilde)
        _stats = [float(x) for x in jax.device_get(jnp.stack([
            _lowdin_lmin, _lowdin_lmax, _ortho_before, _ortho_after,
            _lowdin_move]))]
        lmin, lmax, ortho_before, ortho_after, lowdin_move = _stats
        if (not np.isfinite(lmin) or not np.isfinite(lmax) or lmax <= 0.0
                or lmin <= 1.0e-12 * lmax):
            raise ValueError(
                "streaming_galerkin_solve: the requested reduced cross-k "
                "basis does not span all bands at every k: the smallest/"
                f"largest eigenvalue of C_k C_k^H is {lmin:.6e}/"
                f"{lmax:.6e}.  The per-k Löwdin map would amplify by more "
                "than 1e6.  Increase htransform_rank_multiplier (or use 0 "
                "for the exact numerical-rank path).")
        log_fn(
            f"  [reduced-galerkin] per-k Löwdin row isometry: "
            f"eig(C C^H) min/max={lmin:.6e}/{lmax:.6e}, "
            f"max|C C^H-I| {ortho_before:.3e} -> {ortho_after:.3e}, "
            f"max relative coefficient move={lowdin_move:.3e}.  The move is "
            "the declared cross-k compression error; final acceptance comes "
            "from the on-grid wavefunction and exciton-spectrum A/B gates.")
        ortho_err = jnp.asarray(ortho_after)

    # B = L⁻¹ Vᴴ, μ-SHARDED end-to-end (replaces the replicated B_at_mu).
    # The triangular solve runs along the rank axis and is independent per μ
    # column, so it stays local on each μ shard; the spinor axis is moved in
    # front as the (broadcast) batch axis so no reshape ever merges the
    # replicated ns axis with the sharded μ axis.  The trailing slice trims
    # the loader μ-pad to the TRUE centroid count; n_μ divides the mesh axis
    # only by luck, so the output sharding is FITTED (announced replication
    # fallback, never a refusal).
    b_shard = _fit(mesh_xy, P(None, None, 'y'), (rank, nspinor, n_mu),
                   "htransform.B_at_mu(mu-axis)")

    @partial(jax.jit, out_shardings=b_shard)
    def _b_at_mu(L, Vh):
        Vt = jnp.moveaxis(Vh, 1, 0)          # (ns, rank, μ_pad)
        B = jax.vmap(lambda rhs: jsp_linalg.solve_triangular(
            L, rhs, lower=True))(Vt)
        return jnp.moveaxis(B, 0, 1)[..., :n_mu]

    B_at_mu = _b_at_mu(L, Vh_sh)
    del Vh_sh
    # ``jnp.eye`` run EAGERLY is not one module but six —
    # ``iota`` x2 (compiled TWICE: the two operands differ only in
    # dimension), ``add``, ``equal``, ``convert_element_type`` x2 — because
    # every eager jnp op is its own single-primitive XLA module (measured,
    # job 7884866: 6 of this driver's 137).  One jit, one module, and an
    # explicit ``rep`` out_sharding rather than whatever the enclosing mesh
    # context happens to infer — the same spelling ``_alloc_G`` and the Q
    # allocation above already use.  Exact identity either way.
    S = jax.jit(lambda: jnp.eye(rank, dtype=jnp.complex128),
                out_shardings=rep)()
    log_fn(f"  ctilde[0] orthogonality error: {float(ortho_err):.3e}")
    log_fn(f"  Total Galerkin: {time.time()-t0:.2f}s")

    if return_full_proj:
        # W_proj = L⁻¹ diag(1/s) U^H — the α-basis-on-full-r projector (see
        # docstring).  (rank, nk·nb) replicated; small (≲ tens of MB).
        W_proj = jsp_linalg.solve_triangular(L, inv_s * UH, lower=True)
        W_proj = jax.device_put(W_proj, rep)
        return S, ctilde, B_at_mu, W_proj
    return S, ctilde, B_at_mu


def _f_params_from_energies(enk_nb_nk: jax.Array, top_band_index: int,
                            a_band_index: int | None = None) -> tuple[float, float, float]:
    """Compute f-transform parameters from eigenvalues.

    Parameters
    ----------
    top_band_index : int
        Index of the highest band (sets epsilon0 = max eigenvalue of this band).
    a_band_index : int or None
        Index of the band whose bandwidth sets `a = 4 * bandwidth`.
        If None, defaults to top_band_index (original behavior).
        Set this to the highest band you want to keep accurately.
    """
    if a_band_index is None:
        a_band_index = top_band_index
    # ONE jit for the two host floats this returns.  Eagerly these five
    # lines are six single-primitive XLA modules — ``dynamic_slice`` and
    # ``squeeze`` for each band index (one compile each, shapes match),
    # ``_reduce_max`` x2, ``_reduce_min``, ``subtract``, ``add`` — measured
    # at 6 of this driver's 137 (job 7884866).  The band indices are static,
    # so the whole thing folds into one module per (shape, index pair).
    #
    # Bit-identical: the ops and their order are unchanged, ``max``/``min``
    # are order-independent reductions, and ``(max - min) + 1e-14`` is a
    # subtract feeding an add with no multiply for a fusion to contract into
    # an FMA.  ``* 4.0`` stays on the host exactly where it was.
    stats = np.asarray(jax.device_get(
        _f_params_jit(enk_nb_nk, int(top_band_index), int(a_band_index))))
    shift = float(stats[0])
    gap = 4.0 * float(stats[1])
    n = 3.0
    return gap, n, shift


@partial(jax.jit, static_argnames=('top_band_index', 'a_band_index'))
def _f_params_jit(enk_nb_nk: jax.Array, top_band_index: int,
                  a_band_index: int) -> jax.Array:
    """``[max ε_top, max ε_a - min ε_a + 1e-14]`` — see the caller."""
    E_top_k = enk_nb_nk[top_band_index]
    E_a_k = enk_nb_nk[a_band_index]
    return jnp.stack([jnp.max(E_top_k),
                      jnp.max(E_a_k) - jnp.min(E_a_k) + 1e-14])


@partial(jax.jit, static_argnames=('a', 'n', 'shift'))
def _fun_jit(a: float, n: float, shift: float, x: jax.Array) -> jax.Array:
    erf_half = erf(n * 0.5)
    y = x - shift
    cond_left = y <= -a
    cond_mid = jnp.logical_and(y < 0, ~cond_left)
    f_left = y + 0.5 * a
    arg = n * (0.5 + y / a)
    term1 = a * (jnp.exp(-(n * 0.5) ** 2) - jnp.exp(-((n * (a + 2 * y)) / (2 * a)) ** 2)) / (2 * n * jnp.sqrt(jnp.pi) * erf_half)
    term2 = (a + 2 * y) * (erf_half - erf(arg)) / (4 * erf_half)
    f_mid = term1 + term2
    f = jnp.where(cond_left, f_left, 0.0)
    f = jnp.where(cond_mid, f_mid, f)
    f = jnp.where(y >= 0, 0.0, f)
    return jnp.where(f > 0, 0.0, f)


def fun(a: float, n: float, shift: float, x: jax.Array) -> jax.Array:
    """Transform function f(x) — matches Fortran fun(). Works in unshifted space."""
    return _fun_jit(float(a), float(n), float(shift), x)


@partial(jax.jit, static_argnames=('a', 'n', 'shift'))
def _dfun_jit(a: float, n: float, shift: float, x: jax.Array) -> jax.Array:
    erf_half = erf(n * 0.5)
    y = x - shift
    cond_left = y <= -a
    cond_mid = jnp.logical_and(y < 0, ~cond_left)
    df = jnp.zeros_like(y)
    df = jnp.where(cond_left, 1.0, df)
    arg = n * (0.5 + y / a)
    df = jnp.where(cond_mid, 0.5 - erf(arg) / (2 * erf_half), df)
    return jnp.where(y >= 0, 0.0, df)


def dfun(a: float, n: float, shift: float, x: jax.Array) -> jax.Array:
    """Derivative of transform function — matches Fortran dfun()."""
    return _dfun_jit(float(a), float(n), float(shift), x)


def f_transform_eigs(enk_nb_nk: jax.Array,
                     a_band_index: int | None = None) -> tuple[jax.Array, float, float, float]:
    """Apply f-transform to eigenvalues. Returns (f_eps, a, n, shift)."""
    nb, _ = enk_nb_nk.shape
    a, n, shift = _f_params_from_energies(enk_nb_nk, top_band_index=nb - 1,
                                          a_band_index=a_band_index)
    f_eps = fun(a, n, shift, enk_nb_nk)
    return f_eps, a, n, shift


def build_R_grid_np(kgrid: tuple[int, int, int]) -> np.ndarray:
    """Lattice-R grid (nk, 3) matching the IFFT-shift convention used by
    ``make_flat_k_ifftn`` — symmetric around 0, with R_i ∈ {-n/2, ..., n/2-1}."""
    def _shift(n):
        a = np.arange(n, dtype=np.float64)
        return np.where(a >= (n + 1) // 2, a - n, a)
    return np.stack(np.meshgrid(
        _shift(kgrid[0]), _shift(kgrid[1]), _shift(kgrid[2]), indexing='ij'),
        axis=-1).reshape(-1, 3)


def build_fH_R(ctilde: jax.Array, enk_sigma: jax.Array,
               kgrid_co: tuple[int, int, int], mesh_xy: Mesh,
               *, a_band_index: int | None = None,
               log_fn=None):
    """f-transformed Hamiltonian in real-space lattice representation.

    Math (htransform paper):
        fH_k  = -Σ_n |sqrt(-f(ε_n,k))|² ctilde_n,k ctilde_n,k^H
              = Σ_n f(ε_n,k) ctilde_n,k ctilde_n,k^H
        fH_R  = (1/N_k) Σ_k e^{-2πi k·R} fH_k                          # IFFT
    where f(ε) is the smooth bandwidth-bound transform from
    ``f_transform_eigs`` (≤0 for ε<shift, =0 for ε≥shift). For any q,
    fH_q = Σ_R e^{-2πi q·R} fH_R recovers the rank-α-basis Hamiltonian
    whose eigenvalues are f(ε_n,q) and whose eigenvectors are c_n,q,
    enabling both bandstructure interpolation (eigvalsh + newton_inv on
    eigvals) and wfn recovery (eigh, then ψ_n,q(r_μ) = Σ_α c_n,q[α]·B[α,s,μ]).

    Args:
        ctilde:    (nk_co, nb, rank) Galerkin coefficients in the rank-α basis,
                   replicated. Output of ``streaming_galerkin_solve``.
        enk_sigma: (nb, nk_co) DFT band energies in Ry.
        kgrid_co:  (nkx, nky, nkz) coarse uniform k-grid.
        mesh_xy:   ('x','y') device mesh.
        a_band_index: optional band index whose bandwidth sets ``a``;
                   defaults to top of the htransform window (nb-1).
        log_fn:    optional logger.

    Returns:
        fH_k:    (nk_co, rank, rank), sharded P(None, 'x', 'y').
        fH_R:    (nk_co, rank, rank), sharded P(None, 'x', 'y') (lattice-R index).
        params:  (a_f, n_f, shift) — for ``newton_inv`` on the eigvals of fH_q.
        f_eps:   (nb, nk_co) f-transformed eigenvalues, replicated.
    """
    if log_fn is None:
        log_fn = lambda *a, **kw: None

    f_eps, a_f, n_f, shift = f_transform_eigs(enk_sigma, a_band_index=a_band_index)
    log_fn(f"  f-transform: a={a_f:.6f} Ry, n={n_f:.2f}, shift={shift:.6f} Ry"
           + (f" (a from band {a_band_index})" if a_band_index is not None else ""))

    flat_xy = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    spec_3d = P(None, None, None, 'x', 'y')
    # 'backward' = 1/N normalisation in IFFT — matches Σ_R e^{-2πik·R} fH_R = fH_k.
    local_ifftn = make_flat_k_ifftn(mesh_xy, kgrid_co, spec_3d, norm='backward')

    # ``f_eps.T`` at the call site was its own eager ``transpose`` module;
    # transposing inside costs nothing and removes one compile (exact — a
    # transpose is a permutation of the same f64 values).
    @partial(jax.jit, out_shardings=(flat_xy, flat_xy))
    def _build(ctilde_in, f_eps_in):
        f_eps_T = f_eps_in.T
        f_eps_ki = jnp.where(f_eps_T > 0, 0.0, f_eps_T)
        bw = jnp.sqrt(jnp.clip(-f_eps_ki, 0.0, None))
        weighted = ctilde_in * bw[..., None]
        fH_k = -jnp.einsum('kim,kin->kmn', weighted, jnp.conj(weighted), optimize=True)
        # Constrain the CONTRACTION OUTPUT, not just the hermitized copy.
        # ``ctilde`` arrives replicated, so without this the SPMD partitioner
        # has no reason to split the (nk, rank, rank) product and materialises
        # it — and then its transpose, and then the hermitized sum — at FULL
        # size on every device.  Measured on MoS2 12x12 / n_mu=2412 / nb=20
        # (rank 2880): the module wanted 57.84 GiB per device and OOMed an
        # A100-80GB, against a true sharded footprint of 2 x 1.2 GiB on a 4x4
        # mesh.  Pinning the einsum output makes the whole chain local.
        # Memory-only: same values, same shardings out.
        fH_k = jax.lax.with_sharding_constraint(fH_k, flat_xy)
        fH_k = 0.5 * (fH_k + jnp.swapaxes(fH_k, -1, -2).conj())
        fH_k = jax.lax.with_sharding_constraint(fH_k, flat_xy)
        fH_R = local_ifftn(fH_k)
        return fH_k, fH_R

    # ── THE ON-GRID RECOVERY GATE ─────────────────────────────────────────
    # fH_k = Σ_n f(ε_n,k) c_n,k c_n,kᴴ has eigenvalues EXACTLY {f(ε_n,k)} (plus
    # rank−nb exact zeros) if and only if the rows of ctilde[k] are
    # orthonormal.  When they are not, the eigenvalues are no longer f(ε_n)
    # and ``newton_inv`` returns wrong ENERGIES on the coarse grid — silently,
    # with a Hermitian positive-semidefinite fH, a successful Cholesky
    # upstream and a plot that looks fine.
    #
    # WHY HERE.  This is the one function BOTH consumers pass through — the
    # ``bandstructure.htransform`` CLI and ``bandstructure.bse_setup.
    # compute_wfns_fi`` (hence ``bse.exciton_bands``).  Gating here covers the
    # exciton path without reaching into it.
    #
    # WHY ALL k.  ``streaming_galerkin_solve`` prints ``ctilde[0]`` only, and
    # k=0 is Γ, the most symmetric point in the zone.  Cheap to fix: this is
    # nk·nb²·rank, four orders below the nk·rank³ of an eigendecomposition.
    #
    # THE THRESHOLD IS MEASURED, not chosen.  Jobs 7883150 / 7883160, MoS2 4×4
    # / n_μ=785 / rtol 1e-8, walking ncond across the capacity limit
    # (nk·nb → nspinor·n_μ = 1570).  ``ortho`` is this number; ``on-grid'' is
    # the order-matched max|Δε| over the whole band window against the stored
    # eigenvalues:
    #
    #   ncond  nk·nb  rank   ortho      on-grid max   on-grid over BSE cond
    #     70    1536  1536   1.87e-14   3.2e-05 meV        8.8e-11 meV
    #     71    1552  1551   2.97e-04   2.955   meV        0.201   meV   ←
    #     72    1568  1562   6.31e-04   6.939   meV        0.242   meV
    #     74    1600  1570   3.11e-03   23.60   meV        0.860   meV
    #     78    1632  1570   9.35e-03   88.92   meV        3.142   meV
    #
    # Losing ONE direction of 1552 (rank 1551) is what crosses the owner's
    # 0.1 meV requirement.  Over that range on-grid max|Δε| ≈ 9.0e3 · ortho
    # (7.6e3 … 1.1e4), so a cap of 1e-6 holds the on-grid energy error under
    # ~0.01 meV with a decade of margin, while every healthy configuration
    # measured — every window from nb=30 to nb=96 — sits at 1.9e-14, EIGHT
    # decades below the cap.  There is no false-positive room here.
    #
    # AND IT IS NOT A CONDITIONING PROBLEM, so do not reach for ``rtol``.
    # Same job, ncond=72, tightening the amplification cap makes it WORSE
    # because it discards more of a basis that already cannot span:
    #     rtol 1e-8 → rank 1562, ortho 6.31e-04, on-grid  6.9 meV
    #     rtol 1e-6 → rank 1494, ortho 1.04e-02, on-grid 80.3 meV
    #     rtol 1e-4 → rank 1086, ortho 2.61e-02, on-grid 219.3 meV
    # The lever for the EXACT-SPAN route is n_μ (or a narrower window), never
    # the truncation tolerance.  ``htransform_rank_multiplier`` is different:
    # it declares a reduced cross-k MODEL and restores the row-isometry with a
    # per-k polar factor before this gate.  Its accuracy is decided by an
    # observable A/B, not by relabeling its cut as numerical noise.
    #
    # ``LORRAX_FH_ORTHO_TOL`` overrides; 0 disables. Never disable to make a
    # run finish — the failure it catches is a wrong number, not a crash.
    rep_ = NamedSharding(mesh_xy, P())

    @partial(jax.jit, out_shardings=rep_)
    def _ortho_all_k(c):
        nb_ = c.shape[1]
        G = jnp.einsum('kim,kjm->kij', c, jnp.conj(c), optimize=True)
        return jnp.max(jnp.abs(G - jnp.eye(nb_, dtype=G.dtype)[None]))

    _ortho = float(_ortho_all_k(ctilde))
    _tol = resolve_fh_ortho_tol(log_fn)
    log_fn(f"  [gate] ctilde orthonormality over ALL {int(ctilde.shape[0])} k: "
           f"max|C Cᴴ − I| = {_ortho:.3e}  (cap {_tol:.1e}; measured "
           f"conversion: on-grid max|Δε| ≈ 9.0e3 × this, so this run's "
           f"on-grid energy error is ≈ {9.0e3 * _ortho:.2e} meV)")
    if _tol > 0.0 and _ortho > _tol:
        raise ValueError(
            f"build_fH_R: the Galerkin coefficients are NOT orthonormal — "
            f"max|C Cᴴ − I| = {_ortho:.3e} over all k, above the {_tol:.1e} "
            f"cap.  fH's eigenvalues are then not f(ε_n) and the recovered "
            f"ENERGIES are wrong on the coarse grid by roughly "
            f"{9.0e3 * _ortho:.2e} meV, silently.  Cause, in order of "
            f"likelihood: (1) ψ-at-centroids cannot span the band window — "
            f"check the rank line from streaming_galerkin_solve for "
            f"rank < nk·nb.  THREE repairs, and which one applies depends on "
            f"WHY the span failed: MORE CENTROIDS or a NARROWER window if the "
            f"basis is genuinely too small for the bands; or an explicit "
            f"reduced model order, htransform_rank_multiplier >= 1, if the "
            f"span failed because nk·nb grew with the K-POINT COUNT.  The "
            f"exact-span route retains up to nk·nb directions, so on a dense "
            f"metal grid it demands a basis that grows with N_k, which the "
            f"method's own scaling contract does not (Wu et al. S1 estimates "
            f"the QRCP basis at 20–40·N_b, INDEPENDENT of N_k, and reports "
            f"N_mu saturating as N_k grows).  Measured: Na 512 k × 12 bands "
            f"with the validated c1016 basis retains the full column rank "
            f"2032 and refuses here at 2.324e-4, while the SAME basis "
            f"completes at three bands — buying centroids is the wrong "
            f"repair for that shape.  Never rtol: same job, tightening the "
            f"amplification cap made ortho WORSE (1e-8 → 6.31e-4, 1e-6 → "
            f"1.04e-2, 1e-4 → 2.61e-2).  (2) the G accumulation summed Q Qᴴ "
            f"per band chunk instead of summing into Q first.  Override with "
            f"LORRAX_FH_ORTHO_TOL only to reproduce a known-bad run.")

    fH_k, fH_R = _build(ctilde, f_eps)
    return fH_k, fH_R, (a_f, n_f, shift), f_eps


def newton_inv(a: float, n: float, shift: float, y: jax.Array,
               max_iter: int = 50) -> jax.Array:
    """Newton inversion — matches Fortran newton_inv(). Works in unshifted space."""
    dxmax = a / 2.0
    x0 = y + shift  # Fortran initial guess: x = y + s

    def body_fun(_, x_curr):
        res = fun(a, n, shift, x_curr) - y
        df_val = dfun(a, n, shift, x_curr)
        dx = jnp.where(jnp.abs(df_val) > 1e-14, -res / df_val, 0.0)
        dx = jnp.clip(dx, -dxmax, dxmax)
        return x_curr + dx

    return lax.fori_loop(0, max_iter, body_fun, x0)


def load_wfns_and_enk_for_sigma(wfn, sym, nval: int, ncond: int, nband: int):
    nelec = int(wfn.nelec)
    nsigmarange = (int(nelec - nval), int(nelec + ncond))
    enk_sigma, _ = get_enk_bandrange(wfn, sym, nsigmarange, nsigmarange)
    return nsigmarange, jnp.asarray(enk_sigma).transpose(1, 0)


def setup_wfn_and_sym(wfn_file: str, mesh_xy: Mesh | None = None):
    # Pass the device mesh so the loader can pick a sharded read backend
    # (the collective phdf5 FFI read, on GPU or the CUDA-free host lib),
    # band-sharding ψ instead of replicating the whole WFN on every rank.
    # Single-process / no-mesh transparently stays eager.
    wfn = WfnLoader(wfn_file, mesh=mesh_xy)
    sym = symmetry_maps.SymMaps(wfn)
    return wfn, sym


def _clean_label(raw: str | None) -> str | None:
    if not raw:
        return None
    label = raw.strip()
    if not label:
        return None
    # Map common aliases to actual Unicode Greek letters so Matplotlib displays them
    if label.lower() in {'gg', 'gamma', '\\u0393'}:
        label = 'Γ'
    elif label.lower() in {'gl', 'lambda', '\\u039b'}:
        label = 'Λ'
    elif label.lower() in {'gs', 'sigma', '\\u03a3'}:
        label = 'Σ'
    return label


def generate_kpath_from_qe_segments(params: dict, wfn) -> tuple[jnp.ndarray, np.ndarray, list[str | None]] | None:
    seginfo = params.get("kpoints_crystal_b")
    if not seginfo:
        return None
    segments = seginfo.get("segments", [])
    if len(segments) < 2:
        return None
    nodes_crys = [np.asarray(seg["k"], dtype=float) for seg in segments]
    labels = [_clean_label(seg.get("label")) for seg in segments]
    pts_crys = [nodes_crys[0]]
    node_indices = [0]
    for i in range(len(nodes_crys) - 1):
        k0 = nodes_crys[i]
        k1 = nodes_crys[i + 1]
        n = max(1, int(segments[i + 1].get("n", 1)))
        for t in range(1, n + 1):
            alpha = t / float(n)
            pts_crys.append((1.0 - alpha) * k0 + alpha * k1)
        node_indices.append(len(pts_crys) - 1)
    kpoints = np.stack(pts_crys, axis=0)
    return jnp.asarray(kpoints, dtype=jnp.float64), np.asarray(node_indices, dtype=int), labels


def _load_centroids(centroids_path: str, fft_grid: tuple[int, int, int]) -> np.ndarray:
    centroids_frac = np.loadtxt(centroids_path, ndmin=2)
    if centroids_frac.size == 0:
        raise ValueError(f"Centroids file {centroids_path} is empty")
    fft_grid = np.asarray(fft_grid, dtype=int)
    centroid_indices = np.round(centroids_frac * fft_grid).astype(int)
    centroid_indices = np.mod(centroid_indices, fft_grid)
    return centroid_indices


def _shift_indices(n: int) -> jnp.ndarray:
    arr = jnp.arange(n, dtype=jnp.float64)
    return jnp.where(arr >= (n + 1) // 2, arr - n, arr)


def _make_logger(verbose: bool):
    return print if verbose else (lambda *_, **__: None)


def read_eqp_energies(eqp_file: str, sym, band_window: tuple[int, int]) -> jax.Array:
    """Full-BZ QP energies from the IRREDUCIBLE-WEDGE ``eqp{0,1}.dat``.

    Reads the BerkeleyGW-columnar eqp file LORRAX's GW writes — one block
    per ``wfn.kpoints`` entry, the crystal coordinate in the block header,
    energies in eV — and returns ``(nb, nk_full)`` in RYDBERG over
    ``band_window``, which is what this module's DFT path returns.

    THE UNFOLD IS THE SERVICE'S, NOT THIS MODULE'S.  Every IBZ→full-BZ
    map in the tree goes through ``symmetry_maps.star_broadcast``, reached
    here by :func:`symmetry_maps.unfold_file_wedge_to_full_bz` — the FILE
    wedge, ``wfn.kpoints``, which is what ``eqp1.dat`` is indexed by and
    what BerkeleyGW means by the IBZ.  It shares its backend with the
    kin_ion read path; no index table crosses into this module.

    WHAT THIS REPLACED, AND WHY.  It used to require a PRE-UNFOLDED
    full-BZ text file (``nk == sym.nk_tot``, refused otherwise) whose
    ``k-point N:`` blocks it paired to full-BZ k BY POSITION, with no
    coordinate ever read.  Nothing in the tree wrote that file: it came
    from an out-of-tree ``make_eqp_htformat.py`` that joined
    ``eqp_g0w0.dat`` against ``eqp1.dat`` to do the unfold by hand.  That
    is bespoke unfolding one hop upstream, plus a positional pairing that
    a re-ordered file passes silently.  Now the wedge file is read
    directly and the service does the unfold, so the converter has no job
    left and the position never enters.

    The block coordinates are CHECKED against the deck's own wedge
    (``sym.unfolded_kpts[sym.kirr_fullids]``) rather than trusted — a
    file from another deck, or in another order, is refused here instead
    of producing a quasiparticle bandstructure with the energies on the
    wrong k.
    """
    start, end = int(band_window[0]), int(band_window[1])
    nb = int(max(0, end - start))
    if nb == 0:
        raise ValueError("Empty band window requested for EQP override")

    from gw.eqp_bgw import read_bgw_eqp
    from symmetry_maps import unfold_file_wedge_to_full_bz

    kpts_file, _e_dft_ev, e_qp_ev, band_offset = read_bgw_eqp(eqp_file)
    nk_file, nb_file = e_qp_ev.shape

    # ---- the file must be THIS deck's wedge, in the wedge's order -------
    kirr = np.asarray(sym.unfolded_kpts, dtype=np.float64)[
        np.asarray(sym.kirr_fullids, dtype=np.int64)]
    if nk_file != kirr.shape[0]:
        raise ValueError(
            f"{os.path.basename(eqp_file)} holds {nk_file} k-blocks but this "
            f"deck's irreducible wedge has {kirr.shape[0]} (full BZ "
            f"{int(sym.nk_tot)}).  This reader takes the wedge file LORRAX's "
            f"GW writes; the pre-unfolded full-BZ form is no longer read.")
    # Written by ``%13.9f``, so equality is to that many places; the
    # comparison is modulo a lattice vector because either side may carry
    # a k in a different periodic image.
    dk = np.asarray(kpts_file, dtype=np.float64) - kirr
    dk -= np.rint(dk)
    worst = float(np.max(np.abs(dk))) if dk.size else 0.0
    if worst > 1e-6:
        bad = int(np.argmax(np.max(np.abs(dk), axis=1)))
        raise ValueError(
            f"{os.path.basename(eqp_file)} block {bad} is at "
            f"{np.asarray(kpts_file)[bad].tolist()} but this deck's wedge "
            f"point {bad} is {kirr[bad].tolist()} (worst |Δk| = {worst:.2e} "
            f"over {nk_file} blocks).  The eqp file does not belong to this "
            f"wavefunction, or its k-order differs — either way its energies "
            f"would land on the wrong k.")

    # ---- absolute band window -> the file's columns ---------------------
    lo, hi = start - band_offset, end - band_offset
    if lo < 0 or hi > nb_file:
        raise ValueError(
            f"{os.path.basename(eqp_file)} covers absolute bands "
            f"[{band_offset}, {band_offset + nb_file}) but the requested "
            f"window is [{start}, {end}) — the eqp file does not span the "
            f"htransform sigma window.")
    window_ev = np.asarray(e_qp_ev)[:, lo:hi]
    if np.isnan(window_ev).any():
        n_missing = int(np.isnan(window_ev).sum())
        raise ValueError(
            f"{os.path.basename(eqp_file)} is missing {n_missing} of "
            f"{window_ev.size} (k, band) entries inside the requested window "
            f"[{start}, {end}) — a short block cannot be silently padded.")

    # ---- THE unfold: wedge -> full BZ, through the service --------------
    # The FILE wedge: ``eqp1.dat`` is indexed by ``wfn.kpoints``, which is a
    # different (and on two of three committed decks a different-LENGTH)
    # k-set from the star wedge — see the register.  The call site says
    # which without the reader needing to know what ``trs_reference`` is.
    full_ev = np.asarray(unfold_file_wedge_to_full_bz(sym, window_ev))
    if full_ev.shape[0] != int(sym.nk_tot):
        raise ValueError(
            f"star_broadcast returned {full_ev.shape[0]} k-points, expected "
            f"{int(sym.nk_tot)}")

    # eV on disk (BGW convention); Ry is this module's internal unit.
    return jnp.asarray(full_ev.T / RYD_TO_EV, dtype=jnp.float64)


def initialize_wfns(input_path: str, params: dict, log_fn, eqp_file: str | None = None,
                    mesh_xy: Mesh | None = None, return_full_proj: bool = False,
                    n_guard_bands: int = 0, centroid_subset_idx=None):
    """Load ψ, build the Galerkin ``ctilde``/``B_at_mu`` over the deck's window.

    ``n_guard_bands`` widens the htransform window ABOVE the deck's
    ``(nelec − nval, nelec + ncond)`` by that many bands, and only that.  It
    is the two-window contract's one lever: ``f(ε)`` is identically zero for
    ε ≥ ``shift`` := ``max_k ε[nb−1]``, so the top of the htransform's OWN
    window contributes nothing to ``fH`` and ``jnp.linalg.eigh`` fills those
    slots out of fH's null space.  A caller that needs the top of its window
    BACK — ``bse.vq_interp.refit_prepare``, whose ζ' must be fitted on exactly
    the producer's window — buys guard bands above it so every band it asks
    for is interior.  Default ``0`` is the historical behaviour, byte for
    byte: no caller that omits it loads a band it did not load before.

    The widening moves ``ncond`` (and, with it, ``nband``, which is what
    ``Meta`` zero-pads ψ above — a guard band past ``nband`` would arrive as
    exact zeros and the Galerkin solve would never say so).  It does NOT move
    ``nval``: the shoulder is at the TOP of the window, and the bottom edge is
    the deck's own.

    Measured, on the two parents of
    ``tests/known_failures/2026-08-11-fifth-wall-is-the-f-transform-shoulder.md``:
    with zero guards the top four bands of a 20-band window collapse to
    ``min_k ‖O[m,:]‖`` = 0.23…0.27 in the α-space overlap, and the on-grid
    tile null reads 1.267 against a 5.0e-02 bracket.
    """
    from file_io.centroids import load_centroids as _shared_load_centroids

    input_dir = os.path.dirname(os.path.abspath(input_path))

    def _resolve(path: str) -> str:
        return path if os.path.isabs(path) else os.path.join(input_dir, path)

    # Build the mesh up front so the loader is mesh-aware (sharded read).
    if mesh_xy is None:
        mesh_xy = _build_mesh_xy()
    wfn_file = _resolve(params["wfn_file"])
    wfn, sym = setup_wfn_and_sym(wfn_file, mesh_xy=mesh_xy)
    centroid_path = _resolve(params.get("centroids_file", "centroids_frac.txt"))
    _, centroid_indices, n_rmu = _shared_load_centroids(
        centroid_path, tuple(int(x) for x in wfn.fft_grid))
    if centroid_subset_idx is not None:
        _n_parent = int(n_rmu)
        _subset = validate_centroid_subset_idx(
            centroid_subset_idx, _n_parent)
        centroid_indices = np.asarray(centroid_indices)[_subset]
        n_rmu = int(_subset.size)
        log_fn(
            f"  [reduced-galerkin] centroid fit born in the downfold basis: "
            f"ordered parent subset {_n_parent} -> {n_rmu} rows.  No "
            "parent-width B_at_mu or projected wavefunction is formed.")

    nval = int(params["nval"])
    ncond = int(params["ncond"])
    nband = int(params["nband"])
    n_guard_bands = int(n_guard_bands)
    if n_guard_bands < 0:
        raise ValueError(
            f"initialize_wfns: n_guard_bands={n_guard_bands} is negative; the "
            f"guard bands sit ABOVE the deck's window and there is no such "
            f"thing as a negative one.  Pass 0 for the historical window.")
    if n_guard_bands:
        _ncond_deck, _nband_deck = ncond, nband
        ncond = _ncond_deck + n_guard_bands
        nband = max(_nband_deck, int(wfn.nelec) + ncond)
        _b_hi = int(wfn.nelec) + ncond
        if _b_hi > int(wfn.nbands):
            raise SystemExit(
                f"initialize_wfns: {n_guard_bands} guard band(s) above the "
                f"deck's window would need bands up to {_b_hi}, and "
                f"{os.path.basename(wfn_file)} carries {int(wfn.nbands)}.  "
                f"A guard band the file cannot supply arrives as EXACT ZEROS "
                f"(``Meta``'s past-mnband pad), which the Galerkin solve "
                f"absorbs without complaint and the f-transform then reports "
                f"as a perfectly representable band.  Re-run the WFN with "
                f"more bands, or drop the guard count to "
                f"{max(0, int(wfn.nbands) - int(wfn.nelec) - _ncond_deck)}.")
        log_fn(f"  [two-window] htransform window WIDENED by "
               f"{n_guard_bands} guard band(s) above the deck's: ncond "
               f"{_ncond_deck} → {ncond} (bands "
               f"[{int(wfn.nelec) - nval}, {_b_hi}) of {int(wfn.nbands)}); "
               f"nband {_nband_deck} → {nband} so the guards are READ rather "
               f"than zero-padded.  f(ε) ≡ 0 for ε ≥ max_k ε[nb−1], so the "
               f"guards absorb the shoulder and every band the caller asks "
               f"for is interior to fH.")
    bispinor = bool(params.get("bispinor", False))
    meta = Meta.from_system(wfn, sym, nval, ncond, nband, n_rmu, bispinor)
    # ``sys_dim`` IS A DECK KEY AND ``Meta`` HAS NO FIELD FOR IT.  The GW
    # driver stamps it on the Meta it builds (``gw_jax.main``:
    # ``meta.sys_dim = config.sys_dim``) and every dimension-aware consumer
    # reads ``meta.sys_dim`` off that stamp.  This is the OTHER Meta in the
    # tree — the one the bandstructure driver, ``bse.bse_densify``'s
    # ``bse_k_grid`` densification, ``bse.exciton_bands``' ``--w-coarse-grid``
    # leg and ``bse.vq_interp.refit_prepare`` all use — so it is stamped here,
    # from the same key, once, rather than at each of those four call sites.
    #
    # WITHOUT THIS the stamp was simply absent, and every consumer that reads
    # it defensively fell back to bulk 3D: ``gw.coulomb.get_kernel(None)``
    # returns ``Bulk3D``, and the former head-densify guard returned without
    # deciding anything.  The second one was the expensive
    # case — on a slab deck the C1 head channel re-attached ``8π/|q|²``, the
    # untruncated 3D pole, where the true 2D head goes as ``8π·z_c/|q_∥|``,
    # and the error GROWS as the fine grid densifies.  See
    # ``gw.head_densify.build_fine_head_scalars`` and
    # ``bse.bse_densify.build_w_head_channel``.
    #
    # ``read_lorrax_input`` resolves ``sys_dim`` from ``gw_config._DEFAULTS``
    # for every deck, so the key is present on every real caller; a params
    # dict assembled by hand and missing it is refused rather than defaulted,
    # because a default here is exactly the silent 3D assumption above.
    if "sys_dim" not in params:
        raise KeyError(
            "initialize_wfns: params carries no 'sys_dim', so the Meta this "
            "builds cannot be stamped with the deck's dimensionality and "
            "every dimension-aware consumer downstream would silently assume "
            "bulk 3D.  Build params with gw.gw_config.read_lorrax_input (it "
            "fills the key from _DEFAULTS for every deck), or set it "
            "explicitly.  Do NOT default it here.")
    meta.sys_dim = int(params["sys_dim"])
    nsigmarange, enk_sigma = load_wfns_and_enk_for_sigma(wfn, sym, nval, ncond, nband)

    # Optionally override energies with EQP values from a file only if explicitly requested via CLI
    if eqp_file:
        # --eqp-file is an explicit request: a file that cannot be found or
        # parsed must refuse, not silently fall back to the DFT energies —
        # the output would be a DFT bandstructure labeled as quasiparticle.
        eqp_path = _resolve(eqp_file)
        if not os.path.isfile(eqp_path):
            raise SystemExit(f"FATAL: --eqp-file was given but not found: {eqp_path}")
        try:
            enk_sigma = read_eqp_energies(eqp_path, sym, nsigmarange)
            log_fn(f"Using EQP energies from {os.path.basename(eqp_path)} for band window {nsigmarange}")
        except Exception as exc:
            raise SystemExit(
                f"FATAL: --eqp-file {os.path.basename(eqp_path)} could not be "
                f"consumed: {exc}\nExpected LORRAX's GW ``eqp1.dat`` as "
                f"written: BerkeleyGW columns on the IRREDUCIBLE WEDGE — "
                f"{getattr(sym, 'nk_red', '?')} k-blocks for this deck, each "
                f"a '(3f13.9,i8)' crystal-coordinate header followed by that "
                f"many '(2i8,2f15.9)' band rows in eV.  Pass it directly; the "
                f"unfold to this deck's {getattr(sym, 'nk_tot', '?')} full-BZ "
                f"k-points happens here, through the symmetry service.  No "
                f"pre-conversion step is needed or accepted."
            ) from exc

    band_range = (int(nsigmarange[0]), int(nsigmarange[1]))
    with mesh_xy:
        out = streaming_galerkin_solve(
            wfn, sym, meta, centroid_indices, mesh_xy, band_range,
            rtol=1e-8, log_fn=log_fn, bispinor=bispinor,
            return_full_proj=return_full_proj,
            # Deck key, same family as the fH_q eigh; the CLI --eigh-backend
            # override is scoped to compute_wfns_fi and deliberately not
            # threaded here — the deck is the source of truth for the fit.
            #
            # RESOLVED, not raw.  ``eigh_backend`` and ``use_low_mem_eigh``
            # are two spellings of ONE axis and ``gw_config.resolve_eigh_
            # backend`` is the single place they combine; reading the raw
            # key here meant a deck that said ``use_low_mem_eigh = true``
            # got the native replicated Gram eigh anyway — the key was
            # parsed, defaulted, stored and read by nobody on this driver.
            eigh_backend=resolve_eigh_backend(params),
            rank_multiplier=params.get("htransform_rank_multiplier", 0.0),
        )
    S, ctilde, B_at_mu = out[:3]
    log_fn(f"Loaded wavefunctions: nk={sym.nk_tot}, nb={band_range[1]-band_range[0]}, rank={ctilde.shape[2]}")
    if return_full_proj:
        return wfn, sym, meta, mesh_xy, S, ctilde, B_at_mu, enk_sigma, out[3]
    return wfn, sym, meta, mesh_xy, S, ctilde, B_at_mu, enk_sigma


def initialize_kpath(wfn, params):
    info = generate_kpath_from_qe_segments(params, wfn)
    if info is None:
        return None, None, None, None, []
    kpath_frac, node_indices, node_labels = info
    bvec = np.asarray(wfn.bvec, dtype=float)
    blat = float(wfn.blat)
    k_cart = np.asarray(kpath_frac) @ bvec * blat * (2.0 * np.pi)
    seg_len = np.linalg.norm(np.diff(k_cart, axis=0), axis=1)
    x_path = np.concatenate([[0.0], np.cumsum(seg_len)])
    # Compare against the CANONICAL label ``_clean_label`` emits (the real
    # 'Γ' character).  Until 2026-08-01 this tested the eight-character
    # literal ``'\\u0393'``, which no cleaned label ever equals, so
    # gamma_positions was always empty and the ``argmin(norm)`` fallback in
    # ``h_transform`` picked index 0 — right only because Γ headed every
    # path used so far and ties (the zero pad rows) resolve to the first
    # index (scorecard BE).
    gamma_positions = [int(idx) for idx, lbl in zip(node_indices, node_labels)
                       if (lbl or '').strip() == 'Γ']
    return kpath_frac, x_path, node_indices, node_labels, gamma_positions


def resolve_local_vbm_index(nelec: int, band_start: int,
                            n_return_bands: int) -> int:
    """Return the VBM column inside one absolute contiguous band window."""
    idx = int(nelec) - 1 - int(band_start)
    if idx < 0 or idx >= int(n_return_bands):
        raise ValueError(
            "htransform: the returned absolute band window does not contain "
            f"the VBM: nelec={int(nelec)}, band_start={int(band_start)}, "
            f"n_return_bands={int(n_return_bands)} gives local index {idx}. "
            "A bandstructure referenced to the VBM must retain that level.")
    return idx


def h_transform(meta, S, ctilde, enk_sigma, wfn, kpath_data, log_fn, mesh_xy: Mesh,
                a_band_index: int | None = None, diagnostics: bool = True,
                band_start: int = 0, n_return_bands: int | None = None):
    from time import perf_counter as _perf   # instrument:
    nk = int(meta.nkx * meta.nky * meta.nkz)
    states = ctilde.shape[1]
    rank = ctilde.shape[2]
    kgrid = (meta.nkx, meta.nky, meta.nkz)
    nb_keep = states if n_return_bands is None else int(n_return_bands)
    if nb_keep <= 0 or nb_keep > states:
        raise ValueError(
            f"htransform: n_return_bands={nb_keep} must be in [1, {states}] "
            "for the fitted Galerkin window.")
    fermi_band_idx = resolve_local_vbm_index(
        int(wfn.nelec), int(band_start), nb_keep)
    n_guard_bands = int(states) - nb_keep
    log_fn(
        f"  [two-window] standalone htransform returns {nb_keep} band(s) "
        f"from absolute window [{int(band_start)}, "
        f"{int(band_start) + nb_keep}) and fits {states} band(s) "
        f"({n_guard_bands} guard band(s) above).")

    rep = NamedSharding(mesh_xy, P())  # fully replicated, used for diagnostics

    _t0 = _perf()                                          # instrument:
    coeffs = ctilde.reshape(nk, states, rank)
    fH_k, fH_R, (a_f, n_f, shift), f_eps = build_fH_R(
        coeffs, enk_sigma, kgrid, mesh_xy, a_band_index=a_band_index, log_fn=log_fn)
    jax.block_until_ready((fH_k, fH_R, f_eps))             # instrument:
    timing.record("ht.build_fH_R", _perf() - _t0)          # instrument:

    # The top of the fit window is identically invisible to fH at the k where
    # it sets ``shift``.  Returning that band gives an arbitrary null-space
    # direction even when ctilde is exactly orthonormal.  The BSE consumer has
    # enforced this two-window contract since the fifth-wall diagnosis; the
    # standalone band writer must pass through the same gate before it can
    # publish a curve.
    from .bse_setup import _f_shoulder_gate
    _f_shoulder_gate(
        f_eps, 0, nb_keep, shift, log_fn, rank=rank,
        where="htransform")

    # Diagnostics. Split into two small jits:
    #   _diag_stats_fast  — sharding-respecting reductions on the full fH_k
    #     (no gather needed; psum on the (x,y) face).
    #   _diag_eig_at_gamma — eigvalsh on fH_k[0] only. Pull fH_k[0:1] to
    #     replicated FIRST (7.4 MB at our scale), so the eigvalsh runs
    #     single-device locally rather than driving an all-gather of the
    #     full (rank, rank) face inside the eigvalsh module.
    @jax.jit
    def _diag_stats_fast(fH_k):
        return (jnp.min(jnp.real(fH_k)),
                jnp.max(jnp.real(fH_k)),
                jnp.max(jnp.abs(jnp.imag(fH_k))))

    # Takes the WHOLE ``f_eps`` and returns the four printed 5-element
    # windows as well: ``f_eps[:, 0]`` and each ``np.array(x[a:b])`` below
    # was its own eager ``dynamic_slice``/``squeeze`` module (4 of this
    # driver's 137, job 7884866).  Slicing is exact, and these values reach
    # the log only — never ``bandstructure.dat``.
    @partial(jax.jit, static_argnames=('states',))
    def _diag_eig_at_gamma(fH_k0_rep, f_eps_in, states):
        eigs0 = jnp.sort(jnp.linalg.eigvalsh(fH_k0_rep))
        f_exp0 = jnp.sort(f_eps_in[:, 0])
        eig_err = jnp.max(jnp.abs(eigs0[:f_exp0.shape[0]] - f_exp0))
        return (eig_err, f_exp0[:5], eigs0[:5], f_exp0[-5:],
                eigs0[states - 5:states])

    _t0 = _perf()                                          # instrument:
    # ── THE DIAGNOSTICS GATE ──────────────────────────────────────────────
    # This block plus ``_gamma_rt`` below is 1.442 s of the 4.327 s cache-cold
    # ``h_transform`` stage — 33 %, 7 XLA programs — and 0.120 s warm
    # (PROFILE_htransform_exciton §1.2).  Every value it computes reaches a
    # ``log_fn`` line and nothing else; none of them reaches
    # ``bandstructure.dat``.  And ``log_fn`` is ``print if verbose else a
    # no-op`` (``_make_logger``), so WITHOUT ``--verbose`` the driver was
    # spending a third of the stage computing numbers it then discarded.
    #
    # ``fH_k`` exists ONLY for this block: the kpath solve consumes ``fH_R``
    # alone.  At the reference shape it is (64, 768, 768) complex128 = 576 MiB
    # global, held alive across the whole solve for four log lines.  Dropping
    # the reference right after the block frees it before the kpath batches
    # allocate their own (bs, rank, rank) temporaries — the two peaks stop
    # overlapping.  (The buffer is still PRODUCED, because it is an output of
    # ``build_fH_R``'s single jit and ``fH_R`` is its ifft; only its residency
    # is at stake here, which is the 576 MiB the profile prices.)
    if diagnostics:
        re_min, re_max, im_max = _diag_stats_fast(fH_k)
        log_fn("fH_k real range: [{:.3e}, {:.3e}], |imag|max={:.3e}".format(
            float(re_min), float(re_max), float(im_max)))
        fH_k0_rep = jax.device_put(fH_k[0], rep)  # (rank, rank) replicated, ~7 MB
        _eig_err, _f_head, _e_head, _f_tail, _e_tail = jax.device_get(
            _diag_eig_at_gamma(fH_k0_rep, f_eps, int(states)))
        fH_eig_err = float(_eig_err)
        log_fn(f"fH(k=0) eigenvalue error vs f(eps): {fH_eig_err:.6f} Ry = {fH_eig_err * 13.6057:.3f} eV")
        log_fn(f"  f(eps) first 5: {np.asarray(_f_head)}")
        log_fn(f"  fH eig first 5: {np.asarray(_e_head)}")
        log_fn(f"  f(eps) last 5:  {np.asarray(_f_tail)}")
        log_fn(f"  fH eig last 5:  {np.asarray(_e_tail)}")
    else:
        log_fn("  [route] fH diagnostics: OFF (fH_k stats, Γ eigen-check and "
               "the Γ round-trip are not computed; --fh-diagnostics=on "
               "restores them)")
    timing.record("ht.diagnostics", _perf() - _t0)         # instrument:

    _t0 = _perf()                                          # instrument:
    # ── THE METRIC ROUTE — S is the identity, and the code already knows it ──
    #
    # ``_kpath_batch`` below reduces a GENERALIZED eigenproblem fH_q c = λ S c
    # by two triangular solves against chol(S) per q.  But S has exactly ONE
    # producer — ``streaming_galerkin_solve`` returns
    #     S = jax.jit(lambda: jnp.eye(rank, dtype=jnp.complex128), …)()
    # (see its closing lines) — because the α basis is Cholesky-orthogonalised
    # upstream, which is the whole point of ``_finalize``'s ``coeffs @ L``.  The
    # other consumer of the identical math, ``bse_setup._q_batch``, already
    # takes S = I for granted: it calls ``jnp.linalg.eigh(fH_q)`` with no
    # metric at all.  The two solves here are vestigial.
    #
    # WHAT THEY COST.  Each ``solve_triangular`` of a (rank, rank) triangular
    # factor against a (rank, rank) right-hand side is rank³/2 complex MACs, so
    # the pair is ~8·rank³ real flops per q against the ~5.3·rank³ of the
    # eigenvalues-only Hermitian eigensolve they feed.  They are the LARGER
    # half of the arithmetic in the batch — and the Fourier sum this whole
    # workstream is about is 8·N_k·rank², a further factor rank/N_k below both.
    #
    # WHAT THEY CHANGE.  With S = I the ridge makes S_sym = (1+1e-10)·I, so
    # chol(S) = sqrt(1+1e-10)·I and the pair divides every eigenvalue by
    # (1+1e-10) — a systematic -1e-10 RELATIVE shift of λ, applied for no
    # reason.  Skipping them is therefore not bit-identical; it is analytically
    # exact and removes a small bias rather than adding one.  The measured
    # energy delta on the reference deck is in HTRANSFORM_FFT.md §6.
    #
    # The route is decided ONCE, from the data, and announced.  A caller that
    # ever hands this driver a non-identity S keeps the Cholesky path
    # unchanged — the branch is on a measured deviation, not on an assumption.
    @partial(jax.jit, out_shardings=rep)
    def _s_dev_from_eye(S_in):
        return jnp.max(jnp.abs(S_in - jnp.eye(rank, dtype=S_in.dtype)))

    _s_dev = float(_s_dev_from_eye(S))
    metric_route = "identity" if _s_dev == 0.0 else "cholesky"
    log_fn(f"  [route] fH_q metric: {metric_route}  (max|S - I| = {_s_dev:.3e}; "
           f"'identity' skips 2 triangular solves of {rank}³/2 complex MACs per q)")

    S_chol = None
    if metric_route == "cholesky":
        # Built ONLY on the path that uses it — on the identity route this
        # compile (one XLA program, 0.34 s cache-cold) is not paid at all,
        # which is what keeps the route program-count-neutral.
        @jax.jit
        def _build_S_chol(S):
            S_sym = (S + S.conj().T) * 0.5
            S_sym += 1e-10 * jnp.mean(jnp.real(jnp.diag(S_sym))) * jnp.eye(rank, dtype=S_sym.dtype)
            return jnp.linalg.cholesky(S_sym)
        S_chol = _build_S_chol(S)

    R_grid = jnp.asarray(build_R_grid_np(kgrid))
    jax.block_until_ready(                                 # instrument:
        (R_grid,) if S_chol is None else (S_chol, R_grid)) # instrument:
    timing.record("ht.S_chol", _perf() - _t0)              # instrument:

    # ── Kpath-batch processing ───────────────────────────────────────────
    # fH_R stays SHARDED P(None, 'x', 'y'): the (rank, rank) face is split
    # across the mesh, the lattice-R axis is not.  The q-Fourier sum contracts
    # ONLY over R, so every device builds its own (i, j) tile with NO
    # communication; the single collective is the reshard onto the q axis just
    # before the eigvalsh, after which each device owns whole (rank, rank)
    # matrices for its own q-rows and the eigvalsh runs ndev-parallel.
    #
    # It used to be ``fH_R_rep = jax.device_put(fH_R, rep)``.  That is
    # nk · rank² · 16 B on EVERY device (MoS2 12×12: ~51 GB/rank at rank 4716,
    # and it is the term that breaks a 90 GB rank envelope at n_μ ≈ 3.1k) and,
    # because the source is sharded, JAX routes it through ``x._value`` — a
    # host gather of the same size per process.  Identical de-replication to
    # ``bandstructure/bse_setup.py::_fourier`` (see its lines 166-178 / 247-259
    # for the measured OOM this removes); the arithmetic is unchanged.

    # Sharding specs for batched (bs, rank, rank) → (bs, rank) eigvalsh.
    batch_mat_shard = NamedSharding(mesh_xy, P(('x', 'y'), None, None))
    batch_eig_shard = NamedSharding(mesh_xy, P(('x', 'y'), None))
    face_ij_shard = NamedSharding(mesh_xy, P(None, 'x', 'y'))

    @partial(jax.jit, out_shardings=batch_eig_shard)
    def _kpath_batch(batch_k, fH_R, S_chol):
        # batch_k: (bs, 3) replicated; fH_R: (nk, rank, rank) at P(None,'x','y');
        # S_chol: (rank, rank) replicated.  The einsum contracts over the
        # (replicated) R axis only → local on each (i, j) tile.
        phase = jnp.exp(-2j * jnp.pi * (batch_k @ R_grid.T))           # (bs, nk)
        mat = 0.5 * jnp.einsum('bk,kij->bij', phase, fH_R)             # (bs, rank, rank)
        # Pin the contraction OUTPUT to the (i, j) layout: left free, XLA
        # materialises the whole (bs, rank, rank) batch on every device before
        # the reshard below (bs·rank²·16 = 11.4 GiB/device at bs=32/rank 4716).
        mat = jax.lax.with_sharding_constraint(mat, face_ij_shard)
        # Reshard (i,j)→q FIRST, then hermitize: on the q-sharded layout each
        # device owns whole matrices, so the transpose is local.  The other
        # order costs a second all-to-all for ``swapaxes``.
        mat = jax.lax.with_sharding_constraint(mat, batch_mat_shard)
        mat = mat + jnp.swapaxes(mat, 1, 2).conj()

        if S_chol is None:
            # S = I (see the metric-route block above).  ``mat`` was hermitized
            # two lines up and ``eigvalsh`` reads one triangle, so the dropped
            # ``(z + zᴴ)/2`` cannot move an eigenvalue either.
            return jax.vmap(jnp.linalg.eigvalsh)(mat)

        def _solve_one(m):
            y = jsp_linalg.solve_triangular(S_chol, m, lower=True)
            z = jsp_linalg.solve_triangular(S_chol, y, lower=True, trans=2)
            return jnp.linalg.eigvalsh((z + z.conj().T) * 0.5)

        return jax.vmap(_solve_one)(mat)

    # Round-trip diagnostic at Γ — single q, kept (i, j)-sharded (never whole
    # on any device: rank²·16/ndev instead of rank²·16).
    # ``q0`` is a numpy constant, not ``jnp.zeros``: eagerly that was two more
    # single-primitive modules (``convert_element_type``, ``broadcast_in_dim``)
    # for three zeros that only ever appear as a jit constant anyway.
    q0 = np.zeros((1, 3), dtype=np.float64)
    face_one_shard = NamedSharding(mesh_xy, P('x', 'y'))

    # The residual reduction is INSIDE the jit.  Eagerly it was four more
    # modules (``subtract``, ``abs``, ``_reduce_max``, ``_reduce_min``); the
    # arithmetic and the shardings of both operands are unchanged, and the
    # scalar reaches the log only.
    @jax.jit
    def _gamma_rt(fH_R, fH_k):
        phase0 = jnp.exp(-2j * jnp.pi * (q0 @ R_grid.T))
        m = 0.5 * jnp.einsum('bk,kij->bij', phase0, fH_R)
        m = jax.lax.with_sharding_constraint(m, face_ij_shard)
        m = (m + jnp.swapaxes(m, 1, 2).conj())[0]
        m = jax.lax.with_sharding_constraint(m, face_one_shard)
        return jnp.min(jnp.max(jnp.abs(fH_k - m), axis=(1, 2)))

    _t0 = _perf()                                          # instrument:
    if diagnostics:
        rt_err = float(_gamma_rt(fH_R, fH_k))
        log_fn(f"FFT Γ round-trip max error: {rt_err:.3e}")
    timing.record("ht.gamma_roundtrip", _perf() - _t0)     # instrument:
    # Last reader of ``fH_k``; see the diagnostics-gate block above.
    del fH_k

    fermi_energy = float(wfn.efermi)
    kpath_frac, x_path, node_indices, node_labels, gamma_positions = kpath_data
    energies_on_path = None
    energies_sorted = None
    path_range = None
    gamma_exact = None

    if kpath_frac is not None:
        # Wrap + pad in ONE jit.  Eagerly this was five single-primitive
        # modules — ``add``, ``remainder``, ``subtract`` for the wrap and
        # ``broadcast_in_dim`` + ``concatenate`` for the pad (job 7884866).
        # Op-for-op identical, so the k-points fed to ``_kpath_batch`` are
        # bit-unchanged: an add feeding a remainder feeding a subtract, with
        # no multiply anywhere for a fusion to contract into an FMA.
        @partial(jax.jit, static_argnames=('n_pad',))
        def _prep_kpath(kf, n_pad):
            wk = (kf + 0.5) % 1.0 - 0.5
            if n_pad:
                wk = jnp.concatenate(
                    [wk, jnp.zeros((n_pad, 3), dtype=wk.dtype)], axis=0)
            return wk

        # Pad nq to a multiple of batch_size — every batch has the same
        # shape, _kpath_batch compiles ONCE.
        #
        # And pad batch_size itself to a multiple of ndev.  ``batch_mat_shard``
        # / ``batch_eig_shard`` above are RAW ``P(('x','y'), …)`` NamedShardings
        # with no fitter, so a width that does not divide px*py does not
        # degrade here — it raises ``IndivisibleError`` inside the jit.  A
        # fixed 32 therefore restricts this driver to device counts dividing
        # 32; at P=64 it cannot run at all.  Third member of the same class as
        # the zero-padded Galerkin r carrier above and the ``bs`` block in
        # ``bandstructure.bse_setup`` (whose 32-wide q-batch, which DOES have a
        # fitter, replicated instead of refusing and cost 866.5 s vs 105.7 s —
        # jobs 7882533 / 7882569).  Padding is the right lever for all three:
        # the extra q are zero rows already appended here and already dropped
        # by the ``[:nq]`` slice in ``_post_kpath``, so this changes how many
        # q are evaluated and where, never a value.
        _t0 = _perf()                                      # instrument:
        batch_size = _pad_to(mesh_xy, ('x', 'y'), 32)
        nq = int(kpath_frac.shape[0])
        n_pad = round_up(nq, batch_size) - nq
        wrapped_k = _prep_kpath(kpath_frac, int(n_pad))
        nq_padded = wrapped_k.shape[0]
        jax.block_until_ready(wrapped_k)                   # instrument:
        timing.record("ht.kpath_prep", _perf() - _t0)      # instrument:
        _t0 = _perf()                                      # instrument:
        lambda_q_list = []
        for i in range(0, nq_padded, batch_size):
            batch_eigs = _kpath_batch(wrapped_k[i:i+batch_size], fH_R, S_chol)
            lambda_q_list.append(batch_eigs)
            jax.block_until_ready(batch_eigs)
        timing.record("ht.kpath_loop", _perf() - _t0,      # instrument:
                      count=len(lambda_q_list))            # instrument:

        # Bundle concat + slice + vmap(newton_inv) + sort into ONE jit so the
        # post-loop processing emits one compile rather than 4 (concatenate,
        # sort, gather, vmap-newton).
        # ``out_shardings=(rep, rep)`` is a CORRECTNESS requirement at P>1, not
        # a placement preference.  ``batches`` arrive q-sharded
        # (``batch_eig_shard = P(('x','y'), None)``); left to inference the
        # concatenate propagates that sharding onto the outputs, and whether
        # the SPMD partitioner then replicates them or leaves them split is
        # decided by whether ``nq`` divides the device count.  When it does,
        # the ``np.asarray`` on the next line raises
        #     RuntimeError: Fetching value for `jax.Array` that spans
        #     non-addressable (non process local) devices is not possible.
        # — the whole kpath solve completes and the run dies on the final host
        # fetch.  Measured on the survey's own reference path lengths at P=4
        # (2x2 mesh): nq=13 OK, nq=40 DIED, nq=139 OK, i.e. TWO OF THE FOUR
        # reference decks could not run multi-process at all
        # (PROFILE_htransform_exciton §1.5).
        #
        # Replication is the right answer and not merely the safe one: BOTH
        # outputs are consumed on the host by every process immediately below
        # (``np.asarray``, ``np.max``, the writer gate), and they are small —
        # ``energies`` is (nq, rank) f64 and ``energies_sorted`` (nq, nb_keep),
        # 854 KB and 13 KB at nq=139 / rank 768, against the (nk, rank, rank)
        # 576 MiB arrays this driver already holds.  Values are untouched:
        # ``out_shardings`` moves data, it does not compute.
        # Gate: ``tests/test_htransform_post_kpath_sharding.py``.
        @partial(jax.jit, static_argnames=('nq', 'nb_keep'),
                 out_shardings=(rep, rep))
        def _post_kpath(batches, nq, nb_keep):
            lambda_q = jnp.concatenate(batches, axis=0)[:nq]
            energies = jax.vmap(lambda row: newton_inv(a_f, n_f, shift, row.real))(lambda_q)
            energies_sorted = jnp.sort(energies, axis=1)[:, :nb_keep]
            return energies, energies_sorted

        _t0 = _perf()                                      # instrument:
        energies_on_path, energies_sorted_jax = _post_kpath(
            tuple(lambda_q_list), int(nq), int(nb_keep))
        # Report the sharding that was ACTUALLY produced, not the one asked
        # for.  Anything other than ``P()`` here is the non-addressable-fetch
        # crash of PROFILE_htransform_exciton §1.5 waiting for a P>1 run with
        # nq divisible by the device count.  ``_post_kpath`` now pins
        # ``out_shardings=(rep, rep)``, so this line should always read ``P()``;
        # it earns its keep by reporting the spec that came back rather than
        # the one that was requested.  Gated by
        # ``tests/test_htransform_kpath_gates.py::test_post_kpath_outputs_are_replicated``.
        log_fn(f"  [gate] _post_kpath out spec: "
               f"{energies_sorted_jax.sharding.spec} "
               f"(must be P() — a q-sharded fetch dies at P>1)")
        # ``gather_to_host``, not ``np.asarray``: the line above DECLARES the
        # convention, this one SURVIVES its violation.  ``np.asarray`` goes
        # through the same ``_value`` path ``device_get`` does and raises
        # identically at P>1, so a future edit that drops the ``out_shardings``
        # pin would turn the gate line into a crash on the very next statement
        # instead of a report.  Both halves are kept deliberately.
        energies_sorted = gather_to_host(energies_sorted_jax)
        timing.record("ht.post_kpath", _perf() - _t0)      # instrument:
        _t0 = _perf()                                      # instrument:
        # The VBM index is LOCAL to the fitted band window.  Using the
        # absolute electron count here silently selected a conduction level
        # whenever the window started above band zero.
        fermi_energy = float(np.max(energies_sorted[:, fermi_band_idx]))
        if not gamma_positions:
            # Label-less path: nearest-to-Γ point, EXCLUDING the batch pad
            # rows (they are exact zeros and would win the tie on any path
            # that does not start at Γ).  Host numpy — two values for a log
            # line and the plot markers, no reason for two eager XLA modules.
            _k_np = np.asarray(jax.device_get(wrapped_k))[:nq]
            gamma_positions = [int(np.argmin(np.linalg.norm(_k_np, axis=1)))]
        gamma_exact = np.sort(np.asarray(enk_sigma[:, 0]))[:nb_keep]
        # Shift all reported energies so VBM is at 0
        if energies_on_path is not None:
            energies_on_path = energies_on_path - fermi_energy
        if energies_sorted is not None:
            energies_sorted = energies_sorted - fermi_energy
        if gamma_exact is not None:
            gamma_exact = gamma_exact - fermi_energy
        # Recompute path range after shift and report
        path_range = (float(energies_sorted.min()), float(energies_sorted.max()))
        log_fn(f"Path energy range: {path_range[0]:.6f} to {path_range[1]:.6f} Ry (VBM@0)")
        # Γ deltas (in mRy) remain unchanged by uniform shift
        delta = (energies_sorted[gamma_positions[0]] - gamma_exact) * 1000.0
        log_fn("Γ Δε (mRy): " + ", ".join(f"{d:+.2f}" for d in delta[:6]))
        # After shifting, the Fermi level indicator is at 0
        fermi_energy = 0.0
        timing.record("ht.kpath_host_tail", _perf() - _t0)  # instrument:

    return {
        "nk_total": nk,
        "nb_keep": nb_keep,
        "nb_fit": int(states),
        "band_start": int(band_start),
        "n_guard_bands": n_guard_bands,
        "fermi_energy": fermi_energy,
        "energies_on_path": energies_on_path,
        "energies_sorted": energies_sorted,
        "path_range": path_range,
        "gamma_exact": gamma_exact,
        "kpath_data": (kpath_frac, x_path, node_indices, node_labels, gamma_positions),
    }


def plot_bands(result):
    kpath_frac, x_path, node_indices, node_labels, gamma_positions = result["kpath_data"]
    energies_sorted = result["energies_sorted"]
    gamma_exact = result["gamma_exact"]
    fermi_energy = result["fermi_energy"]
    nb_keep = result["nb_keep"]

    if kpath_frac is None or energies_sorted is None:
        raise RuntimeError("Plotting requires a K_POINTS {crystal_b} path in the input file")

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required for plotting") from exc

    fig, ax = plt.subplots()
    for band in range(nb_keep):
        ax.plot(x_path, energies_sorted[:, band], lw=1.0, color='C0', alpha=0.9)

    x_ticks = x_path[np.asarray(node_indices, dtype=int)]
    labels = [(lbl or "") for lbl in node_labels]
    for xpos in x_ticks:
        ax.axvline(xpos, color='k', lw=0.6, alpha=0.3)
    ax.set_xticks(x_ticks, labels)

    for pos_idx, idx in enumerate(gamma_positions or [0]):
        xpos = x_path[idx]
        label_exact = 'Exact Γ' if pos_idx == 0 else None
        label_ht = 'HT Γ' if pos_idx == 0 else None
        if gamma_exact is not None:
            ax.scatter(np.full(nb_keep, xpos), gamma_exact, marker='o', facecolors='none', edgecolors='red', label=label_exact)
        ax.scatter(np.full(nb_keep, xpos), energies_sorted[idx], marker='x', color='black', label=label_ht)

    ax.axhline(fermi_energy, color='red', linestyle='--', linewidth=1.0, alpha=0.7, label='$E_F$')
    ax.set_xlabel('k-path arc length (2π-scaled)')
    ax.set_ylabel('Energy (Ry)')
    ax.set_title('Hamiltonian-transform bands')
    ax.grid(True, which='both', axis='y', linestyle='--', alpha=0.3)
    ax.legend(loc='best', fontsize='small')
    fig.tight_layout()
    plt.show() 


def write_bands_to_file(output_path: str, energies_on_path, kpath_frac, x_path,
                        *, band_start: int = 0, nb_fit: int | None = None):
    if energies_on_path is None or kpath_frac is None or x_path is None:
        return
    # Same family as ``bse_io.write_eigenvectors_stream``: a WRITER must not
    # assume the layout of what it is handed.  ``energies_on_path`` comes
    # straight out of ``_post_kpath`` with the q axis tiled over the mesh.
    energies = gather_to_host(energies_on_path)
    kpoints = gather_to_host(kpath_frac)
    with open(output_path, 'w', encoding='utf8') as fh:
        fh.write('# idx_k idx_b kx ky kz s energy\n')
        if nb_fit is not None:
            fh.write(
                f"# absolute_band_window=[{int(band_start)},"
                f"{int(band_start) + energies.shape[1]}) "
                f"fit_bands={int(nb_fit)} "
                f"guard_bands={int(nb_fit) - energies.shape[1]}\n")
        for ik in range(energies.shape[0]):
            for ib in range(energies.shape[1]):
                kx, ky, kz = kpoints[ik]
                s_coord = x_path[ik]
                fh.write(f"{ik:4d} {ib:4d} {kx: .8f} {ky: .8f} {kz: .8f} {s_coord: .8f} {energies[ik, ib]: .8f}\n")


def main(argv=None):
    import time as _time
    # ── The honesty row, and why these three lines exist ──────────────────
    # ``timing.report(wall=…)`` closes the table with
    #     (untimed) = wall - Σ(top-level rows)
    # and PROFILING_TOOLS calls that the honesty row.  On this driver it read
    # -2.414 s (-60.5 %) warm and -1.283 s (-11.3 %) cache-cold at P=4
    # (PROFILE_htransform_exciton §1.6) — a NEGATIVE honesty row, which cannot
    # be read at all.
    #
    # The cause is an attribution boundary, not a mis-measurement.
    # ``initialize_communicator_stack()`` runs in this module's BODY, above
    # every import, and ``prepare_mesh`` records a ``collective_warmup``
    # section (~2.4 s at P=4) into the global collector while doing it.  That
    # is before ``_t_main``, so the row was inside Σ(rows) but outside the wall
    # it is subtracted from, and the difference came out negative — the table
    # was reporting more seconds than the clock it divides by.
    #
    # The fix is the idiom ``gw_jax`` already uses at its own ``_t_main``
    # (gw_jax.py:180-200) and ``exciton_bands`` uses via ``process_elapsed_s``:
    # RESET the collector here so the pre-main section is not double-counted,
    # read the true process start from /proc, and DECOMPOSE that pre-main span
    # into rows at report time rather than adding rows to it.  The table then
    # closes against the process wall, ``(untimed)`` is non-negative by
    # construction, and the 2.4 s that used to be unreadable appears by name.
    _t_main = _time.perf_counter()
    timing.reset()
    _pre_main = timing.process_elapsed_s()
    parser = argparse.ArgumentParser(allow_abbrev=False, description="Hamiltonian interpolation driver")
    parser.add_argument("-i", "--input", default="cohsex_test.in", help="Input file")
    parser.add_argument("-wfn", "--wfn-file", default=None, help="Override WFN file (e.g. WFN_qp.h5)")
    parser.add_argument("--plot", action="store_true", help="Show interpolated band plot")
    parser.add_argument("--eqp-file", default=None, help="Path to EQP/sigX file to override DFT band energies")
    parser.add_argument("--verbose", action="store_true", help="Print diagnostic details")
    parser.add_argument("--a-band", type=int, default=None,
                        help="Band index (0-based) whose bandwidth sets 'a'. "
                             "E.g. nval+ncond_keep-1. Default: top band.")
    parser.add_argument(
        "--guard-bands", type=int, default=4,
        help="Bands fitted above the requested nval+ncond output window. "
             "The f-transform is identically zero at the top of its own "
             "window, so standalone output requires interior returned bands. "
             "Default: 4 (the measured shoulder depth). Zero is retained only "
             "as a red/reproduction arm and will normally refuse.")
    parser.add_argument("--fh-diagnostics", default="auto",
                        choices=("auto", "on", "off"),
                        help="fH_k range stats, the Γ eigenvalue check against "
                             "f(eps) and the Γ round-trip.  They are 33%% of "
                             "the cache-cold h_transform stage (1.442 s, 7 XLA "
                             "programs) and hold fH_k — 576 MiB at the "
                             "reference shape — alive across the whole solve, "
                             "for four log lines that never reach "
                             "bandstructure.dat.  auto (default) = follow "
                             "--verbose, i.e. compute them only if anything "
                             "will print them; on = always; off = never.")
    parser.add_argument("--eigh-backend", default=None,
                        choices=eigh_backend_choices(),
                        help="Eigensolver for the fH_q eigendecomposition of "
                             "the get_centroids_fi handoff.  auto|off = the "
                             "q-batched native path; distributed|cusolvermp|"
                             "slate|scalapack spread ONE (rank, rank) tile "
                             "over the mesh through the distrib_la door (wide "
                             "band windows).  ``distributed`` is the portable "
                             "spelling and the ONLY one that exists on a host "
                             "mesh, where it means ScaLAPACK pzheevd.  "
                             "OVERRIDES the input-file ``eigh_backend`` key "
                             "(default: use the key, which defaults to auto).")
    parser.add_argument(
        "--distrib-la-batched-route", default=None,
        choices=distrib_la_batched_route_choices(),
        help="OVERRIDES the input-file distrib_la_batched_route key for "
             "every Plan.batched call in this driver. auto preserves the "
             "backend's robust distributed route; batch_reshard moves q "
             "onto the mesh and runs whole-matrix local JAX linalg.")
    args = parser.parse_args(argv)
    log = _make_logger(args.verbose)

    # JAX persistent compile cache — the same call/pattern as gw_jax's
    # _warm_start (and run_nscf / run_sternheimer / kmeans_cli).  This CLI
    # was the one driver that never enabled it, so every htransform run paid
    # a cold XLA compile of the Galerkin/G-accum kernels.  It is now safe and
    # effective at EVERY process count (scorecard AH) — measured at P=8 on the
    # fixture, a warm run compiles 0 of 152 modules per rank instead of 152.
    # Opt out with ISDF_JAX_CACHE_DIR="" — that env value is honoured inside
    # ensure_jax_compile_cache, so nothing here needs to test for it.
    # Failures are logged and swallowed.
    try:
        from common.jax_compile_cache import ensure_jax_compile_cache
        ensure_jax_compile_cache()
    except Exception as exc:
        print(f"  [jax compile cache] skipped: {exc}", flush=True)

    from gw.gw_init import read_cohsex_input
    params = read_cohsex_input(args.input)
    # Input file is the source of truth; the CLI flag is an override — and
    # BOTH go through the one resolver, which is where ``use_low_mem_eigh``
    # folds in.  This driver used to inline the precedence and never call
    # it, so the deck key moved nothing here.
    eigh_backend = resolve_eigh_backend(params, override=args.eigh_backend)
    distrib_la_batched_route = resolve_distrib_la_batched_route(
        params, override=args.distrib_la_batched_route)
    # The INTENT travels too.  ``compute_wfns_fi`` refuses at resolve time
    # under this flag rather than falling back to the whole-matrix native
    # path (bse_setup's no-fallback contract); passing only the resolved
    # library name would leave that refusal disarmed on this driver.
    use_low_mem_eigh = bool(params.get("use_low_mem_eigh", False))
    n_return_bands = int(params["nval"]) + int(params["ncond"])
    
    # Override WFN file if provided via CLI
    if args.wfn_file is not None:
        params["wfn_file"] = args.wfn_file
        log(f"Using WFN file from CLI: {args.wfn_file}")

    from common import sanity

    with timing.section("initialize_wfns"):
        wfn, sym, meta, mesh_xy, S, ctilde, B_at_mu, enk_sigma = initialize_wfns(
            args.input, params, log, args.eqp_file,
            n_guard_bands=args.guard_bands)
    # ── Galerkin-input gate ───────────────────────────────────────────
    # S is the ISDF overlap Gram matrix (Hermitian positive-definite by
    # construction) and ``enk_sigma`` is the band energies the whole
    # interpolation is anchored to — including, when ``--eqp-file`` is
    # given, energies read from a GW run that may itself have produced
    # garbage.  A −136 eV QP energy fed into htransform yields a
    # bandstructure.dat that is numerically finite, plots fine, and is
    # wrong.  Bracket it here, where the file name is still in scope.
    sanity.check_finite("htransform S (ISDF overlap)", S, print_fn=log)
    sanity.check_finite("htransform ctilde", ctilde, print_fn=log)
    sanity.check_finite("htransform E_nk (Ry)", enk_sigma, print_fn=log)
    # Bandwidth, not absolute energy: the zero of a pseudopotential
    # eigenvalue is convention-dependent, but the *spread* of a Σ-window
    # band set is not — 20 Ry (272 eV) is far wider than any real
    # semicore-to-conduction window and so only fires on gross
    # corruption.  A subtler check (comparing --eqp-file energies against
    # the DFT ones they replace) is proposed but not implemented here;
    # see the workstream-O report.
    _enk = np.asarray(jax.device_get(enk_sigma), dtype=np.float64)
    if _enk.size:
        _spread = float(_enk.max() - _enk.min())
        log(f"  E_nk spread: {_spread:.4f} Ry over {_enk.size} states")
        sanity.check_in_range(
            "htransform E_nk bandwidth (Ry)", np.array([_spread]),
            0.0, 20.0, unit="Ry", print_fn=log)

    kpath_data = initialize_kpath(wfn, params)
    _diag_on = (args.verbose if args.fh_diagnostics == "auto"
                else args.fh_diagnostics == "on")
    with mesh_xy, timing.section("h_transform"):
        result = h_transform(meta, S, ctilde, enk_sigma, wfn, kpath_data, log, mesh_xy,
                             a_band_index=args.a_band, diagnostics=_diag_on,
                             band_start=int(wfn.nelec) - int(params["nval"]),
                             n_return_bands=n_return_bands)

    # Optional BSE interpolation handoff: fine-k wfns at coarse centroids.
    # Driven by ``get_centroids_fi`` + ``kgrid_fi`` + ``wfn_fi_{min,max}``;
    # see ``bandstructure.bse_setup.compute_wfns_fi`` for the contract.
    if params.get("get_centroids_fi", False):
        from .bse_setup import compute_wfns_fi
        b_min = int(params["wfn_fi_min"])
        b_max = int(params["wfn_fi_max"]) or int(ctilde.shape[1])
        # SPLASH RADIUS OF THE f-SHOULDER, named by
        # ``2026-08-11-fifth-wall-is-the-f-transform-shoulder.md`` §7 and
        # audited here.  ``wfn_fi_max`` unset DEFAULTS to the full band count
        # — zero guard bands, i.e. exactly the configuration that row
        # convicts.  ``compute_wfns_fi``'s f-shoulder gate is what refuses;
        # this warns first, in the vocabulary of the deck key the user would
        # have to change, so the refusal is not the first news of it.
        _n_guard = int(ctilde.shape[1]) - b_max
        if _n_guard < 4:
            log(f"  [warn] wfn_fi_max={b_max} leaves only {_n_guard} guard "
                f"band(s) below the top of the htransform window "
                f"({int(ctilde.shape[1])} bands)"
                + (" — this is the ZERO-GUARD default (wfn_fi_max unset "
                   "means 'the whole window')" if not
                   int(params["wfn_fi_max"]) else "")
                + f".  f(eps) is identically zero at and above "
                f"max_k eps of the window's own top band, so the top of what "
                f"you are asking BACK may be an arbitrary direction out of "
                f"fH's null space.  Raise nband/ncond so the window extends "
                f"above wfn_fi_max; the f-shoulder gate decides.")
        with mesh_xy, timing.section("wfns_fi"):
            wfns_fi = compute_wfns_fi(
                ctilde=ctilde, B_at_mu=B_at_mu, enk_sigma=enk_sigma,
                kgrid_co=(int(meta.nkx), int(meta.nky), int(meta.nkz)),
                kgrid_fi=params["kgrid_fi"],
                band_window_fi=(b_min, b_max),
                mesh_xy=mesh_xy, a_band_index=args.a_band,
                eigh_backend=eigh_backend,
                use_low_mem_eigh=use_low_mem_eigh, log_fn=log,
                distrib_la_batched_route=distrib_la_batched_route,
            )
        log(f"BSE setup: psi_rmu_Y={wfns_fi.psi_rmu_Y.shape} "
            f"P{wfns_fi.psi_rmu_Y.sharding.spec}, "
            f"psi_rmuT_X={wfns_fi.psi_rmuT_X.shape} "
            f"P{wfns_fi.psi_rmuT_X.sharding.spec}, "
            f"enk_full={wfns_fi.enk_full.shape}")

    if args.plot:
        plot_bands(result)

    # ── Writer gate ───────────────────────────────────────────────────
    # bandstructure.dat is the file downstream tooling and the regression
    # gate diff against ground truth; a NaN row silently changes the file
    # length rather than the exit code.
    sanity.check_finite("bandstructure.dat energies",
                        result['energies_sorted'], print_fn=log)

    # Rank-0 writer gate: at P>1 every process reaches this line with the
    # same (replicated) energies and used to write the SAME shared-FS file
    # concurrently — a race that can interleave partial writes.  Same idiom
    # as the gw_output/gw_init writers.
    output_dir = os.path.dirname(os.path.abspath(args.input))
    if jax.process_index() == 0:
        write_bands_to_file(
            os.path.join(output_dir, 'bandstructure.dat'),
            result['energies_sorted'],  # sorted & truncated to nb_keep, not raw eigenvalues
            kpath_data[0],
            kpath_data[1],
            band_start=result["band_start"],
            nb_fit=result["nb_fit"],
        )

    # ── Outputs barrier, then CLOSE THE LOADER EXPLICITLY ─────────────
    # The block above is rank-0-only, so without this barrier the other
    # ranks arrive at the collective close while rank 0 is still writing
    # ``bandstructure.dat``.  The barrier is what makes every rank enter
    # the close at the same point; the close is the load-bearing half.
    #
    # ``initialize_wfns`` hands back a MESH-AWARE ``WfnLoader``
    # (``setup_wfn_and_sym`` -> ``WfnLoader(wfn_file, mesh=mesh_xy)``, and
    # this driver passes ``mesh_xy=None`` so ``initialize_wfns`` builds the
    # mesh itself), so at P>1 the loader picks the phdf5 backend and owns a
    # ``SlabIO`` whose ``close()`` runs an UNCONDITIONAL COLLECTIVE barrier
    # (``file_io/_slab_io_ffi.py``, ``_barrier("slab_io_ffi_close_attrs")``).
    # Left to ``WfnLoader.__del__``, that collective fires whenever the
    # garbage collector happens to drop the object during interpreter
    # shutdown — a moment no two ranks agree on, and rank 0's object graph
    # differs from the others' because of the writer block above.
    #
    # This is the defect measured and cured in ``bse.exciton_bands`` at
    # ``b3813d8f`` (FIX_exciton_exit_hang.md): three ranks parked in
    # ``__del__`` -> ``SlabIO.close`` -> ``sync_global_devices`` while the
    # fourth had already reached ``ffi.io._atexit_close_all`` ->
    # ``H5Fclose`` -> ``MPI_Barrier``; two disjoint collective domains,
    # neither ever satisfied, the payload complete and every output written,
    # and the step holding its GPUs at 4x100% CPU until the allocation died.
    # That report's §6 named THIS driver as the one remaining sibling with
    # the identical shape.  ``close()`` is idempotent and nulls the handles,
    # so the later ``__del__`` becomes a no-op on every rank; at P=1 both
    # the barrier and the SlabIO collective are already no-ops.
    from common.collectives import barrier
    barrier("htransform.outputs_written")
    try:
        wfn.close()
    except Exception as exc:                                  # noqa: BLE001
        print(f"  [htransform] WfnLoader.close() failed "
              f"({type(exc).__name__}: {exc}); continuing to exit")

    summary = (f"HT complete: {result['nb_keep']} returned / "
               f"{result['nb_fit']} fit bands "
               f"({result['n_guard_bands']} guards), nk={result['nk_total']}, "
               f"fermi={result['fermi_energy']:.6f} Ry")
    if result['path_range'] is not None:
        summary += f", path range [{result['path_range'][0]:.6f}, {result['path_range'][1]:.6f}] Ry"
    print(summary)
    # Close the table against the PROCESS wall, not against ``main()``'s.
    # ``_pre_main`` is everything above this function: the module body's
    # ``initialize_communicator_stack()`` (env, jax.distributed, backend init,
    # mesh + clique warm-up) and the import storm under it.  It is decomposed
    # into rows from the numbers the startup call measured for itself, with
    # the remainder attributed to imports — the same shape as
    # ``gw_jax.py``'s epilogue, and for the same reason: recording the phases
    # AND the whole span would double-count and break the
    # "rows + (untimed) == wall" property that makes the table readable.
    _wall = _time.perf_counter() - _t_main
    if _pre_main is not None:
        _phases = {}
        try:
            _phases = dict(RUNTIME.facts.get("elapsed", {}) or {})
        except Exception:      # noqa: BLE001 — observability never kills a run
            _phases = {}
        for _phase, _secs in sorted(_phases.items()):
            if _phase != "total":
                timing.record(f"htransform.runtime_stack.{_phase}", float(_secs))
        timing.record("htransform.imports",
                      max(_pre_main - float(_phases.get("total", 0.0)), 0.0))
        _wall = _pre_main + _wall
    timing.report(title="--- Timing (seconds) ---", wall=_wall)
    return 0


if __name__ == "__main__":
    # ``runtime.finalize_process``, not a bare ``SystemExit``: the explicit
    # ``wfn.close()`` above removes the collective from ``__del__``, and this
    # removes the SECOND unordered collective — ``ffi.io._atexit_close_all``,
    # whose ``H5Fclose`` on the restart/zeta contexts is collective too and
    # otherwise runs at whatever point each rank's interpreter teardown
    # reaches it.  ``finalize_process`` runs the effects barrier, the
    # distributed shutdown and the atexit hooks in ONE stated order on every
    # rank and then ends with ``os._exit``, so GC-driven ``__del__``s at
    # shutdown never run at all.  Same pattern as ``bse.exciton_bands``
    # (``b3813d8f``) and ``gw.gw_jax``, the sibling that has never hung.
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)

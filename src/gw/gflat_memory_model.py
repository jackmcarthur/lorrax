"""ISDF ζ + V_q memory model — the single production chunk planner.

Design: ``reports/gw_refactor_map_2026-07-01/MEMORY_MODEL_DESIGN.md`` (§1a).
Substrate: ``SHARDING_RULES.md`` (the trichotomy: μ² → all-P, μ×nb → √P,
nb² → replicated).

The fit-loop model is two things summed:

    HWM_fit(cr, bc, P) = persistent(P) + max( A, B, C, D )

**persistent(P)** — resident across the entire r-chunk loop (the *floor*):

    L_q          nq·μ²·16 / P            (÷P, μ²  — the rank floor)
    gflat_acc    nq·μ·ngkmax·16 / P      (÷P)
    ψ_copies     resolved legacy (÷√P) or face (÷P) inventory
    loader_tbl   nk·n_rtot·4 + nk·ngkmax·16          (REPLICATED, no ÷P)
    ψ(r)_cache   nk·nb_cache_pad·ns·n_rtot·16 / P    (optional low-mem hoist)

**stage transients** — each stage adds ONE transient on top; they do not
co-exist, so the HWM takes a ``max``, not a sum:

    A  centroid load    fit FFT box (k_tile, bc, ns, n_rtot)  knob: band_chunk
    B  CCT + Cholesky   C_q + full-(μ,μ) pair density
    C  fit_one_rchunk   legacy open-spin or face scalar-pair rank-3 peaks    ← binder
    D  accumulate       accumulate FFT box (cs, n_rtot)       knob: gflat_chunk_size
Post-fit stages use their own smaller base because ``L_q`` and ``gflat_acc``
have been released:

    E  V_q per tile     V_acc + resident/resharded ζ slabs
    F  tensor write     max(V/W0 tile, G-flat ζ tile)

Two-phase picker (§2):

    Phase 1 — rank floor.  ``persistent(P)`` is un-chunkable; the smallest
              mesh ``P`` with ``persistent(P) ≤ util·budget`` is ``P_min``.
              If the requested ``P < P_min`` → infeasible.
    Phase 2 — choose the four live chunks (band, r, q-solve and G-flat
              accumulation), then report the binding stage across A–F.

Stage-C additions (2026-08-22, planner escapes JID 57269074 / 57281385):
the pair-density temps are ONE contiguous arena, additionally capped at
``_ARENA_PLACEMENT_FRAC`` of the post-persistent headroom (placement,
not sum); the full-BZ ``Z_q`` is charged as LIVE ACROSS the solve seam
(``solve_t + Z_q``, a sum not a max); and the budget-derived r-chunk cap
outranks the μ-wide performance floor.  ``r_chunk_override`` still wins
over all of it — the register-documented run-level workaround.

Bispinor (§1b): charge and the 3 transverse fits share one planner but not
one arena count.  Legacy uses the calibrated open-spin ``ns²`` pair density.
Face source completes one scalar spin pair's band sum before its k-IFFTs;
the identity executable retains four rank-3 buffers, while XLA places the
nonidentity-current loop in one ``ns²``-wide arena.  The caller states which
executable it will run so a smaller transverse ``mu_T`` cannot hide that
different placement.

SCOPE, STATED SO IT IS NOT ASSUMED: this model prices Stages A-F — the
ISDF ζ fit through the V_q tensor write — and NOTHING PAST IT.  The
screening stage that follows (``gw.screening.compute_static_w`` ->
``gw.w_isdf.compute_chi0`` / ``minimax_tau_integrate_chi`` -> ``solve_w``)
has no entry here: no persistent term, no transient, no HWM contribution.
That is not an oversight being deferred — it is a real gap this planner
cannot see, and it OOM'd in production exactly where the gap says it
would (``compute_static_w -> chi0_q.block_until_ready()``, screening.py,
27,262,284,032 B requested against a 63.82 GB pool at MoS2 9x9x1/P16,
KNOWN_LORRAX_ISSUES.md "GN-PPM probe chi0 has no bounded two-role
live-set plan at 81 q", 2026-08-20).  The chi0 τ-scan's scratch arena is
``O(nq·μ²/P)`` and legitimately unchunked over q — the flat-k FFT it runs
needs the whole q/k axis local on every rank, so it cannot be capped by a
knob this model owns the way Stage C's r-chunk is.  The mitigation
shipped instead bounds the ONE thing that WAS schedulable — an earlier
screening role's completed W must not sit resident on-device while a
later role pays that same fixed cost — via
``gw.screening.compute_screening``'s role-serialized spill/restore
(``common.collectives.spill_to_host`` / ``restore_from_host``); see that
function's docstring for the measured numbers.  Pricing the chi0/W stage
itself — so a planner could refuse or shrink it before dispatch the way
Phase 1/2 below do for Stages A-F — remains open and is NOT this file's
job today; do not read the absence of a screening-stage row here as "it
fits", only as "unmeasured".

Most terms above are closed-form shape algebra.  The bounded-k Stage-A load
and the separate full-nk psi-r/to_rchunk transform both query the same
production FFT helper at their real shape/mesh, including XLA's buffer peak
plus cuFFT plan workspace, through
``common.fft_helpers.query_fft_peak_bytes``.  Stage D's two-box factor and
the pair-density ``slots`` count (3 GPU / 4 CPU) are HLO-calibrated facts.
Where the Stage-A query is unavailable the model demotes to an analytic bound
and ANNOUNCES it from the rank it happened on — an unmeasured term here is a
silent OOM later.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Optional

import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.gpu_utils import bfc_fragmentation_target_utilization
from runtime.padding import bounded_partition_tile, round_up

# The planner's ONE printing path: every fallback below announces its demotion
# through this, once per process, tagged with the rank it happened on.  (Safe
# at import time — ``runtime.aot_memory`` pulls in no JAX at module scope.)
from runtime.aot_memory import announce_once as _announce


_C128 = 16  # bytes per complex128


def _c128(*dims, shard: int = 1) -> float:
    """Per-rank complex128 bytes for ``dims`` sharded over ``shard`` ranks."""
    n = 1
    for d in dims:
        n *= int(d)
    return _C128 * n / max(int(shard), 1)


def _coupled_mu123_zq_incremental_bytes(
        *, nk: int, nq: int, ns: int, mu: int, face_nb: int,
        r_chunk: int, p_x: int, p_y: int, ngkmax: int = 0,
        n_rtot: int = 0, cache_psi_r: bool = False,
        stack_three_solves: bool = False,
        host_spill_gflat: bool = False) -> dict[str, float]:
    """Production coupled-schedule delta over one transverse face fit.

    The coupled transport returns all three completed ``Z_q`` channels, so
    two additional ``Z_q[q,mu_X,r_Y]`` arrays are live.  Its full-spin X
    owner cache is sharded only over ``mu_X`` and replicated over Y.  The
    production coordinator also retains two extra G-flat outputs and two
    extra transverse CCT factors.  ``host_spill_gflat`` parks those outputs
    in process-local RAM, so their bytes are reported separately and do not
    inflate the device HWM.  When ψ(r) is hoisted, its two additional
    all-P-sharded copies are included too.  The incumbent bounded Y cache and
    one channel's two P carries are unchanged.  ``stack_three_solves`` adds
    the larger of the two nonconcurrent face↔batch solve liveness phases.
    """
    completed_zq = 2.0 * _c128(
        nq, mu, r_chunk, shard=int(p_x) * int(p_y))
    shared_x_face = _c128(nk, ns, mu, face_nb, shard=p_x)
    extra_gflat_host = 2.0 * _c128(
        nq, mu, ngkmax, shard=int(p_x) * int(p_y))
    extra_gflat = 0.0 if host_spill_gflat else extra_gflat_host
    extra_factors = 2.0 * _c128(
        nq, mu, mu, shard=int(p_x) * int(p_y))
    face_nb_transport = round_up(face_nb, int(p_x) * int(p_y))
    extra_psi_r = (2.0 * _c128(
        nk, face_nb_transport, ns, n_rtot,
        shard=int(p_x) * int(p_y)) if cache_psi_r else 0.0)
    stacked_solve = 0.0
    if stack_three_solves:
        p_xy = int(p_x) * int(p_y)
        b_one = round_up(nq, p_xy)
        b_three = round_up(3 * nq, p_xy)
        delta_b = b_three - b_one
        # Two nonconcurrent liveness phases bind.  First, face→batch local
        # RHS input/output plus the larger A arena; second, one enlarged local
        # RHS arena plus all three face outputs retained by the coordinator.
        local_phase = (
            2.0 * _c128(delta_b, mu, r_chunk, shard=p_xy)
            + _c128(delta_b, mu, mu, shard=p_xy))
        face_phase = (
            _c128(delta_b, mu, r_chunk, shard=p_xy)
            + 2.0 * _c128(nq, mu, r_chunk, shard=p_xy))
        stacked_solve = max(local_phase, face_phase)
    return {
        "two_additional_completed_zq": completed_zq,
        "shared_full_spin_x_face": shared_x_face,
        "two_additional_gflat_outputs": extra_gflat,
        "two_additional_host_gflat_outputs": (
            extra_gflat_host if host_spill_gflat else 0.0),
        "three_host_gflat_outputs": (
            1.5 * extra_gflat_host if host_spill_gflat else 0.0),
        "two_additional_transverse_factors": extra_factors,
        "two_additional_psi_r_caches": extra_psi_r,
        "stacked_solve_transient": stacked_solve,
        "total": (completed_zq + shared_x_face + extra_gflat
                  + extra_factors + extra_psi_r + stacked_solve),
    }


def _batch_reshard_operand_floor_bytes(
        *, batch: int, mu: int, nrhs: int, processes: int) -> float:
    """Measured three-arena floor of one local batch-reshard solve."""
    batch_local = (int(batch) + int(processes) - 1) // int(processes)
    return 3.0 * _c128(batch_local, mu, int(mu) + int(nrhs))


def _coupled_route_projected_hwm_bytes(
        *, base_hwm: float, persistent: float, coupled_delta: float,
        solve_operand_floor: float = 0.0) -> float:
    """Maximum of the incumbent stage HWM and route-specific solve stage."""
    delta = float(coupled_delta)
    return max(
        float(base_hwm) + delta,
        float(persistent) + float(solve_operand_floor) + delta,
    )


def _pair_density_slots() -> int:
    """Concurrent rank-5 ``(nk, ns², μ, cr)`` pair-density slots XLA keeps
    live at the Stage-C peak — 3 on GPU, 4 on CPU.  This is a
    BufferAssignment fact (HLO-calibrated), not shape algebra."""
    try:
        import jax
        return 4 if jax.default_backend() == "cpu" else 3
    except Exception as exc:
        _announce("pair-density-slots-default",
                  f"jax.default_backend() unreadable ({type(exc).__name__}: "
                  f"{exc}); assuming 3 pair-density slots (GPU).  CPU really "
                  f"has 4, so Stage C would be one (nk, ns², μ, cr) arena low")
        return 3


def _face_pair_density_slots(*, ns: int, current_vertex: bool) -> int:
    """Rank-3 equivalents in the face pair executable's one HLO arena.

    The identity/charge executable retains four at its calibrated peak:
    old ``Z_R``, the two k-IFFT outputs, and new ``Z_R``.  The nonidentity
    current executable is different: XLA places its scalar-spin-pair loop as
    one open-spin ``(nk, ns, ns, mu/Px, r/Py)`` temporary arena, i.e.
    ``ns**2`` rank-3 equivalents.  Run153 measured the exact ns=4 shape
    ``(36,4,4,200,10368)c128`` (19,110,297,600 B before BFC rounding).

    This function is the sole slot-count owner for both executables.  Keep
    the distinction structural; applying the current count to charge would
    discard the independently measured four-slot charge route.
    """
    ns = int(ns)
    if ns <= 0:
        raise ValueError(f"ns must be positive, got {ns}")
    return ns * ns if bool(current_vertex) else 4


# Analytic FALLBACK factor for the FFT box, used only when it cannot be
# compiled and queried.  A cuFFT out-of-place plan holds ~2 box-sized scratch
# slots on top of the in/out boxes; measured 4.00x at Si 24³/nk=64 and
# 48³/nk=1, 4.25-4.55x on small shards (memory-model.md "FFT peak memory").
# It counts BOX COPIES and does not model the cuFFT plan workspace at all,
# which is why every path reaching it announces itself.
_FFT_CUFFT_FACTOR = 4.0

# ``gflat_chunk_size`` cap: past cs ~ 1000 cuFFT switches plan algorithm and
# scratch grows non-linearly (cs=1414 OOM'd at production CrI3 80Ry).
GFLAT_CHUNK_SIZE_CAP = 100
_GFLAT_CHUNK_FLOOR = 4  # cuFFT plan amortisation

#: Fraction of the post-persistent target headroom the Stage-C pair-density
#: temp arena may claim as ONE contiguous allocation.  The z_q executable's
#: temps are placed as a single BFC request of exactly
#: ``slots · nk · ns² · (μ/p_x) · (cr/p_y) · 16`` bytes — matched to the
#: byte on both measured failures: 32,470,795,776 B at MoS2 8x8 full-BZ
#: (JID 57269074 step lx-Xg4-005932, 30.24 GiB refused against a 63.82-GB
#: pool whose HWM the planner had certified at 54.33 GB) and
#: 27,262,282,752 B at the naturally-unreduced 9x9 81-q run (JID 57281385
#: step .28).  In both, the sum fit but the SINGLE arena could not be
#: placed against the fragmentation the freed Stage-B transients leave
#: behind (the allocator map shows the free space split around ~44% in
#: use).  0.80 of the post-persistent headroom is the measured failing
#: ratio; 0.5 is the margin this model claims, stated as a placement
#: heuristic rather than shape algebra.
_ARENA_PLACEMENT_FRAC = 0.5


def _factor_mesh(pp: int) -> tuple[int, int]:
    """Most-square factorization ``p_x·p_y = pp`` with ``p_x = floor(√pp)``
    decremented until it divides — matches ``gw_jax`` mesh construction."""
    gx = int(math.isqrt(pp))
    while pp % gx != 0:
        gx -= 1
    return gx, pp // gx


def centroid_fft_tile_geometry(
    *, nk: int, band_chunk: int, p_band: int,
) -> tuple[int, int]:
    """Return ``(k_tile, local_flat_rows)`` for a centroid WFN transfer.

    The loader mesh-rounds its global band tile before distributing that
    axis.  Bound ``k_tile * (band_tile / p_band)`` by that same physical band
    tile, so the local FFT-row batch does not grow just because the band axis
    is divided over more ranks.  This pure rule also serves artifact-reuse
    paths, which need a safe centroid resample but deliberately do not run the
    full zeta-fit memory planner.
    """
    nk = int(nk)
    band_chunk = int(band_chunk)
    p_band = int(p_band)
    if nk <= 0 or band_chunk <= 0 or p_band <= 0:
        raise ValueError(
            "centroid FFT geometry requires positive nk, band_chunk, and "
            f"p_band; got nk={nk}, band_chunk={band_chunk}, p_band={p_band}")
    band_tile = round_up(band_chunk, p_band)
    local_bands = band_tile // p_band
    k_tile = min(nk, max(1, band_tile // local_bands))
    return k_tile, k_tile * local_bands


# ---------------------------------------------------------------------------
# The consequential array inventory (§1)
# ---------------------------------------------------------------------------

def _persistent_bytes(*, nk, ns, nq, nq_disk, mu, nb, ngkmax, n_rtot,
                      p_x, p_y, low_mem_bands: bool = False) -> dict:
    """The un-chunkable floor resident across the whole r-chunk loop.

    ``L_q`` and ``gflat_acc`` are ÷P (μ²/μ-family).

    ``psi_copies`` prices the RESOLVED ``Wavefunctions`` layout
    (``gw.wavefunction_bundle``, report
    ``reports/gwjax_low_mem_bands_audit_2026-08-22/report.md`` §verdict/§7):

      ``low_mem_bands=False`` (``layout="legacy"``, the default): the four
      ψ centroid copies are single-axis ÷√P (2 on 'x', 2 on 'y') — the
      corrected centroid term (design §5 bug #4: NOT ÷p_xy).
      ``2·S/Px + 2·S/Py`` where ``S = psi_one``.

      ``low_mem_bands=True`` (``layout="face"``): exactly TWO copies, both
      2-D sharded on the FULL (x, y) mesh (``psi_nmu``, ``psi_mun``).
      ``2·S/(Px·Py)`` — the ``2·√P`` reduction on a square mesh the
      feature exists for.

      As of the zeta-fit face-CCT and r-chunk redesigns
      (feat/zeta-fit-face-psi and feat/zeta-fit-rchunk-face-psi,
      2026-08-22), this term is GENUINELY accurate for Stages A--D
      (centroid load, CCT/Cholesky): ``gw.gw_init.prepare_isdf_and_
      wavefunctions`` builds ``psi_nmu``/``psi_mun`` immediately after the
      fresh load and drops the single-axis ψ_Y copy BEFORE the fit runs,
      and ``gw.isdf_fitting.fit_zeta_to_h5``'s CCT step (STEP 2) reads
      them directly (``isdf.core.c_q_from_psi_sm(layout='face')``, a
      distributed SUMMA GEMM) instead of single-axis Y-forms.  Stages C/D
      read the band-contraction operand from that same persistent face
      carrier; their bounded incremental workspace is disclosed separately
      by ``GFlatChunkPlan.stage_cd_psi_bytes``.

    ``loader_tables`` is the WFN loader's REPLICATED per-k metadata (the
    sparse-G→FFT-box index + the τ-phase row), retained for the loader's
    lifetime and **P-INDEPENDENT** — adding nodes never shrinks it, so it
    belongs in the floor.  Measured history: memory-model.md §"Measured
    corrections behind the G-flat terms" #1."""
    P_ = p_x * p_y
    psi_one = _c128(nk, ns, mu, nb)
    if low_mem_bands:
        psi_copies = 2.0 * psi_one / P_
    else:
        psi_copies = 2 * psi_one / p_x + 2 * psi_one / p_y
    return {
        "L_q":         _c128(nq, mu, mu, shard=P_),
        "gflat_acc":   _c128(nq_disk, mu, ngkmax, shard=P_),
        "psi_copies":  psi_copies,
        "loader_tables": 4.0 * nk * n_rtot + _C128 * nk * ngkmax,
    }


def _fft_box_bytes(*, nk, bc, ns, fft_grid, mesh_xy, p_xy) -> float:
    """Per-rank bytes of one production WFN spatial FFT box.

    The caller supplies the transform's actual live k extent: bounded
    ``centroid_k_chunk`` for Stage A, full ``nk`` for psi-r/to_rchunk stages.
    MEASURED whenever a real ``Mesh`` is available: compiles the production
    WFN spatial ``ifftn(norm='ortho')`` helper at this shape/sharding and
    reads XLA's buffer peak PLUS the cuFFT plan workspace, which is not in
    buffer assignment.  Both halves matter — the analytic factor alone
    under-predicted Si-10³ by 19 GiB (design §6), and the cuFFT half is
    >13.7 GB/rank at the CrI3 V_q box.
    Otherwise: the analytic
    ``nk·(bc/p_xy)·ns·n_rtot·16·4.0`` box-copy bound,
    which does NOT see the plan workspace — and announces that."""
    nx, ny, nz = (int(v) for v in fft_grid)
    n_rtot = nx * ny * nz
    if isinstance(mesh_xy, Mesh):
        try:
            from common.fft_helpers import query_fft_peak_bytes
            return float(query_fft_peak_bytes(
                input_shape=(int(nk), int(bc), int(ns), nx, ny, nz),
                fft_axes=(-3, -2, -1),
                sharding=NamedSharding(
                    mesh_xy, P(None, ('x', 'y'), None, None, None, None)),
                kind="ifftn",
                norm="ortho",
                dtype=jnp.complex128,
            ))
        except Exception as exc:
            why = f"the probe failed to compile ({type(exc).__name__}: {exc})"
    else:
        why = f"mesh_xy is a {type(mesh_xy).__name__}, not a jax Mesh"
    # Analytic fallback (bands sharded over all P; ns + FFT axes replicated).
    _announce(f"fft-box-unmeasured:{why}",
              f"WFN FFT-box term is the analytic {_FFT_CUFFT_FACTOR}x "
              f"box-copy bound because {why}.  It does NOT include cuFFT plan "
              f"workspace, so this planner will UNDER-predict FFT-box stages")
    return _c128(nk, bc, ns, n_rtot, shard=p_xy) * _FFT_CUFFT_FACTOR


#: Concurrent copies of the band-all_gathered FULL-r ψ(r) slab that XLA
#: keeps live inside ``z_q_from_psi_sm``'s scan body.  ONE is unavoidable
#: (the ``lax.all_gather`` output); the historical second came from the
#: ``jnp.take`` band-compaction, now elided at trace time whenever the
#: permutation is the identity (``isdf/core.py`` ``_y_compact_identity``).
#: Kept at 2 because the elision is config-dependent (a short final band
#: chunk re-enables the take) and an under-estimate here is a hard OOM.
_GATHERED_PSI_SLOTS = 2


def _stage_C_slope(*, nk, ns, nq, mu, slots, p_xy, band_chunk, p_y) -> float:
    """Per-``cr`` bytes of the Stage-C transient (the binder): the ``slots``
    concurrent pair-density accumulators, the Z_q output, and the two psi(r)
    slabs the band-gather machinery keeps live.

    THE GATHERED psi(r) SLAB IS SHARDED ON 'y' ONLY -- 1/p_y, not 1/P.
    ``z_q_from_psi_sm`` computes each rank's 1/P band block over the FULL
    r-chunk, then does ``all_to_all('y', split r, concat bands)`` +
    ``all_gather('x', bands)``, so every rank ends up holding
    ``(nk, band_chunk, ns, cr/p_y)`` -- ALL bands, but only ITS r-block.
    A second, smaller slab is live alongside it: this rank's OWN
    ``band_chunk/p_xy`` bands over the FULL r-chunk (the all-to-all source),
    which is unavoidable because other y-ranks need other r-blocks of exactly
    those bands.

    DO NOT REGRESS the two mesh divisions here; what an un-divided gather
    cost is memory-model.md §"Measured corrections behind the G-flat
    terms" #2 (job 7874236, a single 271 GB allocation).
    """
    return (slots * _c128(nk, ns, ns, mu, shard=p_xy)   # pair carry
            + _c128(nq, mu, shard=p_xy)                 # Z_q / zeta_out
            # gathered psi(r): all bands, r-block only  -> /p_y
            + _GATHERED_PSI_SLOTS * _c128(nk, band_chunk, ns, shard=p_y)
            # all-to-all source: own bands, full r      -> /p_xy on bands
            + _c128(nk, max(1, band_chunk // p_xy), ns))


def _stage_C_face_terms(
        *, nk, ns, mu, face_nb, slots, p_x, p_y, p_xy,
        band_chunk, n_band_chunks) -> dict[str, float]:
    """Analytic live-shape census for the scalar-pair face kernel.

    Source evaluates one scalar spin pair at a time, but the executable's
    allocation may still be placed as an ``ns²``-wide arena.  Express both
    cases as a count of rank-3 ``(nk, mu/Px, r/Py)`` equivalents.  ``slots``
    is the *total* count from :func:`_face_pair_density_slots`; accumulated Z
    and k-IFFT outputs are included there and are not additional terms.

    ``face_nb`` is the exact ``psi_mun[..., b0:b4]`` carrier width, distinct
    from both the legacy inventory and a possibly narrowed zeta-fit union.
    The cached route retains one current-r Y slab per padded fit-band chunk;
    the repeated-transform fallback retains one slab and its source.
    """
    x_rows = 1 if int(ns) == 1 else 2
    face_conj = _c128(nk, ns, mu, face_nb, shard=p_xy)
    x_block = _c128(nk, x_rows, max(1, mu // p_x), band_chunk)
    pair_rank3 = _c128(nk, mu, shard=p_xy)
    pair_peak = float(slots) * pair_rank3
    y_block = _c128(nk, band_chunk, ns, shard=p_y)
    y_source = _c128(nk, max(1, band_chunk // p_xy), ns)
    y_cache = float(n_band_chunks) * y_block
    return {
        # psi_mun_conj plus the selected-row gather and psum result.
        "constant": face_conj + 2.0 * x_block,
        # Complete executable-specific rank-3-equivalent arena.
        "pair_arena_slope": pair_peak,
        "y_block_slope": y_block,
        "y_source_slope": y_source,
        "y_cache_slope": y_cache,
        # Build peak: completed stack + last gather/compaction/source.
        "cache_build_slope": (
            y_cache + _GATHERED_PSI_SLOTS * y_block + y_source),
        # Pair peak: cached stack plus the calibrated rank-3 live set.
        "cache_pair_slope": y_cache + pair_peak,
        # Bounded fallback: one gathered block/source plus the pair peak.
        "repeated_pair_slope": (
            _GATHERED_PSI_SLOTS * y_block + y_source + pair_peak),
    }


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class GFlatChunkPlan:
    """Resolved chunk sizes + per-rank HBM high-water estimate."""
    band_chunk: int
    #: Outer k tile for the centroid WFN transfer.  This is derived from the
    #: resolved band tile (not a frontend knob) so the LOCAL flat FFT-row
    #: batch stays bounded as P changes.
    centroid_k_chunk: int
    #: Actual per-rank flat-row extent priced for one centroid FFT batch:
    #: ``centroid_k_chunk * (band_chunk / P_band)``.
    centroid_fft_rows_local: int
    #: Stage-A centroid-load FFT price at the bounded outer-k tile.
    centroid_fft_bytes: float
    #: Full-nk to_rchunk_inner price for psi-r cache/streamed face stages.
    zeta_transform_fft_bytes: float
    r_chunk: int
    n_r_chunks: int
    q_chunk: int
    zeta_solve_memory_route: str
    gflat_chunk_size: int
    hwm_bytes: float
    peak_breakdown: dict          # stage -> total per-rank bytes (persist+transient)
    persistent_bytes: float
    bottleneck: str               # binding stage name
    p_min: int                    # rank floor
    budget_bytes: float
    target_utilization: float
    #: "legacy" | "face" — the ``Wavefunctions`` layout this plan was
    #: priced under (``low_mem_bands`` deck key).  Disclosed on the
    #: startup banner so a run cannot believe it selected the low-memory
    #: path when the planner priced the other one (report §7).
    psi_layout: str = "legacy"
    #: Per-rank bytes of the ψ centroid term (``_persistent_bytes``'s
    #: ``"psi_copies"``) AT THE RESOLVED LAYOUT — the number the banner
    #: prints next to ``psi_layout``.
    psi_layout_bytes: float = 0.0
    #: Per-rank bytes of the r-chunk loop's OWN incremental ψ residency
    #: during Stage C/D (``fit_one_rchunk`` / the accumulate step),
    #: beyond whatever ``psi_layout_bytes`` already counts as persistent
    #: for the run.  LAYOUT-DEPENDENT (unlike its name suggests — kept for
    #: back-compat):
    #:
    #: ``psi_layout="legacy"``: the two surviving X-form single-axis
    #: copies (mu on 'x', bands replicated; ``2·psi_one/p_x``) that are
    #: what is ACTUALLY resident during Stage C/D after
    #: ``gw.isdf_fitting.fit_zeta_to_h5`` frees the Y-form copies right
    #: after CCT (2026-08-22 fresh-fit low-mem psi contract; report
    #: ``reports/gwjax_low_mem_bands_audit_2026-08-22/report.md`` census
    #: row "Fresh centroid load/liveness").
    #:
    #: ``psi_layout="face"``: NO resident single-axis X-form at all — the
    #: r-chunk-loop port (feat/zeta-fit-rchunk-face-psi-2026-08-22,
    #: docs/architecture/zeta_fit_face_psi_cct.md's r-chunk section) reads
    #: the band-contraction operand per-band-chunk out of the PERSISTENT
    #: psi_mun_fresh instead (``isdf.core._z_q_face``).  This field then
    #: prices only the small bounded per-call/per-bc transients that
    #: mechanism adds — see ``plan_gflat_chunks``'s computation for the
    #: exact terms.
    #:
    #: The face value is folded into its separate Stage-C build/pair peaks;
    #: this field discloses that already-priced subset.  The legacy value
    #: remains a compatibility disclosure of the post-CCT inventory.
    stage_cd_psi_bytes: float = 0.0
    #: Whether the ζ fit hoists every band/grid coefficient into one
    #: all-P-sharded ψ(r) cache.  ``False`` selects the existing streamed
    #: band-chunk FFT path; the planner uses it only for ``low_mem_bands``
    #: when even the cache's minimum persistent floor exceeds the target.
    cache_psi_r: bool = True
    #: Face-layout Stage-C structural route.  True stacks the canonical
    #: current-r-chunk Y band slabs once and reuses them for every scalar
    #: spin pair; False repeats one bounded transform/scatter and is the
    #: always-valid large-band fallback.  Planner-owned, never a knob.
    cache_face_y_blocks: bool = False
    #: Global-r width of one internal Y-cache tile.  Equal to ``r_chunk``
    #: for the original one-pass cache.  Smaller means the explicit outer
    #: chunk is partitioned into equal cache-sized transactions, preserving
    #: solve amortization without crossing into the ns² repeated route.
    face_y_cache_r_tile: int = 0
    #: Per-rank bytes in the selected face Y cache at resolved r width.
    face_y_cache_bytes: float = 0.0

    def format(self) -> str:
        bg = self.budget_bytes / 1e9
        hwm = self.hwm_bytes / 1e9
        lines = [
            "  ISDF memory model — chunk plan + HWM estimate",
            f"    psi layout    = {self.psi_layout} "
            f"({self.psi_layout_bytes / 1e9:.3f} GB/dev, ψ centroid copies)",
            f"    Stage C/D ψ floor (post-CCT, {self.psi_layout} r-chunk "
            f"incremental) = {self.stage_cd_psi_bytes / 1e9:.3f} GB/dev "
            + ("[included in face Stage-C peaks]"
               if self.psi_layout == "face" else
               "[legacy compatibility disclosure]"),
            "    ψ(r) source   = " + (
                "hoisted all-band cache" if self.cache_psi_r else
                "streamed band-chunk FFT (low-memory fallback)"),
            "    face Y route  = " + (
                (("current-r cache" if self.face_y_cache_r_tile == self.r_chunk
                  else f"tiled current-r cache ({self.r_chunk // max(self.face_y_cache_r_tile, 1)}"
                       f" x {self.face_y_cache_r_tile})"))
                if self.cache_face_y_blocks else
                ("repeated bounded transform" if self.psi_layout == "face"
                 else "n/a (legacy layout)")),
            *([f"    face Y cache = "
               f"{self.face_y_cache_bytes / 1e9:.3f} GB/dev"]
              if self.psi_layout == "face" else []),
            f"    band_chunk    = {self.band_chunk}",
            f"    centroid FFT = k_tile {self.centroid_k_chunk}, "
            f"{self.centroid_fft_rows_local} local rows, "
            f"{self.centroid_fft_bytes / 1e9:.3f} GB/dev",
            f"    zeta FFT      = full-nk transform, "
            f"{self.zeta_transform_fft_bytes / 1e9:.3f} GB/dev",
            f"    r_chunk       = {self.r_chunk}  ({self.n_r_chunks} chunks)",
            f"    q_chunk       = {self.q_chunk}",
            f"    ζ solve memory = {self.zeta_solve_memory_route}",
            f"    gflat_cs      = {self.gflat_chunk_size}",
            f"    P_min (floor) = {self.p_min}",
            f"    budget        = {bg:.2f} GB/dev  (util {self.target_utilization:.2f})",
            f"    persistent    = {self.persistent_bytes/1e9:.2f} GB/dev",
            f"    HWM estimate  = {hwm:.2f} GB/dev "
            f"({100 * hwm / max(bg, 1e-9):.0f}% of budget) "
            f"[binder: {self.bottleneck}]",
            "    per-stage peaks (persistent + transient, GB/dev):",
        ]
        for name, b in sorted(self.peak_breakdown.items(), key=lambda kv: -kv[1]):
            mark = " <=" if name == self.bottleneck else ""
            lines.append(f"      {name:.<20s} {b/1e9:>7.2f}{mark}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The planner
# ---------------------------------------------------------------------------

def plan_gflat_chunks(
    *,
    meta,
    mesh_xy,
    nb_total: int,
    face_nb_total: int | None = None,
    fit_nb_total: int | None = None,
    ngkmax: int,
    n_q_disk: int,
    budget_gb: float,
    target_utilization: Optional[float] = None,
    is_bispinor: bool = True,
    max_chunks: int = 64,
    r_chunk_override: int | None = None,
    band_chunk_override: int | None = None,
    gflat_chunk_size_override: int | None = None,
    n_q_ibz: int | None = None,
    pair_density_slots: int | None = None,
    distributed_zeta_solve: str = "auto",
    low_mem_bands: bool = False,
    face_current_vertex: bool = False,
) -> GFlatChunkPlan:
    """Pick ``(band_chunk, centroid_k_chunk, r_chunk, q_chunk,
    gflat_chunk_size)`` so the
    per-rank HWM lands under ``util·budget``.  Reports the rank floor
    ``P_min`` and the binding stage.

    ``nb_total`` sizes the legacy resident centroid-wavefunction inventory;
    ``face_nb_total`` is the exact low-memory ``[b0,b4)`` face-carrier width
    (defaults to ``nb_total`` for compatibility);
    ``fit_nb_total`` is the logical union of the two ζ fit windows and owns
    the band-chunk K extent.  Keeping those extents separate matters when
    ``zeta_nband`` narrows only the fit.

    Overrides (each a ``cohsex.in`` knob; ``>0`` wins over the picker):
    ``r_chunk_override``, ``band_chunk_override``, ``gflat_chunk_size_override``.
    The shipping no-key configuration passes ``band_chunk_override=16``:
    the planner mesh-rounds it and caps it at the logical fit window.  An
    explicit deck value ``band_chunk_size=0`` reaches this function as
    ``band_chunk_override=None`` and opts into the full-window-first ladder.

    ``distributed_zeta_solve`` controls the solve-memory inventory.  ``auto``
    is conservatively priced as ``replicated``; the live resolver may choose
    ``per_q`` at execution time, but that cannot make this plan optimistic.

    ``low_mem_bands`` (the deck key of the same name; default ``False``)
    selects which ``Wavefunctions`` layout ``_persistent_bytes`` and the
    Stage-E base price the ψ centroid inventory as — see
    :func:`_persistent_bytes`.  When that face inventory fits but the
    all-band ψ(r) hoist does not, it also selects the canonical streamed
    band-chunk FFT route.  The legacy/default route retains the hoist.

    ``face_current_vertex`` selects the nonidentity-current face executable's
    measured ``ns²`` arena.  False preserves the independently calibrated
    four-slot identity/charge executable.
    """
    p_x = int(mesh_xy.shape['x'])
    p_y = int(mesh_xy.shape['y'])
    p_xy = p_x * p_y

    nk = int(meta.nk_tot)
    ns = int(meta.nspinor)
    mu = int(getattr(meta, "n_rmu_padded", None) or meta.n_rmu)
    nq = nk
    nq_disk = int(n_q_disk)
    n_rtot = int(meta.n_rtot)
    ngkmax = int(ngkmax)
    nb = int(nb_total)
    face_nb = int(face_nb_total if face_nb_total is not None else nb)
    fit_nb = int(fit_nb_total if fit_nb_total is not None else nb)
    if fit_nb <= 0 or face_nb <= 0:
        raise ValueError(
            f"fit_nb_total and face_nb_total must be positive, got "
            f"fit={fit_nb}, face={face_nb}")
    if low_mem_bands and face_nb % p_y:
        raise ValueError(
            f"face_nb_total={face_nb} must divide the face band mesh axis "
            f"Py={p_y}")
    fft_grid = tuple(getattr(meta, 'fft_grid', None)
                     or (int(round(n_rtot ** (1 / 3))),) * 3)
    if n_q_ibz is None:
        n_q_ibz = nq
    n_q_ibz = int(n_q_ibz)

    if target_utilization is None:
        target_utilization = bfc_fragmentation_target_utilization(ns)
    slots = pair_density_slots if pair_density_slots is not None \
        else _pair_density_slots()
    face_slots = _face_pair_density_slots(
        ns=ns, current_vertex=bool(face_current_vertex))

    budget = budget_gb * 1e9
    target = budget * target_utilization

    inventory_nb = face_nb if low_mem_bands else nb
    sys = dict(nk=nk, ns=ns, nq=nq, nq_disk=nq_disk, mu=mu,
               nb=inventory_nb,
               ngkmax=ngkmax, n_rtot=n_rtot, low_mem_bands=bool(low_mem_bands))

    # The hoisted ψ(r) cache has no centroid axis: at large FFT grids it can
    # dominate a low-memory face calculation even though every centroid
    # object is correctly 2-D sharded.  The ζ kernel already owns a canonical
    # cache-free route (one ψ(G)->ψ(r_chunk) transform per band chunk and
    # r chunk).  Select that route only when low_mem_bands was requested AND
    # the smallest possible all-band cache plus the true persistent base
    # cannot fit the target.  Ordinary modes retain the established hoist.
    _persistent_base = _persistent_bytes(p_x=p_x, p_y=p_y, **sys)
    _min_cache_slots = max(
        p_xy, ((fit_nb + p_xy - 1) // p_xy) * p_xy)
    _min_cache_bytes = _c128(
        nk, _min_cache_slots, ns, n_rtot, shard=p_xy)
    cache_psi_r = True
    _cache_probe_peak = sum(_persistent_base.values()) + _min_cache_bytes
    if low_mem_bands:
        _cache_probe_bc = (
            int(band_chunk_override)
            if band_chunk_override and band_chunk_override > 0 else fit_nb)
        _cache_probe_bc = max(
            p_xy, ((_cache_probe_bc + p_xy - 1) // p_xy) * p_xy)
        _cache_probe_bc = min(_cache_probe_bc, _min_cache_slots)
        _cache_probe_r = min(
            n_rtot, max(min(mu, n_rtot), math.ceil(n_rtot / max_chunks)))
        _cache_probe_r = max(
            p_y, _cache_probe_r - _cache_probe_r % max(p_y, 1))
        _cache_probe_face = _stage_C_face_terms(
            nk=nk, ns=ns, mu=mu, face_nb=face_nb,
            slots=face_slots, p_x=p_x, p_y=p_y, p_xy=p_xy,
            band_chunk=_cache_probe_bc,
            n_band_chunks=math.ceil(fit_nb / _cache_probe_bc))
        _cache_probe_peak += (
            _cache_probe_face["constant"]
            + _cache_probe_face["repeated_pair_slope"] * _cache_probe_r)
        cache_psi_r = not (
            sum(_persistent_base.values()) + _min_cache_bytes > target
            or _cache_probe_peak > target)
    if not cache_psi_r:
        _announce(
            "stream-psi-r-cache-lowmem",
            "the all-band ψ(r) cache is disabled for low_mem_bands: its "
            f"minimum persistent floor is "
            f"{(sum(_persistent_base.values()) + _min_cache_bytes) / 1e9:.2f} "
            f"GB/dev and its minimum-performance face peak is "
            f"{_cache_probe_peak / 1e9:.2f} GB/dev vs the "
            f"{target / 1e9:.2f} GB/dev target.  The ζ fit "
            "will use its canonical streamed band-chunk FFT path")

    # ---- Phase 1: the rank floor (un-chunkable ÷P / ÷√P family) ---------
    def _floor_at(pp: int) -> float:
        px, py = _factor_mesh(pp)
        face_nb_pp = ((face_nb + pp - 1) // pp) * pp
        if band_chunk_override and band_chunk_override > 0:
            floor_bc = int(band_chunk_override)
        else:
            floor_bc = fit_nb
        floor_bc = max(pp, ((floor_bc + pp - 1) // pp) * pp)
        fit_padded = max(pp, ((fit_nb + pp - 1) // pp) * pp)
        floor_bc = min(floor_bc, fit_padded)
        cache_slots = math.ceil(fit_nb / floor_bc) * floor_bc
        psi_r_cache = _c128(nk, cache_slots, ns, n_rtot, shard=pp)
        floor_sys = dict(sys)
        if low_mem_bands:
            floor_sys["nb"] = face_nb_pp
        floor = (sum(_persistent_bytes(
                     p_x=px, p_y=py, **floor_sys).values())
                 + (psi_r_cache if cache_psi_r else 0.0))
        if low_mem_bands:
            # P_min must admit one legal face r slab.  Use the universal
            # repeated route at r=Py and the full-nk analytic transform.
            face_floor = _stage_C_face_terms(
                nk=nk, ns=ns, mu=mu, face_nb=face_nb_pp,
                slots=face_slots, p_x=px, p_y=py, p_xy=pp,
                band_chunk=floor_bc,
                n_band_chunks=math.ceil(fit_nb / floor_bc))
            fft_floor = (_c128(
                nk, floor_bc, ns, n_rtot, shard=pp)
                * _FFT_CUFFT_FACTOR)
            pair_floor = (
                face_floor["constant"]
                + face_floor["repeated_pair_slope"] * py
                + (0.0 if cache_psi_r else fft_floor))
            floor += max(fft_floor if cache_psi_r else 0.0, pair_floor)
        return floor

    # ``loader_tables`` is P-INDEPENDENT, so if it alone busts the budget no
    # rank count fixes it — say so at once instead of stepping the search a
    # million times (each step factorises the mesh).
    _p_independent = _persistent_bytes(p_x=1, p_y=1, **sys)["loader_tables"]
    p_min = 1
    if _p_independent > target:
        p_min = 1 << 20
    else:
        while p_min < 1 << 20 and _floor_at(p_min) > target:
            p_min += 1

    persistent = _persistent_bytes(p_x=p_x, p_y=p_y, **sys)
    persistent_total = sum(persistent.values())

    # ---- band_chunk (Stage A FFT box) ----------------------------------
    def _bump_bc(bc: int) -> int:
        bc = int(bc)
        if p_xy > 1 and bc % p_xy != 0:
            bc = ((bc + p_xy - 1) // p_xy) * p_xy
        # The physical transport chunk may extend past the logical ζ edge by
        # at most P-1 zero-masked bands.  Do not clamp a rounded value back to
        # a non-divisible logical edge (e.g. 50 -> 52 -> 50 at P=4).
        fit_nb_padded = ((fit_nb + p_xy - 1) // p_xy) * p_xy
        return min(max(bc, p_xy), max(fit_nb_padded, p_xy))

    _fft_box_cache: dict[tuple[int, int], float] = {}

    def _fft_for_bc(bc: int, *, nk_extent: int) -> float:
        bc = int(bc)
        nk_extent = int(nk_extent)
        key = (nk_extent, bc)
        if key not in _fft_box_cache:
            _fft_box_cache[key] = _fft_box_bytes(
                nk=nk_extent, bc=bc, ns=ns, fft_grid=fft_grid,
                mesh_xy=mesh_xy, p_xy=p_xy)
        return _fft_box_cache[key]

    # The full-window decision accounts for both things whose live size the
    # band chunk changes: the ψ(G)->ψ(r) FFT box (Stage A) and Stage C's
    # pair-density accumulator plus gathered ψ slab.  With an explicit
    # r-chunk override, price exactly that extent.  Otherwise use the
    # planner's existing performance floor / max-chunk floor; Phase 2 may
    # still choose a larger affordable r chunk after band K is resolved.
    r_lo = min(mu, n_rtot)
    r_alignment = p_y if low_mem_bands else p_xy
    if r_chunk_override and r_chunk_override > 0:
        r_for_band_guard = min(int(r_chunk_override), n_rtot)
    else:
        r_for_band_guard = max(r_lo, math.ceil(n_rtot / max_chunks))
    if r_alignment > 1:
        r_for_band_guard = max(
            r_alignment,
            r_for_band_guard - r_for_band_guard % r_alignment)

    def _band_candidate_fits(bc: int) -> bool:
        n_bc = math.ceil(fit_nb / bc)
        # The cache is stacked by uniform transport chunks.  Its band slots
        # are sharded over all P; only the final chunk's at-most-(bc-1) pad
        # rows are extra, and they remain zero-masked.
        psi_r_cache = _c128(
            nk, n_bc * bc, ns, n_rtot, shard=p_xy)
        centroid_k, _ = centroid_fft_tile_geometry(
            nk=nk, band_chunk=bc, p_band=p_xy)
        centroid_fft_t = _fft_for_bc(bc, nk_extent=centroid_k)
        zeta_fft_t = _fft_for_bc(bc, nk_extent=nk)
        if low_mem_bands:
            face = _stage_C_face_terms(
                nk=nk, ns=ns, mu=mu, face_nb=face_nb,
                slots=face_slots, p_x=p_x, p_y=p_y, p_xy=p_xy,
                band_chunk=bc, n_band_chunks=n_bc)
            fit_t = (face["constant"]
                     + face["repeated_pair_slope"] * r_for_band_guard)
        else:
            c_slope = _stage_C_slope(
                nk=nk, ns=ns, nq=nq, mu=mu, slots=slots,
                p_xy=p_xy, band_chunk=bc, p_y=p_y)
            fit_t = c_slope * r_for_band_guard
        # Centroid sampling sees only its bounded k tile.  The incumbent
        # to_rchunk_inner source sees full nk and is a distinct peak when
        # hoisted, or coexists with the streamed pair route when not.
        transient = (max(centroid_fft_t, zeta_fft_t, fit_t)
                     if cache_psi_r else
                     max(centroid_fft_t, zeta_fft_t + fit_t))
        return (persistent_total + (psi_r_cache if cache_psi_r else 0.0)
                + transient <= target)

    if band_chunk_override and band_chunk_override > 0:
        band_chunk = _bump_bc(band_chunk_override)
    else:
        # Prefer one GEMM K dimension spanning the whole logical ζ window.
        # This removes rank-5 carry read/modify/write traffic between band
        # chunks.  If the measured FFT box + existing Stage-C accounting do
        # not fit, retain the old power-of-two family and choose its largest
        # member admitted by the same memory guard.
        full_window = _bump_bc(fit_nb)
        if _band_candidate_fits(full_window):
            band_chunk = full_window
        else:
            band_chunk = _bump_bc(1)
            bc = 1
            while bc * 2 <= fit_nb:
                trial = _bump_bc(bc * 2)
                if trial >= full_window or not _band_candidate_fits(trial):
                    break
                band_chunk = trial
                bc *= 2

    centroid_k_chunk, centroid_fft_rows_local = centroid_fft_tile_geometry(
        nk=nk, band_chunk=band_chunk, p_band=p_xy)
    fft_box_A = _fft_for_bc(
        band_chunk, nk_extent=centroid_k_chunk)
    fft_box_zeta_transform = _fft_for_bc(
        band_chunk, nk_extent=nk)

    # ψ(r) is hoisted across the outer r-chunk loop.  It is a band-flat
    # all-P-sharded cache, never a replicated full-window object.  Price its
    # uniform final-chunk pad exactly; this becomes part of the persistent
    # floor for all r-chunk stages.  The rectangular lax.scan result is one
    # compiled shape family.  For a 50-band window at bc16 it intentionally
    # holds 64 slots (28% pad; 7.25 vs 5.66 GB at P=1 on Si 80 Ry).  A ragged
    # last allocation would split the scan/cache ABI into another executable,
    # so the pad stays until that trade is measured as its own change.
    _cache_n_bc = math.ceil(fit_nb / band_chunk)
    if cache_psi_r:
        persistent["psi_r_cache"] = _c128(
            nk, _cache_n_bc * band_chunk, ns, n_rtot, shard=p_xy)
    persistent_total = sum(persistent.values())

    # ---- Phase 2: dial chunk_r against Stage C's slope ------------------
    # The face route has three non-overlapping peaks: current-r Y-cache
    # build, scalar-pair accumulation/final k FFT, and the all-P-aligned
    # solve seam.  Keep them separate; adding them invents a live set.
    cache_face_y_blocks = False
    face_y_cache_r_tile = 0
    face_terms = None
    face_cache_build_t = 0.0
    face_tile_concat_t = 0.0
    face_y_cache_bytes = 0.0
    if low_mem_bands:
        face_terms = _stage_C_face_terms(
            nk=nk, ns=ns, mu=mu, face_nb=face_nb,
            slots=face_slots, p_x=p_x, p_y=p_y, p_xy=p_xy,
            band_chunk=band_chunk, n_band_chunks=_cache_n_bc)
        # to_rchunk_inner consumes the full-nk carrier.  This is deliberately
        # the 12.96-GB CrI3/P16 term, not Stage A's 5.76-GB bounded k tile.
        streamed_fft = 0.0 if cache_psi_r else fft_box_zeta_transform
        face_headroom = max(
            target - persistent_total - face_terms["constant"], 0.0)

        cache_pair_cap = int(
            face_headroom / face_terms["cache_pair_slope"])
        cache_build_headroom = max(face_headroom - streamed_fft, 0.0)
        cache_build_cap = int(
            cache_build_headroom / face_terms["cache_build_slope"])
        # The calibrated four-slot arena already contains old/new Z and both
        # k-IFFT outputs.  Only the Y cache sits outside its placement bound.
        cache_other_slope = face_terms["y_cache_slope"]
        cache_arena_cap = int(
            _ARENA_PLACEMENT_FRAC * face_headroom
            / (face_terms["pair_arena_slope"]
               + _ARENA_PLACEMENT_FRAC * cache_other_slope))
        cache_cap = min(cache_pair_cap, cache_build_cap, cache_arena_cap)

        repeated_headroom = max(face_headroom - streamed_fft, 0.0)
        repeated_pair_cap = int(
            repeated_headroom / face_terms["repeated_pair_slope"])
        repeated_other_slope = (
            _GATHERED_PSI_SLOTS * face_terms["y_block_slope"]
            + face_terms["y_source_slope"])
        repeated_arena_cap = int(
            _ARENA_PLACEMENT_FRAC * repeated_headroom
            / (face_terms["pair_arena_slope"]
               + _ARENA_PLACEMENT_FRAC * repeated_other_slope))
        repeated_cap = min(repeated_pair_cap, repeated_arena_cap)

        route_width = (min(int(r_chunk_override), n_rtot)
                       if r_chunk_override and r_chunk_override > 0
                       else min(n_rtot, max(
                           r_lo, math.ceil(n_rtot / max_chunks))))
        route_width = max(
            r_alignment,
            route_width - route_width % max(r_alignment, 1))
        route_tile = bounded_partition_tile(
            route_width, cache_cap, r_alignment)
        route_tiles = (route_width // route_tile) if route_tile else 0
        # The bounded fallback executes one transform per scalar spin pair.
        # A tiled cache is useful only while it executes strictly fewer
        # transforms than that fallback; otherwise retain the smaller-memory
        # incumbent.  The outer r chunk is unchanged either way.
        cache_face_y_blocks = bool(
            ns > 1 and route_tile > 0 and route_tiles < ns * ns)
        face_y_cache_r_tile = route_tile if cache_face_y_blocks else 0
        if cache_face_y_blocks:
            C_slope = face_terms["cache_pair_slope"]
            C_constant = face_terms["constant"]
            r_from_budget = min(cache_pair_cap, cache_build_cap)
            r_from_arena = cache_arena_cap
            r_budget_cap = cache_cap
        else:
            C_slope = face_terms["repeated_pair_slope"]
            C_constant = face_terms["constant"] + streamed_fft
            r_from_budget = repeated_pair_cap
            r_from_arena = repeated_arena_cap
            r_budget_cap = repeated_cap
    else:
        C_slope = _stage_C_slope(
            nk=nk, ns=ns, nq=nq, mu=mu, slots=slots,
            p_xy=p_xy, band_chunk=band_chunk, p_y=p_y)
        C_constant = 0.0
        arena_slope = slots * _c128(nk, ns, ns, mu, shard=p_xy)

    headroom_C = max(
        target - persistent_total - C_constant
        - (0.0 if (low_mem_bands or cache_psi_r)
           else fft_box_zeta_transform),
        0.0)
    if r_chunk_override and r_chunk_override > 0:
        # The register-documented run-level workaround: an explicit
        # r_chunk_size wins over every cap below, exactly as before.
        r_chunk = min(int(r_chunk_override), n_rtot)
    else:
        if not low_mem_bands:
            r_from_budget = (
                int(headroom_C / C_slope) if C_slope > 0 else n_rtot)
            r_from_arena = (
                int(_ARENA_PLACEMENT_FRAC * headroom_C / arena_slope)
                if arena_slope > 0 else n_rtot)
            r_budget_cap = min(r_from_budget, r_from_arena)
        # Performance floors — chunks at least μ wide, at most
        # ``max_chunks`` of them.  THE BUDGET OUTRANKS THE FLOORS: until
        # 2026-08-22 ``r_lo = min(μ, n_rtot)`` silently overrode a
        # smaller budget-derived width, which is how the 9x9/626b run
        # was handed r_chunk=5296 (= μ) against a plan its own banner
        # priced at 244% of budget (JID 57281385 step .28) — the
        # instrument measured the overrun and proceeded.  A perf floor
        # that busts the budget is an OOM, not a floor.
        r_floor_perf = max(r_lo, math.ceil(n_rtot / max_chunks))
        r_chunk = min(n_rtot, r_floor_perf)
        if r_budget_cap < r_chunk:
            capped = max(r_alignment, r_budget_cap)
            _announce(
                "stage-c-rchunk-budget-cap",
                f"Stage C r_chunk lowered {r_chunk} -> {capped} by the "
                f"memory budget (sum cap {r_from_budget}, single-arena "
                f"placement cap {r_from_arena} at "
                f"{_ARENA_PLACEMENT_FRAC:.2f}x post-persistent headroom); "
                f"the mu-wide performance floor does not outrank the "
                f"budget.  Explicit r_chunk_size overrides this cap")
            r_chunk = capped
        else:
            r_chunk = max(r_chunk, min(n_rtot, r_budget_cap))
        r_chunk = min(r_chunk, n_rtot)
    if r_alignment > 1:
        r_chunk = max(
            r_alignment, r_chunk - r_chunk % r_alignment)
    n_r_chunks = max(1, math.ceil(n_rtot / r_chunk))
    if cache_face_y_blocks:
        face_y_cache_r_tile = bounded_partition_tile(
            r_chunk, cache_cap, r_alignment)
        if (face_y_cache_r_tile <= 0
                or r_chunk // face_y_cache_r_tile >= ns * ns):
            # This can only differ from ``route_width`` on an auto-selected
            # outer extent.  Fail back to the always-valid bounded route; do
            # not pretend a cache transaction exists when its equal static
            # tiles would execute no fewer transforms.
            cache_face_y_blocks = False
            face_y_cache_r_tile = 0

    # ---- gflat_chunk_size (Stage D FFT box) ----------------------------
    fft_per_row = _c128(n_rtot) * 2.0   # accumulate box has no ns axis
    zeta_chunk_D = _c128(nq_disk, mu, r_chunk, shard=p_xy)
    headroom_D = max(target - persistent_total - zeta_chunk_D, 0.0)
    if gflat_chunk_size_override and gflat_chunk_size_override > 0:
        gflat_cs = int(gflat_chunk_size_override)
    else:
        cs = max(_GFLAT_CHUNK_FLOOR, int(headroom_D / max(fft_per_row, 1.0)))
        cs = min(cs, GFLAT_CHUNK_SIZE_CAP)
        gflat_cs = max(_GFLAT_CHUNK_FLOOR, (cs // 4) * 4)

    # ---- q_chunk + ζ solve peak, at the ACTUAL chunk_r -----------------
    # ``replicated`` gathers one full (μ,μ) factor per q in the compute
    # batch.  ``per_q`` structurally gathers exactly one factor and ignores
    # q_chunk.  ``distributed`` keeps the factor 2-D sharded and also bypasses
    # the replicated batch.  ``auto`` is deliberately priced as replicated:
    # the live resolver may narrow it to per_q, but must never make the memory
    # model optimistic.  Z_col and the donated output accumulator are both
    # live across the solve and therefore contribute two full sharded RHS
    # stacks independently of the factor route.
    _solve_route_requested = str(distributed_zeta_solve).strip().lower()
    if _solve_route_requested not in {
            "auto", "replicated", "per_q", "distributed"}:
        raise ValueError(
            "distributed_zeta_solve must be auto, replicated, per_q, or "
            f"distributed; got {distributed_zeta_solve!r}")
    # Face pair accumulation needs only Py alignment; every solve still sees
    # an all-P carrier.  Price that pad here without imposing it on the face
    # kernel or duplicating the runtime channel resolver.
    _solve_r_chunk = ((r_chunk + p_xy - 1) // p_xy) * p_xy
    _rhs_stacks = 2 * _c128(nq, mu, _solve_r_chunk, shard=p_xy)
    # The full-BZ Z_q the solve is handed as a live input.  Defined HERE
    # rather than beside ``C_t`` because ``q_chunk``'s own headroom has to
    # subtract it: see ``_factor_headroom``.
    _zq_live = _c128(nq, mu, _solve_r_chunk, shard=p_xy)
    if _solve_route_requested == "distributed":
        q_chunk = 1                    # ignored by the distributed route
        solve_t = _rhs_stacks
        _solve_memory_route = "distributed (2-D-sharded factor)"
    elif _solve_route_requested == "per_q":
        q_chunk = 1                    # ignored; one q is structural
        # The inner shard_map holds the replicated tile plus its y-gather
        # row, matching isdf.core's live-byte contract.
        solve_t = (_rhs_stacks
                   + _c128(mu, mu) * (1.0 + 1.0 / p_y))
        _solve_memory_route = "per_q (one replicated factor)"
    else:
        _factor_per_q = _c128(mu, mu)
        # SUBTRACT THE LIVE Z_q TOO.  ``C_t`` charges ``solve_t + Z_q``
        # (the reshard cannot alias, so they coexist), but this headroom
        # subtracted only the RHS stacks — so the planner sized q_chunk
        # against a budget it then priced itself over.  MEASURED at the
        # naturally-unreduced MoS2 9x9 geometry (JID 57405800 step
        # lx-Xg1-030359-905761): q_chunk = 70 -> a 31.41 GB replicated
        # factor term, C_t = 55.30 GB against 40.51 GB of C_fit_t, and a
        # plan HWM of 65.12 GB against its own 64.00 GB budget.  Since
        # 77ab293c's sibling made ``gw_init`` REFUSE an infeasible plan,
        # that arithmetic is now a hard stop on a deck the model could
        # have planned by choosing a smaller q_chunk.  This is the ONE
        # knob that shrinks the term the seam charge made binding; the
        # r-chunk dial cannot (the factor is r-independent) and neither
        # can adding ranks (the factor is REPLICATED -- p-independent by
        # construction), which is why the refusal's own advice would not
        # have helped here.
        _factor_headroom = max(
            target - persistent_total - _rhs_stacks - _zq_live, 0.0)
        q_chunk = max(
            1, min(nq, int(_factor_headroom / _factor_per_q)))
        solve_t = _rhs_stacks + q_chunk * _factor_per_q
        _solve_memory_route = (
            "replicated (auto-conservative)"
            if _solve_route_requested == "auto" else
            "replicated")

    # ---- stage transients + per-stage peaks ----------------------------
    # These are different transforms and different peaks.  Centroid sampling
    # sees only ``centroid_k_chunk``; the psi-r cache / streamed face source
    # passes full nk to the incumbent to_rchunk_inner owner.
    A_t = fft_box_A
    A_psi_r_cache_t = fft_box_zeta_transform if cache_psi_r else 0.0
    B_t = (_c128(nq, mu, mu, shard=p_xy)               # C_q
           + 2 * _c128(nk, ns, ns, mu, mu, shard=p_xy))  # full (μ,μ) pair density
    if low_mem_bands:
        if cache_face_y_blocks:
            _cache_tile = face_y_cache_r_tile
            _completed_tile_z = (
                _c128(nq, mu, shard=p_xy) * (r_chunk - _cache_tile))
            C_fit_t = (face_terms["constant"]
                       + face_terms["cache_pair_slope"] * _cache_tile
                       + _completed_tile_z)
            face_y_cache_bytes = (
                face_terms["y_cache_slope"] * _cache_tile)
            # Every tile invokes the SAME canonical face owner at tile width,
            # so the source/all_to_all/all_gather transaction and stacked
            # cache are both tile-sized.  Previously completed compact Z_q
            # tiles stay live while the next tile is built/contracted.
            face_cache_build_t = (
                face_terms["constant"]
                + (0.0 if cache_psi_r else fft_box_zeta_transform)
                + face_terms["cache_build_slope"] * _cache_tile
                + _completed_tile_z)
            if _cache_tile < r_chunk:
                # Global tile arrays are concatenated outside manual
                # shard_map, then constrained once to the incumbent outer
                # P(None,'x','y') carrier.  Conservatively price live tile
                # inputs plus the redistributed output as two all-P Z stacks.
                face_tile_concat_t = 2.0 * _c128(
                    nq, mu, r_chunk, shard=p_xy)
        else:
            C_fit_t = C_constant + C_slope * r_chunk
    else:
        C_fit_t = C_slope * r_chunk
        if not cache_psi_r:
            C_fit_t += fft_box_zeta_transform
    # THE Z_q/SOLVE SEAM IS A SUM, NOT A MAX.  ``fit_one_rchunk`` hands
    # the full-BZ ``Z_q (nq, μ, cr) P(None,'x','y')`` it just built to
    # ``solve_phase`` as a live input: the solve's Z_col reshard targets a
    # DIFFERENT sharding, so donation cannot alias and Z_q coexists with
    # the solve's two RHS stacks.  Whether full q storage is forced
    # (LORRAX_FORCE_FULL_BZ) or the 81-q mesh is naturally unreduced,
    # Z_q is built at the full BZ (z_q_from_psi_sm's contract) — the two
    # measured escapes this seam-charge closes are JID 57269074 step
    # lx-Xg4-005932 (forced 8x8) and JID 57281385 step .28 (natural 9x9).
    C_t = max(C_fit_t, solve_t + _zq_live)
    D_t = zeta_chunk_D + fft_per_row * gflat_cs
    # Stage E (V_q) has its OWN base: L_q + gflat_acc are freed post-fit.
    # Transient = V_acc + the ζ slabs read from disk (+ their single-axis
    # resharded copies).  The RESIDENT ψ term prices the resolved layout
    # (report §7), same split as ``_persistent_bytes``:
    #
    #   low_mem_bands=False (legacy): only ~2 of the 4 centroid copies are
    #   still live at this point (fit_zeta's own psi_rmu_Y/psi_rmuT_X
    #   fit-input copies — the caller has not yet built the four-copy
    #   bundle, which happens after V_q) — HALF of the fit-loop persistent
    #   term: S/Px + S/Py.
    #
    #   low_mem_bands=True (face): the fresh path converts to the two
    #   face copies and DELETES the fit-input copies before V begins (see
    #   gw.gw_init.prepare_isdf_and_wavefunctions), so the SAME 2S/(Px·Py)
    #   that is live for the rest of the run is already what is resident
    #   here — there is no separate "post-fit narrowing" step to price.
    psi_one = _c128(nk, ns, mu, inventory_nb)
    if low_mem_bands:
        E_base = 2.0 * psi_one / p_xy
    else:
        E_base = psi_one / p_x + psi_one / p_y
    # Stage-C/D disclosure (GFlatChunkPlan.stage_cd_psi_bytes docstring).
    # The face value is already present in the separate build/pair peaks.
    #
    # low_mem_bands=False (legacy): the two surviving X-form single-axis
    # copies (mu on 'x', bands replicated) — what is ACTUALLY resident
    # during Stage C/D (fit_one_rchunk / the accumulate step) after
    # fit_zeta_to_h5 frees the Y-form copies right after CCT (2026-08-22
    # fresh-fit low-mem psi contract).
    #
    # low_mem_bands=True (face): the r-chunk-loop port
    # (feat/zeta-fit-rchunk-face-psi-2026-08-22,
    # docs/architecture/zeta_fit_face_psi_cct.md's r-chunk section) means
    # there is no resident single-axis X-form at all any more — the band-
    # contraction operand is read per-band-chunk out of the PERSISTENT
    # psi_mun_fresh (already priced in ``persistent["psi_copies"]`` /
    # ``psi_layout_bytes`` above; NOT double-counted here).  The only
    # INCREMENTAL Stage-C/D cost is (a) one conjugated copy of psi_mun's
    # own local shard, held for the duration of one z_q_from_psi_sm call
    # (``isdf.core._z_q_face``'s ``psi_mun_conj``), and (b) the
    # band_chunk-bounded per-bc gather/weight transient (tiny against (a)
    # whenever band_chunk << n_rmu, the normal case).  Both scale with P
    # (px·py), not sqrt(P) — the fix this term exists to disclose.
    if low_mem_bands:
        if cache_face_y_blocks:
            face_y_route_bytes = face_y_cache_bytes
        else:
            face_y_route_bytes = (
                (_GATHERED_PSI_SLOTS * face_terms["y_block_slope"]
                 + face_terms["y_source_slope"]) * r_chunk)
        stage_cd_psi_bytes = face_terms["constant"] + face_y_route_bytes
    else:
        stage_cd_psi_bytes = 2.0 * psi_one / p_x
    zeta_slab = _c128(n_q_ibz, mu, ngkmax, shard=p_xy)
    E_t = (_c128(n_q_ibz, mu, mu, shard=p_xy)          # V_acc
           + zeta_slab                                  # ζ_L_all
           + (zeta_slab if is_bispinor else 0.0)        # ζ_R_all (TT off-diag)
           + _c128(mu, ngkmax, shard=p_x)               # ζ resharded on 'x'
           + _c128(mu, ngkmax, shard=p_y))              # ζ resharded on 'y'

    # Stage F — the restart-tensor WRITE (isdf_tensors_<n_rmu>.h5).  SlabIO
    # writes per-rank hyperslabs, so this costs the sharded amount. The
    # deleted h5py allgather branch is not a plan this source can execute and
    # therefore is not an option in the live model. Two tensors cross this
    # seam and the binder is the LARGER: V/W0 ``(n_q_ibz, μ, μ)`` and the
    # G-flat ζ ``(n_q_disk, μ, ngkmax)``. Whenever ngkmax > μ, the ζ
    # write wins.
    # Measured: memory-model.md §"Measured corrections behind the G-flat
    # terms" #3 (a 40,594,046,976 B all-gather, matched to the byte).
    _v_tensor = _c128(n_q_ibz, mu, mu, shard=p_xy)
    _gflat_tensor = _c128(nq_disk, mu, ngkmax, shard=p_xy)
    _f_tensor = max(_v_tensor, _gflat_tensor)
    F_t = _f_tensor

    peaks = {
        "A_centroid_load": persistent_total + A_t,
        "B_cct_chol":      persistent_total + B_t,
        "C_fit_one_rchunk": persistent_total + C_t,
        "D_accumulate":    persistent_total + D_t,
        "E_v_q":           E_base + E_t,
        "F_tensor_write":  E_base + F_t,
    }
    if cache_psi_r:
        peaks["A_psi_r_cache_build"] = (
            persistent_total + A_psi_r_cache_t)
    if cache_face_y_blocks:
        peaks["C_face_y_cache_build"] = (
            persistent_total + face_cache_build_t)
        if face_tile_concat_t:
            peaks["C_face_tile_concat"] = (
                persistent_total + face_tile_concat_t)
    bottleneck = max(peaks, key=peaks.get)
    hwm = peaks[bottleneck]

    return GFlatChunkPlan(
        band_chunk=int(band_chunk),
        centroid_k_chunk=int(centroid_k_chunk),
        centroid_fft_rows_local=int(centroid_fft_rows_local),
        centroid_fft_bytes=float(fft_box_A),
        zeta_transform_fft_bytes=float(fft_box_zeta_transform),
        r_chunk=int(r_chunk),
        n_r_chunks=int(n_r_chunks),
        q_chunk=int(q_chunk),
        zeta_solve_memory_route=_solve_memory_route,
        gflat_chunk_size=int(gflat_cs),
        hwm_bytes=float(hwm),
        peak_breakdown=peaks,
        persistent_bytes=float(persistent_total),
        bottleneck=bottleneck,
        p_min=int(p_min),
        budget_bytes=float(budget),
        target_utilization=float(target_utilization),
        psi_layout=("face" if low_mem_bands else "legacy"),
        psi_layout_bytes=float(persistent["psi_copies"]),
        stage_cd_psi_bytes=float(stage_cd_psi_bytes),
        cache_psi_r=bool(cache_psi_r),
        cache_face_y_blocks=bool(cache_face_y_blocks),
        face_y_cache_r_tile=int(face_y_cache_r_tile),
        face_y_cache_bytes=float(face_y_cache_bytes),
    )

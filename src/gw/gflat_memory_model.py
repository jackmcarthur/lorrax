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
    C  fit_one_rchunk   slots·(nk,ns²,μ,cr) + Z_q (nq,μ,cr)   knob: chunk_r  ← binder
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

Bispinor (§1b): the fit loop (A–D) runs the charge channel only — the 3
transverse channels are *exactly parallel* with μ_T ≤ μ_C, so they are
never the binder.  The model carries the spinor factor ``ns² = nspinor²``
in the pair density and does not size the transverse channels separately.

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

Most terms above are closed-form shape algebra.  Stage A compiles the
production FFT helper at the real shape/mesh and queries XLA's buffer peak
plus cuFFT plan workspace through
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
    """Per-rank bytes of the centroid-load FFT box (Stage A).

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
              f"Stage A FFT-box term is the analytic {_FFT_CUFFT_FACTOR}x "
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
    #: INFORMATIONAL ONLY in both cases: this field is NOT folded into
    #: ``peak_breakdown``/``hwm_bytes``/``bottleneck`` (see
    #: KNOWN_LORRAX_ISSUES.md).  Disclosed on the banner so a run can see
    #: the fix's effect without re-deriving it from HLO.
    stage_cd_psi_bytes: float = 0.0
    #: Whether the ζ fit hoists every band/grid coefficient into one
    #: all-P-sharded ψ(r) cache.  ``False`` selects the existing streamed
    #: band-chunk FFT path; the planner uses it only for ``low_mem_bands``
    #: when even the cache's minimum persistent floor exceeds the target.
    cache_psi_r: bool = True

    def format(self) -> str:
        bg = self.budget_bytes / 1e9
        hwm = self.hwm_bytes / 1e9
        lines = [
            "  ISDF memory model — chunk plan + HWM estimate",
            f"    psi layout    = {self.psi_layout} "
            f"({self.psi_layout_bytes / 1e9:.3f} GB/dev, ψ centroid copies)",
            f"    Stage C/D ψ floor (post-CCT, {self.psi_layout} r-chunk "
            f"incremental) = {self.stage_cd_psi_bytes / 1e9:.3f} GB/dev "
            f"[informational; not folded into hwm_bytes]",
            "    ψ(r) source   = " + (
                "hoisted all-band cache" if self.cache_psi_r else
                "streamed band-chunk FFT (low-memory fallback)"),
            f"    band_chunk    = {self.band_chunk}",
            f"    centroid FFT = k_tile {self.centroid_k_chunk}, "
            f"{self.centroid_fft_rows_local} local rows",
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
) -> GFlatChunkPlan:
    """Pick ``(band_chunk, centroid_k_chunk, r_chunk, q_chunk,
    gflat_chunk_size)`` so the
    per-rank HWM lands under ``util·budget``.  Reports the rank floor
    ``P_min`` and the binding stage.

    ``nb_total`` sizes the resident centroid-wavefunction inventory;
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
    fit_nb = int(fit_nb_total if fit_nb_total is not None else nb)
    if fit_nb <= 0:
        raise ValueError(f"fit_nb_total must be positive, got {fit_nb}")
    fft_grid = tuple(getattr(meta, 'fft_grid', None)
                     or (int(round(n_rtot ** (1 / 3))),) * 3)
    if n_q_ibz is None:
        n_q_ibz = nq
    n_q_ibz = int(n_q_ibz)

    if target_utilization is None:
        target_utilization = bfc_fragmentation_target_utilization(ns)
    slots = pair_density_slots if pair_density_slots is not None \
        else _pair_density_slots()

    budget = budget_gb * 1e9
    target = budget * target_utilization

    sys = dict(nk=nk, ns=ns, nq=nq, nq_disk=nq_disk, mu=mu, nb=nb,
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
    cache_psi_r = not (
        low_mem_bands
        and sum(_persistent_base.values()) + _min_cache_bytes > target)
    if not cache_psi_r:
        _announce(
            "stream-psi-r-cache-lowmem",
            "the all-band ψ(r) cache is disabled for low_mem_bands: its "
            f"minimum persistent floor is "
            f"{(sum(_persistent_base.values()) + _min_cache_bytes) / 1e9:.2f} "
            f"GB/dev vs the {target / 1e9:.2f} GB/dev target.  The ζ fit "
            "will use its canonical streamed band-chunk FFT path")

    # ---- Phase 1: the rank floor (un-chunkable ÷P / ÷√P family) ---------
    def _floor_at(pp: int) -> float:
        px, py = _factor_mesh(pp)
        if band_chunk_override and band_chunk_override > 0:
            floor_bc = int(band_chunk_override)
        else:
            floor_bc = fit_nb
        floor_bc = max(pp, ((floor_bc + pp - 1) // pp) * pp)
        fit_padded = max(pp, ((fit_nb + pp - 1) // pp) * pp)
        floor_bc = min(floor_bc, fit_padded)
        cache_slots = math.ceil(fit_nb / floor_bc) * floor_bc
        psi_r_cache = _c128(nk, cache_slots, ns, n_rtot, shard=pp)
        return (sum(_persistent_bytes(p_x=px, p_y=py, **sys).values())
                + (psi_r_cache if cache_psi_r else 0.0))

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

    def _centroid_fft_geometry(bc: int) -> tuple[int, int]:
        """Return ``(k_tile, local_flat_rows)`` for Stage A.

        ``bc`` is a mesh-divisible GLOBAL band tile whose local band extent
        is ``bc / P_band``.  Bound the product of that extent and the outer k
        tile by the already resolved Stage-A band extent ``bc``.  Therefore a
        shipping ``band_chunk=16`` means 16 local FFT rows at both P=1
        (1 k x 16 bands) and P=16 (16 k x 1 band), instead of silently growing
        to ``nk`` rows as band sharding becomes finer.  This reuses the one
        existing memory-policy knob; it is not a second planner or a new
        frontend choice.
        """
        bc = int(bc)
        local_bands = max(1, bc // p_xy)
        k_tile = min(nk, max(1, bc // local_bands))
        return k_tile, k_tile * local_bands

    _fft_box_cache: dict[tuple[int, int], float] = {}

    def _fft_for_bc(bc: int) -> float:
        bc = int(bc)
        k_tile, _ = _centroid_fft_geometry(bc)
        key = (k_tile, bc)
        if key not in _fft_box_cache:
            _fft_box_cache[key] = _fft_box_bytes(
                nk=k_tile, bc=bc, ns=ns, fft_grid=fft_grid,
                mesh_xy=mesh_xy, p_xy=p_xy)
        return _fft_box_cache[key]

    # The full-window decision accounts for both things whose live size the
    # band chunk changes: the ψ(G)->ψ(r) FFT box (Stage A) and Stage C's
    # pair-density accumulator plus gathered ψ slab.  With an explicit
    # r-chunk override, price exactly that extent.  Otherwise use the
    # planner's existing performance floor / max-chunk floor; Phase 2 may
    # still choose a larger affordable r chunk after band K is resolved.
    r_lo = min(mu, n_rtot)
    if r_chunk_override and r_chunk_override > 0:
        r_for_band_guard = min(int(r_chunk_override), n_rtot)
    else:
        r_for_band_guard = max(r_lo, math.ceil(n_rtot / max_chunks))
    if p_xy > 1:
        r_for_band_guard = max(
            p_xy, r_for_band_guard - r_for_band_guard % p_xy)

    def _band_candidate_fits(bc: int) -> bool:
        n_bc = math.ceil(fit_nb / bc)
        # The cache is stacked by uniform transport chunks.  Its band slots
        # are sharded over all P; only the final chunk's at-most-(bc-1) pad
        # rows are extra, and they remain zero-masked.
        psi_r_cache = _c128(
            nk, n_bc * bc, ns, n_rtot, shard=p_xy)
        c_slope = _stage_C_slope(
            nk=nk, ns=ns, nq=nq, mu=mu, slots=slots,
            p_xy=p_xy, band_chunk=bc, p_y=p_y)
        fft_t = _fft_for_bc(bc)
        fit_t = c_slope * r_for_band_guard
        transient = max(fft_t, fit_t) if cache_psi_r else fft_t + fit_t
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

    fft_box_A = _fft_for_bc(band_chunk)
    centroid_k_chunk, centroid_fft_rows_local = _centroid_fft_geometry(
        band_chunk)

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
    # ``band_chunk`` is resolved above — Stage C's ψ(r) slab is sized by it
    # (the gathered band axis), so the two knobs are coupled.
    C_slope = _stage_C_slope(nk=nk, ns=ns, nq=nq, mu=mu, slots=slots,
                             p_xy=p_xy, band_chunk=band_chunk, p_y=p_y)
    # The z_q executable's pair-density temps are ONE contiguous BFC
    # arena of ``slots`` rank-5 carries; its per-cr slope is priced
    # separately because it carries a PLACEMENT bound on top of the sum
    # (see ``_ARENA_PLACEMENT_FRAC``).
    arena_slope = slots * _c128(nk, ns, ns, mu, shard=p_xy)
    # WHY THE SEAM IS *NOT* IN THIS DIAL, stated because it looks like it
    # should be.  ``C_t`` below is ``max(C_fit_t, solve_t + zq_live)`` and
    # the r-LINEAR part of that second member is ``3·(nq, μ, cr)/P``,
    # against a ``C_slope`` whose pair-carry term alone is
    # ``slots·nk·ns²·μ/P`` and which carries its own ``(nq, μ)/P`` Z_q term
    # besides.  For ``nq ≤ nk`` — every real deck, since q runs over the
    # same mesh as k — and ``slots = 3`` the fit slope is strictly larger,
    # so a seam-derived r cap can never bind.  MEASURED on both geometries
    # this model's Stage-C note cites (JID 57405800 step
    # lx-Xg1-030359-905761): seam/fit = 0.199 at MoS2 8x8 and 0.590 at 9x9.
    # An r cap taken from the seam would be a no-op dressed as a guard.
    #
    # What the seam DOES bind through is the r-INDEPENDENT
    # ``q_chunk·(μ,μ)`` replicated factor — see ``_factor_headroom``.
    # On the streamed route the per-band FFT is inside the pair pipeline
    # and coexists with its r-linear carry arena.  Reserve it before sizing
    # r_chunk; otherwise the picker would spend the same headroom twice.
    headroom_C = max(
        target - persistent_total - (0.0 if cache_psi_r else fft_box_A),
        0.0)
    if r_chunk_override and r_chunk_override > 0:
        # The register-documented run-level workaround: an explicit
        # r_chunk_size wins over every cap below, exactly as before.
        r_chunk = min(int(r_chunk_override), n_rtot)
    else:
        r_from_budget = int(headroom_C / C_slope) if C_slope > 0 else n_rtot
        # Placement cap: the single Stage-C arena must fit the contiguous
        # headroom, not just the sum.
        r_from_arena = (int(_ARENA_PLACEMENT_FRAC * headroom_C / arena_slope)
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
            capped = max(p_xy, r_budget_cap)
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
    if p_xy > 1:
        r_chunk = max(p_xy, r_chunk - r_chunk % p_xy)
    n_r_chunks = max(1, math.ceil(n_rtot / r_chunk))

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
    _rhs_stacks = 2 * _c128(nq, mu, r_chunk, shard=p_xy)
    # The full-BZ Z_q the solve is handed as a live input.  Defined HERE
    # rather than beside ``C_t`` because ``q_chunk``'s own headroom has to
    # subtract it: see ``_factor_headroom``.
    _zq_live = _c128(nq, mu, r_chunk, shard=p_xy)
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
    # The hoisted route pays the FFT box while constructing its persistent
    # cache.  The streamed route pays the same box inside the pair pipeline,
    # alongside its r-chunk carries, and has no separate Stage-A allocation.
    A_t = fft_box_A if cache_psi_r else 0.0
    B_t = (_c128(nq, mu, mu, shard=p_xy)               # C_q
           + 2 * _c128(nk, ns, ns, mu, mu, shard=p_xy))  # full (μ,μ) pair density
    C_fit_t = C_slope * r_chunk
    if not cache_psi_r:
        C_fit_t += fft_box_A
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
    psi_one = _c128(nk, ns, mu, nb)
    if low_mem_bands:
        E_base = 2.0 * psi_one / p_xy
    else:
        E_base = psi_one / p_x + psi_one / p_y
    # Informational Stage-C/D disclosure (GFlatChunkPlan.stage_cd_psi_bytes
    # docstring).  Deliberately NOT substituted into
    # ``persistent_total``/``peaks`` — see that docstring for why.
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
        stage_cd_psi_bytes = (
            psi_one / p_xy
            + 2.0 * _c128(nk, ns, max(1, mu // p_x), band_chunk))
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
    bottleneck = max(peaks, key=peaks.get)
    hwm = peaks[bottleneck]

    return GFlatChunkPlan(
        band_chunk=int(band_chunk),
        centroid_k_chunk=int(centroid_k_chunk),
        centroid_fft_rows_local=int(centroid_fft_rows_local),
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
    )

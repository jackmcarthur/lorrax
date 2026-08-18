"""ISDF ζ + V_q memory model — the single production chunk planner.

Design: ``reports/gw_refactor_map_2026-07-01/MEMORY_MODEL_DESIGN.md`` (§1a).
Substrate: ``SHARDING_RULES.md`` (the trichotomy: μ² → all-P, μ×nb → √P,
nb² → replicated).

The whole model is two things summed:

    HWM(cr, bc, P) = persistent(P) + max( A, B, C, D, E )

**persistent(P)** — resident across the entire r-chunk loop (the *floor*):

    L_q          nq·μ²·16 / P            (÷P, μ²  — the rank floor)
    gflat_acc    nq·μ·ngkmax·16 / P      (÷P)
    4·ψ_copies   nk·μ·nb·ns·16, 2 on 'x' + 2 on 'y'  (÷√P — single-axis)
    loader_tbl   nk·n_rtot·4 + nk·ngkmax·16          (REPLICATED, no ÷P)
    ψ(r)_cache   nk·nb_cache_pad·ns·n_rtot·16 / P    (band-flat over all P)

**stage transients** — each stage adds ONE transient on top; they do not
co-exist, so the HWM takes a ``max``, not a sum:

    A  centroid load    fit FFT box (nk, bc, ns, n_rtot)      knob: band_chunk
    B  CCT + Cholesky   C_q + full-(μ,μ) pair density
    C  fit_one_rchunk   slots·(nk,ns²,μ,cr) + Z_q (nq,μ,cr)   knob: chunk_r  ← binder
    D  accumulate       accumulate FFT box (cs, n_rtot)       knob: gflat_chunk_size
    E  V_q per tile     V_acc + resharded ζ slabs   (own base, post-fit)

Two-phase picker (§2):

    Phase 1 — rank floor.  ``persistent(P)`` is un-chunkable; the smallest
              mesh ``P`` with ``persistent(P) ≤ util·budget`` is ``P_min``.
              If the requested ``P < P_min`` → infeasible.
    Phase 2 — dial ``chunk_r`` down from ``n_rtot`` against Stage C's slope
              so ``HWM ≤ util·budget``; report the binding stage.

Bispinor (§1b): the fit loop (A–D) runs the charge channel only — the 3
transverse channels are *exactly parallel* with μ_T ≤ μ_C, so they are
never the binder.  The model carries the spinor factor ``ns² = nspinor²``
in the pair density and does not size the transverse channels separately.

Everything above is closed-form shape algebra.  TWO terms are MEASURED (§6):
the Stage-A/D FFT box (``_fft_box_bytes`` compiles the production FFT helper
at the real shape/mesh and reads XLA's buffer peak *plus* the cuFFT plan
workspace, via ``common.fft_helpers.query_fft_peak_bytes`` ->
``runtime.aot_memory.aot_kernel_peak_bytes``) and the pair-density ``slots``
count (3 GPU / 4 CPU, an XLA BufferAssignment fact).  Where a measurement is
unavailable the model demotes to an analytic bound and ANNOUNCES it from the
rank it happened on — an un-measured term here is a silent OOM later.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Optional

import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

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
                      p_x, p_y) -> dict:
    """The un-chunkable floor resident across the whole r-chunk loop.

    ``L_q`` and ``gflat_acc`` are ÷P (μ²/μ-family); the four ψ centroid
    copies are single-axis ÷√P (2 on 'x', 2 on 'y') — the corrected
    centroid term (design §5 bug #4: NOT ÷p_xy).

    ``loader_tables`` is the WFN loader's REPLICATED per-k metadata (the
    sparse-G→FFT-box index + the τ-phase row), retained for the loader's
    lifetime and **P-INDEPENDENT** — adding nodes never shrinks it, so it
    belongs in the floor.  Measured history: memory-model.md §"Measured
    corrections behind the G-flat terms" #1."""
    P_ = p_x * p_y
    psi_one = _c128(nk, ns, mu, nb)
    return {
        "L_q":         _c128(nq, mu, mu, shard=P_),
        "gflat_acc":   _c128(nq_disk, mu, ngkmax, shard=P_),
        "psi_copies":  2 * psi_one / p_x + 2 * psi_one / p_y,
        "loader_tables": 4.0 * nk * n_rtot + _C128 * nk * ngkmax,
    }


def _fft_box_bytes(*, nk, bc, ns, fft_grid, mesh_xy, p_xy) -> float:
    """Per-rank bytes of the centroid-load FFT box (Stage A / D).

    MEASURED whenever a real ``Mesh`` is available: compiles the production
    FFT helper at this shape/sharding and reads XLA's buffer peak PLUS the
    cuFFT plan workspace, which is not in buffer assignment.  Both halves
    matter — the analytic factor alone under-predicted Si-10³ by 19 GiB
    (design §6), and the cuFFT half is >13.7 GB/rank at the CrI3 V_q box.
    Otherwise: the analytic ``(bc/p_xy)·ns·n_rtot·16·4.0`` box-copy bound,
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
                dtype=jnp.complex128,
            ))
        except Exception as exc:
            why = f"the probe failed to compile ({type(exc).__name__}: {exc})"
    else:
        why = f"mesh_xy is a {type(mesh_xy).__name__}, not a jax Mesh"
    # Analytic fallback (bands sharded over all P; ns + FFT axes replicated).
    _announce(f"fft-box-unmeasured:{why}",
              f"Stage A/D FFT-box term is the analytic {_FFT_CUFFT_FACTOR}x "
              f"box-copy bound because {why}.  It does NOT include cuFFT plan "
              f"workspace, so this planner will UNDER-predict FFT-box stages")
    return _c128(bc, ns, n_rtot, shard=p_xy) * _FFT_CUFFT_FACTOR


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
    r_chunk: int
    n_r_chunks: int
    q_chunk: int
    gflat_chunk_size: int
    hwm_bytes: float
    peak_breakdown: dict          # stage -> total per-rank bytes (persist+transient)
    persistent_bytes: float
    bottleneck: str               # binding stage name
    p_min: int                    # rank floor
    budget_bytes: float
    target_utilization: float

    def format(self) -> str:
        bg = self.budget_bytes / 1e9
        hwm = self.hwm_bytes / 1e9
        lines = [
            "  ISDF memory model — chunk plan + HWM estimate",
            f"    band_chunk    = {self.band_chunk}",
            f"    r_chunk       = {self.r_chunk}  ({self.n_r_chunks} chunks)",
            f"    q_chunk       = {self.q_chunk}",
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

def _default_util(ns: int) -> float:
    """ns²-aware default utilization (§5 divergence #1).

    Stage C's binding transient is a SINGLE ``slots·nk·ns²·μ·cr`` arena;
    at large ``ns²`` (bispinor ns=4 → ns²=16) it approaches the whole
    budget as one contiguous buffer, which the allocator cannot place
    against BFC fragmentation + the card's MEM_FRACTION.  Leave more
    headroom the larger ns² is (validated: MoS2 bispinor ns=4 at 0.85
    OOM'd on a 40 GB card with a 23 GB single arena; 0.78 fits)."""
    if ns >= 4:
        return 0.78      # bispinor (ns²=16): the big single-arena regime
    if ns == 2:
        return 0.85      # spinor / SOC charge (ns²=4)
    return 0.90          # scalar (ns²=1)


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
) -> GFlatChunkPlan:
    """Pick ``(band_chunk, r_chunk, q_chunk, gflat_chunk_size)`` so the
    per-rank HWM lands under ``util·budget``.  Reports the rank floor
    ``P_min`` and the binding stage.

    ``nb_total`` sizes the resident centroid-wavefunction inventory;
    ``fit_nb_total`` is the logical union of the two ζ fit windows and owns
    the band-chunk K extent.  Keeping those extents separate matters when
    ``zeta_nband`` narrows only the fit.

    Overrides (each a ``cohsex.in`` knob; ``>0`` wins over the picker):
    ``r_chunk_override``, ``band_chunk_override``, ``gflat_chunk_size_override``.
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
        target_utilization = _default_util(ns)
    slots = pair_density_slots if pair_density_slots is not None \
        else _pair_density_slots()

    budget = budget_gb * 1e9
    target = budget * target_utilization

    sys = dict(nk=nk, ns=ns, nq=nq, nq_disk=nq_disk, mu=mu, nb=nb,
               ngkmax=ngkmax, n_rtot=n_rtot)

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
                + psi_r_cache)

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

    _fft_box_cache: dict[int, float] = {}

    def _fft_for_bc(bc: int) -> float:
        bc = int(bc)
        if bc not in _fft_box_cache:
            _fft_box_cache[bc] = _fft_box_bytes(
                nk=nk, bc=bc, ns=ns, fft_grid=fft_grid,
                mesh_xy=mesh_xy, p_xy=p_xy)
        return _fft_box_cache[bc]

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
        return (persistent_total + psi_r_cache
                + max(_fft_for_bc(bc), c_slope * r_for_band_guard)
                <= target)

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

    # ψ(r) is hoisted across the outer r-chunk loop.  It is a band-flat
    # all-P-sharded cache, never a replicated full-window object.  Price its
    # uniform final-chunk pad exactly; this becomes part of the persistent
    # floor for all r-chunk stages.
    _cache_n_bc = math.ceil(fit_nb / band_chunk)
    persistent["psi_r_cache"] = _c128(
        nk, _cache_n_bc * band_chunk, ns, n_rtot, shard=p_xy)
    persistent_total = sum(persistent.values())

    # ---- Phase 2: dial chunk_r against Stage C's slope ------------------
    # ``band_chunk`` is resolved above — Stage C's ψ(r) slab is sized by it
    # (the gathered band axis), so the two knobs are coupled.
    C_slope = _stage_C_slope(nk=nk, ns=ns, nq=nq, mu=mu, slots=slots,
                             p_xy=p_xy, band_chunk=band_chunk, p_y=p_y)
    headroom_C = max(target - persistent_total, 0.0)
    if r_chunk_override and r_chunk_override > 0:
        r_chunk = min(int(r_chunk_override), n_rtot)
    else:
        r_from_budget = int(headroom_C / C_slope) if C_slope > 0 else n_rtot
        r_chunk = max(r_lo, min(n_rtot, r_from_budget))
        r_chunk = max(r_chunk, math.ceil(n_rtot / max_chunks))
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

    # ---- q_chunk (ζ solve batch, sized at the ACTUAL chunk_r) ----------
    # Production cuSolverMp 2-D solve has no replicated-L (§5 #6); the
    # per-q buffer is one μ×cr RHS/output slice.  Fold q/k into this
    # planner at the real chunk_r (fixes the legacy cr inconsistency).
    per_q_solve = _c128(mu, r_chunk, shard=p_xy)
    headroom_q = max(target - persistent_total, 0.0)
    q_chunk = max(1, min(nq, int(headroom_q / per_q_solve))) if per_q_solve > 0 else 1

    # ---- stage transients + per-stage peaks ----------------------------
    A_t = fft_box_A
    B_t = (_c128(nq, mu, mu, shard=p_xy)               # C_q
           + 2 * _c128(nk, ns, ns, mu, mu, shard=p_xy))  # full (μ,μ) pair density
    C_t = C_slope * r_chunk
    D_t = zeta_chunk_D + fft_per_row * gflat_cs
    # Stage E (V_q) has its OWN base: L_q + gflat_acc are freed post-fit;
    # only ~2 ψ centroid copies are retained.  Transient = V_acc + the ζ
    # slabs read from disk (+ their single-axis resharded copies).
    psi_one = _c128(nk, ns, mu, nb)
    E_base = psi_one / p_x + psi_one / p_y
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
    # seam and the binder is the LARGER: V/W0 ``(n_q_ibz, μ, μ)`` and
    # the G-flat ζ
    # ``(n_q_disk, μ, ngkmax)`` — whenever ngkmax > μ the ζ write wins.
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
        r_chunk=int(r_chunk),
        n_r_chunks=int(n_r_chunks),
        q_chunk=int(q_chunk),
        gflat_chunk_size=int(gflat_cs),
        hwm_bytes=float(hwm),
        peak_breakdown=peaks,
        persistent_bytes=float(persistent_total),
        bottleneck=bottleneck,
        p_min=int(p_min),
        budget_bytes=float(budget),
        target_utilization=float(target_utilization),
    )

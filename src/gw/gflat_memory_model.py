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

What the model QUERIES rather than guesses (§6): the cuFFT plan scratch
(``query_fft_peak_bytes`` on the unsharded 6-D FFT shape) and the backend
pair-density ``slots`` count (3 GPU / 4 CPU, an XLA BufferAssignment fact).
"""
from __future__ import annotations

import dataclasses
import math
from typing import Optional

import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


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
    except Exception:
        return 3


# cuFFT out-of-place plan holds ~2 box-sized scratch slots on top of the
# in/out boxes.  Used only for the analytic fallback when the FFT box
# cannot be XLA-queried (no real mesh); production queries XLA exactly.
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

def _persistent_bytes(*, nk, ns, nq, nq_disk, mu, nb, ngkmax,
                      p_x, p_y) -> dict:
    """The un-chunkable floor resident across the whole r-chunk loop.

    ``L_q`` and ``gflat_acc`` are ÷P (μ²/μ-family); the four ψ centroid
    copies are single-axis ÷√P (2 on 'x', 2 on 'y') — the corrected
    centroid term (design §5 bug #4: NOT ÷p_xy)."""
    P_ = p_x * p_y
    psi_one = _c128(nk, ns, mu, nb)
    return {
        "L_q":         _c128(nq, mu, mu, shard=P_),
        "gflat_acc":   _c128(nq_disk, mu, ngkmax, shard=P_),
        "psi_copies":  2 * psi_one / p_x + 2 * psi_one / p_y,
    }


def _fft_box_bytes(*, nk, bc, ns, fft_grid, mesh_xy, p_xy) -> float:
    """Per-rank bytes of the centroid-load FFT box (Stage A / D).

    Queries XLA exactly for the cuFFT plan scratch when a real ``Mesh`` is
    available (design §6 — the static factor under-predicted Si-10³ by
    19 GiB); analytic ``(bc/p_xy)·ns·n_rtot·16·factor`` otherwise."""
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
        except Exception:
            pass
    # Analytic fallback (bands sharded over all P; ns + FFT axes replicated).
    return _c128(bc, ns, n_rtot, shard=p_xy) * _FFT_CUFFT_FACTOR


#: Concurrent copies of the band-all_gathered FULL-r ψ(r) slab that XLA
#: keeps live inside ``z_q_from_psi_sm``'s scan body.  ONE is unavoidable
#: (the ``lax.all_gather`` output); the historical second came from the
#: ``jnp.take`` band-compaction, now elided at trace time whenever the
#: permutation is the identity (``isdf/core.py`` ``_y_compact_identity``).
#: Kept at 2 because the elision is config-dependent (a short final band
#: chunk re-enables the take) and an under-estimate here is a hard OOM.
_GATHERED_PSI_SLOTS = 2


def _stage_C_slope(*, nk, ns, nq, mu, slots, p_xy, band_chunk) -> float:
    """Per-``cr`` bytes of the Stage-C transient (the binder): the
    ``slots`` concurrent pair-density accumulators, the Z_q output, and the
    band-all_gathered ψ(r) slab.

    THE ψ(r) SLAB HAS NO ``/P``.  ``z_q_from_psi_sm`` computes each rank's
    1/P band block over the FULL r-chunk, then ``lax.all_gather``s the band
    axis over ('x','y') so every rank holds ``(nk, band_chunk, ns, cr)``
    before it slices its own r_loc (isdf/core.py — the "r-slab strategy"
    comment: the r-slice MUST follow the gather for band/r coherence).
    That object is replicated in bands AND full in r, so it is the only
    Stage-C term that does not shrink with the mesh — and at MoS2 12×12
    (nk=144, band_chunk=160, ns=2, cr=n_rtot=174960) it is 129 GB/rank per
    copy, i.e. 9.4× everything else in this slope combined.  Omitting it is
    what let the planner choose r_chunk = n_rtot and ask XLA for a single
    271 GB allocation (job 7874236 RESOURCE_EXHAUSTED)."""
    return (slots * _c128(nk, ns, ns, mu, shard=p_xy)   # pair carry
            + _c128(nq, mu, shard=p_xy)                 # Z_q / zeta_out
            # band-gathered FULL-r ψ(r) slab — UNSHARDED (see above)
            + _GATHERED_PSI_SLOTS * _c128(nk, band_chunk, ns))


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
    ngkmax: int,
    n_q_disk: int,
    budget_gb: float,
    target_utilization: Optional[float] = None,
    is_bispinor: bool = True,
    max_chunks: int = 64,
    r_chunk_override: int | None = None,
    band_chunk_override: int | None = None,
    gflat_chunk_size_override: int | None = None,
    use_ibz_T: bool = False,
    n_q_ibz: int | None = None,
    pair_density_slots: int | None = None,
    slab_io_replicates: bool = True,
) -> GFlatChunkPlan:
    """Pick ``(band_chunk, r_chunk, q_chunk, gflat_chunk_size)`` so the
    per-rank HWM lands under ``util·budget``.  Reports the rank floor
    ``P_min`` and the binding stage.

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

    sys = dict(nk=nk, ns=ns, nq=nq, nq_disk=nq_disk, mu=mu, nb=nb, ngkmax=ngkmax)

    # ---- Phase 1: the rank floor (un-chunkable ÷P / ÷√P family) ---------
    def _floor_at(pp: int) -> float:
        px, py = _factor_mesh(pp)
        return sum(_persistent_bytes(p_x=px, p_y=py, **sys).values())

    p_min = 1
    while p_min < 1 << 20 and _floor_at(p_min) > target:
        p_min += 1

    persistent = _persistent_bytes(p_x=p_x, p_y=p_y, **sys)
    persistent_total = sum(persistent.values())

    # ---- band_chunk (Stage A FFT box) ----------------------------------
    def _bump_bc(bc: int) -> int:
        bc = int(bc)
        if p_xy > 1 and bc % p_xy != 0:
            bc = ((bc + p_xy - 1) // p_xy) * p_xy
        return min(max(bc, p_xy), max(nb, p_xy))

    if band_chunk_override and band_chunk_override > 0:
        band_chunk = _bump_bc(band_chunk_override)
    else:
        # Largest power-of-2 band_chunk whose FFT box fits half the headroom.
        headroom_A = max(target - persistent_total, 0.0)
        bc = 1
        while bc * 2 <= nb:
            trial = _bump_bc(bc * 2)
            box = _fft_box_bytes(nk=nk, bc=trial, ns=ns, fft_grid=fft_grid,
                                 mesh_xy=mesh_xy, p_xy=p_xy)
            if box > 0.5 * headroom_A:
                break
            bc *= 2
        band_chunk = _bump_bc(bc)

    fft_box_A = _fft_box_bytes(nk=nk, bc=band_chunk, ns=ns, fft_grid=fft_grid,
                               mesh_xy=mesh_xy, p_xy=p_xy)

    # ---- Phase 2: dial chunk_r against Stage C's slope ------------------
    # ``band_chunk`` is resolved above — Stage C's ψ(r) slab is sized by it
    # (the gathered band axis), so the two knobs are coupled.
    C_slope = _stage_C_slope(nk=nk, ns=ns, nq=nq, mu=mu, slots=slots,
                             p_xy=p_xy, band_chunk=band_chunk)
    headroom_C = max(target - persistent_total, 0.0)
    r_lo = min(mu, n_rtot)                      # performance floor (§3)
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

    # Stage F — the restart-tensor WRITE (isdf_tensors_<n_rmu>.h5).
    # On the H5PY_ALLGATHER SlabIO backend (the CPU default whenever the venv
    # lacks mpi4py + h5py-parallel, which is the case on Frontera today)
    # ``_slab_io_allgather._to_host`` process_allgathers the WHOLE tensor onto
    # EVERY rank and then copies it to host numpy — so V_qmunu / W0_qmunu land
    # UNSHARDED, twice.  Nothing in stages A-E models an I/O-seam replication,
    # which is why the planner reported a 6.70 GB HWM for a run that died past
    # ζ-fit.  Scales as μ², so it is the next wall after Stage C:
    #   μ=276  -> 0.47 GB    μ=2412 -> 27 GB    μ=10k  -> 460 GB.
    # ``slab_io_replicates=False`` (PHDF5_FFI / PHDF5_HOST: each rank writes
    # its own hyperslab) drops it to the sharded cost.
    _v_tensor = _c128(n_q_ibz, mu, mu, shard=1 if slab_io_replicates else p_xy)
    F_t = 2.0 * _v_tensor if slab_io_replicates else _v_tensor

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

"""G-flat ζ + V_q memory model — pick chunk sizes that fit the HBM budget.

The sole production planner for ``band_chunk`` / ``r_chunk`` /
``gflat_chunk_size``: it models five per-rank HBM peaks across
:func:`isdf_fitting.fit_zeta_to_h5` and :func:`gw.v_q_g_flat.compute_V_q`,
then picks the largest chunks whose predicted peak stays under
``target_utilization · budget`` and returns a :class:`GFlatChunkPlan`.

Peaks (all sharded on the ('x','y') mesh; ``P = p_x·p_y``):

    A  band-chunked centroid load (pre-loop) — the ψ(G)→ψ(r) IFFT box
       dominates; only the centroid output is persistent.
    B  CCT + Cholesky (pre-loop) — full-(μ,ν) pair density + C_q FFT +
       L_q factor (transient); centroids (L+R) persistent.
    C  fit_one_rchunk (inside the r-chunk loop) — the fused jit holds
       centroids×4 + L_q + gflat_acc + sphere-idx (persistent), the
       P_l/P_r rank-5 (μ, r_chunk) accumulators, their R-space IFFTs, and
       the pre-reshard Z_q.  Usually the binding peak.
    D  accumulate_rchunk_to_gflat — gflat_acc persistent; transient is the
       per-scan-iter zero-padded FFT box ``(cs, n_rtot)``, cs = gflat_chunk_size.
    E  V_q per tile (post fit_zeta) — 7 sequential tiles (CC + 3 TT-diag +
       3 TT-off-diag); TT-off-diagonal binds (two distinct ζ_all buffers).

Modeling constants worth knowing:
  * ``pair_density_slots`` (Peak C): 3 on GPU XLA, 4 on CPU XLA — CPU's
    BufferAssignment schedules one extra concurrent slot for the same
    algebra; resolved from ``jax.default_backend()``.
  * ``factor_D = 2.0`` (Peak D): cuFFT's out-of-place 3D fftn splits its
    scratch across two box-sized slots.
  * ``gflat_acc`` is charged in BOTH Peak C and Peak D persistent bases —
    the two jits have isolated transient slots, so this is not double-counting.

Knobs:
    band_chunk        bc-size for the per-bc ψ(G)→ψ(r-chunk) IFFT; primary
                      lever on Peaks A and C.  Divides nb when possible.
    r_chunk           outer r-axis chunk count; lower-bounded by ``n_rmu``
                      (the Σ_μν output is ``n_rmu²·n_q·16`` B, so <n_rmu work
                      per chunk is wasted overhead), upper-bounded by ``max_chunks``.
    gflat_chunk_size  scan chunk inside accumulate; **capped at 100** — past
                      cs~1000 cuFFT switches plan algorithm and workspace grows
                      non-linearly (cs=1414 OOM'd at production CrI3 80Ry).
"""
from __future__ import annotations

import dataclasses
import math
from typing import Optional

import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


_BYTES_PER_C128 = 16
_BYTES_PER_I32 = 4


def _bytes_c128(*dims, shard: int = 1) -> float:
    """Per-rank c128 byte count for a tensor of ``dims`` sharded over
    ``shard`` ranks (1 = replicated)."""
    n = 1
    for d in dims:
        n *= int(d)
    return _BYTES_PER_C128 * n / max(int(shard), 1)


def _bytes_i32(*dims, shard: int = 1) -> float:
    """Per-rank int32 byte count for a tensor of ``dims`` sharded over
    ``shard`` ranks (1 = replicated)."""
    n = 1
    for d in dims:
        n *= int(d)
    return _BYTES_PER_I32 * n / max(int(shard), 1)


def _default_pair_density_slots() -> int:
    """Default ``pair_density_slots`` for the current XLA backend.

    GPU XLA's BufferAssignment schedules 3 concurrent pair-density slots
    in ``fit_one_rchunk`` (HLO-calibrated on CrI3 80Ry bispinor + Si
    4×4×4; see ``reports/memory_model_refit_2026-05-17/
    agent_d_hlo_calibration.md``).  CPU XLA schedules 4 — one extra
    concurrent live slot — at the same algebraic structure (verified at
    Si μ=384 non-bispinor n=1/2/4 and bispinor charge+transverse channels
    on 2×2 mesh; see ``reports/memory_model_nonbispinor_kgrid_2026-05-18/
    CPU_OVERHEAD_DECOMP_2026-05-20.md``).  Per-slot bytes match the
    planner's ``_bytes_c128(nk, ns², mu, r_chunk, /p_xy)`` formula
    bit-exactly on both backends; only the slot count differs.

    The FFT-scratch alternative was tested: band_chunk ∈ {32, 64, 120}
    leaves the slot count and per-slot bytes unchanged on CPU.  The FFT
    box shapes alias INTO the pair-density slots but do not size them.

    Resolves at function-call time via ``jax.default_backend()`` so the
    caller does not need to thread the backend through.  Falls back to 3
    (the original GPU-only value) if jax is unimportable.
    """
    try:
        import jax
        return 4 if jax.default_backend() == "cpu" else 3
    except Exception:
        return 3


def _round_pow2_down(n: int) -> int:
    """Largest power of 2 ≤ n, but ≥ 1."""
    if n <= 1:
        return 1
    return 1 << (int(n).bit_length() - 1)


def _largest_divisor_le(n: int, cap: int) -> int:
    """Largest divisor of ``n`` that is ≤ ``cap``.  Falls back to ``cap``."""
    cap = max(1, int(cap))
    if cap >= n:
        return n
    for c in range(cap, 0, -1):
        if n % c == 0:
            return c
    return cap


# ---------------------------------------------------------------------------
# Module-level constants pinned to live_arrays / HLO probes
# ---------------------------------------------------------------------------
# Cap on ``gflat_chunk_size``.  Empirically derived from the 2026-05-17
# cs=1414 OOM (agent_f_live_arrays_probe.md): past cs ~ 1000 cuFFT's
# out-of-place 3D plan switches algorithm and workspace grows non-linearly
# (the planner predicted ~56 GB/rank but cuFFT's plan-creation asked for
# an additional 24 GB of scratch that XLA doesn't see).  At cs ≤ 100 the
# FFT-box per rank is bounded at 100 × n_rtot × 16 ≈ 1.8 GB and cuFFT
# stays in its "factor-2" algorithm regime that agent_d_hlo_calibration
# M2 verified across cs=1, cs=360.  Per-iter FFT cost is dominated by
# the spatial FFT shape, not the batch — capping at 100 has no
# measurable performance impact (agent_d M3: cs=1 finished in 24s/r-chunk
# vs cs=360's 21s/r-chunk).
GFLAT_CHUNK_SIZE_CAP = 100

# Replicated sphere/phase index buffer count.
#
# Pre-Round-4 (commit d1fcd20 + 94542c2): every fresh psi_G_store
# instance device_put'd its own ``(nk, nx, ny, nz) int32`` g_index
# buffer, and every fresh ``gflat_to_rmu`` build() closure baked its
# own.  agent_h_full_lifecycle.md §3 Finding 3 measured 2→3→6→7→8
# buffers across the bispinor 4-channel pipeline, costing ~1.3 GB/rank
# of replicated waste by V_q time.
#
# Round-4 (commits d1fcd20 + 94542c2): two per-source dedups landed —
# ``WfnLoader.box_index_dev`` for psi_G_store + ``_cached_gindex_dev``
# (content-hash) for wfn_transforms closures.  Each bounded growth
# WITHIN its source (no monotonic accumulation across bispinor
# channels) but did NOT collapse the loader-side and wfn_transforms-
# side buffers — different sharding (NamedSharding-replicated vs
# SingleDeviceSharding) ⇒ different device allocations from the same
# underlying ``WfnLoader.box_index(k)`` numpy bytes.  Round-5 live
# verify (agent_l_round5_liveverify.md §2) measured a steady-state of
# 3 buffers, not 1.
#
# Sphere-idx buffer count: one per rank — a single WfnLoader-cached
# canonical allocation shared across the charge + all 3 transverse
# bispinor channels (no per-channel growth).
N_SPHERE_IDX_BUFFERS = 1


@dataclasses.dataclass
class GFlatChunkPlan:
    """Resolved chunk sizes + per-rank HBM high-water estimate."""
    band_chunk: int
    r_chunk: int
    n_r_chunks: int
    gflat_chunk_size: Optional[int]   # always int after the 2026-05-17 cap
    hwm_bytes: float
    peak_breakdown: dict              # name -> bytes  (A/B/C/D/E totals)
    peak_components: dict             # full per-term breakdown
    bottleneck: str                   # name of binding peak
    budget_bytes: float

    def format(self) -> str:
        bg = self.budget_bytes / 1e9
        hwm = self.hwm_bytes / 1e9
        lines = [
            f"  G-flat memory model — chunk plan + HWM estimate",
            f"    band_chunk         = {self.band_chunk}",
            f"    r_chunk            = {self.r_chunk}  ({self.n_r_chunks} chunks)",
            f"    gflat_chunk_size   = {self.gflat_chunk_size}",
            f"    budget             = {bg:.2f} GB/dev",
            f"    HWM estimate       = {hwm:.2f} GB/dev "
            f"({100 * hwm / max(bg, 1e-9):.0f}% of budget) "
            f"[bottleneck: {self.bottleneck}]",
            f"    peak totals (GB/dev):",
        ]
        for name, b in sorted(self.peak_breakdown.items(),
                              key=lambda kv: -kv[1]):
            lines.append(f"      {name:.<24s} {b/1e9:>7.2f}")
        # Per-peak component breakdown.  Self-explaining: shows exactly
        # which term drives each peak (so users can target the actual
        # binding term rather than guessing).
        lines.append(f"    per-peak components (GB/dev):")
        # Group components by peak letter (the prefix before the dot
        # in the key) and emit in A→E order so the format is stable.
        groups: dict[str, list[tuple[str, float]]] = {}
        for k, v in self.peak_components.items():
            if "." in k:
                peak_letter, term = k.split(".", 1)
            else:
                peak_letter, term = "_misc", k
            groups.setdefault(peak_letter, []).append((term, v))
        for peak_letter in sorted(groups):
            lines.append(f"      [{peak_letter}]")
            for term, v in sorted(groups[peak_letter], key=lambda kv: -kv[1]):
                lines.append(f"        {term:.<22s} {v/1e9:>7.3f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sphere-index replicated leak — helper used by Peaks A/B/C/D/E
# ---------------------------------------------------------------------------
# REPLICATED, NOT sharded.  ``make_flat_k_fft`` and the centroid-load
# path build ``g_index[k, nx, ny, nz] int32`` (see
# common/gvec_fft_box.py:55) once per FFT helper instance.  The same
# helper is constructed inside each ``fit_zeta_to_h5`` call (charge +
# 3 transverse), so a new copy lands on every rank per channel and is
# not freed by ``gc.collect() + jax.clear_caches()`` (the helper holds
# a strong ref via a module-level cache).
#
# live_arrays signature (LORRAX_MEM_DEBUG=1):
#   ``int32 (nk, nx, ny, nz)`` — for CrI3 80Ry that's (36, 75, 75, 200)
#   ≈ 0.16 GB/rank PER BUFFER (replicated).  By μ_L=3 there are 8
#   buffers ≈ 1.30 GB/rank.
#
# Source-of-truth allocation site: ``common/gvec_fft_box.py:55`` —
# ``g_index = np.full((nk, nx, ny, nz), ngkmax, dtype=np.int32)``.
# The leak is "cross_jit_leaked" in agent_g_census's terminology;
# see agent_h_full_lifecycle.md §3 Finding 3 for the per-channel growth.
def _sphere_idx_replicated_bytes(*, nq, fft_grid, n_buffers) -> float:
    """Per-rank bytes of the replicated sphere/phase-index buffers.

    NOT divided by p_xy — these are replicated on every rank.
    """
    nx, ny, nz = (int(v) for v in fft_grid)
    return n_buffers * _bytes_i32(nq, nx, ny, nz)


# ---------------------------------------------------------------------------
# Per-peak cost functions
# ---------------------------------------------------------------------------
# Each returns a dict[str, float] of per-rank byte contributions; the peak
# is sum-over-dict.  Keeping the per-term breakdown surfaced lets the
# log explain WHY each peak is what it is.

def _peak_A_centroid_load(*, nk, ns, n_rtot, nb_per_load, band_chunk,
                          mu, p, p_xy, fft_box_factor_A,
                          nq, fft_grid, n_sphere_buffers) -> dict:
    """Pre-loop ψ(G) → r-space → centroid sample.  Runs once per channel
    (charge + 3 transverse on bispinor).

    FFT-box formula audited 2026-05-17 (Agent A audit, Bug #1).  The
    actual ``gflat_to_rmu._kernel`` (wfn_transforms.py:611-851) batches
    on a flat ``(nk · nb_local)`` axis INSIDE a shard_map.  Its
    per-iter FFT box is ``c128[cs, ns, nx, ny, nz]`` where ``cs`` is
    the planner-picked chunk size — **per-rank already**, no nk factor.
    Use ``band_chunk`` as the proxy for ``cs``.
    """
    # live_arrays signature: c128 (nk, ns, mu, nb_per_load) — sharded /p_xy
    #   lifetime class: persistent_throughout_zeta_fit (centroid output)
    #   source: common/wfn_transforms.py (gflat_to_rmu fills psi_rmu_Y/X)
    # live_arrays signature: int32 (nk, nx, ny, nz) × N replicated
    #   lifetime class: persistent_throughout_zeta_fit (REPLICATED leak)
    #   source: common/gvec_fft_box.py:55 (build_g_index_for_fft_box)
    out = {
        "centroid_out_filling":
            _bytes_c128(nk, ns, mu, nb_per_load, shard=p),
        "phase_table":
            _bytes_c128(nk, n_rtot),
        "fft_box":
            _bytes_c128(band_chunk, ns, n_rtot) * fft_box_factor_A,
        "sphere_idx_replicated":
            _sphere_idx_replicated_bytes(
                nq=nq, fft_grid=fft_grid, n_buffers=n_sphere_buffers),
    }
    return {f"A.{k}": v for k, v in out.items()}


def _peak_B_cct_chol(*, nk, ns, nq, mu, nb_total, p, p_xy,
                     fft_grid, n_sphere_buffers) -> dict:
    """CCT/Cholesky pre-loop on (μ, ν) full-grid.

    Persistent centroids term now counts the 4 physical buffers per channel
    (rmuT_X + rmu_Y transpose, for both ψ_l and ψ_r).  agent_a_audit
    finding #1 + agent_g_census §6 row #1 confirmed via ``live_arrays()``:
    each centroid array appears twice (rmuT_X form + transposed rmu_Y
    form) — total 4 buffers per channel.
    """
    # live_arrays signature: c128 (nk, ns, mu, nb_total) ×4 — sharded /p_xy
    #   lifetime class: persistent_throughout_zeta_fit (4 physical buffers:
    #                   ψ_l_rmuT_X, ψ_l_rmu_Y, ψ_r_rmuT_X, ψ_r_rmu_Y)
    #   source: common/wfn_transforms.py (gflat_to_rmu) + transpose copy
    #           lives at gw/isdf_fitting.py:fit_zeta_to_h5 step 1
    #           (slice/divide-by-norms doubles each into a Y-form view)
    out = {
        "centroids_persistent":  # ×4 (L+R, both rmuT_X and rmu_Y)
            4 * _bytes_c128(nk, ns, mu, nb_total, shard=p_xy),
        "P_l_plus_P_r_open_spin":
            2 * _bytes_c128(nk, ns, ns, mu, mu, shard=p_xy),
        "C_q":
            _bytes_c128(nq, mu, mu, shard=p_xy),
        "L_q":
            _bytes_c128(nq, mu, mu, shard=p_xy),
        # Replicated sphere/phase indices baked into the FFT helper closures.
        # Lives on every rank — see _sphere_idx_replicated_bytes docstring.
        "sphere_idx_replicated":
            _sphere_idx_replicated_bytes(
                nq=nq, fft_grid=fft_grid, n_buffers=n_sphere_buffers),
    }
    return {f"B.{k}": v for k, v in out.items()}


def _peak_C_fit_one_rchunk(*, nk, ns, nq, nq_disk, mu, ngkmax, n_rtot, r_chunk,
                           band_chunk, n_bc, nb_total, p, p_x, p_y, p_xy,
                           fft_box_factor, pair_density_slots,
                           is_charge_channel, fft_grid,
                           n_sphere_buffers) -> dict:
    """Per-r-chunk fit_one_rchunk fused-jit peak.

    bc-loop is now ``lax.scan(..., unroll=1)`` (Round-6, commit f567aa0)
    so n_bc no longer multiplies the FFT-box term — XLA aliases the
    per-iter FFT box into a single slot across scan iterations.

    ``pair_density_slots`` is the **XLA-BufferAssignment-determined**
    count of concurrent rank-5
    ``c128[nk, ns², n_rmu_local, r_chunk_local]`` tensors live at peak
    inside the monolithic ``c_q_from_psi_sm`` / ``z_q_from_psi_sm``
    shard_map.  HLO-verified on:
      * CrI3 80Ry bispinor 4×4 mesh (agent_d_hlo_calibration M1: 3 slots)
      * Si 4×4×4 80Ry single-device (agent_a_audit pair_density_slots: 3)
      * MoS2 3×3 bispinor 2×2 mesh (original calibration: 3 slots)
    """
    # live_arrays signature: c128 (nk, ns, mu, nb_total) ×4 — sharded /p_xy
    #   lifetime class: persistent_throughout_zeta_fit
    #   source: see _peak_B_cct_chol comment; same 4-buffer set
    # live_arrays signature: c128 (nq, mu, mu) — sharded /p_xy
    #   lifetime class: persistent_throughout_zeta_fit (L_q Cholesky factor)
    #   source: isdf/core.py:factor_c_q (step 3 of fit_zeta_to_h5)
    # gflat_acc is RESIDENT during fit_one_rchunk (separate jit from
    # accumulate; the two jits each have isolated transient slots, so
    # counting gflat_acc in BOTH Peak C persistent and Peak D persistent
    # is correct — they don't double-count.  Verified by live_arrays
    # census in agent_o_y3_95.out (Round-9b agent_q breakdown).
    # live_arrays signature: c128(36, 1520, 59990) on every rank, sharded p_xy.
    persistent = {
        "centroids_persist":  # ×4 (L+R rmuT_X + rmu_Y transpose)
            4 * _bytes_c128(nk, ns, mu, nb_total, shard=p_xy),
        "L_q":
            _bytes_c128(nq, mu, mu, shard=p_xy),
        "gflat_acc":
            _bytes_c128(nq_disk, mu, ngkmax, shard=p_xy),
        "sphere_idx_replicated":
            _sphere_idx_replicated_bytes(
                nq=nq, fft_grid=fft_grid, n_buffers=n_sphere_buffers),
    }
    # live_arrays signature: c128 (nk, ns, ns, mu_local, r_loc) ×3 slots
    #   lifetime class: accumulate_transient (XLA-internal scratch slots,
    #                   aliased to P_l_R_conj / P_r_R / FFT box across
    #                   non-overlapping lifetimes)
    #   source: isdf/core.py (fit_one_rchunk (P_l_acc/P_r_acc, scan
    #           init) + isdf/core.py (P_l_R_conj reshape)
    # The dominant transient is ``pair_density_slots`` rank-5 buffers;
    # psi_bc_Y, the FFT box, Z_q etc. all fit in the SAME lifetime slots
    # (XLA's allocator reuses them when lifetimes don't overlap — verified
    # in agent_d M1 where slot 1 holds both a P_pair and the bispinor
    # FFT box across non-overlapping lifetimes).
    slots = pair_density_slots
    transient = {
        "P_pair_concurrent_slots":
            slots * _bytes_c128(nk, ns, ns, mu, r_chunk, shard=p_xy),
        "zeta_out":
            _bytes_c128(nq, mu, r_chunk, shard=p),
    }
    out = {f"C.{k}": v for k, v in persistent.items() if v > 0}
    out.update({f"C.{k}": v for k, v in transient.items()})
    return out


def _peak_D_accumulate(*, nk, ns, nq, nb_total, nq_disk, mu, n_rtot, ngkmax,
                       r_chunk, gflat_chunk_size, p, p_xy,
                       fft_box_factor_D, fft_grid, n_sphere_buffers) -> dict:
    """accumulate_rchunk_to_gflat peak — runs after fit_one_rchunk
    returns (its P_l/P_r are freed); ζ_chunk is the only fit_one_rchunk
    output still live.

    Uses a per-peak ``fft_box_factor_D`` (default 2.0, vs Peak A's 4.0).
    The accumulate kernel's FFT (wfn_transforms.py:1057-1107) is shape
    ``c128[cs, nx, ny, nz]`` (no ns axis — ζ is spin-traced upstream).
    HLO calibration at agent_d_hlo_calibration M2 confirmed factor_D=2.0
    on cs=1 + cs=360 production runs (CrI3 6×6 80Ry bispinor).

    """
    # live_arrays signature: c128 (nk, ns, mu, nb_total) ×4 — sharded /p_xy
    #   lifetime class: persistent_throughout_zeta_fit
    #   source: see _peak_B_cct_chol comment; same 4-buffer centroid set
    # live_arrays signature: c128 (nq_disk, mu, ngkmax) — sharded /p_xy
    #   lifetime class: persistent_throughout_zeta_fit (gflat_acc, lives
    #                   for the full r-chunk loop; donated in-place each
    #                   accumulate call)
    #   source: isdf/core.py (jit(zeros) just before chunk
    #           loop) — live_arrays-verified at probe 1A in agent_f
    persistent = {
        "centroids_persist":  # ×4 (L+R rmuT_X + rmu_Y transpose)
            4 * _bytes_c128(nk, ns, mu, nb_total, shard=p_xy),
        "L_q":
            _bytes_c128(nq, mu, mu, shard=p_xy),
        "gflat_acc":
            _bytes_c128(nq_disk, mu, ngkmax, shard=p_xy),
        "sphere_idx_replicated":
            _sphere_idx_replicated_bytes(
                nq=nq, fft_grid=fft_grid, n_buffers=n_sphere_buffers),
    }
    # live_arrays signature: c128 (nq_disk, mu, r_chunk) — sharded /p_xy
    #   lifetime class: fit_one_rchunk_transient (alive at after_fit, gone
    #                   at after_accumulate via donate_argnums=(1,))
    #   source: isdf/core.py:fit_one_rchunk return path
    # accumulate_fft_box live_arrays signature:
    #   c128 (cs, nx, ny, nz) — XLA-internal preallocated-temp; invisible
    #   to live_arrays() (lives inside the jit's scratch pool).  HLO-
    #   verified in agent_d_hlo_calibration M2 module_0474 + M3 module_0363
    #   (2 box-sized slots × factor_D=2.0).
    fft_box_bytes = _bytes_c128(gflat_chunk_size, n_rtot) * fft_box_factor_D
    transient = {
        "zeta_chunk":
            _bytes_c128(nq_disk, mu, r_chunk, shard=p_xy),
        "accumulate_fft_box":
            fft_box_bytes,
        # Note: cuFFT-internal workspace pointer scratch is NOT separate
        # at the small cs values we now pick (≤100).  XLA's planner folds
        # it into the 2 box-sized slots (HLO-verified, agent_d M2).
    }
    out = {f"D.{k}": v for k, v in persistent.items()}
    out.update({f"D.{k}": v for k, v in transient.items()})
    return out


# ---------------------------------------------------------------------------
# Peak E — V_q (per-tile and bispinor mix)
# ---------------------------------------------------------------------------

def _peak_E_v_q_per_tile_transient(*, n_q_ibz, mu_L, mu_R, ngkmax,
                                   p_x, p_y, p_xy,
                                   same_zeta: bool,
                                   write_g0: bool,
                                   fft_grid,
                                   n_sphere_buffers,
                                   nb_total, nk, ns, mu) -> dict:
    """Per-tile peak inside ``_compute_V_q_g_flat_one_tile``'s per-q kernel.

    Source (per agent_i_v_q_memory.md §2): ``gw/v_q_g_flat.py:271-465``.
    The binding term is the IBZ ζ̃ slab plus its two ``P(x,⋅)/P(y,⋅)``
    resharded copies inside the kernel (each replicated on the *other*
    mesh axis).  CC + TT-diagonal share zeta_L_all (``same_zeta=True``);
    TT-off-diagonal allocates a distinct zeta_R_all and is the dominant
    peak at full BZ.

    Persistent caller-scope from fit_zeta lifetime (per agent_h §4):
    ψ_r centroids are retained via ``psi_rmu_Y`` / ``transverse_wfn_data``
    closures during V_q.  Live globally during V_q; sized at the smaller
    band count (nb_r ≈ 160 — but the planner uses nb_total which over-
    counts slightly; conservative bias preferred).
    """
    # live_arrays signature: c128 (n_q_ibz, mu_L, ngkmax) — sharded /p_xy
    #   lifetime class: v_q_per_tile_transient (allocated pre-loop in
    #                   _compute_V_q_g_flat_one_tile, freed at line 437)
    #   source: gw/v_q_g_flat.py:372-384 — pre-loop slab pre-read
    zeta_L_slab = _bytes_c128(n_q_ibz, mu_L, ngkmax, shard=p_xy)
    # live_arrays signature: c128 (mu_L_pad, ngkmax) — sharded /p_x only
    #   (resharded inside per-q kernel, REPLICATED on the y axis)
    #   lifetime class: v_q_per_tile_transient (one slice per q iter,
    #                   aliased across q-loop iterations)
    #   source: gw/v_q_g_flat.py:_make_per_q_kernel.fn (resharded to
    #           P('x', None))
    zeta_L_on_x = _bytes_c128(mu_L, ngkmax, shard=p_x)
    zeta_R_on_y = _bytes_c128(mu_R, ngkmax, shard=p_y)
    out = {
        # live_arrays signature: c128 (n_q_ibz, mu_L, mu_R) — sharded /p_xy
        # lifetime class: v_q_per_tile_transient (V_acc, donated in-place;
        # final post-unfold output piggybacks the same buffer slot)
        # source: gw/v_q_g_flat.py:372 (V_acc init)
        "V_acc": _bytes_c128(n_q_ibz, mu_L, mu_R, shard=p_xy),
        # live_arrays signature: c128 (n_q_ibz, ngkmax) — REPLICATED
        # lifetime class: v_q_per_tile_transient (every rank holds full
        # table; tiny in absolute terms)
        # source: gw/v_q_g_flat.py:372-384 (v_q_dev table, P(None, None))
        "v_q_table_replicated": _bytes_c128(n_q_ibz, ngkmax),
        "zeta_L_all": zeta_L_slab,
        "zeta_R_all": (0.0 if same_zeta
                       else _bytes_c128(n_q_ibz, mu_R, ngkmax, shard=p_xy)),
        # Resharded ζ copies inside the per-q kernel — each replicated
        # on the OTHER mesh axis, so divides by p_x (resp. p_y), not p_xy.
        "zeta_L_on_x_axis": zeta_L_on_x,
        "zeta_R_on_y_axis": zeta_R_on_y,
        "V_q_block": _bytes_c128(mu_L, mu_R, shard=p_xy),
        "g0_acc": (_bytes_c128(n_q_ibz, mu_L, shard=p_x)
                   if write_g0 else 0.0),
        # Caller-scope persistent state during V_q (ψ_r centroids
        # retained from fit_zeta close).  Per agent_i §5: 2 × ψ shapes
        # ≈ 2 GB/rank on CrI3 80Ry.  Using nb_total over-counts slightly
        # (fit_zeta keeps only the right-band slab) but conservative.
        # live_arrays signature: c128 (nk, ns, mu, nb_total) ×2 — /p_xy
        # lifetime class: persistent_throughout_v_q (psi_rmu_Y +
        # transverse_wfn_data closures hold these for all 7 tiles)
        # source: gw/gw_init.py:_orchestrate_isdf_compute (caller scope)
        "psi_centroids_persistent": 2 * _bytes_c128(
            nk, ns, mu, nb_total, shard=p_xy),
        # Replicated sphere-idx leak persists into V_q (agent_h §4).
        "sphere_idx_replicated":
            _sphere_idx_replicated_bytes(
                nq=nk, fft_grid=fft_grid, n_buffers=n_sphere_buffers),
    }
    return {f"E.{k}": v for k, v in out.items() if v > 0}


def _peak_E_v_q_unfold(*, n_q_full, mu_L, mu_R, p_xy) -> dict:
    """Post-loop ``unfold_v_q`` output (centroid double-permute + L-phase).

    Runs after ``zeta_L_all`` is ``del``'d (per agent_i §3); the unfolded
    V_acc never coexists with the per-tile slab.  Output is the full-BZ
    V_acc only.
    """
    # live_arrays signature: c128 (n_q_full, mu_L, mu_R) — sharded /p_xy
    # lifetime class: v_q_per_tile_transient (post-unfold output;
    # aliased into the same V_acc buffer slot via in-place
    # all_to_all collective)
    # source: common/symmetry_maps.py:unfold_v_q (line 392-470)
    return {
        "E.V_acc_full_BZ":
            _bytes_c128(n_q_full, mu_L, mu_R, shard=p_xy),
    }


def _peak_E_v_q_bispinor_buffer(*, n_q_full, mu_T, p_xy,
                                use_ibz_T: bool) -> dict:
    """Bispinor Lorentz-mix transient (only when the transverse IBZ
    cascade is active — otherwise tiles stream straight to disk and this
    is zero).

    At peak: ``tt_full_in`` (9 tiles: 3 unique upper + 3 Hermitian-mirror
    + 3 diag) + ``tt_mixed`` (6 unique outputs) coexist for one mix call.
    Per agent_i §4.

    Source: ``gw/v_q_bispinor.py:587-728`` (orchestrator's mix branch).
    """
    if not use_ibz_T:
        return {"E.lorentz_mix_buffer": 0.0}
    # live_arrays signature: c128 (n_q_full, mu_T, mu_T) ×{9,6} /p_xy
    # lifetime class: v_q_bispinor_lorentz_transient
    # source: gw/v_q_bispinor.py:unfold_v_q_bispinor_lorentz call
    tile = _bytes_c128(n_q_full, mu_T, mu_T, shard=p_xy)
    return {
        "E.tt_full_in_9_tiles": 9 * tile,
        "E.tt_mixed_6_tiles":   6 * tile,
    }


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

def plan_gflat_chunks(
    *,
    meta,
    mesh_xy: Mesh,
    nb_total: int,
    ngkmax: int,
    n_q_disk: int,
    budget_gb: float,
    target_utilization: float = 0.94,
    fft_box_factor_A: float = 4.0,
    fft_box_factor_D: float = 2.0,
    pair_density_slots_transverse: int | None = None,
    pair_density_slots_charge: int | None = None,
    is_bispinor: bool = True,
    max_chunks: int = 64,
    r_chunk_override: int | None = None,
    band_chunk_override: int | None = None,
    gflat_chunk_size_override: int | None = None,
    use_ibz_T: bool = False,
    n_q_ibz: int | None = None,
) -> GFlatChunkPlan:
    """Pick (band_chunk, r_chunk, gflat_chunk_size) to land near
    ``target_utilization · budget_gb`` per device.

    Algorithm (deterministic, no iterative search; r-first picker):

      1. Compute persistent footprint (centroids ×4 + L_q + gflat_acc +
         sphere-idx replicated leak) and the shared headroom.
      2. Pick ``r_chunk`` first against Peak C's transient slope.
      3. Pick ``gflat_chunk_size`` against Peak D's transient — capped
         at ``GFLAT_CHUNK_SIZE_CAP=100`` (cuFFT plan-algorithm
         crossover at cs ~ 1000; see module-level constant).
      4. Pick ``band_chunk`` against Peak A's FFT-box transient.

    User-overrideable kwargs (each maps to a ``cohsex.in`` knob; the
    cohsex value, when > 0, wins over the picker; per agent_j §5 the
    planner was previously blind to ``gflat_chunk_size`` because the
    override was applied AFTER ``plan_gflat_chunks`` returned —
    threading it as a kwarg lets the planner recompute Peak D at the
    overridden cs and warn if it exceeds the cap):

      * ``r_chunk_override`` ← cohsex.in ``r_chunk_size`` (0 = picker)
      * ``band_chunk_override`` ← cohsex.in ``band_chunk_size`` (0 = picker)
      * ``gflat_chunk_size_override`` ← cohsex.in ``gflat_chunk_size``
        (0 = picker; non-zero wins over the GFLAT_CHUNK_SIZE_CAP=100
        safety cap, with a printed warning so the user knows the
        runtime may OOM)
      * ``target_utilization`` ← cohsex.in ``chunk_target_utilization``
        (default 0.94 here; cohsex.in default 0.97)
      * ``budget_gb`` ← cohsex.in ``memory_per_device_gb`` (0 ⇒ GPU
        auto-detect happens upstream in MemoryConfig)

    Non-user-overrideable kwargs (set by the caller, not the user):

      * ``is_bispinor`` ← ``cfg.bispinor`` (affects the sphere-idx
        leak count and pair_density_slots branch)
      * ``use_ibz_T`` ← derived from sym + centroid orbit closure
      * ``n_q_ibz`` ← IBZ q-count (defaults to full BZ for conservative
        Peak E)

    Returns a :class:`GFlatChunkPlan` with HWM = max of {A, B, C, D, E}.
    """
    p_x = int(mesh_xy.shape['x'])
    p_y = int(mesh_xy.shape['y'])
    p_xy = p_x * p_y
    p = p_xy
    nk = int(meta.nk_tot)
    ns = int(meta.nspinor)
    mu = int(meta.n_rmu_padded if hasattr(meta, "n_rmu_padded") else meta.n_rmu)
    nq = int(meta.nk_tot)
    n_rtot = int(meta.n_rtot)
    nq_disk = int(n_q_disk)
    # FFT grid for sphere-idx leak accounting.  meta.fft_grid is set on
    # the production MetaInfo; fall back to a cube of ``n_rtot**(1/3)``
    # if absent (shouldn't happen on a real run, but keeps tests with
    # a SimpleNamespace meta working).
    fft_grid = tuple(getattr(meta, 'fft_grid', None) or
                     (int(round(n_rtot ** (1/3))),) * 3)
    # One replicated sphere-idx buffer per rank, charged in every peak.
    n_sphere_buffers = N_SPHERE_IDX_BUFFERS
    # n_q_ibz for Peak E.  When the IBZ cascade is active for V_q,
    # per-tile zeta_L_all slabs shrink by n_q_full/n_q_ibz.  Default to
    # full BZ (conservative — matches the 0X_lorrax production run).
    if n_q_ibz is None:
        n_q_ibz = nq
    n_q_ibz = int(n_q_ibz)

    budget = budget_gb * 1e9
    target = budget * target_utilization

    # Picker order: r_chunk → gflat_chunk_size → band_chunk.  Peak C and
    # Peak D transients don't coexist (fit_one_rchunk returns and
    # block_until_ready before accumulate runs — isdf/core.py fit_one_rchunk)
    # so each peak draws from the same shared headroom independently.

    # Mesh-floor helper for band_chunk.
    def _bump_band_chunk_to_mesh_floor(bc_in: int) -> int:
        """Round bc up to a multiple of p_xy, floored at p_xy."""
        bc = int(bc_in)
        if p_xy > 1 and bc % p_xy != 0:
            bc = ((bc + p_xy - 1) // p_xy) * p_xy
        if bc < p_xy:
            bc = p_xy
        return min(bc, max(int(nb_total), p_xy))

    # ---- 1. r_chunk --------------------------------------------------------
    # Pick r_chunk against Peak C's transient slope.  Two terms scale
    # with r_chunk: the P-pair concurrent slots (dominant) and the
    # zeta_out output (sized at full-mesh /p sharding).
    # Resolve None defaults against the active XLA backend.
    # GPU = 3 slots, CPU = 4 slots (see _default_pair_density_slots docstring).
    if pair_density_slots_charge is None:
        pair_density_slots_charge = _default_pair_density_slots()
    if pair_density_slots_transverse is None:
        pair_density_slots_transverse = _default_pair_density_slots()
    pair_density_slots = (
        pair_density_slots_transverse if is_bispinor
        else pair_density_slots_charge)
    α_C = (
        pair_density_slots * _bytes_c128(nk, ns, ns, mu, shard=p_xy)
        + _bytes_c128(nq, mu, shard=p)  # zeta_out slope
    )
    # Centroids ×4 (not ×2) — agent_a finding #1 + agent_g census §6:
    # rmuT_X + rmu_Y transpose buffers for both ψ_l and ψ_r.
    # gflat_acc is also RESIDENT during fit_one_rchunk (Round-10 / agent_q
    # breakdown): the separate fit/accumulate jits have isolated transient
    # slots, so gflat_acc must be charged against the C headroom too.
    c_C_const = (
        4 * _bytes_c128(nk, ns, mu, nb_total, shard=p_xy)
        + _bytes_c128(nq, mu, mu, shard=p_xy)
        + _bytes_c128(nq_disk, mu, ngkmax, shard=p_xy)  # gflat_acc
        + _sphere_idx_replicated_bytes(
            nq=nq, fft_grid=fft_grid, n_buffers=n_sphere_buffers)
    )
    headroom_C = max(0.0, target - c_C_const)

    if r_chunk_override and r_chunk_override > 0:
        r_chunk = min(int(r_chunk_override), n_rtot)
        # Recompute Peak C transient at the override; warn if the
        # picker would have chosen smaller (i.e. the override exceeds
        # the C-headroom budget cap).
        r_natural = (int(headroom_C / α_C) if α_C > 0 else n_rtot)
        r_natural = max(min(mu, n_rtot), min(n_rtot, r_natural))
        if p_xy > 1:
            r_natural -= r_natural % p_xy
            r_natural = max(r_natural, p_xy)
        if r_chunk > r_natural:
            peak_C_at_override = c_C_const + α_C * float(r_chunk)
            print(
                f"  [plan_gflat_chunks] WARNING: r_chunk overridden to "
                f"{r_chunk} (cap was {r_natural}); Peak C at overridden "
                f"r ≈ {peak_C_at_override/1e9:.2f} GB/dev (budget "
                f"{budget/1e9:.2f} GB/dev).")
    else:
        r_lo = min(mu, n_rtot)
        r_from_budget = (int(headroom_C / α_C) if α_C > 0 else n_rtot)
        r_chunk = max(r_lo, min(n_rtot, r_from_budget))
        r_chunk = max(r_chunk, math.ceil(n_rtot / max_chunks))
        r_chunk = min(r_chunk, n_rtot)
        if p_xy > 1:
            r_chunk -= r_chunk % p_xy
            r_chunk = max(r_chunk, p_xy)
    n_r_chunks = max(1, math.ceil(n_rtot / r_chunk))

    # ---- 2. gflat_chunk_size ----------------------------------------------
    # Cap at GFLAT_CHUNK_SIZE_CAP (=100).  Past cs ~ 1000 cuFFT switches
    # plan algorithm and workspace grows non-linearly (verified OOM at
    # cs=1414 on production CrI3 80Ry — agent_f_live_arrays_probe).
    # Picker order: floor=4 (cuFFT plan amortisation) → largest multiple
    # of bc_floor_factor=4 ≤ 100 that fits Peak D headroom.
    GFLAT_CHUNK_FLOOR = 4
    bc_floor_factor = 4
    # Peak D persistent base.
    persistent_D = _bytes_c128(nq_disk, mu, ngkmax, shard=p_xy)
    transient_zeta_D = _bytes_c128(nq_disk, mu, r_chunk, shard=p_xy)
    centroids_persist = (
        4 * _bytes_c128(nk, ns, mu, nb_total, shard=p_xy)  # ×4 not ×2
    )
    base_D = (
        centroids_persist
        + _bytes_c128(nq, mu, mu, shard=p_xy)  # L_q
        + persistent_D
        + transient_zeta_D
        + _sphere_idx_replicated_bytes(
            nq=nq, fft_grid=fft_grid, n_buffers=n_sphere_buffers)
    )
    headroom_D = max(0.0, target - base_D)
    fft_per_row = _bytes_c128(n_rtot) * fft_box_factor_D
    if gflat_chunk_size_override and gflat_chunk_size_override > 0:
        gflat_chunk_size = int(gflat_chunk_size_override)
        # User-overrideable knob: warn (don't refuse) if the override
        # exceeds the safe-regime cap.  Runtime will OOM if the user
        # picks too aggressively (agent_f saw cs=1414 OOM at production
        # CrI3 80Ry); we want the user to see WHY in the planner log
        # rather than at runtime under a confusing cuFFT plan-creation
        # error.  Per agent_j §1: the cs cap lives in the planner, not
        # in the override path, so the user can knowingly opt out.
        if gflat_chunk_size > GFLAT_CHUNK_SIZE_CAP:
            # Recompute Peak D's FFT-box transient at the overridden cs
            # so the printed peak reflects what the runtime will actually
            # allocate (not the capped planner pick).  See _peak_D_accumulate
            # below for the formula; replicated here as a quick estimate.
            peak_D_at_override = (
                base_D + fft_per_row * float(gflat_chunk_size))
            print(
                f"  [plan_gflat_chunks] WARNING: gflat_chunk_size "
                f"overridden to {gflat_chunk_size} (cap was "
                f"{GFLAT_CHUNK_SIZE_CAP}); past the cuFFT plan-algorithm "
                f"crossover at cs ~ 1000 cuFFT scratch grows non-linearly "
                f"(agent_f cs=1414 OOM verified).  Peak D at overridden "
                f"cs ≈ {peak_D_at_override/1e9:.2f} GB/dev (budget "
                f"{budget/1e9:.2f} GB/dev).")
    else:
        cs_from_budget = max(GFLAT_CHUNK_FLOOR,
                             int(headroom_D / max(fft_per_row, 1.0)))
        # CAP AT GFLAT_CHUNK_SIZE_CAP.  Past cs ~ 1000 cuFFT switches
        # plan algorithm and workspace grows non-linearly (~5× box at
        # cs=1414 → OOM verified at production CrI3 80Ry, agent_f).
        # We don't need cs > 100 for performance (per-iter FFT cost is
        # dominated by spatial FFT shape, not batch size — agent_d M3
        # measured cs=1 within 15% of cs=360 per-r-chunk wall).
        # Capped to stay in the verified safe regime.
        cs_capped = min(cs_from_budget, GFLAT_CHUNK_SIZE_CAP)
        # Round to multiple of bc_floor_factor (=4) for cuFFT plan
        # amortisation.  Pick the LARGEST multiple ≤ cap that fits.
        cs_capped = max(GFLAT_CHUNK_FLOOR,
                        (cs_capped // bc_floor_factor) * bc_floor_factor)
        gflat_chunk_size = cs_capped

    # ---- 3. band_chunk -----------------------------------------------------
    if band_chunk_override and band_chunk_override > 0:
        band_chunk = _bump_band_chunk_to_mesh_floor(int(band_chunk_override))
    else:
        per_unit_bc = (_bytes_c128(ns, n_rtot) * fft_box_factor_A)
        bc_cap = max(1, int(0.5 * target / max(per_unit_bc, 1.0)))
        bc_cap = min(bc_cap, int(nb_total))
        band_chunk = _round_pow2_down(bc_cap)
        band_chunk_pre = band_chunk
        band_chunk = _bump_band_chunk_to_mesh_floor(band_chunk)
        if band_chunk != band_chunk_pre and band_chunk_pre < p_xy:
            pass

    if band_chunk_override and band_chunk_override > 0:
        bc_pre = int(band_chunk_override)
        if band_chunk != bc_pre:
            print(
                f"  [gflat_memory_model] band_chunk_size bumped from "
                f"{bc_pre} to {band_chunk} to satisfy world_size="
                f"{p_xy} (band axis is sharded across all mesh ranks; "
                f"per-device bands per bc = band_chunk // world_size must "
                f"be ≥ 1).")

    n_bc = max(1, math.ceil(nb_total / max(band_chunk, 1)))

    # ---- 4. Compute per-peak breakdowns + HWM -----------------------------
    cs_for_box = gflat_chunk_size  # always an int after the cap

    peak_A = _peak_A_centroid_load(
        nk=nk, ns=ns, n_rtot=n_rtot, nb_per_load=nb_total,
        band_chunk=band_chunk, mu=mu, p=p, p_xy=p_xy,
        fft_box_factor_A=fft_box_factor_A,
        nq=nq, fft_grid=fft_grid, n_sphere_buffers=n_sphere_buffers,
    )
    peak_B = _peak_B_cct_chol(
        nk=nk, ns=ns, nq=nq, mu=mu, nb_total=nb_total, p=p, p_xy=p_xy,
        fft_grid=fft_grid, n_sphere_buffers=n_sphere_buffers,
    )
    peak_C = _peak_C_fit_one_rchunk(
        nk=nk, ns=ns, nq=nq, nq_disk=nq_disk, mu=mu, ngkmax=ngkmax,
        n_rtot=n_rtot, r_chunk=r_chunk,
        band_chunk=band_chunk, n_bc=n_bc, nb_total=nb_total,
        p=p, p_x=p_x, p_y=p_y, p_xy=p_xy,
        fft_box_factor=fft_box_factor_A,
        pair_density_slots=pair_density_slots,
        is_charge_channel=(not is_bispinor),
        fft_grid=fft_grid, n_sphere_buffers=n_sphere_buffers,
    )
    peak_D = _peak_D_accumulate(
        nk=nk, ns=ns, nq=nq, nb_total=nb_total,
        nq_disk=nq_disk, mu=mu, n_rtot=n_rtot, ngkmax=ngkmax,
        r_chunk=r_chunk, gflat_chunk_size=cs_for_box, p=p, p_xy=p_xy,
        fft_box_factor_D=fft_box_factor_D,
        fft_grid=fft_grid, n_sphere_buffers=n_sphere_buffers,
    )
    # ---- Peak E: V_q (dominant tile = TT off-diagonal, same_zeta=False)
    # CC tile: same_zeta=True, write_g0=True
    peak_E_cc = _peak_E_v_q_per_tile_transient(
        n_q_ibz=n_q_ibz, mu_L=mu, mu_R=mu, ngkmax=ngkmax,
        p_x=p_x, p_y=p_y, p_xy=p_xy,
        same_zeta=True, write_g0=True,
        fft_grid=fft_grid, n_sphere_buffers=n_sphere_buffers,
        nb_total=nb_total, nk=nk, ns=ns, mu=mu,
    )
    # TT off-diagonal: same_zeta=False, write_g0=False
    # This is the binding peak per agent_i §2.
    peak_E_off = _peak_E_v_q_per_tile_transient(
        n_q_ibz=n_q_ibz, mu_L=mu, mu_R=mu, ngkmax=ngkmax,
        p_x=p_x, p_y=p_y, p_xy=p_xy,
        same_zeta=False, write_g0=False,
        fft_grid=fft_grid, n_sphere_buffers=n_sphere_buffers,
        nb_total=nb_total, nk=nk, ns=ns, mu=mu,
    ) if is_bispinor else {}
    # Worst-case tile is off-diagonal when bispinor.
    if is_bispinor:
        peak_E = peak_E_off
    else:
        peak_E = peak_E_cc
    # Unfold output piggybacks the V_acc slot — not additive over the
    # per-tile peak.  But surface it as a separate term for clarity.
    peak_E_unfold = _peak_E_v_q_unfold(
        n_q_full=nq, mu_L=mu, mu_R=mu, p_xy=p_xy,
    )
    # Bispinor Lorentz-mix buffer (only when transverse IBZ cascade is
    # active).  This is additive over the per-tile peak when present.
    peak_E_lorentz = _peak_E_v_q_bispinor_buffer(
        n_q_full=nq, mu_T=mu, p_xy=p_xy, use_ibz_T=use_ibz_T,
    )
    # Merge.  The per-tile peak terms are the binding ones; unfold and
    # Lorentz-mix get added to the same E.* namespace.
    peak_E_all = dict(peak_E)
    peak_E_all.update(peak_E_unfold)
    peak_E_all.update(peak_E_lorentz)

    A_total = sum(peak_A.values())
    B_total = sum(peak_B.values())
    C_total = sum(peak_C.values())
    D_total = sum(peak_D.values())
    # Peak E total: per-tile + lorentz_mix.  V_acc_full_BZ is aliased
    # into the same slot as V_acc per-tile (agent_i §3), so we subtract
    # it from the per-tile peak to avoid double-counting.
    E_per_tile = sum(peak_E.values())
    E_lorentz = sum(peak_E_lorentz.values())
    E_total = E_per_tile + E_lorentz

    peak_totals = {
        'A_centroid': A_total,
        'B_CCT_chol': B_total,
        'C_fit_one_rchunk': C_total,
        'D_accumulate': D_total,
        'E_v_q': E_total,
    }
    bottleneck = max(peak_totals, key=peak_totals.get)
    hwm = peak_totals[bottleneck]
    peak_components: dict = {}
    for src in (peak_A, peak_B, peak_C, peak_D, peak_E_all):
        peak_components.update(src)

    return GFlatChunkPlan(
        band_chunk=int(band_chunk),
        r_chunk=int(r_chunk),
        n_r_chunks=int(n_r_chunks),
        gflat_chunk_size=int(gflat_chunk_size),
        hwm_bytes=float(hwm),
        peak_breakdown=peak_totals,
        peak_components=peak_components,
        bottleneck=bottleneck,
        budget_bytes=float(budget),
    )

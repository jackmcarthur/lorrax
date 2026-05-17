"""G-flat ζ + V_q memory model — pick chunk sizes near the budget.

Four per-rank HBM peaks across :func:`isdf_fitting.fit_zeta_to_h5`:

    Peak A — band-chunked centroid load (pre-loop).
        ψ(G) → IFFT → sample at r_μ.  The ψ(r) FFT box transient is
        the dominant cost, sharded on ('x','y').  Run once per channel.
        Persistent here: only the centroid output being filled.

    Peak B — CCT + Cholesky (pre-loop).
        Pair density on (μ, ν) full-grid + C_q FFT + L_q factor.
        Persistent: centroids (L+R copies).
        Transient: P_l, P_r at full μ², C_q, L_q workspace.

    Peak C — fit_one_rchunk (inside the r-chunk loop).
        Per-bc ψ(G) → ψ(r_chunk) IFFT, pair-density accumulators,
        CCT/ZCT k-convolution, solve.  The fused jit holds:
          • centroids + L_q (persistent base)
          • band-chunked FFT box transient (Python-unrolled bc-loop;
            n_bc copies of the FFT-box workspace stack inside XLA's
            trace per the comment in ``gw_init._fft_moment``)
          • P_l + P_r rank-5 accumulators on (μ, r_chunk)
          • IFFT'd P_l, P_r in R-space (rank-5)
          • Z_q intermediate before reshard

    Peak D — accumulate_rchunk_to_gflat (right after fit_one_rchunk).
        gflat_acc persistent.  Transient: per-scan-iter zero-padded
        FFT box ``(cs, n_rtot)`` where cs = gflat_chunk_size; the
        FFT itself adds cuFFT scratch (up to ~4-8× the box size).

Chunker knobs:

    band_chunk     — bc-size for the per-bc ψ(G)→ψ(r-chunk) IFFT inside
                     fit_one_rchunk.  Primary lever on Peak A and Peak
                     C's FFT-box term.  Must divide nb if possible
                     (remainder handled but creates short tail).

    r_chunk        — r-axis chunk count for the outer loop.  Lower-bounded
                     by ``n_rmu`` (per user spec: the eventual Σ_μν output
                     occupies ``n_rmu²·n_q·16`` bytes, so paying less
                     than ``n_rmu`` work per chunk is wasted iteration
                     overhead).  Upper-bounded by ``max_chunks``.

    gflat_chunk_size — scan chunk size inside accumulate_rchunk_to_gflat.
                       Only affects Peak D.  ``None`` = one-shot (fastest
                       when it fits).

Sharding shorthand throughout:  ``P = p_x · p_y`` (mesh size).
"""
from __future__ import annotations

import dataclasses
import math
from typing import Optional

import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


_BYTES_PER_C128 = 16


def _bytes_c128(*dims, shard: int = 1) -> float:
    """Per-rank c128 byte count for a tensor of ``dims`` sharded over
    ``shard`` ranks (1 = replicated)."""
    n = 1
    for d in dims:
        n *= int(d)
    return _BYTES_PER_C128 * n / max(int(shard), 1)


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


@dataclasses.dataclass
class GFlatChunkPlan:
    """Resolved chunk sizes + per-rank HBM high-water estimate."""
    band_chunk: int
    r_chunk: int
    n_r_chunks: int
    gflat_chunk_size: Optional[int]   # None ⇒ one-shot
    hwm_bytes: float
    peak_breakdown: dict              # name -> bytes  (A/B/C/D totals)
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
        # in the key) and emit in A→D order so the format is stable.
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
# Per-peak cost functions
# ---------------------------------------------------------------------------
# Each returns a dict[str, float] of per-rank byte contributions; the peak
# is sum-over-dict.  Keeping the per-term breakdown surfaced lets the
# log explain WHY each peak is what it is.

def _peak_A_centroid_load(*, nk, ns, n_rtot, nb_per_load, band_chunk,
                          mu, p, p_xy, fft_box_factor_A) -> dict:
    """Pre-loop ψ(G) → r-space → centroid sample.  Runs once per channel
    (charge + 3 transverse on bispinor).

    FFT-box formula audited 2026-05-17 (Agent A audit, Bug #1).  The
    actual ``gflat_to_rmu._kernel`` (wfn_transforms.py:611-851) batches
    on a flat ``(nk · nb_local)`` axis INSIDE a shard_map.  Its
    per-iter FFT box is ``c128[cs, ns, nx, ny, nz]`` where ``cs`` is
    the planner-picked chunk size — **per-rank already**, no nk factor.
    The pre-refit formula multiplied by ``nk`` (over-count ~36× on
    CrI3 6×6) and additionally divided by ``p_xy`` (the kernel is
    already inside shard_map so the box is per-rank natively).  Use
    ``band_chunk`` as the proxy for ``cs`` (the centroid-load caller
    picks ``cs`` from its own per-rank HBM budget; band_chunk is a
    representative upper bound)."""
    out = {
        "centroid_out_filling":
            _bytes_c128(nk, ns, mu, nb_per_load, shard=p),
        "phase_table":
            _bytes_c128(nk, n_rtot),
        "fft_box":
            _bytes_c128(band_chunk, ns, n_rtot) * fft_box_factor_A,
    }
    return {f"A.{k}": v for k, v in out.items()}


def _peak_B_cct_chol(*, nk, ns, nq, mu, p, p_xy) -> dict:
    """CCT/Cholesky pre-loop on (μ, ν) full-grid."""
    out = {
        "centroids_persistent":  # L+R copies, sharded
            2 * _bytes_c128(nk, ns, mu, ns, shard=p_xy),
        "P_l_plus_P_r_open_spin":
            2 * _bytes_c128(nk, ns, ns, mu, mu, shard=p_xy),
        "C_q":
            _bytes_c128(nq, mu, mu, shard=p_xy),
        "L_q":
            _bytes_c128(nq, mu, mu, shard=p_xy),
    }
    return {f"B.{k}": v for k, v in out.items()}


def _peak_C_fit_one_rchunk(*, nk, ns, nq, mu, n_rtot, r_chunk,
                           band_chunk, n_bc, p, p_x, p_y, p_xy,
                           fft_box_factor, pair_density_slots,
                           is_charge_channel) -> dict:
    """Per-r-chunk fit_one_rchunk fused-jit peak.

    bc-loop is Python-unrolled inside the kernel, so n_bc copies of the
    psi_bc_Y slab stack across XLA's fused trace.

    ``pair_density_slots`` is the **XLA-BufferAssignment-determined**
    count of concurrent rank-5
    ``c128[nk, ns², n_rmu_local, r_chunk_local]`` tensors live at peak
    inside the monolithic ``c_q_from_psi_sm`` / ``z_q_from_psi_sm``
    shard_map.  Read from XLA's per-kernel
    ``-memory-usage-report.txt`` dump as the number of distinct
    preallocated-temp slots holding a P-pair-shaped value.

    Default = 3: ``P_l_R_conj``, ``P_r_R``, plus one XLA scratch.
    Verified on MoS2 3×3 bispinor / 2×2 mesh.  If XLA changes its
    buffer assignment in a future version, count slots in a fresh
    ``module_NNNN.jit__kernel.sm_*.memory-usage-report.txt`` and
    update the defaults below.
    """
    # Persistent during the chunk loop (not just one iter):
    persistent = {
        "centroids_persist":
            2 * _bytes_c128(nk, ns, mu, nk, shard=p_xy),  # L+R approx
        "L_q":
            _bytes_c128(nq, mu, mu, shard=p_xy),
        "gflat_acc":
            0.0,  # see Peak D — accounted there to avoid double-count
    }
    # Transient inside fit_one_rchunk.  The dominant term is
    # ``pair_density_slots`` rank-5 buffers; psi_bc_Y, the FFT box,
    # Z_q etc. all fit in the SAME lifetime slots (XLA's allocator
    # reuses them when lifetimes don't overlap — verified in the
    # module_0510 dump where slot 1 holds both a P_pair and the
    # band-chunk FFT box across non-overlapping lifetimes).
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


def _peak_D_accumulate(*, nq_disk, mu, n_rtot, ngkmax, r_chunk,
                       gflat_chunk_size, p, p_xy, fft_box_factor_D) -> dict:
    """accumulate_rchunk_to_gflat peak — runs after fit_one_rchunk
    returns (its P_l/P_r are freed); ζ_chunk is the only fit_one_rchunk
    output still live.

    Uses a per-peak ``fft_box_factor_D`` (default 2.0, vs Peak A's 4.0).
    The accumulate kernel's FFT (wfn_transforms.py:1057-1107) is shape
    ``c128[cs, nx, ny, nz]`` (no ns axis — ζ is spin-traced upstream)
    with a small batch ``cs`` and huge spatial axes.  Runtime debug
    print (chunk_size=360 → 6.48 GB/rank = bare 1× box) shows cuFFT's
    out-of-place 3D fftn needs ~2× the box scratch, not 4× (which was
    calibrated for the 6-D centroid-load IFFT nest in Peak A).
    """
    # gflat_acc is the persistent G-flat ζ accumulator (μ-flat sharded
    # across mesh).
    persistent = {
        "gflat_acc": _bytes_c128(nq_disk, mu, ngkmax, shard=p_xy),
    }
    transient = {
        "zeta_chunk":
            _bytes_c128(nq_disk, mu, r_chunk, shard=p_xy),
        "accumulate_fft_box":
            _bytes_c128(gflat_chunk_size, n_rtot) * fft_box_factor_D,
        # Sphere/phase tables baked into closure: ~tens of MB; ignored.
    }
    out = {f"D.{k}": v for k, v in persistent.items()}
    out.update({f"D.{k}": v for k, v in transient.items()})
    return out


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
    fft_box_factor: float | None = None,    # legacy alias for fft_box_factor_A
    fft_box_factor_A: float = 4.0,
    fft_box_factor_D: float = 2.0,
    pair_density_slots_transverse: int = 3,
    pair_density_slots_charge: int = 3,
    is_bispinor: bool = True,
    max_chunks: int = 64,
    r_chunk_override: int | None = None,
    band_chunk_override: int | None = None,
    gflat_chunk_size_override: int | None = None,
) -> GFlatChunkPlan:
    """Pick (band_chunk, r_chunk, gflat_chunk_size) to land near
    ``target_utilization · budget_gb`` per device.

    Algorithm (deterministic, no iterative search; **r-first picker**
    refit 2026-05-17 per the memory-model refit task — was band-first):

      1. Compute persistent footprint (centroids + L_q + gflat_acc) and
         the shared headroom ``H = target − persistent_common``.
      2. Pick ``r_chunk`` first against Peak C's transient slope
         (``α_C`` = pair_density_slots · per-r rank-5 bytes).  This is
         the expensive axis — its iters are slow (solve dominates
         per-chunk wall) — so we saturate it first.
      3. Pick ``gflat_chunk_size`` against Peak D's transient.  Peak C
         and Peak D transients do **not** coexist (the ``fit_one_rchunk``
         fused jit returns + ``block_until_ready`` before
         ``accumulate_rchunk_to_gflat`` runs — see
         ``isdf_fitting.py:2442–2496``), so each draws from the same
         shared headroom independently.
      4. Pick ``band_chunk`` against Peak A's FFT-box transient
         (centroid load pre-loop, runs once per channel; near-free
         after the per-peak A formula fix that dropped the spurious
         ``nk`` factor).

    Overrides take precedence and skip the corresponding step.

    Per-peak ``fft_box_factor_{A,D}`` allow independent calibration of
    the centroid-load 3D IFFT nest (Peak A, factor 4 — empirically
    validated for 6-D ``(nk, B_b/P, ns, n_r)`` shards) vs the
    accumulate flat-batch 3D FFT (Peak D, factor 2 — runtime debug
    print at cs=360 → 6.48 GB/rank = bare 1× box, cuFFT scratch ~1×).
    Passing ``fft_box_factor=`` is back-compatible: it sets the A
    factor (legacy alias).

    Returns a :class:`GFlatChunkPlan` with HWM = max of {Peak A, B, C, D}.
    """
    # Legacy alias: callers that still pass ``fft_box_factor=`` get the
    # value routed to ``fft_box_factor_A`` (the legacy single-factor
    # behaviour modelled the centroid-load IFFT nest).  ``fft_box_factor_D``
    # keeps its per-peak default unless overridden separately.
    if fft_box_factor is not None:
        fft_box_factor_A = float(fft_box_factor)
    p_x = int(mesh_xy.shape['x'])
    p_y = int(mesh_xy.shape['y'])
    p_xy = p_x * p_y                                # mesh.size
    p = p_xy                                        # alias for clarity
    nk = int(meta.nk_tot)
    ns = int(meta.nspinor)
    mu = int(meta.n_rmu_padded if hasattr(meta, "n_rmu_padded") else meta.n_rmu)
    nq = int(meta.nk_tot)
    n_rtot = int(meta.n_rtot)
    nq_disk = int(n_q_disk)

    budget = budget_gb * 1e9
    target = budget * target_utilization

    # Picker order (r → gflat → band).  Per Agent C §1 (2026-05-17):
    # Peak C and Peak D transients do NOT coexist — ``fit_one_rchunk``
    # returns + ``block_until_ready`` before ``accumulate_rchunk_to_gflat``
    # starts (``isdf_fitting.py:2442-2496``), so each peak's transient
    # draws from the same shared headroom independently.  Joint
    # optimisation is unnecessary; sequential picks against the SAME
    # headroom suffice.  Order r_chunk → gflat_chunk_size → band_chunk
    # because r is the expensive axis (slow-solve iters), gflat is the
    # cheap middle axis, band is near-free after the per-peak A formula
    # fix.

    # Mesh-floor helper for band_chunk (used in step 3).
    def _bump_band_chunk_to_mesh_floor(bc_in: int) -> int:
        """Round bc up to a multiple of p_xy, floored at p_xy, capped
        at max(nb_total, p_xy).  See ``test_band_chunk_size_floor.py``
        for the all_gather_dim=0 regression motivating this contract."""
        bc = int(bc_in)
        if p_xy > 1 and bc % p_xy != 0:
            bc = ((bc + p_xy - 1) // p_xy) * p_xy
        if bc < p_xy:
            bc = p_xy
        return min(bc, max(int(nb_total), p_xy))

    # ---- 1. r_chunk --------------------------------------------------------
    # Pick r_chunk against Peak C's transient slope α_C = slots ·
    # rank-5 P-pair bytes/row.  Shared headroom = target − persistent.
    # Persistent during Peak C: centroids (L+R) + L_q.  (gflat_acc is
    # alive too but its sole accounting lives at Peak D to avoid double
    # counting — Peak C's max-sum already covers it via the persistent
    # mass that's also in D's base.)
    pair_density_slots = (
        pair_density_slots_transverse if is_bispinor
        else pair_density_slots_charge)
    α_C = pair_density_slots * _bytes_c128(nk, ns, ns, mu, shard=p_xy)
    c_C_const = (
        2 * _bytes_c128(nk, ns, mu, nb_total, shard=p_xy)  # centroids L+R
        + _bytes_c128(nq, mu, mu, shard=p_xy)              # L_q
    )
    headroom_C = max(0.0, target - c_C_const)

    if r_chunk_override and r_chunk_override > 0:
        r_chunk = min(int(r_chunk_override), n_rtot)
    else:
        # Lower bound: r_chunk ≥ μ (per user note — Σ_μν output dominates
        # any savings from finer chunking).
        r_lo = min(mu, n_rtot)
        # Upper bound from the budget: r_chunk ≤ headroom / α_C.
        r_from_budget = (int(headroom_C / α_C) if α_C > 0 else n_rtot)
        r_chunk = max(r_lo, min(n_rtot, r_from_budget))
        # Enforce max_chunks cap (so chunk count stays manageable for
        # XLA's scan trace + Python overhead).
        r_chunk = max(r_chunk, math.ceil(n_rtot / max_chunks))
        r_chunk = min(r_chunk, n_rtot)
        # Round down to a multiple of p_xy so the (μ_XY, r_) sharding
        # at the solve output divides cleanly.
        if p_xy > 1:
            r_chunk -= r_chunk % p_xy
            r_chunk = max(r_chunk, p_xy)
    n_r_chunks = max(1, math.ceil(n_rtot / r_chunk))

    # ---- 2. gflat_chunk_size ----------------------------------------------
    # Pick gflat_chunk_size against Peak D's transient (independent of C
    # because C's P-pair slots are freed before D allocates its FFT box).
    # Peak D persistent base = gflat_acc + zeta_chunk(r_chunk) + the
    # centroids+L_q that stay live for the rest of fit_zeta.
    persistent_D = _bytes_c128(nq_disk, mu, ngkmax, shard=p_xy)
    transient_zeta_D = _bytes_c128(nq_disk, mu, r_chunk, shard=p_xy)
    centroids_persist = (
        2 * _bytes_c128(nk, ns, mu, nb_total, shard=p_xy)
    )
    base_D = (
        centroids_persist
        + _bytes_c128(nq, mu, mu, shard=p_xy)
        + persistent_D
        + transient_zeta_D
    )
    headroom_D = max(0.0, target - base_D)
    cs_one_shot = max(1, int(math.ceil(nq_disk * mu / p_xy)))
    fft_per_row = _bytes_c128(n_rtot) * fft_box_factor_D
    # cuFFT plan-amortisation floor: below ~4 rows per batch the plan
    # picks a small-batch algorithm with worse FLOPS/byte (fft_helpers
    # has docstring warning at line 14-23).  Enforce gflat_chunk_size ≥ 4.
    GFLAT_CHUNK_FLOOR = 4
    if gflat_chunk_size_override and gflat_chunk_size_override > 0:
        gflat_chunk_size = int(gflat_chunk_size_override)
    elif fft_per_row * cs_one_shot <= headroom_D:
        gflat_chunk_size = None  # one-shot fits
    else:
        cs_from_budget = max(1, int(headroom_D / max(fft_per_row, 1.0)))
        gflat_chunk_size = max(GFLAT_CHUNK_FLOOR, cs_from_budget)

    # ---- 3. band_chunk -----------------------------------------------------
    # Pick band_chunk against Peak A's FFT-box transient (centroid-load
    # IFFT nest, runs once per channel in the pre-loop).  After the per-
    # peak A formula fix (Bug #1 in Agent A audit) the box is just
    # ``cs · ns · n_rtot · 16 · fft_box_factor_A`` per rank inside the
    # shard_map — no nk factor — so this is near-free.
    if band_chunk_override and band_chunk_override > 0:
        band_chunk = _bump_band_chunk_to_mesh_floor(int(band_chunk_override))
    else:
        # Per-bc FFT-box cost (per-rank).  Compare to remaining headroom
        # against persistent_C.  We split 50% so other Peak-C terms keep
        # their slack — but with the formula fix, even 100% would
        # rarely matter.  Empirical: at bc=32 the term is ~140 MB.
        per_unit_bc = (_bytes_c128(ns, n_rtot)
                       * fft_box_factor_A)
        bc_cap = max(1, int(0.5 * target / max(per_unit_bc, 1.0)))
        bc_cap = min(bc_cap, int(nb_total))
        band_chunk = _round_pow2_down(bc_cap)
        band_chunk_pre = band_chunk
        band_chunk = _bump_band_chunk_to_mesh_floor(band_chunk)
        if band_chunk != band_chunk_pre and band_chunk_pre < p_xy:
            # Auto-pick path hit the mesh floor.  No print — that
            # rounding is internal-detail.  Only the override path
            # speaks up (covered below).
            pass

    # If the override path bumped the value, surface the change.
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
    cs_for_box = (gflat_chunk_size if gflat_chunk_size is not None
                  else cs_one_shot)
    peak_A = _peak_A_centroid_load(
        nk=nk, ns=ns, n_rtot=n_rtot, nb_per_load=nb_total,
        band_chunk=band_chunk, mu=mu, p=p, p_xy=p_xy,
        fft_box_factor_A=fft_box_factor_A,
    )
    peak_B = _peak_B_cct_chol(
        nk=nk, ns=ns, nq=nq, mu=mu, p=p, p_xy=p_xy,
    )
    peak_C = _peak_C_fit_one_rchunk(
        nk=nk, ns=ns, nq=nq, mu=mu, n_rtot=n_rtot, r_chunk=r_chunk,
        band_chunk=band_chunk, n_bc=n_bc, p=p, p_x=p_x, p_y=p_y,
        p_xy=p_xy, fft_box_factor=fft_box_factor_A,
        pair_density_slots=pair_density_slots,
        is_charge_channel=(not is_bispinor),
    )
    peak_D = _peak_D_accumulate(
        nq_disk=nq_disk, mu=mu, n_rtot=n_rtot, ngkmax=ngkmax,
        r_chunk=r_chunk, gflat_chunk_size=cs_for_box, p=p, p_xy=p_xy,
        fft_box_factor_D=fft_box_factor_D,
    )
    A_total = sum(peak_A.values())
    B_total = sum(peak_B.values())
    C_total = sum(peak_C.values())
    D_total = sum(peak_D.values())
    peak_totals = {'A_centroid': A_total, 'B_CCT_chol': B_total,
                   'C_fit_one_rchunk': C_total, 'D_accumulate': D_total}
    bottleneck = max(peak_totals, key=peak_totals.get)
    hwm = peak_totals[bottleneck]
    # Per-component breakdown for the formatted log.  The _peak_X
    # helpers already prefix each component key with the peak letter
    # ("A.fft_box", "C.P_pair_concurrent_slots", etc.) so we just
    # merge them into one dict.
    peak_components: dict = {}
    for src in (peak_A, peak_B, peak_C, peak_D):
        peak_components.update(src)

    return GFlatChunkPlan(
        band_chunk=int(band_chunk),
        r_chunk=int(r_chunk),
        n_r_chunks=int(n_r_chunks),
        gflat_chunk_size=(None if gflat_chunk_size is None
                          else int(gflat_chunk_size)),
        hwm_bytes=float(hwm),
        peak_breakdown=peak_totals,        # high-level: A/B/C/D totals
        peak_components=peak_components,   # per-term breakdown
        bottleneck=bottleneck,
        budget_bytes=float(budget),
    )

"""ISDF fitting orchestration and memory-aware chunk sizing for LORRAX GW.

  compute_optimal_chunks — per-device memory model for the 6-stage ISDF pipeline
  fit_zeta / compute_V_q / build_wavefunction_bundle — pipeline steps
  prepare_isdf_and_wavefunctions — top-level orchestrator called by main()
"""
import os
import math
from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import numpy as np
import h5py

import common.timing as timing
from common import jax_profile


# ---------------------------------------------------------------------------
# Memory model — internal helpers for ``compute_optimal_chunks``.
#
# The chunk-loop peak is described by five ``moments``: each is a snapshot of
# concurrently-live device buffers at a distinct point in the r-chunk loop.
# Every moment cost is linear in ``cr`` (the r-chunk size), so closed-form
# inversion gives ``cr_max`` per moment without any iterative search.  The
# functions and small dataclass below break the model up so each piece is
# individually readable / testable; the public driver is unchanged.
# ---------------------------------------------------------------------------

_BYTES_C128 = 16.0
_FFT_COPIES = 3   # cuFFT input + output + scratch (lower bound)
_ZCT_ADDITIONAL_COEF = 3   # ZCT temp + output past the 2 persistent P_l/P_r
                           # accumulators (AOT-measured; see scripts/validate_fft_workspaces.py)


def _bytes_c128(*dims, shard: int = 1) -> float:
    """Bytes for a complex128 array of shape ``dims`` per-device after sharding."""
    result = _BYTES_C128
    for d in dims:
        result *= d
    return result / shard


@dataclass(frozen=True)
class _ChunkAlphas:
    """Per-cr byte coefficients for the five r-chunk moments.

    Each ``α_*`` is the bytes-per-cr live in one moment after sharding.  ``c_solve``
    is the cr-independent overhead for the per-q triangular solve (replicated L
    + 3× full L slabs).  ``m_psi_G_bc`` is the transient per-bc ψ(G) slab on
    device during one bc's FFT.
    """
    α_pair: float
    α_psi_Y_bc: float
    α_zcol: float
    α_z_slice: float
    α_gather: float
    c_solve: float
    m_psi_G_bc: float


def _build_chunk_alphas(*, nk, ns, mu, nq, band_chunk, p_x, p_y, p, nr) -> _ChunkAlphas:
    return _ChunkAlphas(
        α_pair=_bytes_c128(nk, mu, shard=p_x * p_y),
        α_psi_Y_bc=_bytes_c128(nk, band_chunk, ns, shard=p_y),
        α_zcol=_bytes_c128(nq, mu, shard=p),
        α_z_slice=_bytes_c128(mu, shard=p),
        α_gather=_bytes_c128(mu, shard=p) + 2 * _bytes_c128(mu),
        c_solve=_bytes_c128(mu, mu, shard=p_x * p_x) + 3 * _bytes_c128(mu, mu),
        m_psi_G_bc=_bytes_c128(nk, band_chunk, ns, nr, shard=p),
    )


def _fft_moment(cr, base, fft_inloop_bytes, a: _ChunkAlphas, *, n_bc: int = 1):
    """Peak across the bc-loop's FFT + reshard + accumulate.

    Live during a single bc-iteration: base + 2× pair-density accumulators
    + the FFT workspace + the post-reshard ψ_bc_Y slab.

    BUT: ``_make_fit_one_rchunk_kernel`` Python-unrolls the bc-loop
    (``for bc_idx in band_chunk_ranges`` over ``n_bc`` iterations) inside
    one giant jit, so XLA sees n_bc separate ``io_callback + FFT +
    reshard + accumulate`` traces.  The FFT workspace itself can be
    reused iter-to-iter (one cuFFT plan, scratch reused), but each
    iter's post-reshard ψ_bc_Y is a distinct buffer the scheduler can
    keep live concurrently with later iters' to overlap fetch with
    accumulate — measured cumulative live across unroll matches
    ``n_bc · α_psi_Y_bc · cr`` on CrI3 16-GPU (was missing 17 GB at
    chunk_r=112016, band_chunk=16, n_bc=5).

    Pass ``n_bc=1`` to recover the per-iter peak (the legacy model);
    pass ``n_bc = nb / band_chunk`` to honour the unroll cost.
    """
    return base + 2 * a.α_pair * cr + fft_inloop_bytes + n_bc * a.α_psi_Y_bc * cr


def _zct_moment(cr, base, a: _ChunkAlphas):
    """Peak during the ZCT stage: 2 persistent + 3 transient pair-density buffers."""
    return base + (2 + _ZCT_ADDITIONAL_COEF) * a.α_pair * cr


def _reshard_moment(cr, base, a: _ChunkAlphas):
    """Peak during Z_q → Z_col reshard (input + output + NCCL scratch)."""
    return base + 3 * a.α_zcol * cr


def _solve_moment(cr, base, q_batch, a: _ChunkAlphas):
    """Peak during per-q-batch triangular solve."""
    return base + 2 * a.α_zcol * cr + q_batch * (2 * a.α_z_slice * cr + a.c_solve)


def _gather_moment(cr, base, q_gather, a: _ChunkAlphas):
    """Peak during q-gather + H5 write."""
    return base + a.α_zcol * cr + q_gather * a.α_gather * cr


def _max_cr_per_stage(headroom, fft_cost_in_loop, a: _ChunkAlphas, *,
                      nr_max, m_budget, m_zct_cap, n_bc: int = 1) -> dict:
    """Closed-form max feasible ``cr`` for each moment (linear inversion).

    Each moment is ``base + α·cr + c`` so ``cr ≤ (headroom − c) / α``; the
    caller takes the minimum over moments.  Returns a per-stage dict so the
    bottleneck can be reported.

    ``n_bc`` honours the Python-unrolled bc-loop cost in the FFT moment
    (cumulative ψ_bc_Y across n_bc iters live concurrent under XLA's
    fused jit) — see :func:`_fft_moment` for the full rationale.
    """
    # Optional soft cap on zct stage (env override for tight-memory systems).
    denom_fft = 2 * a.α_pair + n_bc * a.α_psi_Y_bc
    denom_zct = (2 + _ZCT_ADDITIONAL_COEF) * a.α_pair
    denom_solve = 2 * a.α_zcol + 2 * a.α_z_slice
    limits = {
        'fft':     (headroom - fft_cost_in_loop) / denom_fft if denom_fft > 0 else nr_max,
        'zct':     headroom / denom_zct if denom_zct > 0 else nr_max,
        'reshard': headroom / (3 * a.α_zcol) if a.α_zcol > 0 else nr_max,
        'solve':   ((headroom - a.c_solve) / denom_solve) if denom_solve > 0 else nr_max,
        'gather':  headroom / (a.α_zcol + a.α_gather) if (a.α_zcol + a.α_gather) > 0 else nr_max,
    }
    if m_zct_cap is not None and a.α_pair > 0:
        zct_headroom = m_zct_cap - (m_budget - headroom)
        if zct_headroom > 0:
            limits['zct'] = min(limits['zct'], zct_headroom / denom_zct)
    return limits



def compute_optimal_chunks(
    meta, mesh_xy, memory_budget_gb: float,
    target_utilization: float = 0.97,
    verbose: bool = True,
    n_b_left: int | None = None, n_b_right: int | None = None,
    r_chunk_override: int | None = None,
    zct_stage_cap_gb: float | None = None,
) -> dict:
    """Derive ISDF chunk sizes that fit within the per-device memory budget.

    System dimensions come from meta; device grid from mesh_xy.
    The ISDF r-chunk loop has 6 stages, each with cost linear in chunk_r (cr):

        stage_cost(cr) = base + αᵢ·cr + cᵢ

    where base = centroids + L_q (persistent; ψ(G) now host-only, per-bc via io_callback),
    αᵢ is the per-cr byte coefficient, and cᵢ is the cr-independent overhead.
    The optimal cr is min over stages of (headroom − cᵢ) / αᵢ — one division
    per stage, no iterative search.

    The 6 stages and their XLA HLO-calibrated multipliers (CrI3, 16 GPUs):

        FFT:     ψ(r_chunk) accumulator + 3× per-k FFT transient (cuFFT)
        Pair:    ψ(r_chunk) + 2× pair density P(nk, μ, cr)
        ZCT:     4× pair data  (P_r alive outside JIT + 3× donated left IFFT)
        Reshard: 4× Z_col      (input + output + 2× NCCL all-to-all temp)
        Solve:   2× Z_col + per-q (2× Z_slice + L_rep + 3× L_full)
        Gather:  Z_col + per-q (input/p + 2× replicated output incl. NCCL)

    Sharding: centroids and pair densities shard on (p_x, p_y).  Z_col and
    solve arrays shard on p = p_x·p_y combined.  The α coefficients encode
    the per-device sizes after sharding.
    """
    if memory_budget_gb <= 0:
        raise ValueError("memory_per_device_gb must be > 0.")

    p_x, p_y = mesh_xy.devices.shape
    p = p_x * p_y
    nb = meta.band_edges[4] - meta.band_edges[0]  # b4 - b0
    nb_l = int(n_b_left) if n_b_left is not None else nb
    nb_r = int(n_b_right) if n_b_right is not None else nb
    nk = float(meta.nk_tot)
    ns = float(meta.nspinor)
    mu = float(meta.n_rmu)
    nq = float(meta.nk_tot)
    nr = float(meta.n_rtot)

    m_budget = memory_budget_gb * 1e9 * target_utilization
    m_zct_cap = float(zct_stage_cap_gb) * 1e9 if zct_stage_cap_gb and zct_stage_cap_gb > 0 else None

    # ---- Persistent arrays (always resident during chunk loop) ----
    # Centroids: left/right × X-sharded + Y-sharded copies.
    # "full" = during initial load (all nb bands); "persist" = after slicing to nb_l + nb_r.
    m_centroids_full = (
        _bytes_c128(nk, ns, mu, nb, shard=p_y)
        + _bytes_c128(nk, ns, mu, nb, shard=p_x)
    )
    m_centroids = (
        _bytes_c128(nk, ns, mu, nb_l + nb_r, shard=p_y)
        + _bytes_c128(nk, ns, mu, nb_l + nb_r, shard=p_x)
    )
    if m_centroids_full + m_centroids > m_budget:
        raise ValueError(
            f"Centroid storage requires {(m_centroids_full + m_centroids)/1e9:.2f} GB/device "
            f"but only {memory_budget_gb:.2f} GB allocated.")

    # ---- Band-chunked FFT (centroid extraction, runs before chunk loop) ----
    # ψ(G) → IFFT → ψ(r).  ``_FFT_COPIES`` is the nominal lower bound (input +
    # output + one scratch); the in-loop FFT cost is queried exactly via
    # ``query_fft_peak_bytes`` below.  This pre-loop sizing uses the nominal
    # 3× to set ``bpd_max``.
    m_phase = _bytes_c128(nk, nr)
    headroom_fft = m_budget - m_centroids_full
    if headroom_fft <= m_phase:
        raise ValueError("Insufficient memory for even a single-band FFT chunk.")
    one_band = _FFT_COPIES * _bytes_c128(nk, ns, nr)
    bpd_max = max(1, int((headroom_fft - m_phase) / one_band))
    band_chunk = min(nb, bpd_max * p)
    bpd = max(1, -(-band_chunk // p))  # ceil div matching read_Gvecs_to_devices
    peak_centroid_fft_stage = (
        m_centroids_full + _FFT_COPIES * _bytes_c128(nk, bpd, ns, nr) + m_phase
    )

    # ---- C_q build (pair density → C_q → L_q, runs before chunk loop) ----
    stage_cct = (
        m_centroids
        + 2 * _bytes_c128(nk, mu, mu, shard=p_x * p_y)
        + _bytes_c128(nq, mu, mu, shard=p_x * p_y)
    )
    if stage_cct > m_budget:
        raise ValueError(f"C_q build requires {stage_cct/1e9:.2f} GB/device — exceeds budget.")
    m_L_q = _bytes_c128(nq, mu, mu, shard=p_x * p_y)  # L_q persists; same size as C_q

    # ψ(G) lives on HOST (per-rank band-sharded) and is fetched into the jit
    # one bc at a time via io_callback; no persistent device residency.
    # Only the currently active bc's ψ(G) is on device — captured in the
    # ``_fft_moment`` term.  See ``_build_chunk_alphas`` and the moment
    # functions at module top for the full memory model.

    alphas = _build_chunk_alphas(
        nk=nk, ns=ns, mu=mu, nq=nq,
        band_chunk=band_chunk, p_x=p_x, p_y=p_y, p=p, nr=nr,
    )

    def _eval_stages(cr, base, fft_inloop, n_bc):
        """Forward-evaluate each moment at a given cr.

        Returns ``(stages_dict, m_zcol, m_solve_per_q, m_gather_per_q, k_batch)``.
        Note: ``fft_inloop`` already includes the input (=m_psi_G_bc), output,
        and cuFFT scratch via ``query_fft_peak_bytes``.

        ``n_bc`` is the number of bc-iterations the kernel's Python-unrolled
        bc-loop emits — fed into ``_fft_moment`` so the cumulative ψ_bc_Y
        live across iters is honoured.
        """
        m_zcol = alphas.α_zcol * cr
        m_solve_per_q = 2 * alphas.α_z_slice * cr + alphas.c_solve
        m_gather_per_q = alphas.α_gather * cr
        stages = {
            'fft':     _fft_moment(cr, base, fft_inloop, alphas, n_bc=n_bc),
            'zct':     _zct_moment(cr, base, alphas),
            'reshard': _reshard_moment(cr, base, alphas),
            'solve':   base + 2 * m_zcol + m_solve_per_q,    # q_batch=1 in AOT
            'gather':  base + m_zcol + m_gather_per_q,        # q_gather=1 for sizing
        }
        # k_batch sizing — uses per-k FFT cost to pack as many k's as
        # headroom allows.  Only matters for centroid-load FFT, which is a
        # separate pre-loop stage.  The k_batch sizing models a SINGLE bc
        # iter (one centroid load), not the cumulative bc-unroll, so this
        # term keeps the legacy 1× α_psi_Y_bc.
        fft_per_k = _FFT_COPIES * _bytes_c128(bpd, ns, nr) + _bytes_c128(nr)
        fft_head = m_budget - base - (2 * alphas.α_pair + alphas.α_psi_Y_bc) * cr
        k_batch = 1
        if fft_per_k > 0 and fft_head > fft_per_k:
            k_batch = min(int(nk), max(1, int(fft_head * 0.5 / fft_per_k)))
        return stages, m_zcol, m_solve_per_q, m_gather_per_q, k_batch

    def _find_r_chunk(fft_inloop, override=None):
        """Compute optimal cr from closed-form moment inversion.

        Persistent device residency: centroids (L + R copies on X, Y shards)
        + L_q (sharded).  The ψ(G) cache is host-only now and pulled per-bc
        via io_callback, so it doesn't enter ``base``.
        """
        base = m_centroids + m_L_q
        headroom = m_budget - base
        if headroom <= alphas.c_solve:
            return None
        # bc-loop is Python-unrolled inside the fit_one_rchunk kernel;
        # n_bc copies of psi_bc_Y stack across XLA's fused trace.
        n_bc = max(1, -(-int(nb) // int(band_chunk)))
        if override and override > 0:
            cr = min(int(override), int(nr))
        else:
            limits = _max_cr_per_stage(
                headroom, fft_inloop, alphas,
                nr_max=nr, m_budget=m_budget, m_zct_cap=m_zct_cap,
                n_bc=n_bc,
            )
            cr = min(int(nr), max(0, int(min(limits.values()))))
        pt = p_x * p_y  # cr must be divisible by p_total for solve sharding
        if pt > 1 and cr > 0:
            cr -= cr % pt
        if cr <= 0:
            return None
        stages, m_zcol, m_spq, m_gpq, k_batch = _eval_stages(cr, base, fft_inloop, n_bc)
        return {
            'chunk_r': cr, 'peak': max(stages.values()),
            'bottleneck': max(stages, key=stages.get), 'stages': stages,
            'base': base, 'zcol': m_zcol, 'solve_per_q': m_spq,
            'gather_per_q': m_gpq, 'k_batch': k_batch,
            'cache_bytes': 0.0,  # host-only now
        }

    # ---- Solve for chunk sizes: try cache→no-cache, halving band_chunk if needed ----
    # In-loop band-FFT tensor lives as (nk, bpd, ns, nx, ny, nz) — FULL nk,
    # no k-chunking inside ``get_sharded_wfns_rchunk_slice``.  XLA holds
    # FFT_COPIES concurrent copies (input, output, cuFFT scratch) in a
    # single preallocated-temp slot.  The old ``fft_per_k = FFT_COPIES ·
    # _mem(bpd, ns, nr)`` dropped the nk factor, assuming k-chunking that
    # never happens — caused Si 10×10×10 4-GPU to under-predict peak by
    # ~19 GiB (measured in module_0147.jit__kernel memory-usage-report:
    # top preallocated-temp = 18.54 GiB = 3 × (nk·bpd·ns·nr · 16)).
    # In-loop band-FFT workspace: query XLA directly for the exact per-rank
    # peak its emitted FFT thunk would allocate, including cuFFT's planner-
    # dependent scratch.  Pure-JAX, platform-portable, cached per shape — see
    # common/fft_helpers.query_fft_peak_bytes.  Shape + sharding match the
    # actual in-loop FFT in get_sharded_wfns_rchunk_slice: (nk, bpd, ns, nx,
    # ny, nz) sharded on P(None, ('x','y'), None, None, None, None).
    from common.fft_helpers import query_fft_peak_bytes

    def _fft_inloop_bytes(band_chunk_val):
        # Pass the UNSHARDED full shape — query_fft_peak_bytes applies the
        # sharding to compute the per-rank peak.  Band axis is axis 1 of the
        # 6-D tensor (nk, band_chunk, ns, nx, ny, nz); sharded on ('x','y')
        # over p_x·p_y = p ranks so each rank sees bpd = band_chunk/p bands.
        ix, iy, iz = meta.fft_grid
        return query_fft_peak_bytes(
            input_shape=(int(nk), int(band_chunk_val), int(ns),
                         int(ix), int(iy), int(iz)),
            fft_axes=(-3, -2, -1),
            sharding=NamedSharding(
                mesh_xy, P(None, ('x', 'y'), None, None, None, None)),
            dtype=jnp.complex128,
        )

    fft_inloop = _fft_inloop_bytes(band_chunk)
    result = _find_r_chunk(fft_inloop, r_chunk_override)
    while result is None and bpd > 1:
        bpd = max(1, bpd // 2)
        band_chunk = min(nb, bpd * p)
        bpd = max(1, -(-band_chunk // p))
        fft_inloop = _fft_inloop_bytes(band_chunk)
        if verbose:
            print(f"    Reducing band_chunk to {band_chunk} (bands/device={bpd})")
        result = _find_r_chunk(fft_inloop, r_chunk_override)
    if result is None:
        raise ValueError("Unable to find r-chunk that fits the memory budget.")

    # ---- q_chunk (solve) and q_gather (H5 write): same linear model, different base ----
    base, m_zcol = result['base'], result['zcol']
    avail_solve = max(0.0, m_budget - base - 2 * m_zcol)
    q_chunk = max(1, min(int(nq), int(avail_solve / result['solve_per_q']))) if result['solve_per_q'] > 0 else 1
    avail_gather = max(0.0, m_budget - base - m_zcol)
    q_gather = max(1, min(int(nq), int(avail_gather / result['gather_per_q']))) if result['gather_per_q'] > 0 else int(nq)

    # ---- Overall peak across all stages (chunk loop + pre-loop) ----
    overall_peak = max(
        result['peak'],
        base + 2 * m_zcol + q_chunk * result['solve_per_q'],
        base + m_zcol + q_gather * result['gather_per_q'],
        peak_centroid_fft_stage, m_centroids_full + m_centroids, stage_cct)

    k_chunk = result['k_batch']
    if k_chunk < int(nk) and verbose:
        print(f"    K-point chunking: {k_chunk} k-pts per FFT batch (total {int(nk)})")

    return {
        'band_chunk': band_chunk,
        'chunk_r': result['chunk_r'],
        'q_chunk': q_chunk,
        'q_gather': q_gather,
        'k_chunk': k_chunk,
        'memory_estimate': {
            'peak_estimate_gb': overall_peak / 1e9,
            'budget_gb': memory_budget_gb,
            'bottleneck': result['bottleneck'],
            'available_vcoul_gb': max(0.0, m_budget - m_centroids) / 1e9,
            'limit_info': {k: v / 1e9 for k, v in result['stages'].items()},
        },
    }


def _apply_aot_chunk_model(
    chunks: dict, cfg, meta, mesh_xy, *,
    band_range_left: tuple[int, int],
    band_range_right: tuple[int, int],
    print_fn, rank0: bool,
) -> float | None:
    """Run the AOT chunk-chooser, optionally overriding ``chunks`` in place.

    Two modes:

    - ``cfg.memory.use_aot_chunk_chooser=True``: the AOT chooser picks
      ``chunk_r`` and ``band_chunk``; the heuristic's existing values are
      overridden.  ``LORRAX_CHOOSER_MODE=analytic`` swaps the regressed-fit
      analytic chooser in for the default 20/80 heuristic.
    - ``cfg.memory.use_aot_chunk_chooser=False``: the AOT *predictor* runs
      alongside the heuristic and prints its predicted peak — used for γ
      calibration against the runtime nvidia-smi sample taken inside
      ``fit_zeta``.

    Returns the AOT-predicted peak in GB (or ``None`` if the AOT path
    isn't importable; the heuristic still drives sizing in that case).
    """
    mem = cfg.memory
    try:
        from gw.aot_memory_model import (
            predict_kernel_peak, SysDims, MeshSpec, Knobs,
            choose_chunks_analytic, choose_chunks_heuristic,
            describe_chunks,
        )
    except Exception as exc:
        if mem.use_aot_chunk_chooser and rank0:
            print_fn(
                f"    AOT chooser FAILED ({exc!r}); falling back to heuristic."
            )
        return None

    p_x, p_y = mesh_xy.devices.shape
    # n_b = UNION range (bytes of psi_G cache / band-chunk FFT).
    # n_b_sum = nb_L + nb_R (L+R pair-density work + centroid bytes).
    # Symmetric production GW has nb_L == nb_R == nb_full so n_b_sum == 2·n_b;
    # asymmetric windows (pseudobands, sub-valence, extra-cond) are handled.
    nb_L = band_range_left[1] - band_range_left[0]
    nb_R = band_range_right[1] - band_range_right[0]
    nb_full = (max(band_range_left[1], band_range_right[1])
               - min(band_range_left[0], band_range_right[0]))
    aot_sys = SysDims(
        kgrid=tuple(meta.kgrid),
        fft_grid=tuple(meta.fft_grid),
        n_rmu=int(meta.n_rmu),
        n_s=int(meta.nspinor),
        n_b=int(nb_full),
        n_b_sum=int(nb_L + nb_R),
        n_r=int(meta.n_rtot),
    )
    aot_mesh = MeshSpec(p_x=int(p_x), p_y=int(p_y))

    if mem.use_aot_chunk_chooser:
        # 20/80 heuristic is the default — no DoE deps.  Falls back to the
        # regressed-fit analytic chooser when LORRAX_CHOOSER_MODE=analytic.
        chooser_mode = os.environ.get("LORRAX_CHOOSER_MODE", "heuristic")
        budget_bytes = (
            mem.per_device_gb * 1e9 * mem.chunk_target_utilization
        )
        if chooser_mode == "analytic":
            choice = choose_chunks_analytic(
                aot_sys, aot_mesh, budget_bytes=budget_bytes,
                kernel_name="fit_one_rchunk", tag="current",
            )
        else:
            choice = choose_chunks_heuristic(
                aot_sys, aot_mesh, budget_bytes=budget_bytes,
            )
        if rank0:
            print_fn(f"    {describe_chunks(choice)}")
        # Override chunk_r / band_chunk.  Keep q_chunk, q_gather, k_chunk
        # from the heuristic — the AOT model doesn't cover them yet.
        chunks['chunk_r'] = int(choice.chunk_r)
        chunks['band_chunk'] = int(choice.band_chunk)
        return choice.peak_bytes / 1e9

    # Predict-only path: the AOT model logs its prediction next to the
    # heuristic's pick so γ = runtime / AOT-pred can be computed downstream.
    aot_peak_bytes = predict_kernel_peak(
        "fit_one_rchunk", aot_sys,
        Knobs.of(chunk_r=int(chunks['chunk_r']),
                 band_chunk=int(chunks['band_chunk'])),
        aot_mesh, tag="current",
    )
    if rank0:
        print_fn(
            f"    AOT fit_one_rchunk peak (driver-level): "
            f"{aot_peak_bytes / 1e9:.2f} GB"
        )
    return aot_peak_bytes / 1e9


def get_effective_chunk_size(chunk_size: int) -> int | None:
    """Convert chunk_size flag: -1=None (all bands), 0=auto (64), 1-2048=explicit."""
    if chunk_size == -1:
        return None
    if chunk_size == 0:
        return 64
    if 1 <= chunk_size <= 2048:
        return chunk_size
    raise ValueError(f"chunk_size must be -1, 0, or 1-2048, got {chunk_size}")


# Backward-compatible re-exports
from .gw_config import read_lorrax_input, read_cohsex_input  # noqa: F401


def get_bandranges(nv, nc, nband, nelec):
	r"""Return ranges of bands necessary for \sigma_{X,SX,COH}.

	Legacy helper used by psp/get_DFT_mtxels.py.  GW code uses BandSlices instead.
	"""
	nvrange = [int(nelec - nv), int(nelec)]
	ncrange = [int(nelec), int(nelec + nc)]
	nsigmarange = [int(nelec - nv), int(nelec + nc)]
	n_fullrange = [0, int(nband)]
	n_valrange = [0, int(nelec)]
	return nvrange, ncrange, nsigmarange, n_fullrange, n_valrange


def fit_zeta(wfn, sym, meta, centroid_indices, mesh_xy, cfg, band_slices, tmp_dir,
             psi_rmu_Y, psi_rmuT_X, chunks, print_fn=print):
	"""Fit ISDF interpolation vectors ζ and write to HDF5.

	The caller supplies (a) the full-range centroid wavefunctions
	(``psi_rmu_Y`` / ``psi_rmuT_X``, spanning [b0, b4) as returned by
	``load_centroids_band_chunked``) and (b) the chunk plan from
	:func:`compute_optimal_chunks`.  Returns ``(zeta_h5_path, mem_est)``.
	"""
	from common.isdf_fitting import fit_zeta_chunked_to_h5

	# ISDF left/right band windows (pair density needs asymmetric ranges)
	band_range_left = (band_slices.b0, band_slices.b3)   # all val + sigma cond
	band_range_right = (band_slices.b1, band_slices.b4)   # sigma val + all cond

	mem_est = chunks.get('memory_estimate', {})
	aot_peak_gb = None  # filled in below if the AOT model is available
	_rank0 = (jax.process_index() == 0)
	if _rank0 and mem_est:
		print_fn(f"    Memory estimate: peak {mem_est['peak_estimate_gb']:.2f} GB "
		         f"(budget {mem_est['budget_gb']:.2f} GB), bottleneck={mem_est['bottleneck']}")
		stages = mem_est.get('limit_info', {})
		if stages:
			print_fn(f"    Per-stage: " + "  ".join(f"{k}={v:.2f}" for k, v in stages.items()) + " GB")

	# AOT-derived driver-level peak + (optional) chunk-size override.
	# Two responsibilities folded together for symmetry: when
	# ``cfg.use_aot_chunk_chooser`` is set, the AOT chooser overrides
	# chunk_r / band_chunk; otherwise the AOT model just *predicts* the
	# peak alongside the heuristic for γ-calibration logging.  See
	# ``_apply_aot_chunk_model``.
	#
	# CRITICAL: this block runs on EVERY rank.  Historically the rank-0
	# guard caused mismatched ``band_chunk`` across ranks → mismatched
	# NCCL buffers in ``_fft_gather_reshard`` → hang on the 2nd band
	# chunk.  Only ``print_fn`` is rank-0-only.
	aot_peak_gb = _apply_aot_chunk_model(
		chunks, cfg, meta, mesh_xy,
		band_range_left=band_range_left,
		band_range_right=band_range_right,
		print_fn=print_fn, rank0=_rank0,
	)

	zeta_h5_path = os.path.join(tmp_dir, "zeta_q.h5")
	print_fn(f"\n  Chunked ISDF fitting:")
	print_fn(f"    Band chunks: {chunks['band_chunk']}")
	print_fn(f"    R chunks:    {chunks['chunk_r']} (contiguous r-space)")
	print_fn(f"    Q chunks:    {chunks['q_chunk']}")
	print_fn(f"    Zeta output: {zeta_h5_path}")

	# Band norms for pseudobands normalization (1.0 for deterministic bands)
	_band_norms = getattr(wfn, 'band_norms', None)

	with timing.section("gw_jax.zeta_fit_chunked"), jax_profile.trace_section("zeta_fit"):
		peak_bytes = fit_zeta_chunked_to_h5(
			wfn=wfn, sym=sym, meta=meta,
			centroid_indices=centroid_indices, mesh_xy=mesh_xy,
			chunk_r=chunks['chunk_r'], output_file=zeta_h5_path,
			psi_rmu_Y=psi_rmu_Y, psi_rmuT_X=psi_rmuT_X,
			band_chunk_size=chunks['band_chunk'],
			q_chunk_size=chunks['q_chunk'],
			q_gather_size=chunks.get('q_gather', 0),
			bispinor=cfg.bispinor,
			band_range_left=band_range_left,
			band_range_right=band_range_right,
			k_chunk_size=chunks.get('k_chunk', 0),
			band_norms=_band_norms,
			slab_io_backend=cfg.backend.slab_io,
			gspace_mode=cfg.gspace_mode,
		)

	budget_gb = mem_est.get('budget_gb', cfg.memory.per_device_gb)
	if peak_bytes > 0:
		peak_gb = peak_bytes / 1e9
		print_fn(f"    GPU high-water mark: {peak_gb:.2f} GB / {budget_gb:.2f} GB budget "
		         f"({100 * peak_gb / budget_gb:.0f}%)")
		# γ calibration: runtime peak vs AOT-predicted peak.  γ > 1 means
		# AOT under-predicts (expected for FFT-heavy kernels because
		# cuFFT scratch is invisible to memory_analysis); γ < 1 means
		# AOT over-predicts (often XLA remat triggered at runtime).
		# Logged for manual tracking — wire a CLI to roll these into
		# per-kernel γ calibration once we have enough data.
		if aot_peak_gb is not None and aot_peak_gb > 0:
			gamma = peak_gb / aot_peak_gb
			print_fn(f"    γ (runtime / AOT-pred) = {gamma:.3f}  "
			         f"(AOT predicted {aot_peak_gb:.2f} GB)")

	return zeta_h5_path, mem_est


def compute_V_q(zeta_h5_path, wfn, meta, mesh_xy, cfg, mem_est=None, print_fn=print, bgw_v_grid_fn=None):
	"""Compute bare Coulomb V_qmunu from zeta HDF5 and write G0 back.

	Returns (V_qmunu, G0) where V_qmunu has shape (1, npol, npol, nkx, nky, nkz, μ, μ)
	and G0 is (n_rmu,) ζ_μ(G=0) at q=0.
	"""
	from .compute_vcoul import compute_all_V_q

	if jax.process_index() == 0:
		os.sync()
	jax.experimental.multihost_utils.sync_global_devices("zeta_flush")

	bvec = np.asarray(wfn.blat * wfn.bvec, dtype=np.float64)

	# V_q memory budget (per rank).  The unified V_q tile chooser inside
	# ``compute_all_V_q`` derives q_chunk and μ_chunk from this budget and
	# the system geometry (n_rmu, n_G, n_rtot, mesh shape) directly — no
	# need for the caller to pre-compute legacy μ/q-batch hints.
	if mem_est is None:
		mem_est = {}
	budget_gb = float(mem_est.get('available_vcoul_gb', cfg.memory.per_device_gb))
	try:
		from common.gpu_utils import get_device_memory_info
		budget_gb = min(budget_gb, float(get_device_memory_info().get('budget_gb', budget_gb)))
	except Exception:
		pass
	m_budget = max(0.1, budget_gb) * 1e9

	# Default to ecutwfc — matches BGW's screened_coulomb_cutoff convention
	# (BGW truncates the pair-density v(q+G) sphere at ecutwfc).  The previous
	# default of 4·ecutwfc was the strict mathematical cutoff for ψ*ψ but
	# produced a ~4% V_μν body offset vs BGW's vcoul.dat on Si 4×4×4.
	if cfg.head.bare_coulomb_cutoff is None:
		vcoul_cutoff_ry = float(wfn.ecutwfc)
		print_fn(f"    V_q bare cutoff: {vcoul_cutoff_ry:.1f} Ry (auto: ecutwfc)")
	else:
		vcoul_cutoff_ry = float(cfg.head.bare_coulomb_cutoff)
		print_fn(f"    V_q bare cutoff: {vcoul_cutoff_ry:.1f} Ry")

	print_fn(f"    V_q budget:    {budget_gb:.2f} GB")

	from file_io.slab_io import SlabIO
	# Single dispatcher routes to the unified V_q tile driver (one inner
	# kernel handles same/distinct ζ × Case A/B; one outer driver handles
	# both PHDF5/FFI and h5py-allgather backends).
	with timing.section("gw_jax.V_q_compute"), jax_profile.trace_section("V_q_compute"):
		with SlabIO(zeta_h5_path, mode='r', mesh=mesh_xy,
		            backend=cfg.backend.slab_io) as zeta_io:
			with mesh_xy:
				V_q_raw, G0_all = compute_all_V_q(
					zeta_io,
					kgrid=meta.kgrid, fft_grid=meta.fft_grid,
					bvec=bvec, cell_volume=meta.cell_volume,
					mesh_xy=mesh_xy,
					n_rmu=meta.n_rmu, n_rtot=meta.n_rtot,
					sys_dim=meta.sys_dim,
					bdot=np.asarray(wfn.bdot, dtype=np.float64)
						if meta.sys_dim == 0 else None,
					mc_average_vcoul_body=cfg.head.mc_average_vcoul_body,
					bare_coulomb_cutoff=vcoul_cutoff_ry,
					bgw_v_grid_fn=bgw_v_grid_fn,
					budget_bytes=m_budget,
				)

	# Write G0 = ζ_μ(G=0) at q=0 back to zeta file via SlabIO's deferred
	# attr path (small; rank-0-only after MPI-IO file is closed).
	G0_gathered = jax.experimental.multihost_utils.process_allgather(G0_all)
	if G0_gathered.ndim == 5 and G0_gathered.shape[0] == 1:
		G0_gathered = G0_gathered[0]
	if jax.process_index() == 0:
		with h5py.File(zeta_h5_path, 'a') as f:
			if 'g0_mu' in f:
				del f['g0_mu']
			f.create_dataset('g0_mu', data=np.asarray(G0_gathered))
	jax.experimental.multihost_utils.sync_global_devices("g0_write")

	# Add polarization axes: (nkx, nky, nkz, μ, μ) → (1, npol, npol, nkx, nky, nkz, μ, μ)
	nkx, nky, nkz = meta.kgrid
	V_qmunu = jnp.array(jnp.broadcast_to(
		V_q_raw[None, None, None], (1, meta.npol, meta.npol, nkx, nky, nkz, meta.n_rmu, meta.n_rmu)))

	G0 = G0_gathered
	while G0.ndim > 1:
		G0 = G0[0]

	print_fn(f"\n  V_q computed:")
	print_fn(f"    Shape: {V_qmunu.shape}")
	print_fn(f"    V_q=0 trace: {jnp.trace(V_q_raw[0, 0, 0]).real:.4f}")
	return V_qmunu, G0


def build_wavefunction_bundle(
	wfn, sym, meta, band_slices, mesh_xy,
	*, psi_rmu_Y, psi_rmuT_X, enk_full=None, print_fn=print,
):
	"""Build 4-copy Wavefunctions bundle from the two centroid-sampled
	arrays produced by ``load_centroids_band_chunked``.
	"""
	from .wavefunction_bundle import build_wavefunctions
	from common.load_wfns import get_enk_bandrange

	if enk_full is None:
		enk_full, _ = get_enk_bandrange(
			wfn, sym, band_slices.full_range,
			(band_slices.b1, band_slices.b3), nspinor=meta.nspinor)

	wfns = build_wavefunctions(
		psi_rmu_Y, psi_rmuT_X,
		enk_full=enk_full, slices=band_slices, mesh_xy=mesh_xy)

	print_fn(f"  Wavefunctions built (b0:b4={band_slices.nb_full} bands, "
	         f"4 sharded copies: xn/xr/yr/yn)")
	return wfns


def prepare_isdf_and_wavefunctions(
	*, cfg, wfn, sym, meta, centroid_indices, band_slices,
	mesh_xy, tmp_dir, tensors_filename, print0, bgw_v_grid_fn=None, **_ignored,
):
	"""ISDF pipeline (non-restart path reads top-to-bottom):

	  1. ``compute_optimal_chunks`` → chunk plan (band/r/q chunk sizes).
	  2. ``load_centroids_band_chunked`` → ψ at centroids for [b0, b4).
	  3. ``fit_zeta`` → ζ.h5 (consumes ψ slices for pair density).
	  4. ``compute_V_q`` → V_qmunu, G0 (reads ζ from disk).
	  5. Flush V_q / G0 / enk + W0 placeholder to restart H5 (mode="w").
	  6. ``build_wavefunctions`` → 4-copy Wavefunctions bundle (reuses ψ).
	  7. Append ``psi_full_y`` (= wfns.psi_yr) to restart H5 (mode="a").

	Returns SimpleNamespace(V_qmunu, wf_bundle).
	"""
	from file_io import write_restart_state_to_h5, save_restart_state_per_proc
	from common.load_wfns import load_centroids_band_chunked

	if not cfg.restart:
		from common.load_wfns import get_enk_bandrange

		with mesh_xy:
			# Plan chunks (band/r/q sizes).
			mem = cfg.memory
			chunks = compute_optimal_chunks(
				meta, mesh_xy,
				memory_budget_gb=mem.per_device_gb,
				target_utilization=mem.chunk_target_utilization,
				n_b_left=band_slices.b3 - band_slices.b0,
				n_b_right=band_slices.b4 - band_slices.b1,
				r_chunk_override=mem.r_chunk_override if mem.r_chunk_override > 0 else None,
				zct_stage_cap_gb=mem.zct_stage_cap_gb,
			)

			# Load centroid ψ once for the full [b0, b4) range; reused by
			# both the zeta fit (sliced into halves internally) and the
			# downstream Wavefunctions bundle.
			with timing.section("gw_jax.load_centroid_wfns"):
				psi_rmu_Y, psi_rmuT_X = load_centroids_band_chunked(
					wfn, sym, meta, centroid_indices, cfg.bispinor, mesh_xy,
					band_range=band_slices.full_range,
					band_chunk_size=chunks['band_chunk'],
				)

			zeta_path, mem_est = fit_zeta(
				wfn, sym, meta, centroid_indices, mesh_xy,
				cfg, band_slices, tmp_dir,
				psi_rmu_Y, psi_rmuT_X, chunks, print_fn=print0)
			V_qmunu, G0 = compute_V_q(
				zeta_path, wfn, meta, mesh_xy, cfg,
				mem_est=mem_est, print_fn=print0,
				bgw_v_grid_fn=bgw_v_grid_fn)

			enk_full, _ = get_enk_bandrange(
				wfn, sym, band_slices.full_range,
				(band_slices.b1, band_slices.b3), nspinor=meta.nspinor)

			# Flush V_q / G0 / enk + W0 placeholder immediately.
			write_restart_state_to_h5(
				tensors_filename,
				V_qmunu=V_qmunu, G0_mu_nu=G0, enk_full=enk_full,
				init_W0=True, mesh=mesh_xy, backend=cfg.backend.slab_io,
				mode="w",
			)

			with timing.section("gw_jax.wavefunction_setup"):
				wfns = build_wavefunction_bundle(
					wfn, sym, meta, band_slices, mesh_xy,
					psi_rmu_Y=psi_rmu_Y, psi_rmuT_X=psi_rmuT_X,
					enk_full=enk_full, print_fn=print0)

			# Append ψ to the now-open restart file.
			write_restart_state_to_h5(
				tensors_filename,
				psi_full_y=wfns.psi_yr, mesh=mesh_xy,
				backend=cfg.backend.slab_io, mode="a",
			)
		save_restart_state_per_proc(
			os.path.join(tmp_dir, "isdf_tensors"),
			V_qmunu, None, wfns.psi_yr, wfns.enk, meta, mesh_xy)
		V_qmunu.block_until_ready()
		print0("  Chunked ISDF path complete")
	else:
		from file_io import load_restart_state_from_h5
		with timing.section("gw_jax.restart_load"):
			rs = load_restart_state_from_h5(
				tensors_filename, mesh_xy, band_slices=band_slices)
			V_qmunu = rs.V_qmunu
			print0("  Loaded restart tensors from H5.")
			wfns = build_wavefunction_bundle(
				wfn, sym, meta, band_slices, mesh_xy,
				psi_rmu_Y=rs.psi_rmu_Y, psi_rmuT_X=rs.psi_rmuT_X,
				enk_full=rs.enk_full, print_fn=print0)

	return SimpleNamespace(V_qmunu=V_qmunu, wf_bundle=wfns)

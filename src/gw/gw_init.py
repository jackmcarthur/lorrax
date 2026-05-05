"""ISDF fitting orchestration and memory-aware chunk sizing for LORRAX GW.

  compute_optimal_chunks — per-device memory model for the 6-stage ISDF pipeline
  fit_zeta / compute_V_q / build_wavefunction_bundle — pipeline steps
  prepare_isdf_and_wavefunctions — top-level orchestrator called by main()
"""
import os
import math
from types import SimpleNamespace

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import numpy as np
import h5py

import common.timing as timing
from common import jax_profile



def compute_optimal_chunks(
    meta, mesh_xy, memory_budget_gb: float,
    target_utilization: float = 0.80,
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

    B = 16.0  # bytes per complex128
    p_x, p_y = mesh_xy.devices.shape
    p = p_x * p_y
    nb = meta.band_edges[4] - meta.band_edges[0]  # b4 - b0
    nb_l = int(n_b_left) if n_b_left is not None else nb
    nb_r = int(n_b_right) if n_b_right is not None else nb
    nk, ns, mu, nq, nr = float(meta.nk_tot), float(meta.nspinor), float(meta.n_rmu), float(meta.nk_tot), float(meta.n_rtot)
    nx, ny, nz = meta.fft_grid

    def _mem(*dims, shard=1):
        """complex128 array bytes: 16 · ∏(dims) / shard."""
        result = B
        for d in dims:
            result *= d
        return result / shard

    m_budget = memory_budget_gb * 1e9 * target_utilization
    m_zct_cap = float(zct_stage_cap_gb) * 1e9 if zct_stage_cap_gb and zct_stage_cap_gb > 0 else None

    # ---- Persistent arrays (always resident during chunk loop) ----
    # Centroids: left/right × X-sharded + Y-sharded copies.
    # "full" = during initial load (all nb bands); "persist" = after slicing to nb_l + nb_r.
    m_centroids_full = _mem(nk, ns, mu, nb, shard=p_y) + _mem(nk, ns, mu, nb, shard=p_x)
    m_centroids = _mem(nk, ns, mu, nb_l + nb_r, shard=p_y) + _mem(nk, ns, mu, nb_l + nb_r, shard=p_x)
    if m_centroids_full + m_centroids > m_budget:
        raise ValueError(
            f"Centroid storage requires {(m_centroids_full + m_centroids)/1e9:.2f} GB/device "
            f"but only {memory_budget_gb:.2f} GB allocated.")

    # ---- Band-chunked FFT (centroid extraction, runs before chunk loop) ----
    # ψ(G) → IFFT → ψ(r).  FFT_COPIES=3 is the nominal lower bound (input +
    # output + one scratch); the *actual* per-rank peak (incl. cuFFT plan-
    # specific workspace) is measured exactly via query_fft_peak_bytes
    # below when we have a candidate bpd.  This pre-loop sizing loop uses
    # the nominal 3× to set bpd_max.
    FFT_COPIES = 3
    m_phase = _mem(nk, nr)
    headroom_fft = m_budget - m_centroids_full
    if headroom_fft <= m_phase:
        raise ValueError("Insufficient memory for even a single-band FFT chunk.")
    one_band = FFT_COPIES * _mem(nk, ns, nr)
    bpd_max = max(1, int((headroom_fft - m_phase) / one_band))
    band_chunk = min(nb, bpd_max * p)
    bpd = max(1, -(-band_chunk // p))  # ceil div matching read_Gvecs_to_devices
    peak_centroid_fft_stage = m_centroids_full + FFT_COPIES * _mem(nk, bpd, ns, nr) + m_phase

    # ---- C_q build (pair density → C_q → L_q, runs before chunk loop) ----
    stage_cct = m_centroids + 2 * _mem(nk, mu, mu, shard=p_x * p_y) + _mem(nq, mu, mu, shard=p_x * p_y)
    if stage_cct > m_budget:
        raise ValueError(f"C_q build requires {stage_cct/1e9:.2f} GB/device — exceeds budget.")
    m_L_q = _mem(nq, mu, mu, shard=p_x * p_y)  # L_q persists; same size as C_q

    # ψ(G) lives on HOST (per-rank band-sharded) and is fetched into the
    # jit one bc at a time via io_callback; no persistent device residency.
    # Only the currently active bc's ψ(G) is on device — captured in
    # the fft_moment term below.

    # ======================================================================
    #  Chunk-loop memory model — five moments, each grounded in a buffer
    #  visible in the fit_one_rchunk HLO dump.
    #
    #  ψ(G) arrives via io_callback, gets FFT'd to ψ(r)-box, resharded to
    #  an r-chunk slab ψ_bc_Y, then einsum'd into P_l/P_r accumulators.
    #  ZCT is 2× ifftn + 1× fftn over kgrid on the pair-density tensor.
    #  Then reshard Z_q → Z_col → solve L_q · zeta = Z_col, per-q-batch.
    #  Finally q_gather copies are replicated + written to H5.
    # ======================================================================
    α_pair     = _mem(nk, mu, shard=p_x * p_y)      # P_l/P_r per unit cr
    α_psi_Y_bc = _mem(nk, band_chunk, ns, shard=p_y) # ψ_bc_Y per unit cr, post-reshard (Y-sharded on r-axis)
    α_zcol     = _mem(nq, mu, shard=p)              # Z_col per unit cr
    α_z_slice  = _mem(mu, shard=p)                  # per-q Z-slice per cr
    α_gather   = _mem(mu, shard=p) + 2 * _mem(mu)   # q-gather: sharded + replicated copy + NCCL

    # Per-bc transient ψ(G) slab (one bc on device during its FFT only):
    m_psi_G_bc = _mem(nk, band_chunk, ns, nr, shard=p)
    # Replicated-L temps during Cholesky solve (L_batch_rep + 3× L_full):
    c_solve    = _mem(mu, mu, shard=p_x * p_x) + 3 * _mem(mu, mu)

    # ZCT adds N·α_pair·cr on top of the persistent P_l/P_r accumulators.
    # CALIBRATION (2026-05-03, MoS2 r_chunk sweep + CrI3 16-GPU spot
    # checks; see runs/MoS2/B_bispinor_profile/sweep_results.json):
    #
    #   The previous value (3) was an AOT-isolated-kernel measurement of
    #   "temp+out" buffers visible inside compute_ZCT_from_left_right_zchunk,
    #   but XLA donates and reuses buffers inside the larger
    #   fit_one_rchunk fusion — at runtime only ~1 ZCT-internal buffer
    #   stays concurrent with the P_l, P_r accumulators.  Setting
    #   ZCT_ADDITIONAL_COEF=3 gave heur=10.65 GB at MoS2 cr=46080 vs
    #   measured 2.80 GB (3.8× over-prediction); on CrI3 16-GPU
    #   r_chunk=25000 it gave 45.95 GB vs measured 5.91 GB (~8× over-
    #   prediction) and tripped the chooser into picking r_chunk=4992
    #   (sub-optimal performance).
    #
    #   The fitted slope across MoS2 cr={1k…46k} sweep is closer to
    #   1×α_pair (saturation regime, single concurrent ZCT buffer);
    #   on CrI3 16-GPU it rises to ~2.7×α_pair (more aggressive
    #   pipelining at 16-rank scale).  Setting ZCT_ADDITIONAL_COEF=1
    #   (total 2+1=3·α_pair·cr) gives:
    #     MoS2  cr=46080: heur 3.21 GB vs measured 2.80 GB (1.15×)
    #     CrI3  cr=25000: heur ≈ 7.6 GB vs measured 5.91 GB (1.29×)
    #   both within 30 % across the full sweep.
    ZCT_ADDITIONAL_COEF = 1

    def _fft_moment(cr, base, fft_inloop_bytes):
        """Peak during one bc-iteration's FFT + reshard + accumulate.

        Live simultaneously:
          - base (persistent centroids + L_q)
          - 2 pair-density accumulators (P_l, P_r) — kept across bc-loop
          - FFT workspace for the wfn FFT (input + output + cuFFT scratch,
            all captured by ``fft_inloop_bytes`` from query_fft_peak_bytes)
          - post-reshard ψ_bc_Y slab — still live until the pair einsum
            folds it in (α_psi_Y_bc · cr)
        """
        return base + 2 * α_pair * cr + fft_inloop_bytes + α_psi_Y_bc * cr

    def _zct_moment(cr, base):
        """Peak during the ZCT stage.

        Live simultaneously:
          - base (persistent)
          - 2 pair-density accumulators (P_l, P_r — persistent across bc-loop)
          - 3 additional pair-density-sized buffers for ZCT's internal
            working set + Z_q output (AOT-measured, see ZCT_ADDITIONAL_COEF)
        """
        return base + (2 + ZCT_ADDITIONAL_COEF) * α_pair * cr

    def _reshard_moment(cr, base):
        """Peak during the Z_q → Z_col reshard.

        Z_col has size α_zcol·cr; the reshard needs input + output.
        FUDGE: coefficient 3 covers (input + output + NCCL scratch).
        TODO: AOT-measure an isolated reshard jit to calibrate.
        """
        return base + 3 * α_zcol * cr

    def _solve_moment(cr, base, q_batch):
        """Peak during per-q-batch triangular solve.

        Z_col (input) + zeta_acc (donation-compatible output, same size).
        Per-q-batch replicated L slab lives inside the solve jit.
        """
        return base + 2 * α_zcol * cr + q_batch * (2 * α_z_slice * cr + c_solve)

    def _gather_moment(cr, base, q_gather):
        """Peak during q-gather + H5 write.

        Z_col sharded + q_gather replicated copies (one per q being
        written) + NCCL scratch.
        """
        return base + α_zcol * cr + q_gather * α_gather * cr

    def _max_cr(headroom, fft_cost_per_k):
        """Invert each MOMENT's linear cost to get max feasible cr.

        One inversion per distinct live-range snapshot.  The bc-loop's
        ``fft_moment`` and ``zct_moment`` are the two real in-loop peaks;
        reshard / solve / gather are post-loop moments with their own
        linear cr-dependence.
        """
        # fft_moment: base + 2·α_pair·cr + fft_inloop + α_psi_Y_bc·cr
        #   → cr ≤ (headroom - fft_inloop) / (2·α_pair + α_psi_Y_bc)
        denom_fft = 2 * α_pair + α_psi_Y_bc
        # zct_moment: base + (2 + ZCT_ADDITIONAL_COEF)·α_pair·cr
        denom_zct = (2 + ZCT_ADDITIONAL_COEF) * α_pair
        limits = {
            'fft':     (headroom - fft_cost_per_k) / denom_fft if denom_fft > 0 else nr,
            'zct':     headroom / denom_zct if denom_zct > 0 else nr,
            'reshard': headroom / (3 * α_zcol) if α_zcol > 0 else nr,
            # solve/gather live post-loop and have extra per-q terms; size
            # them against their cr-linear part (the 2·α_zcol + α_zcol
            # components dominate; per-q batch costs are capped below).
            'solve':   (headroom - c_solve) / (2 * α_zcol + 2 * α_z_slice) if (2 * α_zcol + 2 * α_z_slice) > 0 else nr,
            'gather':  headroom / (α_zcol + α_gather) if (α_zcol + α_gather) > 0 else nr,
        }
        # Optional soft cap on zct stage (env override for tight-memory systems)
        if m_zct_cap is not None and α_pair > 0:
            zct_headroom = m_zct_cap - (m_budget - headroom)
            if zct_headroom > 0:
                limits['zct'] = min(limits['zct'], zct_headroom / denom_zct)
        return limits

    # Per-k centroid-load FFT cost — used only for k_batch sizing.  The
    # in-loop FFT workspace cost is queried separately via
    # query_fft_peak_bytes (see below).
    fft_per_k = FFT_COPIES * _mem(bpd, ns, nr) + _mem(nr)

    def _eval_stages(cr, base, fft_inloop):
        """Forward-evaluate each moment at a given cr."""
        m_zcol = α_zcol * cr
        m_solve_per_q = 2 * α_z_slice * cr + c_solve
        m_gather_per_q = α_gather * cr
        # Note: fft_inloop already includes the input (=m_psi_G_bc), output,
        # and cuFFT scratch via query_fft_peak_bytes.  Don't add m_psi_G_bc
        # on top — that was double-counting the argument buffer.
        stages = {
            'fft':     _fft_moment(cr, base, fft_inloop),
            'zct':     _zct_moment(cr, base),
            'reshard': _reshard_moment(cr, base),
            'solve':   base + 2 * m_zcol + m_solve_per_q,  # q_batch=1 in AOT
            'gather':  base + m_zcol + m_gather_per_q,      # q_gather=1 for sizing
        }
        # k_batch sizing — uses per-k FFT cost to pack as many k's as
        # headroom allows.  Kept from the old model; only matters for
        # centroid-load FFT, which is a separate pre-loop stage.
        fft_head = m_budget - base - (2 * α_pair + α_psi_Y_bc) * cr
        k_batch = 1
        if fft_per_k > 0 and fft_head > fft_per_k:
            k_batch = min(int(nk), max(1, int(fft_head * 0.5 / fft_per_k)))
        return stages, m_zcol, m_solve_per_q, m_gather_per_q, k_batch

    def _find_r_chunk(use_cache, fft_inloop, override=None):
        """Compute optimal cr from direct formula, round down to p-divisible.

        Persistent device residency at the fit_one_rchunk call site:
        centroids (L + R copies on X, Y shards) + L_q (sharded).  The
        ψ(G) cache is host-only now and pulled per-bc via io_callback,
        so it doesn't enter ``base``.  ``use_cache`` is retained for
        callsite symmetry but no longer affects memory sizing (left for
        the caller to decide ``gspace_mode``).
        """
        del use_cache  # no longer drives memory sizing
        base = m_centroids + m_L_q
        # CALIBRATION: subtract the runtime-overhead floor (allocator pool
        # + NCCL/cuFFT buffers + persistent jit caches; cuFFT plan-cache
        # growth across r-chunks; XLA fragmentation that prevents finding
        # contiguous space for the next-call's largest single buffer) from
        # the per-cr headroom so the auto-picked cr leaves room for the
        # overhead too.  Without this, auto-picked cr saturates the budget
        # and a downstream OOM trips on a single-buffer alloc.
        #
        # 2026-05-04: scaled with the budget after observing CrI3 60 Ry
        # bispinor OOMs at both 8 GPU (chunker pred=29.1 GB, OOM at single
        # 25.4 GiB alloc) and 16 GPU (chunker pred=29.1 GB, OOM at 17.7
        # GiB single alloc).  XLA needs free contiguous memory for the
        # next call's largest transient *on top of* the current peak —
        # 5 % of the budget covers it across the scales we've measured.
        _RUNTIME_OVERHEAD_FOR_HEADROOM = max(0.8e9, 0.05 * memory_budget_gb * 1e9)
        headroom = m_budget - base - _RUNTIME_OVERHEAD_FOR_HEADROOM
        if headroom <= c_solve:
            return None
        if override and override > 0:
            cr = min(int(override), int(nr))
        else:
            cr = min(int(nr), max(0, int(min(_max_cr(headroom, fft_inloop).values()))))
        pt = p_x * p_y  # cr must be divisible by total device count for solve sharding
        if pt > 1 and cr > 0:
            cr -= cr % pt
        if cr <= 0:
            return None
        stages, m_zcol, m_spq, m_gpq, k_batch = _eval_stages(cr, base, fft_inloop)
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

    # ===== Budget split: cap FFT workspace at WFN_WORKSPACE_FRAC of budget =====
    #
    # 2026-05-04: ported the budget-split policy from
    # ``aot_memory_model.choose_chunks_heuristic`` (the AOT-default chooser)
    # into this default heuristic chunker so production runs get the same
    # behavior without the ``use_aot_chunk_chooser`` flag.
    #
    # Why: the prior policy sized ``band_chunk`` against the FULL FFT
    # headroom (``m_budget − m_centroids_full``) and only shrank it when
    # ``_find_r_chunk`` returned None (cr ≤ 0).  At production scale (CrI3
    # 60Ry on 8 GPU) that meant ``fft_inloop`` ate ~22 GB of the 24 GB
    # working budget at band_chunk=16 — feasible by a thread, but it left
    # only ~1.5 GB for the cr-linear stages, forcing chunk_r=4584 (246
    # chunks × 4 bispinor channels = ~16 hour wall time).
    #
    # The new policy reserves ``WFN_WORKSPACE_FRAC × m_budget`` for the
    # FFT workspace (driving ``band_chunk``) and leaves the rest for the
    # cr-linear stages (driving ``chunk_r``).  ``band_chunk`` is halved
    # until ``fft_inloop ≤ wfn_workspace_cap`` or until it hits the
    # mesh-divisibility floor (band_chunk = p, one band per device).
    # Small systems where the FFT is naturally tiny (MoS2 60Ry small)
    # don't notice the cap (it's non-binding); large systems where the
    # FFT was the bottleneck now get a proportional reservation.
    #
    # FRAC=0.30 covers CrI3 60Ry (fft_inloop ≈ 11 GB at band_chunk=8 fits
    # 0.30×24 = 7.2 GB cap if we hit minimum band_chunk; the cap is
    # advisory — we accept the floor when band_chunk = p is the smallest
    # mesh-divisible value).  Lower FRAC reserves more for cr-linear
    # stages but risks band_chunk hitting min ≥ band_chunk_p; higher
    # FRAC keeps band_chunk too coarse and shrinks chunk_r.  See
    # MEMORY_MODEL.md for the calibration data.
    WFN_WORKSPACE_FRAC = 0.30
    wfn_workspace_cap = WFN_WORKSPACE_FRAC * m_budget
    while fft_inloop > wfn_workspace_cap and bpd > 1:
        bpd = max(1, bpd // 2)
        band_chunk = min(nb, bpd * p)
        bpd = max(1, -(-band_chunk // p))
        fft_inloop = _fft_inloop_bytes(band_chunk)
        if verbose:
            print(f"    Capping band_chunk to {band_chunk} (bands/device={bpd}) "
                  f"to keep FFT workspace ≤ {wfn_workspace_cap/1e9:.1f} GB "
                  f"({100*WFN_WORKSPACE_FRAC:.0f}% of budget)")

    result = _find_r_chunk(True, fft_inloop, r_chunk_override) or \
             _find_r_chunk(False, fft_inloop, r_chunk_override)
    while result is None and bpd > 1:
        bpd = max(1, bpd // 2)
        band_chunk = min(nb, bpd * p)
        bpd = max(1, -(-band_chunk // p))
        fft_inloop = _fft_inloop_bytes(band_chunk)
        if verbose:
            print(f"    Reducing band_chunk to {band_chunk} (bands/device={bpd}) "
                  f"to fit cr-linear stages within remaining budget")
        result = _find_r_chunk(True, fft_inloop, r_chunk_override) or \
                 _find_r_chunk(False, fft_inloop, r_chunk_override)
    if result is None:
        raise ValueError("Unable to find r-chunk that fits the memory budget.")

    # ---- q_chunk (solve) and q_gather (H5 write): same linear model, different base ----
    base, m_zcol = result['base'], result['zcol']
    avail_solve = max(0.0, m_budget - base - 2 * m_zcol)
    q_chunk = max(1, min(int(nq), int(avail_solve / result['solve_per_q']))) if result['solve_per_q'] > 0 else 1
    avail_gather = max(0.0, m_budget - base - m_zcol)
    q_gather = max(1, min(int(nq), int(avail_gather / result['gather_per_q']))) if result['gather_per_q'] > 0 else int(nq)

    # CALIBRATION (2026-05-03): the post-loop gather stage's α_gather
    # term claimed 2 × replicated copies of (μ × cr) live concurrently
    # per device, but in practice ``process_allgather`` is a D2H copy
    # — the replicated GPU-side buffer is a fused transient, not a
    # persistent allocation, and ``np.asarray`` immediately pulls it
    # to host.  Empirically the gather stage NEVER drives the peak on
    # nvidia-smi: at MoS2 cr=46080 the heuristic predicted 9·1.09 GB
    # = 9.8 GB for it, but measured peak was 2.80 GB (see
    # sweep_results.json).  Drop the gather contribution from the
    # overall-peak computation; the runtime ``_safe_q_gather`` cap in
    # ``fit_zeta_chunked_to_h5`` already prevents OOM here.
    #
    # CALIBRATION: nvidia-smi memory.used captures the JAX preallocator
    # pool, NCCL ring buffers, cuFFT plan caches, persistent compiled-
    # jit memory, and the kernel working set.  The heuristic above
    # models only the working set; the MoS2 4-GPU sweep shows a
    # consistent ~0.8 GB gap that's independent of cr.  CrI3 60 Ry
    # bispinor at 8/16 GPU showed XLA needing extra contiguous space
    # for the next call's largest transient buffer (~17 GB single alloc)
    # which fragmentation prevents when the persistent peak is too tight.
    # 2026-05-04: scale the floor with budget — 5% of budget tracks the
    # observed 1.5 GB at 30 GB / 0.5 GB at 10 GB pattern.
    RUNTIME_OVERHEAD_BYTES = max(0.8e9, 0.05 * memory_budget_gb * 1e9)

    # CALIBRATION: the post-loop solve runs as a JAX scan over q-points
    # inside one jit body (see ``_solve_zeta_per_q_chunk_with_full_L_q``);
    # only ONE q's Z_slice + L slab is alive at a time inside the scan,
    # not q_chunk simultaneously.  The old formula
    # ``q_chunk * solve_per_q`` over-counted by ~q_chunk× and was the
    # dominant predictor at cr=46080 on MoS2 (5.27 GB heuristic vs
    # actual 2.80 GB).  Use 1× per_q here — matches the scan's live
    # working set.
    overall_peak = max(
        result['peak'],
        base + 2 * m_zcol + result['solve_per_q'],
        peak_centroid_fft_stage, m_centroids_full + m_centroids, stage_cct,
    ) + RUNTIME_OVERHEAD_BYTES

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

	# AOT-derived driver-level peak + chunk-size override.
	#
	# CRITICAL: this block runs on EVERY rank, not just rank 0.  Both
	# ``compute_optimal_chunks`` (above) and the AOT chooser below are
	# deterministic pure-Python on identical inputs, so they yield the
	# same ``chunks`` dict on every rank — but only if we call them on
	# every rank.  Historically this block was wrapped in
	# ``if jax.process_index() == 0``, which caused rank 0 to override
	# ``chunks['band_chunk']`` while ranks 1..p kept the heuristic's
	# pick; the inner ``_fft_gather_reshard`` jit shape depends on
	# ``band_chunk``, so the ranks then posted mismatched NCCL buffers
	# and hung mid-all-gather on the 2nd band chunk.  Keep all control
	# flow that mutates ``chunks`` OUT of the rank-0 guard; only the
	# ``print_fn`` calls are rank-0-only.
	try:
		from gw.aot_memory_model import (
			predict_kernel_peak, SysDims, MeshSpec, Knobs,
			choose_chunks_analytic, choose_chunks_heuristic,
			describe_chunks,
		)
		p_x, p_y = mesh_xy.devices.shape
		# n_b = UNION range (bytes of psi_G cache / band-chunk FFT).
		# n_b_sum = nb_L + nb_R (L+R pair-density work + centroid bytes).
		# Symmetric production GW (nval == nelec, nband == nelec+ncond)
		# has nb_L == nb_R == nb_full so n_b_sum == 2·n_b; asymmetric
		# windows (pseudobands, sub-valence, extra-cond) are picked
		# up correctly here.
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
		if cfg.use_aot_chunk_chooser:
			# 20/80 heuristic is the default — simpler, no DoE deps.
			# Falls back to the regressed-fit analytic chooser if
			# the user sets LORRAX_CHOOSER_MODE=analytic.
			_mode = os.environ.get("LORRAX_CHOOSER_MODE", "heuristic")
			budget = (cfg.memory_per_device_gb * 1e9
			          * cfg.chunk_target_utilization)
			if _mode == "analytic":
				choice = choose_chunks_analytic(
					aot_sys, aot_mesh,
					budget_bytes=budget,
					kernel_name="fit_one_rchunk", tag="current",
				)
			else:
				choice = choose_chunks_heuristic(
					aot_sys, aot_mesh, budget_bytes=budget,
				)
			if _rank0:
				print_fn(f"    {describe_chunks(choice)}")
			# Override the heuristic's chunk_r / band_chunk.  Keep
			# q_chunk, q_gather, k_chunk from the old chooser since
			# the AOT model doesn't cover them yet.
			chunks['chunk_r'] = int(choice.chunk_r)
			chunks['band_chunk'] = int(choice.band_chunk)
			aot_peak_gb = choice.peak_bytes / 1e9
		else:
			aot_peak = predict_kernel_peak(
				"fit_one_rchunk", aot_sys,
				Knobs.of(chunk_r=int(chunks['chunk_r']),
				         band_chunk=int(chunks['band_chunk'])),
				aot_mesh, tag="current",
			)
			if _rank0:
				print_fn(f"    AOT fit_one_rchunk peak (driver-level): "
				         f"{aot_peak / 1e9:.2f} GB")
			aot_peak_gb = aot_peak / 1e9
	except Exception as _aot_exc:
		if cfg.use_aot_chunk_chooser and _rank0:
			print_fn(f"    AOT chooser FAILED ({_aot_exc!r}); "
			         f"falling back to heuristic.")

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
			use_ffi_io=cfg.use_ffi_io,
			gspace_mode=cfg.gspace_mode,
			max_r_chunks=int(getattr(cfg, 'max_r_chunks', -1) or -1),
		)

	budget_gb = mem_est.get('budget_gb', cfg.memory_per_device_gb)
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

	# ── Bispinor: fit ζ^{μ_L=1,2,3} on the current-density centroid set ──
	# Same kernel, swapping in γ̃^i vertex; sequential calls keep peak
	# memory at the scalar-fit level.  Output paths follow the convention
	# zeta_q_mu{1,2,3}.h5 next to the scalar zeta_q.h5.
	if cfg.bispinor and getattr(cfg, 'centroids_file_current', None):
		import dataclasses
		from common.load_wfns import load_centroids_band_chunked
		from file_io.centroids import load_centroids

		print_fn(f"\n  [bispinor] fitting ζ^{{μ_L=1,2,3}} on current-density "
		         f"centroids: {cfg.centroids_file_current}")
		_, cents_curr_idx, n_rmu_curr = load_centroids(
			cfg.centroids_file_current, meta.fft_grid)
		# Round n_rmu_jax to n_proc, matching Meta.from_system convention.
		n_rmu_curr_jax = ((n_rmu_curr + meta.n_proc - 1) // meta.n_proc) * meta.n_proc
		meta_curr = dataclasses.replace(
			meta, n_rmu=int(n_rmu_curr), n_rmu_jax=int(n_rmu_curr_jax))

		with timing.section("gw_jax.load_centroid_wfns_current"):
			psi_curr_rmu_Y, psi_curr_rmuT_X = load_centroids_band_chunked(
				wfn, sym, meta_curr,
				jnp.asarray(cents_curr_idx, dtype=jnp.int32),
				cfg.bispinor, mesh_xy,
				band_range=band_slices.full_range,
				band_chunk_size=chunks['band_chunk'],
			)

		# Caches in isdf_fitting / load_wfns close over tracers from the
		# enclosing jit at first compile.  Re-using those compiled fns from
		# a fresh fit_zeta_chunked_to_h5 invocation triggers
		# UnexpectedTracerError because the closures hold values from a
		# now-closed trace scope.  Clear before each new channel.
		import gc
		from common import isdf_fitting as _isdf, load_wfns as _lw

		def _drop_traced_caches():
			_isdf._fit_one_rchunk_cache.clear()
			_isdf._compute_pair_density_cache.clear()
			_isdf._accum_pair_density_cache.clear()
			_isdf._compute_pair_density_vertex_cache.clear()
			_isdf._accum_pair_density_vertex_cache.clear()
			_lw._rchunk_slice_cache.clear()
			jax.clear_caches()
			gc.collect()

		for mu_L in (1, 2, 3):
			_drop_traced_caches()
			zeta_mu_path = os.path.join(tmp_dir, f"zeta_q_mu{mu_L}.h5")
			print_fn(f"  [bispinor] μ_L={mu_L} → {zeta_mu_path}")
			with timing.section(f"gw_jax.zeta_fit_chunked_mu{mu_L}"), \
			     jax_profile.trace_section(f"zeta_fit_mu{mu_L}"):
				fit_zeta_chunked_to_h5(
					wfn=wfn, sym=sym, meta=meta_curr,
					centroid_indices=jnp.asarray(cents_curr_idx, dtype=jnp.int32),
					mesh_xy=mesh_xy,
					chunk_r=chunks['chunk_r'], output_file=zeta_mu_path,
					psi_rmu_Y=psi_curr_rmu_Y, psi_rmuT_X=psi_curr_rmuT_X,
					band_chunk_size=chunks['band_chunk'],
					q_chunk_size=chunks['q_chunk'],
					q_gather_size=chunks.get('q_gather', 0),
					bispinor=cfg.bispinor,
					band_range_left=band_range_left,
					band_range_right=band_range_right,
					k_chunk_size=chunks.get('k_chunk', 0),
					band_norms=_band_norms,
					use_ffi_io=cfg.use_ffi_io,
					gspace_mode=cfg.gspace_mode,
					vertex_mu_L=mu_L,
					max_r_chunks=int(getattr(cfg, 'max_r_chunks', -1) or -1),
				)

	return zeta_h5_path, mem_est


def compute_V_q(zeta_h5_path, wfn, meta, mesh_xy, cfg, mem_est=None, print_fn=print, bgw_v_grid_fn=None):
	"""Compute bare Coulomb V_qmunu from zeta HDF5 and write G0 back.

	Returns (V_qmunu, G0) where V_qmunu has shape (1, npol, npol, nkx, nky, nkz, μ, μ)
	and G0 is (n_rmu,) ζ_μ(G=0) at q=0.

	When ``cfg.bispinor`` is True, the path branches into the full Lorentz
	V^{μ_L, ν_L}_q tensor (10 non-zero (μ_L, ν_L) tiles, 7 unique kernel
	calls + 3 hermitian-transpose; 6 tiles are zero by Coulomb gauge).
	The return value is a dict in that case — see the bispinor branch
	below for the layout.  Downstream Σ^B integration is not yet wired
	(TODO at the consumer site).
	"""
	from .v_q_driver import (
		compute_all_V_q_from_zeta_h5,
		compute_all_V_q_sharded,
	)

	if jax.process_index() == 0:
		os.sync()
	jax.experimental.multihost_utils.sync_global_devices("zeta_flush")

	bvec = np.asarray(wfn.blat * wfn.bvec, dtype=np.float64)

	# V_q memory model: 2 zeta reads + 1 FFT workspace = 3× (μ × n_G × 16 bytes)
	if mem_est is None:
		mem_est = {}
	budget_gb = float(mem_est.get('available_vcoul_gb', cfg.memory_per_device_gb))
	try:
		from common.gpu_utils import get_device_memory_info
		budget_gb = min(budget_gb, float(get_device_memory_info().get('budget_gb', budget_gb)))
	except Exception:
		pass
	m_budget = max(0.1, budget_gb) * 1e9
	m_per_mu = 3 * 16 * meta.n_rtot
	mu_chunk = max(1, min(meta.n_rmu, int(m_budget / m_per_mu)))
	q_batch = 1
	if mu_chunk >= meta.n_rmu and meta.nk_tot > 1:
		q_batch = max(1, min(4, meta.nk_tot, int(m_budget // max(1.0, 2.0 * 16 * meta.n_rmu * meta.n_rtot))))

	# get_device_memory_info reports this rank's free HBM, which can
	# differ across ranks (pool fragmentation, non-symmetric allocations
	# before this point) → mu_chunk + q_batch can diverge across ranks.
	# compute_all_V_q_from_zeta_h5 then branches on (n_chunks == 1) at
	# line ~884; divergent branches across ranks deadlock on the first
	# collective inside one branch (observed 2026-04-18 w/ FFI read).
	# Fix: broadcast rank 0's numbers to every rank so they agree.
	if jax.process_count() > 1:
		_mq = jnp.asarray([int(mu_chunk), int(q_batch)], dtype=jnp.int64)
		_mq = jax.experimental.multihost_utils.broadcast_one_to_all(_mq)
		mu_chunk = int(jax.device_get(_mq)[0])
		q_batch = int(jax.device_get(_mq)[1])

	# Default to ecutwfc — matches BGW's screened_coulomb_cutoff convention
	# (BGW truncates the pair-density v(q+G) sphere at ecutwfc).  The previous
	# default of 4·ecutwfc was the strict mathematical cutoff for ψ*ψ but
	# produced a ~4% V_μν body offset vs BGW's vcoul.dat on Si 4×4×4.
	if cfg.bare_coulomb_cutoff is None:
		vcoul_cutoff_ry = float(wfn.ecutwfc)
		print_fn(f"    V_q bare cutoff: {vcoul_cutoff_ry:.1f} Ry (auto: ecutwfc)")
	else:
		vcoul_cutoff_ry = float(cfg.bare_coulomb_cutoff)
		print_fn(f"    V_q bare cutoff: {vcoul_cutoff_ry:.1f} Ry")

	print_fn(f"    V_q budget:    {budget_gb:.2f} GB")
	print_fn(f"    V_q mu chunks: {mu_chunk}")
	if q_batch > 1:
		print_fn(f"    V_q q batches: {q_batch}")

	from file_io.slab_io import SlabIO
	# Sharded path = default: ``compute_all_V_q_sharded`` runs each q-batch
	# as one big jit (fit_one_rchunk-style) with sharded FFI reads landing
	# ζ directly in μ-on-('x','y') layout — FFT + gathers + gemm all stay
	# mesh-parallel.  Fallback to the replicated ``compute_all_V_q_from_zeta_h5``
	# when FFI phdf5 isn't available (single-GPU sandbox builds, h5py-only
	# backend).
	use_sharded_v_q = bool(cfg.use_ffi_io)

	if cfg.bispinor:
		# Bispinor V^{μ_L, ν_L}_q — full Lorentz tensor over the four ζ
		# files (zeta_q.h5 + zeta_q_mu{1,2,3}.h5).  Returns a dict
		# keyed by (μ_L, ν_L).  Downstream Σ^B integration not yet
		# wired — see TODO below.
		from .v_q_lorentz import compute_all_V_q_lorentz_sharded
		from .coulomb_kernel import make_v_munu_chunked_kernel

		zeta_dir = os.path.dirname(zeta_h5_path)
		channel_paths = {
			0: zeta_h5_path,
			1: os.path.join(zeta_dir, "zeta_q_mu1.h5"),
			2: os.path.join(zeta_dir, "zeta_q_mu2.h5"),
			3: os.path.join(zeta_dir, "zeta_q_mu3.h5"),
		}
		# Verify every channel file exists before we open file handles.
		for ch, p in channel_paths.items():
			if not os.path.exists(p):
				raise FileNotFoundError(
					f"compute_V_q (bispinor): channel {ch} ζ-file missing: {p}")
		# Centroid counts may differ across channels (scalar = 1800,
		# transverse = 1808 in CrI3 4-density).  SlabIO doesn't expose
		# dataset shape, so peek with h5py rank-0 then broadcast.
		import h5py as _h5py
		_n_rmu_by_ch_local = np.zeros(4, dtype=np.int64)
		if jax.process_index() == 0:
			for ch, p in channel_paths.items():
				with _h5py.File(p, 'r') as _f:
					_n_rmu_by_ch_local[ch] = int(_f['zeta_q'].shape[2])
		_n_rmu_jax = jax.experimental.multihost_utils.broadcast_one_to_all(
			jnp.asarray(_n_rmu_by_ch_local, dtype=jnp.int64))
		n_rmu_by_channel = {ch: int(jax.device_get(_n_rmu_jax)[ch])
		                    for ch in (0, 1, 2, 3)}
		print_fn(f"    V_q (bispinor) channels: n_rmu={n_rmu_by_channel}")

		# Build the Coulomb kernel bundle once — reused for the K_cart
		# helper and v(q+G) closures inside the lorentz driver.
		nkx, nky, nkz = meta.kgrid
		coulomb_kernels = make_v_munu_chunked_kernel(
			meta.fft_grid[0], meta.fft_grid[1], meta.fft_grid[2],
			nkx, nky, nkz,
			bvec, meta.cell_volume, meta.sys_dim,
			bdot=np.asarray(wfn.bdot, dtype=np.float64) if meta.sys_dim == 0 else None,
			mc_average_vcoul_body=cfg.mc_average_vcoul_body,
			vcoul_cutoff_ry=vcoul_cutoff_ry,
		)

		with timing.section("gw_jax.V_q_compute"), jax_profile.trace_section("V_q_compute"):
			# Stack of context managers — open all four ζ SlabIOs.
			ios = []
			try:
				for ch in (0, 1, 2, 3):
					ios.append(SlabIO(channel_paths[ch], mode='r',
					                  mesh=mesh_xy, use_ffi_io=use_sharded_v_q))
				zeta_io_by_channel = {ch: ios[ch] for ch in (0, 1, 2, 3)}
				with mesh_xy:
					V_blocks, G0_all = compute_all_V_q_lorentz_sharded(
						zeta_io_by_channel=zeta_io_by_channel,
						coulomb_kernels=coulomb_kernels,
						mesh_xy=mesh_xy,
						kgrid=meta.kgrid, fft_grid=meta.fft_grid,
						bvec=bvec, cell_volume=meta.cell_volume,
						n_rmu_by_channel=n_rmu_by_channel,
						sys_dim=meta.sys_dim,
						bdot=np.asarray(wfn.bdot, dtype=np.float64) if meta.sys_dim == 0 else None,
						mc_average_vcoul_body=cfg.mc_average_vcoul_body,
						bare_coulomb_cutoff=vcoul_cutoff_ry,
						bgw_v_grid_fn=bgw_v_grid_fn,
						budget_bytes=m_budget,
					)
			finally:
				for io in ios:
					try:
						io.close()
					except Exception:
						pass

		# Write G0 (channel 0 only) back to the scalar ζ file.
		G0_gathered = jax.experimental.multihost_utils.process_allgather(G0_all)
		if G0_gathered.ndim == 5 and G0_gathered.shape[0] == 1:
			G0_gathered = G0_gathered[0]
		if jax.process_index() == 0:
			with h5py.File(zeta_h5_path, 'a') as f:
				if 'g0_mu' in f:
					del f['g0_mu']
				f.create_dataset('g0_mu', data=np.asarray(G0_gathered))
		jax.experimental.multihost_utils.sync_global_devices("g0_write")

		# TODO(bispinor-sigma): downstream Σ^B integration is not yet
		# wired.  The 10 non-zero V^{μ_L, ν_L}_q tiles are returned as
		# a dict; sigma_sx / sigma_coh consume V_qmunu shaped
		# (1, npol, npol, nkx, nky, nkz, μ, μ) which assumes a single
		# Lorentz channel.  The bispinor consumer site must be updated
		# to walk the (μ_L, ν_L) dict — fail loudly until then.
		print_fn(f"\n  V_q (bispinor) computed:")
		print_fn(f"    {len(V_blocks)} non-zero (μ_L, ν_L) tiles "
		         f"(7 unique kernel + 3 hermitian-transpose; "
		         f"6 zero by Coulomb gauge).")
		for (m, n), block in sorted(V_blocks.items()):
			tr = float(jnp.trace(block[0]).real)
			print_fn(f"      ({m},{n}) shape={block.shape}, "
			         f"V_q=0 trace={tr:.4f}")
		G0 = G0_gathered
		while G0.ndim > 1:
			G0 = G0[0]
		# Return the dict directly; consumer must check ``cfg.bispinor``
		# and walk it.  (See TODO above.)
		return V_blocks, G0

	with timing.section("gw_jax.V_q_compute"), jax_profile.trace_section("V_q_compute"):
		with SlabIO(zeta_h5_path, mode='r', mesh=mesh_xy,
		            use_ffi_io=use_sharded_v_q) as zeta_io:
			with mesh_xy:
				if use_sharded_v_q:
					V_q_raw, G0_all = compute_all_V_q_sharded(
						zeta_io, kgrid=meta.kgrid, fft_grid=meta.fft_grid,
						bvec=bvec, cell_volume=meta.cell_volume,
						mesh_xy=mesh_xy,
						n_rmu=meta.n_rmu, n_rtot=meta.n_rtot,
						sys_dim=meta.sys_dim,
						bdot=np.asarray(wfn.bdot, dtype=np.float64) if meta.sys_dim == 0 else None,
						mc_average_vcoul_body=cfg.mc_average_vcoul_body,
						bare_coulomb_cutoff=vcoul_cutoff_ry,
						bgw_v_grid_fn=bgw_v_grid_fn,
						budget_bytes=m_budget,
					)
				else:
					V_q_raw, G0_all = compute_all_V_q_from_zeta_h5(
						zeta_io, kgrid=meta.kgrid, fft_grid=meta.fft_grid,
						bvec=bvec, cell_volume=meta.cell_volume,
						mu_chunk_size=mu_chunk, mesh_xy=mesh_xy,
						sys_dim=meta.sys_dim,
						q_batch_size=q_batch if mu_chunk >= meta.n_rmu else None,
						bdot=np.asarray(wfn.bdot, dtype=np.float64) if meta.sys_dim == 0 else None,
						mc_average_vcoul_body=cfg.mc_average_vcoul_body,
						bare_coulomb_cutoff=vcoul_cutoff_ry,
						bgw_v_grid_fn=bgw_v_grid_fn,
						n_rmu=meta.n_rmu, n_rtot=meta.n_rtot,
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
			chunks = compute_optimal_chunks(
				meta, mesh_xy,
				memory_budget_gb=cfg.memory_per_device_gb,
				target_utilization=cfg.chunk_target_utilization,
				n_b_left=band_slices.b3 - band_slices.b0,
				n_b_right=band_slices.b4 - band_slices.b1,
				r_chunk_override=cfg.r_chunk_override if cfg.r_chunk_override > 0 else None,
				zct_stage_cap_gb=cfg.zct_stage_cap_gb,
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
				init_W0=True, mesh=mesh_xy, use_ffi_io=cfg.use_ffi_io,
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
				use_ffi_io=cfg.use_ffi_io, mode="a",
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

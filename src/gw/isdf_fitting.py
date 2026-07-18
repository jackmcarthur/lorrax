import gc
import os
import subprocess
import time

import numpy as np
import jax
import jax.numpy as jnp
import jax.experimental.multihost_utils  # noqa: F401  (sync_global_devices)
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common import Meta
from common import timing
from common import jax_profile
from common.gamma_matrices import gamma_perm_phase as _gamma_perm_phase_mu

from isdf.core import (
    c_q_from_psi_sm,
    factor_c_q,
    fit_one_rchunk,
    _resolve_solver_kind,
    _band_norms_slice,
)


# Running max of nvidia-smi used MB across all probe points within a run
# (this rank's GPU only).  jax.device_memory_stats() returns None on the
# JAX 0.8 / CUDA 12.9 Perlmutter stack, so nvidia-smi is the only way to
# observe the TRUE per-rank HBM peak including cuFFT plan workspace,
# NCCL collective buffers, and other XLA-arena-external allocations.
_NVSMI_PEAK_MB = 0
_NVSMI_LAST_MB = 0

def mem_probe(label, *, only_rank0=True):
    """``LORRAX_MEM_DEBUG=1`` runtime probe of process-wide HBM at named sites.

    Reports the JAX/XLA allocator ``bytes_in_use+peak`` plus the top-10
    ``jax.live_arrays()`` shapes.  Module-level so both ``fit_zeta_to_h5``
    (r-chunk loop) and ``gw_init.prepare_isdf_and_wavefunctions`` (V_q
    sites) call the SAME helper — single source of truth for the full
    ζ-fit + V_q HBM lifecycle map.  HLO buffer-assignment.txt is per-jit
    and cannot prove cross-jit liveness; this fills the gap.  Cheap when
    unset (env-var check only; no JAX calls in the early-exit path).

    Round-0 (commit 5c884ac) wired this at three points per r-chunk in
    fit_zeta_to_h5; Round-1 extends to zeta_fit_start, pre_rchunk_loop,
    zeta_fit_end, pre_v_q, post_v_q for the full lifecycle.  Round-7
    (faithfulness audit) adds the ``nvidia-smi`` per-rank true-HBM
    sample — the *canonical* OOM-relevance metric since
    ``device.memory_stats()`` returns ``None`` on this stack.
    """
    if not os.environ.get("LORRAX_MEM_DEBUG"):
        return
    if only_rank0 and jax.process_index() != 0:
        return
    dev = jax.devices()[0]
    stats = dev.memory_stats() if hasattr(dev, "memory_stats") else {}
    if stats is None:
        stats = {}
    bytes_in_use = stats.get("bytes_in_use", -1)
    peak_bytes_in_use = stats.get("peak_bytes_in_use", -1)
    live = jax.live_arrays()
    by_shape = {}
    total_live = 0
    for arr in live:
        if not hasattr(arr, "shape"):
            continue
        try:
            sz = int(np.prod(arr.shape)) * arr.dtype.itemsize
        except Exception:
            continue
        total_live += sz
        key = (tuple(arr.shape), str(arr.dtype))
        entry = by_shape.get(key)
        if entry is None:
            by_shape[key] = [1, sz]
        else:
            entry[0] += 1
            entry[1] += sz
    nvsmi_mb = _nvsmi_used_mb_local_gpu()
    print(f"[mem_probe {label}] in_use={bytes_in_use/1e9:.2f} GB  "
          f"peak={peak_bytes_in_use/1e9:.2f} GB  "
          f"live_count={len(live)} live_total={total_live/1e9:.2f} GB  "
          f"nvsmi={nvsmi_mb/1024:.2f} GB nvsmi_peak={_NVSMI_PEAK_MB/1024:.2f} GB",
          flush=True)
    top = sorted(by_shape.items(), key=lambda kv: -kv[1][1])[:10]
    for (shape, dtype), (cnt, sz) in top:
        print(f"[mem_probe {label}]   {dtype} {shape} x {cnt} = "
              f"{sz/1e9:.2f} GB", flush=True)


def _nvsmi_used_mb_local_gpu():
    """Sample nvidia-smi for the local rank's GPU.  Returns used-MB int or 0.

    Uses ``CUDA_VISIBLE_DEVICES`` (or falls back to GPU 0) to query just
    this rank's GPU rather than the whole node.  Updates module-level
    ``_NVSMI_PEAK_MB`` running max.  Silently returns 0 on any failure
    (nvidia-smi missing, parse error, timeout) — never raises.
    """
    global _NVSMI_PEAK_MB, _NVSMI_LAST_MB
    try:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cvd:
            gpu_idx = cvd.split(",")[0].strip()
        else:
            gpu_idx = "0"
        out = subprocess.run(
            ["nvidia-smi", f"--id={gpu_idx}",
             "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        mb = int(out.stdout.strip().split("\n")[0])
        _NVSMI_LAST_MB = mb
        if mb > _NVSMI_PEAK_MB:
            _NVSMI_PEAK_MB = mb
        return mb
    except Exception:
        return 0


def fit_zeta_to_h5(
    wfn,
    sym,
    meta: Meta,
    centroid_indices: jax.Array,
    mesh_xy: Mesh,
    chunk_r: int,
    output_file: str,
    psi_rmu_Y: jax.Array,
    psi_rmuT_X: jax.Array,
    band_chunk_size: int = 16,
    q_chunk_size: int = 1,
    bispinor: bool = True,
    band_range_left: tuple[int, int] | None = None,
    band_range_right: tuple[int, int] | None = None,
    band_norms: np.ndarray | None = None,
    *,
    slab_io_backend=None,
    gspace_mode: str = "host_cache",
    vertex_mu_L: int = 0,
    solver_kind: str = 'auto',
    distributed_cholesky: str = "auto",
    distributed_lu: str = "auto",
    gflat_chunk_size: int = 0,
    write_ibz_only: bool = True,
    zeta_cutoff_ry: float | None = None,
    zeta_ridge_eps: float = 0.0,
):
    """
    Full zeta fitting pipeline with r-chunk loop and HDF5 output.

    For ``vertex_mu_L == 0`` (default) this is the standard spin-traced
    path used by the charge-channel ISDF fit — bit-identical to the
    pre-bispinor implementation.  For ``vertex_mu_L ∈ {1, 2, 3}`` the
    pair-density helpers contract through the Lorentz vertex γ̃^{μ_L}
    instead of the identity, and ``factor_c_q`` /
    ``solve_zeta`` switch from Cholesky to LU because the
    transverse-channel CCT is indefinite.  See
    ``docs/BISPINOR_DHFB_DESIGN.md`` for the math.

    Workflow:
    1. Slice pre-loaded centroid wavefunctions into left/right halves.
    2. Compute C_q from left/right pair density via FFT.
    3. Compute L_q = chol(C_q) using 2D blocked algorithm.
    4. For each r-chunk:
       a. Compute psi_nk,a(r_chunk) via FFT
       b. Compute left/right pair densities at r-chunk
       c. Compute Z_q via ortho FFT with left/right cross-product
       d. Solve zeta_q = L^{-H}(L^{-1} Z_q) (q-chunked)
       e. Write zeta_q chunk to HDF5

    Args:
        wfn: WFNReader object
        sym: SymMaps object
        meta: Meta object with system info
        centroid_indices: ISDF centroid indices
        mesh_xy: 2D device mesh
        chunk_r: Number of flattened r-points per chunk
        output_file: Path to output HDF5 file
        psi_rmu_Y:  Centroid wavefunctions for the full [b0, b4) band range,
                    shape (nk, nb_full, ns, n_rmu), P(None, None, None, 'y'),
                    un-conjugated ψ.  Produced by
                    :func:`common.wfn_transforms.load_centroids_band_chunked`.
        psi_rmuT_X: Same centroid data transposed/sharded for the pair-density
                    kernel, shape (nk, n_rmu, nb_full, ns),
                    P(None, 'x', None, None), conjugated ψ*.
        band_chunk_size: Bands to process at once when FFTing wavefunctions (with global r)
        q_chunk_size: Q-points to solve C_q @ zeta_q = Z_q simultaneously
        bispinor: Whether to use bispinor wavefunctions
        gspace_mode: ``"host_cache"`` (default) loads all ψ(G) band-chunks
                     into host RAM once at startup and pulls per-bc shards
                     into the jit via io_callback.  ``"file_reread"`` drops
                     the host cache between r-chunks and re-reads via
                     phdf5 collective I/O.  In both modes the jit never
                     holds more than one bc's ψ(G) on device.
        band_range_left: (start, end) for left wfns. Default: (b0, b3)
        band_range_right: (start, end) for right wfns. Default: (b0, b4)
        zeta_ridge_eps: Opt-in Tikhonov ridge ε_rel for the ζ-fit
            (cohsex.in ``zeta_ridge_eps``; 0.0 = OFF, bit-identical).
            Charge channel only — loud-fails for vertex_mu_L ≠ 0.
            When ON the fit solves (C²+ε_q²I)ζ = CZ with ε_q = ε_rel ·
            λ̂_max(C_q); see ``isdf.core._zeta_ridge_normal_matrix``
            for the exact spec and large-system guidance.  Memory: the
            IBZ-sliced CCT (n_q_disk · n_rmu_padded² · 16 B) stays
            alive through the r-chunk loop for the RHS premultiply.

    Returns:
        peak_bytes:  GPU high-water mark (peak_bytes_in_use) during chunk loop

    The centroid wavefunctions are inputs, not outputs — the caller is
    expected to hold the single ``load_centroids_band_chunked`` result and
    reuse it for :func:`gw.wavefunction_bundle.build_wavefunctions` after
    the fit completes.
    """
    from gw.gw_config import SlabIOBackend
    if slab_io_backend is None:
        slab_io_backend = SlabIOBackend.H5PY_ALLGATHER

    zeta_ridge_eps = float(zeta_ridge_eps)
    if zeta_ridge_eps > 0.0 and int(vertex_mu_L) != 0:
        # Charge-only for now — factor_c_q raises the same way; catch it
        # here before any expensive setup.  See factor_c_q's
        # zeta_ridge_eps docstring for the transverse rationale.
        raise NotImplementedError(
            f"zeta_ridge_eps={zeta_ridge_eps:g} with transverse channel "
            f"μ_L={int(vertex_mu_L)}: the ζ-ridge is charge-only for now.")

    # P0 — entry of ζ-fit.  Captures the persistent state set up by
    # ``prepare_isdf_and_wavefunctions`` BEFORE ζ-fit starts: ψ at
    # centroids (full [b0, b4) band range, both Y and X transposes),
    # gflat_acc allocation will not have happened yet.  Forms the
    # planner's "Peak C const" baseline.  Round-1 addition.
    mem_probe("zeta_fit_start")

    nx, ny, nz = meta.fft_grid
    # Two μ extents flow through this function (see common/meta.py:38):
    # ``n_rmu`` is the LOGICAL centroid count from the centroid file;
    # ``n_rmu_padded`` rounds up to ``world_size = ∏ p_a`` so any
    # single- or product-axis sharding on the μ dim divides cleanly.
    # ψ is delivered at PADDED extent by ``load_centroids_band_chunked``
    # (Phase 3a) — pad rows zero — and stays there through the
    # in-memory pair-density / CCT chain.  The Cholesky in
    # ``factor_c_q`` slices internally to logical via the
    # ``n_rmu_logical=`` kwarg (Phase 3b-Cholesky) so the factorization
    # sees a non-singular matrix at its true extent.  zeta_q on disk
    # has logical extent (SlabIO ``valid_shape=`` clips the padded
    # output before write).
    n_rmu = meta.n_rmu                      # logical
    n_rmu_padded = meta.n_rmu_padded        # padded
    n_rtot = meta.n_rtot
    nk_tot = meta.nk_tot
    kgrid = meta.kgrid
    nqx, nqy, nqz = kgrid
    nq = nqx * nqy * nqz

    num_chunks = (n_rtot + chunk_r - 1) // chunk_r
    n_rchunk = chunk_r

    # Band ranges for left and right wavefunctions.
    # Defaults here are (b0,b3) and (b0,b4); gw_jax typically passes (b0,b3) and (b1,b4).
    if band_range_left is None:
        band_range_left = (meta.b_id_0, meta.b_id_3)
    if band_range_right is None:
        band_range_right = (meta.b_id_0, meta.b_id_4)

    # Full range for loading (max of left and right)
    band_range_full = (min(band_range_left[0], band_range_right[0]),
                       max(band_range_left[1], band_range_right[1]))

    nb_left = band_range_left[1] - band_range_left[0]
    nb_right = band_range_right[1] - band_range_right[0]
    nb_full = band_range_full[1] - band_range_full[0]

    print(f"\n  Zeta fitting: {num_chunks} r-chunks x {n_rchunk} r-points, "
          f"{nb_full} bands ({nb_left} left + {nb_right} right)")
    print(f"  Output: {output_file}")

    # ========== STEP 1: Slice pre-loaded centroid ψ into left/right halves ==========
    with timing.section("zeta_fit.slice_halves"):
        # Band range arithmetic — left/right are sub-ranges of [b0, b4).
        l_band_start = band_range_left[0] - band_range_full[0]
        l_band_end = l_band_start + nb_left
        r_band_start = band_range_right[0] - band_range_full[0]
        r_band_end = r_band_start + nb_right

        # Cheap views — the caller keeps the full arrays alive for the
        # post-fit wfn bundle build, so we don't need independent copies.
        psi_l_rmu_Y = psi_rmu_Y[:, l_band_start:l_band_end, :, :]
        psi_l_rmuT_X = psi_rmuT_X[:, :, l_band_start:l_band_end, :]
        psi_r_rmu_Y = psi_rmu_Y[:, r_band_start:r_band_end, :, :]
        psi_r_rmuT_X = psi_rmuT_X[:, :, r_band_start:r_band_end, :]

        print(f"  Left wfns:  {psi_l_rmu_Y.shape}")
        print(f"  Right wfns: {psi_r_rmu_Y.shape}")

        # Pseudobands: clamp weights to ``max(1, w_n)`` and apply them to
        # the centroid copies used for CCT.  When band_norms is None the
        # slices are jnp.ones → the *_fit aliases are identical to the
        # *_rmu_Y / *_rmuT_X copies.  See _band_norms_slice for the why.
        norms_l_jax = _band_norms_slice(band_norms, band_range_left, nb_left)
        norms_r_jax = _band_norms_slice(band_norms, band_range_right, nb_right)
        # psi shapes: Y=(nk, nb, ns, n_rmu), X=(nk, n_rmu, nb, ns)
        psi_l_rmu_Y_fit = psi_l_rmu_Y / norms_l_jax[None, :, None, None]
        psi_l_rmuT_X_fit = psi_l_rmuT_X / norms_l_jax[None, None, :, None]
        psi_r_rmu_Y_fit = psi_r_rmu_Y / norms_r_jax[None, :, None, None]
        psi_r_rmuT_X_fit = psi_r_rmuT_X / norms_r_jax[None, None, :, None]
        if band_norms is not None:
            n_weighted = int(np.sum(band_norms > 1.01))
            n_zero = int(np.sum(band_norms < 1e-10))
            print(f"  Pseudobands normalization: {n_weighted} weighted, "
                  f"{n_zero} zero-weight (skipped)")

    # ========== STEP 2: Compute CCT (C_q) from left/right pair densities ==========
    # γ̃^0 = I_4 → vertex_mu_L=0 is the standard spin-traced path.  For
    # vertex_mu_L ∈ {1,2,3} the γ̃^μ vertex is folded into both P_l and
    # P_r so C_q is the proper per-channel interpolation metric for the
    # Lorentz pair density.  CCT^μ for transverse channels is Hermitian
    # indefinite and rank-deficient: TRS in non-magnetic ground states
    # gives near-null transverse-current modes that would be amplified
    # by 10^4–10^6 if we naively LU-solved through them (the original
    # MoS2 σ^B blowup).  The robust solver in :func:`solve_zeta`
    # uses an SVD pseudoinverse with rcond cutoff to drop those null
    # modes instead of inverting through them — the unique min-norm LSQ
    # solution.
    # Force-eager-import gamma_matrices so its module-level
    # ``gammas_sparse = [_to_sparse(g) for g in gammas]`` (which calls
    # ``jnp.nonzero``) runs OUTSIDE any JIT trace; otherwise the first
    # reference comes from inside the per-chunk kernel jit and trips
    # a ConcretizationTypeError.
    if int(vertex_mu_L) != 0:
        from common import gamma_matrices as _gm  # noqa: F401  (warm import)

    # ── Finalize write_ibz_only BEFORE any IBZ slicing (bug fix) ─────────
    # The IBZ cascade slices C_q/L_q to IBZ rows in STEP 2/3 below, and
    # slices Z_q to IBZ inside the per-r-chunk kernel; the two MUST agree.
    # The orbit-closure auto-fallback can flip write_ibz_only=False when the
    # centroid set isn't closed under the WFN sym group, so it must run HERE
    # — before the C_q slice.  (Previously it ran after factor_c_q, so the
    # charge channel sliced L_q to IBZ, then fell back, leaving L_q at IBZ
    # while Z_q stayed full-BZ → the ``B.shape[0]=nq_full != Nq=nq_ibz``
    # distributed-potrs crash.)  Transverse channels can't fall back (the
    # V_q orchestrator assumes IBZ ζ̃_T), so they loud-fail with a hint.
    if write_ibz_only and getattr(sym, 'q_irr_full_idx', None) is not None:
        try:
            from centroid.orbit_syms import (
                compute_centroid_sym_perm as _check_perm,
            )
            _cent_idx_for_check = np.asarray(
                jax.device_get(centroid_indices), dtype=np.int32)
            _ntran_check = int(np.asarray(sym.sym_matrices).shape[0])
            # ``sym.sym_matrices`` holds the spatial ops; the fractional
            # translations live on WFNReader (BGW WFN.h5 layout).
            _check_perm(
                _cent_idx_for_check,
                sym_matrices=np.asarray(sym.sym_matrices[:_ntran_check]),
                translations=np.asarray(wfn.translations[:_ntran_check]),
                fft_grid=np.asarray(meta.fft_grid, dtype=np.int32),
            )
        except RuntimeError as _exc:
            _first = (_exc.args[0].splitlines()[0]
                      if _exc.args else str(_exc))
            if int(vertex_mu_L) != 0:
                raise RuntimeError(
                    f"Bispinor transverse zeta_T (mu_L={int(vertex_mu_L)}) "
                    f"IBZ-write requested, but the transverse centroid set "
                    f"fails the orbit-closure check under the WFN sym group: "
                    f"{_first}.  Regenerate the transverse centroid file with "
                    f"``centroid.kmeans_cli --density-mode current`` "
                    f"(orbit-aware by default for ntran>1) so the set is "
                    f"closed under the spatial sym group, or bypass the "
                    f"bispinor IBZ cascade with ``LORRAX_FORCE_FULL_BZ=1``."
                ) from _exc
            if jax.process_index() == 0:
                print(f"  q-IBZ reduction: centroid orbit closure failed "
                      f"— falling back to full-BZ on disk.  Reason: {_first}")
            write_ibz_only = False

    with timing.section("zeta_fit.CCT"):
        # ψ inputs at PADDED n_rmu (Phase 3a's load_centroids contract).
        # Monolithic shard_map pipeline: open-spin pair density + IFFT
        # + γ̃·γ̃ + FFT fused inside one shard_map.  The rank-5
        # P_l/P_r pair density never exists as a global XLA value, so
        # the rank-3 fused-replicated reshape that pegged the kernel
        # peak under the legacy chain cannot form.  γ̃^μ_L applied at
        # the post-IFFT contraction step (charge: identity short-
        # circuit; transverse: (perm, phase) tuple).  Output C_q is
        # rank-3 (k, μ, ν).
        chan_label = ("charge γ̃^0=I" if vertex_mu_L == 0
                      else f"transverse γ̃^{vertex_mu_L}")
        print(f"  Computing C_q via shard_map pipeline (open-spin, {chan_label})")
        if vertex_mu_L == 0:
            C_q = c_q_from_psi_sm(
                psi_l_rmuT_X_fit, psi_l_rmu_Y_fit,
                psi_r_rmuT_X_fit, psi_r_rmu_Y_fit,
                kgrid=kgrid, mesh_xy=mesh_xy)
        else:
            gamma_mu = _gamma_perm_phase_mu(vertex_mu_L)
            C_q = c_q_from_psi_sm(
                psi_l_rmuT_X_fit, psi_l_rmu_Y_fit,
                psi_r_rmuT_X_fit, psi_r_rmu_Y_fit,
                gamma_mu, gamma_mu,
                kgrid=kgrid, mesh_xy=mesh_xy)
        C_q.block_until_ready()
        # C_q: (nqx, nqy, nqz, n_rmu_padded, n_rmu_padded) with zero
        # pad rows/cols.

        # Flatten for Cholesky.  Reshape uses padded extent (the
        # in-memory shape); factor_c_q slices to logical
        # internally via ``n_rmu_logical=``.
        C_q_flat = C_q.reshape(nq, n_rmu_padded, n_rmu_padded)
        flat_shard = NamedSharding(mesh_xy, P(None, 'x', 'y'))
        C_q_flat = jax.lax.with_sharding_constraint(C_q_flat, flat_shard)

        # IBZ cascade for the per-q factor: slice C_q to IBZ rows *before*
        # ``factor_c_q`` runs so Cholesky / LU factors only ``n_q_ibz``
        # blocks instead of all ``n_q_full``.  C_q has the same (n_q, μ, ν)
        # shape as V_q, and Cholesky is per-q independent — slice-then-
        # factor gives bit-equal L_q rows as factor-then-slice.  The
        # downstream solve still produces ζ_q at IBZ, and V_q unfolds via
        # ``common.symmetry_maps.unfold_v_q`` from IBZ → full BZ.  Same
        # slice helper applies to χ_q for the W_q = (1 − v_q χ_q)^{-1} v_q
        # path once that lands.
        if write_ibz_only and getattr(sym, 'q_irr_full_idx', None) is not None:
            from common.symmetry_maps import slice_q_full_to_ibz
            C_q_flat = slice_q_full_to_ibz(
                C_q_flat, sym.q_irr_full_idx, out_sharding=flat_shard)

    # ========== STEP 3: Compute L_q from CCT ==========
    # μ_L=0 (charge): C_q is PSD → 2D-blocked Cholesky factor L_q.
    # μ_L=1,2,3 (transverse): C_q is Hermitian indefinite — skip the
    # factorization and pass the slice through; the per-chunk
    # solve_zeta dispatches to an SVD pseudoinverse with
    # rcond cutoff (drops null transverse-current modes that would
    # otherwise be amplified by 10^4–10^6).
    with timing.section("zeta_fit.cholesky"):
        # Resolve once so the banner reflects what actually runs and
        # downstream callees skip their own 'auto' fallback.
        _resolved_solver_kind = _resolve_solver_kind(
            mesh_xy, int(vertex_mu_L), solver_kind,
            distributed_cholesky=distributed_cholesky,
            distributed_lu=distributed_lu)
        if int(vertex_mu_L) == 0:
            _ridge_tag = (f", ζ-ridge eps_rel={zeta_ridge_eps:.1e}"
                          if zeta_ridge_eps > 0.0 else "")
            print(f"  Computing L_q = chol(C_q)  [PSD, charge channel, "
                  f"path={_resolved_solver_kind}{_ridge_tag}]")
        else:
            print(f"  Pass through C_q  [γ̃^{vertex_mu_L} indefinite — "
                  f"path={_resolved_solver_kind}]")
        L_q = factor_c_q(
            C_q_flat, mesh_xy, vertex_mu_L=int(vertex_mu_L),
            n_rmu_logical=n_rmu, solver_kind=_resolved_solver_kind,
            zeta_ridge_eps=zeta_ridge_eps)
        L_q.block_until_ready()
        print(f"  L_q: {L_q.shape}")

    # Pre-compute per-q trace of L_q ONCE per channel.  Only the
    # transverse (LU) path uses it (for the ridge ``ε·|tr(L)|/n_rmu``
    # before each per-q LU solve).  Computing inside solve_zeta means an
    # all-reduce across the (mu/p_x, mu/p_y) mesh sharding fires on every
    # r-chunk — 17 s of GPU stream time on MoS2 3×3 bispinor across 4
    # r-chunks × 3 transverse channels.  L_q (which is CCT for the LU
    # path) doesn't change across r-chunks, so the trace is invariant.
    if int(vertex_mu_L) != 0:
        with timing.section("zeta_fit.trace_L_q"):
            # LOGICAL-block trace only: the identity pad block would
            # contribute exactly +mu_pad to the padded trace, making
            # the LU ridge (ε·|tr|/n) depend on the pad extent — i.e.
            # on the device count.  The slice is a no-op when the
            # extent is already logical.  solve_zeta divides by the
            # logical n to match.
            cct_trace_per_q = jnp.einsum(
                'qii->q', L_q[:, :n_rmu, :n_rmu])
            cct_trace_per_q.block_until_ready()
    else:
        cct_trace_per_q = None

    # Free C_q to reclaim GPU memory before z-chunk loop
    # (P_k_mumu was already deleted above)
    # This is critical for fitting within memory budget.
    # ζ-ridge ON: the RAW (zero-pad, IBZ-sliced) CCT must survive the
    # r-chunk loop for the per-chunk RHS premultiply Z → C·Z inside
    # solve_zeta — ``ridge_c_q`` keeps the reference alive
    # (n_q_disk · n_rmu_padded² · 16 B; opt-in cost, see docstring).
    ridge_c_q = C_q_flat if zeta_ridge_eps > 0.0 else None
    del C_q, C_q_flat
    with timing.section("zeta_fit.gc_pre_chunk_loop"):
        gc.collect()
        jax.clear_caches()  # Clear JAX function caches that may hold array refs

    # ========== STEP 4a: q-IBZ reduction + header writes (rank 0) ==========
    # When ``write_ibz_only=True`` (default), ζ is written for IBZ q's
    # only.  V_q at the full BZ is recovered by the reader / V_q
    # orchestrator using sym data from ``mf_header`` (see report.md
    # §2.4).  The on-disk ``zeta_q`` leading axis is ``n_q_disk``
    # rather than ``n_q_full``; the chunk loop slices
    # ``zeta_chunk[q_irr_full_idx]`` before writing.
    #
    # When ``write_ibz_only=False`` (caller forced full-BZ writes via
    # ``LORRAX_FORCE_FULL_BZ=1``), the full-BZ axis is preserved on
    # disk for back-compatibility.
    #
    # ``write_ibz_only`` was finalized above (before the C_q/L_q IBZ slice)
    # by the orbit-closure auto-fallback, so the on-disk q-axis is IBZ when
    # it is True and full-BZ when it fell back — nothing more to decide here.

    # BGW Brillouin-zone wrap (the local ``_bgw_wrap_q`` below, matching
    # the convention the V_q consumer uses): ``q > kgrid/2 → q
    # − kgrid``.  The writer must match so the per-q phase
    # ``exp(-2πi (q/kgrid)·r)`` baked into the G-flat output is the
    # convention the consumer expects.
    def _bgw_wrap_q(q_int_kgrid: np.ndarray) -> np.ndarray:
        kg = np.asarray(meta.kgrid, dtype=np.float64)
        q = np.asarray(q_int_kgrid, dtype=np.float64)
        return np.where(q > kg / 2, q - kg, q)

    if write_ibz_only:
        q_irr_kgrid_int = sym.q_irr_kgrid_int
        q_irr_full_idx = sym.q_irr_full_idx
        n_q_disk = int(q_irr_full_idx.shape[0])
        # IBZ fractional q-vectors for the G-flat accumulator (Phase C1b).
        # BGW wrap THEN divide by kgrid so the writer's per-q phase
        # matches the V_q kernel's ``apply_bloch_phase`` convention.
        _kgrid_arr_for_qfrac = np.asarray(meta.kgrid, dtype=np.float64)
        q_irr_frac = (_bgw_wrap_q(q_irr_kgrid_int)
                       / _kgrid_arr_for_qfrac[None, :])
        print(f"  q-IBZ reduction: {n_q_disk} IBZ q-points / {nq} full-BZ "
              f"(disk shrink {nq / max(1, n_q_disk):.1f}×)")
    else:
        q_irr_full_idx = None
        q_irr_frac = None
        n_q_disk = nq
        print(f"  q axis on disk: full BZ ({nq} q-points) "
              f"(write_ibz_only=False or closure check failed)")

    # ---- G-flat on-disk format ---------------------------------
    # The writer accumulates each r-chunk's contribution into a
    # persistent G-flat buffer via
    # ``common.wfn_transforms.accumulate_rchunk_to_gflat`` and writes
    # the final tensor as ``zeta_q_G`` (shape
    # ``(n_q_disk, n_rmu, ngkmax)``).  The full r-space ζ_q is never
    # materialised on disk or as a persistent device buffer.  When
    # ``zeta_cutoff_ry`` is provided we build the per-q WFN.h5-style
    # sphere ``{G : |q+G|² ≤ cutoff}``, pad to a uniform ``ngkmax``
    # with the sentinel Miller index ``(-nx/2, -ny/2, -nz/2)``, and
    # store both the coeffs and the per-q components on disk.  Without
    # a cutoff the writer falls back to the full flat-FFT axis
    # (n_G_sph = n_rtot) — slow disk path, kept for sanity checks.
    if q_irr_frac is None:
        # Full-BZ q-vectors with BGW wrap, then / kgrid — same convention
        # the V_q consumer's disk→G path (``zeta_loader._do_disk_to_G``)
        # expects, via the local ``_bgw_wrap_q``.
        _kgrid_arr_for_qfrac = np.asarray(meta.kgrid, dtype=np.float64)
        q_irr_frac = (_bgw_wrap_q(sym.kvecs_asints)
                       / _kgrid_arr_for_qfrac[None, :])

    # Build the per-q WFN.h5-style sphere when a cutoff is available.
    # The output is host numpy; the writer threads ``sphere_idx_padded``
    # through ``accumulate_rchunk_to_gflat`` and stashes the components
    # / ngk / cutoff into the isdf_header below.  ``zeta_cutoff_ry``
    # — distinct from V_q's bare-Coulomb cutoff — defines the per-q
    # sphere on disk.  Caller (``gw_init.fit_zeta``) validates
    # ``zeta_cutoff_ry ≥ bare_coulomb_cutoff_ry`` so V_q has every G
    # it needs.
    _gflat_sphere_idx_padded = None      # (n_q_disk, ngkmax) int32
    _gflat_gvec_components = None        # (n_q_disk, 3, ngkmax) int32
    _gflat_ngk_per_q = None              # (n_q_disk,) int32
    _gflat_ngkmax = None
    if zeta_cutoff_ry is not None and int(meta.sys_dim) != 0:
        from common.coulomb_sphere import compute_per_q_bare_coulomb_components
        _bvec_for_sphere = np.asarray(
            wfn.blat * wfn.bvec, dtype=np.float64)
        _sphere_pkg = compute_per_q_bare_coulomb_components(
            fft_grid=meta.fft_grid,
            bvec=_bvec_for_sphere,
            q_irr_frac=q_irr_frac,
            vcoul_cutoff_ry=float(zeta_cutoff_ry),
            sys_dim=int(meta.sys_dim),
        )
        _gflat_sphere_idx_padded = _sphere_pkg["sphere_idx_padded"]
        _gflat_gvec_components = _sphere_pkg["gvec_components_padded"]
        _gflat_ngk_per_q = _sphere_pkg["ngk_per_q"]
        _gflat_ngkmax = int(_sphere_pkg["ngkmax"])
        if jax.process_index() == 0:
            print(
                f"  G-flat ζ sphere: ngkmax={_gflat_ngkmax}, "
                f"min ngk={int(_gflat_ngk_per_q.min())}, "
                f"max ngk={int(_gflat_ngk_per_q.max())} "
                f"({_gflat_ngkmax / float(n_rtot):.3%} of n_rtot)")

    # ``zeta_q.h5`` carries the BGW-style ``mf_header`` verbatim from
    # the source WFN so any downstream consumer (the new
    # :class:`file_io.zeta_loader.ZetaLoader`, or anything else that
    # speaks the WFN.h5 header) sees the same crystal / k-grid / G-grid
    # / symmetry view.  ``isdf_header`` holds ζ-specific metadata only
    # — centroids in FFT-grid + fractional coords, density label,
    # ``vertex_mu_L``.  Everything sym-derivable (q-IBZ list, centroid
    # orbit permutation, G-sphere) is rebuilt at read time via
    # ``SymMaps`` + ``orbit_syms`` and is *not* stored.
    #
    # Sequence: rank 0 pre-stripes the file, writes both header groups
    # in mode='w' (truncate), closes.  Then SlabIO re-opens with
    # mode='a' so the headers survive and ``create_dataset('zeta_q')``
    # appends rather than truncates.
    from file_io.slab_io import SlabIO
    from file_io.mf_header import copy_mf_header
    from file_io.isdf_header import IsdfHeader, write_isdf_header
    from file_io._slab_io_ffi import _lustre_prestripe

    _wfn_src_path = getattr(wfn, '_filename', None)
    if _wfn_src_path is None:
        raise ValueError(
            "fit_zeta_to_h5: wfn must expose '_filename' (the source "
            "WFN.h5 path) so mf_header can be copied verbatim into "
            "zeta_q.h5.")

    # Centroid FFT-grid indices for the isdf_header.  ``centroid_indices``
    # may be a jax.Array on device; pull to host as int32 (n_rmu, 3).
    _cent_idx_np = np.asarray(jax.device_get(centroid_indices),
                              dtype=np.int32)
    if _cent_idx_np.shape != (n_rmu, 3):
        raise ValueError(
            f"fit_zeta_to_h5: centroid_indices has shape "
            f"{_cent_idx_np.shape}, expected ({n_rmu}, 3).")
    _density_label = 'scalar' if int(vertex_mu_L) == 0 else 'current'
    _hdr_kwargs = dict(
        r_mu_fft_idx=_cent_idx_np,
        fft_grid=meta.fft_grid,
        density=_density_label,
        vertex_mu_L=int(vertex_mu_L),
        zeta_layout='G_flat',
    )
    if _gflat_gvec_components is None:
        raise ValueError(
            "G-flat ζ writer requires a ζ sphere — pass "
            "zeta_cutoff_ry to fit_zeta_to_h5.")
    _hdr_kwargs.update(
        gvec_components=_gflat_gvec_components,
        ngk_per_q=_gflat_ngk_per_q,
        zeta_cutoff_ry=float(zeta_cutoff_ry),
    )
    _isdf_hdr = IsdfHeader.build(**_hdr_kwargs)

    with timing.section("zeta_fit.write_headers"):
        if jax.process_index() == 0:
            # Pre-stripe the file (delete + lfs setstripe).  Idempotent
            # no-op on non-Lustre filesystems.  Must happen before any
            # h5py create so the stripe layout survives ``H5Fcreate``.
            stripe_count = int(
                os.environ.get("LORRAX_PHDF5_STRIPE_COUNT", "16"))
            stripe_size = os.environ.get(
                "LORRAX_PHDF5_STRIPE_SIZE_FS", "4M")
            _lustre_prestripe(output_file, stripe_count=stripe_count,
                              stripe_size=stripe_size)
            # Create file with mf_header, then append isdf_header.
            copy_mf_header(_wfn_src_path, output_file, dst_mode='w')
            write_isdf_header(output_file, _isdf_hdr, mode='a')
        jax.experimental.multihost_utils.sync_global_devices(
            "zeta_fit_headers_written")

    # ========== STEP 4b: SlabIO appends zeta_q to the pre-created file ==========
    # zeta_q is stored flat-q: shape (nq, n_rmu, n_rtot) with
    # q_flat = qx*nqy*nqz + qy*nqz + qz.  Flat-q is the ongoing
    # convention across LORRAX; see file_io.slab_io docs.  Chunk by
    # single-q r-slice so per-q reads stay contiguous.
    #
    # Single SlabIO handle reused for both create_dataset and all
    # writes — avoids the ~900 ms cost of a second collective
    # H5Fopen/close pair (measured 2026-04-18 at MoS2 3x3).  The same
    # handle serves BOTH backends: the allgather backend's handle is a
    # cheap rank-0 h5py file object, and routing its final write through
    # ``write_slab`` (instead of a hand-rolled gather + ``[...] =``)
    # applies the shared ``valid_shape`` prefix clip — the bypass used
    # to write the PADDED gathered buffer into the logical-shaped
    # dataset and crashed whenever a μ pad existed (PADDING_AUDIT #2).
    #
    # mode='a' (not 'w') so the pre-written mf_header + isdf_header
    # are preserved.  SlabIO's FFI prestripe step is skipped on 'a'
    # — we already striped above.
    #
    # Dataset layout ``(nq, n_rtot, n_rmu)`` — NOT ``(nq, n_rmu, n_rtot)``.
    # Rationale: per-r-chunk writes span the full innermost axis (n_rmu)
    # under this layout, so each ``(q, r)`` row is contiguous on disk.
    # Under the old ``(nq, n_rmu, n_rtot)`` layout we'd write n_rchunk <
    # n_rtot on the innermost axis, producing 480K × 1920-B scattered
    # strips per rank per write (measured at 0.18 GB/s on Perlmutter
    # pscratch, 8× slower than contiguous).  Per-q reads (V_q) stay
    # contiguous under this layout too: a 6.6 M-element slab at
    # ``(q, 0, 0)`` is a single contiguous block.  Downstream V_q
    # transposes the returned array on GPU to match the kernel's
    # (n_rmu, n_rtot) expectation — ~50 µs per q, negligible.
    with timing.section("zeta_fit.open_file"):
        # G-flat layout: ``zeta_q_G`` dataset (n_q_disk, n_rmu, ngkmax)
        # — WFN.h5 ``wfns/coeffs`` style with a fixed ``ngkmax`` padded
        # G axis.  Per-q components live in
        # ``isdf_header/gvec_components`` (already serialised by the
        # write_isdf_header call above).  Chunking: one row per q ×
        # full μ × full ngkmax keeps per-q reads contiguous.
        _n_G_sph = (int(_gflat_ngkmax)
                     if _gflat_ngkmax is not None else n_rtot)
        zeta_io = SlabIO(output_file, mode='a', mesh=mesh_xy,
                         backend=slab_io_backend)
        zeta_io.create_dataset(
            'zeta_q_G',
            shape=(n_q_disk, n_rmu, _n_G_sph),
            dtype=np.complex128,
            chunks=(1, n_rmu, _n_G_sph),
        )

    # ========== STEP 5: Pre-load G-space for all band chunks (ONCE) ==========
    # This caches the expensive HDF5 read + scatter so we don't repeat it
    # for each r-chunk. Memory cost depends on band_range_full (can be large).
    kgrid_arr = np.array(meta.kgrid)
    kvecs_frac = sym.kvecs_asints / kgrid_arr[None, :]

    # ``gspace_mode`` (cohsex.in ``gspace_mode``; see
    # ``GspaceIO`` enum): ``host_cache`` is the default; ``file_reread``
    # rebuilds the per-rank host ψ(G) buffer at each r-chunk for
    # multi-TB WFN.h5 systems that can't hold ψ(G) resident.

    # Uniform band chunks over [b_full_start, b_full_end]: N-1 of
    # size ``band_chunk_size`` plus one remainder chunk.  This gives
    # the read/FFT pipeline and the pair-density einsum exactly
    # TWO compile shapes, regardless of where the L/R endpoints fall.
    # Chunks that straddle an L/R endpoint get handled in the loop
    # below by padding the left-side ``psi_L_bc`` slice with zero
    # bands — the resulting einsum still runs at the uniform
    # ``bc_size``, so it hits the same JIT cache.
    _bfs, _bfe = band_range_full
    band_chunk_ranges = [
        (_bfs + i * band_chunk_size,
         min(_bfs + (i + 1) * band_chunk_size, _bfe))
        for i in range((_bfe - _bfs + band_chunk_size - 1) // band_chunk_size)
    ]

    # Build the host-resident ψ(G) store.  Both modes keep zero
    # persistent device residency — the jit fetches one bc at a time
    # via io_callback.  See :mod:`common.psi_G_store` for details.
    from common.psi_G_store import build_psi_G_store
    psi_G_store = build_psi_G_store(
        wfn=wfn, sym=sym, mesh_xy=mesh_xy, meta=meta,
        band_chunk_ranges=band_chunk_ranges,
        bispinor=bispinor,
        mode=gspace_mode,
    )

    # ========== STEP 6: Loop over chunks ==========
    # Wall-clock totals for the end-of-fit timing line.  ``t_fit_total``
    # covers the fused fit_one_rchunk jit (load + pair + ZCT + solve) —
    # finer-grained breakdown now lives inside the jit and is only
    # observable via xprof, not perf_counter.
    t_fit_total = 0.0
    t_write_total = 0.0
    t_chunk_start = time.perf_counter()

    # ``LORRAX_MEM_DEBUG=1`` — runtime probe of process-wide HBM at
    # named lifecycle sites.  The module-level ``mem_probe`` helper is
    # reused so the r-chunk loop sites and the gw_init V_q sites all
    # share one source of truth.  HLO's buffer-assignment.txt is per-jit
    # and cannot prove cross-jit liveness — see
    # reports/memory_model_refit_2026-05-17/agent_e_cross_jit_lifetime.md.
    _mem_probe = mem_probe

    # Per-chunk: ``accumulate_rchunk_to_gflat`` adds the chunk's
    # contribution into the donated ``gflat_acc`` in place; no
    # per-chunk SlabIO write.  The single ``zeta_q_G`` write happens
    # once after the loop.

    # GPU high-water tracker — the all-time ``peak_bytes_in_use``, which is what
    # actually determines OOM (it includes JIT caches + prior-stage allocations,
    # not just the chunk-loop arrays).  Prefer JAX's exact per-rank BFC-arena peak
    # from ``memory_stats()``; fall back to THIS rank's nvidia-smi sample only if
    # that's unavailable.  Two traps this avoids: (1) ``--id=0`` reads a *foreign*
    # GPU on a multi-rank / shared node (``_nvsmi_used_mb_local_gpu`` honours
    # CUDA_VISIBLE_DEVICES); (2) a single post-loop nvidia-smi sample MISSES the
    # peak under the cudaMallocAsync allocator (freed transients already returned),
    # so it is only a last-resort floor, never the reported number when stats work.
    _peak_bytes = 0
    def _track_peak():
        nonlocal _peak_bytes
        try:
            stats = jax.local_devices()[0].memory_stats() or {}
            pk = int(stats.get("peak_bytes_in_use", 0) or 0)
            if pk > 0:
                _peak_bytes = max(_peak_bytes, pk)
                return
        except Exception:
            pass
        try:
            _peak_bytes = max(_peak_bytes, _nvsmi_used_mb_local_gpu() * (1024 ** 2))
        except Exception:
            pass  # leave _peak_bytes = 0; caller suppresses the print

    from common.progress import LoopProgress
    r_progress = LoopProgress(
        num_chunks, print, title="zeta fitting",
        item_name="r-chunk", max_updates=min(num_chunks, 20))

    # norms_l_jax / norms_r_jax were built in STEP 1 above — reuse them
    # as the uniform-shape (nb,) inputs to the fit_one_rchunk jit.

    # ---- G-flat accumulator (zero-init, μ-sharded) ----
    # Persistent buffer: (n_q_disk, n_rmu_padded, ngkmax) c128 with
    # μ sharded across ('x', 'y') so each rank holds n_rmu/p per q.
    # Donated to ``accumulate_rchunk_to_gflat`` each iter; in-place add.
    # When the per-q sphere isn't available (no vcoul_cutoff_ry) we
    # fall back to the full flat-FFT axis n_rtot — slow, kept for
    # smoke / sanity tests.
    from common.wfn_transforms import accumulate_rchunk_to_gflat
    # μ allocated at PADDED extent so the ('x','y') sharding divides
    # cleanly.  Pad rows are zero because the back-solve produces
    # zeta_pad = 0 (L_q's pad block is identity).
    _n_rmu_padded = int(meta.n_rmu_padded)
    _gflat_acc_n_G = (int(_gflat_ngkmax)
                       if _gflat_ngkmax is not None else n_rtot)
    _gflat_acc_sharding = NamedSharding(mesh_xy, P(None, ('x', 'y'), None))
    gflat_acc = jax.jit(
        lambda: jnp.zeros(
            (n_q_disk, _n_rmu_padded, _gflat_acc_n_G),
            dtype=jnp.complex128),
        out_shardings=_gflat_acc_sharding,
    )()
    # Flat-axis chunking inside ``accumulate_rchunk_to_gflat``.  The
    # kernel runs inside a ``shard_map`` over ``('x','y')`` and chunks
    # the per-rank flat ``(n_q · n_mu_local)`` axis into rows-per-
    # scan-iteration of ``chunk_size``.  Memory bound:
    # ``chunk_size · n_rtot · 16 B`` for the per-iteration FFT box.
    #
    # ``gflat_chunk_size = 0`` ⇒ one-shot (fine when the full per-rank
    # box ``N · n_rtot · 16 B`` fits; MoS2 3×3 at 4 ranks: 1.1 GB).
    # For CrI3-class FFT grids set cohsex.in ``gflat_chunk_size`` to
    # an integer; the kernel zero-pads N up to a multiple of the chunk
    # size so any value works (no divisibility constraint on either
    # n_q or n_mu_local).
    _gflat_chunk_size = int(gflat_chunk_size) if gflat_chunk_size else None
    if jax.process_index() == 0:
        _p_prod = int(jax.device_count())
        _n_mu_local = int(meta.n_rmu_padded) // _p_prod
        _N = n_q_disk * _n_mu_local
        _cs = _gflat_chunk_size or _N
        print(f"  G-flat ζ accumulator: N={_N} rows/rank "
              f"(n_q={n_q_disk} × n_mu_local={_n_mu_local}); "
              f"chunk_size={_cs} → "
              f"per-iter FFT box {_cs * n_rtot * 16 / 1e9:.2f} GB/rank")
    # Numpy → replicated: avoid the ``jnp.asarray`` wrap that would
    # single-device-stage and turn device_put into an all-reduce.
    _q_irr_frac_dev = jax.device_put(
        np.asarray(q_irr_frac, dtype=np.float64),
        NamedSharding(mesh_xy, P(None, None)))

    # P1 — pre r-chunk loop, after L_q computed AND gflat_acc allocated.
    # This is the persistent baseline the planner's ``_peak_C_const``
    # should match: centroids (ψ_l/ψ_r in both Y and X transposes), L_q
    # (Cholesky factor at IBZ for charge / pass-through CCT for
    # transverse), and the freshly-zeroed gflat_acc.  Round-1 addition.
    if os.environ.get("LORRAX_MEM_DEBUG"):
        jax.block_until_ready(gflat_acc)
        jax.block_until_ready(L_q)
    mem_probe("pre_rchunk_loop")

    with timing.section("zeta_fit.chunk_loop"):
        for chunk_idx in range(num_chunks):
            r_start = chunk_idx * chunk_r
            r_end = min(r_start + chunk_r, n_rtot)
            actual_n_rchunk = r_end - r_start

            # file_reread mode: (re)build the host-side ψ(G) tiles
            # for this r-chunk.  host_cache mode: no-op.
            psi_G_store.begin_rchunk(r_start, r_end)

            _dbg_rchunk = bool(os.environ.get("LORRAX_RCHUNK_DEBUG"))
            _mem_probe(f"rchunk_start chunk={chunk_idx}")
            t0 = time.perf_counter()
            try:
                with timing.section("zeta_fit.chunk.fit_one_rchunk"), \
                     jax_profile.step_annotation("chunk_fit", step_num=chunk_idx):
                    zeta_chunk = fit_one_rchunk(
                        psi_G_store=psi_G_store,
                        psi_l_rmuT_X_fit=psi_l_rmuT_X_fit,
                        psi_r_rmuT_X_fit=psi_r_rmuT_X_fit,
                        L_q=L_q,
                        norms_l=norms_l_jax,
                        norms_r=norms_r_jax,
                        r_start_dyn=jnp.asarray(r_start, dtype=jnp.int32),
                        mesh_xy=mesh_xy,
                        meta=meta,
                        band_chunk_ranges=band_chunk_ranges,
                        band_range_left=band_range_left,
                        band_range_right=band_range_right,
                        band_range_full=band_range_full,
                        actual_n_rchunk=actual_n_rchunk,
                        q_chunk_size=q_chunk_size,
                        kvecs_frac=kvecs_frac,
                        vertex_mu_L=int(vertex_mu_L),
                        solver_kind=_resolved_solver_kind,
                        q_irr_full_idx=q_irr_full_idx,   # Phase B: gather inside the kernel
                        cct_trace_per_q=cct_trace_per_q,
                        ridge_c_q=ridge_c_q,
                    )
                    zeta_chunk.block_until_ready()
            finally:
                # MUST run after block_until_ready — under file_reread
                # the host tiles are freed here and any still-pending
                # io_callback would use-after-free.
                psi_G_store.end_rchunk()
            _t_fit = time.perf_counter() - t0
            t_fit_total += _t_fit
            _mem_probe(f"after_fit_one_rchunk chunk={chunk_idx}")

            # 6e. IBZ-slice → allgather (or FFI) → HDF5 write.
            # ``zeta_chunk`` is computed at full BZ q (the FFT in
            # ``solve_zeta`` naturally outputs all q's).  We slice to
            # Phase B: ``zeta_chunk`` is already IBZ-shape
            # (n_q_disk, n_rmu, n_rchunk) — the gather happens inside
            # ``fit_one_rchunk`` before the triangular solve.  In
            # full-BZ mode (q_irr_full_idx=None) the kernel returns
            # full-BZ shape.  Accumulate this r-chunk's contribution
            # into ``gflat_acc`` in place; the full ``zeta_q_G`` is
            # written once after the loop.
            t0 = time.perf_counter()
            with timing.section("zeta_fit.chunk.h5_write"):
                gflat_acc = accumulate_rchunk_to_gflat(
                    rchunk=zeta_chunk, gflat_acc=gflat_acc,
                    fft_grid=meta.fft_grid, r0=r_start,
                    sphere_idx=_gflat_sphere_idx_padded,
                    qvec_frac=_q_irr_frac_dev,
                    norm='backward',
                    chunk_size=_gflat_chunk_size,
                    mesh=mesh_xy,
                )
                del zeta_chunk
                if os.environ.get("LORRAX_MEM_DEBUG"):
                    jax.block_until_ready(gflat_acc)
            _t_write = time.perf_counter() - t0
            t_write_total += _t_write
            _mem_probe(f"after_accumulate chunk={chunk_idx}")
            if _dbg_rchunk and jax.process_index() == 0:
                print(f"[rchunk_dbg] chunk={chunk_idx+1}/{num_chunks} "
                      f"r=[{r_start},{r_end}) fit={_t_fit*1000:.0f}ms "
                      f"write={_t_write*1000:.0f}ms "
                      f"total={(_t_fit+_t_write)*1000:.0f}ms", flush=True)
            r_progress.step()
            # LORRAX_MAX_RCHUNKS=N: stop the r-chunk loop after N chunks
            # for profiling/sweeping.  Clean python exit avoids the
            # SLURM step-zombie issue you get from killing the python
            # mid-run.  Off when unset.
            _max_rchunks = os.environ.get("LORRAX_MAX_RCHUNKS")
            if _max_rchunks and (chunk_idx + 1) >= int(_max_rchunks):
                if jax.process_index() == 0:
                    print(f"[rchunk_dbg] LORRAX_MAX_RCHUNKS={_max_rchunks} "
                          f"reached after chunk {chunk_idx+1}; "
                          f"breaking r-chunk loop for profiling.",
                          flush=True)
                break


    t_chunks_total = time.perf_counter() - t_chunk_start
    r_progress.finish()
    # Sample GPU memory ONCE after the last chunk's jit settles.  The
    # allocator keeps the peak reservation so this reads close to the
    # all-time high water.
    _track_peak()

    # ---- Write the accumulated G-flat ζ_q ----
    # One collective write of the persistent ``(n_q_disk, n_rmu,
    # ngkmax)`` tensor to disk.
    with timing.section("zeta_fit.write_g_flat"):
        # Pad slot zero-fill (WFN.h5 ``coeffs = 0`` convention).  The
        # per-q gather inside ``accumulate_rchunk_to_gflat`` read the
        # sentinel ``(-nx/2, -ny/2, -nz/2)`` flat-FFT slot into every
        # pad position; those values are physical (not zero) so we
        # mask them here.  Logical slots ``[..., :ngk[q]]`` carry the
        # real coeffs and are untouched.
        if _gflat_ngk_per_q is not None:
            _ngk_dev = jax.device_put(
                np.asarray(_gflat_ngk_per_q, dtype=np.int32),
                NamedSharding(mesh_xy, P(None)))
            _g_axis = jnp.arange(int(gflat_acc.shape[-1]),
                                  dtype=jnp.int32)        # (ngkmax,)
            _mask = (_g_axis[None, None, :] < _ngk_dev[:, None, None])
            gflat_acc = jnp.where(
                _mask, gflat_acc, jnp.zeros_like(gflat_acc))
        jax.block_until_ready(gflat_acc)
        _n_G_sph = int(gflat_acc.shape[-1])
        # On-disk extent is LOGICAL n_rmu; in-memory buffer is
        # PADDED ``n_rmu_padded``.  SlabIO ``valid_shape=`` clips
        # the trailing μ pad rows on write (they are zero by
        # construction — L_q's pad block is identity).  Both backends
        # implement the same prefix clip (FFI hyperslab / allgather
        # rank-0 slice), so this is the single write path.
        zeta_io.write_slab(
            'zeta_q_G', gflat_acc,
            offset=(0, 0, 0),
            global_shape=(n_q_disk, n_rmu, _n_G_sph),
            valid_shape=(n_q_disk, n_rmu, _n_G_sph),
        )
    del gflat_acc

    with timing.section("zeta_fit.close_io"):
        zeta_io.close()

    with timing.section("zeta_fit.sync_global"):
        jax.experimental.multihost_utils.sync_global_devices("zeta_writes_complete")

    # Flip ``isdf_header/zeta_is_done`` to True now that every chunk
    # has drained to disk.  Restart paths key off this flag to decide
    # whether the on-disk ζ is trustable; flipping it here (after the
    # global sync above) guarantees every rank's writes are durable.
    if jax.process_index() == 0:
        from file_io.isdf_header import mark_zeta_done
        mark_zeta_done(output_file)

    # Free the host tiles (host_cache mode only; file_reread's tiles
    # are already empty after the final end_rchunk).  The phdf5 reader
    # itself is cached at module level and survives.
    psi_G_store.close()

    # Per-stage timing breakdown.  ``fit`` is the fused fit_one_rchunk jit;
    # ``H5`` is the allgather+write (or FFI write_slab).  Everything else
    # lives inside the jit — see xprof for the intra-jit breakdown.
    print(f"  Zeta output: {output_file}  shape: "
          f"(n_q_disk={n_q_disk} of {nqx}·{nqy}·{nqz}={nq} full-BZ, "
          f"n_rtot={n_rtot}, n_rmu={n_rmu})")
    print(f"  Timing ({num_chunks} r-chunks, {t_chunks_total:.1f}s total):")
    for label, t in [("fit", t_fit_total), ("H5", t_write_total)]:
        print(f"    {label:<6} {t:6.2f}s  {100*t/t_chunks_total:4.1f}%")

    # P3 — exit of ζ-fit.  Captures what's still alive after the chunk
    # loop completes: gflat_acc was del'd above, zeta_chunk freed, but
    # centroids (psi_l/psi_r) and L_q are still referenced by the
    # caller's closure (they were passed in as args).  V_q runs next
    # against this baseline.  Round-1 addition.
    mem_probe("zeta_fit_end")

    # Return only peak-memory high-water mark; centroid wavefunctions
    # are not returned (see docstring — callers re-load them directly
    # via ``load_centroids_band_chunked``).
    return _peak_bytes

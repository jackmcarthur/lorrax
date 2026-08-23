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
from common.collectives import (
    device_put_process_local as _device_put_process_local,
)
from common.gamma_matrices import gamma_perm_phase as _gamma_perm_phase_mu
from runtime import debug_print_enabled

# Canonical boolean env grammar for this layer (same recognised token set
# as file_io._slab_io_ffi._env_flag and the one isdf.core imports, plus an
# announcement for anything outside it).
# See gw/gw_config.py's module comment and tests/test_env_grammar.py for
# the drift gate.
from .gw_config import (ZETA_RCOND_DEFAULT,
                        TRANSVERSE_ZETA_RCOND_DEFAULT,
                        active_zeta_truncating_knobs, env_bool)

from isdf.core import (
    build_psi_r_cache_sm,
    c_q_from_psi_sm,
    host_rss_gb as _host_rss_gb,
    factor_c_q,
    fit_one_rchunk,
    _resolve_solver_kind,
    _resolve_zeta_gather,
    _band_norms_slice,
)
# The opaque distributed factor.  Re-exported through isdf.core rather than
# imported from the door here: this module never CALLS distrib_la, it only
# has to tell a token from an array in two places (a log line and the
# fused-route trace predicate).
from isdf.core import FactorToken


# Running max of nvidia-smi used MB across all probe points within a run
# (this rank's GPU only).  jax.device_memory_stats() returns None on the
# JAX 0.8 / CUDA 12.9 Perlmutter stack, so nvidia-smi is the only way to
# observe the TRUE per-rank HBM peak including cuFFT plan workspace,
# NCCL collective buffers, and other XLA-arena-external allocations.
_NVSMI_PEAK_MB = 0
_NVSMI_LAST_MB = 0


def mem_probe(label, *, only_rank0=True):
    """Driver-debug runtime probe of process-wide HBM at named sites.

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
    if not debug_print_enabled():
        return
    if only_rank0 and jax.process_index() != 0:
        return
    # local_devices(), not devices(): jax.devices() is the GLOBAL list, so
    # jax.devices()[0] is process 0's device on every rank.  ``only_rank0``
    # is a DEFAULT, not a guarantee — callers pass only_rank0=False to get a
    # per-rank sample, and that sample must describe the rank's own pool.
    dev = jax.local_devices()[0]
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
    psi_rmuT_X: jax.Array | None,
    band_chunk_size: int = 16,
    q_chunk_size: int = 1,
    bispinor: bool = True,
    band_range_left: tuple[int, int] | None = None,
    band_range_right: tuple[int, int] | None = None,
    band_norms: np.ndarray | None = None,
    *,
    vertex_mu_L: int = 0,
    solver_kind: str = 'auto',
    distributed_cholesky: str = "auto",
    distributed_lu: str = "auto",
    zeta_ridge: float = 0.0,
    charge_zeta_solve: str = "cholesky",
    distributed_zeta_solve: str = "auto",
    zeta_rcond: float = ZETA_RCOND_DEFAULT,
    transverse_zeta_solve: str = "ridge",
    transverse_zeta_rcond: float = TRANSVERSE_ZETA_RCOND_DEFAULT,
    distrib_la_batched_route: str = "auto",
    gflat_chunk_size: int = 0,
    write_ibz_only: bool = True,
    zeta_cutoff_ry: float | None = None,
    low_mem_bands: bool = False,
    psi_nmu_fresh: jax.Array | None = None,
    psi_mun_fresh: jax.Array | None = None,
    print_fn=print,
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
                    P(None, 'x', None, None), conjugated ψ*.  Required when
                    ``low_mem_bands=False``; MUST be None when
                    ``low_mem_bands=True`` (STEP 6 reads psi_nmu_fresh/
                    psi_mun_fresh instead -- see the ``low_mem_bands``
                    entry below).
        band_chunk_size: Bands to process at once when FFTing wavefunctions (with global r)
        q_chunk_size: Q-points to solve C_q @ zeta_q = Z_q simultaneously
        bispinor: Whether to use bispinor wavefunctions
        band_range_left: (start, end) for left wfns. Default: (b0, b3)
        band_range_right: (start, end) for right wfns. Default: (b0, b4)
        low_mem_bands: When True (``low_mem_bands`` deck key), the CCT Gram
                    (STEP 2) AND the r-chunk loop's band contraction
                    (STEP 6) are both built from ``psi_nmu_fresh``/
                    ``psi_mun_fresh`` -- the two-face, all-P-sharded
                    carrier -- instead of from any single-axis ψ copy.
                    STEP 2 uses one distributed SUMMA GEMM
                    (``isdf.core._c_q_face``); STEP 6 reads a
                    ``band_chunk_size``-bounded slab per band-chunk,
                    per r-chunk, out of ``psi_mun_fresh`` via a masked
                    gather + ``psum('y')`` (``isdf.core._z_q_face`` --
                    see docs/architecture/zeta_fit_face_psi_cct.md's
                    r-chunk section for why this, not a second SUMMA GEMM,
                    is the design).  Both ``psi_rmu_Y`` AND ``psi_rmuT_X``
                    MUST be None in this mode (neither single-axis form is
                    ever built or held; the caller drops both before even
                    calling this function -- see
                    ``gw.gw_init.prepare_isdf_and_wavefunctions``).
                    Requires ``vertex_mu_L == 0`` and ``band_norms is
                    None`` -- refuses by name otherwise.  Default False:
                    bit-identical to the pre-existing path (``psi_nmu_fresh``/
                    ``psi_mun_fresh`` are ignored).
        psi_nmu_fresh, psi_mun_fresh: the two-face carrier (see
                    ``gw.wavefunction_bundle``'s ``PSI_NMU_SPEC``/
                    ``PSI_MUN_SPEC``), spanning the SAME [b0, b4) range
                    ``psi_rmuT_X`` does.  Required when ``low_mem_bands``;
                    ignored otherwise.
        print_fn: Status-output sink. Drivers pass the runtime rank-zero
                  printer so deterministic progress is emitted once.

    Returns:
        peak_bytes:  GPU high-water mark (peak_bytes_in_use) during chunk loop

    The centroid wavefunctions are inputs, not outputs — the caller is
    expected to hold the single ``load_centroids_band_chunked`` result and
    reuse it for :func:`gw.wavefunction_bundle.build_wavefunctions` (or,
    under ``low_mem_bands``, the pre-built face carrier via
    :func:`gw.wavefunction_bundle.wavefunctions_face_from_restart`) after
    the fit completes.
    """
    if low_mem_bands:
        if int(vertex_mu_L) != 0:
            raise ValueError(
                "fit_zeta_to_h5: low_mem_bands=True requires vertex_mu_L=0 "
                f"(charge channel); got vertex_mu_L={int(vertex_mu_L)}.  "
                "The bispinor transverse channel is refused under "
                "low_mem_bands upstream "
                "(gw_config.refuse_unsupported_low_mem_bands's bispinor "
                "row) -- this call site should never be reached with "
                "both set; fix the caller.")
        if band_norms is not None:
            raise NotImplementedError(
                "fit_zeta_to_h5: low_mem_bands=True with pseudobands "
                "(band_norms is not None) is not supported this session -- "
                "the face CCT path (isdf.core._c_q_face) has no weighted-"
                "norms arm.  Set low_mem_bands=false, or drop pseudobands, "
                "for this deck.  See KNOWN_LORRAX_ISSUES.md, \"zeta-fit "
                "face CCT: pseudobands unported\".")
        if psi_nmu_fresh is None or psi_mun_fresh is None:
            raise ValueError(
                "fit_zeta_to_h5: low_mem_bands=True requires psi_nmu_fresh="
                " and psi_mun_fresh= (the two-face carrier) -- see "
                "gw.gw_init.prepare_isdf_and_wavefunctions.")
        if psi_rmu_Y is not None:
            raise ValueError(
                "fit_zeta_to_h5: low_mem_bands=True expects psi_rmu_Y=None "
                "-- the caller drops the single-axis Y-form before calling "
                "(its only consumer, the CCT Gram, now reads "
                "psi_nmu_fresh/psi_mun_fresh instead).  A non-None "
                "psi_rmu_Y here means the caller kept a single-axis copy "
                "alive the redesign exists to avoid; fix the caller rather "
                "than silently ignoring it.")
        if psi_rmuT_X is not None:
            raise ValueError(
                "fit_zeta_to_h5: low_mem_bands=True expects psi_rmuT_X="
                "None -- the r-chunk loop (STEP 6) now reads its band-"
                "contraction operand out of psi_mun_fresh (the persistent "
                "face carrier), never a resident single-axis X-form (see "
                "docs/architecture/zeta_fit_face_psi_cct.md's r-chunk "
                "section).  A non-None psi_rmuT_X here means the caller "
                "kept the single-axis X-form alive past the point where "
                "psi_mun_fresh/psi_nmu_fresh were built -- fix the caller "
                "(gw.gw_init.prepare_isdf_and_wavefunctions) rather than "
                "silently ignoring it.")
    else:
        if psi_rmu_Y is None:
            raise ValueError(
                "fit_zeta_to_h5: psi_rmu_Y is required when "
                "low_mem_bands=False (the legacy CCT path reads it in "
                "STEP 1).")
        if psi_rmuT_X is None:
            raise ValueError(
                "fit_zeta_to_h5: psi_rmuT_X is required when "
                "low_mem_bands=False (the legacy r-chunk loop reads it in "
                "STEP 1/STEP 6).")

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
    # has logical extent (SlabIO clips the padded output against the
    # dataset's extent on write).
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

    print_fn(f"\n  Zeta fitting: {num_chunks} r-chunks x {n_rchunk} r-points, "
             f"{nb_full} bands ({nb_left} left + {nb_right} right)")
    print_fn(f"  Output: {output_file}")

    # ========== STEP 1: Slice pre-loaded centroid ψ into left/right halves ==========
    with timing.section("zeta_fit.slice_halves"):
        # Band range arithmetic — left/right are sub-ranges of [b0, b4).
        l_band_start = band_range_left[0] - band_range_full[0]
        l_band_end = l_band_start + nb_left
        r_band_start = band_range_right[0] - band_range_full[0]
        r_band_end = r_band_start + nb_right

        # The X-forms are built ONLY when low_mem_bands=False.  Under
        # low_mem_bands=True, psi_rmuT_X is None (validated above) — STEP
        # 6 now reads its band-contraction operand directly out of
        # psi_mun_fresh (the persistent face carrier, already resident
        # for the whole fit) via a bounded per-band-chunk gather, never
        # a resident single-axis X-form slice (see this function's
        # ``low_mem_bands`` docstring and
        # docs/architecture/zeta_fit_face_psi_cct.md's r-chunk section).
        #
        # The Y-forms are the CCT Gram's ONLY consumer.  Under
        # low_mem_bands, psi_rmu_Y is None (validated above) -- STEP 2
        # reads the face carrier (psi_nmu_fresh/psi_mun_fresh) instead, so
        # skipping this slice is not merely dead-code elimination: it is
        # the point of the redesign (no single-axis Y-form is EVER
        # resident under low_mem_bands=True, not even transiently).
        if low_mem_bands:
            psi_l_rmuT_X = psi_r_rmuT_X = None
            psi_l_rmu_Y = psi_r_rmu_Y = None
            print_fn(f"  X-forms: never built (low_mem_bands: STEP 6 reads "
                     f"psi_mun_fresh directly, one band-chunk at a time; "
                     f"CCT reads the face carrier, STEP 2 below)")
            if env_bool("LORRAX_MEM_DEBUG", False) and jax.process_index() == 0:
                # PER-RANK BYTES, not global shape (KNOWN_LORRAX_ISSUES.md's
                # mem_probe row: two arrays with the same GLOBAL shape print
                # the identical figure whether one is genuinely 2-D sharded
                # or single-axis).  Sums this rank's OWN addressable shards
                # -- the same measurement claims/0426.md's shard-level
                # .nbytes check used to confirm the carrier's face arrays
                # are genuinely 2*S/(Px*Py), not a same-shape single-axis
                # replica.
                _nmu_b = sum(int(s.data.nbytes)
                             for s in psi_nmu_fresh.addressable_shards)
                _mun_b = sum(int(s.data.nbytes)
                             for s in psi_mun_fresh.addressable_shards)
                print_fn(f"  [face carrier, per-rank bytes] psi_nmu_fresh="
                         f"{_nmu_b/1e9:.4f} GB  psi_mun_fresh={_mun_b/1e9:.4f} "
                         f"GB  (2*S/(Px*Py) expected)")
        else:
            # Cheap views — the caller keeps the full arrays alive for the
            # post-fit wfn bundle build, so we don't need independent
            # copies.
            psi_l_rmuT_X = psi_rmuT_X[:, :, l_band_start:l_band_end, :]
            psi_r_rmuT_X = psi_rmuT_X[:, :, r_band_start:r_band_end, :]
            psi_l_rmu_Y = psi_rmu_Y[:, l_band_start:l_band_end, :, :]
            psi_r_rmu_Y = psi_rmu_Y[:, r_band_start:r_band_end, :, :]
            print_fn(f"  Left wfns:  {psi_l_rmu_Y.shape}")
            print_fn(f"  Right wfns: {psi_r_rmu_Y.shape}")

        # Pseudobands: clamp weights to ``max(1, w_n)`` and apply them to
        # the centroid copies used for CCT.  When band_norms is None the
        # slices are jnp.ones → the *_fit values are identical to the
        # *_rmu_Y / *_rmuT_X copies.  See _band_norms_slice for the why.
        # (norms_l_jax/norms_r_jax are also STEP 6 inputs in every layout
        # -- cheap either way, so built unconditionally.)
        norms_l_jax = _band_norms_slice(band_norms, band_range_left, nb_left)
        norms_r_jax = _band_norms_slice(band_norms, band_range_right, nb_right)
        # psi shapes: Y=(nk, nb, ns, n_rmu), X=(nk, n_rmu, nb, ns)
        #
        # band_norms is None (no pseudobands -- the common/default case,
        # and the ONLY case low_mem_bands=True accepts -- refused above
        # otherwise): dividing by an all-ones array is bit-identical to
        # the dividend (IEEE754 x/1.0 == x), so the *_fit computation used
        # to allocate a second, fully-redundant single-axis buffer beside
        # its own source for no numerical benefit -- one of the four "ψ
        # centroid copies" gflat_memory_model.py's psi_copies term prices,
        # doubled again.  ALIAS instead of computing: no new device
        # buffer, and the `del ..._fit` cleanup below still frees the
        # shared object once both names' refcounts drop (fresh-fit
        # low-mem psi contract; see
        # reports/gwjax_low_mem_bands_audit_2026-08-22/report.md census row
        # "Fresh centroid load/liveness").
        if band_norms is None:
            psi_l_rmuT_X_fit = psi_l_rmuT_X
            psi_r_rmuT_X_fit = psi_r_rmuT_X
            psi_l_rmu_Y_fit = None if low_mem_bands else psi_l_rmu_Y
            psi_r_rmu_Y_fit = None if low_mem_bands else psi_r_rmu_Y
        else:
            # low_mem_bands=True refuses band_norms is not None above, so
            # psi_l_rmu_Y/psi_r_rmu_Y are real arrays here in every branch
            # that reaches this line.
            psi_l_rmu_Y_fit = psi_l_rmu_Y / norms_l_jax[None, :, None, None]
            psi_l_rmuT_X_fit = psi_l_rmuT_X / norms_l_jax[None, None, :, None]
            psi_r_rmu_Y_fit = psi_r_rmu_Y / norms_r_jax[None, :, None, None]
            psi_r_rmuT_X_fit = psi_r_rmuT_X / norms_r_jax[None, None, :, None]
        if band_norms is not None:
            n_weighted = int(np.sum(band_norms > 1.01))
            n_zero = int(np.sum(band_norms < 1e-10))
            print_fn(f"  Pseudobands normalization: {n_weighted} weighted, "
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
    # (A ``vertex_mu_L != 0`` warm-import of ``common.gamma_matrices`` used to
    # sit here.  Its stated reason was that module's
    # ``gammas_sparse = [_to_sparse(g) for g in gammas]``, whose ``jnp.nonzero``
    # has a data-dependent output shape and so raises ConcretizationTypeError
    # if its first evaluation happens inside a jit trace.  ``gammas_sparse``
    # was dead — defined and never read anywhere in the tree — and was deleted
    # in the 2026-08-09 import audit, taking the hazard with it.  Nothing left
    # in that module's body is trace-hostile: the remaining constants are
    # ``jnp.asarray`` of numpy tables.  The line was also already inert here,
    # because this file imports ``common.gamma_matrices`` at module scope
    # (``_gamma_perm_phase_mu``, top of file), so the module was fully
    # imported long before this function ran.)

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
    #
    # ONE RESOLUTION POINT.  This used to call ``centroid_source_map_and_wrap``
    # for its side effect — build the whole table, throw it away, and read
    # the answer off whether an exception came back — which is both the
    # third spelling of the closure question in ``gw/`` and an
    # exception used as a boolean.  It now asks
    # ``gw.qgrid_symmetry`` the same question every other site asks, and
    # branches on the mode.  The charge channel's fallback line is the
    # SHARED announcement (deduped on the centroid set, so a run that also
    # falls back in V_q says it once, not twice); the transverse channel
    # suppresses it because its consequence is a refusal, not a fallback.
    if write_ibz_only and getattr(sym, 'q_irr_full_idx', None) is not None:
        from .qgrid_symmetry import resolve_qgrid_symmetry_tables
        _is_transverse = int(vertex_mu_L) != 0
        # ``sym.sym_matrices`` holds the spatial ops; the fractional
        # translations live on WFNReader (BGW WFN.h5 layout).
        _res = resolve_qgrid_symmetry_tables(
            sym=sym, centroid_indices=centroid_indices,
            fft_grid=meta.fft_grid, translations=wfn.translations,
            context=("bispinor transverse ζ̃_T IBZ write"
                     if _is_transverse else "ζ̃ IBZ write"),
            announce_fallback=not _is_transverse,
        )
        if not _res.use_ibz:
            if _is_transverse:
                raise RuntimeError(
                    f"Bispinor transverse zeta_T (mu_L={int(vertex_mu_L)}) "
                    f"IBZ-write requested, but the transverse centroid set "
                    f"fails the orbit-closure check under the WFN sym group: "
                    f"{_res.reason}.  Regenerate the transverse centroid "
                    f"file with ``centroid.kmeans_cli --density-mode "
                    f"current`` (orbit-aware by default for ntran>1) so the "
                    f"set is closed under the spatial sym group, or bypass "
                    f"the bispinor IBZ cascade with "
                    f"``LORRAX_FORCE_FULL_BZ=1``.")
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
        print_fn(f"  Computing C_q via shard_map pipeline (open-spin, {chan_label})")
        if low_mem_bands:
            # Face path (report gwjax_low_mem_bands_audit_2026-08-22,
            # docs/architecture/zeta_fit_face_psi_cct.md): the band
            # contraction is a distributed SUMMA GEMM over the FULL
            # [b0, b4) face carrier, band-WEIGHTED (not sliced) to the L/R
            # window -- see isdf.core._c_q_face's docstring for why this
            # sidesteps the sigma-window edge's (band_range_left[1],
            # generally not mesh-divisible) own pad requirement.  ONE
            # plan serves both P_l and P_r (mirrors greens_function_
            # kernel._build_G_face's g_plan reuse across Gv/Gc).
            with timing.section("zeta_fit.CCT.face_gemm_plan"):
                from distrib_la import gemm_plan as _gemm_plan
                _ns_face = int(psi_mun_fresh.shape[1])
                _nb_face = int(psi_mun_fresh.shape[3])
                if int(psi_nmu_fresh.shape[1]) != _nb_face:
                    raise ValueError(
                        f"fit_zeta_to_h5(low_mem_bands=True): psi_mun_fresh "
                        f"band extent {_nb_face} != psi_nmu_fresh's "
                        f"{int(psi_nmu_fresh.shape[1])}.")
                if int(psi_nmu_fresh.shape[2]) != _ns_face:
                    raise ValueError(
                        f"fit_zeta_to_h5(low_mem_bands=True): psi_mun_fresh "
                        f"spin extent {_ns_face} != psi_nmu_fresh's "
                        f"{int(psi_nmu_fresh.shape[2])}.")
                # GEMM-seam merge folds (s, μ) into ONE axis of size μ*ns
                # (common.contract_bands.merge_spin_centroid) -- m/n must
                # match that merged extent, exactly as
                # greens_function_kernel's own g_plan does
                # (``gemm_plan(mesh_xy, m=n_rmu*ns, k=nb_full, n=n_rmu*ns,
                # ...)``).  Missing the ``*ns`` factor here is caught
                # loudly by gemm_plan's own operand-shape check (a
                # ValueError, not a silent wrong contraction) whenever
                # ns > 1 -- verified 2026-08-22 on the MoS2 k6_c50 spinor
                # deck (ns=2): "gemm_plan A: expected shape (36, 676, 76);
                # got (36, 1352, 76)".
                _face_gemm = _gemm_plan(
                    mesh_xy, m=n_rmu_padded * _ns_face, k=_nb_face,
                    n=n_rmu_padded * _ns_face,
                    nq=nk_tot, dtype=jnp.complex128)
                print_fn(f"  {_face_gemm.describe()}")
                # Band-window WEIGHTS over the array's own [band_range_full)
                # extent (matching STEP 1's l_band_start/r_band_start
                # offset exactly -- psi_mun_fresh/psi_nmu_fresh are the
                # SAME band indexing as psi_rmuT_X, just conj+transposed).
                # Zero-weighted bands contribute EXACTLY zero to the band
                # sum (bilinear in ψ); any trailing bands beyond
                # band_range_full's own span (possible under zeta_nband
                # narrowing) are zero in BOTH weights, contributing
                # nothing regardless of nb_face's exact extent.
                _off = int(band_range_full[0])
                _idx = np.arange(_nb_face)
                _w_l_np = np.where(
                    (_idx >= band_range_left[0] - _off)
                    & (_idx < band_range_left[1] - _off), 1.0, 0.0)
                _w_r_np = np.where(
                    (_idx >= band_range_right[0] - _off)
                    & (_idx < band_range_right[1] - _off), 1.0, 0.0)
                weight_l_face = jnp.asarray(_w_l_np, dtype=jnp.float64)
                weight_r_face = jnp.asarray(_w_r_np, dtype=jnp.float64)
            C_q = c_q_from_psi_sm(
                kgrid=kgrid, mesh_xy=mesh_xy, layout='face',
                psi_mun=psi_mun_fresh, psi_nmu=psi_nmu_fresh,
                weight_l=weight_l_face, weight_r=weight_r_face,
                gemm=_face_gemm)
        elif vertex_mu_L == 0:
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
        # ``symmetry_maps.unfold_isdf_operator`` from IBZ → full BZ.  Same
        # slice helper applies to χ_q for the W_q = (1 − v_q χ_q)^{-1} v_q
        # path once that lands.
        if write_ibz_only and getattr(sym, 'q_irr_full_idx', None) is not None:
            from ffi import _services
            _services.ensure_on_path()
            from symmetry_maps import slice_q_full_to_ibz
            C_q_flat = slice_q_full_to_ibz(
                C_q_flat, sym.q_irr_full_idx, out_sharding=flat_shard)

    # Fresh-fit low-mem psi contract (low_mem_bands census row "Fresh
    # centroid load/liveness"; report gwjax_low_mem_bands_audit_2026-08-22).
    # The Y-form centroid copies (mu on 'y', single-axis-sharded, bands
    # replicated -- psi_{l,r}_rmu_Y[_fit]) were consumed ONLY by the CCT
    # contraction just above.  Neither Cholesky (STEP 3, reads C_q_flat,
    # not psi) nor the r-chunk loop (STEP 6, fit_one_rchunk reads only
    # psi_{l,r}_rmuT_X_fit) touches them again.  Left alive, they used to
    # sit as dead Python references for the whole Stage-C r-chunk loop --
    # gflat_memory_model._persistent_bytes's "psi_copies" term prices
    # exactly this (2 Y-form + 2 X-form single-axis copies) as resident
    # for every stage, Stage C included.  Free them HERE, before that loop
    # opens, dropping Stage C's own psi floor to the two X-forms alone.
    # The base (pre-normalization) X arrays are ALSO superseded once
    # *_fit exists (band_norms is not None: a real second copy; band_norms
    # is None: an alias of the same object per STEP 1 above, so this
    # second `del` is a no-op refcount drop, not a second free) --
    # deleting both names is what actually reaches zero refcount.
    #
    # low_mem_bands=True: psi_l_rmu_Y/psi_r_rmu_Y/*_fit AND
    # psi_l_rmuT_X/psi_r_rmuT_X are already None (STEP 1 never built
    # them -- CCT read the face carrier instead, and STEP 6 below reads
    # psi_mun_fresh directly), so these two ``del``s are no-op name
    # drops for that half.  The face GEMM PLAN is a real per-call object
    # worth freeing here (it holds a cuBLASMp communicator + two
    # compiled executables; CCT runs once per fit, so nothing downstream
    # reuses it) -- but ``weight_l_face``/``weight_r_face`` are NOT
    # freed: STEP 6's per-band-chunk gather (``isdf.core._z_q_face``)
    # reuses them unchanged (same L/R window, same band_range_full
    # indexing), and they are two tiny (nb_face,) float64 vectors, not
    # worth rebuilding.  psi_mun_fresh/psi_nmu_fresh are NOT touched:
    # they are the caller's own permanent face bundle
    # (Wavefunctions.psi_mun/psi_nmu after this call returns), not a
    # fit-local intermediate.
    del psi_l_rmu_Y, psi_r_rmu_Y, psi_l_rmu_Y_fit, psi_r_rmu_Y_fit
    del psi_l_rmuT_X, psi_r_rmuT_X
    if low_mem_bands:
        del _face_gemm
    gc.collect()

    # ========== STEP 3: Compute L_q from CCT ==========
    # μ_L=0 (charge): C_q is PSD → 2D-blocked Cholesky factor L_q.
    # μ_L=1,2,3 (transverse): C_q is Hermitian indefinite — skip the
    # factorization and pass the slice through; the per-chunk
    # solve_zeta dispatches to an SVD pseudoinverse with
    # rcond cutoff (drops null transverse-current modes that would
    # otherwise be amplified by 10^4–10^6).
    with timing.section("zeta_fit.cholesky"):
        # Resolve once so the banner reflects what actually runs and
        # downstream callees skip their own 'auto' fallback.  Pass the
        # factor batch (nq = C_q_flat.shape[0], IBZ-sliced above when the
        # cascade fires) and the logical centroid count so the charge
        # resolver can pick the mesh-invariant replicated dense Cholesky for
        # fit-size stacks (see _resolve_solver_kind_charge).
        # ORDER MATTERS (capacity fix 2026-07-29, ladder notes R15.1).
        # The back-solve tier is resolved FIRST because when it is
        # ``distributed`` it REPLACES the charge factor wholesale a few lines
        # below, and the replicated factor's capacity check must therefore not
        # be enforced.  Enforcing it was capping mu at
        # sqrt(4 GiB / 16 B) = 16,384 on the size of a buffer the distributed
        # route never allocates.  ``_resolve_zeta_gather`` does not depend on
        # the factor kind, so the reorder is free.
        _resolved_zeta_gather = _resolve_zeta_gather(
            distributed_zeta_solve,
            n_rmu=int(n_rmu_padded), nq=int(C_q_flat.shape[0]),
            mesh_xy=mesh_xy, vertex_mu_L=int(vertex_mu_L),
            charge_zeta_solve=charge_zeta_solve,
            transverse_zeta_solve=transverse_zeta_solve)
        _resolved_solver_kind = _resolve_solver_kind(
            mesh_xy, int(vertex_mu_L), solver_kind,
            distributed_cholesky=distributed_cholesky,
            distributed_lu=distributed_lu,
            n_rmu=int(n_rmu), nq=int(C_q_flat.shape[0]),
            charge_zeta_solve=charge_zeta_solve,
            replicated_factor_used=(_resolved_zeta_gather != 'distributed'),
            transverse_zeta_solve=transverse_zeta_solve)
        if _resolved_zeta_gather == 'distributed':
            # The tier IS a charge-channel route: it replaces the whole
            # factor+back-solve pair, not just the gather granularity of
            # the replicated one.  Overriding here (and NOT inside
            # _resolve_solver_kind_charge) keeps the two knobs
            # single-purpose: `distributed_cholesky` picks the factor
            # LIBRARY, `distributed_zeta_solve` picks whether the factor is
            # ever replicated.  Refuses rather than downgrades if the user
            # also pinned an incompatible factor route.
            if int(vertex_mu_L) != 0:
                # Transverse rank_truncate family (the ONLY way the
                # transverse channel reaches this tier — the ridge
                # family's resolver returns per_q above): replace the
                # local eigh factor with the pzheevd 2D-sharded C⁺.
                if _resolved_solver_kind != 'transverse_rank_truncate':
                    raise ValueError(
                        f"distributed_zeta_solve='distributed' on a "
                        f"transverse channel expects the rank_truncate "
                        f"family, but the solver resolved to "
                        f"{_resolved_solver_kind!r}.")
                _resolved_solver_kind = 'distributed_transverse_rank_truncate'
            else:
                if _resolved_solver_kind not in ('replicated_rank_truncate',):
                    raise ValueError(
                        f"distributed_zeta_solve='distributed' resolves the "
                        f"charge factor itself, but distributed_cholesky "
                        f"resolved to {_resolved_solver_kind!r}.  Leave "
                        f"distributed_cholesky at 'auto' (which gives "
                        f"replicated_rank_truncate) for this tier.")
                _resolved_solver_kind = 'distributed_rank_truncate'
        if int(vertex_mu_L) == 0:
            _how = ("rank-truncated pinv"
                    if _resolved_solver_kind == 'replicated_rank_truncate'
                    else "distributed rank-truncated pinv (2D-sharded C+)"
                    if _resolved_solver_kind == 'distributed_rank_truncate'
                    else "chol(C_q)")
            print_fn(f"  Computing L_q = {_how}  [PSD, charge channel, "
                     f"path={_resolved_solver_kind}]")
        else:
            _how_t = ("hoisted per-q pivoted LU (once per channel)"
                      if _resolved_solver_kind == 'lu'
                      else "hoisted distributed pXgetrf (block-cyclic, "
                           "once per channel)"
                      if _resolved_solver_kind == 'scalapack_lu'
                      else "rank-truncated pinv (|lambda| cut, explicit "
                           "C+, once per channel, rcond="
                           f"{float(transverse_zeta_rcond):g})"
                      if _resolved_solver_kind == 'transverse_rank_truncate'
                      else "distributed rank-truncated pinv (pzheevd, "
                           "2D-sharded C+, rcond="
                           f"{float(transverse_zeta_rcond):g})"
                      if _resolved_solver_kind
                      == 'distributed_transverse_rank_truncate'
                      else "CCT passthrough (fused per-r-chunk getrf+getrs)")
            print_fn(f"  Computing transverse factor = {_how_t}  "
                     f"[γ̃^{vertex_mu_L} indefinite — "
                     f"path={_resolved_solver_kind}]")
        _gather_gb = (int(C_q_flat.shape[0]) * int(n_rmu_padded) ** 2
                      * 16 / 1e9)
        # per-q tile: the two structural all_gathers inside ``_per_q_block``
        # move μ²/p_y (row block) + μ² (full tile) — measured, not nominal.
        _p_y = int(mesh_xy.shape['y'])
        _tile_gb = (int(n_rmu_padded) ** 2 * 16 * (1.0 + 1.0 / _p_y)) / 1e9
        print_fn(f"  Zeta back-solve tier: {_resolved_zeta_gather} "
                 f"(distributed_zeta_solve={distributed_zeta_solve})  "
                 f"replicated (nq,μ,μ) gather would be {_gather_gb:.2f} GB/rank; "
                 f"per-q tile {_tile_gb:.3f} GB (×nq executions/r-chunk); "
                 f"distributed tier gathers NO (μ,μ) object")
        if int(vertex_mu_L) != 0:
            # Transverse: the factor stage is HOISTED (2026-08) — one
            # pivoted LU per q per CHANNEL instead of per r-chunk.
            # factor_c_q returns (factor, piv): (LU, perm) on the local
            # plan, a distrib_la FactorToken (ipiv inside it) on the
            # scalapack plan, (CCT passthrough, None) on the cusolvermp
            # plan (fused per-r-chunk getrf+getrs kept there).
            L_q, lu_piv = factor_c_q(
                C_q_flat, mesh_xy, vertex_mu_L=int(vertex_mu_L),
                n_rmu_logical=n_rmu, solver_kind=_resolved_solver_kind,
                zeta_ridge=zeta_ridge, zeta_rcond=zeta_rcond,
                transverse_zeta_rcond=float(transverse_zeta_rcond),
                distrib_la_batched_route=distrib_la_batched_route)
        else:
            L_q = factor_c_q(
                C_q_flat, mesh_xy, vertex_mu_L=int(vertex_mu_L),
                n_rmu_logical=n_rmu, solver_kind=_resolved_solver_kind,
                zeta_ridge=zeta_ridge, zeta_rcond=zeta_rcond,
                distrib_la_batched_route=distrib_la_batched_route)
            lu_piv = None
        # A distributed library factor is an OPAQUE token: no ``.shape``
        # to print and no single buffer to block on (a ScaLAPACK token
        # holds factors AND pivots).  Report what it does publish.
        if isinstance(L_q, FactorToken):
            print_fn(f"  L_q: {L_q!r}")
        else:
            L_q.block_until_ready()
            print_fn(f"  L_q: {L_q.shape}")

    # Pre-compute per-q trace of the CCT ONCE per channel — needed ONLY
    # by the remaining FUSED transverse route (cusolvermp passthrough,
    # piv None), whose per-r-chunk ridge is ``ε·|tr(CCT)|/n``.  On the
    # hoisted routes the ridge is baked into the factor, so the trace
    # (and its per-r-chunk all-reduce, 17 s of GPU stream at MoS2 3×3
    # bispinor) is gone.
    if (int(vertex_mu_L) != 0 and lu_piv is None
            and not isinstance(L_q, FactorToken)
            and _resolved_solver_kind not in (
                'transverse_rank_truncate',
                'distributed_transverse_rank_truncate')):
        # ``not isinstance(...)`` is the ScaLAPACK hoist: its ridge is
        # baked into the factored matrix, so it needs no trace operand —
        # the condition used to read that off ``lu_piv is not None``, and
        # the pivots live in the token now.  Rank-truncate kinds also
        # return piv=None but carry no fused LU path — no ridge, hence no
        # trace operand (their L_q is C⁺, whose trace would be a
        # different, meaningless quantity here).
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
    # This is critical for fitting within memory budget
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
        print_fn(f"  q-IBZ reduction: {n_q_disk} IBZ q-points / {nq} full-BZ "
                 f"(disk shrink {nq / max(1, n_q_disk):.1f}×)")
    else:
        q_irr_full_idx = None
        q_irr_frac = None
        n_q_disk = nq
        print_fn(f"  q axis on disk: full BZ ({nq} q-points) "
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
    # with the shared FFT-box pad sentinel (Nyquist-corner cell; the
    # Miller index is ``(-n/2)`` per even axis and ``+(n-1)/2`` per odd
    # one — ``common.gvec_fft_box.fft_box_pad_sentinel``), and
    # store both the coeffs and the per-q components on disk.  Without
    # a cutoff the writer falls back to the full flat-FFT axis
    # (n_G_sph = n_rtot) — slow disk path, kept for sanity checks.
    if q_irr_frac is None:
        # Full-BZ q-vectors with BGW wrap, then / kgrid — the convention
        # the per-q sphere below is built in, via the local
        # ``_bgw_wrap_q``.  (It used to be stated as "what the V_q
        # consumer's disk→G path expects"; that path,
        # ``zeta_loader._do_disk_to_G``, was deleted on 2026-08-07 — this
        # writer bakes the phase in and the reader does no FFT at all.)
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
        # The Cartesian reciprocal ROWS off the vcoul door's geometry rather
        # than a hand-written ``blat * bvec``, which ``docs/services/vcoul.md``
        # names as an antipattern for a reason worth restating: a product
        # every caller has to remember to take is a footgun, and the day one
        # of them forgets, every G in this sphere is off by the lattice
        # constant with no shape error to say so.  ``from_wfn`` is duck-typed
        # on ``blat``/``bvec``/``cell_volume``; ``WfnLoader`` binds all three
        # off the mf_header.  Only ``.bvec`` is read here — this site takes no
        # Ω at all, so there is no ``cell_volume`` question to answer.
        from ffi import _services
        _services.ensure_on_path()
        from vcoul import CoulombGeometry
        _bvec_for_sphere = CoulombGeometry.from_wfn(wfn).bvec
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
            print_fn(
                f"  G-flat ζ sphere: ngkmax={_gflat_ngkmax}, "
                f"min ngk={int(_gflat_ngk_per_q.min())}, "
                f"max ngk={int(_gflat_ngk_per_q.max())} "
                f"({_gflat_ngkmax / float(n_rtot):.3%} of n_rtot)")

    # ``zeta_q.h5`` carries the BGW-style ``mf_header`` verbatim from
    # the source WFN so any downstream consumer (the new
    # :class:`zeta_loader.ZetaLoader`, or anything else that
    # speaks the WFN.h5 header) sees the same crystal / k-grid / G-grid
    # / symmetry view.  ``isdf_header`` holds ζ-specific metadata only
    # — centroids in FFT-grid + fractional coords, density label,
    # ``vertex_mu_L``.  Everything sym-derivable (q-IBZ list, centroid
    # orbit permutation, G-sphere) is rebuilt at read time via
    # ``SymMaps`` + ``orbit_syms`` and is *not* stored.
    #
    # Sequence: SlabIO(mode='w') replaces the inode (rank-0 unlink +
    # barrier) and CREATES it collectively, so H5Fcreate applies the
    # Lustre striping hints; it closes immediately.  Rank 0 then appends
    # both header groups with h5py mode='a'.  Then SlabIO re-opens with
    # mode='a' so the headers survive and ``create_dataset('zeta_q')``
    # appends rather than truncates.  The create-order matters and is
    # explained at the write_headers section below.
    from file_io.slab_io import SlabIO
    from file_io.mf_header import copy_mf_header
    from file_io.isdf_header import IsdfHeader, write_isdf_header

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
        # WHICHEVER CALL CREATES THE INODE DECIDES THE STRIPE LAYOUT.
        # A Lustre layout is fixed at inode create, and the striping
        # hints live in the MPI_Info that phdf5's H5Fcreate builds — so
        # the inode has to be created by MPI-IO, and everything after
        # inherits what it chose.  (The old ``lfs setstripe`` prestripe
        # helper was deleted 2026-07-31: ``lfs`` is absent in the
        # production container, so the hints are the only lever left.)
        #
        # This used to be ``_replace_inode_for_write`` followed straight
        # by ``copy_mf_header(..., dst_mode='w')`` — i.e. rank-0 SERIAL
        # h5py created the inode, MPI-IO never saw it, and the file took
        # the DIRECTORY DEFAULT while the comment here claimed it got the
        # hints.  MEASURED 2026-08-07: zeta_q.h5 stripe_count=1 against a
        # resolved policy of 4 and a sibling isdf_tensors_*.h5 of 4
        # (``lfs getstripe -c -S``; /pscratch directory default is 1).
        # Cost at 1 rank is exactly zero, which is how it survived — it
        # bites at P>1, where ROMIO sets cb_nodes = min(stripe_count,
        # nranks), so a 1-stripe file pins collective buffering to a
        # SINGLE aggregator however many ranks are writing.
        #
        # So: SlabIO(mode='w') creates and closes the file collectively
        # (that H5Fcreate carries the striping hints), and only THEN does
        # rank-0 h5py append the two header groups with mode='a', which
        # keeps the inode it finds.  Serial h5py appending to a
        # parallel-HDF5-created file is fine — verified through both
        # readers by
        # tests/test_file_io.py::test_zeta_q_inode_gets_striping_policy.
        # Measured cost of the extra collective create+close on an empty
        # file: 4 ms at 1 rank.
        #
        # SlabIO mode='w' runs the same rank-0 unlink + barrier helper
        # (``_replace_inode_for_write``) internally, so the explicit call
        # that used to be here would be doing its job twice.
        with SlabIO(output_file, mode='w', mesh=mesh_xy):
            pass
        if jax.process_index() == 0:
            # Append both header groups to the inode SlabIO just created.
            copy_mf_header(_wfn_src_path, output_file, dst_mode='a')
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
    # applies the shared logical-extent clip — the bypass used
    # to write the PADDED gathered buffer into the logical-shaped
    # dataset and crashed whenever a μ pad existed (PADDING_AUDIT #2).
    #
    # mode='a' (not 'w') so the pre-written mf_header + isdf_header
    # are preserved.  SlabIO's mode='w' inode replace is skipped on 'a'
    # — the inode was already replaced above.
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
        # write_isdf_header call above).  The row-major axis order keeps one
        # full q slab contiguous without requesting an HDF5 chunk layout.
        _n_G_sph = (int(_gflat_ngkmax)
                     if _gflat_ngkmax is not None else n_rtot)
        zeta_io = SlabIO(output_file, mode='a', mesh=mesh_xy,
                         )
        zeta_io.create_dataset(
            'zeta_q_G',
            shape=(n_q_disk, n_rmu, _n_G_sph),
            dtype=np.complex128,
        )

    # ========== STEP 5: Pre-load G-space for all band chunks (ONCE) ==========
    # This caches the expensive HDF5 read + scatter so we don't repeat it
    # for each r-chunk. Memory cost depends on band_range_full (can be large).
    kgrid_arr = np.array(meta.kgrid)
    kvecs_frac = sym.kvecs_asints / kgrid_arr[None, :]

    # Uniform band chunks over [b_full_start, b_full_end]: N-1 of
    # size ``band_chunk_size`` plus one remainder chunk.  This gives
    # the read/FFT pipeline and the pair-density einsum exactly
    # TWO compile shapes, regardless of where the L/R endpoints fall.
    # Chunks that straddle an L/R endpoint get handled in the loop
    # below by padding the left-side ``psi_L_bc`` slice with zero
    # bands — the resulting einsum still runs at the uniform
    # ``bc_size``, so it hits the same JIT cache.
    _bfs, _bfe = band_range_full
    _p_xy = int(mesh_xy.shape['x']) * int(mesh_xy.shape['y'])
    if band_chunk_size % _p_xy != 0:
        raise ValueError(
            f"band_chunk_size={band_chunk_size} is not divisible by the "
            f"mesh size {_p_xy}")
    # The physics window remains exactly [_bfs, _bfe): masks below use the
    # logical endpoints.  Only the ψ(G) transport range is padded so every
    # per-chunk band shard is P-divisible.  The loader and the pair masks
    # zero/ignore these at-most-P-1 tail bands.
    _bfe_transport = _bfs + ((_bfe - _bfs + _p_xy - 1) // _p_xy) * _p_xy
    band_chunk_ranges = [
        (_bfs + i * band_chunk_size,
         min(_bfs + (i + 1) * band_chunk_size, _bfe_transport))
        for i in range(
            (_bfe_transport - _bfs + band_chunk_size - 1) // band_chunk_size)
    ]

    # Build the host-resident ψ(G) staging store.  It supplies one band
    # chunk at a time while the all-P-sharded ψ(r) cache is built below,
    # then its host tiles are released before the r-chunk loop.
    from common.psi_G_store import build_psi_G_store
    psi_G_store = build_psi_G_store(
        wfn=wfn, mesh_xy=mesh_xy, meta=meta,
        band_chunk_ranges=band_chunk_ranges,
        bispinor=bispinor,
    )

    # Hoist the full-grid ψ(G)->ψ(r) transforms once.  The cache's band
    # axis is sharded over the complete ('x','y') mesh; no rank holds the
    # full window at P>1, and there is no μ axis (hence no N_μ² object).
    # Block until every io_callback has drained before freeing the host tiles.
    with timing.section("zeta_fit.build_psi_r_cache"):
        psi_r_cache = build_psi_r_cache_sm(
            psi_G_store, mesh_xy=mesh_xy)
        psi_r_cache.block_until_ready()
    _cache_local_bytes = sum(
        int(shard.data.size) * int(shard.data.dtype.itemsize)
        for shard in psi_r_cache.addressable_shards)
    if jax.process_index() == 0:
        print_fn(f"  ψ(r) cache: {psi_r_cache.shape}, "
                 f"band-sharded over ('x','y'), "
                 f"{_cache_local_bytes / 1e9:.2f} GB local")
    # The r-chunk loop consumes only the device cache.  Drop host ψ(G) now;
    # g_index/kvecs remain staged properties used as compatibility operands.
    psi_G_store.close()

    # ========== STEP 6: Loop over chunks ==========
    # Wall-clock totals for the end-of-fit timing line.  ``t_fit_total``
    # covers the fused fit_one_rchunk jit (load + pair + ZCT + solve) —
    # finer-grained breakdown now lives inside the jit and is only
    # observable via xprof, not perf_counter.
    t_fit_total = 0.0
    t_write_total = 0.0
    t_chunk_start = time.perf_counter()

    # Driver debug — runtime probe of process-wide HBM at
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
        num_chunks, print_fn, title="zeta fitting",
        item_name="r-chunk", max_updates=min(num_chunks, 20)).start()

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
        print_fn(f"  G-flat ζ accumulator: N={_N} rows/rank "
                 f"(n_q={n_q_disk} × n_mu_local={_n_mu_local}); "
                 f"chunk_size={_cs} → "
                 f"per-iter FFT box {_cs * n_rtot * 16 / 1e9:.2f} GB/rank")
    # Numpy → replicated, process-locally: ``jax.device_put(numpy, <a
    # multi-process NamedSharding>)`` fires JAX's hidden
    # ``multihost_utils.assert_equal`` all-gather (see
    # ``common.collectives.device_put_process_local``).
    _q_irr_frac_dev = _device_put_process_local(
        np.asarray(q_irr_frac, dtype=np.float64),
        NamedSharding(mesh_xy, P(None, None)))

    # P1 — pre r-chunk loop, after L_q computed AND gflat_acc allocated.
    # This is the persistent baseline the planner's ``_peak_C_const``
    # should match: centroids (ψ_l/ψ_r in both Y and X transposes), L_q
    # (Cholesky factor at IBZ for charge / pass-through CCT for
    # transverse), and the freshly-zeroed gflat_acc.  Round-1 addition.
    if debug_print_enabled():
        jax.block_until_ready(gflat_acc)
        jax.block_until_ready(L_q)
    mem_probe("pre_rchunk_loop")

    # glibc heap trim hook (see the comment at the call site in the loop
    # below).  DEFAULT ON — one ``malloc_trim(0)`` per r-chunk costs a few
    # ms and is the second half of the workstream-T ramp cure (the first
    # half is ``runtime.tune_glibc_malloc``, applied before ``import jax``).
    # ``LORRAX_MALLOC_TRIM=0`` disables; ``malloc_trim`` is glibc-only, so
    # a missing symbol just disables the hook.
    #
    # The parse used to be a case-SENSITIVE ``not in ("0", "off", "false")``,
    # which missed ``""``, ``"no"`` and every uppercase spelling — so
    # ``LORRAX_MALLOC_TRIM=OFF`` left the hook ON while its documented
    # sibling ``LORRAX_MALLOC_TUNE=OFF`` (runtime._env_falsy) correctly
    # turned OFF.  The two knobs are advertised together in
    # docs/dev/env_vars.md:116-117 and now answer identically.
    _trim_fn = None
    if env_bool("LORRAX_MALLOC_TRIM", True):
        try:
            import ctypes
            _trim_fn = ctypes.CDLL("libc.so.6").malloc_trim
        except Exception:
            _trim_fn = None

    with timing.section("zeta_fit.chunk_loop"):
        for chunk_idx in range(num_chunks):
            r_start = chunk_idx * chunk_r
            r_end = min(r_start + chunk_r, n_rtot)
            actual_n_rchunk = r_end - r_start

            _dbg_rchunk = debug_print_enabled()
            _rss0 = _host_rss_gb() if _dbg_rchunk else 0.0
            _mem_probe(f"rchunk_start chunk={chunk_idx}")
            t0 = time.perf_counter()
            with timing.section("zeta_fit.chunk.fit_one_rchunk"), \
                 jax_profile.step_annotation("chunk_fit", step_num=chunk_idx):
                zeta_chunk = fit_one_rchunk(
                    psi_G_store=psi_G_store,
                    psi_r_cache=psi_r_cache,
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
                    zeta_gather=_resolved_zeta_gather,
                    lu_piv=lu_piv,
                    distrib_la_batched_route=distrib_la_batched_route,
                    layout=('face' if low_mem_bands else 'legacy'),
                    psi_mun=(psi_mun_fresh if low_mem_bands else None),
                    weight_l=(weight_l_face if low_mem_bands else None),
                    weight_r=(weight_r_face if low_mem_bands else None),
                )
                zeta_chunk.block_until_ready()
            _t_fit = time.perf_counter() - t0
            t_fit_total += _t_fit
            _rss1 = _host_rss_gb() if _dbg_rchunk else 0.0
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
                if debug_print_enabled():
                    jax.block_until_ready(gflat_acc)
            _t_write = time.perf_counter() - t0
            t_write_total += _t_write
            _mem_probe(f"after_accumulate chunk={chunk_idx}")
            _rss2 = _host_rss_gb() if _dbg_rchunk else 0.0
            # Return glibc's free-but-still-mapped heap to the OS at the
            # end of each r-chunk.  Together with
            # ``runtime.tune_glibc_malloc`` this is the cure for the
            # workstream-T per-r-chunk anonymous-memory ramp: XLA:CPU's
            # transients are ordinary malloc/free, and the memory glibc
            # keeps in its per-thread arenas after ``free`` is what
            # ratchets RSS up chunk after chunk while
            # ``jax.live_arrays()`` stays flat.  MEASURED at MoS2 12x12 /
            # 606c / P=80 / 81 r-chunks: +0.35 GB/rank/chunk -> 0.00.
            _trimmed = _rss2
            if _trim_fn is not None:
                try:
                    _trim_fn(0)
                    _trimmed = _host_rss_gb()
                except Exception:
                    pass
            if _dbg_rchunk and jax.process_index() == 0:
                # ``rss`` is the leak observable: on CPU the XLA arena is
                # invisible to ``memory_stats()``, so per-chunk anonymous
                # growth only shows up here.  ``live`` is the JAX-side
                # array total — rss growing while live stays flat means
                # the accumulation is NOT a retained jax.Array.
                _live_gb = 0.0
                try:
                    for _a in jax.live_arrays():
                        _live_gb += (int(np.prod(_a.shape))
                                     * _a.dtype.itemsize) / 1e9
                except Exception:
                    _live_gb = -1.0
                print_fn(f"[rchunk_dbg] chunk={chunk_idx+1}/{num_chunks} "
                         f"r=[{r_start},{r_end}) fit={_t_fit*1000:.0f}ms "
                         f"write={_t_write*1000:.0f}ms "
                         f"total={(_t_fit+_t_write)*1000:.0f}ms "
                         f"rss={_rss2:.3f}GB live={_live_gb:.3f}GB "
                         f"d_fit={_rss1 - _rss0:+.3f} "
                         f"d_acc={_rss2 - _rss1:+.3f} "
                         f"rss_trim={_trimmed:.3f}GB")
            r_progress.step()
            # LORRAX_MAX_RCHUNKS=N: stop the r-chunk loop after N chunks
            # for profiling/sweeping.  Clean python exit avoids the
            # SLURM step-zombie issue you get from killing the python
            # mid-run.  Off when unset.
            #
            # STRICT PARSE.  The old ``if _max_rchunks and ... >= int(...)``
            # took any non-empty string as a request: ``=0`` — the natural
            # spelling of "no limit" — is a truthy STRING whose int is 0, so
            # ``(chunk_idx+1) >= 0`` fired on the very first chunk and
            # truncated the fit to one r-chunk.  A non-numeric value raised a
            # bare ``invalid literal for int()`` from inside the loop.  Both
            # produce a PARTIAL ζ that the writer still marks complete, so
            # they must refuse up front instead.
            _max_rchunks = os.environ.get("LORRAX_MAX_RCHUNKS")
            _max_rchunks = _max_rchunks.strip() if _max_rchunks else ""
            if _max_rchunks:
                try:
                    _max_n = int(_max_rchunks)
                except ValueError:
                    raise ValueError(
                        f"LORRAX_MAX_RCHUNKS={_max_rchunks!r} is not an "
                        f"integer.  It truncates the ζ fit after N r-chunks "
                        f"(profiling only); unset it to fit every chunk."
                    ) from None
                if _max_n < 1:
                    raise ValueError(
                        f"LORRAX_MAX_RCHUNKS={_max_rchunks!r} must be >= 1.  "
                        f"To disable the truncation, UNSET the variable — "
                        f"'0' used to be accepted and truncated the fit to a "
                        f"single r-chunk, silently."
                    )
            if _max_rchunks and (chunk_idx + 1) >= _max_n:
                if jax.process_index() == 0:
                    print_fn(f"[rchunk_dbg] LORRAX_MAX_RCHUNKS={_max_rchunks} "
                             f"reached after chunk {chunk_idx+1}; "
                             f"breaking r-chunk loop for profiling.")
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
        # FFT-box pad sentinel's flat slot into every
        # pad position; those values are physical (not zero) so we
        # mask them here.  Logical slots ``[..., :ngk[q]]`` carry the
        # real coeffs and are untouched.
        if _gflat_ngk_per_q is not None:
            _ngk_dev = _device_put_process_local(
                np.asarray(_gflat_ngk_per_q, dtype=np.int32),
                NamedSharding(mesh_xy, P(None)))
            _g_axis = jnp.arange(int(gflat_acc.shape[-1]),
                                  dtype=jnp.int32)        # (ngkmax,)
            _mask = (_g_axis[None, None, :] < _ngk_dev[:, None, None])
            gflat_acc = jnp.where(
                _mask, gflat_acc, jnp.zeros_like(gflat_acc))
        jax.block_until_ready(gflat_acc)
        _n_G_sph = int(gflat_acc.shape[-1])
        # On-disk extent is LOGICAL n_rmu (stated once, at the
        # ``create_dataset`` above); in-memory buffer is PADDED
        # ``n_rmu_padded``.  SlabIO clips the trailing μ pad rows
        # against the dataset's own extent (they are zero by
        # construction — L_q's pad block is identity), identically on
        # every backend, with nothing said here.
        zeta_io.write_slab('zeta_q_G', gflat_acc)
    del gflat_acc

    with timing.section("zeta_fit.close_io"):
        zeta_io.close()

    with timing.section("zeta_fit.sync_global"):
        jax.experimental.multihost_utils.sync_global_devices("zeta_writes_complete")

    # Flip ``isdf_header/zeta_is_done`` to True now that every chunk
    # has drained to disk.  Restart paths key off this flag to decide
    # whether the on-disk ζ is trustable; flipping it here (after the
    # global sync above) guarantees every rank's writes are durable.
    #
    # NOT when a truncating knob was in force.  ``LORRAX_MAX_RCHUNKS=N``
    # breaks the r-chunk loop above after N chunks, so the tensor written
    # a few lines up is a PARTIAL ζ — and this stamp is the file's own
    # claim about itself.  ``gw_init`` already refuses to add
    # ``fit_provenance`` in that case, which blocks REUSE; this stops the
    # file lying in the first place.  The two guards are deliberately
    # independent: provenance is about "may a later run reuse this", and
    # ``zeta_is_done`` is about "did the writer finish", which it did not.
    # Both read the one list in ``gw_config.ZETA_TRUNCATING_ENV_KNOBS``.
    _trunc = active_zeta_truncating_knobs()
    if _trunc and jax.process_index() == 0:
        _names = ", ".join(f"{k}={v}" for k, v in _trunc)
        print_fn("")
        print_fn("  " + "!" * 68)
        print_fn(f"  *** LORRAX SANITY: {_names} truncated this ζ fit "
                 f"(fewer than {num_chunks} r-chunks). ***")
        print_fn(f"  {output_file} holds a PARTIAL ζ and is NOT being marked")
        print_fn("  complete (isdf_header/zeta_is_done stays False), so no")
        print_fn("  restart or reuse path will trust it.  Profiling only —")
        print_fn("  delete this file before any production run from this")
        print_fn("  directory.")
        print_fn("  " + "!" * 68)
    elif jax.process_index() == 0:
        from file_io.isdf_header import mark_zeta_done
        mark_zeta_done(output_file)

    # Idempotent after the post-cache-build close above.  The phdf5 reader
    # itself is cached at module level and survives.
    psi_G_store.close()

    # Per-stage timing breakdown.  ``fit`` is the fused fit_one_rchunk jit;
    # ``H5`` is the allgather+write (or FFI write_slab).  Everything else
    # lives inside the jit — see xprof for the intra-jit breakdown.
    print_fn(f"  Zeta output: {output_file}  shape: "
             f"(n_q_disk={n_q_disk} of {nqx}·{nqy}·{nqz}={nq} full-BZ, "
             f"n_rtot={n_rtot}, n_rmu={n_rmu})")
    print_fn(f"  Timing ({num_chunks} r-chunks, {t_chunks_total:.1f}s total):")
    for label, t in [("fit", t_fit_total), ("H5", t_write_total)]:
        print_fn(f"    {label:<6} {t:6.2f}s  {100*t/t_chunks_total:4.1f}%")

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

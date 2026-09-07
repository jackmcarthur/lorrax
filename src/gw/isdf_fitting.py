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
from common import collectives
from common import timing
from common import jax_profile
from common.collectives import (
    device_put_process_local as _device_put_process_local,
)
from runtime import debug_print_enabled
from runtime.padding import bounded_partition_tile

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
    complete_ordered_pair_normal_equations,
    host_rss_gb as _host_rss_gb,
    factor_c_q,
    fit_one_rchunk,
    solve_zeta,
    _z_q_face_parent,
    _resolve_solver_kind,
    _resolve_zeta_gather,
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


def _coupled_gflat_spill(value, *, enabled):
    """Park one coupled-channel G-flat accumulator in process-local RAM."""
    if not enabled:
        return value
    return collectives.spill_to_host(value)


def _coupled_gflat_restore(value, *, enabled):
    """Restore one parked accumulator for its canonical device operation."""
    if not enabled:
        return value
    return collectives.restore_from_host(value)


def _local_shard_nbytes(arr) -> int:
    """Exact bytes this process will hold after spilling ``arr``."""
    return sum(
        int(np.prod(shard.data.shape)) * np.dtype(shard.data.dtype).itemsize
        for shard in arr.addressable_shards
    )


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


def _collect_fit_setup_garbage():
    """Collect dead setup arrays without flushing process-wide JIT caches."""
    gc.collect()


def _prepare_zeta_fit_geometry(
        _coupled_mu123_coordinator, _coupled_rank_gate, _spill_coupled_gflat_to_host,
        _stack_coupled_solve_inputs, band_norms, band_range_left, band_range_right,
        cache_face_y_blocks, centroid_indices, chunk_r, k_unfold_plan, meta, output_file,
        print_fn, psi_mun_parent, psi_nmu_parent, sym, vertex_mu_L, wfn, write_ibz_only):
    """Produce validated fit dimensions, band windows and the q-domain decision."""
    if k_unfold_plan is None or psi_nmu_parent is None or psi_mun_parent is None:
        raise ValueError("fit_zeta_to_h5 requires a typed plan and both raw-parent faces.")
    if band_norms is not None:
        raise NotImplementedError("Raw-parent zeta fitting does not support pseudobands.")
    if (int(psi_mun_parent.shape[0]) != int(k_unfold_plan.n_parent)
            or int(psi_mun_parent.shape[2]) != int(k_unfold_plan.n_centroid_packed)):
        raise ValueError("fit_zeta_to_h5: parent face extent differs from its typed plan.")
    if (_coupled_mu123_coordinator is not None
            and (not cache_face_y_blocks
                 or int(vertex_mu_L) not in (1, 2, 3))):
        raise ValueError(
            "fit_zeta_to_h5: the private coupled-mu123 coordinator is "
            "transverse-only and requires the "
            "planner-selected bounded face-Y cache")
    if ((_coupled_mu123_coordinator is None)
            != (_coupled_rank_gate is None)):
        raise ValueError(
            "fit_zeta_to_h5: the private coupled coordinator and rank "
            "gate must be supplied together")
    if (_spill_coupled_gflat_to_host
            and _coupled_mu123_coordinator is None):
        raise ValueError(
            "fit_zeta_to_h5: coupled G-flat host spill is private to the "
            "three-channel transverse coordinator")
    if (_stack_coupled_solve_inputs
            and _coupled_mu123_coordinator is None):
        raise ValueError(
            "fit_zeta_to_h5: stacked coupled solves are private to the "
            "three-channel transverse coordinator")
    mem_probe("zeta_fit_start")
    nx, ny, nz = meta.fft_grid
    n_rmu = meta.n_rmu                      # logical (the file extent)
    n_rmu_padded = meta.n_rmu_padded        # padded (the in-memory carrier)
    n_rmu_solve = int(getattr(meta, 'mu_solve_extent', n_rmu))
    mu_basis = getattr(meta, 'mu_basis', None)
    n_rtot = meta.n_rtot
    nk_tot = meta.nk_tot
    kgrid = meta.kgrid
    nqx, nqy, nqz = kgrid
    nq = nqx * nqy * nqz
    num_chunks = (n_rtot + chunk_r - 1) // chunk_r
    n_rchunk = chunk_r
    if band_range_left is None:
        band_range_left = (meta.b_id_0, meta.b_id_3)
    if band_range_right is None:
        band_range_right = (meta.b_id_0, meta.b_id_4)
    _complete_charge_pairs = (
        int(vertex_mu_L) == 0
        and tuple(band_range_left) != tuple(band_range_right))
    if _complete_charge_pairs:
        from ffi import _services
        _services.ensure_on_path()
        from symmetry_maps import q_negation_index
        _q_neg_idx = q_negation_index(kgrid)
        print("  Charge pair training domain: ordered LR + RL "
              "(conjugation-closed normal equations)")
    else:
        _q_neg_idx = None
    band_range_full = (min(band_range_left[0], band_range_right[0]),
                       max(band_range_left[1], band_range_right[1]))
    nb_left = band_range_left[1] - band_range_left[0]
    nb_right = band_range_right[1] - band_range_right[0]
    nb_full = band_range_full[1] - band_range_full[0]
    print_fn(f"\n  Zeta fitting: {num_chunks} r-chunks x {n_rchunk} r-points, "
             f"{nb_full} bands ({nb_left} left + {nb_right} right)")
    print_fn(f"  Output: {output_file}")
    if write_ibz_only and getattr(sym, 'q_irr_full_idx', None) is not None:
        from .qgrid_symmetry import resolve_qgrid_symmetry_tables
        _is_transverse = int(vertex_mu_L) != 0
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
                    f"set is closed under the spatial sym group.")
            write_ibz_only = False
    return (n_rmu, n_rmu_padded, n_rmu_solve, mu_basis, n_rtot, nk_tot, kgrid, nq, num_chunks, n_rchunk, band_range_left, band_range_right, band_range_full, _q_neg_idx, write_ibz_only)


def _build_zeta_fit_gram(
        _q_neg_idx, band_range_full, band_range_left, band_range_right, k_unfold_plan,
        kgrid, mesh_xy, mu_basis, n_rmu, n_rmu_padded, n_rmu_solve, nk_tot, nq, print_fn,
        psi_mun_parent, psi_nmu_parent, sym, vertex_mu_L, write_ibz_only):
    """Produce the selected-q parent Gram and its band weights."""
    with timing.section("zeta_fit.CCT"):
        chan_label = ("charge γ̃^0=I" if vertex_mu_L == 0
                      else f"transverse γ̃^{vertex_mu_L}")
        print_fn(f"  Computing C_q via shard_map pipeline (open-spin, {chan_label})")
        with timing.section("zeta_fit.CCT.face_gemm_plan"):
            from distrib_la import gemm_plan as _gemm_plan
            _ns_face, _nb_face = int(psi_mun_parent.shape[1]), int(psi_mun_parent.shape[3])
            if (int(psi_nmu_parent.shape[1]) != _nb_face
                    or int(psi_nmu_parent.shape[2]) != _ns_face):
                raise ValueError("fit_zeta_to_h5: parent face band/spin extents differ.")
            _mu_gemm = int(k_unfold_plan.n_centroid_packed)
            _face_gemm = _gemm_plan(
                mesh_xy, m=_mu_gemm * _ns_face, k=_nb_face,
                n=_mu_gemm * _ns_face, nq=int(k_unfold_plan.n_parent),
                dtype=jnp.complex128)
            print_fn(f"  {_face_gemm.describe()}")
            _off = int(band_range_full[0])
            _idx = np.arange(_nb_face)
            weight_l_face = jnp.asarray(np.where(
                (_idx >= band_range_left[0] - _off)
                & (_idx < band_range_left[1] - _off), 1.0, 0.0), dtype=jnp.float64)
            weight_r_face = jnp.asarray(np.where(
                (_idx >= band_range_right[0] - _off)
                & (_idx < band_range_right[1] - _off), 1.0, 0.0), dtype=jnp.float64)
        print_fn(f"  C_q on raw parents: {k_unfold_plan.n_parent} -> {nk_tot} k rows")
        C_q = c_q_from_psi_sm(
            kgrid=kgrid, mesh_xy=mesh_xy,
            psi_mun_parent=psi_mun_parent, psi_nmu_parent=psi_nmu_parent,
            weight_l=weight_l_face, weight_r=weight_r_face,
            gemm=_face_gemm, k_unfold_plan=k_unfold_plan,
            gamma_L=int(vertex_mu_L), gamma_R=int(vertex_mu_L))
        C_q_flat = C_q.reshape(nq, n_rmu_padded, n_rmu_padded)
        del C_q
        flat_shard = NamedSharding(mesh_xy, P(None, 'x', 'y'))
        C_q_flat = jax.lax.with_sharding_constraint(C_q_flat, flat_shard)
        if n_rmu_solve == n_rmu_padded and n_rmu_padded > n_rmu:
            _pad_scale = (jnp.trace(C_q_flat, axis1=-2, axis2=-1).real
                          / float(n_rmu)).astype(C_q_flat.dtype)
            _pad_diag = jnp.diag(jnp.asarray(
                ~mu_basis.active_mask, dtype=C_q_flat.dtype))
            C_q_flat = jax.lax.with_sharding_constraint(
                C_q_flat + _pad_scale[:, None, None] * _pad_diag[None],
                flat_shard)
        if _q_neg_idx is not None:
            C_q_flat = complete_ordered_pair_normal_equations(
                C_q_flat, _q_neg_idx)
        if write_ibz_only and getattr(sym, 'q_irr_full_idx', None) is not None:
            from ffi import _services
            _services.ensure_on_path()
            from symmetry_maps import slice_q_full_to_ibz
            C_q_flat = slice_q_full_to_ibz(
                C_q_flat, sym.q_irr_full_idx, out_sharding=flat_shard)
        C_q_flat.block_until_ready()
    del _face_gemm
    return (C_q_flat, weight_l_face, weight_r_face)


def _report_zeta_factor_route(
        C_q_flat, _resolved_solver_kind, _resolved_zeta_gather, distributed_zeta_solve,
        mesh_xy, n_rmu_padded, print_fn, transverse_zeta_rcond, vertex_mu_L):
    """Report the resolved factor and back-solve route with its memory price."""
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
                  else "hoisted distributed getrf (block-cyclic, "
                       "once per channel)"
                  if _resolved_solver_kind in
                  ('scalapack_lu', 'cusolvermp_lu')
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
    _p_y = int(mesh_xy.shape['y'])
    _tile_gb = (int(n_rmu_padded) ** 2 * 16 * (1.0 + 1.0 / _p_y)) / 1e9
    print_fn(f"  Zeta back-solve tier: {_resolved_zeta_gather} "
             f"(distributed_zeta_solve={distributed_zeta_solve})  "
             f"replicated (nq,μ,μ) gather would be {_gather_gb:.2f} GB/rank; "
             f"per-q tile {_tile_gb:.3f} GB (×nq executions/r-chunk); "
             f"distributed tier gathers NO (μ,μ) object")


def _factor_zeta_fit_gram(
        C_q_flat, _coupled_mu123_coordinator, _stack_coupled_solve_inputs,
        charge_zeta_solve, distrib_la_batched_route, distributed_cholesky, distributed_lu,
        distributed_zeta_solve, mesh_xy, n_rmu_padded, n_rmu_solve, print_fn, solver_kind,
        transverse_zeta_rcond, transverse_zeta_solve, vertex_mu_L, zeta_rcond, zeta_ridge):
    """Produce the existing factor and back-solve contract."""
    with timing.section("zeta_fit.cholesky"):
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
            n_rmu=n_rmu_solve, nq=int(C_q_flat.shape[0]),
            charge_zeta_solve=charge_zeta_solve,
            replicated_factor_used=(_resolved_zeta_gather != 'distributed'),
            transverse_zeta_solve=transverse_zeta_solve)
        if _resolved_zeta_gather == 'distributed':
            if int(vertex_mu_L) != 0:
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
        factor_trace_per_q = None
        if (int(vertex_mu_L) != 0 and _resolved_solver_kind in
                ('cusolvermp_lu', 'scalapack_lu')):
            with timing.section("zeta_fit.trace_L_q"):
                factor_trace_per_q = jnp.einsum(
                    'qii->q', C_q_flat[:, :n_rmu_solve, :n_rmu_solve])
                factor_trace_per_q.block_until_ready()
        _report_zeta_factor_route(
            C_q_flat, _resolved_solver_kind, _resolved_zeta_gather, distributed_zeta_solve, mesh_xy,
            n_rmu_padded, print_fn, transverse_zeta_rcond, vertex_mu_L)
        _coupled_factor = bool(
            _coupled_mu123_coordinator is not None
            and _stack_coupled_solve_inputs
            and _resolved_solver_kind in ('cusolvermp_lu', 'scalapack_lu'))
        if _coupled_factor:
            L_q, lu_piv = C_q_flat, None
            print_fn("  Deferring current factor to one coupled LU factor")
        elif int(vertex_mu_L) != 0:
            L_q, lu_piv = factor_c_q(
                C_q_flat, mesh_xy, vertex_mu_L=int(vertex_mu_L),
                n_rmu_logical=n_rmu_solve, solver_kind=_resolved_solver_kind,
                zeta_ridge=zeta_ridge, zeta_rcond=zeta_rcond,
                transverse_zeta_rcond=float(transverse_zeta_rcond),
                distrib_la_batched_route=distrib_la_batched_route,
                transverse_trace_per_q=factor_trace_per_q)
        else:
            L_q = factor_c_q(
                C_q_flat, mesh_xy, vertex_mu_L=int(vertex_mu_L),
                n_rmu_logical=n_rmu_solve, solver_kind=_resolved_solver_kind,
                zeta_ridge=zeta_ridge, zeta_rcond=zeta_rcond,
                distrib_la_batched_route=distrib_la_batched_route)
            lu_piv = None
        if isinstance(L_q, FactorToken):
            print_fn(f"  L_q: {L_q!r}")
        else:
            L_q.block_until_ready()
            print_fn(f"  L_q: {L_q.shape}")
    if _coupled_factor:
        cct_trace_per_q = factor_trace_per_q
    elif (int(vertex_mu_L) != 0 and lu_piv is None
            and not isinstance(L_q, FactorToken)
            and _resolved_solver_kind not in (
                'transverse_rank_truncate',
                'distributed_transverse_rank_truncate')):
        with timing.section("zeta_fit.trace_L_q"):
            cct_trace_per_q = jnp.einsum(
                'qii->q', L_q[:, :n_rmu_solve, :n_rmu_solve])
            cct_trace_per_q.block_until_ready()
    else:
        cct_trace_per_q = None
    _coupled_stacked_solve = _coupled_factor
    _coupled_solve_inputs = (
        (L_q, cct_trace_per_q) if _coupled_stacked_solve else None)
    del C_q_flat
    return (L_q, lu_piv, _resolved_solver_kind, _resolved_zeta_gather, cct_trace_per_q, _coupled_stacked_solve, _coupled_solve_inputs)


def _prepare_zeta_output_sphere(
        meta, n_rtot, nq, print_fn, sym, wfn, write_ibz_only, zeta_cutoff_ry):
    """Produce the disk q rows and their Coulomb sphere."""
    from ffi import _services
    _services.ensure_on_path()
    from symmetry_maps import bgw_integer_q_to_fractional
    if write_ibz_only:
        q_irr_kgrid_int = sym.q_irr_kgrid_int
        q_irr_full_idx = sym.q_irr_full_idx
        n_q_disk = int(q_irr_full_idx.shape[0])
        q_irr_frac = bgw_integer_q_to_fractional(
            q_irr_kgrid_int, meta.kgrid)
        print_fn(f"  q-IBZ reduction: {n_q_disk} IBZ q-points / {nq} full-BZ "
                 f"(disk shrink {nq / max(1, n_q_disk):.1f}×)")
    else:
        q_irr_full_idx = None
        q_irr_frac = None
        n_q_disk = nq
        print_fn(f"  q axis on disk: full BZ ({nq} q-points) "
                 f"(write_ibz_only=False or closure check failed)")
    if q_irr_frac is None:
        q_irr_frac = bgw_integer_q_to_fractional(
            sym.kvecs_asints, meta.kgrid)
    _gflat_sphere_idx_padded = None      # (n_q_disk, ngkmax) int32
    _gflat_gvec_components = None        # (n_q_disk, 3, ngkmax) int32
    _gflat_ngk_per_q = None              # (n_q_disk,) int32
    _gflat_ngkmax = None
    if zeta_cutoff_ry is not None and int(meta.sys_dim) != 0:
        from common.coulomb_sphere import compute_per_q_bare_coulomb_components
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
    return (q_irr_full_idx, q_irr_frac, n_q_disk, _gflat_sphere_idx_padded, _gflat_gvec_components, _gflat_ngk_per_q, _gflat_ngkmax)


def _open_zeta_fit_output(
        _gflat_gvec_components, _gflat_ngk_per_q, _gflat_ngkmax, centroid_indices, mesh_xy,
        meta, n_q_disk, n_rmu, n_rtot, output_file, vertex_mu_L, wfn, zeta_cutoff_ry):
    """Produce the collective zeta output handle after writing its headers."""
    from file_io.slab_io import SlabIO
    from file_io.mf_header import copy_mf_header
    from file_io.isdf_header import IsdfHeader, write_isdf_header
    _wfn_src_path = getattr(wfn, '_filename', None)
    if _wfn_src_path is None:
        raise ValueError(
            "fit_zeta_to_h5: wfn must expose '_filename' (the source "
            "WFN.h5 path) so mf_header can be copied verbatim into "
            "zeta_q.h5.")
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
        with SlabIO(output_file, mode='w', mesh=mesh_xy):
            pass
        if jax.process_index() == 0:
            copy_mf_header(_wfn_src_path, output_file, dst_mode='a')
            write_isdf_header(output_file, _isdf_hdr, mode='a')
        jax.experimental.multihost_utils.sync_global_devices(
            "zeta_fit_headers_written")
    with timing.section("zeta_fit.open_file"):
        _n_G_sph = (int(_gflat_ngkmax)
                     if _gflat_ngkmax is not None else n_rtot)
        zeta_io = SlabIO(output_file, mode='a', mesh=mesh_xy,
                         )
        zeta_io.create_dataset(
            'zeta_q_G',
            shape=(n_q_disk, n_rmu, _n_G_sph),
            dtype=np.complex128,
        )
    return (zeta_io, _n_G_sph)


def _prepare_zeta_fit_wavefunctions(
        band_chunk_size, band_range_full, bispinor, bispinor_lift, cache_psi_r, chunk_r,
        k_unfold_plan, mesh_xy, meta, n_rtot, nk_tot, print_fn, wfn):
    """Produce the shared wavefunction store, optional real-space cache and real-grid tiles."""
    _bfs, _bfe = band_range_full
    from runtime.padding import padded_axis
    band_chunk_size = padded_axis(
        band_chunk_size, mesh_xy, name="zeta-fit band chunk").carrier
    _bfe_transport = _bfs + padded_axis(
        _bfe - _bfs, mesh_xy, name="zeta-fit band transport").carrier
    band_chunk_ranges = [
        (_bfs + i * band_chunk_size,
         min(_bfs + (i + 1) * band_chunk_size, _bfe_transport))
        for i in range(
            (_bfe_transport - _bfs + band_chunk_size - 1) // band_chunk_size)
    ]
    from common.psi_G_store import build_psi_G_store
    psi_G_store = build_psi_G_store(
        wfn=wfn, mesh_xy=mesh_xy, meta=meta,
        band_chunk_ranges=band_chunk_ranges,
        bispinor=bispinor,
        bispinor_lift=bispinor_lift,
        k_domain=k_unfold_plan.sym.parent_k_domain,
    )
    real_grid_tiles = k_unfold_plan.real_grid_tiles(
        target_width=int(chunk_r))
    num_chunks = int(real_grid_tiles.n_tiles)
    n_rchunk = int(real_grid_tiles.width)
    if n_rchunk > int(chunk_r):
        print_fn(
            f"  *** LORRAX SANITY: orbit-closed r tiles are {n_rchunk} "
            f"slots wide but the memory plan priced chunk_r = "
            f"{int(chunk_r)}; the Z_q / solve / zeta chunk live set is "
            f"{n_rchunk / max(1, int(chunk_r)):.2f}x the planned one. ***")
    print_fn(
        f"  Z_q on raw parents: {k_unfold_plan.n_parent} parent k rows "
        f"-> {nk_tot} full k by the typed unfold on {num_chunks} "
        f"orbit-closed real-grid tiles x {n_rchunk} slots "
        f"(planned chunk_r {int(chunk_r)}; pad slots "
        f"{num_chunks * n_rchunk - n_rtot})")
    psi_r_cache = None
    if cache_psi_r:
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
        psi_G_store.release_host_tiles()
    elif jax.process_index() == 0:
        print_fn("  ψ(r) cache: disabled by the low-memory plan; streaming "
                 "one ψ(G) band chunk per r chunk")
    return (band_chunk_ranges, psi_G_store, real_grid_tiles, num_chunks, n_rchunk, psi_r_cache)


def _prepare_zeta_accumulator(
        L_q, _coupled_mu123_coordinator, _coupled_rank_gate, _coupled_solve_inputs,
        _coupled_stacked_solve, _gflat_ngkmax, _spill_coupled_gflat_to_host,
        cct_trace_per_q, gflat_chunk_size, mesh_xy, meta, n_q_disk, n_rtot, print_fn,
        q_irr_frac, vertex_mu_L):
    """Produce the sharded G-flat accumulator and coupled preparation boundary."""
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
    _gflat_chunk_size = int(gflat_chunk_size) if gflat_chunk_size else None
    if jax.process_index() == 0:
        from runtime.padding import mesh_divisor
        _p_prod = mesh_divisor(mesh_xy)
        _n_mu_local = int(meta.n_rmu_padded) // _p_prod
        _N = n_q_disk * _n_mu_local
        _cs = _gflat_chunk_size or _N
        print_fn(f"  G-flat ζ accumulator: N={_N} rows/rank "
                 f"(n_q={n_q_disk} × n_mu_local={_n_mu_local}); "
                 f"chunk_size={_cs} → "
                 f"per-iter FFT box {_cs * n_rtot * 16 / 1e9:.2f} GB/rank")
    _q_irr_frac_dev = _device_put_process_local(
        np.asarray(q_irr_frac, dtype=np.float64),
        NamedSharding(mesh_xy, P(None, None)))
    if debug_print_enabled():
        jax.block_until_ready(gflat_acc)
        jax.block_until_ready(L_q)
    mem_probe("pre_rchunk_loop")
    if _coupled_mu123_coordinator is not None:
        if _spill_coupled_gflat_to_host:
            _local_gflat_host_bytes = _local_shard_nbytes(gflat_acc)
            with timing.section("zeta_fit.gflat_spill_initial"):
                gflat_acc = _coupled_gflat_spill(
                    gflat_acc, enabled=True)
            if (jax.process_index() == 0 and int(vertex_mu_L) == 1):
                print_fn(
                    "  [bispinor] coupled G-flat host spill: "
                    f"{_local_gflat_host_bytes} bytes/channel/rank, "
                    f"{3 * _local_gflat_host_bytes} bytes/rank with all "
                    "three accumulators parked")
            mem_probe("pre_rchunk_loop_host_spilled")
        _coupled_rank_gate("prepared")
        _coupled_mu123_coordinator.channel_prepared(
            int(vertex_mu_L), solve_inputs=_coupled_solve_inputs)
        if _coupled_stacked_solve:
            L_q = None
            cct_trace_per_q = None
            _coupled_solve_inputs = None
    return (gflat_acc, _gflat_chunk_size, _q_irr_frac_dev, L_q, cct_trace_per_q, _coupled_solve_inputs)


def _prepare_coupled_zeta_tile(
        _coupled_mu123_coordinator, _coupled_stacked_solve, _resolved_solver_kind,
        _resolved_zeta_gather, _tile_args, actual_n_rchunk, band_chunk_ranges, chunk_idx,
        distrib_la_batched_route, k_unfold_plan, kgrid, mesh_xy, n_rmu_padded, n_rmu_solve,
        psi_G_store, psi_mun_parent, psi_r_cache, q_chunk_size, q_irr_full_idx,
        transverse_zeta_rcond, vertex_mu_L, weight_l_face, weight_r_face, zeta_rcond,
        zeta_ridge):
    """Produce the existing coupled tile factor or solve result."""
    _prebuilt_Z_q = None
    _prebuilt_zeta = None
    if _coupled_mu123_coordinator is not None:
        def _build_coupled_Z_q():
            return _z_q_face_parent(
                psi_mun_parent, psi_G_store, psi_r_cache,
                weight_l_face, weight_r_face,
                band_chunk_ranges=band_chunk_ranges,
                kgrid=kgrid, mesh_xy=mesh_xy,
                k_unfold_plan=k_unfold_plan, coupled_mu123=True,
                tile_r_index=_tile_args["tile_r_index"],
                tile_local_perm=_tile_args["tile_local_perm"],
                tile_wraps=_tile_args["tile_wraps"])
        if _coupled_stacked_solve:
            def _build_coupled_zeta():
                with timing.section(
                        "zeta_fit.chunk.coupled_z_q_build"):
                    Z_mu_q = _build_coupled_Z_q()
                    if q_irr_full_idx is not None:
                        Z_mu_q = Z_mu_q[:, jnp.asarray(
                            np.asarray(
                                q_irr_full_idx, dtype=np.int32))]
                def factorize(C, trace):
                    return factor_c_q(
                        C, mesh_xy, vertex_mu_L=1,
                        n_rmu_logical=n_rmu_solve,
                        solver_kind=_resolved_solver_kind,
                        zeta_ridge=zeta_ridge, zeta_rcond=zeta_rcond,
                        transverse_zeta_rcond=transverse_zeta_rcond,
                        distrib_la_batched_route=distrib_la_batched_route,
                        transverse_trace_per_q=trace)
                L_mu_q, piv_mu_q = (
                    _coupled_mu123_coordinator.
                    stacked_solve_inputs(factorize=factorize))
                n_q_solve = int(Z_mu_q.shape[1])
                Z_flat = Z_mu_q.reshape(
                    3 * n_q_solve, n_rmu_padded,
                    actual_n_rchunk)
                with timing.section(
                        "zeta_fit.chunk.coupled_solve"):
                    zeta_flat = solve_zeta(
                        L_mu_q, Z_flat, mesh_xy, q_chunk_size,
                        vertex_mu_L=1,
                        solver_kind=_resolved_solver_kind,
                        lu_piv=piv_mu_q,
                        n_rmu_logical=n_rmu_solve,
                        zeta_gather=_resolved_zeta_gather,
                        distrib_la_batched_route=(
                            distrib_la_batched_route))
                    zeta_mu_q = zeta_flat.reshape(
                        3, n_q_solve, n_rmu_padded,
                        actual_n_rchunk)
                return zeta_mu_q
            _prebuilt_zeta = (
                _coupled_mu123_coordinator.acquire_channel_zeta(
                    int(vertex_mu_L), chunk_idx,
                    _build_coupled_zeta))
        else:
            _prebuilt_Z_q = (
                _coupled_mu123_coordinator.acquire_channel_Z_q(
                    int(vertex_mu_L), chunk_idx,
                    _build_coupled_Z_q))
    return (_prebuilt_Z_q, _prebuilt_zeta)


def _fit_zeta_tile(
        L_q, _coupled_rank_gate, _prebuilt_Z_q, _prebuilt_zeta, _q_neg_idx,
        _resolved_solver_kind, _resolved_zeta_gather, _tile_args, actual_n_rchunk,
        band_chunk_ranges, cct_trace_per_q, chunk_idx, distrib_la_batched_route,
        logical_n_rchunk, lu_piv, mesh_xy, meta, psi_G_store, psi_mun_parent, psi_r_cache,
        q_chunk_size, q_irr_full_idx, vertex_mu_L, weight_l_face, weight_r_face):
    """Produce a solved zeta tile and its elapsed fit time."""
    t0 = time.perf_counter()
    with timing.section("zeta_fit.chunk.fit_one_rchunk"), \
         jax_profile.step_annotation("chunk_fit", step_num=chunk_idx):
        if _prebuilt_zeta is not None:
            zeta_chunk = _prebuilt_zeta
        else:
            zeta_chunk = fit_one_rchunk(
                psi_G_store=psi_G_store,
                psi_r_cache=psi_r_cache,
                L_q=L_q,
                mesh_xy=mesh_xy,
                meta=meta,
                band_chunk_ranges=band_chunk_ranges,
                actual_n_rchunk=actual_n_rchunk,
                q_chunk_size=q_chunk_size,
                vertex_mu_L=int(vertex_mu_L),
                solver_kind=_resolved_solver_kind,
                q_irr_full_idx=q_irr_full_idx,
                q_neg_idx=_q_neg_idx,
                cct_trace_per_q=cct_trace_per_q,
                zeta_gather=_resolved_zeta_gather,
                lu_piv=lu_piv,
                distrib_la_batched_route=distrib_la_batched_route,
                psi_mun=psi_mun_parent,
                weight_l=weight_l_face,
                weight_r=weight_r_face,
                _prebuilt_Z_q=_prebuilt_Z_q,
                **_tile_args,
            )
        if actual_n_rchunk != logical_n_rchunk:
            zeta_chunk = zeta_chunk[..., :logical_n_rchunk]
        zeta_chunk.block_until_ready()
    if _coupled_rank_gate is not None:
        _coupled_rank_gate(f"rchunk {chunk_idx} solve")
    _t_fit = time.perf_counter() - t0
    return (zeta_chunk, _t_fit)


def _accumulate_zeta_tile(
        _gflat_chunk_size, _gflat_sphere_idx_padded, _q_irr_frac_dev,
        _spill_coupled_gflat_to_host, _tile_r_indices_dev, accumulate_rchunk_to_gflat,
        gflat_acc, mesh_xy, meta, zeta_chunk):
    """Produce the updated G-flat accumulator and elapsed accumulation time."""
    t0 = time.perf_counter()
    with timing.section("zeta_fit.chunk.h5_write"):
        if _spill_coupled_gflat_to_host:
            with timing.section("zeta_fit.gflat_restore_accumulate"):
                gflat_acc = _coupled_gflat_restore(
                    gflat_acc, enabled=True)
        gflat_acc = accumulate_rchunk_to_gflat(
            rchunk=zeta_chunk, gflat_acc=gflat_acc,
            fft_grid=meta.fft_grid,
            r0=None,
            r_indices=_tile_r_indices_dev,
            sphere_idx=_gflat_sphere_idx_padded,
            qvec_frac=_q_irr_frac_dev,
            norm='backward',
            chunk_size=_gflat_chunk_size,
            mesh=mesh_xy,
        )
        del zeta_chunk
        if _spill_coupled_gflat_to_host:
            with timing.section("zeta_fit.gflat_spill_accumulated"):
                gflat_acc = _coupled_gflat_spill(
                    gflat_acc, enabled=True)
        elif debug_print_enabled():
            jax.block_until_ready(gflat_acc)
    _t_write = time.perf_counter() - t0
    return (gflat_acc, _t_write)


def _report_zeta_tile_memory(
        _dbg_rchunk, _rss0, _rss1, _rss2, _t_fit, _t_write, _trim_fn, chunk_idx, num_chunks,
        print_fn, r_end, r_start):
    """Report the existing tile memory observations after heap trimming."""
    _trimmed = _rss2
    if _trim_fn is not None:
        try:
            _trim_fn(0)
            _trimmed = _host_rss_gb()
        except Exception:
            pass
    if _dbg_rchunk and jax.process_index() == 0:
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


def _run_zeta_fit_tiles(
        L_q, _coupled_mu123_coordinator, _coupled_rank_gate, _coupled_stacked_solve,
        _gflat_chunk_size, _gflat_sphere_idx_padded, _mem_probe, _q_irr_frac_dev,
        _q_neg_idx, _resolved_solver_kind, _resolved_zeta_gather,
        _spill_coupled_gflat_to_host, _trim_fn, accumulate_rchunk_to_gflat,
        band_chunk_ranges, cct_trace_per_q, distrib_la_batched_route, gflat_acc,
        k_unfold_plan, kgrid, lu_piv, mesh_xy, meta, n_rmu_padded, n_rmu_solve, n_rtot,
        num_chunks, print_fn, psi_G_store, psi_mun_parent, psi_r_cache, q_chunk_size,
        q_irr_full_idx, r_progress, real_grid_tiles, t_fit_total, t_write_total,
        transverse_zeta_rcond, vertex_mu_L, weight_l_face, weight_r_face, zeta_rcond,
        zeta_ridge):
    """Produce the accumulated G-flat fit and fit/write timing totals."""
    with timing.section("zeta_fit.chunk_loop"):
        for chunk_idx in range(num_chunks):
            _tile_args = {}
            _tile_r_indices_dev = None
            _tile_row = np.asarray(real_grid_tiles.r_index[chunk_idx])
            _tile_perm, _tile_wraps = real_grid_tiles.source_tables(
                chunk_idx)
            _rep = NamedSharding(mesh_xy, P())
            _tile_args = dict(
                k_unfold_plan=k_unfold_plan,
                tile_r_index=_device_put_process_local(
                    _tile_row.astype(np.int32), _rep),
                tile_local_perm=_device_put_process_local(
                    _tile_perm.astype(np.int32), _rep),
                tile_wraps=_device_put_process_local(
                    _tile_wraps.astype(np.int32), _rep),
            )
            _tile_r_indices_dev = _device_put_process_local(
                np.where(
                    _tile_row >= 0, _tile_row,
                    n_rtot + np.arange(_tile_row.size)).astype(np.int32),
                _rep)
            r_start = 0
            r_end = n_rtot
            logical_n_rchunk = actual_n_rchunk = int(real_grid_tiles.width)
            _dbg_rchunk = debug_print_enabled()
            _rss0 = _host_rss_gb() if _dbg_rchunk else 0.0
            _mem_probe(f"rchunk_start chunk={chunk_idx}")
            (_prebuilt_Z_q, _prebuilt_zeta) = _prepare_coupled_zeta_tile(
                _coupled_mu123_coordinator, _coupled_stacked_solve, _resolved_solver_kind,
                _resolved_zeta_gather, _tile_args, actual_n_rchunk, band_chunk_ranges, chunk_idx,
                distrib_la_batched_route, k_unfold_plan, kgrid, mesh_xy, n_rmu_padded, n_rmu_solve,
                psi_G_store, psi_mun_parent, psi_r_cache, q_chunk_size, q_irr_full_idx,
                transverse_zeta_rcond, vertex_mu_L, weight_l_face, weight_r_face, zeta_rcond, zeta_ridge)
            (zeta_chunk, _t_fit) = _fit_zeta_tile(
                L_q, _coupled_rank_gate, _prebuilt_Z_q, _prebuilt_zeta, _q_neg_idx,
                _resolved_solver_kind, _resolved_zeta_gather, _tile_args, actual_n_rchunk,
                band_chunk_ranges, cct_trace_per_q, chunk_idx, distrib_la_batched_route,
                logical_n_rchunk, lu_piv, mesh_xy, meta, psi_G_store, psi_mun_parent, psi_r_cache,
                q_chunk_size, q_irr_full_idx, vertex_mu_L, weight_l_face, weight_r_face)
            t_fit_total += _t_fit
            _rss1 = _host_rss_gb() if _dbg_rchunk else 0.0
            _mem_probe(f"after_fit_one_rchunk chunk={chunk_idx}")
            (gflat_acc, _t_write) = _accumulate_zeta_tile(
                _gflat_chunk_size, _gflat_sphere_idx_padded, _q_irr_frac_dev,
                _spill_coupled_gflat_to_host, _tile_r_indices_dev, accumulate_rchunk_to_gflat,
                gflat_acc, mesh_xy, meta, zeta_chunk)
            del zeta_chunk
            t_write_total += _t_write
            _mem_probe(f"after_accumulate chunk={chunk_idx}")
            if _coupled_mu123_coordinator is not None:
                _coupled_mu123_coordinator.finish_chunk(
                    int(vertex_mu_L), chunk_idx)
            _rss2 = _host_rss_gb() if _dbg_rchunk else 0.0
            _report_zeta_tile_memory(
                _dbg_rchunk, _rss0, _rss1, _rss2, _t_fit, _t_write, _trim_fn, chunk_idx, num_chunks,
                print_fn, r_end, r_start)
            r_progress.step()
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
    return (gflat_acc, t_fit_total, t_write_total)


def _write_zeta_fit_result(
        _coupled_mu123_coordinator, _gflat_ngk_per_q, _spill_coupled_gflat_to_host,
        gflat_acc, mesh_xy, mu_basis, vertex_mu_L, zeta_io):
    """Write the completed G-flat tensor through the collective output handle."""
    if _coupled_mu123_coordinator is not None:
        _coupled_mu123_coordinator.wait_finalize(int(vertex_mu_L))
    if _spill_coupled_gflat_to_host:
        with timing.section("zeta_fit.gflat_restore_final_write"):
            gflat_acc = _coupled_gflat_restore(gflat_acc, enabled=True)
    with timing.section("zeta_fit.write_g_flat"):
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
        if mu_basis is not None:
            gflat_acc = mu_basis.unpack_axis(gflat_acc, 1)
        zeta_io.write_slab('zeta_q_G', gflat_acc)


def _close_zeta_fit_output(
        n_q_disk, n_rmu, n_rtot, nq, num_chunks, output_file, print_fn, psi_G_store,
        t_chunks_total, t_fit_total, t_write_total, zeta_io):
    """Close collective fit resources and report the existing completion diagnostics."""
    with timing.section("zeta_fit.close_io"):
        zeta_io.close()
    with timing.section("zeta_fit.sync_global"):
        jax.experimental.multihost_utils.sync_global_devices("zeta_writes_complete")
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
    psi_G_store.close()
    print_fn(f"  Zeta output: {output_file}  shape: "
             f"(n_q_disk={n_q_disk} of {nqx}·{nqy}·{nqz}={nq} full-BZ, "
             f"n_rtot={n_rtot}, n_rmu={n_rmu})")
    print_fn(f"  Timing ({num_chunks} r-chunks, {t_chunks_total:.1f}s total):")
    for label, t in [("fit", t_fit_total), ("H5", t_write_total)]:
        print_fn(f"    {label:<6} {t:6.2f}s  {100*t/t_chunks_total:4.1f}%")
    mem_probe("zeta_fit_end")


def fit_zeta_to_h5(
    wfn,
    sym,
    meta: Meta,
    centroid_indices: jax.Array,
    mesh_xy: Mesh,
    chunk_r: int,
    output_file: str,
    band_chunk_size: int = 16,
    q_chunk_size: int = 1,
    bispinor: bool = False,
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
    distrib_la_batched_route: str = "batch_reshard",
    gflat_chunk_size: int = 0,
    write_ibz_only: bool = True,
    zeta_cutoff_ry: float | None = None,
    cache_psi_r: bool = True,
    cache_face_y_blocks: bool = False,
    bispinor_lift: str = "raw",
    _coupled_mu123_coordinator=None,
    _coupled_rank_gate=None,
    _spill_coupled_gflat_to_host: bool = False,
    _stack_coupled_solve_inputs: bool = False,
    k_unfold_plan=None,
    psi_nmu_parent: jax.Array | None = None,
    psi_mun_parent: jax.Array | None = None,
    print_fn=print,
):
    """Fit canonical parent-face zeta; see docs/architecture/zeta_fit_face_psi_cct.md."""
    (n_rmu, n_rmu_padded, n_rmu_solve, mu_basis, n_rtot, nk_tot, kgrid, nq, num_chunks, n_rchunk, band_range_left, band_range_right, band_range_full, _q_neg_idx, write_ibz_only) = _prepare_zeta_fit_geometry(
        _coupled_mu123_coordinator, _coupled_rank_gate, _spill_coupled_gflat_to_host,
        _stack_coupled_solve_inputs, band_norms, band_range_left, band_range_right,
        cache_face_y_blocks, centroid_indices, chunk_r, k_unfold_plan, meta, output_file,
        print_fn, psi_mun_parent, psi_nmu_parent, sym, vertex_mu_L, wfn, write_ibz_only)
    (C_q_flat, weight_l_face, weight_r_face) = _build_zeta_fit_gram(
        _q_neg_idx, band_range_full, band_range_left, band_range_right, k_unfold_plan, kgrid,
        mesh_xy, mu_basis, n_rmu, n_rmu_padded, n_rmu_solve, nk_tot, nq, print_fn,
        psi_mun_parent, psi_nmu_parent, sym, vertex_mu_L, write_ibz_only)
    gc.collect()
    (L_q, lu_piv, _resolved_solver_kind, _resolved_zeta_gather, cct_trace_per_q, _coupled_stacked_solve, _coupled_solve_inputs) = _factor_zeta_fit_gram(
        C_q_flat, _coupled_mu123_coordinator, _stack_coupled_solve_inputs, charge_zeta_solve,
        distrib_la_batched_route, distributed_cholesky, distributed_lu, distributed_zeta_solve,
        mesh_xy, n_rmu_padded, n_rmu_solve, print_fn, solver_kind, transverse_zeta_rcond,
        transverse_zeta_solve, vertex_mu_L, zeta_rcond, zeta_ridge)
    del C_q_flat
    with timing.section("zeta_fit.gc_pre_chunk_loop"):
        _collect_fit_setup_garbage()
    (q_irr_full_idx, q_irr_frac, n_q_disk, _gflat_sphere_idx_padded, _gflat_gvec_components, _gflat_ngk_per_q, _gflat_ngkmax) = _prepare_zeta_output_sphere(
        meta, n_rtot, nq, print_fn, sym, wfn, write_ibz_only, zeta_cutoff_ry)
    (zeta_io, _n_G_sph) = _open_zeta_fit_output(
        _gflat_gvec_components, _gflat_ngk_per_q, _gflat_ngkmax, centroid_indices, mesh_xy,
        meta, n_q_disk, n_rmu, n_rtot, output_file, vertex_mu_L, wfn, zeta_cutoff_ry)
    (band_chunk_ranges, psi_G_store, real_grid_tiles, num_chunks, n_rchunk, psi_r_cache) = _prepare_zeta_fit_wavefunctions(
        band_chunk_size, band_range_full, bispinor, bispinor_lift, cache_psi_r, chunk_r,
        k_unfold_plan, mesh_xy, meta, n_rtot, nk_tot, print_fn, wfn)
    t_fit_total = 0.0
    t_write_total = 0.0
    t_chunk_start = time.perf_counter()
    _mem_probe = mem_probe
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
    from common.wfn_transforms import accumulate_rchunk_to_gflat
    (gflat_acc, _gflat_chunk_size, _q_irr_frac_dev, L_q, cct_trace_per_q, _coupled_solve_inputs) = _prepare_zeta_accumulator(
        L_q, _coupled_mu123_coordinator, _coupled_rank_gate, _coupled_solve_inputs,
        _coupled_stacked_solve, _gflat_ngkmax, _spill_coupled_gflat_to_host, cct_trace_per_q,
        gflat_chunk_size, mesh_xy, meta, n_q_disk, n_rtot, print_fn, q_irr_frac, vertex_mu_L)
    _trim_fn = None
    if env_bool("LORRAX_MALLOC_TRIM", True):
        try:
            import ctypes
            _trim_fn = ctypes.CDLL("libc.so.6").malloc_trim
        except Exception:
            _trim_fn = None
    (gflat_acc, t_fit_total, t_write_total) = _run_zeta_fit_tiles(
        L_q, _coupled_mu123_coordinator, _coupled_rank_gate, _coupled_stacked_solve,
        _gflat_chunk_size, _gflat_sphere_idx_padded, _mem_probe, _q_irr_frac_dev, _q_neg_idx,
        _resolved_solver_kind, _resolved_zeta_gather, _spill_coupled_gflat_to_host, _trim_fn,
        accumulate_rchunk_to_gflat, band_chunk_ranges, cct_trace_per_q,
        distrib_la_batched_route, gflat_acc, k_unfold_plan, kgrid, lu_piv, mesh_xy, meta,
        n_rmu_padded, n_rmu_solve, n_rtot, num_chunks, print_fn, psi_G_store, psi_mun_parent,
        psi_r_cache, q_chunk_size, q_irr_full_idx, r_progress, real_grid_tiles, t_fit_total,
        t_write_total, transverse_zeta_rcond, vertex_mu_L, weight_l_face, weight_r_face,
        zeta_rcond, zeta_ridge)
    t_chunks_total = time.perf_counter() - t_chunk_start
    r_progress.finish()
    _track_peak()
    _write_zeta_fit_result(
        _coupled_mu123_coordinator, _gflat_ngk_per_q, _spill_coupled_gflat_to_host, gflat_acc,
        mesh_xy, mu_basis, vertex_mu_L, zeta_io)
    del gflat_acc
    _close_zeta_fit_output(
        n_q_disk, n_rmu, n_rtot, nq, num_chunks, output_file, print_fn, psi_G_store,
        t_chunks_total, t_fit_total, t_write_total, zeta_io)
    return _peak_bytes

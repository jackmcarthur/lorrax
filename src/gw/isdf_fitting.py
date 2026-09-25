import gc
import os
import subprocess
import time
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
import jax.experimental.multihost_utils  # noqa: F401  (sync_global_devices)
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common import Meta
from common import timing
from common.collectives import (
    device_put_process_local as _device_put_process_local,
)
from runtime import debug_print_enabled

# Canonical boolean env grammar for this layer (same recognised token set
# as file_io._slab_io_ffi._env_flag and the one isdf.core imports, plus an
# announcement for anything outside it).
# See gw/gw_config.py's module comment and tests/test_env_grammar.py for
# the drift gate.
from .gw_config import (ZETA_RCOND_DEFAULT,
                        active_zeta_truncating_knobs)

from isdf.core import (
    c_q_from_psi_sm,
    complete_ordered_pair_normal_equations,
    factor_c_q,
    _resolve_solver_kind,
    _resolve_zeta_gather,
    zeta_factor_resident,
)
# The opaque distributed factor.  Re-exported through isdf.core rather than
# imported from the door here: this module never CALLS distrib_la, it only
# has to tell a token from an array (route G refuses one).
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




def add_pad_diagonal_sharded(C, active_mask, n_logical, *, mesh_xy):
    """``C + (tr C / n_logical) diag(~active_mask)`` with every operand a rank-local tile.

    ``C`` is ``(nq, n, n)`` at ``P(None, 'x', 'y')``.  Each rank sums the
    global-diagonal entries that fall inside its own ``(mu_X, nu_Y)`` tile
    (a tile touches the diagonal only where its row and column ranges
    overlap), ``psum`` gives ``tr C`` per q, and the pad diagonal is the
    same row-equals-column test masked by the pad rows of the tile.  The
    values equal the former replicated ``trace``/``diag`` expression; only
    the summation order of the trace differs (per-rank partials, then the
    mesh reduction).  No ``(n, n)`` or ``(nq, n, n)`` value is formed
    outside the rank's tile.
    """
    from jax.sharding import PartitionSpec as P
    from common.shard_map import shard_map
    nq, n, n2 = (int(v) for v in C.shape)
    px, py = int(mesh_xy.shape['x']), int(mesh_xy.shape['y'])
    if n != n2 or n % px or n % py:
        raise ValueError(
            "add_pad_diagonal_sharded: C must be a square (nq, n, n) carrier "
            f"with n divisible by the mesh; got {tuple(C.shape)} on {px}x{py}")
    mu_loc, nu_loc = n // px, n // py
    pad = jnp.asarray(~np.asarray(active_mask, dtype=bool))
    if pad.shape != (n,):
        raise ValueError(
            f"add_pad_diagonal_sharded: active_mask has shape {pad.shape}, want ({n},)")
    inv_n = 1.0 / float(n_logical)

    def _local(c, pad_rows_full):
        rows = jax.lax.axis_index('x') * mu_loc + jnp.arange(mu_loc)
        cols = jax.lax.axis_index('y') * nu_loc + jnp.arange(nu_loc)
        on_diag = rows[:, None] == cols[None, :]
        local_trace = jnp.sum(jnp.where(on_diag[None], c, 0), axis=(-2, -1))
        trace = jax.lax.psum(local_trace, ('x', 'y'))
        scale = (trace.real * inv_n).astype(c.dtype)
        pad_here = jnp.take(pad_rows_full, rows)
        mask = (on_diag & pad_here[:, None]).astype(c.dtype)
        return c + scale[:, None, None] * mask[None]

    kernel = jax.jit(shard_map(
        _local, mesh=mesh_xy, in_specs=(P(None, 'x', 'y'), P()),
        out_specs=P(None, 'x', 'y'), check_vma=False))
    return kernel(C, pad)


def _host_mem(label):
    """Rank-local host line: this process's VmRSS and the node's MemAvailable."""
    from common.gpu_utils import get_host_memory_available_gb
    from isdf.core import host_rss_gb
    avail = get_host_memory_available_gb()
    return (f"  [host mem] {label}: RSS {host_rss_gb():.1f} GiB, node MemAvailable "
            f"{'?' if avail is None else f'{avail:.1f}'} GB")


class ZetaChannel(NamedTuple):
    """One ζ channel of a route-G fit: its vertex, factor and output."""
    vertex: int              # μ_L of γ̃^{μ_L}: 0 charge, 1..3 the currents
    L_q: object              # the whole-tile factor (B, C⁺ or LU)
    lu_piv: object           # the LU pivots ('lu' seam), else None
    solver_kind: str
    write: bool              # write ``zeta_q_G`` into ``output_file``
    output_file: str


def _fit_mubatch(
    *, wfn, meta, centroid_indices, mesh_xy, plan, parent_psi,
    band_range_full, bispinor, bispinor_lift, k_unfold_plan,
    weight_l_face, weight_r_face, channels,
    zeta_gather, distrib_la_batched_route, n_rmu_solve,
    q_irr_full_idx, q_neg_idx, q_frac, sphere_idx, ngk_per_q,
    mu_basis, gvec_components, scratch_dir, print_fn,
):
    """The μ-batch fit: Z^{μ_L}_q(G) of every channel by batches, C⁻¹ on the sphere.

    See docs/architecture/zeta_fit_mubatch.md.  ``plan`` is the planner's
    :class:`gw.gflat_memory_model.MuBatchPlan`; ``channels`` the
    :class:`ZetaChannel` list.  The channels share every stage of a batch but
    the k-convolution and the accumulate; each has its own Z store.  Writes
    ``zeta_q_G`` into each channel's file G tile by G tile, one file at a
    time, and returns ``({μ_L: ZetaG}, n_batches_run, n_batches)``.
    """
    from isdf import zeta_mubatch as zmb
    from common.wfn_transforms import psi_cylinder_tables
    from runtime.padding import mesh_divisor, pad_to_axis, padded_axis

    P_ = int(mesh_divisor(mesh_xy))
    nk = int(meta.nk_tot)
    ns = int(meta.nspinor)
    mu_pad = int(meta.n_rmu_padded)
    fft_grid = tuple(int(v) for v in meta.fft_grid)
    Q = int(sphere_idx.shape[0])
    ngkmax = int(sphere_idx.shape[1])
    kgrid = tuple(int(v) for v in meta.kgrid)
    vertices = tuple(int(ch.vertex) for ch in channels)
    # The fit window's LOGICAL bands: band_range_full is the P-padded transport
    # range, whose tail past the user's last band is an exact-zero pad on the
    # faces (PsiGStore zeroes it the same way); route G reads the file, so it
    # stops at the logical edge and the pad weights are zero.
    b_lo = int(band_range_full[0])
    b_hi = min(int(band_range_full[1]),
               int(getattr(meta, "b_id_4_user", 0) or band_range_full[1]))
    nb = b_hi - b_lo
    rep = NamedSharding(mesh_xy, P())
    axis = int(np.argmax(fft_grid))

    # ---- conj ψ(G) of the raw parents, G slots over the mesh -------------
    # The pair GEMM runs on the parents; each owner unfolds its pair
    # projectors to the full zone with the plan's typed transport in G space
    # (zmb.typed_child_G_tables: the exact Fourier image of the r-space
    # transport C_q and the faces use -- rotation as a G permutation, the
    # grid-snapped τ phase, spinor U, conjugation on antiunitary rows).
    with timing.section("zeta_fit.mubatch.psi_G"):
        _pd = k_unfold_plan.sym.parent_k_domain
        if parent_psi is None:
            # The one ψ read (common.psi_G_store.load_parent_psi_G) when the
            # caller did not hand its G-slot store over with the faces.
            from common.psi_G_store import load_parent_psi_G
            parent_psi = load_parent_psi_G(
                wfn=wfn, mesh_xy=mesh_xy, meta=meta, band_range=(b_lo, b_hi),
                band_chunk=int(plan.band_chunk), centroid_indices=None,
                placement="device", bispinor=bool(bispinor),
                bispinor_lift=(bispinor_lift if bispinor else "raw"),
                k_domain=_pd, print_fn=print_fn)
        _br = tuple(int(v) for v in parent_psi.band_range)
        if _br[0] != b_lo or _br[1] < b_hi:
            raise ValueError(
                f"_fit_mubatch: the ψ(G) store holds bands {_br}, the fit "
                f"window is {(b_lo, b_hi)}")
        n_par, nb_p, _, ngk_c = (int(v) for v in parent_psi.psi_G.shape)
        g_spec = NamedSharding(mesh_xy, P(None, None, None, ('x', 'y')))
        cbar = jax.jit(jnp.conj, out_shardings=g_spec, donate_argnums=0)(
            parent_psi.psi_G)
        sphere_par = np.asarray(jax.device_get(parent_psi.sphere_index), dtype=np.int64)
        ngk_psi = ngk_c
        s_ax = padded_axis(ngk_c, P_, name="route-G ψ sphere slots")
        kv = np.asarray(wfn.kvecs(k="full_bz"), dtype=np.float64)
        kin = np.rint(kv * np.asarray(kgrid)).astype(int) % np.asarray(kgrid)
        if not np.array_equal(np.ravel_multi_index(kin.T, kgrid), np.arange(nk)):
            raise ValueError("_fit_mubatch: the loader's full-BZ rows are not the "
                             "C-order k grid the k-convolution assumes")
        kpar = np.asarray(k_unfold_plan.k_parent_frac, dtype=np.float64)
        fgv = np.asarray(fft_grid, dtype=np.int64)
        flat = np.where(sphere_par < int(np.prod(fgv)), sphere_par, 0)
        g3 = np.stack([flat // (fgv[1] * fgv[2]), (flat // fgv[2]) % fgv[1],
                       flat % fgv[2]], axis=-1).astype(np.int32)
        g3 = jax.make_array_from_callback(
            g3.shape, NamedSharding(mesh_xy, P(None, ('x', 'y'), None)),
            lambda idx: g3[idx])
        pslot, phase, anti = zmb.typed_child_G_tables(
            k_unfold_plan, fft_grid=fft_grid, sphere_par=sphere_par,
            gvec_child=np.asarray(wfn.gvecs(k="full_bz")),
            ngk_child=np.asarray(wfn.ngk_valid(k="full_bz")), k_child=kv)
        unf = tuple(_device_put_process_local(np.asarray(a), rep) for a in (
            np.asarray(k_unfold_plan.irr_idx, np.int32),
            np.asarray(k_unfold_plan.sym_idx, np.int32), anti,
            np.asarray(k_unfold_plan.spin_action_full, np.complex128), pslot, phase, kv))
        cyl = psi_cylinder_tables(wfn.box_index(k="full_bz"), fft_grid, axis,
                                  ngkmax=int(wfn.ngkmax))
    w = lambda wf: np.asarray(pad_to_axis(jnp.asarray(
        np.asarray(jax.device_get(wf))[:nb]), padded_axis(nb, nb_p, name="bands"),
        axis=0))
    w_l, w_r = w(weight_l_face), w(weight_r_face)
    if debug_print_enabled() and jax.process_index() == 0:
        print_fn(f"[mubatch_dbg] route G bands {b_lo}:{b_hi} (carrier {nb_p}) "
                 f"w_l={w_l.real.tolist()} w_r={w_r.real.tolist()} "
                 f"face weights {np.asarray(jax.device_get(weight_l_face)).tolist()} / "
                 f"{np.asarray(jax.device_get(weight_r_face)).tolist()}")

    # ---- the named pads shared by kernel, store and finalize --------------
    q_axis = padded_axis(Q, P_, name="μ-batch stored q rows")
    g_axis = padded_axis(ngkmax, int(plan.g_tile), name="μ-batch ζ-sphere G tiles")
    zt = zmb.zeta_plane_tables(gvec_components, ngk_per_q, fft_grid, axis, g_axis)
    # μ-owned rows: rank p owns slots p·c + [0, c) of every batch, whole
    # orbits per owner (the owner unfolds its own pair projectors).
    mb = zmb.best_owner_orbit_batches(k_unfold_plan, mu_pad, P_,
                                      c_max=max(1, int(plan.b) // P_))
    b = int(mb.b)
    # The widest orbit can make the bins wider than planned: the owner then
    # streams its rows through the planes at the planned width (or refuses).
    from gw.gflat_memory_model import route_g_plane_chunk
    c_out, n_blk = route_g_plane_chunk(plan, int(mb.c), P_)
    kern_args = dict(
        mesh=mesh_xy, kgrid=kgrid, fft_grid=fft_grid, ns=ns, b=b,
        q_sel=q_irr_full_idx, q_axis=q_axis, q_neg=q_neg_idx, qvec_frac=q_frac,
        n_col=int(cyl[0].shape[1]), n_s=int(cyl[0].shape[2]),
        n_pg=int(plan.r_sub), axis=axis, n_src=n_par, vertices=vertices, c_out=c_out,
        n_blk=n_blk)
    kernel = zmb.make_route_g_kernel(**kern_args)
    split_kernels = {}
    if debug_print_enabled():
        # Debug split timers: the same kernel truncated after each stage.
        for stage in ('x', 'gemm', 'a2a', 'planes', 'kconv'):
            split_kernels[stage] = zmb.make_route_g_kernel(**kern_args, stop_at=stage)
    stores = [zmb.ZStore(
        mesh=mesh_xy, q_axis=q_axis, mu_pad=mu_pad, g_axis=g_axis, b=b,
        placement=plan.placement,
        packed_from_slot=mb.slot_of_packed, n_batch=int(mb.n_batch),
        scratch_path=os.path.join(
            scratch_dir, "zeta_Z_store.scratch.h5" if v == 0
            else f"zeta_Z_store_mu{v}.scratch.h5")) for v in vertices]
    canon = np.asarray(k_unfold_plan.layout.axis.packed_to_canonical)
    x_cent = np.asarray(centroid_indices, dtype=np.float64) / np.asarray(fft_grid)
    ops = (_device_put_process_local(w_l, rep), _device_put_process_local(w_r, rep),
           _device_put_process_local(kpar, rep))
    rank_sh = NamedSharding(mesh_xy, P(('x', 'y')))
    tabs = (tuple(_device_put_process_local(np.asarray(a), rep) for a in cyl),
            tuple(_device_put_process_local(a, rep) for a in zt))
    n_batch = int(mb.n_batch)
    print_fn(f"  μ-batch fit (route G): {n_batch} batches of {b} centroids "
             f"(whole orbits per owner; planned {int(plan.b)}; planes {c_out} "
             f"of each owner's {int(mb.c)} rows at a time, {n_blk} plane block(s)), "
             f"{int(plan.r_sub)} planes per group, "
             f"{n_par} parent k -> {nk}, ψ sphere {ngk_psi} "
             f"slots ({s_ax.carrier // P_}/rank), channels μ_L={list(vertices)}, "
             f"Z store {plan.placement}")

    print_fn(_host_mem("fit start"))
    from common.progress import LoopProgress
    progress = LoopProgress(n_batch, print_fn, title="zeta fitting",
                            item_name="μ-batch",
                            max_updates=min(n_batch, 20)).start()
    _max = os.environ.get("LORRAX_MAX_RCHUNKS", "").strip()
    _max_n = int(_max) if _max else None
    if _max_n is not None and _max_n < 1:
        raise ValueError(
            f"LORRAX_MAX_RCHUNKS={_max!r} must be >= 1; unset it to fit every "
            "μ batch.")
    t_batch = 0.0
    n_run = 0
    n_go = n_batch if _max_n is None else min(n_batch, _max_n)

    def launch_args(beta):
        slots = mb.mu[beta]
        live = (slots >= 0).astype(np.float64)
        xmu = x_cent[canon[np.clip(slots, 0, None)]] * live[:, None]
        lt = tuple(jax.make_array_from_callback(a.shape, rank_sh, lambda i, a=a: a[i])
                   for a in (mb.left_perm[beta], mb.left_L[beta]))
        return (cbar, *ops, g3, _device_put_process_local(xmu, rep),
                _device_put_process_local(live, rep), *tabs, unf, lt)

    def launch(beta):
        """Dispatch batch β (asynchronous); the caller writes it later."""
        return kernel(*launch_args(beta))

    if debug_print_enabled():
        # The collective count of one batch, read from the compiled HLO.
        hlo = kernel.lower(*launch_args(0)).compile().as_text()
        n_coll = {k: hlo.count(k + '(') + hlo.count(k + '-start(')
                  for k in ('all-to-all', 'all-reduce', 'all-gather',
                            'reduce-scatter', 'collective-permute')}
        if jax.process_index() == 0:
            print_fn(f"[mubatch_dbg] route G collectives per batch (HLO): {n_coll}")
    with timing.section("zeta_fit.mubatch.loop"):
        # One batch of lookahead: β+1 is on the device while β is written.
        pending = launch(0)
        for beta in range(n_go):
            t0 = time.perf_counter()
            rows = pending
            pending = launch(beta + 1) if beta + 1 < n_go else None
            for store, r in zip(stores, rows):
                store.write_batch(beta, r)
                if len(stores) > 1:
                    store.sync()          # one handle's collective writes at a time
            del rows
            t_batch += time.perf_counter() - t0
            n_run += 1
            progress.step()
            if n_run % 10 == 0:
                print_fn(_host_mem(f"after μ-batch {n_run}"))
            if split_kernels and beta in (1, 2):
                args = launch_args(beta)
                t_stage = {}
                for stage, kfn in list(split_kernels.items()) + [('full', kernel)]:
                    ts = time.perf_counter()
                    jax.block_until_ready(kfn(*args))
                    t_stage[stage] = time.perf_counter() - ts
                if jax.process_index() == 0:
                    names = ['x', 'gemm', 'a2a', 'planes', 'kconv', 'full']
                    print_fn(f"[mubatch_dbg] batch {beta + 1} route-G split (s): " + " ".join(
                        f"{n}={t_stage[n] - (t_stage[names[i - 1]] if i else 0.0):.3f}"
                        for i, n in enumerate(names)) + f" total={t_stage['full']:.3f}")
            if debug_print_enabled() and jax.process_index() == 0:
                print_fn(f"[mubatch_dbg] batch={beta + 1}/{n_batch} "
                         f"{1e3 * (time.perf_counter() - t0):.0f}ms "
                         f"write_total={sum(st.t_write for st in stores):.2f}s")
    if n_go < n_batch:
        print_fn(f"[mubatch_dbg] LORRAX_MAX_RCHUNKS={_max_n} reached after "
                 f"μ-batch {n_run}; the fit is truncated.")
    progress.finish()
    del cbar
    print_fn(_host_mem("pre-V_q (Z store full)"))
    print_fn(f"  μ-batch timing: {n_run} batches {t_batch:.2f}s (store write incl. "
             f"the wait on the in-flight batch {sum(st.t_write for st in stores):.2f}s, "
             f"{plan.placement}, overlapped with the next batch)")

    # ---- ζ = C⁻¹ Z, held lazily; written only for a file consumer --------
    zetas = {}
    for ch, store in zip(channels, stores):
        zeta_g = zmb.ZetaG(
            store, mesh=mesh_xy, L_q=ch.L_q, lu_piv=ch.lu_piv,
            solver_kind=ch.solver_kind,
            zeta_gather=zeta_gather, batched_route=distrib_la_batched_route,
            n_rmu_solve=n_rmu_solve, n_rmu=int(meta.n_rmu), mu_basis=mu_basis,
            ngk_per_q=ngk_per_q, gvec_components=gvec_components,
            path=ch.output_file, print_fn=print_fn)
        if ch.write:
            # ONE ζ file open at a time: the SlabIO writer is asynchronous
            # and its writes are collective MPI-IO, so two handles with
            # queued writes can reach MPI in different orders on different
            # ranks and deadlock (CrI3 6x6 bispinor, three files open at
            # once: rc 137 after 25 min in H5Fclose).  Open, write, close.
            from file_io.slab_io import SlabIO
            t_w = time.perf_counter()
            with timing.section("zeta_fit.mubatch.write_zeta"):
                with SlabIO(ch.output_file, mode='a', mesh=mesh_xy) as zeta_io:
                    zeta_io.create_dataset(
                        'zeta_q_G', shape=(Q, int(meta.n_rmu), ngkmax),
                        dtype=np.complex128)
                    zeta_g.write_file(zeta_io, print_fn=print_fn)
                jax.experimental.multihost_utils.sync_global_devices(
                    "zeta_writes_complete")
            print_fn(f"  μ-batch ζ file (μ_L={ch.vertex}) written in "
                     f"{time.perf_counter() - t_w:.2f}s")
        print_fn(store.receipt())
        zetas[int(ch.vertex)] = zeta_g
    return zetas, n_run, n_batch


def fit_zeta_to_h5(
    wfn,
    sym,
    meta: Meta,
    centroid_indices: jax.Array,
    mesh_xy: Mesh,
    output_files: dict,
    *,
    band_range_left: tuple[int, int] | None = None,
    band_range_right: tuple[int, int] | None = None,
    band_norms: np.ndarray | None = None,
    bispinor: bool = False,
    bispinor_lift: str = "raw",
    solver_kind: str = 'auto',
    distributed_cholesky: str = "auto",
    distributed_lu: str = "auto",
    zeta_ridge: float = 0.0,
    charge_zeta_solve: str = "cholesky",
    distributed_zeta_solve: str = "auto",
    zeta_rcond: float = ZETA_RCOND_DEFAULT,
    distrib_la_batched_route: str = "batch_reshard",
    write_ibz_only: bool = True,
    zeta_cutoff_ry: float | None = None,
    k_unfold_plan=None,
    psi_nmu_parent: jax.Array | None = None,
    psi_mun_parent: jax.Array | None = None,
    layout="face",
    mubatch_plan=None,
    parent_psi=None,
    write_zeta_file: bool = True,
    print_fn=print,
):
    """Fit canonical q-IBZ ζ for each channel of ``output_files`` on route G.

    ``output_files`` maps μ_L to its ζ file: ``{0: path}`` for the charge
    channel, ``{1: …, 2: …, 3: …}`` (or the missing subset) for the bispinor
    current channels, which share one μ-batch loop
    (docs/architecture/zeta_fit_mubatch.md).  Each channel owns its C_q, its
    factor and its file.  Returns ``(peak_bytes, {μ_L: ZetaG})``.
    """
    if k_unfold_plan is None or psi_nmu_parent is None or psi_mun_parent is None:
        raise ValueError("fit_zeta_to_h5 requires a typed plan and both raw-parent faces.")
    if mubatch_plan is None:
        raise ValueError("fit_zeta_to_h5 requires the route-G μ-batch plan "
                         "(gw.gflat_memory_model.plan_zeta_route_g).")
    if band_norms is not None:
        raise NotImplementedError("Raw-parent zeta fitting does not support pseudobands.")
    if (int(psi_mun_parent.shape[0]) != int(k_unfold_plan.n_parent)
            or int(psi_mun_parent.shape[2]) != int(k_unfold_plan.n_centroid_packed)):
        raise ValueError("fit_zeta_to_h5: parent face extent differs from its typed plan.")
    vertices = tuple(sorted(int(v) for v in output_files))
    if not vertices or (vertices != (0,) and 0 in vertices) or any(
            v not in (0, 1, 2, 3) for v in vertices):
        raise ValueError(
            f"fit_zeta_to_h5: channels {vertices} must be (0,) (charge) or a "
            "subset of the current channels (1, 2, 3).")
    transverse = vertices != (0,)
    mem_probe("zeta_fit_start")

    # Two μ extents (common/meta.py): ``n_rmu`` is the LOGICAL centroid count
    # (the file extent); ``n_rmu_padded`` the in-memory carrier.  The dense
    # factor/solve extent is the whole carrier when the packed centroid order
    # interleaves its pad slots per shard (the pad diagonal below keeps them
    # inert), the logical prefix otherwise.
    n_rmu = meta.n_rmu
    n_rmu_padded = meta.n_rmu_padded
    n_rmu_solve = int(getattr(meta, 'mu_solve_extent', n_rmu))
    mu_basis = getattr(meta, 'mu_basis', None)
    n_rtot = meta.n_rtot
    nk_tot = meta.nk_tot
    kgrid = meta.kgrid
    nqx, nqy, nqz = kgrid
    nq = nqx * nqy * nqz

    if band_range_left is None:
        band_range_left = (meta.b_id_0, meta.b_id_3)
    if band_range_right is None:
        band_range_right = (meta.b_id_0, meta.b_id_4)

    # The production charge fit uses asymmetric serving windows: L contains
    # all occupied states plus the Sigma conduction window, while R contains
    # the Sigma occupied window plus all empty states.  Complex conjugation
    # swaps those ordered endpoints, so LR alone is not a conjugation-closed
    # training space.  Complete the *normal equations* before factor/solve;
    # no fitted zeta, V, or W is projected downstream.  The q involution is
    # owned by the symmetry service and passed into neutral ``isdf.core``.
    # The current channels train on LR alone, as they always have.
    _complete_charge_pairs = (
        not transverse and tuple(band_range_left) != tuple(band_range_right))
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
    print_fn(f"\n  Zeta fitting (route G, μ_L={list(vertices)}): {nb_full} bands "
             f"({nb_left} left + {nb_right} right)")
    for v in vertices:
        print_fn(f"  Output μ_L={v}: {output_files[v]}")

    # ── Finalize write_ibz_only BEFORE any IBZ slicing ──────────────────
    # C_q is sliced to IBZ rows below and Z_q to IBZ rows inside the kernel;
    # the two MUST agree.  The orbit-closure fallback can flip
    # write_ibz_only=False for the charge channel; the current channels
    # cannot fall back (the V_q orchestrator assumes IBZ ζ̃_T), so they
    # refuse with a hint.  One resolution point: gw.qgrid_symmetry.
    if write_ibz_only and getattr(sym, 'q_irr_full_idx', None) is not None:
        from .qgrid_symmetry import resolve_qgrid_symmetry_tables
        _res = resolve_qgrid_symmetry_tables(
            sym=sym, centroid_indices=centroid_indices,
            fft_grid=meta.fft_grid, translations=wfn.translations,
            context=("bispinor transverse ζ̃_T IBZ write"
                     if transverse else "ζ̃ IBZ write"),
            announce_fallback=not transverse,
        )
        if not _res.use_ibz:
            if transverse:
                raise RuntimeError(
                    f"Bispinor transverse zeta_T (mu_L={list(vertices)}) "
                    f"IBZ-write requested, but the transverse centroid set "
                    f"fails the orbit-closure check under the WFN sym group: "
                    f"{_res.reason}.  Regenerate the transverse centroid "
                    f"file with ``centroid.kmeans_cli --density-mode "
                    f"current`` (orbit-aware by default for ntran>1) so the "
                    f"set is closed under the spatial sym group.")
            write_ibz_only = False

    # Band windows as 0/1 weights over the face band extent (C_q and Z_q alike).
    _ns_face, _nb_face = int(psi_mun_parent.shape[1]), int(psi_mun_parent.shape[3])
    if (int(psi_nmu_parent.shape[1]) != _nb_face
            or int(psi_nmu_parent.shape[2]) != _ns_face):
        raise ValueError("fit_zeta_to_h5: parent face band/spin extents differ.")
    _off = int(band_range_full[0])
    _idx = np.arange(_nb_face)
    weight_l_face = jnp.asarray(np.where(
        (_idx >= band_range_left[0] - _off)
        & (_idx < band_range_left[1] - _off), 1.0, 0.0), dtype=jnp.float64)
    weight_r_face = jnp.asarray(np.where(
        (_idx >= band_range_right[0] - _off)
        & (_idx < band_range_right[1] - _off), 1.0, 0.0), dtype=jnp.float64)
    flat_shard = NamedSharding(mesh_xy, P(None, 'x', 'y'))

    # ---- q rows, the per-q ζ sphere (shared by every channel) ------------
    from ffi import _services
    _services.ensure_on_path()
    from symmetry_maps import bgw_integer_q_to_fractional
    if write_ibz_only:
        q_irr_full_idx = sym.q_irr_full_idx
        n_q_disk = int(q_irr_full_idx.shape[0])
        # BGW wrap THEN divide by kgrid, the V_q kernel's phase convention.
        q_irr_frac = bgw_integer_q_to_fractional(sym.q_irr_kgrid_int, meta.kgrid)
        print_fn(f"  q-IBZ reduction: {n_q_disk} IBZ q-points / {nq} full-BZ "
                 f"(disk shrink {nq / max(1, n_q_disk):.1f}×)")
    else:
        q_irr_full_idx = None
        n_q_disk = nq
        q_irr_frac = bgw_integer_q_to_fractional(sym.kvecs_asints, meta.kgrid)
        print_fn(f"  q axis on disk: full BZ ({nq} q-points) "
                 f"(write_ibz_only=False or closure check failed)")
    if zeta_cutoff_ry is None or int(meta.sys_dim) == 0:
        raise ValueError(
            "G-flat ζ writer requires a ζ sphere — pass zeta_cutoff_ry to "
            "fit_zeta_to_h5.")
    # The per-q WFN.h5-style sphere {G : |q+G|² ≤ zeta_cutoff}; ``zeta_cutoff_ry``
    # (≥ the bare-Coulomb cutoff, checked in gw_init.fit_zeta) defines it on disk.
    # The Cartesian reciprocal rows come off the vcoul door's geometry.
    from common.coulomb_sphere import compute_per_q_bare_coulomb_components
    from vcoul import CoulombGeometry
    _sphere_pkg = compute_per_q_bare_coulomb_components(
        fft_grid=meta.fft_grid, bvec=CoulombGeometry.from_wfn(wfn).bvec,
        q_irr_frac=q_irr_frac, vcoul_cutoff_ry=float(zeta_cutoff_ry),
        sys_dim=int(meta.sys_dim))
    _gflat_sphere_idx_padded = _sphere_pkg["sphere_idx_padded"]
    _gflat_gvec_components = _sphere_pkg["gvec_components_padded"]
    _gflat_ngk_per_q = _sphere_pkg["ngk_per_q"]
    _gflat_ngkmax = int(_sphere_pkg["ngkmax"])
    if jax.process_index() == 0:
        print_fn(f"  G-flat ζ sphere: ngkmax={_gflat_ngkmax}, "
                 f"min ngk={int(_gflat_ngk_per_q.min())}, "
                 f"max ngk={int(_gflat_ngk_per_q.max())} "
                 f"({_gflat_ngkmax / float(n_rtot):.3%} of n_rtot)")

    from file_io.slab_io import SlabIO
    from file_io.mf_header import copy_mf_header
    from file_io.isdf_header import IsdfHeader, write_isdf_header
    _wfn_src_path = getattr(wfn, '_filename', None)
    if _wfn_src_path is None:
        raise ValueError(
            "fit_zeta_to_h5: wfn must expose '_filename' (the source "
            "WFN.h5 path) so mf_header can be copied verbatim into "
            "zeta_q.h5.")
    _cent_idx_np = np.asarray(jax.device_get(centroid_indices), dtype=np.int32)
    if _cent_idx_np.shape != (n_rmu, 3):
        raise ValueError(
            f"fit_zeta_to_h5: centroid_indices has shape "
            f"{_cent_idx_np.shape}, expected ({n_rmu}, 3).")
    # A ζ file is written only for a consumer that reads it: restart / reuse
    # (write_restart_tensors), or a four-current V_q that forms some tiles
    # from files (a family accepted for reuse).  The caller decides.
    _write_file = bool(write_zeta_file)

    # ========== per channel: C_q, its factor, its file ==========
    from distrib_la import gemm_plan as _gemm_plan
    _mu_gemm = int(k_unfold_plan.n_centroid_packed)
    # C's enclosing JIT compiles this GEMM; standalone dummy warmup would
    # compile and execute it again without reusing that executable.
    _face_gemm = _gemm_plan(
        mesh_xy, m=_mu_gemm * _ns_face, k=_nb_face, n=_mu_gemm * _ns_face,
        nq=int(k_unfold_plan.n_parent), dtype=jnp.complex128, layout=layout,
        warmup=False)
    print_fn(f"  {_face_gemm.describe()}")
    channels = []
    _resolved_zeta_gather = None
    for v in vertices:
        with timing.section("zeta_fit.CCT"):
            # γ̃^{μ_L} on both endpoints after the typed unfold: C_q is the
            # channel's interpolation metric (Hermitian indefinite for μ_L ≠ 0).
            print_fn(f"  C_q on raw parents ({'charge γ̃^0=I' if v == 0 else f'current γ̃^{v}'}): "
                     f"{k_unfold_plan.n_parent} -> {nk_tot} k rows")
            C_q = c_q_from_psi_sm(
                kgrid=kgrid, mesh_xy=mesh_xy,
                psi_mun_parent=psi_mun_parent, psi_nmu_parent=psi_nmu_parent,
                weight_l=weight_l_face, weight_r=weight_r_face,
                gemm=_face_gemm, k_unfold_plan=k_unfold_plan,
                gamma_L=v, gamma_R=v)
            C_q_flat = jax.lax.with_sharding_constraint(
                C_q.reshape(nq, n_rmu_padded, n_rmu_padded), flat_shard)
            del C_q
            if n_rmu_solve == n_rmu_padded and n_rmu_padded > n_rmu:
                # Interleaved pad slots (orbit-packed order): C_q's pad rows and
                # columns are exact zeros.  Put C's own MEAN DIAGONAL (tr C/n per
                # q) on the pad diagonal: the factor is nonsingular, Z's zero pad
                # rows give zeta_pad = 0, and the pad eigenvalues sit inside the
                # active spectrum (a unit pad would BE lambda_max when C's scale
                # is small, and the cut would drop real modes: Si leg 20, 39 meV).
                # Rank-local by construction (Fe3GeTe2 P36/P16 OOM, 2026-09-21).
                C_q_flat = add_pad_diagonal_sharded(
                    C_q_flat, mu_basis.active_mask, float(n_rmu), mesh_xy=mesh_xy)
            if _q_neg_idx is not None:
                C_q_flat = complete_ordered_pair_normal_equations(
                    C_q_flat, _q_neg_idx)
            # IBZ cascade: slice C_q to the stored rows before the per-q factor.
            if write_ibz_only and getattr(sym, 'q_irr_full_idx', None) is not None:
                from symmetry_maps import slice_q_full_to_ibz
                C_q_flat = slice_q_full_to_ibz(
                    C_q_flat, sym.q_irr_full_idx, out_sharding=flat_shard)
            C_q_flat.block_until_ready()

        with timing.section("zeta_fit.cholesky"):
            _resolved_zeta_gather = _resolve_zeta_gather(
                distributed_zeta_solve,
                n_rmu=int(n_rmu_padded), nq=int(C_q_flat.shape[0]),
                mesh_xy=mesh_xy)
            # Route G applies a WHOLE-TILE factor on each G tile, so a current
            # channel always takes the local pivoted LU (a block-cyclic provider
            # token cannot be applied per tile).
            _kind = 'lu' if v != 0 else _resolve_solver_kind(
                mesh_xy, v, solver_kind,
                distributed_cholesky=distributed_cholesky,
                distributed_lu=distributed_lu,
                n_rmu=n_rmu_solve, nq=int(C_q_flat.shape[0]),
                charge_zeta_solve=charge_zeta_solve)
            _route = 'batch_reshard' if v != 0 else distrib_la_batched_route
            _factor = factor_c_q(
                C_q_flat, mesh_xy, vertex_mu_L=v,
                n_rmu_logical=n_rmu_solve, solver_kind=_kind,
                zeta_ridge=zeta_ridge, zeta_rcond=zeta_rcond,
                distrib_la_batched_route=_route)
            # The charge factor is one array; a current factor is (factor, piv).
            L_q, lu_piv = _factor if v != 0 else (_factor, None)
            if isinstance(L_q, FactorToken):
                raise ValueError(
                    f"fit_zeta_to_h5: μ_L={v} factor resolved to {L_q!r}; route G "
                    "applies a whole-tile factor on each G tile.")
            jax.block_until_ready(L_q)
            print_fn(f"  μ_L={v} factor: {_kind} -> "
                     f"{'hoisted pivoted LU' if lu_piv is not None else 'whole-tile'} "
                     f"{tuple(L_q.shape)}, back-solve tier {_resolved_zeta_gather}")
        with timing.section("zeta_fit.factor_residency"):
            L_q, lu_piv = zeta_factor_resident(
                L_q, lu_piv, mesh_xy, zeta_gather=_resolved_zeta_gather,
                solver_kind=_kind, distrib_la_batched_route=_route)
        del C_q_flat
        gc.collect()

        # zeta_q.h5 carries the source WFN's mf_header verbatim and an
        # isdf_header with the ζ-specific metadata.  SlabIO(mode='w') creates
        # the inode collectively so H5Fcreate applies the Lustre striping
        # hints (a rank-0 h5py create took the directory default: 1 stripe,
        # one ROMIO aggregator at P>1, measured 2026-08-07); rank 0 then
        # appends both header groups with mode='a'.  The ζ dataset is appended
        # after the loop, one file at a time (_fit_mubatch).
        if _write_file:
            path = output_files[v]
            _isdf_hdr = IsdfHeader.build(
                r_mu_fft_idx=_cent_idx_np, fft_grid=meta.fft_grid,
                density='scalar' if v == 0 else 'current', vertex_mu_L=v,
                zeta_layout='G_flat', gvec_components=_gflat_gvec_components,
                ngk_per_q=_gflat_ngk_per_q, zeta_cutoff_ry=float(zeta_cutoff_ry))
            with timing.section("zeta_fit.write_headers"):
                with SlabIO(path, mode='w', mesh=mesh_xy):
                    pass
                if jax.process_index() == 0:
                    copy_mf_header(_wfn_src_path, path, dst_mode='a')
                    write_isdf_header(path, _isdf_hdr, mode='a')
                jax.experimental.multihost_utils.sync_global_devices(
                    "zeta_fit_headers_written")
        channels.append(ZetaChannel(v, L_q, lu_piv, _kind, _write_file, output_files[v]))
    del _face_gemm
    gc.collect()

    print_fn(mubatch_plan.format())
    t_mb0 = time.perf_counter()
    zetas, n_run, n_total = _fit_mubatch(
        wfn=wfn, meta=meta, centroid_indices=centroid_indices,
        mesh_xy=mesh_xy, plan=mubatch_plan, parent_psi=parent_psi,
        band_range_full=band_range_full, bispinor=bispinor,
        bispinor_lift=bispinor_lift, k_unfold_plan=k_unfold_plan,
        weight_l_face=weight_l_face, weight_r_face=weight_r_face,
        channels=channels,
        zeta_gather=_resolved_zeta_gather,
        distrib_la_batched_route=distrib_la_batched_route,
        n_rmu_solve=n_rmu_solve, q_irr_full_idx=q_irr_full_idx,
        q_neg_idx=_q_neg_idx, q_frac=q_irr_frac,
        sphere_idx=_gflat_sphere_idx_padded, ngk_per_q=_gflat_ngk_per_q,
        mu_basis=mu_basis, gvec_components=_gflat_gvec_components,
        scratch_dir=os.path.dirname(os.path.abspath(output_files[vertices[0]])),
        print_fn=print_fn)
    # ``isdf_header/zeta_is_done`` is the file's own claim that the writer
    # finished; a truncating knob (gw_config.ZETA_TRUNCATING_ENV_KNOBS) leaves
    # it False so no restart or reuse path trusts a PARTIAL ζ.
    _trunc = active_zeta_truncating_knobs()
    if _trunc and jax.process_index() == 0:
        print_fn(f"  *** LORRAX SANITY: {_trunc} truncated this ζ fit "
                 f"({n_run} of {n_total} μ batches); ζ is PARTIAL"
                 + (" and its files are NOT marked complete." if _write_file else "."))
    for ch in channels:
        if ch.write and not _trunc and jax.process_index() == 0:
            from file_io.isdf_header import mark_zeta_done
            mark_zeta_done(ch.output_file)
    _track_peak_mb = 0
    try:
        _st = jax.local_devices()[0].memory_stats() or {}
        _track_peak_mb = int(_st.get("peak_bytes_in_use", 0) or 0)
    except Exception:
        pass
    print_fn(f"  Zeta output μ_L={list(vertices)} (n_q_disk={n_q_disk}, "
             f"n_rmu={n_rmu}); μ-batch fit {time.perf_counter() - t_mb0:.1f}s, "
             f"device peak {_track_peak_mb / 1e9:.2f} GB")
    mem_probe("zeta_fit_end")
    return _track_peak_mb, zetas

"""Parallel WFN.h5 read via phdf5 FFI + safe-corner scatter into FFT box.

End-to-end correctness + timing harness: compares the existing
``common.load_wfns.read_Gvecs_to_devices`` pipeline against a phdf5-FFI
parallel-read path that fans the wavefunction coefficients directly into
device-local band shards without a rank-0 round trip.

Two paths, both produce a real-space band-sharded wavefunction array of
shape ``(nk, nb_padded, nspinor, nx, ny, nz)`` with
``P(None, ('x','y'), None, None, None, None)``:

  (B) baseline
      WFNReader(path)  →  read_Gvecs_to_devices  →  jitted iFFT
      WFNReader eagerly slurps ``wfns/coeffs[:]`` into host RAM on every
      rank, then read_Gvecs_to_devices does per-k host-side scatter-into-
      G-space-FFT-box + make_array_from_process_local_data.

  (F) FFI parallel read
      SlabIO(mode='r', use_ffi_io=True)  →
          per-k collective H5Dread of (nb, ns, ngkmax, 2) sharded along
          the band axis over the combined ('x','y') mesh axes, no h5py
          slurp, no host copy of the whole file.
      jitted post-read kernel: mask trailing rows to zero + remap their
      gvec indices to the safe corner ``fft_grid // 2`` (guaranteed unused
      for files with ``ecutrho >= 4*ecutwfc``).
      jitted scatter into the (nb_padded, ns, nx, ny, nz) FFT box.
      same iFFT.

v1 limits: nosym files only (``ntran == 1``) so we can skip symmetry
unfolding on the (F) path.  Planned v2 ports the U_spinor + Umklapp + τ
phase from ``SymMaps.get_cnk_fullzone_batch`` into the jitted kernel.

Correctness check: bitwise-equal real-space outputs.
Timing: per-stage walls, iters-mean summary, peak host RSS before/after.

Usage (4-GPU on Perlmutter — requires the phdf5 FFI + PMIx):

    lxalloc
    export SLURM_JOBID=<from lxalloc>
    LORRAX_NGPU=4 LORRAX_MPI_TYPE=pmix \
        src/ffi/common/cpp/run_shifter.sh env \
        XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async \
        HDF5_USE_FILE_LOCKING=FALSE \
        python3 -u -m common.phdf5_wfn_read_test \
            --wfn /pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/qe/nscf/WFN.h5
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

import argparse
import resource
import sys
import time
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

# Multi-process bootstrap (SLURM-aware, same pattern as phdf5_write_test).
_DIST_SENTINEL = "_LORRAX_JAX_DISTRIBUTED_DONE"
def _maybe_init_jax_distributed():
    if os.environ.get(_DIST_SENTINEL):
        return
    proc_count = int(os.environ.get("JAX_PROCESS_COUNT",
                         os.environ.get("JAX_NUM_PROCESSES",
                         os.environ.get("SLURM_NTASKS", "1"))))
    if proc_count > 1:
        try:
            jax.distributed.initialize()
        except Exception:
            pass
    os.environ[_DIST_SENTINEL] = "1"
_maybe_init_jax_distributed()

from jax.experimental import multihost_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.wfnreader import WFNReader
from common.symmetry_maps import SymMaps
from common.load_wfns import read_Gvecs_to_devices
from common.fft_helpers import make_jittable_local_ifftn_3d
from file_io.slab_io import SlabIO
from ffi.phdf5.read import (
    read_kchunk_sharded, read_kchunk_union_sharded, ffi_read_call,
)
from ffi.common import ffi_loader


# =========================================================================
# helpers
# =========================================================================
def _log(msg: str) -> None:
    if jax.process_index() == 0:
        print(msg, flush=True)


def _rss_gb() -> float:
    """Peak host RSS this process has seen, in GB.  ru_maxrss is KiB on Linux."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def _sync(tag: str) -> None:
    try:
        multihost_utils.sync_global_devices(tag)
    except Exception:
        pass


def _time(fn, tag: str, *args, wait_array: bool = True, **kw):
    """Sync-time a callable.  Returns (dt_seconds, result).

    ``wait_array`` controls whether we call ``jax.block_until_ready`` on
    the return value — enable for stages whose output is a jax.Array,
    disable for things like file-handle construction that return opaque
    Python objects.
    """
    _sync(f"{tag}_start")
    t0 = time.perf_counter()
    out = fn(*args, **kw)
    if wait_array and isinstance(out, jax.Array):
        jax.block_until_ready(out)
    _sync(f"{tag}_end")
    return time.perf_counter() - t0, out


# =========================================================================
# (F) FFI parallel-read path
# =========================================================================
def _fft_safe_corner(fft_grid: tuple[int, int, int]) -> np.ndarray:
    """The FFT-box slot guaranteed unused by any WFN G-vector when
    ecutrho >= 4*ecutwfc.  See reports/... (TBD).
    """
    nx, ny, nz = (int(x) for x in fft_grid)
    return np.asarray([nx // 2, ny // 2, nz // 2], dtype=np.int32)


def _assert_safe_corner_ok(wfn: WFNReader) -> None:
    """Fail fast if ecutrho < 4*ecutwfc OR any WFN G-vec wraps to the midpoint."""
    ecutrho = float(wfn.ecutrho)
    ecutwfc = float(wfn.ecutwfc)
    if ecutrho + 1e-8 < 4.0 * ecutwfc:
        raise RuntimeError(
            f"safe-corner scatter requires ecutrho >= 4*ecutwfc; "
            f"got ecutrho={ecutrho}, ecutwfc={ecutwfc}")
    # Belt + suspenders: empirical check on the actual gvecs in the file.
    fft = np.asarray(wfn.fft_grid, dtype=np.int64)
    wrapped = wfn.gvecs % fft[None, :]
    mid = fft // 2
    hits = np.all(wrapped == mid[None, :], axis=1)
    if np.any(hits):
        raise RuntimeError(
            f"FFT midpoint {tuple(int(x) for x in mid)} is hit by "
            f"{int(hits.sum())} WFN G-vectors — can't use as safe corner")


def _build_global_gather_idx(wfn: WFNReader) -> np.ndarray:
    """Precompute ``global_idx[k, nx, ny, nz]`` = position in the
    ngktot-long G axis of ``big`` where ``(k, nx, ny, nz)``'s coefficient
    lives, or ``ngktot`` (sentinel) if this FFT cell has no coefficient
    for this k.  At runtime we pad ``big`` with a zero row at index
    ngktot, then do ONE ``jnp.take`` gather of shape
    ``(bpr, ns, ngktot+1, 2) → (bpr, ns, nk, nx, ny, nz, 2)``, transpose
    to ``(nk, bpr, ns, nx, ny, nz, 2)``, and combine re+im to complex.
    No per-k vmap, no per-k dynamic_slice — one kernel launch.
    """
    nk = int(wfn.nkpts)
    ngk = np.asarray(wfn.ngk, dtype=np.int64)
    kpt_starts = np.asarray(wfn.kpt_starts, dtype=np.int64)
    ngktot = int(kpt_starts[-1] + ngk[-1]) if nk > 0 else 0
    fft_grid = np.asarray(wfn.fft_grid, dtype=np.int64)
    nx, ny, nz = (int(x) for x in fft_grid)
    gvecs_all = np.asarray(wfn.gvecs, dtype=np.int32)  # (ngktot, 3)

    # Sentinel = ngktot means "gather from the zero-pad row at the end".
    global_idx = np.full((nk, nx, ny, nz), ngktot, dtype=np.int32)
    for k in range(nk):
        start = int(kpt_starts[k])
        n = int(ngk[k])
        gv = gvecs_all[start:start + n]
        wrapped = gv % fft_grid[None, :]
        for g_idx, (gx, gy, gz) in enumerate(wrapped):
            global_idx[k, int(gx), int(gy), int(gz)] = int(start + g_idx)
    return global_idx


def _build_gvecs_slabs(wfn: WFNReader, ngkmax: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute per-k fixed-width gvec slabs and mask-bounds on host.

    Returns
    -------
    gvecs_slabs: (nk, ngkmax, 3) int32 — padded with safe-corner rows where
        the slab extends past ng[k] (or where we had to backshift the slab
        for the last k to avoid overflowing ngktot).
    slab_offsets: (nk,) int64 — the slab_start used for the FFI read
        (kpt_starts[k] except for the last k where it's ngktot-ngkmax).
    valid_starts: (nk,) int32 — start of valid rows within each slab.
    valid_ends:   (nk,) int32 — end   of valid rows within each slab (exclusive).
    """
    nk = int(wfn.nkpts)
    ngk = np.asarray(wfn.ngk, dtype=np.int64)
    kpt_starts = np.asarray(wfn.kpt_starts, dtype=np.int64)
    ngktot = int(kpt_starts[-1] + ngk[-1]) if nk > 0 else 0
    gvecs_all = np.asarray(wfn.gvecs, dtype=np.int32)  # (ngktot, 3)

    slab_offsets = np.zeros(nk, dtype=np.int64)
    valid_starts = np.zeros(nk, dtype=np.int32)
    valid_ends   = np.zeros(nk, dtype=np.int32)
    for k in range(nk):
        # If reading ngkmax rows from kpt_starts[k] would overflow the
        # dataset, backshift the slab so we read the final ngkmax rows
        # and track where the valid data lives within that slab.
        if kpt_starts[k] + ngkmax <= ngktot:
            slab_start = int(kpt_starts[k])
            valid_start = 0
            valid_end = int(ngk[k])
        else:
            slab_start = int(ngktot - ngkmax)
            valid_start = int(kpt_starts[k] - slab_start)
            valid_end = valid_start + int(ngk[k])
        slab_offsets[k] = slab_start
        valid_starts[k] = valid_start
        valid_ends[k] = valid_end

    # Host-build gvecs_slabs with safe-corner padding at the invalid rows
    # so the jitted scatter can unconditionally dispatch — cheap host op
    # since gvecs is small (MB).
    corner = _fft_safe_corner(tuple(int(x) for x in wfn.fft_grid))
    gvecs_slabs = np.broadcast_to(corner[None, None, :],
                                  (nk, ngkmax, 3)).copy()
    for k in range(nk):
        s = int(slab_offsets[k])
        gvecs_slabs[k] = gvecs_all[s:s + ngkmax]
        vs = int(valid_starts[k])
        ve = int(valid_ends[k])
        # Zero-out any padding rows at the slab ends that lie outside
        # [vs, ve) — those will match the safe corner after the remap.
        if vs > 0:
            gvecs_slabs[k, :vs] = corner
        if ve < ngkmax:
            gvecs_slabs[k, ve:] = corner

    return gvecs_slabs, slab_offsets, valid_starts, valid_ends


def _make_bigread_kernel(
    mesh: Mesh, ngktot: int, nb: int, nspinor: int,
    ctx_handle: int, ds_id: int,
):
    """One FFI read → ``(nb, ns, ngktot, 2)`` sharded on band axis.

    Separating this from the scatter kernel gives XLA two small HLO
    modules to optimize instead of one big one; combining them into the
    same shard_map body dropped scatter throughput from ~200 ms
    (standalone) to ~380 ms (combined) at MoS2 3×3 scale.
    """
    p = int(mesh.shape["x"])
    q = int(mesh.shape["y"])
    world = p * q
    if nb % world != 0:
        raise RuntimeError(f"band count {nb} not divisible by world={world}")
    bpr = nb // world

    out_struct_shard = jax.ShapeDtypeStruct(
        (bpr, nspinor, ngktot, 2), jnp.float64)
    mesh_shape_attr = (p, q)
    axis_count_per_dim = (2, 0, 0, 0)
    axis_flat = (0, 1)

    def _per_rank(offset_base):
        return ffi_read_call(
            out_struct_shard, offset_base,
            ctx_handle=int(ctx_handle), ds_id=int(ds_id),
            mesh_shape=mesh_shape_attr,
            axis_count_per_dim=axis_count_per_dim,
            axis_flat=axis_flat,
        )
    sm_bare = shard_map(
        _per_rank, mesh=mesh,
        in_specs=(P(),),
        out_specs=P(("x", "y"), None, None, None),
        check_rep=False,
    )
    return jax.jit(sm_bare)


def _make_bigread_scatter_kernel(
    mesh: Mesh, nk: int, ngktot: int, ngkmax: int, nb: int, nspinor: int,
    fft_grid: tuple[int, int, int],
):
    """Scatter half (LEGACY — replaced by _make_gather_kernel).  Kept
    around as a reference for the unique_indices=True scatter path."""
    nx, ny, nz = (int(x) for x in fft_grid)
    p = int(mesh.shape["x"])
    q = int(mesh.shape["y"])
    world = p * q
    bpr = nb // world

    corner_np = _fft_safe_corner(fft_grid)
    trace_counter = [0]

    def _per_rank(big, slab_offsets, gvec_stack, vs, ve):
        trace_counter[0] += 1
        idx = jnp.arange(ngkmax, dtype=jnp.int32)
        corner = jnp.asarray(corner_np, dtype=jnp.int32)
        n_rtot = nx * ny * nz

        def body_one_k(slab_start, vs_k, ve_k, gvec_k):
            slab = jax.lax.dynamic_slice(
                big,
                (jnp.int32(0), jnp.int32(0),
                 slab_start.astype(jnp.int32), jnp.int32(0)),
                (bpr, nspinor, ngkmax, 2))
            mask = (idx >= vs_k) & (idx < ve_k)
            cnk = slab[..., 0] + 1j * slab[..., 1]
            gx = gvec_k[:, 0].astype(jnp.int32)
            gy = gvec_k[:, 1].astype(jnp.int32)
            gz = gvec_k[:, 2].astype(jnp.int32)
            lin = gx * (ny * nz) + gy * nz + gz
            lin = jnp.where(mask, lin, jnp.int32(-1))
            psi_flat = jnp.zeros((bpr, nspinor, n_rtot), dtype=jnp.complex128)
            psi_flat = psi_flat.at[:, :, lin].set(
                cnk, mode="drop", unique_indices=True)
            return psi_flat.reshape(bpr, nspinor, nx, ny, nz)

        return jax.vmap(body_one_k, in_axes=(0, 0, 0, 0))(
            slab_offsets, vs, ve, gvec_stack)

    sm_bare = shard_map(
        _per_rank, mesh=mesh,
        in_specs=(P(("x", "y"), None, None, None),
                  P(None), P(None, None, None), P(None), P(None)),
        out_specs=P(None, ("x", "y"), None, None, None, None),
        check_rep=False,
    )
    jitted = jax.jit(sm_bare)
    jitted._trace_counter = trace_counter  # type: ignore[attr-defined]
    return jitted


def _build_within_k_inv(wfn: WFNReader, ngkmax: int) -> np.ndarray:
    """Per-k inverse index ``inv[k, nx, ny, nz]`` — position within k's
    ngkmax-wide slab where the coefficient at FFT cell (nx, ny, nz)
    lives, or ``ngkmax`` (sentinel) if this cell has no coefficient.

    Intended for the union-read path: the output is
    ``(bpr, ns, nk, ngkmax, 2)`` and for each (k, nx, ny, nz) we gather
    from the ngkmax-wide slab dedicated to that k (zero-padded past
    ngk[k]).  No need to know file offsets here.
    """
    nk = int(wfn.nkpts)
    ngk = np.asarray(wfn.ngk, dtype=np.int64)
    kpt_starts = np.asarray(wfn.kpt_starts, dtype=np.int64)
    fft_grid = np.asarray(wfn.fft_grid, dtype=np.int64)
    nx, ny, nz = (int(x) for x in fft_grid)
    gvecs_all = np.asarray(wfn.gvecs, dtype=np.int32)

    inv = np.full((nk, nx, ny, nz), ngkmax, dtype=np.int32)
    for k in range(nk):
        start = int(kpt_starts[k])
        n = int(ngk[k])
        gv = gvecs_all[start:start + n]
        wrapped = gv % fft_grid[None, :]
        for g_idx, (gx, gy, gz) in enumerate(wrapped):
            inv[k, int(gx), int(gy), int(gz)] = int(g_idx)
    return inv


def _make_union_gather_kernel(
    mesh: Mesh, nk: int, ngkmax: int, nb: int, nspinor: int,
    fft_grid: tuple[int, int, int],
):
    """Gather kernel for the UNION-read output.

    Input ``big`` shape: ``(bpr, ns, nk, ngkmax, 2)`` — per-k slabs with
    G axis zero-padded past ngk[k].  Input ``inv`` shape:
    ``(nk, nx, ny, nz)`` — per-k g-local index or ngkmax sentinel.

    Gather: for each (b, s, k, x, y, z),
      out[b, s, k, x, y, z] = cnk_padded[b, s, k, inv[k, x, y, z]]
    where cnk_padded has one extra zero row at index ngkmax.  Implement
    as a single ``jnp.take`` along axis 2 of the flattened
    ``(bpr, ns, nk*(ngkmax+1))`` using ``flat_idx[k,x,y,z] =
    k*(ngkmax+1) + inv[k,x,y,z]``.
    """
    nx, ny, nz = (int(x) for x in fft_grid)
    p = int(mesh.shape["x"])
    q = int(mesh.shape["y"])
    world = p * q
    bpr = nb // world
    trace_counter = [0]

    def _per_rank(big, inv):
        trace_counter[0] += 1
        # big: (bpr, ns, nk, ngkmax, 2) f64
        # Complex conversion, then zero-pad an extra slot at ngkmax.
        cnk = big[..., 0] + 1j * big[..., 1]               # (bpr, ns, nk, ngkmax)
        zero_pad = jnp.zeros((bpr, nspinor, nk, 1), dtype=jnp.complex128)
        cnk_padded = jnp.concatenate([cnk, zero_pad], axis=-1)  # (bpr, ns, nk, ngkmax+1)
        # Flatten (nk, ngkmax+1) → nk*(ngkmax+1) for a 1-D gather.
        cnk_flat = cnk_padded.reshape(bpr, nspinor, nk * (ngkmax + 1))
        # Flat index combining (k, g_local).
        flat_idx = (jnp.arange(nk, dtype=jnp.int32)[:, None, None, None]
                    * (ngkmax + 1)) + inv
        # Gather → (bpr, ns, nk, nx, ny, nz)
        gathered = jnp.take(cnk_flat, flat_idx, axis=2)
        # Move nk to front: (nk, bpr, ns, nx, ny, nz)
        return jnp.transpose(gathered, (2, 0, 1, 3, 4, 5))

    sm_bare = shard_map(
        _per_rank, mesh=mesh,
        in_specs=(P(("x", "y"), None, None, None, None),   # big (bpr-sharded)
                  P(None, None, None, None)),                # inv (replicated)
        out_specs=P(None, ("x", "y"), None, None, None, None),
        check_rep=False,
    )
    jitted = jax.jit(sm_bare)
    jitted._trace_counter = trace_counter  # type: ignore[attr-defined]
    return jitted


def _make_gather_kernel(
    mesh: Mesh, nk: int, ngktot: int, ngkmax: int, nb: int, nspinor: int,
    fft_grid: tuple[int, int, int],
):
    """Gather half — ONE global gather for all k-points at once.

    Takes the ``(bpr, ns, ngktot, 2)`` big-read output + the precomputed
    ``global_idx[k, nx, ny, nz]`` index array (with ``ngktot`` sentinel
    for empty FFT cells), pads big with one zero row at index ngktot,
    and does ONE ``jnp.take`` along axis 2 → ``(bpr, ns, nk, nx, ny, nz, 2)``.
    Transpose + combine re+im → ``(nk, bpr, ns, nx, ny, nz)``.

    No per-k dynamic_slice, no vmap over k — single kernel launch for
    the whole k-chunk.
    """
    nx, ny, nz = (int(x) for x in fft_grid)
    p = int(mesh.shape["x"])
    q = int(mesh.shape["y"])
    world = p * q
    bpr = nb // world
    trace_counter = [0]

    def _per_rank(big, global_idx):
        trace_counter[0] += 1
        # Pad with one zero row at the end of the ngktot axis so the
        # sentinel index (== ngktot) gathers exact zero for empty cells.
        zero_pad = jnp.zeros((bpr, nspinor, 1, 2), dtype=jnp.float64)
        big_padded = jnp.concatenate([big, zero_pad], axis=2)
        # Single gather: (bpr, ns, ngktot+1, 2) → take indices global_idx
        # of shape (nk, nx, ny, nz) along axis 2 → (bpr, ns, nk, nx, ny, nz, 2)
        gathered = jnp.take(big_padded, global_idx, axis=2)
        # Move nk to front: (nk, bpr, ns, nx, ny, nz, 2)
        gathered = jnp.transpose(gathered, (2, 0, 1, 3, 4, 5, 6))
        # Real + imag → complex, trailing dim 2 collapses.
        return gathered[..., 0] + 1j * gathered[..., 1]

    sm_bare = shard_map(
        _per_rank, mesh=mesh,
        in_specs=(P(("x", "y"), None, None, None),       # big
                  P(None, None, None, None)),             # global_idx
        out_specs=P(None, ("x", "y"), None, None, None, None),
        check_rep=False,
    )
    jitted = jax.jit(sm_bare)
    jitted._trace_counter = trace_counter  # type: ignore[attr-defined]
    return jitted


def _make_kchunk_scatter_kernel(
    mesh: Mesh, nk: int, ngkmax: int, nb: int, nspinor: int,
    fft_grid: tuple[int, int, int],
):
    """Per-rank post-read scatter kernel for a full k-chunk.

    Takes the raw ``(nk, bpr, ns, ngkmax, 2)`` f64 slab returned by
    ``read_kchunk_sharded`` (sharded on dim 1) + per-k gvec/mask tables
    and produces the ``(nk, bpr, ns, nx, ny, nz)`` c128 G-space FFT box.

    The k-loop is Python-unrolled inside the shard_map body — for modest
    nk (≲ tens), this gives XLA a fully unrolled HLO and a single jit
    compile; for large nk (hundreds/thousands at Si 10×10×10 etc.) we'd
    switch to ``jax.lax.scan`` to keep HLO size bounded.  One compile,
    regardless.  Trace counter exposed on the returned callable.
    """
    nx, ny, nz = (int(x) for x in fft_grid)
    p = int(mesh.shape["x"])
    q = int(mesh.shape["y"])
    world = p * q
    if nb % world != 0:
        raise RuntimeError(f"band count {nb} not divisible by world={world}")
    bpr = nb // world

    corner_np = _fft_safe_corner(fft_grid)

    trace_counter = [0]

    def _per_rank(raw_local, gvec_stack, vs, ve):
        trace_counter[0] += 1
        # raw_local: (nk, bpr, ns, ngkmax, 2) f64 — per-rank view of the
        # band-sharded kchunk output.
        # gvec_stack: (nk, ngkmax, 3) int32, replicated.
        # vs, ve    : (nk,) int32, replicated — per-k valid-row bounds.
        idx = jnp.arange(ngkmax, dtype=jnp.int32)
        corner = jnp.asarray(corner_np, dtype=jnp.int32)

        psi_outs = []
        for k in range(nk):
            raw_k = raw_local[k]                  # (bpr, ns, ngkmax, 2)
            gvec_k = gvec_stack[k]                # (ngkmax, 3)
            mask = (idx >= vs[k]) & (idx < ve[k]) # (ngkmax,)
            cnk = (raw_k[..., 0] + 1j * raw_k[..., 1]) * mask
            gvec = jnp.where(mask[:, None], gvec_k, corner[None, :])
            psi = jnp.zeros((bpr, nspinor, nx, ny, nz), dtype=jnp.complex128)
            psi = psi.at[:, :, gvec[:, 0], gvec[:, 1], gvec[:, 2]].set(cnk)
            psi_outs.append(psi)
        return jnp.stack(psi_outs, axis=0)         # (nk, bpr, ns, nx, ny, nz)

    sm_bare = shard_map(
        _per_rank, mesh=mesh,
        in_specs=(P(None, ("x", "y"), None, None, None),   # raw: nk_rep, bands on XY
                  P(None, None, None),                      # gvec_stack replicated
                  P(None), P(None)),                        # vs, ve replicated
        out_specs=P(None, ("x", "y"), None, None, None, None),
        check_rep=False,
    )
    jitted = jax.jit(sm_bare)
    jitted._trace_counter = trace_counter  # type: ignore[attr-defined]
    return jitted


def run_ffi_path(
    wfn_path: str, wfn: WFNReader, band_range: tuple[int, int],
    mesh: Mesh, ngkmax: int,
):
    """(F) path: phdf5 parallel read → static-shape scatter → iFFT.

    Returns
    -------
    psi_r_global: (nk, nb_padded, nspinor, nx, ny, nz) c128, sharded
    stage_times: dict — per-stage wall times (open, read+scatter, fft)
    """
    nk = int(wfn.nkpts)
    nspinor = int(wfn.nspinor)
    fft_grid = tuple(int(x) for x in wfn.fft_grid)
    nx, ny, nz = fft_grid
    b_lo, b_hi = band_range
    nb = b_hi - b_lo

    p = int(mesh.shape["x"])
    q = int(mesh.shape["y"])
    world = p * q
    bands_per_shard = (nb + world - 1) // world
    nb_padded = bands_per_shard * world

    # Must divide evenly across mesh for the band-axis shard_map.
    if nb % world != 0:
        raise RuntimeError(
            f"band count {nb} not divisible by world={world}; pick "
            f"a band_range multiple of world for this test.")

    # Host prep: per-k within-slab inverse index (nk, nx, ny, nz) int32,
    # plus per-k (offset, count) tables sorted ascending by file offset
    # (the union handler requires sorted).
    inv_within_np = _build_within_k_inv(wfn, ngkmax)
    inv_within_dev = jax.device_put(
        jnp.asarray(inv_within_np),
        NamedSharding(mesh, P(None, None, None, None)))

    kpt_starts_np = np.asarray(wfn.kpt_starts, dtype=np.int64)
    ngk_np = np.asarray(wfn.ngk, dtype=np.int64)
    # MoS2 nosym: kpt_starts already ascending (k-space-parallel file).
    # For symmetric files with a sparse / out-of-order k subset, the
    # caller would sort here and permute output k afterward; skipped
    # for this test since nosym is already sorted.
    offsets_u_np = np.stack([
        np.array([b_lo, 0, int(kpt_starts_np[k]), 0], dtype=np.int64)
        for k in range(nk)
    ], axis=0)
    counts_u_np = np.stack([
        np.array([bands_per_shard, nspinor, int(ngk_np[k]), 2], dtype=np.int64)
        for k in range(nk)
    ], axis=0)
    offsets_u_dev = jax.device_put(jnp.asarray(offsets_u_np),
                                   NamedSharding(mesh, P(None, None)))
    counts_u_dev = jax.device_put(jnp.asarray(counts_u_np),
                                  NamedSharding(mesh, P(None, None)))

    # iFFT.
    ifftn = make_jittable_local_ifftn_3d(
        mesh,
        P(None, ("x", "y"), None, None, None, None),
        P(None, ("x", "y"), None, None, None, None),
    )

    stage_times = {"open": 0.0, "read_scatter": 0.0, "fft": 0.0}

    # ---- stage: open ----
    def _open():
        return SlabIO(wfn_path, mode="r", mesh=mesh, use_ffi_io=True)
    t_open, io = _time(_open, "wfn_ffi_open", wait_array=False)
    stage_times["open"] = t_open

    try:
        # Union kchunk read: ONE H5Dread pulls per-k ngk[k]-sized slabs
        # via compound H5S_SELECT_OR, placing each k's data in its own
        # stripe of a (bpr, ns, nk, ngkmax, 2) per-rank buffer.
        ctx_handle = io._backend.fh
        ngktot = int(wfn.kpt_starts[-1] + wfn.ngk[-1])

        union_read = read_kchunk_union_sharded(
            ctx_handle, "wfns/coeffs",
            n_kchunk=nk,
            file_global_shape=(int(wfn.nbands), nspinor, ngktot, 2),
            per_rank_file_shape=(bands_per_shard, nspinor, ngkmax, 2),
            kchunk_axis=2,                             # between ns and G
            dtype=np.float64, mesh=mesh,
            file_partition_spec=P(("x", "y"), None, None, None),
        )
        gather_j = _make_union_gather_kernel(
            mesh, nk, ngkmax, nb_padded, nspinor, fft_grid)

        # ---- stage: read + gather (internal split timing) ----
        t_rs0 = time.perf_counter()
        big = union_read(offsets_u_dev, counts_u_dev)    # (bpr, ns, nk, ngkmax, 2)
        jax.block_until_ready(big)
        t_r1 = time.perf_counter()
        psi_G_global = gather_j(big, inv_within_dev)
        jax.block_until_ready(psi_G_global)
        t_r2 = time.perf_counter()
        stage_times["read_scatter"] = t_r2 - t_rs0
        stage_times["_read_only"] = t_r1 - t_rs0
        stage_times["_gather_only"] = t_r2 - t_r1

        # ---- stage: iFFT to real space ----
        t_fft, psi_r = _time(lambda psi_G: ifftn(psi_G), "ffi_fft", psi_G_global)
        stage_times["fft"] = t_fft
        stage_times["_trace_count"] = int(gather_j._trace_counter[0])
    finally:
        io.close()
    return psi_r, stage_times


# =========================================================================
# (B) baseline path
# =========================================================================
def run_baseline_path(
    wfn_path: str, band_range: tuple[int, int], mesh: Mesh,
):
    """(B) path: WFNReader slurp + read_Gvecs_to_devices + iFFT.

    Returns (psi_r, stage_times) matching run_ffi_path.
    """
    stage_times = {"open": 0.0, "read_scatter": 0.0, "fft": 0.0}

    # stage: open (WFNReader also slurps coeffs here).
    def _open():
        wfn = WFNReader(wfn_path)
        sym = SymMaps(wfn)
        return wfn, sym
    t_open, (wfn, sym) = _time(_open, "base_open", wait_array=False)
    stage_times["open"] = t_open

    nspinor = int(wfn.nspinor)
    fft_grid = tuple(int(x) for x in wfn.fft_grid)

    # stage: read + scatter — the FFT box in G-space.
    from common.meta import Meta
    # Minimal Meta-like struct: only the fields read_Gvecs_to_devices reads.
    # We hand-build one instead of full Meta.from_system to avoid requiring
    # nval/ncond/nband — this test only probes the read pipeline.
    import types
    meta = types.SimpleNamespace(
        fft_grid=fft_grid,
        nspinor=nspinor,
        nspinor_wfnfile=nspinor,
        nk_tot=int(sym.nk_tot),
    )

    def _read():
        psi_G, _ = read_Gvecs_to_devices(
            wfn, sym, band_range, meta, bispinor=False, mesh_xy=mesh)
        return psi_G
    t_rs, psi_G = _time(_read, "base_read")
    stage_times["read_scatter"] = t_rs

    # stage: iFFT to real space.
    ifftn = make_jittable_local_ifftn_3d(
        mesh,
        P(None, ("x", "y"), None, None, None, None),
        P(None, ("x", "y"), None, None, None, None),
    )
    t_fft, psi_r = _time(lambda: ifftn(psi_G), "base_fft")
    stage_times["fft"] = t_fft

    return psi_r, stage_times


# =========================================================================
# comparison
# =========================================================================
def compare_sharded(psi_B: jax.Array, psi_F: jax.Array) -> float:
    """Max |B - F| reduced across ranks.  Both inputs must have the same
    sharding — returned value is global."""
    diff = jnp.abs(psi_B - psi_F).max()
    diff_host = float(jax.device_get(diff))
    # All-reduce the scalar across processes (they all see the same value
    # for a replicated scalar, but be defensive).
    all_diffs = np.asarray(
        multihost_utils.process_allgather(jnp.asarray(diff_host), tiled=False))
    return float(np.max(all_diffs))


# =========================================================================
# main
# =========================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wfn",
                    default="/pscratch/sd/j/jackm/lorrax_sandbox/runs/"
                            "MoS2/02_mos2_3x3_nosym/qe/nscf/WFN.h5",
                    help="WFN.h5 path (nosym file for v1).")
    ap.add_argument("--band-range", type=int, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="Band range [LO, HI).  Default: all bands, trimmed "
                         "down to a multiple of world size.")
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    world = jax.process_count()
    if world == 4:
        p, q = 2, 2
    elif world == 1:
        p, q = 1, 1
    else:
        p, q = world, 1

    _log(f"world={world}, mesh=({p},{q}), wfn={args.wfn}")
    _log(f"rss_pre_any_read = {_rss_gb():.3f} GB")

    devices = np.asarray(jax.devices()).reshape(p, q)
    mesh = Mesh(devices, axis_names=("x", "y"))

    # Peek at the file to get nbands + ntran + ngkmax for band-range defaults.
    # This is a small per-rank read via h5py (just metadata datasets, not
    # coeffs).  We keep it OUT of timed stages.
    import h5py
    with h5py.File(args.wfn, "r") as f:
        nbands = int(f["mf_header/kpoints/mnband"][()])
        ntran = int(f["mf_header/symmetry/ntran"][()])
        ngk = np.asarray(f["mf_header/kpoints/ngk"][:], dtype=np.int64)
        ngkmax = int(ngk.max())
    if ntran != 1:
        _log(f"ERROR: v1 of this test only supports nosym files (ntran=1); "
             f"got ntran={ntran}.  Port symmetry unfolding for v2.")
        return 1

    if args.band_range is None:
        nb_trim = (nbands // world) * world
        band_range = (0, nb_trim)
    else:
        band_range = tuple(args.band_range)
    nb = band_range[1] - band_range[0]
    _log(f"band_range = {band_range}  (nb = {nb})")
    _log(f"ngkmax     = {ngkmax}")

    # Safe-corner sanity check (reads fft_grid, ecutrho, ecutwfc).
    wfn_peek = WFNReader(args.wfn)
    _assert_safe_corner_ok(wfn_peek)
    wfn_peek_nk = int(wfn_peek.nkpts)
    _log(f"safe corner  = {tuple(int(x) for x in _fft_safe_corner(tuple(int(x) for x in wfn_peek.fft_grid)))}")
    _log(f"nk           = {wfn_peek_nk}")
    del wfn_peek  # don't keep the slurp around

    # ============================================================
    # Warmup — trigger all compile caches before timed iters so the
    # timings reflect steady-state dispatch, not XLA compile time.
    # ============================================================
    _log("\n--- warmup ---")
    _log(f"rss_pre_baseline_warmup = {_rss_gb():.3f} GB")
    psi_B_warm, tB_warm = run_baseline_path(args.wfn, band_range, mesh)
    _log(f"  baseline warmup: open={tB_warm['open']*1e3:7.1f} ms  "
         f"read+scatter={tB_warm['read_scatter']*1e3:7.1f} ms  "
         f"fft={tB_warm['fft']*1e3:7.1f} ms")
    _log(f"rss_post_baseline_warmup = {_rss_gb():.3f} GB")

    psi_F_warm, tF_warm = run_ffi_path(args.wfn, WFNReader(args.wfn),
                                       band_range, mesh, ngkmax)
    _log(f"  FFI      warmup: open={tF_warm['open']*1e3:7.1f} ms  "
         f"read+scatter={tF_warm['read_scatter']*1e3:7.1f} ms  "
         f"fft={tF_warm['fft']*1e3:7.1f} ms  "
         f"trace_count={tF_warm.get('_trace_count', '?')}  (nk={wfn_peek_nk})")
    _log(f"rss_post_ffi_warmup = {_rss_gb():.3f} GB")

    # ---- correctness ----
    diff = compare_sharded(psi_B_warm, psi_F_warm)
    _log(f"\nmax |psi_B - psi_F| = {diff:.3e}   "
         f"{'PASS (bit-identical)' if diff == 0.0 else ('PASS (tol)' if diff < 1e-12 else 'FAIL')}")
    del psi_B_warm, psi_F_warm

    # ============================================================
    # Timed iters.
    # ============================================================
    _log(f"\n--- timed iters ({args.iters}) ---")
    B_times = {"open": [], "read_scatter": [], "fft": []}
    F_times = {"open": [], "read_scatter": [], "fft": []}

    for it in range(args.iters):
        _log(f"\n[iter {it}]")
        _, tB = run_baseline_path(args.wfn, band_range, mesh)
        for k in B_times:
            B_times[k].append(tB[k])
        _log(f"  (B) open={tB['open']*1e3:7.1f}  "
             f"r+s={tB['read_scatter']*1e3:7.1f}  "
             f"fft={tB['fft']*1e3:7.1f}   (ms)")

        _, tF = run_ffi_path(args.wfn, WFNReader(args.wfn), band_range, mesh, ngkmax)
        for k in F_times:
            F_times[k].append(tF[k])
        split = ""
        if "_read_only" in tF and "_gather_only" in tF:
            split = (f"  [read={tF['_read_only']*1e3:.1f} "
                     f"gather={tF['_gather_only']*1e3:.1f}]")
        _log(f"  (F) open={tF['open']*1e3:7.1f}  "
             f"r+s={tF['read_scatter']*1e3:7.1f}  "
             f"fft={tF['fft']*1e3:7.1f}   (ms){split}")

    def _mean_ms(ts): return 1e3 * float(np.mean(ts))

    _log("\n=== summary (mean over iters) ===")
    _log(f"{'stage':>14} {'(B) base':>12} {'(F) ffi':>12} {'speedup':>10}")
    total_B = total_F = 0.0
    for k in ("open", "read_scatter", "fft"):
        mB = _mean_ms(B_times[k])
        mF = _mean_ms(F_times[k])
        total_B += mB
        total_F += mF
        sp = f"{mB/mF:6.2f}x" if mF > 0 else "  inf "
        _log(f"{k:>14} {mB:>9.1f} ms {mF:>9.1f} ms {sp:>10}")
    _log(f"{'TOTAL':>14} {total_B:>9.1f} ms {total_F:>9.1f} ms "
         f"{total_B/total_F:>6.2f}x")

    _log(f"\nrss_final = {_rss_gb():.3f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

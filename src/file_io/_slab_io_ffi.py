"""FFI SlabIO backend — collective MPI-IO via ``ffi.phdf5``.

Opt-in path (``use_ffi_io=True``).  Imported lazily by
:mod:`file_io.slab_io` so the default allgather path works without
``liblorrax_ffi.so`` being built.

Every operation derives per-rank hyperslab offsets from the sharding
spec of the JAX array being written (or a caller-provided one for
reads) plus a global-origin ``offset`` argument.  The C++ handler
un-ravels the rank id through ``mesh_shape`` and advances along every
sharded dim.  See ``ffi/phdf5/cpp/write_ffi.cc`` for the C++ side.
"""
from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def _lustre_prestripe(path: str, stripe_count: int = 16,
                      stripe_size: str = "4M") -> None:
    """Pre-create ``path`` with an explicit Lustre stripe layout.

    Must be called on rank 0 only, BEFORE ``open_file``.  On Lustre,
    the MPI-IO hints ``striping_factor`` / ``striping_unit`` passed via
    ``H5Pset_fapl_mpio`` are frequently ignored by Cray MPICH when the
    containing directory has a fixed stripe (default 1×1 MiB on
    Perlmutter's ``pscratch``) — the file inherits the directory
    layout and the hint is silently dropped.  Pre-striping the file
    with ``lfs setstripe`` forces the desired layout; HDF5's
    ``H5Fcreate`` with ``H5F_ACC_TRUNC`` then truncates it in place and
    the stripe metadata survives.

    Measured on Si 10³ zeta_q.h5 write: with default 1×1 MiB stripe,
    per-write effective bandwidth was ~32 MB/s/rank (64 GB total in
    515 s).  With 16×4 MiB stripe we expect ~500 MB/s/rank on
    Perlmutter's HDD OSTs.

    Best-effort: if ``lfs`` isn't on PATH or striping fails (e.g.,
    non-Lustre filesystem), this is a no-op.
    """
    if shutil.which("lfs") is None:
        return
    try:
        # Remove any existing file so lfs can set the stripe.  Safe for
        # mode='w' callers since that mode is about to truncate anyway.
        if os.path.exists(path):
            os.remove(path)
        subprocess.run(
            ["lfs", "setstripe", "-c", str(stripe_count),
             "-S", stripe_size, path],
            check=False, capture_output=True, timeout=10,
        )
    except Exception:
        # Best-effort; fall through to plain H5Fcreate.  A debug
        # message would be nice here but we don't want to pollute
        # stdout on non-Lustre targets.
        pass

# Lazy imports happen inside the class methods; module-level imports
# of ffi.phdf5 would break users who don't build the FFI .so.


def _sharding_to_axis_info(
    sharding: NamedSharding, ndim: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Encode a NamedSharding's per-dim axis lists for the FFI attrs.

    Returns ``(axis_count_per_dim, axis_flat)``:
      - axis_count_per_dim[d]: number of mesh axes sharding dim d
        (0 = replicated).
      - axis_flat: concatenation of per-dim axis index lists, in dim
        order, each list preserving JAX's leftmost-is-slowest order.

    JAX canonicalises ``PartitionSpec(None, None)`` to
    ``PartitionSpec()``, so iterate by the array's ndim and treat
    missing trailing entries as ``None``.
    """
    axis_names = list(sharding.mesh.axis_names)
    spec = list(sharding.spec)
    counts: list[int] = []
    flat: list[int] = []
    for i in range(ndim):
        s = spec[i] if i < len(spec) else None
        if s is None:
            counts.append(0)
        elif isinstance(s, str):
            if s not in axis_names:
                raise ValueError(
                    f"sharding spec dim {i}: axis '{s}' not in mesh "
                    f"axis_names {axis_names}")
            counts.append(1)
            flat.append(axis_names.index(s))
        elif isinstance(s, (list, tuple)):
            counts.append(len(s))
            for a in s:
                if a not in axis_names:
                    raise ValueError(
                        f"sharding spec dim {i}: axis '{a}' not in mesh "
                        f"axis_names {axis_names}")
                flat.append(axis_names.index(a))
        else:
            raise ValueError(f"unrecognised spec element at dim {i}: {s!r}")
    return tuple(counts), tuple(flat)


def _replicated_sharding(mesh: Mesh, ndim: int) -> NamedSharding:
    """All-None PartitionSpec on `mesh` for an ndim-D array."""
    return NamedSharding(mesh, P(*([None] * ndim)))


def _replicated_i64_vector(values: Sequence[int], mesh: Mesh) -> jax.Array:
    """Small int64 control buffer, explicitly replicated on ``mesh``.

    Do not rely on JAX's default placement for these vectors: the PHDF5
    write path passes offsets through a cached jitted shard_map, and an
    implicitly placed offset buffer once arrived in C++ with dimensions
    permuted in the real CrI3 driver.  Replicating the control buffer is
    both the intended semantics and the safest JIT cache key.
    """
    return jax.device_put(
        np.asarray(tuple(int(v) for v in values), dtype=np.int64),
        NamedSharding(mesh, P()),
    )


def _normalize_slab_request(
    *,
    op: str,
    name: str,
    offset: Sequence[int] | None,
    slab_shape: Sequence[int],
    global_shape: Sequence[int] | None,
    check_bounds: bool = True,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Return ``(offset, slab_shape, global_shape)`` after basic checks."""
    shape = tuple(int(s) for s in slab_shape)
    if not shape:
        raise ValueError(f"{op} {name!r}: slab shape must be non-empty")
    if any(s < 0 for s in shape):
        raise ValueError(f"{op} {name!r}: negative slab shape {shape}")

    off = tuple(int(o) for o in (offset if offset is not None
                                else (0,) * len(shape)))
    gshape = tuple(int(s) for s in (global_shape if global_shape is not None
                                   else shape))

    if len(off) != len(shape) or len(gshape) != len(shape):
        raise ValueError(
            f"{op} {name!r}: rank mismatch offset={off}, "
            f"slab_shape={shape}, global_shape={gshape}")
    if any(o < 0 for o in off):
        raise ValueError(f"{op} {name!r}: negative offset {off}")
    if any(g < 0 for g in gshape):
        raise ValueError(f"{op} {name!r}: negative global shape {gshape}")

    if check_bounds:
        over = [
            (i, off[i], shape[i], gshape[i])
            for i in range(len(shape))
            if off[i] + shape[i] > gshape[i]
        ]
        if over:
            details = ", ".join(
                f"dim {i}: {o}+{s}>{g}" for i, o, s, g in over)
            raise ValueError(
                f"{op} {name!r}: slab exceeds global shape ({details})")
    return off, shape, gshape


def _normalize_valid_shape(
    *,
    op: str,
    name: str,
    valid_shape: Sequence[int] | None,
    slab_shape: Sequence[int],
    offset: Sequence[int],
    global_shape: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Return the logical on-file extent inside a possibly padded slab.

    ``slab_shape`` is the physical JAX array shape.  ``valid_shape`` is
    the prefix of that array that should map to file data; omitted means
    the whole slab is valid.  The valid shape itself need not be
    divisible by the mesh because the C++ FFI clips the last rank(s).
    """
    shape = tuple(int(s) for s in slab_shape)
    vshape = tuple(int(s) for s in (valid_shape if valid_shape is not None
                                    else shape))
    off = tuple(int(o) for o in offset)
    if len(vshape) != len(shape):
        raise ValueError(
            f"{op} {name!r}: valid_shape rank mismatch "
            f"valid_shape={vshape}, slab_shape={shape}")
    if any(s < 0 for s in vshape):
        raise ValueError(f"{op} {name!r}: negative valid_shape {vshape}")
    too_large = [
        (i, vshape[i], shape[i])
        for i in range(len(shape))
        if vshape[i] > shape[i]
    ]
    if too_large:
        details = ", ".join(f"dim {i}: {v}>{s}"
                            for i, v, s in too_large)
        raise ValueError(
            f"{op} {name!r}: valid_shape exceeds slab shape ({details})")
    if global_shape is not None:
        gshape = tuple(int(s) for s in global_shape)
        over = [
            (i, off[i], vshape[i], gshape[i])
            for i in range(len(shape))
            if off[i] + vshape[i] > gshape[i]
        ]
        if over:
            details = ", ".join(
                f"dim {i}: {o}+{s}>{g}" for i, o, s, g in over)
            raise ValueError(
                f"{op} {name!r}: valid slab exceeds global shape ({details})")
    return vshape


def _validate_block_divisible(
    *,
    op: str,
    name: str,
    shape: Sequence[int],
    axis_count_per_dim: Sequence[int],
    axis_flat: Sequence[int],
    mesh_shape: Sequence[int],
) -> None:
    """Reject sharded dimensions that cannot form equal block shards."""
    flat_idx = 0
    for d, size in enumerate(tuple(int(s) for s in shape)):
        na = int(axis_count_per_dim[d])
        div = 1
        axes = []
        for k in range(na):
            ax = int(axis_flat[flat_idx + k])
            axes.append(ax)
            div *= int(mesh_shape[ax])
        flat_idx += na
        if div > 1 and size % div:
            raise ValueError(
                f"{op} {name!r}: dimension {d} size {size} is not "
                f"divisible by mesh axes {axes} product {div}")


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------
class _FfiBackend:
    """Collective MPI-IO SlabIO backend."""

    def __init__(self, path: str, mesh: Mesh, mode: str = "w") -> None:
        # Lazy import — keeps file_io importable without the FFI built.
        from ffi.phdf5 import open_file as _open_file, close_file as _close_file
        from ffi.common import ffi_loader as _loader

        self._open_file = _open_file
        self._close_file = _close_file
        self._loader = _loader

        self.path = path
        self.mesh = mesh
        self.mode = mode
        # Pre-stripe the file on rank 0 so Lustre's per-stripe layout
        # actually matches the MPI_Info hints the FFI passes to ROMIO.
        # Only for 'w' mode — existing files in 'a'/'r' already have
        # their layout and we'd lose it by unlinking.  Barrier after
        # so all ranks see the new inode before H5Fcreate.
        if mode == "w" and jax.process_index() == 0:
            stripe_count = int(os.environ.get("LORRAX_PHDF5_STRIPE_COUNT", "16"))
            stripe_size = os.environ.get("LORRAX_PHDF5_STRIPE_SIZE_FS", "4M")
            _lustre_prestripe(path, stripe_count=stripe_count,
                              stripe_size=stripe_size)
        if mode == "w":
            try:
                multihost_utils.sync_global_devices("slab_io_ffi_prestripe")
            except Exception:
                pass
        self.fh: int = self._open_file(path, mesh=mesh, mode=mode)
        self._ds_ids: dict[str, int] = {}
        # write_attr needs plain h5py (the FFI doesn't expose a
        # collective attr-write path), so we defer attr writes to
        # close() — concurrent h5py + MPI-IO on the same file would
        # corrupt HDF5 metadata.
        self._deferred_attrs: list[tuple[str, object]] = []
        # Python-level async writer.  ``write_slab`` enqueues a callable
        # here; ``_dispatch_worker`` pops it and calls
        # ``jax.jit(shard_map(_per_rank))(A).block_until_ready()``.
        # Rationale: XLA's ``ffi::Future`` async mechanism registers the
        # Future with XLA's scheduler but still blocks the caller
        # (Python main thread) of ``jit(...)(A)`` until the Future
        # resolves — i.e. until ``H5Dwrite`` completes.  By doing the
        # jit on a dedicated Python worker thread, we leave the main
        # Python thread free to build the next chunk while the current
        # one is still writing.  One worker per backend (FIFO) ensures
        # every rank dispatches in the same order, which is the MPI-IO
        # collective rendezvous requirement.  See
        # ``reports/session_2026-04-18_async_probe/report.md``.
        # Cache of compiled shard_map closures keyed on the full FFI
        # attr + shape/dtype signature, so repeat writes at identical
        # signatures reuse the jit cache instead of recompiling.
        self._sm_cache: dict = {}
        # Bound the write-dispatch queue to prevent GPU memory growth
        # across chunks.  Each queued ``_task`` closure captures its
        # input ``A`` (the jax.Array being written) by Python reference
        # — XLA's allocator counts A as live-in-use until the closure
        # runs and returns.  With H5Dwrite at ~11 s per chunk and
        # chunk-compute at ~1-2 s, an unbounded queue grows ~1 task per
        # chunk at steady state: each chunk's A accumulates on GPU and
        # ``bytes_in_use`` rises by ~1 zeta_chunk/rank/chunk until OOM.
        #
        # Total in-flight A-holding at queue-cap K = (K queued +
        # 1 being processed + 1 in main-thread transpose view).
        # Throughput cost vs unbounded is small above K=2; writer is
        # already the bottleneck on typical H5Dwrite rates.
        #
        # Measured at Si 4x4x4 60Ry / 2400c / mem16:
        #   K=0 unbounded: 12.91 → 22.48+ GB / 28 s zeta_fit (OOM-bound)
        #   K=2:           12.91 → 16.47 GB (flat) / 97 s zeta_fit
        #   K=4:           12.91 → 18.50 GB (flat) / 92 s zeta_fit
        # K=2 gives identical throughput to K=4 on this system (writer
        # saturates) while saving 2 × zeta_chunk/rank — picked as the
        # default.  Override via ``LORRAX_WRITE_QUEUE_MAXSIZE``; 0 =
        # legacy unbounded (not recommended).
        _qmax = int(os.environ.get('LORRAX_WRITE_QUEUE_MAXSIZE', '2'))
        self._dispatch_queue: queue.Queue = queue.Queue(maxsize=_qmax)
        self._dispatch_pending: int = 0          # protected by _pending_mu
        self._pending_mu = threading.Lock()
        self._pending_cv = threading.Condition(self._pending_mu)
        self._dispatch_error: BaseException | None = None
        self._dispatch_worker = threading.Thread(
            target=self._dispatch_loop,
            name=f"phdf5-dispatch-{path}",
            daemon=True,
        )
        self._dispatch_worker.start()

    # ------------------------------------------------------------------
    def create_dataset(
        self,
        name: str,
        *,
        shape: Sequence[int],
        dtype,
        chunks: Sequence[int] | None = None,
        attrs: dict | None = None,
    ) -> None:
        ds_id = self._loader.phdf5_ensure_dataset(
            self.fh, name, tuple(int(s) for s in shape),
            str(jnp.dtype(dtype).name),
        )
        self._ds_ids[name] = ds_id
        # chunks + attrs are runtime-set on the underlying H5 dataset.
        # The FFI backend doesn't yet expose a collective "set chunks"
        # after H5Dcreate (it would need a new ctypes entry and some
        # care around MPI-IO dataset transfer property lists).  The
        # caller's `chunks=` argument is a hint for the writer; when the
        # dataset is created by the FFI path the H5 library picks
        # contiguous layout + the FAPL-level alignment set in ctx
        # init.  For v1 this matches the OpenMPI-stack perf ceiling;
        # user can pre-create with h5py + chunks if needed.
        if chunks is not None or attrs is not None:
            import warnings
            warnings.warn(
                "FFI backend: chunks/attrs on create_dataset currently no-op; "
                "pre-create with h5py if you need explicit chunking or attrs.")

    # ------------------------------------------------------------------
    def write_attr(self, name: str, value) -> None:
        # Deferred to close() to avoid interleaving rank-0 h5py with
        # active MPI-IO on the same file.  Small arrays only; this is
        # not meant for large data.
        self._deferred_attrs.append((name, value))

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _dispatch_loop(self) -> None:
        """Drain ``_dispatch_queue`` in FIFO order.

        Each queue entry is a callable ``() -> None`` that performs the
        jit dispatch + block_until_ready.  We catch and stash any
        exception so the main thread can re-raise it on the next
        enqueue or at close() time, rather than silently losing it.
        """
        while True:
            task = self._dispatch_queue.get()
            if task is None:
                return
            try:
                task()
            except BaseException as exc:  # noqa: BLE001
                with self._pending_cv:
                    if self._dispatch_error is None:
                        self._dispatch_error = exc
            finally:
                with self._pending_cv:
                    self._dispatch_pending -= 1
                    self._pending_cv.notify_all()

    def _drain_pending(self) -> None:
        """Block main thread until all queued tasks finish."""
        with self._pending_cv:
            while self._dispatch_pending > 0:
                self._pending_cv.wait()
        if self._dispatch_error is not None:
            err = self._dispatch_error
            self._dispatch_error = None
            raise err

    # ------------------------------------------------------------------
    def _ds_id(self, name: str, readonly: bool = False) -> int:
        if name in self._ds_ids:
            return self._ds_ids[name]
        if readonly:
            ds_id = self._loader.phdf5_open_dataset_ro(self.fh, name)
        else:
            raise RuntimeError(
                f"dataset '{name}' not registered — call create_dataset first")
        self._ds_ids[name] = ds_id
        return ds_id

    # ------------------------------------------------------------------
    def write_slab(
        self,
        name: str,
        A,
        *,
        offset: Sequence[int] | None = None,
        global_shape: Sequence[int] | None = None,
        valid_shape: Sequence[int] | None = None,
        dtype=None,
        chunks: Sequence[int] | None = None,
        k_chunk_size: int | None = None,
    ) -> None:
        from ffi.phdf5.write import ffi_write_call

        if not isinstance(A, jax.Array):
            A = jnp.asarray(A)
        # Ensure placement: if not sharded on our mesh, put as replicated.
        if not isinstance(A.sharding, NamedSharding) or A.sharding.mesh is not self.mesh:
            A = jax.device_put(A, _replicated_sharding(self.mesh, A.ndim))

        axis_count_per_dim, axis_flat = _sharding_to_axis_info(
            A.sharding, A.ndim)
        off, slab_shape, gshape = _normalize_slab_request(
            op="write_slab", name=name, offset=offset,
            slab_shape=A.shape, global_shape=global_shape,
            check_bounds=False)
        vshape = _normalize_valid_shape(
            op="write_slab", name=name, valid_shape=valid_shape,
            slab_shape=slab_shape, offset=off, global_shape=gshape)
        mesh_shape = tuple(self.mesh.shape[ax] for ax in self.mesh.axis_names)
        _validate_block_divisible(
            op="write_slab", name=name, shape=slab_shape,
            axis_count_per_dim=axis_count_per_dim,
            axis_flat=axis_flat, mesh_shape=mesh_shape)

        if name not in self._ds_ids:
            ds_id = self._loader.phdf5_ensure_dataset(
                self.fh, name, tuple(int(s) for s in gshape),
                str(jnp.dtype(A.dtype).name),
            )
            self._ds_ids[name] = ds_id

        if os.environ.get("LORRAX_FFI_DEBUG_SHARDS"):
            import sys
            local_shapes = [tuple(s.data.shape) for s in A.addressable_shards]
            sys.__stdout__.write(
                f"[ffi-debug proc={jax.process_index()}] "
                f"name={name} shape={tuple(A.shape)} dtype={A.dtype} "
                f"spec={getattr(A.sharding, 'spec', None)} "
                f"offset={off} valid_shape={vshape} gshape={gshape} "
                f"local_shapes={local_shapes}\n")
            sys.__stdout__.flush()


        ds_id = self._ds_ids[name]
        ctx_handle = self.fh
        in_specs = A.sharding.spec  # PartitionSpec

        # ── shard_map cache ──
        # Keyed on everything that's a compile-time FFI attr + array
        # shape/dtype/sharding, so repeat writes at identical signatures
        # reuse the compiled module.  ``offset_base`` is now a RUNTIME
        # Buffer (not an Attr), so it's intentionally NOT in the key —
        # chunks with different offsets hit the same cached compile.
        cache_key = (
            int(ctx_handle), int(ds_id), mesh_shape,
            axis_count_per_dim, axis_flat,
            A.shape, str(A.dtype), in_specs,
        )
        sm = self._sm_cache.get(cache_key)
        if sm is None:
            def _per_rank(A_local, offset_local, valid_shape_local,
                          _ds_id=int(ds_id),
                          _mesh_shape=mesh_shape,
                          _axis_count=axis_count_per_dim,
                          _axis_flat=axis_flat,
                          _ctx_handle=int(ctx_handle)):
                return ffi_write_call(
                    A_local, offset_local, valid_shape_local,
                    ctx_handle=_ctx_handle,
                    ds_id=_ds_id,
                    mesh_shape=_mesh_shape,
                    axis_count_per_dim=_axis_count,
                    axis_flat=_axis_flat,
                )
            sm_bare = shard_map(
                _per_rank, mesh=self.mesh,
                in_specs=(in_specs, P(), P()), out_specs=P(),
                check_rep=False,
            )
            # DIAGNOSTIC: LORRAX_WRITE_NO_JIT=1 bypasses the jax.jit
            # wrapper — tests whether jit's argument-retention cache is
            # what's leaking ~1 zeta_chunk/rank per write.
            if os.environ.get('LORRAX_WRITE_NO_JIT'):
                sm = sm_bare
            else:
                # Wrap in jax.jit so the trace+compile is cached at the
                # JAX jit level.  Without this, shard_map re-traces on each
                # call even though we reuse the same ``sm`` object — visible
                # in the HLO dump as multiple identical-signature modules.
                sm = jax.jit(sm_bare)
            self._sm_cache[cache_key] = sm

        # Enqueue dispatch onto the Python worker thread.  Main thread
        # returns in ~0.2ms; the worker thread calls ``sm(A, offset)``
        # in FIFO order.  The offset Buffer is tiny (ndim × 8 bytes).
        offset_arr = _replicated_i64_vector(off, self.mesh)
        valid_shape_arr = _replicated_i64_vector(vshape, self.mesh)

        def _task():
            tok = sm(A, offset_arr, valid_shape_arr)
            tok.block_until_ready()

        with self._pending_cv:
            self._dispatch_pending += 1
        self._dispatch_queue.put(_task)

    # ------------------------------------------------------------------
    def read_slab(
        self,
        name: str,
        *,
        shape: Sequence[int],
        dtype,
        offset: Sequence[int] | None = None,
        valid_shape: Sequence[int] | None = None,
        mesh: Mesh | None = None,
        partition_spec: P | None = None,
        as_numpy: bool = False,  # accepted for signature compatibility;
        # the public SlabIO.read_slab handles the numpy conversion.
    ) -> jax.Array:
        from ffi.phdf5.read import ffi_read_call

        mesh = mesh or self.mesh
        if shape is None:
            raise ValueError(
                f"read_slab {name!r}: shape is required for FFI reads")
        off, read_shape, _ = _normalize_slab_request(
            op="read_slab", name=name, offset=offset,
            slab_shape=shape, global_shape=None, check_bounds=False)
        vshape = _normalize_valid_shape(
            op="read_slab", name=name, valid_shape=valid_shape,
            slab_shape=read_shape, offset=off, global_shape=None)

        # Default: fully replicated.  Caller can provide partition_spec
        # to shard the read.
        if partition_spec is None:
            partition_spec = P(*([None] * len(read_shape)))
        sharding = NamedSharding(mesh, partition_spec)
        axis_count_per_dim, axis_flat = _sharding_to_axis_info(
            sharding, len(read_shape))
        mesh_shape = tuple(mesh.shape[ax] for ax in mesh.axis_names)
        _validate_block_divisible(
            op="read_slab", name=name, shape=read_shape,
            axis_count_per_dim=axis_count_per_dim,
            axis_flat=axis_flat, mesh_shape=mesh_shape)

        # Per-rank output shape: divide by the product of the mesh
        # sizes of all axes sharding that dim.
        local_shape = list(read_shape)
        _flat_idx = 0
        for d in range(len(read_shape)):
            na = axis_count_per_dim[d]
            if na > 0:
                div = 1
                for k in range(na):
                    div *= int(mesh_shape[axis_flat[_flat_idx + k]])
                local_shape[d] = int(local_shape[d]) // div
                _flat_idx += na
        out_struct = jax.ShapeDtypeStruct(tuple(local_shape), jnp.dtype(dtype))

        ds_id = self._ds_id(name, readonly=True)
        ctx_handle = self.fh

        # Cache key: same as writes, plus the local_shape + partition_spec
        # (since reads are parameterised by output shape + sharding spec).
        cache_key = (
            "read", int(ctx_handle), int(ds_id), mesh_shape,
            axis_count_per_dim, axis_flat,
            tuple(local_shape), str(jnp.dtype(dtype)),
            partition_spec,
        )
        sm = self._sm_cache.get(cache_key)
        if sm is None:
            def _per_rank(offset_local, valid_shape_local,
                          _ds_id=int(ds_id),
                          _mesh_shape=mesh_shape,
                          _axis_count=axis_count_per_dim,
                          _axis_flat=axis_flat,
                          _ctx_handle=int(ctx_handle),
                          _out_struct=out_struct):
                return ffi_read_call(
                    _out_struct,
                    offset_local,
                    valid_shape_local,
                    ctx_handle=_ctx_handle,
                    ds_id=_ds_id,
                    mesh_shape=_mesh_shape,
                    axis_count_per_dim=_axis_count,
                    axis_flat=_axis_flat,
                )
            sm_bare = shard_map(
                _per_rank, mesh=mesh,
                in_specs=(P(), P()), out_specs=partition_spec,
                check_rep=False,
            )
            sm = jax.jit(sm_bare)
            self._sm_cache[cache_key] = sm

        offset_arr = _replicated_i64_vector(off, mesh)
        valid_shape_arr = _replicated_i64_vector(vshape, mesh)
        result = sm(offset_arr, valid_shape_arr)
        result.block_until_ready()
        return result

    # ------------------------------------------------------------------
    def accumulate_slab(
        self,
        name: str,
        A,
        *,
        offset: Sequence[int] | None = None,
    ) -> None:
        """dset[off:off+A.shape] += A — collective read-modify-write."""
        if not isinstance(A, jax.Array):
            A = jnp.asarray(A)
        if not isinstance(A.sharding, NamedSharding) or A.sharding.mesh is not self.mesh:
            A = jax.device_put(A, _replicated_sharding(self.mesh, A.ndim))

        off = tuple(offset) if offset is not None else tuple([0] * A.ndim)
        existing = self.read_slab(
            name, shape=tuple(A.shape), dtype=A.dtype,
            offset=off, mesh=self.mesh, partition_spec=A.sharding.spec,
        )
        A_new = existing + A
        self.write_slab(name, A_new, offset=off, global_shape=None)

    # ------------------------------------------------------------------
    def close(self) -> None:
        # Drain pending writes on the Python worker thread, then stop
        # the worker, THEN close the MPI-IO handle.  Order matters:
        # close_ctx() in C++ also drains its own task queue, but an
        # in-flight Python-side jit dispatch could still be holding a
        # reference to ctx_handle when we call close_file below.
        #
        # The drain can take minutes for multi-GB writes (N collective
        # MPI-IO calls serialised through one writer thread per ctx).
        # Print per-stage timings on rank 0 so a long drain doesn't
        # look like a hang.
        import time as _time
        _rank0 = (jax.process_index() == 0)
        _verbose = _rank0 and bool(
            os.environ.get("LORRAX_PHDF5_CLOSE_VERBOSE", "1") != "0")
        with self._pending_mu:
            _pending = self._dispatch_pending
        if _verbose:
            print(f"  [SlabIO.close] draining {_pending} pending writes "
                  f"for {os.path.basename(self.path)} …", flush=True)
        _t0 = _time.perf_counter()
        self._drain_pending()
        _t_drain = _time.perf_counter() - _t0
        if _verbose:
            print(f"  [SlabIO.close] Python dispatch drained in "
                  f"{_t_drain:.1f} s; joining writer thread", flush=True)
        _t0 = _time.perf_counter()
        self._dispatch_queue.put(None)          # shutdown sentinel
        self._dispatch_worker.join()
        _t_join = _time.perf_counter() - _t0
        if self.fh:
            if _verbose:
                print(f"  [SlabIO.close] writer thread joined in "
                      f"{_t_join:.1f} s; calling H5Fclose collectively",
                      flush=True)
            _t0 = _time.perf_counter()
            self._close_file(self.fh)
            self.fh = 0
            _t_close = _time.perf_counter() - _t0
            if _verbose:
                print(f"  [SlabIO.close] H5Fclose returned in "
                      f"{_t_close:.1f} s", flush=True)
        # Now that MPI-IO has released the file, rank 0 can safely
        # reopen with h5py to tack on any deferred small-metadata
        # datasets (omega_ev and friends).
        if self._deferred_attrs:
            import h5py
            import numpy as np
            if jax.process_index() == 0:
                with h5py.File(self.path, "a") as h5:
                    for name, value in self._deferred_attrs:
                        if name in h5:
                            del h5[name]
                        host = value
                        if not isinstance(host, np.ndarray):
                            host = np.asarray(jax.device_get(host))
                        h5.create_dataset(name, data=host)
            try:
                multihost_utils.sync_global_devices("slab_io_ffi_close_attrs")
            except Exception:
                pass
            self._deferred_attrs = []

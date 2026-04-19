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

import queue
import threading
from typing import Sequence

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

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
        self._dispatch_queue: queue.Queue = queue.Queue()
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
        off = tuple(offset) if offset is not None else tuple([0] * A.ndim)
        gshape = tuple(global_shape) if global_shape is not None else tuple(A.shape)

        if name not in self._ds_ids:
            ds_id = self._loader.phdf5_ensure_dataset(
                self.fh, name, tuple(int(s) for s in gshape),
                str(jnp.dtype(A.dtype).name),
            )
            self._ds_ids[name] = ds_id

        ds_id = self._ds_ids[name]
        ctx_handle = self.fh
        mesh_shape = tuple(self.mesh.shape[ax] for ax in self.mesh.axis_names)
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
            def _per_rank(A_local, offset_local,
                          _ds_id=int(ds_id),
                          _mesh_shape=mesh_shape,
                          _axis_count=axis_count_per_dim,
                          _axis_flat=axis_flat,
                          _ctx_handle=int(ctx_handle)):
                return ffi_write_call(
                    A_local, offset_local,
                    ctx_handle=_ctx_handle,
                    ds_id=_ds_id,
                    mesh_shape=_mesh_shape,
                    axis_count_per_dim=_axis_count,
                    axis_flat=_axis_flat,
                )
            sm_bare = shard_map(
                _per_rank, mesh=self.mesh,
                in_specs=(in_specs, P()), out_specs=P(),
                check_rep=False,
            )
            # Wrap in jax.jit so the trace+compile is cached at the
            # JAX jit level.  Without this, shard_map re-traces on each
            # call even though we reuse the same ``sm`` object — visible
            # in the HLO dump as multiple identical-signature modules.
            sm = jax.jit(sm_bare)
            self._sm_cache[cache_key] = sm

        # Enqueue dispatch onto the Python worker thread.  Main thread
        # returns in ~0.2ms; the worker thread calls ``sm(A, offset)``
        # in FIFO order.  The offset Buffer is tiny (ndim × 8 bytes).
        offset_arr = jnp.asarray(off, dtype=jnp.int64)

        def _task():
            tok = sm(A, offset_arr)
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
        mesh: Mesh | None = None,
        partition_spec: P | None = None,
        as_numpy: bool = False,  # accepted for signature compatibility;
        # the public SlabIO.read_slab handles the numpy conversion.
    ) -> jax.Array:
        from ffi.phdf5.read import ffi_read_call

        mesh = mesh or self.mesh
        off = tuple(offset) if offset is not None else tuple([0] * len(shape))

        # Default: fully replicated.  Caller can provide partition_spec
        # to shard the read.
        if partition_spec is None:
            partition_spec = P(*([None] * len(shape)))
        sharding = NamedSharding(mesh, partition_spec)
        axis_count_per_dim, axis_flat = _sharding_to_axis_info(
            sharding, len(shape))
        mesh_shape = tuple(mesh.shape[ax] for ax in mesh.axis_names)

        # Per-rank output shape: divide by the product of the mesh
        # sizes of all axes sharding that dim.
        local_shape = list(shape)
        _flat_idx = 0
        for d in range(len(shape)):
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

        def _per_rank():
            return ffi_read_call(
                out_struct,
                ctx_handle=int(ctx_handle),
                ds_id=int(ds_id),
                offset_base=off,
                mesh_shape=mesh_shape,
                axis_count_per_dim=axis_count_per_dim,
                axis_flat=axis_flat,
            )

        result = shard_map(
            _per_rank, mesh=mesh,
            in_specs=(), out_specs=partition_spec,
            check_rep=False,
        )()
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
        self._drain_pending()
        self._dispatch_queue.put(None)          # shutdown sentinel
        self._dispatch_worker.join()
        if self.fh:
            self._close_file(self.fh)
            self.fh = 0
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

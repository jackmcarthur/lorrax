"""MPI-IO SlabIO backend — host-side parallel HDF5 via mpi4py + h5py.

Spiritual equivalent of :mod:`file_io._slab_io_ffi` for the JAX CPU
backend.  Each rank writes its own hyperslab via collective MPI-IO
through h5py(parallel) + mpi4py — no allgather, no rank-0 bottleneck.

Architecture deliberately parallels :class:`_FfiBackend` *except* for
the async writer thread.  We do writes synchronously on the main
thread because:

1. There's no D2H to overlap (the FFI's worker thread exists to
   overlap CUDA memcpy with H5Dwrite; on CPU XLA the "device" memory
   IS host memory, no copy needed).
2. Cray MPICH defaults to ``MPI_THREAD_SINGLE``; doing MPI-IO calls
   from a worker thread then ``H5Fclose`` on the main thread deadlocks
   because MPI's internal state is rank-thread-specific.  Forcing
   ``MPI_THREAD_MULTIPLE`` via ``mpi4py.rc.thread_level = "multiple"``
   would let us mirror the FFI's threaded design, but the perf win
   isn't worth the added complexity at this scale.

MPI-collectives coexistence (scorecard AS, 2026-07-27) — synchronous
writes are NOT sufficient once JAX's own CPU collectives ride MPI:

* With ``JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi`` the jaxlib wheel's
  MPItrampoline runtime drives the SAME ``libmpi`` as this module,
  from XLA's collectives thread — so this module's "main thread only"
  discipline no longer serializes anything.  That concurrency is only
  defined at ``MPI_THREAD_MULTIPLE``, and XLA requests FUNNELED:
  measured at P=16 x 8 nodes, ~29% of runs segfault/hang at the
  ζ-write/V_q boundary (two threads concurrently inside
  ``MPID_Progress_wait``; provider-independent — AS.4b).
* The certified stack (AS.4c): ``MPITRAMPOLINE_LIB`` must point at the
  THREAD_MULTIPLE-patched MPIwrapper build
  (``wk_AS/mpiw_thr_install/lib64/libmpiwrapper.so`` — it upgrades
  every init request to MULTIPLE, never downgrades), plus
  ``LORRAX_MPI_FINALIZE_FIX=skip_atexit`` via the overlay
  ``sitecustomize`` (jax registers an atexit ``collectives.Finalize``
  AND the C++ destructor finalizes → double-finalize rc=1 without it).
  These are harness-level machine facts, not LORRAX policy: LORRAX
  reads neither variable (see docs/dev/env_vars.md §6).
* Under the certified default (gloo collectives) none of this applies:
  mpi4py's import here is the first MPI init and requests MULTIPLE.
  The C++ FFI twin guards the same hazard at open time
  (``ffi/phdf5/cpp/context.cc`` MPI_Query_thread warning).

What we DO match from :class:`_FfiBackend`:

* Same Lustre prestripe on rank 0 before any rank opens the file.
* Same per-rank hyperslab arithmetic derived from JAX's
  ``Array.addressable_shards[i].index`` (a tuple of ``slice`` per dim
  giving the global slab of each local shard).
* Same ``valid_shape`` clipping for arrays padded for even sharding.
* Same deferred ``write_attr`` mechanism for tiny replicated metadata.

The only thing this backend doesn't do that :class:`_FfiBackend` does
is the cudaMemcpy D2H — on JAX's CPU backend the "device" memory IS
host memory, so the local shard is already in numpy-addressable RAM
and we hand the pointer straight to h5py.

Selected by :class:`gw.gw_config.SlabIOBackend.PHDF5_HOST`.  This is now
the SECOND CPU tier, not the first: since workstream AE the phdf5 write
core also compiles into the CUDA-free host FFI lib, and
``gw_config._route_cpu_slab_io`` prefers ``PHDF5_FFI`` whenever that lib
exports ``PhdfWriteHostFfi`` — because the FFI path needs no extra Python
environment, whereas this module needs ``mpi4py`` + an ``HDF5_MPI=ON``
h5py (the ``lorrax_env_mpi_overlay`` prefix, workstream AB).  Kept as the
fallback for a deployed host lib without the write handler, and as the
A/B control for the writer.
"""
from __future__ import annotations

import os
from typing import Sequence

import h5py
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ._slab_io_ffi import (
    _barrier,
    _lustre_prestripe,
    _normalize_slab_request,
    _normalize_valid_shape,
    _rank0,
)

_TRUE = ("1", "true", "yes", "on")


def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in _TRUE


def _stripe_size_bytes() -> int:
    """Lustre stripe size in bytes.

    ``LORRAX_PHDF5_STRIPE_SIZE_FS`` is the ``lfs setstripe -S`` spelling
    ("4M"); MPI-IO's ``striping_unit`` hint wants bytes, so accept both
    and normalise here.
    """
    raw = os.environ.get("LORRAX_PHDF5_STRIPE_SIZE_FS", "4M").strip()
    mult = 1
    if raw and raw[-1] in "kKmMgG":
        mult = {"k": 1 << 10, "m": 1 << 20, "g": 1 << 30}[raw[-1].lower()]
        raw = raw[:-1]
    try:
        return int(float(raw) * mult)
    except ValueError:
        return 4 << 20


def _mpi_io_hints(MPI):
    """The ``MPI_Info`` handed to ``H5Pset_fapl_mpio``.

    **This is the striping lever that actually works on Frontera.**
    :func:`_lustre_prestripe` shells out to ``lfs``, which does NOT
    EXIST inside the apptainer container every production run uses, so
    it has always been a silent no-op on this machine — MEASURED: both
    files job 7876423 wrote (``zeta_q.h5``, ``isdf_tensors_2406.h5``)
    came back ``lmm_stripe_count: 1``, i.e. 13.3 GB of ``V_qmunu``
    funnelled through a SINGLE OST.  ROMIO sets the layout through
    ``llapi`` at file-create time instead, so the hints below reach
    Lustre from inside the container with no binary on PATH.

    Same keys, same defaults as the C++ writer's
    ``ffi/phdf5/cpp/context.cc`` (which has always set them) — the h5py
    backend set NONE, which is the single biggest divergence between
    the two "equivalent" writers.
    """
    info = MPI.Info.Create()
    sc = int(os.environ.get("LORRAX_PHDF5_STRIPE_COUNT", "16"))
    if sc > 0:
        info.Set("striping_factor", str(sc))
        info.Set("striping_unit", str(_stripe_size_bytes()))
    # romio_cb_write / romio_ds_write / cb_nodes / cb_buffer_size are
    # left at ROMIO's automatic policy unless asked for.  MEASURED
    # (wk_AI, P=16, the production tile): forcing
    # ``romio_cb_write=enable`` on top of collective + stripe32 was
    # 1826 MB/s vs 2066 MB/s for ROMIO's own choice, i.e. the explicit
    # hint buys nothing here and may cost a little.  Knobs kept,
    # defaults not opinionated.  Since workstream AW the C++ writer
    # (``ffi/phdf5/cpp/context.cc``) follows the SAME policy — its old
    # Perlmutter-era forced defaults (cb_write=enable, ds_write=disable,
    # cb_buffer_size=64M, cb_nodes=world_size) are gone, so an unset env
    # means "ROMIO decides" in every writer and a set one reaches both.
    for key, env in (("romio_cb_write", "LORRAX_PHDF5_CB_WRITE"),
                     ("romio_ds_write", "LORRAX_PHDF5_DS_WRITE"),
                     ("cb_nodes", "LORRAX_PHDF5_CB_NODES"),
                     ("cb_buffer_size", "LORRAX_PHDF5_CB_BUFFER_SIZE")):
        val = os.environ.get(env, "")
        if val:
            info.Set(key, val)
    return info


def _local_shard_and_global_offset(
    A: jax.Array,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return ``(local_numpy, global_offset)`` for the process-local shard.

    LORRAX runs one JAX device per process under multi-process (mesh on
    ``mesh_xy``), so each process has exactly one addressable shard.

    The shard's ``.index`` is a tuple of ``slice`` objects giving the
    GLOBAL start/stop along each axis.  Slabs are always contiguous
    along each axis (no broadcast tiling) so ``.start`` is the offset
    within A.shape.  Replicated axes give ``slice(0, A.shape[ax])`` —
    every process holds the full axis and writes the same overlapping
    rows; under independent MPI-IO that's a redundant write but
    semantically correct (every rank writes identical bytes).
    """
    shards = A.addressable_shards
    if len(shards) != 1:
        # Multi-device-per-process (e.g. GPU with N visible devices
        # under a single process).  Not the LORRAX CPU mesh-xy regime
        # but worth a clear error rather than silent wrong data.
        raise RuntimeError(
            f"_slab_io_mpi_host expects 1 addressable shard per process; "
            f"got {len(shards)} for A.shape={tuple(A.shape)}.  Did you "
            f"set --xla_force_host_platform_device_count > 1 on a "
            f"multi-process run?")
    shard = shards[0]
    local = np.asarray(shard.data)
    # Replicated axes have ``slice(None, None)`` (no explicit bounds);
    # treat ``start=None`` as 0 (the full-axis slab starts at 0).
    offset = tuple(int(s.start) if s.start is not None else 0
                   for s in shard.index)
    return local, offset


def _clip_shard_to_valid(
    local: np.ndarray,
    shard_offset: tuple[int, ...],
    slab_offset: tuple[int, ...],
    valid_shape: tuple[int, ...],
) -> tuple[np.ndarray, tuple[int, ...]] | None:
    """Clip a local shard to the valid (non-padded) region of the slab.

    Returns ``(clipped_local, dataset_offset)`` or ``None`` if this
    shard is entirely outside the valid region (its data should not be
    written at all).

    Arithmetic:

      For axis ``d`` the shard covers global indices
      ``[shard_offset[d], shard_offset[d] + local.shape[d])`` inside
      the slab; the slab covers dataset indices
      ``[slab_offset[d], slab_offset[d] + valid_shape[d])``; the
      intersection is what we actually write.  Below the valid region
      means ``shard_end <= 0`` (impossible because shard_offset >= 0);
      above means ``shard_start >= valid_shape[d]``, in which case the
      shard is in the padded tail and we skip the write.

    All axes must have non-empty intersection or we return ``None``.
    """
    ndim = local.ndim
    ds_offsets: list[int] = []
    local_slices: list[slice] = []
    for d in range(ndim):
        shard_start_in_slab = shard_offset[d]
        shard_end_in_slab = shard_offset[d] + local.shape[d]
        valid_end_in_slab = valid_shape[d]
        # Intersection within the slab coordinate system:
        write_start_in_slab = shard_start_in_slab
        write_end_in_slab = min(shard_end_in_slab, valid_end_in_slab)
        if write_end_in_slab <= write_start_in_slab:
            return None  # shard fully in padded tail
        # Map slab-local to dataset-global:
        ds_offsets.append(int(slab_offset[d] + write_start_in_slab))
        # Slice into the local shard's first-axis-N elements:
        keep = write_end_in_slab - write_start_in_slab
        local_slices.append(slice(0, keep))
    return local[tuple(local_slices)], tuple(ds_offsets)


# ---------------------------------------------------------------------------
class _MpiHostBackend:
    """Collective MPI-IO SlabIO via parallel h5py + mpi4py.

    The interface mirrors :class:`_slab_io_ffi._FfiBackend` so they
    plug in interchangeably from :class:`SlabIO`.  See module docstring
    for the architectural rationale.
    """

    def __init__(self, path: str, mesh: Mesh, mode: str = "w") -> None:
        # Importing mpi4py inits MPI on first use.  We lazy-import here
        # so file_io.slab_io stays importable when MPI isn't built into
        # the venv (single-process tests, non-MPI smoke runs).
        from mpi4py import MPI
        self._MPI = MPI
        self._comm = MPI.COMM_WORLD

        self.path = path
        self.mesh = mesh
        self.mode = mode

        # Pre-stripe the file on rank 0 so Lustre's per-stripe layout
        # actually matches the MPI-IO hints we'll use below.  Only on
        # 'w'; 'a'/'r' inherit the existing inode's stripe layout.
        # Barrier after so all ranks see the inode before H5Fopen.
        if mode == "w" and _rank0():
            # UNLINK, don't just truncate.  A Lustre stripe layout is a
            # property of the INODE and is fixed at create time: both
            # ``lfs setstripe`` and MPI-IO's ``striping_factor`` hint are
            # no-ops against an existing file, and ``H5Fcreate(TRUNC)``
            # reuses the inode.  So a rerun in a run dir that already
            # holds a 1-stripe ``isdf_tensors_*.h5`` would silently keep
            # 1 stripe for ever.  ``mode='w'`` means "replace", so
            # replacing the inode is also the honest reading.
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass
            stripe_count = int(os.environ.get("LORRAX_PHDF5_STRIPE_COUNT", "16"))
            stripe_size = os.environ.get("LORRAX_PHDF5_STRIPE_SIZE_FS", "4M")
            _lustre_prestripe(path, stripe_count=stripe_count,
                              stripe_size=stripe_size)
        if mode == "w":
            _barrier("slab_io_mpi_host_prestripe")

        h5_mode = {"w": "w", "a": "a", "r": "r"}[mode]
        self._info = _mpi_io_hints(MPI)
        self._fh = h5py.File(path, h5_mode, driver="mpio",
                             comm=self._comm, info=self._info)

        # Collective (two-phase) MPI-IO for the bulk writes.  DEFAULT ON,
        # and this is the fix for scorecard AF.4c's 1.7 MB/s.  Rationale,
        # measured (wk_AI microbench, the production tile geometry):
        #
        #   ``V_qmunu`` is (nq, mu, mu) sharded P(None,'x','y'), i.e. a
        #   2-D TILE of a CONTIGUOUS dataset.  A rank's tile is not one
        #   contiguous file region — its innermost run is only
        #   ``(mu/Py)*16`` = 3.2 kB, and there are ``nq*(mu/Px)`` = 28 800
        #   of them per rank.  Under INDEPENDENT MPI-IO HDF5 issues one
        #   ``MPI_File_write_at`` per run, so 144 ranks hammer Lustre with
        #   4.1 M tiny strided writes.
        #
        #   ``zeta_q_G`` is (nq, mu, ngkmax) sharded on mu only AND
        #   created with ``chunks=(1, mu, ngkmax)``, so a rank's tile is
        #   one contiguous 2.3 MB region per chunk.  Same transport, same
        #   job, ~3 orders faster — the defect was never the transport.
        #
        # H5FD_MPIO_COLLECTIVE makes ROMIO aggregate the strided tiles in
        # its two-phase exchange and emit a few large contiguous writes
        # per aggregator instead.  ``LORRAX_PHDF5_COLLECTIVE_WRITES=0``
        # restores the pre-AI behaviour exactly.
        self._collective = _env_flag("LORRAX_PHDF5_COLLECTIVE_WRITES", True)
        # Collective transfer mode requires EVERY rank to enter every
        # H5Dwrite, so ranks with nothing to write need a null selection
        # rather than an early return.  Build the dxpl once.
        self._dxpl_coll = None
        if self._collective:
            from h5py import h5fd, h5p
            dxpl = h5p.create(h5p.DATASET_XFER)
            dxpl.set_dxpl_mpio(h5fd.MPIO_COLLECTIVE)
            self._dxpl_coll = dxpl
        # Drop redundant writes of replicated axes.  A replicated axis
        # gives every rank in that mesh row the SAME hyperslab and the
        # SAME bytes; under independent MPI-IO that was merely wasteful,
        # under collective MPI-IO overlapping selections are also
        # undefined behaviour.  One canonical writer per distinct
        # hyperslab fixes both (and cuts psi_full_y's traffic by Px).
        self._dedup = _env_flag("LORRAX_PHDF5_DEDUP_REPLICAS", True)
        if _rank0() and _env_flag("LORRAX_PHDF5_LOG", True):
            print(f"  [SlabIO.phdf5_host] {os.path.basename(path)} mode={mode} "
                  f"collective_write={self._collective} dedup_replicas="
                  f"{self._dedup} stripe_count="
                  f"{os.environ.get('LORRAX_PHDF5_STRIPE_COUNT', '16')} "
                  f"stripe_unit={_stripe_size_bytes()} B", flush=True)

        # write_attr accumulates small (rank-0-only) writes that we
        # defer to close() so they don't interleave with collective
        # MPI-IO on the same file handle.  Same hazard as the FFI
        # path's _deferred_attrs.
        self._deferred_attrs: list[tuple[str, object]] = []

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
        # H5Dcreate is collective; all ranks must reach it before any
        # rank writes.  Synchronous writes elsewhere in this module
        # guarantee that ordering naturally.
        if name in self._fh:
            # idempotent on 'a' mode: respect the existing dataset
            return
        shape_t = tuple(int(s) for s in shape)
        h5_dtype = jnp.dtype(dtype) if not isinstance(dtype, np.dtype) else dtype
        # Collective H5Dcreate via h5py.  All ranks participate;
        # parallel HDF5 broadcasts the dataset metadata from rank 0.
        ds = self._fh.create_dataset(
            name, shape=shape_t, dtype=h5_dtype,
            chunks=tuple(chunks) if chunks else None,
        )
        if attrs:
            # h5py attribute write is small + replicated; all ranks
            # write identical bytes, parallel HDF5 dedup's.
            for k, v in attrs.items():
                ds.attrs[k] = v

    def write_attr(self, name: str, value) -> None:
        # Tiny rank-0-only writes — defer to close() to avoid mixing
        # h5py serial writes with active MPI-IO on the same file.
        self._deferred_attrs.append((name, value))

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
        k_chunk_size: int | None = None,  # noqa: ARG002 — allgather-only knob
    ) -> None:
        """Write A as a per-rank hyperslab of dataset ``name``.

        ``A`` is a sharded ``jax.Array``; each rank pulls its local
        shard via ``A.addressable_shards[0]``, clips it to the
        ``valid_shape`` extent (drops the padded tail), and writes the
        clipped block at ``(slab_offset + shard_offset_within_slab)``
        in the dataset.

        Independent MPI-IO writes — no rank-rank collective synchro
        at write time.  Equivalent to the FFI backend's default
        (``LORRAX_PHDF5_INDEPENDENT=1`` rationale, see
        ``ffi/phdf5/cpp/ctx.h``).
        """
        if not isinstance(A, jax.Array):
            A = jnp.asarray(A)

        # Resolve offset / shapes via the same helpers the FFI uses.
        off, slab_shape, gshape = _normalize_slab_request(
            op="write_slab", name=name, offset=offset,
            slab_shape=tuple(A.shape), global_shape=global_shape,
            check_bounds=False)
        vshape = _normalize_valid_shape(
            op="write_slab", name=name, valid_shape=valid_shape,
            slab_shape=slab_shape, offset=off, global_shape=gshape)

        # Ensure dataset exists (collective if it needs creating).
        if name not in self._fh:
            self.create_dataset(name, shape=gshape, dtype=A.dtype,
                                chunks=chunks)

        local, shard_offset = _local_shard_and_global_offset(A)
        clipped = _clip_shard_to_valid(local, shard_offset, off, vshape)
        dset = self._fh[name]
        if clipped is None:
            # This rank's shard is entirely in the padded tail: nothing
            # of its own to write.  Under INDEPENDENT MPI-IO it could
            # simply return; under COLLECTIVE it must still enter the
            # H5Dwrite, with a null selection.
            arr, ds_offset = None, None
        else:
            arr, ds_offset = clipped

        if self._dedup:
            # One canonical writer per distinct (offset, shape).  Ranks
            # that are duplicates of a lower rank drop to a null
            # selection.  The allgather is a few hundred bytes.
            key = (None if arr is None
                   else (tuple(int(o) for o in ds_offset),
                         tuple(int(s) for s in arr.shape)))
            keys = self._comm.allgather(key)
            me = self._comm.Get_rank()
            if key is not None and keys.index(key) != me:
                arr, ds_offset = None, None

        self._write_hyperslab(dset, ds_offset, arr)

    # ------------------------------------------------------------------
    def _write_hyperslab(self, dset, ds_offset, arr) -> None:
        """One ``H5Dwrite`` of ``arr`` at ``ds_offset`` (or a null
        selection when ``arr is None``), in the configured transfer mode.

        Uses the low-level h5py API rather than ``dset[slc] = arr`` for
        two reasons: a *null* selection has no high-level spelling, and
        the collective dxpl has to be the same object on every rank.
        """
        from h5py import h5s

        if arr is None:
            if not self._collective:
                return                      # independent: just skip
            fspace = dset.id.get_space()
            fspace.select_none()
            mspace = h5s.create_simple((1,))
            mspace.select_none()
            buf = np.zeros((1,), dtype=dset.dtype)
            dset.id.write(mspace, fspace, buf, dxpl=self._dxpl_coll)
            return

        # Clipping an inner axis makes ``local[...]`` a strided VIEW;
        # H5Dwrite wants a contiguous buffer.  No-op when nothing clipped.
        arr = np.ascontiguousarray(arr)
        fspace = dset.id.get_space()
        fspace.select_hyperslab(tuple(int(o) for o in ds_offset),
                                tuple(int(s) for s in arr.shape))
        mspace = h5s.create_simple(arr.shape)
        dset.id.write(mspace, fspace, arr,
                      dxpl=self._dxpl_coll if self._collective else None)

    # ------------------------------------------------------------------
    def read_slab(
        self,
        name: str,
        *,
        shape: Sequence[int] | None = None,
        dtype=None,
        offset: Sequence[int] | None = None,
        valid_shape: Sequence[int] | None = None,
        mesh: Mesh | None = None,
        partition_spec: P | None = None,
        as_numpy: bool = False,
    ) -> jax.Array:
        """Read a hyperslab into a sharded JAX array.

        Each rank reads its own piece from the dataset (independent
        MPI-IO via h5py) and assembles the result via JAX's
        ``make_array_from_single_device_arrays`` (no allgather across
        processes — each process delivers its local block sized to
        match ``partition_spec``).
        """
        mesh = mesh or self.mesh
        if shape is None or dtype is None:
            ds = self._fh[name]
            ds_shape = tuple(int(s) for s in ds.shape)
            ds_dtype = np.dtype(ds.dtype)
            if shape is None:
                shape = ds_shape
            if dtype is None:
                dtype = ds_dtype

        off, read_shape, _ = _normalize_slab_request(
            op="read_slab", name=name, offset=offset,
            slab_shape=tuple(shape), global_shape=None,
            check_bounds=False)
        vshape = _normalize_valid_shape(
            op="read_slab", name=name, valid_shape=valid_shape,
            slab_shape=read_shape, offset=off, global_shape=None)

        if partition_spec is None:
            partition_spec = P(*([None] * len(read_shape)))

        # Build a target sharding so the resulting JAX array advertises
        # the right partition spec; we'll plumb each rank's local
        # block into it via make_array_from_single_device_arrays.
        sharding = NamedSharding(mesh, partition_spec)

        # Ask the sharding directly which hyperslab THIS rank owns.
        # `addressable_devices_indices_map` is PURE METADATA — it
        # allocates nothing and communicates nothing — and it is the same
        # idiom `common.collectives.device_put_process_local` uses.
        #
        # It replaces a probe that built a zero array of the WHOLE GLOBAL
        # SHAPE and `device_put` it onto `sharding` just to read
        # `.addressable_shards[0].index`.  That did two bad things at
        # once: it materialised the full tensor on every rank, AND
        # `device_put(numpy, <multi-process NamedSharding>)` trips JAX's
        # hidden `multihost_utils.assert_equal` (scorecard AA.1), i.e. a
        # `process_allgather` of P x the global array.  MEASURED: job
        # 7876346 died here in `compute_V_q` at MoS2 12x12 / c2406 with
        #   nq*mu_pad*ngkmax*16 = 144*2448*8603*16 = 48.52 GB/rank
        #   x P=144                                = 6.99 TB
        # reported verbatim as `RESOURCE_EXHAUSTED: Out of memory
        # allocating 6987250335744 bytes`.  That run was the first ever to
        # reach `compute_V_q` at this size, which is why a read path in
        # production since wk_AB had never shown it: the cost is
        # O(P x global tensor) and only bites once the tensor is large.
        idx_map = sharding.addressable_devices_indices_map(tuple(read_shape))
        index = idx_map[jax.local_devices()[0]]
        # Same None-handling as _local_shard_and_global_offset: replicated
        # axes come back as slice(None, None).
        local_shape = tuple(
            int(s.stop - s.start) if s.start is not None else int(full)
            for s, full in zip(index, read_shape))
        shard_offset = tuple(int(s.start) if s.start is not None else 0
                             for s in index)

        # Read the local hyperslab.  Clip to valid_shape: anything in the
        # padded tail is zero-filled by JAX's array assembly.
        clipped = _clip_shard_to_valid(
            np.zeros(local_shape, dtype=np.dtype(dtype)),  # shape carrier
            shard_offset, off, vshape,
        )
        host_local = np.zeros(local_shape, dtype=np.dtype(dtype))
        if clipped is not None:
            _, ds_offset = clipped
            keep_shape = tuple(min(ls, ve - so)
                               for ls, ve, so in zip(local_shape, vshape,
                                                     shard_offset))
            slc = tuple(slice(o, o + s)
                        for o, s in zip(ds_offset, keep_shape))
            # Read into the prefix of host_local; the padded tail stays
            # zeroed out.  h5py.dataset[slc] returns a freshly-allocated
            # ndarray; we copy into host_local's prefix.
            read_block = self._fh[name][slc]
            sel = tuple(slice(0, s) for s in keep_shape)
            host_local[sel] = read_block

        # Assemble per-process locals back into a globally-sharded array.
        device = jax.local_devices()[0]
        local_arr = jax.device_put(host_local, device)
        result = jax.make_array_from_single_device_arrays(
            tuple(read_shape), sharding, [local_arr],
        )
        if as_numpy:
            return np.asarray(host_local)
        return result

    # ------------------------------------------------------------------
    def close(self) -> None:
        # Flush deferred small writes via h5py.  All ranks call
        # create_dataset (it's collective under parallel HDF5); we
        # broadcast the rank-0 value to all ranks so every rank writes
        # the same bytes.
        if self._deferred_attrs:
            for ds_name, value in self._deferred_attrs:
                arr = np.asarray(value) if _rank0() else None
                arr = self._comm.bcast(arr, root=0)
                # Default: top-level dataset.  Could extend to
                # "dset:attr" form if a caller ever needs per-dataset
                # attrs deferred; today no caller does.
                if ds_name not in self._fh:
                    self._fh.create_dataset(ds_name, data=arr)
        # All ranks close — collective MPI-IO H5Fclose.
        self._fh.close()

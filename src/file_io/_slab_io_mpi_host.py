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
* The production stack: ``MPITRAMPOLINE_LIB`` must point at the
  MPIwrapper built by ``config/frontera/build_mpiwrapper.sh`` — it
  upgrades every init request to MULTIPLE and never downgrades — plus
  ``LORRAX_MPI_FINALIZE_FIX=skip_atexit`` via the overlay
  ``sitecustomize`` (jax registers an atexit ``collectives.Finalize``
  AND the C++ destructor finalizes → double-finalize rc=1 without it).
  These are harness-level machine facts, not LORRAX policy: LORRAX
  reads neither variable (see docs/dev/env_vars.md and
  docs/dev/mpi_collectives.md).
* The C++ FFI twin guards the same hazard at open time
  (``ffi/cpp/phdf5/context.cc`` MPI_Query_thread warning), which is what
  fires if the wrapper is missing from the path.

What we DO match from :class:`_FfiBackend`:

* Same ``mode='w'`` inode replace before any rank opens the file (the
  shared ``_replace_inode_for_write``; the layout that actually sticks
  comes from the MPI-IO striping hints — see :func:`_mpi_io_hints`).
* Same collective-write default (``LORRAX_PHDF5_COLLECTIVE_WRITES``,
  on) with replica dedup (``LORRAX_PHDF5_DEDUP_REPLICAS``, on) — see
  ``__init__`` for the wk_AI measurements.
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
    _assert_mpi_world,
    _barrier,
    _DatasetGeometry,
    _normalize_slab_request,
    _normalize_valid_shape,
    _rank0,
    _replace_inode_for_write,
    _shard_read_plan,
    _stripe_count,
    _stripe_size_bytes,
)

_TRUE = ("1", "true", "yes", "on")


def _env_flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in _TRUE


# ``_stripe_count`` and ``_stripe_size_bytes`` are imported, not defined
# here, since 2026-08-06.  They used to live in two modules with two
# different notions of "the default" (this one said 1 MiB, the C++ header
# comment said 4 MiB), which is the shape of bug where two writers agree
# on the environment and disagree on the layout.  Both are now one policy
# function of the rank count in ``_slab_io_ffi``; see ``_stripe_policy``.


def _mpi_io_hints(MPI):
    """The ``MPI_Info`` handed to ``H5Pset_fapl_mpio``.

    **This is the striping lever that actually works on Frontera.**
    The deleted ``lfs setstripe`` prestripe helper shelled out to
    ``lfs``, which does NOT EXIST inside the apptainer container every
    production run uses, so it was a silent no-op on this machine —
    MEASURED: both files job 7876423 wrote (``zeta_q.h5``,
    ``isdf_tensors_2406.h5``) came back ``lmm_stripe_count: 1``, i.e.
    13.3 GB of ``V_qmunu`` funnelled through a SINGLE OST.  ROMIO sets
    the layout through ``llapi`` at file-create time instead, so the
    hints below reach Lustre from inside the container with no binary
    on PATH.

    Same keys, same defaults as the C++ writer's
    ``ffi/cpp/phdf5/context.cc`` (which has always set them) — the h5py
    backend set NONE, which is the single biggest divergence between
    the two "equivalent" writers.
    """
    info = MPI.Info.Create()
    sc = _stripe_count()
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
    # (``ffi/cpp/phdf5/context.cc``) follows the SAME policy — its old
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
class _MpiHostBackend(_DatasetGeometry):
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

        # mode='w' must REPLACE the inode (rank-0 unlink + barrier,
        # shared with _FfiBackend — see _replace_inode_for_write for the
        # full rationale): a Lustre stripe layout is a property of the
        # INODE, fixed at create time; ``lfs setstripe``, the
        # ``striping_factor`` hint and ``H5Fcreate(TRUNC)`` are all
        # no-ops against an existing file, so a rerun over a 1-stripe
        # ``isdf_tensors_*.h5`` would silently keep 1 stripe for ever.
        # The helper uses lexists (dangling symlinks too) and RAISES on
        # a failed unlink instead of the old silent ``except OSError:
        # pass`` (audit 2026-07-28).  'a'/'r' inherit the existing
        # inode's layout by design.  Barrier after the rank-0 unlink
        # so all ranks see the inode state before H5Fopen.
        if mode == "w":
            _replace_inode_for_write(path)
            _barrier("slab_io_mpi_host_prestripe")

        h5_mode = {"w": "w", "a": "a", "r": "r"}[mode]
        self._info = _mpi_io_hints(MPI)
        self._fh = h5py.File(path, h5_mode, driver="mpio",
                             comm=self._comm, info=self._info)
        # The MPI world this handle ACTUALLY collects on, vs the JAX world
        # the hyperslabs are derived from.  Same guard as _FfiBackend, and
        # it was missing here until 2026-08-06 — tier 2 is a real
        # multi-process path (``gw_config._route_cpu_slab_io`` falls to it
        # whenever the host FFI lib lacks ``PhdfWriteHostFfi``), so the
        # PMI-mismatch failure that guard exists for was reachable with no
        # guard on it at all.  ``mpi4py``'s COMM_WORLD is handed in
        # explicitly rather than probed through ctypes: this backend
        # already holds the authoritative communicator, and it is the SAME
        # one h5py's mpio driver collects on, so there is no ABI guess to
        # get wrong.  Crucially it is still a different frame of reference
        # from ``jax.process_count()`` — the whole point of the check.
        _assert_mpi_world(mesh, mpi_size=int(self._comm.Get_size()),
                          probe_detail="mpi4py MPI.COMM_WORLD.Get_size()",
                          tier="PHDF5_HOST")

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
                  f"{self._dedup} stripe_count={_stripe_count()} "
                  f"stripe_unit={_stripe_size_bytes()} B", flush=True)

        # write_attr accumulates small (rank-0-only) writes that we
        # defer to close() so they don't interleave with collective
        # MPI-IO on the same file handle.  Same hazard as the FFI
        # path's _deferred_attrs.
        self._deferred_attrs: list[tuple[str, object]] = []
        # Replicated record of each dataset's LOGICAL geometry; here it
        # is only a cache, because ``self._fh`` is a parallel handle that
        # every rank can query directly.
        self._geom_init()

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
        shape_t = tuple(int(s) for s in shape)
        h5_dtype = jnp.dtype(dtype) if not isinstance(dtype, np.dtype) else dtype
        if name in self._fh:
            # Idempotent only when the geometry MATCHES: decisions.md
            # 2026-08-04.  The old unconditional early return let an
            # ``mode='a'`` rerun at a different μ write into the previous
            # extent — wrong physics with no symptom.  Every rank holds
            # the same parallel handle, so this verdict is rank-invariant
            # and either everyone refuses or nobody does.
            existing = self._fh[name]
            self._refuse_geometry_change(
                op="create_dataset", name=name,
                want_shape=shape_t, want_dtype=h5_dtype,
                have_shape=existing.shape, have_dtype=existing.dtype)
            self._remember_geom(name, existing.shape, existing.dtype)
            return
        # Collective H5Dcreate via h5py.  All ranks participate;
        # parallel HDF5 broadcasts the dataset metadata from rank 0.
        ds = self._fh.create_dataset(
            name, shape=shape_t, dtype=h5_dtype,
            chunks=tuple(chunks) if chunks else None,
        )
        self._remember_geom(name, shape_t, h5_dtype)
        if attrs:
            # h5py attribute write is small + replicated; all ranks
            # write identical bytes, parallel HDF5 dedup's.
            for k, v in attrs.items():
                ds.attrs[k] = v

    # ------------------------------------------------------------------
    def _dataset_shape(self, name: str) -> tuple[int, ...] | None:
        """LOGICAL extent of ``name``, or None if it does not exist yet.

        Read straight off the parallel handle, so it is the same tuple on
        every rank — the precondition for deriving ``valid_shape`` from
        it (decisions.md 2026-08-04).
        """
        known = self._known_shape(name)
        if known is not None:
            return known
        if name not in self._fh:
            return None
        ds = self._fh[name]
        self._remember_geom(name, ds.shape, ds.dtype)
        return tuple(int(s) for s in ds.shape)

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

        COLLECTIVE transfer mode by default (``H5FD_MPIO_COLLECTIVE``;
        wk_AI measurements at the ``self._collective`` comment in
        ``__init__`` — the fix for AF.4c's 1.7 MB/s strided-tile
        writes).  That makes this call rank-synchronous: EVERY rank
        must enter it for every dataset — ranks with nothing to write
        participate with a null selection, and an early return would
        deadlock the two-phase exchange.  Additionally, with
        ``LORRAX_PHDF5_DEDUP_REPLICAS`` (default on) each call runs a
        small blocking ``comm.allgather`` to elect one canonical
        writer per distinct hyperslab.
        ``LORRAX_PHDF5_COLLECTIVE_WRITES=0`` restores independent
        MPI-IO writes (the pre-AI behaviour).  Note
        ``LORRAX_PHDF5_INDEPENDENT`` is unrelated: it controls FFI
        READS only (``force_indep_read``, ``ffi/cpp/phdf5/context.cc``).
        """
        if not isinstance(A, jax.Array):
            A = jnp.asarray(A)

        # Resolve offset / shapes via the same helpers the FFI uses.
        off, slab_shape, req_gshape = _normalize_slab_request(
            op="write_slab", name=name, offset=offset,
            slab_shape=tuple(A.shape), global_shape=global_shape,
            check_bounds=False)

        # Ensure dataset exists (collective if it needs creating), then
        # derive the logical extent from the DATASET, not from the
        # caller (decisions.md 2026-08-04).
        ds_shape = self._dataset_shape(name)
        if ds_shape is None:
            self.create_dataset(name, shape=req_gshape, dtype=A.dtype,
                                chunks=chunks)
            ds_shape = req_gshape
        elif global_shape is not None and req_gshape != ds_shape:
            raise ValueError(
                f"write_slab {name!r}: global_shape={req_gshape} contradicts "
                f"the dataset's extent {ds_shape}.  global_shape is only "
                f"needed to CREATE a dataset; drop it and SlabIO uses the "
                f"dataset's own shape.")
        vshape = _normalize_valid_shape(
            op="write_slab", name=name, valid_shape=valid_shape,
            slab_shape=slab_shape, offset=off, ds_shape=ds_shape)

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
        ds = self._fh[name]
        ds_shape = tuple(int(s) for s in ds.shape)
        ds_dtype = np.dtype(ds.dtype)
        self._remember_geom(name, ds_shape, ds_dtype)
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
            slab_shape=read_shape, offset=off, ds_shape=ds_shape)

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
        # Shared index→(local block, valid-clipped hyperslab) arithmetic
        # — single source of truth with the allgather backend's sharded
        # fast path (``_slab_io_ffi._shard_read_plan``; audit
        # 2026-07-28, QUALITY_PATTERNS #3).
        local_shape, dst, disk = _shard_read_plan(
            index, read_shape, off, vshape)
        host_local = np.zeros(local_shape, dtype=np.dtype(dtype))
        if dst is not None:
            # Read the valid part into the block prefix; the padded
            # tail stays zeroed.  h5py returns a fresh ndarray; we copy
            # into host_local in place.
            host_local[dst] = self._fh[name][disk]

        # Assemble per-process locals back into a globally-sharded array.
        device = jax.local_devices()[0]
        local_arr = jax.device_put(host_local, device)
        result = jax.make_array_from_single_device_arrays(
            tuple(read_shape), sharding, [local_arr],
        )
        if as_numpy:
            # ``host_local`` is this rank's BLOCK, not the slab.  Returning
            # it for a sharded request would hand the caller 1/P of the
            # data typed as the whole thing — the allgather backend's
            # ``as_numpy`` returns the full replicated slab, so the two
            # backends would silently disagree on what they returned.
            # (Unreachable through ``SlabIO.read_slab``, which does not
            # forward ``as_numpy`` on the phdf5 paths and converts the
            # returned array itself; reachable by using this backend
            # directly, which the tests do.)
            if any(s is not None for s in partition_spec):
                raise ValueError(
                    f"_MpiHostBackend.read_slab({name!r}): as_numpy=True is "
                    f"only defined for a replicated request; got "
                    f"partition_spec={partition_spec!r}, whose per-rank "
                    f"block is 1/P of the slab.  Drop as_numpy and convert "
                    f"the sharded jax.Array, or ask for P().")
            return np.asarray(host_local)
        return result

    # ------------------------------------------------------------------
    def close(self) -> None:
        # Flush deferred small writes via h5py.  All ranks call
        # create_dataset (it's collective under parallel HDF5); we
        # broadcast the rank-0 value to all ranks so every rank writes
        # the same bytes.
        #
        # The loop body is TWO collectives per attr (``comm.bcast`` and a
        # collective ``create_dataset``), so its trip count must be the
        # same on every rank or close() deadlocks.  ``_deferred_attrs`` is
        # a per-rank Python list built by per-rank ``write_attr`` calls;
        # every call site in the tree is SPMD today, but that is a property
        # of the callers, not of this method.  Agree the count first — one
        # unconditional allgather of a small int, on every rank — so a
        # divergence becomes a NAMED refusal on every rank instead of a
        # hang with no message (the failure mode that costs runs: stderr
        # is lost under srun+apptainer at teardown).
        names = [n for n, _ in self._deferred_attrs]
        all_names = self._comm.allgather(names)
        # A rank must not skip a collective because of its own error
        # (decisions.md 2026-08-04).  The divergence verdict IS
        # rank-invariant (every rank sees the same allgather result), but
        # the H5Fclose below is collective either way, so run the
        # teardown first and raise after — otherwise a future
        # rank-dependent refusal here would strand the file handle.
        divergent = any(n != names for n in all_names)
        try:
            if not divergent:
                for ds_name, value in self._deferred_attrs:
                    arr = np.asarray(value) if _rank0() else None
                    arr = self._comm.bcast(arr, root=0)
                    # Default: top-level dataset.  Could extend to
                    # "dset:attr" form if a caller ever needs per-dataset
                    # attrs deferred; today no caller does.
                    if ds_name not in self._fh:
                        self._fh.create_dataset(ds_name, data=arr)
        finally:
            # All ranks close — collective MPI-IO H5Fclose.
            self._fh.close()
        if divergent:
            raise RuntimeError(
                f"SlabIO(phdf5_host).close: deferred write_attr lists differ "
                f"across ranks — rank {self._comm.Get_rank()} has {names!r}, "
                f"rank 0 has {all_names[0]!r}.  The flush loop is "
                f"collective, so a divergent list would deadlock every rank. "
                f"write_attr must be called identically on all ranks.")

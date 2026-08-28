"""Serial SlabIO backend — the EMULATED-MESH tier, and nothing else.

WHAT IT IS.  ``_SerialBackend`` serves the same seven operations as
:class:`file_io._slab_io_ffi._FfiBackend` over ONE ``h5py.File`` handle,
with no MPI, no collective open and no per-rank hyperslab.  It is selected
by :class:`file_io.slab_io.SlabIO` — and only there — when
``common.collectives.mesh_is_emulated(mesh)`` holds: one process owning
every cell of a multi-device mesh.

WHY THIS IS NOT THE TIER THAT WAS DELETED (2026-08-06, ``slab_io.py:21``).
``H5PY_ALLGATHER`` gathered the whole global array **onto rank 0** — an OOM
at the design envelope, refused at seven separate doors and deleted for it.
The difference here is checkable rather than rhetorical: at
``process_count() == 1`` this process ALREADY holds every shard of every
mesh-sharded array, because all N devices are its own.  So this backend
never gathers anything across a process boundary, and it is written to
move **one shard at a time** —

* :meth:`_SerialBackend.write_slab` iterates ``A.addressable_shards`` and
  writes each shard's own hyperslab, so the host-resident peak is ONE
  shard, not the global array;
* :meth:`_SerialBackend.read_slab` builds its result through
  ``jax.make_array_from_callback``, which asks for one shard at a time, so
  the host-resident peak is again ONE shard.

The scaling doctrine ("no N_mu^2-class object required to fit on one
process") is therefore not weakened BY THIS MODULE: whatever the emulation
already forced onto this process is what it moves, plus one shard.  The
doctrine is spent by the emulation itself, which is also why this tier is
scoped to ``P == 1`` and can never be selected on a real multi-process run
— the seven-doors failure mode.

WHAT IT REFUSES.  ``process_count() > 1``, at construction, by name.  It is
a tier chosen from a PREDICATE, never a fallback from a failed transport:
nothing in this module or in ``slab_io`` catches an FFI error and lands
here.  The four ``p*q != process_count()`` refusals in the tree
(``ffi/io.py``'s phdf5 open, ``ffi/cublasmp/batched.py``,
``distrib_la/_slate.py``, ``distrib_la/matmul.py``) are untouched and still
fire on an emulated mesh — that is what makes this a second door rather
than a relaxation of the first, and
``tests/test_slab_io_emulated_mesh.py`` observes it.

WHAT IT IS NOT FOR.  Production.  An emulated run is device-parallel inside
each jit and serial across everything a real run parallelises over
processes (``common.collectives.local_share``), and this backend's h5py
writes are serial by construction.  It validates SHAPES and NUMBERS on a
multi-device mesh without four ranks; it is not a performance proxy for one.

ONE HDF5 LIBRARY INSTANCE PER OPEN FILE still holds (audit A1;
``docs/architecture/slab_io.md#one-owner``).  This backend declares its open
to :mod:`file_io.hdf5_owner` under ``STACK_H5PY`` — the stack it actually
uses — so a caller holding its own h5py handle on the same path is refused
by the same door that refuses it against the FFI.
"""
from __future__ import annotations

import os
from typing import Sequence

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.collectives import mesh_is_emulated, process_count
from runtime import debug_print_enabled

from . import h5_journal as _journal
from ._slab_io_ffi import (_DatasetGeometry, _apply_dataset_attrs,
                           _normalize_slab_request, _normalize_valid_shape,
                           _normalize_window_tables, _replace_inode_for_write,
                           _sharding_to_axis_info, _shard_divisors,
                           _validate_block_divisible, mesh_divisible_shape)

#: Journal stack name.  Every HDF5 call this module makes goes through
#: h5py's bundled libhdf5, so it is journaled under the same token
#: ``file_io.hdf5_owner`` and ``file_io.h5_journal`` use for that library —
#: a journal that called it something else would not line up with the
#: registry's verdict on the same path.
_J_H5PY = "h5py"

#: Announced once per process, not once per file: the tier is a property of
#: the run's (process, device) geometry, and one line per zeta chunk would
#: bury it.
_ANNOUNCED: set = set()


def announce_tier(mesh) -> None:
    """One line naming the tier and why, on the first emulated open.

    Same shape as ``WfnLoader``'s ``read backend = eager …`` line
    (``services/wfn_loader/src/wfn_loader/loader.py``): the tier, the
    geometry that selected it, and the mechanism — so a log says which
    transport moved the bytes without the reader having to know the
    predicate.
    """
    if not debug_print_enabled():
        return
    key = (int(mesh.devices.size), int(process_count()))
    if key in _ANNOUNCED:
        return
    _ANNOUNCED.add(key)
    print(f"  [SlabIO] transport = serial (auto, {key[1]} process, "
          f"{key[0]} devices) — an emulated mesh has no per-process MPI "
          f"world for the phdf5 hyperslab (ffi.io.open_file refuses "
          f"p*q != process_count); host h5py moves one shard at a time, "
          f"and every shard is this process's own.", flush=True)


def _index_bounds(index, shape: Sequence[int]) -> tuple[tuple[int, ...],
                                                        tuple[int, ...]]:
    """``(start, stop)`` per dim for a shard index, replicated dims full.

    A shard's ``.index`` is a tuple of ``slice``; a REPLICATED dim gives
    ``slice(None, None)``, which means the whole axis and must be read as
    ``(0, shape[d])`` rather than skipped — every device holds that axis
    entire.  ``jax.make_array_from_callback`` hands the same object shape,
    so one reader serves both directions.
    """
    starts, stops = [], []
    for d, sl in enumerate(index):
        lo = 0 if sl.start is None else int(sl.start)
        hi = int(shape[d]) if sl.stop is None else int(sl.stop)
        starts.append(lo)
        stops.append(hi)
    return tuple(starts), tuple(stops)


class _SerialBackend(_DatasetGeometry):
    """Single-process h5py SlabIO backend.  See the module docstring."""

    def __init__(self, path: str, mesh: Mesh, mode: str = "w") -> None:
        # THE TIER IS SELECTABLE ONLY AT P == 1, and it says so here rather
        # than trusting its one caller.  A fallback for a broken
        # multi-process launch is exactly what this must not become: at
        # P > 1 every rank would open the same file with serial h5py and
        # write its own shard on top of its peers' (last flush wins, no
        # error anywhere).  Two ways out, named, in the style of
        # distrib_la.resolve's guard 4.
        if int(process_count()) > 1:
            raise RuntimeError(
                "SlabIO serial tier refusal\n"
                f"  file  : {path}\n"
                f"  got   : jax.process_count() = {int(process_count())} with "
                f"a {'x'.join(str(int(mesh.shape[a])) for a in mesh.axis_names)}"
                f" mesh.\n"
                "  want  : the serial tier is the EMULATED-mesh tier and is "
                "selectable only at process_count() == 1, where every mesh "
                "cell is this process's own device.  Above one process it "
                "would have each rank overwrite its peers' hyperslabs "
                "through separate h5py handles, with no error anywhere.\n"
                "  fix   : build the mesh from the process count and let the "
                "phdf5 FFI transport serve it (that is the production path "
                "at P > 1), or run single-process with "
                "--xla_force_host_platform_device_count for the emulated "
                "geometry.\n"
                "  doc   : docs/architecture/slab_io.md")
        if not mesh_is_emulated(mesh):
            raise RuntimeError(
                "SlabIO serial tier refusal\n"
                f"  file  : {path}\n"
                f"  got   : mesh.devices.size = {int(mesh.devices.size)} at "
                f"process_count() = {int(process_count())}, which is not an "
                f"emulated mesh.\n"
                "  want  : common.collectives.mesh_is_emulated(mesh).  A "
                "1x1 mesh at one process is served by the phdf5 FFI "
                "transport like every other non-emulated geometry; there is "
                "no second path for it and this tier must not become one.\n"
                "  fix   : construct SlabIO normally — file_io.slab_io picks "
                "the transport from the same predicate.\n"
                "  doc   : docs/architecture/slab_io.md")
        announce_tier(mesh)

        self.path = str(path)
        self.mesh = mesh
        self.mode = mode
        # SAME INODE CONTRACT AS THE FFI TIER.  ``mode='w'`` REPLACES the
        # inode rather than truncating it, so a rerun over an existing file
        # cannot inherit its Lustre layout.  h5py's own ``'w'`` truncates in
        # place, which would make the two tiers disagree about what
        # ``mode='w'`` means — a difference no caller asked for.
        if mode == "w":
            _replace_inode_for_write(self.path)

        import h5py

        from .hdf5_owner import STACK_H5PY, note_close, note_open
        self._owner_token: int | None = note_open(
            self.path, STACK_H5PY, mode,
            where=f"SlabIO/_SerialBackend({os.path.basename(self.path)}, "
                  f"mode={mode!r})")
        try:
            with _journal.op_scope("open", self.path, stack=_J_H5PY,
                                   mode=mode):
                self._h5 = h5py.File(self.path, mode)
        except BaseException:
            note_close(self.path, self._owner_token)
            self._owner_token = None
            raise
        # ``fh`` is read by ``SlabIO._handle()`` for the journal's handle
        # field.  There is no ctx address to report — the handle this tier
        # owns is a Python object, not an int the C++ side could name — so
        # it stays 0 and the journal lines carry no handle, which is the
        # truth rather than a fabricated one.
        self.fh = 0
        self._geom_init()
        self._deferred_attrs: list[tuple[str, object]] = []
        self._deferred_ds_attrs: list[tuple[str, dict]] = []

    # ------------------------------------------------------------------
    def _drain_pending(self) -> int:
        """No queue to drain: this tier's writes are synchronous.

        Kept because ``SlabIO.sync_writes`` and the read paths call it.
        Returning 0 is not a stub — zero bytes are in flight, always, which
        is a STRONGER guarantee than the FFI tier's and the reason nothing
        here needs the read-after-write drain that path documents.
        """
        return 0

    def _dataset_geom(self, name: str) -> tuple[tuple[int, ...], "np.dtype"]:
        got = self._ds_geom.get(str(name))
        if got is not None:
            return got
        if name not in self._h5:
            raise KeyError(
                f"SlabIO({os.path.basename(self.path)}): no dataset "
                f"{name!r} in the file")
        ds = self._h5[name]
        shape, dtype = tuple(int(s) for s in ds.shape), np.dtype(ds.dtype)
        self._remember_geom(name, shape, dtype)
        return shape, dtype

    # ------------------------------------------------------------------
    def create_dataset(
        self,
        name: str,
        *,
        shape: Sequence[int],
        dtype,
        attrs: dict | None = None,
    ) -> None:
        """Pre-create at the given LOGICAL shape; reuse-or-refuse on a clash.

        The rule is ``lrx_phdf5_ensure_dataset``'s, restated here because
        this tier does not go through it: identical shape AND dtype reuse
        the dataset, anything else REFUSES naming both.  Never
        delete-and-recreate (data loss) and never write into the previous
        geometry (wrong extent, no symptom) — decisions.md 2026-08-04.
        """
        want_shape = tuple(int(s) for s in shape)
        want_dtype = np.dtype(dtype)
        with _journal.op_scope("create", self.path, stack=_J_H5PY, ds=name,
                               cnt=want_shape, mode=self.mode):
            if name in self._h5:
                ds = self._h5[name]
                got_shape = tuple(int(s) for s in ds.shape)
                got_dtype = np.dtype(ds.dtype)
                if got_shape != want_shape or got_dtype != want_dtype:
                    raise ValueError(
                        f"SlabIO.create_dataset({name!r}) on "
                        f"{os.path.basename(self.path)}: the dataset exists "
                        f"as {got_shape}/{got_dtype.name} and you asked for "
                        f"{want_shape}/{want_dtype.name}.  SlabIO neither "
                        f"deletes-and-recreates (data loss) nor writes into "
                        f"the previous geometry (wrong extent, no symptom); "
                        f"pick one shape.")
            else:
                self._h5.create_dataset(name, shape=want_shape,
                                        dtype=want_dtype)
        self._remember_geom(name, want_shape, want_dtype)
        if attrs:
            self._deferred_ds_attrs.append((name, dict(attrs)))

    def write_attr(self, name: str, value) -> None:
        """QUEUE a small replicated dataset for close, as the FFI tier does.

        DEFERRED even though nothing here holds the file collectively.  The
        deferral is part of ``SlabIO.write_attr``'s published contract —
        "the dataset does not exist until ``close()`` RETURNS" — and a tier
        that landed it earlier would let a caller depend on timing the other
        tier does not offer.
        """
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
    ) -> None:
        """Write ``A``'s shards into ``name``, one shard at a time.

        Same three-step contract as the FFI writer, with the per-rank
        arithmetic replaced by the shard's own ``.index``:

        1. the dataset's LOGICAL extent comes from the record, then the
           caller's ``global_shape``, then ``A.shape``;
        2. ``valid_shape`` defaults to ``min(A.shape, dataset - offset)``,
           so pad rows past the dataset are dropped with no argument;
        3. each shard writes only the part of itself inside that valid
           extent, at ``offset + <its global start>``.

        REPLICAS ARE DE-DUPLICATED by shard index, the same rule the
        collective writer applies for a different reason (overlapping
        hyperslab selections are undefined under MPI-IO; here they are
        merely N identical writes of identical bytes).  A fully replicated
        4-device array therefore costs ONE write, not four.
        """
        host = not isinstance(A, jax.Array)
        slab_shape_in = tuple(int(s) for s in np.shape(A))
        off, slab_shape, req_gshape = _normalize_slab_request(
            op="write_slab", name=name, offset=offset,
            slab_shape=slab_shape_in, global_shape=global_shape,
            check_bounds=False)
        if not host:
            # The same JAX constraint the FFI tier restates: a (shape, spec)
            # pair that cannot form equal block shards is one JAX itself
            # refuses to build, so name the numbers here rather than let it
            # surface from a traced buffer.
            axis_count_per_dim, axis_flat = _sharding_to_axis_info(
                A.sharding, A.ndim) if isinstance(
                    A.sharding, NamedSharding) else ((0,) * A.ndim, ())
            mesh_shape = tuple(int(self.mesh.shape[ax])
                               for ax in self.mesh.axis_names)
            _validate_block_divisible(
                op="write_slab", name=name, shape=slab_shape,
                axis_count_per_dim=axis_count_per_dim,
                axis_flat=axis_flat, mesh_shape=mesh_shape)

        ds_shape = self._known_shape(name)
        if ds_shape is None:
            ds_shape = req_gshape
            self.create_dataset(
                name, shape=ds_shape,
                dtype=np.dtype(dtype) if dtype is not None
                else np.dtype(A.dtype))
        elif global_shape is not None and req_gshape != ds_shape:
            raise ValueError(
                f"write_slab {name!r}: global_shape={req_gshape} contradicts "
                f"the dataset's extent {ds_shape}.  global_shape is only "
                f"needed to CREATE a dataset; drop it and SlabIO uses the "
                f"dataset's own shape.")

        vshape = _normalize_valid_shape(
            op="write_slab", name=name, valid_shape=valid_shape,
            slab_shape=slab_shape, offset=off, ds_shape=ds_shape)

        with _journal.op_scope("write", self.path, stack=_J_H5PY, ds=name,
                               off=off, cnt=slab_shape, mode=self.mode):
            ds = self._h5[name]
            if host:
                self._write_block(ds, np.asarray(A), (0,) * len(slab_shape),
                                  off, vshape)
                return
            seen: set = set()
            for shard in A.addressable_shards:
                starts, _stops = _index_bounds(shard.index, slab_shape)
                if starts in seen:
                    continue                     # replica of one already written
                seen.add(starts)
                self._write_block(ds, np.asarray(shard.data), starts, off,
                                  vshape)

    @staticmethod
    def _write_block(ds, block: np.ndarray, starts: Sequence[int],
                     offset: Sequence[int], vshape: Sequence[int]) -> None:
        """Write the part of one shard that lies inside the valid extent.

        ``starts`` is where the block begins inside the (possibly padded)
        slab; ``vshape`` is how much of that slab is real.  The intersection
        is what reaches the file, at ``offset + <its slab coordinate>`` —
        i.e. exactly the clip ``_derive_valid_shape`` expresses for the
        whole slab, evaluated per shard.  An empty intersection writes
        nothing, which is the pad-row semantics both tiers share.
        """
        src, dst = [], []
        for d, n in enumerate(block.shape):
            lo = int(starts[d])
            hi = min(lo + int(n), int(vshape[d]))
            if hi <= lo:
                return
            src.append(slice(0, hi - lo))
            dst.append(slice(int(offset[d]) + lo, int(offset[d]) + hi))
        ds[tuple(dst)] = block[tuple(src)]

    # ------------------------------------------------------------------
    def padded_shape_for(self, name: str, *, mesh: Mesh, partition_spec: P
                         ) -> tuple[int, ...]:
        ds_shape, _ = self._dataset_geom(name)
        return mesh_divisible_shape(ds_shape, mesh, partition_spec)

    def read_whole(self, name: str, *, dtype=None):
        """The whole small dataset as a host ``np.ndarray``.

        The scalar door: an H5 rank-0 dataspace has no hyperslab, so
        :meth:`read_slab` cannot express the request on either tier.
        """
        arr = np.asarray(self._h5[name][()])
        return arr.astype(np.dtype(dtype), copy=False) if dtype is not None \
            else arr

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
        as_numpy: bool = False,   # handled by the public SlabIO.read_slab
    ) -> jax.Array:
        """One hyperslab, sharded ``partition_spec``, built shard by shard.

        ``jax.make_array_from_callback`` asks for each addressable shard's
        global index in turn, so the host never holds more than one shard —
        the same tile guarantee the collective reader gets from its per-rank
        H5Dread, reached a different way.  Past ``valid_shape`` the result
        is zero, per shard, which is the padded consumer buffer both tiers
        promise.
        """
        mesh = mesh or self.mesh
        ds_shape, ds_dtype = self._dataset_geom(name)
        if shape is None:
            shape = ds_shape
        if dtype is None:
            dtype = ds_dtype
        off, read_shape, _ = _normalize_slab_request(
            op="read_slab", name=name, offset=offset,
            slab_shape=shape, global_shape=None, check_bounds=False)
        if partition_spec is None:
            partition_spec = P(*([None] * len(read_shape)))
        sharding = NamedSharding(mesh, partition_spec)
        axis_count_per_dim, axis_flat = _sharding_to_axis_info(
            sharding, len(read_shape))
        mesh_shape = tuple(int(mesh.shape[ax]) for ax in mesh.axis_names)
        vshape = _normalize_valid_shape(
            op="read_slab", name=name, valid_shape=valid_shape,
            slab_shape=read_shape, offset=off, ds_shape=ds_shape)
        _validate_block_divisible(
            op="read_slab", name=name, shape=read_shape,
            axis_count_per_dim=axis_count_per_dim,
            axis_flat=axis_flat, mesh_shape=mesh_shape)

        ds = self._h5[name]
        want = np.dtype(dtype)
        # REPLICAS READ ONCE.  A fully replicated spec gives every device the
        # same index; without this cache a 4-device replicated read is four
        # identical H5Dreads of the same bytes.
        cache: dict = {}

        def _block(index):
            starts, stops = _index_bounds(index, read_shape)
            got = cache.get(starts)
            if got is not None:
                return got
            out = np.zeros(tuple(hi - lo for lo, hi in zip(starts, stops)),
                           dtype=want)
            src, dst = [], []
            empty = False
            for d in range(len(read_shape)):
                lo, hi = starts[d], min(stops[d], int(vshape[d]))
                if hi <= lo:
                    empty = True
                    break
                dst.append(slice(0, hi - lo))
                src.append(slice(int(off[d]) + lo, int(off[d]) + hi))
            if not empty:
                out[tuple(dst)] = np.asarray(ds[tuple(src)], dtype=want)
            cache[starts] = out
            return out

        with _journal.op_scope("read", self.path, stack=_J_H5PY, ds=name,
                               off=off, cnt=read_shape, mode=self.mode):
            return jax.make_array_from_callback(
                tuple(int(s) for s in read_shape), sharding, _block)

    # ------------------------------------------------------------------
    def read_slabs(
        self,
        name: str,
        *,
        shape: Sequence[int],
        offsets,
        valid_shapes,
        partition_spec: P,
        window_axis: int,
        dtype=None,
        mesh: Mesh | None = None,
    ) -> jax.Array:
        """n windows of ONE slab shape, packed along ``window_axis``.

        The collective tier serves this as one ``H5S_SELECT_OR`` compound
        hyperslab because n per-window collectives cost 1.4 s of rendezvous
        at the production deck (``SlabIO.read_slabs``).  Serially there is
        no rendezvous to amortise, so it is n reads per shard — same bytes,
        same packing, same per-window clip
        (``_derive_window_counts``' arithmetic, evaluated here per shard).

        The caller's precondition is UNCHANGED and still theirs: windows
        pairwise disjoint and sorted ascending in file order.  This tier
        happens not to need it (each window is its own read), and that is
        exactly why it must not be relaxed — a caller that got away with it
        here would break on the collective tier with no message.
        """
        mesh = mesh or self.mesh
        ds_shape, ds_dtype = self._dataset_geom(name)
        slab_shape = tuple(int(s) for s in shape)
        want = np.dtype(dtype if dtype is not None else ds_dtype)
        offsets_t, valid_t = _normalize_window_tables(
            name=name, offsets=offsets, valid_shapes=valid_shapes,
            ndim=len(slab_shape))
        n_win = int(offsets_t.shape[0])

        sharding = NamedSharding(mesh, partition_spec)
        axis_count_per_dim, axis_flat = _sharding_to_axis_info(
            sharding, len(slab_shape))
        mesh_shape = tuple(int(mesh.shape[ax]) for ax in mesh.axis_names)
        _validate_block_divisible(
            op="read_slabs", name=name, shape=slab_shape,
            axis_count_per_dim=axis_count_per_dim,
            axis_flat=axis_flat, mesh_shape=mesh_shape)
        divs = _shard_divisors(
            axis_count_per_dim=axis_count_per_dim, axis_flat=axis_flat,
            mesh_shape=mesh_shape, ndim=len(slab_shape))
        per_rank_shape = tuple(s // d for s, d in zip(slab_shape, divs))

        out_shape = (slab_shape[:window_axis] + (n_win,)
                     + slab_shape[window_axis:])
        out_spec = P(*(tuple(partition_spec[:window_axis]) + (None,)
                       + tuple(partition_spec[window_axis:])))
        out_sharding = NamedSharding(mesh, out_spec)
        ds = self._h5[name]
        cache: dict = {}

        def _block(index):
            # Drop the window axis: the rest of the index is the shard's
            # block inside ONE window, which is what the per-(rank, window)
            # clip is expressed in.
            file_index = tuple(index[:window_axis]) + tuple(
                index[window_axis + 1:])
            starts, _stops = _index_bounds(file_index, slab_shape)
            got = cache.get(starts)
            if got is not None:
                return got
            blk = np.zeros((n_win,) + per_rank_shape, dtype=want)
            for w in range(n_win):
                src, dst, empty = [], [], False
                for d in range(len(slab_shape)):
                    cnt = min(int(per_rank_shape[d]),
                              int(valid_t[w, d]) - int(starts[d]))
                    if cnt <= 0:
                        empty = True
                        break
                    o = int(offsets_t[w, d]) + int(starts[d])
                    dst.append(slice(0, cnt))
                    src.append(slice(o, o + cnt))
                if not empty:
                    blk[w][tuple(dst)] = np.asarray(ds[tuple(src)], dtype=want)
            packed = np.moveaxis(blk, 0, window_axis)
            cache[starts] = packed
            return packed

        with _journal.op_scope("read", self.path, stack=_J_H5PY, ds=name,
                               off=f"nwin{n_win}", cnt=slab_shape,
                               mode=self.mode):
            return jax.make_array_from_callback(out_shape, out_sharding,
                                                _block)

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Close the handle, then land the deferred metadata.

        SAME ORDER AS THE FFI TIER, for a weaker reason that is still a
        reason: there, the reopen must wait for ``H5Fclose`` because
        collective MPI-IO holds the file; here the two opens are the same
        library instance and could be one.  They are kept separate anyway so
        both tiers publish deferred metadata at exactly the same moment in
        the lifecycle — a caller must not be able to tell which tier wrote
        its file by when ``omega_ev`` appeared.
        """
        from .hdf5_owner import STACK_H5PY, note_close, open_scope
        with _journal.op_scope("close", self.path, stack=_J_H5PY,
                               mode=self.mode):
            self._h5.close()
        if self._owner_token is not None:
            note_close(self.path, self._owner_token)
            self._owner_token = None
        if self._deferred_attrs or self._deferred_ds_attrs:
            import h5py

            with open_scope(self.path, STACK_H5PY, "a",
                            where="_SerialBackend.close deferred-attr reopen"), \
                    _journal.op_scope(
                        "attr_w", self.path, stack=_J_H5PY, mode="a",
                        cnt=(len(self._deferred_attrs),
                             len(self._deferred_ds_attrs))), \
                    h5py.File(self.path, "a") as h5:
                for name, value in self._deferred_attrs:
                    if name in h5:
                        del h5[name]
                    host = value
                    if not isinstance(host, np.ndarray):
                        host = np.asarray(jax.device_get(host))
                    h5.create_dataset(name, data=host)
                _apply_dataset_attrs(h5, self._deferred_ds_attrs)
                h5.flush()
        self._deferred_attrs = []
        self._deferred_ds_attrs = []

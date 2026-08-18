"""SlabIO — sharded-slab HDF5 I/O.  One transport, no choices.

Every rank writes its own hyperslab and reads its own hyperslab, through
collective MPI-IO in the phdf5 FFI (``file_io._slab_io_ffi``).  Measured
2.919 GiB/s on a cold disjoint-node read at 16 ranks on Lustre
(CLAIMS 69).  There is no other path, and that is the point of this
module: nothing larger than one rank's tile is ever materialised, so the
guarantee holds by construction rather than by a check.

    with SlabIO(path, mode="w", mesh=mesh) as io:
        io.create_dataset("A", shape=(n_q, n_mu, n_G), dtype=c128)
        io.write_slab("A", A)
        B = io.read_slab("A", partition_spec=P(None, "x", "y"))

That is the whole interface a caller needs.  A caller does NOT choose a
backend, does not pass striping or ROMIO hints, does not assert anything
about the MPI world, and does not compute a mesh-divisible extent — those
are properties of the DEPLOYMENT, not of the call site, and they are
settled here and in ``_slab_io_ffi`` once.

WHAT HAPPENED TO THE OTHER TIERS (2026-08-06, owner ruling)
-----------------------------------------------------------
There used to be three, selected by a ``slab_io`` deck key through an
``auto`` router:

* ``H5PY_ALLGATHER`` gathered the whole global array onto rank 0 and
  wrote it with serial h5py.  It was refused above one process at SEVEN
  separate doors, each closure landed and reported as complete, and an
  eighth door kept being found.  A tier that must be refused at seven
  doors is not a tier; it is dead code wearing a safety label.  DELETED
  — the module, the enum member, the deck value, and all seven refusals.
  The doctrine it kept violating ("there should always exist a valid path
  that does not materialize N_mu^2 on any proc because it is a guaranteed
  OOM") is now enforced by there being nothing else to select.
* ``PHDF5_HOST`` drove the same collective MPI-IO from Python via
  mpi4py + h5py(parallel).  Its ONLY selection condition was a host FFI
  library built before workstream AE, which exports the phdf5 READ
  symbols and not ``PhdfWriteHostFfi``.  The deployed host library
  exports it (measured 2026-08-06, ``nm -D
  liblorrax_ffi_host.so`` → ``PhdfWriteHostFfi``), so the condition is
  false on the live stack; and the tier needs an extra two-package
  environment overlay that the FFI path does not.  A tier that requires
  MORE to do the SAME thing, selected only by a stale artifact, is not a
  fallback — the correct response to a stale ``.so`` is a refusal naming
  it, which is the repo-wide contract (CLAIMS 81).  DELETED.

So there is no ``backend=`` argument, no ``use_ffi_io=`` argument, no
``SlabIOBackend`` enum, and no ``slab_io`` deck key.  A deployment that
cannot serve the tile path REFUSES at open, naming the probe that
declined and the repair for that probe — :func:`assert_available` is
that refusal, and :meth:`SlabIO.__init__` CALLS IT FOR YOU.  It is
re-exported here so a caller can pre-flight a deployment before building
a mesh, and so the two refusal messages elsewhere in the tree have a
name to point at; nothing has to call it.

WHO CLOSES WHAT, AND WHAT MAY BE OPEN AT THE SAME TIME
------------------------------------------------------
A ``SlabIO`` owns exactly one collective HDF5 handle.  :meth:`SlabIO.
close` closes it and nothing else does; the caller never touches the
underlying ``PhdfCtx``.  Use ``with``.

The harder half is what ELSE may be open.  This process maps TWO
independent HDF5 library instances — h5py's bundled libhdf5 and the
FFI's cray parallel one — each with its own metadata cache and open-file
table.  A file held by both AT ONCE, with either side able to write, is
undefined behaviour, and it is REFUSED BY NAME at :meth:`SlabIO.
__init__` (:mod:`file_io.hdf5_owner`; audit A1, sandbox claims/0110).
Cross-stack READ-ONLY concurrency is allowed and counted.  So: close
your h5py handle on a path before opening SlabIO on it, not after — the
ordering ``close`` itself uses for its rank-0 deferred-attr reopen.
``docs/architecture/slab_io.md#one-owner`` is the whole rule.

EVERY OPERATION IS JOURNALED.  One line per op, per rank, written
BEFORE the call, to ``h5_journal.rank<R>.log``
(:mod:`file_io.h5_journal`; ``LORRAX_H5_JOURNAL=0`` turns it off).  The
three measured failure signatures on this transport are native deaths or
native refusals, so the journal is the only instrument that survives
them — ``docs/architecture/slab_io.md#journal`` says how to read one.

PADDING is SlabIO's business, not the caller's (decisions.md
2026-08-04).  A caller states LOGICAL shapes:

* ``write_slab(name, A, offset=...)`` accepts any ``A``.  The extent that
  reaches the file is ``min(A.shape, dataset - offset)`` per dim, derived
  from the dataset SlabIO already knows about — so a buffer padded for
  mesh divisibility needs no argument at all.
* ``read_slab(name, partition_spec=spec)`` with NO ``shape`` returns the
  dataset rounded UP to the mesh-divisible extent, zero-filled past the
  dataset.  That is the padded consumer buffer, and asking for it is now
  the easy call as well as the correct one — see :meth:`SlabIO.read_slab`.
* ``valid_shape`` survives ONLY as the ragged-chunk override.
* ``global_shape`` is only needed to CREATE a dataset inside
  ``write_slab``.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import jax

from . import h5_journal as _journal
from ._slab_io_ffi import (_FfiBackend, assert_available, mesh_divisible_shape,
                           probe_availability, probe_read_availability)

__all__ = ["SlabIO", "assert_available", "mesh_divisible_shape",
           "probe_availability", "probe_read_availability"]

#: Every op through this class is one line in the per-rank HDF5 journal
#: (``file_io.h5_journal``; ``LORRAX_H5_JOURNAL=0`` turns it off).  The
#: line is written BEFORE the call, so a rank that dies inside HDF5
#: leaves a file whose last line names the op that killed it — which is
#: the whole instrument, given that all three failure signatures in
#: ``SLAB_IO_ROOT_CAUSE_AUDIT.md`` are native deaths.
_STACK = "ffi"


class SlabIO:
    """Sharded-slab HDF5 file handle.  One transport; see module docstring.

    Parameters
    ----------
    path : str | os.PathLike
        HDF5 file path on a shared filesystem.  ``Path`` is accepted; the
        FFI bindings ``.encode()`` the string, so SlabIO does the
        ``str()`` once rather than making every caller remember it.
    mode : {"w", "a", "r"}
        HDF5 open mode.
    mesh : jax.sharding.Mesh
        The device mesh this file's slabs are sharded over.  Required:
        every read and write is collective over it.

    COLLECTIVE.  Constructing a ``SlabIO`` opens the file collectively
    over ``mesh``; every rank must construct it, in the same order, with
    the same path and mode.  Three things happen before any byte moves,
    and each has its own refusal:

    * :func:`assert_available` — a deployment that cannot serve the tile
      path refuses HERE, naming the probe that declined.  It is called
      for you; a caller invoking it directly is doing a pre-flight, not
      meeting a requirement.
    * ``mode="w"`` REPLACES the inode (rank-0 unlink + barrier).  A
      Lustre layout is fixed at inode create, so ``H5Fcreate(TRUNC)``
      over an old file would silently keep its stripe count.  ``"a"`` and
      ``"r"`` keep the existing inode and its layout, by design.
    * the open is declared to :mod:`file_io.hdf5_owner`, which REFUSES BY
      NAME if h5py already holds a live handle on this path and either
      side can write.  One HDF5 library instance per open file — audit
      A1, ``docs/architecture/slab_io.md#one-owner``.  So a caller must
      close its h5py handle on this path before constructing this, not
      after.

    The handle is closed by :meth:`close` (which ``__exit__`` calls) and
    by nothing else.  The caller never owns, closes or reuses the
    underlying ``PhdfCtx``.
    """

    def __init__(self, path, *, mode: str = "w", mesh=None) -> None:
        self.path = str(path) if not isinstance(path, str) else path
        self.mode = mode
        if mesh is None:
            raise ValueError(
                "SlabIO(mesh=...) is required: every slab read and write is "
                "collective over the device mesh the slabs are sharded on.  "
                "Pass the run's mesh (drivers hold it as `mesh_xy`).")
        self.mesh = mesh
        # The COMPLETION line for this open: it is the only one that can
        # carry the ctx handle, because the handle does not exist until
        # ``_FfiBackend`` returns.  The issue-time line is written inside
        # ``_slab_io_ffi`` around ``open_file`` itself, and the registry
        # writes a third naming the ownership verdict the open walked
        # into.  Three facts, three lines, one per choke point.
        try:
            self._backend = _FfiBackend(self.path, mesh=mesh, mode=mode)
        except BaseException as exc:
            _journal.fail("open", self.path, exc, stack=_STACK, mode=mode)
            raise
        _journal.record("open", self.path, stack=_STACK, mode=mode,
                        handle=self._handle())

    def _handle(self):
        """The live ctx handle, or ``None`` once closed / before open."""
        return getattr(getattr(self, "_backend", None), "fh", None) or None

    # ------------------------------------------------------------------
    def __enter__(self) -> "SlabIO":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        """Drain, close collectively, then stamp the deferred metadata.

        COLLECTIVE, and it is four steps rather than one — which is why
        it is the method that shows up in crash reports:

        1. drain THIS rank's queued writes.  The FFI writer is
           asynchronous, so this is the first moment any byte is on disk,
           and the drain can take MINUTES for a multi-GB file.  It blocks
           on collective MPI-IO, so every rank must reach it.
        2. join the writer thread;
        3. ``H5Fclose``, collectively;
        4. release the :mod:`file_io.hdf5_owner` claim — then RANK 0
           reopens the file with serial h5py to write the deferred
           :meth:`write_attr` datasets and :meth:`stamp_dataset_attrs`
           attributes, flushes, and every rank meets an unconditional
           barrier after it.  That reopen is the other HDF5 library
           instance touching this path, and it is legal only because
           step 3 already let go.

        A worker-thread exception does NOT skip the teardown.  It is
        recorded, the collective completes on every rank, and it is
        re-raised at the end (decisions.md 2026-08-04): a rank that
        raised out of the middle would leave its peers inside
        ``H5Fclose`` with no message.

        ``__exit__`` calls this, so ``with SlabIO(...) as io:`` is the
        normal spelling; an explicit call is for a handle whose lifetime
        does not match a block.  This is the ONLY thing that closes the
        underlying ``PhdfCtx``.

        WHERE FAILURE SIGNATURE S1 SURFACES
        (``docs/architecture/slab_io.md#s1``).  A raced or stale ctx
        handle reaches ``close_ctx`` as a thread join and a mutex destroy
        at the wrong offsets, so the corruption is DETECTED here and
        CREATED somewhere earlier — do not start reading at this method.
        """
        with _journal.op_scope("close", self.path, stack=_STACK,
                               mode=self.mode, handle=self._handle()):
            self._backend.close()

    # ------------------------------------------------------------------
    def create_dataset(
        self,
        name: str,
        *,
        shape: Sequence[int],
        dtype,
        attrs: dict | None = None,
    ) -> None:
        """Pre-create a dataset with the given LOGICAL shape and dtype.

        COLLECTIVE over the constructor's mesh: every rank calls it, in
        the same order, with the same ``name`` and ``shape``.  It drains
        pending writes first, because ``phdf5_ensure_dataset`` is itself
        an MPI rendezvous on the file handle.

        ``shape`` is what the file stores, so it can be re-read on any
        process count.  Everything a subsequent ``write_slab`` needs to
        know about padding is derived from it.

        On an existing dataset: identical shape and dtype reuse it
        (idempotent); anything else REFUSES, naming both shapes, on
        every rank.  SlabIO never deletes-and-recreates (data loss) and
        never writes into the previous geometry (wrong extent, no
        symptom) — decisions.md 2026-08-04.

        ``attrs`` ARE WRITTEN, as H5 attributes on this dataset, by rank
        0 when the file closes — the transport cannot stamp them while
        collective MPI-IO holds the file, so they land beside the
        deferred :meth:`write_attr` metadata in the one reopen.  They are
        therefore visible to a reader after ``close()`` and not before,
        which is the only moment any reader has ever looked.  They used
        to be DISCARDED with a warning, which is how a cluster-written
        wedge Σ cube lost its ``k_storage`` stamp and read back as
        full-BZ; see ``_slab_io_ffi._apply_dataset_attrs``.

        Datasets created by this transport are contiguous.  The native
        collective create has no chunk-layout argument, so SlabIO does not
        expose a no-op ``chunks=`` compatibility parameter.
        """
        with _journal.op_scope("create", self.path, stack=_STACK, ds=name,
                               cnt=tuple(int(s) for s in shape),
                               mode=self.mode, handle=self._handle()):
            self._backend.create_dataset(
                name, shape=shape, dtype=dtype, attrs=attrs)

    def write_attr(self, name: str, value) -> None:
        """QUEUE a small replicated dataset (e.g. ``omega_ev``) for close.

        IT DOES NOT WRITE.  The value is held and written by RANK 0 in
        the one serial-h5py reopen :meth:`close` performs after
        ``H5Fclose`` — the transport cannot stamp metadata while
        collective MPI-IO holds the file.  So the dataset does not exist
        until ``close()`` RETURNS, which is the only moment any reader
        has ever looked, and a device-resident ``value`` is
        ``device_get``'d then, not now.

        Two consequences, stated because neither is checkable here:

        * an existing dataset of this name is DELETED and recreated at
          close, so a queued stamp replaces whatever else wrote ``name``;
        * every rank queues and only RANK 0's copy lands.  Call sites are
          SPMD today so the lists agree — but that is a property of the
          callers, not of this method, and a rank-dependent value passed
          here is silently discarded on every rank but one.

        For scalars and small metadata arrays that are replicated or
        already on host.  Bulk data goes through :meth:`write_slab`.
        """
        with _journal.op_scope("attr_w", self.path, stack=_STACK, ds=name,
                               mode=self.mode, handle=self._handle()):
            self._backend.write_attr(name, value)

    def stamp_dataset_attrs(self, name: str, attrs: dict) -> None:
        """Stamp H5 attributes onto a dataset this handle did not create.

        Rides the SAME deferral as ``create_dataset(attrs=...)`` — the one
        rank-0 h5py reopen after ``H5Fclose`` — so a stamp costs no extra
        open of the file and, in particular, no serial-h5py open while the
        collective handle is live (``file_io.hdf5_owner``, audit A1).

        For metadata ABOUT a dataset that ``write_attr`` wrote: those land
        in the same reopen, before this, so ``omega_ev`` exists by the
        time its attrs are applied.  A name absent at that moment raises,
        exactly as it does for ``create_dataset``.
        """
        _journal.record("attr_w", self.path, stack=_STACK, ds=name,
                        cnt=len(attrs), mode=self.mode,
                        handle=self._handle())
        self._backend._deferred_ds_attrs.append((str(name), dict(attrs)))

    def sync_writes(self) -> None:
        """Wait for queued writes without closing the collective handle.

        Use this only when the next operation enters another HDF5 handle on
        the same ranks.  The FFI writer is asynchronous, so program order at
        the Python call site alone does not serialize those two HDF5 calls.

        EFFECTIVELY COLLECTIVE.  The wait itself is local — it joins this
        rank's queue — but what is queued is collective MPI-IO, so a rank
        that calls this while its peers do not blocks until they enter the
        same writes.  Call it on every rank or on none.
        """
        self._backend._drain_pending()

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
        """Write A as a hyperslab of dataset ``name``.

        COLLECTIVE over the constructor's mesh, and ASYNCHRONOUS: the
        call ENQUEUES the write and returns.  The bytes are on disk after
        :meth:`sync_writes` or :meth:`close`, not after this returns, so
        timing this call times the enqueue and the rate it implies is a
        fiction of the queue depth.

        ``A`` is an N-D ``jax.Array`` (possibly sharded) or numpy array;
        **its sharding is read from ``A.sharding``** — this method does
        not reshard, gather or replicate, and each rank contributes
        exactly the tile it already holds.  A replicated axis is
        de-duplicated to one canonical writer
        (``LORRAX_PHDF5_DEDUP_REPLICAS``), because overlapping hyperslab
        selections are undefined under collective MPI-IO.

        ``offset`` (default all zeros) is where the slab starts in the
        dataset.  The extent written is ``min(A.shape, dataset - offset)``
        per dim — pad rows past the dataset's extent are dropped, with no
        argument from the caller.

        ``valid_shape`` is the ragged-chunk OVERRIDE and nothing else:
        pass it only when ``A``'s trailing rows are not part of THIS
        write even though the dataset has room for them (a chunk buffer
        whose last chunk is short).  An override that overruns the
        dataset refuses, on every rank.  ``global_shape`` is likewise
        only needed when ``write_slab`` has to create the dataset.
        """
        with _journal.op_scope("write", self.path, stack=_STACK, ds=name,
                               off=offset, cnt=tuple(A.shape) if hasattr(
                                   A, "shape") else None,
                               mode=self.mode, handle=self._handle()):
            self._backend.write_slab(
                name, A,
                offset=offset, global_shape=global_shape,
                valid_shape=valid_shape,
                dtype=dtype,
            )

    # ------------------------------------------------------------------
    def read_slab(
        self,
        name: str,
        *,
        shape: Sequence[int] | None = None,
        dtype=None,
        offset: Sequence[int] | None = None,
        valid_shape: Sequence[int] | None = None,
        mesh=None,
        partition_spec=None,
        as_numpy: bool = False,
    ) -> jax.Array:
        """Read a hyperslab, sharded ``partition_spec`` on ``mesh``.

        COLLECTIVE over ``mesh`` (the constructor's, unless overridden),
        and SYNCHRONOUS — unlike :meth:`write_slab`, the array is
        readable when this returns.  Each rank reads its own hyperslab
        and no one else's; nothing larger than one rank's tile is
        materialised.  The return is a ``jax.Array`` sharded
        ``partition_spec`` on ``mesh``, or a host ``np.ndarray`` under
        ``as_numpy=True``.

        Reading a dataset THIS handle did not create costs one serial
        h5py introspect of the file (cached per name) to learn its shape
        and dtype.  That is a legal cross-stack read-only touch and the
        registry counts it; it is the reason a read on a handle opened
        for writing is a different risk from a read on one opened ``"r"``.

        THE EASY CALL IS THE CORRECT CALL.  Omit ``shape`` and SlabIO
        returns the dataset rounded UP to the mesh-divisible extent under
        ``partition_spec``, zero-filled past the dataset — which is
        exactly the padded consumer buffer every sharded consumer wants.
        A caller no longer has to know ``runtime.padding.padded_mu_extent``,
        the device count, or which axes ``partition_spec`` shards to ask
        for a legal shape.

        This used to REFUSE.  ``shape=None`` defaulted to the dataset's
        own shape, and a dataset whose μ extent is not divisible by the
        mesh (the normal case — N_mu is a physics number, not a multiple
        of the device count) then hit ``_validate_block_divisible``'s
        ValueError.  So every caller computed the rounded-up extent
        itself, which is the "nitty-gritty" the owner objected to, and
        two of them computed it differently.

        ``shape`` may still be given EXACTLY, and is returned exactly.  It
        must be mesh-divisible under ``partition_spec``: the return value
        is a ``jax.Array`` of that shape sharded that way, and JAX will
        not build one at a non-divisible extent, so there is nothing to
        trim TO.  It may exceed the dataset; the overhang comes back
        zero-filled.

        ``valid_shape`` is the ragged-chunk override; see
        :meth:`write_slab`.  Routine reads do not pass it.

        ``as_numpy=True`` forces a host ``np.ndarray`` via ``device_get``.
        """
        mesh = mesh or self.mesh
        if shape is None and partition_spec is not None:
            shape = self._backend.padded_shape_for(
                name, mesh=mesh, partition_spec=partition_spec)
        with _journal.op_scope("read", self.path, stack=_STACK, ds=name,
                               off=offset,
                               cnt=None if shape is None else tuple(
                                   int(s) for s in shape),
                               mode=self.mode, handle=self._handle()):
            arr = self._backend.read_slab(
                name, shape=shape, dtype=dtype, offset=offset,
                valid_shape=valid_shape,
                mesh=mesh, partition_spec=partition_spec)
        if as_numpy and not isinstance(arr, np.ndarray):
            arr = np.asarray(jax.device_get(arr))
        return arr

    # ------------------------------------------------------------------
    def read_slabs(
        self,
        name: str,
        *,
        shape: Sequence[int],
        offsets,
        valid_shapes,
        partition_spec,
        window_axis: int,
        dtype=None,
        mesh=None,
    ) -> jax.Array:
        """Read n windows of ONE slab ``shape`` in ONE collective H5Dread.

        COLLECTIVE over ``mesh``, synchronous, same tile guarantee as
        :meth:`read_slab`.

        ITS PRODUCTION CONSUMER IS ``services/wfn_loader`` (the k-chunk
        union read), which lives outside ``src/`` — so a census that
        greps only ``src/`` reports this method as having no caller.  It
        has one; see ``docs/services/wfn_loader.md``.

        The windows share a shape and differ only in where they start and
        how much of that shape is real:

        * ``shape`` — the common (mesh-padded, global) slab shape, exactly
          as :meth:`read_slab` takes it;
        * ``offsets`` — ``(n, ndim)``, each window's origin in the dataset;
        * ``valid_shapes`` — ``(n, ndim)``, each window's LOGICAL extent
          from its own origin.  Same meaning as :meth:`read_slab`'s
          ``valid_shape`` and the same consequence: past it the output is
          zero, per window, with no caller-side arithmetic.  It is not
          optional here because a per-window extent is not derivable from
          the dataset — the raggedness is the request.

        The result is a ``jax.Array`` of ``shape`` with an n-long window
        axis inserted at ``window_axis``, sharded by ``partition_spec``
        with ``None`` inserted at the same position.  ``window_axis`` must
        sit immediately before the dim that VARIES across windows, so the
        packed output iterates in the file's own order.

        PRECONDITION the caller owns: the windows must be pairwise
        DISJOINT in the file and sorted ascending in row-major file order.
        They are selected into one ``H5S_SELECT_OR`` compound hyperslab, so
        an overlap is a double-selection and a wrong order permutes the
        packing.  A caller whose windows are not sorted argsorts the tables
        and permutes the window axis back — both cheap.

        WHY THIS IS A DOOR PRIMITIVE and not n :meth:`read_slab` calls
        (MEASURED 2026-08-07, CPU milan, 2x2 mesh, both arms through the
        same handle; artifacts ``_measure_fold/``).  The fold-down loses on
        every deck measured: warm-min ratios 3.58x / 3.22x / 1.44x against
        this call, and at the production deck (144 windows, 15.6 GB) it
        costs +2.1 s per read, of which ~1.4 s is per-call collective
        H5Dread overhead and ~0.6 s an extra stack this path never needs.
        The cost axis is n — the number of windows — which is exactly what
        grows with the deck.  So n windows is a REQUEST SlabIO serves, not
        a loop a caller writes.
        """
        # ``off`` is the WINDOW COUNT here, not an origin: n windows share
        # one call and their origins are a table, which does not belong on
        # a log line.  The origins are the caller's ``offsets`` argument.
        with _journal.op_scope("read", self.path, stack=_STACK, ds=name,
                               off=f"nwin{len(offsets)}",
                               cnt=tuple(int(s) for s in shape),
                               mode=self.mode, handle=self._handle()):
            return self._backend.read_slabs(
                name, shape=shape, offsets=offsets, valid_shapes=valid_shapes,
                partition_spec=partition_spec, window_axis=window_axis,
                dtype=dtype, mesh=mesh or self.mesh)

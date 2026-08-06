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
declined and the repair for that probe — see :func:`assert_available`.

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

from ._slab_io_ffi import (_FfiBackend, assert_available, mesh_divisible_shape,
                           probe_availability)

__all__ = ["SlabIO", "assert_available", "mesh_divisible_shape",
           "probe_availability"]


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
        self._backend = _FfiBackend(self.path, mesh=mesh, mode=mode)

    # ------------------------------------------------------------------
    def __enter__(self) -> "SlabIO":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self._backend.close()

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
        """Pre-create a dataset with the given LOGICAL shape + dtype + chunks.

        ``shape`` is what the file stores, so it can be re-read on any
        process count.  Everything a subsequent ``write_slab`` needs to
        know about padding is derived from it.

        On an existing dataset: identical shape and dtype reuse it
        (idempotent); anything else REFUSES, naming both shapes, on
        every rank.  SlabIO never deletes-and-recreates (data loss) and
        never writes into the previous geometry (wrong extent, no
        symptom) — decisions.md 2026-08-04.
        """
        self._backend.create_dataset(
            name, shape=shape, dtype=dtype, chunks=chunks, attrs=attrs)

    def write_attr(self, name: str, value) -> None:
        """Write a small rank-0-only dataset (e.g. omega_ev).

        For scalars / small metadata arrays that are replicated or
        already on host.
        """
        self._backend.write_attr(name, value)

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
    ) -> None:
        """Write A as a hyperslab of dataset ``name``.

        ``A`` is an N-D ``jax.Array`` (possibly sharded) or numpy array;
        its sharding is read from ``A.sharding``.

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
        self._backend.write_slab(
            name, A,
            offset=offset, global_shape=global_shape,
            valid_shape=valid_shape,
            dtype=dtype, chunks=chunks,
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
        arr = self._backend.read_slab(
            name, shape=shape, dtype=dtype, offset=offset,
            valid_shape=valid_shape,
            mesh=mesh, partition_spec=partition_spec)
        if as_numpy and not isinstance(arr, np.ndarray):
            arr = np.asarray(jax.device_get(arr))
        return arr

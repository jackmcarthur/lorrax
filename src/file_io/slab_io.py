"""SlabIO — unified sharded-slab HDF5 I/O with pluggable backend.

One helper for all large-sharded-array writes/reads: replaces the
ad-hoc ``process_allgather`` → rank-0 ``h5py`` patterns sprinkled
across the codebase.

Two backends, selected by ``backend: SlabIOBackend``:

- :attr:`SlabIOBackend.H5PY_ALLGATHER` — :mod:`file_io._slab_io_allgather`.
  Gather to rank 0 via ``jax.experimental.multihost_utils.process_allgather``,
  write with plain serial ``h5py``.  **Reachable at exactly one process**
  (owner ruling 2026-08-05): at P=1 the gather and the per-rank write are
  the same operation, above P=1 rank 0 would have to hold the whole global
  array, which the production envelope makes an OOM rather than a slow
  path.  Named explicitly it is honoured at P=1 and refused above; left
  unstated it is refused above P=1 by :func:`_normalize_slab_backend`.
- :attr:`SlabIOBackend.PHDF5_FFI` — :mod:`file_io._slab_io_ffi`.
  Collective MPI-IO via ``ffi.phdf5``; each rank writes its own
  hyperslab directly.  Lazy import; only loads the FFI when selected.
  Works on BOTH backends: the C++ core compiles into the CUDA lib and
  into the CUDA-free host lib (``LORRAX_FFI_NO_CUDA`` — the D2H staging
  becomes an in-place read of the XLA host buffer).  Preferred on CPU
  whenever the deployed host lib exports ``PhdfWriteHostFfi``.
- :attr:`SlabIOBackend.PHDF5_HOST` — :mod:`file_io._slab_io_mpi_host`.
  Same per-rank collective MPI-IO, driven from Python by mpi4py +
  h5py(parallel) instead of the FFI.  Fallback tier on CPU for a host
  lib built without the write handler; needs the mpi4py overlay.

The legacy ``use_ffi_io: bool`` kwarg is still accepted on every entry
point and silently coerced via :func:`_normalize_slab_backend` — pass
the enum in new code.

One primitive:

- ``SlabIO`` — context manager for files that see multiple writes
  (the isdf_fitting zeta loop, ppm_sigma stream).  Opens once,
  creates/ensures datasets, writes/reads, closes.

Padding is SlabIO's business, not the caller's (decisions.md
2026-08-04).  A caller states LOGICAL shapes and nothing else:

    io.create_dataset("A", shape=(n_q, n_mu, n_G), dtype=c128)
    io.write_slab("A", A_padded)                       # pad rows dropped
    B = io.read_slab("A", shape=(n_q, n_mu_pad, n_G),  # pad rows zeroed
                     dtype=c128, mesh=mesh, partition_spec=spec)

* ``write_slab(name, A, offset=...)`` accepts any ``A``.  The extent
  that reaches the file is ``min(A.shape, dataset - offset)`` per dim,
  derived from the dataset SlabIO already knows about — so a buffer
  padded for mesh divisibility needs no argument at all.
* ``read_slab(name, shape=...)`` returns exactly ``shape``.  Whatever
  padding the backend needs (mesh divisibility) is read and trimmed
  internally; positions past the dataset come back zero-filled.
* ``valid_shape`` survives ONLY as the ragged-chunk override: a chunk
  buffer whose tail is genuinely not part of this write, which SlabIO
  cannot derive because both extents are legitimate.  An override that
  overruns the dataset is a refusal, on every rank.
* ``global_shape`` is only needed to CREATE a dataset inside
  ``write_slab``.  Once the dataset exists it is redundant, and a
  ``global_shape`` that contradicts the dataset is a refusal.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import jax

from ._slab_io_allgather import _AllgatherBackend

__all__ = ["SlabIO"]


def _normalize_slab_backend(backend, use_ffi_io):
    """Resolve ``(backend, use_ffi_io)`` to a :class:`SlabIOBackend`.

    Accepts:
    - ``backend=SlabIOBackend.PHDF5_FFI | H5PY_ALLGATHER`` (preferred)
    - ``use_ffi_io=True | False`` (legacy boolean — coerced)
    - both ``None``: allgather at one process, a REFUSAL above one

    The legacy boolean is the only place strings/bools cross over into
    the enum world for SlabIO; everywhere else stays in enum land.

    THE DEFAULT IS GATED BY PROCESS COUNT (2026-08-06).  ``0d8e50c``
    turned the allgather demotion into a refusal in the config routers
    and recorded, in its own message, that this function was left
    pointing the same wrong way: ``SlabIO(path, mesh=mesh)`` with no
    backend named lands on ``H5PY_ALLGATHER``, which gathers the whole
    global array onto rank 0, and it does so BELOW the parse-time gate
    that refuses exactly that.  No in-tree caller took the path — every
    one of them passes ``backend=`` — so it was latent rather than live,
    but "no caller does it today" is not a property a default can rely
    on, and the failure it produces is an OOM minutes later with rc=0
    behind it.  The refusal is raised HERE, at construction, where the
    file is about to be opened, so a direct caller hits it in the same
    place the config path does.

    A named ``backend=H5PY_ALLGATHER`` is NOT re-refused here: that is
    ``from_input_file``'s ``_refuse_explicit_h5py_allgather``, and the
    tier legitimately serves single-process tests and fixtures which
    construct it by name.  What is refused is the *unstated* default.
    """
    from gw.gw_config import SlabIOBackend  # avoid circular import at module load
    if backend is not None:
        if isinstance(backend, SlabIOBackend):
            return backend
        raise TypeError(
            f"backend={backend!r} must be SlabIOBackend, "
            f"not {type(backend).__name__}"
        )
    if use_ffi_io is None:
        return _refuse_or_allgather("backend=None, use_ffi_io=None")
    # The legacy ``use_ffi_io=True`` boolean predates the host-side
    # PHDF5 backend; on CPU runs the auto-router in
    # ``LorraxConfig.from_input_file`` has already resolved this to a
    # concrete backend enum.  Anyone still calling SlabIO directly with
    # ``use_ffi_io=True`` (a few tests + ad-hoc scripts) intends the
    # GPU FFI; preserve that mapping here.
    if bool(use_ffi_io):
        return SlabIOBackend.PHDF5_FFI
    return _refuse_or_allgather("use_ffi_io=False")


def _refuse_or_allgather(how: str):
    """H5PY_ALLGATHER at one process; raise above one.  ``how`` names the
    argument shape that got here, because the repair differs by caller.

    Process count comes from ``gw_config._slab_io_process_count`` — the
    SAME function the parse-time refusals use, deliberately imported
    rather than re-derived.  It takes the max of ``jax.process_count()``
    and the launcher's own world size, because a caller that constructs
    SlabIO before ``jax.distributed.initialize`` would otherwise see 1 on
    every rank of a 16-rank job, which is the exact reading that lets the
    gather through.
    """
    from gw.gw_config import (SlabIOBackend, _slab_io_process_count,
                              _SLAB_IO_DOC, _SLAB_IO_RULE)
    processes, evidence = _slab_io_process_count()
    if processes <= 1:
        return SlabIOBackend.H5PY_ALLGATHER
    raise ValueError(
        f"\n*** SlabIO() with no backend named REFUSED at {processes} "
        f"processes. ***\n"
        f"  rule    {_SLAB_IO_RULE}\n"
        f"  got     SlabIO(...) constructed with {how}, at {processes} "
        f"process(es) ({evidence}).  The unstated default is "
        f"H5PY_ALLGATHER.\n"
        f"  wanted  a backend that writes one tile per rank: "
        f"SlabIOBackend.PHDF5_FFI (GPU or CPU, needs the FFI lib) or "
        f"SlabIOBackend.PHDF5_HOST (CPU, needs the mpi4py overlay).\n"
        f"  fix     pass backend=<SlabIOBackend> explicitly.  Callers "
        f"inside the driver chain should take it from the resolved "
        f"config (LorraxConfig.slab_io_backend), which has already "
        f"probed this stack and will name the reason if neither "
        f"parallel tier is available — do not re-derive it here.\n"
        f"  escape  backend=SlabIOBackend.H5PY_ALLGATHER is still "
        f"honoured by name at one process; above one it is refused by "
        f"the config router for the same reason.\n"
        f"  doc     {_SLAB_IO_DOC}\n")


# ---------------------------------------------------------------------------
# Public context manager
# ---------------------------------------------------------------------------
class SlabIO:
    """Unified sharded-slab HDF5 file handle.

    Usage::

        from gw.gw_config import SlabIOBackend
        with SlabIO(path, mode="w", mesh=mesh,
                    backend=SlabIOBackend.PHDF5_FFI) as io:
            io.create_dataset("A", shape=(N, M), dtype=jnp.complex128)
            io.write_slab("A", A_sharded)          # whole dataset
            io.write_slab("A", chunk, offset=(i, 0))  # sub-slab
            B = io.read_slab("A", shape=(N, M), dtype=jnp.complex128)

    Parameters
    ----------
    path : str
        HDF5 file path on a shared filesystem.
    mode : {"w", "a", "r"}
        HDF5 open mode.  Must be ``"w"`` to create, ``"a"`` to
        append, ``"r"`` to read.
    mesh : jax.sharding.Mesh, optional
        Required by the FFI backend; the allgather backend ignores
        it (rank-0 always owns the file).
    backend : SlabIOBackend, optional
        Selects the underlying I/O path.  Omitting it is only legal at
        ONE process (where it means allgather); above one process an
        unstated backend is a refusal, not a default — see
        :func:`_normalize_slab_backend`.  The legacy ``use_ffi_io: bool``
        kwarg is also accepted for back-compat and gated the same way.
    """

    def __init__(
        self,
        path,
        *,
        mode: str = "w",
        mesh=None,
        backend=None,
        use_ffi_io: bool | None = None,
    ) -> None:
        from gw.gw_config import SlabIOBackend
        # Accept Path | str | bytes uniformly.  The FFI backend's C
        # bindings ``.encode()`` the string, which raises
        # ``AttributeError: 'PosixPath' object has no attribute 'encode'``
        # — historically callers had to remember to ``str()`` paths
        # themselves; now SlabIO does it once.
        self.path = str(path) if not isinstance(path, str) else path
        self.mode = mode
        self.mesh = mesh
        self.backend = _normalize_slab_backend(backend, use_ffi_io)
        # Boolean shortcut still used internally by ``read_slab`` for
        # branch dispatch — kept as a derived attribute, not the source
        # of truth.  Both PHDF5 variants follow the "per-rank parallel
        # write" semantics that read_slab branches on; the allgather
        # backend is the only one that needs the rank-0-gather code path.
        self.use_ffi_io = (self.backend in (
            SlabIOBackend.PHDF5_FFI, SlabIOBackend.PHDF5_HOST))
        if self.backend is SlabIOBackend.PHDF5_FFI:
            if mesh is None:
                raise ValueError(
                    "backend=SlabIOBackend.PHDF5_FFI requires mesh")
            from ._slab_io_ffi import _FfiBackend
            self._backend = _FfiBackend(self.path, mesh=mesh, mode=mode)
        elif self.backend is SlabIOBackend.PHDF5_HOST:
            if mesh is None:
                raise ValueError(
                    "backend=SlabIOBackend.PHDF5_HOST requires mesh")
            from ._slab_io_mpi_host import _MpiHostBackend
            self._backend = _MpiHostBackend(self.path, mesh=mesh, mode=mode)
        else:
            self._backend = _AllgatherBackend(self.path, mode=mode)

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

        Skips the allgather; good for scalars / small metadata arrays
        that are replicated or already on host.
        """
        self._backend.write_attr(name, value)

    # Padding contract for sharded producers: hand over the array you
    # have.  SlabIO clips it to the dataset's own extent, so a buffer
    # padded for mesh divisibility writes its logical prefix and nothing
    # else, with no argument from the caller.
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
        """Write A as a hyperslab of dataset ``name``.

        ``A`` is an N-D ``jax.Array`` (possibly sharded) or numpy
        array.  Its sharding is inferred from ``A.sharding`` on the
        FFI path; ignored on the allgather path (which gathers the
        whole thing to rank 0).

        ``offset`` (default all zeros) is where the slab starts in the
        dataset.  The extent written is ``min(A.shape, dataset - offset)``
        per dim — pad rows past the dataset's extent are dropped, with
        no argument from the caller.

        ``valid_shape`` is the ragged-chunk OVERRIDE and nothing else:
        pass it only when ``A``'s trailing rows are not part of THIS
        write even though the dataset has room for them (a chunk buffer
        whose last chunk is short).  An override that overruns the
        dataset refuses, on every rank.  ``global_shape`` is likewise
        only needed when ``write_slab`` has to create the dataset.

        ``k_chunk_size`` is an allgather-backend-only knob that
        streams the rank-0 write along axis 1 to keep memory bounded
        for large-omega writes (matches the legacy sigma_output
        k_chunk pattern).  Ignored by the FFI backend.
        """
        self._backend.write_slab(
            name, A,
            offset=offset, global_shape=global_shape,
            valid_shape=valid_shape,
            dtype=dtype, chunks=chunks, k_chunk_size=k_chunk_size,
        )

    # Padding contract for sharded consumers: ask for the shape you want
    # to consume.  SlabIO fills the part the dataset covers and zeroes
    # the rest, and handles mesh divisibility internally, so a consumer
    # that runs at a padded μ extent just asks for the padded shape.
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
        """Read a hyperslab.

        On the allgather path, returns a replicated host-backed JAX
        array (or a plain ``np.ndarray`` if ``as_numpy=True`` — useful
        for readers that feed host-side numpy stacks straight into a
        GPU compute kernel later, skipping a pointless H2D+D2H round
        trip).  On the FFI path, returns a sharded array with
        ``partition_spec`` on ``mesh`` (``as_numpy`` still forces a
        host ndarray via ``device_get``).

        ``shape`` is returned EXACTLY.  It may exceed the dataset (a
        μ-padded consumer buffer): the overhang comes back zero-filled.
        Defaults to the dataset's own shape.

        ``shape`` MUST be mesh-divisible under ``partition_spec``.  This
        docstring used to promise the opposite — "it need not be
        mesh-divisible either, SlabIO reads the rounded-up extent and
        trims" — and that was never true of any backend (measured, job
        56389339, 4 nodes / 16 ranks: ``read_slab(shape=(17,16),
        partition_spec=P('x','y'))`` on a 4x4 mesh REFUSES on PHDF5_FFI
        with ``_validate_block_divisible``'s ValueError, and on
        H5PY_ALLGATHER with JAX's own ``IndivisibleError``).  It cannot
        be true: the return value is a ``jax.Array`` of exactly
        ``shape`` sharded by ``partition_spec``, and JAX will not build
        that array at a non-divisible extent, so there is nothing to
        trim TO.  Ask for the rounded-up shape — that IS the padded
        consumer buffer, and the overhang comes back zero-filled, which
        is the padding contract stated above.

        ``valid_shape`` is the ragged-chunk override; see
        :meth:`write_slab`.  Routine reads do not pass it.
        """
        if self.use_ffi_io:
            arr = self._backend.read_slab(
                name, shape=shape, dtype=dtype, offset=offset,
                valid_shape=valid_shape,
                mesh=mesh or self.mesh, partition_spec=partition_spec)
        else:
            arr = self._backend.read_slab(
                name, shape=shape, dtype=dtype, offset=offset,
                valid_shape=valid_shape,
                mesh=mesh, as_numpy=as_numpy,
                partition_spec=partition_spec)
        if as_numpy and not isinstance(arr, np.ndarray):
            arr = np.asarray(jax.device_get(arr))
        return arr

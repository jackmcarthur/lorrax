"""FFI SlabIO backend — collective MPI-IO via ``ffi.phdf5``.

FIRST tier of the ``slab_io = auto`` capability router on BOTH JAX
backends (``gw_config._route_cpu_slab_io`` / ``_route_gpu_slab_io``:
PHDF5_FFI → PHDF5_HOST → H5PY_ALLGATHER), also selectable explicitly
via ``slab_io = phdf5_ffi``.  The legacy ``use_ffi_io`` boolean no
longer routes here: ``true`` is a deprecated no-op, ``false`` forces
the allgather backend.  Imported lazily by :mod:`file_io.slab_io` so
the fallback tiers work without ``liblorrax_ffi*.so`` being built.

PLATFORM-AGNOSTIC.  Nothing in this module is CUDA-specific: the
``jax.ffi.ffi_call`` sites name only the target string, and
``ffi_loader`` registers ``liblorrax_ffi.so``'s handlers under
platform="CUDA" and ``liblorrax_ffi_host.so``'s under platform="cpu"
against those same strings.  So this backend drives the GPU collective
write and (since workstream AE) the host one — on CPU the C++ side
skips the D2H staging entirely and H5Dwrite reads the XLA buffer in
place.  ``gw_config._route_cpu_slab_io`` capability-probes for it.

Every operation derives per-rank hyperslab offsets from the sharding
spec of the JAX array being written (or a caller-provided one for
reads) plus a global-origin ``offset`` argument.  The C++ handler
un-ravels the rank id through ``mesh_shape`` and advances along every
sharded dim.  See ``ffi/cpp/phdf5/write_ffi.cc`` for the C++ side.
"""
from __future__ import annotations

import functools
import os
import shutil
import subprocess
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.collectives import barrier as _barrier, device_put_process_local


def _rank0() -> bool:
    return jax.process_index() == 0


# ``_barrier`` is ``common.collectives.barrier``.  It used to be a local
# copy whose whole body was ``try: sync_global_devices(tag); except
# Exception: pass`` — seven lines below an import of the very module that
# owns the correct one.  The swallow is the defect: every use of it here
# guards a WRITE ordering (the inode replacement before the Lustre
# prestripe hints, the prestripe before the collective open, the rank-0
# deferred-attr write before close), so a barrier that silently did
# nothing would let a rank read a file whose stripe layout or metadata
# the writer has not finished changing — a data defect that reports rc=0.
# ``collectives.barrier`` instead returns False for the legitimate
# single-process no-op and RAISES (after naming the barrier) when a real
# multi-process barrier fails.


def _stripe_count() -> int:
    """``LORRAX_PHDF5_STRIPE_COUNT`` (Lustre ``striping_factor``), default 16.

    Unset/empty → default; anything else must be a plain int.  Shared by
    both PHDF5 writers.  A typo here used to crash with a bare
    ``ValueError`` while the sibling ``LORRAX_PHDF5_STRIPE_SIZE_FS`` was
    silently replaced by ITS default — two neighbouring knobs, opposite
    failure modes (audit 2026-07-28).  Both now refuse loudly, naming
    the variable and the accepted grammar (doctrine 3).
    """
    raw = os.environ.get("LORRAX_PHDF5_STRIPE_COUNT", "").strip()
    if not raw:
        return 16
    try:
        return int(raw)
    except ValueError:
        raise ValueError(
            f"LORRAX_PHDF5_STRIPE_COUNT={raw!r} is not a valid stripe "
            f"count: expected a plain integer (e.g. 16; <=0 disables "
            f"the striping hints).") from None


def _replace_inode_for_write(path: str) -> None:
    """Rank-0 unlink + barrier so ``mode='w'`` REPLACES the file's inode.

    Called by BOTH PHDF5 writers (:class:`_FfiBackend` and
    ``_slab_io_mpi_host._MpiHostBackend``) before any rank opens the
    file, UNCONDITIONALLY of ``lfs`` availability.  Rationale: a Lustre
    stripe layout is a property of the INODE, fixed at create time —
    ``lfs setstripe``, MPI-IO's ``striping_factor`` hint and
    ``H5Fcreate(H5F_ACC_TRUNC)`` are all no-ops against an existing
    inode, so a rerun over an existing 1-stripe file keeps 1 stripe
    forever (measured: job 7876423 funnelled 13.3 GB of ``V_qmunu``
    through a single OST).  The only unlink used to live inside
    :func:`_lustre_prestripe` AFTER its lfs-missing early return, i.e.
    it never ran in the production apptainer image (audit 2026-07-28).

    ``os.path.lexists`` (not ``exists``) so a dangling symlink is also
    replaced; a live symlink is announced before removal because the
    new file lands at ``path`` itself, not at the link's old target.
    A failed unlink RAISES rather than falling through: proceeding
    would H5F_ACC_TRUNC the old inode and silently inherit its stripe
    layout — ``mode='w'`` is a replace contract.
    """
    if _rank0() and os.path.lexists(path):
        if os.path.islink(path):
            try:
                target = os.readlink(path)
            except OSError:
                target = "<unreadable>"
            print(f"  [SlabIO] mode='w': {path} is a symlink -> "
                  f"{target!r}; removing the LINK — the new file is "
                  f"created at {path}, not at the old target.",
                  flush=True)
        try:
            os.remove(path)
        except OSError as e:
            raise OSError(
                f"SlabIO mode='w': could not replace existing file "
                f"{path!r}: {e}.  Refusing to open with H5F_ACC_TRUNC "
                f"instead — truncation reuses the inode, so the file "
                f"would silently keep its existing Lustre stripe layout "
                f"and ignore the striping_factor/striping_unit hints "
                f"(the 1-stripe single-OST defect, job 7876423).  "
                f"Delete the file or fix its permissions, then rerun."
            ) from e
    _barrier("slab_io_replace_inode")


def _shard_read_plan(
    index: Sequence[slice],
    out_shape: Sequence[int],
    offset: Sequence[int],
    valid_shape: Sequence[int],
) -> tuple[tuple[int, ...],
           tuple[slice, ...] | None,
           tuple[slice, ...] | None]:
    """Map one entry of ``Sharding.addressable_devices_indices_map`` to
    a per-device hyperslab read plan.

    Returns ``(local_shape, dst, disk)``:

    * ``local_shape`` — the device-local block shape (replicated axes
      come back from JAX as ``slice(None, None)`` and span the full
      ``out_shape`` axis);
    * ``dst`` — slices INTO the local block for the part overlapping
      the valid (on-disk) region ``[0, valid_shape)`` of the slab, or
      ``None`` when the block lies wholly in the padded tail (leave
      the zero-filled block untouched);
    * ``disk`` — the matching dataset slices, shifted by the caller's
      global ``offset``.

    Single source of truth for the index→(offset, shape) + valid-clip
    arithmetic that was previously copy-pasted per backend
    (``_slab_io_allgather`` sharded fast path and
    ``_slab_io_mpi_host.read_slab`` — audit 2026-07-28,
    QUALITY_PATTERNS #3).  ``common.collectives.
    device_put_process_local`` keeps its own offsets-only two-liner.
    """
    ndim = len(out_shape)
    los: list[int] = []
    his: list[int] = []
    for ax in range(ndim):
        sl = index[ax]
        los.append(0 if sl.start is None else int(sl.start))
        his.append(int(out_shape[ax]) if sl.stop is None else int(sl.stop))
    local_shape = tuple(h - l for l, h in zip(los, his))
    r_lo = [min(l, int(v)) for l, v in zip(los, valid_shape)]
    r_hi = [min(h, int(v)) for h, v in zip(his, valid_shape)]
    if not all(b > a for a, b in zip(r_lo, r_hi)):
        return local_shape, None, None
    disk = tuple(slice(int(offset[ax]) + r_lo[ax],
                       int(offset[ax]) + r_hi[ax]) for ax in range(ndim))
    dst = tuple(slice(r_lo[ax] - los[ax], r_hi[ax] - los[ax])
                for ax in range(ndim))
    return local_shape, dst, disk


_PRESTRIPE_WARNED = False


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

    **It IS a no-op on Frontera.**  Every production run executes inside
    the apptainer image, which does not ship ``lfs`` (nor libraries for
    it), so ``shutil.which`` returns None and this function has never
    once set a stripe here — MEASURED on job 7876423's output:
    ``zeta_q.h5`` and ``isdf_tensors_2406.h5`` both came back
    ``lmm_stripe_count: 1``.  The no-op used to be silent, which is why
    it survived; it now says so once.  The layout is instead requested
    through MPI-IO's ``striping_factor``/``striping_unit`` hints, which
    ROMIO applies via ``llapi`` with no binary on PATH
    (``_slab_io_mpi_host._mpi_io_hints``; ``ffi/cpp/phdf5/context.cc``).

    The ``mode='w'`` inode replace no longer depends on this function:
    :func:`_replace_inode_for_write` unlinks unconditionally before
    either PHDF5 writer opens (audit 2026-07-28 — the ``os.remove``
    below sat behind the lfs early-return above and never ran in the
    container), so on the lfs-available path the remove below merely
    re-runs harmlessly against an already-removed file.
    """
    if shutil.which("lfs") is None:
        global _PRESTRIPE_WARNED
        if not _PRESTRIPE_WARNED and _rank0():
            _PRESTRIPE_WARNED = True
            print("  [SlabIO] `lfs` not on PATH (container): lfs prestripe "
                  "SKIPPED; Lustre layout comes from the MPI-IO "
                  "striping_factor/striping_unit hints instead.", flush=True)
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
    # Process-local placement: plain ``jax.device_put`` of host numpy onto
    # a multi-process sharding runs JAX's hidden ``assert_equal``
    # all-gather (scorecard AA.1) — a per-call blocking collective on a
    # control buffer that is identical on every rank by construction.
    # ``LORRAX_CHECK_REPLICA=1`` re-arms the assertion.
    return device_put_process_local(
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


# ``valid_shape`` is the logical file prefix inside a padded physical
# slab.  It may be ragged; only the physical slab must divide the mesh.
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
# Module-level shard_map kernel factories (read / write)
# ---------------------------------------------------------------------------
#
# A jit'd shard_map per ``(mesh, sharding, ctx_handle, ds_id, mesh_shape,
# axis layout, dtype/shape)`` signature.  Caching at module scope means
# all ``_FfiBackend`` instances share one cache — re-opening the same
# file, or opening any file with a matching FFI signature, reuses the
# compile.  The closure is built INSIDE the cached factory so its
# Python ``id()`` is stable per cache entry (vs ``functools.partial``,
# which constructs a fresh wrapper each call and defeats JAX's
# trace-cache identity test).
#
# FFI attrs (``ctx_handle``, ``ds_id``, mesh layout) remain compile-time
# constants — distinct ``(file, dataset)`` tuples genuinely produce
# distinct HLO modules (the FFI handler picks the dataset by ``ds_id``).
# Coalescing across files would require making those attrs runtime args
# on the C++ side, which is out of scope for this refactor.

@functools.lru_cache(maxsize=None)
def _get_read_sm(mesh, partition_spec, *,
                 ds_id, ctx_handle, mesh_shape,
                 axis_count_per_dim, axis_flat, out_struct):
    """One H5Dread per rank.  Returns a jit'd shard_map; identity-stable
    via lru_cache so JAX's trace cache hits on repeat invocation."""
    from ffi.phdf5.read import ffi_read_call

    def _per_rank(offset_local, valid_shape_local):
        return ffi_read_call(
            out_struct, offset_local, valid_shape_local,
            ctx_handle=ctx_handle, ds_id=ds_id,
            mesh_shape=mesh_shape,
            axis_count_per_dim=axis_count_per_dim,
            axis_flat=axis_flat,
        )
    sm_bare = shard_map(
        _per_rank, mesh=mesh,
        in_specs=(P(), P()), out_specs=partition_spec,
        check_rep=False,
    )
    return jax.jit(sm_bare)


@functools.lru_cache(maxsize=None)
def _get_write_sm(mesh, in_specs, *,
                  ds_id, ctx_handle, mesh_shape,
                  axis_count_per_dim, axis_flat, no_jit):
    """One H5Dwrite per rank.  ``LORRAX_WRITE_NO_JIT=1`` (passed via
    ``no_jit``) skips the jit wrapper — diagnostic for chasing the
    jit-argument-retention buffer leak on long write loops."""
    from ffi.phdf5.write import ffi_write_call

    def _per_rank(A_local, offset_local, valid_shape_local):
        return ffi_write_call(
            A_local, offset_local, valid_shape_local,
            ctx_handle=ctx_handle, ds_id=ds_id,
            mesh_shape=mesh_shape,
            axis_count_per_dim=axis_count_per_dim,
            axis_flat=axis_flat,
        )
    sm_bare = shard_map(
        _per_rank, mesh=mesh,
        in_specs=(in_specs, P(), P()), out_specs=P(),
        check_rep=False,
    )
    return sm_bare if no_jit else jax.jit(sm_bare)


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
        # mode='w' must REPLACE the inode (rank-0 unlink + barrier,
        # shared with _MpiHostBackend — see _replace_inode_for_write):
        # Lustre layout is fixed at inode create, so H5Fcreate(TRUNC)
        # over an old file silently keeps its stripe count and the
        # MPI_Info striping hints no-op.  Unconditional of `lfs`: the
        # old unlink sat behind _lustre_prestripe's lfs-missing early
        # return and never ran in the production container (audit
        # 2026-07-28, job 7876423 1-stripe evidence).  'a'/'r' keep the
        # existing inode and its layout by design.  Barrier after the
        # rank-0 prestripe so all ranks see the inode state before
        # H5Fcreate.
        if mode == "w":
            _replace_inode_for_write(path)
            if _rank0():
                _lustre_prestripe(
                    path, stripe_count=_stripe_count(),
                    stripe_size=os.environ.get(
                        "LORRAX_PHDF5_STRIPE_SIZE_FS", "4M"))
            _barrier("slab_io_ffi_prestripe")
        self.fh: int = self._open_file(path, mesh=mesh, mode=mode)
        self._ds_ids: dict[str, int] = {}
        # write_attr needs plain h5py (the FFI doesn't expose a
        # collective attr-write path), so we defer attr writes to
        # close() — concurrent h5py + MPI-IO on the same file would
        # corrupt HDF5 metadata.
        self._deferred_attrs: list[tuple[str, object]] = []
        # Python-level async writer.  ``write_slab`` enqueues a callable
        # here; the ``AsyncDispatcher`` worker pops it and calls
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
        # Compiled shard_map cache lives at module level — see
        # ``_get_read_sm`` / ``_get_write_sm``.  Instance no longer
        # carries its own ``_sm_cache``; the module-level lru_cache is
        # shared across all _FfiBackend instances, so re-opening the
        # same (or another) file with matching FFI signature reuses
        # the cached compile.
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
        # saturates) while saving 2 × zeta_chunk/rank.
        from common.async_io import AsyncDispatcher
        self._dispatcher = AsyncDispatcher(
            name=f"phdf5-dispatch-{path}", maxsize=2)

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
        # ``phdf5_ensure_dataset`` is a collective HDF5 op (H5Dcreate)
        # that goes through the same MPI file handle as the writer
        # thread's H5Dwrite.  If we issue it while the writer is still
        # in flight, MPI's datatype-cache state on the file handle
        # interleaves and the next H5Dwrite trips ``MPI_File_set_view:
        # Invalid datatype``.  Drain first.
        self._drain_pending()
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
    def _drain_pending(self) -> None:
        """Block main thread until all queued write tasks finish."""
        self._dispatcher.drain()

    # ------------------------------------------------------------------
    def _introspect_dataset(self, name: str) -> tuple[tuple[int, ...], "np.dtype"]:
        """Return ``(shape, dtype)`` of an existing dataset.

        Uses h5py for the metadata read (cheap, parallel-safe with
        ``HDF5_USE_FILE_LOCKING=FALSE`` already set process-wide).
        Cached so repeated lookups for the same name are free.

        Symmetry with the allgather backend: callers don't have to
        pre-compute shape just because the FFI write thunk needs it
        as an FFI attr — we look it up here.
        """
        cache = getattr(self, "_introspect_cache", None)
        if cache is None:
            cache = {}
            self._introspect_cache = cache
        if name in cache:
            return cache[name]
        import h5py
        with h5py.File(self.path, "r") as f:
            ds = f[name]
            shape = tuple(int(s) for s in ds.shape)
            dtype = np.dtype(ds.dtype)
        cache[name] = (shape, dtype)
        return shape, dtype

    def _ds_id(self, name: str, readonly: bool = False) -> int:
        if name in self._ds_ids:
            return self._ds_ids[name]
        # ``phdf5_open_dataset_ro`` is collective on the file handle
        # — same MPI rendezvous + datatype-cache hazard as
        # ``phdf5_ensure_dataset`` (see :meth:`create_dataset`).
        self._drain_pending()
        if readonly:
            ds_id = self._loader.phdf5_open_dataset_ro(self.fh, name)
        else:
            raise RuntimeError(
                f"dataset '{name}' not registered — call create_dataset first")
        self._ds_ids[name] = ds_id
        return ds_id

    # ------------------------------------------------------------------
    # FFI write padding contract: shard ``A`` with equal local blocks,
    # then let C++ clip each rank's file hyperslab to ``valid_shape``.
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
        if not isinstance(A, jax.Array):
            A = jnp.asarray(A)
        # Ensure placement: if not sharded on our mesh, put as replicated.
        # Process-local (see _replicated_i64_vector): a host/uncommitted
        # operand here would otherwise pay the hidden assert_equal
        # all-gather at P × A.nbytes — on a WRITE-path tensor, the
        # single biggest assertion payload in the codebase (AA.1 class).
        # A replicated write requires rank-identical A anyway (the
        # collective writer dedups replicas); LORRAX_CHECK_REPLICA=1
        # re-arms the assertion.
        if not isinstance(A.sharding, NamedSharding) or A.sharding.mesh is not self.mesh:
            A = device_put_process_local(
                A, _replicated_sharding(self.mesh, A.ndim))

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

        # Module-level lru_cache shared across all _FfiBackend instances.
        # Keys on the FFI signature (ctx_handle / ds_id / mesh / sharding);
        # offset is a RUNTIME arg so different chunks reuse one compile.
        sm = _get_write_sm(
            self.mesh, in_specs,
            ds_id=int(ds_id), ctx_handle=int(ctx_handle),
            mesh_shape=mesh_shape,
            axis_count_per_dim=axis_count_per_dim,
            axis_flat=axis_flat,
            no_jit=bool(os.environ.get('LORRAX_WRITE_NO_JIT')),
        )

        # Enqueue dispatch onto the Python worker thread.  Main thread
        # returns in ~0.2ms; the worker thread calls ``sm(A, offset)``
        # in FIFO order.  The offset Buffer is tiny (ndim × 8 bytes).
        offset_arr = _replicated_i64_vector(off, self.mesh)
        valid_shape_arr = _replicated_i64_vector(vshape, self.mesh)

        def _task():
            tok = sm(A, offset_arr, valid_shape_arr)
            tok.block_until_ready()

        self._dispatcher.submit(_task)

    # ------------------------------------------------------------------
    # FFI read padding contract: output ``shape`` is equal-block
    # sharded; C++ reads only ``valid_shape`` and zero-fills the rest.
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
        as_numpy: bool = False,  # accepted for signature compatibility;
        # the public SlabIO.read_slab handles the numpy conversion.
    ) -> jax.Array:
        mesh = mesh or self.mesh
        if shape is None or dtype is None:
            # Symmetry with the allgather backend: callers that don't
            # need padding shouldn't have to compute shape themselves.
            # Cheap h5py introspect (collective-free, locking off);
            # the result is also the file's intrinsic shape, which is
            # what the kernel will read into when valid_shape isn't
            # set.
            ds_shape, ds_dtype = self._introspect_dataset(name)
            if shape is None:
                shape = ds_shape
            if dtype is None:
                dtype = ds_dtype
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

        # Module-level lru_cache shared across all _FfiBackend instances.
        sm = _get_read_sm(
            mesh, partition_spec,
            ds_id=int(ds_id), ctx_handle=int(ctx_handle),
            mesh_shape=mesh_shape,
            axis_count_per_dim=axis_count_per_dim,
            axis_flat=axis_flat,
            out_struct=out_struct,
        )

        offset_arr = _replicated_i64_vector(off, mesh)
        valid_shape_arr = _replicated_i64_vector(vshape, mesh)
        result = sm(offset_arr, valid_shape_arr)
        result.block_until_ready()
        return result

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
        _pending = self._dispatcher.pending
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
        self._dispatcher.close()                # drain + poison pill + join
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
            # Same barrier, same reason as the write-ordering ones above:
            # rank 0 has just rewritten datasets in this file with serial
            # h5py, and no other rank may reopen it until that is durable.
            _barrier("slab_io_ffi_close_attrs")
            self._deferred_attrs = []

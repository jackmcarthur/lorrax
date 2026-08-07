"""The two cross-process primitives wfn_loader needs, and nothing else.

PROVENANCE, both halves ported at e9340d1:

* :func:`device_put_process_local` — from
  ``services/distrib_la/src/distrib_la/_collectives.py`` (itself the
  ``device_put_process_local`` half of LORRAX ``src/common/collectives.py``
  at 96a6399).  distrib_la's ``broadcast_bytes`` is NOT here: it exists in
  that service for the cuSOLVERMp ``ncclUniqueId`` bootstrap, and this
  package has no collective whose context has to be shipped from rank 0.
* :func:`_local_shard_and_global_offset` — from LORRAX
  ``src/file_io/_slab_io_ffi.py:862``, body verbatim; the only edit is the
  dropped ``A: jax.Array`` parameter annotation, which would name a module
  this file deliberately does not import at module scope.  It is the
  second half of the "learn my own slab" idiom
  :meth:`wfn_loader.loader.WfnLoader._assemble_process_local` runs on, and
  reaching back into ``file_io`` for it would be exactly the monorepo
  dependency this package exists to not have.

Both are private copies by coordination ruling 3: a standalone package
cannot import the monorepo, and one service must not import another to
avoid it.  CONSOLIDATION INTO :mod:`lxkit` IS REGISTERED POST-WAVE — these
functions are cross-process substrate with no service-specific content, so
three copies is the wrong end state and this docstring is the record that
somebody said so before the third one was written.
"""

from __future__ import annotations

import os

import numpy as np

__all__ = ["device_put_process_local", "_local_shard_and_global_offset"]


def device_put_process_local(host_array, sharding, *, check: bool | None = None):
    """Place a host array that EVERY process already holds onto ``sharding``
    **without a collective**.

    Drop-in for ``jax.device_put(host_array, sharding)`` on a multi-process
    ``NamedSharding``.  Each process slices out only the shard(s) its own
    devices own and declares them via
    ``jax.make_array_from_single_device_arrays``.

    Why it exists here: a plain ``device_put`` of host numpy onto a
    multi-process sharding fires JAX's hidden ``assert_equal`` all-gather
    at ``P × x.nbytes`` (scorecard AA.1).  For THIS package's tables that
    is the larger of the two "P-LINEAR loader allgathers" of scorecard
    Y.3: the replicated ``box_index`` table is ``P·nk·n_rtot·4`` B (6.45 GB
    at P=64) and the τ-phase table ``P·nk·ngkmax·16`` B (1.27 GB at P=64),
    on every rank, per call.

    **Correctness precondition**, the same one ``device_put`` was spending
    17 GB/rank to assert: ``host_array`` must be bit-identical on every
    process.  ``LORRAX_CHECK_REPLICA=1`` (or ``check=True``) re-enables
    JAX's assertion for a debugging run; it costs exactly the all-gather
    described above, so it is OFF by default.
    """
    import jax

    # An input that is ALREADY a globally-sharded jax.Array is a genuine
    # reshard, not host staging: device_put handles it on the committed /
    # non-fully-addressable branch, which never calls assert_equal.  Hand
    # it straight back -- and never np.asarray it, which would BE the host
    # gather this function exists to avoid.
    if isinstance(host_array, jax.Array) and not host_array.is_fully_addressable:
        return jax.device_put(host_array, sharding)

    arr = np.asarray(host_array)
    if jax.process_count() <= 1 or bool(
            getattr(sharding, "is_fully_addressable", False)):
        return jax.device_put(arr, sharding)

    if check is None:
        # The same falsy vocabulary as every other LORRAX knob.  A parse
        # that recognised only ""/"0"/"false" meant LORRAX_CHECK_REPLICA=off
        # silently ENABLED a P-linear all-gather (7.8 GB/rank at P=64,
        # scorecard Y.5) via a string that reads as "disabled".
        check = os.environ.get("LORRAX_CHECK_REPLICA", "0").strip().lower() \
            not in ("", "0", "false", "no", "off")
    if check:
        return jax.device_put(arr, sharding)

    shape = tuple(int(s) for s in arr.shape)
    idx_map = sharding.addressable_devices_indices_map(shape)
    if not idx_map:
        # This process owns no device in the target sharding.  JAX skips
        # its own assertion in that case too, so device_put is already
        # collective-free.
        return jax.device_put(arr, sharding)
    shards = [jax.device_put(np.ascontiguousarray(arr[idx]), dev)
              for dev, idx in idx_map.items()]
    return jax.make_array_from_single_device_arrays(shape, sharding, shards)


def _local_shard_and_global_offset(A):
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
            f"SlabIO expects 1 addressable shard per process; "
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

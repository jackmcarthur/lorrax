"""The two cross-process primitives wfn_loader needs, and nothing else.

PROVENANCE, both halves ported at e9340d1:

* :func:`device_put_process_local` — the shared lxkit placement primitive.
  distrib_la's ``broadcast_bytes`` is NOT here: it exists in that service
  for the cuSOLVERMp ``ncclUniqueId`` bootstrap, and this package has no
  collective whose context has to be shipped from rank 0.
* :func:`_local_shard_and_global_offset` — from LORRAX
  ``src/file_io/_slab_io_ffi.py:862``, body verbatim; the only edit is the
  dropped ``A: jax.Array`` parameter annotation, which would name a module
  this file deliberately does not import at module scope.  It is the
  second half of the "learn my own slab" idiom
  :meth:`wfn_loader.loader.WfnLoader._assemble_process_local` runs on, and
  reaching back into ``file_io`` for it would be exactly the monorepo
  dependency this package exists to not have.

The slab-offset helper remains local because it is loader-specific.  The
placement primitive is cross-process substrate with no service-specific
content and therefore lives once in :mod:`lxkit`.
"""

from __future__ import annotations

import numpy as np
from lxkit import device_put_process_local

__all__ = ["device_put_process_local", "_local_shard_and_global_offset"]


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

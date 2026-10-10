"""The one cross-process primitive wfn_loader needs, and nothing else.

:func:`device_put_process_local` is the shared lxkit placement primitive.
distrib_la's ``broadcast_bytes`` is NOT here: it exists in that service for
the cuSOLVERMp ``ncclUniqueId`` bootstrap, and this package has no collective
whose context has to be shipped from rank 0.  A rank's own slab of a sharded
array comes from the sharding itself
(:meth:`wfn_loader.loader.WfnLoader._assemble_process_local`).
"""

from __future__ import annotations

from lxkit import device_put_process_local

__all__ = ["device_put_process_local"]

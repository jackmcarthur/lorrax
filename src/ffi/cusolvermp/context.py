"""Re-export shim — implementation moved to ``distrib_la._cusolvermp``.

Deletion is the replumb-complete gate.  No in-tree reacher: the NCCL /
cuSOLVERMp context lifecycle is now internal to the service, which is
where a per-process singleton over a vendor communicator belongs.
"""
from distrib_la._cusolvermp import get_or_init_context  # noqa: F401

__all__ = ["get_or_init_context"]

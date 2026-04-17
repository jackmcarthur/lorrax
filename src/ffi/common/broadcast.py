"""Byte-exact broadcast via the JAX distributed KV store.

Used to ship opaque library handles (ncclUniqueId for cuSOLVERMp, MPI
communicator descriptors for ELPA, PMI handles, ...) from rank 0 to
every JAX process before any collective work begins.

Why not ``jax.experimental.multihost_utils.broadcast_one_to_all``?
Under ``jax_enable_x64=True`` that helper silently promotes ``uint8``
inputs to ``uint64``, which scrambles opaque byte payloads.  The
distributed runtime client's KV store is byte-exact by construction
(strings over a network) and is already live once
``jax.distributed.initialize()`` returns.

Typical use::

    buf = np.zeros(n_bytes, dtype=np.uint8)
    if jax.process_index() == 0:
        lib.fill_unique_id(buf.ctypes.data)      # library-specific
    buf = broadcast_bytes(buf, key='lorrax_ffi/cusolvermp/nccl_uid/v0')
"""
from __future__ import annotations

import numpy as np
import jax

__all__ = ["broadcast_bytes"]


def broadcast_bytes(buf: np.ndarray, *, key: str,
                    timeout_ms: int = 60_000) -> np.ndarray:
    """Broadcast a ``uint8`` numpy buffer from rank 0 to all JAX processes.

    Rank 0's contents of ``buf`` are replicated into every rank's returned
    buffer.  Byte-exact — no dtype promotion.  Single-process jobs are a
    no-op pass-through of the input.

    Parameters
    ----------
    buf
        ``uint8`` numpy array.  On rank 0, the payload to broadcast; on
        other ranks, any initial contents are overwritten.
    key
        Unique KV-store key.  Each call site should use a distinct key
        to avoid collisions.
    timeout_ms
        Max milliseconds to wait for rank 0's `set` before giving up.

    Returns
    -------
    np.ndarray (uint8, same shape as ``buf``), equal on every rank.
    """
    if buf.dtype != np.uint8:
        raise TypeError(
            f"broadcast_bytes: expected uint8 input, got {buf.dtype}")
    if jax.process_count() == 1:
        return buf

    from jax._src.distributed import global_state
    client = global_state.client
    if int(jax.process_index()) == 0:
        client.key_value_set(key, buf.tobytes().hex())
    payload_hex = client.blocking_key_value_get(key, timeout_ms)
    payload = bytes.fromhex(payload_hex)
    if len(payload) != buf.size:
        raise RuntimeError(
            f"broadcast_bytes: received {len(payload)} bytes, "
            f"expected {buf.size}")
    return np.frombuffer(payload, dtype=np.uint8).copy()

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

import time
import numpy as np
import jax

__all__ = ["broadcast_bytes", "reduce_bytes_to_all"]


def _is_wait_timeout(exc: BaseException) -> bool:
    text = str(exc).upper()
    return (isinstance(exc, TimeoutError) or "DEADLINE" in text
            or "TIMED OUT" in text or "TIMEOUT" in text)


def _wait_for_key(client, key: str, timeout_ms: int, *, what: str) -> bytes:
    """Wait for one host-coordination key, with visible 60 s heartbeats."""
    started = time.monotonic()
    while True:
        elapsed_ms = int((time.monotonic() - started) * 1000)
        chunk_ms = 60_000
        if timeout_ms > 0:
            chunk_ms = min(chunk_ms, timeout_ms - elapsed_ms)
            if chunk_ms <= 0:
                raise TimeoutError(
                    f"{what}: not within {timeout_ms / 1000:g} seconds")
        try:
            return client.blocking_key_value_get_bytes(key, max(1, chunk_ms))
        except Exception as exc:                         # noqa: BLE001
            elapsed_ms = int((time.monotonic() - started) * 1000)
            if not _is_wait_timeout(exc) or (
                    timeout_ms > 0 and elapsed_ms >= timeout_ms):
                raise
            print(f"  [host control] still waiting for {what} "
                  f"({elapsed_ms / 1000:.0f} s elapsed)", flush=True)


def reduce_bytes_to_all(buf: np.ndarray, *, key: str, reduce,
                        max_bytes: int = 4096,
                        timeout_ms: int = 0) -> np.ndarray:
    """Gather bounded byte records on rank zero and broadcast one reduction.

    This is the control-plane counterpart of tensor collectives.  It exists
    for small receipts that must remain usable while a device collective is
    failed or busy; payload arrays do not belong here.  Rank zero alone reads
    the P inputs and applies ``reduce(list[np.ndarray])``.  Every process reads
    the one reduced result, keeping both RPC count and traffic O(P).  Every
    reader acknowledges that result before rank zero removes the transaction's
    keys, so a long run does not retain one record per I/O transaction.

    ``key`` must identify one occurrence and be identical on every process.
    ``timeout_ms <= 0`` retains the collective convention of waiting without
    a deadline while emitting one heartbeat per minute.
    """
    arr = np.ascontiguousarray(buf)
    if arr.dtype != np.uint8 or arr.ndim != 1:
        raise TypeError(
            "reduce_bytes_to_all: expected a contiguous 1-D uint8 array, got "
            f"shape={arr.shape} dtype={arr.dtype}")
    if arr.nbytes > int(max_bytes):
        raise ValueError(
            f"reduce_bytes_to_all: {arr.nbytes} bytes exceeds bound "
            f"{max_bytes}")

    n_proc = int(jax.process_count())
    proc_idx = int(jax.process_index())
    if n_proc <= 1:
        result = np.ascontiguousarray(reduce([arr]))
        if result.dtype != np.uint8 or result.ndim != 1:
            raise TypeError("reduce_bytes_to_all: reduction must return 1-D uint8")
        if result.nbytes > int(max_bytes):
            raise ValueError(
                f"reduce_bytes_to_all: reduced {result.nbytes} bytes exceeds "
                f"bound {max_bytes}")
        return result

    from jax._src.distributed import global_state
    client = global_state.client
    if client is None:
        raise RuntimeError(
            "reduce_bytes_to_all: JAX distributed coordination client is "
            "absent "
            f"at process_count={n_proc}")

    value_keys = [f"{key}/value/{rank}" for rank in range(n_proc)]
    ack_keys = [f"{key}/ack/{rank}" for rank in range(n_proc)]
    result_key = f"{key}/result"
    client.key_value_set_bytes(value_keys[proc_idx], arr.tobytes())
    if proc_idx == 0:
        try:
            payloads = [
                _wait_for_key(client, value_key, timeout_ms,
                              what=f"host receipt from rank {rank} for {key}")
                for rank, value_key in enumerate(value_keys)
            ]
            sizes = {len(payload) for payload in payloads}
            if sizes != {arr.nbytes}:
                raise RuntimeError(
                    f"ranks published sizes {sorted(sizes)}, expected "
                    f"{arr.nbytes}")
            records = [np.frombuffer(payload, dtype=np.uint8).copy()
                       for payload in payloads]
            result = np.ascontiguousarray(reduce(records))
            if result.dtype != np.uint8 or result.ndim != 1:
                raise TypeError("reduction must return 1-D uint8")
            if result.nbytes > int(max_bytes):
                raise ValueError(
                    f"reduced {result.nbytes} bytes exceeds bound {max_bytes}")
            envelope = b"\0" + result.tobytes()
        except BaseException as exc:                     # noqa: BLE001
            message = (f"{type(exc).__name__}: {exc}").encode(
                "utf-8", errors="replace")[:max_bytes]
            envelope = b"\1" + message
        client.key_value_set_bytes(result_key, envelope)

    envelope = _wait_for_key(
        client, result_key, timeout_ms, what=f"host reduction for {key}")
    client.key_value_set_bytes(ack_keys[proc_idx], b"1")
    if proc_idx == 0:
        for rank, ack_key in enumerate(ack_keys):
            _wait_for_key(client, ack_key, timeout_ms,
                          what=f"host receipt acknowledgement from rank {rank} "
                               f"for {key}")
        for item in value_keys + [result_key] + ack_keys:
            client.key_value_delete(item)
    if not envelope or envelope[0] not in (0, 1):
        raise RuntimeError(
            f"reduce_bytes_to_all: invalid result envelope for {key}")
    if envelope[0] == 1:
        raise RuntimeError(
            f"reduce_bytes_to_all: root reduction failed for {key}: "
            f"{envelope[1:].decode('utf-8', errors='replace')}")
    return np.frombuffer(envelope[1:], dtype=np.uint8).copy()


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

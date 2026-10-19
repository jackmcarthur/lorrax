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

It also owns the all-rank gather every host control agreement uses
(:func:`publish_rank_record`, :func:`collect_rank_records`): each rank
stores one record and bumps one atomic arrival counter, the rank that
completes the set marks the directory complete, and the collector waits on
that one key and reads every record with one directory get.  The collector
therefore makes O(1) coordination RPCs at any P; one blocking get per rank
cost about 53 ms per agreement at P64.
"""
from __future__ import annotations

import time
import numpy as np
import jax

__all__ = ["broadcast_bytes", "reduce_bytes_to_all", "wait_for_key",
           "publish_rank_record", "rank_records", "collect_rank_records"]

_HEARTBEAT_MS = 60_000


def _is_wait_timeout(exc: BaseException) -> bool:
    text = str(exc).upper()
    return (isinstance(exc, TimeoutError) or "DEADLINE" in text
            or "TIMED OUT" in text or "TIMEOUT" in text)


def wait_for_key(client, key: str, timeout_ms: int, *, what: str,
                 pending=None) -> bytes:
    """Wait for one host-coordination key, with visible 60 s heartbeats.

    ``timeout_ms <= 0`` waits without a deadline.  ``pending()``, when
    given, returns a short text naming what is still missing; it is read
    only at a heartbeat or a deadline, never on the fast path.
    """
    started = time.monotonic()
    while True:
        elapsed_ms = int((time.monotonic() - started) * 1000)
        chunk_ms = _HEARTBEAT_MS
        if timeout_ms > 0:
            chunk_ms = min(chunk_ms, timeout_ms - elapsed_ms)
            if chunk_ms <= 0:
                detail = f"; {pending()}" if pending is not None else ""
                raise TimeoutError(
                    f"{what}: not within {timeout_ms / 1000:g} seconds"
                    f"{detail}")
        try:
            return client.blocking_key_value_get_bytes(key, max(1, chunk_ms))
        except Exception as exc:                         # noqa: BLE001
            elapsed_ms = int((time.monotonic() - started) * 1000)
            if not _is_wait_timeout(exc):
                raise
            if timeout_ms > 0 and elapsed_ms >= timeout_ms:
                continue                     # the deadline branch above raises
            detail = f"; {pending()}" if pending is not None else ""
            print(f"  [host control] still waiting for {what} "
                  f"({elapsed_ms / 1000:.0f} s elapsed{detail})", flush=True)


def _arrive(client, directory: str, n_arrivals: int) -> None:
    """Count one arrival; the ``n_arrivals``-th marks ``directory`` complete."""
    if client.key_value_increment(f"{directory}/arrived", 1) == n_arrivals:
        client.key_value_set_bytes(f"{directory}/complete", b"1")


def publish_rank_record(client, directory: str, rank: int, n_proc: int,
                        payload: bytes) -> None:
    """Store this rank's record under ``directory`` and count its arrival.

    The record is stored before the counter moves, so once ``directory/
    complete`` exists every rank's record can be read.
    """
    client.key_value_set_bytes(f"{directory}/rank/{rank}", payload)
    _arrive(client, directory, n_proc)


def rank_records(client, directory: str, n_proc: int) -> list:
    """The records published under ``directory`` in rank order, one RPC.

    A rank that has not published yet reads as ``None``.
    """
    records = [None] * n_proc
    for name, value in client.key_value_dir_get_bytes(f"{directory}/rank"):
        rank = name.rsplit("/", 1)[-1]
        if rank.isdigit() and int(rank) < n_proc:
            records[int(rank)] = value
    return records


def _missing_ranks_text(records) -> str:
    missing = [rank for rank, record in enumerate(records) if record is None]
    shown = ", ".join(str(rank) for rank in missing[:16])
    more = f", ... ({len(missing)} ranks)" if len(missing) > 16 else ""
    return f"missing rank(s) {shown}{more}"


def collect_rank_records(client, directory: str, n_proc: int,
                         timeout_ms: int, *, what: str) -> list:
    """Every rank's record under ``directory``, in rank order.

    Waits on the one completion key (heartbeats name the missing ranks),
    then reads all records with one directory get.
    """
    wait_for_key(
        client, f"{directory}/complete", timeout_ms, what=what,
        pending=lambda: _missing_ranks_text(
            rank_records(client, directory, n_proc)))
    records = rank_records(client, directory, n_proc)
    if any(record is None for record in records):
        raise RuntimeError(
            f"{what}: arrival count complete but "
            f"{_missing_ranks_text(records)}")
    return records


def reduce_bytes_to_all(buf: np.ndarray, *, key: str, reduce,
                        max_bytes: int = 4096,
                        timeout_ms: int = 0) -> np.ndarray:
    """Gather bounded byte records on rank zero and broadcast one reduction.

    This is the control-plane counterpart of tensor collectives.  It exists
    for small receipts that must remain usable while a device collective is
    failed or busy; payload arrays do not belong here.  Rank zero alone reads
    the P inputs (one directory get, :func:`collect_rank_records`) and
    applies ``reduce(list[np.ndarray])``.  Every other process reads the one
    reduced result, so rank zero makes O(1) coordination RPCs and the total
    traffic is O(P).  Every reader acknowledges that result through one
    arrival counter before rank zero removes the transaction's directory
    with one delete, so a long run does not retain one record per I/O
    transaction.

    ``key`` must identify one occurrence and be identical on every process;
    the transaction owns every key under ``key/``.  ``timeout_ms <= 0``
    retains the collective convention of waiting without a deadline while
    emitting one heartbeat per minute.
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

    result_key = f"{key}/result"
    publish_rank_record(client, key, proc_idx, n_proc, arr.tobytes())
    if proc_idx == 0:
        try:
            payloads = collect_rank_records(
                client, key, n_proc, timeout_ms,
                what=f"host receipts for {key}")
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
        wait_for_key(
            client, f"{key}/read/complete", timeout_ms,
            what=f"every rank to read the host reduction for {key}")
        client.key_value_delete(key)          # the whole key/ directory
    else:
        envelope = wait_for_key(
            client, result_key, timeout_ms, what=f"host reduction for {key}")
        _arrive(client, f"{key}/read", n_proc - 1)
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

"""Private cross-process transport for small symmetry receipts.

The two-component reference check is host NumPy work over large coefficient
slabs.  In a multi-process run that work belongs on one process; only its
small, immutable report is replicated.  The JAX distributed key-value store
is already live at that point and transports bytes without a device
collective or dtype conversion.

This helper stays in ``symmetry_maps`` for the same reason ``distrib_la``
owns its NCCL-id broadcast: use of JAX's private distributed client is a
service-local portability bet, not shared ``lxkit`` policy.
"""

from __future__ import annotations


# Large-WFN host overlap has no separately certified wall-time bound.  Keep
# the receipt wait well above an ordinary RPC timeout so a slow valid check is
# not turned into a false distributed failure.
DEFAULT_TIMEOUT_MS = 2 * 60 * 60 * 1000


def _transport_context(
    *,
    operation: str,
    timeout_ms: int,
    client,
    process_index: int | None,
    process_count: int | None,
):
    """Resolve and validate one byte-KV collective's runtime context."""
    if int(timeout_ms) <= 0:
        raise ValueError(f"{operation}: timeout_ms must be positive")
    if process_index is None or process_count is None:
        import jax
        if process_index is None:
            process_index = int(jax.process_index())
        if process_count is None:
            process_count = int(jax.process_count())
    rank = int(process_index)
    world = int(process_count)
    if world < 1 or not 0 <= rank < world:
        raise ValueError(
            f"{operation}: invalid process geometry rank={rank}, world={world}")
    if world == 1:
        return None, rank, world

    if client is None:
        from jax._src.distributed import global_state
        client = global_state.client
    if client is None:
        raise RuntimeError(
            f"{operation}: JAX distributed client is unavailable for "
            f"world={world}; initialize jax.distributed first")
    required = (
        "key_value_set_bytes", "blocking_key_value_get_bytes",
        "wait_at_barrier",
    )
    missing = [
        name for name in required if not callable(getattr(client, name, None))
    ]
    if missing:
        raise RuntimeError(
            f"{operation}: distributed client lacks the byte-exact KV API "
            f"{missing}; this route requires JAX 0.9.x")
    return client, rank, world


def broadcast_root_bytes(
    payload: bytes | None,
    *,
    key: str,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    client=None,
    process_index: int | None = None,
    process_count: int | None = None,
) -> bytes:
    """Publish rank 0's bytes and return them on every JAX process.

    ``client`` and the process coordinates are injectable solely so the
    transport contract can be tested without starting a distributed JAX
    runtime.  Production callers omit them.
    """
    if not isinstance(key, str) or not key:
        raise ValueError("broadcast_root_bytes: key must be a non-empty string")
    client, rank, world = _transport_context(
        operation="broadcast_root_bytes", timeout_ms=timeout_ms,
        client=client, process_index=process_index,
        process_count=process_count)

    if world == 1:
        if payload is None:
            raise TypeError(
                "broadcast_root_bytes: rank 0 must supply a bytes payload")
        return bytes(payload)

    if rank == 0:
        if payload is None:
            raise TypeError(
                "broadcast_root_bytes: rank 0 must supply a bytes payload")
        data = bytes(payload)
        client.key_value_set_bytes(key, data)
    else:
        if payload is not None:
            raise TypeError(
                "broadcast_root_bytes: non-root processes must pass "
                "payload=None")
        data = bytes(client.blocking_key_value_get_bytes(key, int(timeout_ms)))

    # The payload may be an error that rank 0 will raise immediately after
    # this call.  Commit the read collectively first: otherwise rank 0 can
    # tear down the coordination client while peers are still in KV get.
    client.wait_at_barrier(f"{key}/commit", timeout_in_ms=int(timeout_ms))
    return data


def gather_root_bytes(
    payload: bytes,
    *,
    key: str,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    client=None,
    process_index: int | None = None,
    process_count: int | None = None,
) -> tuple[bytes, ...] | None:
    """Gather one small byte proposal per process onto rank 0.

    Every process publishes before either barrier.  The second barrier is
    load-bearing: peers cannot enter a subsequent verdict broadcast until
    rank 0 has retrieved every proposal, so a root-side decision can always
    be encoded and published before any caller raises.
    """
    if not isinstance(key, str) or not key:
        raise ValueError("gather_root_bytes: key must be a non-empty string")
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("gather_root_bytes: every process must supply bytes")
    data = bytes(payload)
    client, rank, world = _transport_context(
        operation="gather_root_bytes", timeout_ms=timeout_ms,
        client=client, process_index=process_index,
        process_count=process_count)
    if world == 1:
        return (data,)

    client.key_value_set_bytes(f"{key}/rank/{rank}", data)
    client.wait_at_barrier(f"{key}/published", timeout_in_ms=int(timeout_ms))
    gathered = None
    if rank == 0:
        gathered = tuple(
            data if peer == 0 else bytes(client.blocking_key_value_get_bytes(
                f"{key}/rank/{peer}", int(timeout_ms)))
            for peer in range(world)
        )
    client.wait_at_barrier(f"{key}/read", timeout_in_ms=int(timeout_ms))
    return gathered


__all__ = [
    "broadcast_root_bytes", "gather_root_bytes", "DEFAULT_TIMEOUT_MS",
]

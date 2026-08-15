"""The cross-process primitives distrib_la needs, and nothing else.

Ported from LORRAX ``src/ffi/common/broadcast.py`` and the
``device_put_process_local`` half of ``src/common/collectives.py``, both at
96a6399.  Survey 3 (a) ruled ``broadcast_bytes`` OUT of lxkit — it reaches
``jax._src.distributed.global_state``, which is a private-API bet lxkit
must not take on every service's behalf — and INTO the service that needs
it, which is this one: the cuSOLVERMp context bootstrap ships an
``ncclUniqueId`` from rank 0 to every process before any collective work.
"""

from __future__ import annotations

import os

import numpy as np

__all__ = ["broadcast_bytes", "device_put_process_local",
           "warm_mesh_cliques"]


def broadcast_bytes(buf: np.ndarray, *, key: str,
                    timeout_ms: int = 60_000) -> np.ndarray:
    """Broadcast a ``uint8`` numpy buffer from rank 0 to all JAX processes.

    Byte-exact, and that is the whole point.  Under
    ``jax_enable_x64=True`` — which LORRAX always runs —
    ``jax.experimental.multihost_utils.broadcast_one_to_all`` silently
    promotes ``uint8`` to ``uint64``, which scrambles an opaque payload
    like an ``ncclUniqueId``.  The distributed runtime client's KV store is
    byte-exact by construction (strings over a network) and is already live
    once ``jax.distributed.initialize()`` returns.

    ``key`` must be unique per call site; single-process jobs pass through.
    """
    import jax
    if buf.dtype != np.uint8:
        raise TypeError(
            f"broadcast_bytes: expected uint8 input, got {buf.dtype}")
    if jax.process_count() == 1:
        return buf

    from jax._src.distributed import global_state
    client = global_state.client
    if int(jax.process_index()) == 0:
        client.key_value_set(key, buf.tobytes().hex())
    payload = bytes.fromhex(client.blocking_key_value_get(key, timeout_ms))
    if len(payload) != buf.size:
        raise RuntimeError(
            f"broadcast_bytes: received {len(payload)} bytes, "
            f"expected {buf.size}")
    return np.frombuffer(payload, dtype=np.uint8).copy()


def device_put_process_local(host_array, sharding, *, check: bool | None = None):
    """Place a host array that EVERY process already holds onto ``sharding``
    **without a collective**.

    Drop-in for ``jax.device_put(host_array, sharding)`` on a multi-process
    ``NamedSharding``.  Each process slices out only the shard(s) its own
    devices own and declares them via
    ``jax.make_array_from_single_device_arrays``.

    Why it exists here: a plain ``device_put`` of host numpy onto a
    multi-process sharding fires JAX's hidden ``assert_equal`` all-gather
    at ``P × x.nbytes`` (scorecard AA.1) — for an ``(n, n)`` c128 FFI
    operand that is exactly the class of silent cost
    :func:`distrib_la.plan.ensure_sharding` exists to prevent.

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


# CPU/MPI creates collective communicators on first use.  jaxlib refuses to
# create them from an intra-op worker because that worker is not MPI's main
# thread.  This is the service-local sibling of
# ``common.collectives.warm_mesh_cliques``: distrib_la is independently
# installable and must not reach upward into LORRAX's common package.
_WARMED_MESHES: set = set()


def warm_mesh_cliques(mesh, *, print_fn=print) -> float:
    """Create each mesh-axis MPI communicator on the calling main thread.

    A no-op off JAX's CPU ``impl=mpi`` transport, in single-process runs,
    and for an already-warmed mesh.  Call synchronously on every rank before
    compiling a route containing explicit ``all_to_all`` collectives.
    """
    if os.environ.get(
            "JAX_CPU_COLLECTIVES_IMPLEMENTATION", "").strip().lower() != "mpi":
        return 0.0

    import time

    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.sharding import PartitionSpec as P

    from distrib_la._shard_map import shard_map

    if jax.process_count() <= 1:
        return 0.0
    key = (tuple(int(d.id) for d in mesh.devices.ravel()),
           tuple(mesh.axis_names))
    if key in _WARMED_MESHES:
        return 0.0

    t0 = time.perf_counter()
    tiny = jnp.zeros(1)
    axes = list(mesh.axis_names)
    groups = list(axes) + ([tuple(axes)] if len(axes) > 1 else [])
    for ax in groups:
        f = jax.jit(shard_map(
            lambda a, ax=ax: lax.psum(a, ax), mesh=mesh,
            in_specs=(P(None),), out_specs=P(None), check_vma=False))
        jax.block_until_ready(f(tiny))

    dt = time.perf_counter() - t0
    _WARMED_MESHES.add(key)
    if jax.process_index() == 0:
        print_fn(
            f"[distrib_la] warmed {len(groups)} MPI cliques for mesh "
            f"{tuple(mesh.devices.shape)} axes={axes} in {dt * 1e3:.0f} ms")
    return dt

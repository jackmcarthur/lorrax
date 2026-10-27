"""The source tree's face of ``lxkit.memory``: the run budget from this card's readers.

``lxkit.memory`` is the one memory model (price, room, admission, receipts); it
cannot import ``runtime``, so the card total and the bytes outside the allocator
are read here (``runtime.xla_memory``) and handed to :func:`lxkit.memory.resolve`.
Every memory-driven route goes through :func:`lxkit.memory.admit`.  The host
helpers (``MemAvailable`` shares, host holds) stay here.
"""

import os


def minimum_process_budget_gb(local_gb: float) -> float:
    """Agree the smallest device budget before choosing collective shapes.

    An allocation may contain different GPU memory capacities; even a fixed
    allocator limit is then rank-local. Every process must enter this call.
    """
    import numpy as np
    from common.collectives import all_gather_processes

    budgets = np.asarray(all_gather_processes(np.asarray(local_gb, dtype=np.float64)))
    if budgets.size == 0 or not np.all(np.isfinite(budgets) & (budgets >= 0)):
        raise ValueError("process memory budgets must be finite and nonnegative")
    return float(np.min(budgets))


# ============================================================================
# The run's device budget (lxkit.memory.resolve on this card's readers)
# ============================================================================

def resolve_device_budget_gb(deck_gb: float = 0.0, linalg: str = "local") -> float:
    """THE run budget in decimal GB per device, recorded once in ``lxkit.memory``
    (:func:`lxkit.memory.resolve`): the card total and the bytes outside the allocator now
    from this process's leg readers; CPU: ``MemTotal`` over the processes on the node.
    CPU room also follows live cgroup-aware available RAM beside non-JAX
    consumers. Every process must enter."""
    import jax
    from lxkit import memory
    available = None
    if jax.default_backend() == "cpu":
        processes = _processes_on_this_node()
        total, outside = (_meminfo_gb("MemTotal") or 0.0) * 1e9 / processes, 0
        devices = max(1, jax.local_device_count())
        available = lambda: (_meminfo_gb("MemAvailable") or 0.0) * 1e9 / (processes * devices)
    else:
        from runtime.xla_memory import device_total_bytes, outside_pool_bytes
        ordinal = int(getattr(jax.local_devices()[0], "local_hardware_id", 0) or 0)
        total, outside = device_total_bytes(ordinal), outside_pool_bytes(ordinal)
    return memory.resolve(float(deck_gb or 0.0) * 1e9, str(linalg), total=total,
                          outside=outside, available=available,
                          agree=lambda v: minimum_process_budget_gb(v / 1e9) * 1e9) / 1e9


def device_budget_bytes() -> float:
    """THE per-device budget in bytes every planner prices against; a driver without a
    ``memory_per_device_gb`` key resolves it on its first call, which every process enters."""
    from lxkit import memory
    if "budget" not in memory.run():
        resolve_device_budget_gb()
    return memory.run()["budget"]


def shared_device_room(stage: str):
    """A planner's room beside what is allocated now (:func:`lxkit.memory.room`), or
    ``distrib_la.RoomExhausted`` naming ``stage`` when the allocator already fills the budget
    (its eigh stacks then take the whole mesh with a warning).  Every process must enter."""
    from distrib_la import room_or_exhausted
    from lxkit import memory
    budget, room = int(device_budget_bytes()), memory.room()
    return room_or_exhausted(room, stage=stage, priced_bytes=budget - room, available_bytes=budget)


_HOST_HOLDS: dict[str, float] = {}


def record_host_hold(name: str, nbytes: float) -> None:
    """Name host memory a stage keeps across stages, bytes per process.

    The stage-memory table lists every hold beside the host run peak, so a
    resident host copy is not read as a stage's own rise.
    """
    _HOST_HOLDS[str(name)] = float(nbytes)


def host_holds() -> dict[str, float]:
    """Every host hold recorded this run (:func:`record_host_hold`)."""
    return dict(_HOST_HOLDS)


def _meminfo_gb(field: str, *, proc: str = '/proc',
                cgroup_root: str = '/sys/fs/cgroup') -> float | None:
    """``MemTotal`` or ``MemAvailable`` in GB (1e9 B), capped by this process's
    memory cgroups, or None if ``/proc/meminfo`` is unreadable.

    A batch job's cgroup can hold far less than the node, and its OOM killer
    acts on the cgroup.  ``MemTotal`` is capped by every enclosing cgroup's
    limit and ``MemAvailable`` by its limit minus its usage, page cache it can
    reclaim (``inactive_file``) not counted as usage: cgroup v2
    ``memory.max``/``memory.current``, v1
    ``memory.limit_in_bytes``/``memory.usage_in_bytes``.
    """
    value = None
    try:
        with open(os.path.join(proc, 'meminfo')) as f:
            for line in f:
                if line.startswith(field + ':'):
                    value = int(line.split()[1]) * 1024
                    break
        with open(os.path.join(proc, 'self', 'cgroup')) as f:
            groups = [line.rstrip('\n').split(':', 2) for line in f]
    except (OSError, ValueError, IndexError):
        groups = []
    if value is None:
        return None
    cap = float(value)
    for row in groups:
        if len(row) != 3:
            continue
        if row[1] == '':
            base, names, stat = cgroup_root, ('memory.max', 'memory.current'), 'inactive_file'
        elif 'memory' in row[1].split(','):
            base, stat = os.path.join(cgroup_root, 'memory'), 'total_inactive_file'
            names = ('memory.limit_in_bytes', 'memory.usage_in_bytes')
        else:
            continue
        parts = [p for p in row[2].split('/') if p]
        for depth in range(len(parts) + 1):
            here = os.path.join(base, *parts[:depth])
            try:
                with open(os.path.join(here, names[0])) as f:
                    limit = f.read().strip()
                with open(os.path.join(here, names[1])) as f:
                    used = int(f.read())
            except (OSError, ValueError):
                continue
            if limit == 'max':
                continue
            if field == 'MemTotal':
                cap = min(cap, int(limit))
                continue
            try:
                with open(os.path.join(here, 'memory.stat')) as f:
                    used -= next((int(line.split()[1]) for line in f
                                  if line.split()[0] == stat), 0)
            except (OSError, ValueError, IndexError):
                pass
            cap = min(cap, max(0, int(limit) - used))
    return cap / 1e9


def get_host_memory_available_gb() -> float | None:
    """This node's ``MemAvailable`` in GB (1e9 B), or None if unreadable.

    Whole-node, not per process: a caller whose processes share the node
    divides it among them.
    """
    return _meminfo_gb('MemAvailable')


def _processes_on_this_node() -> int:
    """How many processes of the run share this node (one small gather; every
    process must enter)."""
    import socket
    import zlib
    import numpy as np
    from common.collectives import all_gather_processes
    host = zlib.crc32(socket.gethostname().encode())
    hosts = np.asarray(all_gather_processes(np.asarray(host, dtype=np.int64)))
    return max(1, int(np.sum(hosts == host)))


def host_bytes_per_process(fraction: float = 0.9) -> float:
    """``fraction`` of the node's live ``MemAvailable`` over the processes
    sharing the node, in bytes, agreed (minimum) across processes. Every
    process must enter this call.  For residence choices (host or file), not
    for sizes.
    """
    avail_gb = get_host_memory_available_gb()
    per_node = _processes_on_this_node()
    local_gb = float('inf') if avail_gb is None else float(fraction) * avail_gb / per_node
    return minimum_process_budget_gb(min(local_gb, 1e12)) * 1e9


__all__ = ["device_budget_bytes", "get_host_memory_available_gb", "host_bytes_per_process",
           "minimum_process_budget_gb", "resolve_device_budget_gb", "shared_device_room"]

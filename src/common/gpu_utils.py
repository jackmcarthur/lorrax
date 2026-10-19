"""The run's device budget, the planners' prices against it, and the routes it chooses.

The budget is the one memory rule (``runtime.planner_budget_bytes``) on
the card total, so it is the same on every rank and every run; a positive
deck ``memory_per_device_gb`` can only lower it.  No planner reads free
memory or a fragmentation factor; the pool limit caps the rule only when the
caller's allocator enforces it (docs/architecture/memory-model.md, "Budget").
Every memory-driven route goes through :func:`admit_local`, which prints and
records the decision.
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
# The run's device budget: ONE number every planner prices against
# ============================================================================

#: The run's per-device budget in decimal GB, set once when the deck's
#: ``memory_per_device_gb`` resolves (``gw.gw_config``); None until then.
_RUN_DEVICE_BUDGET_GB: float | None = None
#: The deck's resolved ``linalg`` the budget was ruled for: under ``local`` the
#: rule withheld no room for the distributed providers' contexts.
_RUN_LINALG: str = "local"


def set_device_budget_gb(gb: float) -> None:
    """Record the run's per-device budget (decimal GB), once it resolves
    (:func:`resolve_device_budget_gb`)."""
    global _RUN_DEVICE_BUDGET_GB
    value = float(gb)
    if not value > 0:
        raise ValueError(f"the device budget must be positive GB, got {gb!r}")
    _RUN_DEVICE_BUDGET_GB = value


def resolve_device_budget_gb(deck_gb: float = 0.0, linalg: str = "local") -> float:
    """THE run budget in decimal GB per device, recorded once.

    The memory rule's budget for the deck's resolved ``linalg``
    (:func:`get_device_memory_gb`), the minimum over processes so static tile
    shapes agree on every process.  A positive deck ``memory_per_device_gb``
    below it lowers the budget; above it the rule stands, with one warning
    naming both numbers, because a budget the card cannot hold would admit
    routes and batches that run out of memory mid-run.  Every process must
    enter.
    """
    global _RUN_LINALG
    rule = minimum_process_budget_gb(get_device_memory_gb(linalg))
    deck = float(deck_gb or 0.0)
    if deck > rule:
        import warnings
        from runtime import _resolve_proc_id
        if _resolve_proc_id() == 0:
            warnings.warn(
                f"memory_per_device_gb = {deck:g} exceeds the memory rule's "
                f"{rule:.2f} GB on this card (linalg = {linalg}); running at "
                f"{rule:.2f} GB (docs/architecture/memory-model.md#rule)",
                RuntimeWarning, stacklevel=2)
    budget = min(deck, rule) if deck > 0 else rule
    _RUN_LINALG = str(linalg)
    _ROUTES.clear()     # a new run (the hsuite runs its drivers in one process)
    set_device_budget_gb(budget)
    return budget


def device_budget_bytes() -> float:
    """THE per-device budget in bytes (1 GB = 1e9 B) every planner prices against.

    As the config resolved it; a driver without a ``memory_per_device_gb``
    key (kmeans, htransform, bse, exciton_bands, a tool) resolves the rule on
    its first call, which every process must enter.
    """
    if _RUN_DEVICE_BUDGET_GB is None:
        resolve_device_budget_gb()
    return _RUN_DEVICE_BUDGET_GB * 1e9


def shared_device_room_bytes() -> int:
    """The room beside what is live now: the run budget less this process's live device bytes
    per local device, the minimum over processes so every rank decides alike. Every process must
    enter this call."""
    import jax
    live = sum(s.data.nbytes for a in jax.live_arrays() for s in a.addressable_shards)
    return int(minimum_process_budget_gb(
        max(0.0, device_budget_bytes() - live / len(jax.local_devices())) / 1e9) * 1e9)


def shared_device_room(stage: str):
    """A planner's room beside what is live now: :func:`shared_device_room_bytes`, bucketed
    (:func:`room_bucket`), or ``distrib_la.RoomExhausted`` naming ``stage`` when the live set
    already fills the budget (its eigh stacks then take the whole mesh with a warning, not a
    refusal). Every process must enter this call."""
    from distrib_la import room_or_exhausted
    budget, room = int(device_budget_bytes()), shared_device_room_bytes()
    return room_or_exhausted(room_bucket(room), stage=stage, priced_bytes=budget - room,
                             available_bytes=budget)


def room_bucket(room) -> int:
    """A room as planners key on it: floored to a whole GiB at or above 1 GiB, exact below
    (a sub-GiB room that fits is not rounded to none), 0 when negative."""
    room = int(room)
    return (room >> 30) << 30 if room >= 1 << 30 else max(room, 0)


# ============================================================================
# Routes: every memory-driven local-or-distributed choice, decided and recorded
# ============================================================================

_ROUTES: dict = {}


def record_route(site: str, route: str, reason: str, *, warning: str | None = None) -> None:
    """Record one route decision; rank 0 prints it the first time it is made.

    The one printer of route decisions (:func:`admit_local`, and the service
    decisions a driver hands on).  A repeat of the same decision (an SC map,
    a frequency) only counts; the run report lists every row (:func:`routes`).
    """
    key = (str(site), str(route), str(reason))
    row = _ROUTES.get(key)
    if row is not None:
        row["count"] += 1
        return
    line = f"route {site}: {route} ({reason})" + (f"; WARNING {warning}" if warning else "")
    _ROUTES[key] = dict(site=str(site), route=str(route), reason=str(reason),
                        warning=warning, line=line, count=1)
    from runtime import _resolve_proc_id
    if _resolve_proc_id() == 0:
        print(("  WARNING: " if warning else "  ") + line, flush=True)


def routes() -> list[dict]:
    """Every route decision recorded this run, in first-decision order."""
    return [dict(row) for row in _ROUTES.values()]


_LAST_DECISION: dict = {}


def last_decision() -> dict:
    """``route``, ``reason`` and ``warning`` of the latest :func:`admit_local` call."""
    return dict(_LAST_DECISION)


def admit_local(site: str, local_bytes: float, room_bytes: float, *,
                distributed_bytes: float = 0.0, distributed_room: float | None = None,
                provider: bool = True, fits: bool | None = None, probe=None,
                forced: str | None = None, names=("local", "distributed"),
                quiet: bool = False) -> bool:
    """Whether ``site`` runs its local route (True) or its distributed one (False).

    Local when ``local_bytes``, the local route's full per-rank price, fits
    ``room_bytes`` (``fits``: the service's own answer, when it prices a
    workspace the caller cannot see).  Otherwise the distributed route, priced
    at ``distributed_bytes`` against ``distributed_room`` (default
    ``room_bytes``).  A run ruled for ``linalg = local`` withheld no room for a
    provider's contexts outside the pool (``provider``: the route opens
    cuSOLVERMp), so they are charged here
    (``runtime.DISTRIBUTED_LINALG_OUTSIDE_BYTES``).  When neither fits, it refuses
    by name (GATE route_capacity) naming both prices; ``distributed_bytes``
    may be a callable, priced only then.  ``probe()`` raises
    when the distributed route cannot run here (platform, mesh, build): the
    local route stays, warned.  ``forced`` is a route the deck named.  Every
    input is a shape price or the agreed budget, so every rank decides alike;
    the decision is recorded and printed once (:func:`record_route`), unless
    ``quiet``: a caller that records its final route itself.
    """
    gb = lambda v: f"{float(v) / 1e9:.2f} GB"
    local_name, other_name = names
    def record(site, route, reason, warning=None):
        _LAST_DECISION.update(route=route, reason=reason, warning=warning)
        if not quiet:
            record_route(site, route, reason, warning=warning)
    if forced is not None:
        record(site, forced, "linalg = distributed" if forced == other_name else "named")
        return forced == local_name
    room = float(room_bytes)
    if (local_bytes <= room) if fits is None else bool(fits):
        record(site, local_name, f"{gb(local_bytes)}/rank of room {gb(room)}")
        return True
    if probe is not None:
        try:
            probe()
        except Exception as exc:                                  # noqa: BLE001
            record(site, local_name, f"{gb(local_bytes)}/rank over room {gb(room)}",
                   warning=f"the {other_name} route is unavailable here "
                           f"({type(exc).__name__}: {exc})")
            return True
    from runtime import DISTRIBUTED_LINALG_OUTSIDE_BYTES
    contexts = DISTRIBUTED_LINALG_OUTSIDE_BYTES if provider and _RUN_LINALG == "local" else 0.0
    other_room = room if distributed_room is None else float(distributed_room)
    if callable(distributed_bytes):
        distributed_bytes = distributed_bytes()
    other = float(distributed_bytes) + contexts
    priced = gb(distributed_bytes) + (f" + provider contexts {gb(contexts)}" if contexts else "")
    reason = f"local {gb(local_bytes)}/rank over room {gb(room)}; {other_name} {priced}"
    if other <= other_room:
        record(site, other_name, reason + f" of room {gb(other_room)}")
        return False
    record(site, other_name, reason + f" over room {gb(other_room)}",
           warning="both routes are over their room")
    raise RuntimeError(f"GATE route_capacity: {site}: {reason} over room {gb(other_room)}: no "
                       f"route fits" + ("; linalg = distributed reserves the provider contexts' "
                                        "room at startup" if contexts else ""))


# ============================================================================
# Planner prices: what each planner said its stage would hold, per rank
# ============================================================================

_STAGE_PRICES: list[dict] = []


def record_stage_price(stage: str, price_bytes: float, *, section: str | None = None) -> None:
    """Record a planner's per-rank price for the stage it plans.

    ``price_bytes`` is the live set the planner compared against its budget;
    ``section`` names the timing section whose device peak the price is judged
    against (default: the innermost open section at the call).  The run's
    stage-memory table prints peak / price as γ, or "no planner".
    """
    from common import timing
    path = timing.current_path()
    _STAGE_PRICES.append({"stage": str(stage), "bytes": float(price_bytes),
                          "path": path, "section": section or (path[-1] if path else None)})


def stage_prices() -> list[dict]:
    """Every price recorded this run, in call order."""
    return [dict(row) for row in _STAGE_PRICES]


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


_OVER_BUDGET_WARNED: dict[str, float] = {}


def warn_over_budget(stage: str, need_bytes: float, budget_bytes: float, *,
                     local: bool = False) -> None:
    """The one line a planner prints when its price exceeds its budget; the run goes on.

    Owner ruling 2026-10-01: a price over ``memory_per_device_gb`` (or a
    planner's tile) never stops a run.  The planner takes its smallest
    size, this warns once per stage kind and process (``stage`` less a
    trailing ``.N`` or ``:N`` counter; again only when a later call of that
    kind needs more), and the stage runs; if the device really lacks the
    room, the allocator OOMs.  Rank 0 warns for a rank-invariant
    price; ``local=True`` (a rank-local price) warns on the rank it is on.

    Rank 0's line is a :class:`RuntimeWarning`, so a production driver's
    report keeps it in its WARNINGS block (``runtime.production_stream``
    discards incidental stdout); another rank's local line goes to stderr,
    which production leaves untouched.
    """
    import re
    import sys
    import warnings
    from runtime import _resolve_proc_id
    kind = re.sub(r"([.:]\d+)+$", "", str(stage))
    need, have = float(need_bytes), float(budget_bytes)
    if need <= _OVER_BUDGET_WARNED.get(kind, -1.0):
        return
    _OVER_BUDGET_WARNED[kind] = need
    rank = _resolve_proc_id()
    if rank != 0 and not local:
        return
    where = f" (rank {rank})" if local else ""
    message = (f"memory over budget at {stage}{where}: needs {need / 1e9:.2f} GB/rank, "
               f"budget {have / 1e9:.2f} GB/rank, over by {(need - have) / 1e9:.2f} GB; "
               "continuing (an OOM is possible)")
    if rank == 0:
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    else:
        print("WARNING: " + message, file=sys.stderr, flush=True)


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


def get_device_memory_gb(linalg: str = "local") -> float:
    """This device's budget under the memory rule, decimal GB.

    ``runtime.planner_budget_bytes(T, linalg)``.  GPU: ``T`` is the card total
    (``cuDeviceTotalMem``).  CPU: ``T`` is the node's
    ``MemTotal`` over the processes on the node, and the budget is shared by
    the process's devices; collective, so every process must enter.
    Rank-local otherwise; the run budget is the minimum over processes
    (:func:`resolve_device_budget_gb`).
    """
    import jax
    from runtime import planner_budget_bytes
    if jax.default_backend() == 'cpu':
        total = (_meminfo_gb('MemTotal') or 0.0) * 1e9 / _processes_on_this_node()
        return planner_budget_bytes(total, linalg) / max(1, jax.local_device_count()) / 1e9
    from runtime.xla_memory import device_total_bytes
    device = jax.local_devices()[0]
    # The card total through the device leg's declared reader (lxkit.LEGS
    # device_memory_api: cuDeviceTotalMem on CUDA, hipDeviceTotalMem on ROCm).
    total = device_total_bytes(int(getattr(device, 'local_hardware_id', 0) or 0))
    if total is None:
        # No card total from the leg: the client's pool limit stands in for the card.
        total = int((device.memory_stats() or {}).get('bytes_limit', 0))
        from runtime.aot_memory import announce_once
        announce_once("gpu-budget-pool-limit",
                      f"device total unreadable (this device's leg declares no "
                      f"reader, or it did not answer); the XLA pool limit "
                      f"{total / 1e9:.2f} GB stands in for the card")
    budget = planner_budget_bytes(total, linalg)
    limit = _enforced_pool_limit(device)
    if limit:
        # A BFC arena refuses past its limit, so the pool's own overshoot
        # allowance has to fit inside it, as it fits inside the card.
        from runtime import MEMORY_MARGIN, POOL_OVERSHOOT
        budget = min(budget, limit / (1.0 + MEMORY_MARGIN * POOL_OVERSHOOT))
    return budget / 1e9


def _enforced_pool_limit(device) -> int:
    """The allocator's ``bytes_limit`` when it is a hard cap (a caller-owned
    BFC arena), else 0: under ``cuda_async`` (the runtime's pool policy) the
    reservation is a release threshold, not a limit, and the card total stands."""
    import os
    allocator = os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", "default").lower()
    if allocator not in ("default", "bfc"):
        return 0
    return int((device.memory_stats() or {}).get("bytes_limit", 0))


__all__ = ["admit_local", "device_budget_bytes", "get_device_memory_gb",
           "get_host_memory_available_gb", "host_bytes_per_process", "last_decision",
           "minimum_process_budget_gb", "record_route", "resolve_device_budget_gb", "routes"]

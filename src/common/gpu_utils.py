"""The run's device budget, and the planners' prices against it.

The budget is the one memory rule (``runtime.planner_budget_bytes``) on
the card total, so it is the same on every rank and every run; a positive
deck ``memory_per_device_gb`` is used as given.  No planner reads free memory, the
pool limit or a fragmentation factor (docs/architecture/memory-model.md,
"Budget").
"""


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

    A positive deck ``memory_per_device_gb`` is used as given; above the
    memory rule's budget for the deck's resolved ``linalg``
    (:func:`get_device_memory_gb`) it warns once, with both numbers.  Without
    one, the rule's budget.  The rule is the minimum over processes, so static
    tile shapes agree on every process.  Every process must enter.
    """
    rule = minimum_process_budget_gb(get_device_memory_gb(linalg))
    deck = float(deck_gb or 0.0)
    if deck > rule:
        import warnings
        from runtime import POOL_OVERSHOOT, _resolve_proc_id
        if _resolve_proc_id() == 0:
            warnings.warn(
                f"memory_per_device_gb = {deck:g} exceeds the memory rule's "
                f"{rule:.2f} GB on this card (linalg = {linalg}); the XLA pool has "
                f"overshot its budget by up to {100 * POOL_OVERSHOOT:.0f} % (P4) and "
                f"15 % (P64 CrI3 24x24); continuing at {deck:g} "
                "(docs/architecture/memory-model.md#budget)", RuntimeWarning, stacklevel=2)
    budget = deck if deck > 0 else rule
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


def _meminfo_gb(field: str) -> float | None:
    """This node's ``/proc/meminfo`` ``field`` in GB (1e9 B), or None if unreadable."""
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith(field + ':'):
                    return int(line.split()[1]) * 1024 / 1e9
    except (FileNotFoundError, ValueError, IndexError):
        pass
    return None


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
    from lxkit import device_vendor
    from runtime.xla_memory import cuda_device_total_bytes
    device = jax.local_devices()[0]
    total = (cuda_device_total_bytes(int(getattr(device, 'local_hardware_id', 0) or 0))
             if device_vendor(device) == 'cuda' else None)
    if total is None:
        # No CUDA card total: the client's pool limit stands in for the card.
        total = int((device.memory_stats() or {}).get('bytes_limit', 0))
        from runtime.aot_memory import announce_once
        announce_once("gpu-budget-pool-limit",
                      f"device total unreadable (not a CUDA device); the XLA pool "
                      f"limit {total / 1e9:.2f} GB stands in for the card")
    return planner_budget_bytes(total, linalg) / 1e9


__all__ = ["device_budget_bytes", "get_device_memory_gb", "get_host_memory_available_gb",
           "host_bytes_per_process", "minimum_process_budget_gb", "resolve_device_budget_gb"]

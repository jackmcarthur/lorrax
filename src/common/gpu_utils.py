"""Memory detection helpers for JAX device budget.

Used by ``gw.gw_init`` and ``gw.gw_config`` to size chunking parameters.
This module also owns the one BFC fragmentation target shared by planners
whose large stage allocations scale with a small field-width factor, and the
worst-process residency reduction required when allocator state sizes static
multi-process control flow.
"""

import os
import subprocess


# ============================================================================
# Memory Detection for Auto-sizing Chunk Parameters
# ============================================================================

def bfc_fragmentation_target_utilization(width_factor: int) -> float:
    """Return the conservative stage fraction of an available BFC budget.

    This is a *second* bound after live available memory has been measured;
    it is not an estimate of array bytes.  A stage can fit by arithmetic and
    still fail when BFC must place one large arena among earlier transient
    allocations.  ``width_factor`` is the caller-visible multiplier of the
    stage's large buffers (the physics meaning stays at the caller).

    The table is measured production policy, formerly private to the G-flat
    planner: factor 4 at 0.85 failed with a 23-GB single arena on a 40-GB
    device, while 0.78 fit.  Factors 2 and 1 retain progressively more of the
    budget.  Keeping the table here prevents independent GW and htransform
    copies from drifting.
    """
    try:
        factor = int(width_factor)
    except (TypeError, ValueError):
        raise ValueError(
            f"width_factor must be a positive integer, got {width_factor!r}") \
            from None
    if factor != width_factor or factor <= 0:
        raise ValueError(
            f"width_factor must be a positive integer, got {width_factor!r}")
    if factor >= 4:
        return 0.78
    if factor == 2:
        return 0.85
    return 0.90


def worst_process_resident_bytes(local_bytes: int) -> int:
    """Return one rank-invariant allocator-residency floor.

    Allocator residency is process-local and can differ because JIT arenas are
    released asynchronously.  Any value that sizes a static executable or
    host-loop shape must therefore be derived from the same worst-process
    floor on every rank.  Keep the communication in the canonical process-
    collective service and the memory policy here.
    """
    import numpy as np

    from common.collectives import all_gather_processes

    local_i = int(local_bytes)
    if local_i < 0:
        raise ValueError(f"resident bytes must be nonnegative, got {local_i}")
    gathered = np.asarray(
        all_gather_processes(np.asarray(local_i, dtype=np.int64)),
        dtype=np.int64,
    )
    if gathered.size == 0 or np.any(gathered < 0):
        raise ValueError(
            "process residency gather returned no values or a negative value"
        )
    return int(np.max(gathered))


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
    """Record the run's per-device budget (``memory_per_device_gb``, decimal GB), once,
    when the deck value (or the collective auto-detection at 0) resolves."""
    global _RUN_DEVICE_BUDGET_GB
    value = float(gb)
    if not value > 0:
        raise ValueError(f"the device budget must be positive GB, got {gb!r}")
    _RUN_DEVICE_BUDGET_GB = value


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


def _query_nvidia_smi_memory(field: str) -> int | None:
    """Query this rank's visible GPU memory field, returned in bytes."""
    try:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        gpu_id = visible.split(",", 1)[0].strip() if visible else "0"
        result = subprocess.run(
            ['nvidia-smi', f'--id={gpu_id}', f'--query-gpu={field}',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            value_mib = float(result.stdout.strip().split('\n')[0])
            return int(value_mib * 2**20)  # nvidia-smi reports MiB
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


def get_gpu_memory_nvidia_smi() -> float | None:
    """Query currently free GPU memory via nvidia-smi (GB = 1e9 B)."""
    free = _query_nvidia_smi_memory('memory.free')
    return None if free is None else free / 1e9


def get_gpu_used_memory_bytes_nvidia_smi() -> int | None:
    """Conservative whole-device residency for this rank's visible GPU.

    Unlike JAX allocator accounting this includes the CUDA context, library
    workspaces and any unrelated process sharing the device.  It is therefore
    an upper bound suitable for a capacity refusal, not an attribution tool.
    """
    return _query_nvidia_smi_memory('memory.used')


def _get_jax_gpu_memory_bytes() -> tuple[float | None, float | None, float | None]:
    """Return (bytes_limit, bytes_in_use, bytes_available) for **this process's** JAX device.

    ``jax.local_devices()``, NOT ``jax.devices()``: the latter is the GLOBAL
    device list, so ``jax.devices()[0]`` is process 0's device on every rank
    (the hazard ``common.wfn_transforms.process_local_mesh`` names).  A
    non-addressable device's ``memory_stats()`` does not describe this rank's
    pool, and this function feeds MEMORY BUDGETS — every rank but 0 would have
    been sizing its chunks against another process's allocator.  At ``P == 1``
    the two lists are the same object, so single-process behaviour is
    unchanged.
    """
    try:
        import jax
        import jax.numpy as jnp
        _ = jnp.zeros(1).block_until_ready()
        devices = jax.local_devices()
        if not devices or not hasattr(devices[0], 'memory_stats'):
            return None, None, None
        stats = devices[0].memory_stats()
        bytes_limit = float(stats.get('bytes_limit', 0.0))
        bytes_in_use = float(stats.get('bytes_in_use', 0.0))
        if bytes_limit <= 0.0:
            return None, None, None
        bytes_available = max(0.0, bytes_limit - max(0.0, bytes_in_use))
        return bytes_limit, bytes_in_use, bytes_available
    except Exception:
        return None, None, None


#: XLA's ``GpuAllocatorConfig.memory_fraction`` when neither
#: ``XLA_CLIENT_MEM_FRACTION`` nor ``XLA_PYTHON_CLIENT_MEM_FRACTION`` is set.
_XLA_DEFAULT_MEM_FRACTION = 0.75


def _derived_pool_bytes() -> tuple[int | None, int | None, str]:
    """(limit, in_use, source) when the live client reports no ``bytes_limit``.

    jaxlib 0.9's ``cuda_async`` client reports ``bytes_limit`` only when its
    pool is reserved (``PREALLOCATE=true``, the runtime's GPU pool policy):
    unreserved, ``bytes_limit`` is 0 (the 2026-09-23 cuda_async logs,
    runs/runtime/zeta_mubatch_20260923 in the sandbox).  The limit is then
    derived as the reservation would have been, memory_fraction x total
    device memory, in bytes, honouring the fraction variable jaxlib reads
    (:func:`runtime.xla_memory.resolve_xla_gpu_memory_env`).  ``in_use`` is
    the bytes of this process's live arrays on its device.  It is not
    nvidia-smi's used memory, because the async pool keeps freed blocks
    reserved and nvidia-smi counts them as used.
    """
    from runtime.xla_memory import resolve_xla_gpu_memory_env

    total = _query_nvidia_smi_memory('memory.total')
    if total is None:
        return None, None, 'nvidia-smi memory.total unavailable'
    env = resolve_xla_gpu_memory_env()
    fraction = (float(env.mem_fraction) if env.mem_fraction
                else _XLA_DEFAULT_MEM_FRACTION)
    in_use = _live_array_bytes()
    source = (f"{env.mem_fraction_var or 'XLA default fraction'} {fraction:g} "
              f"x nvidia-smi memory.total {total/1e9:.2f} GB "
              f"(live arrays in_use={0.0 if in_use is None else in_use/1e9:.2f} GB)")
    return int(fraction * total), in_use, source


def _live_array_bytes() -> int | None:
    """Bytes of this process's live JAX arrays on its first local device."""
    try:
        import math
        import jax
        device = jax.local_devices()[0]
        total = 0
        for array in jax.live_arrays():
            if array.is_deleted() or device not in array.sharding.addressable_devices:
                continue
            total += (math.prod(array.sharding.shard_shape(array.shape))
                      * array.dtype.itemsize)
        return int(total)
    except Exception:
        return None


def get_cpu_memory_total() -> float | None:
    """Query total system memory in GB (or None if unavailable)."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        pass

    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    parts = line.split()
                    kb = int(parts[1])
                    return kb / (1024**2)
    except (FileNotFoundError, ValueError, IndexError):
        pass

    return None


def get_host_memory_available_gb() -> float | None:
    """This node's ``MemAvailable`` in GB (1e9 B), or None if unreadable.

    Whole-node, not per process: a caller whose processes share the node
    divides it among them.
    """
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024 / 1e9
    except (FileNotFoundError, ValueError, IndexError):
        pass
    return None


def host_bytes_per_process() -> float:
    """0.9 of the node's live ``MemAvailable`` over the processes sharing the
    node, in bytes, agreed (minimum) across processes. Every process must
    enter this call.
    """
    import socket
    import zlib
    import numpy as np
    from common.collectives import all_gather_processes
    avail_gb = get_host_memory_available_gb()
    host = zlib.crc32(socket.gethostname().encode())
    hosts = np.asarray(all_gather_processes(np.asarray(host, dtype=np.int64)))
    per_node = max(1, int(np.sum(hosts == host)))
    local_gb = float('inf') if avail_gb is None else 0.9 * avail_gb / per_node
    return minimum_process_budget_gb(min(local_gb, 1e12)) * 1e9


def get_device_memory_gb(n_devices: int | None = None) -> float:
    """Get per-device memory budget in GB for JAX computations.

    GPU policy: budget = 0.9 * bytes_limit from jax.memory_stats().
    Uses bytes_limit (pool size, constant across ranks) rather than
    bytes_available (which can vary by rank due to JIT timing).  When the
    client reports no limit (``cuda_async``, ``platform``), the limit is
    MEM_FRACTION x total device memory (:func:`_derived_pool_bytes`).
    GB means 1e9 bytes everywhere in this module.
    """
    try:
        import jax
        backend = jax.default_backend()
        if n_devices is None:
            n_devices = jax.device_count()
    except ImportError:
        backend = 'cpu'
        if n_devices is None:
            n_devices = 1

    if backend in ('gpu', 'cuda'):
        bytes_limit, _, _ = _get_jax_gpu_memory_bytes()
        if bytes_limit is None:
            bytes_limit, _, source = _derived_pool_bytes()
            from runtime.aot_memory import announce_once
            announce_once(
                "gpu-budget-derived-limit",
                "XLA client reports no bytes_limit; device limit "
                + (f"{bytes_limit/1e9:.2f} GB = {source}" if bytes_limit
                   else f"unknown ({source}), budget from nvidia-smi free"))
        if bytes_limit is not None and bytes_limit > 0:
            return max(0.1, 0.90 * bytes_limit / 1e9)

        # Last resort when not even the device total is readable.
        mem_free_gb = get_gpu_memory_nvidia_smi()
        if mem_free_gb is not None:
            return max(0.1, mem_free_gb * 0.90)

        return 4.0

    else:  # CPU backend
        total_mem = get_cpu_memory_total()
        if total_mem is not None:
            usable = total_mem * 0.9
            return usable / max(1, n_devices)

        return 4.0


def get_device_memory_info() -> dict:
    """Get detailed memory information for current JAX backend.

    Returns a dict with keys:
    - backend: 'gpu' or 'cpu'
    - total_gb: total visible memory per device in GB (if known)
    - available_gb: currently available memory per device in GB
    - budget_gb: memory budget used by get_device_memory_gb
    - source: detection source
    - n_devices: number of JAX devices
    """
    try:
        import jax
        backend = jax.default_backend()
        n_devices = jax.device_count()
    except ImportError:
        backend = 'cpu'
        n_devices = 1

    source = 'default'
    total_gb = 8.0
    available_gb = 4.0
    budget_gb = 4.0

    if backend in ('gpu', 'cuda'):
        bytes_limit, bytes_in_use, bytes_available = _get_jax_gpu_memory_bytes()
        if bytes_limit is not None and bytes_available is not None:
            source = f'jax.memory_stats (in_use={bytes_in_use/1e9:.2f} GB)'
        else:
            bytes_limit, bytes_in_use, source = _derived_pool_bytes()
            bytes_available = (None if bytes_limit is None or bytes_in_use is None
                               else max(0, bytes_limit - bytes_in_use))
        if bytes_limit is not None and bytes_available is not None:
            total_gb = bytes_limit / 1e9
            available_gb = bytes_available / 1e9
            budget_gb = max(0.1, 0.90 * available_gb)
        else:
            mem_free_gb = get_gpu_memory_nvidia_smi()
            mem_total = _query_nvidia_smi_memory('memory.total')
            if mem_free_gb is not None:
                available_gb = mem_free_gb
                budget_gb = max(0.1, mem_free_gb * 0.90)
                if mem_total is not None:
                    total_gb = mem_total / 1e9
                source = 'nvidia-smi memory.free'
    else:
        try:
            import psutil
            total_gb = psutil.virtual_memory().total / (1024**3)
            available_gb = total_gb / max(1, n_devices)
            budget_gb = available_gb * 0.90
            total_gb = available_gb
            source = 'psutil'
        except ImportError:
            mem = get_cpu_memory_total()
            if mem is not None:
                available_gb = mem / max(1, n_devices)
                budget_gb = available_gb * 0.90
                total_gb = available_gb
                source = '/proc/meminfo'

    return {
        'backend': backend,
        'total_gb': total_gb,
        'available_gb': available_gb,
        'budget_gb': budget_gb,
        'source': source,
        'n_devices': n_devices,
    }


__all__ = ["bfc_fragmentation_target_utilization",
           "get_device_memory_gb", "get_device_memory_info",
           "get_host_memory_available_gb",
           "get_gpu_memory_nvidia_smi",
           "get_gpu_used_memory_bytes_nvidia_smi", "get_cpu_memory_total",
           "worst_process_resident_bytes"]

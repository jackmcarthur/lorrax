"""Memory detection helpers for JAX device budget.

Used by ``gw.gw_init`` and ``gw.gw_config`` to size chunking parameters.
This module also owns the one BFC fragmentation target shared by planners
whose large stage allocations scale with a small field-width factor.
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

def _query_nvidia_smi_memory(field: str) -> float | None:
    """Query a single GPU memory field (in MiB) from nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', f'--query-gpu={field}', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            value_mib = float(result.stdout.strip().split('\n')[0])
            return value_mib / 1024.0  # MiB -> GB
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


def get_gpu_memory_nvidia_smi() -> float | None:
    """Query currently free GPU memory via nvidia-smi (GB)."""
    return _query_nvidia_smi_memory('memory.free')


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


def get_device_memory_gb(n_devices: int | None = None) -> float:
    """Get per-device memory budget in GB for JAX computations.

    GPU policy: budget = 0.9 * bytes_limit from jax.memory_stats().
    Uses bytes_limit (pool size, constant across ranks) rather than
    bytes_available (which can vary by rank due to JIT timing).
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
        if bytes_limit is not None and bytes_limit > 0:
            return max(0.1, 0.90 * bytes_limit / 1e9)

        # Fallback to currently free memory if JAX stats are unavailable.
        # WARNING: With PREALLOCATE=true, nvidia-smi free memory is AFTER the
        # BFC pool allocation, so it's much smaller than the actual pool.
        import sys
        print(f"  [gpu_utils] WARNING: bytes_limit={bytes_limit}, falling back to nvidia-smi "
              f"(pid={os.getpid()})", file=sys.stderr, flush=True)
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
            total_gb = bytes_limit / 1e9
            available_gb = bytes_available / 1e9
            budget_gb = max(0.1, 0.90 * available_gb)
            source = f'jax.memory_stats (in_use={bytes_in_use/1e9:.2f} GB)'
        else:
            mem_free_gb = get_gpu_memory_nvidia_smi()
            mem_total_gb = _query_nvidia_smi_memory('memory.total')
            if mem_free_gb is not None:
                available_gb = mem_free_gb
                budget_gb = max(0.1, mem_free_gb * 0.90)
                if mem_total_gb is not None:
                    total_gb = mem_total_gb
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
           "get_gpu_memory_nvidia_smi", "get_cpu_memory_total"]

import os
import subprocess
import numpy as np

# CPU-first default; enable CUDA explicitly via ISDF_ENABLE_CUDA=1
_ENABLE_CUDA = str(os.environ.get("ISDF_ENABLE_CUDA", "")).lower() in ("1", "true", "yes", "on")

if _ENABLE_CUDA:
    try:
        import cupy as _cupy
        import cupyx.scipy.fft as _cufft
        _cupy.cuda.runtime.getDeviceCount()
        cp = _cupy
        cufft = _cufft
        xp = cp
        GPU_AVAILABLE = True
    except Exception:
        cp = np
        cufft = np.fft
        xp = cp
        GPU_AVAILABLE = False
else:
    cp = np
    cufft = np.fft
    xp = cp
    GPU_AVAILABLE = False

# Minimal compatibility surface
def _asnumpy(x):
    return x
cp.asarray = getattr(cp, 'asarray', np.asarray)
cp.asnumpy = getattr(cp, 'asnumpy', _asnumpy)
def _get_array_module(x):
    return np
cp.get_array_module = getattr(cp, 'get_array_module', _get_array_module)

if not _ENABLE_CUDA or not GPU_AVAILABLE:
    class _DummyRuntime:
        def getDeviceCount(self):
            return 0
        def memGetInfo(self):
            return (0, 0)
        def getDeviceProperties(self, _):
            return {"name": "CPU"}
    class _DummyCuda:
        runtime = _DummyRuntime()
        def is_available(self):
            return False
    cp.cuda = getattr(cp, 'cuda', _DummyCuda())


# ============================================================================
# Memory Detection for Auto-sizing Chunk Parameters
# ============================================================================

def get_gpu_memory_nvidia_smi() -> float | None:
    """Query GPU memory via nvidia-smi.
    
    Returns:
        Total GPU memory in GB, or None if nvidia-smi unavailable.
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            # Returns memory in MiB; take first GPU
            mem_mib = float(result.stdout.strip().split('\n')[0])
            return mem_mib / 1024.0  # Convert to GB
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


def get_cpu_memory_total() -> float | None:
    """Query total system memory.
    
    Returns:
        Total system memory in GB, or None if unavailable.
    """
    # Try psutil first (most reliable)
    try:
        import psutil
        return psutil.virtual_memory().total / (1024**3)
    except ImportError:
        pass
    
    # Fallback: parse /proc/meminfo on Linux
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    # Format: "MemTotal:       16384000 kB"
                    parts = line.split()
                    kb = int(parts[1])
                    return kb / (1024**2)  # kB to GB
    except (FileNotFoundError, ValueError, IndexError):
        pass
    
    return None


def get_device_memory_gb(n_devices: int = 1) -> float:
    """Get available memory per device for JAX computations.
    
    Detection strategy:
    1. If JAX backend is 'gpu' or 'cuda': use nvidia-smi with 50% factor
       (CUDA driver, XLA runtime, and fragmentation consume ~50%)
    2. If JAX backend is 'cpu': use system RAM / n_devices with 80% factor
    3. Fallback: return conservative 4 GB default
    
    Args:
        n_devices: Number of devices to divide memory among (for CPU)
    
    Returns:
        Memory per device in GB (usable for computation)
    """
    # Lazy import to avoid circular dependencies
    try:
        import jax
        backend = jax.default_backend()
    except ImportError:
        backend = 'cpu'
    
    if backend in ('gpu', 'cuda'):
        # For GPU, try nvidia-smi first (works even with XLA device count override)
        mem = get_gpu_memory_nvidia_smi()
        if mem is not None:
            # GPU memory is significantly reduced by:
            # - CUDA driver overhead (~500 MB)
            # - XLA runtime and JIT compilation buffers
            # - Memory fragmentation during computation
            # Empirically, ~50% of total is safely usable
            return mem * 0.50
        
        # Fallback to JAX device info if available
        try:
            import jax
            devices = jax.devices()
            if devices and hasattr(devices[0], 'memory_stats'):
                # Some backends expose memory stats
                stats = devices[0].memory_stats()
                if 'bytes_limit' in stats:
                    return stats['bytes_limit'] / (1024**3) * 0.50
        except Exception:
            pass
        
        # Default GPU memory if detection fails
        return 4.0
    
    else:  # CPU backend
        total_mem = get_cpu_memory_total()
        if total_mem is not None:
            # For CPU, divide total memory among logical devices
            # Leave 20% for OS/other processes
            usable = total_mem * 0.8
            return usable / max(1, n_devices)
        
        # Conservative default
        return 4.0


def get_device_memory_info() -> dict:
    """Get detailed memory information for current JAX backend.
    
    Returns:
        Dictionary with:
        - backend: 'gpu' or 'cpu'
        - total_gb: Total memory per device in GB
        - source: How memory was detected ('nvidia-smi', 'psutil', '/proc/meminfo', 'default')
        - n_devices: Number of JAX devices
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
    
    if backend in ('gpu', 'cuda'):
        mem = get_gpu_memory_nvidia_smi()
        if mem is not None:
            total_gb = mem * 0.50  # 50% factor for GPU (driver + XLA + fragmentation)
            source = 'nvidia-smi (50% usable)'
    else:
        # Try psutil
        try:
            import psutil
            total_gb = psutil.virtual_memory().total / (1024**3)
            total_gb = total_gb * 0.8 / max(1, n_devices)
            source = 'psutil'
        except ImportError:
            # Try /proc/meminfo
            mem = get_cpu_memory_total()
            if mem is not None:
                total_gb = mem * 0.8 / max(1, n_devices)
                source = '/proc/meminfo'
    
    return {
        'backend': backend,
        'total_gb': total_gb,
        'source': source,
        'n_devices': n_devices,
    }


__all__ = ["cp", "xp", "cufft", "GPU_AVAILABLE", 
           "get_device_memory_gb", "get_device_memory_info"]

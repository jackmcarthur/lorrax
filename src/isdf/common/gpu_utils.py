import os
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

__all__ = ["cp", "xp", "cufft", "GPU_AVAILABLE"]

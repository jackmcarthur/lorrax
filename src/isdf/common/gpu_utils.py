import types
import numpy as np

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

    def _asnumpy(x):
        return x
    cp.asarray = np.asarray
    cp.asnumpy = _asnumpy
    def _get_array_module(x):
        return np
    cp.get_array_module = _get_array_module

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
    cp.cuda = _DummyCuda()

try:
    import fftx as _fftx
    fftx = _fftx
except Exception:
    fftx = types.SimpleNamespace(fft=np.fft)

__all__ = ["cp", "xp", "cufft", "fftx", "GPU_AVAILABLE"]

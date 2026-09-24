"""The CUDA FFT leg's Python-side table — **C++ only, by design.**

``cpp/cufft/kconv_mathdx_cuda_ffi.cc`` is the whole CUDA FFT leg since
2026-09-24: the nvidia-mathdx k-convolution family (the ``lorrax_mathdx_*``
targets below), reached only through the ``ffi.fft`` router, which never
lowers them on a cpu mesh.  It replaced the cuFFT strided flat-k/gw_conv
handler and the direct-DFT conv_klead/conv_kminor kernels (decisions.md
2026-09-24).  The directory keeps its historical ``cufft`` name.  The Python
for both platforms lives in ``ffi.fft``; the names below are re-exported so a
reader arriving here can confirm the table without opening the loader.
"""

#: The target strings this library's FFT TU registers: the CUDA-only
#: nvidia-mathdx k-convolution family (``ffi.fft`` router, decisions.md
#: 2026-09-24).  The flat-k transform target ``lorrax_mklfft_flat_k`` is
#: host-only since the cuFFT strided handler was replaced (same date).
CUDA_TARGETS = ("lorrax_mathdx_kconv_pair", "lorrax_mathdx_kconv_parent",
                "lorrax_mathdx_kconv_plane", "lorrax_mathdx_kconv_klead", "lorrax_mathdx_kfft_klead",
                "lorrax_mathdx_kconv_kminor", "lorrax_mathdx_kfft_kminor")

#: target → the C++ symbol THIS library exports (host exports different
#: symbols for the same targets; see ``ffi_loader._CUDA_TARGET_SYMBOLS``).
CUDA_SYMBOLS = {
    "lorrax_mathdx_kconv_pair":   "KConvMathdxPairCudaFfi",
    "lorrax_mathdx_kconv_parent": "KConvMathdxParentCudaFfi",
    "lorrax_mathdx_kconv_plane":  "KConvMathdxPlaneCudaFfi",
    "lorrax_mathdx_kconv_klead":  "KConvMathdxKleadCudaFfi",
    "lorrax_mathdx_kfft_klead":   "KFftMathdxKleadCudaFfi",
    "lorrax_mathdx_kconv_kminor": "KConvMathdxKminorCudaFfi",
    "lorrax_mathdx_kfft_kminor":  "KFftMathdxKminorCudaFfi",
}

__all__ = ["CUDA_TARGETS", "CUDA_SYMBOLS"]

_REDIRECT = frozenset({
    "make_flat_k_fft_ffi", "require_fft_ffi",
    "fft_ffi_enabled", "fft_ffi_mode", "GATE", "flat_k",
})


def __getattr__(name: str):
    """Point the obvious wrong import at the right module instead of a bare
    ``ImportError``.  Someone WILL type ``from ffi.cufft import
    make_flat_k_fft_ffi``; the module docstring explains why that name is
    not here, but only if they read it."""
    if name in _REDIRECT:
        raise AttributeError(
            f"ffi.cufft has no {name!r}: the cuFFT handlers register the "
            f"SAME XLA target strings as the host MKL-DFTI handlers, so ONE "
            f"platform-agnostic wrapper serves both platforms.  Import it "
            f"from ffi.fft instead (`from ffi.fft import {name}`) — "
            f"it lowers to cuFFT on a CUDA mesh.  See ffi/cufft/__init__.py.")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

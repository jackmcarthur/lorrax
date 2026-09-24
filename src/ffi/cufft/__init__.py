"""cuFFT strided flat-k FFI subpackage — **C++ only, by design.**

This package has no ``shard_map``/``ffi_call`` wrapper of its own, and that
is a deliberate design choice rather than a gap.  ``fft_flat_k_cuda_ffi.cc``
registers the SAME XLA target STRINGS as the host MKL-DFTI handlers —

    lorrax_mklfft_flat_k    CufftFlatKCudaFfi   (CUDA)
                            MklFftFlatKHostFfi  (host)

— so ONE platform-agnostic ``jax.ffi.ffi_call`` per site resolves the right
handler from the LOWERING platform, exactly the way jaxlib splits its own
cpu (lapack) vs CUDA (cusolver) kernels and the way ``ffi/phdf5`` does it
in-tree.  A ``ffi/cufft/flat_k.py`` mirroring ``ffi/fft.py`` would
duplicate every line of it and force call sites to branch on a platform they
are not supposed to know about.

THE ``lorrax_mathdx_*`` TARGETS ARE **NOT** MIRRORED, and their names say so:
they are the nvidia-mathdx k-convolution family
(``cpp/cufft/kconv_mathdx_cuda_ffi.cc``), the CUDA leg of the ``ffi.fft``
router, which never lowers them on a cpu mesh (it takes the plan route there).
They replaced the cuFFT strided gw_conv handler and the hand-written direct-DFT
conv_klead/conv_kminor kernels on 2026-09-24 (decisions.md).

**The Python for both platforms lives in** ``ffi.fft`` (the target strings
were coined by the CPU prototype and kept; the name is historical, the
dispatch is not).  The target names below are re-exported so a reader
arriving here can confirm the mirror without opening the loader.

What IS cuFFT-specific, and lives only in ``cpp/``:

* a grow-only ``cudaMalloc`` arena OUTSIDE the XLA allocator, with cuFFT
  auto-allocation disabled (``fft_flat_k_cuda_ffi.cc:435``, ``:401-425``)
  and a ``cudaDeviceSynchronize()`` before every growth (``:412``);
* plan-cache / arena / enqueue serialized under one process mutex, with the
  single-compute-stream assumption stated honestly (``:72-78``, ``:519``);
* NVRTC compilation of the norm-scale kernel (no nvcc at build time);
* the MKL host handler's compact-chunk L2 staging is deliberately NOT
  mirrored here (``:20-23``: "the host engine's per-thread L2 buffer was a
  CLX cache artifact, not part of the contract") — the right call, and the
  right way to say it.

Those three memory/concurrency models (cuFFT plan workspace, MKL OpenMP
chunking, a scratch-free BLAS call) are why there is no shared C++ handler
base either: see ``src/ffi/TEMPLATE.md:188-195``.

GPU evidence for this handler: job 7879378, 1× Quadro RTX 5000 sm_75,
15-case correctness gate at 0-3.7e-16 (``wk_REL/cufft_unit.log:11-33``).
Multi-GPU / sharded meshes are UNMEASURED — every GPU log is
``[CudaDevice(id=0)]``.
"""

#: The target strings this library's cuFFT/mathdx TUs register.  The first is
#: identical to the host table; the rest are the CUDA-only nvidia-mathdx
#: k-convolution family (``ffi.fft`` router, decisions.md 2026-09-24).
CUDA_TARGETS = ("lorrax_mklfft_flat_k",
                "lorrax_mathdx_kconv_pair", "lorrax_mathdx_kconv_parent",
                "lorrax_mathdx_kconv_klead", "lorrax_mathdx_kfft_klead",
                "lorrax_mathdx_kconv_kminor", "lorrax_mathdx_kfft_kminor")

#: target → the C++ symbol THIS library exports (host exports different
#: symbols for the same targets; see ``ffi_loader._CUDA_TARGET_SYMBOLS``).
CUDA_SYMBOLS = {
    "lorrax_mklfft_flat_k":       "CufftFlatKCudaFfi",
    "lorrax_mathdx_kconv_pair":   "KConvMathdxPairCudaFfi",
    "lorrax_mathdx_kconv_parent": "KConvMathdxParentCudaFfi",
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

"""Flat-k batched 3-D FFT FFI subpackage (cpu MKL DFTI + CUDA cuFFT).

Public API::

    from ffi.mklfft import (
        fft_ffi_enabled,          # LORRAX_FFT_FFI          (factory-time)
        fused_fft_ffi_enabled,    # LORRAX_FFT_FFI_FUSED    (factory-time)
        require_fft_ffi,          # announce-or-REFUSE on a mesh
        make_flat_k_fft_ffi,      # (nk,*trail) -> (nk,*trail)
        make_gw_conv_ffi,         # fused ifftn·(G·W)·fftn
    )

ONE Python module serves BOTH platforms on purpose: the host and CUDA
libraries register the same target strings under different C++ symbols, so
the ``ffi_call`` sites never name a platform.  ``src/ffi/cufft/`` therefore
holds C++ only — see ``ffi/cufft/__init__.py`` for that design note.

Contract: ``docs/dev/flat_k_fft_service.md``.  Gate doctrine:
``docs/dev/ffi_gate_contract.md``.
"""
from .flat_k import (
    FLAT_K_TARGET,
    FUSED_GATE,
    GATE,
    GW_CONV_TARGET,
    ffi_fft_scale,
    fft_ffi_enabled,
    fft_ffi_mode,
    fused_fft_ffi_enabled,
    fused_fft_ffi_mode,
    make_flat_k_fft_ffi,
    make_gw_conv_ffi,
    require_fft_ffi,
    validate_flat_spec,
)

__all__ = [
    "FLAT_K_TARGET", "GW_CONV_TARGET", "GATE", "FUSED_GATE",
    "fft_ffi_enabled", "fft_ffi_mode",
    "fused_fft_ffi_enabled", "fused_fft_ffi_mode",
    "require_fft_ffi", "make_flat_k_fft_ffi", "make_gw_conv_ffi",
    "ffi_fft_scale", "validate_flat_spec",
]

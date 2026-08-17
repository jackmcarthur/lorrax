"""Re-export shim — the flat-k FFT service moved to ``ffi.fft``.

Retained one wave so existing ``ffi.mklfft`` imports keep working
(docs/architecture/ffi_layout.md §4); new code imports ``ffi.fft``.  The
vendor name was historical anyway: ONE python module serves BOTH platforms
(MKL DFTI on cpu, cuFFT on CUDA) under the same jax.ffi target names —
see ``ffi/fft.py``'s docstring.  C++ handlers: ``src/ffi/cpp/mklfft/`` and
``src/ffi/cpp/cufft/``.
"""
import sys as _sys

from ffi import fft as flat_k  # noqa: F401  (ffi.mklfft.flat_k module alias)
from ffi.fft import (  # noqa: F401
    FLAT_K_TARGET,
    FUSED_GATE,
    GATE,
    GW_CONV_TARGET,
    GW_CONV_REAL_W_TARGET,
    ffi_fft_scale,
    fft_ffi_enabled,
    fft_ffi_mode,
    fused_fft_ffi_enabled,
    fused_fft_ffi_mode,
    make_flat_k_fft_ffi,
    make_gw_conv_ffi,
    make_gw_conv_real_w_ffi,
    require_fft_ffi,
    validate_flat_spec,
)

# `from ffi.mklfft.flat_k import X` must keep working for the shim wave.
_sys.modules[__name__ + ".flat_k"] = flat_k

__all__ = [
    "FLAT_K_TARGET", "GW_CONV_TARGET", "GW_CONV_REAL_W_TARGET",
    "GATE", "FUSED_GATE",
    "fft_ffi_enabled", "fft_ffi_mode",
    "fused_fft_ffi_enabled", "fused_fft_ffi_mode",
    "require_fft_ffi", "make_flat_k_fft_ffi", "make_gw_conv_ffi",
    "make_gw_conv_real_w_ffi",
    "ffi_fft_scale", "validate_flat_spec",
]

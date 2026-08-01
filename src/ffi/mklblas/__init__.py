"""Re-export shim — the vendor-BLAS batched-GEMM service moved to ``ffi.gemm``.

Retained one wave so existing ``ffi.mklblas`` imports keep working
(docs/architecture/ffi_layout.md §4); new code imports ``ffi.gemm``.  The
public names keep the ``bands_gemm`` prefix they have carried since the
dial shipped.  Not a general GEMM service — see ``ffi/gemm.py``'s docstring
and ``docs/dev/vendor_gemm_service.md``.  C++ handler:
``src/ffi/cpp/mklblas/gemm_batch_ffi.cc``.
"""
import sys as _sys

from ffi import gemm as gemm_module
from ffi.gemm import (  # noqa: F401
    GATE,
    GEMM_TARGET,
    gemm_batch,
    gemm_ffi_enabled as bands_gemm_ffi_enabled,
    gemm_ffi_mode as bands_gemm_ffi_mode,
    require_gemm_ffi as require_bands_gemm_ffi,
)

# `from ffi.mklblas.gemm import X` must keep working for the shim wave.
_sys.modules[__name__ + ".gemm"] = gemm_module
del gemm_module

__all__ = [
    "GATE", "GEMM_TARGET", "gemm_batch",
    "bands_gemm_ffi_enabled", "bands_gemm_ffi_mode",
    "require_bands_gemm_ffi",
]

"""Vendor-BLAS batched-GEMM FFI subpackage (host platform only).

Public API::

    from ffi.mklblas import (
        bands_gemm_ffi_enabled,   # the LORRAX_BANDS_GEMM_FFI dial (tier 1)
        bands_gemm_ffi_mode,      # its raw grammar: on | off | auto
        require_bands_gemm_ffi,   # announce-or-REFUSE for an explicit =1
        gemm_batch,               # C[i] = A[i] @ B[i % BB]
    )

The public names keep the ``bands_gemm`` prefix they have carried since the
dial shipped — they are re-exported unchanged from
``common.contract_bands`` (``__all__`` there, and
``tests/test_contract_bands.py:56-57`` imports them), so this move is a
relocation of the implementation, not a rename of the API.

Not a general GEMM service — see ``gemm.py``'s docstring and
``docs/dev/vendor_gemm_service.md``.
"""
from .gemm import (
    GATE,
    GEMM_TARGET,
    gemm_batch,
    gemm_ffi_enabled as bands_gemm_ffi_enabled,
    gemm_ffi_mode as bands_gemm_ffi_mode,
    require_gemm_ffi as require_bands_gemm_ffi,
)

__all__ = [
    "GATE", "GEMM_TARGET", "gemm_batch",
    "bands_gemm_ffi_enabled", "bands_gemm_ffi_mode",
    "require_bands_gemm_ffi",
]

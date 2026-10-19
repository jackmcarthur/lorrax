"""LORRAX FFI subpackage: JAX ↔ external parallel linear algebra libraries.

See `AGENTS.md` for the directory layout and how to add a new target.
"""


#: The per-process dials, by name.  Each is read from ``os.environ`` at
#: kernel-FACTORY time and changes the emitted HLO body (one vendor-GEMM
#: ``ffi_call`` against a native ``dot``), so a rank whose dial differs from
#: its peers' compiles a different module.  Consumers fold :func:`ffi_dial_key`
#: into their kernel cache keys.  Add a dial here when you add one below.
FFI_DIAL_ENV = (
    "LORRAX_BANDS_GEMM_FFI",
)
# The k-convolution router has no dial: it chooses its backend from the
# mesh's device vendor only, so it adds nothing here.


def ffi_dial_key() -> tuple:
    """The one cache-key component capturing every factory-time FFI dial.

    ``contract_bands`` reads its backend dial at factory time, so a kernel
    cache that omits it serves a stale backend after a mid-process flip.
    Consumers fold the returned tuple into their cache keys
    (``gw.ppm_tau_kernel``, ``gw.cohsex_sigma``, ``gw.w_isdf``).  Tier-1
    lexical: no JAX backend init.
    """
    from ffi.gemm import gemm_ffi_enabled as bands_gemm_ffi_enabled
    return (("bands_gemm_ffi", bands_gemm_ffi_enabled()),)


__all__ = ["FFI_DIAL_ENV", "ffi_dial_key"]

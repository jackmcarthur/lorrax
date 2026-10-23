"""LORRAX FFI subpackage: the Python half of the rank-local native kernels.

Each module owns the ``jax.ffi.ffi_call``s of one family and the rule that
picks its backend:

* ``ffi.fft``: the k-convolution router.  Every k-axis convolution and
  k-axis transform (the χ₀ time node, Σ = G⋆W, the W wedge into R space, the
  BSE W term, the ζ-fit pair Gram, the route-G plane FFT) and the
  ``common.fourier_plan.LocalFourierPlan`` custom call.  The backend follows
  the mesh's device vendor only (``ffi.fft.kconv_backend``): nvidia-mathdx on
  CUDA, the host FFTW3-ABI plan on cpu, kdft where it serves on ROCm, XLA
  elsewhere.  No environment variable selects it.
* ``ffi.gemm``: the host CBLAS batched GEMM behind the right contraction of
  ``common.contract_bands`` (dial ``LORRAX_BANDS_GEMM_FFI``, cpu meshes only).
* ``ffi.io``: the parallel-HDF5 slab handlers behind ``file_io.slab_io``.
* ``ffi.gate``: the env-dial resolver (``lxkit.gate`` bound to this
  package's loader) and the vendor platform key.
* ``ffi.common``: library location and target registration (``ffi_loader``),
  the per-leg family table (``families``), a byte-exact broadcast.

Distributed and local dense linear algebra (GEMM plans, active-range GEMMs,
LU, eigh) lives in ``services/distrib_la``, not here.  Contracts:
``docs/architecture/ffi_layout.md`` (operations and targets) and
``docs/architecture/kconv.md`` (the k-convolution modes); directory layout
and how to add a target: ``AGENTS.md``.
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

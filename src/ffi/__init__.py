"""LORRAX FFI subpackage: JAX ↔ external parallel linear algebra libraries.

See `AGENTS.md` for the directory layout and how to add a new target.
"""


#: THE PER-PROCESS DIALS, BY NAME — the declaration this subpackage owes the
#: rest of the tree.
#:
#: Every name here is read from ``os.environ`` at kernel-FACTORY time, and
#: flipping one changes the emitted HLO BODY, not merely which cached
#: callable is handed back: ``LORRAX_FFT_FFI_FUSED`` is the difference
#: between one fused host-FFI ``ffi_call`` custom call and a native
#: three-FFT ``jnp`` chain.  So a rank whose dial differs from its peers'
#: compiles a DIFFERENT MODULE, holds a different persistent-cache key, and
#: — because JAX writes entries from process 0 only — misses where its peers
#: hit.  That is ``jit__multi_slice``'s divergence
#: (FIX_multislice_cachekey.md) arriving through the environment instead of
#: through a shard offset, and nothing used to compare it across ranks.
#:
#: ``common/jax_compile_cache.py::RANK_FINGERPRINT_ENV`` now folds these into
#: the cross-rank fingerprint, so a non-uniform dial turns the cache off on
#: every rank LOUDLY instead of silently diverging.  The two lists are kept
#: in agreement by ``tests/cache_key_lint.py``'s ``env-dial`` rule, which is
#: why this tuple exists as data rather than being spelled inside
#: :func:`ffi_dial_key`'s body: a lint that has to execute the function
#: cannot run on a machine with no FFI library, which is exactly the machine
#: someone adds a dial on.
#:
#: ADD A DIAL HERE WHEN YOU ADD ONE BELOW.  The lint fails otherwise.
FFI_DIAL_ENV = (
    "LORRAX_FFT_FFI",
    "LORRAX_FFT_FFI_FUSED",
    "LORRAX_BANDS_GEMM_FFI",
    # The fused-conv family's k-minor member.  Default auto may select either
    # the CUDA custom call or the caller's reference chain, so rank agreement
    # is part of the compiled-op contract.
    "LORRAX_CONV_KMINOR_FFI",
    # Direct k-leading candidate, default off.  No production Sigma caller
    # exists until its separate seam lands; retain the dial in cross-rank and
    # factory fingerprints for the candidate API and future caller.
    "LORRAX_CONV_KLEAD_FFI",
    # ISDF CCT/ZCT post-pair accelerator.  The mode changes the post-einsum
    # HLO body inside a shard_map, so every rank must compile the same arm.
    "LORRAX_CONV_KPAIR_FFI",
)


def ffi_dial_key() -> tuple:
    """The ONE cache-key component capturing every factory-time FFI dial.

    ``make_flat_k_*`` / ``make_flat_k_gw_conv`` / the ``contract_bands``
    primitive all read their backend dial (``LORRAX_FFT_FFI``,
    ``LORRAX_FFT_FFI_FUSED``, ``LORRAX_BANDS_GEMM_FFI``) at FACTORY time, so
    a kernel cache that omits the dials serves a stale backend after a
    mid-process flag flip (tests flip them; the service contract —
    ``docs/dev/flat_k_fft_service.md`` — says the dial MUST be in every
    consumer cache key).  This helper is the single owner of "which dials
    were live when this factory ran"; consumers fold the returned tuple into
    their cache keys instead of each re-listing the dials (and drifting when
    a dial is added):

        ``gw.ppm_tau_kernel``   (pipeline_key + tau cache_key)
        ``gw.cohsex_sigma._make_cohsex_kernels``
        ``gw.w_isdf._get_chi_minimax_kernel``

    All six reads are tier-1 lexical (no JAX backend init) and O(1) —
    safe in any cache-lookup path at any P.
    """
    from ffi.fft import (conv_klead_mode, conv_kminor_mode, conv_kpair_mode,
                         fft_ffi_enabled, fused_fft_ffi_enabled)
    from ffi.gemm import gemm_ffi_enabled as bands_gemm_ffi_enabled
    return (
        ("fft_ffi", fft_ffi_enabled()),
        ("fft_ffi_fused", fused_fft_ffi_enabled()),
        ("bands_gemm_ffi", bands_gemm_ffi_enabled()),
        # THE MODE, not a bool: `auto` and `on` both "enable" the dial but
        # emit different programs on a mesh where the capability is absent,
        # so a boolean key would let one be served from the other's cache.
        ("conv_kminor_ffi", conv_kminor_mode()),
        ("conv_klead_ffi", conv_klead_mode()),
        ("conv_kpair_ffi", conv_kpair_mode()),
    )


__all__ = ["FFI_DIAL_ENV", "ffi_dial_key"]

"""The native families per platform leg: one table, read by the loaders, routers and docs.

A *family* is a set of LORRAX FFI targets sharing a target-name prefix and one
C++ source tree (``src/ffi/cpp/<dir>/``).  For every leg of :data:`lxkit.LEGS`
a family is in exactly one state:

* ``built``    the leg's library registers the family's targets; a router
               may route to them, and every such route is gated against the
               same-device XLA route at ``tol`` (``gate`` names the check);
* ``xla``      the operation runs through XLA on this leg (no native code);
* ``elsewhere`` another family serves the operation on this leg (``via``);
* ``host``     the operation runs on the host through a ``jax.pure_callback``
               (a stated state, not an XLA route; it makes programs uncacheable);
* ``refused``  the operation refuses on this leg by a named GATE (``via``).

Owner: ``ffi.common``.  It is the one package both FFI loaders' callers and the
routers already import (``ffi_loader``, ``ffi.fft``, ``ffi.gemm``, ``ffi.io``),
it stays stdlib-only so tests and docs read it without JAX, and it is the
layer that knows which handlers each leg's library registers.  ``lxkit``
carries policy, never tables, so the table is not there; ``distrib_la``'s
families are declared here too and its loader is checked against this table
by ``tests/test_platform_legs.py`` (a service cannot import ``src``).
What stays OUT: target -> C++ symbol maps (the loaders'), backend preference
by operation (``distrib_la.resolve``), and launch bounds (each router's).
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional

__all__ = ["Family", "FAMILIES", "STATES", "state", "built_prefixes", "family_of"]

STATES = ("built", "xla", "elsewhere", "host", "refused")


@dataclass(frozen=True)
class Family:
    """One native family (see the module docstring)."""

    prefixes: tuple            #: target-name prefixes the family owns
    source: str                #: src/ffi/cpp/<dir> (or the service that holds its Python side)
    legs: Mapping              #: leg key -> (state, via)
    gate: str = ""             #: the same-device XLA check of a ``built`` route
    tol: Optional[float] = None  #: its tolerance, relative to max|XLA| (0.0 = bytes equal)
    note: str = ""


def _legs(cuda, rocm, cpu):
    return MappingProxyType({"CUDA": cuda, "ROCM": rocm, "cpu": cpu})


FAMILIES: Mapping[str, Family] = MappingProxyType({
    "phdf5": Family(
        ("lorrax_phdf5_",), "phdf5",
        _legs(("built", ""), ("built", ""), ("built", "")),
        gate="byte round trip against h5py (tests/test_file_io.py)", tol=0.0,
        note="IO, not arithmetic: no XLA route; the reference is the file."),
    "mathdx": Family(
        ("lorrax_mathdx_", "lorrax_fourier_plan_mathdx"), "cufft",
        _legs(("built", ""), ("xla", "ffi.fft.kconv_backend -> 'xla'"),
              ("elsewhere", "fftw")),
        gate="tests/test_kconv_xla_gate.py (ffi.fft.xla_reference)", tol=1e-12,
        note="cuFFTDx k-convolution; NVRTC-built per (mode, grid, ns)."),
    "kdft": Family(
        ("lorrax_kdft_",), "kdft",
        _legs(("elsewhere", "mathdx"), ("built", ""), ("elsewhere", "fftw")),
        gate="kdft vs ffi.fft.xla_reference on the same GCD (modes 2/3; the k-box modes 7/9/11: "
             "tests/test_kdft_kbox_gate.py)", tol=1e-12,
        note="k-leading and k-minor pairs (ffi.fft._ROCM_KLEAD_TARGETS) and the k-box single-arm "
             "modes 7/9/11 (ffi.fft._ROCM_KBOX_TARGETS); every other k-conv factory takes XLA on "
             "ROCm.  Measured max|dU|/max|U| <= 7.1e-16 (c128)."),
    "fourier_plan": Family(
        ("lorrax_fourier_plan",), "cufft",
        _legs(("built", ""), ("xla", "common.fourier_plan XLA leg"), ("xla", "common.fourier_plan XLA leg")),
        note="LocalFourierPlan's CUDA leg (cuBLAS Fourier GEMMs + one cuFFT group); "
             "same-device XLA gate owed."),
    "fftw": Family(
        ("lorrax_mklfft_",), "fftw",
        _legs(("elsewhere", "mathdx"), ("xla", "ffi.fft.kconv_backend -> 'xla'"), ("built", "")),
        gate="tests/test_kconv_xla_gate.py (cpu 'plan' backend)", tol=1e-12),
    "cblas": Family(
        ("lorrax_mklblas_",), "cblas",
        _legs(("xla", "ffi.gemm refuses the dial on a device mesh"),
              ("xla", "ffi.gemm refuses the dial on a device mesh"), ("built", "")),
        gate="common.contract_bands XLA body", tol=None,
        note="Opt-in dial (LORRAX_BANDS_GEMM_FFI); tolerance not declared in code yet."),
    "cublas_local": Family(
        ("lorrax_cublas_local_",), "cublas",
        _legs(("built", ""), ("xla", "distrib_la._active_local (masked: ACTIVE_LOCAL_ROUTE)"),
              ("xla", "distrib_la._active_local (interval: ACTIVE_LOCAL_ROUTE)")),
        gate="tests/test_active_local_routes.py (masked vs interval vs dense, CPU)", tol=1e-13,
        note="Sigma's active-range GEMM. ROCM takes the masked route (one batched dot, no device "
             "control flow); the interval route's while/cond trips are launch-bound there."),
    "active_subspace": Family(
        ("lorrax_active_subspace_",), "active_subspace",
        _legs(("built", ""), ("xla", "distrib_la.subspace XlaSubspacePlan (SUBSPACE_ROUTE)"),
              ("host", "distrib_la.subspace pure_callback (LAPACK)")),
        note="Provider per leg: distrib_la.subspace.SUBSPACE_ROUTE.  ROCM takes the XLA plan "
             "(device products + eigh_local); the host route cost 1.5-2.1 s per BSE solve "
             "and made jit_solve uncacheable."),
    "cusolvermp": Family(
        ("lorrax_cusolvermp_",), "cusolvermp",
        _legs(("built", ""), ("elsewhere", "slate (distrib_la.resolve eigh/ROCM row)"),
              ("elsewhere", "scalapack, slate")),
        note="NVIDIA-only library; no ROCm equivalent."),
    "slate": Family(
        ("lorrax_slate_",), "slate",
        _legs(("built", ""), ("built", ""), ("built", "")),
        note="ROCM: SLATE v2025.05.28-1 gpu_backend=hip (config/frontier/build_slate_hip.sh), "
             "sequential LibSci; eigh/potrf gated against the native route at P=4, n=512-8192. "
             "ROCM also builds the LU family (getrf PartialPiv + getrs: distributed solve_lu, "
             "gated against jnp.linalg.solve at P=4); CUDA's LU is cusolvermp, the host's scalapack."),
    "scalapack": Family(
        ("lorrax_scalapack_",), "scalapack",
        _legs(("elsewhere", "cusolvermp"), ("elsewhere", "slate"), ("built", "")),
        note="Host-only."),
})


def family_of(target: str) -> Optional[str]:
    """The family owning ``target`` (longest prefix), or None."""
    hits = [(len(p), name) for name, fam in FAMILIES.items() for p in fam.prefixes
            if target.startswith(p)]
    return max(hits)[1] if hits else None


def state(family: str, leg: str) -> tuple:
    """``(state, via)`` of ``family`` on ``leg`` (a :data:`lxkit.LEGS` key)."""
    return FAMILIES[family].legs[leg]


def built_prefixes(leg: str) -> tuple:
    """Target prefixes of every family ``built`` on ``leg``, in table order."""
    return tuple(p for fam in FAMILIES.values() if fam.legs[leg][0] == "built"
                 for p in fam.prefixes)

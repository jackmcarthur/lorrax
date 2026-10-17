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
             "modes 7/9/11 (ffi.fft._ROCM_KBOX_TARGETS), the BSE outer-product load and its fused "
             "decode (ffi.fft._ROCM_OUTER_TARGETS; tests/test_kdft_outer_gate.py), k-box mode 6 (the ISDF "
             "route-G pair, ffi.fft._ROCM_PAIR_TARGETS; tests/test_kdft_plane_gate.py); every other k-conv "
             "factory takes XLA on ROCm.  Measured max|dU|/max|U| <= 7.1e-16 (c128); the outer load and "
             "its decode: <= 1.6e-15 (tests/test_kdft_outer_gate.py, 1.524e-15)."),
    "fourier_plan": Family(
        ("lorrax_fourier_plan",), "cufft",
        _legs(("built", ""), ("elsewhere", "fourier_plan_rocm"), ("xla", "common.fourier_plan XLA leg")),
        gate="tests/test_fourier_plan_xla_gate.py (ffi leg vs XLA leg of one plan, same device)",
        tol=1e-12,
        note="LocalFourierPlan's CUDA leg (cuBLAS Fourier GEMMs + one cuFFT group); the gate "
             "serves both device legs and is unrun on CUDA (no CUDA device since wave 7)."),
    "fourier_plan_rocm": Family(
        ("lorrax_fourier_plan_rocm",), "hipfft",
        _legs(("elsewhere", "fourier_plan"), ("built", ""), ("xla", "common.fourier_plan XLA leg")),
        gate="tests/test_fourier_plan_xla_gate.py (ffi leg vs XLA leg of one plan, same GCD)",
        tol=1e-12,
        note="LocalFourierPlan's ROCM leg, ported from cufft/fourier_plan_cuda_ffi.cc: rocBLAS "
             "ZGEMM axes (stride-0 matrix, no transposes) + one hipFFT group + a hipRTC remap; no "
             "fused pair (cuBLASDx has no ROCm twin: GEMM pairs run as two rocBLAS calls).  Route "
             "row: ffi.fft.FOURIER_PLAN_TARGETS['ROCM']; GEMM axes per common.fourier_plan."
             "GEMM_CROSSOVER."),
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
        _legs(("built", ""), ("built", ""),
              ("xla", "distrib_la._active_local (interval: ACTIVE_LOCAL_ROUTE)")),
        gate="tests/test_active_local_native_gate.py (native vs masked vs interval vs dense on "
             "the same device, P=1 and P=4); tests/test_active_local_routes.py (XLA routes, CPU)",
        tol=1e-13,
        note="Sigma's active-range GEMM: one leg-neutral source (cublas/local_active_gemm_ffi.cc "
             "over common/gpu_blas.h), cuBLAS on CUDA, rocBLAS on ROCM (the mapped "
             "librocblas.so.5).  Pointer views into the original buffers; the bounds metadata "
             "is read to the host per call.  ROCM's ACTIVE_LOCAL_ROUTE row is 'device' (active_gemm): "
             "that per-call read drains the stream in Sigma's tau loop (Fe Sigma tau 13.5 s vs 9.1 s "
             "masked); masked and interval stay as the XLA references."),
    "active_gemm": Family(
        ("lorrax_active_gemm_",), "active_gemm",
        _legs(("built", ""), ("built", ""),
              ("xla", "distrib_la._active_local (interval: ACTIVE_LOCAL_ROUTE)")),
        gate="tests/test_active_local_native_gate.py (device vs masked vs interval vs dense on "
             "the same device, P=1 and P=4)", tol=1e-13,
        note="Sigma's active-range GEMM with DEVICE-side bounds: one run-time-compiled kernel "
             "(hipRTC / NVRTC via rtc_build), K loop clamped per row from the device bounds, "
             "weights applied on load; no host read or sync, no operand copy, no workspace. "
             "Inner product on the FP64 matrix cores, complex as the exact 4M form: MFMA f64 16x16x4 "
             "on gfx90a / gfx94x, DMMA m8n8k4 f64 on CUDA compute capability >= 8.0; the FMA "
             "path elsewhere. "
             "ROCM's ACTIVE_LOCAL_ROUTE row."),
    "active_subspace": Family(
        ("lorrax_active_subspace_",), "active_subspace",
        _legs(("built", ""), ("built", ""),
              ("host", "distrib_la.subspace pure_callback (LAPACK)")),
        gate="tests/test_subspace_native_gate.py (native vs XlaSubspacePlan and CpuSubspacePlan "
             "on the same device: project, eigh, reconstruct, Gram, CGS2, subtract, store, "
             "NaN-poisoned inactive rows, one compilation per plan)", tol=1e-12,
        note="Provider per leg: distrib_la.subspace.SUBSPACE_ROUTE; implementation per leg: "
             "distrib_la.active_subspace.NATIVE_HANDLERS.  CUDA: cuBLAS/cuSOLVERDn on the active "
             "dimensions (active_ops.cc, active_eigh.cc; host-read descriptors).  ROCM: "
             "active_devsize.cc, run-time compiled kernels reading the descriptors on the device "
             "(no per-call host sync), and the shifted, reflected capacity eigensolve through "
             "lorrax_rocsolver_batched_eigh.  distributed_ortho needs cuSolverMp's communicator: "
             "CUDA-only (ROCM: JAX psum; native_collectives refuses by GATE)."),
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
    "rocsolver": Family(
        ("lorrax_rocsolver_",), "rocsolver",
        _legs(("xla", "jnp.linalg.eigh (jaxlib cuSOLVER)"), ("built", ""), ("xla", "jnp.linalg.eigh (LAPACK)")),
        gate="tests/test_rocsolver_eigh_gate.py (residual, orthogonality, eigenvalues and "
             "eigenspaces vs jnp.linalg.eigh on the same GCD)", tol=1e-12,
        note="ROCM batched local eigh: one rocsolver_{z,d}{heevd,syevd}_strided_batched call per "
             "stack (jaxlib loops zheevd per matrix), a matrix rocSOLVER leaves non-finite with "
             "info = 0 (its silent zheevd failure) is NaN in w and v; distrib_la.local_eigh."
             "BATCHED_EIGH_ROUTE holds the drivers and stacks.  Also the one-device LU solve "
             "(lorrax_rocsolver_lu_solve / _tri_solve: getrf + triangular solves blocked so no rocBLAS operand "
             "spans 2^31 bytes; distrib_la.local_lu.LOCAL_LU_ROUTE, gate "
             "tests/test_rocsolver_lu_gate.py).  rocSOLVER is the process's mapped copy."),
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

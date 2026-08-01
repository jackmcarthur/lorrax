"""Vendor-BLAS batched GEMM — the ``LORRAX_BANDS_GEMM_FFI`` service.

The Python half of ``src/ffi/cpp/mklblas/gemm_batch_ffi.cc``: the gate that
decides whether the dial is on, and the single ``jax.ffi.ffi_call`` that
issues the handler.  Per ``src/ffi/AGENTS.md:149-150`` and
``TEMPLATE.md:10-20`` this is where an FFI target's Python lives; it used to
live in ``common/contract_bands.py``, which is a *physics* module.

Scope, stated up front so nobody "adopts" this everywhere: this is NOT a
general GEMM service.  It is the body of ONE contraction — the large right
contraction of ``common.contract_bands.contract_bands_block_reshard`` — and
nothing else in the tree is routed through it.  Full contract:
``docs/dev/vendor_gemm_service.md``.

    C[i] = A[i] @ B[i % BB]      A (BA, M, K), B (BB, K, N), BA % BB == 0

row-major NN, dtype dispatched inside the ``.so`` from the buffer element
type onto ``cblas_{d,s,z,c}gemm[_batch]`` (all four BLAS precisions since
2026-07-29).  No ``input_output_aliases``: a ``(BA, M, N)`` output can never
legally alias a ``(BA, M, K)``/``(BB, K, N)`` operand buffer — contrast the
shape-preserving mklfft handlers, which do alias.

Host platform ONLY.  ``lorrax_mklblas_gemm_batch`` is in ``ffi_loader``'s
host table and not the CUDA one (``ffi_loader.py:146`` vs ``:92-116``), so a
forced CUDA probe reports *unknown target*; XLA:GPU's own ``dot`` lowering
already dispatches cuBLAS and is deliberately untouched.
"""

from __future__ import annotations

import jax

from ffi.gate import Gate

__all__ = ["GEMM_TARGET", "GATE", "gemm_ffi_mode", "gemm_ffi_enabled",
           "require_gemm_ffi", "gemm_batch"]

GEMM_TARGET = "lorrax_mklblas_gemm_batch"

#: The ``LORRAX_BANDS_GEMM_FFI`` dial.  AUTO by default (owner order
#: 2026-07-29, env-vars doctrine #8: capability detection, not policy).
#:
#: ``silent_platform_demote`` is the one place this gate is allowed to keep
#: quiet, and the reason is recorded in the field itself: on a GPU run the
#: dial is not merely off, it does not exist (host symbol table only), so an
#: announcement would report a non-decision on every GPU run and name a
#: variable the user has no reason to have set.
GATE = Gate(
    env="LORRAX_BANDS_GEMM_FFI",
    target=GEMM_TARGET,
    platforms=("cpu",),
    modes=("auto", "off", "on"),
    default="auto",
    off_label="default XLA GEMM lowering",
    label={"cpu": "MKL batched-GEMM host"},
    lexical_default="CUDA",
    auto_on_msg=(
        "[bands_gemm] AUTO-ON: CPU platform and FFI target "
        "'{target}' resolves in the host .so -> "
        "contract_bands right-GEMMs at vendor-BLAS rate "
        "(1.6-1.9x over the XLA:CPU Eigen dots at full "
        "threads, jobs 7879008/7879010).  "
        "LORRAX_BANDS_GEMM_FFI=0 disables."),
    auto_unavailable_msg=(
        "[bands_gemm] auto: FFI target '{target}' "
        "unavailable -> default XLA GEMM lowering "
        "({reason})"),
    resolved_msg={"cpu": (
        "[bands_gemm] contract_bands right-GEMMs -> vendor-BLAS "
        "GEMM host FFI handler ({target}): "
        "dgemm/zgemm at BLAS rate (memo Sec. 4.4 exit (b)); "
        "collectives, channel algebra and left dots untouched.")},
    silent_platform_demote=(
        "the target is not in ffi_loader's CUDA table at all; XLA:GPU's dot "
        "lowering already dispatches cuBLAS, which is optimal there, so a "
        "GPU user has nothing to act on"),
    refuse_platform_msg=(
        "LORRAX_BANDS_GEMM_FFI requested the MKL batched-GEMM host "
        "backend, but the mesh devices are '{platform}'.  This dial is "
        "CPU-only BY DESIGN: the native XLA dot lowering on CUDA "
        "already dispatches cuBLAS (optimal there) and is deliberately "
        "untouched.  Unset LORRAX_BANDS_GEMM_FFI on non-CPU meshes — "
        "explicit requests are never silently downgraded."),
    refuse_probe_msg=(
        "LORRAX_BANDS_GEMM_FFI requested the MKL batched-GEMM host "
        "backend, but FFI target '{target}' is unusable: "
        "{reason}"),
)


def gemm_ffi_mode() -> str:
    """``"on"`` | ``"off"`` | ``"auto"`` — the raw ``LORRAX_BANDS_GEMM_FFI``
    grammar (:meth:`ffi.gate.Gate.mode`)."""
    return GATE.mode()


def gemm_ffi_enabled() -> bool:
    """True when the dial is on.  Read at FACTORY time and **safe before
    ``jax.distributed.initialize``** — consumers key kernel caches on it
    (tier 1 of the gate contract; ``gw.ppm_tau_kernel``'s ``pipeline_key``).
    """
    return GATE.enabled()


def require_gemm_ffi(mesh) -> str:
    """Announce-or-REFUSE for an EXPLICIT ``=1``: raises with the fix named
    on a non-CPU mesh or a host ``.so`` without the handler (quoting the
    ``probe_target`` reason)."""
    return GATE.require(mesh)


def gemm_batch(a3, b3):
    """``C[i] = A[i] @ B[i % BB]`` for A ``(BA, M, K)``, B ``(BB, K, N)``.

    Row-major NN batched GEMM through the host handler; dtype
    (f64/f32/c128/c64) is dispatched inside the ``.so`` from the buffer
    element type.  The broadcast rule (B cycling with period BB) is what
    lets one handler serve both the plain per-k batch (BA == BB == nk) and
    the extra-stacked batch (BA == E·nk against the k-only ψ, extra axis
    OUTERMOST).  No ``input_output_aliases`` — a (BA, M, N) GEMM output can
    never legally alias a (BA, M, K)/(BB, K, N) operand buffer.

    No ``shard_map`` here: this is a rank-LOCAL handler with no
    communicator, called from inside the caller's own ``shard_map`` body
    (``shard_map`` cannot nest).  That is the structural difference from the
    ``ffi/{slate,scalapack,cusolvermp,cublasmp}`` families, whose wrappers
    own their ``shard_map`` because their handlers hold an MPI/NCCL context.
    """
    ba, m, k = (int(d) for d in a3.shape)
    bb, k2, n = (int(d) for d in b3.shape)
    if k != k2 or (bb and ba % bb):
        raise ValueError(
            f"bands_gemm: incompatible batch shapes A{tuple(a3.shape)} vs "
            f"B{tuple(b3.shape)} (need A.K == B.K and BA % BB == 0).")
    if a3.dtype != b3.dtype:
        raise TypeError(
            f"bands_gemm: mixed operand dtypes {a3.dtype} vs {b3.dtype} — "
            f"the de-promotion policy should have split them upstream.")
    out_t = jax.ShapeDtypeStruct((ba, m, n), a3.dtype)
    return jax.ffi.ffi_call(GEMM_TARGET, out_t)(a3, b3)

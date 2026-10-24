"""Locate, load and register optional distrib_la FFI provider libraries.

Ported from LORRAX ``src/ffi/common/ffi_loader.py`` at 96a6399, carrying
only what the distributed-linalg backends need: the two platform library
specs, the linalg target→C++-symbol tables, the ctypes declarations for
the cuSOLVERMp and MPI mesh contexts, and the probe.  The one native GEMM here is
the CUDA local active-range cuBLAS product; the distributed GEMM is
:func:`distrib_la.panel_matmul` in XLA.  Only unrelated FFT, host
bands-GEMM and phdf5 surfaces remain in LORRAX's loader.

THIS IS THE QUARANTINE BOUNDARY.  Everything in this package that knows a
vendor library exists knows it through this module: one ``dlopen``, one
symbol table, one probe.  ``pyproject.toml`` declares no ScaLAPACK, ELPA
or cuSOLVERMp because none is linked here — they are inside
a ``.so`` that this file opens by path.

The tables are the SERVICE's, the policy is lxkit's
--------------------------------------------------
:mod:`lxkit.probe` ships the ABSENT-vs-BROKEN split
(:class:`~lxkit.probe.LibraryNotBuilt` / :class:`~lxkit.probe.LibraryUnusable`)
and the three *unusable* reason constructors; this module supplies the
tables they describe.  That division is the reason a harness can act on a
distrib_la refusal without knowing whose library it was.

Environment
-----------
``LORRAX_FFI_SO``       absolute path to ``liblorrax_ffi.so`` (CUDA)
``LORRAX_FFI_ROCM_SO``  absolute path to ``liblorrax_ffi_rocm.so`` (ROCM)
``LORRAX_FFI_HOST_SO``  absolute path to ``liblorrax_ffi_host.so`` (cpu)

The legs (file, pin, build dir, C-ABI suffix, ABI symbol) are
:data:`lxkit.legs.LEGS`, shared with LORRAX's own loader; this module owns
its target tables, ctypes declarations and build hints.

These historical names identify the currently supported provider ABI, whose
C++ implementation lives in LORRAX.  A standalone installation may pin a
compatible provider by absolute path; it need not import or install the
LORRAX Python package.  The variables GRANT A CAPABILITY and never select a
backend — backend choice is an API argument, and :mod:`distrib_la.resolve`
reads no environment at all.

An explicit pin that is missing is a REFUSAL, never a fall-through
(2026-08-06): a stale or mistyped pin used to be appended to the candidate
list, so the run silently used the in-tree build instead and the pin left
no trace.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, Optional

from lxkit import native_provider as _native
from lxkit.legs import LEGS, accelerator_leg_to_open_first
from lxkit.probe import (AVAILABLE, LibraryNotBuilt, LibraryUnusable,
                         ProbeResult, missing_symbol, not_loadable,
                         unknown_target)

__all__ = ["get_lib", "has_target", "probe_target", "loaded_lib_path",
           "context_key", "bind_context", "unbind_context",
           "loaded_platforms_in_order", "LibraryNotBuilt", "LibraryUnusable",
           "LORRAX_FFI_ABI_VERSION", "AbiMismatch"]

#: The handler-signature ABI this package speaks.
#:
#: A MIRROR of ``src/ffi/cpp/common/lorrax_ffi_abi.h`` in the LORRAX repo,
#: which is the definition and carries the bump rule.  This package is
#: independently installable and must not reach into that tree at import time,
#: so the number is copied — and ``tests/test_so_acceptance.py`` parses the
#: header whenever the monorepo is reachable and FAILS if the two disagree.
#: That is the whole mechanism keeping a copy honest: a copy nothing compares
#: is how the ten dispatch sites drifted.
LORRAX_FFI_ABI_VERSION = 11

#: platform -> the C entry point reporting the library's ABI.  Per leg,
#: because both libraries are dlopened RTLD_GLOBAL and already share sixteen
#: symbol names; a shared spelling would let the first-loaded library answer
#: for the second, which is the exact failure being detected.
_ABI_SYMBOLS = {key: leg.abi_symbol for key, leg in LEGS.items()}

class AbiMismatch(LibraryUnusable):
    """The library loaded fine and speaks a DIFFERENT handler ABI.

    A subclass of :class:`~lxkit.probe.LibraryUnusable` because it is the same
    kind of fact — present, and not usable — so every existing handler stays
    correct; a distinct class because the fix is different (pair it, do not
    debug it).
    """


_LIBS: Dict[str, ctypes.CDLL] = {}
#: platform -> the .so path actually loaded (for diagnostics).
_LIB_PATHS: Dict[str, str] = {}

# ---------------------------------------------------------------------------
# The tables.  target name (what a jax.ffi.ffi_call names) -> C++ symbol
# (what the .so exports).  FROZEN: both halves are invariant 1 of
# docs/architecture/ffi_layout.md, and no C++ moved in this extraction.
#
# One handler per routine covers every supported dtype -- dispatch happens
# inside the .so on the input buffer's element type.
# ---------------------------------------------------------------------------
_CUDA_TARGET_SYMBOLS = {
    "lorrax_active_subspace_store":        "ActiveSubspaceStoreFfi",
    "lorrax_active_subspace_eigh":         "ActiveSubspaceEighFfi",
    "lorrax_active_subspace_project":      "ActiveSubspaceProjectFfi",
    "lorrax_active_subspace_reconstruct":  "ActiveSubspaceReconstructFfi",
    "lorrax_active_subspace_gram": "ActiveSubspaceGramFfi",
    "lorrax_active_subspace_ortho":        "ActiveSubspaceOrthoFfi",
    "lorrax_active_subspace_distributed_ortho": "ActiveSubspaceDistributedOrthoFfi",
    "lorrax_active_subspace_subtract": "ActiveSubspaceSubtractFfi",
    "lorrax_active_subspace_subtract_gram": "ActiveSubspaceSubtractGramFfi",

    "lorrax_cublas_local_active_range_gemm": "CublasLocalActiveRangeGemmFfi",
    "lorrax_cublas_local_active_range_gemm_out": "CublasLocalActiveRangeGemmOutFfi",
    "lorrax_cublas_local_prepared_active_range_gemm": "CublasLocalPreparedActiveRangeGemmFfi",
    # The active-range GEMM with DEVICE-side bounds (cpp/active_gemm/, leg-neutral, run-time
    # compiled; DMMA f64 on compute capability >= 8.0): ACTIVE_LOCAL_ROUTE's "device" row.
    "lorrax_active_gemm_device": "ActiveGemmDeviceFfi",
    "lorrax_active_gemm_device_out": "ActiveGemmDeviceOutFfi",
    "lorrax_cusolvermp_eigh":             "EighMpFfi",
    "lorrax_cusolvermp_batched_potrf":    "CusolverMpBatchedPotrfFfi",
    "lorrax_cusolvermp_batched_potrs":    "CusolverMpBatchedPotrsFfi",
    "lorrax_cusolvermp_batched_solve_lu_factors": "CusolverMpBatchedSolveLuFactorsFfi",
    "lorrax_cusolvermp_batched_getrf":    "CusolverMpBatchedGetrfFfi",
    "lorrax_cusolvermp_batched_getrs":    "CusolverMpBatchedGetrsFfi",
}

# The host variants: the SAME target names registered under platform="cpu"
# (the split jaxlib itself uses for lapack vs cusolver), so a call site
# never mentions a platform and the lowering platform picks the handler.
# Only the C++ SYMBOLS differ, so the two .so's do not collide with EACH
# OTHER under RTLD_GLOBAL — but their DEPENDENCIES do, and one of those
# collisions is load-bearing: see the SONAME note above ``get_lib``.
# ScaLAPACK is host-only: there is no CUDA row for it anywhere.
_HOST_TARGET_SYMBOLS = {
    "lorrax_scalapack_eigh":                "ScalapackEighHostFfi",
    "lorrax_scalapack_batched_solve_lu_factors": "ScalapackBatchedSolveLuFactorsHostFfi",
    "lorrax_scalapack_batched_getrf":       "ScalapackBatchedGetrfHostFfi",
    "lorrax_scalapack_batched_getrs":       "ScalapackBatchedGetrsHostFfi",
}

#: Per-platform library spec.  ``build_subdir`` is relative to
#: ``src/ffi/cpp/`` of the LORRAX checkout, when there is one (see
#: :func:`_candidate_paths`).
#: The batched local Hermitian eigh (cpp/rocsolver/batched_eigh_ffi.cc: one rocSOLVER
#: strided-batched heevd/syevd per stack), the target of local_eigh.BATCHED_EIGH_ROUTE's ROCM
#: row.  CUDA solves the same stacks with jaxlib's cuSOLVER, so only this leg builds it.
_ROCM_LOCAL_EIGH_SYMBOLS: dict = {
    "lorrax_rocsolver_batched_eigh": "RocsolverBatchedEighFfi",
}
#: The one-device LU solve (cpp/rocsolver/lu_solve_ffi.cc: rocSOLVER getrf + a triangular solve
#: blocked so that no rocBLAS operand spans 2^31 bytes), the target of local_lu.LOCAL_LU_ROUTE's
#: ROCM row.  CUDA solves through jaxlib's cuSOLVER (exact at every n measured): ROCM only.
_ROCM_LOCAL_LU_SYMBOLS: dict = {
    "lorrax_rocsolver_lu_solve": "RocsolverLuSolveFfi",
    "lorrax_rocsolver_tri_solve": "RocsolverTriSolveFfi",
}
#: The active-subspace family on this leg (cpp/active_subspace/active_devsize.cc: device-side
#: sizes, no host read) -- the CUDA symbols less the one-communicator CGS2 handler (needs
#: cuSolverMp's NCCL context) and the active-block eigh (the ROCM plan solves the capacity
#: matrix through lorrax_rocsolver_batched_eigh: distrib_la.active_subspace.NATIVE_HANDLERS).
_ROCM_ACTIVE_SUBSPACE_SYMBOLS: dict = {
    t: s for t, s in _CUDA_TARGET_SYMBOLS.items()
    if t.startswith("lorrax_active_subspace_")
    and t not in ("lorrax_active_subspace_distributed_ortho", "lorrax_active_subspace_eigh")}
#: The local active-range GEMM (cpp/cublas/local_active_gemm_ffi.cc) is leg-neutral: the
#: ROCM leg compiles the CUDA leg's source on rocBLAS (common/gpu_blas.h), so it exports
#: the same three symbols (ACTIVE_LOCAL_ROUTE's "native" row on both legs).
#: The active-range GEMM with device-side bounds (cpp/active_gemm/: leg-neutral, run-time
#: compiled) is built on both legs: ACTIVE_LOCAL_ROUTE's "device" row, in the CUDA table above.
#: The ELPA eigh (cpp/elpa/eigh_ffi.cc over config/frontier/build_elpa_hip.sh's ELPA): the
#: ROCM ``distributed`` eigh above the one-device bound (distrib_la.resolve).  ROCM only.
_ROCM_ELPA_SYMBOLS: dict = {"lorrax_elpa_eigh": "ElpaEighFfi"}
_ROCM_TARGET_SYMBOLS: dict = {**{t: s for t, s in _CUDA_TARGET_SYMBOLS.items()
                                 if t.startswith(("lorrax_cublas_local_", "lorrax_active_gemm_"))},
                              **_ROCM_LOCAL_EIGH_SYMBOLS, **_ROCM_LOCAL_LU_SYMBOLS,
                              **_ROCM_ACTIVE_SUBSPACE_SYMBOLS, **_ROCM_ELPA_SYMBOLS}

#: Leg key -> this loader's target table and not-found build command.
_TARGETS = {"CUDA": _CUDA_TARGET_SYMBOLS, "ROCM": _ROCM_TARGET_SYMBOLS,
            "cpu": _HOST_TARGET_SYMBOLS}
_BUILD_HINTS = {
    "CUDA": 'src/ffi/cpp/run_shifter.sh bash src/ffi/cpp/build.sh',
    "ROCM": "bash config/frontier/build_ffi_rocm.sh",
    "cpu": 'bash config/perlmutter/build_ffi_host.sh --fresh',
}
#: Per-target overrides of the leg's not-found build command.  A target whose handler needs a
#: library the leg's command does not name gets that library's script first: ROCM's ELPA eigh
#: (cpp/elpa/) is ELPA's (config/frontier/build_elpa_hip.sh).  Every other target keeps the
#: leg's command above.
_TARGET_BUILD_HINTS = {
    ("ROCM", t): "bash config/frontier/build_elpa_hip.sh && bash config/frontier/build_ffi_rocm.sh"
    for t in _ROCM_ELPA_SYMBOLS
}


def target_build_hint(platform: str, target_name: Optional[str] = None) -> str:
    """The rebuild command a missing ``target_name`` on ``platform`` names: its own, else the leg's."""
    return _TARGET_BUILD_HINTS.get((platform, target_name), _BUILD_HINTS[platform])

_PLATFORMS = {
    key: dict(so_name=leg.so_name, env=leg.so_env, build_subdir=leg.build_subdir,
              targets=_TARGETS[key], build_hint=_BUILD_HINTS[key])
    for key, leg in LEGS.items()
}

# Error buffer size for the lrx_ wrappers.
_ERR_CAP = 512


def _checkout_build_dir(platform: str) -> Optional[Path]:
    """``<lorrax>/src/ffi/cpp/<build_subdir>``, when this service sits
    inside a LORRAX checkout (``lxkit.native_provider.checkout_build_dir``).

    distrib_la installs standalone too, where there is no ``src/ffi/cpp``;
    ``None`` then leaves the env pin (or ``sys.path``) as the only route,
    which is what the not-found message says.
    """
    return _native.checkout_build_dir(
        __file__, _PLATFORMS[platform]["build_subdir"])


def _candidate_paths(platform: str) -> list[Path]:
    spec = _PLATFORMS[platform]
    paths: list[Path] = []
    env = os.environ.get(spec["env"])
    if env:
        paths.append(Path(env))
    build_dir = _checkout_build_dir(platform)
    if build_dir is not None:
        paths.append(build_dir / spec["so_name"])
    for p in sys.path:
        pp = Path(p)
        if pp.is_dir():
            paths.append(pp / spec["so_name"])
    seen, unique = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _locate_so(platform: str) -> Path:
    return _native.locate_library(
        platform, specs=_PLATFORMS,
        candidates={name: _candidate_paths(name) for name in _PLATFORMS},
        expected_abi=LORRAX_FFI_ABI_VERSION,
        not_built_cls=LibraryNotBuilt, unusable_cls=LibraryUnusable)


# ---------------------------------------------------------------------------
# THE C ABI IS PER-LIBRARY; THE PYTHON NAME IS NOT.
# ---------------------------------------------------------------------------
# ``cpp/mesh/context.cc`` is pure MPI; the host leg includes it with ScaLAPACK
# and the ROCM leg with ELPA.  Until 2026-08-08 both .so files defined the same
# ``extern "C"`` names (and five more from ``cpp/phdf5/api.cc``, which this
# package does not call).  Both are dlopened RTLD_GLOBAL, one name with two
# definitions behind it -- the C half of the cross-.so ODR violation
# lorrax's tests/KNOWN_FAILURES.md registered as L1.  ``cpp/common/c_abi.h``
# appends ``_host`` on the host leg; binding it under the plain name here
# keeps every call site below leg-agnostic.  Mirrors
# ``ffi.common.ffi_loader._bind_c_abi`` -- the two loaders open the SAME
# FILES, so both have to know the same naming rule.
_C_ENTRY_POINTS = (
    "lrx_mesh_context_create",
    "lrx_mesh_context_destroy",
    "lrx_mesh_init_mpi",
    "lrx_ctx_bind",
    "lrx_ctx_unbind",
    # cpp/active_subspace/active_eigh.cc: the eigensolver workspace query.
    "lrx_active_eigh_lwork",
)

#: platform -> the suffix ``LRX_C_ENTRY`` appends on that leg.
_C_ABI_SUFFIX = {key: leg.c_abi_suffix for key, leg in LEGS.items()}


def _bind_c_abi(lib: ctypes.CDLL, platform: str) -> None:
    """Bind this leg's suffixed C entry points under their plain names.

    A no-op on a pre-2026-08-08 library, on purpose: the unsuffixed symbol
    is still there and ``ctypes.CDLL.__getattr__`` still finds it.  The
    ratchet is on the artifact, in ``tests/test_so_acceptance.py``'s
    check 6, not here -- see ``ffi_loader._bind_c_abi`` for the argument.
    """
    suffix = _C_ABI_SUFFIX.get(platform, "")
    if not suffix:
        return
    for base in _C_ENTRY_POINTS:
        try:
            fn = getattr(lib, base + suffix)
        except AttributeError:
            continue
        setattr(lib, base, fn)


def _declare_cusolvermp(lib: ctypes.CDLL) -> None:
    """NCCL + cuSOLVERMp context lifecycle — ``liblorrax_ffi.so`` only.

    Under a ``hasattr`` guard so a partial build (or the host library,
    which has none of these) loads instead of failing.
    """
    if not hasattr(lib, "lrx_create_cusolvermp_context"):
        return
    lib.lrx_nccl_unique_id_bytes.argtypes = []
    lib.lrx_nccl_unique_id_bytes.restype = ctypes.c_int

    lib.lrx_fill_nccl_unique_id.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
    ]
    lib.lrx_fill_nccl_unique_id.restype = ctypes.c_int

    lib.lrx_create_cusolvermp_context.argtypes = [
        ctypes.c_int,                       # rank
        ctypes.c_int,                       # world_size
        ctypes.c_void_p,                    # nccl_unique_id_addr
        ctypes.c_int,                       # nccl_unique_id_nbytes
        ctypes.c_int,                       # p
        ctypes.c_int,                       # q
        ctypes.c_int,                       # grid_layout_col_major
        ctypes.POINTER(ctypes.c_int64),     # ctx_out
        ctypes.c_char_p,                    # err_out
        ctypes.c_int,                       # err_cap
    ]
    lib.lrx_create_cusolvermp_context.restype = ctypes.c_int

    lib.lrx_destroy_cusolvermp_context.argtypes = [ctypes.c_int64]
    lib.lrx_destroy_cusolvermp_context.restype = None


def _declare_ctx_registry(lib: ctypes.CDLL) -> None:
    """The configuration-key registry (``cpp/common/ctx_registry.h``), both legs."""
    if not hasattr(lib, "lrx_ctx_bind"):
        return
    lib.lrx_ctx_bind.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_char_p,
                                 ctypes.c_char_p, ctypes.c_int]
    lib.lrx_ctx_bind.restype = ctypes.c_int
    lib.lrx_ctx_unbind.argtypes = [ctypes.c_int64]
    lib.lrx_ctx_unbind.restype = None


def _declare_mesh(lib: ctypes.CDLL) -> None:
    """The MPI mesh context lifecycle (``cpp/mesh/context.cc``), when the platform
    build includes it (host: ScaLAPACK; ROCM: ELPA)."""
    if not hasattr(lib, "lrx_mesh_context_create"):
        return
    lib.lrx_mesh_context_create.argtypes = [
        ctypes.c_int,           # rank
        ctypes.c_int,           # world_size
        ctypes.c_int,           # p
        ctypes.c_int,           # q
        ctypes.c_char_p,        # err_buf
        ctypes.c_int,           # err_buf_len
    ]
    lib.lrx_mesh_context_create.restype = ctypes.c_int64
    lib.lrx_mesh_context_destroy.argtypes = [ctypes.c_int64]
    lib.lrx_mesh_context_destroy.restype = None
    lib.lrx_mesh_init_mpi.argtypes = []
    lib.lrx_mesh_init_mpi.restype = None


def _register_ffi_targets(lib: ctypes.CDLL, platform: str) -> None:
    import jax.ffi
    for target_name, cpp_symbol in _PLATFORMS[platform]["targets"].items():
        # A partial build omits some handler symbols.  Skip the ones this
        # .so does not export instead of failing the whole load; the
        # capability probe is what turns that into a refusal, at resolve
        # time, naming the target.
        if not hasattr(lib, cpp_symbol):
            continue
        try:
            jax.ffi.register_ffi_target(
                target_name, jax.ffi.pycapsule(getattr(lib, cpp_symbol)),
                platform=platform)
        except Exception as exc:                              # noqa: BLE001
            # "already registered" is the normal case when lorrax's own
            # loader opened the same .so first: same file, same symbol,
            # same capsule target.  Anything else is real.
            if "already registered" not in str(exc).lower():
                raise


# ---------------------------------------------------------------------------
# LOAD ORDER: a device-capable process opens its device leg before the host leg.
# ---------------------------------------------------------------------------
# Introduced (2026-08-07) for a SONAME race between two builds of one third-party
# library that both legs linked; that library was removed on 2026-10-23
# (UPGRADE_NOTES.md), and the cross-.so ODR defect it exposed is fixed in C++
# (``exports_{cuda,host}.map``, ``cpp/common/c_abi.h``, per-leg ``PhdfCtx`` tags;
# ``test_so_acceptance.py`` check 6 is the ratchet).  The order is kept until
# ``test_so_acceptance.py`` check 5 is re-run on both clusters without it.  A
# CPU-platform process opens the host library only: a pre-open there cost the
# host phdf5 path (KNOWN_FAILURES B1).
#: Device legs this loader has already tried to open ahead of the host leg.
_ACCELERATOR_FIRST_TRIED: set = set()


def _open_accelerator_before_host() -> None:
    """Open the device leg ahead of the host leg (the load-order note above).

    THIS READS THE ENVIRONMENT AND STILL SELECTS NO BACKEND.  The module
    docstring's rule -- env grants a capability, a deck key picks the
    backend -- is intact: what is decided here is which FILES this process
    dlopens (:func:`lxkit.legs.accelerator_leg_to_open_first`), never which
    linalg backend answers a call.  A CPU-platform process opens the host
    library and nothing else.
    """
    key = accelerator_leg_to_open_first("cpu", _LIBS, _ACCELERATOR_FIRST_TRIED)
    if key is None:
        return
    try:
        get_lib(key)
    except Exception:                                          # noqa: BLE001
        pass


def loaded_platforms_in_order() -> list:
    """The platforms whose library this process has opened, in ORDER.

    The ordering above is the whole contract, so it is readable rather than
    inferable: ``tests`` assert on this list, and a diagnostic can print it.
    """
    return list(_LIBS)


def get_lib(platform: str) -> ctypes.CDLL:
    """Return the loaded FFI library for ``platform``; idempotent.

    ``platform`` is a :data:`lxkit.legs.LEGS` key.  It is REQUIRED — unlike
    lorrax's loader this one has no ``None`` default, because resolving it
    from the JAX default backend calls ``jax.default_backend()``, which
    INITIALIZES the XLA backend, and every caller here already knows the
    platform from its mesh.

    In a process that can host a device leg, opening the ``cpu`` library
    opens that leg's first — see the SONAME note above; it is a correctness
    requirement, not a warm-up.  In a CPU-platform process it does NOT:
    there is no device handler to protect and the mixed load costs that
    process its host phdf5 path.
    """
    if platform not in _PLATFORMS:
        raise LibraryNotBuilt(
            f"distrib_la: no FFI leg for platform {platform!r} "
            f"(known: {sorted(_PLATFORMS)}).")
    lib = _LIBS.get(platform)
    if lib is not None:
        return lib
    if platform == "cpu":
        _open_accelerator_before_host()
        lib = _LIBS.get(platform)
        if lib is not None:                    # re-entrancy, not reachable today
            return lib
    path = _locate_so(platform)
    lib, actual_origin = _native.open_and_attest(
        path, platform=platform, expected_abi=LORRAX_FFI_ABI_VERSION,
        abi_symbols=_ABI_SYMBOLS,
        build_hint=str(_PLATFORMS[platform]["build_hint"]),
        unusable_cls=LibraryUnusable, mismatch_cls=AbiMismatch)
    _bind_c_abi(lib, platform)
    _declare_cusolvermp(lib)
    _declare_mesh(lib)
    _declare_ctx_registry(lib)
    _register_ffi_targets(lib, platform)
    _LIBS[platform] = lib
    _LIB_PATHS[platform] = str(actual_origin)
    return lib


def loaded_lib_path(platform: str) -> Optional[str]:
    """The ``.so`` path loaded for ``platform``, or ``None``.  Pure lookup —
    never loads, so a caller can report an already-probed library without
    repeating the load side effects."""
    return _LIB_PATHS.get(platform)


def probe_target(target_name: str, platform: str):
    """``(usable, reason)`` for XLA target ``target_name`` on ``platform``.

    The three ways a target can be unusable have three different fixes, so
    they get three different reasons (:mod:`lxkit.probe` owns the wording):
    *unknown target* (a typo or a wrong-platform request), *library could
    not be loaded* (a dependency, an ``LD_LIBRARY_PATH``, a bind mount —
    and it says NOTHING about whether the handler is compiled), *loaded
    but does not export* (the genuine partial build, and the only one that
    means "rebuild").

    Conflating the middle case with the last produces an error that tells
    somebody to rebuild a library that is perfectly fine; on Frontera that
    silently downgraded an N×1 mesh to ``native`` with a "not compiled"
    diagnosis (wk_P G4, 2026-07-25).

    Never raises.
    """
    spec = _PLATFORMS.get(platform)
    if spec is None:
        return ProbeResult(False, (
            f"distrib_la builds no FFI library for platform {platform!r} "
            f"(known: {', '.join(sorted(_PLATFORMS))})."))
    sym = spec["targets"].get(target_name)
    if sym is None:
        return unknown_target(target_name, platform, spec["targets"])
    try:
        lib = get_lib(platform)
    except Exception as exc:                                   # noqa: BLE001
        return not_loadable(
            platform, exc, target=target_name,
            hint=(f"{spec['env']} selects the .so; LD_LIBRARY_PATH must "
                  f"cover every library that .so needs (run `ldd <so>` to "
                  f"see which are missing)."))
    if not hasattr(lib, sym):
        return missing_symbol(target_name, sym, _LIB_PATHS.get(platform, "?"),
                              build_hint=target_build_hint(platform, target_name))
    return AVAILABLE


def has_target(target_name: str, platform: str) -> bool:
    """True when ``platform``'s library is loadable AND exports the handler
    for ``target_name``.  Never raises.

    A bool is all an auto-policy decision needs.  Anything that REPORTS a
    refusal to a human uses :func:`probe_target` and quotes its reason — a
    bare False cannot distinguish "not built" from "LD_LIBRARY_PATH is
    wrong".
    """
    return bool(probe_target(target_name, platform)[0])


# ---------------------------------------------------------------------------
# Thin Pythonic helpers over the lrx_* entry points, so the wrappers do not
# each re-write ctypes plumbing.
# ---------------------------------------------------------------------------
def _check_err(rc: int, err_buf: ctypes.Array) -> None:
    if rc != 0:
        msg = err_buf.value.decode("utf-8", errors="replace")
        raise RuntimeError(f"lorrax_ffi error ({rc}): {msg}")


def nccl_unique_id_bytes() -> int:
    return int(get_lib("CUDA").lrx_nccl_unique_id_bytes())


def fill_nccl_unique_id(addr: int) -> None:
    err = ctypes.create_string_buffer(_ERR_CAP)
    rc = get_lib("CUDA").lrx_fill_nccl_unique_id(addr, err, _ERR_CAP)
    _check_err(rc, err)


def create_cusolvermp_context(rank: int, world_size: int,
                              nccl_unique_id_addr: int,
                              nccl_unique_id_nbytes: int,
                              p: int, q: int,
                              grid_layout_col_major: bool = True) -> int:
    lib = get_lib("CUDA")
    ctx_out = ctypes.c_int64(0)
    err = ctypes.create_string_buffer(_ERR_CAP)
    rc = lib.lrx_create_cusolvermp_context(
        int(rank), int(world_size),
        int(nccl_unique_id_addr), int(nccl_unique_id_nbytes),
        int(p), int(q),
        1 if grid_layout_col_major else 0,
        ctypes.byref(ctx_out), err, _ERR_CAP)
    _check_err(rc, err)
    return int(ctx_out.value)


def destroy_cusolvermp_context(ctx_handle: int) -> None:
    get_lib("CUDA").lrx_destroy_cusolvermp_context(int(ctx_handle))


def create_mesh_context(rank: int, world_size: int, p: int, q: int,
                        platform: str) -> int:
    """Collective create of an MPI mesh context; returns an opaque int64 handle.

    Inits ``MPI_THREAD_MULTIPLE`` if nothing else got there first, then splits
    ``MPI_COMM_WORLD`` with the JAX rank remap.  ``platform`` selects which
    library serves the call, so a CPU-mesh op never forces a device library to load.
    """
    lib = get_lib(platform)
    if not hasattr(lib, "lrx_mesh_context_create"):
        raise LibraryUnusable(
            f"MPI mesh context unavailable on {platform}: the selected FFI "
            "library does not export lrx_mesh_context_create")
    err = ctypes.create_string_buffer(_ERR_CAP)
    h = lib.lrx_mesh_context_create(
        int(rank), int(world_size), int(p), int(q), err, _ERR_CAP)
    if int(h) == 0:
        raise RuntimeError("lorrax_ffi mesh.context_create failed: "
                           + err.value.decode("utf-8", errors="replace"))
    return int(h)


def context_key(config: str) -> int:
    """The FFI ``ctx_key`` of a native context: a pure function of its configuration.

    An FFI attribute is baked into the HLO, so it must not be an address:
    the heap ``ctx_handle`` it replaces made every process emit a new module
    and the persistent compile cache miss on every run.  SHA-256 of the
    canonical configuration string, low 63 bits (never Python's salted
    ``hash``), so every rank and every run compute the same key.
    """
    digest = hashlib.sha256(config.encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)) or 1


def bind_context(platform: str, config: str, handle: int) -> int:
    """Bind ``handle`` under ``context_key(config)`` in ``platform``'s library; return the key.

    The registry stores ``config`` beside the key and refuses a collision or
    a second live context for one configuration (``ctx_registry.h``).
    """
    lib = get_lib(platform)
    key = context_key(config)
    if not hasattr(lib, "lrx_ctx_bind"):
        raise LibraryUnusable(
            f"{_LIB_PATHS.get(platform)} has no lrx_ctx_bind: it predates the ctx_key "
            "registry (handler ABI 6); rebuild and reseal both legs")
    err = ctypes.create_string_buffer(_ERR_CAP)
    if lib.lrx_ctx_bind(int(key), int(handle), config.encode("utf-8"), err, _ERR_CAP) != 0:
        raise RuntimeError("lorrax_ffi ctx_registry refused: "
                           + err.value.decode("utf-8", errors="replace"))
    return key


def unbind_context(platform: str, key: int) -> None:
    lib = _LIBS.get(platform)
    if lib is not None and hasattr(lib, "lrx_ctx_unbind"):
        lib.lrx_ctx_unbind(int(key))


def destroy_mesh_context(ctx_handle: int, platform: str) -> None:
    get_lib(platform).lrx_mesh_context_destroy(int(ctx_handle))


def mesh_init_mpi(platform: str) -> None:
    """Eagerly init ``MPI_THREAD_MULTIPLE`` outside the hot path.
    Idempotent; no-op if MPI is already initialized."""
    get_lib(platform).lrx_mesh_init_mpi()


def dial_key() -> tuple:
    """The ONE cache-key component capturing every factory-time distrib_la
    dial.

    distrib_la has no capability GATES — backend choice is a deck key and
    this package reads no environment for it (env grants capability only).
    What it does read, once, at library-load time, is WHICH ``.so`` is
    pinned; and a plan cache that omits that serves a stale backend after a
    mid-process pin flip, which is exactly what a test suite does.  So the
    dial key is the pins.

    Both reads are lexical (no JAX backend init) and O(1) — safe in any
    cache-lookup path at any P, which is the property
    :func:`lxkit.gate.dial_key` exists to preserve for gates.
    """
    return tuple((spec["env"], os.environ.get(spec["env"]))
                 for _, spec in sorted(_PLATFORMS.items()))


def active_eigh_workspace(capacity: int, platform: str = "CUDA") -> int:
    """Query the declared local eigensolver scratch (complex128 elements) on a device leg.

    No vendor edge escapes here: the leg's library answers through its own (suffixed,
    ``_bind_c_abi``) ``lrx_active_eigh_lwork``.  A negative answer is the solver's status
    (ROCM: -1000 = rocBLAS/rocSOLVER not mapped, GATE rocblas-mapped).
    """
    lib = get_lib(platform)
    try:
        query = lib.lrx_active_eigh_lwork
    except AttributeError as exc:
        raise LibraryUnusable(f"active subspace requires a rebuilt {platform} provider") from exc
    query.argtypes = [ctypes.c_int]
    query.restype = ctypes.c_int
    size = int(query(capacity))
    if size <= 0:
        raise LibraryUnusable(f"active eigensolver workspace query failed: {size}")
    return size

"""Locate, load and register the LORRAX FFI library — the LINALG half.

Ported from LORRAX ``src/ffi/common/ffi_loader.py`` at 96a6399, carrying
only what the distributed-linalg backends need: the two platform library
specs, the linalg target→C++-symbol tables, the ctypes declarations for
the cuSOLVERMp and SLATE context lifecycles, and the probe.  The FFT,
GEMM and phdf5 halves of that loader stay in lorrax, which still owns
them.

THIS IS THE QUARANTINE BOUNDARY.  Everything in this package that knows a
vendor library exists knows it through this module: one ``dlopen``, one
symbol table, one probe.  ``pyproject.toml`` declares no ScaLAPACK, no
SLATE and no cuSOLVERMp because none of them is linked here — they are
inside a ``.so`` that this file opens by path.

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
``LORRAX_FFI_HOST_SO``  absolute path to ``liblorrax_ffi_host.so`` (cpu)

The SAME two variables lorrax's loader reads, deliberately: a run pins its
library once and both loaders resolve the same file.  They GRANT A
CAPABILITY and never select a backend — backend choice is a deck key, and
:mod:`distrib_la.resolve` reads no environment at all.

An explicit pin that is missing is a REFUSAL, never a fall-through
(2026-08-06): a stale or mistyped pin used to be appended to the candidate
list, so the run silently used the in-tree build instead and the pin left
no trace.
"""

from __future__ import annotations

import ctypes
import glob
import os
import sys
from pathlib import Path
from typing import Dict, Optional

from lxkit.probe import (AVAILABLE, LibraryNotBuilt, LibraryUnusable,
                         ProbeResult, missing_symbol, not_loadable,
                         unknown_target)

__all__ = ["get_lib", "has_target", "probe_target", "loaded_lib_path",
           "loaded_platforms_in_order", "LibraryNotBuilt", "LibraryUnusable"]

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
    "lorrax_cusolvermp_eigh":             "EighMpFfi",
    "lorrax_cusolvermp_batched_potrf":    "CusolverMpBatchedPotrfFfi",
    "lorrax_cusolvermp_batched_potrs":    "CusolverMpBatchedPotrsFfi",
    "lorrax_cusolvermp_batched_solve_lu": "CusolverMpBatchedSolveLuFfi",
    "lorrax_slate_eigh":                  "SlateEighFfi",
    "lorrax_slate_potrf":                 "SlatePotrfFfi",
    "lorrax_slate_trsm":                  "SlateTrsmFfi",
    "lorrax_slate_batched_potrf":         "SlateBatchedPotrfFfi",
    "lorrax_slate_batched_trsm":          "SlateBatchedTrsmFfi",
}

# The host variants: the SAME target names registered under platform="cpu"
# (the split jaxlib itself uses for lapack vs cusolver), so a call site
# never mentions a platform and the lowering platform picks the handler.
# Only the C++ SYMBOLS differ, so the two .so's do not collide with EACH
# OTHER under RTLD_GLOBAL — but their DEPENDENCIES do, and one of those
# collisions is load-bearing: see the SONAME note above ``get_lib``.
# ScaLAPACK is host-only: there is no CUDA row for it anywhere.
_HOST_TARGET_SYMBOLS = {
    "lorrax_slate_eigh":                    "SlateEighHostFfi",
    "lorrax_slate_potrf":                   "SlatePotrfHostFfi",
    "lorrax_slate_trsm":                    "SlateTrsmHostFfi",
    "lorrax_slate_batched_potrf":           "SlateBatchedPotrfHostFfi",
    "lorrax_slate_batched_trsm":            "SlateBatchedTrsmHostFfi",
    "lorrax_scalapack_eigh":                "ScalapackEighHostFfi",
    "lorrax_scalapack_batched_solve_lu":    "ScalapackBatchedSolveLuHostFfi",
    "lorrax_scalapack_batched_getrf":       "ScalapackBatchedGetrfHostFfi",
    "lorrax_scalapack_batched_getrs":       "ScalapackBatchedGetrsHostFfi",
}

#: Per-platform library spec.  ``build_subdir`` is relative to
#: ``src/ffi/cpp/`` of the LORRAX checkout, when there is one (see
#: :func:`_candidate_paths`).
_PLATFORMS = {
    "CUDA": dict(
        so_name="liblorrax_ffi.so",
        env="LORRAX_FFI_SO",
        build_subdir="build",
        targets=_CUDA_TARGET_SYMBOLS,
        build_hint="src/ffi/cpp/run_shifter.sh bash src/ffi/cpp/build.sh",
    ),
    "cpu": dict(
        so_name="liblorrax_ffi_host.so",
        env="LORRAX_FFI_HOST_SO",
        build_subdir="build_host",
        targets=_HOST_TARGET_SYMBOLS,
        build_hint="bash config/perlmutter/build_ffi_host.sh --fresh",
    ),
}

#: What a request for a platform LORRAX builds no library for gets told.
#: ``rocm`` is in :data:`distrib_la.resolve._DISTRIBUTED_DEFAULT` as a
#: DECLARED-UNTESTED tier — the preference row exists so the day a ROCm
#: mesh appears the answer is a named refusal about the .so rather than a
#: nonsense "host-only" message about the backend.
_NO_LIBRARY = {
    "rocm": ("LORRAX builds no ROCm FFI library.  The ROCm tier is "
             "DECLARED-UNTESTED: distrib_la carries the backend preference "
             "rows (SLATE everywhere) so the routing question has an "
             "answer, but no ROCm .so has ever been built or measured."),
}

# Error buffer size for the lrx_ wrappers.
_ERR_CAP = 512


def _checkout_build_dir(platform: str) -> Optional[Path]:
    """``<lorrax>/src/ffi/cpp/<build_subdir>``, when this service sits
    inside a LORRAX checkout.

    The ``.so`` is a LORRAX build artifact, so an in-tree build directory
    is a real candidate whenever there is an in-tree — but distrib_la
    installs standalone too, where there is no ``src/ffi/cpp`` at all.
    Walk up rather than hard-code a depth: the answer must not change if
    the service directory moves.  ``None`` means "installed on its own",
    and then the env pin (or ``sys.path``) is the only route, which is
    what the not-found message says.
    """
    tail = Path("src") / "ffi" / "cpp" / _PLATFORMS[platform]["build_subdir"]
    for parent in Path(__file__).resolve().parents:
        cand = parent / tail
        if cand.is_dir():
            return cand
    return None


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
    spec = _PLATFORMS[platform]

    # AN EXPLICIT PIN THAT IS MISSING IS A REFUSAL, NEVER A FALL-THROUGH.
    # Same shape as lxkit.gate's "an explicit request that cannot be
    # honored refuses rather than silently downgrading".
    pinned = os.environ.get(spec["env"])
    if pinned:
        pin = Path(pinned)
        if pin.is_file():
            return pin
        raise LibraryUnusable(
            f"{spec['env']} is set to {pinned!r}, which is not a file.  "
            f"Refusing to fall back to another {spec['so_name']}: an "
            f"explicit pin that cannot be honored is a refusal, not a hint.  "
            f"Fix the path, or unset {spec['env']} to search the default "
            f"locations.")

    for c in _candidate_paths(platform):
        if c.is_file():
            return c
    hints = "\n  ".join(str(p) for p in _candidate_paths(platform)) or "(none)"
    raise LibraryNotBuilt(
        f"Could not locate {spec['so_name']} (platform={platform}).  "
        f"Build with:\n    {spec['build_hint']}\n"
        f"or pin it with {spec['env']}.\n"
        "Paths searched:\n  " + hints)


# ---------------------------------------------------------------------------
# THE C ABI IS PER-LIBRARY; THE PYTHON NAME IS NOT.
# ---------------------------------------------------------------------------
# ``cpp/slate/context.cc`` is pure MPI and compiles into BOTH platform
# libraries, so until 2026-08-08 both .so files defined these four
# ``extern "C"`` names (and five more from ``cpp/phdf5/api.cc``, which this
# package does not call).  Both are dlopened RTLD_GLOBAL, one name with two
# definitions behind it -- the C half of the cross-.so ODR violation
# lorrax's tests/KNOWN_FAILURES.md registered as L1.  ``cpp/common/c_abi.h``
# appends ``_host`` on the host leg; binding it under the plain name here
# keeps every call site below leg-agnostic.  Mirrors
# ``ffi.common.ffi_loader._bind_c_abi`` -- the two loaders open the SAME
# FILES, so both have to know the same naming rule.
_SLATE_C_ENTRY_POINTS = (
    "lrx_slate_context_create",
    "lrx_slate_subrow_context_create",
    "lrx_slate_context_destroy",
    "lrx_slate_init_mpi",
)

#: platform -> the suffix ``LRX_C_ENTRY`` appends on that leg.
_C_ABI_SUFFIX = {"CUDA": "", "cpu": "_host"}


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
    for base in _SLATE_C_ENTRY_POINTS:
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


def _declare_slate(lib: ctypes.CDLL) -> None:
    """SLATE context lifecycle — exported by BOTH platform libraries
    (``cpp/slate/context.cc`` is pure MPI and compiled into each).  Absent
    from a build made without SLATE, hence the guard."""
    if not hasattr(lib, "lrx_slate_context_create"):
        return
    lib.lrx_slate_context_create.argtypes = [
        ctypes.c_int,           # rank
        ctypes.c_int,           # world_size
        ctypes.c_int,           # p
        ctypes.c_int,           # q
        ctypes.c_char_p,        # err_buf
        ctypes.c_int,           # err_buf_len
    ]
    lib.lrx_slate_context_create.restype = ctypes.c_int64
    lib.lrx_slate_subrow_context_create.argtypes = [
        ctypes.c_int,           # rank (full-world)
        ctypes.c_int,           # world_size
        ctypes.c_int,           # Px
        ctypes.c_int,           # Py
        ctypes.c_char_p,        # err_buf
        ctypes.c_int,           # err_buf_len
    ]
    lib.lrx_slate_subrow_context_create.restype = ctypes.c_int64
    lib.lrx_slate_context_destroy.argtypes = [ctypes.c_int64]
    lib.lrx_slate_context_destroy.restype = None
    lib.lrx_slate_init_mpi.argtypes = []
    lib.lrx_slate_init_mpi.restype = None


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
# THE TWO PLATFORM LIBRARIES SHARE THEIR SLATE.  Open CUDA first — IN A
# PROCESS THAT CAN USE CUDA, and in no other.
# ---------------------------------------------------------------------------
# ``liblorrax_ffi.so`` and ``liblorrax_ffi_host.so`` do not merely differ in
# their C++ symbols (the claim two comments in this file used to make).  BOTH
# carry ``NEEDED libslate.so.2`` and ``NEEDED libblaspp.so.2``, and they
# resolve those SONAMEs out of DIFFERENT BUILDS:
#
#   host  DT_RPATH   .../slate_builds/cpu/install/lib64   gpu_backend=none
#   CUDA  DT_RUNPATH /lorrax_slate/lib, .../slate/install/lib64   CUDA
#
# ld.so keys a loaded object by SONAME, so whichever of the two is dlopened
# FIRST decides which ``libblaspp.so.2`` the OTHER one calls for the rest of
# the process.  The CPU build's ``blas::get_device_count()`` is a
# compiled-in 0 -- it has no ``libcudart`` to ask.
#
# MEASURED, Perlmutter 2026-08-07, ``dladdr`` of ``blas::get_device_count``
# at a failing cell, EXACTLY ONE visible GPU in both legs:
#
#   full-suite ``-m distrib_la``  host opened first
#       -> .../slate_builds/cpu/install/lib64/libblaspp.so.2, returns 0
#       -> FAILED_PRECONDITION: slate.potrf: blas::get_device_count()=0 but
#          JAX one-process-per-GPU model requires exactly 1.       8 CELLS RED
#   service-only (by path)        CUDA opened first
#       -> .../slate/install/lib64/libblaspp.so.2, returns 1       ALL GREEN
#
# What opened the host library first was a MODULE-SCOPE probe during
# COLLECTION: ``tests/test_fft_flat_k_numerics.py`` calls
# ``ffi_loader.probe_target(FLAT_K_TARGET, "cpu")`` at import time.  ``-m
# distrib_la`` deselects that file's CELLS; it does not un-import the module.
#
# So this is NOT a harness artefact.  Any GPU run whose first FFI touch is a
# host target -- a phdf5 host read, an mklfft probe -- silently gets a
# device-less SLATE, and the refusal it eventually prints names a device
# count nobody set.
#
# ONE SAFE ORDER EXISTS AMONG PROCESSES THAT OPEN BOTH, and the asymmetry is
# the reason: the host SLATE handlers run ``slate::Target::HostTask`` and
# never consult ``get_device_count``, so a CUDA blaspp in charge costs the
# host path nothing (measured: the service-only leg is 244 cells / 0 failed
# with exactly that arrangement).  The reverse costs the CUDA path
# everything.
#
# AND THE CHEAPER ORDER IS TO OPEN ONLY ONE.  MEASURED, Perlmutter
# 2026-08-07 (KNOWN_FAILURES B1): a pre-open with no CUDA arm to protect
# costs a CPU-platform process its host phdf5 path outright.  Bisect on
# ``tests/test_file_io.py``, CPU platform (JAX_PLATFORMS=cpu), four emulated
# host devices, same pins on every arm:
#
#   96a6399 branch base                  42 passed / 1 skipped
#   b3f3675 the commit before this rule  42 passed / 1 skipped
#   32e61fe this rule, unconditional     3 failed, Fatal Python error: Aborted
#
# and where it refused instead of aborting:
#
#   phdf5 read: logical slab out of bounds
#     extent=[2,4,1,6]
#     offset_base=[0,0,0,4596944070643295330]
#     valid_shape=[3,6,6,4609783128842618077]
#
# -- IEEE-754 float64 bit patterns (~0.19, ~1.87) decoded as int64: one
# library's handler reading the other's argument layout.  See INTERPOSITION
# below.
#
# SO THE PRE-OPEN IS TWO-ARMED, on the one question that decides whether
# there is a CUDA handler in this process for the order to protect:
#
#   CUDA-capable process  ->  CUDA first, then host.  SLATE survives, and
#                             the ``-m distrib_la`` leg stays 130 cells /
#                             0 failed.
#   CPU-platform process  ->  host only.  The CUDA library is NEVER dlopened.
#
# Best effort on top: no CUDA library built (every CPU-only tree, the WSL
# box) or one that cannot load (no driver on this node) leaves the
# environment untouched, and is tried once.
#
# ---------------------------------------------------------------------------
# INTERPOSITION -- THE LATENT DEFECT UNDERNEATH, registered not chased
# ---------------------------------------------------------------------------
# ``libslate``/``libblaspp`` are not the only SONAME-scale collision between
# the two files.  MEASURED 2026-08-07, ``nm -D --defined-only`` on the two
# pinned builds: SIXTEEN symbol names are DEFINED BY BOTH -- the nine
# C-linkage ``lrx_phdf5_*`` / ``lrx_slate_*`` entry points and seven mangled
# ``lorrax_ffi::phdf5::{open_ctx,close_ctx,ensure_dataset,ensure_read_buf,
# ensure_pinned,ensure_mpi_initialized,open_dataset_ro}``.  Both files are
# dlopened RTLD_GLOBAL, so once both are open the FIRST answers those names
# for BOTH, including for the other library's own internal calls.
#
# And they are not the same functions: lorrax's ``cpp/phdf5/ctx.h`` compiles
# ``PhdfCtx`` with CUDA stream / event / pinned-buffer members under
# ``#ifndef LORRAX_FFI_NO_CUDA`` and the host build defines that macro and
# drops them.  One C++ type name, two struct layouts, both exporting
# ``open_ctx(...) -> PhdfCtx*`` -- a cross-.so ODR violation, and reading one
# build's ``PhdfCtx`` at the other's offsets is what the garbage
# ``offset_base`` above looks like.
#
# NOT FIXED HERE and not chased: the fix is in C++ (distinct symbol
# namespaces per build, or a hidden-visibility phdf5 core), it is not this
# branch's regression, and the gate above keeps it latent by never putting a
# host-only process into the mixed state.  Registered in
# tests/KNOWN_FAILURES.md as L1.
_CUDA_FIRST_TRIED = False


def _nvidia_device_visible() -> bool:
    """A GPU device node this process can see.

    Split out from :func:`_process_can_use_cuda` so its cells can construct
    both answers without monkeypatching ``glob``/``os.path`` for the whole
    session -- the hardware half of the question is the one a pure test
    cannot arrange for real.
    """
    return (bool(glob.glob("/dev/nvidia[0-9]*"))
            or os.path.exists("/dev/nvidiactl"))


def _process_can_use_cuda() -> bool:
    """Can THIS process run CUDA work at all?  Environment only.

    TRUTHFUL AT LOAD TIME is the whole requirement, and it rules out the
    obvious answer: ``jax.default_backend()`` INITIALIZES the XLA backend --
    the same reason :func:`get_lib` refuses to default its ``platform``
    argument -- so asking it from inside a loader call would make the loader
    decide the process's backend.  Both signals below are ones JAX itself
    acts on and both are readable before any backend exists:

      * ``JAX_PLATFORMS``.  The first entry wins, mirroring how JAX picks
        its default backend; ``cpu`` there forbids a GPU backend outright,
        so no CUDA handler in this process can ever run and there is
        nothing for the load order to protect.
      * A VISIBLE NVIDIA DEVICE.  ``CUDA_VISIBLE_DEVICES=""`` is the
        spelling that means "no GPU, deliberately", and JAX's own
        ``backends()`` skips cuda when no device node is visible
        (``xla_bridge.py``: ``if platform == "cuda" and not
        has_visible_nvidia_gpu(): continue``).

    THIS READS THE ENVIRONMENT AND STILL SELECTS NO BACKEND.  The module
    docstring's rule -- env grants a capability, a deck key picks the
    backend -- is intact: what is being decided here is which FILES this
    process dlopens, never which linalg backend answers a call.

    Deliberately NOT a probe of the library.  "Is the CUDA .so loadable
    here" is a different question and the best-effort ``try`` in
    :func:`_open_cuda_before_host` is what answers it.
    """
    first = os.environ.get("JAX_PLATFORMS", "").split(",")[0].strip().lower()
    if first and first not in ("cuda", "gpu"):
        return False
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip() == "":
        return False
    return _nvidia_device_visible()


def _open_cuda_before_host() -> None:
    """Win the ``libslate``/``libblaspp`` SONAME for the CUDA build.

    Only where there is a CUDA build in the running to win it: a
    CPU-platform process opens the host library and nothing else.
    """
    global _CUDA_FIRST_TRIED
    if _CUDA_FIRST_TRIED or "CUDA" in _LIBS:
        return
    if not _process_can_use_cuda():
        return
    _CUDA_FIRST_TRIED = True
    try:
        get_lib("CUDA")
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

    ``platform`` is ``"CUDA"`` or ``"cpu"``.  It is REQUIRED — unlike
    lorrax's loader this one has no ``None`` default, because resolving it
    from the JAX default backend calls ``jax.default_backend()``, which
    INITIALIZES the XLA backend, and every caller here already knows the
    platform from its mesh.

    In a CUDA-CAPABLE process, opening the ``cpu`` library opens the
    ``CUDA`` one first — see the SONAME note above; it is a correctness
    requirement, not a warm-up.  In a CPU-platform process it does NOT:
    there is no CUDA handler to protect and the mixed load costs that
    process its host phdf5 path.
    """
    if platform not in _PLATFORMS:
        raise LibraryNotBuilt(
            f"distrib_la: no FFI library for platform {platform!r} "
            f"(known: {sorted(_PLATFORMS)})."
            + (f"  {_NO_LIBRARY[platform]}" if platform in _NO_LIBRARY else ""))
    lib = _LIBS.get(platform)
    if lib is not None:
        return lib
    if platform == "cpu":
        _open_cuda_before_host()
        lib = _LIBS.get(platform)
        if lib is not None:                    # re-entrancy, not reachable today
            return lib
    path = _locate_so(platform)
    # Load h5py's HDF5 FIRST.  Both LORRAX libraries link the site's
    # parallel HDF5 and are dlopened RTLD_GLOBAL, which publishes every
    # ``H5*`` symbol globally; h5py ships its OWN, ABI-incompatible HDF5,
    # and a later ``import h5py`` binds to the site symbols and dies with
    # ``ValueError: Not a datatype`` (measured, wk_AG job 7876369).
    # Production imports h5py long before any FFI probe, so this never
    # fired in a driver -- but every diagnostic that probes the FFI first
    # hits it, and nothing said why.
    try:
        import h5py                                            # noqa: F401
    except Exception:                                          # noqa: BLE001
        pass
    try:
        lib = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:
        # The file is THERE; its dependency closure is not resolvable.
        raise LibraryUnusable(
            f"{path} exists but could not be loaded: {exc}.  This is a BROKEN "
            f"BUILD OR ENVIRONMENT, not an absent library: check "
            f"LD_LIBRARY_PATH and, in a container, that every RPATH "
            f"directory is actually bind-mounted "
            f"(ldd {path} | grep 'not found').") from exc
    _bind_c_abi(lib, platform)
    _declare_cusolvermp(lib)
    _declare_slate(lib)
    _register_ffi_targets(lib, platform)
    _LIBS[platform] = lib
    _LIB_PATHS[platform] = str(path)
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
            f"(known: {', '.join(sorted(_PLATFORMS))})."
            + (f"  {_NO_LIBRARY[platform]}" if platform in _NO_LIBRARY else "")))
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
                              build_hint=spec["build_hint"])
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


def create_slate_context(rank: int, world_size: int, p: int, q: int,
                         platform: str) -> int:
    """Collective create of a SLATE context; returns an opaque int64 handle.

    Inits ``MPI_THREAD_MULTIPLE`` if nothing else got there first, then
    dups ``MPI_COMM_WORLD`` for SLATE's exclusive use.  ``platform``
    selects which library serves the call, so a CPU-mesh op never forces
    the CUDA library to load.
    """
    lib = get_lib(platform)
    err = ctypes.create_string_buffer(_ERR_CAP)
    h = lib.lrx_slate_context_create(
        int(rank), int(world_size), int(p), int(q), err, _ERR_CAP)
    if int(h) == 0:
        raise RuntimeError("lorrax_ffi slate.context_create failed: "
                           + err.value.decode("utf-8", errors="replace"))
    return int(h)


def create_slate_subrow_context(rank: int, world_size: int,
                                Px: int, Py: int, platform: str) -> int:
    """Collective create of a per-X-row SLATE sub-comm context.

    ``MPI_COMM_WORLD`` split by x-coordinate (color=x_rank, key=y_rank),
    giving one comm of size ``Py`` per X-row — what the batched wrappers
    need, where each row independently processes a slice of a
    ``(Nbatch, N, N)`` input distributed as ``P('x', None, 'y')``.
    """
    lib = get_lib(platform)
    err = ctypes.create_string_buffer(_ERR_CAP)
    h = lib.lrx_slate_subrow_context_create(
        int(rank), int(world_size), int(Px), int(Py), err, _ERR_CAP)
    if int(h) == 0:
        raise RuntimeError("lorrax_ffi slate.subrow_context_create failed: "
                           + err.value.decode("utf-8", errors="replace"))
    return int(h)


def destroy_slate_context(ctx_handle: int, platform: str) -> None:
    get_lib(platform).lrx_slate_context_destroy(int(ctx_handle))


def slate_init_mpi(platform: str) -> None:
    """Eagerly init ``MPI_THREAD_MULTIPLE`` from outside SLATE's hot path.
    Idempotent; no-op if MPI is already initialized."""
    get_lib(platform).lrx_slate_init_mpi()


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

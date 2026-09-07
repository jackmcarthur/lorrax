"""Locate, load, and register the LORRAX FFI shared libraries.

The libraries are plain C shared objects (no pybind/nanobind).  We load
them with :class:`ctypes.CDLL`, declare the entry-point signatures, and
register each XLA FFI handler symbol with ``jax.ffi.register_ffi_target``
on first use.

There is one library per XLA platform, registering the SAME target names
under different ``platform=`` strings — the same split jaxlib uses for
its cpu (lapack) vs CUDA (cusolver) kernels, so ``jax.ffi.ffi_call``
sites resolve the right handler from the lowering platform and never
mention a platform themselves:

    CUDA  liblorrax_ffi.so       cuSOLVERMp/cuBLASMp/phdf5/slate/cuFFT flat-k
    cpu   liblorrax_ffi_host.so  phdf5 read+write / slate (Target::HostTask)
                                 / ScaLAPACK / MKL-DFTI flat-k / MKL GEMM

Public API
----------
get_lib(platform=None) : ctypes.CDLL
    The loaded library for ``platform`` ("CUDA" or "cpu"; default = the
    JAX default backend's platform) with ctypes argtypes/restype set on
    its ``lrx_*`` wrapper functions.  The XLA FFI handlers remain raw
    ``ctypes._FuncPtr`` — we pass them to ``jax.ffi.pycapsule`` rather
    than calling them directly.

Environment overrides
---------------------
LORRAX_FFI_SO
    Absolute path to ``liblorrax_ffi.so`` (CUDA).  Takes precedence.
LORRAX_FFI_HOST_SO
    Absolute path to ``liblorrax_ffi_host.so`` (cpu).  Takes precedence.

Where the library's own dependencies come from (audited 2026-07-27, AW)
----------------------------------------------------------------------
Loading the FFI library also loads everything it needs, searched first
in the directory list baked into the library at link time (its *built-in
search path*, set by ``config/frontera/build_ffi_host.sh``) and then in
``LD_LIBRARY_PATH``.  On Frontera the unified host lib's built-in path
already covers Intel MPI ``lib/release``, SLATE's ``lib64`` and MKL —
what ``LD_LIBRARY_PATH`` MUST add is only what it does not cover: the
parallel-HDF5 ``lib`` (``libhdf5.so.310``), Intel MPI's
``libfabric/lib``, and (for the overlay h5py, not for this .so) the
Intel compiler runtime.  Two ordering facts, so nobody re-derives
them:

* **"SLATE first" is convention, not requirement.**  The SLATE dir
  holds only ``libslate``/``libblaspp``/``liblapackpp`` (no libmpi, no
  libhdf5, no MKL), so its position can shadow nothing; libslate's own
  built-in search path finds its blaspp/lapackpp.  Ordering among the
  LORRAX entries is not load-bearing — PRESENCE is (a missing entry is
  the ``probe_target`` "library could not be loaded" case).
* The one measured ordering hazard is host-lib STAGING into the
  container: a bare ``/hostlibs`` (host ``/usr/lib64``) *prepended* to
  ``LD_LIBRARY_PATH`` shadows the container glibc and kills every
  binary — staged RDMA userspace must be APPENDED (scorecard AP.2 /
  AS.1; the working pattern is wk_AS ``as_inner.sh``).

MPI coexistence (scorecard AS): this library's handlers MPI_Init the
process libmpi with ``MPI_THREAD_MULTIPLE`` if nothing else got there
first.  Under ``JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi`` jax's
MPItrampoline runtime inits FIRST (requesting only FUNNELED), so that
site must arrange a live ``MPI_THREAD_MULTIPLE`` grant before the FFI's
concurrent MPI work starts — measured ~29% multi-node failure rate below
that level (AS.4b). Frontera upgrades the request in its patched wrapper;
Perlmutter uses an unmodified wrapper plus Cray async progress. LORRAX
deliberately does NOT default ``MPITRAMPOLINE_LIB`` itself: it is a
harness-level machine fact naming an adapter built for the site MPI.
``cpp/phdf5/context.cc`` refuses an insufficient live grant at open time.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Dict, Optional

from ffi import _services

_services.ensure_on_path()

import jax
import jax.ffi
from lxkit import native_provider as _native

__all__ = ["get_lib", "has_target", "probe_target", "has_phdf5_read",
           "has_phdf5_write", "loaded_lib_path",
           "LORRAX_FFI_ABI_VERSION", "FfiAbiMismatch"]

_LIBS: Dict[str, ctypes.CDLL] = {}
#: platform -> the .so path actually loaded (for diagnostics).
_LIB_PATHS: Dict[str, str] = {}

# Symbols the XLA FFI side exports (plain C via XLA_FFI_DEFINE_HANDLER_SYMBOL),
# per platform.  One handler per routine covers all supported dtypes —
# dispatch is done inside the .so based on the input buffer's element type.
_CUDA_TARGET_SYMBOLS = {
    "lorrax_cusolvermp_eigh":       "EighMpFfi",
    "lorrax_cusolvermp_batched_potrf":    "CusolverMpBatchedPotrfFfi",
    "lorrax_cusolvermp_batched_potrs":    "CusolverMpBatchedPotrsFfi",
    "lorrax_cusolvermp_batched_solve_lu": "CusolverMpBatchedSolveLuFfi",
    "lorrax_cublasmp_batched_gemm":       "CublasMpBatchedGemmFfi",
    "lorrax_cublasmp_batched_w_solve":    "CublasMpBatchedWSolveFfi",
    # cuFFT strided flat-k batched-FFT handlers (cpp/cufft) — the CUDA
    # platform mirror of the mklfft host handlers below.  The target STRINGS
    # deliberately keep the host table's "mklfft" names (they were coined by
    # the CPU prototype): common.fft_helpers issues ONE platform-agnostic
    # ffi_call per site and the lowering platform resolves the handler —
    # exactly the phdf5 same-target/different-symbol split.
    "lorrax_mklfft_flat_k":         "CufftFlatKCudaFfi",
    "lorrax_mklfft_gw_conv":        "CufftGwConvCudaFfi",
    # k-MINOR fused conv (cpp/cufft/conv_kminor_cuda_ffi.cc) — ifft·multiply·
    # fft over a MINOR-most k axis, one kernel, no layout change on the
    # caller's side.  CUDA-ONLY BY CONSTRUCTION and named accordingly: unlike
    # the two rows above there is no host twin, so borrowing the "mklfft"
    # target string would promise a cpu handler that does not exist.  A cpu
    # mesh must refuse here, not resolve to nothing (ffi.fft.CONV_KMINOR_GATE).
    "lorrax_cufft_conv_kminor":     "CufftConvKMinorCudaFfi",
    # Direct k-leading candidate: public T/W/U remain k-leading; the handler
    # coalesces loads into resident rows and writes k-leading output, with no
    # global pack or transpose.  CUDA-only; no production Sigma caller exists
    # until its separate seam lands.
    "lorrax_cufft_conv_klead":      "CufftConvKLeadCudaFfi",
    # ISDF CCT/ZCT post-pair convolution.  Both rank-7 inputs keep the three
    # k axes leading; the handler contracts the two spin axes and returns the
    # rank-5 tile.  CUDA-only, with the unchanged JAX chain as its fallback.
    "lorrax_cufft_conv_kpair":      "CufftConvKPairCudaFfi",
    "lorrax_cufft_conv_kparent":    "CufftConvKParentCudaFfi",
    "lorrax_phdf5_write":           "PhdfWriteFfi",
    "lorrax_phdf5_read":            "PhdfReadFfi",
    "lorrax_phdf5_read_kchunk":       "PhdfReadKchunkFfi",
    "lorrax_phdf5_read_kchunk_union": "PhdfReadKchunkUnionFfi",
    "lorrax_slate_eigh":              "SlateEighFfi",
    "lorrax_slate_potrf":             "SlatePotrfFfi",
    "lorrax_slate_trsm":              "SlateTrsmFfi",
    "lorrax_slate_batched_potrf":     "SlateBatchedPotrfFfi",
    "lorrax_slate_batched_trsm":      "SlateBatchedTrsmFfi",
}

# Host variants of the slate targets (src/ffi/cpp/slate/host_ffi.cc) —
# same target names as the CUDA table, registered under platform="cpu" —
# plus the host-only ScaLAPACK targets (src/ffi/cpp/scalapack/, MKL/LibSci;
# their python side is services/distrib_la now, but the SYMBOLS are in both
# loaders' tables because both loaders open the same .so),
# plus the phdf5 read AND write handlers (src/ffi/cpp/phdf5/{read,write}_ffi.cc
# compiled with -DLORRAX_FFI_NO_CUDA).  The phdf5 target STRINGS are identical
# to the CUDA table so the ffi.phdf5.{read,write} ffi_call sites resolve by
# lowering platform; only the C++ SYMBOL names differ (Phdf*HostFfi vs
# Phdf*Ffi) so the two platform .so's can co-exist under RTLD_GLOBAL.
#
# ``lorrax_phdf5_write`` is what makes SlabIO's tile path reachable on
# the CPU backend (workstream AE) — a host lib built before that port exports
# the three read symbols only, and ``has_phdf5_write('cpu')`` is False, which
# means the tile path is unavailable, and SlabIO then REFUSES rather than
# moving the bytes some other way.  There is no demotion: the tiers this
# comment used to name (PHDF5_HOST, H5PY_ALLGATHER) and the gw_config
# router that chose between them were deleted in the one-backend port.
# An allgather is a refusal, not a fallback -- owner ruling 2026-08-05 --
# because the design envelope is arrays needing hundreds of GPUs to hold,
# where a rank-0 gather is an OOM and not a slow path.  A host lib without
# the write symbol is a BUILD defect to fix, not a routing condition.
_HOST_TARGET_SYMBOLS = {
    "lorrax_slate_eigh":              "SlateEighHostFfi",
    "lorrax_slate_potrf":             "SlatePotrfHostFfi",
    "lorrax_slate_trsm":              "SlateTrsmHostFfi",
    "lorrax_slate_batched_potrf":     "SlateBatchedPotrfHostFfi",
    "lorrax_slate_batched_trsm":      "SlateBatchedTrsmHostFfi",
    "lorrax_scalapack_batched_solve_lu": "ScalapackBatchedSolveLuHostFfi",
    "lorrax_scalapack_batched_getrf": "ScalapackBatchedGetrfHostFfi",
    "lorrax_scalapack_batched_getrs": "ScalapackBatchedGetrsHostFfi",
    "lorrax_scalapack_eigh":          "ScalapackEighHostFfi",
    # MKL FFT (DFTI API) flat-k batched-FFT handlers (cpp/mklfft) — the
    # LORRAX_FFT_FFI backend of common.fft_helpers (FFT-FFI prototype).
    "lorrax_mklfft_flat_k":           "MklFftFlatKHostFfi",
    "lorrax_mklfft_gw_conv":          "MklFftGwConvHostFfi",
    # MKL batched-GEMM handler (cpp/mklblas) — the LORRAX_BANDS_GEMM_FFI
    # body of common.contract_bands (contract_bands_block_reshard).
    "lorrax_mklblas_gemm_batch":      "MklBlasGemmBatchHostFfi",
    "lorrax_phdf5_read":              "PhdfReadHostFfi",
    "lorrax_phdf5_read_kchunk":       "PhdfReadKchunkHostFfi",
    "lorrax_phdf5_read_kchunk_union": "PhdfReadKchunkUnionHostFfi",
    "lorrax_phdf5_write":             "PhdfWriteHostFfi",
}

# Per-platform library spec: .so filename, env-var override, in-tree build
# dir (relative to src/ffi/cpp/, the one C++ tree), target table, and the
# build command for the not-found error.
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
        build_hint="bash src/ffi/cpp/build_host.sh",
    ),
}

# Error buffer size for the lrx_ wrappers.
_ERR_CAP = 512


def _default_platform() -> str:
    # NOTE: jax.default_backend() INITIALIZES the XLA backend.  Code that
    # runs before jax.distributed.initialize (multi-rank CLI drivers) must
    # not call get_lib(None) — use get_lib(platform_from_env()) instead.
    backend = jax.default_backend()
    if backend in ("gpu", "cuda"):
        return "CUDA"
    if backend == "cpu":
        return "cpu"
    raise RuntimeError(
        f"lorrax_ffi: no FFI library for JAX backend {backend!r} "
        f"(supported: cuda, cpu).")


def platform_from_env(default: str = "CUDA") -> str:
    """Resolve the FFI platform from ``JAX_PLATFORMS`` WITHOUT touching the
    JAX backend — safe before ``jax.distributed.initialize``.  The first
    entry wins, mirroring how JAX picks its default backend from the list.
    """
    first = os.environ.get("JAX_PLATFORMS", "").split(",")[0].strip().lower()
    if not first:
        return default
    return "CUDA" if first in ("cuda", "gpu") else "cpu"


def _candidate_paths(platform: str) -> list[Path]:
    spec = _PLATFORMS[platform]
    paths: list[Path] = []
    env = os.environ.get(spec["env"])
    if env:
        paths.append(Path(env))
    here = Path(__file__).resolve()
    build_dir = here.parent.parent / "cpp" / spec["build_subdir"]
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


class FfiLibraryNotBuilt(FileNotFoundError):
    """No FFI .so exists for this platform anywhere on the search path.

    This is the LEGITIMATE platform gate: a CPU-only laptop, or a checkout
    that was never built.  A caller may reasonably skip on it.
    """


#: The handler-signature ABI this Python tree speaks.
#:
#: A MIRROR.  The definition is ``src/ffi/cpp/common/lorrax_ffi_abi.h``, which
#: also carries the bump rule and the history of the two bumps in two days that
#: motivated the mechanism.  A C++ macro cannot be imported, so a mirror across
#: this boundary is unavoidable; what IS avoidable is the mirror drifting
#: silently, and ``tests/test_ffi_abi_stamp.py`` parses the header and fails if
#: these two numbers disagree.
LORRAX_FFI_ABI_VERSION = 3

#: platform -> the C entry point that reports the library's ABI.  Per leg on
#: purpose: both libraries are dlopened RTLD_GLOBAL in a GPU process and
#: already share sixteen symbol names (KNOWN_FAILURES L1), so a shared spelling
#: here would let the first-loaded library answer for the second — and this is
#: the one question where being answered by the wrong library is the exact
#: failure being detected.
_ABI_SYMBOLS = {"cpu": "lorrax_ffi_host_abi_version",
                "CUDA": "lorrax_ffi_cuda_abi_version"}

class FfiLibraryUnusable(OSError):
    """An FFI .so WAS located but cannot be used.

    Distinct from :class:`FfiLibraryNotBuilt` on purpose.  "The library is
    absent" and "the library is present and broken" want opposite
    responses -- the first is an environment fact, the second is a DEFECT,
    and collapsing them into one exception is how a broken dependency
    closure gets reported as a platform skip.  Live example, measured
    2026-08-06 inside the Shifter container on nid001644: the host .so at
    src/ffi/cpp/build_host/liblorrax_ffi_host.so exists, but /opt/cray/pe
    is not bind-mounted, so all three libfftw3*.so.mpi31.3 RPATH deps are
    "not found" and 19 Tier-1 cells reported themselves as skipped.
    """


class FfiAbiMismatch(FfiLibraryUnusable):
    """The library loaded fine and speaks a DIFFERENT handler ABI.

    A subclass of :class:`FfiLibraryUnusable` because it is the same kind of
    fact — the file is present and cannot be used — and every existing
    handler of that exception stays correct.  A distinct class because this
    one has a distinct fix (rebuild the .so from this tree, or pin the .so
    this tree was built for) and because a caller that wants to say
    "mispaired" rather than "broken" now can.
    """


def _check_abi(lib: ctypes.CDLL, platform: str, path: str) -> None:
    """Refuse a library whose handler signatures are not this tree's.

    CALLED IMMEDIATELY AFTER ``CDLL`` AND BEFORE ANYTHING ELSE TOUCHES THE
    LIBRARY.  Order is the requirement, not an optimisation: registering
    targets or setting argtypes on a mispaired library is exactly the state
    whose only symptom is a runtime ``INVALID_ARGUMENT`` several minutes into
    an allocated run.

    THE TWO OUTCOMES ARE DELIBERATELY DIFFERENT.

    *Stamped and different* is a REFUSAL.  The two sides disagree about what
    crosses the boundary; nothing good happens next.  What used to happen
    instead was::

        INVALID_ARGUMENT: Wrong number of arguments: expected 3 but got 4

    which names neither library, neither version, nor what to do.

    *Not stamped at all* is an ANNOUNCEMENT, once, and the load proceeds.  A
    library built before 2026-08-08 carries no stamp, and "unstamped" is not
    evidence of "wrong" — roughly nine worktrees pin such libraries today and
    refusing them wholesale would break every one of them to fix none.  A site
    that wants the ratchet closed sets ``LORRAX_FFI_ABI_STRICT=1``.
    """
    _native.check_abi(
        lib, platform, path, expected_abi=LORRAX_FFI_ABI_VERSION,
        abi_symbols=_ABI_SYMBOLS,
        build_hint=str(_PLATFORMS[platform]["build_hint"]),
        strict_unstamped=os.environ.get("LORRAX_FFI_ABI_STRICT", "") == "1",
        mismatch_cls=FfiAbiMismatch)


def _locate_so(platform: str) -> Path:
    return _native.locate_library(
        platform, specs=_PLATFORMS,
        candidates={name: _candidate_paths(name) for name in _PLATFORMS},
        expected_abi=LORRAX_FFI_ABI_VERSION,
        not_built_cls=FfiLibraryNotBuilt,
        unusable_cls=FfiLibraryUnusable)


# ---------------------------------------------------------------------------
# THE C ABI IS PER-LIBRARY; THE PYTHON NAME IS NOT.
# ---------------------------------------------------------------------------
# ``cpp/phdf5/api.cc`` and ``cpp/slate/context.cc`` are CUDA-free and compile
# into BOTH platform libraries, so until 2026-08-08 both .so files defined
# these NINE ``extern "C"`` names.  Both are dlopened RTLD_GLOBAL, which puts
# one name and two definitions into one process namespace -- the C half of
# the cross-.so ODR violation KNOWN_FAILURES registered as L1.
#
# ``cpp/common/c_abi.h``'s ``LRX_C_ENTRY`` now appends ``_host`` on the host
# leg, the same per-library renaming the ``*HostFfi`` handlers already used,
# so ``nm -D --defined-only`` on the two libraries intersects in nothing.
# Binding the suffixed symbol under the unsuffixed PYTHON name here, once,
# keeps that a build-level fact: no call site in this module, in
# ``distrib_la.loader``, in ``file_io._slab_io_ffi`` or in the bench scripts
# has to know which leg it is talking to.
_SHARED_C_ENTRY_POINTS = (
    "lrx_phdf5_open",
    "lrx_phdf5_close",
    "lrx_phdf5_init_mpi",
    "lrx_phdf5_ensure_dataset",
    "lrx_phdf5_open_dataset_ro",
    # The metadata half (2026-08-22).  Absent from any library built before
    # that date; both this binder and ``_declare_phdf5`` are hasattr-guarded,
    # so an older .so simply does not acquire the names and
    # ``file_io._slab_io_ffi`` announces the degraded introspect route.
    "lrx_phdf5_dataset_geometry",
    "lrx_phdf5_read_whole",
    "lrx_slate_context_create",
    "lrx_slate_subrow_context_create",
    "lrx_slate_context_destroy",
    "lrx_slate_init_mpi",
)

#: platform -> the suffix ``LRX_C_ENTRY`` appends on that leg.
_C_ABI_SUFFIX = {"CUDA": "", "cpu": "_host"}


def _bind_c_abi(lib: ctypes.CDLL, platform: str) -> None:
    """Bind this leg's suffixed C entry points under their plain names.

    A NO-OP ON A PRE-2026-08-08 LIBRARY, ON PURPOSE.  An older host .so
    exports the unsuffixed names, ``getattr`` below raises, and every call
    site then resolves the unsuffixed symbol through ``ctypes.CDLL``'s own
    ``__getattr__`` exactly as it always did.  Refusing here instead would
    strand every worktree pinned to the deployed Aug-7 pair over a defect
    those libraries have had since they were built -- the ratchet belongs
    on the ARTIFACT, and it is there:
    ``services/distrib_la/tests/test_so_acceptance.py``'s check 6 fails on
    any pinned pair that still shares a defined symbol.
    """
    suffix = _C_ABI_SUFFIX.get(platform, "")
    if not suffix:
        return
    for base in _SHARED_C_ENTRY_POINTS:
        try:
            fn = getattr(lib, base + suffix)
        except AttributeError:
            continue
        # ctypes.CDLL.__getattr__ only fires when normal lookup fails, so an
        # instance attribute wins from here on -- including for the
        # ``hasattr`` guards in _declare_phdf5 / _declare_slate below.
        setattr(lib, base, fn)


def _set_argtypes(lib: ctypes.CDLL, platform: str) -> None:
    """Declare argtypes/restype for the lrx_* entry points ``lib`` exports."""
    _bind_c_abi(lib, platform)
    if platform == "CUDA":
        _declare_cuda_stack(lib)
    # phdf5 lifecycle (CUDA-free) + slate lifecycle (pure MPI) are exported by
    # WHICHEVER platform library was built with them — the phdf5 host read
    # path drives lrx_phdf5_* through liblorrax_ffi_host.so.  Both declare
    # under hasattr guards so a partial build is fine.
    _declare_phdf5(lib)
    _declare_slate(lib)


def _declare_cuda_stack(lib: ctypes.CDLL) -> None:
    """NCCL / cuSOLVERMp / phdf5 lifecycle — liblorrax_ffi.so only."""

    lib.lrx_nccl_unique_id_bytes.argtypes = []
    lib.lrx_nccl_unique_id_bytes.restype  = ctypes.c_int

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
    lib.lrx_destroy_cusolvermp_context.restype  = None

    lib.lrx_smoke_allreduce_sum.argtypes = [
        ctypes.c_int64, ctypes.c_void_p, ctypes.c_int,
    ]
    lib.lrx_smoke_allreduce_sum.restype = ctypes.c_int

    lib.lrx_version_info.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.lrx_version_info.restype = ctypes.c_int


def _declare_phdf5(lib: ctypes.CDLL) -> None:
    """parallel-HDF5 lifecycle — exported by whichever platform library was
    built with the phdf5 subpackage (the CUDA-free wrappers in
    cpp/phdf5/api.cc compile into BOTH liblorrax_ffi.so and, for the host
    read path, liblorrax_ffi_host.so).  Absent from a
    -DLORRAX_FFI_HAVE_PHDF5=0 build; declared under a hasattr guard so
    loading such a .so doesn't fail."""
    if not hasattr(lib, "lrx_phdf5_open"):
        return
    lib.lrx_phdf5_open.argtypes = [
        ctypes.c_char_p,                     # path
        ctypes.c_int, ctypes.c_int,          # p, q
        ctypes.c_int, ctypes.c_int,          # rank, world_size
        ctypes.c_int,                        # mode_flag
        ctypes.POINTER(ctypes.c_int64),      # ctx_out
        ctypes.c_char_p, ctypes.c_int,       # err_out, err_cap
    ]
    lib.lrx_phdf5_open.restype = ctypes.c_int
    lib.lrx_phdf5_close.argtypes = [ctypes.c_int64]
    lib.lrx_phdf5_close.restype  = None
    lib.lrx_phdf5_init_mpi.argtypes = []
    lib.lrx_phdf5_init_mpi.restype  = None

    lib.lrx_phdf5_ensure_dataset.argtypes = [
        ctypes.c_int64,                      # ctx_handle
        ctypes.c_char_p,                     # ds_name
        ctypes.POINTER(ctypes.c_int64),      # shape[] (N ints)
        ctypes.c_int,                        # ndim
        ctypes.c_int,                        # dtype_tag
        ctypes.POINTER(ctypes.c_int64),      # ds_id_out (hid_t)
        ctypes.c_char_p, ctypes.c_int,       # err_out, err_cap
    ]
    lib.lrx_phdf5_ensure_dataset.restype = ctypes.c_int

    lib.lrx_phdf5_open_dataset_ro.argtypes = [
        ctypes.c_int64,                      # ctx_handle
        ctypes.c_char_p,                     # ds_name
        ctypes.POINTER(ctypes.c_int64),      # ds_id_out (hid_t)
        ctypes.c_char_p, ctypes.c_int,       # err_out, err_cap
    ]
    lib.lrx_phdf5_open_dataset_ro.restype = ctypes.c_int

    # The metadata half.  hasattr-guarded INDIVIDUALLY, not folded into the
    # ``lrx_phdf5_open`` guard above: a library predating 2026-08-22 exports
    # the lifecycle names and not these two, so one guard for the whole
    # group would refuse to bind anything on an otherwise-fine .so.
    if hasattr(lib, "lrx_phdf5_dataset_geometry"):
        lib.lrx_phdf5_dataset_geometry.argtypes = [
            ctypes.c_int64,                      # ctx_handle
            ctypes.c_char_p,                     # ds_name
            ctypes.POINTER(ctypes.c_int64),      # shape_out[cap_ndim]
            ctypes.c_int,                        # cap_ndim
            ctypes.POINTER(ctypes.c_int),        # ndim_out
            ctypes.POINTER(ctypes.c_int),        # dtype_tag_out
            ctypes.c_char_p, ctypes.c_int,       # err_out, err_cap
        ]
        lib.lrx_phdf5_dataset_geometry.restype = ctypes.c_int
    if hasattr(lib, "lrx_phdf5_read_whole"):
        lib.lrx_phdf5_read_whole.argtypes = [
            ctypes.c_int64,                      # ctx_handle
            ctypes.c_char_p,                     # ds_name
            ctypes.c_int,                        # dtype_tag
            ctypes.c_void_p,                     # out
            ctypes.c_int64,                      # out_nelem
            ctypes.c_char_p, ctypes.c_int,       # err_out, err_cap
        ]
        lib.lrx_phdf5_read_whole.restype = ctypes.c_int


def _declare_slate(lib: ctypes.CDLL) -> None:
    """SLATE context lifecycle — exported by BOTH platform libraries
    (cpp/slate/context.cc is pure MPI and compiled into each).  Absent
    from a build made without SLATE (e.g. the Frontera eigh-only .so)."""

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
    lib.lrx_slate_context_create.restype  = ctypes.c_int64
    lib.lrx_slate_subrow_context_create.argtypes = [
        ctypes.c_int,           # rank (full-world)
        ctypes.c_int,           # world_size
        ctypes.c_int,           # Px
        ctypes.c_int,           # Py
        ctypes.c_char_p,        # err_buf
        ctypes.c_int,           # err_buf_len
    ]
    lib.lrx_slate_subrow_context_create.restype  = ctypes.c_int64
    lib.lrx_slate_context_destroy.argtypes = [ctypes.c_int64]
    lib.lrx_slate_context_destroy.restype  = None
    lib.lrx_slate_init_mpi.argtypes = []
    lib.lrx_slate_init_mpi.restype  = None


def _register_ffi_targets(lib: ctypes.CDLL, platform: str) -> None:
    for target_name, cpp_symbol in _PLATFORMS[platform]["targets"].items():
        # A partial build (e.g. the Frontera eigh-only .so with
        # -DLORRAX_FFI_HAVE_PHDF5=0 and no SLATE) omits some handler
        # symbols.  Skip the ones this .so doesn't export instead of
        # failing the whole load; the corresponding Python wrappers raise
        # a clear AttributeError only if actually called.
        if not hasattr(lib, cpp_symbol):
            continue
        fn = getattr(lib, cpp_symbol)
        try:
            jax.ffi.register_ffi_target(
                target_name,
                jax.ffi.pycapsule(fn),
                platform=platform,
            )
        except Exception as exc:
            if "already registered" not in str(exc).lower():
                raise


def probe_target(target_name: str, platform: str) -> tuple[bool, str]:
    """``(usable, reason)`` for XLA target ``target_name`` on ``platform``.

    The reason DISTINGUISHES the three ways a target can be unusable,
    because they have three different fixes:

    ``unknown target``
        Not a target of this platform's library at all — a typo or a
        wrong-platform request.
    ``library could not be loaded``
        The ``.so`` is missing, or loading it failed: one of the
        libraries it in turn needs could not be found, a glibc/GLIBCXX
        mismatch, a wrong ``LD_LIBRARY_PATH``.  **The handler may well
        be compiled; nothing about the build is wrong.**
    ``loaded but does not export <symbol>``
        The genuine partial-build case — this, and ONLY this, is a
        "rebuild the library" problem.

    Why the distinction is worth a function: conflating the middle case
    with the last one produces an error that tells you to rebuild a
    library that is perfectly fine, while the actual cause (one missing
    entry in ``LD_LIBRARY_PATH``) goes unmentioned.  On Frontera the
    unified host lib needs MKL **and** the Intel compiler runtime
    (``libimf``/``libsvml``/``libintlc``/``libirng``) **and** the phdf5
    ``libhdf5.so.310`` **and** ``libfabric`` — only some of which are
    covered by the library's built-in search path.  Getting that wrong
    silently downgraded an N×1 mesh to
    ``native`` with a "not compiled" diagnosis (found by wk_P G4,
    2026-07-25).

    Never raises.
    """
    spec = _PLATFORMS.get(platform)
    if spec is None:
        return False, (f"unknown FFI platform {platform!r} "
                       f"(known: {sorted(_PLATFORMS)})")
    sym = spec["targets"].get(target_name)
    if sym is None:
        return False, (f"unknown target: {target_name!r} is not a target of "
                       f"the {platform} FFI library (known: "
                       f"{', '.join(sorted(spec['targets']))})")
    try:
        lib = get_lib(platform)
    except Exception as exc:
        return False, (
            f"the {platform} FFI library could not be loaded: "
            f"{type(exc).__name__}: {exc}  "
            f"NOTE: this says nothing about whether {target_name} is "
            f"compiled — fix the library path/dependencies first "
            f"(LORRAX_FFI_SO / LORRAX_FFI_HOST_SO select the .so; "
            f"LD_LIBRARY_PATH must cover every library that .so needs — "
            f"on Frontera that is the vendor BLAS/ScaLAPACK (MKL) + the "
            f"Intel compiler runtime + parallel HDF5 + libfabric + "
            f"SLATE's lib64; run `ldd <so>` to see which are missing).")
    if not hasattr(lib, sym):
        return False, (
            f"loaded {_loaded_path(platform)} but it does not export {sym} — "
            f"this library was built WITHOUT the {target_name} handler.  "
            f"Rebuild with {spec['build_hint']}.")
    return True, "available"


def has_target(target_name: str, platform: str) -> bool:
    """True when ``platform``'s FFI library is loadable AND exports the C++
    handler for XLA target ``target_name`` (per that platform's symbol
    table).  Never raises.

    This is THE capability probe for backend auto-pick logic (e.g.
    ``WfnLoader._auto_pick_backend``), where a bool is all a fallback
    decision needs.  Anything that REPORTS a refusal to a human should
    use :func:`probe_target` instead and quote its reason — a bare False
    cannot distinguish "not built" from "LD_LIBRARY_PATH is wrong".

    Callers must not reach into the private ``_CUDA_TARGET_SYMBOLS`` /
    ``_HOST_TARGET_SYMBOLS`` tables: the target-name → C++-symbol mapping
    is per-platform and owned here, so adding a new FFI target platform
    only touches ``_PLATFORMS``.
    """
    return probe_target(target_name, platform)[0]


def has_phdf5_read(platform: str) -> bool:
    """True when ``platform``'s FFI library can serve the collective
    kchunk-union WFN read — the probe ``WfnLoader._auto_pick_backend``
    uses to pick the ``phdf5`` backend on that platform."""
    return has_target("lorrax_phdf5_read_kchunk_union", platform)


def has_phdf5_write(platform: str) -> bool:
    """True when ``platform``'s FFI library can serve the collective
    sharded-slab WRITE — the probe ``gw.gw_config`` uses to route
    ``file_io.slab_io.assert_available`` refuses on.

    False on a host lib built before workstream AE (read-only), which is
    why the router demotes rather than failing at the first ζ write; use
    :func:`probe_target` when reporting the reason to a human."""
    return has_target("lorrax_phdf5_write", platform)


# ---------------------------------------------------------------------------
# THE TWO PLATFORM LIBRARIES SHARE THEIR SLATE.  Open CUDA first — IN A
# PROCESS THAT CAN USE CUDA, and in no other.
# ---------------------------------------------------------------------------
# The target tables above say the two .so's "can co-exist under RTLD_GLOBAL"
# because only their C++ SYMBOLS differ.  That is true of the symbols and
# FALSE of the dependency closure: both carry ``NEEDED libslate.so.2`` and
# ``NEEDED libblaspp.so.2`` and resolve those SONAMEs out of DIFFERENT builds
# (host DT_RPATH -> a gpu_backend=none blaspp; CUDA DT_RUNPATH -> the CUDA
# one).  ld.so keys a loaded object by SONAME, so the FIRST of the two to be
# dlopened decides which ``libblaspp.so.2`` the OTHER one calls for the rest
# of the process, and the CPU build's ``blas::get_device_count()`` is a
# compiled-in 0.
#
# MEASURED, Perlmutter 2026-08-07 (``dladdr`` at a failing cell, ONE visible
# GPU in both legs).  THIS module lost the race, at COLLECTION time:
# ``tests/test_fft_flat_k_numerics.py`` calls ``probe_target(FLAT_K_TARGET,
# "cpu")`` at MODULE SCOPE, which opened the host lib before any CUDA cell
# ran, and every CUDA SLATE handler then refused:
#
#   FAILED_PRECONDITION: slate.potrf: blas::get_device_count()=0 but JAX
#   one-process-per-GPU model requires exactly 1.
#
# Eight cells.  Not a harness artefact: any GPU run whose first FFI touch is
# a host target (a phdf5 host read, an mklfft probe) gets the same
# device-less SLATE, and the message names a device count nobody set.
#
# ONE SAFE ORDER EXISTS -- AMONG PROCESSES THAT OPEN BOTH.  The host SLATE
# handlers run ``slate::Target::HostTask`` and never consult
# ``get_device_count``, so a CUDA blaspp in charge costs the host path
# nothing; the reverse costs the CUDA path everything.
# ``distrib_la.loader`` carries the same rule for the libraries it opens --
# the two loaders share the FILES, so both have to.
#
# AND THE CHEAPER ORDER IS TO OPEN ONLY ONE.  MEASURED, Perlmutter
# 2026-08-07 (KNOWN_FAILURES B1): a pre-open with no CUDA arm to protect
# costs a CPU-platform process its host phdf5 path outright.
# ``tests/test_file_io.py`` on a CPU-platform leg (JAX_PLATFORMS=cpu, four
# emulated host devices, same pins throughout):
#
#   96a6399 branch base                    42 passed / 1 skipped
#   b3f3675 the commit before this rule     42 passed / 1 skipped
#   32e61fe this rule, unconditional        3 failed, Fatal Python error: Aborted
#
# and where it refused instead of aborting it printed
#
#   phdf5 read: logical slab out of bounds
#     extent=[2,4,1,6]
#     offset_base=[0,0,0,4596944070643295330]
#     valid_shape=[3,6,6,4609783128842618077]
#
# -- IEEE-754 float64 bit patterns (~0.19, ~1.87) decoded as int64: one
# library's handler reading the other's argument layout.  See INTERPOSITION
# below; that cross-wiring is a defect of its own and this rule must not
# hand it any new processes.
#
# SO THE PRE-OPEN IS TWO-ARMED, on the one question that decides whether
# there is a CUDA handler in this process for the order to protect:
#
#   CUDA-capable process  ->  CUDA first, then host.  SLATE survives.
#   CPU-platform process  ->  host only.  The CUDA library is NEVER dlopened,
#                             so there is no second libslate/libblaspp, no
#                             second phdf5, and nothing to cross-wire.
#
# FALSIFIED at HEAD before the fix, on the same leg: point ``LORRAX_FFI_SO``
# at a path that cannot be dlopened, so the best-effort ``get_lib("CUDA")``
# raises and the process stays host-only -- 42 passed / 1 skipped, byte for
# byte the base result, against a 300 s wall with the CUDA .so pinned.  That
# arm is what this gate makes deliberate instead of accidental.
#
# Best effort on top: no CUDA library built, or one that cannot load on this
# node, leaves the process untouched and is tried once.
#
# ---------------------------------------------------------------------------
# INTERPOSITION -- THE DEFECT UNDERNEATH.  FIXED 2026-08-08; this block is
# the record of what it was and of what the load-order rule above does NOT
# cover.
# ---------------------------------------------------------------------------
# ``libslate``/``libblaspp`` were not the only collision between the two
# files.  MEASURED 2026-08-07, ``nm -D --defined-only`` on the two pinned
# builds (deployed device lib; build_host_h200, md5 4c4422b8...): 259 symbol
# names DEFINED BY BOTH, **25 of them LORRAX's own** -- the nine C-linkage
# ``lrx_phdf5_*`` / ``lrx_slate_*`` entry points and sixteen mangled
# ``lorrax_ffi::phdf5::*`` (``open_ctx``, ``close_ctx``, ``ensure_dataset``,
# ``open_dataset_ro``, ``ensure_pinned``, ``ensure_read_buf``,
# ``ensure_mpi_initialized``, ``env_flag``, ``~PhdfCtx``, the ``dt::`` HDF5
# type singletons and their guards).  Both are dlopened RTLD_GLOBAL, so once
# both were open the FIRST one answered those names for BOTH -- including for
# the other library's own internal calls.
#
# And they were not the same functions.  ``cpp/phdf5/ctx.h`` compiles
# ``PhdfCtx`` with the CUDA stream / event / pinned-buffer members under
# ``#ifndef LORRAX_FFI_NO_CUDA``; the host build defines that macro and drops
# them.  One C++ type name, two struct layouts, both exporting
# ``open_ctx(...) -> PhdfCtx*`` -- a cross-.so ODR violation.  A handler from
# one build handed a ``PhdfCtx*`` minted by the other read its fields at the
# wrong offsets, which is what the float64-as-int64 ``offset_base`` above and
# the ``phdf5 read: ctx_handle is null`` in the xdist arm both are.
#
# FIXED, IN C++, THREE WAYS (branch fix/ffi-odr-2026-08-08):
#   * ``src/ffi/cpp/exports_{cuda,host}.map`` -- linker version scripts that
#     make every ``lorrax_ffi``-namespaced definition LOCAL to the library
#     that compiled it.  Third-party vague-linkage symbols are deliberately
#     left global; localising those was measured to segfault the SLATE host
#     handlers, and the map carries that five-arm table.
#   * ``cpp/common/c_abi.h`` -- the nine ``lrx_*`` entry points cannot be
#     hidden (this module dlsyms them), so the HOST leg's carry a ``_host``
#     suffix.  ``_bind_c_abi`` above binds them under the plain Python name.
#   * ``cpp/phdf5/ctx.h`` -- ``PhdfCtx``'s struct TAG is now per-leg
#     (``PhdfCtxCudaV1`` / ``PhdfCtxHostV1``), so the cross-layout aliasing
#     is unconstructible even without the version scripts.
#
# MEASURED AFTER, same two commands: 234 shared names, **0 of them LORRAX's
# own and 0 with C linkage**.  And the mixed state is now survivable --
# ``tests/test_file_io.py`` on a CPU-platform leg with the capability gate
# below DISABLED (i.e. 32e61fe's unconditional pre-open, the configuration
# that aborted):
#
#     pre-fix pair, gate off   Fatal Python error: Aborted, no junitxml
#     rebuilt pair, gate off   46 passed / 1 skipped, 0 abort signatures
#
# ``services/distrib_la/tests/test_so_acceptance.py`` check 6 is the ratchet.
#
# THE GATE ABOVE STAYS ANYWAY, and this is why: it was written for the
# ``libslate``/``libblaspp`` SONAME race, and that is a DIFFERENT defect --
# about which BUILD of a third-party library answers, not about which of our
# own definitions does.  ~234 third-party names are still shared (they have
# to be: they are ODR-correct weak COMDAT copies the C++ ABI merges on
# purpose), and there is still ONE ``libblaspp.so.2`` per process whose
# ``blas::get_device_count()`` is a compiled-in 0 in the host build.  Nothing
# in the ODR fix touches that.  Retiring the pre-open would re-open the eight
# ``FAILED_PRECONDITION: slate.potrf: blas::get_device_count()=0`` cells.
# ``test_so_acceptance.py``'s check 5 is what will say when it CAN go.
_CUDA_FIRST_TRIED = False


def _nvidia_device_visible() -> bool:
    """A GPU device node this process can see.

    Split out from :func:`_process_can_use_cuda` so its cells can construct
    both answers without monkeypatching ``glob``/``os.path`` for the whole
    session -- the hardware half of the question is the one a pure test
    cannot arrange for real.
    """
    return (any(Path("/dev").glob("nvidia[0-9]*"))
            or Path("/dev/nvidiactl").exists())


def _process_can_use_cuda() -> bool:
    """Can THIS process run CUDA work at all?  Environment only.

    TRUTHFUL AT LOAD TIME is the whole requirement, and it rules out the
    obvious answer: ``jax.default_backend()`` INITIALIZES the XLA backend
    (the reason :func:`get_lib` takes an explicit platform at all), so
    asking it from inside a loader call would make the loader decide the
    process's backend.  Both signals below are ones JAX itself acts on and
    both are readable before any backend exists:

      * ``JAX_PLATFORMS`` -- :func:`platform_from_env` resolves it exactly
        as JAX picks its default backend (first entry wins).  ``cpu`` there
        forbids a GPU backend outright, so no CUDA handler in this process
        can ever run and there is nothing for the load order to protect.
      * A VISIBLE NVIDIA DEVICE.  ``CUDA_VISIBLE_DEVICES=""`` is this
        tree's spelling of "no GPU, deliberately", and JAX's own
        ``backends()`` skips cuda when no device node is visible
        (``xla_bridge.py``: ``if platform == "cuda" and not
        has_visible_nvidia_gpu(): continue``).  Same two signals in the
        same order as ``runtime._gpu_is_present`` -- the call this tree
        already makes, made the same way, so the two cannot drift into
        disagreeing about what platform a process is.

    Deliberately NOT a probe of the library.  "Is the CUDA .so loadable
    here" is a different question and the best-effort ``try`` in
    :func:`_open_cuda_before_host` is what answers it.
    """
    return _native.process_can_use_cuda(device_visible=_nvidia_device_visible)


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
    """The platforms whose library this process has opened, in ORDER."""
    return list(_LIBS)


def _assert_one_mapped_mpi_runtime() -> None:
    """Refuse a process in which more than one MPI runtime is already mapped.

    The link-time ``gate_one_mpi.sh`` proves one artifact's dependency
    closure.  It cannot see a foreign MPI loaded earlier by an inherited
    Python overlay or preload.  Check the live address space immediately
    after the FFI ``dlopen`` and before any native collective starts.
    Filename aliases onto one real object are deduplicated by ``realpath``.
    """
    _native.assert_one_mapped_mpi_runtime(unusable_cls=FfiLibraryUnusable)


def get_lib(platform: Optional[str] = None) -> ctypes.CDLL:
    """Return the loaded FFI library for ``platform``; idempotent.

    ``platform`` is "CUDA" or "cpu"; ``None`` follows the JAX default
    backend, so wrapper call sites stay platform-agnostic.

    In a CUDA-CAPABLE process, opening the ``cpu`` library opens the
    ``CUDA`` one first — see the SONAME note above; it is a correctness
    requirement, not a warm-up.  In a CPU-platform process it does NOT:
    there is no CUDA handler to protect and the mixed load costs that
    process its host phdf5 path.
    """
    if platform is None:
        platform = _default_platform()
    if platform not in _PLATFORMS:
        raise ValueError(
            f"lorrax_ffi: unknown platform {platform!r} "
            f"(known: {sorted(_PLATFORMS)}).")
    lib = _LIBS.get(platform)
    if lib is not None:
        return lib
    if platform == "cpu":
        _open_cuda_before_host()
        lib = _LIBS.get(platform)
        if lib is not None:                    # re-entrancy, not reachable today
            return lib
    path = _locate_so(platform)
    lib, actual_origin = _native.open_and_attest(
        path, platform=platform, expected_abi=LORRAX_FFI_ABI_VERSION,
        abi_symbols=_ABI_SYMBOLS,
        build_hint=str(_PLATFORMS[platform]["build_hint"]),
        strict_unstamped=os.environ.get("LORRAX_FFI_ABI_STRICT", "") == "1",
        unusable_cls=FfiLibraryUnusable, mismatch_cls=FfiAbiMismatch)
    _set_argtypes(lib, platform)
    _register_ffi_targets(lib, platform)
    _LIBS[platform] = lib
    _LIB_PATHS[platform] = str(actual_origin)
    return lib


def _loaded_path(platform: str) -> str:
    """The .so path loaded for ``platform`` (for error messages)."""
    return _LIB_PATHS.get(platform, f"<{platform} library>")


def library_provenance(platform: str) -> str:
    """WHICH BUILD of the .so this process loaded, as one printable line.

    A path is not an identity.  Stage dirs are hand-named
    (``build_host_ONE``, ``build_host_PADFIX``, ...), a harness echoes the
    path it *intends* from a shell variable that a later line may override,
    and nothing in a log has ever recorded the bytes that were actually
    dlopened.  That cost real time on 2026-08-02: two 32-node legs were
    debugged against a lib whose revision nobody had checked, and the
    certified ``build_host_ONE`` turned out to predate the tree it was being
    compared with (453 exported symbols vs 475).

    So: report the ``PROVENANCE`` file ``build_ffi_host.sh`` stamps beside
    the .so, and when there is none, fall back to facts measured from the
    file itself (size + sha256 prefix) so an unstamped legacy lib is still
    identified rather than merely located.  Never raises — an unmeasurable
    provenance is a string, not an exception.
    """
    path = _LIB_PATHS.get(platform)
    if not path:
        return f"{platform}: no library loaded"
    return _native.describe_library(path)


def loaded_lib_path(platform: str) -> Optional[str]:
    """The .so path loaded for ``platform``, or None if none is loaded yet.

    Pure lookup — never loads. Exists for callers that need the file path of
    an already-probed library for the subprocess MPI-bootstrap probe without
    repeating the load side effects.
    """
    return _LIB_PATHS.get(platform)


# ---------------------------------------------------------------------------
# Thin Pythonic helpers (so call sites don't keep re-writing ctypes plumbing)
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


def create_cusolvermp_context(
    rank: int, world_size: int,
    nccl_unique_id_addr: int, nccl_unique_id_nbytes: int,
    p: int, q: int, grid_layout_col_major: bool = True,
) -> int:
    lib = get_lib("CUDA")
    ctx_out = ctypes.c_int64(0)
    err = ctypes.create_string_buffer(_ERR_CAP)
    rc = lib.lrx_create_cusolvermp_context(
        int(rank), int(world_size),
        int(nccl_unique_id_addr), int(nccl_unique_id_nbytes),
        int(p), int(q),
        1 if grid_layout_col_major else 0,
        ctypes.byref(ctx_out),
        err, _ERR_CAP,
    )
    _check_err(rc, err)
    return int(ctx_out.value)


def destroy_cusolvermp_context(ctx_handle: int) -> None:
    get_lib("CUDA").lrx_destroy_cusolvermp_context(int(ctx_handle))


def smoke_allreduce_sum(ctx_handle: int, device_ptr: int, nelems: int) -> int:
    return int(get_lib("CUDA").lrx_smoke_allreduce_sum(
        int(ctx_handle), int(device_ptr), int(nelems)))


def version_info() -> dict:
    lib = get_lib("CUDA")
    rt   = ctypes.c_int(0)
    drv  = ctypes.c_int(0)
    nccl = ctypes.c_int(0)
    lib.lrx_version_info(ctypes.byref(rt), ctypes.byref(drv), ctypes.byref(nccl))
    return {"cuda_runtime": rt.value, "cuda_driver": drv.value, "nccl": nccl.value}


# ---- phdf5 ----------------------------------------------------------------
# ``platform`` selects the library that owns the collective context: "CUDA"
# for the GPU MPI-IO path, "cpu" for the host read path (liblorrax_ffi_host.so
# built with the phdf5 subpackage).  ``None`` follows the JAX default backend,
# so the WFN loader's phdf5 backend works on both GPU and CPU nodes without a
# platform argument.  A PhdfCtx handle is a plain int64 address, so the
# lifecycle calls just need to reach the .so that created it.
def phdf5_open(path: str, p: int, q: int, rank: int, world_size: int,
               mode_flag: int, platform: Optional[str] = None) -> int:
    """Collective open/create of a parallel-HDF5 file.

    mode_flag: 0 = truncate ('w'), 1 = append-or-create ('a'), 2 = read-only ('r').
    """
    lib = get_lib(platform)
    ctx_out = ctypes.c_int64(0)
    err = ctypes.create_string_buffer(_ERR_CAP)
    rc = lib.lrx_phdf5_open(
        path.encode("utf-8"),
        int(p), int(q), int(rank), int(world_size), int(mode_flag),
        ctypes.byref(ctx_out),
        err, _ERR_CAP,
    )
    _check_err(rc, err)
    return int(ctx_out.value)


def phdf5_close(ctx_handle: int, platform: Optional[str] = None) -> None:
    get_lib(platform).lrx_phdf5_close(int(ctx_handle))


def phdf5_init_mpi(platform: Optional[str] = None) -> None:
    """Eagerly init MPI_THREAD_MULTIPLE so the first ``open_file`` in the
    hot path doesn't pay the ~400 ms MPI_Init_thread cost.  Collective
    across all ranks; idempotent after first call.
    """
    get_lib(platform).lrx_phdf5_init_mpi()


# Mapping from numpy/jax dtype to the integer tag matching xla::ffi::DataType.
_DTYPE_TAG = {
    "float32":   1,
    "float64":   2,
    "int32":     3,
    "int64":     4,
    "complex64": 5,
    "complex128": 6,
}


def phdf5_ensure_dataset(ctx_handle: int, ds_name: str,
                         shape, dtype_name: str,
                         platform: Optional[str] = None) -> int:
    """Collective create/open of an N-D HDF5 dataset.  Returns hid_t as int64.

    ``shape`` is any sequence of non-negative ints (tuple, list, numpy array).
    """
    if dtype_name not in _DTYPE_TAG:
        raise ValueError(f"phdf5_ensure_dataset: unsupported dtype {dtype_name}")
    shape = tuple(int(s) for s in shape)
    if not shape:
        raise ValueError("phdf5_ensure_dataset: shape must be non-empty")
    lib = get_lib(platform)
    ShapeArr = ctypes.c_int64 * len(shape)
    shape_buf = ShapeArr(*shape)
    ds_id_out = ctypes.c_int64(0)
    err = ctypes.create_string_buffer(_ERR_CAP)
    rc = lib.lrx_phdf5_ensure_dataset(
        int(ctx_handle),
        ds_name.encode("utf-8"),
        shape_buf, int(len(shape)),
        _DTYPE_TAG[dtype_name],
        ctypes.byref(ds_id_out),
        err, _ERR_CAP,
    )
    _check_err(rc, err)
    return int(ds_id_out.value)


def phdf5_open_dataset_ro(ctx_handle: int, ds_name: str,
                          platform: Optional[str] = None) -> int:
    """Collective H5Dopen of an existing dataset.  Returns hid_t as int64."""
    lib = get_lib(platform)
    ds_id_out = ctypes.c_int64(0)
    err = ctypes.create_string_buffer(_ERR_CAP)
    rc = lib.lrx_phdf5_open_dataset_ro(
        int(ctx_handle),
        ds_name.encode("utf-8"),
        ctypes.byref(ds_id_out),
        err, _ERR_CAP,
    )
    _check_err(rc, err)
    return int(ds_id_out.value)


#: dtype tag -> numpy dtype name.  The inverse of ``_DTYPE_TAG``; kept
#: beside it so the two cannot drift.
_TAG_DTYPE = {tag: name for name, tag in _DTYPE_TAG.items()}


def has_phdf5_metadata_api(platform: Optional[str] = None) -> bool:
    """Does the loaded library export the metadata half of the transport?

    ``lrx_phdf5_dataset_geometry`` / ``lrx_phdf5_read_whole`` landed
    2026-08-22.  A library built before then exports neither, and the
    callers (``file_io._slab_io_ffi``) need to know WHICH route they are on
    so they can say so rather than silently taking the serial-h5py one.
    Both names are checked: a half-built library is not a capability.
    """
    try:
        lib = get_lib(platform)
    except Exception:
        return False
    return (hasattr(lib, "lrx_phdf5_dataset_geometry")
            and hasattr(lib, "lrx_phdf5_read_whole"))


#: The rank ceiling ``phdf5_dataset_geometry`` will report.  Nothing in this
#: tree stores above 6-D (the WFN coefficient dataset is 4-D, Sigma cubes are
#: 5-D); a higher-rank dataset REFUSES in C++ rather than truncating.
_GEOM_MAX_NDIM = 8


def phdf5_dataset_geometry(ctx_handle: int, ds_name: str,
                           platform: Optional[str] = None
                           ) -> tuple[tuple, str]:
    """``(shape, dtype_name)`` of an existing dataset, through the FFI.

    The replacement for opening the same path a second time with serial
    h5py just to learn a geometry — that read is legal only while BOTH
    stacks are read-only and is refused by ``file_io.hdf5_owner`` whenever
    the live FFI handle can write, which is the measured
    ``--parallel-transport-out`` failure.  Here the answer comes from the
    library that already holds the file open.

    Collective in the same sense as :func:`phdf5_open_dataset_ro` (it
    routes through that same cached H5Dopen): every rank calls it with the
    same name.  A SCALAR dataset returns ``shape == ()``.
    """
    lib = get_lib(platform)
    if not hasattr(lib, "lrx_phdf5_dataset_geometry"):
        raise RuntimeError(
            "phdf5_dataset_geometry: the loaded FFI library does not export "
            "lrx_phdf5_dataset_geometry.  It predates 2026-08-22; rebuild it "
            "(src/ffi/cpp/build.sh for the CUDA leg, "
            "config/perlmutter/build_ffi_host.sh for the host leg).")
    ShapeArr = ctypes.c_int64 * _GEOM_MAX_NDIM
    shape_buf = ShapeArr()
    ndim_out = ctypes.c_int(0)
    tag_out = ctypes.c_int(0)
    err = ctypes.create_string_buffer(_ERR_CAP)
    rc = lib.lrx_phdf5_dataset_geometry(
        int(ctx_handle),
        ds_name.encode("utf-8"),
        shape_buf, int(_GEOM_MAX_NDIM),
        ctypes.byref(ndim_out), ctypes.byref(tag_out),
        err, _ERR_CAP,
    )
    _check_err(rc, err)
    ndim = int(ndim_out.value)
    tag = int(tag_out.value)
    if tag not in _TAG_DTYPE:
        raise RuntimeError(
            f"phdf5_dataset_geometry({ds_name!r}): the library returned "
            f"dtype tag {tag}, which this build of the Python side does not "
            f"know ({sorted(_TAG_DTYPE)}).  The two sides of the tag table "
            f"have drifted — rebuild the .so from this tree.")
    return tuple(int(shape_buf[i]) for i in range(ndim)), _TAG_DTYPE[tag]


def phdf5_read_whole(ctx_handle: int, ds_name: str, *, shape, dtype_name: str,
                     platform: Optional[str] = None):
    """Read a WHOLE small dataset into a fresh host ``np.ndarray``.

    Serves the case the sharded read handler structurally cannot: a
    rank-0 (scalar) dataset has no hyperslab to select, so
    ``read_slab`` refuses it before it reaches HDF5 at all
    (``_normalize_slab_request``: "slab shape must be non-empty").  Every
    scalar stamp ``SlabIO.write_attr`` publishes is such a dataset.

    FOR SCALARS AND SMALL REPLICATED VECTORS ONLY.  Every rank gets the
    whole thing, so the payload must be O(1) in the design envelope —
    schema stamps, band counts, a k-grid triple.  Bulk data is a
    hyperslab and belongs in ``SlabIO.read_slab``; the C++ side checks the
    element count against the dataset's own so a mis-sized buffer refuses
    instead of overrunning, but it cannot check that you meant it.
    """
    import numpy as _np
    lib = get_lib(platform)
    if not hasattr(lib, "lrx_phdf5_read_whole"):
        raise RuntimeError(
            "phdf5_read_whole: the loaded FFI library does not export "
            "lrx_phdf5_read_whole.  It predates 2026-08-22; rebuild it "
            "(src/ffi/cpp/build.sh for the CUDA leg, "
            "config/perlmutter/build_ffi_host.sh for the host leg).")
    if dtype_name not in _DTYPE_TAG:
        raise ValueError(f"phdf5_read_whole: unsupported dtype {dtype_name}")
    out = _np.zeros(tuple(int(s) for s in shape), dtype=_np.dtype(dtype_name))
    err = ctypes.create_string_buffer(_ERR_CAP)
    rc = lib.lrx_phdf5_read_whole(
        int(ctx_handle),
        ds_name.encode("utf-8"),
        _DTYPE_TAG[dtype_name],
        out.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int64(int(out.size)),
        err, _ERR_CAP,
    )
    _check_err(rc, err)
    return out


# ---- slate ----------------------------------------------------------------
# The slate lifecycle exists in BOTH platform libraries (context.cc is pure
# MPI); ``platform`` selects which one serves the call.  A SlateCtx handle is
# platform-agnostic — pass the platform of the mesh being operated on so the
# call never forces the OTHER platform's library to load (e.g. a CPU-mesh op
# on a machine whose CUDA library is absent).
def create_slate_context(rank: int, world_size: int, p: int, q: int,
                         platform: Optional[str] = None) -> int:
    """Collective create of a SLATE context; returns opaque int64 handle.

    Inits MPI_THREAD_MULTIPLE if not already inited, then dups
    MPI_COMM_WORLD for SLATE's exclusive use.  Raises RuntimeError on
    failure with a message from the .so's error buffer.
    """
    lib = get_lib(platform)
    err = ctypes.create_string_buffer(_ERR_CAP)
    h = lib.lrx_slate_context_create(
        int(rank), int(world_size), int(p), int(q), err, _ERR_CAP)
    if int(h) == 0:
        msg = err.value.decode("utf-8", errors="replace")
        raise RuntimeError(f"lorrax_ffi slate.context_create failed: {msg}")
    return int(h)


def create_slate_subrow_context(rank: int, world_size: int,
                                Px: int, Py: int,
                                platform: Optional[str] = None) -> int:
    """Collective create of a SLATE sub-row context; returns int64 handle.

    The sub-comm is MPI_COMM_WORLD split by x-coordinate (color=x_rank,
    key=y_rank), producing one comm of size ``Py`` per X-row.  Intended
    for batched ops where each X-row independently processes a slice of
    a 3-D (Nbatch, N, N) input distributed as P('x', None, 'y').
    """
    lib = get_lib(platform)
    err = ctypes.create_string_buffer(_ERR_CAP)
    h = lib.lrx_slate_subrow_context_create(
        int(rank), int(world_size), int(Px), int(Py), err, _ERR_CAP)
    if int(h) == 0:
        msg = err.value.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"lorrax_ffi slate.subrow_context_create failed: {msg}")
    return int(h)


def destroy_slate_context(ctx_handle: int,
                          platform: Optional[str] = None) -> None:
    get_lib(platform).lrx_slate_context_destroy(int(ctx_handle))


def slate_init_mpi(platform: Optional[str] = None) -> None:
    """Eagerly init MPI_THREAD_MULTIPLE from outside SLATE's hot path.
    Idempotent; no-op if MPI is already initialized.
    """
    get_lib(platform).lrx_slate_init_mpi()

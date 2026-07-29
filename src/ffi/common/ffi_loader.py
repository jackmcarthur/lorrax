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

Dynamic-linking environment (audited 2026-07-27, workstream AW)
---------------------------------------------------------------
The dlopen here resolves the library's ``DT_NEEDED`` through its
RUNPATH plus ``LD_LIBRARY_PATH``.  On Frontera the unified host lib's
RUNPATH covers Intel MPI ``lib/release``, SLATE's ``lib64`` and MKL —
what ``LD_LIBRARY_PATH`` MUST add is only what has no RUNPATH
coverage: the parallel-HDF5 ``lib`` (``libhdf5.so.310``), Intel MPI's
``libfabric/lib``, and (for the overlay h5py, not for this .so) the
Intel compiler runtime.  Two ordering facts, so nobody re-derives
them:

* **"SLATE first" is convention, not requirement.**  The SLATE dir
  holds only ``libslate``/``libblaspp``/``liblapackpp`` (no libmpi, no
  libhdf5, no MKL), so its position can shadow nothing; libslate's own
  RUNPATH resolves its transitive blaspp/lapackpp.  Ordering among the
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
stack must run the THREAD_MULTIPLE-patched MPIwrapper as
``MPITRAMPOLINE_LIB`` (+ ``LORRAX_MPI_FINALIZE_FIX=skip_atexit`` in
the overlay sitecustomize) or the FFI's concurrent MPI threads race —
measured ~29% multi-node failure rate (AS.4b).  LORRAX deliberately
does NOT default ``MPITRAMPOLINE_LIB`` itself: it is a harness-level
machine fact (a build artifact path), the unpatched build is a
measured hazard, and ``phdf5/cpp/context.cc`` announces the hazardous
level at open time.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import jax
import jax.ffi

__all__ = ["get_lib", "has_target", "probe_target", "has_phdf5_read",
           "has_phdf5_write"]

_LIBS: Dict[str, ctypes.CDLL] = {}
#: platform -> the .so path actually dlopen'd (for diagnostics).
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
    # cuFFT strided flat-k batched-FFT handlers (cufft/cpp) — the CUDA
    # platform mirror of the mklfft host handlers below.  The target STRINGS
    # deliberately keep the host table's "mklfft" names (they were coined by
    # the CPU prototype): common.fft_helpers issues ONE platform-agnostic
    # ffi_call per site and the lowering platform resolves the handler —
    # exactly the phdf5 same-target/different-symbol split.
    "lorrax_mklfft_flat_k":         "CufftFlatKCudaFfi",
    "lorrax_mklfft_gw_conv":        "CufftGwConvCudaFfi",
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

# Host variants of the slate targets (src/ffi/slate/cpp/host_ffi.cc) —
# same target names as the CUDA table, registered under platform="cpu" —
# plus the host-only ScaLAPACK targets (src/ffi/scalapack/, MKL/LibSci),
# plus the phdf5 read AND write handlers (src/ffi/phdf5/cpp/{read,write}_ffi.cc
# compiled with -DLORRAX_FFI_NO_CUDA).  The phdf5 target STRINGS are identical
# to the CUDA table so the ffi.phdf5.{read,write} ffi_call sites resolve by
# lowering platform; only the C++ SYMBOL names differ (Phdf*HostFfi vs
# Phdf*Ffi) so the two platform .so's can co-exist under RTLD_GLOBAL.
#
# ``lorrax_phdf5_write`` is what makes ``SlabIOBackend.PHDF5_FFI`` reachable on
# the CPU backend (workstream AE) — a host lib built before that port exports
# the three read symbols only, and ``has_phdf5_write('cpu')`` is False, which
# is exactly how the gw_config router demotes gracefully to PHDF5_HOST /
# H5PY_ALLGATHER instead of dying at the first ζ write.
_HOST_TARGET_SYMBOLS = {
    "lorrax_slate_eigh":              "SlateEighHostFfi",
    "lorrax_slate_potrf":             "SlatePotrfHostFfi",
    "lorrax_slate_trsm":              "SlateTrsmHostFfi",
    "lorrax_slate_batched_potrf":     "SlateBatchedPotrfHostFfi",
    "lorrax_slate_batched_trsm":      "SlateBatchedTrsmHostFfi",
    "lorrax_scalapack_batched_solve_lu": "ScalapackBatchedSolveLuHostFfi",
    "lorrax_scalapack_eigh":          "ScalapackEighHostFfi",
    # MKL FFT (DFTI API) flat-k batched-FFT handlers (mklfft/cpp) — the
    # LORRAX_FFT_FFI backend of common.fft_helpers (FFT-FFI prototype).
    "lorrax_mklfft_flat_k":           "MklFftFlatKHostFfi",
    "lorrax_mklfft_gw_conv":          "MklFftGwConvHostFfi",
    # MKL batched-GEMM handler (mklblas/cpp) — the LORRAX_BANDS_GEMM_FFI
    # body of common.contract_bands (contract_bands_block_reshard).
    "lorrax_mklblas_gemm_batch":      "MklBlasGemmBatchHostFfi",
    "lorrax_phdf5_read":              "PhdfReadHostFfi",
    "lorrax_phdf5_read_kchunk":       "PhdfReadKchunkHostFfi",
    "lorrax_phdf5_read_kchunk_union": "PhdfReadKchunkUnionHostFfi",
    "lorrax_phdf5_write":             "PhdfWriteHostFfi",
}

# Per-platform library spec: .so filename, env-var override, in-tree build
# dir (relative to this file's cpp/), target table, and the build command
# for the not-found error.
_PLATFORMS = {
    "CUDA": dict(
        so_name="liblorrax_ffi.so",
        env="LORRAX_FFI_SO",
        build_subdir="build",
        targets=_CUDA_TARGET_SYMBOLS,
        build_hint="src/ffi/common/cpp/run_shifter.sh bash src/ffi/common/cpp/build.sh",
    ),
    "cpu": dict(
        so_name="liblorrax_ffi_host.so",
        env="LORRAX_FFI_HOST_SO",
        build_subdir="host/build",
        targets=_HOST_TARGET_SYMBOLS,
        build_hint="bash src/ffi/common/cpp/host/build_host.sh",
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
    build_dir = here.parent / "cpp" / spec["build_subdir"]
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
    for c in _candidate_paths(platform):
        if c.is_file():
            return c
    spec = _PLATFORMS[platform]
    hints = "\n  ".join(str(p) for p in _candidate_paths(platform)) or "(none)"
    raise FileNotFoundError(
        f"Could not locate {spec['so_name']} (platform={platform}).  "
        f"Build with:\n    {spec['build_hint']}\n"
        "Paths searched:\n  " + hints
    )


def _set_argtypes(lib: ctypes.CDLL, platform: str) -> None:
    """Declare argtypes/restype for the lrx_* entry points ``lib`` exports."""
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
    phdf5/cpp/api.cc compile into BOTH liblorrax_ffi.so and, for the host
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


def _declare_slate(lib: ctypes.CDLL) -> None:
    """SLATE context lifecycle — exported by BOTH platform libraries
    (slate/cpp/context.cc is pure MPI and compiled into each).  Absent
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
        The ``.so`` is missing, or ``dlopen`` failed (an unresolved
        ``DT_NEEDED``, a glibc/GLIBCXX mismatch, a wrong
        ``LD_LIBRARY_PATH``).  **The handler may well be compiled;
        nothing about the build is wrong.**
    ``loaded but does not export <symbol>``
        The genuine partial-build case — this, and ONLY this, is a
        "rebuild the library" problem.

    Why the distinction is worth a function: conflating the middle case
    with the last one produces an error that tells you to rebuild a
    library that is perfectly fine, while the actual cause (one missing
    entry in ``LD_LIBRARY_PATH``) goes unmentioned.  On Frontera the
    unified host lib needs MKL **and** the Intel compiler runtime
    (``libimf``/``libsvml``/``libintlc``/``libirng``) **and** the phdf5
    ``libhdf5.so.310`` **and** ``libfabric`` — only some of which are in
    its RUNPATH.  Getting that wrong silently downgraded an N×1 mesh to
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
            f"LD_LIBRARY_PATH must cover its DT_NEEDED — on Frontera that "
            f"means MKL + the Intel compiler runtime + parallel HDF5 + "
            f"libfabric + SLATE's lib64; `ldd <so>` lists what is missing).")
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
    ``slab_io=auto`` to :attr:`SlabIOBackend.PHDF5_FFI`.

    False on a host lib built before workstream AE (read-only), which is
    why the router demotes rather than failing at the first ζ write; use
    :func:`probe_target` when reporting the reason to a human."""
    return has_target("lorrax_phdf5_write", platform)


def get_lib(platform: Optional[str] = None) -> ctypes.CDLL:
    """Return the loaded FFI library for ``platform``; idempotent.

    ``platform`` is "CUDA" or "cpu"; ``None`` follows the JAX default
    backend, so wrapper call sites stay platform-agnostic.
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
    path = _locate_so(platform)
    # Load h5py's HDF5 FIRST.  Both FFI libraries link the site's Intel
    # parallel HDF5 (``libhdf5.so.310`` = 1.14.6 on Frontera) and we dlopen
    # them RTLD_GLOBAL, which publishes every ``H5*`` symbol into the global
    # namespace.  h5py ships its OWN, ABI-incompatible HDF5
    # (``h5py.libs/libhdf5-*.so.320`` = 2.0.0); if the FFI lands first, the
    # later ``import h5py`` binds its extension modules to the Intel symbols
    # and dies at import with the useless
    # ``ValueError: Not a datatype (not a datatype)``  (measured, wk_AG job
    # 7876369).  Production happens to import h5py long before any FFI probe,
    # so this never fired in a driver — but every diagnostic script that
    # probes the FFI first hits it, and nothing said why.  Importing h5py
    # here makes the safe order unconditional; it is already a hard
    # dependency, so this costs nothing after the first call.
    try:
        import h5py  # noqa: F401
    except Exception:
        pass
    lib = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    _set_argtypes(lib, platform)
    _register_ffi_targets(lib, platform)
    _LIBS[platform] = lib
    _LIB_PATHS[platform] = str(path)
    return lib


def _loaded_path(platform: str) -> str:
    """The .so path dlopen'd for ``platform`` (for error messages)."""
    return _LIB_PATHS.get(platform, f"<{platform} library>")


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

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

    CUDA  liblorrax_ffi.so       cuSOLVERMp/cuBLASMp/cusolverMg/phdf5/slate
    cpu   liblorrax_ffi_host.so  slate host handlers (Target::HostTask)

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
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Dict, Optional

import jax
import jax.ffi

__all__ = ["get_lib"]

_LIBS: Dict[str, ctypes.CDLL] = {}

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
    "lorrax_cusolvermg_eigh_f64":   "EighMgF64",
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
# same target names as the CUDA table, registered under platform="cpu".
_HOST_TARGET_SYMBOLS = {
    "lorrax_slate_eigh":              "SlateEighHostFfi",
    "lorrax_slate_potrf":             "SlatePotrfHostFfi",
    "lorrax_slate_trsm":              "SlateTrsmHostFfi",
    "lorrax_slate_batched_potrf":     "SlateBatchedPotrfHostFfi",
    "lorrax_slate_batched_trsm":      "SlateBatchedTrsmHostFfi",
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
    backend = jax.default_backend()
    if backend in ("gpu", "cuda"):
        return "CUDA"
    if backend == "cpu":
        return "cpu"
    raise RuntimeError(
        f"lorrax_ffi: no FFI library for JAX backend {backend!r} "
        f"(supported: cuda, cpu).")


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

    # phdf5 lifecycle
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
    (slate/cpp/context.cc is pure MPI and compiled into each)."""

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
    lib = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    _set_argtypes(lib, platform)
    _register_ffi_targets(lib, platform)
    _LIBS[platform] = lib
    return lib


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
def phdf5_open(path: str, p: int, q: int, rank: int, world_size: int,
               mode_flag: int) -> int:
    """Collective open/create of a parallel-HDF5 file.

    mode_flag: 0 = truncate ('w'), 1 = append-or-create ('a'), 2 = read-only ('r').
    """
    lib = get_lib("CUDA")
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


def phdf5_close(ctx_handle: int) -> None:
    get_lib("CUDA").lrx_phdf5_close(int(ctx_handle))


def phdf5_init_mpi() -> None:
    """Eagerly init MPI_THREAD_MULTIPLE so the first ``open_file`` in the
    hot path doesn't pay the ~400 ms MPI_Init_thread cost.  Collective
    across all ranks; idempotent after first call.
    """
    get_lib("CUDA").lrx_phdf5_init_mpi()


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
                         shape, dtype_name: str) -> int:
    """Collective create/open of an N-D HDF5 dataset.  Returns hid_t as int64.

    ``shape`` is any sequence of non-negative ints (tuple, list, numpy array).
    """
    if dtype_name not in _DTYPE_TAG:
        raise ValueError(f"phdf5_ensure_dataset: unsupported dtype {dtype_name}")
    shape = tuple(int(s) for s in shape)
    if not shape:
        raise ValueError("phdf5_ensure_dataset: shape must be non-empty")
    lib = get_lib("CUDA")
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


def phdf5_open_dataset_ro(ctx_handle: int, ds_name: str) -> int:
    """Collective H5Dopen of an existing dataset.  Returns hid_t as int64."""
    lib = get_lib("CUDA")
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
def create_slate_context(rank: int, world_size: int, p: int, q: int) -> int:
    """Collective create of a SLATE context; returns opaque int64 handle.

    Inits MPI_THREAD_MULTIPLE if not already inited, then dups
    MPI_COMM_WORLD for SLATE's exclusive use.  Raises RuntimeError on
    failure with a message from the .so's error buffer.
    """
    lib = get_lib()
    err = ctypes.create_string_buffer(_ERR_CAP)
    h = lib.lrx_slate_context_create(
        int(rank), int(world_size), int(p), int(q), err, _ERR_CAP)
    if int(h) == 0:
        msg = err.value.decode("utf-8", errors="replace")
        raise RuntimeError(f"lorrax_ffi slate.context_create failed: {msg}")
    return int(h)


def create_slate_subrow_context(rank: int, world_size: int,
                                Px: int, Py: int) -> int:
    """Collective create of a SLATE sub-row context; returns int64 handle.

    The sub-comm is MPI_COMM_WORLD split by x-coordinate (color=x_rank,
    key=y_rank), producing one comm of size ``Py`` per X-row.  Intended
    for batched ops where each X-row independently processes a slice of
    a 3-D (Nbatch, N, N) input distributed as P('x', None, 'y').
    """
    lib = get_lib()
    err = ctypes.create_string_buffer(_ERR_CAP)
    h = lib.lrx_slate_subrow_context_create(
        int(rank), int(world_size), int(Px), int(Py), err, _ERR_CAP)
    if int(h) == 0:
        msg = err.value.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"lorrax_ffi slate.subrow_context_create failed: {msg}")
    return int(h)


def destroy_slate_context(ctx_handle: int) -> None:
    get_lib().lrx_slate_context_destroy(int(ctx_handle))


def slate_init_mpi() -> None:
    """Eagerly init MPI_THREAD_MULTIPLE from outside SLATE's hot path.
    Idempotent; no-op if MPI is already initialized.
    """
    get_lib().lrx_slate_init_mpi()

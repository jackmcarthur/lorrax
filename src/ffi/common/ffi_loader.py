"""Locate, load, and register the LORRAX FFI shared library.

The library is a plain C shared object (no pybind/nanobind).  We load it
with :class:`ctypes.CDLL`, declare the entry-point signatures, and
register each XLA FFI handler symbol with ``jax.ffi.register_ffi_target``
on first use.

Public API
----------
get_lib() : ctypes.CDLL
    The loaded library with all ctypes argtypes/restype set on the
    ``lrx_*`` wrapper functions.  The XLA FFI handlers (``EighF64``,
    ``EighC128``) remain as raw ``ctypes._FuncPtr`` — we pass them to
    ``jax.ffi.pycapsule`` rather than calling them directly.

Environment overrides
---------------------
LORRAX_FFI_SO
    Absolute path to ``liblorrax_ffi.so``.  Takes precedence over search.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Optional

import jax
import jax.ffi

__all__ = ["get_lib"]

_LIB: Optional[ctypes.CDLL] = None

# Symbols the XLA FFI side exports (plain C via XLA_FFI_DEFINE_HANDLER_SYMBOL).
_FFI_TARGET_SYMBOLS = {
    "lorrax_cusolvermp_eigh_f64":  "EighF64",
    "lorrax_cusolvermp_eigh_c128": "EighC128",
    "lorrax_cusolvermg_eigh_f64":  "EighMgF64",
}

# Error buffer size for the lrx_ wrappers.
_ERR_CAP = 512


def _candidate_paths() -> list[Path]:
    paths: list[Path] = []
    env = os.environ.get("LORRAX_FFI_SO")
    if env:
        paths.append(Path(env))
    here = Path(__file__).resolve()
    build_dir = here.parent / "cpp" / "build"
    if build_dir.is_dir():
        paths.extend(sorted(build_dir.glob("liblorrax_ffi*.so")))
    for p in sys.path:
        pp = Path(p)
        if pp.is_dir():
            paths.extend(sorted(pp.glob("liblorrax_ffi*.so")))
    seen, unique = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _locate_so() -> Path:
    for c in _candidate_paths():
        if c.is_file():
            return c
    hints = "\n  ".join(str(p) for p in _candidate_paths()) or "(none)"
    raise FileNotFoundError(
        "Could not locate liblorrax_ffi*.so.  Build with:\n"
        "    bash src/ffi/common/cpp/build.sh\n"
        "Paths searched:\n  " + hints
    )


def _set_argtypes(lib: ctypes.CDLL) -> None:
    """Declare argtypes/restype for all lrx_* entry points."""

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


def _register_ffi_targets(lib: ctypes.CDLL) -> None:
    for target_name, cpp_symbol in _FFI_TARGET_SYMBOLS.items():
        fn = getattr(lib, cpp_symbol)
        try:
            jax.ffi.register_ffi_target(
                target_name,
                jax.ffi.pycapsule(fn),
                platform="CUDA",
            )
        except Exception as exc:
            if "already registered" not in str(exc).lower():
                raise


def get_lib() -> ctypes.CDLL:
    """Return the loaded lorrax_ffi.so; idempotent."""
    global _LIB
    if _LIB is not None:
        return _LIB
    path = _locate_so()
    lib = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    _set_argtypes(lib)
    _register_ffi_targets(lib)
    _LIB = lib
    return lib


# ---------------------------------------------------------------------------
# Thin Pythonic helpers (so call sites don't keep re-writing ctypes plumbing)
# ---------------------------------------------------------------------------
def _check_err(rc: int, err_buf: ctypes.Array) -> None:
    if rc != 0:
        msg = err_buf.value.decode("utf-8", errors="replace")
        raise RuntimeError(f"lorrax_ffi error ({rc}): {msg}")


def nccl_unique_id_bytes() -> int:
    return int(get_lib().lrx_nccl_unique_id_bytes())


def fill_nccl_unique_id(addr: int) -> None:
    err = ctypes.create_string_buffer(_ERR_CAP)
    rc = get_lib().lrx_fill_nccl_unique_id(addr, err, _ERR_CAP)
    _check_err(rc, err)


def create_cusolvermp_context(
    rank: int, world_size: int,
    nccl_unique_id_addr: int, nccl_unique_id_nbytes: int,
    p: int, q: int, grid_layout_col_major: bool = True,
) -> int:
    lib = get_lib()
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
    get_lib().lrx_destroy_cusolvermp_context(int(ctx_handle))


def smoke_allreduce_sum(ctx_handle: int, device_ptr: int, nelems: int) -> int:
    return int(get_lib().lrx_smoke_allreduce_sum(
        int(ctx_handle), int(device_ptr), int(nelems)))


def version_info() -> dict:
    lib = get_lib()
    rt   = ctypes.c_int(0)
    drv  = ctypes.c_int(0)
    nccl = ctypes.c_int(0)
    lib.lrx_version_info(ctypes.byref(rt), ctypes.byref(drv), ctypes.byref(nccl))
    return {"cuda_runtime": rt.value, "cuda_driver": drv.value, "nccl": nccl.value}

"""OpenBLAS spin tuning: set at import of ``runtime``, verified per library, gated.

WHAT.  OpenBLAS keeps its worker threads in a busy-wait after a call so the
next call finds them hot.  LORRAX drivers interleave short LAPACK calls with
Python work, so the spinning workers hold the cores the calling thread needs:
one 140x140 ``cho_factor`` took 28.7 ms in situ against 0.067 ms standalone
(``minimax.uniform_rule``, DEV/327), and the Fe V2 rank-0 tangential rule
build on Frontier ran 264.7 s untuned against 131.8 s with
``OPENBLAS_THREAD_TIMEOUT=1`` (HOSTLAT lane, 2026-10-15, eqp bit-identical).
The setting costs 22 % only in a back-to-back BLAS loop, which no driver is.

ONE ENVIRONMENT READ PER LIBRARY.  Each OpenBLAS copy reads
``OPENBLAS_THREAD_TIMEOUT`` once, in its own constructor, when it is mapped.
A LORRAX process holds two: numpy's ``libscipy_openblas64_`` (mapped by
``import numpy``) and scipy's ``libscipy_openblas`` (mapped by the first
``scipy.linalg`` import; the minimax rule builders' LAPACK runs there).  The
variable therefore has to be in the environment before EACH library maps.
:func:`tune_at_import` runs as the first statement of ``runtime`` and always
``setdefault``\\ s it, so every library mapped afterwards reads it — including
scipy's when numpy came first (until 2026-10-15 that case skipped the
``setdefault`` and left scipy's copy untuned as well: the 2x above).

THE WITNESS IS THE LIBRARY, NOT ``os.environ``.  :func:`census` states, for
every OpenBLAS mapped in the process, the value it actually read: live from
the library where it exports ``openblas_thread_timeout`` (scipy's copy does),
otherwise from the environment the library could have seen — the process's
initial environment (``/proc/self/environ``) for a library already mapped
when ``runtime`` was imported, the tuned environment for one mapped later.

THE GATE.  :func:`enforce` (GATE ``blas_spin``, called by
``runtime.bootstrap``) refuses a process in which an OpenBLAS copy is proven
untuned and the caller made no choice: that happens only when numpy was
imported before any LORRAX package and the launch did not export the
variable.  An exported value (any value) is the caller's choice and is
honoured; ``LORRAX_BLAS_TUNE=0`` declines the tuning and the gate with it.

PLATFORM-NEUTRAL.  This is a host-CPU property of OpenBLAS, identical under
the cpu, CUDA and ROCM legs, so it is not a ``lxkit.LEGS`` field and not a
site row: ``runtime`` owns it for every leg.  Thread COUNT is not set here
(launcher's; :func:`runtime.default_blas_threads`).

Stdlib only: this module runs before numpy, jax and every other import.
"""
from __future__ import annotations

import os
import sys

from .env_flags import env_falsy

VAR = "OPENBLAS_THREAD_TIMEOUT"
DEFAULT = "1"
GATE = "blas_spin"
#: Exported by OpenBLAS's ``driver/others/openblas_env.c``: the value the
#: library read from its environment at load (0 when unset).
LIVE_SYMBOLS = ("openblas_thread_timeout",)

#: Filled in by :func:`tune_at_import`; ``runtime._BLAS_TUNE`` is this dict.
STATE: dict = {"applied": None, "timeout": None, "reason": None,
               "numpy_already_imported": None, "caller_value": None,
               "preloaded": None, "libs": None}


def tune_cpu_malloc_arenas(platforms: str) -> dict:
    """Bound glibc's CPU worker arenas before JAX creates its worker pools.

    MPI ranks need independent data, but hundreds of per-worker malloc arenas
    retain freed preprocessing scratch in each rank. Two arenas preserve a
    main/worker arena while bounding that fragmentation. Caller arena settings
    and the existing malloc-tuning opt-out win. No backend is initialized here.
    """
    result = dict(applied=False, maximum=None, reason=None)
    bridge = sys.modules.get("jax._src.xla_bridge")
    backend = getattr(bridge, "_default_backend", None)
    if platforms.strip().lower() != "cpu" or backend not in (None, "cpu"):
        result["reason"] = "not a resolved CPU client"
        return result
    if env_falsy("LORRAX_MALLOC_TUNE"):
        result["reason"] = "disabled by LORRAX_MALLOC_TUNE"
        return result
    if "MALLOC_ARENA_MAX" in os.environ:
        result["reason"] = "caller supplied MALLOC_ARENA_MAX"
        return result
    if any(s.startswith("glibc.malloc.arena_max=")
           for s in os.environ.get("GLIBC_TUNABLES", "").split(":")):
        result["reason"] = "caller supplied glibc.malloc.arena_max"
        return result
    try:
        import ctypes
        libc = ctypes.CDLL(None)
        if not hasattr(libc, "gnu_get_libc_version"):
            result["reason"] = "not glibc"
            return result
        mallopt = libc.mallopt
        mallopt.argtypes = [ctypes.c_int, ctypes.c_int]
        mallopt.restype = ctypes.c_int
        if mallopt(-8, 2) != 1:             # M_ARENA_MAX, malloc.h
            result["reason"] = "glibc rejected M_ARENA_MAX=2"
            return result
    except (AttributeError, OSError) as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
        return result
    result.update(applied=True, maximum=2)
    return result


def mapped_openblas() -> list | None:
    """Paths of every OpenBLAS shared object mapped in this process, or None
    when the process map is unreadable (no ``/proc``)."""
    try:
        with open("/proc/self/maps") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return None
    paths = []
    for line in lines:
        parts = line.split(None, 5)
        if len(parts) == 6:
            path = parts[5].strip()
            if "openblas" in os.path.basename(path).lower() and ".so" in path \
                    and path not in paths:
                paths.append(path)
    return paths


def initial_environ() -> dict | None:
    """The environment this process was exec'd with (``/proc/self/environ``),
    which is what a library mapped before any in-process ``setenv`` read; None
    when unreadable."""
    try:
        with open("/proc/self/environ", "rb") as fh:
            raw = fh.read()
    except OSError:
        return None
    env = {}
    for item in raw.split(b"\0"):
        key, sep, value = item.partition(b"=")
        if sep:
            env[key.decode(errors="replace")] = value.decode(errors="replace")
    return env


def live_timeout(path: str) -> int | None:
    """The timeout ``path`` read at load, from its exported symbol, or None
    when it exports none (numpy's prefixed ILP64 copy does not)."""
    try:
        import ctypes
        lib = ctypes.CDLL(path, mode=getattr(os, "RTLD_NOLOAD", 0) | os.RTLD_NOW)
    except OSError:
        return None
    for name in LIVE_SYMBOLS:
        fn = getattr(lib, name, None)
        if fn is not None:
            fn.restype = ctypes.c_int
            return int(fn())
    return None


def tune_at_import(env=os.environ) -> bool:
    """Set ``OPENBLAS_THREAD_TIMEOUT`` (``setdefault``) for every OpenBLAS not
    yet mapped, and record which copies were mapped before it.  Returns True
    when the default is in the environment."""
    already = "numpy" in sys.modules
    caller = env.get(VAR)
    STATE.update(numpy_already_imported=already, caller_value=caller,
                 preloaded=mapped_openblas())
    if env_falsy("LORRAX_BLAS_TUNE"):
        STATE.update(applied=False, reason="disabled by LORRAX_BLAS_TUNE")
        return False
    env.setdefault(VAR, DEFAULT)
    STATE.update(applied=True, timeout=env[VAR], reason=None)
    return True


def classify(path: str, *, live: int | None, preloaded: bool,
             caller_value: str | None, initial_value: str | None,
             initial_known: bool, env_value: str | None) -> tuple:
    """``(state, value, witness)`` of one library; ``state`` is ``tuned``,
    ``caller`` (an explicit choice, honoured), ``untuned`` or ``unknown``.
    Pure, so the decision table is unit-tested without a process."""
    if live is not None:
        if live > 0:
            return ("tuned" if str(live) == DEFAULT else "caller"), str(live), "live symbol"
        if caller_value is not None:
            return "caller", caller_value, "live symbol 0; caller's export"
        return "untuned", "0", "live symbol"
    if preloaded:
        if initial_value is not None:
            return ("tuned" if initial_value == DEFAULT else "caller",
                    initial_value, "initial environment (mapped before runtime)")
        if caller_value is not None:
            return "caller", caller_value, "set in-process before runtime; load order unknown"
        if initial_known:
            return "untuned", None, "absent from the initial environment; mapped before runtime"
        return "unknown", None, "mapped before runtime; initial environment unreadable"
    if env_value is not None:
        return ("tuned" if env_value == DEFAULT else "caller", env_value,
                "mapped after runtime set the environment")
    return "untuned", None, "mapped after runtime; variable removed from the environment"


def census() -> list:
    """One row per mapped OpenBLAS: ``{lib, state, value, witness}``."""
    mapped = mapped_openblas()
    if mapped is None:
        return []
    preloaded = set(STATE.get("preloaded") or ())
    init = initial_environ()
    rows = []
    for path in mapped:
        state, value, witness = classify(
            path, live=live_timeout(path), preloaded=path in preloaded,
            caller_value=STATE.get("caller_value"),
            initial_value=(init or {}).get(VAR), initial_known=init is not None,
            env_value=os.environ.get(VAR))
        rows.append({"lib": os.path.basename(path), "state": state,
                     "value": value, "witness": witness})
    STATE["libs"] = rows
    return rows


def enforce() -> list:
    """GATE ``blas_spin``: refuse when an OpenBLAS copy in this process is
    proven untuned and the caller chose nothing.  Returns the census."""
    rows = census()
    if STATE.get("applied") is False:          # LORRAX_BLAS_TUNE=0: declined
        return rows
    bad = [r for r in rows if r["state"] == "untuned"]
    if bad:
        libs = ", ".join(f"{r['lib']} ({r['witness']})" for r in bad)
        raise RuntimeError(
            f"GATE {GATE}: OpenBLAS idle workers spin in this process: {libs}. "
            f"OpenBLAS reads {VAR} once, when the library is mapped, and this "
            f"copy was mapped (numpy imported) before any LORRAX package set it. "
            f"Untuned, the Fe V2 rank-0 rule build measured 264.7 s against "
            f"131.8 s. Remedy: import runtime (or any LORRAX driver package, "
            f"e.g. gw) before numpy, or export {VAR}=1 in the launch "
            f"environment; LORRAX_BLAS_TUNE=0 runs untuned on purpose.")
    return rows

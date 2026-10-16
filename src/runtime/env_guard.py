"""The process environment is frozen once LORRAX starts native helper threads (GATE env-frozen).

glibc's ``getenv`` takes no lock.  ``setenv`` of a NEW key reallocates the
``environ`` array and frees the old one, so a native thread scanning
``environ`` inside ``getenv`` while Python's main thread runs
``os.environ[k] = v`` (``putenv``) for a key not yet set reads freed memory.
Overwriting an existing key swaps one pointer (glibc never frees the old
string) and ``unsetenv`` shifts entries in place, so neither frees what a
reader holds; a reader may see the old or the new value.  ROCm's HSA
runtime calls ``getenv`` on every code-object load (``hipModuleLoadData``), and
MPI-IO, HDF5 and XLA read the environment from their own threads, so after
start-up the race is reachable on any leg.

Measured (2026-10-16, Frontier MI250X, Fe V2 P=4): the intermittent start-up
SIGSEGV (rc 139, about 10 s in, at the first SlabIO open) was exactly this:
the rocBLAS warm-up thread (:mod:`runtime.vendor_kernels`) faulted in
``rocblas_dgemm -> hipModuleLoadData -> libhsa-runtime64 -> getenv`` while
the main thread exported ``LORRAX_PHDF5_STRIPE_COUNT``/``_SIZE_FS`` (new keys)
before opening WFN.h5.  Under load the Tensile code-object load is slow, so
the window widened: 5 crashes in 60 concurrent start-ups.  Evidence (scratch
/lustre/orion/mat280/scratch/jackmc/crash): native backtraces from an LD_PRELOAD
SIGSEGV handler in s2/B.3/run.r1.log and s2/F.1/run.r0.log; faulthandler stacks
naming the faulting thread in s1/{C.2,E.3,F.3}/run.r*.log; the main thread's
environment writes with the live threads in envtrace/run.r1.log.

Policy, so the class cannot come back: every LORRAX environment write happens
before :func:`freeze_process_environment`, which ``runtime.initialize_communicator_stack``
calls immediately before it starts the first native helper thread (step 7b).
Afterwards a write that ADDS a key from LORRAX's own code (``src/`` or
``services/*/src`` of this checkout) refuses with ``GATE env-frozen``; one from
any other code is announced once per key and caller and performed (the residual
risk is named, not hidden).  Overwrites of existing keys and unsets pass.  Values that later code needs travel in-process
(``common.sanity.forced_level``, function arguments), never through ``os.environ``.
Reading the environment is always safe.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = ["freeze_process_environment", "frozen", "thaw_for_tests"]

_STATE: dict = {"frozen": False, "where": "", "announced": set()}
_ORIG: dict = {}
_HERE = Path(__file__).resolve()
_CHECKOUT = _HERE.parents[2]                       # <checkout>/src/runtime/env_guard.py
_OWN_ROOTS = tuple(str(p) + os.sep for p in (_CHECKOUT / "src", _CHECKOUT / "services"))


def frozen() -> bool:
    return bool(_STATE["frozen"])


def _resolved(path: str) -> str:
    try:
        return str(Path(path).resolve())
    except OSError:
        return path


def _caller() -> tuple[str, int]:
    """The first frame outside this module and ``os`` (resolved: HOME may be a symlink)."""
    f = sys._getframe(2)
    while f is not None:
        name = f.f_code.co_filename
        if not (name.endswith(("os.py", "<frozen os>")) or _resolved(name) == str(_HERE)):
            return name, f.f_lineno
        f = f.f_back
    return "?", 0


def _own(path: str) -> bool:
    p = _resolved(path)
    return p.startswith(_OWN_ROOTS) and os.sep + "tests" + os.sep not in p


def _guard(kind: str, key, call, *args):
    k = key.decode(errors="replace") if isinstance(key, bytes) else str(key)
    # os.environ updates its mapping after putenv returns, so a key absent here is new to environ.
    if _STATE["frozen"] and kind == "write" and k not in os.environ:
        path, line = _caller()
        msg = (f"GATE env-frozen: got a process-environment write adding {k!r} from {path}:{line} after "
               f"{_STATE['where']} started native helper threads; want every LORRAX environment write "
               "before that point (runtime.env_guard); why: glibc getenv takes no lock, so a setenv "
               "of a new key reallocates environ while a native thread (HSA/HIP module load, MPI-IO, "
               "XLA) runs getenv is a use-after-free -- the rc 139 start-up SIGSEGV of 2026-10-16; fix: "
               "set it before runtime.initialize_communicator_stack step 7b, or pass the value in-process")
        if _own(path):
            raise RuntimeError(msg)
        if (k, path, line) not in _STATE["announced"]:
            _STATE["announced"].add((k, path, line))
            print(f"  [runtime] WARNING {msg} (not LORRAX code: performed)", file=sys.stderr, flush=True)
    return call(*args)


def freeze_process_environment(where: str) -> None:
    """From now on a LORRAX write adding a key refuses (GATE env-frozen); others are announced."""
    if not _ORIG:
        _ORIG["putenv"], _ORIG["unsetenv"] = os.putenv, os.unsetenv
        os.putenv = lambda key, value: _guard("write", key, _ORIG["putenv"], key, value)
        os.unsetenv = lambda key: _guard("unset", key, _ORIG["unsetenv"], key)
    _STATE["frozen"], _STATE["where"] = True, where


def thaw_for_tests() -> None:
    """Undo the freeze (tests only: a test process starts the runtime more than once)."""
    _STATE["frozen"], _STATE["where"] = False, ""

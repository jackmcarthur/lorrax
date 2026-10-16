"""Ordered process end for every P > 1 entry point, not only the ones that call
:func:`runtime.finalize_process` themselves.

The failure this removes (measured, Frontier, 2026-10-16)
---------------------------------------------------------
A P > 1 process that ends by interpreter teardown (``sys.exit(main())``,
``raise SystemExit(...)`` at module level, or falling off the end of the
script) aborted after all of its work had succeeded, on every rank::

    terminate called after throwing an instance of '\\x12'
    FATAL: exception not rethrown

rc 134 (once rc 139).  Rate without this module: 8 of 9 P=16 runs and 3 of 3
P=4 runs of a collectives probe that ends by falling off the end of the
script; ``run_main_and_finalize`` was clean 6/6.  The aborting thread is not
the main thread (native backtrace of the SIGABRT: ``start_thread`` ->
libpython ``thread_run`` -> Python -> ``libjax_common.so`` -> Python ->
``libjax_common.so`` -> ``std::terminate``, which the symbolizer labels with
the nearest symbol ``std::unexpected``): a non-main Python thread is inside
jaxlib when CPython finalizes.  The reading that CPython 3.12 ends it with
``pthread_exit`` when it re-takes the GIL and that the forced unwind meets a
``noexcept`` jaxlib frame fits ``FATAL: exception not rethrown``; which thread
it is (and whether it is a daemon) was not identified.  With this module the
same probe ended cleanly 5/5 at P=16 and 2/2 at P=4 (an unpatched P=4 control
in the same allocation aborted).

The fix is the one :func:`runtime.finalize_process` already applies to the
drivers that call it: finish every duty in order, then ``os._exit`` before
interpreter finalization can start.  This module makes that the process end
of every entry point, so no CLI has to remember it.

How
---
* :func:`install` registers :func:`_at_shutdown` with
  ``threading._register_atexit``.  CPython runs those hooks at the start of
  ``Py_FinalizeEx`` (``threading._shutdown``), before any ``atexit`` callback
  and before any daemon thread can be torn down, which is the last point
  where an ordered teardown is still possible.  (The same hook point
  ``concurrent.futures`` uses.)
* The exit status is not visible to Python at that point (CPython keeps it on
  the C stack between ``handle_system_exit`` and ``Py_Exit``), so a
  ``sys.monitoring`` ``PY_UNWIND`` callback records the exception that leaves
  the ``__main__`` module frame.  ``PY_UNWIND`` cannot be a local event in
  CPython 3.12 (``set_local_events`` refuses it), so the callback is global;
  it is one comparison per exception-unwound frame and touches no device.
* Status 0 (the script ended, or ``SystemExit(0)`` / ``SystemExit(None)``)
  -> ``finalize(0)``: effects barrier, the collective finalize barrier,
  ``jax.distributed.shutdown``, every ``atexit`` duty exactly once,
  ``os._exit(0)``.
* Any other status -> flush and ``os._exit(status)`` with NO collective
  teardown, the contract :func:`runtime.run_main_and_finalize` has for a
  nonzero exit at P > 1 (a rank-local failure must not wait in a collective
  its peers will never join).  An uncaught non-``SystemExit`` exception
  normally never gets here at P > 1: the fail-fast excepthook has already
  ended the process.
* Idempotent: one install per process; an explicit
  :func:`runtime.finalize_process` (``run_main_and_finalize``) ends the process
  first, so the hook never runs there.

Standard library only; JAX is reached only through the injected ``finalize``.
"""
from __future__ import annotations

import os
import sys
import threading

#: ``sys.monitoring`` tool name.  Ids 0 (debugger), 1 (coverage), 2 (profiler)
#: and 5 (optimizer) are reserved by convention; 3 and 4 are free.
TOOL_NAME = "lorrax-ordered-exit"
_FREE_TOOL_IDS = (4, 3)

_state: dict = {"installed": False, "reason": None, "main_code": None,
                "exit_exc": None, "tool": None, "finalize": None, "ran": False}


def status() -> dict:
    """What :func:`install` did, for the start-up report and tests."""
    return {"installed": _state["installed"], "reason": _state["reason"],
            "tool": _state["tool"]}


def _main_module_code(frame):
    """Code object of the outermost frame running ``__main__``'s module body, or None."""
    code = None
    while frame is not None:
        if frame.f_globals.get("__name__") == "__main__" and frame.f_code.co_name == "<module>":
            code = frame.f_code
        frame = frame.f_back
    return code


def exit_status(exc) -> tuple[int, str | None]:
    """(status, message) CPython would exit with for ``exc`` leaving ``__main__``.

    None (the module body completed) -> 0.  ``SystemExit``: None -> 0, an int
    -> that int, anything else -> 1 (CPython's ``handle_system_exit``
    prints that value itself).  Any other exception -> 1.
    """
    if exc is None:
        return 0, None
    if isinstance(exc, SystemExit):
        code = exc.code
        if code is None:
            return 0, None
        if isinstance(code, int) and not isinstance(code, bool):
            return code, None
        if isinstance(code, bool):
            return int(code), None
        return 1, str(code)
    return 1, None


def _on_unwind(code, offset, exc):
    if code is _state["main_code"]:
        _state["exit_exc"] = exc


def _flush():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:                                  # noqa: BLE001
            pass


def _at_shutdown():
    """``threading`` shutdown hook: the ordered end of the process."""
    if _state["ran"]:
        return
    _state["ran"] = True
    # A non-integer SystemExit value was already printed by CPython's
    # handle_system_exit before Py_Exit started the shutdown that called us.
    rc, _message = exit_status(_state["exit_exc"])
    if rc != 0:
        _flush()
        try:
            os._exit(rc)
        except OverflowError:                              # CPython exits 1 on such a code too
            os._exit(1)
    _state["finalize"](0)
    # A finalize that returned (a stub, or an already-running finalize) still
    # must not hand the process back to interpreter teardown.
    _flush()
    os._exit(0)


def install(finalize, *, frame=None) -> dict:
    """Make ``finalize(0)`` (or a bare ``os._exit(status)``) the end of this process.

    ``finalize``: a callable that performs the ordered teardown and ends the
    process (:func:`runtime.finalize_process`).  ``frame``: where to look for the
    ``__main__`` module frame (default: the caller's stack).  Returns
    :func:`status`.  Not installed, with the reason recorded, when there is no
    ``__main__`` module frame on the stack (embedded or interactive use), when
    ``sys.monitoring`` is unavailable (Python < 3.12), or when no monitoring
    tool id is free; the process then ends as before.
    """
    if _state["installed"]:
        return status()
    code = _main_module_code(frame if frame is not None else sys._getframe(1))
    if code is None:
        _state["reason"] = "no __main__ module frame on the stack"
        return status()
    mon = getattr(sys, "monitoring", None)
    if mon is None:
        _state["reason"] = "sys.monitoring unavailable (Python < 3.12)"
        return status()
    tool = next((t for t in _FREE_TOOL_IDS if mon.get_tool(t) is None), None)
    if tool is None:
        _state["reason"] = "no free sys.monitoring tool id (3 and 4 in use)"
        return status()
    register = getattr(threading, "_register_atexit", None)
    if register is None:
        _state["reason"] = "threading._register_atexit unavailable"
        return status()
    mon.use_tool_id(tool, TOOL_NAME)
    mon.register_callback(tool, mon.events.PY_UNWIND, _on_unwind)
    mon.set_events(tool, mon.get_events(tool) | mon.events.PY_UNWIND)
    try:
        register(_at_shutdown)
    except RuntimeError as exc:                            # interpreter already shutting down
        mon.set_events(tool, 0)
        mon.register_callback(tool, mon.events.PY_UNWIND, None)
        mon.free_tool_id(tool)
        _state["reason"] = f"threading shutdown already started ({exc})"
        return status()
    _state.update(installed=True, reason=None, main_code=code, tool=tool, finalize=finalize)
    return status()

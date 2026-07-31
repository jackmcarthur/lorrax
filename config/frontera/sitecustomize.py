"""wk_AS: rc=0 teardown fix for JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi.

Problem (AP.5b wart): with the jaxlib CPU 'mpi' collectives (MPItrampoline ->
MPIwrapper -> Intel MPI), a fully-correct run exits rc=1: jax registers
`atexit.register(collectives.Finalize)` (xla_bridge.py make_cpu_client);
that MPI_Finalize runs, then post-atexit interpreter/XLA C++ teardown makes
one more MPI call -> Intel MPI prints "Attempting to use an MPI routine
after finalizing MPICH" and the process exits 1 AFTER SUCCESS.

Gate: env LORRAX_MPI_FINALIZE_FIX
  hard_exit   - register an atexit hook at interpreter startup (site import
                time), i.e. FIRST, so it runs LAST -- after jax's
                collectives.Finalize has run cleanly.  It flushes stdio and
                os._exit(status): 0 on the normal path, preserved nonzero
                for uncaught exceptions and sys.exit(n).  The only thing it
                skips is the already-finalized-MPI C++ teardown.
  skip_atexit - diagnostic variant: intercept the atexit registration of
                <collectives>.Finalize so only one finalize path remains.
  (unset/other) - no-op.

Known hole (documented): `raise SystemExit(n)` NOT via sys.exit() is seen as
a clean exit by hard_exit and would return rc=0.  LORRAX CLIs signal failure
via exceptions or sys.exit(); both are preserved.
"""
import os

# --- init-order fix (wk_AS gw16 segfault): if XLA's MpiCollectives calls
# MPI_Init first, the granted thread level is below MPI_THREAD_MULTIPLE and
# concurrent MPI use (XLA collectives thread + mpi4py/h5py collective I/O)
# segfaults in the OFI progress engine (MPIDI_OFI_dispatch_function).
# LORRAX_MPI_INIT_FIRST=mpi4py imports mpi4py.MPI at interpreter startup —
# mpi4py requests MPI_THREAD_MULTIPLE, and every later Init (XLA's, guarded
# by MPI_Initialized) inherits it.
if os.environ.get("LORRAX_MPI_INIT_FIRST", "") == "mpi4py":
    try:
        from mpi4py import MPI as _asMPI  # noqa: F401  (MPI_Init_thread here)
    except Exception as _e:  # announce, never crash the interpreter
        print(f"[sitecustomize] LORRAX_MPI_INIT_FIRST=mpi4py FAILED: {_e}",
              flush=True)

_mode = os.environ.get("LORRAX_MPI_FINALIZE_FIX", "")

if _mode == "hard_exit":
    import atexit
    import sys

    _st = {"code": 0}

    _orig_excepthook = sys.excepthook

    def _hook(t, v, tb):
        _st["code"] = 1
        _orig_excepthook(t, v, tb)

    sys.excepthook = _hook

    _orig_exit = sys.exit

    def _exit(code=None):
        if code is None:
            _st["code"] = 0
        elif isinstance(code, int):
            _st["code"] = code
        else:
            _st["code"] = 1
        _orig_exit(code)

    sys.exit = _exit

    def _final():
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        if _st["code"] == 0:
            os._exit(0)
        # nonzero: fall through so the true failure rc propagates

    atexit.register(_final)

elif _mode == "skip_atexit":
    import atexit

    _orig_register = atexit.register

    def _register(f, *a, **k):
        if getattr(f, "__name__", "") == "Finalize" and hasattr(f, "__self__"):
            print("[sitecustomize] skip_atexit: NOT registering",
                  type(f.__self__).__name__ + ".Finalize", flush=True)
            return f
        return _orig_register(f, *a, **k)

    atexit.register = _register

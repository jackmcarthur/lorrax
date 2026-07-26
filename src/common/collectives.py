"""Cross-process barriers that refuse to be silently skipped.

Why this module exists
----------------------
The codebase had seven copies of this shape, wrapped around the barrier
that follows a rank-0-only HDF5 write::

    try:
        from jax.experimental import multihost_utils as _mh
        _mh.sync_global_devices("restart_head_scalars")
    except Exception:
        pass

The intent was benign — "don't blow up in a single-process run" — but the
handler catches far more than that.  ``sync_global_devices`` raises when
the distributed client is *broken*: a peer has died, a timeout expired, a
rank reached a barrier its peers never will.  Swallowing that is the worst
possible response, because the surviving rank then sails past a barrier
its peers are still blocked in.  The result is the campaign's signature
failure: the job hangs or half-completes, srun reaps it, and the exit code
at the sbatch level is **0**.

The correct split is:

* **single process** → the barrier is genuinely a no-op; skip it, silently.
* **``multihost_utils`` unavailable** → ditto (there is nothing to sync).
* **anything else** → fatal.  Let it propagate so the rank dies loudly and
  takes the job's exit code with it.

That is what :func:`barrier` implements, and it is the *only* thing it
does differently from the code it replaces.
"""
from __future__ import annotations

from typing import Any, Callable


__all__ = ["barrier", "process_count"]


def process_count() -> int:
    """Number of participating processes (1 when JAX is not distributed)."""
    try:
        import jax
    except ImportError:
        return 1
    try:
        return int(jax.process_count())
    except Exception:
        # jax present but no backend initialised yet — treat as local.
        return 1


def barrier(name: str, *, print_fn: Callable[..., Any] = print) -> bool:
    """Synchronize all processes at ``name``; fatal if the collective fails.

    Returns True when a real barrier was executed, False when it was
    correctly skipped as a single-process no-op.  Never returns after a
    *failed* barrier — that raises.

    Parameters
    ----------
    name
        Barrier label.  Must be identical on every rank; JAX uses it as
        the collective key, so a typo on one rank is itself a hang.
    print_fn
        Used only to annotate the failure before re-raising, so the log
        names the barrier that broke rather than just the traceback.
    """
    try:
        from jax.experimental import multihost_utils as _mh
    except ImportError:
        # No multihost support compiled in: nothing to synchronize.
        return False
    if process_count() <= 1:
        return False
    try:
        _mh.sync_global_devices(name)
    except Exception as exc:
        # Do NOT swallow.  A broken barrier in a multi-process run means
        # this rank is about to diverge from its peers; continuing turns a
        # crash into a hang-then-rc-0.
        print_fn(
            f"  *** LORRAX SANITY FAILURE: collective barrier {name!r} failed "
            f"across {process_count()} processes ({type(exc).__name__}: "
            f"{exc}).  This rank is now out of step with its peers — "
            f"aborting rather than continuing into an undefined collective "
            f"state. ***"
        )
        raise
    return True

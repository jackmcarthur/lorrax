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

The module's second job — :func:`device_put_process_local` — is the
mirror image: a cross-process placement that must NOT become a
collective.  See its docstring for the JAX behaviour it works around.
"""
from __future__ import annotations

from typing import Any, Callable


__all__ = ["barrier", "process_count", "device_put_process_local"]


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


# ---------------------------------------------------------------------------
# The anti-collective: staging a host table onto a multi-process sharding
# ---------------------------------------------------------------------------
#
# ``jax.device_put(host_numpy, NamedSharding(multi_process_mesh, spec))``
# LOOKS like a pure local placement — every rank already holds the same
# host table, each rank owns one device, so nothing needs to move.  It is
# not.  ``jax/_src/dispatch.py::_device_put_sharding_impl`` takes this
# branch whenever the target sharding is not fully addressable AND the
# operand is a numpy array (or an *uncommitted* ``jax.Array``)::
#
#     if xla_bridge.process_count() == len(s. ... .process_indices):
#         multihost_utils.assert_equal(x, fail_message=...)
#
# and ``assert_equal`` is ``process_allgather(x, tiled=True)`` — a REAL
# all-gather that concatenates every process's copy and materialises
# ``P × x.nbytes`` on **every** rank, on device AND (via the closing
# ``np.asarray``) on the host.  It is a debug assertion, it is silent, it
# is P-LINEAR, and it is charged to whatever stage happened to call
# ``device_put``.
#
# MEASURED (wk_Y probe, MoS2 12x12, rank-0 optimized HLO — scorecard Y.3):
# the WFN loader's two big tables did exactly this inside the
# ``A_centroid_load`` stage:
#
#     jit(_identity_fn)  s32[P*nk, 36,36,135]  =  P*nk*n_rtot*4
#                        1.61 GB @P=16 | 6.45 GB @P=64 | 14.5 GB @P=144
#     jit(_identity_fn)  c128[P*nk, ngkmax]    =  P*nk*ngkmax*16
#                        0.32 GB @P=16 | 1.27 GB @P=64 |  2.85 GB @P=144
#
# i.e. 17.4 GB/rank of pure assertion traffic at the production mesh, and
# the reason the planner "under-predicted" A_centroid_load by ~2x at
# 606c/P=64 (scorecard Y.5 / anomaly 5) — nothing in the memory model
# knows about a debug all-gather.
#
# The cure is the idiom the codebase already uses in three other places
# (``WfnLoader._assemble_process_local``, ``kin_ion_io.replicate_to_mesh``,
# the slab-io process-local writers): declare each rank's own shard with
# ``jax.make_array_from_single_device_arrays``.  Zero collectives, zero
# transient, and identical semantics for the values this is used on.

def device_put_process_local(host_array, sharding, *, check: bool | None = None):
    """Place a host array that EVERY process already holds onto ``sharding``
    **without a collective**.

    Drop-in replacement for ``jax.device_put(host_array, sharding)`` on a
    multi-process ``NamedSharding``.  Each process slices out only the
    shard(s) its own devices own (the full array for a replicated spec, a
    per-rank hyperslab for a sharded one) and declares them via
    ``jax.make_array_from_single_device_arrays``.

    **Correctness precondition** (the same one ``jax.device_put`` was
    silently spending 17 GB/rank to assert): ``host_array`` must be
    bit-identical on every process.  Every current caller satisfies it by
    construction — the tables are pure functions of the input file plus
    the mesh shape, computed identically on every rank.

    Set ``LORRAX_CHECK_REPLICA=1`` (or pass ``check=True``) to re-enable
    JAX's assertion for a debugging run; it costs exactly the all-gather
    described above, so it is OFF by default.

    Parameters
    ----------
    host_array
        numpy array (or anything ``np.asarray`` accepts) of the GLOBAL
        shape, identical on every process.
    sharding
        Target ``jax.sharding.Sharding``.
    check
        Force the cross-process equality assertion on/off.  ``None``
        (default) reads ``LORRAX_CHECK_REPLICA``.

    Returns
    -------
    jax.Array with the requested ``sharding``.
    """
    import os

    import jax
    import numpy as np

    # An input that is ALREADY a globally-sharded jax.Array is a genuine
    # reshard, not a host staging: ``device_put`` handles it on the
    # committed / non-fully-addressable branch, which never calls
    # ``assert_equal``.  Hand it straight back (and never ``np.asarray``
    # it — that would be the host gather this function exists to avoid).
    if isinstance(host_array, jax.Array) and not host_array.is_fully_addressable:
        return jax.device_put(host_array, sharding)

    arr = np.asarray(host_array)

    # Fully addressable target (single process, or a mesh this process
    # owns entirely): plain device_put takes the assertion-free path.
    if process_count() <= 1 or bool(getattr(sharding, "is_fully_addressable",
                                            False)):
        return jax.device_put(arr, sharding)

    if check is None:
        # Same falsy vocabulary as every other LORRAX knob ("", 0, false,
        # no, off — case-insensitive).  The old parse recognised only
        # ""/"0"/"false"/"False", so LORRAX_CHECK_REPLICA=off (or NO, OFF,
        # False in another case) silently ENABLED the debug all-gather —
        # a P-linear hidden collective (7.8 GB/rank at P=64, scorecard Y.5)
        # turned on by a string that reads as "disabled" (workstream AT).
        check = os.environ.get("LORRAX_CHECK_REPLICA", "0").strip().lower() \
            not in ("", "0", "false", "no", "off")
    if check:
        # Opt-in debug path — this IS the P-linear all-gather.
        return jax.device_put(arr, sharding)

    shape = tuple(int(s) for s in arr.shape)
    idx_map = sharding.addressable_devices_indices_map(shape)
    if not idx_map:
        # This process owns no device in the target sharding.  JAX's own
        # assertion is skipped in that case too, so device_put is already
        # collective-free; hand it over rather than build an empty array.
        return jax.device_put(arr, sharding)
    shards = [
        jax.device_put(np.ascontiguousarray(arr[idx]), dev)
        for dev, idx in idx_map.items()
    ]
    return jax.make_array_from_single_device_arrays(shape, sharding, shards)

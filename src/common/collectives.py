"""THE cross-process layer: one wrapped API for everything that spans ranks.

A driver that needs another process's data — or needs to agree with it —
calls this module and nothing below it.  Nothing here requires the caller to
know what a ``Mesh``, a ``NamedSharding``, a ``shard_map`` or
``multihost_utils`` is; that is the point.  Every function imports jax
*inside its body*, so importing this module costs nothing and is safe before
``jax.distributed.initialize``.

In the order a driver meets it: **topology** (:func:`process_rank_world`,
:func:`device_count`); **the mesh and its warm-up** (:func:`prepare_mesh`,
:func:`resolve_mesh`, :func:`single_device_mesh`); **staging onto a mesh
without a collective** (:func:`device_put_process_local`,
:func:`replicate_to_mesh`, :func:`shard_over_k`); **reductions and gathers**
(:func:`psum_replicate`, :func:`all_gather_processes`,
:func:`gather_to_host`, :func:`gather_indexed_blocks`,
:func:`psum_scatter_checked`); **the k-partitioned sweep**
(:func:`local_share`, :func:`sweep_local_k`, :func:`gather_k_blocks`); and
**barriers** (:func:`barrier`, :func:`warm_mesh_cliques`).

Two of these are the ones a physics driver actually wants:
:func:`prepare_mesh` (*"give me the run's mesh, with every communicator it
will need already created"*) and :func:`gather_k_blocks` (*"run this per-k
block on every k and hand me all of them"*), with the partition, the
pipelining, the warm-up and the gather inside.

Two pairs here look interchangeable and are not — read their docstrings
before choosing.  :func:`all_gather_processes` vs :func:`gather_to_host`
(different-value-per-process vs one-global-array; the wrong branch silently
multiplies the leading axis by P), and ``warm_mesh_cliques`` vs
``runtime.nccl_warmup`` (only :func:`prepare_mesh`'s CALL SITE unifies them;
their bodies must stay separate).

Why the barrier is the module's founding member
-----------------------------------------------
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


__all__ = [
    # topology
    "process_count",
    "process_rank",
    "process_rank_world",
    "device_count",
    # barriers
    "barrier",
    # mesh construction (and the warm-up that must accompany it)
    "resolve_mesh",
    "single_device_mesh",
    "prepare_mesh",
    # staging a host table onto a mesh, without a collective
    "device_put_process_local",
    "replicate_to_mesh",
    "shard_over_k",
    # reductions and gathers
    "psum_replicate",
    "all_gather_processes",
    "gather_to_host",
    "gather_indexed_blocks",
    "psum_scatter_checked",
    "report_collective_residual",
    "COLLECTIVE_RTOL",
    # the k-partitioned sweep
    "local_share",
    "sweep_local_k",
    "gather_k_blocks",
    "sweep_lookahead",
    "SWEEP_LOOKAHEAD_ENV",
    # impl=mpi clique warm-up
    "warm_mesh_cliques",
]


def _jax_int(attr: str, default: int) -> int:
    """``int(jax.<attr>())``, or ``default`` when jax is absent OR its backend
    is not initialised yet — both of which mean "treat this as local"."""
    try:
        import jax
        return int(getattr(jax, attr)())
    except Exception:                                          # noqa: BLE001
        return default


def process_count() -> int:
    """Number of participating processes (1 when JAX is not distributed)."""
    return _jax_int("process_count", 1)


def process_rank() -> int:
    """This process's index in the world (0 when JAX is not distributed)."""
    return _jax_int("process_index", 0)


def process_rank_world() -> tuple[int, int]:
    """``(rank, world)`` for a work partition — PROCESS granularity.

    The unit of parallelism for every sweep in this module is the process
    (one full CPU socket on Frontera, one device), not the device, because
    the sweeps run on process-local single-device arrays.  A build with
    more than one addressable device per process would break that
    identity; :func:`psum_replicate` asserts it rather than silently
    mis-assembling.
    """
    return process_rank(), process_count()


def device_count() -> int:
    """Global device count — what a run should print, not ``len(local)``."""
    import jax
    return int(jax.device_count())


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
# Mesh construction
# ---------------------------------------------------------------------------


def _require_addressable(mesh, *, origin: str):
    """Refuse a mesh in which THIS process owns no device.

    21 production sites build their own ``Mesh`` (enumerated in
    ``wk_REL/API_LINKAGE_AUDIT.md`` §B.1, plus 19 more in ``common/*_test.py``);
    five build the 1x1 one as
    ``Mesh(jax.devices()[:1].reshape(1,1), …)``, which is the standing bug —
    ``jax.devices()`` is the GLOBAL list, so every rank builds a mesh over
    *process 0's* device and every rank >= 1 gets a mesh with ZERO
    addressable devices.  Nothing complains until the first array is placed
    on it, at which point ``jax.make_array_from_process_local_data`` runs
    ``next(iter(addressable_shards.values()))`` (jax/_src/array.py:1017) over
    an empty map and raises a BARE ``StopIteration`` — no message, no rank,
    no mention of a mesh.

    ``Mesh.local_devices`` (jax/_src/mesh.py:425) keeps only devices whose
    ``process_index`` matches this client's, so an empty list is exactly that
    case.  Refusing here names the caller, which is the only place the mesh's
    provenance is still known.
    """
    if not list(mesh.local_devices):
        raise RuntimeError(
            f"{origin}: the mesh over devices "
            f"{[int(d.id) for d in mesh.devices.flat]} contains NO device "
            f"addressable by process {process_rank()} of {process_count()}.  "
            f"Nothing can be computed on it, and the first array placed on it "
            f"would raise a bare StopIteration from "
            f"jax.make_array_from_process_local_data.  The usual cause is "
            f"``jax.devices()[:1]`` (the GLOBAL list — process 0's device on "
            f"every rank) where ``collectives.single_device_mesh()`` was "
            f"meant.")
    return mesh


def resolve_mesh(mesh=None, *, axis_names=("x", "y")):
    """THE run's device mesh: pass your own, or get the canonical one.

    Callers that already hold the run's mesh (the GW driver) pass it, so no
    second mesh — and no second communicator — is created.  A CLI has none
    and gets the most-square factorisation of ``jax.devices()``, degenerating
    to 1x1 on a single device so every code path is identical at P=1.

    Either way the result is checked with :func:`_require_addressable`, so a
    mesh this process cannot compute on is refused here rather than
    manifesting as a bare ``StopIteration`` several layers down.
    """
    import jax
    import numpy as np
    from jax.sharding import Mesh

    if mesh is not None:
        return _require_addressable(mesh, origin="resolve_mesh(caller's mesh)")
    devices = jax.devices()
    total = len(devices)
    gx = int(np.sqrt(total))
    while gx > 1 and total % gx != 0:
        gx -= 1
    return _require_addressable(
        Mesh(np.asarray(devices).reshape(gx, total // gx),
             axis_names=tuple(axis_names)),
        origin="resolve_mesh")


#: The one cached process-local mesh.  A module global, not a
#: ``functools.lru_cache``, so it is greppable and so the identity contract
#: below is visible at the point it is enforced.
_PROCESS_LOCAL_MESH = None


def single_device_mesh():
    """THE 1x1 ``('x','y')`` mesh over **this process's own** device.

    Use wherever a jit must run process-locally: every rank gets a different
    mesh holding a different device, so the program compiles and runs with no
    collective and no cross-rank agreement on shapes.  Safe at any P — which
    is what distinguishes it from a same-named helper that means "collapse to
    ONE device for the whole run" and must refuse at P>1.

    **There must be exactly one of these objects per process, and this module
    owns it.**  That is a correctness requirement, not tidiness: the
    shape-keyed jit caches in ``common.wfn_transforms`` (``to_box`` and
    friends) key on the output ``NamedSharding``, which embeds the mesh
    OBJECT, so a second equal-but-distinct 1x1 ``Mesh`` doubles every one of
    them.  ``common.wfn_transforms.process_local_mesh`` is an alias for this
    function — the same object, not a second implementation.

    It was defined there until 2026-07-31, and this module reached *up* into
    that ψ-transform kernel to borrow it: the cross-process service importing
    a physics module to learn where its own device is.  The identity contract
    is unchanged; only the direction of the import is.

    NOTE the difference from ``jax.devices()[:1]``, the spelling this helper
    replaces at its call sites: ``jax.devices()`` is the GLOBAL device list,
    so ``jax.devices()[0]`` is process 0's device on every rank — a mesh no
    rank but 0 can compute on.
    """
    global _PROCESS_LOCAL_MESH
    if _PROCESS_LOCAL_MESH is None:
        import jax
        import numpy as np
        from jax.sharding import Mesh
        _PROCESS_LOCAL_MESH = Mesh(
            np.asarray(jax.local_devices()[:1]).reshape(1, 1),
            axis_names=('x', 'y'))
    return _PROCESS_LOCAL_MESH


def prepare_mesh(mesh=None, *, axis_names=("x", "y"), print_fn=print):
    """:func:`resolve_mesh`, then warm every communicator the run will need.

    THE call a driver should make once, before any jit.  Nobody owned
    collective warm-up, so it was happening by luck: ``gw.gw_jax`` calls only
    ``runtime.nccl_warmup``, and under
    ``JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi`` its mesh gets clique-warmed
    *incidentally*, by whichever physics kernel fires a collective first
    (``common/zeta_projection.py:422,511,547,596,864``;
    ``common/contract_bands.py:542``).  That works only while those early
    programs stay small enough for XLA's SEQUENTIAL thunk executor; a larger
    one takes the parallel executor and lands on the very refusal that killed
    the BSE TDA Lanczos outright (32 refusals at P=16, gate 7881216).  A
    physics kernel performing distributed-runtime initialisation is the
    layering inverted; this function is where it belongs.

    THE TWO WARM-UPS ARE NOT MERGED, AND MUST NOT BE.  Only the CALL SITE is
    unified here.

    * :func:`warm_mesh_cliques` (CPU/MPI) is correct *because* its jit is
      small enough that XLA takes ``ThunkExecutor::ExecuteSequential`` and
      runs the thunk inline on the calling thread — that inlining IS the
      mechanism, not an implementation detail.
    * ``runtime.nccl_warmup`` (GPU/NCCL) deliberately routes through
      ``jax.jit(jnp.sum)`` on a sharded array, because ``lax.psum`` is not
      callable from a top-level jit, and its purpose is to force
      ``ncclCommInitRank`` topology discovery, which has no thread constraint.

    A merged body would have to satisfy both, and the CPU constraint bounds
    program size in a way the GPU one does not need.

    Both are no-ops off their platform, at P=1, and on an already-warmed mesh,
    so this is safe to call unconditionally.  It IS a collective: call it
    synchronously on every rank.
    """
    from common import timing

    mesh = resolve_mesh(mesh, axis_names=axis_names)
    if process_count() > 1:
        with timing.section("collective_warmup"):
            warm_mesh_cliques(mesh, print_fn=print_fn)
            from runtime import nccl_warmup
            nccl_warmup(mesh)
    return mesh


def shard_over_k(arr, mesh, *, axes=None):
    """Constrain ``arr`` to be split along its LEADING axis over the mesh.

    The k-major sharding constraint every ψ consumer applies:  ``arr`` is
    ``(nk, …)``, the k axis rides the flattened mesh axes and every other axis
    is replicated.  ``P(('x','y'), …)`` flattens row-major, so the mesh's
    factorisation changes only how a 2-D spec would split — not which device
    holds which k.

    Exists so a physics driver never has to name ``NamedSharding`` or
    ``PartitionSpec`` to say "one k per device" (request R2 from
    ``psp/_dist.py``).
    """
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P

    ax = tuple(mesh.axis_names) if axes is None else tuple(axes)
    spec = P(ax, *((None,) * (int(arr.ndim) - 1)))
    return jax.lax.with_sharding_constraint(arr, NamedSharding(mesh, spec))


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


def replicate_to_mesh(host_array, mesh):
    """A host array every rank already holds → a REPLICATED global array.

    The fully-replicated case of :func:`device_put_process_local`, and the
    third of the three call sites its header names.  No communication: each
    rank declares its identical copy as the replica on its own device.  Use
    it on anything produced by :func:`psum_replicate` or
    :func:`gather_indexed_blocks` before mixing it with a driver's global
    arrays — ``jnp.asarray`` would make a *single-device* array instead,
    which is indistinguishable from the right thing at P=1 and an
    operand-sharding error at P>1.

    Correctness precondition: the input must be bit-identical on every rank.
    Both producers above satisfy that by construction (a psum and an
    all-gather are rank-invariant).
    """
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P

    if process_count() <= 1:
        return jnp.asarray(host_array)
    spec = P(*([None] * len(np.shape(host_array))))
    return device_put_process_local(host_array, NamedSharding(mesh, spec))


# ---------------------------------------------------------------------------
# Reductions and gathers over PROCESSES
# ---------------------------------------------------------------------------


def psum_replicate(local_np, mesh):
    """Per-rank partial → the summed whole, on every rank.  ONE psum.

    ``local_np`` is this rank's partial (any shape); the return is the
    element-wise sum over ranks, as a host array identical on all of them.
    Cost is exactly one all-reduce of ``local_np.nbytes``.

    Mechanism: the per-rank partials are declared as the shards of a
    ``(world, *shape)`` global array via
    ``make_array_from_single_device_arrays`` (a metadata operation — no data
    moves), then a ``shard_map`` body whose only op is ``lax.psum`` reduces
    them.  Spelling it as an explicit psum instead of ``jnp.sum(axis=0)``
    keeps the optimized HLO to a single ``all-reduce`` with no
    reduce-scatter/all-gather pair, which is what the communication audit
    checks.

    P=1 short-circuits to the identity: no mesh, no collective, and therefore
    bit-identical to a serial accumulation.

    Reading a timer wrapped around this: a collective cannot complete before
    the LAST rank arrives, so a rank-0 measurement is (rank-0 idle waiting for
    the slowest peer) + (the actual reduce, microseconds).  It is a
    load-imbalance meter, not a bandwidth meter — do not read a large value as
    "the network is slow".
    """
    import jax
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P

    world = process_count()
    if world <= 1:
        return np.asarray(local_np)
    if int(jax.local_device_count()) != 1:
        raise RuntimeError(
            f"psum_replicate assumes one addressable device per process; "
            f"this process has {int(jax.local_device_count())}.")
    if int(mesh.devices.size) != world:
        raise RuntimeError(
            f"mesh has {int(mesh.devices.size)} devices but there are "
            f"{world} processes; they must match for a process-partial psum.")

    shape = tuple(int(s) for s in np.shape(local_np))
    sharding = NamedSharding(mesh, P(tuple(mesh.axis_names),
                                     *((None,) * len(shape))))
    local_dev = jax.device_put(np.asarray(local_np)[None],
                               jax.local_devices()[0])
    g = jax.make_array_from_single_device_arrays(
        (world,) + shape, sharding, [local_dev])
    return np.asarray(_psum_kernel(len(shape), mesh)(g))


_PSUM_KERNELS: dict = {}


def _psum_kernel(ndim: int, mesh):
    """Cached ``jit(shard_map(psum))`` — one compile per (rank, mesh).

    Built once and memoised because the QSGW loop calls the V_H kernel every
    SC iteration; a fresh closure per call would retrace and recompile the
    reduction each time, and per-rank compiles are the documented dominant
    startup cost of this code base.

    ``id(mesh)`` is safe as a key here (and everywhere else in the tree that
    uses it — isdf/core's five kernel caches do the same): the CACHED VALUE
    closes over ``mesh`` through the shard_map, so the Mesh object cannot be
    collected while its entry lives and its id cannot be recycled underneath
    us.  The failure mode is therefore a redundant compile on a second, equal
    Mesh object — never a stale hit.  Use ``ffi.slate.batched._mesh_key``
    instead wherever the cached value does NOT retain the mesh.
    """
    from functools import partial

    import jax
    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P

    key = (int(ndim), id(mesh))
    fn = _PSUM_KERNELS.get(key)
    if fn is None:
        rest = (None,) * int(ndim)
        axes = tuple(mesh.axis_names)

        @partial(shard_map, mesh=mesh, in_specs=(P(axes, *rest),),
                 out_specs=P(*rest), check_rep=False)
        def _body(x):
            return jax.lax.psum(x[0], axes)

        fn = jax.jit(_body)
        _PSUM_KERNELS[key] = fn
    return fn


def all_gather_processes(x, *, tiled: bool = False):
    """Every process's ``x`` → the ``(world, *x.shape)`` host array on all.

    THE wrapper for ``jax.experimental.multihost_utils.process_allgather``,
    which six sites in this tree hand-rolled with their own ``device_put`` and
    their own ``np.asarray``.  Two things it fixes for all of them:

    * the operand is placed on **this process's own** device
      (``jax.local_devices()[0]``), never ``jax.devices()[0]``;
    * P=1 never touches jax at all, so a single-process run of a distributed
      routine is bit-identical to the serial one and cannot be perturbed by a
      backend that is not up yet.

    ``tiled=False`` stacks (leading axis = process index); ``tiled=True``
    concatenates along axis 0, as in the underlying call.
    """
    import numpy as np

    if process_count() <= 1:
        arr = np.asarray(x)
        return arr if tiled else arr[None]
    import jax
    from jax.experimental import multihost_utils as _mh

    if not isinstance(x, jax.Array):
        x = jax.device_put(np.asarray(x), jax.local_devices()[0])
    return np.asarray(_mh.process_allgather(x, tiled=tiled))


def gather_to_host(x):
    """ONE global ``jax.Array`` → the full host array, on every process.

    The OTHER gather, and keeping it apart from :func:`all_gather_processes`
    is the whole point:

    * :func:`all_gather_processes` — every process holds a DIFFERENT
      process-local value; stack or concatenate them.
    * this one — there is a single logical array, and the question is only
      whether this process happens to hold all of it.

    Getting that branch wrong is SILENT.  ``jax.device_get`` raises on an
    array whose shards live on other processes, so that direction is loud.
    The other direction is not: ``process_allgather(tiled=True)`` on a
    FULLY-ADDRESSABLE (replicated, or single-process) array concatenates each
    process's complete copy and MULTIPLIES THE LEADING AXIS BY P — e.g.
    ``eps_c (144, ·) → (2304, ·)`` at P=16, a plausible array of the wrong
    shape (``bse/exciton_bands.py:117-119`` documents having been bitten).

    ``is_fully_addressable`` is a GLOBAL property — identical on every
    process — so the branch keeps the collective in lockstep.  Five wrapper
    functions in this tree hand-roll this, four of them naming another in
    their own docstring, and they disagree on ``tiled``; this is the one they
    should call.

    THREE ARMS, NOT TWO — and the middle one is the whole point.
    ``is_fully_replicated`` and ``is_fully_addressable`` are DIFFERENT
    predicates and a two-arm version gets replicated arrays wrong.  jax's own
    ``Array`` docstring says it: *"fully replicated is not equal to fully
    addressable; a fully replicated array can span multiple hosts."*  A
    ``P()``-sharded array at P=64 has 64 devices in its sharding and 1
    addressable, so ``is_fully_addressable`` is **False** — measured, in this
    tree, by ``bse/exciton_bands.py``'s own log line:

        [dist] evs sharding=NamedSharding(mesh=Mesh('x': 8, 'y': 8),
               spec=P(), ...), fully_addressable=False        (job 7882507)

    A two-arm version therefore pushed every replicated array through a
    collective it does not need.  That is wasted communication at best, and at
    worst a refusal: under ``JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi`` it took
    down the ``exciton_bands`` on-grid gate on every cell of jobs 7882523,
    7882531 and 7882555 at P=16 with "Communicator requested from a thread
    that is not the one MPI was initialized from".

    When the array is fully replicated, this process ALREADY holds all of it:
    read shard 0 locally and issue nothing.  ``file_io._slab_io_allgather``
    has carried this arm and its rationale all along
    (``PROCESS_ALLGATHER_DESIGN_REVIEW_2026-05-20.md``); the service did not,
    which is why it was the least correct of the five wrappers it was meant to
    replace.  Strictly LESS communication than before in every variable.
    """
    import numpy as np

    import jax

    if getattr(x, "is_fully_addressable", True):
        return np.asarray(jax.device_get(x))
    if getattr(x, "is_fully_replicated", False):
        return np.asarray(x.addressable_data(0))
    from jax.experimental import multihost_utils as _mh
    return np.asarray(_mh.process_allgather(x, tiled=True))


def gather_indexed_blocks(local_vals, local_idx, n_total: int):
    """Per-rank ``(blk, …)`` blocks + their global indices → the whole array.

    Each rank passes the values it computed and, alongside them, the global
    slot each one belongs in (``-1`` marks a padding slot on ranks that drew
    fewer items).  Carrying the index in the payload is what frees the work
    assignment from having to be contiguous — see :func:`local_share` — and
    removes any dependence on the process ordering of the gather.

    ONE all-gather of ``world · blk · prod(item_shape)`` elements.  For
    ⟨mk|V_H|nk⟩ that is nk·nb²·16 B ≈ 15 MB at 80 bands / 33 MB at 120 bands
    (MoS₂ 12×12) — deliberately accepted rather than engineered away: it is
    3 % of one matrix-element sweep's flops-equivalent time, it happens once
    per invocation, and it hands every rank the object both consumers want (a
    file write on rank 0; a replicated operand for a driver).
    """
    import numpy as np

    vals = np.asarray(local_vals)
    out = np.zeros((int(n_total),) + tuple(int(s) for s in vals.shape[1:]),
                   dtype=vals.dtype)
    vals_g = all_gather_processes(vals)
    idx_g = all_gather_processes(np.asarray(local_idx, dtype=np.int32))
    for r in range(idx_g.shape[0]):
        for j in range(idx_g.shape[1]):
            ik = int(idx_g[r, j])
            if ik >= 0:
                out[ik] = vals_g[r, j]
    return out


# ---------------------------------------------------------------------------
# The k-partitioned sweep
# ---------------------------------------------------------------------------
#
# Why the work assignment is round-robin (``seq[rank::world]``) rather than
# contiguous blocks: the item count need not divide P (the cohsex fixture has
# nk=9), and contiguous blocking would idle whole ranks — nk=9 at P=4 gives
# 3/3/3/0 contiguous but 3/2/2/2 round-robin.  :func:`gather_indexed_blocks`
# carries each item's index with it, so the assignment is free to be any
# permutation.

#: Host-ahead-of-device depth for :func:`sweep_local_k`.  Named for the
#: driver it was first measured on, and KEPT at that name because the live
#: D10 harness (``wk_REL/d10_inner.sh``) sets it to run the un-pipelined
#: control cell.
SWEEP_LOOKAHEAD_ENV = "LORRAX_KIN_ION_LOOKAHEAD"

_DEFAULT_LOOKAHEAD = 2


def sweep_lookahead() -> int:
    """The resolved :func:`sweep_local_k` lookahead depth (>= 1, default 2).

    ``1`` restores the serialised behaviour (no read/compute overlap) and
    exists so the overlap's contribution can be MEASURED against the
    pipelined default rather than asserted.

    Malformed input REFUSES, naming the variable — the same rule
    ``common.timing`` applies to ``LORRAX_TIMING_TRACE_DEPTH``.  The parse
    this replaces swallowed ``ValueError`` and returned the default, so
    ``LORRAX_KIN_ION_LOOKAHEAD=one`` silently ran the PIPELINED path while
    the operator believed the control cell had disabled it, and a control
    that measures the same thing twice is worse than no control.
    """
    import os

    raw = os.environ.get(SWEEP_LOOKAHEAD_ENV, "").strip()
    if not raw:
        return _DEFAULT_LOOKAHEAD
    try:
        depth = int(raw)
    except ValueError:
        depth = 0                    # falls into the refusal below, with raw
    if depth < 1:
        raise ValueError(
            f"{SWEEP_LOOKAHEAD_ENV}={raw!r} is not a usable depth.  It is how "
            f"many k the host may run ahead of the device in the per-k sweep: "
            f"an integer >= 1, where 1 disables the read/compute overlap "
            f"(default {_DEFAULT_LOOKAHEAD}).")
    return depth


def local_share(items):
    """This process's round-robin share of ``items`` (see the note above)."""
    rank, world = process_rank_world()
    return list(items)[rank::world]


def sweep_local_k(ks_local, blk: int, item_shape, per_k, *,
                  lookahead: int | None = None,
                  print_fn=print, label: str = "k"):
    """Run ``per_k(ik)`` over this rank's local k — **ONE host readback**.

    ``per_k`` returns k's ``item_shape`` block (``(nb, nb)`` for every caller
    today) as a **device** array and must not force it.  The blocks are
    collected lazily and copied to the host in a single transfer at the end,
    so a sweep over ``n`` local k costs ONE device→host copy instead of ``n``
    blocking ones.  Returns ``(vals, idx)`` in exactly the layout
    :func:`gather_indexed_blocks` consumes: ``vals`` is ``(blk, *item_shape)``
    with unused rows zero, ``idx`` is ``(blk,)`` with ``-1`` in the unused
    slots.

    ``item_shape`` is REQUIRED rather than inferred from the first block
    because a rank can legitimately draw NO k: ``world > nk`` (P=64 on a 16-k
    deck) leaves ``world - nk`` ranks with an empty list, and the gather is a
    collective — every rank must contribute the same shape or it
    mis-assembles.  An empty rank has no block to infer from.

    WHY THE PER-K READBACK WAS THE COST, not the transfer volume.  Each
    ``np.asarray(H_k)`` is a full host synchronisation: it drains the device
    queue, so the *next* k's ψ read (h5py, host-side, hundreds of ms) could
    not start until the *current* k's FFTs had finished.  The two are
    independent — a k block of ⟨mk|O|nk⟩ touches no other k — and deferring
    the copy lets them overlap.  This is only legal once every k presents the
    SAME operand shapes; with a ragged ``ngk`` each k was a distinct XLA
    executable and the loop was a sequence of unrelated dispatches (owner
    decision D10).

    ``lookahead`` bounds how far the host may run ahead of the device.  It
    matters because the operand being held in flight is the ψ FFT box —
    ``nb·nspinor·nx·ny·nz·16`` B, 189 MB at MoS₂ 4×4/256 bands — and an
    unbounded queue would hold one per pending k.  ``2`` is the smallest value
    that still overlaps read *k+1* with compute *k*; it caps the extra
    residency at one box.  Note ``block_until_ready`` is a synchronisation,
    NOT a transfer, so the one-readback contract holds for any lookahead.
    """
    import jax.numpy as jnp
    import numpy as np

    ks_local = [int(v) for v in ks_local]
    blk = int(blk)
    shape = (blk,) + tuple(int(s) for s in item_shape)
    depth = sweep_lookahead() if lookahead is None else max(1, int(lookahead))
    blocks: list = []
    for j, ik in enumerate(ks_local):
        if j % 16 == 0:
            print_fn(f"    {label}: local {j + 1}/{len(ks_local)} "
                     f"(global k={ik + 1})...")
        blocks.append(per_k(ik))
        if j >= depth:
            blocks[j - depth].block_until_ready()

    vals = np.zeros(shape, dtype=np.complex128)
    if blocks:
        stacked = np.asarray(jnp.stack(blocks))          # <-- the ONE readback
        if stacked.shape[1:] != shape[1:]:
            raise ValueError(
                f"{label}: per_k returned {stacked.shape[1:]} but "
                f"item_shape={shape[1:]} was declared; the gather is a "
                f"collective and would mis-assemble.")
        vals[: stacked.shape[0]] = stacked
    idx = np.full((blk,), -1, dtype=np.int32)
    idx[: len(ks_local)] = np.asarray(ks_local, dtype=np.int32)
    return vals, idx


def gather_k_blocks(n_k: int, per_k, *, item_shape, label: str,
                    lookahead: int | None = None, print_fn=print):
    """``per_k(ik)`` for EVERY ``ik`` in ``range(n_k)`` → the whole array.

    The entire k-partition contract in one call, and the only one a physics
    driver should need: the work is spread round-robin over the processes,
    each rank's share is swept behind a single host readback
    (:func:`sweep_local_k`), and the result is returned as an
    ``(n_k, *item_shape)`` host array **identical on every rank**
    (:func:`gather_indexed_blocks`).  No reduction is involved anywhere — a k
    block of ⟨mk|O|nk⟩ is independent of every other, so this is
    embarrassingly parallel with one indexed gather at the end.

    ``per_k`` must return a device array of ``item_shape`` and must not force
    it.  ``label`` names the two timing sections (``{label}_k`` and
    ``{label}_gather``) and the progress lines, so a driver gets its sweep and
    its gather in the timing table without opening a section by hand.
    """
    from common import timing

    n_k = int(n_k)
    world = process_count()
    ks_local = local_share(range(n_k))
    print_fn(f"\nProcessing {n_k} k-points over P={world} rank"
             f"{'s' if world != 1 else ''} ({len(ks_local)} on this rank)...")
    with timing.section(f"{label}_k"):
        vals, idx = sweep_local_k(ks_local, max(1, -(-n_k // world)),
                                  item_shape, per_k, lookahead=lookahead,
                                  print_fn=print_fn, label=label)
    with timing.section(f"{label}_gather"):
        return gather_indexed_blocks(vals, idx, n_k)


# ---------------------------------------------------------------------------
# The checked reduce-scatter
# ---------------------------------------------------------------------------
#
# WHY
# ---
# ``jax.lax.psum_scatter`` under ``JAX_CPU_COLLECTIVES_IMPLEMENTATION=gloo``
# silently returns wrong data in ~5 % of executions — always output segment 0,
# always a plausible magnitude, always with a zero exit code
# (``wk_REL/UPSTREAM_gloo_psum_scatter_corruption.md``).  The transport verdict
# is that mpi is clean in 584/584, so this is not a live fire; it is the
# durable answer to "what makes the NEXT one detectable".  There are fifteen
# reduce-scatter sites in this codebase and none of them carries an invariant.
# This function is the ONE place that changes for all fifteen — no per-call-site
# carve-outs, one primitive, one tolerance, one message.
#
# THE CHECK — Freivalds, not a plain sum
# --------------------------------------
# A reduce-scatter over a replica group of P ranks satisfies, by definition,
#
#     concat_r out_r  ==  sum_p x_p
#
# so for ANY weight tensor ``w`` the two weighted sums must agree:
#
#     sum_{global j} w_j (sum_p x_p)_j  ==  sum_r sum_{local j} w_{rL+j} out_r[j]
#
# A PLAIN sum would not do: it is invariant under moving mass between segments,
# and it cancels for a zero-mean perturbation — which is exactly the observed
# failure shape (a stale / dropped / duplicated SEGMENT, §4.6a).  With an
# incoherent ``w``, a corrupted segment escapes only on a measure-zero
# coincidence.
#
# Both sides are built from ``psum``, which the corruption study established as
# the clean path: the all-reduce-only formulation of the same reduction was
# 200/200 clean in the same harness that corrupted the reduce-scatter.  The
# checker therefore does not rest on the primitive it is checking.
#
# ``w`` must NOT be materialised — it would be as large as the payload (253 MB
# at the corrupting shape) — so it is RANK-1 FACTORISED,
# ``w[i0,i1,…] = prod_d v_d[i_d]``.  The contraction is then a sequence of
# axis-wise matrix-vector products: ONE pass over the operand, O(1) extra
# memory, and no large intermediate (the "no replicated large intermediates"
# rule is respected by construction).
#
# The two checksums ride ONE scalar all-reduce, not two: only
# ``|psum(s_in − s_out)|`` is needed, so the difference is formed locally and a
# single 2-element ``psum`` carries it together with the scale.
#
# COST — measured, wk_REL/collective_invariants_notes.md §2.3.  In the form
# below (two contractions, no separate scale pass) it is +0.7 % on the 0.41 MB
# BSE collective under mpi and +1.3 % under gloo — free at the payload BSE
# actually issues.  It is NOT free at Sigma-class payloads: +107 % at 41 MB and
# +56 % at 253 MB under mpi, because the contraction is a memory-bandwidth
# sweep over the operand while the mpi reduce-scatter of the same operand is
# fast.  Under gloo everything is under +5 %.  Read §2.3-2.4 before wiring it
# anywhere; §2.4 prices the sub-sampling lever that brings the large payloads
# down.
#
# WIRING STATUS: no call site uses this yet.  Ten of the fifteen reduce-scatter
# sites live in ``contract_bands.py`` and ``zeta_projection.py``, which the
# all-MPI migration workstream owns.  Wiring is an owner decision on the
# measured cost, and it should land as ONE change across all fifteen.

COLLECTIVE_RTOL = 1e-10

# Six mutually-incommensurate multipliers, one per tensor axis, so the
# per-axis weight sequences are independent.  Values are irrelevant beyond
# being irrational-ish and distinct; determinism is what matters.
_W_MULT = (1.6180339887498949, 1.3247179572447460, 1.2207440846057595,
           1.1673039782614187, 1.1347241384015194, 1.1127756842787055)


def _weight_vec(length: int, axis_pos: int):
    """Deterministic unit-modulus weights for one axis of the checksum.

    A pure function of ``(length, axis_pos)``, so every rank in the replica
    group produces identical weights with **no communication**, and a failure
    reproduces exactly.  Unit modulus keeps the checksum's dynamic range equal
    to the data's — no amplification, and no way for a large-magnitude weight
    to hide a small-magnitude corrupted segment.
    """
    import jax.numpy as jnp
    import numpy as np

    i = jnp.arange(int(length), dtype=jnp.float64)
    m = _W_MULT[axis_pos % len(_W_MULT)]
    return jnp.exp(2j * np.pi * jnp.mod(i * m, 1.0))


def _weighted_sum(t, glob_len, offsets):
    """``sum_idx (prod_d v_d[idx_d + offsets_d]) * t[idx]`` in ONE pass.

    ``glob_len[d]`` is the GLOBAL extent of axis ``d`` and must be a Python
    int; the weight vector is generated at global length (so every rank agrees
    on the weights) and then ``dynamic_slice``d at this rank's offset, which
    MAY be a tracer.  Generating it at ``t.shape[d] + offset`` instead puts a
    tracer inside ``jnp.arange`` and raises ``ConcretizationTypeError``.

    Axes are contracted one at a time, so the working set shrinks
    monotonically and the total flop count is ~ ``t.size``.
    """
    import jax.numpy as jnp
    from jax import lax

    shp = tuple(int(v) for v in t.shape)
    out = t
    for d in range(len(shp)):
        v = _weight_vec(int(glob_len[d]), d).astype(t.dtype)
        v = lax.dynamic_slice(v, (offsets[d],), (shp[d],))
        out = jnp.tensordot(out, v, axes=([0], [0]))
    return out


def psum_scatter_checked(x, axis, scatter_dimension, tiled=True, *, name):
    """``lax.psum_scatter`` carrying a Freivalds checksum of its own output.

    Drop-in for ``jax.lax.psum_scatter`` except that it returns a PAIR.

    Parameters
    ----------
    x
        Local partial, FULL extent along ``scatter_dimension``.
    axis
        Mesh axis name to reduce over (the replica group).
    scatter_dimension
        Tensor axis to tile the result over, as in ``lax.psum_scatter``.
    tiled
        As in ``lax.psum_scatter``.  ``tiled=False`` is not supported: the
        segment→global index map below assumes the contiguous tiled layout.
    name
        Label used in the failure message.  Must identify the call site.

    Returns
    -------
    (out, residual)
        ``out`` is exactly what ``lax.psum_scatter`` would have returned.
        ``residual`` is a REPLICATED real scalar,
        ``|Σ w·in − Σ w·out| / Σ |w·in|``, at round-off when the collective is
        correct.  You cannot raise inside ``shard_map``, so the caller must
        accumulate ``residual`` (``jnp.maximum``) into a small device buffer
        and hand it to :func:`report_collective_residual` at a coarse boundary
        — per τ node, per Lanczos iteration — where a host value is affordable.
    """
    import jax.numpy as jnp
    from jax import lax

    if not tiled:
        raise ValueError(
            f"psum_scatter_checked({name!r}): tiled=False is not supported — "
            f"the checksum's segment→global index map assumes the contiguous "
            f"tiled layout.  Use lax.psum_scatter directly, unchecked, and say "
            f"so at the call site.")

    out = lax.psum_scatter(x, axis, scatter_dimension=scatter_dimension,
                           tiled=tiled)

    glob = [int(v) for v in x.shape]           # x carries the FULL extent
    seg = int(out.shape[scatter_dimension])
    off_in = [0] * x.ndim
    off_out = [0] * out.ndim
    off_out[scatter_dimension] = lax.axis_index(axis) * seg

    s_in = _weighted_sum(x, glob, off_in)
    s_out = _weighted_sum(out, glob, off_out)

    # ONE scalar all-reduce, 2 complex doubles.  psum, never psum_scatter —
    # the checker must not be built on the primitive it is checking.
    packed = lax.psum(jnp.stack([s_in - s_out, s_in]), axis)
    # SCALE = |checksum of the correct reduction|, NOT the operand's L1 mass.
    # Measured (wk_REL job 7880840): a separate L1-mass pass costs +65 % on the
    # 0.41 MB BSE collective and +157 % at 41 MB, because `abs(x)` materialises
    # a second full-size array — it was the ENTIRE overhead at the BSE payload
    # (+65 % with it, +0.7 % without).  Using |s_in| is free, and it is also
    # the better-conditioned choice: numerator and denominator are then the
    # SAME kind of quantity (a length-N random-walk sum of unit-modulus-weighted
    # terms), so the ratio is a pure relative error with no sqrt(N) offset
    # between them.  Measured floor at the 253 MB payload: 4.7e-13 this way vs
    # 2.3e-11 with the extra pass.
    denom = jnp.maximum(jnp.abs(packed[1]), jnp.finfo(jnp.float64).tiny)
    return out, (jnp.abs(packed[0]) / denom).astype(jnp.float64)


def report_collective_residual(name: str, residual, *,
                               rtol: float = COLLECTIVE_RTOL,
                               print_fn: Callable[..., Any] = print) -> None:
    """Coarse-boundary check of an accumulated :func:`psum_scatter_checked`
    residual.  Traced-safe: one unordered ``jax.debug.callback``, no sync.

    Tolerance — derived, then measured (same shape of argument as
    ``solvers.lanczos.ALPHA_HERM_RTOL``).  Numerator and denominator are both
    length-``N`` sums of unit-modulus-weighted terms, so the ratio is a pure
    relative error whose round-off floor is ``~sqrt(N)·u`` with pairwise
    summation (``u = 1.11e-16``).  MEASURED clean floors (wk_REL job 7880840,
    P=4, c128): **3.6e-14** at 0.41 MB, **2.0e-13** at 41 MB, **4.7e-13** at
    253 MB — i.e. the floor grows like sqrt(payload), as predicted.

    MEASURED sensitivity to a segment-0 fault of relative size ``rel``
    (job 7880905): residual ``= 0.53 x rel`` at 0.41 MB and ``= 1.12 x rel``
    at 41 MB — essentially 1:1, because the fault and the checksum live on the
    same scale.

    ``1e-10`` therefore sits 200-2800x above the measured floor and detects a
    fault of relative size ``~2e-10``.  The corruption it exists to catch is
    ``rel ~ 0.24`` (upstream S3), a margin of ``~1e9``.  Re-derive if a payload
    much larger than 253 MB appears: the floor scales as sqrt(N).
    """
    import jax
    import numpy as np

    def _cb(r):
        from common import sanity
        r = float(np.asarray(r))
        if not np.isfinite(r) or r > float(rtol):
            sanity.warn(
                f"{name}: reduce-scatter checksum residual {r:.3e} > "
                f"{float(rtol):.1e}.  A Freivalds weighted checksum of a "
                f"psum_scatter cannot disagree with its own input unless the "
                f"COLLECTIVE returned data that is not the reduction it was "
                f"asked for.  Known cause on this stack: "
                f"JAX_CPU_COLLECTIVES_IMPLEMENTATION=gloo corrupts output "
                f"segment 0 in ~5% of executions (see "
                f"wk_REL/UPSTREAM_gloo_psum_scatter_corruption.md); mpi is "
                f"clean in 584/584.  Every number downstream of this "
                f"collective is suspect.",
                print_fn=print_fn)

    jax.debug.callback(_cb, residual)


# ---------------------------------------------------------------------------
# impl=mpi mesh-clique warm-up
# ---------------------------------------------------------------------------

_WARMED_MESHES: set = set()


def warm_mesh_cliques(mesh, *, print_fn=print) -> float:
    """Create every mesh-axis MPI communicator on the calling (main) thread.

    THE PROBLEM.  jaxlib 0.9.1's
    ``xla::cpu::MpiCollectives::CreateCommunicators`` refuses with ``MPI:
    Communicator requested from a thread that is not the one MPI was
    initialized from`` unless ``MPI_Is_thread_main`` is true, and XLA:CPU's
    PARALLEL ``ThunkExecutor`` issues collective thunks from intra-op pool
    workers.  So under ``JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi`` any clique
    whose FIRST use is inside a real jitted program dies on every rank.  This
    is what killed the BSE TDA Lanczos (job 7879458).

    THE MECHANISM THAT FIXES IT.  The guard fires only on communicator
    CREATION, and ``xla::cpu::AcquireCommunicator`` caches communicators in a
    process-global clique map keyed *only* by the participating-device set:
    it takes the map lock and calls ``CreateCommunicator`` only on a MISS.
    Creating each clique once here — from the main thread, inside a jit small
    enough (one 8-byte buffer, <= 8 thunks) that XLA takes
    ``ThunkExecutor::ExecuteSequential`` and runs the thunk inline on the
    caller — makes every later acquisition a cache hit, so the guard is never
    evaluated again.  Verified live with ``TF_CPP_VMODULE=cpu_cliques=3``:
    three cliques per rank, created during warm-up, never re-created.

    WHICH CLIQUES.  Exactly the ones the program will touch: one per mesh
    axis, plus one over all axes together.  This is **per-clique**, and the
    discriminating controls (job 7881053) are unambiguous — warming the world
    clique ALONE fails, ``x`` alone fails, ``x+y`` without the world fails,
    and only ``x + y + world`` passes.  An earlier helper in
    ``common.contract_bands`` warmed the world clique only, which is exactly
    the failing cell; that is why the "world-collective-first contract" looked
    falsified.  Warm-up does matter — it was warming the wrong device sets.

    COST.  Three 8-byte ``psum``s for a 2-D mesh; ~150 ms once per process at
    P=4, ``O(log P)`` in latency, independent of ``N_mu``/``N_k``/``N_q``.  It
    adds nothing to any solver iteration and leaves the compiled HLO of every
    downstream kernel untouched, so it cannot move the collective table or the
    allocation table.

    WHY NOT THE WRAPPER OVERRIDE.  ``LORRAX_MPI_FORCE_THREAD_MAIN`` also
    works, but it is a patched MPI shim every user must build, and it is
    strictly less safe: it lets XLA call ``MPI_Comm_split`` — which is
    collective over ``MPI_COMM_WORLD`` — from arbitrary pool workers.
    ``AcquireCommunicator`` serialises creation *within* a process but nothing
    serialises it *across* ranks, so two cliques becoming ready in different
    orders on different ranks is a latent deadlock.  Warming from one thread
    in a program-defined order before any jit runs removes that exposure.
    (The wrapper's OTHER override, the ``MPI_THREAD_MULTIPLE`` upgrade, is
    still required and is unrelated — see docs/dev/mpi_collectives.md.)

    MUST be called synchronously on every rank: it is a collective.  A no-op
    off the mpi implementation, in single-process runs, and on a mesh already
    warmed in this process.  Returns the elapsed seconds (0.0 when a no-op).
    """
    import os
    if os.environ.get(
            "JAX_CPU_COLLECTIVES_IMPLEMENTATION", "").strip().lower() != "mpi":
        return 0.0
    import time
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.sharding import PartitionSpec as P
    from jax.experimental.shard_map import shard_map
    if jax.process_count() <= 1:
        return 0.0
    key = (tuple(int(d.id) for d in mesh.devices.ravel()),
           tuple(mesh.axis_names))
    if key in _WARMED_MESHES:
        return 0.0
    t0 = time.perf_counter()
    tiny = jnp.zeros(1)
    axes = list(mesh.axis_names)
    groups = list(axes) + ([tuple(axes)] if len(axes) > 1 else [])
    for ax in groups:
        f = jax.jit(shard_map(lambda a, ax=ax: lax.psum(a, ax), mesh=mesh,
                              in_specs=(P(None),), out_specs=P(None),
                              check_rep=False))
        jax.block_until_ready(f(tiny))
    dt = time.perf_counter() - t0
    # Mark AFTER the collectives succeed: marking first would skip the warm-up
    # forever if one raised, and the next caller would hit the refusal with no
    # indication why.  (Same ordering rule the old contract_bands helper used.)
    _WARMED_MESHES.add(key)
    if jax.process_index() == 0:
        print_fn(f"[collectives] warmed {len(groups)} MPI cliques for mesh "
                 f"{tuple(mesh.devices.shape)} axes={axes} in {dt*1e3:.0f} ms")
    return dt

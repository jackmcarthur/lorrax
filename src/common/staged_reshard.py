"""Explicit volume-preserving staged reshard services.

``face_to_batch_reshard`` — the STAGED face→batch reshard.

The movement-only sibling of :mod:`common.contract_bands`.  Where that
module stages a *contraction* so no ``(μ, μ)``-class tile ever lands on
one rank, this one stages a pure **layout change** that the SPMD
partitioner cannot perform in one step and silently degrades to
replicate-then-repartition::

    (B, M, N)   P(None, 'x', 'y')      →     P(('x','y'), None, None)
    "face":  every rank owns the whole  "batch": every rank owns WHOLE
    batch on its own (M/px, N/py) tile   (M, N) matrices for its own B-rows

Both layouts hold exactly ``B·M·N / (px·py)`` elements per rank, so this
is volume-preserving in both directions.  XLA still refuses it: the tile
multisets are ``{1, px, py}`` and ``{px·py, 1, 1}``, which are not a
permutation of one another, so ``GetReshardAllToAllSourceTargetDims``
does not match and GSPMD takes its documented last resort — replicate
the tensor on every device, then slice.  Its own diagnostic, from the
91-Q MoS2 4x4 exciton path (job 7882974, ``rank`` 672, ``bs`` 64)::

    [SPMD] Involuntary full rematerialization.  The compiler cannot go
    from sharding {devices=[1,8,8]<=[64]} to {devices=[64,1,1]<=[64]}
    efficiently for ... c128[64,84,84] ... jit(_q_batch)/
    sharding_constraint ...  As the last resort, SPMD will replicate the
    tensor and then partition it, which is inefficient.  (b/433785288)

One warning per rank per site; the replicated intermediate is
``B·M·N·16`` bytes per device (462 MB at P=64 on that deck, 231 MB at
P=16) against a 7.2 MB true shard — a 64× per-rank residency blow-up
that is exactly the failure mode the LORRAX scaling target forbids.

WHY STAGING WORKS
-----------------
Split the move into the two single-mesh-axis exchanges it really is, and
each one IS a tile permutation::

    stage 1   P(None, 'x', 'y')  →  P('x', None, 'y')     all_to_all over 'x'
              tiles [1, px, py]  →  [px, 1, py]           split B, concat M
    stage 2   P('x', None, 'y')  →  P(('x','y'), None, None)
              tiles [px, 1, py]  →  [px·py, 1, 1]         split B, concat N

Stage 2's tile multiset still is not a permutation *of the whole mesh* —
but it is one **within each 'x' group**, which is precisely what
``lax.all_to_all(..., 'y', ...)`` expresses and what the partitioner
could not infer.  Issuing both exchanges by hand inside ONE ``shard_map``
(``check_vma=False``) is therefore not an optimisation hint but a
structural guarantee: the partitioner cannot hoist, merge or re-plan
collectives written inside a manual-axis region (the per_q lesson,
QUALITY_PATTERNS §4; the same reason
``contract_bands_block_reshard`` puts its psum_scatters there).

PER-RANK RESIDENCY (the owner's hard constraint)
------------------------------------------------
Every intermediate in the chain is exactly one shard::

    in       (B,        M/px, N/py)   =  B·M·N/(px·py)
    stage 1  (B/px,     M,    N/py)   =  B·M·N/(px·py)
    stage 2  (B/(px·py), M,   N)      =  B·M·N/(px·py)

so the count of live shard-class intermediates is 2 at each collective
(source + destination) and never more — the same count the unstaged form
already had, with the *replicated* member of that pair removed.  Peak
per-rank residency for this array therefore FALLS by ``px·py``; it
cannot rise.  No staging buffer is introduced.

TWO ROUTES THROUGH THE SAME MOVE  (``route=``)
---------------------------------------------
There is more than one pair of exchanges that lands on the output
layout, and which one is faster is a MEASUREMENT, not a derivation.  Both
are implemented; ``route`` selects::

  "split_b_first"  (default, the shipped one)
      (B, M_x, N_y)  →  (B_x, M, N_y)  →  (B_xy, M, N)
      all_to_all ax_x  (split B, concat M)      1 exchange, p_x peers
      all_to_all ax_y  (split B, concat N)      1 exchange, p_y peers

  "flatten_m_first"  (proposed by the owner, 2026-07-31:
      "q,mu_X,nu_Y -> q,mu_XY,nu -> q_XY,mu,nu ... i think that's most
      efficient")
      (B, M_x, N_y)  →  (B, M_xy, N)  →  (B_xy, M, N)
      all_to_all ax_y            (split M, concat N)   p_y peers
      all_to_all (ax_x, ax_y)    (split B, concat M)   p_x·p_y peers

Both are volume-preserving at every step and both are pure movement, so
they are the same bit-exact parity class.  They differ in

* **divisibility.**  ``split_b_first`` needs ``M % p_x`` and ``N % p_y``
  (the input layout's own requirement) plus ``B % (p_x·p_y)``.
  ``flatten_m_first`` additionally needs ``M % (p_x·p_y)``, because its
  intermediate cuts M over the FLATTENED axis.  On the MoS2 4x4 exciton
  deck M = rank = nk·nb = 672 and 672 % 64 = 32, so at P=64 that route
  needs M padded to 704 (+4.76 %).  ``pad_m`` does exactly that, locally,
  and drops the pad rows again with a static gather at the end — see
  :func:`face_to_batch_reshard` — so the padding cost is paid by the
  route that needs it and by nothing else.
* **peer count.**  Two exchanges over p_x and p_y peers versus one over
  p_y and one over p_x·p_y.  Payload is ``(1 − 1/p)`` of a shard each
  way, so ``split_b_first`` moves 1.75 shards at 8x8 against 1.86, and
  its largest replica group is 8 rather than 64.  On this machine these
  collectives measured COUNT-bound rather than bandwidth-bound, which
  predicts ``split_b_first`` wins — a hypothesis the harness tests, not a
  reason to skip testing it.

ORDER OF THE TWO STAGES (§3.2 of the staged-reshard doctrine)
--------------------------------------------------------------
The two payloads within ``split_b_first`` are equal — ``(1 − 1/p)`` of
one shard each — so the axis order is not a payload decision as it is in
``contract_bands_block_reshard``.  It is fixed by the OUTPUT layout
instead: ``P((ax, ay), None, None)`` numbers its B-blocks ``ax``-major,
so B must be cut by ``ax`` first and by ``ay`` second, and rank
``(x, y)`` then holds block ``x·py + y`` exactly as the spec says.  The
happy consequence is that the FINAL exchange rides ``ay`` — the mesh's
minor axis, whose replica groups are consecutive ranks (node-local pairs
at 2 ranks/node on the production layout).  The module still REFUSES a
mesh whose minor axis is not ``ay``, because on such a mesh the block
numbering the caller asked for and the groups the collectives use stop
agreeing.  ``flatten_m_first`` reaches the same numbering by a different
schedule: its first exchange gives rank ``(x, y)`` M-block ``x·py + y``,
and its second hands out B over the flattened axis in the same
``ax_x``-major order, so the two routes' outputs are element-identical.

Evidence pointer for the pattern itself: ``symmetry_maps``'s
``unfold_isdf_operator`` has shipped the same ``shard_map`` +
``lax.all_to_all``
volume-preserving redistribution (with the same explicit divisibility
refusal) since the TRS/umklapp work.  This module is that idiom lifted
out of one call site and given a contract.

MEASURED, first adopter (``bandstructure.bse_setup``'s ``_q_batch``;
job 7883154 against 7882974, same deck, same args, same harness,
cache-cold, coll=mpi).  ``htransform_psi_cQ`` on the 91-Q MoS2 4x4 path::

    P=64   231.39 -> 145.53 s   1.59x    involuntary-remat lines 64 -> 0
    P=16   175.96 -> 186.13 s   0.95x    involuntary-remat lines 16 -> 0

so the P=64/P=16 ratio goes 1.32 -> 0.78.  The P=16 row is stated as
measured: at 16 devices the replicated batch this removes is only
231 MB and the two exchanges run over 4-device groups, so the trade is
close to even and came out ~6% the wrong way.  The primitive's value is
therefore claim-scoped to the regime it was built for — device counts
where ``B·M·N`` replicated per rank is the binding cost — and the
measured domain is CPU/``impl=mpi``/2 ranks per node; no GPU walltime of
this module exists.  Values: byte-identical .dat at both P.

Those two ``involuntary-remat lines N -> 0`` counts are HISTORY, not a
check that can be re-run.  They were read off Frontera's XLA in
2026-07-31, where the partitioner still printed the line quoted above.
The shipped stack no longer prints it at all — see below — so an A/B that
greps for it today reads zero on both arms.

Gate: ``tests/test_staged_reshard.py`` — value parity against the
unstaged chain, and a pin on the compiled module: exactly two
``all-to-all`` ISSUE SITES and no other collective, no per-device buffer
larger than one shard, and a temp allocation below the size of the whole
array.  Its RED TWIN puts the unstaged chain through the same assertion
helper and requires it to raise, in-process and again as a cold compile at
this docstring's own deck geometry.  An instrument that has not been shown
failing is not an instrument (wk_REL README §5.1).

That pin used to be written as a count of the string ``all-to-all(`` plus
a grep of the compiler's ``Involuntary full rematerialization`` warning,
and BOTH were vacuous on CUDA — the count because XLA:GPU spells its
collectives ``all-to-all-start``/``-done`` and read 0 of 2, the grep
because the warning is absent on this XLA on every platform and under both
partitioners even while the replication it announced is still there.  Both
were repaired GPU-first on 2026-08-10; the measurement is
``/pscratch/sd/j/jackm/reshard_instr_0810/`` and the reasoning is at the
head of each section of the gate file.
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.collectives import warm_mesh_cliques
from runtime.padding import spec_divisor

#: The two exchange schedules that land on ``P((ax_x, ax_y), None, None)``.
#: ``split_b_first`` is the default and the measured one; ``flatten_m_first``
#: is the owner's proposal (see the module docstring).
ROUTES = ("split_b_first", "flatten_m_first")
DEFAULT_ROUTE = "split_b_first"

__all__ = [
    "band_to_product_r_reshard",
    "face_to_batch_reshard",
    "face_to_batch_reshard_supported",
    "shard_local_slice_pad",
    "shard_local_update",
    "ROUTES",
    "DEFAULT_ROUTE",
]


def band_to_product_r_reshard(
    mesh: Mesh, *, axes: tuple[str, str] = ("x", "y")
) -> Callable:
    """Factory for the exact band-product → r-product wavefunction move.

    The input and output contracts are::

        (k, b, s, r)  P(None, (x,y), None, None)
                  ->  P(None, None, None, (y,x))

    This is the two-axis extension of the established ISDF band→r exchange.
    Both stages are inside one manual-axis region, so GSPMD cannot replace
    them with its replicate-then-slice fallback::

        y: split r, concat b  ->  P(None, x, None, y)
        x: split r, concat b  ->  P(None, None, None, (y,x))

    If the incoming flat band-block number is ``x*p_y + y``, the first
    exchange concatenates the ``y`` blocks for each fixed ``x`` and the
    second concatenates those groups in ``x`` order.  The final band order
    is therefore exactly the original global order.  The real-space block
    is first selected by ``y`` and then by ``x``, hence the deliberately
    reversed output tuple ``(y,x)``.  Every intermediate contains exactly
    ``k*b*s*r/(p_x*p_y)`` elements per rank: no axis is ever replicated.

    Callers must zero-pad the free r extent through :mod:`runtime.padding`
    before this service.  This routine owns movement only; it neither pads
    nor changes values.
    """
    from common.shard_map import shard_map

    ax_x, ax_y = axes
    names = tuple(mesh.axis_names)
    if ax_x not in names or ax_y not in names:
        raise ValueError(
            "band_to_product_r_reshard: mesh axes "
            f"{names} do not contain axes={axes!r}")
    p_x = int(mesh.shape[ax_x])
    p_y = int(mesh.shape[ax_y])
    in_spec = P(None, (ax_x, ax_y), None, None)
    out_spec = P(None, None, None, (ax_y, ax_x))
    band_divisor = spec_divisor(mesh, in_spec, axis=1)
    r_divisor = spec_divisor(mesh, out_spec, axis=3)

    def _body(a):
        if p_y > 1:
            a = jax.lax.all_to_all(
                a, ax_y, split_axis=3, concat_axis=1, tiled=True)
        if p_x > 1:
            a = jax.lax.all_to_all(
                a, ax_x, split_axis=3, concat_axis=1, tiled=True)
        return a

    sm = shard_map(
        _body, mesh=mesh, in_specs=(in_spec,), out_specs=out_spec,
        check_vma=False)
    compiled = jax.jit(
        sm,
        in_shardings=NamedSharding(mesh, in_spec),
        out_shardings=NamedSharding(mesh, out_spec),
    )

    def _reshard(a):
        shape = tuple(int(s) for s in a.shape)
        if len(shape) != 4:
            raise ValueError(
                "band_to_product_r_reshard expects rank-4 (k,b,s,r), "
                f"got shape {shape}")
        # Fail BEFORE the shard_map boundary: accepting a replicated or
        # r-sharded array here would let that boundary synthesize the exact
        # implicit generic reshard this service exists to forbid.  Normalize
        # PartitionSpec's elided trailing Nones by the established
        # wfn_transforms convention, then require the run's one Mesh object.
        sharding = getattr(a, "sharding", None)
        if isinstance(sharding, NamedSharding):
            actual_spec = tuple(sharding.spec)
            actual_spec += (None,) * (len(shape) - len(actual_spec))
        else:
            actual_spec = (None,) * len(shape)
        expected_spec = tuple(in_spec)
        expected_spec += (None,) * (len(shape) - len(expected_spec))
        same_mesh = (isinstance(sharding, NamedSharding)
                     and sharding.mesh is mesh)
        if actual_spec != expected_spec or not same_mesh:
            raise ValueError(
                "band_to_product_r_reshard requires an array already "
                f"committed to {in_spec} on the supplied Mesh; got "
                f"spec={P(*actual_spec)} on {type(sharding).__name__}. "
                "Refusing an implicit pre-reshard at the shard_map boundary.")
        from runtime.padding import authenticate_padded_axis
        authenticate_padded_axis(
            shape[1], shape[1], band_divisor,
            name="band-to-r band carrier")
        authenticate_padded_axis(
            shape[3], shape[3], r_divisor,
            name="band-to-r real-space carrier")
        return compiled(a)

    # Production calls enter through ``_reshard`` so the concrete array's
    # committed source layout is checked before any staged/JIT boundary.
    # HLO instrumentation must lower the already-pinned executable directly:
    # wrapping ``_reshard`` in another jit would replace ``a`` by a tracer and
    # erase the concrete ``a.sharding`` fact the guard deliberately requires.
    _reshard.lower = compiled.lower

    warm_mesh_cliques(mesh)
    return _reshard


def shard_local_slice_pad(
    mesh: Mesh,
    *,
    spec: P,
    axis: int,
    mesh_axis: str,
    local_size: int,
) -> Callable:
    """Factory for a device-local slice with an exact-zero tail.

    A normal global slice of a ``P(mesh_axis)`` axis changes the global
    extent and therefore repartitions ownership.  XLA may implement that as
    full replication followed by slicing.  This primitive instead enters a
    manual ``shard_map`` first and indexes *within each already-owned shard*.
    The output keeps ``spec`` and has ``local_size`` entries per shard on
    ``axis``.  Out-of-range tail entries are exact zeros.

    ``start`` is a replicated scalar measured in LOCAL-shard coordinates.
    The same compiled executable therefore serves every tile offset, including
    the padded tail, without a global reshard or a second padding convention.
    """
    from common.shard_map import shard_map

    size = int(local_size)
    if size <= 0:
        raise ValueError(f"shard_local_slice_pad local_size must be >0; got {size}")
    spec_t = tuple(spec)
    rank = len(spec_t)
    ax = int(axis) % rank
    spec_full = spec_t + (None,) * (rank - len(spec_t))
    if mesh_axis not in tuple(mesh.axis_names):
        raise ValueError(
            f"shard_local_slice_pad mesh has no axis {mesh_axis!r}: "
            f"{tuple(mesh.axis_names)!r}")
    if spec_full[ax] != mesh_axis:
        raise ValueError(
            "shard_local_slice_pad requires the sliced axis to be sharded "
            f"only by {mesh_axis!r}; got spec={spec}, axis={ax}")

    sh = NamedSharding(mesh, spec)
    rep = NamedSharding(mesh, P())

    def _body(a, start):
        n_local = int(a.shape[ax])
        ids = jnp.asarray(start, dtype=jnp.int32) + jnp.arange(
            size, dtype=jnp.int32)
        safe = jnp.minimum(ids, jnp.int32(n_local - 1))
        out = jnp.take(a, safe, axis=ax)
        mask_shape = [1] * a.ndim
        mask_shape[ax] = size
        valid = (ids < jnp.int32(n_local)).reshape(mask_shape)
        return jnp.where(valid, out, jnp.zeros((), dtype=a.dtype))

    sm = shard_map(
        _body, mesh=mesh, in_specs=(spec, P()), out_specs=spec,
        check_vma=False)
    compiled = jax.jit(
        sm, in_shardings=(sh, rep), out_shardings=sh)

    def _slice(a, start):
        sharding = getattr(a, "sharding", None)
        actual = tuple(sharding.spec) if isinstance(sharding, NamedSharding) else ()
        actual += (None,) * (a.ndim - len(actual))
        expected = tuple(spec) + (None,) * (a.ndim - len(tuple(spec)))
        if not isinstance(sharding, NamedSharding) \
                or sharding.mesh is not mesh or actual != expected:
            raise ValueError(
                "shard_local_slice_pad requires its input already committed "
                f"to {spec} on the supplied Mesh; got {sharding!r}")
        return compiled(a, start)

    _slice.lower = compiled.lower
    return _slice


def shard_local_update(mesh: Mesh, *, spec: P) -> Callable:
    """Factory for donated, device-local tile insertion into a sharded face.

    ``dst`` and ``tile`` share ``spec``.  ``starts`` contains one LOCAL-shard
    offset per array axis.  Out-of-range tail indices are dropped, so a
    zero-padded final tile can update a logical-sized destination without
    backward clamping.  The body is inside ``shard_map`` and donates ``dst``;
    it cannot communicate or allocate a second global face.
    """
    from common.shard_map import shard_map

    sh = NamedSharding(mesh, spec)
    rep = NamedSharding(mesh, P())

    def _body(dst, tile, starts):
        grids = jnp.ix_(*(
            jnp.asarray(starts[d], dtype=jnp.int32)
            + jnp.arange(tile.shape[d], dtype=jnp.int32)
            for d in range(dst.ndim)
        ))
        return dst.at[grids].set(tile, mode="drop")

    sm = shard_map(
        _body, mesh=mesh, in_specs=(spec, spec, P()), out_specs=spec,
        check_vma=False)
    compiled = jax.jit(
        sm, in_shardings=(sh, sh, rep), out_shardings=sh,
        donate_argnums=(0,))

    def _update(dst, tile, starts):
        for name, value in (("dst", dst), ("tile", tile)):
            sharding = getattr(value, "sharding", None)
            actual = (tuple(sharding.spec)
                      if isinstance(sharding, NamedSharding) else ())
            actual += (None,) * (value.ndim - len(actual))
            expected = tuple(spec) + (None,) * (value.ndim - len(tuple(spec)))
            if not isinstance(sharding, NamedSharding) \
                    or sharding.mesh is not mesh or actual != expected:
                raise ValueError(
                    f"shard_local_update {name} must already be committed "
                    f"to {spec} on the supplied Mesh; got {sharding!r}")
        if dst.ndim != tile.ndim or tuple(starts.shape) != (dst.ndim,):
            raise ValueError(
                "shard_local_update requires dst/tile of equal rank and "
                f"starts.shape=({dst.ndim},); got {dst.shape}, {tile.shape}, "
                f"{tuple(starts.shape)}")
        return compiled(dst, tile, starts)

    _update.lower = compiled.lower
    return _update


def face_to_batch_reshard_supported(mesh: Mesh, shape, *,
                                    axes: tuple[str, str] = ("x", "y"),
                                    route: str = DEFAULT_ROUTE) -> bool:
    """True when :func:`face_to_batch_reshard` can serve ``shape`` on ``mesh``.

    Pure arithmetic — no JAX call, no collective — so a caller can branch
    on it identically on every rank before building anything.  Callers use
    it to keep their historical (degraded) path alive for a shape this
    primitive must refuse, instead of turning a working-but-slow run into
    a crash.

    Both routes are answered: ``flatten_m_first`` accepts an M that is not
    a multiple of ``p_x·p_y`` only because the factory pads it locally
    (``m_loc`` up to a multiple of ``p_y``), which needs ``M % p_x == 0``
    and nothing more — the same requirement the INPUT layout already has.
    """
    ax_x, ax_y = axes
    names = tuple(mesh.axis_names)
    if route not in ROUTES:
        return False
    if ax_x not in names or ax_y not in names or names[-1] != ax_y:
        return False
    if len(shape) != 3:
        return False
    b, m, n = (int(s) for s in shape)
    from runtime.padding import padded_axis
    return (
        padded_axis(
            b, mesh, name="face batch carrier",
            spec=P(('x', 'y'), None, None), axis=0).carrier == b
        and padded_axis(
            m, mesh, name="face row carrier",
            spec=P(None, 'x', None), axis=1).carrier == m
        and padded_axis(
            n, mesh, name="face column carrier",
            spec=P(None, None, 'y'), axis=2).carrier == n)


def face_to_batch_reshard(mesh: Mesh, *,
                          axes: tuple[str, str] = ("x", "y"),
                          route: str = DEFAULT_ROUTE,
                          log_fn=None) -> Callable:
    """Factory: a ``shard_map``'d ``(B, M, N)`` face→batch reshard.

    Parameters
    ----------
    mesh
        2-D device mesh.  ``axes[1]`` MUST be the mesh's minor axis.
    axes
        ``(major, minor)`` mesh axis names.  ``axes[0]`` carries the M
        (row) face and cuts B first; ``axes[1]`` carries the N (column)
        face and cuts B second.
    route
        One of :data:`ROUTES` — which pair of exchanges to issue.  See the
        module docstring; the two are element-identical and differ only in
        schedule, peer counts and (for ``flatten_m_first``) a local M pad.
    log_fn
        Optional rank-0 logger; used to announce the ``flatten_m_first``
        pad, which is a real cost and must never be silent.

    Returns
    -------
    ``reshard(a)`` — takes ``(B, M, N)`` laid out ``P(ax_x_face)`` i.e.
    ``P(None, ax_x, ax_y)`` and returns the same values laid out
    ``P((ax_x, ax_y), None, None)``.  Values are UNCHANGED: the body
    issues two ``lax.all_to_all``s and no arithmetic whatsoever, so this
    is the bit-exact parity class (staged-reshard doctrine §4.1(a)).

    The factory must be invoked synchronously on every rank — it warms
    this mesh's MPI cliques (``common.collectives.warm_mesh_cliques``,
    a no-op off ``impl=mpi`` and on an already-warm mesh), for the same
    reason ``contract_bands_block_reshard`` does: ``all_to_all`` acquires
    a per-mesh-axis communicator, and under ``impl=mpi`` a clique first
    created from an intra-op pool worker dies on jaxlib's
    ``MPI_Is_thread_main`` guard.
    """
    from common.shard_map import shard_map

    if route not in ROUTES:
        raise ValueError(
            f"face_to_batch_reshard: route={route!r} unknown; expected one "
            f"of {' | '.join(ROUTES)}.")
    ax_x, ax_y = axes
    names = tuple(mesh.axis_names)
    if ax_x not in names or ax_y not in names:
        raise ValueError(
            f"face_to_batch_reshard: mesh axes {names} do not contain "
            f"axes={axes!r}")
    if names[-1] != ax_y:
        raise ValueError(
            f"face_to_batch_reshard: the mesh's minor axis is {names[-1]!r} "
            f"but the batch axis is numbered {ax_x!r}-major by "
            f"P(({ax_x!r}, {ax_y!r}), ...), so the SECOND exchange must run "
            f"over {ax_y!r}.  On the standard process-ordered device layout "
            f"only the LAST mesh axis has consecutive-rank (node-local) "
            f"replica groups; on an inverted mesh the block numbering the "
            f"caller asked for and the groups these collectives use stop "
            f"agreeing.  Build the mesh with {ax_y!r} minor, or pass "
            f"axes=(major, minor) matching your mesh.")
    p_x, p_y = int(mesh.shape[ax_x]), int(mesh.shape[ax_y])
    from runtime.padding import mesh_divisor
    ndev = mesh_divisor(mesh)
    _log = log_fn if log_fn is not None else (lambda *a, **k: None)

    def _body_split_b_first(a):
        # a: (B, M/p_x, N/p_y) — this rank's face tile of the WHOLE batch.
        #
        # Stage 1, over ax_x: hand out B in p_x pieces, collect the M axis.
        # lax.all_to_all(x, axis, split_axis, concat_axis, tiled=True) gives
        # the rank at index r along ``axis`` every peer's split-chunk r,
        # laid down along ``concat_axis`` in peer order — so afterwards rank
        # (x, y) holds B-chunk x, all of M, and its own N tile.
        if p_x > 1:
            a = jax.lax.all_to_all(a, ax_x, split_axis=0, concat_axis=1,
                                   tiled=True)                # (B/px, M, N/py)
        # Stage 2, over ax_y: cut that B-chunk again in p_y pieces and
        # collect N.  Rank (x, y) ends holding B-block x·py + y whole —
        # which is what P((ax_x, ax_y), None, None) means.
        if p_y > 1:
            a = jax.lax.all_to_all(a, ax_y, split_axis=0, concat_axis=2,
                                   tiled=True)             # (B/(px·py), M, N)
        return a

    def _make_body_flatten_m_first(m_loc, m_pad, keep_idx):
        """Owner's route: gather N first, then ONE flattened B exchange.

        ``m_loc`` is the incoming per-rank M extent (M/p_x).  Stage 1 cuts
        it p_y ways, so when ``m_loc % p_y`` it is zero-padded to ``m_pad``
        FIRST — locally, no collective — and the pad rows are removed at
        the end with a static gather (``keep_idx``).  Padding locally
        interleaves the pad into the global M axis (rows m_loc..m_pad-1 of
        every p_x-block), which is why the removal is a gather and not a
        slice.
        """
        def _body(a):
            if m_pad != m_loc:
                a = jnp.pad(a, ((0, 0), (0, m_pad - m_loc), (0, 0)))
            # Stage 1, over ax_y: split THIS rank's M tile p_y ways and
            # collect N.  Rank (x, y) ends with M-block x·p_y + y (of the
            # padded axis) and the whole N axis: P(None, (ax_x, ax_y), None).
            if p_y > 1:
                a = jax.lax.all_to_all(a, ax_y, split_axis=1, concat_axis=2,
                                       tiled=True)      # (B, M_pad/ndev, N)
            # Stage 2, over the FLATTENED (ax_x, ax_y): hand out B in ndev
            # pieces and collect M.  Peer order over a tuple axis is
            # ax_x-major, the same numbering P((ax_x, ax_y), ...) uses, so
            # rank (x, y) ends with B-block x·p_y + y and the whole padded M.
            if ndev > 1:
                a = jax.lax.all_to_all(a, (ax_x, ax_y), split_axis=0,
                                       concat_axis=1, tiled=True)
            if keep_idx is not None:
                a = jnp.take(a, keep_idx, axis=1)
            return a
        return _body

    def _make_sm(body):
        return shard_map(body, mesh=mesh,
                         in_specs=(P(None, ax_x, ax_y),),
                         out_specs=P((ax_x, ax_y), None, None),
                         check_vma=False)

    _sm_cache: dict = {}

    def _reshard(a):
        shape = tuple(int(s) for s in a.shape)
        if len(shape) != 3:
            raise ValueError(
                f"face_to_batch_reshard expects a rank-3 (B, M, N) array, "
                f"got shape {shape}")
        b, m, n = shape
        from runtime.padding import authenticate_padded_axis
        authenticate_padded_axis(
            b, b, ndev, name="face-to-batch batch carrier")
        authenticate_padded_axis(
            m, m, p_x, name="face-to-batch row carrier")
        authenticate_padded_axis(
            n, n, p_y, name="face-to-batch column carrier")
        sm = _sm_cache.get(shape)
        if sm is None:
            if route == "split_b_first":
                sm = _make_sm(_body_split_b_first)
            else:
                m_loc = m // p_x
                from runtime.padding import padded_axis
                m_pad = padded_axis(
                    m_loc, p_y, name="face-to-batch local row carrier").carrier
                if m_pad == m_loc:
                    keep = None
                else:
                    # NUMPY on purpose: a closed-over jax.Array inside a
                    # shard_map body is a replicated operand, a host array is
                    # folded in as a constant.
                    keep = np.concatenate(
                        [np.arange(x * m_pad, x * m_pad + m_loc,
                                   dtype=np.int32) for x in range(p_x)])
                    _log(f"  face_to_batch_reshard[flatten_m_first]: M={m} "
                         f"gives m_loc={m_loc} which is not a multiple of "
                         f"p_y={p_y}; zero-padding the local M tile to "
                         f"{m_pad} (global {m_pad * p_x}, +"
                         f"{100.0 * (m_pad * p_x - m) / m:.2f}%) and dropping "
                         f"the {m_pad * p_x - m} pad rows with a static "
                         f"gather.  This cost belongs to this route only; "
                         f"'split_b_first' needs no M pad.")
                sm = _make_sm(_make_body_flatten_m_first(m_loc, m_pad, keep))
            _sm_cache[shape] = sm
        return sm(a)

    # Same policy, same reason, as contract_bands_block_reshard's factory:
    # create this mesh's cliques on the MAIN thread now so the all_to_alls
    # inside the returned kernel hit XLA's clique cache rather than its
    # MPI_Is_thread_main guard.  No-op off impl=mpi / already-warm mesh.
    warm_mesh_cliques(mesh)

    return _reshard

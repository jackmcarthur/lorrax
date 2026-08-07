"""``face_to_batch_reshard`` — the STAGED face→batch reshard.

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
``unfold_v_q`` has shipped the same ``shard_map`` + ``lax.all_to_all``
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

Gate: ``tests/test_staged_reshard.py`` — value parity against the
unstaged chain, an HLO pin (2 ``all-to-all``, zero full-batch
``all-gather``), and a RED TWIN that compiles the unstaged chain in a
subprocess and requires the compiler's own
``Involuntary full rematerialization`` line to appear.  An instrument
that has not been shown failing is not an instrument (wk_REL README §5.1).
"""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P

from common.collectives import warm_mesh_cliques

#: The two exchange schedules that land on ``P((ax_x, ax_y), None, None)``.
#: ``split_b_first`` is the default and the measured one; ``flatten_m_first``
#: is the owner's proposal (see the module docstring).
ROUTES = ("split_b_first", "flatten_m_first")
DEFAULT_ROUTE = "split_b_first"

__all__ = ["face_to_batch_reshard", "face_to_batch_reshard_supported",
           "ROUTES", "DEFAULT_ROUTE"]


_DIVISIBILITY_MSG = (
    "face_to_batch_reshard: the staged exchange tiles B over {ax_x}·{ax_y}, "
    "M over {ax_x} and N over {ax_y}, so all three must divide exactly; got "
    "B={b} (% {ndev} = {b_rem}), M={m} (% {px} = {m_rem}), N={n} "
    "(% {py} = {n_rem}) on a {px}x{py} mesh.  Fix by PADDING the offending "
    "extent to the mesh (common.sharding_fit.padded_extent), not by degrading "
    "the spec: a degraded spec here is a replicated batch, which is the "
    "failure this primitive exists to remove.{hint}"
)


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
    p_x, p_y = int(mesh.shape[ax_x]), int(mesh.shape[ax_y])
    b, m, n = (int(s) for s in shape)
    return b % (p_x * p_y) == 0 and m % p_x == 0 and n % p_y == 0


def face_to_batch_reshard(mesh: Mesh, *,
                          axes: tuple[str, str] = ("x", "y"),
                          route: str = DEFAULT_ROUTE,
                          divisibility_hint: str = "",
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
    divisibility_hint
        Appended to the divisibility refusal so a call site can name its
        own padding lever.
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
    ndev = p_x * p_y
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
        if b % ndev or m % p_x or n % p_y:
            raise ValueError(_DIVISIBILITY_MSG.format(
                b=b, m=m, n=n, px=p_x, py=p_y, ndev=ndev,
                b_rem=b % ndev, m_rem=m % p_x, n_rem=n % p_y,
                ax_x=ax_x, ax_y=ax_y,
                hint=(("  " + divisibility_hint) if divisibility_hint
                      else "")))
        sm = _sm_cache.get(shape)
        if sm is None:
            if route == "split_b_first":
                sm = _make_sm(_body_split_b_first)
            else:
                m_loc = m // p_x
                m_pad = -(-m_loc // p_y) * p_y          # ceil to a p_y multiple
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

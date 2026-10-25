"""Distributed BSE kernels (ring collectives and sharded matvecs)."""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from jax import shard_map as _shard_map_fn

from common.collectives import prepare_mesh, resolve_mesh
from common.contract_bands import contract_bands_block_reshard
from common.fft_helpers import make_kconv_kminor
from lxkit import mark_varying
from lxkit import ordered_psum, ordered_psum_scatter

from .bse_preconditioner import exchange_spin_weight
from common.log import get_logger, message

log = get_logger(__name__)


jax.config.update("jax_enable_x64", True)


def _make_ring_rung(mesh_xy: Mesh, kgrid, w_decode):
    """Shared ladder-W field and bra decode factories.

        U  = fftn_k( ifftn_k(T) · W_R )          (both 'ortho')
        WX = (1/√Nk) Σ conj(ψ_c) ψ_v U           (the decode)

    ``T`` ``(b, μ, ν, t, s, nk)`` (μ on 'x', ν on 'y'), ``W_R``
    ``(μ, ν, kx, ky, kz)`` already in R space.  The convolution is ONE call of
    the k-convolution router's k-minor factory (nvidia-mathdx on CUDA, the plan
    route on cpu) whose store emits the decode's ``(b, k, t, μ, s, ν)`` layout
    (``out_layout=1``); ``w_decode`` is the caller's
    ``contract_bands_block_reshard(mesh, extra="leading")``.
    """
    nkx, nky, nkz = (int(v) for v in kgrid)
    nk = nkx * nky * nkz
    conv = make_kconv_kminor(mesh_xy, (nkx, nky, nkz),
                             P(None, "x", "y", None, None, None), P("x", "y", None),
                             norm="ortho", out_layout=1)

    def field(T, W_R):
        return conv(T, W_R.reshape(W_R.shape[0], W_R.shape[1], nk))

    def decode(O_b, psi_c_X, psi_v_Y):
        sqrt_nk = jnp.sqrt(jnp.asarray(nk, dtype=O_b.real.dtype))
        psi_v_snv = jnp.transpose(psi_v_Y, (0, 2, 3, 1))
        out = w_decode(psi_c_X, O_b, psi_v_snv)
        return jnp.transpose(out, (0, 2, 3, 1)) / sqrt_nk

    return field, decode


def create_mesh_xy(px: int, py: int, devices: Optional[list] = None) -> Mesh:
    """THE ('x','y') BSE device mesh, warmed — for a driver that names px/py.

    Every BSE entry point must get its mesh from here or from
    :func:`create_mesh_2d` (which is this function with the square shape
    chosen for it).  There is exactly one mesh factory in ``src/bse/`` and
    this is it.  Only SQUARE shapes are accepted (owner ruling, repo
    ``docs/architecture/decisions.md`` 2026-08-01): a ``px != py`` request
    refuses rather than building a rectangular mesh.

    The shape must consume the whole device list — ``px * py ==
    len(devices)``, both directions guarded (2026-08-27).  Over-request was
    always refused; under-request was not, so ``--px 2 --py 2`` on a
    16-device job was accepted and quietly built a mesh over ``devices[:4]``,
    leaving ranks 4..15 entering the same jit from outside the mesh.
    ``RuntimeStack.reshape`` (``runtime/__init__.py``) already had this rule;
    the two now cannot drift.  With the equality enforced, the
    ``devices[: px * py]`` slice below is the identity on the list it was
    handed.  The comparison is against the list in play, not against
    ``jax.devices()``: a caller that deliberately passes a sub-list
    (``create_mesh_2d([dev0])``, the 1x1 control leg of a 1x1/2x2 mesh
    parametrization) is asserting a different job, and does so greppably.
    A CLI never does it — its ``devices`` is ``None``.

    Not the factory for a process-local 1x1: use
    ``common.collectives.single_device_mesh``.  Passing ``devices[:1]`` here
    looks device-count independent and is not — the canonical-reuse branch
    below calls ``resolve_mesh``, which refuses outright on a device count
    that is not a perfect square (measured 2026-08-27: ``create_mesh_xy(1, 1,
    jax.local_devices()[:1])`` raises ``resolve_mesh: this run has 5 devices
    ... not a perfect square`` at ``--xla_force_host_platform_device_count=5``,
    while ``single_device_mesh()`` returns a 1x1 at every count).

    Why this function exists at all.  Four byte-identical
    ``_create_mesh_xy(px, py)`` bodies used to live in ``bse_w_exact``,
    ``bse_feast``, ``bse_pseudopoles`` (and, by import, ``bse_kpm``), none of
    which warmed anything, while only ``create_mesh_2d`` did.  So five BSE
    drivers — ``exciton_bands``, ``bse_w_exact``, ``bse_feast``, ``bse_kpm``,
    ``bse_pseudopoles`` — silently ran WITHOUT the clique warm-up that
    ``bse_jax`` depends on, and each one dies at P>1 under
    ``JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi`` the moment a collective is
    first issued from inside a jitted program (see
    :func:`common.collectives.warm_mesh_cliques`: 32 refusals at P=16, gate
    7881216).  A duplicated constructor is how a load-bearing side effect goes
    missing without anybody deleting a line.

    The warm-up is setup-time and size-independent.  ``prepare_mesh`` fires
    three 1-element ``psum``s (one per mesh axis + one over all axes) plus,
    on GPU, ``runtime.nccl_warmup``'s three tiny reductions.  ``O(log P)``
    latency, once per process, independent of ``N_mu`` / ``N_k·N_q`` / the
    trial-block width; it emits no HLO into any solver kernel, so the BSE
    communication pattern and every intermediate's sharding are untouched.
    That is what makes it admissible under the standing BSE rule.

    Collective: every rank must reach this together.  Do not make the call
    rank-conditional.
    """
    if devices is None:
        devices = jax.devices()
    px, py = int(px), int(py)
    if px != py:
        raise ValueError(
            f"create_mesh_xy: a {px}x{py} mesh was requested, but only "
            f"square 2-D meshes are supported (repo "
            f"docs/architecture/decisions.md, 2026-08-01) — rectangular "
            f"meshes complicate ScaLAPACK grid geometry and the "
            f"divisibility contracts for no measured benefit.  Request "
            f"px == py (and a square process count).")
    if px * py > len(devices):
        raise ValueError(
            f"Requested px*py={px*py} devices, only {len(devices)} available")
    if px * py < len(devices):
        # The fix field has to be a fix.  ``isqrt`` always returns something,
        # so an unguarded "request --px s --py s" tells a user on 8 devices to
        # type the 2x2 they just typed, and "or omit --px/--py" routes to
        # :func:`create_mesh_2d`, which refuses in different words.  Only a
        # perfect square has a square mesh to offer at all.
        s = math.isqrt(len(devices))
        if s * s == len(devices):
            fix = (f"Request --px {s} --py {s}, or omit --px/--py and get "
                   f"that mesh by default "
                   f"(:func:`create_mesh_xy_from_flags`).")
        else:
            fix = (f"{len(devices)} is not a perfect square, so NO --px/--py "
                   f"names a square mesh over this list and omitting them "
                   f"refuses too: relaunch on {s * s} devices (a {s}x{s} "
                   f"mesh) or {(s + 1) * (s + 1)} (a {s + 1}x{s + 1} mesh).  "
                   f"That is :func:`create_mesh_2d`'s refusal, and it is the "
                   f"one that applies here.")
        raise ValueError(
            f"create_mesh_xy: a {px}x{py} mesh was requested, but the device "
            f"list in play has {len(devices)} device(s) over "
            f"{jax.process_count()} process(es), so {len(devices) - px * py} "
            f"of them would sit OUTSIDE the mesh.  Idle-rank truncation is "
            f"not implemented on this stack, for the reason "
            f"``common.collectives.resolve_mesh`` states: under the "
            f"production transport JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi, "
            f"communicator creation is MPI_Comm_split — collective over "
            f"MPI_COMM_WORLD — so an out-of-mesh rank that never runs the "
            f"warm-up jits leaves every in-mesh rank blocked in the split, "
            f"and psum_replicate / the k-sweep gathers assume mesh size == "
            f"process count.  " + fix)
    # Reuse the run's canonical mesh when the request IS that mesh: since
    # the BSE drivers start with ``runtime.initialize_communicator_stack``
    # (2026-08-01), the canonical square ('x','y') mesh already exists — an
    # equal-but-distinct copy here would be a second set of communicators
    # and a second copy of every shape-keyed jit cache (the
    # ``collectives.single_device_mesh`` identity contract).  A different
    # requested shape is a deliberate second mesh and is built as before.
    canonical = resolve_mesh(axis_names=("x", "y"))
    if (tuple(int(n) for n in canonical.devices.shape) == (px, py)
            and list(canonical.devices.flat) == list(devices[: px * py])):
        mesh = canonical
    else:
        mesh = Mesh(np.array(devices[: px * py]).reshape(px, py),
                    axis_names=("x", "y"))
    # resolve_mesh (pass-through here) + warm_mesh_cliques + nccl_warmup.
    mesh = prepare_mesh(mesh)
    _warm_process_allgather(mesh)
    return mesh


def _warm_process_allgather(mesh: Mesh) -> None:
    """Create the ``multihost_utils.process_allgather`` clique on this thread.

    ``warm_mesh_cliques`` warms the cliques of OUR mesh: one per axis plus the
    world set, all created by ``shard_map(psum)`` over a ``('x','y')`` Mesh.
    ``jax.experimental.multihost_utils.process_allgather`` does not use that
    mesh — it builds its OWN one-dimensional ``Mesh(jax.devices(), 'processes')``
    and issues an all-gather on it.  That is a different collective on a
    differently-shaped mesh, and the x/y/world warm-up does not cover it.

    MEASURED, not reasoned: with the mesh warm-up in place and no allgather
    warm-up, every cell of job 7882523 (P=16) died at
    ``exciton_bands.py`` line 592 -> ``_gather_host`` ->
    ``common/collectives.py`` ``process_allgather`` with
    "MPI: Communicator requested from a thread that is not the one MPI was
    initialized from" — 4 of 4 cells, same frame every time.  The gather in
    question is the driver's on-grid diagnostic gate on the mu-sharded
    Q-shifted psi_c, which is genuinely process-spanning, so it MUST take the
    allgather arm; it cannot be avoided by branching differently.

    THE WARM-UP MUST TRAVEL THE REAL PATH.  A first attempt handed
    ``process_allgather`` a HOST numpy array; job 7882531 then failed in
    exactly the same frame as before (4 of 4 cells), because a host operand
    never creates the device clique the real call needs.  So the warm-up now
    builds a genuinely mesh-SHARDED (hence, at P>1, non-fully-addressable)
    array and pushes it through ``gather_to_host`` — the same function, the
    same arm, the same trailing ``np.asarray`` that materialises the result.
    A warm-up that does not reproduce the failing call is not a warm-up; that
    is the same lesson as a negative control that withholds the wrong thing.

    COST.  One all-gather of ``P`` float64 — 512 B at P=64 — once per process
    at driver setup.  ``O(log P)`` latency, independent of ``N_mu``,
    ``N_k*N_q`` and the trial-block width, and it emits no HLO into any solver
    kernel.  Same admissibility argument as ``warm_mesh_cliques``.

    No-op at ``process_count() <= 1``.  Best-effort: a failure here is not
    itself fatal (the real call will raise with a better frame), but it is
    announced.
    """
    if jax.process_count() <= 1:
        return
    try:
        from common.collectives import gather_to_host
        n = int(mesh.devices.size)
        probe = jnp.zeros(n, jnp.float64, device=NamedSharding(mesh, P(tuple(mesh.axis_names))))
        out = gather_to_host(probe)
        assert out.shape == (n,), out.shape
    except Exception as exc:                      # noqa: BLE001
        log.warning(f"  [collectives] process_allgather warm-up failed "
                    f"({type(exc).__name__}: {exc}); a later gather may refuse "
                    f"under JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi",
                    extra=message(
                        id="process_allgather_warmup_failed",
                        found=("process_allgather warm-up",
                               f"{type(exc).__name__}: {exc}"),
                        needs="a working cross-process gather",
                        fix="check JAX_CPU_COLLECTIVES_IMPLEMENTATION and the launcher"))


def create_mesh_2d(devices: Optional[list] = None) -> Mesh:
    """Create the square 2-D device mesh for BSE sharding.

    Thin wrapper over :func:`create_mesh_xy` — same warm-up, same contract,
    and the same square-only rule as ``common.collectives.resolve_mesh``:
    a device count that is not a perfect square refuses with the square
    count to request (the refusal fires in ``resolve_mesh``'s vocabulary
    via :func:`create_mesh_xy`'s canonical-mesh reuse, or here).
    """
    if devices is None:
        devices = jax.devices()

    n_devices = len(devices)
    s = math.isqrt(n_devices)
    if s * s != n_devices:
        raise RuntimeError(
            f"create_mesh_2d: {n_devices} devices is not a perfect square, "
            f"and only square 2-D meshes are supported (repo "
            f"docs/architecture/decisions.md, 2026-08-01).  Request "
            f"{s * s} (a {s}x{s} mesh) or {(s + 1) * (s + 1)} "
            f"(a {s + 1}x{s + 1} mesh) processes.")

    return create_mesh_xy(s, s, devices)


def create_mesh_xy_from_flags(px: Optional[int],
                              py: Optional[int]) -> Mesh:
    """Resolve ``--px/--py`` for every BSE driver's ``main()``.

    Omitted means the run's mesh.  ``px is None and py is None`` — the
    argparse default in all five bse-family drivers since 2026-08-27
    (``bse_jax``, ``bse_feast``, ``bse_kpm``, ``bse_w_exact``,
    ``exciton_bands``: 10 declarations, gated by
    ``tools/lint/test_layering.py::test_no_bse_driver_defaults_its_mesh_shape``) —
    is a request for the job's canonical square mesh, i.e. exactly what
    ``runtime.initialize_communicator_stack`` already built and announced in
    the startup report.  It used to mean ``1x1``: ``--px``/``--py``
    defaulted to ``1``, so ``python -m bse.bse_jax -i deck.in`` with no flags
    ran the whole solve on one device of a four-GPU node (three idle, no
    warning) and, at P>1, built a mesh over ``jax.devices()[:1]`` — process
    0's device on every rank — which every rank >= 1 then refuses in
    ``collectives._require_addressable``.  Every launch recipe in-tree
    hard-codes ``--px 2 --py 2`` (``tests/fast_gate.py``,
    ``tests/test_bse_bgw_regression.py``) — that is the shape of a
    default nobody can rely on.

    A given shape is an assertion, not a request.  Both must be given
    together, and :func:`create_mesh_xy` then refuses anything that is not
    square and does not consume the job.  What survives both guards is the
    canonical shape, so the returned object is the canonical mesh by identity
    (``resolve_mesh``'s ``_CANONICAL_MESHES`` cache), not an equal-but-
    distinct twin.  That is the property that matters: ``Mesh.__eq__`` is
    true for twins, while the shape-keyed jit caches key on the mesh object
    embedded in each ``NamedSharding``.

    The ``None`` arm goes through :func:`create_mesh_2d`, not through a bare
    ``resolve_mesh()``, because ``bse_feast`` / ``bse_w_exact`` / ``bse_kpm``
    call only ``runtime.bootstrap()`` and hold no
    ``RuntimeStack``: for them a bare resolve would hand back an unwarmed
    mesh and drop both :func:`common.collectives.prepare_mesh`'s cliques and
    the :func:`_warm_process_allgather` warm-up this module documents (32
    refusals at P=16, job 7881216; 4-of-4 cells at P=16, job 7882523).  Same
    object either way; only the warm-up differs.

    Collective: every rank must reach this together.
    """
    if (px is None) != (py is None):
        raise ValueError(
            f"create_mesh_xy_from_flags: --px and --py must be given "
            f"together (got px={px!r}, py={py!r}).  Omit BOTH to run on the "
            f"job's canonical square mesh, or give both — and then they must "
            f"name that same mesh, because a partial mesh strands the ranks "
            f"outside it (see create_mesh_xy).")
    if px is None:
        return create_mesh_2d()
    return create_mesh_xy(int(px), int(py))


def make_bse_shardings(mesh_xy: Mesh) -> SimpleNamespace:
    return SimpleNamespace(
        X=NamedSharding(mesh_xy, P(None, "x", "y", None)),
        X_full=NamedSharding(mesh_xy, P(None, None, "x", "y", None)),
        psi_x=NamedSharding(mesh_xy, P(None, None, None, "x")),
        psi_y=NamedSharding(mesh_xy, P(None, None, None, "y")),
        # The exchange pair amplitude M (k, c, v, μ): transitions tiled like X
        # (c on x, v on y), μ whole -- 1/P per rank, one copy for both legs.
        M=NamedSharding(mesh_xy, P(None, "x", "y", None)),
        V=NamedSharding(mesh_xy, P("x", "y")),
        W=NamedSharding(mesh_xy, P("x", "y", None, None, None)),
        eps=NamedSharding(mesh_xy, P(None, None)),
        R=NamedSharding(mesh_xy, P(None, "x", None, None, "y")),
        T=NamedSharding(mesh_xy, P(None, "x", "y", None, None, None)),
        U=NamedSharding(mesh_xy, P(None, "x", "y", None, None, None)),
        A=NamedSharding(mesh_xy, P(None, "x", "y", None, None)),
        D=NamedSharding(mesh_xy, P(None, "x", "y", None, None)),
        S=NamedSharding(mesh_xy, P(None, "y", None)),
        S_k0=NamedSharding(mesh_xy, P(None, "y")),
        U_mu=NamedSharding(mesh_xy, P(None, "x", None)),
        d_mu=NamedSharding(mesh_xy, P(None, "x")),
    )


def _ring_perm(axis_size: int) -> tuple[tuple[int, int], ...]:
    return tuple((i, (i + 1) % axis_size) for i in range(axis_size))


# ---------------------------------------------------------------------------
# Why the ring accumulators below are ``mark_varying``-ed
# ---------------------------------------------------------------------------
# Every ring sum here is ``acc = jnp.zeros(...)`` followed by a fori_loop whose
# body does ``acc = acc + <something built from a sharded operand>``.  From jax
# 0.7.0 ``shard_map`` tracks varying manual axes, and the loop carry's VMA set
# must be EQUAL at input and output -- so the empty set on the zeros init does
# not match the {x,y} the body produces, and the loop is rejected:
#
#   TypeError: scan body function carry input and carry output must have equal
#   types ... complex128[1,2,399,9] vs complex128[1,2,399,9]{x,y}
#
# (fori_loop with static bounds lowers to scan, which is why the message says
# "scan body".)  These shard_maps go through ``jax.shard_map`` with checking
# ON, so the marking is required; the ``check_vma=False`` shard_maps elsewhere
# in the tree are exempt.  See ``lxkit.mark_varying``.
#
# ('x','y') is what jax itself names in the error, and it is what the bodies
# introduce: the ring buffers come from X (in_spec P(None,'x','y',None)) and
# the psi slices from operands on 'x' or 'y'.  Do NOT widen a mark "to be
# safe" -- a carry marked over an axis it does not vary on fails when it
# leaves through a replicated out_specs entry.
def _ring_sum_valence(
    X: jax.Array,
    psi_v_Y: jax.Array,
    v_chunk: int,
    py: int,
    nu_local: int,
) -> jax.Array:
    axis_index_y = jnp.asarray(lax.axis_index("y"), dtype=jnp.int32)
    nk, _, nspinor, _ = psi_v_Y.shape

    R0 = mark_varying(
        jnp.zeros((X.shape[0], X.shape[1], nk, nspinor, nu_local), dtype=X.dtype),
        ("x", "y"))
    perm = _ring_perm(py)

    def step(i, carry):
        buf, R = carry
        origin = (axis_index_y - jnp.asarray(i, dtype=jnp.int32)) % py
        v_start = origin * jnp.asarray(v_chunk, dtype=jnp.int32)
        z = jnp.int32(0)
        psi_v_slice = lax.dynamic_slice(
            psi_v_Y, (z, v_start, z, z), (nk, v_chunk, nspinor, nu_local)
        )
        R = R + jnp.einsum("kv sN,bcvk->bcksN", jnp.conj(psi_v_slice), buf)
        buf = lax.ppermute(buf, axis_name="y", perm=perm)
        return buf, R

    _, R_total = lax.fori_loop(0, py, step, (X, R0))
    return R_total


def _ring_sum_conduction(
    R: jax.Array,
    psi_c_X: jax.Array,
    c_chunk: int,
    px: int,
    mu_local: int,
) -> jax.Array:
    axis_index_x = jnp.asarray(lax.axis_index("x"), dtype=jnp.int32)
    nk, _, nspinor, _ = psi_c_X.shape
    nu_local = R.shape[4]

    T0 = mark_varying(
        jnp.zeros((R.shape[0], mu_local, nu_local, nspinor, nspinor, nk),
                  dtype=R.dtype),
        ("x", "y"))
    perm = _ring_perm(px)

    def step(i, carry):
        buf, T = carry
        origin = (axis_index_x - jnp.asarray(i, dtype=jnp.int32)) % px
        c_start = origin * jnp.asarray(c_chunk, dtype=jnp.int32)
        z = jnp.int32(0)
        psi_c_slice = lax.dynamic_slice(
            psi_c_X, (z, c_start, z, z), (nk, c_chunk, nspinor, mu_local)
        )
        T = T + jnp.einsum("kctM,bcksN->bMNtsk", psi_c_slice, buf)
        buf = lax.ppermute(buf, axis_name="x", perm=perm)
        return buf, T

    _, T_total = lax.fori_loop(0, px, step, (R, T0))
    return T_total


# ---------------------------------------------------------------------------
# WHY THE COUPLING (B) ENCODE CANNOT BE A TWO-STAGE RING LIKE THE A ENCODE
# ---------------------------------------------------------------------------
# A ring stage ``ppermute``s a buffer along one mesh axis and slices the
# *stationary* psi at the origin rank's orbital block.  That is only sound if
# the moving buffer carries NO axis that is sharded on the rotation axis other
# than the orbital axis being consumed -- a ppermute moves every axis of the
# buffer, including a zeta (interpolation-vector) axis, and the accumulator it
# lands in belongs to the LOCAL zeta shard.
#
# The A encode satisfies that by construction.  Its stage 1 rings over 'y' to
# consume v and produces nu from psi_v_Y, but nu is produced INTO the
# accumulator, which never moves.  Its stage 2 rings over 'x' to consume c
# while the moving buffer carries nu on 'y' -- orthogonal to the rotation, and
# every rank in an 'x' row holds the same 'y' shard, so nu survives the trip.
#
# The B encode has Henneke's j_c <-> j_v swap on the encode side, so the zeta
# index produced by contracting c (sharded on 'x') is nu (sharded on 'y') and
# the one produced by contracting v (sharded on 'y') is mu (sharded on 'x').
# The pairing is CROSSED, and no ordering of two ring stages avoids rotating an
# intermediate along the very axis its zeta index lives on:
#
#     consume c first (ring 'x') -> R(v on 'y', nu on 'y'); ring 'y' next
#         drags nu with v                                    <- the 2026-08-08 defect
#     consume v first (ring 'y') -> R(c on 'x', mu on 'x'); ring 'x' next
#         drags mu with c
#
# The defect version of the first line was silent at P=1 (one shard, ppermute
# is the identity) and lost two thirds of the coupling correction at P=2x2:
# every 'y' rank accumulated the SAME sum
# ``sum_j sum_{v in block j} psi_v(mu) R(v, nu + j*nu_chunk)`` and stored it
# against its own nu shard, so the nu tiles of T came out bit-identical and the
# block lost its complex symmetry (0.69 vs 2.9e-11).
#
# THE FIX: move the communication onto the TRIAL VECTOR, which carries no zeta
# axis at all.  X is gathered along c on 'x' (X is by far the smallest tensor
# in this chain -- T carries two zeta axes) and then ring-rotated along 'y' to
# bring the v blocks round, so mu and nu are both produced into stationary
# accumulators and never travel.  Collective count stays O(px + py), matching
# the A encode; the T-sized and R-sized tensors never go on the wire.
def _ring_sum_B_encode(
    X: jax.Array,
    psi_c_Y: jax.Array,
    psi_v_X: jax.Array,
    v_chunk: int,
    px: int,
    py: int,
    mu_local: int,
    nu_local: int,
) -> jax.Array:
    """Coupling-block encode ``T(mu,nu,t,s,k) = sum_{c,v} psi_v(mu) X(c,v,k)
    conj(psi_c(nu))`` -- the c' <-> v' swapped partner of ``_encode_T``.

    Only ``X`` moves: an all-gather over 'x' (the c axis) and a ``py``-step
    ring over 'y' (the v axis).  ``mu`` and ``nu`` stay on their owning ranks.
    """
    axis_index_y = jnp.asarray(lax.axis_index("y"), dtype=jnp.int32)
    nk, _, nspinor, _ = psi_c_Y.shape
    z = jnp.int32(0)

    # c is contracted in full on every rank: gather the trial vector's
    # conduction axis once, rather than rotating a partially-contracted R.
    X_full_c = lax.all_gather(X, "x", axis=1, tiled=True)

    T0 = mark_varying(
        jnp.zeros((X.shape[0], mu_local, nu_local, nspinor, nspinor, nk),
                  dtype=X.dtype),
        ("x", "y"))
    perm = _ring_perm(py)

    def step(i, carry):
        buf, T = carry
        origin = (axis_index_y - jnp.asarray(i, dtype=jnp.int32)) % py
        v_start = origin * jnp.asarray(v_chunk, dtype=jnp.int32)
        # buf holds the origin rank's v block for EVERY c; nu is local and
        # stationary, so conj(psi_c_Y) is this rank's nu shard throughout.
        R = jnp.einsum("kcsN,bcvk->bvksN", jnp.conj(psi_c_Y), buf)
        psi_v_slice = lax.dynamic_slice(
            psi_v_X, (z, v_start, z, z), (nk, v_chunk, nspinor, mu_local)
        )
        T = T + jnp.einsum("kvtM,bvksN->bMNtsk", psi_v_slice, R)
        buf = lax.ppermute(buf, axis_name="y", perm=perm)
        return buf, T

    _, T_total = lax.fori_loop(0, py, step, (X_full_c, T0))
    return T_total


def apply_V_ring(
    X: jax.Array,
    psi_c_Y: jax.Array,
    psi_v_Y: jax.Array,
    M: jax.Array,
    V_q0: jax.Array,
    nk: int,
    px: int,
    py: int,
    exchange_channel: str = "singlet",
) -> jax.Array:
    # ``M`` (nk, c_loc, v_loc, μ): the hoisted decode-side exchange pair
    # amplitude ``Σ_s conj(ψ_c^X) ψ_v^X``, precomputed ONCE per solve (audit P3)
    # rather than rebuilt from ψ every matvec, in the TRANSITION layout ``sh.M``
    # (c on x, v on y, μ whole): the local tile is exactly this rank's X tile, so
    # the decode below is rank-local and reads the one 1/P copy in place.  The
    # encode side stays fused into the ψ_c^Y/ψ_v^Y ring sum below (it is
    # X-dependent, not hoistable).
    nb_trial = X.shape[0]
    sqrt_nk = jnp.sqrt(jnp.asarray(nk, dtype=X.real.dtype))

    c_chunk = X.shape[1]
    v_chunk = X.shape[2]

    axis_index_x = jnp.asarray(lax.axis_index("x"), dtype=jnp.int32)
    axis_index_y = jnp.asarray(lax.axis_index("y"), dtype=jnp.int32)

    nk_local = psi_c_Y.shape[0]
    nspinor = psi_c_Y.shape[2]
    nu_local = psi_c_Y.shape[-1]

    c_start_local = axis_index_x * jnp.asarray(c_chunk, dtype=jnp.int32)
    z = jnp.int32(0)
    psi_c_slice = lax.dynamic_slice(
        psi_c_Y, (z, c_start_local, z, z), (nk_local, c_chunk, nspinor, nu_local)
    )

    A0 = mark_varying(
        jnp.zeros((nb_trial, c_chunk, nu_local, nk_local), dtype=X.dtype),
        ("x", "y"))
    perm_y = _ring_perm(py)

    def step_y(i, carry):
        buf, A = carry
        origin = (axis_index_y - jnp.asarray(i, dtype=jnp.int32)) % py
        v_start = origin * jnp.asarray(v_chunk, dtype=jnp.int32)
        psi_v_slice = lax.dynamic_slice(
            psi_v_Y, (z, v_start, z, z), (nk_local, v_chunk, nspinor, nu_local)
        )
        # Forward (encode) leg carries the CONJUGATED vertex conj(M) =
        # ψ_c·conj(ψ_v) — see the K^x = M V M† note at the back-contract below.
        R_v = jnp.einsum("kvsN,bcvk->bcksN", jnp.conj(psi_v_slice), buf)
        A = A + jnp.einsum("kcsN,bcksN->bcNk", psi_c_slice, R_v)
        buf = lax.ppermute(buf, axis_name="y", perm=perm_y)
        return buf, A

    _, A_local = lax.fori_loop(0, py, step_y, (X, A0))

    S0 = mark_varying(
        jnp.zeros((nb_trial, nu_local, nk_local), dtype=X.dtype),
        ("x", "y"))
    perm_x = _ring_perm(px)

    def step_x(i, carry):
        buf, S = carry
        S = S + jnp.sum(buf, axis=1)
        buf = lax.ppermute(buf, axis_name="x", perm=perm_x)
        return buf, S

    _, S_total = lax.fori_loop(0, px, step_x, (A_local, S0))

    # Exchange (q=0) is DENSE in (k,k') — SUM over k' into a k-free
    # transition-Hartree density, one V_q0 solve, then broadcast back at every
    # k in the decode (VERDICT.md).  k is a replicated (unsharded) axis here, so
    # the reduction is device-local.  The two 1/sqrt_nk compose to 1/Nk.
    # exchange_spin_weight: the scalar-singlet 2 (bse_preconditioner owns it).
    S_total = (jnp.sum(S_total, axis=2) / sqrt_nk) * exchange_spin_weight(nspinor, exchange_channel)

    U_partial = jnp.einsum("MN,bN->bM", V_q0, S_total)  # (b, mu_local)
    U = ordered_psum(U_partial, axis_name="y")
    # U whole on μ: one all_gather of a (b, μ) vector over x.  The decode then
    # contracts M's own (c_x, v_y) tile over all of μ -- no reshard of M and no
    # psum_scatter of a (b, c, v, k) partial.  (Until 2026-09-26 the decode read
    # M μ-on-x; after M moved to the transition layout that cost a per-call
    # relayout of M inside every matvec: +1.5 ms on the Si 4x4x4 ladder.)
    U = lax.all_gather(U, "x", axis=1, tiled=True)  # (b, μ)
    # Back-contract carries the BARE vertex: K^x = M V M†, fixed by the
    # transition density <0|ρ̂|Ψ> = Σ A_cvk ψ_ck ψ*_vk.  (The reverse assignment
    # builds conj(K^x), which cannot be covariant alongside the correct W term.)
    # The optical coupling block's B = M V M^T is bse_stack_matvec's pair applier.
    VX = jnp.einsum("kcvM,bM->bcvk", M, U)  # broadcast over k

    return VX / sqrt_nk


def build_bse_ring_matvec_full(
    mesh_xy: Mesh,
    nkx: int,
    nky: int,
    nkz: int,
    low_mem: bool = True,
    include_W: bool = True,
    screening: bool = True,
    ladder_rung_slots: bool = False,
    fuse_ladder_rung: bool = True,
    physical_branches: bool = False,
    exchange_channel: str = "singlet",
):
    """Build the non-TDA density-response Hamiltonian, sharded over all P.

    physical_branches=True takes (X_full, payload), where the plain-array
    payload from ladder_matvec_operands names both physical R/A vertices.
    All four Hartree and direct blocks use the same encode/decode owners.
    The include_W=False default retains the original folded RPA control.
    Optical BSE uses bse_stack_matvec instead.
    """
    if include_W and not physical_branches:
        raise ValueError("Ladder screening requires physical_branches=True and "
                         "the physical R/A payload from build_finite_q_data")
    if ladder_rung_slots and not (screening and include_W):
        raise ValueError(
            "ladder_rung_slots=True is only meaningful for the ladder "
            "screening operator (screening=True, include_W=True): it extends "
            "the matvec signature with the four PHYSICAL (rolled, un-flipped) "
            "psi operands the direct rung consumes on a build_finite_q_data "
            "payload, and no other operator both takes that payload and "
            "carries a rung. Got screening=%r, include_W=%r." % (screening, include_W))
    if not screening:
        raise ValueError(
            "build_bse_ring_matvec_full serves the screening operator only "
            "(screening=True); the optical full BSE is "
            "bse_stack_matvec.build_bse_stack_matvec_full")
    px, py = mesh_xy.devices.shape
    sh = make_bse_shardings(mesh_xy)
    nk = nkx * nky * nkz

    def _encode_T(X, psi_c_X, psi_v_Y):
        c_chunk = X.shape[1]
        v_chunk = X.shape[2]
        n_rmu_local_X = psi_c_X.shape[-1]
        n_rmu_local_Y = psi_v_Y.shape[-1]
        R = _ring_sum_valence(X, psi_v_Y, v_chunk, py, n_rmu_local_Y)
        T = _ring_sum_conduction(R, psi_c_X, c_chunk, px, n_rmu_local_X)
        return T

    encode_T_ring = _shard_map_fn(
        _encode_T,
        mesh=mesh_xy,
        in_specs=(P(None, "x", "y", None), P(None, None, None, "x"), P(None, None, None, "y")),
        out_specs=P(None, "x", "y", None, None, None),
    )

    def _encode_T_gather(X, psi_c_X, psi_v_Y):
        X_full_v = lax.all_gather(X, "y", axis=2, tiled=True)
        R = jnp.einsum("kvsN,bcvk->bcksN", jnp.conj(psi_v_Y), X_full_v)
        R_full_c = lax.all_gather(R, "x", axis=1, tiled=True)
        T = jnp.einsum("kctM,bcksN->bMNtsk", psi_c_X, R_full_c)
        return T

    encode_T_gather = _shard_map_fn(
        _encode_T_gather,
        mesh=mesh_xy,
        in_specs=(P(None, "x", "y", None), P(None, None, None, "x"), P(None, None, None, "y")),
        out_specs=P(None, "x", "y", None, None, None),
    )

    def _encode_T_B(X, psi_c_Y, psi_v_X):
        v_chunk = X.shape[2]
        n_rmu_local_X = psi_v_X.shape[-1]
        n_rmu_local_Y = psi_c_Y.shape[-1]
        return _ring_sum_B_encode(X, psi_c_Y, psi_v_X, v_chunk, px, py,
                                  n_rmu_local_X, n_rmu_local_Y)

    encode_T_ring_B = _shard_map_fn(
        _encode_T_B,
        mesh=mesh_xy,
        in_specs=(P(None, "x", "y", None), P(None, None, None, "y"), P(None, None, None, "x")),
        out_specs=P(None, "x", "y", None, None, None),
    )

    def _encode_T_B_gather(X, psi_c_Y, psi_v_X):
        # Gather the TRIAL VECTOR on both axes, never the partially-contracted
        # R: R carries nu on 'y', and an all_gather of R along v concatenates
        # tiles whose nu shards differ -- the same defect as the ring version
        # (see _ring_sum_B_encode).  X carries no zeta axis, so gathering it is
        # sound, and it is the smallest tensor in the chain.
        X_full_c = lax.all_gather(X, "x", axis=1, tiled=True)
        X_full = lax.all_gather(X_full_c, "y", axis=2, tiled=True)
        R = jnp.einsum("kcsN,bcvk->bvksN", jnp.conj(psi_c_Y), X_full)
        T = jnp.einsum("kvtM,bvksN->bMNtsk", psi_v_X, R)
        return T

    encode_T_gather_B = _shard_map_fn(
        _encode_T_B_gather,
        mesh=mesh_xy,
        in_specs=(P(None, "x", "y", None), P(None, None, None, "y"), P(None, None, None, "x")),
        out_specs=P(None, "x", "y", None, None, None),
    )

    def _apply_V_ring_only(X, psi_c_Y, psi_v_Y, M, V_q0):
        return apply_V_ring(X, psi_c_Y, psi_v_Y, M, V_q0, nk, px, py, exchange_channel)

    apply_V_ring_only = _shard_map_fn(
        _apply_V_ring_only,
        mesh=mesh_xy,
        in_specs=(P(None, "x", "y", None), P(None, None, None, "y"), P(None, None, None, "y"),
                  P(None, "x", "y", None), P("x", "y")),
        out_specs=P(None, "x", "y", None),
    )

    # ψ†Uψ decode = contract_bands_block_reshard, extra="leading" (owner
    # order 2026-07-29; adoption map wk_REL/contract_bands_notes.md §6.2):
    # the structural stacked psum_scatter chain -- large partial over the
    # node-local 'y' groups, small final over 'x', all b trials on ONE
    # collective per mesh axis.  The rung's k-convolution emits the decode's
    # canonical O layout (b, k, t, M, s, N) straight from its store
    # (:func:`_make_ring_rung`), so no transpose sits between them.
    _w_decode = contract_bands_block_reshard(mesh_xy, extra="leading")
    rung_field, rung_decode = _make_ring_rung(mesh_xy, (nkx, nky, nkz), _w_decode)

    def _apply_D_term(X, eps_c, eps_v):
        delta_E = eps_c.T[None, :, None, :] - eps_v.T[None, None, :, :]
        return delta_E * X

    apply_D_term = jax.jit(
        _apply_D_term,
        in_shardings=(sh.X, sh.eps, sh.eps),
        out_shardings=sh.X,
    )

    def _T_term_A(X, psi_cW_X, psi_vW_Y):
        """The rung's ``(mu, nu, s, s, k)`` intermediate for ``W_d X``.

        Split out of :func:`_w_term_A` so a caller that applies SEVERAL rung
        terms with the same ``(psi, W_R)`` can sum their ``T`` first and run the
        FFT chain once — see ``_impl_core``'s fused row.  No behaviour of its
        own: ``_w_term_A`` is still exactly ``apply_W_from_T(_T_term_A(...))``.
        """
        return (encode_T_ring(X, psi_cW_X, psi_vW_Y) if low_mem
                else encode_T_gather(X, psi_cW_X, psi_vW_Y))

    def _T_term_B(X, psi_cW_Y, psi_vW_X):
        """The same, for the c'<->v'-swapped (coupling) rung ``W_d^B X``."""
        return (encode_T_ring_B(X, psi_cW_Y, psi_vW_X) if low_mem
                else encode_T_gather_B(X, psi_cW_Y, psi_vW_X))

    if physical_branches:
        from common.collectives import transpose_xy

        def physical(X_full, p):
            """Physical R=(c[k+q],v[k]), A=(v[k+q],c[k]) density operator.

            All four direct blocks encode their ket into the same centroid
            field and decode with the physical bra. The anti branch stays
            in (c,v,k) storage; its decode alone transposes the band owners.
            """
            X, Y = X_full
            acy, avy = jnp.conj(p["psi_c_A_Y"]), jnp.conj(p["psi_v_A_Y"])
            vr = (apply_V_ring_only(X, p["psi_c_Y"], p["psi_v_Y"], p["M"], p["V_q0"])
                  + apply_V_ring_only(Y, acy, avy, p["M"], p["V_q0"]))
            va = (apply_V_ring_only(X, p["psi_c_Y"], p["psi_v_Y"], p["M_A"], p["V_q0"])
                  + apply_V_ring_only(Y, acy, avy, p["M_A"], p["V_q0"]))

            def rung_trial(carry, rows):
                x, y = rows[0][None], rows[1][None]
                t = (_T_term_A(x, p["psi_c_X"], p["psi_v_Y"])
                     + _T_term_B(y, p["psi_c_A_Y"], p["psi_v_A_X"]))
                field = rung_field(t, p["W_R"])
                wr = rung_decode(field, p["psi_c_X"], p["psi_v_Y"])
                # The same mixed kernel's anti bra has (v_X,c_Y) output.
                wa = rung_decode(field, p["psi_v_A_X"], p["psi_c_A_Y"])
                wa = transpose_xy(jnp.transpose(wa, (0, 3, 1, 2)), mesh_xy)
                return carry, (wr[0], jnp.transpose(wa, (0, 2, 3, 1))[0])

            if include_W:
                _, (wr, wa) = lax.scan(rung_trial, None, (X, Y), unroll=1)
            else:
                wr, wa = jnp.zeros_like(X), jnp.zeros_like(Y)
            return jnp.stack((apply_D_term(X, p["eps_c"], p["eps_v"]) + vr - wr,
                              -apply_D_term(Y, p["eps_c_A"], p["eps_v_A"]) - va + wa))

        return jax.jit(physical, out_shardings=sh.X_full)

    def rpa(X_full, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c,
            eps_v, W_R, V_q0, M):
        X, Y = X_full
        vx = apply_V_ring_only(X, psi_c_Y, psi_v_Y, M, V_q0)
        vy = apply_V_ring_only(Y, psi_c_Y, psi_v_Y, M, V_q0)
        return jnp.stack((apply_D_term(X, eps_c, eps_v) + vx + vy,
                          -vx - (apply_D_term(Y, eps_c, eps_v) + vy)))

    return jax.jit(rpa,
        in_shardings=(sh.X_full, sh.psi_x, sh.psi_y, sh.psi_x, sh.psi_y,
                      sh.eps, sh.eps, sh.W, sh.V, sh.M),
        out_shardings=sh.X_full)


def build_realspace_random_transition_generator(
    mesh_xy: Mesh,
    nkx: int,
    nky: int,
    nkz: int,
    n_cond_pad: int,
    n_val_pad: int,
):
    """Build a sharded map: r(nu,k) -> V_q0 @ r -> X(c,v,k) (local c/v chunks).

    Assumes r is column-sharded on the Y axis and V_q0 is tiled (x,y).
    The resulting X is sharded like the BSE matvec (c on x, v on y).
    """
    px, py = mesh_xy.devices.shape
    sh = make_bse_shardings(mesh_xy)
    nk = nkx * nky * nkz

    from runtime.padding import authenticate_padded_axis
    authenticate_padded_axis(
        n_cond_pad, n_cond_pad, mesh_xy, name="BSE conduction carrier",
        spec=P(None, "x", None, None, None), axis=1)
    authenticate_padded_axis(
        n_val_pad, n_val_pad, mesh_xy, name="BSE valence carrier",
        spec=P(None, None, "y", None, None), axis=2)

    v_chunk = n_val_pad // py

    def _map(r, psi_c_X, psi_v_X, V_q0):
        # r: (batch, nu_local, nk), column-sharded on y.
        U_partial = jnp.einsum("MN,bNk->bMk", V_q0, r)
        U = ordered_psum(U_partial, axis_name="y")  # (batch, mu_local_x, nk), mu on x

        axis_index_y = jnp.asarray(lax.axis_index("y"), dtype=jnp.int32)
        v_start = axis_index_y * jnp.asarray(v_chunk, dtype=jnp.int32)
        z = jnp.int32(0)

        # v is a free output index (tiled on y) -> pre-slice its y-block.  c is
        # NOT pre-sliced: the centroid (mu) contraction below is over the LOCAL
        # x-slice of mu, so it must be completed across x by a psum.  Slicing c to
        # this x-rank's block first (as an earlier version did) silently drops the
        # off-rank mu contributions to every c — correct only at px=1 (~50% wrong
        # at px=2).  Mirror apply_V_ring: full c, psum_scatter over x.
        nk_local, _, nspinor, mu_local = psi_c_X.shape
        psi_v_slice = lax.dynamic_slice(
            psi_v_X, (z, v_start, z, z), (nk_local, v_chunk, nspinor, mu_local)
        )

        M_X = jnp.einsum("kcsm,kvsm->kcvm", jnp.conj(psi_c_X), psi_v_slice)  # (k,c_full,v,mu_x)
        # Decode leg of K^x = M V M† carries the BARE vertex (the conjugate
        # belongs on the encode leg — see build_density_snapshot_operator, the
        # partner half of the v(ω−H)⁻¹v form these two assemble together).
        VX_partial = jnp.einsum("kcvM,bMk->bcvk", M_X, U)                    # local-mu, c full
        # psum over x completes the mu sum across x-slices AND scatters the
        # conduction index onto x -> (b, c_chunk_x, v_chunk_y, k).
        X_local = ordered_psum_scatter(
            VX_partial, axis_name="x", scatter_dimension=1, tiled=True)

        sqrt_nk = jnp.sqrt(jnp.asarray(nk, dtype=X_local.real.dtype))
        return X_local / sqrt_nk

    _gen = _shard_map_fn(
        _map,
        mesh=mesh_xy,
        in_specs=(P(None, "y", None), P(None, None, None, "x"), P(None, None, None, "x"), P("x", "y")),
        out_specs=P(None, "x", "y", None),
    )
    # jit the seed (zeta->pair) reshard boundary.  A bare shard_map re-traces and
    # re-lowers to HLO on EVERY eager call (~2.5 s on the MoS2 fixture) because the
    # trace is not memoized; wrapping it in jax.jit caches the compiled executable
    # so repeated calls dispatch in <1 ms.  The BSE matvec is jitted for exactly
    # this reason — the seed/project boundaries were the only un-jitted ones and
    # dominated the W-column solve (see PHASE2_LOG "W-column resolvent profiling").
    # Bit-faithful: same shard_map body, same in/out shardings.
    return jax.jit(
        _gen,
        in_shardings=(sh.S, sh.psi_x, sh.psi_x, sh.V),
        out_shardings=sh.X,
    )


def build_density_snapshot_operator(
    mesh_xy: Mesh,
    nkx: int,
    nky: int,
    nkz: int,
    scatter_nu_on_y: bool = False,
    exchange_channel: str = "singlet",
):
    """Build a sharded map: s(b,c,v,k) -> d(b,mu) = V_q0 @ sum_k R s.

    The projection back from the (c,v,k) pair basis to the ISDF density
    (centroid) basis.  ``b`` is a batch axis (one probe/trial vector per slot),
    replicated across the mesh; ``mu`` is the centroid output index.

    Two output layouts, selected at build time:

    * ``scatter_nu_on_y=False`` (default): the V_q0 contraction over N is
      completed with a plain ``psum('y')`` and ``d`` returns replicated on the
      batch axis with ``mu`` on ``x`` — ``P(None, 'x')`` = ``(b, mu_X)``.

    * ``scatter_nu_on_y=True``: the W-column / screened-W(omega) path, where the
      batch axis IS the probe index ``nu`` and we want the assembled tile
      ``W(mu_X, nu_Y)`` with no replicated ``(mu, nu)`` intermediate.  This
      mirrors the shared band-projection owner
      (``common.contract_bands.contract_bands_block_reshard``): the same mesh
      axis ``y`` carries BOTH the reduction (finish the V_q0 N-contraction,
      whose partials are scattered across the ``y`` shards) AND the output
      tiling (scatter the ``nu`` batch onto ``y``), fused into ONE
      ``psum_scatter`` — same NCCL volume as the plain ``psum`` but the tile
      lands ``P('x', 'y')`` = ``(mu_X, nu_Y)`` = ``sh.V`` device-local.
      Requires ``b % py == 0`` (pad the probe block; ``padded_mu_extent`` already
      rounds the full-basis centroid count to a multiple of ``px*py``).

    Returns density-space vectors summed over k.
    """
    px, py = mesh_xy.devices.shape
    sh = make_bse_shardings(mesh_xy)
    nk = nkx * nky * nkz

    def _map(s, psi_c_Y, psi_v_Y, V_q0):
        nb_trial = s.shape[0]
        sqrt_nk = jnp.sqrt(jnp.asarray(nk, dtype=s.real.dtype))

        c_chunk = s.shape[1]
        v_chunk = s.shape[2]

        axis_index_x = jnp.asarray(lax.axis_index("x"), dtype=jnp.int32)
        axis_index_y = jnp.asarray(lax.axis_index("y"), dtype=jnp.int32)

        nk_local = psi_c_Y.shape[0]
        nspinor = psi_c_Y.shape[2]
        nu_local = psi_c_Y.shape[-1]

        c_start_local = axis_index_x * jnp.asarray(c_chunk, dtype=jnp.int32)
        z = jnp.int32(0)
        psi_c_slice = lax.dynamic_slice(
            psi_c_Y, (z, c_start_local, z, z), (nk_local, c_chunk, nspinor, nu_local)
        )

        A0 = mark_varying(
            jnp.zeros((nb_trial, c_chunk, nu_local, nk_local), dtype=s.dtype),
            ("x", "y"))
        perm_y = _ring_perm(py)

        def step_y(i, carry):
            buf, A = carry
            origin = (axis_index_y - jnp.asarray(i, dtype=jnp.int32)) % py
            v_start = origin * jnp.asarray(v_chunk, dtype=jnp.int32)
            psi_v_slice = lax.dynamic_slice(
                psi_v_Y, (z, v_start, z, z), (nk_local, v_chunk, nspinor, nu_local)
            )
            # Encode leg carries the CONJUGATED vertex conj(M) = ψ_c·conj(ψ_v),
            # matching K^x = M V M† (the decode half lives in
            # build_realspace_random_transition_generator).
            R_v = jnp.einsum("kvsN,bcvk->bcksN", jnp.conj(psi_v_slice), buf)
            A = A + jnp.einsum("kcsN,bcksN->bcNk", psi_c_slice, R_v)
            buf = lax.ppermute(buf, axis_name="y", perm=perm_y)
            return buf, A

        _, A_local = lax.fori_loop(0, py, step_y, (s, A0))

        S0 = mark_varying(
            jnp.zeros((nb_trial, nu_local, nk_local), dtype=s.dtype),
            ("x", "y"))
        perm_x = _ring_perm(px)

        def step_x(i, carry):
            buf, S = carry
            S = S + jnp.sum(buf, axis=1)
            buf = lax.ppermute(buf, axis_name="x", perm=perm_x)
            return buf, S

        _, S_total = lax.fori_loop(0, px, step_x, (A_local, S0))

        # exchange_spin_weight(nspinor): this ENCODE is the resolvent's own
        # outer readout (v(M s), the density-snapshot vertex named in
        # bse_w_exact._build_rpa_resolvent's docstring) and needs the SAME
        # missing weight as apply_V_ring's identically-shaped S_total — see
        # exchange_spin_weight's docstring.
        S_total = (S_total / sqrt_nk) * exchange_spin_weight(nspinor, exchange_channel)

        # V_q0 @ S: local N-slice partials, N tiled on y (mu on x from V_q0).
        U_partial = jnp.einsum("MN,bNk->bMk", V_q0, S_total)  # (b, mu_local_x, k)

        if not scatter_nu_on_y:
            U = ordered_psum(U_partial, axis_name="y")  # finish N sum, replicate on y
            return jnp.sum(U, axis=-1)              # (b, mu_local_x)

        # Reduce-scatter tail (mirror of ppm_tau_kernel reduce-scatter): sum the
        # local-N partials across y (completes the V_q0 N-contraction) AND scatter
        # the nu batch onto y in one collective, so W lands (mu_X, nu_Y) with no
        # replicated (mu, nu).  Local transpose first (no comms) so nu is the
        # scatter axis; psum_scatter reduces over y and tiles nu across y.
        d_partial = jnp.sum(U_partial, axis=-1)          # (b, mu_local_x)
        d_local = jnp.swapaxes(d_partial, 0, 1)          # (mu_local_x, b)
        return ordered_psum_scatter(
            d_local, axis_name="y", scatter_dimension=1, tiled=True
        )                                                # (mu_local_x, b/py)

    _snap = _shard_map_fn(
        _map,
        mesh=mesh_xy,
        in_specs=(P(None, "x", "y", None), P(None, None, None, "y"), P(None, None, None, "y"), P("x", "y")),
        out_specs=(P("x", "y") if scatter_nu_on_y else P(None, "x")),
    )
    # jit the projection (pair->zeta) reshard boundary — same rationale as
    # build_realspace_random_transition_generator: cache the compiled executable
    # instead of re-lowering the shard_map on every eager call (~2.8 s -> <3 ms).
    # Bit-faithful.  Output sharding follows the build-time scatter_nu_on_y branch.
    return jax.jit(
        _snap,
        in_shardings=(sh.X, sh.psi_y, sh.psi_y, sh.V),
        out_shardings=(sh.V if scatter_nu_on_y else sh.d_mu),
    )


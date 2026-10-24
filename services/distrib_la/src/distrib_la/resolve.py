"""Backend resolution — the ONE guard ladder.

A requested backend name (``auto | off | distributed | native2d |
cusolvermp | elpa | scalapack``, per op) becomes a concrete,
*guaranteed-callable* backend, or a clear resolve-time error.  Every guard
lives here, applied in ONE fixed order — SEVEN steps, all at resolve time:

    1. vocabulary   — is the name a backend of this op at all?
                      (``distributed`` resolves to the platform's default
                      distributed library FIRST, then runs the whole
                      ladder as if it had been named explicitly)
    1b. size        — ``distributed`` on a leg with a size row
                      (:data:`ONE_DEVICE_MAX_N`, ROCM; or a room on a
                      :data:`ONE_DEVICE_BY_ROOM` leg, CUDA) and a known
                      ``n`` at or below the row's bound resolves to
                      ``one_device`` instead: the matrix fits one device,
                      where the library's fixed per-call cost is the whole
                      cost (§ "Distributed is a capacity route").  Declared,
                      recorded on the plan and announced once; never silent.
    2. platform     — does the backend run on this mesh's device kind?
                      (cusolvermp is CUDA-only; scalapack is host-only)
    2c. known-broken— cusolvermp eigh with ``compute_evecs=False``
                      (bug L-3: cusolverMpSyevd status=7 at every n)
    3. capability   — is the backend's FFI handler actually usable?
                      (``loader.probe_target``, which separates "the
                      library would not load" from "the library has no
                      such handler" — partial builds legitimately omit
                      handlers, and either way it must fail HERE, not
                      minutes later at the first distributed call, with
                      the reason that names the actual fix)
    4. coverage     — the FFI backends run ONE JAX process per device
                      (their MPI/NCCL context is per-process); a mesh
                      with more devices than processes cannot drive them
    5. geometry     — square-mesh where required (cusolverMpSyevd
                      DEADLOCKS inside a collective on rectangular blocks
                      instead of returning an error), ScaLAPACK's square-block descriptor requirement
    6. divisibility — ``n`` divisible by both mesh axes (only checked
                      when the caller passes ``n``)

After all six pass, an explicit ``eigh`` on cuSOLVERMp below
:data:`_EIGH_DISTRIBUTED_ANNOUNCE_BELOW` gets a rank-0 NOTICE
(:func:`_announce_eigh_fixed_cost`).  A notice is not a guard: the
resolution already succeeded and the caller gets exactly the backend they
named.  It exists because that route is a capacity route, not a speed
route, and nothing in this module would otherwise say so.

PROMISE SEMANTICS ARE LAW.  A returned FFI name means every guard passed:
the subsequent call cannot fail for an availability or geometry reason.
Explicit requests REFUSE; only ``auto`` may demote, and it announces on
rank 0.  The worst measured defect in this tree — a full MoS2 12×12 G0W0
that ran to completion with rc=0 and a QP gap of −161 eV — was a silent
route change, so a silent demote is not a convenience here, it is the
failure mode.

``auto`` and ``off`` resolve to ``native`` (pure JAX, available
everywhere) except that ``auto`` for cholesky / solve_lu picks cusolvermp
on a true-2D CUDA mesh when it is compiled.  An eigh ``auto`` plan that carries
a room (``plan(..., budget_bytes=)``) also names the capacity route for the
stacks that do not fit it: :func:`resolve_eigh_capacity`.  ``off`` is an *override*, not
a guard: it is honored unconditionally.

ZERO ENVIRONMENT READS.  Nothing in this module consults ``os.environ``.
Backend choice is a deck key; the environment grants a CAPABILITY (which
``.so`` is pinned — :mod:`distrib_la.loader`) and never selects a route.
"""
from __future__ import annotations

import logging
import math

from lxkit import memory


from lxkit import FFI_PLATFORM_MAP, announce_once, mesh_ffi_platform

from distrib_la import loader

#: ``distrib_la`` is independently installable: it logs on the ``lorrax.`` namespace directly.
_LOG = logging.getLogger("lorrax.distrib_la")

__all__ = [
    "OPS",
    "NATIVE",
    "NATIVE2D",
    "ONE_DEVICE",
    "ONE_DEVICE_MAX_N",
    "ONE_DEVICE_BY_ROOM",
    "ONE_DEVICE_ROUND_N",
    "one_device_round_n",
    "one_device_max_n",
    "size_route_note",
    "BACKEND_CHOICES",
    "EIGH_BACKENDS",
    "CHOLESKY_BACKENDS",
    "LU_BACKENDS",
    "FFI_PLATFORMS",
    "mesh_platform",
    "mesh_is_cpu",
    "mesh_key",
    "resolve_backend",
    "resolve_eigh_capacity",
    "list_backends",
    "backend_module",
]

#: The operations the facade dispatches.
OPS = ("eigh", "cholesky", "solve_lu")

#: The resolved name of the in-tree pure-JAX implementations.  They are
#: first-class backends: always available, on every platform, and the
#: measured default for every op at production tile sizes.
NATIVE = "native"

#: The 2-D block-distributed tiled Cholesky (:mod:`distrib_la._native2d`).
#: Also pure JAX and also available on every platform, but a DIFFERENT
#: algorithm with a different memory profile, which is why it is a backend
#: and not a mode of ``native``: at n=10k on P=128 the replicated-reshard
#: route costs 1.6 GB/device and this one 5 MB/device.  ``auto`` never
#: picks it -- choosing an algorithm whose cost model differs by three
#: orders of magnitude in BOTH directions is the caller's decision.
NATIVE2D = "native2d"

#: Each matrix moved WHOLE onto one device (route (c)'s staged exchange,
#: :mod:`distrib_la._batch_reshard`), solved there by the leg's native JAX
#: kernel, and returned in the caller's face layout ``P(None,'x','y')``
#: (:mod:`distrib_la._one_device`).  Pure JAX, like ``native2d``, but not a
#: mode of ``native``: it takes the FFI backends' operand contract, so a
#: ``distributed`` call site runs it unchanged.  It is what ``distributed``
#: resolves to on a leg with a size row (:data:`ONE_DEVICE_MAX_N`) for a
#: matrix within the row's bound; beyond the bound the leg's library runs.
#: A batch is spread over the mesh (``ceil(B/P)`` whole matrices per rank,
#: in rounds bounded by the row), so it keeps the P-scaling the
#: ``distributed`` name exists for; one matrix is solved by one rank.
ONE_DEVICE = "one_device"

#: Per-op user vocabulary (requested names).  ``auto``/``off`` resolve to
#: ``native``; the rest name a concrete implementation.  ``distributed``
#: is the PLATFORM-DEFAULT distributed backend (:data:`_DISTRIBUTED_DEFAULT`);
#: it exists for ``eigh`` and ``solve_lu`` because both have exactly one
#: right distributed library per platform, so naming the library at the
#: call site is redundant and drifts (scorecard Z.2 was precisely that
#: drift).  ``cholesky`` deliberately does NOT: its CPU story is a
#: channel-policy ladder in the caller, not one library.
BACKEND_CHOICES = {
    "eigh":     ("auto", "off", "distributed", "cusolvermp", "elpa",
                 "scalapack", ONE_DEVICE),
    "cholesky": ("auto", "off", "native2d", "cusolvermp"),
    "solve_lu": ("auto", "off", "distributed", "cusolvermp", "scalapack",
                 ONE_DEVICE),
}

#: Backend names that no longer exist: {name: note}.  A request refuses with the
#: repo's one-line "`x` was removed: ..." (docs/dev/renaming.md), not a vocabulary dump.
REMOVED_BACKENDS = {
    "slate": "the SLATE backend left LORRAX on 2026-10-23 (UPGRADE_NOTES.md); use 'distributed' "
             "(CPU: scalapack, CUDA: cusolvermp, ROCm: elpa eigh / one_device LU) or "
             "'native2d' for cholesky",
}
EIGH_BACKENDS = BACKEND_CHOICES["eigh"]
CHOLESKY_BACKENDS = BACKEND_CHOICES["cholesky"]
LU_BACKENDS = BACKEND_CHOICES["solve_lu"]

#: Device vendor → the FFI platform key this package speaks: lxkit's one
#: map, one entry per leg of ``lxkit.LEGS`` (``cuda`` → ``CUDA``, ``rocm`` →
#: ``ROCM``, ``cpu``); a vendor with no leg passes through unmapped.  The
#: vendor is read from the device client (:func:`lxkit.device_vendor`).  The
#: ROCM leg has its own library (``LORRAX_FFI_ROCM_SO``,
#: ``config/frontier/build_ffi_rocm.sh``); which of this package's targets it
#: registers is :data:`_IMPL` below and ``ffi.common.families``.
FFI_PLATFORMS = FFI_PLATFORM_MAP

#: ``distributed`` → the platform's permanent default distributed backend.
#:
#: On **cpu** that is ScaLAPACK ``pzheevd``.  On **CUDA** it is cuSOLVERMp, the
#: only library there with a distributed syevd.  On **ROCM** eigh is ELPA
#: (``distrib_la._elpa``: ELPA 2026.02.001's one-stage GPU solver,
#: config/frontier/build_elpa_hip.sh), gated against the native route at P=4
#: (``ffi.common.families`` 'elpa' row); measured 5-8x faster than the library it
#: replaced at n = 20480-30720 on MI250X (ELPA_ROCM_REPORT.md).
#:
#: ROCM has NO distributed LU library: ``('solve_lu', 'ROCM')`` is the room route
#: alone (:data:`ONE_DEVICE_BY_ROOM`), so ``distributed`` LU there serves every ``n``
#: whose one-device program fits the caller's room and refuses beyond it.  One device
#: was 4-5x faster than the former 2x2 distributed LU wherever both ran: 0.72 / 1.53
#: / 4.10 s against 3.63 / 10.4 / 16.1 s at n = 8192 / 12288 / 16384 complex128, and
#: it runs to n = 28672 (22.7 s) on one GCD (ROCSOLVER_LU_REPORT.md section 2).
_DISTRIBUTED_DEFAULT = {
    ("eigh", "cpu"):      "scalapack",
    ("eigh", "CUDA"):     "cusolvermp",
    ("eigh", "ROCM"):     "elpa",
    # solve_lu: the same one-library-per-platform rule.  ScaLAPACK's
    # pXgetrf/pXgetrs on the host lib, cuSOLVERMp's batched getrf/getrs
    # on CUDA — the two backends _IMPL already lists for this op.
    ("solve_lu", "cpu"):  "scalapack",
    ("solve_lu", "CUDA"): "cusolvermp",
}

#: ``distributed`` by SIZE — (op, leg) → the largest ``n`` that
#: ``distributed`` serves with ``one_device`` (each matrix whole on one
#: device, the leg's native solver).  ONE row per leg that measured it; a
#: leg without a row keeps its library for every ``n``.  The bound is in
#: ``n`` and was measured at complex128, the larger of the two dtypes, so
#: float64 is inside it.  A caller's room still caps the route
#: (:func:`_size_route`), so a row never puts a matrix on a rank that the
#: caller found it does not fit.
#:
#: ROCM (MI250X GCD, 64 GB; pool logs ``distla_bound1``, ``distla_lubig*``):
#:   eigh 16384: the one-device program at n = 16384 complex128 takes 42.4 s
#:     (native 42.1 s; the former 2x2 library 76.4 s), peak 27.9 GB per rank,
#:     resid 9e-15.  Beyond 16384 ELPA takes over (the capacity route).  16384 is a
#:     MEASURED-COVERAGE bound, not a correctness one: jnp.linalg.eigh (rocSOLVER
#:     zheevd) is exact at n = 20480 (60.5 s, peak 35.2 GB, resid 8.4e-15) and
#:     24576 (120.2 s, 50.75 GB, 8.1e-15) complex128 (pool logs lurc_e_*).
#:   solve_lu: the CAPACITY of one GCD, a room leg (:data:`ONE_DEVICE_BY_ROOM`) as on CUDA:
#:     the largest n whose priced one-device program (``_one_device.one_device_bytes``:
#:     5 n^2 x 16 B, complex128) fits the caller's room.  There is no speed crossover below
#:     it: one-device (distrib_la.local_lu) vs the former 2x2 library, complex128,
#:     n = 8192 0.72 vs 3.63 s, 12288 1.53 vs 10.4 s, 16384 4.10 vs 16.1 s; one-device
#:     20480 6.18 s, 24576 13.3 s, 26624 14.2 s, 28672 22.7 s (peak 52.9 GB,
#:     resid <= 1.1e-14), 30720 out of memory (pool logs lurc_f_*, lurc_k_*,
#:     distla_lubigs).  The former 8192 was a correctness cap: rocBLAS trsm's
#:     in-place path is wrong past a 2^31-byte span (ROCSOLVER_LU_REPORT.md);
#:     local_lu solves such systems on a span-blocked path, so the native LU is
#:     exact at every n here.
ONE_DEVICE_MAX_N = {("eigh", "ROCM"): 16384}

#: ``one_device`` ROUNDS by leg row: the per-rank element allowance of one round (whole
#: matrices of n x max(n, width) elements up to ROUND_N^2 per rank, at least one), DECOUPLED
#: from the selection bound above so that a stack's per-rank peak never rises above main's
#: (owner rule): these are the rows main 4bae15f55 planned its rounds with.  A single matrix
#: above ROUND_N that ONE_DEVICE_MAX_N selects runs one matrix per rank per round.
ONE_DEVICE_ROUND_N = {
    ("eigh", "ROCM"):     16384,
    ("solve_lu", "ROCM"): 8192,
}


def one_device_round_n(op: str, platform: str, bound: int) -> int:
    """The rounds' bound for a selection bound ``bound``: ``min(bound, ONE_DEVICE_ROUND_N row)``
    (the bound itself on a leg without a round row, e.g. a room leg).  A shape rule."""
    r = ONE_DEVICE_ROUND_N.get((op, str(platform)))
    return int(bound) if r is None else min(int(bound), int(r))

#: ``distributed`` by ROOM -- the (op, leg) pairs whose ``one_device`` bound is not a number
#: but the caller's per-rank room: the largest ``n`` whose one whole matrix's priced program
#: (:func:`distrib_la._one_device.one_device_bytes`) fits it, and the rounds hold whole
#: matrices within the same room.  Without a room (or a zero room) the leg's library runs, so
#: no matrix is ever put on a rank that a caller has not priced; the size ceiling is the
#: library's.  CUDA: jaxlib's cuSOLVER per device against cuSOLVERMp, measured on A100-SXM4
#: (P=4 2x2, complex128, ``dla_sweep``): eigh n=312 x64 0.022 s vs 1.97 s, n=8192 x4 2.05 vs
#: 6.53 s, n=16384 x4 13.7 vs 29.0 s; solve_lu n=312 x64 0.0075 vs 0.307 s, n=8192 x4 0.44 vs
#: 1.72 s, n=16384 x4 2.88 s (cuSOLVERMp ran out of device memory); residuals <= 1.7e-14 on
#: both, cuSOLVER's LU exact through 16384 (the rocBLAS trsm defect of distrib_la.local_lu
#: does not occur there).
ONE_DEVICE_BY_ROOM = frozenset({("eigh", "CUDA"), ("solve_lu", "CUDA"), ("solve_lu", "ROCM")})
#: ``auto`` cholesky / solve_lu: the leg's FFI candidate on a true-2-D mesh
#: (else native).  CUDA only: cuSOLVERMp.  A leg without a row resolves
#: ``auto`` to native as its documented policy (no probe, no notice) -- the
#: ROCM leg used to probe the CUDA handler here and print a CUDA-specific
#: "NOT AVAILABLE" notice on every ROCm 2x2 mesh.
_AUTO_FFI = {
    ("cholesky", "CUDA"): "cusolvermp",
    ("solve_lu", "CUDA"): "cusolvermp",
}

#: What the pure-JAX backends mean per op (for messages / list_backends).
_NATIVE_IMPL = {
    "eigh":     "jnp.linalg.eigh (q-batched, every device solves its shard)",
    "cholesky": "replicated dense / sharded cholesky, owned by the caller",
    "solve_lu": "per-q jnp.linalg.solve + ridge, owned by the caller",
}

#: (op, backend) → (FFI target(s) to probe, platforms it exists on).
#: A tuple of targets means ALL of them are required.
_SPEC = {
    ("eigh", "cusolvermp"):     ("lorrax_cusolvermp_eigh",          ("CUDA",)),
    # ELPA (cpp/elpa/, config/frontier/build_elpa_hip.sh): built on the ROCM leg only.
    ("eigh", "elpa"):           ("lorrax_elpa_eigh",        ("ROCM",)),
    ("eigh", "scalapack"):      ("lorrax_scalapack_eigh",           ("cpu",)),
    ("cholesky", "cusolvermp"): ("lorrax_cusolvermp_batched_potrf", ("CUDA",)),
    ("solve_lu", "cusolvermp"): (("lorrax_cusolvermp_batched_solve_lu_factors",
                                    "lorrax_cusolvermp_batched_getrf",
                                    "lorrax_cusolvermp_batched_getrs"),
                                   ("CUDA",)),
    # The scalapack LU family is THREE handlers since the transverse
    # factor hoist (2026-08): the fused solve (legacy callers) plus the
    # split getrf/getrs pair the hoisted ζ factor stage requires.  All
    # three are probed — an old host .so without the pair refuses at
    # resolve time instead of dying mid-run inside the factor stage.
    ("solve_lu", "scalapack"):  (("lorrax_scalapack_batched_solve_lu_factors",
                                  "lorrax_scalapack_batched_getrf",
                                  "lorrax_scalapack_batched_getrs"),
                                 ("cpu",)),
}


def mesh_platform(mesh_xy) -> str:
    """The FFI platform key for the mesh's devices.

    ``"cpu"`` / ``"CUDA"`` / ``"ROCM"`` (a :data:`lxkit.legs.LEGS` key) from
    the device vendor; anything else passes through UNMAPPED so a refusal can name what it actually saw
    (``'tpu'`` reads better than ``'unknown'``).
    """
    return mesh_ffi_platform(mesh_xy, FFI_PLATFORMS)


def mesh_key(mesh_xy) -> tuple:
    """A STABLE, hashable identity for a mesh: axes, extents, devices.

    THE cache key to use wherever the cached value does NOT retain the mesh.
    ``id(mesh)`` is only safe when it does — then the entry keeps the Mesh
    alive and the id cannot be recycled — and the failure mode when it does
    not is a stale HIT, not a redundant miss.

    Device IDENTITY, not just shape.  Two meshes with the same axis names
    and extents over DIFFERENT devices (a GPU 1x1 and a CPU 1x1 in one
    process) lower to different platform handlers, so a shape-only key
    aliases them into each other's compiled kernels.

    Public because LORRAX needs it and there is nowhere else honest to get
    it: ``isdf/core.py`` keys two announcement sets (which do not retain
    their mesh) and used to import a backend-private ``_mesh_key`` — a
    reach past this package's public API into a backend-private module, for a
    function that is not private in any real sense.  Three
    of this package's four backends had their own copy of it; they all call
    this one now.
    """
    return (tuple(mesh_xy.axis_names),
            tuple(int(s) for s in mesh_xy.shape.values()),
            mesh_xy.devices.flat[0].platform,
            tuple(d.id for d in mesh_xy.devices.flat))


def mesh_is_cpu(mesh_xy) -> bool:
    """True when the mesh's devices are the CPU backend.

    Keeps every ``auto`` policy from selecting the CUDA-only cuSOLVERMp
    paths on a host-only run.  Never raises (a malformed mesh reads as
    non-CPU, i.e. no special-casing).
    """
    try:
        return bool(mesh_xy.devices.flat[0].platform == "cpu")
    except Exception:                                          # noqa: BLE001
        return False


def _vocab_error(op: str, requested) -> str:
    if requested in REMOVED_BACKENDS:
        return f"`{requested}` was removed: {REMOVED_BACKENDS[requested]}"
    return (f"{op} backend must be one of "
            f"{'|'.join(BACKEND_CHOICES[op])}, got {requested!r}")


def _mesh_shape(mesh_xy) -> tuple[int, int]:
    return int(mesh_xy.shape["x"]), int(mesh_xy.shape["y"])


def _process_count() -> int:
    import jax
    try:
        return int(jax.process_count())
    except Exception:                                          # noqa: BLE001
        return 1


def _process_index() -> int:
    import jax
    try:
        return int(jax.process_index())
    except Exception:                                          # noqa: BLE001
        return 0


def _platform_phrase(platforms: tuple[str, ...]) -> str:
    """"CUDA-only" / "host-only" / "available on CUDA, cpu, rocm only".

    The two-way phrasing this replaces produced a NONSENSE message on any
    platform that was neither: a backend declared on ``('CUDA', 'cpu')``
    was described as "host-only" to a caller on an unknown platform, which
    is both false and unactionable.  Say what the row says.
    """
    if platforms == ("CUDA",):
        return "CUDA-only"
    if platforms == ("cpu",):
        return "host-only"
    return f"available on {', '.join(platforms)} only"


def _check_geometry(op: str, backend: str, px: int, py: int) -> None:
    """Guard 5 — mesh-geometry constraints, per (op, backend).  Raises
    ValueError with the reason; silent-pass otherwise."""
    if op == "eigh" and backend in ("cusolvermp", "elpa"):
        # These FFI eigh wrappers reject p != q; for cusolvermp this is
        # a DEADLOCK guard (cusolverMpSyevd hangs in a collective on
        # rectangular one-tile-per-rank blocks — observed 4x1/1x4,
        # 2026-07-10).
        # ScaLAPACK pXheevd is NOT in this class: like pXgetrf it only
        # needs square descriptor BLOCKS, so square-or-1-D is enough (its
        # rule is the ``scalapack`` branch below).
        if px != py:
            raise ValueError(
                f"eigh backend {backend!r} needs a SQUARE mesh "
                f"(cusolverMpSyevd DEADLOCKS on rectangular blocks; ELPA's wrapper relabels one tile edge for "
                f"both axes); got {px}x{py}.")
    elif backend == "scalapack":
        # Mirrors _scalapack.validate_eigh_mesh / the LU family's own check
        # — keep them in sync (bug L-1 again).
        if px > 1 and py > 1 and px != py:
            _routine = "pXheevd" if op == "eigh" else "pXgetrf"
            raise ValueError(
                f"{op} backend 'scalapack': mesh {px}x{py} unsupported — "
                f"{_routine} needs square descriptor blocks (MB == NB), "
                f"which the one-tile-per-rank layout only gives on square "
                f"or 1-D meshes.")


#: (op, px, py) triples whose auto→native geometry demote has already been
#: announced this process — resolve_backend runs per plan() call and the
#: announcement is per-decision, not per-call.
_AUTO_GEOMETRY_DEMOTE_ANNOUNCED: set[tuple[str, int, int]] = set()

#: (op, target) pairs whose auto→native CAPABILITY demote has already been
#: announced this process.
_AUTO_CAPABILITY_DEMOTE_ANNOUNCED: set[tuple[str, str]] = set()


def _announce_auto_geometry_demote(op: str, px: int, py: int) -> None:
    """Rank-0, once-per-(op, geometry) announcement that ``auto`` demoted a
    compiled cusolvermp to native because the mesh is 1-D ('auto' may
    demote but must announce).  Mesh geometry cannot differ per rank, so
    rank 0 speaks for all."""
    key = (op, int(px), int(py))
    if key in _AUTO_GEOMETRY_DEMOTE_ANNOUNCED:
        return
    _AUTO_GEOMETRY_DEMOTE_ANNOUNCED.add(key)
    if _process_index() == 0:
        _LOG.info(
            f"  [distrib_la.resolve] {op} backend 'auto': cusolvermp is "
            f"compiled but the {px}x{py} mesh is 1-D (block-cyclic 2-D "
            f"layout needs px >= 2 and py >= 2) — demoting to native.  "
            f"An explicit 'cusolvermp' request on this mesh refuses "
            f"instead.")


def _announce_auto_capability_demote(op: str, target: str, platform: str) -> None:
    """Rank-0, once-per-(op, target) announcement that ``auto`` demoted to
    native because the handler is NOT LOADABLE — the mesh geometry was fine.

    Why this exists (2026-08-05, Perlmutter port).  This arm used to be the
    one silent demote left in the resolver: ``has_target()`` swallows every
    reason it can answer "no" for (library missing, library present but
    unloadable, library loaded but the symbol absent from a partial build),
    so a stale or half-built ``liblorrax_ffi.so`` on a true-2-D GPU mesh
    quietly ran the replicated JAX path with no print, no ledger entry and
    no exception.  Only the startup report witnessed it.

    That is precisely the failure class that produced the worst measured
    defect in this tree — a full MoS2 12x12 G0W0 that ran to completion
    with rc=0 and a QP gap of -161 eV after a config rewrite silently took
    the only route carrying the cure.  We quote ``probe_target``'s
    three-way reason because it names the ``.so`` and distinguishes "not
    built" from "built but unloadable" — completely different fixes.
    """
    key = (op, target)
    if key in _AUTO_CAPABILITY_DEMOTE_ANNOUNCED:
        return
    _AUTO_CAPABILITY_DEMOTE_ANNOUNCED.add(key)
    if _process_index() != 0:
        return
    try:
        reason = loader.probe_target(target, platform)
    except Exception as exc:               # noqa: BLE001 — never let the
        reason = f"(probe_target itself failed: {exc!r})"   # witness raise
    _LOG.warning(
        f"  [distrib_la.resolve] {op} backend 'auto': the mesh geometry "
        f"supports '{target}' but the handler is NOT AVAILABLE — "
        f"demoting to the replicated native path.  Reason: {reason}.  "
        f"This is a PERFORMANCE AND MEMORY cliff, not a correctness "
        f"fallback; an explicit backend request refuses here instead.  "
        f"Fix the FFI build or set the backend explicitly to make this "
        f"fail loudly.",
        # the ``common.log.message`` extra, by hand: a service imports no ``common``
        extra={"lorrax": {"id": "distrib_la_auto_demote",
                          "fix": "fix the FFI build, or set the backend explicitly"}})


#: Below this ``n``, an EXPLICIT distributed eigh gets the capacity-route
#: notice below.  It is the order of ``n`` at which one 40 GB device stops
#: holding an n x n complex128 matrix, its eigenvectors and workspace; being
#: wrong costs a line of stdout, not a route.
_EIGH_DISTRIBUTED_ANNOUNCE_BELOW = 16384


def _announce_eigh_fixed_cost(px: int, py: int, n: int) -> None:
    """Rank-0, once-per-(op, geometry): the distributed eigh is a capacity route.

    NOT a demote and never a demote.  The caller asked for cuSOLVERMp
    explicitly (or via ``distributed``, which is explicit); promise
    semantics (module docstring, doctrine C7) say they get exactly that.
    This prints the advice and returns.

    The notice quotes no timing.  Provenance of the measurements behind the
    advice: an older cuSOLVERMp (Perlmutter jobid 56447670, 2x2 A100,
    complex128, warm medians) cost ~1.55 s per matrix at n = 64..2048,
    almost all inside ``cusolverMpSyevd``, where the replicated native path
    took milliseconds.  The deployed cuSOLVERMp 0.9.1 (2026-09-30, lane
    EIGHBATCH) took 0.17-0.45 s per first call at n = 16..800 on 2x2 and
    0.28-0.98 s at n = 32..1024 on 4x4, each timing one cold call including
    plan and compile; no warm table exists for 0.9.1.  Neither set covers a
    tile too large for one device or a multi-node mesh, which is the regime
    ``distributed`` eigh exists for.
    """
    announce_once(
        ("distrib_la.resolve", "eigh", "cusolvermp", "fixed-cost", px, py),
        f"  [distrib_la.resolve] eigh backend 'cusolvermp' on the {px}x{py} "
        f"mesh: HONORED (explicit requests are never demoted).  Notice, "
        f"once per mesh geometry, at n={int(n)}: 'distributed' eigh is for "
        f"CAPACITY (a tile too large for one device, or a multi-node mesh), "
        f"not for speed.  A matrix that fits one device pays the library's "
        f"per-call charge; the replicated native path (backend 'off'/'auto') "
        f"is the faster choice there.", scope="rank0")


def resolve_backend(op: str, requested: str, mesh_xy, *,
                    n: int | None = None,
                    compute_evecs: bool = True,
                    size_route: bool = True,
                    room: int | None = None) -> str:
    """Resolve a requested backend name for ``op`` on ``mesh_xy``.

    ``distributed`` takes the leg's size route first (guard 1b,
    :data:`ONE_DEVICE_MAX_N`); :func:`size_route_note` says what it decided.
    ``size_route=False`` resolves ``distributed`` to the leg's library whatever
    ``n`` (the capacity provider, :func:`resolve_eigh_capacity`).  ``room``
    (bytes per rank beside the caller's live set, ``plan(budget_bytes=)``) caps
    the size route too: ``one_device`` only where one whole matrix's priced
    program (:func:`distrib_la._one_device.one_device_bytes`) fits it.

    Returns the concrete backend to run: ``"native"``, ``"native2d"``, or
    one of the FFI backend names (``"cusolvermp" | "elpa" | "scalapack"``).
    A returned FFI name is a *promise*: its handler is compiled into the
    loaded library and every mesh guard passed — the subsequent call cannot
    fail for an availability/geometry reason.

    Raises ``ValueError`` (bad name / platform / coverage / geometry /
    divisibility) or ``RuntimeError`` (backend not compiled into the FFI
    library) at RESOLVE time, with a message that names the failed guard
    and what is available instead.

    Parameters
    ----------
    op
        One of :data:`OPS`.
    requested
        A name from ``BACKEND_CHOICES[op]`` (``"native"`` is accepted as
        an alias of ``"off"``, so already-resolved names round-trip).
    mesh_xy
        The ('x','y') device mesh the op will run on.
    n
        Optional matrix extent, DECOUPLED from the operands on purpose:
        a caller that will pad to ``n`` can ask about ``n`` before it has
        built anything (``gw.sc_iteration`` does exactly that).  When
        given, the FFI backends additionally require ``n`` divisible by
        both mesh axes (guard 6) and ``native2d`` a legal tile
        decomposition.
    compute_evecs
        ``eigh`` only: whether the caller wants eigenVECTORS as well as
        eigenvalues.  ``False`` is a documented parameter of both eigh
        wrappers and is BROKEN on cuSOLVERMp 0.7.2 — guard 2c refuses it
        (bug L-3).  Defaulted True because that is what LORRAX wants
        everywhere (owner, 2026-08-07); a caller that does not pass it is
        unaffected by the guard.

    Notes
    -----
    ``auto`` policy — per op, mirroring the measured production defaults:

    * ``eigh``: always ``native``.  The q-batched jnp path solves ndev
      matrices concurrently; the FFI path solves one matrix ndev-ways and
      walks the batch serially, measured hundreds of times slower per
      fit-size matrix.  ``auto`` means "the fastest eigh for this call",
      and for a batch of tiles that fit on one device that is still the
      native one.  **``distributed`` is the name for "spread ONE tile over
      the mesh"**, and its CPU default is permanently ScaLAPACK
      ``pzheevd`` — the only way out of the replicated factor's
      O(nq·μ³)-with-no-P-scaling wall.
    * ``cholesky`` / ``solve_lu``: ``cusolvermp`` on a true-2D
      (px>=2 and py>=2) CUDA mesh when its handler is compiled, else
      ``native`` (announced on rank 0 when the demote is geometry-driven,
      i.e. the handler IS compiled but the mesh is 1-D).  Never an FFI
      backend on a CPU mesh.  An EXPLICIT ``cusolvermp`` (or
      ``distributed`` resolving to it) on a 1-D mesh refuses at guard 5
      instead of demoting.
    """
    if op not in BACKEND_CHOICES:
        raise ValueError(f"unknown linalg op {op!r} (known: {'|'.join(OPS)})")
    requested = ("auto" if requested is None else str(requested)).strip().lower()
    if requested == NATIVE:
        requested = "off"
    if requested not in BACKEND_CHOICES[op]:
        raise ValueError(_vocab_error(op, requested))

    if requested == "off":
        return NATIVE

    px, py = _mesh_shape(mesh_xy)

    if requested == NATIVE2D:
        # Pure JAX: no library to probe, no one-process-per-device rule,
        # every platform.  Its one real constraint is that the matrix must
        # decompose into tiles the mesh divides, and that is checked HERE
        # rather than at call time for the same reason every other guard is
        # (bug L-1: a rule enforced only at call time turns a returned
        # backend name into a broken promise).
        if n is not None:
            from distrib_la._native2d import block_size_for
            block_size_for(int(n), px, py)          # raises with the reason
        return NATIVE2D

    requested_spelling = requested   # keep the user's spelling for messages
    if requested == "distributed" and size_route:
        sized = _size_route(op, mesh_xy, n, room)
        if sized is not None:
            requested = sized
    if requested == ONE_DEVICE:
        return _resolve_one_device(op, mesh_xy, n, requested_spelling, room)
    if requested == "distributed":
        # "Spread ONE tile over the whole mesh, with whatever library this
        # platform's distributed eigh is."  Explicit — every guard below
        # applies and a failure raises rather than downgrading, because a
        # silent downgrade to the replicated path is exactly the O(nq·μ³)
        # no-P-scaling wall this name exists to escape.
        requested = _DISTRIBUTED_DEFAULT.get((op, mesh_platform(mesh_xy)))
        if requested is None:
            platform = mesh_platform(mesh_xy)
            if (op, platform) in ONE_DEVICE_BY_ROOM:
                from distrib_la._one_device import one_device_bytes
                need = ("" if n is None else
                        f": n={int(n)} needs {one_device_bytes(op, int(n)) / 1e9:.2f} GB/rank")
                raise ValueError(
                    f"{op} on {platform} has no distributed library, only one whole matrix per "
                    f"device within the caller's room (budget_bytes"
                    f"{'; none given' if room is None else ''}){need}; use fewer ISDF "
                    f"centroids, more device memory per rank, or a CUDA run.")
            raise ValueError(
                f"{op} backend 'distributed' is not available on platform {platform!r} "
                f"(it is defined for {sorted(_DISTRIBUTED_DEFAULT)}); use backend 'native'.")
    if requested == "auto":
        if op == "eigh" or mesh_is_cpu(mesh_xy):
            return NATIVE
        platform = mesh_platform(mesh_xy)
        candidate = _AUTO_FFI.get((op, platform))
        if candidate is None:
            return NATIVE                  # this leg's policy: no FFI auto
        target, _ = _SPEC[(op, candidate)]
        if px >= 2 and py >= 2 and loader.has_target(target, platform):
            return candidate
        # 'auto' MAY demote, but must announce.  The only demote here that
        # is not the documented default policy ("native unless a compiled
        # cusolvermp on a true-2D CUDA mesh") is the geometry one: handler
        # compiled, mesh 1-D.  Mesh geometry is identical on every rank, so
        # a rank-0 line suffices; deduped so repeated plan() calls do not
        # spam.
        if loader.has_target(target, platform) and not (px >= 2 and py >= 2):
            _announce_auto_geometry_demote(op, px, py)
        elif px >= 2 and py >= 2:
            # Geometry was fine, so the ONLY reason we are here is that the
            # handler could not be resolved.  See the docstring of
            # _announce_auto_capability_demote for why this used to be
            # silent and why that was the highest-consequence failure mode
            # in the resolver.
            _announce_auto_capability_demote(op, target, platform)
        return NATIVE

    # ── explicit FFI backend: run the full guard ladder ──────────────
    platform = mesh_platform(mesh_xy)
    target, platforms = _SPEC[(op, requested)]

    # 2. platform
    if platform not in platforms:
        raise ValueError(
            f"{op} backend {requested!r} is {_platform_phrase(platforms)} "
            f"but the mesh devices are {platform!r}.  "
            f"Available {op} backends on this mesh: "
            f"{', '.join(_available(op, mesh_xy))}.")

    # 2c. KNOWN-BROKEN combination (bug L-3, found by the step-6 eigh
    # investigation).  ``compute_evecs=False`` (jobz='N') is a DOCUMENTED
    # parameter of _cusolvermp.distributed_eigh and it does not work:
    # cuSOLVERMp 0.7.2 returns status=7 (CUSOLVER_STATUS_INTERNAL_ERROR)
    # from cusolverMpSyevd at EVERY n tried — 64, 256, 1024, 4096 — on a
    # real 4-process 2x2 CUDA mesh.  Reproduced twice (jobid 56447670, both
    # the plain and the LORRAX_FFI_PROFILE=1 decomposition legs;
    # _reports/perf_gpu2x2_decomp.json + perf_gpu2x2prof_decomp.json, rows
    # "g2. distributed_eigh jobz='N'").
    #
    # NOT OURS.  cusolverMpSyevd_bufferSize SUCCEEDS with jobz='N' and the
    # wrapper is a straight pass-through (src/ffi/cpp/cusolvermp/
    # eigh_ffi.cc:106, `const char jobz = compute_evecs ? 'V' : 'N'`), so
    # the library sizes the eigenvalues-only solve and then fails it.  No
    # workaround flag was found and the library bug is NOT CHASED
    # (deliberate scope call).
    #
    # A REFUSAL, PERMANENTLY — not a stopgap.  The owner confirms
    # (2026-08-07) that LORRAX wants compute_evecs=True in every case they
    # can think of, so the value of this parameter on this backend is
    # exactly "a way to get an unexplained INTERNAL_ERROR three call
    # frames deep".  Refusing at resolve time turns that into a message
    # naming the library, the status code and the artifact.
    if op == "eigh" and requested == "cusolvermp" and not compute_evecs:
        raise RuntimeError(
            "eigh backend 'cusolvermp' with compute_evecs=False is REJECTED: "
            "cuSOLVERMp 0.7.2's cusolverMpSyevd returns status=7 "
            "(CUSOLVER_STATUS_INTERNAL_ERROR) for jobz='N' at every n "
            "measured (64/256/1024/4096, real 4-process 2x2 CUDA mesh, "
            "jobid 56447670; _reports/perf_gpu2x2_decomp.json rows "
            "\"g2. distributed_eigh jobz='N'\").  bufferSize succeeds and "
            "the wrapper only forwards the flag, so this is a library "
            "defect, not a LORRAX one.  Use compute_evecs=True (the "
            "default, and what LORRAX wants everywhere) and ignore the "
            "eigenvectors, or 'off'/'auto' for jnp.linalg.eigvalsh.")

    # 3. compiled capability
    #
    # ``probe_target`` (not ``has_target``) because the REASON matters to
    # whoever reads this: "the .so would not load" and "the .so has no
    # such handler" have completely different fixes, and reporting the
    # first as the second sends people to rebuild a library that is fine.
    # That happened (wk_P G4, 2026-07-25): a refusal said "not compiled into
    # the cpu FFI library" while `nm -D` showed the handler present — the real
    # cause was an incomplete LD_LIBRARY_PATH.  ``target`` may be one name or a tuple
    # of REQUIRED names (the scalapack LU family: fused solve + split
    # getrf/getrs) — every one must probe usable, and the refusal names
    # the one that failed.
    for _t in (target if isinstance(target, tuple) else (target,)):
        usable, why = loader.probe_target(_t, platform)
        if not usable:
            raise RuntimeError(
                f"{op} backend {requested!r} requested but its FFI handler "
                f"({_t}) is not usable: {why}  "
                f"Available {op} backends on this mesh: "
                f"{', '.join(_available(op, mesh_xy))}.")

    # 4. process coverage (one JAX process per device)
    ndev = int(mesh_xy.devices.size)
    nproc = _process_count()
    if nproc != ndev:
        raise ValueError(
            f"{op} backend {requested!r} needs ONE JAX process per mesh "
            f"device (its MPI/NCCL context is per-process), but the "
            f"{px}x{py} mesh has {ndev} devices across {nproc} "
            f"process(es).  Use backend 'off' (native) on host-device "
            f"meshes, or relaunch with {ndev} processes.")

    # 5. geometry
    if requested == "cusolvermp" and op in ("cholesky", "solve_lu"):
        # cuSOLVERMp's block-cyclic 2-D layout degenerates on a 1-D mesh.
        # This used to ``return NATIVE`` silently — a promise-semantics
        # violation that forced the one strict consumer to re-check
        # ``p.is_native`` after planning.  Explicit requests refuse HERE;
        # 'auto' demotes with an announcement above.
        if not (px >= 2 and py >= 2):
            _via = ("" if requested_spelling == "cusolvermp"
                    else f" (requested as {requested_spelling!r})")
            raise ValueError(
                f"{op} backend 'cusolvermp'{_via} needs a true-2D mesh "
                f"(px >= 2 and py >= 2): its block-cyclic 2-D layout "
                f"degenerates on the {px}x{py} mesh.  Use a true-2D mesh, "
                f"or 'auto' (demotes to native with an announcement), or "
                f"'off' for the native path.")
    _check_geometry(op, requested, px, py)

    # 6. divisibility
    if n is not None and (int(n) % px or int(n) % py):
        raise ValueError(
            f"{op} backend {requested!r} needs n ({n}) divisible by both "
            f"mesh axes ({px}, {py}) — the one-tile-per-rank block-cyclic "
            f"layout has no ragged tiles.  Pad n or change the mesh.")

    # ── the resolution SUCCEEDED; say what it costs ──────────────────
    # Strictly after every guard, so a refusal is never preceded by a cost
    # notice about a route the caller is not going to get.  Only when ``n``
    # is known: with n unknown there is no size to compare against the
    # crossover, and guessing would make ``list_backends`` (which resolves
    # every backend with n=None to build a startup report) print this once
    # per report.
    if (op == "eigh" and requested == "cusolvermp" and n is not None
            and int(n) < _EIGH_DISTRIBUTED_ANNOUNCE_BELOW):
        _announce_eigh_fixed_cost(px, py, int(n))

    return requested


#: Every size-route decision of this process, (op, leg, px, py, n) -> one line
#: for a driver's route ledger (read through :func:`distrib_la.new_stack_routes`:
#: production drivers discard component stdout, so the rank-0 notice alone
#: would not reach a run's log).
_SIZE_DECISIONS: dict = {}


def one_device_max_n(op: str, platform: str, room: int | None = None) -> int | None:
    """The leg's ``one_device`` bound for ``op``, or None (the leg's library).

    A size row (:data:`ONE_DEVICE_MAX_N`) is the bound.  A room leg
    (:data:`ONE_DEVICE_BY_ROOM`) derives it from the caller's per-rank ``room``: the
    largest ``n`` with ``one_device_bytes(op, n) <= room``; no room, no bound.  A shape
    rule on a value every rank shares: no exchange, no sync."""
    bound = ONE_DEVICE_MAX_N.get((op, str(platform)))
    if bound is not None:
        return bound
    if (op, str(platform)) in ONE_DEVICE_BY_ROOM and room is not None:
        import math
        from distrib_la._one_device import one_device_bytes
        from distrib_la.plan import room_test
        n = room_test(lambda r: math.isqrt(int(r) // one_device_bytes(op, 1)) if int(r) > 0 else 0, room)
        return n if n >= 1 else None
    return None


def _size_route(op: str, mesh_xy, n, room: int | None = None) -> str | None:
    """Guard 1b: ``one_device`` when the leg has a size row, ``n`` is known and
    within it, and (given a ``room``) one whole matrix's priced program fits that
    room; None (the leg's library) otherwise.  Announced once per decision.

    The room cap is the capacity half of the route: ``one_device`` holds whole
    matrices on a rank, exactly what a caller that hands ``distributed`` a room
    has found does not fit (``lxkit.memory.admit``, memory-model.md#routes).
    A shape price against a caller value every rank shares: no exchange, no sync."""
    platform = mesh_platform(mesh_xy)
    bound = one_device_max_n(op, platform, room)
    if bound is None or n is None:
        return None
    px, py = _mesh_shape(mesh_xy)
    # A room leg with no library (ROCM LU) refuses past the room, below.
    library = _DISTRIBUTED_DEFAULT.get((op, platform)) or "no library (refused)"
    chosen = ONE_DEVICE if int(n) <= bound else library
    priced = ""
    if chosen == ONE_DEVICE and room is not None:
        from distrib_la._one_device import one_device_bytes
        need = one_device_bytes(op, int(n))
        from distrib_la.plan import room_test
        if room_test(lambda r: need > int(r), room):
            chosen = library
        priced = (f"; one whole matrix {need / 1e9:.2f} GB/rank "
                  f"{'over' if chosen != ONE_DEVICE else 'within'} room {int(room) / 1e9:.2f} GB")
    _SIZE_DECISIONS.setdefault(
        (op, platform, px, py, int(n), None if room is None else int(room)),
        f"{op} 'distributed' n={int(n)} on the {px}x{py} {platform} mesh: {chosen}, "
        f"size route ({'room' if (op, platform) in ONE_DEVICE_BY_ROOM else 'ONE_DEVICE_MAX_N'}: "
        f"n <= {bound} -> {ONE_DEVICE}, larger -> {library}"
        f"{priced})")
    announce_once(
        ("distrib_la.resolve", op, "size-route", platform, px, py, chosen),
        f"  [distrib_la.resolve] {op} 'distributed' on the {px}x{py} {platform} "
        f"mesh: size route ({'the room' if (op, platform) in ONE_DEVICE_BY_ROOM else 'leg row ONE_DEVICE_MAX_N'}, n <= {bound} -> "
        f"'{ONE_DEVICE}', larger -> '{library}'): n={int(n)} -> '{chosen}'.  "
        f"'{ONE_DEVICE}' solves each matrix whole on one device with the native "
        f"kernel and returns the caller's P(None,'x','y') layout; an explicit "
        f"'{library}' is honoured as named{priced}.", scope="rank0")
    return ONE_DEVICE if chosen == ONE_DEVICE else None


def size_route_note(op: str, requested: str, mesh_xy, n, backend: str,
                    room: int | None = None) -> str | None:
    """The plan record of guard 1b: what the size route decided, or None when it
    did not apply (no leg row, ``n`` unknown, or a request other than ``distributed``)."""
    if str(requested).strip().lower() != "distributed":
        return None
    platform = mesh_platform(mesh_xy)
    bound = one_device_max_n(op, platform, room)
    if bound is None or n is None:
        return None
    rel = "<=" if int(n) <= bound else ">"
    over = " (one whole matrix over the room)" if int(n) <= bound and backend != ONE_DEVICE else ""
    return f"size route {platform}: n={int(n)} {rel} {bound} -> {backend}{over}"


def _resolve_one_device(op: str, mesh_xy, n, spelling: str, room: int | None = None) -> str:
    """The ``one_device`` guards: a leg row (its bound plans the rounds), the bound,
    and divisibility (route (c) moves face tiles).  Refusals name GATE one_device."""
    platform = mesh_platform(mesh_xy)
    bound = one_device_max_n(op, platform, room)
    via = "" if spelling == ONE_DEVICE else f" (requested as {spelling!r})"
    if bound is None:
        raise ValueError(
            f"GATE one_device: {op} backend '{ONE_DEVICE}'{via} has no bound on "
            f"platform {platform!r}: no size row (ONE_DEVICE_MAX_N rows: "
            f"{sorted(ONE_DEVICE_MAX_N)}) and no room for a room leg "
            f"(ONE_DEVICE_BY_ROOM: {sorted(ONE_DEVICE_BY_ROOM)}); its per-device bound is "
            f"a measured leg fact or the caller's room.  Use 'off' (native) or 'distributed'.")
    px, py = _mesh_shape(mesh_xy)
    if n is not None and int(n) > bound:
        raise ValueError(
            f"GATE one_device: {op} backend '{ONE_DEVICE}'{via} at n={int(n)} exceeds "
            f"the {platform} bound {bound} (ONE_DEVICE_MAX_N or the room).  Use 'distributed'.")
    if n is not None and (int(n) % px or int(n) % py):
        raise ValueError(
            f"GATE one_device: {op} backend '{ONE_DEVICE}'{via} needs n ({n}) "
            f"divisible by both mesh axes ({px}, {py}): the operand arrives as face "
            f"tiles.  Pad n or change the mesh.")
    return ONE_DEVICE


#: (platform, px, py) triples whose missing capacity eigh was announced.
_CAPACITY_UNAVAILABLE_ANNOUNCED: set[tuple[str, int, int]] = set()


def resolve_eigh_capacity(mesh_xy, *, n: int | None = None) -> tuple[str, str | None]:
    """The whole-mesh eigh behind an ``auto`` request that carries a room.

    ``plan('eigh', backend='auto', budget_bytes=room)`` asks for the local
    route wherever it fits and the capacity route (one matrix over the whole
    mesh) only for a stack that does not.  The capacity route is the
    platform's ``distributed`` backend (:data:`_DISTRIBUTED_DEFAULT`),
    resolved HERE through the full guard ladder, exactly as an explicit
    ``distributed`` request is.  Returns ``(backend, None)`` when it passes,
    so such a plan is the same plan an explicit ``distributed`` request
    builds.  When it does not (no library on this leg, a partial build,
    geometry, divisibility), returns ``(NATIVE, reason)``: the plan runs
    route (c) and refuses by name at the first stack that needs the whole
    mesh (``GATE eigh_capacity_route``, :meth:`distrib_la.Plan.require_capacity`).
    That is the promise of ``auto``: it may choose, and it announces (rank 0,
    once per platform and geometry); it never runs a stack on a route the
    caller did not ask for.  An explicit ``distributed`` still refuses at
    resolve time.
    """
    try:
        # The LIBRARY, never the size route: this provider runs only the
        # stacks of which not one whole matrix per rank fits the caller's
        # room, and the one-device route would need exactly that.
        return resolve_backend("eigh", "distributed", mesh_xy, n=n,
                               size_route=False), None
    except (ValueError, RuntimeError) as exc:
        # The first line names the failed guard; a loader's search path follows it.
        reason = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
    px, py = _mesh_shape(mesh_xy)
    key = (str(mesh_platform(mesh_xy)), px, py)
    if key not in _CAPACITY_UNAVAILABLE_ANNOUNCED:
        _CAPACITY_UNAVAILABLE_ANNOUNCED.add(key)
        if _process_index() == 0:
            _LOG.info(
                f"  [distrib_la.resolve] eigh 'auto' on the {px}x{py} "
                f"{key[0]} mesh: no capacity (whole-mesh) eigh here ({reason}).  "
                f"Stacks run whole matrices per rank (route (c)) where they "
                f"fit the room; a stack that does not refuses "
                f"(GATE eigh_capacity_route).")
    return NATIVE, reason


def _available(op: str, mesh_xy) -> list[str]:
    """Names of the backends that would pass guards 2–4 for ``op`` on this
    mesh (the pure-JAX ones always qualify)."""
    out = [NATIVE] + ([NATIVE2D] if NATIVE2D in BACKEND_CHOICES[op] else [])
    platform = mesh_platform(mesh_xy)
    if (op, platform) in ONE_DEVICE_MAX_N or (op, platform) in ONE_DEVICE_BY_ROOM:
        out.append(ONE_DEVICE)
    for (o, b), (target, platforms) in _SPEC.items():
        if o != op or platform not in platforms:
            continue
        names = target if isinstance(target, tuple) else (target,)
        if all(loader.has_target(t, platform) for t in names):
            out.append(b)
    return out


def list_backends(op: str, mesh_xy) -> dict[str, str]:
    """Introspection: every concrete backend of ``op`` mapped to a status
    string — ``"available"``/``"available (…)"`` or ``"unavailable: <first
    failed guard>"`` for this mesh.

    NEVER raises for an availability reason (an unknown ``op`` still does):
    this is what a startup report calls, and a report that can take the run
    down is worse than no report.
    """
    if op not in BACKEND_CHOICES:
        raise ValueError(f"unknown linalg op {op!r} (known: {'|'.join(OPS)})")
    out = {NATIVE: f"available ({_NATIVE_IMPL[op]})"}
    for backend in BACKEND_CHOICES[op]:
        if backend in ("auto", "off"):
            continue
        try:
            resolved = resolve_backend(op, backend, mesh_xy)
            out[backend] = ("available" if resolved == backend
                            else f"available (resolves to {resolved} on this mesh)")
        except (ValueError, RuntimeError) as exc:
            out[backend] = f"unavailable: {exc}"
    return out


def backend_module(backend: str):
    """THE import seam for the backend implementation modules.

    Everything that needs a backend implementation goes through here (after
    :func:`resolve_backend` said it is usable), so one place knows the
    package layout::

        mod = backend_module("cusolvermp")
        L = mod.batched_distributed_cholesky(C_q, mesh=mesh)

    ``native`` is the one backend with NO module: it is ``jnp.linalg.eigh``
    plus the caller's own channel-policy routes, which is exactly what
    "native" means.  Everything else — including the pure-JAX ``native2d``
    — has an implementation module, and this is where its name lives.
    """
    if backend == NATIVE2D:
        from distrib_la import _native2d
        return _native2d
    if backend == ONE_DEVICE:
        from distrib_la import _one_device
        return _one_device
    if backend == "cusolvermp":
        from distrib_la import _cusolvermp
        return _cusolvermp
    if backend == "elpa":
        from distrib_la import _elpa
        return _elpa
    if backend == "scalapack":
        from distrib_la import _scalapack
        return _scalapack
    raise ValueError(
        f"no FFI backend module named {backend!r} "
        f"(known: cusolvermp, elpa, scalapack, native2d, one_device; 'native' is "
        f"jnp + the caller's own routes and has no module).")

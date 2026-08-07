"""Backend resolution — the ONE guard ladder.

A requested backend name (``auto | off | distributed | native2d |
cusolvermp | slate | scalapack``, per op) becomes a concrete,
*guaranteed-callable* backend, or a clear resolve-time error.  Every guard
lives here, applied in ONE fixed order — SEVEN steps, all at resolve time:

    1. vocabulary   — is the name a backend of this op at all?
                      (``distributed`` resolves to the platform's default
                      distributed library FIRST, then runs the whole
                      ladder as if it had been named explicitly)
    2. platform     — does the backend run on this mesh's device kind?
                      (cusolvermp is CUDA-only; scalapack is host-only)
    2b. known-broken— slate eigh on a CPU mesh (bug L-2: SIGSEGV)
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
                      instead of returning an error), SLATE's
                      square-or-N×1 tile rule + the 1×q stride-assert
                      guard (both mirroring ``_slate.validate_tile_layout``),
                      ScaLAPACK's square-block descriptor requirement
    6. divisibility — ``n`` divisible by both mesh axes (only checked
                      when the caller passes ``n``)

PROMISE SEMANTICS ARE LAW.  A returned FFI name means every guard passed:
the subsequent call cannot fail for an availability or geometry reason.
Explicit requests REFUSE; only ``auto`` may demote, and it announces on
rank 0.  The worst measured defect in this tree — a full MoS2 12×12 G0W0
that ran to completion with rc=0 and a QP gap of −161 eV — was a silent
route change, so a silent demote is not a convenience here, it is the
failure mode.

``auto`` and ``off`` resolve to ``native`` (pure JAX, available
everywhere) except that ``auto`` for cholesky / solve_lu picks cusolvermp
on a true-2D CUDA mesh when it is compiled.  ``off`` is an *override*, not
a guard: it is honored unconditionally.

ZERO ENVIRONMENT READS.  Nothing in this module consults ``os.environ``.
Backend choice is a deck key; the environment grants a CAPABILITY (which
``.so`` is pinned — :mod:`distrib_la.loader`) and never selects a route.
"""
from __future__ import annotations

from types import MappingProxyType

from lxkit.gate import mesh_ffi_platform

from distrib_la import loader

__all__ = [
    "OPS",
    "NATIVE",
    "NATIVE2D",
    "BACKEND_CHOICES",
    "EIGH_BACKENDS",
    "CHOLESKY_BACKENDS",
    "LU_BACKENDS",
    "FFI_PLATFORMS",
    "mesh_platform",
    "mesh_is_cpu",
    "resolve_backend",
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

#: Per-op user vocabulary (requested names).  ``auto``/``off`` resolve to
#: ``native``; the rest name a concrete implementation.  ``distributed``
#: is the PLATFORM-DEFAULT distributed backend (:data:`_DISTRIBUTED_DEFAULT`);
#: it exists for ``eigh`` and ``solve_lu`` because both have exactly one
#: right distributed library per platform, so naming the library at the
#: call site is redundant and drifts (scorecard Z.2 was precisely that
#: drift).  ``cholesky`` deliberately does NOT: its CPU story is a
#: channel-policy ladder in the caller, not one library.
BACKEND_CHOICES = {
    "eigh":     ("auto", "off", "distributed", "cusolvermp", "slate",
                 "scalapack"),
    "cholesky": ("auto", "off", "native2d", "cusolvermp", "slate"),
    "solve_lu": ("auto", "off", "distributed", "cusolvermp", "scalapack"),
}
EIGH_BACKENDS = BACKEND_CHOICES["eigh"]
CHOLESKY_BACKENDS = BACKEND_CHOICES["cholesky"]
LU_BACKENDS = BACKEND_CHOICES["solve_lu"]

#: jax device platform → the FFI platform key this package speaks.
#:
#: ``rocm`` is present and passes through UNMAPPED, deliberately.  It is a
#: DECLARED-UNTESTED tier: LORRAX builds no ROCm ``.so``, nobody has run one,
#: and the preference rows below exist so the routing question has an answer
#: rather than a nonsense one.  MEASUREMENT GAP, named so nobody has to
#: rediscover it: on the jaxes this tree runs, ``Device.platform`` is
#: ``'gpu'`` for BOTH vendors, so a real ROCm mesh would land on the
#: ``'gpu' -> CUDA`` row above and never reach the rocm tier.  Disambiguating
#: them needs a machine to measure on; this map is where that edit goes, and
#: it is ONE row.  Guessing from ``device_kind`` strings without a ROCm
#: machine would be a compatibility path nobody can see failing, which is
#: the defect class this package exists to remove.
FFI_PLATFORMS = MappingProxyType({
    "cpu": "cpu",
    "gpu": "CUDA",
    "cuda": "CUDA",
    "rocm": "rocm",
})

#: ``distributed`` → the platform's permanent default distributed backend.
#:
#: On **cpu** that is ScaLAPACK ``pzheevd``, FOREVER: SLATE's host ``heev``
#: SIGSEGVs deterministically down to a 1×1 mesh (bug L-2) while
#: ScaLAPACK's routines on the same library and MPI context are clean, so
#: there is no configuration in which slate is the right CPU eigh.  On
#: **CUDA** it is cuSOLVERMp, the only library there with a distributed
#: syevd.  On **rocm** it is SLATE, which is the only one of the three that
#: builds for ROCm at all — declared-untested (see :data:`FFI_PLATFORMS`).
#:
#: There is deliberately NO ``('solve_lu', 'rocm')`` row: SLATE ships no
#: getrf/getrs handler in this tree, so a row would promise a backend that
#: does not exist.  ``distributed`` LU on ROCm therefore refuses by naming
#: the platform, which is the true answer.
_DISTRIBUTED_DEFAULT = {
    ("eigh", "cpu"):      "scalapack",
    ("eigh", "CUDA"):     "cusolvermp",
    ("eigh", "rocm"):     "slate",
    # solve_lu: the same one-library-per-platform rule.  ScaLAPACK's
    # pXgetrf/pXgetrs on the host lib, cuSOLVERMp's batched getrf/getrs
    # on CUDA — the two backends _IMPL already lists for this op.
    ("solve_lu", "cpu"):  "scalapack",
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
    ("eigh", "slate"):          ("lorrax_slate_eigh",       ("CUDA", "cpu", "rocm")),
    ("eigh", "scalapack"):      ("lorrax_scalapack_eigh",           ("cpu",)),
    ("cholesky", "cusolvermp"): ("lorrax_cusolvermp_batched_potrf", ("CUDA",)),
    ("cholesky", "slate"):      ("lorrax_slate_potrf",      ("CUDA", "cpu", "rocm")),
    ("solve_lu", "cusolvermp"): ("lorrax_cusolvermp_batched_solve_lu", ("CUDA",)),
    # The scalapack LU family is THREE handlers since the transverse
    # factor hoist (2026-08): the fused solve (legacy callers) plus the
    # split getrf/getrs pair the hoisted ζ factor stage requires.  All
    # three are probed — an old host .so without the pair refuses at
    # resolve time instead of dying mid-run inside the factor stage.
    ("solve_lu", "scalapack"):  (("lorrax_scalapack_batched_solve_lu",
                                  "lorrax_scalapack_batched_getrf",
                                  "lorrax_scalapack_batched_getrs"),
                                 ("cpu",)),
}


def mesh_platform(mesh_xy) -> str:
    """The FFI platform key for the mesh's devices.

    ``"cpu"`` / ``"CUDA"`` / ``"rocm"`` per :data:`FFI_PLATFORMS`; anything
    else passes through UNMAPPED so a refusal can name what it actually saw
    (``'tpu'`` reads better than ``'unknown'``).
    """
    return mesh_ffi_platform(mesh_xy, FFI_PLATFORMS)


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
    if op == "eigh" and backend in ("cusolvermp", "slate"):
        # These two FFI eigh wrappers reject p != q; for cusolvermp this is
        # a DEADLOCK guard (cusolverMpSyevd hangs in a collective on
        # rectangular one-tile-per-rank blocks — observed 4x1/1x4,
        # 2026-07-10), for SLATE heev a hard library requirement.
        # ScaLAPACK pXheevd is NOT in this class: like pXgetrf it only
        # needs square descriptor BLOCKS, so square-or-1-D is enough (its
        # rule is the ``scalapack`` branch below).
        if px != py:
            raise ValueError(
                f"eigh backend {backend!r} needs a SQUARE mesh "
                f"(cusolverMpSyevd DEADLOCKS on rectangular blocks; SLATE "
                f"heev rejects them); got {px}x{py}.")
    elif backend == "slate":
        # These two rules MIRROR ``_slate.validate_tile_layout`` (the
        # call-time guard) exactly, in the same order.  Keep them in sync:
        # a rule enforced only at call time turns a returned 'slate' into a
        # broken promise (bug L-1, 2026-07-25: cholesky+slate on a 2x4 mesh
        # resolved, then raised on the very next call).
        if px > 1 and py > 1 and px != py:
            raise ValueError(
                f"{op} backend 'slate': mesh {px}x{py} unsupported — with "
                f"both axes > 1 the square SLATE tile size cannot give one "
                f"tile per rank on both axes unless px == py.  Use a square "
                f"or Nx1 mesh, or a different backend.")
        if px == 1 and py > 1:
            raise ValueError(
                f"{op} backend 'slate': 1x{py} meshes hit a SLATE stride "
                f"assert (guarded; see _slate.validate_tile_layout).  Use a "
                f"{py}x1 or square mesh, or a different backend.")
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
        print(
            f"  [distrib_la.resolve] {op} backend 'auto': cusolvermp is "
            f"compiled but the {px}x{py} mesh is 1-D (block-cyclic 2-D "
            f"layout needs px >= 2 and py >= 2) — demoting to native.  "
            f"An explicit 'cusolvermp' request on this mesh refuses "
            f"instead.", flush=True)


def _announce_auto_capability_demote(op: str, target: str) -> None:
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
        reason = loader.probe_target(target, "CUDA")
    except Exception as exc:               # noqa: BLE001 — never let the
        reason = f"(probe_target itself failed: {exc!r})"   # witness raise
    print(
        f"  [distrib_la.resolve] {op} backend 'auto': the mesh geometry "
        f"supports '{target}' but the handler is NOT AVAILABLE — "
        f"demoting to the replicated native path.  Reason: {reason}.  "
        f"This is a PERFORMANCE AND MEMORY cliff, not a correctness "
        f"fallback; an explicit backend request refuses here instead.  "
        f"Fix the FFI build or set the backend explicitly to make this "
        f"fail loudly.", flush=True)


def resolve_backend(op: str, requested: str, mesh_xy, *,
                    n: int | None = None) -> str:
    """Resolve a requested backend name for ``op`` on ``mesh_xy``.

    Returns the concrete backend to run: ``"native"``, ``"native2d"``, or
    one of the FFI backend names (``"cusolvermp" | "slate" | "scalapack"``).
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
    if requested == "distributed":
        # "Spread ONE tile over the whole mesh, with whatever library this
        # platform's distributed eigh is."  Explicit — every guard below
        # applies and a failure raises rather than downgrading, because a
        # silent downgrade to the replicated path is exactly the O(nq·μ³)
        # no-P-scaling wall this name exists to escape.
        requested = _DISTRIBUTED_DEFAULT.get((op, mesh_platform(mesh_xy)))
        if requested is None:
            raise ValueError(
                f"{op} backend 'distributed' has no default backend on "
                f"platform {mesh_platform(mesh_xy)!r} "
                f"(defined for: {sorted(_DISTRIBUTED_DEFAULT)}).")
    if requested == "auto":
        if op == "eigh" or mesh_is_cpu(mesh_xy):
            return NATIVE
        target, _ = _SPEC[(op, "cusolvermp")]
        if px >= 2 and py >= 2 and loader.has_target(target, "CUDA"):
            return "cusolvermp"
        # 'auto' MAY demote, but must announce.  The only demote here that
        # is not the documented default policy ("native unless a compiled
        # cusolvermp on a true-2D CUDA mesh") is the geometry one: handler
        # compiled, mesh 1-D.  Mesh geometry is identical on every rank, so
        # a rank-0 line suffices; deduped so repeated plan() calls do not
        # spam.
        if loader.has_target(target, "CUDA") and not (px >= 2 and py >= 2):
            _announce_auto_geometry_demote(op, px, py)
        elif px >= 2 and py >= 2:
            # Geometry was fine, so the ONLY reason we are here is that the
            # handler could not be resolved.  See the docstring of
            # _announce_auto_capability_demote for why this used to be
            # silent and why that was the highest-consequence failure mode
            # in the resolver.
            _announce_auto_capability_demote(op, target)
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

    # 2b. KNOWN-BROKEN combination (bug L-2).  SLATE's host ``heev``
    # SIGSEGVs deterministically — n = 64/512/1200, meshes 1x1 / 2x2 /
    # 4x4, intra- and inter-node, compute_evecs both ways, SLATE built
    # both threaded and sequential.  It reproduces on ONE rank, so it is
    # neither MPI nor the layout contract.  The handler IS compiled and the
    # capability probe passes, so nothing below would catch it and
    # resolve_backend would hand back a name whose first call kills the
    # job with no Python traceback.  Refuse here, and name the replacement.
    if op == "eigh" and requested == "slate" and platform == "cpu":
        raise RuntimeError(
            "eigh backend 'slate' is REJECTED on CPU meshes: SLATE's host "
            "heev SIGSEGVs deterministically (bug L-2, reproduced on a 1x1 "
            "mesh at n=64 — not an MPI or layout problem).  Use "
            "'distributed' (= ScaLAPACK pzheevd, the permanent CPU default) "
            "or 'off'/'auto' for the replicated jnp.linalg.eigh.")

    # 3. compiled capability
    #
    # ``probe_target`` (not ``has_target``) because the REASON matters to
    # whoever reads this: "the .so would not load" and "the .so has no
    # such handler" have completely different fixes, and reporting the
    # first as the second sends people to rebuild a library that is fine.
    # That happened: wk_P G4 (2026-07-25) refused slate cholesky on a
    # legal 8x1 mesh with "not compiled into the cpu FFI library" while
    # `nm -D` showed SlatePotrfHostFfi present — the real cause was an
    # incomplete LD_LIBRARY_PATH.  ``target`` may be one name or a tuple
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

    return requested


def _available(op: str, mesh_xy) -> list[str]:
    """Names of the backends that would pass guards 2–4 for ``op`` on this
    mesh (the pure-JAX ones always qualify)."""
    out = [NATIVE] + ([NATIVE2D] if NATIVE2D in BACKEND_CHOICES[op] else [])
    platform = mesh_platform(mesh_xy)
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
    if backend == "cusolvermp":
        from distrib_la import _cusolvermp
        return _cusolvermp
    if backend == "slate":
        from distrib_la import _slate
        return _slate
    if backend == "scalapack":
        from distrib_la import _scalapack
        return _scalapack
    raise ValueError(
        f"no FFI backend module named {backend!r} "
        f"(known: cusolvermp, slate, scalapack, native2d; 'native' is "
        f"jnp + the caller's own routes and has no module).")

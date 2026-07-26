"""Backend resolution for the distributed dense-linalg facade.

This module is the ONE place where a requested backend name
(``auto | off | distributed | cusolvermp | slate | scalapack``, per op) is
turned into a concrete, *guaranteed-callable* backend — or a clear
resolve-time error.  Every guard lives here, applied in one fixed order:

    1. vocabulary   — is the name a backend of this op at all?
                      (``distributed`` is resolved to the platform's
                      default distributed library FIRST, then runs the
                      whole ladder as if it had been named explicitly)
    2. platform     — does the backend run on this mesh's device kind?
                      (cusolvermp is CUDA-only; scalapack is host-only)
    2b. known-broken— slate eigh on a CPU mesh (bug L-2: SIGSEGV)
    3. capability   — is the backend's FFI handler actually usable?
                      (``ffi_loader.probe_target``, which separates "the
                      library would not load" from "the library has no
                      such handler" — partial builds legitimately omit
                      handlers, and either way it must fail HERE, not
                      minutes later at the first distributed call, with
                      the reason that names the actual fix)
    4. coverage     — the FFI backends run ONE JAX process per device
                      (their MPI/NCCL context is per-process); a mesh
                      with more devices than processes cannot drive them
    5. geometry     — square-mesh where required (cusolverMpSyevd
                      DEADLOCKS inside a collective on rectangular
                      blocks instead of returning an error), SLATE's
                      square-or-N×1 tile rule + the 1×q stride-assert
                      guard (both mirroring
                      ``ffi/slate/context.validate_tile_layout``),
                      ScaLAPACK's square-block descriptor requirement
    6. divisibility — ``n`` divisible by both mesh axes (only checked
                      when the caller passes ``n``)

``auto`` and ``off`` always resolve to the ``native`` backend paths
(pure JAX, available everywhere) except that ``auto`` for cholesky /
solve_lu picks cusolvermp on a true-2D CUDA mesh when it is compiled —
mirroring the production ζ-fit policy.  ``off`` is an *override*, not a
guard: it is honored unconditionally.  (For the GW ζ-fit,
``distributed_cholesky = off`` also bypasses the replicated
rank-truncation route and can silently destroy the physics — see
``docs/dev/linalg_ffi.md`` "Sharp edges".)

The GW ζ-fit resolvers (``isdf/core._resolve_channel_ladder`` and
friends) implement a richer, channel-specific policy (replication cap,
charge vs transverse route strings) as a thin layer ON TOP of this
module: their explicit slate / scalapack handlers call
:func:`resolve_backend` for the availability + guard work and only add
the route-string mapping.
"""
from __future__ import annotations

from jax.sharding import Mesh

from ..common import ffi_loader

__all__ = [
    "OPS",
    "NATIVE",
    "BACKEND_CHOICES",
    "EIGH_BACKENDS",
    "CHOLESKY_BACKENDS",
    "LU_BACKENDS",
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
#: measured default for every op at production tile sizes (see the
#: per-op notes in ``dispatch.dispatch_eigh`` and ``isdf/core``).
NATIVE = "native"

#: Per-op user vocabulary (requested names).  ``auto``/``off`` resolve
#: to ``native``; the rest name distributed FFI libraries.  ``distributed``
#: (eigh only) is the PLATFORM-DEFAULT distributed backend — see
#: :data:`_DISTRIBUTED_DEFAULT`.
BACKEND_CHOICES = {
    "eigh":     ("auto", "off", "distributed", "cusolvermp", "slate",
                 "scalapack"),
    "cholesky": ("auto", "off", "cusolvermp", "slate"),
    "solve_lu": ("auto", "off", "cusolvermp", "scalapack"),
}

#: ``distributed`` → the platform's permanent default distributed backend.
#: On **cpu** that is ScaLAPACK ``pzheevd``, FOREVER: SLATE's host ``heev``
#: SIGSEGVs deterministically down to a 1×1 mesh (bug L-2, scorecard L §4)
#: while ScaLAPACK's routines on the same library and MPI context are
#: clean, so there is no configuration in which slate is the right CPU
#: eigh.  On CUDA it is cuSOLVERMp, the only library there with a
#: distributed syevd.
_DISTRIBUTED_DEFAULT = {
    ("eigh", "cpu"):  "scalapack",
    ("eigh", "CUDA"): "cusolvermp",
}
EIGH_BACKENDS = BACKEND_CHOICES["eigh"]
CHOLESKY_BACKENDS = BACKEND_CHOICES["cholesky"]
LU_BACKENDS = BACKEND_CHOICES["solve_lu"]

#: What ``native`` means per op (for messages / list_backends).
_NATIVE_IMPL = {
    "eigh":     "jnp.linalg.eigh (q-batched, every device solves its shard)",
    "cholesky": "replicated dense / in-tree sharded_cholesky (isdf/core)",
    "solve_lu": "per-q jnp.linalg.solve + ridge (isdf/core)",
}

#: (op, backend) → (FFI target to probe, platforms it exists on).
#: Platform names follow ffi_loader: "CUDA" (liblorrax_ffi.so) and
#: "cpu" (liblorrax_ffi_host.so).
_SPEC = {
    ("eigh", "cusolvermp"):     ("lorrax_cusolvermp_eigh",             ("CUDA",)),
    ("eigh", "slate"):          ("lorrax_slate_eigh",                  ("CUDA", "cpu")),
    ("eigh", "scalapack"):      ("lorrax_scalapack_eigh",              ("cpu",)),
    ("cholesky", "cusolvermp"): ("lorrax_cusolvermp_batched_potrf",    ("CUDA",)),
    ("cholesky", "slate"):      ("lorrax_slate_potrf",                 ("CUDA", "cpu")),
    ("solve_lu", "cusolvermp"): ("lorrax_cusolvermp_batched_solve_lu", ("CUDA",)),
    ("solve_lu", "scalapack"):  ("lorrax_scalapack_batched_solve_lu",  ("cpu",)),
}

# (The per-platform build command used to be duplicated here.  It now
# lives ONLY in ffi_loader._PLATFORMS[...]["build_hint"], which
# probe_target quotes — one copy, next to the paths it refers to.)


def mesh_platform(mesh_xy: Mesh) -> str:
    """The ffi_loader platform key for the mesh's devices: ``"cpu"`` for a
    host-device mesh, ``"CUDA"`` for a gpu/cuda mesh."""
    plat = mesh_xy.devices.flat[0].platform
    return "CUDA" if plat in ("gpu", "cuda") else "cpu" if plat == "cpu" else plat


def mesh_is_cpu(mesh_xy: Mesh) -> bool:
    """True when the mesh's devices are the CPU backend.

    Keeps every ``auto`` policy from selecting the CUDA-only cuSOLVERMp
    paths on a host-only run.  Never raises (a malformed mesh reads as
    non-CPU, i.e. no special-casing).
    """
    try:
        return bool(mesh_xy.devices.flat[0].platform == "cpu")
    except Exception:
        return False


def _vocab_error(op: str, requested) -> str:
    return (f"{op} backend must be one of "
            f"{'|'.join(BACKEND_CHOICES[op])}, got {requested!r}")


def _mesh_shape(mesh_xy: Mesh) -> tuple[int, int]:
    return int(mesh_xy.shape["x"]), int(mesh_xy.shape["y"])


def _process_count() -> int:
    import jax
    try:
        return int(jax.process_count())
    except Exception:
        return 1


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
        # These two rules MIRROR ``ffi/slate/context.validate_tile_layout``
        # (the call-time guard) exactly, in the same order.  Keep them in
        # sync: a rule enforced only at call time turns a returned 'slate'
        # into a broken promise (bug L-1, 2026-07-25: cholesky+slate on a
        # 2x4 mesh resolved, then raised on the very next call).
        if px > 1 and py > 1 and px != py:
            raise ValueError(
                f"{op} backend 'slate': mesh {px}x{py} unsupported — with "
                f"both axes > 1 the square SLATE tile size cannot give one "
                f"tile per rank on both axes unless px == py.  Use a square "
                f"or Nx1 mesh, or a different backend.")
        if px == 1 and py > 1:
            raise ValueError(
                f"{op} backend 'slate': 1x{py} meshes hit a SLATE stride "
                f"assert (guarded; see src/ffi/slate/README.md).  Use a "
                f"{py}x1 or square mesh, or a different backend.")
    elif backend == "scalapack":
        # Mirrors ffi.scalapack.eigh.validate_eigh_mesh / solve_lu's own
        # check — keep them in sync (bug L-1: a rule enforced only at call
        # time turns a returned backend name into a broken promise).
        if px > 1 and py > 1 and px != py:
            _routine = "pXheevd" if op == "eigh" else "pXgetrf"
            raise ValueError(
                f"{op} backend 'scalapack': mesh {px}x{py} unsupported — "
                f"{_routine} needs square descriptor blocks (MB == NB), "
                f"which the one-tile-per-rank layout only gives on square "
                f"or 1-D meshes.")


def resolve_backend(op: str, requested: str, mesh_xy: Mesh,
                    *, n: int | None = None) -> str:
    """Resolve a requested backend name for ``op`` on ``mesh_xy``.

    Returns the concrete backend to run: ``"native"`` or one of the FFI
    backend names (``"cusolvermp" | "slate" | "scalapack"``).  A returned
    FFI name is a *promise*: its handler is compiled into the loaded
    library and every mesh guard passed — the subsequent call cannot fail
    for an availability/geometry reason.

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
        Optional matrix extent; when given, the FFI backends additionally
        require ``n`` divisible by both mesh axes (guard 6).

    Notes
    -----
    ``auto`` policy — per op, mirroring the measured production defaults:

    * ``eigh``: always ``native``.  The q-batched jnp path solves ndev
      matrices concurrently; the FFI path solves one matrix ndev-ways and
      walks the batch serially, measured 100–600x slower per fit-size
      matrix (``common.eigh_benchmark --mode dispatch``).  ``auto`` means
      "the fastest eigh for this call", and for a batch of tiles that fit
      on one device that is still the native one — the htransform /
      vq_interp consumers depend on it and are gated bit-exact.
      **``distributed`` is the name for "spread ONE tile over the mesh"**,
      and its CPU default is permanently ScaLAPACK ``pzheevd``
      (:data:`_DISTRIBUTED_DEFAULT`).  That is the backend the ζ-fit's
      ``distributed_zeta_solve = distributed`` tier uses, and the only way
      out of the replicated factor's O(nq·μ³)-with-no-P-scaling wall.
    * ``cholesky`` / ``solve_lu``: ``cusolvermp`` on a true-2D
      (px>=2 and py>=2) CUDA mesh when its handler is compiled, else
      ``native``.  Never an FFI backend on a CPU mesh.  (The GW ζ-fit
      layer adds the replication-cap / rank-truncation refinement on
      top — see ``isdf/core._resolve_solver_kind_charge``.)
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
        if px >= 2 and py >= 2 and ffi_loader.has_target(target, "CUDA"):
            return "cusolvermp"
        return NATIVE

    # ── explicit FFI backend: run the full guard ladder ──────────────
    platform = mesh_platform(mesh_xy)
    target, platforms = _SPEC[(op, requested)]

    # 2. platform
    if platform not in platforms:
        raise ValueError(
            f"{op} backend {requested!r} is "
            f"{'CUDA-only' if platforms == ('CUDA',) else 'host-only'} but the "
            f"mesh devices are {platform!r}.  "
            f"Available {op} backends on this mesh: "
            f"{', '.join(_available(op, mesh_xy))}.")

    # 2b. KNOWN-BROKEN combination (bug L-2, scorecard L §4).  SLATE's host
    # ``heev`` SIGSEGVs deterministically — n = 64/512/1200, meshes 1x1 /
    # 2x2 / 4x4, intra- and inter-node, compute_evecs both ways, SLATE built
    # both threaded and sequential.  It reproduces on ONE rank, so it is
    # neither MPI nor the layout contract.  The handler IS compiled and the
    # capability probe passes, so nothing below would catch it and
    # ``resolve_backend`` would hand back a name whose first call kills the
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
    # whoever reads this: "the .so would not dlopen" and "the .so has no
    # such handler" have completely different fixes, and reporting the
    # first as the second sends people to rebuild a library that is fine.
    # That happened: wk_P G4 (2026-07-25) refused slate cholesky on a
    # legal 8x1 mesh with "not compiled into the cpu FFI library" while
    # `nm -D` showed SlatePotrfHostFfi present — the real cause was an
    # incomplete LD_LIBRARY_PATH for the lib's DT_NEEDED.
    usable, why = ffi_loader.probe_target(target, platform)
    if not usable:
        raise RuntimeError(
            f"{op} backend {requested!r} requested but its FFI handler "
            f"({target}) is not usable: {why}  "
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
        # Documented legacy semantics of the ζ-fit ladder: explicit
        # cusolvermp still falls back to native on a 1-D mesh (the
        # block-cyclic 2-D layout degenerates there).
        if not (px >= 2 and py >= 2):
            return NATIVE
    _check_geometry(op, requested, px, py)

    # 6. divisibility
    if n is not None and (int(n) % px or int(n) % py):
        raise ValueError(
            f"{op} backend {requested!r} needs n ({n}) divisible by both "
            f"mesh axes ({px}, {py}) — the one-tile-per-rank block-cyclic "
            f"layout has no ragged tiles.  Pad n or change the mesh.")

    return requested


def _available(op: str, mesh_xy: Mesh) -> list[str]:
    """Names of the backends that would pass guards 2–4 for ``op`` on this
    mesh (native always qualifies)."""
    out = [NATIVE]
    platform = mesh_platform(mesh_xy)
    for (o, b), (target, platforms) in _SPEC.items():
        if o != op or platform not in platforms:
            continue
        if ffi_loader.has_target(target, platform):
            out.append(b)
    return out


def list_backends(op: str, mesh_xy: Mesh) -> dict[str, str]:
    """Introspection: every concrete backend of ``op`` mapped to a status
    string — ``"available"``/``"available (…)"`` or ``"unavailable: <first
    failed guard>"`` for this mesh.  Purely informational; never raises
    for an available-or-not reason (unknown ``op`` still raises)."""
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
    """THE import seam for the FFI backend packages.

    Everything outside ``ffi/`` that needs a backend implementation goes
    through here (after :func:`resolve_backend` said it is usable), so
    the facade is the single place that knows the package layout:

        mod = backend_module("cusolvermp")
        L = mod.batched_distributed_cholesky(C_q, mesh=mesh)

    Raises ``ValueError`` for names that are not FFI backends (``native``
    has no module — its impls are the in-tree call sites).
    """
    if backend == "cusolvermp":
        from .. import cusolvermp
        return cusolvermp
    if backend == "slate":
        from .. import slate
        return slate
    if backend == "scalapack":
        from .. import scalapack
        return scalapack
    raise ValueError(
        f"no FFI backend module named {backend!r} "
        f"(known: cusolvermp, slate, scalapack; 'native' is in-tree).")

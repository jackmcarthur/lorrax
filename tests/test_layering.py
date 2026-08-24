"""THE layering gate: every module in the tree has a level, and the levels hold.

Three levels, assigned by *what a module is allowed to know about*:

===  =====================  ==================================================
L1   physics                knows about bands, q-points, ζ, Σ, decks.  The
                            drivers and the physics kernels.
L2   numerical routines     knows about matrices, quadrature nodes, residuals
                            and convergence — and **nothing physical**.
                            Davidson, Lanczos, MINRES, minimax quadrature,
                            Anderson/CROP mixing, blocked Cholesky.
L3   substrate              knows about devices, meshes, processes, native
                            libraries and files — and **nothing mathematical**.
                            ``runtime``, ``ffi``, ``common.collectives``, the
                            FFT kernels, the sharded-HDF5 transport.
===  =====================  ==================================================

Imports run **downhill only**: L1 → L2 → L3.  ``docs/architecture/layers.md``
states the rules in prose and lists every sanctioned exception with its
reason; this file is the same content in a form that fails.

WHAT "THE TREE" MEANS SINCE THE SERVICE PHASE.  The scan covers ``src/``
**and every ``services/*/src``** (:data:`ROOTS`).  A one-root scan would not
have failed when the linalg stack moved out of ``src/ffi/linalg`` (now
deleted) — it would
have gone on passing while measuring 37 fewer files, which is coverage
evaporating silently, the exact shape this file exists to prevent.  Module
names are root-RELATIVE, so ``services/lxkit/src/lxkit/gate.py`` is
``lxkit.gate``, which is what it is called at an import site.

Services get one extra rule of their own (§8): lorrax may import a service's
TOP-LEVEL package and nothing else.  That is the charter's measurable
criterion — "direct-dependency import edges from lorrax → 0 outside doors" —
and until this edit nothing counted them.

WHY A TEST AND NOT A CONVENTION.  Every finding this file pins was found by
hand at least twice: five hand-rolled host gathers each documenting that they
were copied from another; twenty-one mesh constructors in five mutually
incompatible dialects; ``jax.devices()[:1]`` — process 0's device on every
rank — propagating to three more files after being written down as a hazard
in two.  A layering rule nobody can check decays inside a week.

WHY IT IS PURE AST.  Nothing here imports jax, or any module under ``src``.
That is deliberate: this suite must run on a login node, in a container, and
in CI, on any Python that can parse the tree — so there is never a reason not
to run it.  All 241 files under ``src/`` parse under Python 3.7 through 3.13
(checked).

EVERY SCANNER HAS A RED TWIN.  Each ``test_*_can_fail`` feeds the SAME
scanner function a synthetic source that does contain the banned construct
and asserts it is flagged.  Re-implementing the scan inside the twin would
test the twin, not the instrument, which is standing lesson #1 in
``wk_REL/README.md``: six checks in one 2026-07-30 session were void, four of
them cheerfully green.

THE ALLOWLISTS ARE RATCHETS, NOT ESCAPE HATCHES.  Every sanctioned exception
below is asserted to be *still needed*.  A stale entry fails the suite, so an
exception cannot outlive the violation it excuses, and the numbers in
``_DRIVER_PLUMBING_BUDGET`` can only go down.
"""
import ast
import os
import pathlib

import pytest


_REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = _REPO / "src"
_SERVICES = _REPO / "services"


def _roots():
    """``[src/, services/*/src/, …]`` — every source root in this monorepo.

    Discovered, not listed: a service added next week is scanned the day it
    lands, which is the only way an allowlist-free rule stays true.
    """
    out = [SRC]
    if _SERVICES.is_dir():
        out += [p / "src" for p in sorted(_SERVICES.iterdir())
                if (p / "src").is_dir()]
    return out


ROOTS = _roots()


# ===========================================================================
# 0.  The map
# ===========================================================================
#
# Everything under ``src/`` that is not named here, and is not a bench/test
# driver, is L1.  Physics is the default because physics is the bulk; a new
# module has to ARGUE its way down, not drift down.

#: L3 by module.  Whole packages are in ``_L3_PACKAGES``.
_L3_MODULES = frozenset({
    # process bootstrap, and the shape arithmetic that exists only because a
    # mesh has to divide.  ``runtime.jax_support`` is the startup version /
    # jax._src-arity gate; it moved here from ``common/`` on 2026-08-06 for
    # the reason ``runtime.xla_memory`` did on 2026-07-31 (numbered request
    # R9) -- ``runtime`` was reaching UP into it through a function-local
    # import, and a startup fact that only ``runtime`` consumes is a sibling
    # of ``runtime``, not something above it.
    "runtime", "runtime.aot_memory", "runtime.padding", "runtime.xla_memory",
    "runtime.jax_support", "runtime.pjrt_log_filter",
    # ``runtime.env_flags`` — THE boolean-env grammar, and its level is the
    # whole reason it is here rather than in ``gw.gw_config``.  The parsers
    # that still swallowed an unrecognised token silently
    # (``file_io._slab_io_ffi._env_flag``, ``runtime._env_falsy``) are L3
    # and may not import an L1 module, so a grammar owned at L1 is a
    # grammar the substrate re-invents -- which is exactly what those two
    # were.  It knows about ``os.environ`` and nothing else: no jax, no
    # config, no physics.
    "runtime.env_flags",
    # the cross-process service and its one policy client
    "common.collectives", "centroid.distribution",
    # movement primitives.  Named, owned patterns — NOT generic ``shard_map``
    # wrappers, which is a refusal recorded in docs/architecture/layers.md.
    "common.contract_bands", "common.staged_reshard", "common.sharding_fit",
    # jax glue and instrumentation.  ``common.shard_map`` is a VERSION SHIM,
    # not the generic wrapper layers.md rule 2 forbids -- it selects a symbol
    # and a kwarg spelling and forwards in_specs/out_specs untouched; see its
    # docstring.  It is L3 because L3 kernels (fft_helpers, ffi.*, file_io)
    # import it.
    #
    # ``common.vma`` is its twin and was left unclassified when it landed
    # (``agent/vma-pvary-marking``), so the default put a jax version shim at
    # **L1 -- physics**, the most permissive level there is.  Its own
    # docstring names the relation: ``common.shard_map`` "owns the
    # symbol-and-kwarg decision the way this module owns the marking
    # decision".  Its whole subject is MESH AXES and per-device variation --
    # L3's declared vocabulary -- and it contains no mathematics at all,
    # which is what separates L3 from L2.  Naming it here is a TIGHTENING:
    # at L1 it could import anything in the tree; at L3 it may import
    # nothing above the substrate, and today it imports nothing from
    # ``src/`` at all.
    "common.shard_map", "common.vma",
    "common.jax_compile_cache", "common.jax_profile", "common.timing",
    "common.progress", "common.gpu_utils", "common.async_io", "common.sanity",
    # FFT kernels
    "common.fft_helpers",
    # the sharded-HDF5 transport (NOT the format readers above it).
    # ``file_io.hdf5_owner`` is at the transport's level and not above it:
    # its whole subject is which NATIVE LIBRARY INSTANCE, in this PROCESS,
    # holds a file open — the same class of fact ``_slab_io_ffi`` is made
    # of, and one no format reader is allowed to reason about.  It imports
    # ``file_io.h5_journal`` and nothing else from ``src/``.
    #
    # ``file_io.h5_journal`` is the same level for the same reason: it is
    # the per-rank text log of HDF5 handles, offsets and ctx pointers
    # (SLAB_IO_ROOT_CAUSE_AUDIT.md §C), i.e. transport facts, and the two
    # modules journal through each other — the registry writes the lines,
    # the journal asks the registry who holds the file.  Both imports are
    # function-local for that cycle.  Its only other ``src/`` import is
    # ``common.collectives.process_rank`` (L3), for the rank in the
    # filename.
    "file_io.slab_io", "file_io._slab_io_ffi", "file_io.paths",
    "file_io.hdf5_owner", "file_io.h5_journal",
})
#: Whole packages at L3.  Two of the three services are here for the same
#: reason ``ffi`` is: their entire subject is devices, meshes, processes,
#: native libraries and files.  ``lxkit`` owns capability gates and the probe
#: vocabulary; ``distrib_la`` owns distributed dense linalg over a device
#: mesh, and the mathematics inside it (a blocked Cholesky) is the same
#: mathematics ``ffi.linalg`` always carried at L3 — the level is decided by
#: what a module is allowed to KNOW ABOUT, and both know about ranks.
#:
#: ``zeta_loader`` is DELIBERATELY NOT HERE.  Being a service says nothing
#: about a level; it knows about q-points, centroids and ζ, which is physics,
#: so it takes the L1 default that ``file_io.zeta_loader`` always had
#: (docs/architecture/layers.md:163 — format readers are L1, only the
#: transport ``slab_io``/``_slab_io_ffi``/``paths`` below them is L3).  The
#: default is silent, so this note is where the decision is recorded.
#:
#: ``ffi`` stays only while ``src/ffi`` does;
#: :func:`test_every_package_in_the_map_exists` is what forces its removal
#: at the right moment instead of leaving a dead no-op entry.
#:
#: ``vcoul`` IS A SERVICE AND IS DELIBERATELY NOT HERE.  Being extracted is
#: not a level: the level is decided by what a module may KNOW ABOUT, and
#: vcoul's whole subject is q-points, reciprocal lattices and the Coulomb
#: interaction — physics, so L1 by the default, which is a decision and not
#: an oversight.  It owns no ``.so``, probes no device and counts no
#: processes; there is nothing at L3 for it to be.  It gets the §8 door rule
#: (``_SERVICE_DOORS``) like every service, and no level entry.
_L3_PACKAGES = ("ffi", "lxkit", "distrib_la")

#: L2 by module.  Whole packages are in ``_L2_PACKAGES``.
_L2_MODULES = frozenset({
    "common.rank_criterion",   # the pseudo-inverse truncation criterion
    "common.spectral_closure",  # where that truncation is allowed to land
    "centroid.kmeans_isdf",    # density-weighted Lloyd loop under a metric
})
# ``common.minimax`` was here ("minimax quadrature: a Remez problem").  It is
# the ``minimax`` SERVICE now — ``services/minimax/src/minimax/solver.py`` —
# and the row had to go the moment the file did, because
# ``test_every_module_in_the_map_exists`` refuses a map entry for a module
# that is not there.  Same mechanism that deleted ``common.cholesky_2d``
# below, and same reason: the level conflict dissolved by extraction rather
# than by a ruling.
#
# NO SHIM AT THE OLD PATH, deliberately, and the ratchet is
# ``tests/test_service_path_bootstrap.py::test_the_retired_shim_files_are_gone``.
# The phase-wide cleanup (`029da824`) deleted the five wave-1 shims and
# pinned them gone; re-creating ``src/common/minimax.py`` to green a branch
# would be that decision silently un-happening.  The module had exactly ONE
# production importer — ``gw.minimax_screening`` — and it is repointed at the
# door in the same commit, so there was nothing for a shim to bridge.
# ``common.cholesky_2d`` was here.  It is ``distrib_la``'s ``native2d``
# backend now, so the L2-vs-L3 conflict layers.md:196-200 records
# (blocked Cholesky is mathematics, but it is distributed mathematics)
# dissolved by extraction rather than by a ruling: it left src/ entirely,
# and inside the service it is L3 like the rest of the package.
# ``test_every_module_in_the_map_exists`` is what forced this line to be
# deleted at the right moment instead of rotting into a phantom.
_L2_PACKAGES = ("solvers", "mixing")


def _is_standalone_driver(mod: str) -> bool:
    """True for a bench / smoke-test driver shape inside ``src/``.

    There are ZERO of them: the 26 exempt drivers moved to ``tests/bench/``
    (2026-07-31), where the exemption is structural instead of a rule
    carve-out.  The classifier stays so that
    :func:`test_the_src_tree_grows_no_new_bench_drivers` can keep the count
    at 0 — a new driver shape under ``src/`` belongs in ``tests/bench/``.
    """
    last = mod.split(".")[-1]
    return (last.endswith("_test") or last.startswith("test_")
            or last.endswith("_bench") or "benchmark" in last
            or last.startswith("profile_")
            or ".tests." in mod or ".archive." in mod)


_RANK = {"L1": 1, "L2": 2, "L3": 3}


def layer_of(mod: str) -> str:
    """``"L1"``/``"L2"``/``"L3"``, or ``"X"`` for an exempt bench driver."""
    if _is_standalone_driver(mod):
        return "X"
    if mod in _L3_MODULES or mod.split(".")[0] in _L3_PACKAGES:
        return "L3"
    if mod in _L2_MODULES or mod.split(".")[0] in _L2_PACKAGES:
        return "L2"
    return "L1"


# ===========================================================================
# 1.  The scanners.  Each is used by a rule AND by that rule's red twin.
# ===========================================================================

#: What "reaching under jax" means for a driver.  Import spellings, not
#: identifier uses: a driver may pass a mesh around, it may not build one or
#: name a partition.  ``jax.shard_map`` is the new canonical location and
#: ``jax.experimental.shard_map`` the legacy one; both are banned, because a
#: driver that says either is doing distribution, not physics.
_PLUMBING_MODULES = (
    "jax.sharding",
    "jax.experimental.shard_map",
    "jax.experimental.multihost_utils",
    "jax.experimental.mesh_utils",
    "jax._src",
)
#: Names that mean the same thing when pulled off ``jax`` / ``jax.experimental``
#: directly, e.g. ``from jax import shard_map``.
_PLUMBING_NAMES = ("shard_map", "Mesh", "NamedSharding", "PartitionSpec",
                   "multihost_utils", "mesh_utils", "make_mesh")


def module_name(path: pathlib.Path, root: pathlib.Path = SRC) -> str:
    """The dotted name ``path`` is imported by, RELATIVE to its source root.

    Root-relative is what makes multi-root scanning meaningful:
    ``services/lxkit/src/lxkit/gate.py`` is ``lxkit.gate``, which is the
    name an import site writes and the name the layer map has to key on.
    """
    parts = list(path.relative_to(root).parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][:-3]
    return ".".join(parts)


def _modules_under(root: pathlib.Path):
    """``{module name: source text}`` for every ``.py`` under one root."""
    out = {}
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs
                   if d not in ("__pycache__", ".ipynb_checkpoints")]
        for fn in sorted(files):
            if fn.endswith(".py"):
                p = pathlib.Path(base) / fn
                out[module_name(p, root)] = p.read_text()
    return out


def all_modules():
    """``{module name: source text}`` for every ``.py`` under :data:`ROOTS`.

    A name collision between two roots would silently drop one of them, so
    it raises instead: two files claiming one import name is a defect
    whatever the layer rules say about them.
    """
    out = {}
    for root in ROOTS:
        for mod, text in _modules_under(root).items():
            if mod in out:
                raise AssertionError(
                    f"two source roots both define module {mod!r}; one of "
                    f"them would be invisible to every rule in this file")
            out[mod] = text
    return out


def modules_under_src(sources):
    """The subset of ``sources`` that lives in ``src/`` (i.e. is lorrax).

    Rule 8 is about what LORRAX may import, so it needs the split; nothing
    else does, and computing it from the filesystem once keeps the rule
    from having to guess from a name.
    """
    return set(_modules_under(SRC))


def packages_in(sources):
    """The module names in ``sources`` that are packages (an ``__init__.py``).

    ``all_modules`` maps ``runtime/__init__.py`` to ``"runtime"``, so a name is
    a package exactly when some other name is under it.  Needed by
    :func:`_absolute_target`; see the note there.
    """
    return {m for m in sources if any(k.startswith(m + ".") for k in sources)}


def _absolute_target(node: ast.ImportFrom, mod: str, is_pkg: bool = False) -> str:
    """Resolve a possibly-relative ``from ... import`` to an absolute name.

    ``is_pkg`` says whether ``mod`` is a package (i.e. came from an
    ``__init__.py``), and it is not a detail: for a plain module ``pkg.m``,
    ``from .x import y`` means ``pkg.x``, so one component is dropped — but for
    a PACKAGE ``pkg`` it means ``pkg.x``, and dropping a component yields a
    bare ``x`` that matches no module in the tree, so the edge is silently
    SKIPPED.  There are 24 package ``__init__.py`` files under ``src/`` and 34
    such imports among them, ``runtime/__init__.py``'s two startup siblings
    included; every one of them was exempt from rule 3.  Nothing was hiding
    there when this was fixed (2026-08-06) — the scan reports the same two
    excused edges before and after — but "the rule did not apply to a whole
    file shape" is the kind of hole that only stays empty by luck, and the
    default level is L1, so an unmapped ``runtime/<new>.py`` reached by
    ``from .<new> import …`` was invisible by construction.
    """
    if node.level == 0:
        return node.module or ""
    base = mod.split(".") if is_pkg else mod.split(".")[:-1]
    for _ in range(node.level - 1):
        base = base[:-1]
    if node.module:
        base = base + node.module.split(".")
    return ".".join(base)


def scan_plumbing_imports(source: str, mod: str = "m", is_pkg: bool = False):
    """Every import of a jax distribution/plumbing module. ``[(spelling, line)]``.

    Walks the WHOLE tree, so a lazy import inside a function is caught too —
    a module-scope-only walk missed exactly those in an earlier version of
    the ``kin_ion_io`` census.
    """
    hits = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith(_PLUMBING_MODULES):
                    hits.append((a.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            tgt = _absolute_target(node, mod, is_pkg)
            if tgt.startswith(_PLUMBING_MODULES):
                hits.append((tgt, node.lineno))
            elif tgt in ("jax", "jax.experimental"):
                for a in node.names:
                    if a.name in _PLUMBING_NAMES:
                        hits.append((tgt + "." + a.name, node.lineno))
    return sorted(hits, key=lambda h: h[1])


def scan_imports(source: str, mod: str, is_pkg: bool = False):
    """Every imported dotted name. ``[(target, line, is_lazy)]``."""
    hits = []

    class V(ast.NodeVisitor):
        def __init__(self):
            self.depth = 0

        def visit_FunctionDef(self, node):
            self.depth += 1
            self.generic_visit(node)
            self.depth -= 1

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Import(self, node):
            for a in node.names:
                hits.append((a.name, node.lineno, self.depth > 0))
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            tgt = _absolute_target(node, mod, is_pkg)
            for a in node.names:
                hits.append(((tgt + "." + a.name) if tgt else a.name,
                             node.lineno, self.depth > 0))
            self.generic_visit(node)

    V().visit(ast.parse(source))
    return hits


def _literal(node):
    """``ast.literal_eval`` of a subscript key, across ast versions."""
    if node.__class__.__name__ == "Index":          # Python <= 3.8
        node = node.value
    try:
        return ast.literal_eval(node)
    except Exception:                                # noqa: BLE001
        return "<dynamic>"


def scan_env(source: str):
    """Every ``os.environ`` touch. ``[(kind, key, line)]``.

    ``kind`` is ``"read"`` or ``"write"``.  Covers ``os.environ.get``,
    ``os.getenv``, ``os.environ[...]`` on both sides of an assignment,
    ``setdefault`` and ``pop``.
    """
    hits = []
    tree = ast.parse(source)
    written = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Subscript) and "environ" in ast.dump(t.value):
                    written.add(id(t))
                    hits.append(("write", _literal(t.slice), node.lineno))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr, val = node.func.attr, node.func.value
            dumped = ast.dump(val)
            if attr in ("setdefault", "pop") and "environ" in dumped:
                key = _literal(node.args[0]) if node.args else "<none>"
                hits.append(("write", key, node.lineno))
            elif attr in ("get", "getenv") and (
                    "environ" in dumped
                    or (isinstance(val, ast.Name) and val.id == "os")):
                key = _literal(node.args[0]) if node.args else "<none>"
                hits.append(("read", key, node.lineno))

    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and id(node) not in written:
            if "environ" in ast.dump(node.value):
                hits.append(("read", _literal(node.slice), node.lineno))
    return sorted(hits, key=lambda h: h[2])


def scan_module_scope_env_writes(source: str):
    """Env WRITES that execute at import time. ``[(key, line)]``.

    Import-time only: a write inside a function is a runtime choice a caller
    made, but a write at module scope happens to anyone who so much as names
    the module, and jax reads its environment at ITS import — which is the
    whole reason the ordering matters.
    """
    hits = []

    class V(ast.NodeVisitor):
        """Visits module scope only: recursion stops at any ``def``/``class``.

        Deliberately does NOT stop at ``if``/``try``/``with``/``for`` — the
        real sites include ``if "--cpu" in sys.argv: os.environ[...] = ...``
        (``psp.orbital_magnetization``) and a ``try`` around an optional
        import.  Those run at import time and so are in scope.
        """

        def visit_FunctionDef(self, node):
            return

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_ClassDef = visit_FunctionDef
        visit_Lambda = visit_FunctionDef

        def visit_Assign(self, node):
            for t in node.targets:
                if isinstance(t, ast.Subscript) and "environ" in ast.dump(t.value):
                    hits.append((_literal(t.slice), node.lineno))
            self.generic_visit(node)

        def visit_Call(self, node):
            fn = node.func
            if isinstance(fn, ast.Attribute) and \
                    fn.attr in ("setdefault", "pop") and \
                    "environ" in ast.dump(fn.value):
                key = _literal(node.args[0]) if node.args else "<none>"
                hits.append((key, node.lineno))
            self.generic_visit(node)

    V().visit(ast.parse(source))
    return sorted(set(hits), key=lambda h: (h[1], str(h[0])))


def scan_mesh_construction(source: str):
    """Every call to a name/attribute ``Mesh``. ``[line, ...]``."""
    out = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else (
                fn.attr if isinstance(fn, ast.Attribute) else None)
            if name == "Mesh":
                out.append(node.lineno)
    return sorted(out)


def has_main_block(source: str) -> bool:
    """True when the module declares itself an entry point."""
    for node in ast.parse(source).body:
        if isinstance(node, ast.If) and "'__main__'" in ast.dump(node.test):
            return True
    return False


# ===========================================================================
# 2.  RULE 1 — an L1 driver names no jax plumbing
# ===========================================================================
#
# The owner's goal in one sentence: "gw_jax, bse_jax, kin_ion_io and the
# kmeans drivers should appear to contain only physics on inspection".  A
# physicist reading a driver should not have to know what a ``Mesh`` is, what
# ``in_specs`` means, or that ``jax.devices()`` is the GLOBAL list.
#
# THE NUMBERS ARE A RATCHET.  Each budget is what that file carries TODAY.
# Going over fails; so does coming under without editing this table, which is
# what stops a budget from quietly becoming a licence.

_DRIVER_PLUMBING_BUDGET = {
    # --- clean.  These are the proof the target shape is reachable. -------
    "gw.gw_jax": 0,
    "bse.bse_jax": 0,
    "gw.kin_ion_io": 0,
    "centroid.kmeans_cli": 0,
    "psp.run_nscf": 0,
    "psp.get_DFT_mtxels": 0,
    "psp.get_dipole_mtxels": 0,
    "psp.run_sternheimer": 0,
    "psp.kpm_dos": 0,
    "psp.orbital_magnetization": 0,
    "psp.finite_q_head_interp": 0,
    "gw.eqp_bgw": 0,
    "gw.compute_vcoul_0d": 0,
    "postprocess.rotate_wfn_to_qp": 0,
    "gw.downfold_cli": 0,

    # --- not clean, with the reason ---------------------------------------
    # ``from jax.sharding import Mesh, NamedSharding, PartitionSpec`` at :90.
    # exciton_bands is a driver AND a library: it owns the interpolated-band
    # assembly its own CLI consumes.  Splitting it is the same operation that
    # freed ``kin_ion_io``, and it is a physics decision about where the
    # exciton-band assembly lives.  Numbered request.
    "bse.exciton_bands": 1,
    # ``jax.sharding`` at :20.  Same shape as exciton_bands, larger:
    # htransform is the H-matrix interpolation LIBRARY with a CLI bolted on.
    # Its projected-Gram shard_map and r/band schedule now live in the named
    # ``isdf.galerkin`` kernel owner, but the remaining basis fit and fH
    # interpolation still carry one ``jax.sharding`` import.  Its hand-rolled
    # ``_build_mesh_xy`` is gone (2026-07-31; since 2026-08-01 it hands back
    # the mesh ``initialize_communicator_stack`` built).
    "bandstructure.htransform": 1,
    # One ``from jax.sharding import ...`` each.  The four BSE CLIs also
    # default ``--px/--py`` to 1 and slice ``devices[:px*py]``, so on 16
    # devices with no flags they run a 1x1 mesh with no warning.  That is a
    # DEFAULTS bug, not a plumbing one; fixing the default in place comes
    # first (see docs/architecture/layers.md, "what not to unify").
    "bse.bse_feast": 1,
    "bse.bse_pseudopoles": 1,
    "bse.bse_w_exact": 1,
    "bse.bse_kpm": 1,
}


@pytest.fixture(scope="module")
def sources():
    return all_modules()


def test_every_budgeted_driver_exists(sources):
    """A renamed driver must not silently drop out of the gate."""
    missing = sorted(set(_DRIVER_PLUMBING_BUDGET) - set(sources))
    assert not missing, (
        f"{missing} are budgeted here but no longer exist under src/; the "
        f"budget stopped guarding anything the moment they were renamed")


@pytest.mark.parametrize("mod", sorted(_DRIVER_PLUMBING_BUDGET))
def test_driver_stays_within_its_plumbing_budget(sources, mod):
    hits = scan_plumbing_imports(sources[mod], mod)
    budget = _DRIVER_PLUMBING_BUDGET[mod]
    assert len(hits) <= budget, (
        f"{mod} reaches under jax at {len(hits)} sites, budget {budget}: "
        f"{hits}.  A driver should read as physics — route this through "
        f"common.collectives (mesh, gather, barrier) or a named kernel.")


@pytest.mark.parametrize("mod", sorted(_DRIVER_PLUMBING_BUDGET))
def test_the_plumbing_budget_is_tight(sources, mod):
    """The ratchet.  A budget above the real count is a licence to regress."""
    hits = scan_plumbing_imports(sources[mod], mod)
    budget = _DRIVER_PLUMBING_BUDGET[mod]
    assert len(hits) == budget, (
        f"{mod} now has {len(hits)} plumbing imports but is budgeted "
        f"{budget}.  Lower the budget in this file to {len(hits)} and keep "
        f"the win.")


def test_the_plumbing_scan_can_fail():
    """RED TWIN for rule 1 — the same scanner, on sources that violate it.

    Five spellings, because the tree contains five.  A scan that catches
    only ``from jax.sharding import Mesh`` would have passed
    ``bse.bse_ring_comm``, which reaches its ``shard_map`` through a
    ``try: from jax import shard_map / except ImportError`` shim.
    """
    cases = [
        ("from jax.sharding import Mesh, NamedSharding\n", "jax.sharding"),
        ("from jax.experimental.shard_map import shard_map\n",
         "jax.experimental.shard_map"),
        ("from jax import shard_map as _sm\n", "jax.shard_map"),
        ("import jax.experimental.multihost_utils\n",
         "jax.experimental.multihost_utils"),
        ("def f():\n    from jax.sharding import PartitionSpec\n",
         "jax.sharding"),
    ]
    for src, expected in cases:
        hits = scan_plumbing_imports(src)
        assert hits and hits[0][0] == expected, (
            f"the plumbing scan does not detect {expected!r} in {src!r} — "
            f"it found {hits}")


def test_the_plumbing_scan_does_not_cry_wolf():
    """...and it must stay quiet on what a driver IS allowed to say."""
    ok = ("import jax\n"
          "import jax.numpy as jnp\n"
          "from jax import lax\n"
          "from common.collectives import prepare_mesh, gather_to_host\n"
          "mesh = prepare_mesh()\n")
    assert scan_plumbing_imports(ok) == [], (
        "the plumbing scan flags the sanctioned spelling; it would make the "
        "gate unusable and it would be turned off")


# ===========================================================================
# 3.  RULE 2 — an L2 module reads no environment
# ===========================================================================
#
# L2 is physics-agnostic mathematics.  A Davidson solve that behaves
# differently because of an exported variable is not a function of its
# arguments, and the caller cannot see why.  Every dial an L2 routine needs
# is a PARAMETER; if it must come from the environment, the driver reads it
# and passes it.
#
# The model already exists one layer down: ``common.contract_bands`` makes no
# ``os.environ`` call at all — it consumes ``ffi.mklblas.gemm.GATE``, a typed
# capability object.  A kernel taking a ``Gate`` rather than a string is the
# target state.

_L2_ENV_EXCEPTIONS = {
    # ``bool(int(os.environ.get('STERN_DEBUG', '0')))`` at module scope, so a
    # word spelling raises ValueError on IMPORT.  This module also imports
    # ``psp.dft_operators`` (see _L2_UPWARD_EXCEPTIONS) — it is really an L1
    # physics kernel wrapped around an L2 CG solve, and the two questions
    # have one answer.  Numbered request.
    "solvers.sternheimer_solve": {"STERN_DEBUG"},
    # ``os.environ.setdefault("JAX_ENABLE_X64", "1")`` at module scope: a
    # LIBRARY mutating global jax configuration for whoever imports it.  Same
    # class as ``centroid.kmeans_isdf``'s ``config.update`` at module scope,
    # which was removed on 2026-07-30 once ``kmeans_cli.bootstrap()`` owned
    # it.  Deleting it here is NOT free: the sole consumer
    # (``gw.sc_iteration:577``) is lazy and always post-bootstrap, but a
    # bare ``import mixing.acceleration`` in a fresh process would then run
    # Anderson/CROP in f32 silently.  Physics decision.  Numbered request.
    "mixing.acceleration": {"JAX_ENABLE_X64"},
}


def test_no_l2_module_reads_the_environment(sources):
    offenders = {}
    for mod, src in sources.items():
        if layer_of(mod) != "L2":
            continue
        keys = {k for _, k, _ in scan_env(src)}
        keys -= _L2_ENV_EXCEPTIONS.get(mod, set())
        if keys:
            offenders[mod] = sorted(keys)
    assert not offenders, (
        f"L2 is physics-agnostic mathematics and must be a function of its "
        f"arguments; these read the environment: {offenders}.  Pass the dial "
        f"in, or take a typed capability object the way "
        f"common.contract_bands takes ffi.mklblas.gemm.GATE.")


def test_the_l2_env_exceptions_are_all_still_needed(sources):
    """The ratchet.  An exception that outlives its violation is a lie."""
    stale = {}
    for mod, keys in _L2_ENV_EXCEPTIONS.items():
        assert mod in sources, f"{mod} is excepted here but does not exist"
        assert layer_of(mod) == "L2", (
            f"{mod} is excepted from an L2 rule but is classified "
            f"{layer_of(mod)}")
        live = {k for _, k, _ in scan_env(sources[mod])}
        gone = sorted(keys - live)
        if gone:
            stale[mod] = gone
    assert not stale, (
        f"these env exceptions no longer describe anything real — delete "
        f"them: {stale}")


def test_the_env_scan_can_fail():
    """RED TWIN for rule 2, over all five spellings the tree actually uses."""
    cases = [
        ("x = os.environ.get('A')\n", "read", "A"),
        ("x = os.getenv('B')\n", "read", "B"),
        ("x = os.environ['C']\n", "read", "C"),
        ("os.environ['D'] = '1'\n", "write", "D"),
        ("os.environ.setdefault('E', '1')\n", "write", "E"),
    ]
    for src, kind, key in cases:
        hits = scan_env(src)
        assert (kind, key) in [(k, n) for k, n, _ in hits], (
            f"the env scan missed {kind} of {key} in {src!r}: {hits}")


def test_the_env_scan_separates_reads_from_writes():
    """``os.environ['X']`` on the LEFT of ``=`` is a write, not a read.

    Without this the write rule below double-counts and the read rule fires
    on a module that only ever sets.
    """
    hits = scan_env("os.environ['X'] = '1'\n")
    assert hits == [("write", "X", 1)], hits


# ===========================================================================
# 3b.  RULE 2b — a budgeted L1 LIBRARY's env reads are pinned + resolver-scoped
# ===========================================================================
#
# layers.md's acceptance test for a driver says "no ``os.environ``", and L2
# is covered by rule 2 above — but the L1 *libraries* (htransform,
# bse_setup) had no ratchet at all, and grew three env reads in one wave
# (P1.4), one of them a MODULE-SCOPE ``float(os.environ.get(...))`` whose
# malformed export crashed ``import bandstructure.htransform`` itself.
#
# THE TABLE IS A RATCHET, same rules as _DRIVER_PLUMBING_BUDGET: what each
# file reads TODAY, by name.  A new env read fails; removing one without
# editing the table fails; and every read must live inside a module-level
# ``resolve*`` function (the named-resolver pattern: blank→default,
# announce-or-refuse, callable from the entry layer so a driver can pass
# the value DOWN instead) — module scope is the import-time-crash class and
# is banned outright.

_L1_LIBRARY_ENV_READS = {
    # resolve_galerkin_chunk_bytes / resolve_extra_rank_pad /
    # resolve_fh_ortho_tol / resolve_rank_policy_mode — one resolver each,
    # refuse-on-garbage; the kwarg/entry layer can override by passing values
    # down.  The last one carries the AUTHORITY of the rank-truncation gate
    # (docs/dev/rank_truncation_policy.md); its name and grammar live in
    # common/rank_criterion, which is L2 and may not read the environment, so
    # the driver looks it up and passes it in.
    "bandstructure.htransform": {
        "LORRAX_GALERKIN_CHUNK_GIB",
        "LORRAX_EXTRA_RANK_PAD",
        "LORRAX_FH_ORTHO_TOL",
        "LORRAX_RANK_POLICY",
    },
    # resolve_reshard_route — explicit kwarg wins, env is the A/B path the
    # production exciton_bands driver cannot plumb an argument to (file
    # ownership); unknown tokens announce, never silently.
    # resolve_fi_fshoulder_tol — the f-shoulder gate's floor, one resolver,
    # refuse-on-garbage, same shape as ``resolve_fh_ortho_tol`` in the sibling
    # module four lines up.  It is deliberately NOT a driver flag: the gate
    # catches a returned band that is ABSENT from fH, which is not a
    # tolerance question, and the only value that changes its verdict is a
    # negative one, whose whole purpose is reproducing a known-bad run.
    "bandstructure.bse_setup": {
        "LORRAX_FACE_TO_BATCH_ROUTE",
        "LORRAX_FI_FSHOULDER_TOL",
    },
}


#: Grammar helpers that read the environment on the caller's behalf — an
#: ``env_float("LORRAX_X", ...)`` call is an env read exactly as much as
#: ``os.environ.get`` is, and the P1.3 funnelling of raw reads through
#: these is precisely what blinded the raw-read-only audit (the
#: false-clean sweep tools/env_audit.py documents).
_ENV_HELPER_NAMES = {"env_bool", "env_float"}


def _helper_env_reads(tree):
    """``[(node, key, line)]`` for grammar-helper calls with a literal key."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else (
                fn.attr if isinstance(fn, ast.Attribute) else None)
            if name in _ENV_HELPER_NAMES and node.args:
                out.append((node, _literal(node.args[0]), node.lineno))
    return out


def scan_l1_env_reads(source: str):
    """Every env read in an L1 library, raw OR helper-mediated.

    ``[(key, line)]`` — the union rule 2b's pinned sets are compared
    against.
    """
    tree = ast.parse(source)
    out = [(k, ln) for kind, k, ln in scan_env(source) if kind == "read"]
    out.extend((k, ln) for _, k, ln in _helper_env_reads(tree))
    return sorted(out, key=lambda h: h[1])


def scan_env_reads_outside_resolvers(source: str):
    """Env READS (raw or helper-mediated) not inside a ``resolve*`` function.

    ``[(key, line)]``.  Reads inside any function whose name starts with
    ``resolve`` (public or ``_resolve``-private) are the sanctioned
    pattern; everything else — module scope included — is reported.
    py3.7-safe: membership is computed by walking each resolver's SUBTREE
    (no ``end_lineno`` on that interpreter).
    """
    tree = ast.parse(source)
    sanctioned = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and \
                node.name.lstrip("_").startswith("resolve"):
            for sub in ast.walk(node):
                sanctioned.add(id(sub))
    out = []
    for node in ast.walk(tree):
        if id(node) in sanctioned:
            continue
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr, val = node.func.attr, node.func.value
            if attr in ("get", "getenv") and (
                    "environ" in ast.dump(val)
                    or (isinstance(val, ast.Name) and val.id == "os")):
                key = _literal(node.args[0]) if node.args else "<none>"
                out.append((key, node.lineno))
        elif isinstance(node, ast.Subscript) and \
                "environ" in ast.dump(node.value):
            out.append((_literal(node.slice), node.lineno))
    for node, key, line in _helper_env_reads(tree):
        if id(node) not in sanctioned:
            out.append((key, line))
    return sorted(out, key=lambda h: h[1])


@pytest.mark.parametrize("mod", sorted(_L1_LIBRARY_ENV_READS))
def test_l1_library_env_reads_are_pinned(sources, mod):
    """Exact-set ratchet, both directions."""
    assert mod in sources, f"{mod} is budgeted here but no longer exists"
    live = {k for k, _ in scan_l1_env_reads(sources[mod])}
    pinned = _L1_LIBRARY_ENV_READS[mod]
    new = sorted(live - pinned)
    assert not new, (
        f"{mod} grew env reads {new}.  An L1 library dial is a PARAMETER; "
        f"read it in the entry layer and pass it down, or add a resolve* "
        f"function AND this table entry in the same commit (layers.md "
        f"acceptance test).")
    gone = sorted(pinned - live)
    assert not gone, (
        f"{mod} no longer reads {gone} — lower the table here and keep the "
        f"win.")


@pytest.mark.parametrize("mod", sorted(_L1_LIBRARY_ENV_READS))
def test_l1_library_env_reads_live_in_resolvers(sources, mod):
    """No module-scope read (import-time crash class), no scattered reads."""
    stray = scan_env_reads_outside_resolvers(sources[mod])
    assert not stray, (
        f"{mod} reads the environment outside a resolve* function at "
        f"{stray}.  A module-scope parse crashes the IMPORT on a malformed "
        f"export (the LORRAX_GALERKIN_CHUNK_GIB defect, P1.4); move the "
        f"read into a named resolver with refuse-on-garbage.")


def test_the_resolver_scan_can_fail():
    """RED TWIN for rule 2b — module scope, in-function, and subscript."""
    bad_module_scope = "import os\nX = float(os.environ.get('A', '6'))\n"
    assert scan_env_reads_outside_resolvers(bad_module_scope) == [("A", 2)]
    bad_fn = "import os\ndef work():\n    return os.getenv('B')\n"
    assert scan_env_reads_outside_resolvers(bad_fn) == [("B", 3)]
    bad_sub = "import os\ndef work():\n    return os.environ['C']\n"
    assert scan_env_reads_outside_resolvers(bad_sub) == [("C", 3)]
    bad_helper = ("def work():\n"
                  "    return env_float('E', 1e-6, refuse=True)\n")
    assert scan_env_reads_outside_resolvers(bad_helper) == [("E", 2)], (
        "a helper-mediated read outside a resolver must be reported — the "
        "helper funnel is how the raw-read audit went blind")
    ok = ("import os\ndef resolve_thing():\n"
          "    a = os.environ.get('D', '1')\n"
          "    return a, env_bool('D2', False)\n")
    assert scan_env_reads_outside_resolvers(ok) == [], (
        "the scan flags the sanctioned resolver pattern; the gate would be "
        "turned off")


# ===========================================================================
# 4.  RULE 3 — imports run downhill
# ===========================================================================
#
# L3 must not import L1; L2 must not import L1.  This is the rule that makes
# the levels mean anything: a substrate module that reaches up into the GW
# driver is not a substrate, it is part of GW with a misleading address.

# EMPTY since 2026-08-06.  The one entry was ``SlabIOBackend`` — the enum
# that typed the file-IO service's OWN backend parameter, defined in the GW
# driver's 2.3 kLoC deck parser, which the transport imported lazily at two
# sites with the comment "avoid circular import at module load".  The
# numbered request asked where the enum should live, given that ``file_io``
# imports jax and h5py at package scope while ``gw_config`` is (nearly)
# jax-free.  The answer turned out to be nowhere: with one transport there
# is no enum, so the uphill import is deleted rather than rehomed.
_L3_UPWARD_EXCEPTIONS = {}

_L2_UPWARD_EXCEPTIONS = {
    # ``from psp.dft_operators import apply_H_k_from_G``, module scope, used
    # at :143.  A solver that applies a specific DFT Hamiltonian is not
    # physics-agnostic: the CG core is L2, this file is L1 wrapped around it.
    # Either split ``SternheimerOp``'s operator out, or move the file to
    # ``psp/``.  Physics decision.  Numbered request.
    ("solvers.sternheimer_solve", "psp.dft_operators"): 1,
}
# R3 -- ``centroid.kmeans_isdf`` -> ``centroid.orbit_syms``, 1 lazy site --
# was the second entry.  DELETED, not moved: ``centroid/orbit_syms.py`` left
# ``src/`` for ``services/symmetry_maps/``, and ``kmeans_isdf.py:583`` now
# takes ``canonicalize_orbit`` from the ``symmetry_maps`` DOOR, where §8's
# rule 6 governs it.  See :func:`upward_edges` for why an edge into a
# service door is not ranked, and ``docs/architecture/layers.md`` §5.
# The prescribed fix (inject the orbit map; a signature change) is
# unaffected and stays registered -- extraction changed the module's
# address, not its argument list.


def upward_edges(sources):
    """Every import that goes UP a level. ``[(from_layer, to_layer, mod, target, line, lazy)]``.

    A SERVICE DOOR IS NOT RANKED, and that carve-out is measured, not
    assumed.  Levels order modules within ONE distribution unit; a service
    under ``services/`` is a separate installable package reached through a
    declared door, and §8's rule 6 is what governs an edge into it.
    Ranking one anyway forces a false choice, which ``symmetry_maps`` is
    the first service to expose because it is the first that is not
    substrate.  MEASURED at the extraction commit, both ways::

        symmetry_maps at the L1 default (honest: it is crystal physics)
            L2->L1 centroid.kmeans_isdf:583 (lazy) imports symmetry_maps

        symmetry_maps forced into _L3_PACKAGES (the distrib_la shape)
            L3->L1 symmetry_maps.density_symmetry_check:658 (lazy)
                   imports psp.get_DFT_mtxels
            L3->L1 symmetry_maps.density_symmetry_check:749 (lazy)
                   imports psp.get_DFT_mtxels

    Two exceptions to choose between, both describing the same fact from
    opposite ends -- and the second one buys its green by calling a
    package full of Brillouin zones "substrate".  The package stays at the
    L1 default, which is what it is, and the edge INTO it stops being a
    level question.

    NOTHING IS LEFT UNWATCHED BY THIS.  Reaching past a door still counts
    (rule 6, and the shims' budgets below are the checklist).  A service
    reaching back into ``src/`` still ranks -- the psp edges above are
    real, they are ``symmetry_maps``'s one remaining lorrax dependency,
    and the instrument that owns them is the mandatory import-isolation
    test (import the service with lorrax off ``sys.path``) plus the
    ``dependencies`` list in its ``pyproject.toml``, neither of which a
    level assignment could have replaced.
    """
    mods = set(sources)
    pkgs = packages_in(sources)
    out = []
    for mod, src in sources.items():
        here = layer_of(mod)
        if here == "X":
            continue
        for target, line, lazy in scan_imports(src, mod, mod in pkgs):
            parts = target.split(".")
            resolved = None
            for i in range(len(parts), 0, -1):
                cand = ".".join(parts[:i])
                if cand in mods:
                    resolved = cand
                    break
            if resolved is None or resolved == mod:
                continue
            if resolved in _SERVICE_DOORS and mod.split(".")[0] != resolved:
                continue                      # rule 6's business; see above
            there = layer_of(resolved)
            if there == "X" or _RANK[there] < _RANK[here]:
                out.append((here, there, mod, resolved, line, lazy))
    return out


def test_no_module_imports_a_lower_numbered_layer(sources):
    excused = dict(_L3_UPWARD_EXCEPTIONS)
    excused.update(_L2_UPWARD_EXCEPTIONS)
    bad = []
    seen = {}
    for here, there, mod, target, line, lazy in upward_edges(sources):
        key = (mod, target)
        seen[key] = seen.get(key, 0) + 1
        if key in excused:
            continue
        bad.append(f"{here}->{there} {mod}:{line}"
                   f"{' (lazy)' if lazy else ''} imports {target}")
    assert not bad, (
        "imports must run downhill (L1 -> L2 -> L3); these go up:\n  "
        + "\n  ".join(sorted(bad)))
    over = {k: (n, excused[k]) for k, n in seen.items()
            if k in excused and n > excused[k]}
    assert not over, (
        f"an excused upward edge grew new sites — the exception was for the "
        f"count on the left, not for the direction: {over}")


def test_the_upward_exceptions_are_all_still_needed(sources):
    """The ratchet again: no exception outlives the edge it excuses."""
    live = {(m, t) for _, _, m, t, _, _ in upward_edges(sources)}
    excused = set(_L3_UPWARD_EXCEPTIONS) | set(_L2_UPWARD_EXCEPTIONS)
    stale = sorted(excused - live)
    assert not stale, (
        f"these upward-import exceptions describe edges that no longer "
        f"exist — delete them: {stale}")


def test_the_upward_edge_scan_can_fail(sources):
    """RED TWIN for rule 3.

    Seeded with a REAL pair from this tree, so it cannot pass by testing a
    fiction: ``common.collectives`` is L3 and ``gw.gw_config`` is L1, and the
    scanner must call that an upward edge from a lazy, function-scope import
    — which is how every one of the four live ones is written.
    """
    assert layer_of("common.collectives") == "L3"
    assert layer_of("gw.gw_config") == "L1"
    fake = dict(sources)
    fake["common.collectives"] = (
        "def f():\n    from gw.gw_config import read_lorrax_input\n")
    edges = [(m, t) for _, _, m, t, _, _ in upward_edges(fake)]
    assert ("common.collectives", "gw.gw_config") in edges, (
        f"the upward-edge scan does not see an L3 module importing an L1 "
        f"one from inside a function: {edges}")


def test_the_service_door_carve_out_is_narrow(sources):
    """RED TWIN for the carve-out in :func:`upward_edges`.

    A skip is a hole until something proves where its edges are.  Three
    facts, on the REAL modules the extraction created:

    1. ``centroid.kmeans_isdf`` (L2) importing the ``symmetry_maps`` DOOR
       (L1 by the default, and correctly so) is not an upward edge.
    2. The SUBMODULE spelling of the same import IS still seen — by rule 6,
       which is the rule that owns it.  So the carve-out forgives reaching
       the door, never reaching past it.
    3. An in-tree L2->L1 edge in the same file is still flagged, so the
       carve-out is a service-door exemption and not a kmeans_isdf one.
    """
    assert layer_of("symmetry_maps") == "L1"       # physics, not substrate
    assert layer_of("centroid.kmeans_isdf") == "L2"
    assert "symmetry_maps" in _SERVICE_DOORS

    live = [(m, t) for _, _, m, t, _, _ in upward_edges(sources)]
    assert ("centroid.kmeans_isdf", "symmetry_maps") not in live

    # Visited by the 2026-08-08 rename sweep and deliberately unchanged:
    # ``canonicalize_orbit`` is one of the thirteen names the survey ruled
    # KEEP (pure mathematics, correctly named), so the literal below still
    # names a live export.  It is recorded here because the name appears
    # as SOURCE TEXT the ratchet parses, where a rename tool would not
    # have seen it — the next sweep should re-check it rather than assume.
    past = scan_service_door(
        "def f():\n    from symmetry_maps.orbit_syms import "
        "canonicalize_orbit\n",
        "centroid.kmeans_isdf", "symmetry_maps",
        service_submodules(sources, "symmetry_maps"),
        door_names(sources, "symmetry_maps"))
    assert past == [("symmetry_maps.orbit_syms", 2)], past

    fake = dict(sources)
    fake["centroid.kmeans_isdf"] = (
        "def f():\n    from psp.dft_operators import apply_H_k_from_G\n")
    edges = [(m, t) for _, _, m, t, _, _ in upward_edges(fake)]
    assert ("centroid.kmeans_isdf", "psp.dft_operators") in edges, edges


def test_the_upward_edge_scan_sees_package_relative_imports(sources):
    """RED TWIN for :func:`_absolute_target`'s package case.

    ``from .x import y`` inside a package ``__init__.py`` means ``pkg.x``.  A
    resolver that drops a component instead yields a bare ``x``, which matches
    no module in the tree, so :func:`upward_edges` skips the edge ENTIRELY.
    That was true here until 2026-08-06 for all 34 such imports in the 24
    package ``__init__.py`` files under ``src/`` — ``runtime/__init__.py``'s
    two startup siblings among them.

    The seed is the shape that actually happens, not a fiction: a new file
    under ``runtime/`` that nobody added to the map DEFAULTS TO L1 (§0), and
    ``runtime`` reaches its siblings relatively.  Both of the upward edges
    fixed on 2026-08-06 arose exactly this way.
    """
    assert layer_of("runtime") == "L3"
    assert layer_of("runtime.deck_reader") == "L1", (
        "an unmapped runtime submodule must default to L1, or this twin is "
        "asserting nothing")
    fake = dict(sources)
    fake["runtime"] = "from .deck_reader import read_deck\n"
    fake["runtime.deck_reader"] = "read_deck = None\n"
    edges = [(m, t) for _, _, m, t, _, _ in upward_edges(fake)]
    assert ("runtime", "runtime.deck_reader") in edges, (
        f"a package-relative import inside an __init__.py is invisible to "
        f"rule 3: {edges}")


def test_the_upward_edge_scan_ignores_downhill_imports(sources):
    """...and a downhill import must not be reported, or nobody reads it."""
    fake = dict(sources)
    fake["gw.gw_jax"] = (
        "from common.collectives import prepare_mesh\n"
        "from solvers.davidson import davidson\n"
        "from runtime import bootstrap\n")
    edges = [(m, t) for _, _, m, t, _, _ in upward_edges(fake)
             if m == "gw.gw_jax"]
    assert edges == [], f"downhill imports reported as violations: {edges}"


# ===========================================================================
# 5.  RULE 4 — only the substrate builds a mesh
# ===========================================================================
#
# 21 production ``Mesh(`` sites in five mutually incompatible dialects, as of
# 2026-07-30.  Three of them built ``Mesh(jax.devices()[:1].reshape(1,1))`` —
# process 0's device on EVERY rank — which the addressability guard in
# ``collectives`` now names as "the usual cause" of a bare ``StopIteration``
# at P>1.  The hazard had been written down twice before it propagated to
# three more files.  This is the rule that stops the sixth dialect.

_MESH_OWNERS = {
    "common.collectives",     # resolve_mesh + single_device_mesh: THE two
    "runtime",                # RuntimeStack.reshape, which then calls
                              # prepare_mesh on the result
    # ``create_mesh_2d`` — the ONLY other one left, and it is not a dialect,
    # it is a contract: it builds the mesh AND warms the impl=mpi
    # communicator cliques, without which the TDA Lanczos dies on every rank
    # at P=16 (32 refusals, job 7881216; 0 with it).  Folding it into
    # ``prepare_mesh`` is the open warm-up-contract decision, and doing it by
    # deleting the call site is how the refusals come back.
    "bse.bse_ring_comm",
    # GATE 10 must build a 1x1 mesh over ``jax.devices("cpu")`` INSIDE a
    # process whose default backend is CUDA — that inversion is the thing the
    # gate exists to probe (a CUDA process doing host phdf5 IO).  Both
    # sanctioned constructors resolve the DEFAULT backend, so each would hand
    # this process its GPU: ``single_device_mesh()`` is
    # ``jax.local_devices()[:1]`` and ``resolve_mesh`` squares over
    # ``jax.devices()``.  A build gate is also not runtime code — it runs
    # single-process under ``verify_ffi_build``, so the one-object-per-process
    # jit-cache contract those helpers protect is not in play.
    "ffi.cpp.gate_one_odr",
}


def test_only_the_substrate_constructs_a_mesh(sources):
    offenders = {}
    for mod, src in sources.items():
        if layer_of(mod) == "X" or mod in _MESH_OWNERS:
            continue
        lines = scan_mesh_construction(src)
        if lines:
            offenders[mod] = lines
    assert not offenders, (
        f"these build their own Mesh: {offenders}.  Call "
        f"common.collectives.prepare_mesh (mesh + warm-up) or resolve_mesh "
        f"(mesh only).  ``jax.devices()[:1]`` is process 0's device on every "
        f"rank; ``single_device_mesh()`` is the one you want.")


def test_the_mesh_owners_all_still_build_one(sources):
    """The ratchet: an owner that stopped constructing is no longer an owner."""
    idle = sorted(m for m in _MESH_OWNERS
                  if not scan_mesh_construction(sources[m]))
    assert not idle, (
        f"{idle} are sanctioned mesh owners but construct none — remove them "
        f"from _MESH_OWNERS so the next module cannot inherit the licence")


def test_the_mesh_scan_can_fail():
    """RED TWIN for rule 4, including the attribute spelling."""
    for src in ("m = Mesh(devs.reshape(2, 2), ('x', 'y'))\n",
                "m = jax.sharding.Mesh(d, ('x','y'))\n",
                "def f():\n    return Mesh(np.asarray(jax.devices()[:1]), ())\n"):
        assert scan_mesh_construction(src), (
            f"the mesh scan does not detect a construction in {src!r}")
    assert scan_mesh_construction("mesh = prepare_mesh()\nx = mesh.shape\n") == [], (
        "the mesh scan flags a mesh USE; only construction is banned")


# ===========================================================================
# 6.  RULE 5 — only an entry point writes the environment at import time
# ===========================================================================
#
# jax reads its environment when IT is imported, so a ``setdefault`` is only
# meaningful before that — which is exactly why it keeps being written at the
# top of files.  ``runtime.set_default_env`` is the one place that is allowed
# to make that choice, plus the CLIs that are also standalone scripts and say
# so.  A LIBRARY doing it changes the run for anyone who merely names it.

_MODULE_SCOPE_ENV_WRITE_EXCEPTIONS = {
    # A package ``__init__`` setting jax x64 for ``import gw.<anything>``.
    # Argued in place at :5-18 and shown to be inert for both drivers (they
    # call bootstrap() and then jax.config.update explicitly); kept for the
    # import paths with no bootstrap at all.  Low value to remove, nonzero
    # risk, and the reasoning is already written down.
    "gw": {"JAX_ENABLE_X64"},
    # ``MPLBACKEND=Agg`` before any matplotlib import, because the container
    # has no interactive backend.  Not a compute knob and not jax; a plotting
    # helper choosing a headless renderer is the correct owner of that.
    "centroid.kmeans_plot": {"MPLBACKEND"},
    # See _L2_ENV_EXCEPTIONS: same line, and it fails rule 2 as well.
    "mixing.acceleration": {"JAX_ENABLE_X64"},
}


def test_only_entry_points_write_the_environment_at_import(sources):
    offenders = {}
    for mod, src in sources.items():
        if layer_of(mod) == "X":
            continue
        if mod == "runtime" or mod.startswith("runtime."):
            continue                      # the module whose job this is
        if has_main_block(src):
            continue                      # a declared entry point
        keys = {k for k, _ in scan_module_scope_env_writes(src)}
        keys -= _MODULE_SCOPE_ENV_WRITE_EXCEPTIONS.get(mod, set())
        if keys:
            offenders[mod] = sorted(keys)
    assert not offenders, (
        f"these set environment variables just by being imported, and are "
        f"not entry points: {offenders}.  runtime.set_default_env is the one "
        f"place that decides which variables LORRAX ships.")


def test_the_module_scope_env_write_exceptions_are_still_needed(sources):
    stale = {}
    for mod, keys in _MODULE_SCOPE_ENV_WRITE_EXCEPTIONS.items():
        assert mod in sources, f"{mod} is excepted here but does not exist"
        live = {k for k, _ in scan_module_scope_env_writes(sources[mod])}
        gone = sorted(keys - live)
        if gone:
            stale[mod] = gone
    assert not stale, f"delete these dead exceptions: {stale}"


def test_the_module_scope_env_write_scan_can_fail():
    """RED TWIN for rule 5 — and it must NOT fire on a write inside a def."""
    assert scan_module_scope_env_writes(
        "import os\nos.environ.setdefault('X', '1')\n") == [("X", 2)]
    assert scan_module_scope_env_writes(
        "import os\nif True:\n    os.environ['Y'] = '2'\n") == [("Y", 3)]
    assert scan_module_scope_env_writes(
        "import os\ndef f():\n    os.environ['Z'] = '3'\n") == [], (
        "a write inside a function is a caller's runtime choice and must not "
        "be reported; reporting it would make the rule unenforceable")


# ===========================================================================
# 7.  Map hygiene — the classification itself cannot rot
# ===========================================================================

def test_every_module_in_the_map_exists(sources):
    """A renamed module must not leave a phantom assignment behind."""
    named = set(_L3_MODULES) | set(_L2_MODULES)
    missing = sorted(named - set(sources))
    assert not missing, (
        f"the layer map names modules that no longer exist: {missing}")


def test_every_package_in_the_map_exists(sources):
    """The PACKAGE half of the same ratchet, which did not exist.

    ``test_every_module_in_the_map_exists`` checks ``_L*_MODULES`` only, so
    ``_L3_PACKAGES = ("ffi",)`` could go dead-no-op the day ``src/ffi``
    leaves and nothing would say so — a whole level's worth of files
    silently reverting to the L1 default, which is the most permissive
    level there is.  A package entry is a claim that the package is there.
    """
    pkgs = packages_in(sources)
    for level, names in (("L3", _L3_PACKAGES), ("L2", _L2_PACKAGES)):
        missing = sorted(n for n in names if n not in pkgs)
        assert not missing, (
            f"{level} names package(s) {missing} that are not packages in "
            f"this tree.  Either the package moved (update the entry) or it "
            f"is gone (delete it) — leaving it makes the level a no-op.")


# ===========================================================================
# 8.  RULE 6 — lorrax imports a service through its DOOR, and nothing else
# ===========================================================================
#
# The charter's standalone criterion, made measurable: "Lorrax imports a
# service ONLY through its door module.  Measurable criterion per service:
# direct-dependency import edges from lorrax -> 0 outside doors."  Nothing
# counted them before this rule.
#
# The door of a service is its TOP-LEVEL package.  ``import distrib_la`` and
# ``from distrib_la import plan`` are the contract; ``import
# distrib_la.plan``, ``from distrib_la.plan import plan`` and ``from
# distrib_la import loader`` reach past it — and a reach past the door is
# how a service stops being replaceable, because the thing on the other side
# is no longer the API anybody agreed to.
#
# THE EXEMPTIONS ARE THE REPLUMB CHECKLIST.  Every entry is a module that
# exists only to be deleted; the count can go down and the ratchet below
# fails if an entry stops describing something real.

#: Service top-level package -> the name its door is spelled.  (Identity
#: today; a service whose door is a submodule would say so here.)
#:
#: ``zeta_loader`` joined on 2026-08-07 with its extraction.  Its shim,
#: ``src/file_io/zeta_loader.py``, does NOT appear in the exception table
#: below and must not: it re-exports ``ZetaLoader`` off the top-level
#: package, which is the door, so it is not a past-the-door edge and
#: ``test_the_service_door_exceptions_are_all_still_needed`` would fail on
#: an entry claiming otherwise.
#:
#: ``vcoul`` joined 2026-08-07.  Its shims — ``gw.compute_vcoul``,
#: ``gw.coulomb.*``, ``gw.vcoul``, ``gw.compute_vcoul_0d``,
#: ``file_io.read_bgw_vcoul``, ``common.coulomb_sphere`` — all say ``import
#: vcoul`` / ``from vcoul import <public name>`` and none reaches a
#: submodule, so vcoul needs NO row in ``_SERVICE_DOOR_EXCEPTIONS`` below.
#: That is the point of putting ``_minibz_kernel_bare`` and
#: ``_round_up_fft_size`` on the door despite the underscore: they have real
#: cross-package consumers, and a shim reaching ``vcoul.minibz`` for one of
#: them would have been this rule's first new violation since the replumb.
#: ``minimax`` joined 2026-08-08 with the sixth extraction, and it joined
#: with NO shim and NO row below.  ``gw.minimax_screening`` — the module's
#: one production importer, then and now — says ``import minimax as _mm``
#: and reaches only ``serve`` / ``Quadrature`` / the refusal types, all of
#: which are top-level names.  The offline solver half (``G_hgl``,
#: ``noncrossing_grids``, …) is on the door too, behind a PEP-562 lazy
#: ``__getattr__``, precisely so that the generator tool and the
#: certification tier can have those names without anybody writing
#: ``from minimax.solver import ...`` — which would have been this rule's
#: first new violation since the replumb.
_SERVICE_DOORS = {"lxkit": "lxkit", "distrib_la": "distrib_la",
                  "wfn_loader": "wfn_loader", "zeta_loader": "zeta_loader",
                  "symmetry_maps": "symmetry_maps", "vcoul": "vcoul",
                  "minimax": "minimax"}

#: ``src/`` module -> how many past-the-door edges it is allowed.  These are
#: the transitional re-export shims.
#:
#: WAS 45 EDGES ACROSS 9 MODULES before the replumb (step 3).  Six of the
#: nine are deleted: ``ffi.linalg.{plan,resolve,dispatch,_slate,_scalapack}``
#: (3 + 12 + 2 + 12 + 6) and ``common.cholesky_2d`` (3), with the
#: ``ffi.slate`` and ``ffi.scalapack`` re-export PACKAGES that reached them.
#: Every consumer imports the top-level ``distrib_la`` door now, which this
#: rule does not count because it is not a violation.
#:
#: THE REMAINING SEVEN ARE NOT A REPLUMB LEFTOVER.  ``ffi.cusolvermp`` has a
#: live in-``src/`` reacher that is out of distrib_la's scope entirely --
#: ``ffi/cublasmp/batched.py:33`` takes ``get_or_init_context`` from it, and
#: cuBLASMp is a future ``gemm`` service, not this one.  Converting that edge
#: would mean lorrax importing ``distrib_la._cusolvermp`` directly, i.e.
#: trading three counted exceptions for one uncounted violation.  It goes
#: when gemm is extracted; until then the number is 7 and it is honest.
#:
#: 7 -> 10 AT THE symmetry_maps EXTRACTION.  Three forwarding shims, one
#: past-the-door edge each: ``from symmetry_maps import <submodule> as
#: _impl``, which is what makes their PEP 562 ``__getattr__`` able to
#: forward the PRIVATE names (``_I_SIGMA_Y``, ``_star_conj_flags``,
#: ``TOL_TRS``, ``_CACHE``, …) that tests, ``misc/`` and uncollected gates
#: still take from the old paths.  Their public re-exports come through the
#: door and are not counted, because they are not violations.
#:
#: THE GATE, NAMED: all three are deleted by the phase-wide cleanup commit
#: after all four wave-1 branches land (WAVE1_BRIEF ruling 2).  Unlike
#: distrib_la, whose own replumb deleted its shims, wave-1 services are
#: cross-consumed by files on sibling branches, so the deletion is one
#: commit after the owner merges all four.  Until then these three budgets
#: are 1 apiece and cannot grow: a shim's job is to forward, not to
#: acquire reasons to exist.
#: THE THREE symmetry_maps ROWS ARE GONE, and their absence is the record
#: that the phase-wide shim deletion actually happened.  ``common.symmetry_maps``,
#: ``common.density_symmetry_check`` and ``centroid.orbit_syms`` were
#: ``__getattr__``-forwarding shims that reached past the door to keep the
#: service's private names resolving at the old paths; all three were
#: deleted at the 2026-08-08 landing and this ratchet is what noticed --
#: it failed with "these modules no longer reach past a service door --
#: delete the exception, and probably the module" the moment they went.
#: An exception table that outlives its shims is how a replumb gets
#: declared finished while the old paths are still there.
_SERVICE_DOOR_EXCEPTIONS = {
    "ffi.cusolvermp.batched": 5,
    "ffi.cusolvermp.eigh":    1,
    "ffi.cusolvermp.context": 1,
}


def service_submodules(sources, service: str) -> set:
    """Leaf names directly under ``service`` that are MODULES.

    Needed because ``from distrib_la import plan`` is ambiguous in the AST:
    ``plan`` is both a submodule and a name the door re-exports, and the
    door's binding is what an importer actually gets.  So a name is a
    past-the-door reach only if it is a submodule the door does NOT
    re-export.
    """
    n = len(service) + 1
    return {m[n:] for m in sources
            if m.startswith(service + ".") and "." not in m[n:]}


def door_names(sources, service: str) -> set:
    """The strings in the service door's ``__all__``.

    Read from the AST, not by importing: this suite must keep running on
    any interpreter that can parse the tree, with no jax and no service
    installed.
    """
    src = sources.get(service)
    if src is None:
        return set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "__all__"
                for t in node.targets):
            try:
                return set(ast.literal_eval(node.value))
            except Exception:                                # noqa: BLE001
                return set()
    return set()


def scan_service_door(source: str, mod: str, service: str,
                      submodules, doors, is_pkg: bool = False):
    """Imports of ``service`` that go PAST its door. ``[(spelling, line)]``.

    Three spellings, because the tree can write three:
    ``import svc.sub`` / ``from svc.sub import x`` / ``from svc import
    <submodule>``.  The last one is the one a hurried replumb writes.
    """
    hits = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith(service + "."):
                    hits.append((a.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            tgt = _absolute_target(node, mod, is_pkg)
            if tgt.startswith(service + "."):
                hits.append((tgt, node.lineno))
            elif tgt == service:
                for a in node.names:
                    if a.name in submodules and a.name not in doors:
                        hits.append((service + "." + a.name, node.lineno))
    return sorted(hits, key=lambda h: h[1])


def _door_violations(sources):
    """``{src module: [(spelling, line), …]}`` for every past-the-door edge."""
    pkgs = packages_in(sources)
    lorrax = modules_under_src(sources)
    per_service = {svc: (service_submodules(sources, svc),
                         door_names(sources, door))
                   for svc, door in _SERVICE_DOORS.items()}
    out = {}
    for mod in sorted(lorrax):
        hits = []
        for svc, (subs, doors) in per_service.items():
            hits += scan_service_door(sources[mod], mod, svc, subs, doors,
                                      mod in pkgs)
        if hits:
            out[mod] = sorted(hits, key=lambda h: h[1])
    return out


def test_lorrax_reaches_a_service_only_through_its_door(sources):
    bad = {m: h for m, h in _door_violations(sources).items()
           if m not in _SERVICE_DOOR_EXCEPTIONS}
    assert not bad, (
        f"these src/ modules import past a service's door: {bad}.  A service "
        f"is imported by its TOP-LEVEL package only ("
        f"{', '.join(sorted(_SERVICE_DOORS))}); reaching a submodule is what "
        f"stops it being replaceable.")
    over = {m: (len(h), _SERVICE_DOOR_EXCEPTIONS[m])
            for m, h in _door_violations(sources).items()
            if m in _SERVICE_DOOR_EXCEPTIONS
            and len(h) > _SERVICE_DOOR_EXCEPTIONS[m]}
    assert not over, (
        f"a transitional shim grew new past-the-door edges — the exception "
        f"was for the count on the left, not for the direction: {over}")


def test_the_service_door_exceptions_are_all_still_needed(sources):
    """The ratchet.  A shim that stopped reaching past the door is a shim
    that can be deleted, and this is where that is noticed."""
    live = _door_violations(sources)
    stale = sorted(set(_SERVICE_DOOR_EXCEPTIONS) - set(live))
    assert not stale, (
        f"these modules no longer reach past a service door — delete the "
        f"exception, and probably the module: {stale}")
    loose = {m: (len(live[m]), n) for m, n in _SERVICE_DOOR_EXCEPTIONS.items()
             if m in live and len(live[m]) < n}
    assert not loose, (
        f"budgets above the real count are a licence to regress; lower them "
        f"and keep the win: {loose}")


def test_the_retired_shim_modules_are_gone(sources):
    """DELETED, not merely unreachable.

    The replumb's own gate, in the form its subjects asked for: every one
    of these modules carried "DELETING THIS PACKAGE IS THE REPLUMB-COMPLETE
    GATE" or "Deletion is the replumb-complete gate" in its docstring.  A
    re-export shim that nothing imports is not harmless -- it is a second
    spelling of the door that still works, so the next call site can be
    written against it and the layering rule above will not fire, because
    a shim importing past the door is exactly what the exception list
    forgives.  The only enforceable end state is absence.

    Checked against the AST source map rather than by importing, so this
    stays runnable with no jax and no service installed.

    RED ARM: ``test_the_service_door_exceptions_are_all_still_needed`` is
    the other direction -- it fails if one of the three SURVIVING
    exceptions stops describing something real.  Between them a module
    cannot be quietly resurrected or quietly abandoned.
    """
    retired = (
        "ffi.linalg", "ffi.linalg.plan", "ffi.linalg.resolve",
        "ffi.linalg.dispatch", "ffi.linalg._slate", "ffi.linalg._scalapack",
        "ffi.slate", "ffi.slate.batched", "ffi.slate.cholesky",
        "ffi.slate.context", "ffi.slate.eigh", "ffi.slate.trsm",
        "ffi.scalapack", "ffi.scalapack.eigh", "ffi.scalapack.solve_lu",
        "common.cholesky_2d",
    )
    alive = sorted(m for m in retired if m in sources)
    assert not alive, (
        f"these re-export shims were deleted by the replumb and are back: "
        f"{alive}.  Each one is a second spelling of a service door; the "
        f"door is the package (import distrib_la), and nothing else.")


def test_the_service_door_scan_can_fail(sources):
    """RED TWIN for rule 6, over all three spellings the tree can write —
    seeded with REAL names, so it cannot pass by testing a fiction."""
    subs = service_submodules(sources, "distrib_la")
    doors = door_names(sources, "distrib_la")
    assert "loader" in subs and "loader" not in doors, (
        "this twin needs a real distrib_la submodule that the door does NOT "
        "re-export; 'loader' stopped being one")
    assert "plan" in subs and "plan" in doors, (
        "this twin needs a name that is BOTH a submodule and a door export, "
        "or the third spelling's carve-out is untested")
    cases = [
        ("import distrib_la.loader\n",                "distrib_la.loader"),
        ("from distrib_la.plan import plan\n",        "distrib_la.plan"),
        ("from distrib_la import loader\n",           "distrib_la.loader"),
        ("def f():\n    from distrib_la import _slate\n", "distrib_la._slate"),
    ]
    for src, expected in cases:
        hits = scan_service_door(src, "m", "distrib_la", subs, doors)
        assert hits and hits[0][0] == expected, (
            f"the service-door scan does not detect {expected!r} in {src!r} "
            f"— it found {hits}")


def test_the_service_door_scan_does_not_cry_wolf(sources):
    """...and the DOOR itself must stay quiet, or the rule is unusable."""
    subs = service_submodules(sources, "distrib_la")
    doors = door_names(sources, "distrib_la")
    ok = ("import distrib_la\n"
          "from distrib_la import plan, Plan, factor, solve, BACKEND_CHOICES\n"
          "p = plan('eigh', mesh)\n")
    assert scan_service_door(ok, "m", "distrib_la", subs, doors) == [], (
        "the service-door scan flags the door itself; the gate would be "
        "turned off")


def test_the_wfn_loader_door_scan_can_fail(sources):
    """RED TWIN for rule 6 over the SECOND extracted service.

    A per-service twin, not a second copy of the same one: the rule is
    parameterized by ``_SERVICE_DOORS`` and each service supplies its own
    real submodule names, so a twin seeded from distrib_la says nothing
    about whether ``wfn_loader`` was wired into the map at all.  The
    ``subs`` assertion below is what makes that concrete — it fails if the
    service is absent, renamed, or reduced to a single-module package with
    nothing behind its door.

    ``wfn_loader`` has no name that is BOTH a submodule and a door export
    (distrib_la's ``plan`` is), so the third spelling is seeded with a
    plain submodule and the ambiguous case stays covered by the
    distrib_la twin above, which is the one that has it.
    """
    subs = service_submodules(sources, "wfn_loader")
    doors = door_names(sources, "wfn_loader")
    assert "loader" in subs and "loader" not in doors, (
        "this twin needs a real wfn_loader submodule that the door does "
        "NOT re-export; 'loader' stopped being one")
    assert "WfnLoader" in doors, (
        "the wfn_loader door exports no WfnLoader — either the service is "
        "gone or its __all__ stopped being readable from the AST, and in "
        "both cases the carve-out below tests nothing")
    cases = [
        ("import wfn_loader.loader\n",                    "wfn_loader.loader"),
        ("from wfn_loader.loader import WfnLoader\n",     "wfn_loader.loader"),
        ("from wfn_loader import loader\n",               "wfn_loader.loader"),
        ("def f():\n    from wfn_loader import _collectives\n",
         "wfn_loader._collectives"),
    ]
    for src, expected in cases:
        hits = scan_service_door(src, "m", "wfn_loader", subs, doors)
        assert hits and hits[0][0] == expected, (
            f"the service-door scan does not detect {expected!r} in {src!r} "
            f"— it found {hits}")


def test_the_wfn_loader_door_scan_does_not_cry_wolf(sources):
    """...and the shim that KEEPS the old spelling alive must stay quiet.

    ``src/file_io/wfn_loader.py`` is a transitional re-export shim whose
    whole body is the second line below, underscored helper names
    included.  If those read as past-the-door reaches the rule fires on
    the one module written specifically to obey it, and the gate gets an
    exception entry it does not need — which is how ``_SERVICE_DOOR_
    EXCEPTIONS`` stops being a replumb checklist.
    """
    subs = service_submodules(sources, "wfn_loader")
    doors = door_names(sources, "wfn_loader")
    ok = ("import wfn_loader\n"
          "from wfn_loader import WfnLoader, KSpec, "
          "_phdf5_unfold_kernel\n"
          "w = WfnLoader(path)\n")
    assert scan_service_door(ok, "m", "wfn_loader", subs, doors) == [], (
        "the service-door scan flags the door itself; the gate would be "
        "turned off")


def test_no_module_is_in_two_levels():
    both = sorted(_L3_MODULES & _L2_MODULES)
    assert not both, both
    for pkg in _L3_PACKAGES:
        assert pkg not in _L2_PACKAGES
    for mod in _L3_MODULES:
        assert mod.split(".")[0] not in _L2_PACKAGES, mod
    for mod in _L2_MODULES:
        assert mod.split(".")[0] not in _L3_PACKAGES, mod


def test_every_module_gets_a_level(sources):
    """No module may be unclassified — the default is L1 and that is a
    decision, not an accident."""
    for mod in sources:
        assert layer_of(mod) in ("L1", "L2", "L3", "X"), mod


def test_the_src_tree_grows_no_new_bench_drivers(sources):
    """The 26 exempt bench/smoke drivers moved to ``tests/bench/``
    (2026-07-31), taking their copy-pasted jax preambles — 104 of
    ``common/``'s 121 environment-read matches — with them.  This pins the
    exemption count at ZERO: a module under ``src/`` whose name has a bench
    shape would be exempt from every rule above, so none may exist.
    """
    found = sorted(m for m in sources if _is_standalone_driver(m))
    assert not found, (
        f"{len(found)} bench/test driver(s) under src/, must be 0: "
        f"{found}.  Bench and smoke drivers live in tests/bench/.")


def test_the_bench_driver_classifier_can_fail():
    """RED TWIN: the classifier must actually recognise the shapes it lists."""
    for name in ("common.slate_eigh_test", "common.eigh_benchmark",
                 "common.slate_chol_trsm_bench", "psp.tests.test_x",
                 "ffi.cusolvermp.profile_batched"):
        assert _is_standalone_driver(name), name
    for name in ("gw.gw_jax", "common.collectives", "solvers.davidson",
                 "bse.bse_jax", "common.contract_bands"):
        assert not _is_standalone_driver(name), (
            f"{name} would be exempted from every layering rule")


def test_the_three_levels_are_each_populated(sources):
    """A level with nothing in it means the map stopped being maintained."""
    counts = {}
    for mod in sources:
        counts[layer_of(mod)] = counts.get(layer_of(mod), 0) + 1
    for level in ("L1", "L2", "L3"):
        assert counts.get(level, 0) >= 4, (level, counts)


# ===========================================================================
# Standalone runner — ``python3 tests/test_layering.py`` on any interpreter
# that can parse the tree (the point of a pure-AST suite).  pytest collection
# is unaffected: nothing below starts with ``test_``.
# ===========================================================================

def _main():
    import inspect

    srcs = all_modules()
    n_run, failures = 0, []
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        cases = [("", {})]
        for mark in getattr(fn, "pytestmark", []):
            if getattr(mark, "name", "") == "parametrize":
                argname, values = mark.args[0], mark.args[1]
                cases = [("[%s]" % v, {argname: v}) for v in values]
        needs_sources = "sources" in inspect.signature(fn).parameters
        for suffix, kw in cases:
            if needs_sources:
                kw = dict(kw, sources=srcs)
            n_run += 1
            try:
                fn(**kw)
            except Exception as exc:        # noqa: BLE001 — this IS the tally
                msg = (str(exc).splitlines() or [""])[0]
                failures.append("FAIL %s%s — %s: %s"
                                % (name, suffix, type(exc).__name__, msg))
                print(failures[-1])
            else:
                print("ok   %s%s" % (name, suffix))
    print("\n%d/%d passed, %d failed"
          % (n_run - len(failures), n_run, len(failures)))
    for line in failures:
        print(line)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())

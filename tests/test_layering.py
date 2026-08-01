"""THE layering gate: every module in ``src/`` has a level, and the levels hold.

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


SRC = pathlib.Path(__file__).resolve().parents[1] / "src"


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
    # mesh has to divide
    "runtime", "runtime.aot_memory", "runtime.padding", "runtime.xla_memory",
    # the cross-process service and its one policy client
    "common.collectives", "centroid.distribution",
    # movement primitives.  Named, owned patterns — NOT generic ``shard_map``
    # wrappers, which is a refusal recorded in docs/architecture/layers.md.
    "common.contract_bands", "common.staged_reshard", "common.sharding_fit",
    # jax glue and instrumentation
    "common.jax_compile_cache", "common.jax_profile", "common.timing",
    "common.progress", "common.gpu_utils", "common.async_io", "common.sanity",
    # FFT kernels
    "common.fft_helpers",
    # the sharded-HDF5 transport (NOT the format readers above it)
    "file_io.slab_io", "file_io._slab_io_ffi", "file_io._slab_io_allgather",
    "file_io._slab_io_mpi_host", "file_io.paths",
})
_L3_PACKAGES = ("ffi",)

#: L2 by module.  Whole packages are in ``_L2_PACKAGES``.
_L2_MODULES = frozenset({
    "common.minimax",          # minimax quadrature: a Remez problem
    "common.cholesky_2d",      # blocked Cholesky of an arbitrary SPD matrix
    "common.rank_criterion",   # the pseudo-inverse truncation criterion
    "centroid.kmeans_isdf",    # density-weighted Lloyd loop under a metric
})
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


def module_name(path: pathlib.Path) -> str:
    rel = path.relative_to(SRC)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][:-3]
    return ".".join(parts)


def all_modules():
    """``{module name: source text}`` for every ``.py`` under ``src/``."""
    out = {}
    for root, dirs, files in os.walk(SRC):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".ipynb_checkpoints")]
        for fn in sorted(files):
            if fn.endswith(".py"):
                p = pathlib.Path(root) / fn
                out[module_name(p)] = p.read_text()
    return out


def _absolute_target(node: ast.ImportFrom, mod: str) -> str:
    """Resolve a possibly-relative ``from ... import`` to an absolute name."""
    if node.level == 0:
        return node.module or ""
    base = mod.split(".")[:-1]
    for _ in range(node.level - 1):
        base = base[:-1]
    if node.module:
        base = base + node.module.split(".")
    return ".".join(base)


def scan_plumbing_imports(source: str, mod: str = "m"):
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
            tgt = _absolute_target(node, mod)
            if tgt.startswith(_PLUMBING_MODULES):
                hits.append((tgt, node.lineno))
            elif tgt in ("jax", "jax.experimental"):
                for a in node.names:
                    if a.name in _PLUMBING_NAMES:
                        hits.append((tgt + "." + a.name, node.lineno))
    return sorted(hits, key=lambda h: h[1])


def scan_imports(source: str, mod: str):
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
            tgt = _absolute_target(node, mod)
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

    # --- not clean, with the reason ---------------------------------------
    # ``from jax.sharding import Mesh, NamedSharding, PartitionSpec`` at :90.
    # exciton_bands is a driver AND a library: it owns the interpolated-band
    # assembly its own CLI consumes.  Splitting it is the same operation that
    # freed ``kin_ion_io``, and it is a physics decision about where the
    # exciton-band assembly lives.  Numbered request.
    "bse.exciton_bands": 1,
    # ``jax.sharding`` + ``jax.experimental.shard_map`` at :18-19.  Same
    # shape, larger: htransform is the H-matrix interpolation LIBRARY with a
    # CLI bolted on, and its Galerkin solve is written in ``shard_map``.  Its
    # hand-rolled ``_build_mesh_xy`` is gone (2026-07-31; since 2026-08-01 it
    # hands back the mesh ``initialize_communicator_stack`` built) — so the
    # residue is the kernel half only.
    "bandstructure.htransform": 2,
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
    # resolve_fh_ortho_tol — one resolver each, refuse-on-garbage; the
    # kwarg/entry layer can override by passing values down.
    "bandstructure.htransform": {
        "LORRAX_GALERKIN_CHUNK_GIB",
        "LORRAX_EXTRA_RANK_PAD",
        "LORRAX_FH_ORTHO_TOL",
    },
    # resolve_reshard_route — explicit kwarg wins, env is the A/B path the
    # production exciton_bands driver cannot plumb an argument to (file
    # ownership); unknown tokens announce, never silently.
    "bandstructure.bse_setup": {
        "LORRAX_FACE_TO_BATCH_ROUTE",
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

_L3_UPWARD_EXCEPTIONS = {
    # ``SlabIOBackend`` — the enum that types the file-IO service's OWN
    # backend parameter — is defined in the GW driver's 2.3 kLoC deck parser,
    # so the transport imports it lazily at two sites with the comment
    # "avoid circular import at module load", and slab_io's own public
    # docstring tells users to ``from gw.gw_config import SlabIOBackend``.
    #
    # The audit called this a low-risk enum move.  It is not quite: the
    # obvious new home, ``file_io``, imports jax and h5py at package scope,
    # and gw_config is (nearly) jax-free by design.  The enum needs a home
    # that neither package owns.  Numbered request.
    ("file_io.slab_io", "gw.gw_config"): 2,
}

_L2_UPWARD_EXCEPTIONS = {
    # ``from psp.dft_operators import apply_H_k_from_G``, module scope, used
    # at :143.  A solver that applies a specific DFT Hamiltonian is not
    # physics-agnostic: the CG core is L2, this file is L1 wrapped around it.
    # Either split ``SternheimerOp``'s operator out, or move the file to
    # ``psp/``.  Physics decision.  Numbered request.
    ("solvers.sternheimer_solve", "psp.dft_operators"): 1,
    # ``from .orbit_syms import ...``, lazy, inside the orbit-aware branch.
    # k-means itself is pure clustering under a metric tensor — genuinely L2
    # — but the orbit-aware variant folds a real-space point group into the
    # assignment step, and THAT is physics.  Injecting the orbit map as a
    # parameter is the fix and it is a signature change.  Numbered request.
    ("centroid.kmeans_isdf", "centroid.orbit_syms"): 1,
}


def upward_edges(sources):
    """Every import that goes UP a level. ``[(from_layer, to_layer, mod, target, line, lazy)]``."""
    mods = set(sources)
    out = []
    for mod, src in sources.items():
        here = layer_of(mod)
        if here == "X":
            continue
        for target, line, lazy in scan_imports(src, mod):
            parts = target.split(".")
            resolved = None
            for i in range(len(parts), 0, -1):
                cand = ".".join(parts[:i])
                if cand in mods:
                    resolved = cand
                    break
            if resolved is None or resolved == mod:
                continue
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

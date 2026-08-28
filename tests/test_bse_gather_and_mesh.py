"""Three BSE service contracts, each with a control built to go red.

1. **ONE mesh factory in ``src/bse/``.**  Four byte-identical
   ``_create_mesh_xy(px, py)`` bodies used to build the ``('x','y')`` Mesh
   directly and return it WITHOUT warming the MPI cliques, while only
   ``bse_ring_comm.create_mesh_2d`` warmed.  Five drivers reached the
   un-warmed copies — ``exciton_bands``, ``bse_w_exact``, ``bse_feast``,
   ``bse_kpm``, ``bse_pseudopoles`` — so each of them dies at P>1 under
   ``JAX_CPU_COLLECTIVES_IMPLEMENTATION=mpi`` with
   "Communicator requested from a thread that is not the one MPI was
   initialized from" (32 refusals at P=16, gate 7881216).  A duplicated
   constructor is how a load-bearing side effect goes missing without anybody
   deleting a line, so the gate is structural AND behavioural.

2. **``exciton_bands._gather_host`` delegates to
   ``common.collectives.gather_to_host`` and keeps the branch.**  The wrong
   arm is SILENT in one direction: ``process_allgather(tiled=True)`` on a
   fully-addressable array concatenates each process's complete copy and
   multiplies the leading axis by P.  At the 64-rank run this file was written
   for that is a 64x-too-long array of plausible dtype and finite values.

3. **``--px``/``--py`` omitted means the run's mesh.**  They defaulted to
   ``1``, so every bse-family driver launched with no mesh flags asked the
   (correct, shared) factory for a 1x1 and ran a four-GPU node on one device
   while the startup report announced 2x2; at P>1 the same default builds
   over ``jax.devices()[:1]`` — process 0's device on every rank — and every
   rank >= 1 dies in ``collectives._require_addressable``.  The gate has to
   be behavioural and it has to run wide: at one device every arm of that bug
   agrees, and ``--px 2 --py 2`` (what every launch recipe in-tree passes) is
   the input that masks it.

WHY EACH INSTRUMENT CARRIES A CONTROL.  Both gates here are of the class that
is indistinguishable from a no-op when it passes: a structural scan that
matches nothing, and a spy that never fires.  ``wk_REL/README.md`` standing
lesson 1 — six checks in one session were void, four of them cheerfully green.
Every ``assert`` below has a sibling that runs the SAME instrument against
input it must reject.
"""
from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# Scan the tree that is actually IMPORTABLE, not the one next to this file.
# These tests are run against a frozen source snapshot on PYTHONPATH (the
# wk_REL srcpin model), and a scan hard-wired to ``__file__``'s sibling would
# audit the live worktree in both legs of a pre/post gate — i.e. return the
# same verdict either way, which is the void-instrument failure this file is
# built to avoid.  Falling back to the sibling keeps a plain in-tree pytest
# working.
def _bse_dir() -> pathlib.Path:
    import importlib

    try:
        return pathlib.Path(importlib.import_module("bse").__file__).parent
    except Exception:
        return pathlib.Path(__file__).resolve().parents[1] / "src" / "bse"


_BSE = _bse_dir()

# The ONE place in src/bse/ allowed to construct the ('x','y') Mesh.
_FACTORY_FILE = "bse_ring_comm.py"
_FACTORY_FUNCS = {"create_mesh_xy"}


# ---------------------------------------------------------------------------
# the instrument: find every Mesh(...) construction in a module's source
# ---------------------------------------------------------------------------
def _mesh_constructions(src_text: str) -> list[tuple[str, int]]:
    """(enclosing function name, lineno) for every ``Mesh(...)`` call.

    An un-warmed mesh factory is exactly "a ``Mesh(...)`` call that is not
    inside the one blessed constructor", so that is what this looks for.
    Walks the AST rather than grepping: a grep negative is not evidence of
    absence (README standing lesson 2), and the enclosing-function attribution
    is what makes the finding actionable.
    """
    tree = ast.parse(src_text)
    out: list[tuple[str, int]] = []

    class V(ast.NodeVisitor):
        def __init__(self):
            self.fn = "<module>"

        def visit_FunctionDef(self, node):
            prev, self.fn = self.fn, node.name
            self.generic_visit(node)
            self.fn = prev

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node):
            f = node.func
            name = getattr(f, "id", None) or getattr(f, "attr", None)
            if name == "Mesh":
                out.append((self.fn, node.lineno))
            self.generic_visit(node)

    V().visit(tree)
    return out


def test_instrument_finds_a_mesh_construction_when_there_is_one():
    """CONTROL for the AST scan itself.

    A structural check that has never matched anything is indistinguishable
    from one that scans nothing.  Feed it a module that DOES build a mesh and
    require it to be found, with the right enclosing function.
    """
    bait = (
        "from jax.sharding import Mesh\n"
        "def _create_mesh_xy(px, py):\n"
        "    return Mesh(devs.reshape(px, py), axis_names=('x','y'))\n"
    )
    found = _mesh_constructions(bait)
    assert found == [("_create_mesh_xy", 3)], found


def test_only_one_bse_module_constructs_the_xy_mesh():
    """The gate: no un-warmed mesh factory anywhere in ``src/bse/``."""
    offenders: list[str] = []
    for path in sorted(_BSE.glob("*.py")):
        for fn, lineno in _mesh_constructions(path.read_text()):
            if path.name == _FACTORY_FILE and fn in _FACTORY_FUNCS:
                continue
            offenders.append(f"{path.name}:{lineno} in {fn}()")
    assert not offenders, (
        "Mesh built outside bse_ring_comm.create_mesh_xy — that copy does not "
        "warm the MPI cliques and its driver dies at P>1 under impl=mpi:\n  "
        + "\n  ".join(offenders))


def test_the_blessed_factory_really_is_in_that_file():
    """CONTROL for the test above: if the factory were renamed or moved, the
    scan would find zero constructions everywhere and pass VACUOUSLY.  Pin
    that the allow-listed site exists and is the only allow-listed one."""
    found = _mesh_constructions((_BSE / _FACTORY_FILE).read_text())
    inside = [f for f, _ in found if f in _FACTORY_FUNCS]
    assert len(inside) == 1, (
        f"expected exactly one Mesh(...) inside {_FACTORY_FUNCS} of "
        f"{_FACTORY_FILE}; found {found}")


# ---------------------------------------------------------------------------
# behavioural: every BSE mesh entry point reaches prepare_mesh
# ---------------------------------------------------------------------------
_ENTRY_POINTS = [
    ("bse.bse_ring_comm", "create_mesh_xy"),
    ("bse.bse_ring_comm", "create_mesh_2d"),
    ("bse.bse_ring_comm", "create_mesh_xy_from_flags"),
    # This list is now the whole set, and that is the point.  Three
    # ``_create_mesh_xy`` one-line aliases had rows here — ``bse_pseudopoles``
    # (deleted 2026-08-27 with its module's move to
    # ``create_mesh_xy_from_flags``), then ``bse_feast`` and ``bse_w_exact``,
    # deleted in the same fix round once their last real callers moved to
    # ``collectives.single_device_mesh``.  A row aimed at a function no
    # production route can reach measures the gate keeping the code alive, not
    # the code: exactly the tautology section 5b of tests/test_layering.py
    # argues against.  The three surviving names are the reachable set —
    # ``create_mesh_xy_from_flags`` is what all six drivers' ``main()`` call,
    # and the structural gate below (no ``Mesh(`` outside bse_ring_comm) is
    # what stops a fourth entry point appearing without a row.
]


@pytest.mark.parametrize("modname,fname", _ENTRY_POINTS)
def test_every_bse_mesh_entry_point_goes_through_prepare_mesh(
        monkeypatch, modname, fname):
    """``prepare_mesh`` = resolve + ``warm_mesh_cliques`` + ``nccl_warmup``.

    Spying on it (rather than on ``warm_mesh_cliques``) is deliberate: the
    warm-up short-circuits at ``process_count() <= 1``, so a spy one level
    deeper would never fire in a single-process test and the gate would be
    void — the exact failure this file exists to avoid.

    ``create_mesh_xy_from_flags`` is called with ``(None, None)``, which is
    what every bse driver's ``main()`` passes when the user omits
    ``--px/--py`` — the arm that carries the production default since
    2026-08-27.  A ``(1, 1)`` call here would test the explicit arm twice and
    leave the default one unwarmed-and-unwatched.

    Every arm names its devices.  ``create_mesh_xy(1, 1)`` with an implicit
    list is now a refusal in any process with more than one device, and this
    file cannot assume one: a module-scope ``XLA_FLAGS=
    --xla_force_host_platform_device_count=4`` in a sibling collected into the
    same worker (tests/test_contract_bands.py, tests/test_sanity_gates_jax.py)
    widens the whole session with no mesh marker, and tests/harness.py records
    that class as measured — "a module-scope write never unwinds".  The
    explicit ``jax.local_devices()[:1]`` is also the arm worth spying on: it
    is the only one that takes ``create_mesh_xy``'s non-canonical branch, i.e.
    the branch that actually constructs a ``Mesh`` and could skip the warm-up.
    """
    import importlib

    mod = importlib.import_module(modname)
    ring = importlib.import_module("bse.bse_ring_comm")

    seen: list[object] = []
    real = ring.prepare_mesh

    def _spy(mesh=None, **kw):
        seen.append(mesh)
        return real(mesh, **kw)

    monkeypatch.setattr(ring, "prepare_mesh", _spy)
    fn = getattr(mod, fname)
    if fname == "create_mesh_2d":
        mesh = fn(jax.local_devices()[:1])
    elif fname == "create_mesh_xy_from_flags":
        mesh = fn(None, None)
    else:
        mesh = fn(1, 1, jax.local_devices()[:1])
    assert len(seen) == 1, (
        f"{modname}.{fname} built a mesh without going through prepare_mesh — "
        f"its cliques are never warmed and it dies at P>1 under impl=mpi")
    assert tuple(mesh.axis_names) == ("x", "y")


def test_the_prepare_mesh_spy_can_miss(monkeypatch):
    """CONTROL for the spy above.  A hand-rolled factory — the code that was
    there before this change — must leave the spy at zero.  Without this the
    parametrized test would also pass against a spy that fires on import."""
    import importlib

    from jax.sharding import Mesh

    ring = importlib.import_module("bse.bse_ring_comm")
    seen: list[object] = []
    monkeypatch.setattr(ring, "prepare_mesh",
                        lambda mesh=None, **kw: (seen.append(mesh), mesh)[1])

    def _old_unwarmed_create_mesh_xy(px, py):
        devices = jax.devices()
        return Mesh(np.array(devices[: px * py]).reshape(px, py),
                    axis_names=("x", "y"))

    _old_unwarmed_create_mesh_xy(1, 1)
    assert seen == [], (
        "the spy fired for a factory that never calls prepare_mesh — it is "
        "not measuring delegation")


# ---------------------------------------------------------------------------
# exciton_bands._gather_host -> common.collectives.gather_to_host
# ---------------------------------------------------------------------------
def _gather_host():
    eb = pytest.importorskip("bse.exciton_bands")
    return eb._gather_host


def test_gather_host_delegates_to_the_service(monkeypatch):
    """The unification: the driver's wrapper must be the service, not a
    second copy that can drift from it."""
    from common import collectives as C

    calls: list[object] = []
    monkeypatch.setattr(C, "gather_to_host",
                        lambda x: (calls.append(x), np.zeros((2, 2)))[1])
    out = _gather_host()(jnp.arange(6.0).reshape(2, 3))
    assert len(calls) == 1, (
        "bse.exciton_bands._gather_host did not call "
        "common.collectives.gather_to_host — it is a local copy again")
    assert out.shape == (2, 2)


def test_gather_host_keeps_the_device_get_arm_for_addressable_arrays(
        monkeypatch):
    """The arm whose wrong side is SILENT.  A fully-addressable array must not
    reach ``process_allgather``."""
    from jax.experimental import multihost_utils as mh

    def _boom(*a, **k):
        raise AssertionError(
            "process_allgather on a fully-addressable array — this is the "
            "leading-axis x P bug the branch exists to prevent")

    monkeypatch.setattr(mh, "process_allgather", _boom)
    x = jnp.arange(12.0).reshape(4, 3)
    out = _gather_host()(x)
    assert out.shape == (4, 3)
    assert np.allclose(out, np.arange(12.0).reshape(4, 3))


def test_gather_host_keeps_the_tiled_allgather_arm_for_remote_shards(
        monkeypatch):
    """NEGATIVE CONTROL for the arm above: flip ``is_fully_addressable`` and
    the OTHER path must be taken, with ``tiled=True``.  Without this, the test
    above passes on an implementation that never allgathers at all."""
    from jax.experimental import multihost_utils as mh

    seen: list[bool] = []

    def _fake(x, tiled=False):
        seen.append(tiled)
        return np.zeros((2, 2))

    monkeypatch.setattr(mh, "process_allgather", _fake)

    class _Remote:
        is_fully_addressable = False

    out = _gather_host()(_Remote())
    assert seen == [True], "the remote arm must use tiled=True"
    assert out.shape == (2, 2)


# ---------------------------------------------------------------------------
# the OTHER four wrappers: same service, and the tiled=False trio is gone
# ---------------------------------------------------------------------------
_WRAPPERS = [
    ("bse.exciton_bands", "_gather_host"),
    ("bse.vq_interp", "_to_host"),
    ("bse.bse_davidson_helpers", "_gather_to_host"),
    ("bse.davidson_absorption", "_gather_to_host"),
    ("solvers.davidson", "_to_host"),
]


@pytest.mark.parametrize("modname,fname", _WRAPPERS)
def test_every_hand_rolled_gather_wrapper_delegates(monkeypatch, modname, fname):
    """Five wrappers hand-rolled this; three of them used
    ``process_allgather(tiled=False)`` and branched on ``process_count()``
    instead of ``is_fully_addressable``.

    ``tiled=False`` RAISES on a process-spanning array — and their callers feed
    them exactly that (``bse_davidson_helpers.init_bse_subspace`` on the
    loader's k-sharded ``eps_c``/``eps_v``; ``davidson_absorption`` on
    ``eigvecs``).  Three of the five also cited
    ``common.collectives.gather_to_host`` as the pattern they mirrored, but
    that one uses ``tiled=True``, so the citation was wrong too.  One
    implementation, one branch, one explanation.
    """
    import importlib

    mod = pytest.importorskip(modname)
    from common import collectives as C

    calls = []
    monkeypatch.setattr(C, "gather_to_host",
                        lambda x: (calls.append(x), np.zeros((2, 2)))[1])
    out = getattr(mod, fname)(jnp.arange(6.0).reshape(2, 3))
    assert len(calls) == 1, (
        f"{modname}.{fname} did not reach common.collectives.gather_to_host")
    assert out.shape == (2, 2)
    del importlib


def test_no_tiled_false_allgather_survives_in_bse_or_solvers():
    """Structural sweep with its own control.

    Counts ``process_allgather(..., tiled=False)`` CALLS (AST, not grep — a
    grep negative is not evidence of absence) under bse/ and solvers/.  One is
    expected and legitimate: ``vq_interp``'s 2-element (sum, count) all-reduce,
    where every rank holds a DIFFERENT value and stacking is the point.  Any
    other is a wrapper that regressed.
    """
    import pathlib as _pl

    def _tiled_false_calls(text):
        hits = []
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if name != "process_allgather":
                continue
            for kw in node.keywords:
                if kw.arg == "tiled" and isinstance(kw.value, ast.Constant) \
                        and kw.value.value is False:
                    hits.append(node.lineno)
        return hits

    # CONTROL first: the detector must see one when there is one.
    bait = "process_allgather(x, tiled=False)\n"
    assert _tiled_false_calls(bait) == [1], "the AST detector matches nothing"

    solvers = _BSE.parent / "solvers"
    found = []
    for d in (_BSE, solvers):
        for path in sorted(d.glob("*.py")):
            for ln in _tiled_false_calls(path.read_text()):
                found.append((path.name, ln))
    unexpected = [f"{n}:{ln}" for n, ln in found if n != "vq_interp.py"]
    assert not unexpected, (
        f"tiled=False allgather outside the one legitimate site: {unexpected}. "
        f"tiled=False RAISES on a process-spanning array; a gather of ONE "
        f"logical array must go through common.collectives.gather_to_host.")
    assert len(found) <= 1, (
        f"more tiled=False sites in vq_interp than the single (sum, count) "
        f"all-reduce: {found}")
    del _pl


def test_a_replicated_but_not_addressable_array_issues_no_collective(
        monkeypatch):
    """THE MIDDLE ARM.  ``is_fully_replicated`` != ``is_fully_addressable``.

    A ``P()``-sharded array at P>1 has every device in its sharding and only
    one addressable, so the addressable predicate says False while the process
    already holds the whole thing.  Measured in this tree: ``exciton_bands``
    logs ``spec=P(), fully_addressable=False`` at P=64 (job 7882507).  Sending
    that through ``process_allgather`` is a collective for nothing — and under
    impl=mpi it refused outright (jobs 7882523 / 7882531 / 7882555).
    """
    from jax.experimental import multihost_utils as mh
    from common import collectives as C

    def _boom(*a, **k):
        raise AssertionError(
            "process_allgather on a FULLY REPLICATED array — this process "
            "already holds all of it; the middle arm is missing")

    monkeypatch.setattr(mh, "process_allgather", _boom)

    payload = np.arange(6.0).reshape(2, 3)

    class _Replicated:
        is_fully_addressable = False
        is_fully_replicated = True

        def addressable_data(self, i):
            assert i == 0
            return payload

    out = C.gather_to_host(_Replicated())
    assert out.shape == (2, 3)
    assert np.array_equal(out, payload)


def test_the_middle_arm_does_not_swallow_a_genuinely_sharded_array(
        monkeypatch):
    """CONTROL for the arm above: not-replicated must still allgather, or the
    fix would silently return one process's shard as if it were the whole
    array — a wrong answer with the right shape on one rank."""
    from jax.experimental import multihost_utils as mh
    from common import collectives as C

    seen: list[bool] = []

    def _fake(x, tiled=False):
        seen.append(tiled)
        return np.zeros((4, 3))

    monkeypatch.setattr(mh, "process_allgather", _fake)

    class _Sharded:
        is_fully_addressable = False
        is_fully_replicated = False

        def addressable_data(self, i):
            raise AssertionError("must not read a local shard for a sharded "
                                 "array")

    out = C.gather_to_host(_Sharded())
    assert seen == [True]
    assert out.shape == (4, 3)


@pytest.mark.parametrize("P", [16, 64])
def test_the_wrong_arm_really_multiplies_the_leading_axis(monkeypatch, P):
    """SHOW THE COMPARATOR GOING RED.

    The claim "a wrong branch silently multiplies a leading axis by P" is
    load-bearing for the whole unification, so measure it rather than assert
    it.  Emulate ``process_allgather(tiled=True)`` on a REPLICATED array —
    which is what it does: concatenate every process's complete copy along
    axis 0 — and require the result to be P x too long, and the guarded path
    to be unaffected.
    """
    from jax.experimental import multihost_utils as mh

    x = jnp.arange(4 * 3 * 1.0).reshape(4, 3)

    def _tiled_on_replicated(a, tiled=False):
        assert tiled is True
        return np.concatenate([np.asarray(a)] * P, axis=0)

    monkeypatch.setattr(mh, "process_allgather", _tiled_on_replicated)

    # (a) the unguarded call — what the four hand-rolled tiled=True wrappers
    #     would do if they dropped the is_fully_addressable branch.
    bad = _tiled_on_replicated(x, tiled=True)
    assert bad.shape == (4 * P, 3), (
        "the emulator did not reproduce the failure it is supposed to detect")

    # (b) the guarded call — same monkeypatch in place, correct shape.
    good = _gather_host()(x)
    assert good.shape == (4, 3), (
        f"_gather_host let a fully-addressable array through the tiled "
        f"allgather: got {good.shape}, expected (4, 3)")
    assert np.allclose(good, np.arange(12.0).reshape(4, 3))


# ---------------------------------------------------------------------------
# 3. behavioural: omitted --px/--py resolve to the run's mesh, and a shape
#    that does not consume the job refuses
# ---------------------------------------------------------------------------
#
# This cell must run in a wide process or it proves nothing.  On one device
# the canonical mesh is 1x1, so the pre-fix default, the post-fix default and
# the refusal all agree.  The suite pins one GPU per worker
# (``conftest.pytest_configure``), so the wide process is made here: a child
# with ``JAX_PLATFORMS=cpu
# XLA_FLAGS=--xla_force_host_platform_device_count=4``.
#
# Emulated devices are admissible for this question and only this one.  The
# four-GPU rule forbids substituting host emulation for a parallel-physics
# claim; nothing here is a physics claim.  What is asserted is which Mesh
# object a resolver returns and which shapes it refuses — device-kind
# independent, and false on the pre-fix tree in exactly this process shape
# (measured 2026-08-27: startup mesh (2, 2), ``create_mesh_xy(1, 1)`` ->
# (1, 1) over device [0], ``is RUNTIME.mesh`` False).
#
# The child does not call ``initialize_communicator_stack``: this is a
# question about ``bse_ring_comm`` and ``collectives``, and going through the
# runtime entry point would drag in the required-FFI gate and make an absent
# ``.so`` look like a mesh defect.

_WIDE_CHILD = r"""
import jax
from bse.bse_ring_comm import (create_mesh_xy, create_mesh_xy_from_flags,
                               make_bse_shardings)
from common.collectives import resolve_mesh

print("DEVICES", len(jax.devices()))
canonical = resolve_mesh(axis_names=("x", "y"))
m = create_mesh_xy_from_flags(None, None)
print("OMITTED_SHAPE", tuple(int(n) for n in m.devices.shape))
print("OMITTED_IS_CANONICAL", m is canonical)
print("SHARDINGS_EMBED_CANONICAL",
      all(s.mesh is canonical for s in vars(make_bse_shardings(m)).values()))
try:
    create_mesh_xy_from_flags(1, 1)
    print("EXPLICIT_ONE_BY_ONE ACCEPTED")
except ValueError as exc:
    print("EXPLICIT_ONE_BY_ONE REFUSED", str(len(jax.devices())) in str(exc))
try:
    create_mesh_xy_from_flags(2, None)
    print("HALF_REQUEST ACCEPTED")
except ValueError:
    print("HALF_REQUEST REFUSED")
# CONTROL: the identity assertions above must be capable of returning False.
# This is the pre-fix behaviour spelled out — the prefix slice of the device
# list — and it must not be the canonical mesh.
twin = create_mesh_xy(1, 1, jax.devices()[:1])
print("CONTROL_TWIN_SHAPE", tuple(int(n) for n in twin.devices.shape))
print("CONTROL_TWIN_IS_CANONICAL", twin is canonical)
"""


def _run_wide_child(n_devices: int = 4):
    """Run ``_WIDE_CHILD`` in a fresh process that sees ``n_devices``."""
    import os
    import subprocess
    import sys

    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={n_devices}"
    # The child must import what this process imports; under ``lx test`` the
    # source root arrives on sys.path, not necessarily in PYTHONPATH.
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    proc = subprocess.run([sys.executable, "-c", _WIDE_CHILD],
                          capture_output=True, text=True, env=env, timeout=900)
    assert proc.returncode == 0, (
        f"the widened child died (rc={proc.returncode}).\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr[-3000:]}")
    facts = {}
    for line in proc.stdout.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0].isupper():
            facts[parts[0]] = parts[1].strip()
    return facts, proc.stdout


def test_omitted_px_py_resolve_to_the_runs_own_mesh():
    """The default path lands on the canonical mesh, by identity."""
    facts, out = _run_wide_child(4)
    assert facts.get("DEVICES") == "4", (
        f"the child did not widen — every arm of this defect agrees at one "
        f"device, so the cell would pass vacuously.  Child said:\n{out}")
    assert facts.get("OMITTED_SHAPE") == "(2, 2)", (
        f"--px/--py omitted gave {facts.get('OMITTED_SHAPE')} on a 4-device "
        f"job; the run's mesh is 2x2.  Child said:\n{out}")
    # ``Mesh.__eq__`` is true for equal-but-distinct twins, and a twin is
    # exactly the defect (a second set of communicators, a second copy of
    # every shape-keyed jit cache), so the assertion is identity.
    assert facts.get("OMITTED_IS_CANONICAL") == "True", (
        f"the resolved mesh is not THE canonical object.  Child said:\n{out}")
    assert facts.get("SHARDINGS_EMBED_CANONICAL") == "True", (
        f"make_bse_shardings embedded a mesh that is not the run's.  Child "
        f"said:\n{out}")


def test_a_shape_that_does_not_consume_the_job_refuses():
    """``--px 1 --py 1`` on a 4-device job is the case that used to succeed."""
    facts, out = _run_wide_child(4)
    assert facts.get("DEVICES") == "4", f"the child did not widen:\n{out}"
    assert facts.get("EXPLICIT_ONE_BY_ONE", "").startswith("REFUSED"), (
        f"an explicit 1x1 was accepted on a 4-device job — the mesh covers "
        f"one device and the other three enter the same jit from outside it. "
        f"Child said:\n{out}")
    assert facts.get("EXPLICIT_ONE_BY_ONE") == "REFUSED True", (
        f"the refusal does not name the job's device count, so it cannot "
        f"tell the user what to ask for.  Child said:\n{out}")
    assert facts.get("HALF_REQUEST") == "REFUSED", (
        f"--px without --py was accepted.  Child said:\n{out}")


def test_the_mesh_identity_assertion_can_be_false():
    """CONTROL for the two cells above.

    ``x is canonical`` would also hold if some layer collapsed every mesh to
    one object, in which case the assertions above would be tautologies.
    Build the mesh the pre-fix code built — ``create_mesh_xy`` over the
    prefix of the device list — and require it to be a different object of a
    different shape.
    """
    facts, out = _run_wide_child(4)
    assert facts.get("CONTROL_TWIN_SHAPE") == "(1, 1)", out
    assert facts.get("CONTROL_TWIN_IS_CANONICAL") == "False", (
        f"a 1x1 mesh over device 0 compared IDENTICAL to the run's 2x2 mesh; "
        f"the identity instrument used above cannot fail, so it is not "
        f"evidence.  Child said:\n{out}")

"""Two BSE service contracts, each with a control built to go red.

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
    ("bse.bse_w_exact", "_create_mesh_xy"),
    ("bse.bse_feast", "_create_mesh_xy"),
    ("bse.bse_pseudopoles", "_create_mesh_xy"),
]


@pytest.mark.parametrize("modname,fname", _ENTRY_POINTS)
def test_every_bse_mesh_entry_point_goes_through_prepare_mesh(
        monkeypatch, modname, fname):
    """``prepare_mesh`` = resolve + ``warm_mesh_cliques`` + ``nccl_warmup``.

    Spying on it (rather than on ``warm_mesh_cliques``) is deliberate: the
    warm-up short-circuits at ``process_count() <= 1``, so a spy one level
    deeper would never fire in a single-process test and the gate would be
    void — the exact failure this file exists to avoid.
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
    mesh = fn(1, 1) if fname != "create_mesh_2d" else fn([jax.devices()[0]])
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
    ``file_io._slab_io_allgather._to_host`` as the pattern they mirrored, but
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

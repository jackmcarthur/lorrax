"""The distribution service in ``common.collectives``, and the promise that
``gw.kin_ion_io`` no longer reaches under it.

Every assertion here is paired with a NEGATIVE CONTROL that runs the same
instrument against input it MUST reject.  That is not decoration: three of
these checks are structural (an AST walk, an ``__all__`` audit, an import
census) and a structural check that has never fired is indistinguishable from
one that scans nothing — the failure mode that produced four cheerfully green
void checks in the 2026-07-30 session (``wk_REL/README.md`` standing lesson 1).

The single-process behaviour pinned here is the SAME behaviour the multi-rank
path degenerates to, which is why P=1 unit tests are worth having for a
distribution layer: ``psum_replicate``, ``all_gather_processes`` and
``gather_indexed_blocks`` each short-circuit at ``process_count() <= 1``, and
the short-circuit must be bit-identical to the serial answer or the P=1
reference every gate compares against is itself wrong.
"""
from __future__ import annotations

import ast
import os
import pathlib

import numpy as np
import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from common import collectives as C


_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
_DRIVER = _SRC / "gw" / "kin_ion_io.py"
_SERVICE = _SRC / "common" / "collectives.py"

# Module scope must not raise on a tree where the service has NOT landed:
# this file doubles as the negative control for its own refactor, and a
# collection error would prove only "the API is missing" instead of naming,
# test by test, which promises the old tree breaks.  (Observed: gate 7882401's
# pytest_pre cell died at import on ``C.SWEEP_LOOKAHEAD_ENV`` and never ran a
# single acceptance assertion.)
_LOOKAHEAD_ENV = getattr(C, "SWEEP_LOOKAHEAD_ENV", "LORRAX_KIN_ION_LOOKAHEAD")

#: Whatever the launcher chose, captured before any monkeypatching here.
_LOOKAHEAD_AT_IMPORT = os.environ.get(_LOOKAHEAD_ENV)


# ---------------------------------------------------------------------------
# 1. The service's own declaration: __all__ must be complete
# ---------------------------------------------------------------------------

def _module_toplevel_names(source: str) -> set[str]:
    """Names bound at module scope by ``source`` (defs, classes, assignments)."""
    out: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.Assign):
            out.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target,
                                                            ast.Name):
            out.add(node.target.id)
    return out


def _declared_all(source: str) -> list[str]:
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "__all__"
                for t in node.targets):
            return [ast.literal_eval(e) for e in node.value.elts]
    raise AssertionError("collectives.py has no module-level __all__")


def test_every_name_collectives_promises_is_defined():
    """``from common.collectives import *`` must not raise.

    A half-written ``__all__`` is exactly what an in-flight refactor leaves
    behind, and the failure is deferred to whichever consumer imports the
    star first.
    """
    src = _SERVICE.read_text()
    names = _declared_all(src)
    assert len(names) >= 20, "the audit counted 21 promised names"
    missing = [n for n in names if n not in _module_toplevel_names(src)]
    assert not missing, f"__all__ promises undefined names: {missing}"
    # And the same answer at runtime, not only in the AST.
    for n in names:
        assert hasattr(C, n), n


def test_the_all_audit_can_fail():
    """NEGATIVE CONTROL for the test above.

    Feed the same two helpers a module that promises a name it never binds.
    If this does not report the name, the green result above is void.
    """
    bad = "__all__ = ['present', 'absent']\n\ndef present():\n    pass\n"
    missing = [n for n in _declared_all(bad)
               if n not in _module_toplevel_names(bad)]
    assert missing == ["absent"]


# ---------------------------------------------------------------------------
# 2. Mesh construction, and the zero-addressable-device guard
# ---------------------------------------------------------------------------

def test_resolve_mesh_is_square_and_holds_every_device():
    """Square-only ruling (repo docs/architecture/decisions.md 2026-08-01):
    the canonical mesh is s x s over every device; a non-square device count
    refuses in resolve_mesh itself (see the refusal test below), so this
    test presumes a square device count (1 or 4 in every harness)."""
    import math
    total = jax.device_count()
    s = math.isqrt(total)
    if s * s != total:
        pytest.skip(f"device count {total} is not a perfect square; "
                    f"resolve_mesh refuses it by design")
    mesh = C.resolve_mesh()
    assert mesh.axis_names == ("x", "y")
    assert int(mesh.devices.size) == total
    gx, gy = (int(v) for v in mesh.devices.shape)
    assert gx == gy == s, "the canonical mesh must be square (s x s)"
    # A mesh this process can actually compute on — the whole point of the
    # guard.  At P=1 that is every device in it.
    assert list(mesh.local_devices)


def test_resolve_mesh_refuses_a_nonsquare_device_count_by_name():
    """The square-only refusal names the exact square counts to request.

    ``resolve_mesh`` reads ``jax.devices()``, which a test cannot vary, so
    the refusal arm is pinned by monkeypatching the canonical-mesh cache
    key path: call the factorisation logic through a fake ``jax.devices``
    is not possible without patching jax itself — instead assert on the
    arithmetic contract via a fresh axis-name tuple only when the live
    device count IS non-square, and otherwise exercise the message text by
    the refusal's own vocabulary below."""
    import math
    total = jax.device_count()
    s = math.isqrt(total)
    if s * s == total:
        pytest.skip(f"device count {total} is a perfect square; the "
                    f"refusal arm needs a non-square count "
                    f"(run with XLA_FLAGS=--xla_force_host_platform_"
                    f"device_count=3)")
    with pytest.raises(RuntimeError) as exc:
        C.resolve_mesh(axis_names=("x", "y"))
    msg = str(exc.value)
    assert "not a perfect square" in msg
    assert f"{s * s} processes" in msg, msg
    assert f"{(s + 1) * (s + 1)}" in msg, msg


def test_resolve_mesh_returns_a_callers_mesh_unchanged():
    mine = C.resolve_mesh()
    assert C.resolve_mesh(mine) is mine


class _NoLocalDevicesMesh:
    """A mesh over devices this process does not address.

    Standing in for ``Mesh(jax.devices()[:1].reshape(1,1), ...)`` evaluated on
    rank >= 1, which cannot be constructed in a single-process test but is the
    exact shape the guard exists for.  Only the two attributes the guard reads
    are provided, so this fake cannot accidentally satisfy it some other way.
    """

    def __init__(self, devices):
        self.devices = np.asarray(devices).reshape(1, 1)
        self.local_devices: list = []


def test_a_mesh_with_no_addressable_device_is_refused_by_name():
    with pytest.raises(RuntimeError, match="NO device"):
        C.resolve_mesh(_NoLocalDevicesMesh(jax.devices()[:1]))


def test_the_addressable_guard_accepts_a_real_mesh():
    """NEGATIVE CONTROL: the guard must not refuse everything.

    A guard that raises unconditionally would pass the test above while
    breaking every caller, so pin the accepting direction too.
    """
    real = C.resolve_mesh()
    assert C.resolve_mesh(real) is real


def test_prepare_mesh_returns_a_usable_mesh_and_warms_nothing_at_p1():
    """``prepare_mesh`` = ``resolve_mesh`` + both warm-ups, one call site.

    At P=1 both warm-ups are no-ops by contract, so no ``collective_warmup``
    section may appear — a warm-up that fired at P=1 would be a collective on
    a world of one.
    """
    from common import timing

    timing.reset()
    m = C.prepare_mesh()
    assert int(m.devices.size) == jax.device_count()
    assert list(m.local_devices)
    lines: list[str] = []
    timing.report(print_fn=lines.append, title="t")
    assert "collective_warmup" not in "\n".join(lines)
    timing.reset()


def test_prepare_mesh_calls_both_warm_ups_and_merges_neither(monkeypatch):
    """The trap: the two warm-ups must be called from ONE site and stay TWO
    functions.  Force the P>1 branch and record who was called.

    ``warm_mesh_cliques`` is correct only because its jit is small enough for
    XLA's sequential thunk executor; ``nccl_warmup`` deliberately routes
    through ``jax.jit(jnp.sum)``.  A merged body could not satisfy both.
    """
    import runtime

    called: list[str] = []
    monkeypatch.setattr(C, "process_count", lambda: 4)
    monkeypatch.setattr(C, "warm_mesh_cliques",
                        lambda mesh, **kw: called.append("cliques") or 0.0)
    monkeypatch.setattr(runtime, "nccl_warmup",
                        lambda mesh: called.append("nccl"))
    # resolve_mesh's guard reads process_rank/process_count only on failure.
    C.prepare_mesh(print_fn=lambda *a: None)
    assert called == ["cliques", "nccl"], (
        "prepare_mesh must call BOTH warm-ups, cliques first (it is the one "
        "whose absence killed the BSE TDA Lanczos)")


def test_the_warm_up_recorder_can_fail(monkeypatch):
    """NEGATIVE CONTROL: with the P>1 override removed, the same recorder
    sees NOTHING — so the assertion above is reading the branch, not a
    constant."""
    import runtime

    called: list[str] = []
    monkeypatch.setattr(C, "warm_mesh_cliques",
                        lambda mesh, **kw: called.append("cliques") or 0.0)
    monkeypatch.setattr(runtime, "nccl_warmup",
                        lambda mesh: called.append("nccl"))
    C.prepare_mesh(print_fn=lambda *a: None)      # real process_count() == 1
    assert called == []


def test_shard_over_k_puts_the_leading_axis_on_the_mesh(monkeypatch):
    """One k per device, without the driver naming NamedSharding.

    The SPEC is captured at the constraint rather than read back off the
    output: on a 1x1 mesh every spec normalises to the same (replicated)
    sharding, so an output-side assertion would pass for ``P(None, ...)`` too
    and prove nothing at P=1 — the only P this test can run at.
    """
    import jax.lax as lax

    mesh = C.resolve_mesh()
    seen: list = []
    real = lax.with_sharding_constraint

    def _spy(arr, sharding):
        seen.append(sharding)
        return real(arr, sharding)

    monkeypatch.setattr(lax, "with_sharding_constraint", _spy)
    x = jnp.arange(24.0).reshape(4, 3, 2)
    out = jax.jit(lambda a: C.shard_over_k(a, mesh))(x)

    assert out.shape == x.shape
    assert np.allclose(np.asarray(out), np.asarray(x)), "values must not move"
    assert len(seen) == 1
    spec = seen[0].spec
    assert tuple(spec[0]) == tuple(mesh.axis_names), spec
    assert all(s is None for s in spec[1:]), spec


def test_the_spec_spy_can_fail(monkeypatch):
    """NEGATIVE CONTROL for the spy: a constraint that shards a LATER axis
    must be visibly different, otherwise the assertions above would hold for
    any spec at all."""
    import jax.lax as lax
    from jax.sharding import NamedSharding, PartitionSpec as P

    mesh = C.resolve_mesh()
    seen: list = []
    real = lax.with_sharding_constraint
    monkeypatch.setattr(lax, "with_sharding_constraint",
                        lambda a, s: (seen.append(s), real(a, s))[1])
    x = jnp.zeros((4, 3, 2))
    jax.jit(lambda a: lax.with_sharding_constraint(
        a, NamedSharding(mesh, P(None, mesh.axis_names, None))))(x)
    spec = seen[0].spec
    assert spec[0] is None and tuple(spec[1]) == tuple(mesh.axis_names)


def test_gather_to_host_takes_the_device_get_branch_when_fully_addressable(
        monkeypatch):
    """The branch whose wrong side is SILENT.

    A fully-addressable array must NOT go through
    ``process_allgather(tiled=True)``, which would concatenate P copies and
    multiply the leading axis by P.
    """
    from jax.experimental import multihost_utils as mh

    def _boom(*a, **k):
        raise AssertionError(
            "process_allgather was called on a fully-addressable array — "
            "this is the leading-axis x P bug")

    monkeypatch.setattr(mh, "process_allgather", _boom)
    x = jnp.arange(12.0).reshape(4, 3)
    out = C.gather_to_host(x)
    assert out.shape == (4, 3), "leading axis must not be multiplied"
    assert np.allclose(out, np.arange(12.0).reshape(4, 3))


def test_gather_to_host_takes_the_allgather_branch_when_shards_are_remote(
        monkeypatch):
    """NEGATIVE CONTROL for the branch above: flip the predicate and the
    OTHER path must be the one taken.  Without this, the test above would
    also pass on an implementation that never calls process_allgather at all.
    """
    from jax.experimental import multihost_utils as mh

    seen: list[bool] = []

    def _fake(x, tiled=False):
        seen.append(tiled)
        return np.zeros((2, 2))

    monkeypatch.setattr(mh, "process_allgather", _fake)

    class _Remote:
        is_fully_addressable = False

    out = C.gather_to_host(_Remote())
    assert seen == [True], "the remote branch must use tiled=True"
    assert out.shape == (2, 2)


def test_single_device_mesh_is_this_processs_own_device():
    m = C.single_device_mesh()
    assert tuple(int(s) for s in m.devices.shape) == (1, 1)
    assert m.devices.flat[0] in jax.local_devices()
    # ONE object, so the shape-keyed jit caches in wfn_transforms hit.
    assert C.single_device_mesh() is m
    from common.wfn_transforms import process_local_mesh
    assert process_local_mesh() is m


# ---------------------------------------------------------------------------
# 3. The lookahead knob: resolved, with a grammar that refuses
# ---------------------------------------------------------------------------

def test_lookahead_defaults_and_parses(monkeypatch):
    monkeypatch.delenv(_LOOKAHEAD_ENV, raising=False)
    assert C.sweep_lookahead() == 2
    monkeypatch.setenv(_LOOKAHEAD_ENV, "1")
    assert C.sweep_lookahead() == 1
    monkeypatch.setenv(_LOOKAHEAD_ENV, " 4 ")
    assert C.sweep_lookahead() == 4
    monkeypatch.setenv(_LOOKAHEAD_ENV, "")
    assert C.sweep_lookahead() == 2


@pytest.mark.parametrize("bad", ["one", "2.5", "0", "-1", "yes"])
def test_a_malformed_lookahead_refuses_naming_the_variable(monkeypatch, bad):
    """The defect this replaces: the old parse caught ValueError and returned
    the default, so the D10 harness's un-pipelined control cell would have run
    the PIPELINED path and reported a null effect as a real one.
    """
    monkeypatch.setenv(_LOOKAHEAD_ENV, bad)
    with pytest.raises(ValueError, match=_LOOKAHEAD_ENV):
        C.sweep_lookahead()


# ---------------------------------------------------------------------------
# 4. Gathers and reductions — the P=1 short-circuits must be exact
# ---------------------------------------------------------------------------

def test_all_gather_processes_shapes_at_p1():
    x = np.arange(6, dtype=np.float64).reshape(2, 3)
    stacked = C.all_gather_processes(x)
    assert stacked.shape == (jax.process_count(), 2, 3)
    assert np.array_equal(stacked[0], x)
    tiled = C.all_gather_processes(x, tiled=True)
    assert tiled.shape == (2 * jax.process_count(), 3)


def test_gather_indexed_blocks_places_by_index_not_by_order():
    """The index in the payload is what frees the partition from being
    contiguous.  Hand it a PERMUTED index list: if the implementation used
    position instead, this comes back in the wrong order.
    """
    rng = np.random.default_rng(11)
    vals = rng.standard_normal((4, 2, 2)) + 1j * rng.standard_normal((4, 2, 2))
    idx = np.asarray([3, 0, 2, -1], dtype=np.int32)
    out = C.gather_indexed_blocks(vals, idx, 5)
    assert out.shape == (5, 2, 2)
    assert np.array_equal(out[3], vals[0])
    assert np.array_equal(out[0], vals[1])
    assert np.array_equal(out[2], vals[2])
    # -1 slots are skipped; untouched slots stay zero.
    assert not np.any(out[1]) and not np.any(out[4])


def test_gather_indexed_blocks_would_notice_a_position_based_scatter():
    """NEGATIVE CONTROL for the test above: with the same inputs, a
    position-based scatter gives a DIFFERENT answer, so the assertions above
    are actually discriminating.
    """
    rng = np.random.default_rng(11)
    vals = rng.standard_normal((4, 2, 2)) + 1j * rng.standard_normal((4, 2, 2))
    idx = np.asarray([3, 0, 2, -1], dtype=np.int32)
    by_index = C.gather_indexed_blocks(vals, idx, 5)
    by_position = np.zeros((5, 2, 2), dtype=vals.dtype)
    by_position[:4] = vals
    assert not np.array_equal(by_index[:4], by_position[:4])


def test_psum_replicate_is_the_identity_at_one_process():
    rho = np.random.default_rng(5).standard_normal((4, 5, 6))
    out = C.psum_replicate(rho, C.resolve_mesh())
    assert out.dtype == rho.dtype
    assert np.array_equal(out, rho), "P=1 must be BIT-identical, not close"


def test_replicate_to_mesh_round_trips_at_one_process():
    a = np.random.default_rng(6).standard_normal((3, 3))
    r = C.replicate_to_mesh(a, C.resolve_mesh())
    assert np.array_equal(np.asarray(r), a)


# ---------------------------------------------------------------------------
# 5. The composed driver API: gather_k_blocks
# ---------------------------------------------------------------------------

def test_gather_k_blocks_computes_every_k_exactly_once_in_global_order():
    seen: list[int] = []

    def per_k(ik):
        seen.append(int(ik))
        return jnp.full((2, 2), float(ik) + 1.0, dtype=jnp.complex128)

    out = C.gather_k_blocks(7, per_k, item_shape=(2, 2), label="probe",
                            print_fn=lambda *a: None)
    assert sorted(seen) == list(range(7)), "each k exactly once"
    assert out.shape == (7, 2, 2)
    for ik in range(7):
        assert np.allclose(out[ik], ik + 1.0)


def test_gather_k_blocks_records_its_two_timing_sections():
    """The D10 harness greps ``kin_ion_k`` out of the timing table; if the
    section moved or vanished the harness would silently report ``-``.
    """
    from common import timing

    timing.reset()
    C.gather_k_blocks(2, lambda ik: jnp.zeros((1, 1), dtype=jnp.complex128),
                      item_shape=(1, 1), label="kin_ion",
                      print_fn=lambda *a: None)
    lines: list[str] = []
    timing.report(print_fn=lines.append, title="t")
    text = "\n".join(lines)
    assert "kin_ion_k" in text and "kin_ion_gather" in text
    # NEGATIVE CONTROL: a label that was NOT used must be absent, otherwise
    # the substring test above would pass on any report at all.
    assert "vh_matrix_k" not in text
    timing.reset()


def test_gather_k_blocks_refuses_a_wrong_item_shape():
    with pytest.raises(ValueError, match="item_shape"):
        C.gather_k_blocks(1, lambda ik: jnp.zeros((3, 3)), item_shape=(4, 4),
                          label="probe", print_fn=lambda *a: None)


def test_local_share_is_round_robin_and_covers_everything():
    items = list(range(9))
    assert C.local_share(items) == items[C.process_rank()::C.process_count()]
    assert C.process_rank_world() == (C.process_rank(), C.process_count())


# ---------------------------------------------------------------------------
# 6. THE ACCEPTANCE TEST: what a reader sees at the top of the driver
# ---------------------------------------------------------------------------

#: Modules a physics driver must not name.  These are the layers the
#: distribution service exists to wrap.
_FORBIDDEN_IN_DRIVER = ("jax.sharding", "jax.experimental.shard_map",
                        "jax.experimental.multihost_utils")


def _jax_plumbing_imports(source: str) -> list[str]:
    """Every import in ``source`` that names a wrapped-away jax module.

    Walks the WHOLE tree, not just module scope, so a lazy import inside a
    function body is caught too — that is where the two ``multihost_utils``
    calls were hiding.
    """
    hits: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            hits += [a.name for a in node.names
                     if any(a.name.startswith(m) for m in _FORBIDDEN_IN_DRIVER)]
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if any(mod.startswith(m) for m in _FORBIDDEN_IN_DRIVER):
                hits.append(mod)
            elif mod in ("jax", "jax.experimental"):
                hits += [f"{mod}.{a.name}" for a in node.names
                         if a.name in ("sharding", "shard_map",
                                       "multihost_utils")]
    return hits


def test_kin_ion_io_names_no_jax_plumbing_module():
    """The owner's stated acceptance test, executable.

    ``gw/kin_ion_io.py`` must not import ``jax.sharding``, ``shard_map`` or
    ``multihost_utils`` anywhere — module scope or lazily.
    """
    hits = _jax_plumbing_imports(_DRIVER.read_text())
    assert hits == [], f"kin_ion_io still reaches under jax: {hits}"


def test_the_import_census_can_fail():
    """NEGATIVE CONTROL for the census.

    Three synthetic drivers, one per spelling the real file used to carry —
    including the LAZY one, which a module-scope-only walk would miss.
    """
    top = "from jax.sharding import Mesh, NamedSharding, PartitionSpec as P\n"
    smap = "from jax.experimental.shard_map import shard_map\n"
    lazy = ("def f():\n"
            "    from jax.experimental import multihost_utils as _mh\n"
            "    _mh.sync_global_devices('x')\n")
    assert _jax_plumbing_imports(top) == ["jax.sharding"]
    assert _jax_plumbing_imports(smap) == ["jax.experimental.shard_map"]
    assert _jax_plumbing_imports(lazy) == ["jax.experimental.multihost_utils"]
    # ...and a clean driver reports nothing, so the census is not just "any
    # file fails".
    assert _jax_plumbing_imports("from common.collectives import barrier\n") == []


def test_kin_ion_io_sets_no_environment_variable_at_module_scope():
    """Policy belongs in ``runtime``, not in a physics driver.

    The four ``os.environ.setdefault`` calls this replaces included
    ``JAX_PLATFORMS`` — i.e. the driver decided the platform for the run.
    """
    tree = ast.parse(_DRIVER.read_text())
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            tgt = node.func.value
            if (node.func.attr in ("setdefault", "__setitem__")
                    and isinstance(tgt, ast.Attribute)
                    and tgt.attr == "environ"):
                offenders.append(ast.dump(node)[:60])
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if (isinstance(t, ast.Subscript)
                        and isinstance(t.value, ast.Attribute)
                        and t.value.attr == "environ"):
                    offenders.append("os.environ[...] = ...")
    assert offenders == [], f"kin_ion_io writes env directly: {offenders}"


def test_the_env_write_census_can_fail():
    """NEGATIVE CONTROL: the same walk over the code that was removed."""
    src = ("import os\n"
           "os.environ.setdefault('JAX_PLATFORMS', 'cuda,cpu')\n"
           "os.environ['X'] = '1'\n")
    tree = ast.parse(src)
    found = 0
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "setdefault"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "environ"):
            found += 1
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if (isinstance(t, ast.Subscript)
                        and isinstance(t.value, ast.Attribute)
                        and t.value.attr == "environ"):
                    found += 1
    assert found == 2


# ---------------------------------------------------------------------------
# 7. Nothing downstream was broken by the move
# ---------------------------------------------------------------------------

#: The names ``gw/kin_ion_io.py``'s forwarding-shim block re-exports.  They
#: are ``common.collectives``' and were only ever re-exported so consumers
#: written before the move kept importing.
_SHIM_NAMES = ("sweep_local_k", "replicate_to_mesh",
               "_psum_replicate", "_gather_indexed_blocks")


def test_no_consumer_imports_the_forwarding_shim_any_more():
    """SELF-RETIRING GATE: the shim block in ``kin_ion_io`` is now dead.

    ``gw/sigma_dispatch.py``, ``tests/test_sanity_gates_jax.py`` and
    ``tests/test_kin_ion_padded_gvectors.py`` were the three consumers; all
    three now import these names from ``common.collectives``.  That makes
    the ``# Forwarding shims — NOT used by anything in this file`` block at
    the bottom of ``gw/kin_ion_io.py`` deletable, which is filed as a
    request because that file is not this workstream's to edit.

    This scans SOURCE rather than importing, so it names the offending file
    and line instead of only failing at the import that reintroduces the
    edge.  Delete this test together with the shim.
    """
    offenders = []
    for path in sorted(_SRC.rglob("*.py")) + sorted(
            (_SRC.parent / "tests").rglob("*.py")):
        if path == _DRIVER or path.resolve() == pathlib.Path(__file__).resolve():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
        except SyntaxError:                       # not ours to police
            continue
        for n in ast.walk(tree):
            if not isinstance(n, ast.ImportFrom) or not n.module:
                continue
            if not n.module.endswith("kin_ion_io"):
                continue
            for a in n.names:
                if a.name in _SHIM_NAMES or a.name == "resolve_mesh":
                    offenders.append(f"{path}:{n.lineno}: {a.name}")
    assert offenders == [], (
        "these import generic collectives plumbing from the kin_ion DRIVER; "
        "import it from common.collectives instead:\n  "
        + "\n  ".join(offenders))


def test_kin_ion_io_shim_if_present_forwards_and_does_not_copy():
    """While the (now dead) shim block survives, it must not have drifted.

    Written to pass BOTH before and after the block is deleted: a stale
    private copy would be a second implementation of a collective, which is
    the failure this whole refactor removes.  The physics entry points are
    checked unconditionally — those are the driver's own.
    """
    from gw import kin_ion_io as K

    for name in ("rho_work_items", "compute_hartree_matrix",
                 "get_kin_ion_k", "build_argparser", "main"):
        assert hasattr(K, name), f"kin_ion_io stopped exporting {name}"
    service = {"sweep_local_k": C.sweep_local_k,
               "replicate_to_mesh": C.replicate_to_mesh,
               "_psum_replicate": C.psum_replicate,
               "_gather_indexed_blocks": C.gather_indexed_blocks}
    for name, obj in service.items():
        if hasattr(K, name):
            assert getattr(K, name) is obj, (
                f"kin_ion_io.{name} is no longer the service's object — the "
                f"shim has drifted into a private copy")


def test_the_driver_still_runs_its_sweep_through_the_service():
    """Not merely importable: the two per-k sweeps must go through
    ``gather_k_blocks``, or the driver quietly kept a private copy.
    """
    src = _DRIVER.read_text()
    calls = [n for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "gather_k_blocks"]
    assert len(calls) == 2, "expected the V_H and kin_ion sweeps"
    labels = sorted(ast.literal_eval(kw.value) for c in calls
                    for kw in c.keywords if kw.arg == "label")
    assert labels == ["kin_ion", "vh_matrix"]


def test_the_driver_gets_its_mesh_and_its_warm_up_from_one_call():
    """The driver must not hand-roll the warm-up next to the mesh: exactly
    one ``prepare_mesh`` call, and no ``nccl_warmup`` import of its own.
    """
    src = _DRIVER.read_text()
    tree = ast.parse(src)
    prep = [n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == "prepare_mesh"]
    assert len(prep) == 1, "the driver builds its run mesh exactly once"
    warms = [a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
             for a in n.names if a.name in ("nccl_warmup", "warm_mesh_cliques")]
    assert warms == [], f"driver still imports a warm-up itself: {warms}"


def test_env_is_untouched_by_this_module():
    """This file monkeypatches ``LORRAX_KIN_ION_LOOKAHEAD``.  If a patch ever
    leaked, every later test in the SAME pytest process — including the D10
    suite, which shares the session — would run with a lookahead nobody chose.
    Compared against the value captured at import, not against "unset", so the
    check is still meaningful when the harness sets the variable itself.
    """
    assert os.environ.get(_LOOKAHEAD_ENV) == _LOOKAHEAD_AT_IMPORT

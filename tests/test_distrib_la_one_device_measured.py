"""The ``distributed`` size route decides each call shape against MAIN's INCUMBENT (owner rules
2026-10-16: no stage's per-rank device peak may go up; 2026-10-17: never regress main).

The incumbent is what today's main runs on that leg and shape: the leg library on a room leg
(CUDA), ``one_device`` in the round rule's count on a size-row leg (ROCM).  A plan that
``distributed`` resolved to ``one_device`` prices, per call signature, the incumbent and the
``one_device`` round counts that could beat it by XLA's ``memory_analysis`` (temp + output per
device) and runs the fewest-round ``one_device`` whose bytes are within the incumbent's and the
room, else the incumbent (``distrib_la.plan._measured_rounds``).  No library resolves on a CPU
host, so its cost is stood in for (``_library_bytes``); the ``one_device`` costs are this host's
compiled programs unless a test stands in figures by round count.  The ROCM size row and the
CUDA room leg are stood in for on a CPU mesh.

Run with ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4``.
"""
import importlib

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update("jax_enable_x64", True)

import distrib_la
resolve_mod = importlib.import_module("distrib_la.resolve")
od = importlib.import_module("distrib_la._one_device")
plan_mod = importlib.import_module("distrib_la.plan")

LIBRARY = {"ROCM": "slate", "CUDA": "cusolvermp"}




@pytest.fixture
def mesh():
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 host devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))


@pytest.fixture(params=["ROCM", "CUDA"])
def leg(request, monkeypatch):
    monkeypatch.setattr(resolve_mod, "mesh_platform", lambda m: request.param)
    monkeypatch.setattr(od, "mesh_platform", lambda m: request.param)
    plan_mod._MEASURED.clear()
    for k in [k for k in resolve_mod._SIZE_DECISIONS if k and k[0] == "measured"]:
        del resolve_mod._SIZE_DECISIONS[k]
    return request.param


def _room(leg):
    return 1 << 40          # both legs' LU are room legs: the room is the size bound


def _lu(nb=16, n=8, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(nb, n, n)) + 1j * rng.normal(size=(nb, n, n)) + 4 * np.eye(n)
    B = rng.normal(size=(nb, n, n)) + 0j
    return A, B


def _face(mesh, x):
    return jax.device_put(x, NamedSharding(mesh, P(None, "x", "y") if x.ndim == 3 else P("x", "y")))


def _abstract(mesh, *xs):
    face = NamedSharding(mesh, P(None, "x", "y"))
    return tuple(jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=face) for x in xs)


def _chosen(p, A, B):
    """The size route's decision for this call: arithmetic on the memoised costs."""
    return plan_mod._measured_rounds(p, A, (B,))


def _od_costs(p, mesh, A, B):
    return {r: plan_mod._program_bytes(lambda a, b, _r=r: p.batched(a, b, _rounds=_r),
                                       _abstract(mesh, A, B))
            for r in plan_mod._round_candidates(A.shape[0], 4)}


def test_this_host_reports_compiled_bytes(mesh, leg):
    A, B = _lu()
    p = distrib_la.plan("solve_lu", mesh, backend="distributed", n=8, batched_route="auto",
                        budget_bytes=_room(leg))
    assert p.backend == "one_device"
    costs = _od_costs(p, mesh, A, B)
    assert set(costs) == {1, 2, 4} and all(c is not None and c > 0 for c in costs.values())


@pytest.mark.parametrize("leg", ["CUDA"], indirect=True)
def test_one_device_dearer_than_the_library_is_refused(mesh, leg, monkeypatch):
    """Every one_device round costs more than the library: the stack runs on the library (on
    this host refused by name), never one_device."""
    monkeypatch.setattr(plan_mod, "_library_bytes", lambda p, abstract, single: 1)
    A, B = _lu()
    p = distrib_la.plan("solve_lu", mesh, backend="distributed", n=8, batched_route="auto",
                        budget_bytes=_room(leg))
    assert p.backend == "one_device"
    with pytest.raises((RuntimeError, ValueError), match=LIBRARY[leg]):
        p.batched(_face(mesh, A), _face(mesh, B))
    assert _chosen(p, A, B) == 0
    lines = [l for l in distrib_la.new_stack_routes() if "measured" in l]
    assert lines and lines[-1].endswith(f"-> {LIBRARY[leg]} (the incumbent)")


@pytest.mark.parametrize("leg", ["CUDA"], indirect=True)
def test_fewest_rounds_within_the_library_cost_run(mesh, leg, monkeypatch):
    A, B = _lu(seed=1)
    p = distrib_la.plan("solve_lu", mesh, backend="distributed", n=8, batched_route="auto",
                        budget_bytes=_room(leg))
    costs = _od_costs(p, mesh, A, B)
    for lib in sorted(set(costs.values())):
        plan_mod._MEASURED.clear()
        monkeypatch.setattr(plan_mod, "_library_bytes", lambda p_, a_, s_, _l=lib: _l)
        want = next(r for r in sorted(costs) if costs[r] <= lib)
        X = p.batched(_face(mesh, A), _face(mesh, B))
        assert _chosen(p, A, B) == want
        assert np.allclose(np.asarray(X), np.linalg.solve(A, B), atol=1e-11)


@pytest.mark.parametrize("leg", ["CUDA"], indirect=True)
def test_the_room_is_a_further_cap(mesh, leg, monkeypatch):
    """An ample library cost but a room below every one_device round: the library."""
    monkeypatch.setattr(plan_mod, "_library_bytes", lambda p, abstract, single: 1 << 50)
    A, B = _lu(seed=2)
    one = od.one_device_bytes("solve_lu", 8)
    p = distrib_la.plan("solve_lu", mesh, backend="distributed", n=8, batched_route="auto",
                        budget_bytes=one)
    assert p.backend == "one_device"
    costs = _od_costs(p, mesh, A, B)
    if min(costs.values()) <= one:
        pytest.skip("this host prices a round within one whole matrix")
    with pytest.raises((RuntimeError, ValueError), match=LIBRARY[leg]):
        p.batched(_face(mesh, A), _face(mesh, B))


def test_no_library_on_this_host_keeps_the_round_rule(mesh, leg):
    A, B = _lu(seed=3)
    p = distrib_la.plan("solve_lu", mesh, backend="distributed", n=8, batched_route="auto",
                        budget_bytes=_room(leg))
    X = p.batched(_face(mesh, A), _face(mesh, B))
    assert _chosen(p, A, B) is None
    assert np.allclose(np.asarray(X), np.linalg.solve(A, B), atol=1e-11)


@pytest.mark.parametrize("leg", ["CUDA"], indirect=True)
def test_one_matrix_eigh_is_measured_too(mesh, leg, monkeypatch):
    monkeypatch.setattr(plan_mod, "_library_bytes", lambda p, abstract, single: 1)
    rng = np.random.default_rng(4)
    a = rng.normal(size=(8, 8)); a = a + a.T
    p = distrib_la.plan("eigh", mesh, backend="distributed", n=8, batched_route="auto",
                        budget_bytes=_room(leg))
    assert p.backend == "one_device"
    with pytest.raises((RuntimeError, ValueError), match=LIBRARY[leg]):
        p(_face(mesh, a))


def test_explicit_one_device_is_not_measured(mesh, leg, monkeypatch):
    monkeypatch.setattr(plan_mod, "_library_bytes", lambda p, abstract, single: 1)
    A, B = _lu(seed=5)
    p = distrib_la.plan("solve_lu", mesh, backend="one_device", n=8, batched_route="auto",
                        budget_bytes=_room(leg) or (1 << 40))
    X = p.batched(_face(mesh, A), _face(mesh, B))
    assert not plan_mod._MEASURED
    assert np.allclose(np.asarray(X), np.linalg.solve(A, B), atol=1e-11)


@pytest.mark.parametrize("leg", ["CUDA"], indirect=True)
def test_costs_are_cached_on_disk_and_the_room_applied_at_lookup(mesh, leg, monkeypatch, tmp_path):
    """A warm process compiles nothing to decide: the costs come from the on-disk cache, and a
    different room is decided from the same cached costs."""
    lib = lambda p, abstract, single: 1 << 50
    monkeypatch.setattr(plan_mod, "_library_bytes", lib)
    monkeypatch.setattr(plan_mod, "_LIBRARY_BYTES_DEFAULT", lib)
    monkeypatch.setattr(plan_mod, "_measured_cache_dir", lambda: str(tmp_path))
    compiled = []
    real = plan_mod._program_bytes
    monkeypatch.setattr(plan_mod, "_program_bytes", lambda fn, ab: compiled.append(1) or real(fn, ab))
    A, B = _lu(seed=6)
    p = distrib_la.plan("solve_lu", mesh, backend="distributed", n=8, batched_route="auto",
                        budget_bytes=_room(leg))
    X = p.batched(_face(mesh, A), _face(mesh, B))
    first = len(compiled)
    assert first == len(plan_mod._round_candidates(16, 4)) and len(list(tmp_path.glob("*.json"))) == 1
    assert _chosen(p, A, B) == 1
    plan_mod._MEASURED.clear()
    p.batched(_face(mesh, A), _face(mesh, B))
    assert len(compiled) == first                       # warm: nothing compiled to decide
    assert _chosen(p, A, B) == 1
    assert np.allclose(np.asarray(X), np.linalg.solve(A, B), atol=1e-11)


def _cached_setup(mesh, leg, monkeypatch, tmp_path):
    lib = lambda p, abstract, single: 1 << 50
    monkeypatch.setattr(plan_mod, "_library_bytes", lib)
    monkeypatch.setattr(plan_mod, "_LIBRARY_BYTES_DEFAULT", lib)
    monkeypatch.setattr(plan_mod, "_measured_cache_dir", lambda: str(tmp_path))
    compiled = []
    real = plan_mod._program_bytes
    monkeypatch.setattr(plan_mod, "_program_bytes", lambda fn, ab: compiled.append(1) or real(fn, ab))
    A, B = _lu(seed=7)
    p = distrib_la.plan("solve_lu", mesh, backend="distributed", n=8, batched_route="auto",
                        budget_bytes=_room(leg))
    p.batched(_face(mesh, A), _face(mesh, B))
    return p, A, B, compiled


@pytest.mark.parametrize("leg", ["CUDA"], indirect=True)
@pytest.mark.parametrize("damage", ["truncate", "garbage", "digest", "stale_key", "missing_round"])
def test_a_damaged_or_stale_cache_entry_is_re_measured(mesh, leg, monkeypatch, tmp_path, damage):
    """A partial, corrupt, tampered or stale entry is a miss: the costs are measured again and the
    route is the same; it is never used to route."""
    import json
    p, A, B, compiled = _cached_setup(mesh, leg, monkeypatch, tmp_path)
    first, want = len(compiled), _chosen(p, A, B)
    (f,) = tmp_path.glob("*.json")
    entry = json.loads(f.read_text())
    if damage == "truncate":
        f.write_text(f.read_text()[: len(f.read_text()) // 2])
    elif damage == "garbage":
        f.write_text("not json")
    elif damage == "digest":
        entry["value"]["library"] = 1
        f.write_text(json.dumps(entry))
    elif damage == "stale_key":
        entry["key"] = entry["key"].replace('"', "'", 1)
        f.write_text(json.dumps(entry))
    else:
        entry["value"]["rounds"].pop(next(iter(entry["value"]["rounds"])))
        entry["digest"] = plan_mod._payload_digest(entry["key"], entry["value"])
        f.write_text(json.dumps(entry))
    plan_mod._MEASURED.clear()
    p.batched(_face(mesh, A), _face(mesh, B))
    assert len(compiled) == 2 * first                    # re-measured
    assert _chosen(p, A, B) == want                      # the same route


def test_the_agreement_is_conservative_and_refuses_a_mixed_library():
    peers = [{"library": 100, "rounds": {"1": 120, "2": 90}},
             {"library": 80, "rounds": {"1": 110, "2": 95}}]
    assert plan_mod._agreed(peers) == {"library": 80, "rounds": {"1": 120, "2": 95},
                                       "rounds_min": {"1": 110, "2": 90}}
    assert plan_mod._agreed([{"library": None, "rounds": {}}] * 2) is None
    # a one_device incumbent: no library asked for; candidates charged their largest bytes,
    # the incumbent credited its smallest
    assert plan_mod._agreed([{"library": None, "rounds": {"1": 5, "2": 9}},
                             {"library": None, "rounds": {"1": 7, "2": 8}}], need_library=False) == \
        {"library": None, "rounds": {"1": 7, "2": 9}, "rounds_min": {"1": 5, "2": 8}}
    with pytest.raises(RuntimeError, match="some"):
        plan_mod._agreed([{"library": 1, "rounds": {"1": 1}}, {"library": None, "rounds": {}}])


@pytest.mark.parametrize("leg", ["CUDA"], indirect=True)
def test_ranks_with_divergent_figures_take_one_route(mesh, leg, monkeypatch, tmp_path):
    """Two ranks whose own figures would route differently (this rank: one round fits; its peer:
    not even the last) decide together: every rank takes the library."""
    monkeypatch.setattr(plan_mod, "_measured_cache_dir", lambda: str(tmp_path))
    A, B = _lu(seed=8)
    p = distrib_la.plan("solve_lu", mesh, backend="distributed", n=8, batched_route="auto",
                        budget_bytes=_room(leg))
    costs = _od_costs(p, mesh, A, B)
    lib = max(costs.values())                             # alone, this rank would take 1 round
    monkeypatch.setattr(plan_mod, "_library_bytes", lambda p_, a_, s_: lib)
    seen = []
    def two_ranks(tag, payload):
        seen.append(tag.rsplit("/", 1)[-1])
        if "hit" in payload:
            return [payload, dict(payload, hit=False)]
        peer = {"library": lib, "rounds": {r: lib + 1 for r in payload["rounds"]}}
        return [payload, peer]
    monkeypatch.setattr(plan_mod, "_exchange", two_ranks)
    plan_mod._MEASURED.clear()
    with pytest.raises((RuntimeError, ValueError), match=LIBRARY[leg]):
        p.batched(_face(mesh, A), _face(mesh, B))
    assert seen == ["probe", "measure"] and _chosen(p, A, B) == 0


@pytest.mark.parametrize("leg", ["CUDA"], indirect=True)
def test_the_native_closure_is_part_of_the_cache_key(mesh, leg, monkeypatch, tmp_path):
    """Costs measured against one native library are another key against another: re-measured."""
    p, A, B, compiled = _cached_setup(mesh, leg, monkeypatch, tmp_path)
    first = len(compiled)
    plan_mod._MEASURED.clear()
    monkeypatch.setattr(plan_mod, "native_closure_identity", lambda platform: "/other/liblorrax_ffi.so:0123")
    p.batched(_face(mesh, A), _face(mesh, B))
    assert len(compiled) == 2 * first and len(list(tmp_path.glob("*.json"))) == 2


def test_native_closure_identity_is_lxkits_enforced_identity(monkeypatch):
    """The cost-cache key is lxkit's enforced closure identity of the platform's leg
    (cs/dep-enforce): another recorded library set is another key, it is not cached here,
    and a leg not yet enforced is "none"."""
    from lxkit import native_provider as npv
    seen = []

    def fake(platform=None):
        seen.append(platform)
        if platform == "CUDA":
            return "a" * 64
        raise npv.ClosureNotEnforced(f"the {platform} leg has not been opened and enforced")

    monkeypatch.setattr(npv, "native_closure_identity", fake)
    assert plan_mod.native_closure_identity("CUDA") == "a" * 64
    assert plan_mod.native_closure_identity("ROCM") == "none"
    monkeypatch.setattr(npv, "native_closure_identity", lambda platform=None: "b" * 64)
    assert plan_mod.native_closure_identity("CUDA") == "b" * 64      # a rebuilt closure: another key
    assert seen == ["CUDA", "ROCM"]



# ---- main's incumbent (owner rule 2026-10-17) ------------------------------------------------
ROCM = pytest.mark.parametrize("leg", ["ROCM"], indirect=True)


def _fake_costs(monkeypatch, table, tmp_path, lib=None):
    """Stand-in compiled bytes by round count (the one_device program at ``_rounds=r``); the
    library's when given.  Returns the round counts compiled, in order."""
    calls = []

    def prog(fn, abstract):
        r = (fn.__kwdefaults__ or {}).get("_r", 1)
        calls.append(r)
        return table[r]
    monkeypatch.setattr(plan_mod, "_program_bytes", prog)
    monkeypatch.setattr(plan_mod, "_measured_cache_dir", lambda: str(tmp_path))
    if lib is not None:
        monkeypatch.setattr(plan_mod, "_library_bytes", lambda p, a, s: lib)
    return calls


def _row(monkeypatch, bound, leg="ROCM"):
    """The ROCM solve_lu size row stood in at ``bound`` (n=8, 16 matrices, 2x2: bound 11 -> the
    round rule runs 4 rounds, 12 -> 2, the real row -> 1)."""
    rows = dict(resolve_mod.ONE_DEVICE_MAX_N)
    rows[("solve_lu", leg)] = bound
    monkeypatch.setattr(resolve_mod, "ONE_DEVICE_MAX_N", rows)


def _lu_plan(mesh, leg):
    p = distrib_la.plan("solve_lu", mesh, backend="distributed", n=8, batched_route="auto",
                        budget_bytes=_room(leg))
    assert p.backend == "one_device"
    return p


def _ledger():
    """This test's size-route decision lines (the route ledger's measured entries)."""
    return [l for k, l in resolve_mod._SIZE_DECISIONS.items() if k and k[0] == "measured"]


def test_the_incumbent_is_mains_route(mesh, leg, monkeypatch):
    p = _lu_plan(mesh, leg)
    lu = ((16, 8, 8), (16, 8, 8))
    assert plan_mod._incumbent(p, lu, False) == 0                    # a room leg: the library
    _row(monkeypatch, 11, leg)                                       # a size row stood in
    assert plan_mod._incumbent(p, lu, False) == 4
    _row(monkeypatch, 12, leg)
    assert plan_mod._incumbent(p, lu, False) == 2


@ROCM
def test_a_one_round_one_device_incumbent_is_never_measured(mesh, leg, monkeypatch, tmp_path):
    """A size-row leg whose main route is one_device in 1 round: nothing can have fewer rounds, so
    nothing is compiled, no library is priced and no rank exchanges anything; main's program runs."""
    _row(monkeypatch, 1 << 20)
    calls = _fake_costs(monkeypatch, {1: 100, 2: 10, 4: 1}, tmp_path, lib=1)
    asked = []
    monkeypatch.setattr(plan_mod, "_library_bytes", lambda p, a, s: asked.append(1) or 1)
    monkeypatch.setattr(plan_mod, "_exchange", lambda tag, payload: pytest.fail("exchanged"))
    A, B = _lu(seed=11)
    p = _lu_plan(mesh, leg)
    X = p.batched(_face(mesh, A), _face(mesh, B))
    assert calls == [] and asked == [] and _chosen(p, A, B) is None
    assert np.allclose(np.asarray(X), np.linalg.solve(A, B), atol=1e-11)
    (line,) = _ledger()
    assert "nothing measured" in line and line.endswith("-> one_device in 1 round(s) (the incumbent)")
    assert "rule: the fewest-round one_device whose per-rank footprint is within the incumbent's" in line


@ROCM
def test_lower_memory_but_more_rounds_than_the_one_device_incumbent_keeps_it(mesh, leg, monkeypatch, tmp_path):
    """Incumbent one_device in 2 rounds; 4 rounds would cost less memory but has more rounds (never
    a candidate, never compiled), 1 round costs more: the incumbent (main's round rule)."""
    _row(monkeypatch, 12)
    calls = _fake_costs(monkeypatch, {1: 100, 2: 50, 4: 10}, tmp_path)
    A, B = _lu(seed=12)
    p = _lu_plan(mesh, leg)
    X = p.batched(_face(mesh, A), _face(mesh, B))
    assert sorted(set(calls)) == [1, 2] and _chosen(p, A, B) is None
    assert np.allclose(np.asarray(X), np.linalg.solve(A, B), atol=1e-11)
    (line,) = _ledger()
    assert "incumbent one_device in 2 round(s) 0.0000" in line
    assert line.endswith("-> one_device in 2 round(s) (the incumbent)")


@ROCM
@pytest.mark.parametrize("table,want", [({1: 100, 2: 60, 4: 70}, 2),
                                        ({1: 70, 2: 60, 4: 70}, 1),
                                        ({1: 100, 2: 80, 4: 70}, None)])
def test_fewer_rounds_within_the_one_device_incumbent_take_it(mesh, leg, monkeypatch, tmp_path, table, want):
    """Incumbent one_device in 4 rounds: the fewest-round candidate within its bytes runs (ties
    included), else the incumbent."""
    _row(monkeypatch, 11)
    calls = _fake_costs(monkeypatch, table, tmp_path)
    A, B = _lu(seed=13)
    p = _lu_plan(mesh, leg)
    X = p.batched(_face(mesh, A), _face(mesh, B))
    assert sorted(set(calls)) == [1, 2, 4] and _chosen(p, A, B) == want
    assert np.allclose(np.asarray(X), np.linalg.solve(A, B), atol=1e-11)
    (line,) = _ledger()
    assert line.endswith("(the incumbent)" if want is None else f"-> one_device in {want} round(s)")


@pytest.mark.parametrize("leg", ["CUDA"], indirect=True)
@pytest.mark.parametrize("lib,want", [(60, 4), (90, 2), (100, 1), (40, 0)])
def test_a_library_incumbent_takes_one_device_in_k_rounds_within_it(mesh, leg, monkeypatch, tmp_path, lib, want):
    """CUDA's main route is the library: the fewest one_device rounds within its bytes run, else
    the library (refused by name on this host)."""
    calls = _fake_costs(monkeypatch, {1: 100, 2: 80, 4: 50}, tmp_path, lib=lib)
    A, B = _lu(seed=14)
    p = _lu_plan(mesh, leg)
    if want == 0:
        with pytest.raises((RuntimeError, ValueError), match=LIBRARY[leg]):
            p.batched(_face(mesh, A), _face(mesh, B))
    else:
        X = p.batched(_face(mesh, A), _face(mesh, B))
        assert np.allclose(np.asarray(X), np.linalg.solve(A, B), atol=1e-11)
    assert sorted(set(calls)) == [1, 2, 4] and _chosen(p, A, B) == want
    (line,) = _ledger()
    assert f"incumbent cusolvermp {lib / 1e9:.4f}" in line
    assert line.endswith("-> cusolvermp (the incumbent)" if want == 0 else f"-> one_device in {want} round(s)")


@ROCM
def test_a_k_round_incumbent_figure_is_cached_and_measured_once(mesh, leg, monkeypatch, tmp_path):
    """The one_device incumbent's measured figures are cached like the library's: a warm process
    compiles nothing to decide and takes the same route."""
    _row(monkeypatch, 11)
    calls = _fake_costs(monkeypatch, {1: 100, 2: 60, 4: 70}, tmp_path)
    A, B = _lu(seed=15)
    p = _lu_plan(mesh, leg)
    p.batched(_face(mesh, A), _face(mesh, B))
    first = len(calls)
    assert first == 3 and len(list(tmp_path.glob("*.json"))) == 1
    plan_mod._MEASURED.clear()
    p.batched(_face(mesh, A), _face(mesh, B))
    assert len(calls) == first and _chosen(p, A, B) == 2


# ---- the decision a caller keys on (Plan.decision, audit C4) ---------------------------------
@pytest.mark.parametrize("leg", ["CUDA"], indirect=True)
def test_the_decision_is_what_the_program_runs_and_not_the_room(mesh, leg, monkeypatch):
    """CUDA room leg, no library on this host: one_device's round rule against the room's bound.
    Two rooms with the same rounds are one decision (the bound itself moves with the room and
    reaches no program); a room that changes the rounds flips it; and the program batched
    traces runs the decision's rounds."""
    ran = []
    real = od.batched_distributed_solve_lu
    monkeypatch.setattr(od, "batched_distributed_solve_lu",
                        lambda *a, rounds=None, **k: ran.append(rounds) or real(*a, rounds=rounds, **k))
    A, B = _lu(seed=17)
    one = od.one_device_bytes("solve_lu", 8)

    face = NamedSharding(mesh, P(None, "x", "y"))

    def decided(room):
        p = distrib_la.plan("solve_lu", mesh, backend="distributed", n=8, batched_route="auto",
                            budget_bytes=room)
        d = p.decision(*_abstract(mesh, A, B))
        X = distrib_la.checked_program(lambda a, b: p.batched(a, b), mesh, face,
                                       in_shardings=(face, face))(_face(mesh, A), _face(mesh, B))
        assert ran[-1] == d[2] and np.allclose(np.asarray(X), np.linalg.solve(A, B), atol=1e-11)
        return d
    assert resolve_mod.one_device_max_n("solve_lu", "CUDA", 10 ** 10) != \
        resolve_mod.one_device_max_n("solve_lu", "CUDA", 10 ** 10 + 10 ** 7)
    assert decided(10 ** 10) == decided(10 ** 10 + 10 ** 7) == ("one_device", "backend_batched", 1)
    assert decided(one) == ("one_device", "backend_batched", 4)      # bound 8: one matrix a round


@pytest.mark.parametrize("leg", ["CUDA"], indirect=True)
def test_a_new_room_is_decided_from_the_memoised_costs(mesh, leg, monkeypatch, tmp_path):
    """The agreed costs are memoised without the room: another room exchanges nothing, reads
    and compiles nothing, and applies itself to the same figures."""
    one = od.one_device_bytes("solve_lu", 8)
    calls = _fake_costs(monkeypatch, {1: 3 * one, 2: 2 * one, 4: one}, tmp_path, lib=2 * one + 1)
    seen = []
    real = plan_mod._exchange
    monkeypatch.setattr(plan_mod, "_exchange", lambda tag, payload: seen.append(tag) or real(tag, payload))
    A, B = _lu(seed=18)
    plans = [distrib_la.plan("solve_lu", mesh, backend="distributed", n=8, batched_route="auto",
                             budget_bytes=room) for room in (1 << 40, one)]
    assert _chosen(plans[0], A, B) == 2
    first = (len(calls), len(seen))
    assert _chosen(plans[1], A, B) == 4                               # only 4 rounds fit the room
    assert (len(calls), len(seen)) == first == (3, 2)


@pytest.mark.parametrize("leg", ["CUDA"], indirect=True)
def test_a_room_that_differs_across_ranks_is_refused(mesh, leg, monkeypatch, tmp_path):
    """The room rides the probe: ranks that disagree on it are refused by name, never routed."""
    _fake_costs(monkeypatch, {1: 100, 2: 80, 4: 50}, tmp_path, lib=90)
    monkeypatch.setattr(plan_mod, "_exchange",
                        lambda tag, payload: [payload, dict(payload, room=payload["room"] + 1)])
    with pytest.raises(RuntimeError, match="room that differs across processes"):
        _chosen(_lu_plan(mesh, leg), *_lu(seed=19))

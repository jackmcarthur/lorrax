"""The ``distributed`` size route decides each call shape by MEASURED cost (owner rule
2026-10-16: no stage's per-rank device peak may go up, even inside a caller's room).

A plan that ``distributed`` resolved to ``one_device`` compiles, per call signature, the leg
library's program and the ``one_device`` program at each candidate round count, prices both by
XLA's ``memory_analysis`` (temp + output per device) and runs ``one_device`` in the fewest
rounds whose bytes are within the library's and the room, else the library
(``distrib_la.plan._measured_rounds``).  No library resolves on a CPU host, so its cost is
stood in for (``_library_bytes``); the ``one_device`` costs are this host's compiled programs.
The ROCM size row and the CUDA room leg are stood in for on a CPU mesh.

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
    return request.param


def _room(leg):
    return None if leg == "ROCM" else 1 << 40          # CUDA: the room is its size bound


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
    assert list(plan_mod._MEASURED.values()) == [0]
    lines = [l for l in distrib_la.new_stack_routes() if "measured" in l]
    assert lines and lines[-1].endswith(f"-> {LIBRARY[leg]}")


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
        assert list(plan_mod._MEASURED.values()) == [want]
        assert np.allclose(np.asarray(X), np.linalg.solve(A, B), atol=1e-11)


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
    assert list(plan_mod._MEASURED.values()) == [None]
    assert np.allclose(np.asarray(X), np.linalg.solve(A, B), atol=1e-11)


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


def test_costs_are_cached_on_disk_and_the_room_applied_at_lookup(mesh, leg, monkeypatch, tmp_path):
    """A warm process compiles nothing to decide: the costs come from the on-disk cache, and a
    different room is decided from the same cached costs."""
    lib = lambda p, abstract, single: 1 << 50
    monkeypatch.setattr(plan_mod, "_library_bytes", lib)
    monkeypatch.setattr(plan_mod, "_LIBRARY_BYTES_DEFAULT", lib)
    monkeypatch.setattr(plan_mod, "_measured_cache_path", lambda: str(tmp_path / "measured_cost.json"))
    compiled = []
    real = plan_mod._program_bytes
    monkeypatch.setattr(plan_mod, "_program_bytes", lambda fn, ab: compiled.append(1) or real(fn, ab))
    A, B = _lu(seed=6)
    p = distrib_la.plan("solve_lu", mesh, backend="distributed", n=8, batched_route="auto",
                        budget_bytes=_room(leg))
    X = p.batched(_face(mesh, A), _face(mesh, B))
    first = len(compiled)
    assert first == len(plan_mod._round_candidates(16, 4)) and (tmp_path / "measured_cost.json").exists()
    assert list(plan_mod._MEASURED.values()) == [1]
    plan_mod._MEASURED.clear()
    p.batched(_face(mesh, A), _face(mesh, B))
    assert len(compiled) == first                       # warm: nothing compiled to decide
    assert list(plan_mod._MEASURED.values()) == [1]
    # A room below every round, decided from the cached costs: the library.
    plan_mod._MEASURED.clear()
    if leg == "ROCM":            # (CUDA: the room is its bound, so a tiny room resolves the library)
        q = distrib_la.plan("solve_lu", mesh, backend="distributed", n=8, batched_route="auto",
                            budget_bytes=od.one_device_bytes("solve_lu", 8))
        assert q.backend == "one_device"
        with pytest.raises((RuntimeError, ValueError), match=LIBRARY[leg]):
            q.batched(_face(mesh, A), _face(mesh, B))
        assert len(compiled) == first
    assert np.allclose(np.asarray(X), np.linalg.solve(A, B), atol=1e-11)

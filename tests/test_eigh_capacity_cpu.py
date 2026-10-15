"""The deck's eigh request at the capacity sites (``linalg`` -> one owner).

``gw.gw_config.resolve_linalg`` is the one owner of the eigh backend the
whole-mesh sites request (``LinalgResolution.eigh_backend``): ``auto`` under
``linalg = local``, ``distributed`` under ``linalg = distributed``.  An ``auto``
eigh plan with a room (``distrib_la.plan(..., budget_bytes=)``) is the plan an
explicit ``distributed`` request builds wherever that resolves, and a route (c)
plan that refuses by name at a stack needing the whole mesh where it does not
(a CPU mesh with no host library here; a ROCm leg without SLATE on Frontier).

Run with ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4``.
"""
import dataclasses
import importlib

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import distrib_la
plan_mod = importlib.import_module("distrib_la.plan")
resolve_mod = importlib.import_module("distrib_la.resolve")


@pytest.fixture
def mesh():
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 host devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))


def _no_library(mesh):
    try:
        resolve_mod.resolve_backend("eigh", "distributed", mesh, n=8)
    except (ValueError, RuntimeError):
        return True
    return False


def _hermitian_stack(mesh, nb=4, n=8, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(nb, n, n)) + 1j * rng.normal(size=(nb, n, n))
    a = a + np.conj(np.swapaxes(a, -1, -2))
    return jax.device_put(a, NamedSharding(mesh, P(None, "x", "y")))


def test_linalg_dial_owns_the_eigh_request():
    from gw.gw_config import resolve_linalg
    assert resolve_linalg({"linalg": "local"}).eigh_backend == "auto"
    assert resolve_linalg({"linalg": "distributed"}).eigh_backend == "distributed"


def test_distributed_still_refuses_where_no_library(mesh):
    if not _no_library(mesh):
        pytest.skip("this host carries the distributed eigh library")
    with pytest.raises((ValueError, RuntimeError)):
        distrib_la.plan("eigh", mesh, n=8, backend="distributed", budget_bytes=1 << 30)


def test_auto_with_room_runs_route_c_and_refuses_by_name(mesh):
    if not _no_library(mesh):
        pytest.skip("this host carries the distributed eigh library")
    plan_mod._STACK_ROUTES.clear()
    p = distrib_la.plan("eigh", mesh, n=8, backend="auto", budget_bytes=1 << 30)
    assert p.is_native and p.capacity_refusal
    assert p.batch_in_sharding == NamedSharding(mesh, P(None, "x", "y"))
    assert p.stack_route((4, 8, 8), np.complex128).route == plan_mod.ROUTE_BATCH_RESHARD
    a = _hermitian_stack(mesh)
    w, v = p.batched(a)
    ref = distrib_la.plan("eigh", mesh, n=8, backend="off", batched_route="batch_reshard")
    w0, v0 = ref.batched(a)
    # route (c) is one program whatever plan carries it: bitwise the explicit native route
    assert np.array_equal(np.asarray(w), np.asarray(w0)) and np.array_equal(np.asarray(v), np.asarray(v0))
    tiny = dataclasses.replace(p, budget_bytes=10)
    with pytest.raises(RuntimeError, match="GATE eigh_capacity_route"):
        tiny.stack_route((4, 8, 8), np.complex128)
    with pytest.raises(RuntimeError, match="GATE eigh_capacity_route"):
        dataclasses.replace(p, budget_bytes=0).stack_route((4, 8, 8), np.complex128)
    with pytest.raises(RuntimeError, match="GATE eigh_capacity_route"):
        p.require_capacity("a whole-mesh stack")


def test_auto_with_room_equals_distributed_where_it_resolves(mesh, monkeypatch):
    """CUDA / CPU-with-library: the auto plan IS the distributed plan (bar its spelling)."""
    real = resolve_mod.resolve_backend
    lib = lambda op, requested, mesh_xy, **kw: (
        "scalapack" if requested == "distributed" else real(op, requested, mesh_xy, **kw))
    monkeypatch.setattr(resolve_mod, "resolve_backend", lib)
    monkeypatch.setattr(plan_mod, "resolve_backend", lib)
    plan_mod._STACK_ROUTES.clear()
    a = distrib_la.plan("eigh", mesh, n=8, backend="auto", budget_bytes=1 << 30)
    d = distrib_la.plan("eigh", mesh, n=8, backend="distributed", budget_bytes=1 << 30)
    assert a.capacity_refusal is None and a.backend == d.backend == "scalapack"
    assert dataclasses.replace(a, requested="distributed") == d
    for room in (1 << 30, 10, 0):
        ra = dataclasses.replace(a, budget_bytes=room).stack_route((4, 8, 8), np.complex128)
        rd = dataclasses.replace(d, budget_bytes=room).stack_route((4, 8, 8), np.complex128)
        assert ra == rd


def test_staged_whole_mesh_refuses_by_name_without_library(mesh):
    if not _no_library(mesh):
        pytest.skip("this host carries the distributed eigh library")
    from gw import shared_pole_execution as spe
    spe._mesh_eigh.cache_clear()
    with pytest.raises(RuntimeError, match="GATE eigh_capacity_route"):
        spe._mesh_eigh(mesh, 8, "auto").require_capacity("a staged stack")
    with pytest.raises((ValueError, RuntimeError)):
        spe._mesh_eigh(mesh, 8, "distributed")


def test_polar_route_c_is_independent_of_the_eigh_backend(mesh, monkeypatch):
    """parallel transport under linalg = local: batch_reshard never calls the eigh provider."""
    n = 8
    rng = np.random.default_rng(3)
    a = rng.normal(size=(4, n, n)) + 1j * rng.normal(size=(4, n, n))
    a = jax.device_put(a, NamedSharding(mesh, P(None, "x", "y")))
    auto = distrib_la.plan_polar_factor(mesh, n=n, backend="auto", rcond=1e-10,
                                        batched_route="batch_reshard")
    real = resolve_mod.resolve_backend
    lib = lambda op, requested, mesh_xy, **kw: (
        "scalapack" if requested == "distributed" else real(op, requested, mesh_xy, **kw))
    monkeypatch.setattr(plan_mod, "resolve_backend", lib)
    dist = distrib_la.plan_polar_factor(mesh, n=n, backend="distributed", rcond=1e-10,
                                        batched_route="batch_reshard")
    assert auto.batched_route == dist.batched_route == plan_mod.ROUTE_BATCH_RESHARD
    la, sa = auto.batched(a)
    ld, sd = dist.batched(a)
    assert np.array_equal(np.asarray(la), np.asarray(ld)) and np.array_equal(np.asarray(sa), np.asarray(sd))


def test_sigma_face_gemm_query_matches_the_old_eigh_carrier(mesh, monkeypatch):
    """mpa.sigma prices the face GEMM through the GEMM service: same bytes as the eigh-plan carrier."""
    from distrib_la import workspace as ws
    from distrib_la.plan import Plan
    monkeypatch.setattr(ws, "mesh_platform", lambda m: "CUDA")
    face = NamedSharding(mesh, P(None, "x", "y"))
    carrier = Plan(op="eigh", requested="distributed", backend="cusolvermp", mesh=mesh, n=8,
                   in_sharding=NamedSharding(mesh, P("x", "y")), batch_in_sharding=face,
                   requested_batched_route="auto")
    shapes = ((3, 8, 4), (3, 4, 8))
    old = distrib_la.workspace_bytes_per_rank(carrier, "gemm", shapes, np.complex128)
    new = distrib_la.matmul_workspace_bytes_per_rank(mesh, shapes, np.complex128,
                                                     backend="distributed", batched_route="auto")
    assert old == new == 0

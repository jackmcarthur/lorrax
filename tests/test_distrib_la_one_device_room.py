"""The room cap of the ``distributed`` size route (resolve guard 1b).

``one_device`` holds whole matrices on a rank.  A caller that hands
``distributed`` a room (``plan(budget_bytes=)``) has found that whole matrices
per rank do not fit it (``lxkit.memory.admit``, memory-model.md#routes),
so the size route takes ``one_device`` only where ONE whole matrix's priced
program (``distrib_la._one_device.one_device_bytes``) fits that room; otherwise
the leg's library runs.  Without a room the route is unchanged; on CUDA (no size
row) the room is the bound.  The ROCM leg is stood in for on a CPU mesh.

Run with ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4``.
"""
import importlib

import jax
import numpy as np
import pytest
from jax.sharding import Mesh

import distrib_la
resolve_mod = importlib.import_module("distrib_la.resolve")
od = importlib.import_module("distrib_la._one_device")
plan_mod = importlib.import_module("distrib_la.plan")




@pytest.fixture
def mesh():
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 host devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))


@pytest.fixture
def rocm(monkeypatch):
    monkeypatch.setattr(resolve_mod, "mesh_platform", lambda m: "ROCM")
    monkeypatch.setattr(od, "mesh_platform", lambda m: "ROCM")


def test_one_device_price_is_the_route_c_stack_price():
    n = 1024
    assert od.one_device_bytes("eigh", n) == plan_mod.BATCH_EIGH_TILES * n * n * 16
    assert od.one_device_bytes("solve_lu", n) == od.ONE_DEVICE_LU_TILES * n * n * 16
    with pytest.raises(ValueError):
        od.one_device_bytes("cholesky", n)


@pytest.mark.parametrize("op,n", [("eigh", 64), ("solve_lu", 64)])
def test_room_caps_the_size_route(mesh, rocm, op, n):
    need = od.one_device_bytes(op, n)
    # No room: eigh keeps its size row; solve_lu (a room leg) refuses by name.  Room for one
    # whole matrix: one_device.
    if op == "eigh":
        assert resolve_mod.resolve_backend(op, "distributed", mesh, n=n) == "one_device"
    else:
        with pytest.raises(ValueError, match="GATE distrib_la_room"):
            resolve_mod.resolve_backend(op, "distributed", mesh, n=n)
    assert resolve_mod.resolve_backend(op, "distributed", mesh, n=n, room=need) == "one_device"
    # One byte short (and the no-room 0 of a whole-mesh caller): the library
    # (no SLATE handler on this host, so the ladder names it).
    library = {"eigh": "elpa", "solve_lu": "slate"}[op]      # ROCM's distributed library per op
    for room in (need - 1, 0):
        with pytest.raises((RuntimeError, ValueError), match=f"{op} backend '{library}'"):
            resolve_mod.resolve_backend(op, "distributed", mesh, n=n, room=room)
    # An explicit library or the capacity provider is untouched by the room.
    with pytest.raises((RuntimeError, ValueError), match="slate"):
        resolve_mod.resolve_backend(op, "slate", mesh, n=n, room=10 * need)


def test_plan_passes_its_budget_to_the_size_route(mesh, rocm):
    n = 64
    need = od.one_device_bytes("eigh", n)
    p = distrib_la.plan("eigh", mesh, n=n, backend="distributed", budget_bytes=need)
    assert p.backend == "one_device"
    with pytest.raises((RuntimeError, ValueError), match="eigh backend 'elpa'"):
        distrib_la.plan("eigh", mesh, n=n, backend="distributed", budget_bytes=need - 1)
    note = resolve_mod.size_route_note("eigh", "distributed", mesh, n, "slate")
    assert note.endswith("-> slate (one whole matrix over the room)")


def test_room_is_the_cuda_bound(mesh, monkeypatch):
    """CUDA has no size row: without a room it keeps cuSOLVERMp; a room that holds one whole
    matrix is its one_device bound (resolve.ONE_DEVICE_BY_ROOM)."""
    monkeypatch.setattr(resolve_mod, "mesh_platform", lambda m: "CUDA")
    monkeypatch.setattr(od, "mesh_platform", lambda m: "CUDA")
    for room in (None, 0):
        with pytest.raises((RuntimeError, ValueError), match="cusolvermp"):
            resolve_mod.resolve_backend("eigh", "distributed", mesh, n=64, room=room)
    assert resolve_mod.resolve_backend("eigh", "distributed", mesh, n=64, room=1 << 40) == "one_device"

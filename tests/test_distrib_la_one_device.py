"""The ``distributed`` size route (resolve guard 1b) and its ``one_device`` backend.

``distributed`` on a leg with a ``ONE_DEVICE_MAX_N`` row (the ROCM leg) and a known ``n`` within
it resolves to ``one_device``: each matrix whole on one device, the native kernel, the caller's
face layout.  Beyond the bound, with ``n`` unknown, and for the capacity provider it stays the
leg's library.  ROCM has size rows; on CUDA the bound is the caller's room.  The legs are stood in
for by patching the platform key on a CPU mesh; the numerics are route (c)'s on any device.

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
@pytest.fixture
def mesh():
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 host devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))


#: The room a ROCM caller hands its LU plan: one GCD's budget (48.99 of 60.18 GB).
def _stand_in(monkeypatch, platform):
    monkeypatch.setattr(resolve_mod, "mesh_platform", lambda m: platform)
    monkeypatch.setattr(od, "mesh_platform", lambda m: platform)


def _face(mesh, a):
    spec = P(None, "x", "y") if a.ndim == 3 else P("x", "y")
    return jax.device_put(a, NamedSharding(mesh, spec))


def _hermitian(nb, n, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(nb, n, n)) + 1j * rng.normal(size=(nb, n, n))
    return a + np.conj(np.swapaxes(a, -1, -2))


def _eigh_ok(a, w, v):
    w, v = np.asarray(w), np.asarray(v)
    assert np.allclose(w, np.linalg.eigvalsh(a), atol=1e-11)
    r = np.linalg.norm(a @ v - v * w[..., None, :]) / np.linalg.norm(a)
    assert r < 1e-12


# ---- the router: one table (rows, GCD room, CUDA bound, room cap) ------------------------------
_LIB = (RuntimeError, ValueError)     # the library: absent here (RuntimeError) or refused later (ValueError)
#: (platform, op, requested, n, room, expected); n and room are keys of _n / _room below; the
#: expectation is ("=", backend) or ("raises", regex).
#: main 4bae15f55's ONE_DEVICE_MAX_N rows: the bound its round rule planned with.
# ---- the size route returns the right numbers --------------------------------------------------
def test_one_device_numerics_eigh_lu(mesh, monkeypatch):
    # eigh: whole matrices per rank, in rounds (8: one matrix per rank per round), face layout out
    a = _hermitian(5, 8)                              # ragged against P = 4
    for max_n in (8, 64):
        w, v = od.batched_distributed_eigh(_face(mesh, a), mesh=mesh, max_n=max_n)
        _eigh_ok(a, w, v)
        assert v.sharding.is_equivalent_to(NamedSharding(mesh, P(None, "x", "y")), 3)
        w1, v1 = od.distributed_eigh(_face(mesh, a[2]), mesh=mesh, max_n=max_n)
        _eigh_ok(a[2], w1, v1)
        assert v1.sharding.is_equivalent_to(NamedSharding(mesh, P("x", "y")), 2)
    # LU: one round and two rounds equal numpy
    rng = np.random.default_rng(5)
    A = rng.normal(size=(6, 8, 8)) + 1j * rng.normal(size=(6, 8, 8))
    B = rng.normal(size=(6, 8, 4)) + 1j * rng.normal(size=(6, 8, 4))
    ref = np.linalg.solve(A, B)
    one = od.batched_distributed_solve_lu(_face(mesh, A), _face(mesh, B), mesh=mesh, max_n=64)
    two = od.batched_distributed_solve_lu(_face(mesh, A), _face(mesh, B), mesh=mesh, max_n=8)
    assert od.rounds_for(6, 8, 4, 8, width=4) == 2
    for x in (one, two):
        assert np.allclose(np.asarray(x), ref, rtol=0, atol=1e-11)
        assert x.sharding.is_equivalent_to(NamedSharding(mesh, P(None, "x", "y")), 3)
    # through the plan, the room both selects one_device and bounds its rounds (one matrix a round);
    # a room one byte short of one matrix never builds the plan on one_device
    _stand_in(monkeypatch, "CUDA")
    room = od.one_device_bytes("eigh", 8)
    p = distrib_la.plan("eigh", mesh, backend="distributed", n=8, batched_route="auto", budget_bytes=room)
    assert p.backend == "one_device"
    ah = _hermitian(6, 8, seed=4)
    w, v = p.batched(_face(mesh, ah))
    _eigh_ok(ah, w, v)
    with pytest.raises(_LIB, match="cusolvermp"):
        distrib_la.plan("eigh", mesh, backend="distributed", n=8, batched_route="auto",
                        budget_bytes=room - 1)
    A5 = rng.normal(size=(5, 8, 8)) + 1j * rng.normal(size=(5, 8, 8))
    B5 = rng.normal(size=(5, 8, 8)) + 0j
    lu = distrib_la.plan("solve_lu", mesh, backend="distributed", n=8, batched_route="auto",
                         budget_bytes=od.one_device_bytes("solve_lu", 8))
    assert lu.backend == "one_device"
    X = lu.batched(_face(mesh, A5), _face(mesh, B5))
    assert np.allclose(np.asarray(X), np.linalg.solve(A5, B5), atol=1e-11)

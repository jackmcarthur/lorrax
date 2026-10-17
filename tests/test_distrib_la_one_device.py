"""The ``distributed`` size route (resolve guard 1b) and its ``one_device`` backend.

``distributed`` on a leg with a ``ONE_DEVICE_MAX_N`` row (the ROCM leg) and a
known ``n`` within it resolves to ``one_device``: each matrix whole on one
device, the native kernel, the caller's face layout.  Beyond the bound, with
``n`` unknown, and for the capacity provider it stays the leg's library; a leg
without a row (CUDA) is unchanged.  The ROCM leg is stood in for by patching
the platform key on a CPU mesh; the numerics are route (c)'s on any device.

Run with ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4``.
"""
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import shard_map
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


@pytest.fixture
def rocm(monkeypatch):
    """The ROCM leg's resolution on a CPU mesh (its library has no handler here)."""
    monkeypatch.setattr(resolve_mod, "mesh_platform", lambda m: "ROCM")
    monkeypatch.setattr(od, "mesh_platform", lambda m: "ROCM")


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


def test_size_route_rows_are_rocm_only():
    assert set(resolve_mod.ONE_DEVICE_MAX_N) == {("eigh", "ROCM"), ("solve_lu", "ROCM")}
    assert resolve_mod.one_device_max_n("eigh", "CUDA") is None
    assert "one_device" in resolve_mod.EIGH_BACKENDS and "one_device" in resolve_mod.LU_BACKENDS


def test_distributed_takes_the_size_route_on_rocm(mesh, rocm, capsys):
    bound = resolve_mod.ONE_DEVICE_MAX_N[("eigh", "ROCM")]
    assert resolve_mod.resolve_backend("eigh", "distributed", mesh, n=64) == "one_device"
    lu_bound = resolve_mod.ONE_DEVICE_MAX_N[("solve_lu", "ROCM")]
    assert resolve_mod.resolve_backend("solve_lu", "distributed", mesh, n=lu_bound) == "one_device"
    # Beyond the bound and with n unknown: the leg library, refused on this host by name.
    for n in (bound + 4, None):
        # RuntimeError: the loaded library has no SLATE handler; ValueError: it has one
        # and the ladder refuses later (one JAX process per mesh device).
        with pytest.raises((RuntimeError, ValueError), match="slate"):
            resolve_mod.resolve_backend("eigh", "distributed", mesh, n=n)
    # The capacity provider is the library, never the size route.
    backend, why = resolve_mod.resolve_eigh_capacity(mesh, n=64)
    assert backend == "native" and "slate" in why
    assert resolve_mod.size_route_note("eigh", "distributed", mesh, 64, "one_device") == \
        f"size route ROCM: n=64 <= {bound} -> one_device"
    assert resolve_mod.size_route_note("eigh", "slate", mesh, 64, "slate") is None
    assert "size route" in capsys.readouterr().out
    lines = [l for l in distrib_la.new_stack_routes() if "size route" in l]
    assert any(l.startswith("eigh 'distributed' n=64 on the 2x2 ROCM mesh: one_device,") for l in lines)
    assert not [l for l in distrib_la.new_stack_routes() if "size route" in l]   # reported once


def test_cuda_resolution_is_unchanged(mesh, monkeypatch):
    monkeypatch.setattr(resolve_mod, "mesh_platform", lambda m: "CUDA")
    # RuntimeError: no loadable CUDA library on this host; ValueError: it loads (a GPU node)
    # and the ladder refuses later (one JAX process per mesh device), as the ROCM case above.
    with pytest.raises((RuntimeError, ValueError), match="cusolvermp"):
        resolve_mod.resolve_backend("eigh", "distributed", mesh, n=64)
    with pytest.raises(ValueError, match="GATE one_device"):
        resolve_mod.resolve_backend("eigh", "one_device", mesh, n=64)
    assert resolve_mod.size_route_note("eigh", "distributed", mesh, 64, "cusolvermp") is None


def test_auto_lu_on_rocm_probes_no_cuda_handler(mesh, rocm, capsys):
    resolve_mod._AUTO_CAPABILITY_DEMOTE_ANNOUNCED.clear()
    assert resolve_mod.resolve_backend("solve_lu", "auto", mesh) == "native"
    assert resolve_mod.resolve_backend("cholesky", "auto", mesh) == "native"
    assert "NOT AVAILABLE" not in capsys.readouterr().out


def test_explicit_one_device_guards(mesh, rocm):
    bound = resolve_mod.ONE_DEVICE_MAX_N[("eigh", "ROCM")]
    assert resolve_mod.resolve_backend("eigh", "one_device", mesh) == "one_device"
    with pytest.raises(ValueError, match="GATE one_device"):
        resolve_mod.resolve_backend("eigh", "one_device", mesh, n=bound + 2)
    with pytest.raises(ValueError, match="GATE one_device"):
        resolve_mod.resolve_backend("solve_lu", "one_device", mesh, n=9)
    status = distrib_la.list_backends("eigh", mesh)
    assert status["one_device"] == "available"


@pytest.mark.parametrize("max_n", [8, 64])          # 8: one matrix per rank per round
def test_one_device_eigh_matches_numpy(mesh, max_n):
    a = _hermitian(5, 8)                              # ragged against P = 4
    w, v = od.batched_distributed_eigh(_face(mesh, a), mesh=mesh, max_n=max_n)
    _eigh_ok(a, w, v)
    assert v.sharding.is_equivalent_to(NamedSharding(mesh, P(None, "x", "y")), 3)
    w1, v1 = od.distributed_eigh(_face(mesh, a[2]), mesh=mesh, max_n=max_n)
    _eigh_ok(a[2], w1, v1)
    assert v1.sharding.is_equivalent_to(NamedSharding(mesh, P("x", "y")), 2)


def test_one_device_rounds_rule():
    assert od.rounds_for(5, 8, 4, 8) == 2 and od.rounds_for(5, 8, 4, 64) == 1
    assert od.held_per_rank(64, 4096, 4, 8192) == 4 and od.held_per_rank(64, 16384, 4, 16384) == 1
    assert od.held_per_rank(3, 32, 4, 16384) == 1


def test_one_device_lu_matches_numpy_in_rounds(mesh):
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


def test_facade_records_and_runs_the_size_route(mesh, rocm):
    a = _hermitian(4, 8, seed=2)
    p = distrib_la.plan("eigh", mesh, backend="distributed", n=8, batched_route="auto")
    assert p.backend == "one_device" and not p.is_native
    assert "size route ROCM: n=8 <=" in p.describe()
    w, v = p.batched(_face(mesh, a))
    _eigh_ok(a, w, v)
    w1, v1 = p(_face(mesh, a[1]))
    _eigh_ok(a[1], w1, v1)
    rng = np.random.default_rng(7)
    A = rng.normal(size=(3, 8, 8)); B = rng.normal(size=(3, 8, 8))
    lu = distrib_la.plan("solve_lu", mesh, backend="distributed", n=8, batched_route="auto")
    assert lu.backend == "one_device"
    X = lu.batched(_face(mesh, A), _face(mesh, B))
    assert np.allclose(np.asarray(X), np.linalg.solve(A, B), atol=1e-11)


def test_slate_lu_px1_rows_come_back_in_order():
    """The SLATE LU relabel on a p x 1 mesh: SLATE writes X's row i at the slot
    JAX reads as global row r*(n/p) + c*nb + t, where i = (c*p + r)*nb + t;
    ``_rows_in_order`` must return X itself.  (Square meshes need no movement.)"""
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 host devices")
    from distrib_la import _slate
    mesh41 = Mesh(np.array(jax.devices()[:4]).reshape(4, 1), ("x", "y"))
    p, n, nb, nq, c = 4, 32, 2, 2, 3
    X = np.arange(nq * n * c, dtype=np.float64).reshape(nq, n, c)
    slot = np.empty_like(X)
    for i in range(n):
        blk, t = divmod(i, nb)
        cc, r = divmod(blk, p)
        slot[:, r * (n // p) + cc * nb + t] = X[:, i]
    spec = P(None, "x", "y")
    fn = shard_map(lambda x: _slate._rows_in_order(x, n=n, p=p, q=1, nb=nb), mesh=mesh41,
                   in_specs=spec, out_specs=spec, check_vma=False)
    got = jax.jit(fn)(jax.device_put(slot, NamedSharding(mesh41, spec)))
    assert np.array_equal(np.asarray(got), X)
    assert _slate.lu_layout(8192, 2, 2) == 256 and _slate.lu_layout(512, 2, 2) == 256
    assert _slate.lu_layout(4 * 1009, 2, 2) == 2018    # prime-ish edge: one tile per rank

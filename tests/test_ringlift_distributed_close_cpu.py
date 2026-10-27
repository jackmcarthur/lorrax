"""Right-Dyson closure: generic complex algebra, face storage and recovery."""
import importlib
import re

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update("jax_enable_x64", True)


def ringlift_fixture(n, logical=13):
    rng = np.random.default_rng(384)
    t = .025 * (rng.normal(size=(logical, logical)) +
                1j * rng.normal(size=(logical, logical)))
    x = rng.normal(size=t.shape) + 1j * rng.normal(size=t.shape)
    v = np.eye(logical) + .03 / max(1., np.sqrt(logical/13)) * (x + x.conj().T)
    y = np.linalg.solve((np.eye(logical) - t).T, t.T).T
    expected = v @ y
    # Both an accidental adjoint and a reordered Coulomb prefactor are wrong.
    assert np.max(np.abs(expected - y @ v)) > 1e-3
    assert np.max(np.abs(expected - v @ np.linalg.solve(
        (np.eye(logical) - t).conj().T, t.T).T)) > 1e-3
    pi = np.linalg.solve(v.T, t.T).T
    np.testing.assert_allclose(expected,
        np.linalg.solve(np.eye(logical) - v @ pi, v) - v, atol=1e-14, rtol=0)
    pad = ((0, n-logical), (0, n-logical))
    return np.pad(t, pad), np.pad(v, pad), np.pad(expected, pad)


@pytest.mark.parametrize("side", [1, 2, 3])
def test_right_dyson_faces_padding_singular_and_recovery(side, monkeypatch):
    if len(jax.devices()) < side*side:
        pytest.skip(f"needs {side*side} virtual devices")
    from bse import w_ladder as owner
    from lxkit import memory
    resolve = importlib.import_module("distrib_la.resolve")
    # Virtual CPUs have no installed ScaLAPACK; exercise the real mesh backend.
    monkeypatch.setitem(resolve._DISTRIBUTED_DEFAULT, ("solve_lu", "cpu"), "native2d")
    memory.set_budget(1 << 30)
    mesh = Mesh(np.asarray(jax.devices()[:side*side]).reshape(side, side), ("x", "y"))
    face = NamedSharding(mesh, P("x", "y"))
    n = -(-13 // side) * side
    t, v, expected = ringlift_fixture(n)
    put = lambda a: jax.device_put(a, face)
    result = owner.dyson_close_tile(put(t), put(v))
    assert result.sharding.is_equivalent_to(face, 2)
    np.testing.assert_allclose(np.asarray(result), expected, atol=2e-14, rtol=0)
    assert np.count_nonzero(np.asarray(result)[13:]) == 0
    assert np.count_nonzero(np.asarray(result)[:, 13:]) == 0
    count = len(owner._DYSON_CACHE)
    singular = np.pad(np.eye(13, dtype=np.complex128), ((0, n-13), (0, n-13)))
    with pytest.raises(ValueError, match="distrib_la_result_check"):
        owner.dyson_close_tile(put(singular), put(v))
    np.testing.assert_allclose(np.asarray(owner.dyson_close_tile(put(t), put(v))),
                               expected, atol=2e-14, rtol=0)
    assert len(owner._DYSON_CACHE) == count
    for key, (_, executable, price) in owner._DYSON_CACHE.items():
        if key[0] == t.shape and key[3] == face:
            assert price.bytes() >= executable.memory_analysis().temp_size_in_bytes


def test_closure_matches_full_resonant_antiresonant_resolvent(monkeypatch):
    from bse.w_ladder import dyson_close_tile
    from lxkit import memory
    resolve = importlib.import_module("distrib_la.resolve")
    monkeypatch.setitem(resolve._DISTRIBUTED_DEFAULT, ("solve_lu", "cpu"), "native2d")
    memory.set_budget(1 << 30)
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    face = NamedSharding(mesh, P("x", "y"))
    rng = np.random.default_rng(290)
    f = .1 * (rng.normal(size=(4, 3)) + 1j*rng.normal(size=(4, 3)))
    v = np.diag([.6, .8, 1.1])
    metric = np.diag([1., 1., -1., -1.])
    d = np.diag([1.2, 1.7, -1.2, -1.7])
    h = d + metric @ f @ v @ f.conj().T
    for z in (0j, .3j, .2+.7j):
        t = f.conj().T @ np.linalg.solve(z*np.eye(4)-d, metric @ f @ v)
        expected = v @ f.conj().T @ np.linalg.solve(z*np.eye(4)-h, metric @ f @ v)
        result = dyson_close_tile(jax.device_put(t, face), jax.device_put(v, face))
        np.testing.assert_allclose(np.asarray(result), expected, atol=3e-15, rtol=0)


def test_actual_room_routes_and_complete_program_quote(monkeypatch):
    if len(jax.devices()) < 4:
        pytest.skip("needs four virtual CPUs")
    from bse import w_ladder as owner
    from lxkit import memory
    resolve = importlib.import_module("distrib_la.resolve")
    local = importlib.import_module("distrib_la._one_device")
    # Substitute only the leg policy, preserving real Plans, moves and algebra.
    monkeypatch.setattr(resolve, "mesh_platform", lambda _: "ROCM")
    monkeypatch.setattr(local, "mesh_platform", lambda _: "ROCM")
    memory.set_budget(1 << 30, linalg="distributed")
    mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2), ("x", "y"))
    face = NamedSharding(mesh, P("x", "y"))
    t, v, expected = ringlift_fixture(512)
    operands = (jax.device_put(t, face), jax.device_put(v, face))
    room = [1 << 30]
    monkeypatch.setattr(memory, "room", lambda **_: room[0])
    selected, admit = [], memory.admit
    def observe(*args, **kwargs):
        chosen = admit(*args, **kwargs)
        selected.append(chosen)
        return chosen
    monkeypatch.setattr(memory, "admit", observe)
    for value in (1 << 30, 1, 0):
        room[0] = value
        result = owner.dyson_close_tile(*operands)
        np.testing.assert_allclose(np.asarray(result), expected, atol=2e-14, rtol=0)
    assert selected == ["one_device", "native2d", "native2d"]
    for key, (_, executable, price) in owner._DYSON_CACHE.items():
        if key[0] != (512, 512) or key[3] != face:
            continue
        ma = executable.memory_analysis()
        assert price.xla.at(4) == ma.temp_size_in_bytes + ma.output_size_in_bytes - ma.alias_size_in_bytes
        if key[-1][0] == "one_device":
            assert price.xla.spread == 0 and price.whole == 512


@pytest.mark.parametrize("side", [2, 3, 12])
def test_whole_close_abstract_program_has_only_faces_and_fixed_panels(side):
    if len(jax.devices()) < side*side:
        pytest.skip(f"needs {side*side} virtual CPUs")
    from bse.w_ladder import _dyson_close_program
    from distrib_la import plan
    mesh = Mesh(np.asarray(jax.devices()[:side*side]).reshape(side, side), ("x", "y"))
    face = NamedSharding(mesh, P("x", "y"))
    n, width = side*171, 128
    linear = plan("solve_lu", mesh, backend="native2d", n=n, batched_route="auto")
    arg = jax.ShapeDtypeStruct((n, n), np.complex128, sharding=face)
    exe = _dyson_close_program(mesh, linear, n).lower(arg, arg).compile()
    bound = max(171**2, 2*width*(171+171)+width**2, n*8)
    for dims in re.findall(r"c128\[([0-9,]+)\]", exe.as_text()):
        count = int(np.prod(tuple(map(int, dims.split(",")))))
        assert count <= bound, (side, dims, bound)
        assert count != n*n, "the complete closure must not replicate a matrix"

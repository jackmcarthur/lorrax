"""Host-only factory ownership/oracles; no JAX compilation or GPU claim."""
import gc
import weakref

import numpy as np
import pytest


@pytest.fixture
def programs(monkeypatch):
    import jax
    import jax.sharding
    import gw.shared_pole_screening as owner
    import gw.shared_pole_execution as execution
    import gw.shared_pole_pencil as pencil

    compiled = []

    def jit(fn, **kwargs):
        compiled.append((fn, kwargs))
        return fn  # Exercise the actual equations without compiling a device program.

    def mm(a, b, *, transa=None, transb=None):
        if transa == "C":
            a = a.conj().swapaxes(-1, -2)
        if transb == "C":
            b = b.conj().swapaxes(-1, -2)
        return a @ b

    owner._general_frequency_programs.cache_clear()
    monkeypatch.setattr(jax, "jit", jit)
    monkeypatch.setattr(jax.numpy, "conj", np.conj)
    monkeypatch.setattr(jax.numpy, "swapaxes", np.swapaxes)
    monkeypatch.setattr(jax.sharding, "NamedSharding", lambda mesh, spec: (mesh, spec))
    monkeypatch.setattr(execution, "face_matmul", lambda mesh: mm)
    monkeypatch.setattr(pencil, "jnp", np)
    monkeypatch.setattr(pencil, "tile_join", lambda arrays, axis, face: np.concatenate(arrays, axis))
    monkeypatch.setattr(pencil, "tile_block", lambda blocks, face: np.concatenate(
        [np.concatenate(row, axis=-1) for row in blocks], axis=-2))
    yield owner._general_frequency_programs, compiled
    owner._general_frequency_programs.cache_clear()


def test_factory_reuses_callables_and_binds_mesh_layout_and_dials(programs):
    factory, compiled = programs
    key = ("mesh-a", (None, "x", "y"), ("native",))
    first = factory(*key)
    assert factory(*key) is first
    assert len(compiled) == 4
    face = (key[0], key[1])
    assert [kw["out_shardings"] for _, kw in compiled] == [(face,)*4, (face,)*4, face, face]
    for other in (("mesh-b", key[1], key[2]),
                  (key[0], (None, "y", "x"), key[2]),
                  (key[0], key[1], ("changed-dial",))):
        assert factory(*other) is not first
    for i in range(12):
        factory(f"mesh-{i}", key[1], key[2])
    assert factory.cache_info().maxsize == 8
    assert factory.cache_info().currsize == 8


def test_current_pencil_and_infinity_operands_are_runtime_values(programs):
    factory, compiled = programs
    pencil, infinity, adjoint, normal = factory("mesh", (None, "x", "y"), ("native",))
    rng = np.random.default_rng(497)

    def random(shape):
        return rng.normal(size=shape)+1j*rng.normal(size=shape)

    previous = None
    refs = []
    for iteration in range(2):
        mu = np.array([1+2j, 3+4j])+iteration
        lam = np.array([2+3j, 4+5j, 6+7j])-iteration
        ql, il = random((2, 4, 2)), random((2, 2, 4))
        qr, wr = random((2, 4, 3)), random((2, 4, 3))
        refs.extend(map(weakref.ref, (mu, lam, ql, il, qr, wr)))
        left, right = il @ qr, ql.conj().swapaxes(-1, -2) @ wr
        denom = mu[:, None]-lam[None, :]
        expected = (-(left-right)/denom,
                    -(mu[:, None]*left-lam[None, :]*right)/denom, il, wr)
        actual = pencil(mu, lam, ql, il, qr, wr)
        for a, e in zip(actual, expected):
            np.testing.assert_array_equal(a, e)
        if previous is not None:
            assert not np.array_equal(actual[0], previous)
        previous = actual[0].copy()
        E, A, B, C = actual
        qi, c1, c2 = random((2, 4, 1)), random((2, 4, 4)), random((2, 4, 4))
        qih, qlh = qi.conj().swapaxes(-1, -2), ql.conj().swapaxes(-1, -2)
        ei, ef, ci, bi = qih @ wr, il @ qi, c1 @ qi, qih @ c1
        eii, aii = qih @ ci, qih @ (c2 @ qi)
        ai, af = ei*lam[None, None]-bi @ qr, mu[None, :, None]*ef-qlh @ ci
        expected_augmented = (
            np.concatenate((np.concatenate((E, ef), -1), np.concatenate((ei, eii), -1)), -2),
            np.concatenate((np.concatenate((A, af), -1), np.concatenate((ai, aii), -1)), -2),
            np.concatenate((B, bi), -2), np.concatenate((C, ci), -1))
        augmented = infinity(mu, lam, E, A, B, C, ql, il, qr, wr, qi, c1, c2)
        for a, e in zip(augmented, expected_augmented):
            np.testing.assert_array_equal(a, e)
        np.testing.assert_array_equal(adjoint(c1), c1.conj().swapaxes(-1, -2))
        np.testing.assert_array_equal(normal(c1), .5*(c1+c1.conj().swapaxes(-1, -2)))
    assert len(compiled) == 4  # No new factory when current operands change.
    del mu, lam, ql, il, qr, wr, actual, expected, E, A, B, C
    gc.collect()
    assert all(ref() is None for ref in refs)  # The program cache owns no current arrays.

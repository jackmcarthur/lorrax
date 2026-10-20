"""An eager route-(c) first attempt is admitted at the executable that runs: over the room,
twice the rounds (smaller slices), compiled only then (distrib_la.Plan._admitted)."""
import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

pytestmark = pytest.mark.skipif(jax.device_count() < 4, reason="needs 4 devices")


def test_over_the_room_doubles_the_rounds():
    import distrib_la
    import importlib
    pl = importlib.import_module("distrib_la.plan")
    from lxkit import memory
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))
    nb, n = 8, 64
    a = np.random.default_rng(0).standard_normal((nb, n, n)) * (1 + 1j)
    a = a + np.conj(np.swapaxes(a, 1, 2))
    A = jax.device_put(a, NamedSharding(mesh, P(None, "x", "y")))
    program = lambda phase, r=1: pl._reshard_stack_program("eigh", mesh, (nb, n, n), "complex128", int(r), phase)
    need = {r: pl._analysis_bytes(pl._executable(program("first", r), (A,))) for r in (1, nb)}
    assert need[nb] < need[1], need
    p = distrib_la.plan("eigh", mesh, n=n, backend="off")
    memory.reset(run=True)
    try:
        memory.set_budget(memory._in_use() + (need[1] + need[nb]) // 2)
        rounds, ex = p._admitted(program, A, 1)
        assert rounds > 1 and ex is pl._executable(program("first", rounds), (A,))
        (w, _), failed = ex(A)
        np.testing.assert_allclose(np.asarray(w), np.linalg.eigvalsh(a), atol=1e-9)
    finally:
        memory.reset(run=True)

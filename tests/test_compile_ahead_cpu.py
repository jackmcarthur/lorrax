"""``common.jax_compile_cache.compile_ahead`` on CPU (PARCOMP, 2026-10-10).

A program compiled ahead on a helper thread is the program the live call runs: one
XLA compile per module per process, whichever thread asked first, and the same
values as the eager call. P=1 here, so the agreement is a no-op and only the
counter, the slot reservation and the in-flight dedupe are exercised.
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update("jax_enable_x64", True)


def _compiles():
    from common.jax_compile_cache import compile_cache_stats
    return compile_cache_stats()["compiles"]


def test_compile_ahead_is_the_live_program():
    from common import jax_compile_cache as jcc
    jcc.install_compile_agreement()
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))
    face = NamedSharding(mesh, P("x", "y"))
    x = jax.device_put(np.arange(64, dtype=np.float64).reshape(8, 8) + 1.0, face)

    @jax.jit
    def f(a):
        return jnp.linalg.norm(a @ a.T, axis=1) + 0.5

    before = _compiles()
    futures = [jcc.compile_ahead(f, x) for _ in range(3)]     # the same module three times
    compiled = [fut.result() for fut in futures]
    assert _compiles() == before + 1, "a module in flight must not compile twice"
    live = f(x)
    assert _compiles() == before + 1, "the live call must reuse the ahead compile"
    reference = np.linalg.norm(np.asarray(x) @ np.asarray(x).T, axis=1) + 0.5
    np.testing.assert_array_equal(np.asarray(live), np.asarray(compiled[0](x)))
    np.testing.assert_allclose(np.asarray(live), reference, rtol=1e-12)
    assert 1 <= jcc.compile_threads() <= jcc.COMPILE_THREADS_MAX
    assert jcc.compile_cache_stats()["compile_threads"] == jcc.compile_threads()


def test_compile_ahead_runs_programs_beside_each_other():
    from common import jax_compile_cache as jcc
    jcc.install_compile_agreement()
    programs = [jax.jit(lambda a, k=k: jnp.sin(a * k).sum()) for k in range(1, 5)]
    x = jnp.asarray(np.linspace(0.0, 1.0, 1000))
    before = _compiles()                                      # eager ops above are compiles too
    futures = [jcc.compile_ahead(p, x) for p in programs]
    values = [float(fut.result()(x)) for fut in futures]
    assert _compiles() == before + 4
    assert values == [float(p(x)) for p in programs]          # the live calls hit, bitwise
    assert _compiles() == before + 4


def test_compile_ahead_surfaces_a_failing_compile():
    import pytest
    from common import jax_compile_cache as jcc
    jcc.install_compile_agreement()

    @jax.jit
    def g(a):
        return a + 1.0

    lowered_ok = jcc.compile_ahead(g, jnp.ones(3))
    assert lowered_ok.result() is not None
    with pytest.raises(TypeError):
        # Lowering runs on the calling thread: a bad argument refuses here, not in the pool.
        jcc.compile_ahead(g, object())


def test_live_call_during_the_ahead_compile_takes_no_number(monkeypatch):
    """The live jit call of a program whose helper has not finished waits for
    that compile and takes no request number: whether a rank's live call waits
    or finds the executable on the lowering depends on timing, so a number
    taken here would differ across ranks and the agreement would refuse."""
    import threading
    from common import jax_compile_cache as jcc
    jcc.install_compile_agreement()
    release = threading.Event()
    original = jcc._compile_in_slot

    def held(*args):
        release.wait(10.0)                 # the helper starts only after the live call
        return original(*args)

    monkeypatch.setattr(jcc, "_compile_in_slot", held)

    @jax.jit
    def h(a):
        return jnp.cos(a) @ jnp.sin(a).T + 2.0

    x = jnp.asarray(np.linspace(0.0, 1.0, 36).reshape(6, 6))
    before, numbers = _compiles(), jcc._STATE._compile_sequence
    future = jcc.compile_ahead(h, x)
    timer = threading.Timer(0.3, release.set)
    timer.start()
    live = h(x)                            # reaches the request hook while the helper waits
    timer.join()
    future.result()
    assert jcc._STATE._compile_sequence == numbers + 1, "the live call took a number"
    assert _compiles() == before + 1, "the live call compiled the program again"
    np.testing.assert_array_equal(np.asarray(live), np.asarray(future.result()(x)))


def test_repeated_ahead_calls_take_one_compile_and_one_number_each():
    from common import jax_compile_cache as jcc
    jcc.install_compile_agreement()

    @jax.jit
    def k(a):
        return jnp.tanh(a).sum(axis=0)

    x = jnp.asarray(np.linspace(-1.0, 1.0, 64).reshape(8, 8))
    before, numbers = _compiles(), jcc._STATE._compile_sequence
    futures = [jcc.compile_ahead(k, x) for _ in range(3)]
    [f.result() for f in futures]
    np.asarray(k(x))
    assert _compiles() == before + 1
    # One number per ahead call, finished or not, the same on every rank.
    assert jcc._STATE._compile_sequence == numbers + 3

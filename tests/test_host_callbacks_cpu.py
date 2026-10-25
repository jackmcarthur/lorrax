"""Host callbacks hand their host functions NumPy arrays, never jax.Arrays.

JAX hands a host callback ``jax.Array`` arguments.  A slice or reshape of one is
a new computation dispatched to the CPU device that is running the caller's
loop; when the CPU client's pool has no thread free, the two wait on each other
for ever.  A Si 4^3 singlet BSE Davidson sat for 39 min with every thread in a
futex (fixed in 4aba67e0d); with NumPy views it converged in 6 iterations, 3 min.
Every callback now goes through ``lxkit.host_*_callback`` (the lint
``tools/lint/test_host_callbacks.py`` refuses a raw one).
"""
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from distrib_la import subspace  # noqa: E402
from lxkit import host_debug_callback, host_io_callback, host_pure_callback  # noqa: E402


def test_callbacks_receive_numpy():
    seen = []

    def work(x, n):
        seen.append((type(x), type(n)))
        return x[:2].reshape(2) * int(n)

    spec = jax.ShapeDtypeStruct((2,), jnp.float32)

    @jax.jit
    def run(x):
        n = jnp.int32(3)
        host_debug_callback(lambda x, n: work(x, n), x, n)
        return host_pure_callback(work, spec, x, n) + host_io_callback(work, spec, x, n)

    out = jax.block_until_ready(run(jnp.arange(4.0, dtype=jnp.float32)))
    assert np.allclose(out, [0.0, 6.0])
    assert len(seen) == 3 and all(t is np.ndarray for pair in seen for t in pair)


def test_cpu_provider_project_matches_numpy():
    jax.config.update("jax_enable_x64", True)
    cap = 6
    plan = subspace.CpuSubspacePlan(capacity=cap, n_eig=2)
    rng = np.random.default_rng(1)
    v = rng.normal(size=(cap, 3, 4)) + 1j * rng.normal(size=(cap, 3, 4))

    def run(v, h):
        return jax.lax.fori_loop(0, cap, lambda i, h: plan.project(
            v, v, jnp.int32(cap), h, i, jnp.int32(1)), h)

    h = jax.jit(run)(jnp.asarray(v), jnp.zeros((cap, cap), jnp.complex128))
    flat = v.reshape(cap, -1)
    np.testing.assert_allclose(np.asarray(h), flat.conj() @ flat.T, atol=1e-12)


# Four CPU devices on two cores starve the pool.  A raw ``jax.pure_callback``
# whose host function slices its argument never returns here when it is the
# process's first program (2.5e5..4e6 elements per device, 3/3 runs); through
# the helper it takes under a second.  The pivoted-Cholesky progress report runs
# after it, on the same starved pool.
_STARVED = textwrap.dedent("""
    import os
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax import shard_map
    from jax.sharding import Mesh, PartitionSpec as P
    from lxkit import host_pure_callback
    from common.pivoted_cholesky import make_sharded_group_block_pivoted_cholesky_select

    jax.config.update("jax_enable_x64", True)
    mesh = Mesh(np.array(jax.devices()), ("x",))
    n_dev = mesh.devices.size

    print("stage: slicing host callback in a sharded loop", flush=True)
    def host(x):
        n = (x.shape[0] // 2) * 2
        return np.asarray((x[:n].reshape(2, -1) @ np.ones(n // 2)).sum(), np.float64)
    spec = jax.ShapeDtypeStruct((), jnp.float64)
    loop = lambda xs: jax.lax.fori_loop(0, 3, lambda i, c: c + host_pure_callback(host, spec, c), xs)
    run = jax.jit(lambda x: shard_map(loop, mesh=mesh, in_specs=P("x"), out_specs=P("x"))(x))
    jax.block_until_ready(run(jnp.ones(250_000 * n_dev)))

    print("stage: pivoted-Cholesky progress", flush=True)
    M, n_groups = 16 * n_dev, 4
    A = np.random.default_rng(0).standard_normal((M, 2 * M))
    calls = []
    step = make_sharded_group_block_pivoted_cholesky_select(
        mesh, M, M, n_groups, progress_fn=lambda *a: calls.append(a))
    jax.block_until_ready(step(jnp.asarray(A @ A.T),
                               jnp.asarray(np.arange(M) % n_groups, dtype=jnp.int32)))
    assert calls and all(type(v) is int for a in calls for v in a), calls
    print("stage: done", flush=True)
""")


def test_host_callbacks_complete_on_a_starved_cpu_pool():
    env = dict(os.environ, JAX_PLATFORMS="cpu",
               XLA_FLAGS="--xla_force_host_platform_device_count=4")
    try:
        r = subprocess.run([sys.executable, "-c", _STARVED], env=env, timeout=120,
                           capture_output=True, text=True)
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stage = ([line for line in out.splitlines() if line.startswith("stage:")] or ["start"])[-1]
        raise AssertionError(f"host callback hung at {stage!r} (a host function "
                             "dispatched to the device running its loop)") from None
    assert r.returncode == 0, r.stderr[-2000:]
    assert "stage: done" in r.stdout

"""The CPU subspace provider's callbacks work on NumPy arrays, never jax.Arrays.

JAX hands a ``pure_callback`` ``jax.Array`` arguments.  The CPU provider's
callbacks slice and reshape them; on a jax.Array each of those is a new
computation dispatched to the CPU device that is running the Davidson while
loop, and the two waited on each other: a Si 4^3 singlet BSE Davidson sat for
39 min with every thread in a futex (Lanczos: 10 min).  With NumPy views it
converged in 6 iterations, 3 min.
"""
import ast
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from distrib_la import subspace  # noqa: E402

SOURCE = Path(subspace.__file__)


def test_callbacks_receive_numpy():
    seen = []

    def work(x, n):
        seen.append((type(x), type(n)))
        return x[:2].reshape(2) * int(n)

    out = jax.jit(lambda x: subspace._host_callback(
        work, jax.ShapeDtypeStruct((2,), x.dtype), x, jnp.int32(3)))(jnp.arange(4.0))
    assert np.allclose(out, [0.0, 3.0])
    assert seen and all(t is np.ndarray for pair in seen for t in pair)


def test_cpu_provider_has_no_raw_pure_callback():
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CpuSubspacePlan")
    raw = [n.lineno for n in ast.walk(cls) if isinstance(n, ast.Attribute)
           and n.attr == "pure_callback"]
    assert not raw, f"CpuSubspacePlan calls jax.pure_callback directly at lines {raw}"


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

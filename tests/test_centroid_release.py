"""Run directly under MPI to check centroid working-set/cache lifetimes."""


def main():
    from centroid.kmeans_cli import _release_lloyd_before_prune
    import jax
    import jax.numpy as jnp
    import numpy as np
    import runtime

    traces = []

    @jax.jit
    def shared_backend(x):
        traces.append(None)
        return jnp.sin(x) + x * x

    x = jnp.arange(17, dtype=jnp.float64) / 17
    expected = np.sin(np.arange(17) / 17) + (np.arange(17) / 17) ** 2
    np.testing.assert_allclose(shared_backend(x), expected, rtol=1e-14)
    owned = jnp.arange(43, dtype=jnp.float64)
    _release_lloyd_before_prune(owned, owned, None, np.ones(3))
    assert owned.is_deleted(), "completed preparation buffer remains live"
    assert not x.is_deleted(), "release invalidated an unrelated live buffer"
    np.testing.assert_allclose(shared_backend(x), expected, rtol=1e-14)
    assert len(traces) == 1, "preparation release invalidated a shared program"
    print("centroid buffer release preserves shared compiled programs: PASS", flush=True)
    runtime.finalize_process(0)


def test_requires_distributed_launcher():
    import pytest
    pytest.skip("run this module directly with the configured MPI/GPU launcher")


if __name__ == "__main__":
    main()

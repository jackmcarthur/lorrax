"""Runtime-prefix native algebra must ignore poisoned inactive storage."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from distrib_la import plan_local_subspace


@pytest.mark.parametrize('active', [12, 17, 35, 64])
def test_active_prefix_ignores_nan_tail(active):
    if jax.default_backend() != 'gpu':
        pytest.skip('active subspace requires CUDA')
    rng = np.random.default_rng(78)
    cap, d, b = 64, 128, 12
    z = rng.normal(size=(d, cap))+1j*rng.normal(size=(d, cap))
    v = np.linalg.qr(z)[0].T.copy()
    hv = v*np.arange(d)[None, :]
    c = rng.normal(size=(cap, b))+1j*rng.normal(size=(cap, b))
    p = rng.normal(size=(b, d))+1j*rng.normal(size=(b, d))
    vv, hh, cc = v.copy(), hv.copy(), c.copy()
    vv[active:], hh[active:], cc[active:] = np.nan, np.nan, np.nan
    plan = plan_local_subspace(capacity=cap, n_eig=b)
    h = jnp.full((cap, cap), jnp.nan, dtype=jnp.complex128)
    projected = jax.jit(plan.project)(jnp.asarray(vv), jnp.asarray(hh), active, h, 0, active)
    reference = v[:active].conj()@hv[:active].T
    np.testing.assert_allclose(np.asarray(projected)[:active, :active], reference, atol=2e-12)
    e, coeff = jax.jit(plan.eigh)(projected, active)
    np.testing.assert_allclose(e, np.linalg.eigvalsh(reference)[:b], atol=2e-12)
    np.testing.assert_allclose(reference@np.asarray(coeff)[:active], np.asarray(coeff)[:active]*e, atol=2e-11)
    xx, hxx = jax.jit(plan.reconstruct)(jnp.asarray(vv), jnp.asarray(hh), jnp.asarray(cc), active, jnp.asarray(p))
    np.testing.assert_allclose(xx, c[:active].T@v[:active], atol=2e-12)
    np.testing.assert_allclose(hxx, c[:active].T@hv[:active], atol=2e-11)
    pp = jax.jit(plan.orthogonalize)(jnp.asarray(vv), jnp.asarray(p), active)
    ref = p.copy()
    for _ in range(2):
        ref -= (v[:active].conj()@ref.T).T@v[:active]
    np.testing.assert_allclose(pp, ref, atol=2e-12)


def test_normalization_ignores_nan_tail():
    if jax.default_backend() != 'gpu':
        pytest.skip('active subspace requires CUDA')
    plan = plan_local_subspace(capacity=12, n_eig=12)
    p = jnp.eye(32, dtype=jnp.complex128)[:12]
    p = p.at[1].multiply(2.).at[2:].set(jnp.nan)
    result = jax.jit(plan.normalize)(p, jnp.int32(2))
    assert np.isfinite(np.asarray(result)).all()
    np.testing.assert_allclose(np.asarray(result)[:2].conj()@np.asarray(result)[:2].T, np.eye(2), atol=1e-14)
    np.testing.assert_array_equal(np.asarray(result)[2:], 0)


@pytest.mark.parametrize('count', [0, 2])
def test_active_store_preserves_other_rows_and_input_values(count):
    if jax.default_backend() != 'gpu':
        pytest.skip('active subspace requires CUDA')
    plan = plan_local_subspace(capacity=12, n_eig=4)
    original = np.arange(96).reshape(12, 8).astype(np.complex128)
    original[10:] = np.nan
    v, hv = jnp.asarray(original), jnp.asarray(2*original)
    p = jnp.full((4, 8), 3+2j, dtype=jnp.complex128)
    vv, hh = jax.jit(plan.store)(v, hv, p, 2*p, 5, count)
    reference = original.copy()
    reference[5:5+count] = np.asarray(p)[:count]
    np.testing.assert_allclose(vv, reference, equal_nan=True)
    np.testing.assert_allclose(hh, 2*reference, equal_nan=True)
    # Aliases are internal to XLA: an undonated public input stays immutable.
    np.testing.assert_allclose(v, original, equal_nan=True)
    np.testing.assert_allclose(hv, 2*original, equal_nan=True)

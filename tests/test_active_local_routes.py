"""Active-range local GEMM: the masked route equals the interval route and a dense reference.

``distrib_la._active_local`` serves two XLA routes for the product
``a[q][:, lo:hi] diag(w[q]) b[q][lo:hi, :]`` (``ACTIVE_LOCAL_ROUTE``): the
interval route (power-of-two slices under device control flow) and the masked
route (one batched dot with the inactive storage selected to zero).  Both are
the references the native routes (ROCM ``device``, CUDA ``native``) are gated against.  Both must ignore whatever the inactive storage holds, NaN included.
"""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from distrib_la._active_local import (ACTIVE_LOCAL_ROUTE, active_local_matmul,
                                      active_local_route, local_active_impl,
                                      masked_local_matmul)

NQ, M, K, N = 5, 7, 37, 6
ONE, ZERO = np.complex128(1.0), np.complex128(0.0)


def _operands(rng, dtype=np.complex128):
    def draw(*shape):
        x = rng.standard_normal(shape)
        if np.dtype(dtype).kind == "c":
            x = x + 1j * rng.standard_normal(shape)
        return x.astype(dtype)
    return draw(NQ, M, K), draw(NQ, K, N), draw(NQ, K), draw(NQ, M, N)


def _reference(a, b, w, bounds, c, alpha, beta):
    out = np.zeros((NQ, M, N), a.dtype)
    for q, (lo, hi) in enumerate(bounds):
        out[q] = (a[q][:, lo:hi] * w[q][lo:hi]) @ b[q][lo:hi]
    out = alpha * out
    return out + beta * c if beta != 0 else out


def _poison(a, b, w, bounds):
    """NaN in every inactive column of a, row of b, entry of w."""
    a, b, w = a.copy(), b.copy(), w.copy()
    for q, (lo, hi) in enumerate(bounds):
        for x, axis in ((a, 1), (w, 0)):
            sl = [slice(None)] * (x.ndim - 1)
            sl[axis] = np.r_[0:lo, hi:K]
            x[q][tuple(sl)] = np.nan
        b[q][np.r_[0:lo, hi:K]] = np.nan
    return a, b, w


BOUNDS = {
    "mixed": np.array([[0, 37], [3, 20], [11, 12], [5, 5], [16, 37]], np.int32),
    "equal": np.array([[4, 29]] * NQ, np.int32),
    "full": np.array([[0, K]] * NQ, np.int32),
    "empty": np.array([[9, 9]] * NQ, np.int32),
}


@pytest.mark.parametrize("name", sorted(BOUNDS))
@pytest.mark.parametrize("beta", [ZERO, np.complex128(0.5 - 0.25j)])
def test_masked_equals_interval_and_reference(name, beta):
    rng = np.random.default_rng(7)
    a, b, w, c = _operands(rng)
    bounds = BOUNDS[name]
    a, b, w = _poison(a, b, w, bounds)
    alpha = np.complex128(1.25 + 0.5j)
    a0, b0, w0, _ = _operands(np.random.default_rng(7))
    ref = _reference(a0, b0, w0, bounds, c, alpha, beta)
    args = (jnp.asarray(a), jnp.asarray(b), jnp.asarray(bounds), jnp.asarray(w), jnp.asarray(c))
    got_m = np.asarray(jax.jit(lambda *x: masked_local_matmul(*x, alpha=alpha, beta=beta))(*args))
    got_i = np.asarray(jax.jit(lambda *x: active_local_matmul(*x, alpha=alpha, beta=beta))(*args))
    scale = max(np.abs(ref).max(), 1.0)
    assert np.all(np.isfinite(got_m))
    assert np.abs(got_m - ref).max() / scale < 1e-13
    assert np.abs(got_m - got_i).max() / scale < 1e-13


def test_real_operands():
    rng = np.random.default_rng(3)
    a, b, w, c = _operands(rng, np.float64)
    bounds = BOUNDS["mixed"]
    ref = _reference(a, b, w, bounds, c, 1.0, 0.0)
    got = masked_local_matmul(jnp.asarray(a), jnp.asarray(b), jnp.asarray(bounds), jnp.asarray(w),
                              alpha=ONE, beta=ZERO)
    assert got.dtype == jnp.float64
    assert np.abs(np.asarray(got) - ref).max() < 1e-13 * max(np.abs(ref).max(), 1.0)


@pytest.mark.parametrize("bad", [[[-1, 4]], [[6, 3]], [[0, K + 1]]])
def test_invalid_bounds_give_nan(bad):
    rng = np.random.default_rng(1)
    a, b, w, _ = _operands(rng)
    bounds = np.array(bad * NQ, np.int32)
    got = masked_local_matmul(jnp.asarray(a), jnp.asarray(b), jnp.asarray(bounds), jnp.asarray(w),
                              alpha=ONE, beta=ZERO)
    assert np.all(np.isnan(np.asarray(got)))


def test_route_rows():
    assert ACTIVE_LOCAL_ROUTE == {"cpu": "interval", "CUDA": "native", "ROCM": "device"}
    assert active_local_route("tpu") == "interval"
    assert local_active_impl("cpu") is active_local_matmul


def test_masked_has_no_device_control_flow():
    """The masked program lowers with no while/cond (the interval route's do)."""
    rng = np.random.default_rng(2)
    a, b, w, _ = _operands(rng)
    args = (jnp.asarray(a), jnp.asarray(b), jnp.asarray(BOUNDS["mixed"]), jnp.asarray(w))
    text = jax.jit(lambda *x: masked_local_matmul(*x, alpha=ONE, beta=ZERO)).lower(*args).as_text()
    assert "stablehlo.while" not in text and "stablehlo.case" not in text
    text_i = jax.jit(lambda *x: active_local_matmul(*x, alpha=ONE, beta=ZERO)).lower(*args).as_text()
    assert "stablehlo.while" in text_i

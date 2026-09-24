"""build_G_tau with a prepared (immutable-interval) GEMM, the chi0 route's form."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gw.greens_function_kernel import build_G_tau


class _DenseGemm:
    """Small test double implementing the public dense and active GEMM seams."""

    def __call__(self, a, b):
        return a @ b

    def active_range(self, a, b, lo, hi, *, weights):
        index = jnp.arange(a.shape[-1])
        active = (index >= lo[:, None]) & (index < hi[:, None])
        return (a * jnp.where(active, weights, 0)[:, None, :]) @ b


def _prepared_dense(lo, hi):
    lo = jnp.asarray(lo, dtype=jnp.int32)
    hi = jnp.asarray(hi, dtype=jnp.int32)

    def call(a, b, *, weights):
        index = jnp.arange(a.shape[-1])
        active = (index >= lo[:, None]) & (index < hi[:, None])
        return (a * jnp.where(active, weights, 0)[:, None, :]) @ b

    return call


def test_build_g_tau_prepared_callable_matches_dynamic_interval():
    rng = np.random.default_rng(91)
    nq, ns, mu, nb = 2, 1, 3, 6
    left = jnp.asarray(
        rng.standard_normal((nq, ns, mu, nb))
        + 1j * rng.standard_normal((nq, ns, mu, nb)))
    right = jnp.transpose(left, (0, 3, 1, 2))
    energy = jnp.asarray([
        [10.0, -1.0, 0.5, 3.0, -4.0, 9.0],
        [2.0, 8.0, -2.0, 0.0, 5.0, 1.0],
    ])
    weight = jnp.asarray([
        [0.0, -0.2, 1.0, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.3, -0.7, 0.0, 0.0],
    ])
    lo = jnp.asarray([1, 2], dtype=jnp.int32)
    hi = jnp.asarray([5, 4], dtype=jnp.int32)
    gemm = _DenseGemm()

    dynamic = build_G_tau(
        left, right, energy, 0.2 + 0.1j, gemm=gemm,
        band_weight=weight, trim_zero_bands=True)
    prepared = build_G_tau(
        left, right, energy, 0.2 + 0.1j, gemm=gemm,
        band_weight=weight, trim_zero_bands=True,
        prepared_active_gemm=_prepared_dense(lo, hi))

    np.testing.assert_array_equal(np.asarray(prepared), np.asarray(dynamic))


def test_prepared_route_removes_runtime_band_reductions():
    nq, ns, mu, nb = 1, 1, 2, 4
    left = jnp.ones((nq, ns, mu, nb), dtype=jnp.complex128)
    right = jnp.ones((nq, nb, ns, mu), dtype=jnp.complex128)
    energy = jnp.arange(nb, dtype=jnp.float64)[None]
    selector = jnp.asarray([[False, True, True, False]])
    gemm = _DenseGemm()
    prepared = _prepared_dense([1], [3])

    traced = jax.make_jaxpr(lambda t: build_G_tau(
        left, right, energy, t, gemm=gemm, mask=selector,
        trim_zero_bands=True, prepared_active_gemm=prepared))(
            jnp.asarray(0.1))
    text = str(traced)

    assert "reduce_min" not in text
    assert "reduce_max" not in text


def test_prepared_callable_cannot_be_combined_with_dynamic_bounds():
    gemm = _DenseGemm()
    left = jnp.ones((1, 1, 1, 2), dtype=jnp.complex128)
    right = jnp.ones((1, 2, 1, 1), dtype=jnp.complex128)
    energy = jnp.zeros((1, 2))

    with pytest.raises(ValueError, match="mutually exclusive"):
        build_G_tau(
            left, right, energy, 0.0, gemm=gemm,
            mask=jnp.ones_like(energy, dtype=jnp.bool_),
            band_range=(0, 2), prepared_active_gemm=_prepared_dense([0], [2]))

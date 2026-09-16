"""Correctness gates for preparing one active band interval per Sigma window."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gw.greens_function_kernel import (
    build_G_tau,
    prepare_tau_band_range,
    windowed_exp_iEt,
)


def _reference_intervals(energy, selector, e_ref, evolution_times):
    intervals = []
    for evolution_time in np.asarray(evolution_times):
        phase = np.asarray(windowed_exp_iEt(
            jnp.asarray(energy), jnp.asarray(evolution_time), e_ref=e_ref))
        if np.asarray(selector).dtype == np.bool_:
            phase = np.where(selector, phase, 0.0)
        else:
            phase = np.where(selector != 0, phase * selector, 0.0)
        parents = []
        for live in phase != 0:
            index = np.flatnonzero(live)
            parents.append((0, 0) if not index.size
                           else (int(index[0]), int(index[-1]) + 1))
        intervals.append(tuple(parents))
    return tuple(intervals)


def test_invariant_intervals_allow_time_dependent_interior_underflow():
    # The large positive energy disappears at evolution_time=1.  It is an
    # interior hole, so the smallest enclosing interval remains unchanged.
    energy = jnp.asarray([
        [0.0, 2.0, 900.0, -3.0, 0.0],
        [4.0, 0.0, -2.0, 0.0, 5.0],
    ])
    selector = jnp.asarray([
        [True, False, True, False, True],
        [False, True, False, True, False],
    ])
    times = jnp.asarray([0.0, 1.0, jnp.nan])
    fn = jax.jit(prepare_tau_band_range)

    lo, hi, invariant = fn(energy, selector, 0.0, times, 2)

    np.testing.assert_array_equal(np.asarray(lo), [0, 1])
    np.testing.assert_array_equal(np.asarray(hi), [5, 4])
    assert bool(invariant)
    reference = _reference_intervals(energy, selector, 0.0, times[:2])
    assert reference == (((0, 5), (1, 4)), ((0, 5), (1, 4)))


def test_edge_underflow_requires_dynamic_fallback():
    energy = jnp.asarray([[900.0, 0.0, -1.0]])
    selector = jnp.asarray([[True, False, True]])
    times = jnp.asarray([0.0, 1.0])

    lo, hi, invariant = jax.jit(prepare_tau_band_range)(
        energy, selector, 0.0, times, 2)

    np.testing.assert_array_equal(np.asarray(lo), [0])
    np.testing.assert_array_equal(np.asarray(hi), [3])
    assert not bool(invariant)
    assert _reference_intervals(energy, selector, 0.0, times) == (
        ((0, 3),), ((2, 3),))


def test_signed_and_complex_weights_use_exact_nonzero_support():
    energy = jnp.asarray([[3.0, -2.0, 8.0, 1.0, -4.0]])
    selector = jnp.asarray(
        [[0.0, -0.25, 0.0, 0.0 + 0.5j, 0.0]], dtype=jnp.complex128)
    times = jnp.asarray([0.0 + 0.0j, 0.0 + 0.7j], dtype=jnp.complex128)

    lo, hi, invariant = jax.jit(prepare_tau_band_range)(
        energy, selector, 1.5, times, 2)

    np.testing.assert_array_equal(np.asarray(lo), [1])
    np.testing.assert_array_equal(np.asarray(hi), [4])
    assert bool(invariant)


@pytest.mark.parametrize("selector", [
    jnp.zeros((2, 4), dtype=jnp.bool_),
    jnp.zeros((2, 4), dtype=jnp.float64),
])
def test_all_empty_parents_have_the_canonical_empty_interval(selector):
    energy = jnp.arange(8, dtype=jnp.float64).reshape(2, 4)
    times = jnp.asarray([0.0, 0.5])

    lo, hi, invariant = prepare_tau_band_range(
        energy, selector, 0.0, times, 2)

    np.testing.assert_array_equal(np.asarray(lo), [0, 0])
    np.testing.assert_array_equal(np.asarray(hi), [0, 0])
    assert bool(invariant)


@pytest.mark.parametrize("n_active", [0, -1, 4])
def test_empty_or_invalid_active_count_refuses_preparation(n_active):
    energy = jnp.zeros((1, 2))
    selector = jnp.ones((1, 2), dtype=jnp.bool_)
    times = jnp.asarray([0.0, 1.0, 2.0])

    _lo, _hi, invariant = jax.jit(prepare_tau_band_range)(
        energy, selector, 0.0, times, n_active)

    assert not bool(invariant)


def test_zero_capacity_time_storage_returns_empty_invalid_bounds():
    energy = jnp.zeros((2, 3))
    selector = jnp.ones_like(energy, dtype=jnp.bool_)

    lo, hi, invariant = prepare_tau_band_range(
        energy, selector, 0.0,
        jnp.empty((0,), dtype=jnp.complex128), jnp.asarray(0, jnp.int32))

    np.testing.assert_array_equal(np.asarray(lo), [0, 0])
    np.testing.assert_array_equal(np.asarray(hi), [0, 0])
    assert not bool(invariant)


def test_one_compiled_shape_accepts_different_active_prefix_lengths():
    energy = jnp.asarray([[0.0, 900.0, 0.0]])
    selector = jnp.ones_like(energy, dtype=jnp.bool_)
    times = jnp.asarray([0.0, 1.0, jnp.nan, jnp.nan])
    # Use a fresh outer wrapper so compilations performed by earlier tests do
    # not enter this cache count.
    compiled = jax.jit(
        lambda e, s, r, t, n: prepare_tau_band_range(e, s, r, t, n))

    first = compiled(energy, selector, 0.0, times, 1)
    after_first = compiled._cache_size()
    second = compiled(energy, selector, 0.0, times, 2)

    assert bool(first[2])
    assert bool(second[2])  # Only an interior band changes support.
    assert compiled._cache_size() == after_first


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


@pytest.mark.parametrize("bad_selector", [
    jnp.ones((1, 2), dtype=jnp.bool_),
    jnp.ones((2, 3), dtype=jnp.bool_),
])
def test_preparation_refuses_shape_mismatch(bad_selector):
    with pytest.raises(ValueError, match="selector must match"):
        prepare_tau_band_range(
            jnp.ones((2, 2)), bad_selector, 0.0, jnp.ones((2,)), 2)

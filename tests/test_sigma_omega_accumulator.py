"""Semantics for compact omega folds in the Sigma accumulator.

The ordinary suite exercises the same kernels on one CPU device. A P4 run
uses a real 2x2 band mesh, so the anti-Hermitian checks cross the ``x``/``y``
band shards. Performance measurements live in Run394 rather than this file.
"""
from __future__ import annotations

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.collectives import device_put_process_local
import gw.ppm_accumulators as ppm_accumulators
from gw.ppm_accumulators import DeviceOmegaAccumulator


def _mesh() -> Mesh:
    ndev = jax.device_count()
    if ndev == 4:
        devices = np.asarray(jax.devices()).reshape(2, 2)
    elif ndev == 1:
        devices = np.asarray(jax.devices()).reshape(1, 1)
    else:
        raise RuntimeError(
            f"omega-accumulator gate expects exactly 1 or 4 devices, got {ndev}")
    return Mesh(devices, ("x", "y"))


def _to_host(value) -> np.ndarray:
    if jax.process_count() > 1:
        from jax.experimental import multihost_utils as mhu
        return np.asarray(mhu.process_allgather(value, tiled=True))
    return np.asarray(value)


def _put(value, sharding):
    return device_put_process_local(np.asarray(value), sharding)


def _window_reference(*, omega, times, alpha, base, slope, omega_sign,
                      prefactor, e_ref_sum, indices=None, omega_values=None,
                      omega_axis=0, antihermitian=False):
    if indices is None:
        indices = np.arange(len(omega), dtype=np.int64)
        omega_values = np.asarray(omega, np.complex128)
    else:
        indices = np.asarray(indices, np.int64)
        omega_values = np.asarray(omega_values, np.complex128)

    compact_shape = list(base.shape)
    compact_shape.insert(omega_axis, len(indices))
    compact = np.zeros(compact_shape, np.complex128)
    for t, a in zip(times, alpha):
        sigma = base + t * slope
        coeff = ((prefactor * a)
                 * np.exp(-1j * (e_ref_sum - omega_sign * omega_values) * t))
        coeff_shape = ((1,) * omega_axis + (len(indices),)
                       + (1,) * (compact.ndim - omega_axis - 1))
        compact = compact + coeff.reshape(coeff_shape) * np.expand_dims(
            sigma, axis=omega_axis)
    if antihermitian:
        compact = (compact - np.conj(np.swapaxes(compact, -1, -2))) / 2j

    result_shape = list(base.shape)
    result_shape.insert(omega_axis, len(omega))
    result = np.zeros(result_shape, np.complex128)
    where = (slice(None),) * omega_axis + (indices,)
    result[where] = compact
    return result


def _run_affine_window(mesh, *, omega_axis=0, indices=None,
                       omega_values=None, antihermitian=False,
                       omega_sign=1.0, precompile=False):
    rng = np.random.default_rng(20260913 + omega_axis)
    omega = np.asarray([-1.1, -0.55, -0.1, 0.0, 0.23, 0.8, 1.4])
    times = np.asarray([
        0.07 + 0.02j, 0.19 - 0.03j, 0.31 + 0.01j,
        0.48 - 0.04j, 0.72 + 0.02j, 0.91 - 0.01j, 1.13 + 0.03j,
    ], np.complex128)
    alpha = np.asarray([
        0.4 - 0.2j, -0.7 + 0.1j, 0.15 + 0.33j,
        0.8 - 0.05j, -0.21 - 0.17j, 0.06 + 0.11j, -0.3 + 0.09j,
    ], np.complex128)
    leading = (2,) if omega_axis == 1 else ()
    sigma_shape = (*leading, 2, 4, 4)
    base = (rng.standard_normal(sigma_shape)
            + 1j * rng.standard_normal(sigma_shape))
    slope = (rng.standard_normal(sigma_shape)
             + 1j * rng.standard_normal(sigma_shape)) * 0.2
    sigma_spec = P(None, None, "x", "y") if omega_axis == 1 else P(None, "x", "y")
    output_spec = (P(None, None, None, "x", "y")
                   if omega_axis == 1 else P(None, None, "x", "y"))
    sigma_sharding = NamedSharding(mesh, sigma_spec)
    output_sharding = NamedSharding(mesh, output_spec)
    shape = list(sigma_shape)
    shape.insert(omega_axis, omega.size)
    acc = DeviceOmegaAccumulator(
        omega, shape=tuple(shape), sharding=output_sharding,
        omega_axis=omega_axis)
    kwargs = {}
    if indices is not None:
        kwargs = {"omega_indices": np.asarray(indices),
                  "omega_values": np.asarray(omega_values)}
    acc.begin_window(
        times, alpha, omega_sign=omega_sign, prefactor=-0.63,
        e_ref_sum=0.27, antihermitian=antihermitian, **kwargs)
    if precompile:
        assert acc.precompile_tau_add(
            sigma_shape=sigma_shape, sigma_sharding=sigma_sharding) is True
        assert acc.precompile_tau_add(
            sigma_shape=sigma_shape, sigma_sharding=sigma_sharding) is False
    for t in times:
        acc.add_tau(_put(base + t * slope, sigma_sharding))
    acc.end_window()
    got = _to_host(jax.block_until_ready(acc.finalize()))
    want = _window_reference(
        omega=omega, times=times, alpha=alpha, base=base, slope=slope,
        omega_sign=omega_sign, prefactor=-0.63, e_ref_sum=0.27,
        indices=indices, omega_values=omega_values, omega_axis=omega_axis,
        antihermitian=antihermitian)
    return got, want


def test_noncontiguous_active_frequencies_match_independent_numpy():
    indices = np.asarray([5, 1, 4])
    got, want = _run_affine_window(
        _mesh(), indices=indices, omega_values=np.asarray([0.83, 0.12, 0.61]))
    np.testing.assert_allclose(got, want, rtol=3e-14, atol=3e-14)
    np.testing.assert_array_equal(
        got[np.setdiff1d(np.arange(got.shape[0]), indices)], 0.0)


@pytest.mark.parametrize("omega_axis", [0, 1])
def test_contiguous_nonzero_start_frequencies_match_numpy(omega_axis):
    indices = np.asarray([2, 3, 4])
    got, want = _run_affine_window(
        _mesh(), omega_axis=omega_axis, indices=indices,
        omega_values=np.asarray([0.31, 0.57, 0.92]))
    np.testing.assert_allclose(got, want, rtol=3e-14, atol=3e-14)
    inactive = np.setdiff1d(np.arange(7), indices)
    np.testing.assert_array_equal(
        got[inactive] if omega_axis == 0 else got[:, inactive], 0.0)


def test_contiguous_antihermitian_completion_matches_numpy():
    indices = np.asarray([1, 2, 3, 4])
    got, want = _run_affine_window(
        _mesh(), omega_axis=1, indices=indices,
        omega_values=np.asarray([0.18, 0.36, 0.73, 1.22]),
        antihermitian=True, omega_sign=-1.0)
    np.testing.assert_allclose(got, want, rtol=4e-14, atol=4e-14)
    np.testing.assert_allclose(
        got[:, indices], np.conj(np.swapaxes(got[:, indices], -1, -2)),
        rtol=4e-14, atol=4e-14)


def test_descending_interval_preserves_order_via_general_index_path():
    mesh = _mesh()
    indices = np.asarray([4, 3, 2])
    values = np.asarray([0.92, 0.57, 0.31])
    got, want = _run_affine_window(
        mesh, indices=indices, omega_values=values)
    np.testing.assert_allclose(got, want, rtol=3e-14, atol=3e-14)

    sharding = NamedSharding(mesh, P(None, None, "x", "y"))
    acc = DeviceOmegaAccumulator(
        np.arange(7), shape=(7, 1, 4, 4), sharding=sharding, omega_axis=0)
    acc.begin_window(
        [0.2], [0.5], omega_sign=1.0, prefactor=1.0,
        omega_indices=indices, omega_values=values)
    assert acc._contiguous is False


def test_full_negative_frequency_branch_matches_numpy():
    got, want = _run_affine_window(_mesh(), omega_sign=-1.0)
    np.testing.assert_allclose(got, want, rtol=3e-14, atol=3e-14)


def test_general_active_antihermitian_crosses_band_shards():
    indices = np.asarray([6, 0, 3])
    got, want = _run_affine_window(
        _mesh(), omega_axis=1, indices=indices,
        omega_values=np.asarray([1.5, 0.08, 0.44]),
        antihermitian=True, omega_sign=-1.0)
    np.testing.assert_allclose(got, want, rtol=4e-14, atol=4e-14)
    inactive = np.setdiff1d(np.arange(got.shape[1]), indices)
    np.testing.assert_array_equal(got[:, inactive], 0.0)


def test_empty_frequency_window_is_an_exact_noop():
    mesh = _mesh()
    omega = np.asarray([-0.2, 0.0, 0.4])
    output_sharding = NamedSharding(mesh, P(None, None, "x", "y"))
    sigma_sharding = NamedSharding(mesh, P(None, "x", "y"))
    acc = DeviceOmegaAccumulator(
        omega, shape=(3, 1, 4, 4), sharding=output_sharding, omega_axis=0)
    times = np.asarray([0.2 + 0.1j, 0.7 - 0.2j])
    acc.begin_window(
        times, np.asarray([0.4, -0.3j]), omega_sign=1.0, prefactor=0.7,
        antihermitian=True, omega_indices=np.asarray([], np.int32),
        omega_values=np.asarray([], np.float64))
    assert acc.precompile_tau_add(
        sigma_shape=(1, 4, 4), sigma_sharding=sigma_sharding) is False
    poison = _put(np.full((1, 4, 4), np.nan + 1j * np.nan), sigma_sharding)
    for _ in times:
        acc.add_tau(poison)
    acc.end_window()
    np.testing.assert_array_equal(
        _to_host(acc.finalize()), np.zeros((3, 1, 4, 4), np.complex128))


def test_scalar_updates_preserve_cancellation_sensitive_node_order():
    mesh = _mesh()
    omega = np.asarray([-0.9, -0.1, 0.2, 0.8, 1.3])
    indices = np.asarray([1, 2, 3])
    times = np.zeros(7, np.complex128)
    alpha = np.asarray(
        [1e16, 1.0, -1e16, 3.0, -2.0, 0.25, -0.125], np.complex128)
    sigma_np = (np.arange(16).reshape(1, 4, 4) + 0.25j).astype(np.complex128)
    sigma_sharding = NamedSharding(mesh, P(None, "x", "y"))
    output_sharding = NamedSharding(mesh, P(None, None, "x", "y"))
    acc = DeviceOmegaAccumulator(
        omega, shape=(5, 1, 4, 4), sharding=output_sharding, omega_axis=0)
    acc.begin_window(
        times, alpha, omega_sign=1.0, prefactor=1.0,
        omega_indices=indices, omega_values=np.asarray([0.9, 0.2, 0.8]))
    sigma = _put(sigma_np, sigma_sharding)
    for _ in times:
        acc.add_tau(sigma)
    acc.end_window()
    want = _window_reference(
        omega=omega, times=times, alpha=alpha, base=sigma_np,
        slope=np.zeros_like(sigma_np), omega_sign=1.0, prefactor=1.0,
        e_ref_sum=0.0, indices=indices,
        omega_values=np.asarray([0.9, 0.2, 0.8]))
    np.testing.assert_array_equal(_to_host(acc.finalize()), want)


@pytest.mark.parametrize("route", ["dense", "contiguous", "general", "anti"])
def test_scalar_prewarm_accepts_each_shape_route(route):
    kwargs = {}
    if route == "contiguous":
        kwargs = {"indices": np.asarray([2, 3, 4]),
                  "omega_values": np.asarray([0.2, 0.4, 0.7])}
    elif route == "general":
        kwargs = {"indices": np.asarray([5, 1, 4]),
                  "omega_values": np.asarray([0.8, 0.1, 0.6])}
    elif route == "anti":
        kwargs = {"indices": np.asarray([1, 2, 3]),
                  "omega_values": np.asarray([0.1, 0.2, 0.3]),
                  "antihermitian": True}
    got, want = _run_affine_window(_mesh(), precompile=True, **kwargs)
    np.testing.assert_allclose(got, want, rtol=4e-14, atol=4e-14)


def test_explicit_ordered_full_frequency_set_selects_dense_fast_path():
    mesh = _mesh()
    omega = np.asarray([-0.5, 0.0, 0.4])
    sharding = NamedSharding(mesh, P(None, None, "x", "y"))
    acc = DeviceOmegaAccumulator(
        omega, shape=(3, 1, 4, 4), sharding=sharding, omega_axis=0)
    acc.begin_window(
        [0.2], [0.5], omega_sign=1.0, prefactor=1.0,
        omega_indices=np.arange(omega.size), omega_values=omega)
    assert acc._indices is None
    acc.add_tau(jnp.zeros((1, 4, 4), jnp.complex128))
    acc.end_window()


def test_active_frequency_input_and_lifecycle_guards():
    mesh = _mesh()
    omega = np.asarray([-0.2, 0.0, 0.4])
    sharding = NamedSharding(mesh, P(None, None, "x", "y"))

    def fresh():
        return DeviceOmegaAccumulator(
            omega, shape=(3, 1, 4, 4), sharding=sharding, omega_axis=0)

    with pytest.raises(ValueError, match="distinct"):
        fresh().begin_window(
            [0.2], [0.5], omega_sign=1.0, prefactor=1.0,
            omega_indices=[1, 1], omega_values=[0.0, 0.0])
    with pytest.raises(ValueError, match="invalid active frequency"):
        fresh().begin_window(
            [0.2], [0.5], omega_sign=1.0, prefactor=1.0,
            omega_indices=[3], omega_values=[0.4])
    with pytest.raises(ValueError, match="invalid active frequency"):
        fresh().begin_window(
            [0.2], [0.5], omega_sign=1.0, prefactor=1.0,
            omega_indices=[1, 2], omega_values=[0.0])
    with pytest.raises(RuntimeError, match="no open"):
        fresh().add_tau(jnp.zeros((1, 4, 4), jnp.complex128))

    acc = fresh()
    acc.begin_window([0.2], [0.5], omega_sign=1.0, prefactor=1.0)
    with pytest.raises(RuntimeError, match="still open"):
        acc.begin_window([0.2], [0.5], omega_sign=1.0, prefactor=1.0)
    with pytest.raises(RuntimeError, match="before all tau nodes"):
        acc.end_window()
    with pytest.raises(RuntimeError, match="open frequency window"):
        acc.finalize()


@pytest.mark.parametrize("contiguous, expected_update", [
    (True, "dynamic-update-slice"),
    (False, "scatter"),
])
def test_active_scalar_hlo_uses_expected_update_and_alias(
        contiguous, expected_update):
    mesh = _mesh()
    output_sharding = NamedSharding(mesh, P(None, None, "x", "y"))
    sigma_sharding = NamedSharding(mesh, P(None, "x", "y"))
    replicated = NamedSharding(mesh, P())
    acc = _put(np.zeros((7, 1, 4, 4), np.complex128), output_sharding)
    sigma = _put(np.ones((1, 4, 4), np.complex128), sigma_sharding)
    coeff = _put(np.asarray([0.2, -0.3j, 0.7]), replicated)
    indices = _put(
        np.asarray([2, 3, 4] if contiguous else [5, 1, 4], np.int32),
        replicated)
    compiled = ppm_accumulators._device_active_omega_add(
        output_sharding, 0, contiguous).lower(
            acc, sigma, coeff, indices).compile()
    hlo = compiled.as_text().lower()
    assert expected_update in hlo
    if contiguous:
        assert not re.search(r"\bscatter(?:-start|-done)?\(", hlo)
    for collective in (
            "all-gather", "all-reduce", "reduce-scatter", "all-to-all",
            "collective-permute"):
        assert collective not in hlo
    assert "input_output_alias" in hlo
    local_bytes = int(acc.addressable_shards[0].data.nbytes)
    assert int(compiled.memory_analysis().alias_size_in_bytes) >= local_bytes

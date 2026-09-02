import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from types import SimpleNamespace

from gw.ppm_accumulators import (DeviceOmegaAccumulator,
                                 _MemoryTileSink, _TauAccumulator,
                                 _device_tau_window_loop)
from gw.ppm_tau_kernel import build_shared_w_tau


def _mesh():
    return Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))


def test_shared_w_sums_selected_poles_before_the_spatial_kernel():
    B = np.asarray([
        [[[1.0 + 0.2j, 0.3], [0.4j, 0.8 - 0.1j]]],
        [[[0.5 - 0.1j, 0.7j], [0.2, 1.1 + 0.4j]]],
    ], dtype=np.complex128)
    Omega = np.asarray([
        [[[1.0 - 0.2j, 2.0 - 0.1j], [0.5 - 0.3j, 1.5 - 0.4j]]],
        [[[0.8 - 0.05j, 1.2 - 0.25j], [2.2 - 0.1j, 0.4 - 0.6j]]],
    ], dtype=np.complex128)
    poles = np.asarray([0, 1], dtype=np.int32)
    bounds = np.asarray([
        [0.0, 1.6, 0.15, -np.inf, np.inf, np.inf],
        [0.0, np.inf, -np.inf, -np.inf, 0.2, np.inf],
    ])
    real_phase = np.asarray([False, True])
    t, e_ref = 0.37 - 0.11j, 0.06

    got = np.asarray(build_shared_w_tau(
        B, Omega, poles, bounds, real_phase, e_ref, t))
    want = np.zeros_like(B[0])
    for row, pole in enumerate(poles):
        omega = Omega[pole]
        a, gamma = omega.real, -omega.imag
        b = bounds[row]
        selected = ((a > b[0]) & (a <= b[1])
                    & (gamma >= b[2]) & (gamma > b[3])
                    & (gamma < b[4]) & (gamma <= b[5]))
        phase = a if real_phase[row] else omega
        want += np.where(
            selected, B[pole] * np.exp(-1j * (phase - e_ref) * t), 0.0)
    np.testing.assert_allclose(got, want, rtol=2e-15, atol=2e-15)


def test_selector_boundary_is_strict_below_and_inclusive_above():
    B = np.ones((1, 1, 1, 1), dtype=np.complex128)
    Omega = np.asarray([[[[1.0 - 0.2j]]]])
    bounds = np.asarray([
        [0.0, np.inf, -np.inf, -np.inf, 0.2, np.inf],
        [0.0, np.inf, 0.2, -np.inf, np.inf, np.inf],
    ])
    got = np.asarray(build_shared_w_tau(
        B, Omega, np.asarray([0, 0]), bounds,
        np.asarray([False, False]), 0.0, 0.3))
    want = np.exp(-1j * Omega[0] * 0.3)
    np.testing.assert_allclose(got, want, rtol=2e-15, atol=2e-15)


def test_device_frequency_fold_and_one_sided_completion():
    mesh = _mesh()
    sigma_sharding = NamedSharding(mesh, P(None, "x", "y"))
    output_sharding = NamedSharding(mesh, P(None, None, "x", "y"))
    omega = np.asarray([-0.4, 0.2, 0.9])
    sigma = jax.device_put(
        np.arange(8, dtype=np.float64).reshape(2, 2, 2) * (1.0 + 0.3j),
        sigma_sharding)
    t = np.asarray([0.2 + 0.1j, 0.7 - 0.2j])
    alpha = np.asarray([0.4 - 0.1j, -0.2 + 0.3j])
    e_ref, sign, pref = 0.35, -1.0, 0.6
    shape = (omega.size, *sigma.shape)

    acc = DeviceOmegaAccumulator(omega, shape=shape,
                                 sharding=output_sharding, omega_axis=0)
    acc.begin_window(t, alpha, omega_sign=sign, prefactor=pref,
                     e_ref_sum=e_ref, antihermitian=True)
    for _ in t:
        acc.add_tau(sigma)
    acc.end_window()
    got = np.asarray(acc.finalize())

    coeff = ((pref * alpha[:, None])
             * np.exp(-1j * (e_ref - sign * omega[None, :]) * t[:, None]))
    Z = coeff.sum(axis=0).reshape((-1, 1, 1, 1)) * np.asarray(sigma)[None]
    want = (Z - np.conj(np.swapaxes(Z, -1, -2))) / 2j
    np.testing.assert_allclose(got, want, rtol=2e-14, atol=2e-14)

    broken = DeviceOmegaAccumulator(omega, shape=shape,
                                    sharding=output_sharding, omega_axis=0)
    broken.begin_window(t, alpha, omega_sign=sign, prefactor=pref)
    broken.add_tau(sigma)
    with pytest.raises(RuntimeError, match="before all tau nodes"):
        broken.end_window()


def test_device_frequency_fold_can_target_one_causal_half():
    mesh = _mesh()
    sigma_sharding = NamedSharding(mesh, P(None, "x", "y"))
    output_sharding = NamedSharding(mesh, P(None, None, "x", "y"))
    omega = np.asarray([-0.6, -0.2, 0.0, 0.4])
    sigma = jax.device_put(
        np.asarray([[[2.0 + 0.5j]]]), sigma_sharding)
    acc = DeviceOmegaAccumulator(
        omega, shape=(4, 1, 1, 1), sharding=output_sharding, omega_axis=0)
    acc.begin_window(
        np.asarray([0.3]), np.asarray([0.7]), omega_sign=1.0,
        prefactor=-1.0, omega_indices=np.asarray([2, 3]),
        omega_values=np.asarray([0.0, 0.4]))
    acc.add_tau(sigma)
    acc.end_window()
    got = np.asarray(acc.finalize())[:, 0, 0, 0]
    assert np.array_equal(got[:2], np.zeros(2, np.complex128))
    want = -0.7 * np.exp(1j * np.asarray([0.0, 0.4]) * 0.3) * (2 + 0.5j)
    np.testing.assert_allclose(got[2:], want, rtol=2e-14, atol=2e-14)


def test_device_frequency_fold_preserves_a_leading_bracket_axis():
    mesh = _mesh()
    sigma_sharding = NamedSharding(mesh, P(None, None, "x", "y"))
    output_sharding = NamedSharding(
        mesh, P(None, None, None, "x", "y"))
    omega = np.asarray([-0.5, 0.25, 0.75])
    sigma_np = np.arange(16, dtype=np.float64).reshape(2, 2, 2, 2)
    sigma = jax.device_put(sigma_np * (1.0 - 0.2j), sigma_sharding)
    acc = DeviceOmegaAccumulator(
        omega, shape=(2, 3, 2, 2, 2), sharding=output_sharding,
        omega_axis=1)
    acc.begin_window(
        np.asarray([0.4]), np.asarray([0.7]), omega_sign=-1.0,
        prefactor=0.5)
    acc.add_tau(sigma)
    acc.end_window()

    got = np.asarray(acc.finalize())
    coeff = 0.35 * np.exp(-1j * omega * 0.4)
    want = coeff.reshape(1, 3, 1, 1, 1) * np.asarray(sigma)[:, None]
    np.testing.assert_allclose(got, want, rtol=2e-14, atol=2e-14)


def test_dynamic_tau_loop_reuses_one_signature_for_active_counts():
    mesh = _mesh()
    sigma_sharding = NamedSharding(mesh, P(None, "x", "y"))
    output_sharding = NamedSharding(mesh, P(None, None, "x", "y"))
    omega = np.asarray([-0.7, 0.1, 0.8])
    sigma_np = (
        np.arange(8, dtype=np.float64).reshape(2, 2, 2) + 1.0) * (1 - 0.3j)
    sigma = jax.device_put(sigma_np, sigma_sharding)

    @jax.jit
    def tau_kernel(scale, t_node):
        return scale * (1.0 + 0.2j * t_node)

    loop = _device_tau_window_loop(tau_kernel, output_sharding, 0)
    loop.clear_cache()
    capacity = 5
    for n_tau in (1, 3, capacity):
        t = np.linspace(0.2, 0.8, n_tau).astype(np.complex128)
        alpha = np.linspace(0.4, 0.9, n_tau).astype(np.complex128)
        acc = DeviceOmegaAccumulator(
            omega, shape=(3, 2, 2, 2), sharding=output_sharding,
            omega_axis=0)
        acc.begin_window(
            t, alpha, omega_sign=1.0, prefactor=-0.6,
            e_ref_sum=0.25, capacity=capacity)
        acc.add_tau_loop(tau_kernel, (sigma,))
        acc.end_window()
        got = np.asarray(acc.finalize())

        coeff = (-0.6 * alpha[:, None]
                 * np.exp(-1j * (0.25 - omega[None, :]) * t[:, None]))
        scale = 1.0 + 0.2j * t
        want_coeff = np.sum(coeff * scale[:, None], axis=0)
        want = want_coeff.reshape(3, 1, 1, 1) * sigma_np[None]
        np.testing.assert_allclose(got, want, rtol=2e-14, atol=2e-14)

    assert loop._cache_size() == 1


def test_mpa_host_and_device_accumulators_match_both_causal_halves():
    mesh = _mesh()
    sigma_sharding = NamedSharding(mesh, P(None, "x", "y"))
    output_sharding = NamedSharding(mesh, P(None, None, "x", "y"))
    omega = np.asarray([-0.8, -0.2, 0.2, 0.8])
    shape = (4, 2, 2, 2)
    sigma_np = (
        np.arange(8, dtype=np.float64).reshape(2, 2, 2) + 0.5) * (1 + 0.4j)
    sigma = jax.device_put(sigma_np, sigma_sharding)
    host = _TauAccumulator(
        omega_vec=omega,
        sink=_MemoryTileSink(
            shape=shape, sharding=output_sharding, omega_axis=0),
        lag=1)
    device = DeviceOmegaAccumulator(
        omega, shape=shape, sharding=output_sharding, omega_axis=0)

    windows = (
        SimpleNamespace(
            t=np.asarray([0.25 + 0.1j, 0.7 + 0.2j]),
            alpha=np.asarray([0.4 - 0.1j, -0.2 + 0.3j]),
            omega_sign=1.0, prefactor=0.7, project_code=0,
            omega_indices=np.asarray([2, 3]),
            omega_values=np.asarray([0.2, 0.8]), e_ref_sum=0.3),
        SimpleNamespace(
            t=np.asarray([0.15, 0.55]),
            alpha=np.asarray([0.6 + 0.2j, 0.1 - 0.4j]),
            omega_sign=-1.0, prefactor=-0.5, project_code=1,
            omega_indices=np.asarray([0, 1]),
            omega_values=np.asarray([0.8, 0.2]), e_ref_sum=-0.2),
    )
    for window in windows:
        host.begin_window(window)
        device.begin_window(
            window.t, window.alpha,
            omega_sign=window.omega_sign, prefactor=window.prefactor,
            e_ref_sum=window.e_ref_sum,
            antihermitian=(window.project_code == 1),
            omega_indices=window.omega_indices,
            omega_values=window.omega_values)
        for t, alpha in zip(window.t, window.alpha):
            sigma_tau = sigma * (1.0 + 0.1j * t)
            alpha_eff = alpha * np.exp(-1j * window.e_ref_sum * t)
            host.add_tau(
                sigma_tau, None, complex(t), complex(alpha_eff))
            device.add_tau(sigma_tau)
        host.end_window()
        device.end_window()

    np.testing.assert_allclose(
        np.asarray(host.finalize()), np.asarray(device.finalize()),
        rtol=3e-14, atol=3e-14)

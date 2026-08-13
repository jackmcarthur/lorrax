import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from gw.ppm_accumulators import DeviceOmegaAccumulator
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
                                 sharding=output_sharding)
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
                                    sharding=output_sharding)
    broken.begin_window(t, alpha, omega_sign=sign, prefactor=pref)
    broken.add_tau(sigma)
    with pytest.raises(RuntimeError, match="before all tau nodes"):
        broken.end_window()

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from gw.ppm_accumulators import DeviceOmegaAccumulator
from gw.ppm_tau_kernel import (
    build_shared_w_tau,
    get_shared_sigma_spatial_kernel,
    get_shared_w_tau_kernel,
)
from gw.mpa.sigma import _admit_w_time_factor, _frozen_w_time_factor


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


def test_frozen_w_builder_is_the_owned_w_tau_algebra():
    """The cacheable seam produces the exact established W(tau) tile."""
    mesh = _mesh()
    B = np.asarray([[[[1.0 + 0.25j]]]], dtype=np.complex128)
    Omega = np.asarray([[[[0.8 - 0.1j]]]], dtype=np.complex128)
    poles = np.asarray([0], dtype=np.int32)
    bounds = np.asarray([[0.0, np.inf, -np.inf, -np.inf,
                          np.inf, np.inf]])
    real_phase = np.asarray([False])
    args = (B, Omega, poles, bounds, real_phase, 0.04, 0.3 - 0.05j)
    cached_seam = get_shared_w_tau_kernel(mesh_xy=mesh)(*args)
    direct = build_shared_w_tau(*args)
    np.testing.assert_array_equal(np.asarray(cached_seam), np.asarray(direct))


def test_frozen_w_cache_allows_reordered_subset_but_refuses_new_identity():
    first, second, third = object(), object(), object()
    bank = {
        "signatures": [("a",), ("b",), ("c",)],
        "factors": [first, second, third],
        "complete": True,
    }

    assert _frozen_w_time_factor(bank, ("c",)) is third
    assert _frozen_w_time_factor(bank, ("a",)) is first
    with pytest.raises(RuntimeError, match="sc_w_time_cache_identity"):
        _frozen_w_time_factor(bank, ("new",))


def test_frozen_w_cache_keeps_identity_when_device_budget_evicts_factor():
    mesh = _mesh()
    first = jax.device_put(np.ones((1, 2, 2), dtype=np.complex128))
    second = jax.device_put(np.ones((1, 2, 2), dtype=np.complex128))
    bank = {
        "signatures": [], "factors": [], "factor_index": {},
        "complete": False,
    }
    owner = {
        "capacity_per_device_bytes": int(first.nbytes),
        "resident_per_device_bytes": 0,
    }

    assert _admit_w_time_factor(bank, ("first",), first, owner, mesh)
    assert not _admit_w_time_factor(bank, ("second",), second, owner, mesh)
    assert _frozen_w_time_factor(bank, ("first",)) is first
    assert _frozen_w_time_factor(bank, ("second",)) is None
    assert owner["resident_per_device_bytes"] == first.nbytes
    assert owner["total_evictions"] == 1


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


def test_device_frequency_fold_carries_a_leading_band_bracket_axis():
    """The shared fold inserts omega behind, never through, brackets."""
    mesh = _mesh()
    sigma_sharding = NamedSharding(mesh, P(None, None, "x", "y"))
    output_sharding = NamedSharding(
        mesh, P(None, None, None, "x", "y"))
    omega = np.asarray([-0.3, 0.5])
    sigma_host = (np.arange(16, dtype=np.float64).reshape(2, 2, 2, 2)
                  * (0.4 - 0.2j))
    sigma = jax.device_put(sigma_host, sigma_sharding)
    t = np.asarray([0.25, 0.8])
    alpha = np.asarray([0.7, -0.1])
    acc = DeviceOmegaAccumulator(
        omega, shape=(2, omega.size, 2, 2, 2),
        sharding=output_sharding, omega_axis=1)
    acc.begin_window(t, alpha, omega_sign=1.0, prefactor=-1.0)
    for _ in t:
        acc.add_tau(sigma)
    acc.end_window()

    got = np.asarray(acc.finalize())
    coeff = (-alpha[:, None]
             * np.exp(1j * omega[None, :] * t[:, None])).sum(axis=0)
    want = sigma_host[:, None, ...] * coeff[None, :, None, None, None]
    assert got.shape == (2, 2, 2, 2, 2)
    np.testing.assert_allclose(got, want, rtol=2e-14, atol=2e-14)


def test_shared_tau_factory_forwards_the_band_partition(monkeypatch):
    """The MPA door reaches the existing bracketed spatial kernel."""
    import gw.ppm_tau_kernel as tau

    seen = {}

    def fake_spatial(**kwargs):
        seen.update(kwargs)
        return lambda *_args: np.zeros((2, 1, 1, 1), np.complex128)

    monkeypatch.setattr(tau, "_get_sigma_kij_kernel", fake_spatial)
    tau._sigma_shared_tau_kernel_cache.clear()
    brackets = ((0, 3), (3, 7))
    tau.get_shared_sigma_tau_kernel(
        mesh_xy=_mesh(), kgrid=(1, 1, 1), brackets=brackets,
        pack_brackets=False)
    assert seen["brackets"] == brackets
    assert seen["merged_x"] is True
    assert seen["pack_brackets"] is False
    tau._sigma_shared_tau_kernel_cache.clear()


def test_prepared_w_factory_reuses_the_same_spatial_kernel(monkeypatch):
    """Prepared W enters the existing convolution owner, not a twin."""
    import gw.ppm_tau_kernel as tau

    seen = {}
    sentinel = object()

    def fake_spatial(**kwargs):
        seen.update(kwargs)
        return sentinel

    monkeypatch.setattr(tau, "_get_sigma_kij_kernel", fake_spatial)
    got = get_shared_sigma_spatial_kernel(
        mesh_xy=_mesh(), kgrid=(1, 2, 3), brackets=((0, 2),),
        pack_brackets=False)
    assert got is sentinel
    assert seen["kgrid"] == (1, 2, 3)
    assert seen["merged_x"] is True
    assert seen["brackets"] == ((0, 2),)
    assert seen["layout"] == "legacy"
    assert seen["face_shape"] is None
    assert seen["pack_brackets"] is False

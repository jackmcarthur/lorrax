"""Σ window table: device coefficient == host coefficient (rounding), exact operands, device counter."""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P


def _mesh():
    n = 4 if len(jax.devices()) >= 4 else 1
    return jax.sharding.Mesh(np.array(jax.devices()[:n]).reshape((2, 2) if n == 4 else (1, 1)), ("x", "y"))


def _accumulator(omega):
    from gw.ppm_accumulators import DeviceOmegaAccumulator
    mesh = _mesh()
    return DeviceOmegaAccumulator(omega, shape=(omega.size, 2, 2),
                                  sharding=NamedSharding(mesh, P()), omega_axis=0)


def test_window_pick_matches_host_coefficient():
    from gw.ppm_accumulators import _omega_coefficient
    rng = np.random.default_rng(3)
    omega = np.linspace(-2.0, 1.0, 11)
    acc = _accumulator(omega)
    windows = []
    for n, cols in ((5, None), (3, np.array([1, 4, 7])), (7, np.array([0, 10]))):
        t = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        a = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        windows.append(dict(t=t, alpha=a, omega_indices=cols,
                            omega_values=None if cols is None else np.abs(omega[cols]),
                            omega_sign=-1, prefactor=-1.0, e_ref_sum=0.37))
    extras = dict(active=[np.int32(k) for k in (2, 5, 1)],
                  mask=[np.array([True, False, True]), np.array([False, False, True]), np.array([True, True, True])])
    table = acc.window_table(windows, capacity=8, extras=extras)
    assert table.nodes == (5, 3, 7)
    counter = acc.window_counter()
    for i, w in enumerate(windows):
        t_pad, coeff, n, ex, counter = acc.window_pick(table, counter)
        cols = np.arange(omega.size) if w["omega_indices"] is None else w["omega_indices"]
        om = omega.astype(np.complex128) if w["omega_indices"] is None else w["omega_values"].astype(np.complex128)
        want = np.zeros((8, omega.size), np.complex128)
        want[:w["t"].size, cols] = _omega_coefficient(np, om[None, :], w["t"][:, None], w["alpha"][:, None],
                                                      float(w["omega_sign"]), float(w["prefactor"]),
                                                      float(w["e_ref_sum"]))
        got = np.asarray(coeff)
        assert np.array_equal(got == 0, want == 0)
        np.testing.assert_allclose(got, want, rtol=1e-13, atol=0)
        tp = np.zeros(8, np.complex128); tp[:w["t"].size] = w["t"]
        assert np.array_equal(np.asarray(t_pad), tp) and int(n) == w["t"].size
        assert int(ex["active"]) == int(extras["active"][i])
        assert np.array_equal(np.asarray(ex["mask"]), extras["mask"][i])
    assert int(counter) == len(windows)


def test_jax_slice_axis_static_and_numpy():
    from runtime.padding import jax_slice_axis
    a = np.arange(24.0).reshape(4, 6)
    np.testing.assert_array_equal(np.asarray(jax_slice_axis(jnp.asarray(a), 4, axis=1)), a[:, :4])
    np.testing.assert_array_equal(jax_slice_axis(a, 3, axis=0), a[:3])


def test_check_factor_split_equals_device_check():
    from file_io import shared_pole_store as sps
    rng = np.random.default_rng(0)
    b = np.zeros((2, 3, 1, 5), np.complex128)
    K = np.array([3, 5], np.int64)
    b[0, :, :, :3] = rng.standard_normal((3, 1, 3))
    b[1] = rng.standard_normal((3, 1, 5))
    poles = np.ones((2, 5)); poles[0, :3] = [0.1, 0.2, 0.3]; poles[1] = [0.1, 0.2, 0.2, 0.5, 0.9]
    bd = jax.device_put(b, NamedSharding(_mesh(), P()))
    for p2 in (poles, np.where(np.arange(5) == 4, 2.0, poles), np.where(np.arange(5) == 1, -1.0, poles)):
        split = sps._poles_valid_host(p2, K) and bool(sps._factor_b_valid(bd, sps._placed_counts(bd, K)))
        assert split == bool(sps._factor_valid(bd, p2, K))

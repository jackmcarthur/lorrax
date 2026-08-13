"""The bounded, no-I/O MPA sample-tile to pole boundary."""

from __future__ import annotations

import inspect
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from gw.mpa import model, pade_fit, sample_plan, tiling  # noqa: E402


def _poles(n_p, seed):
    rng = np.random.default_rng(seed)
    Omega = (np.linspace(0.3, 3.2, n_p)
             - 1j * np.linspace(0.04, 0.09, n_p))
    B = ((0.4 + rng.random(n_p))
         * np.exp(2j * np.pi * rng.random(n_p)))
    return Omega, B


def _samples(plan, n_q=2, n_mu=2):
    z = sample_plan.plan_z(plan)
    n_p = len(z) // 2
    body = np.empty((len(z), n_q, n_mu, n_mu), np.complex128)
    for q in range(n_q):
        for mu in range(n_mu):
            for nu in range(n_mu):
                Omega, B = _poles(n_p, 100 * q + 10 * mu + nu)
                body[:, q, mu, nu] = pade_fit.synthesize_w_samples(
                    Omega, B, z)
    h_Omega, h_B = _poles(n_p, 999)
    head = pade_fit.synthesize_w_samples(h_Omega, h_B, z)
    return body, head


def test_producer_walks_bounded_tiles_and_returns_sigma_ready_poles():
    plan = sample_plan.mpa_plan(4, 4.0, energy_unit="Ry")
    body, head = _samples(plan)
    calls = []

    def producer(got_plan, q, columns):
        calls.append((got_plan, q, tuple(columns)))
        return np.take(body[:, q], columns, axis=-1)

    got = model.produce_model(
        plan, 2, 2, producer, head, energy_unit="Ry")
    expected_walk = [
        (q, tuple(range(lo, hi)))
        for q, lo, hi in tiling.fit_schedule(2, 2, len(got.z_samples))]
    assert [(q, cols) for _, q, cols in calls] == expected_walk
    assert all(got_plan is plan for got_plan, _, _ in calls)
    assert max(len(cols) for _, _, cols in calls) == 1

    assert got.Omega_p.shape == (4, 2, 2, 2)
    assert got.B_p.shape == got.Omega_p.shape
    assert got.head_Omega_p.shape == (4,)
    assert got.head_B_p.shape == (4,)
    assert got.energy_unit == "Ry"
    np.testing.assert_array_equal(got.z_samples, sample_plan.plan_z(plan))

    z = got.z_samples
    for q in range(2):
        for mu in range(2):
            for nu in range(2):
                rebuilt = np.asarray(pade_fit.eval_mpa_model(
                    got.Omega_p[:, q, mu, nu],
                    got.B_p[:, q, mu, nu], z))
                np.testing.assert_allclose(
                    rebuilt, body[:, q, mu, nu], rtol=2e-7, atol=2e-7)
    rebuilt_head = np.asarray(pade_fit.eval_mpa_model(
        got.head_Omega_p, got.head_B_p, z))
    np.testing.assert_allclose(rebuilt_head, head, rtol=2e-7, atol=2e-7)

    source = inspect.getsource(model)
    assert "mpa_store" not in source
    assert "h5py" not in source
    assert "Wc_body_samples" not in inspect.signature(
        model.produce_model).parameters


def test_tile_shape_and_head_contract_refuse_at_the_bounded_seam():
    plan = sample_plan.mpa_plan(2, 4.0, energy_unit="Ry")
    body, head = _samples(plan, n_q=1, n_mu=2)

    with pytest.raises(ValueError, match="returned shape"):
        model.produce_model(
            plan, 1, 2,
            lambda _plan, q, cols: np.take(
                body[:, q], cols, axis=-1)[..., :0], head)
    with pytest.raises(ValueError, match="Wc_head_samples"):
        model.produce_model(
            plan, 1, 2,
            lambda _plan, q, cols: np.take(
                body[:, q], cols, axis=-1), head[:-1])

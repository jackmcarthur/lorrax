"""Cheap CPU tests of the full-density frozen-Hamiltonian snapshot solve.

Run with four CPU devices to exercise the propagator's distributed k layout:
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
    python -m pytest tests/test_tdagw_collision_snapshot.py -q
"""
import runtime  # noqa: F401 -- establish runtime defaults before JAX
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from tdagw.dynsig import collision_basis_state, collision_snapshot_state, slow_basis_step

jax.config.update("jax_enable_x64", True)


def _state():
    mesh = Mesh(np.asarray(jax.devices()).reshape((1, -1)), ("x", "y"))
    nk = len(jax.devices())
    blocks = NamedSharding(mesh, P(None, ("x", "y"), None, None))
    reference_sh = NamedSharding(mesh, P(("x", "y"), None, None))
    energy_sh = NamedSharding(mesh, P(("x", "y"), None))
    reference = np.broadcast_to(np.diag([1.0, 0.0]), (nk, 2, 2)).astype(complex).copy()
    rho = np.broadcast_to(np.array([[0.8, 0.12 + 0.07j], [0.12 - 0.07j, 0.2]]),
                          (2, nk, 2, 2)).copy()
    energies = np.broadcast_to(np.array([-0.3, 0.4]), (nk, 2)).copy()
    put = lambda value: jax.device_put(value, blocks)
    reference = jax.device_put(reference, reference_sh)
    energies = jax.device_put(energies, energy_sh)
    occupations = jax.device_put(np.tile([1.0, 0.0], (nk, 1)), energy_sh)
    return put, put(rho), reference, energies, occupations


def _apply(params, rho, t):
    kernel_scale, reference, held = params
    return kernel_scale * (rho - reference[None]) + held


def test_snapshot_basis_keeps_full_density_coherences():
    put, rho, reference, energies, occupations = _state()
    step = slow_basis_step(_apply, reference, energies, occupations)
    params = (0.3, reference, put(np.zeros(rho.shape, complex)))
    U, E, f, dsigma, receipt = collision_basis_state(step, params, rho, 0.0)
    H = energies[..., :, None] * jnp.eye(2)[None] + _apply(params, rho, 0.0)
    np.testing.assert_allclose(H[None] @ U if H.ndim == 3 else H @ U,
                               U * E[..., None, :], atol=2e-14)
    np.testing.assert_allclose(dsigma[..., 0, 1], 0.3 * rho[..., 0, 1], atol=2e-14)
    rotated = jnp.conj(jnp.swapaxes(U, -1, -2)) @ rho @ U
    np.testing.assert_allclose(f, jnp.real(jnp.diagonal(rotated, axis1=-2, axis2=-1)))
    assert receipt["iterations"] == 1


@pytest.mark.parametrize("acceleration,slope", [("linear", -1.5), ("anderson", -3.0)])
def test_snapshot_linear_fixed_point_shares_final_hamiltonian(acceleration, slope):
    put, rho, reference, energies, occupations = _state()
    step = slow_basis_step(_apply, reference, energies, occupations)
    B = put(np.broadcast_to(np.array([[0.02, 0.004j], [-0.004j, -0.01]]), rho.shape))
    calls = []

    def evaluate(held):
        U, E, f, dsigma, _ = collision_basis_state(
            step, (0.3, reference, held), rho, 0.0)
        H = (U * E[..., None, :]) @ jnp.conj(jnp.swapaxes(U, -1, -2))
        # Recover the feedback from the spectrum, so this tests a joint
        # H/eigenpair/coherent-map solve against a known linear fixed point.
        base = energies[..., :, None] * jnp.eye(2)[None] + 0.3 * (rho - reference[None])
        calls.append(np.asarray(held))
        return B + slope * (H - base), (U, E, f, dsigma)

    held, (U, E, _, dsigma), receipt = collision_snapshot_state(
        evaluate, put(np.zeros(rho.shape, complex)), steps=32, tol_ha=1e-11, mixing=0.5,
        acceleration=acceleration)
    np.testing.assert_allclose(held, B / (1.0 - slope), atol=1e-11)
    np.testing.assert_array_equal(held, calls[-1])
    H = energies[..., :, None] * jnp.eye(2)[None] + _apply((0.3, reference, held), rho, 0.0)
    np.testing.assert_allclose(H @ U, U * E[..., None, :], atol=2e-14)
    np.testing.assert_allclose(dsigma, _apply((0.3, reference, held), rho, 0.0), atol=2e-14)
    assert receipt["converged"] and 1 < receipt["iterations"] <= 32
    assert receipt["residual"] <= receipt["tolerance_ha"]
    assert receipt["iterations"] == len(calls)
    if acceleration == "anderson":
        assert len(calls) <= 4  # The linearly mixed map is a noncontractive two-cycle.


@pytest.mark.parametrize("steps", [1, 3])
def test_anderson_snapshot_enforces_exact_evaluation_budget(steps):
    put, rho, _, _, _ = _state()
    calls = []

    def evaluate(held):
        calls.append(np.asarray(held))
        return held + 0.01, None

    with pytest.raises(ValueError, match="GATE dynsig_collision_snapshot_convergence"):
        collision_snapshot_state(evaluate, put(np.zeros(rho.shape, complex)),
                                 steps=steps, tol_ha=1e-12, acceleration="anderson")
    assert len(calls) == steps


def test_anderson_snapshot_refuses_nonfinite_before_an_extra_evaluation():
    put, rho, _, _, _ = _state()
    calls = []

    def evaluate(held):
        calls.append(held)
        return held * np.nan, None

    with pytest.raises(ValueError, match="GATE dynsig_collision_snapshot_convergence"):
        collision_snapshot_state(evaluate, put(np.zeros(rho.shape, complex)),
                                 steps=4, acceleration="anderson")
    assert len(calls) == 1


@pytest.mark.parametrize("slope", [1.2, np.nan])
def test_snapshot_refuses_unconverged_or_nonfinite_updates(slope):
    put, rho, _, _, _ = _state()
    initial = put(np.zeros(rho.shape, complex))
    before = np.asarray(initial).copy()
    with pytest.raises(ValueError, match="GATE dynsig_collision_snapshot_convergence"):
        collision_snapshot_state(lambda h: (0.01 + slope * h, None), initial,
                                 steps=4, tol_ha=1e-12, mixing=0.5)
    np.testing.assert_array_equal(initial, before)


def test_snapshot_full_matrix_static_counterterm_cancels_once():
    from tdagw.collision import hermitian_update
    put, rho, reference, _, _ = _state()
    static_eq = put(np.broadcast_to(np.array([[-0.1, 0.02j], [-0.02j, 0.04]]), rho.shape))
    delta_static = 0.2 * (rho - reference[None])
    baseline = static_eq + 0.03 * jnp.eye(2)

    def evaluate(held):
        target = hermitian_update(baseline + delta_static, baseline,
                                  static_eq + delta_static, static_eq)
        return target, None

    held, _, receipt = collision_snapshot_state(evaluate, jnp.zeros_like(rho))
    np.testing.assert_allclose(held, 0.0, atol=1e-14)
    assert receipt["iterations"] == 1


@pytest.mark.parametrize("controls", [{"steps": 0}, {"tol_ha": 0.0}, {"mixing": 1.1},
                                     {"acceleration": "invalid"}])
def test_snapshot_invalid_controls_refused_before_sweep(controls):
    def forbidden(_):
        raise AssertionError("invalid controls must not evaluate a snapshot")
    with pytest.raises(ValueError, match="GATE dynsig_collision_snapshot_controls"):
        collision_snapshot_state(forbidden, jnp.zeros((1, 1, 2, 2)), **controls)

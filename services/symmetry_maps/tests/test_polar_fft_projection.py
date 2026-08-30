"""Polar FFT-field projection: affine, antiunitary, and device contracts."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import symmetry_maps.orbit_syms as orbit_syms
from symmetry_maps import fft_grid_pullback_perm, project_polar_fft_field


def _symmetry(spatial, rotations, translations, *, trs_allowed):
    spatial = np.asarray(spatial, dtype=np.int32)
    rotations = np.asarray(rotations, dtype=np.float64)
    return SimpleNamespace(
        sym_matrices=spatial,
        translations=np.asarray(translations, dtype=np.float64),
        R_cart_forward=np.concatenate([rotations, -rotations], axis=0),
        trs_allowed=bool(trs_allowed),
    )


def _reference(field, sym):
    """Independent, deliberately eager definition of the documented action."""
    value = np.asarray(field)
    n_spatial = int(sym.sym_matrices.shape[0])
    n_rows = 2 * n_spatial if sym.trs_allowed else n_spatial
    pullback = fft_grid_pullback_perm(
        sym.sym_matrices, sym.translations, value.shape[-3:])
    flat = value.reshape(3, -1)
    out = np.zeros_like(flat, dtype=np.result_type(value.dtype, np.float64))
    for row in range(n_rows):
        operand = np.conj(flat) if row >= n_spatial else flat
        out += sym.R_cart_forward[row] @ operand[:, pullback[row % n_spatial]]
    return (out / float(n_rows)).reshape(value.shape)


def _device_put_replicated(value):
    mesh = Mesh(np.asarray(jax.devices()), ("x",))
    sharding = NamedSharding(mesh, P(None, None, None, None))
    return jax.make_array_from_callback(
        value.shape, sharding, lambda index: value[index])


@pytest.mark.parametrize("trs_allowed", [False, True])
@pytest.mark.parametrize("complex_field", [False, True])
def test_device_matches_host_for_nonsymmorphic_polar_field(
        trs_allowed, complex_field):
    """The glide pullback, polar rotation and optional Θ share one algebra."""
    spatial = np.stack([
        np.eye(3, dtype=np.int32),
        np.diag([1, 1, -1]).astype(np.int32),
    ])
    rotations = spatial.astype(np.float64)
    translations = np.asarray([[0.0, 0.0, 0.0], [np.pi, 0.0, 0.0]])
    sym = _symmetry(
        spatial, rotations, translations, trs_allowed=trs_allowed)
    rng = np.random.default_rng(20260830)
    value = rng.standard_normal((3, 4, 4, 2))
    if complex_field:
        value = value + 1j * rng.standard_normal(value.shape)

    expected = _reference(value, sym)
    host = project_polar_fft_field(value, sym)
    device_input = _device_put_replicated(value)
    device = project_polar_fft_field(device_input, sym)

    np.testing.assert_allclose(host.field, expected, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(
        np.asarray(device.field), expected, rtol=2e-14, atol=2e-14)
    assert device.field.sharding == device_input.sharding
    assert device.relative_residual <= device.relative_residual_tolerance
    assert host.relative_residual <= host.relative_residual_tolerance


def test_time_reversal_is_antilinear_but_has_only_one_time_odd_sign():
    """ΘJ=-conj(J): the -I row owns parity and conjugation owns anti-linearity."""
    identity = np.eye(3, dtype=np.int32)[None]
    translations = np.zeros((1, 3))
    value = (
        np.arange(72, dtype=np.float64).reshape(3, 3, 4, 2)
        + 1j * np.linspace(-0.7, 0.9, 72).reshape(3, 3, 4, 2))

    sym_off = _symmetry(
        identity, identity, translations, trs_allowed=False)
    sym_on = _symmetry(
        identity, identity, translations, trs_allowed=True)
    np.testing.assert_array_equal(
        project_polar_fft_field(value, sym_off).field, value)
    np.testing.assert_allclose(
        project_polar_fft_field(value, sym_on).field,
        1j * np.imag(value), rtol=0.0, atol=2e-15)


def test_two_component_orbital_current_is_polar_not_axial():
    """An nspinor=2 plane wave discriminates polar current from magnetization."""
    grid = (4, 2, 2)
    x = np.arange(grid[0], dtype=np.float64)[:, None, None] / grid[0]
    phase = np.exp(2j * np.pi * x) * np.ones(grid, dtype=np.complex128)
    psi = np.stack([phase, (0.3 + 0.2j) * phase])
    # Im psi^dagger grad(psi) for this two-component plane wave.  It is a
    # polar orbital current; it is not the axial Pauli spin density.
    current = np.zeros((3, *grid), dtype=np.float64)
    current[0] = 2.0 * np.pi * np.sum(np.abs(psi) ** 2, axis=0)

    inversion = -np.eye(3, dtype=np.int32)
    spatial = np.stack([np.eye(3, dtype=np.int32), inversion])
    sym = _symmetry(
        spatial, spatial, np.zeros((2, 3)), trs_allowed=False)
    projected = project_polar_fft_field(current, sym)
    np.testing.assert_allclose(projected.field, 0.0, rtol=0.0, atol=1e-14)
    assert projected.relative_movement == pytest.approx(1.0)


def test_scalar_grid_keeps_the_incumbent_shape_refusal_on_host_and_device():
    """A scalar density has a different symmetry owner, never a guessed axis."""
    sym = _symmetry(
        np.eye(3, dtype=np.int32)[None],
        np.eye(3, dtype=np.float64)[None],
        np.zeros((1, 3)), trs_allowed=False)
    scalar = np.ones((4, 4, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="field must have shape"):
        project_polar_fft_field(scalar, sym)
    with pytest.raises(ValueError, match="field must have shape"):
        project_polar_fft_field(jax.device_put(scalar), sym)


def test_nonfinite_refusal_is_identical_on_host_and_device():
    """Fusing the device predicate into projection must not weaken refusal."""
    sym = _symmetry(
        np.eye(3, dtype=np.int32)[None],
        np.eye(3, dtype=np.float64)[None],
        np.zeros((1, 3)), trs_allowed=False)
    field = np.ones((3, 4, 4, 2), dtype=np.float64)
    field[1, 2, 1, 0] = np.nan
    with pytest.raises(ValueError, match="contains non-finite"):
        project_polar_fft_field(field, sym)
    with pytest.raises(ValueError, match="contains non-finite"):
        project_polar_fft_field(jax.device_put(field), sym)


def test_static_affine_plan_builds_once_and_never_aliases_metadata(monkeypatch):
    """Warm SC calls reuse one pullback; changed grid/group gets a new plan."""
    orbit_syms._POLAR_FIELD_PLAN_CACHE.clear()
    orbit_syms._POLAR_FIELD_JIT_CACHE.clear()
    incumbent = orbit_syms.fft_grid_pullback_perm
    calls = []

    def _counted(*args, **kwargs):
        calls.append(tuple(int(n) for n in np.asarray(args[2])))
        return incumbent(*args, **kwargs)

    monkeypatch.setattr(orbit_syms, "fft_grid_pullback_perm", _counted)
    identity = np.eye(3, dtype=np.int32)[None]
    sym_identity = _symmetry(
        identity, identity, np.zeros((1, 3)), trs_allowed=False)
    field_a = np.ones((3, 4, 4, 2), dtype=np.float64)
    project_polar_fft_field(field_a, sym_identity)
    project_polar_fft_field(2.0 * field_a, sym_identity)
    assert calls == [(4, 4, 2)]

    field_b = np.ones((3, 6, 4, 2), dtype=np.float64)
    project_polar_fft_field(field_b, sym_identity)
    assert calls == [(4, 4, 2), (6, 4, 2)]

    inversion = -np.eye(3, dtype=np.int32)
    spatial = np.stack([np.eye(3, dtype=np.int32), inversion])
    sym_inversion = _symmetry(
        spatial, spatial, np.zeros((2, 3)), trs_allowed=False)
    project_polar_fft_field(field_a, sym_inversion)
    assert calls == [(4, 4, 2), (6, 4, 2), (4, 4, 2)]
    assert len(orbit_syms._POLAR_FIELD_PLAN_CACHE) == 3
    assert all(len(token) == 32
               for token in orbit_syms._POLAR_FIELD_PLAN_CACHE)

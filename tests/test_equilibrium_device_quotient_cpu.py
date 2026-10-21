"""Tiny numerical diagnostics of the device quotient; not GPU qualification.

The independent oracle lives in two well-conditioned latent states. Their
nearly parallel fitting columns give the SAME ill-conditioned H for a weak
output and an important output. A metric spectrum cannot distinguish them.
"""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture
def device_quotient(monkeypatch):
    """Exercise production orchestration with only tiny CPU service stand-ins."""
    import distrib_la
    import gw.shared_pole_constructor as owner
    import gw.shared_pole_execution as execution

    def product(a, b, *, transa='N', transb='N'):
        if transa != 'N':
            a = jnp.swapaxes(a, -1, -2)
            if transa == 'C':
                a = a.conj()
        if transb != 'N':
            b = jnp.swapaxes(b, -1, -2)
            if transb == 'C':
                b = b.conj()
        return jnp.matmul(a, b)

    def plan(op, *args, **kwargs):
        assert op == 'eigh'
        return SimpleNamespace(batched=lambda a, **kw: jnp.linalg.eigh(a))

    monkeypatch.setattr(distrib_la, 'plan', plan)
    monkeypatch.setattr(execution, 'face_matmul', lambda mesh: product)
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ('x', 'y'))

    def reduce(g, h, output, *, output_gram=None, **kwargs):
        with jax.enable_x64(True):
            gram = output.conj().T @ output if output_gram is None else output_gram
            arrays = [jnp.asarray(a[None], dtype=jnp.complex128)
                      for a in (g, h, gram)]
            return owner.reduce_equilibrium_pencil_device(*arrays, mesh_xy=mesh,
                max_order=len(g), rank_tolerance=1e-7, gamma_ry=.03, **kwargs)
    return reduce


def _case(output_strength):
    # H's condition number is about 4e10, entirely due to fitting coordinates.
    x = np.asarray([[1., 1.], [0., 1e-5]], complex)
    latent_g = np.diag([1., -1.])
    c = np.asarray([[1., 0.], [.2j, output_strength], [.3, .4j*output_strength]])
    return x.conj().T @ latent_g @ x, x.conj().T @ x, c @ x, c


def _factors(model, output):
    poles = np.asarray(model['poles_ry'])
    left = np.asarray(model['output_map']).reshape(output.shape[1], -1)[:, :len(poles)]
    right = np.asarray(model['input_map']).reshape(output.shape[1], -1)[:, :len(poles)]
    return output @ left, output @ right, poles


def _value(model, output, z):
    left, right, poles = _factors(model, output)
    return (left / (z-poles)) @ right.conj().T


def _exact(c, z):
    return (c / np.asarray([z+.03j-1, -(z+.03j)-1])) @ c.conj().T


def _column_response(value):
    """Complex band-column contraction of a fixed simple Green-function slice.

    This checks sensitivity to imaginary/offdiagonal W errors. It is an
    oracle diagnostic, not a contour-deformation Sigma error certificate.
    """
    vertices = np.asarray([[1., .2j, .3], [.4j, .8, -.1j]])
    energies = [.23+.11j, -.37+.08j]
    return np.column_stack([(vertices @ value(z) @ vertices.conj().T)[:, j]
                            for j, z in enumerate(energies)])


@pytest.mark.parametrize('strength,negligible', [(1e-4, True), (1., False)])
def test_same_small_metric_mode_can_be_negligible_or_physically_important(
        device_quotient, strength, negligible):
    g, h, output, c = _case(strength)
    model = device_quotient(g, h, output)
    assert len(model['poles_ry']) == 1
    assert np.linalg.cond(h) > 1e10
    exact = lambda z: _exact(c, z)
    approx = lambda z: _value(model, output, z)
    w_error = max(np.linalg.norm(approx(z)-exact(z)) / np.linalg.norm(exact(z))
                  for z in [.23+.11j, -.37+.08j, 1.2+.19j])
    sigma_error = np.linalg.norm(_column_response(approx)-_column_response(exact))
    if negligible:
        assert w_error < 1e-7
        assert sigma_error < 1e-7
    else:
        # Algebraically identical rank/metric evidence requires rejection by
        # the caller's original full-W checks, not silent physical admission.
        assert w_error > .05
        assert sigma_error > 1e-3
    assert model['receipt']['sigma_accuracy_certified'] is False
    assert model['receipt']['physical_accuracy_certified'] is False
    assert model['receipt']['approximation_requires_full_W_validation'] is True


def test_held_common_span_uses_current_metric(device_quotient):
    g, h, output, _ = _case(1e-4)
    first = device_quotient(g, h, output)
    current_h = 1.7*h
    current = device_quotient(g, current_h, output, fixed_spaces=first['spaces'])
    space = np.asarray(first['spaces']['right']).reshape(2, -1)
    for z in [.23+.11j, -.37+.08j]:
        target = output @ space @ np.linalg.solve(
            (z+.03j)*(space.conj().T @ g @ space)
            - space.conj().T @ current_h @ space, space.conj().T @ output.conj().T)
        np.testing.assert_allclose(_value(current, output, z), target, rtol=2e-10, atol=2e-10)
        assert np.linalg.norm(_value(current, output, z)-_value(first, output, z)) > .1
    np.testing.assert_array_equal(current['spaces']['right'], first['spaces']['right'])
    np.testing.assert_array_equal(current['spaces']['left'], current['spaces']['right'])


def test_cut_drops_whole_near_degenerate_metric_block(device_quotient):
    # A Fourier basis gives constant H diagonals, so diagonal equilibration
    # does not move the proposed cut away from this intentionally split pair.
    u = np.exp(2j*np.pi*np.outer(np.arange(3), np.arange(3))/3) / np.sqrt(3)
    eigen = np.asarray([1e-7*(1-2e-8), 1e-7*(1+2e-8), 1.])
    h = (u*eigen) @ u.conj().T
    model = device_quotient(np.eye(3), h, np.eye(3), multiplet_tolerance=1e-6)
    assert len(model['poles_ry']) == 1


def test_common_equilibration_preserves_complex_coordinate_units(device_quotient):
    g, h, output, _ = _case(1e-4)
    original = device_quotient(g, h, output)
    transform = np.diag(np.asarray([.125, 32.])*np.exp(1j*np.asarray([.31, -.77])))
    changed_output = output @ transform
    changed = device_quotient(transform.conj().T @ g @ transform,
        transform.conj().T @ h @ transform, changed_output)
    for z in [.23+.11j, -.37+.08j]:
        np.testing.assert_allclose(_value(changed, changed_output, z),
            _value(original, output, z), rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize('bad_gram', [
    np.diag([1., np.nan]),
    np.diag([1., -.1]),
    np.asarray([[1., .2j], [.2j, 1.]]),
])
def test_invalid_output_evidence_cannot_admit_unresolved_mu(device_quotient, bad_gram):
    with pytest.raises(ValueError, match='output|finite|Hermitian'):
        device_quotient(np.diag([1., 0.]), np.eye(2), np.eye(2), output_gram=bad_gram)


def test_resolved_negative_metric_is_not_projected_to_psd(device_quotient):
    with pytest.raises(ValueError, match='negative H'):
        device_quotient(np.eye(2), np.diag([1., -.1]), np.eye(2))


@pytest.mark.parametrize('unit', [1., 1e8])
def test_live_constant_output_refusal_is_invariant_to_state_units(device_quotient, unit):
    # These are the SAME two-state transfer after an invertible coordinate
    # congruence. The second state contributes a constant -1 for either unit.
    # Raw O.H O changes by 1e16; H-normalized output weights do not change.
    with pytest.raises(ValueError, match='constant feedthrough'):
        device_quotient(np.diag([unit**2, 0.]), np.diag([unit**2, 1.]),
            np.diag([unit, 1.]))


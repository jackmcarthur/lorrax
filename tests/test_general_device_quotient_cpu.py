"""Tiny general-transfer oracles; CPU execution does not qualify CUDA/P4."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture
def general_device(monkeypatch):
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

    def reduce(e, a, **kwargs):
        with jax.enable_x64(True):
            return owner.reduce_general_pencil_device(
                jnp.asarray(e[None], dtype=jnp.complex128),
                jnp.asarray(a[None], dtype=jnp.complex128), mesh_xy=mesh,
                max_order=len(e), rank_tolerance=1e-8, **kwargs)
    return reduce


def _case():
    poles = np.asarray([-.8-.03j, .3+.17j, 1.4-.04j])
    v = np.asarray([[1., .8j, .3], [.2, 1., .4j], [.1j, .3, 1.]])
    generator = (v*poles) @ np.linalg.inv(v)
    e = np.diag([.5, 2., 1.3]).astype(complex)
    a = e @ generator
    b = np.asarray([[1., .2j], [.3j, 1.], [.4, -.3j]])
    c = np.asarray([[.2j, 1., .3], [1., .4j, -.2]])
    return e, a, b, c, poles


def _value(model, b, c, z):
    poles = np.asarray(model['poles_ry'])
    output = np.asarray(model['output_map']).reshape(c.shape[1], -1)[:, :len(poles)]
    inputs = np.asarray(model['input_map']).reshape(b.shape[0], -1)[:, :len(poles)]
    return ((c @ output)/(z-poles)) @ (inputs.conj().T @ b)


def _check_transfer(model, e, a, b, c):
    for z in [.23+.31j, -.41+.46j, 1.8+.29j]:
        expected = c @ np.linalg.solve(z*e-a, b)
        np.testing.assert_allclose(_value(model, b, c, z), expected, rtol=3e-10, atol=3e-10)


def test_complex_nonnormal_transfer_and_genuine_gain_are_preserved(general_device):
    e, a, b, c, poles = _case()
    model = general_device(e, a)
    _check_transfer(model, e, a, b, c)
    np.testing.assert_allclose(np.sort_complex(model['poles_ry']), np.sort_complex(poles),
                               rtol=2e-11, atol=2e-11)
    assert np.max(np.asarray(model['poles_ry']).imag) > .16
    assert model['receipt']['physical_accuracy_certified'] is False
    assert model['receipt']['sigma_accuracy_certified'] is False


def test_independent_equation_unit_changes_preserve_full_transfer(general_device):
    e, a, b, c, _ = _case()
    dl = np.diag([1e3, .02, 7.])
    dr = np.diag([.03, 50., .2])
    es, ass, bs, cs = dl@e@dr, dl@a@dr, dl@b, c@dr
    model = general_device(es, ass)
    _check_transfer(model, es, ass, bs, cs)
    original = general_device(e, a)
    for z in [.23+.31j, -.41+.46j]:
        np.testing.assert_allclose(_value(model, bs, cs, z), _value(original, b, c, z),
                                   rtol=3e-10, atol=3e-10)


def test_held_spaces_project_current_pencil_without_reusing_old_modes(general_device):
    e, a, b, c, _ = _case()
    first = general_device(e, a)
    current_e = e+.07*np.eye(3)
    current_a = a+.11*e
    current = general_device(current_e, current_a, fixed_spaces=first['spaces'])
    _check_transfer(current, current_e, current_a, b, c)
    for side in ('left', 'right'):
        np.testing.assert_array_equal(current['spaces'][side], first['spaces'][side])
    assert np.linalg.norm(_value(current, b, c, .23+.31j)
                          - _value(first, b, c, .23+.31j)) > .1


def test_singular_current_held_descriptor_refuses_without_rank_drop(general_device):
    e, a, _, _, _ = _case()
    first = general_device(e, a)
    with pytest.raises(ValueError, match='held|descriptor|balance'):
        general_device(np.zeros_like(e), a, fixed_spaces=first['spaces'])

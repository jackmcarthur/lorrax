"""Independent causal scalar-charge sphere transport and typed-image guards."""
from dataclasses import replace

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.collectives import gather_to_host

from gw.mixed_basis_pair_convolution import (
    SphereSet, SphereTransport, transport_sphere_response,
)


@pytest.fixture
def mesh():
    devices = np.asarray(jax.devices())
    shape = (2, 2) if jax.process_count() == 4 and devices.size >= 4 else (1, 1)
    return Mesh(devices[:np.prod(shape)].reshape(shape), ('x', 'y'))


def face(value, mesh, *, dtype=np.complex128):
    return jax.device_put(np.asarray(value, dtype), NamedSharding(mesh, P(None, 'x', 'y')))


def small_host(value):
    # This independent toy oracle permits at most 2 KiB; no physical response
    # or production face is gathered for a verification.
    assert value.size * value.dtype.itemsize <= 2048
    return gather_to_host(value)


def plant():
    """Two unequal spheres, wrapped children, nontrivial translations and TR rows."""
    parent = SphereSet(
        np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [77, 78, 79]],
                  [[0, 0, 0], [-1, 0, 1], [77, 78, 79], [77, 78, 79]]]),
        np.array([3, 2]), np.array([[.25, .125, 0.], [.125, 0., .25]]))
    rows = np.array([0, 1, 0, 1, 0])
    anti = np.array([False, True, True, False, False])
    cycle = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    rotations = np.array([np.eye(3, dtype=int), cycle, cycle.T,
                          -np.eye(3, dtype=int), cycle])
    sources = np.array([[2, 0, 1, -47], [1, 0, -47, -47],
                        [1, 2, 0, -47], [0, 1, -47, -47], [1, 0, 2, -47]])
    images = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1], [1, 1, 0], [-1, 0, 1]])
    child_g = np.full((5, 4, 3), 53, dtype=int)
    child_frac = np.zeros((5, 3))
    translations = np.array([[.3, .2, .7], [.4, .9, .1], [.8, .5, .2],
                             [.7, .4, .6], [.2, .6, .9]])
    phases = np.full((5, 4), complex(np.nan, np.nan))
    for child, owner in enumerate(rows):
        sign = -1 if anti[child] else 1
        count = parent.ngk[owner]
        child_frac[child] = sign * rotations[child].T @ parent.frac[owner] - images[child]
        child_g[child, :count] = (sign * (rotations[child].T @
            parent.gvecs[owner, sources[child, :count]].T).T + images[child])
        parent_k = parent.frac[owner] + parent.gvecs[owner, sources[child, :count]]
        phases[child, :count] = np.exp(-1j * (parent_k @ translations[child]))
    children = SphereSet(child_g, parent.ngk[rows], child_frac)
    transport = SphereTransport(rows, anti, np.ones((5, 1, 1)), sources,
        phases, 2, parent, rotations, translations)
    rng = np.random.default_rng(497231)
    values = np.zeros((2, 4, 4), np.complex128)
    # Deliberately neither symmetric nor Hermitian at one complex frequency.
    for owner, count in enumerate(parent.ngk):
        values[owner, :count, :count] = (rng.normal(size=(count, count)) +
            1j * rng.normal(size=(count, count))) / complex(.7, .25)
    return values, transport, children


def literal(values, transport, children, selected, partner):
    result = np.zeros((len(selected), children.width, children.width), complex)
    for out, child in enumerate(selected):
        source = partner if transport.anti[child] else values
        owner = transport.row[child]
        for left in range(children.ngk[child]):
            for right in range(children.ngk[child]):
                a = transport.phase[child, left]
                b = transport.phase[child, right]
                if transport.anti[child]:
                    a, b = a.conjugate(), b.conjugate()
                result[out, left, right] = (a * source[owner,
                    transport.src[child, left], transport.src[child, right]] * b.conjugate())
    return result


def test_complex_same_site_transpose_transport_and_arbitrary_child_order(mesh):
    values, transport, children = plant()
    selected = np.array([4, 0, 2, 1])
    partner = values.swapaxes(-1, -2).copy()
    actual = transport_sphere_response(face(values, mesh), transport, children,
        mesh=mesh, child_rows=selected, transposed_parent_same_z=face(partner, mesh))
    expected = literal(values, transport, children, selected, partner)
    host = small_host(actual)
    np.testing.assert_allclose(host, expected, rtol=3e-13, atol=3e-13)
    assert actual.sharding == NamedSharding(mesh, P(None, 'x', 'y'))
    for row, child in enumerate(selected):
        count = children.ngk[child]
        assert np.count_nonzero(host[row, count:]) == 0
        assert np.count_nonzero(host[row, :, count:]) == 0
    # Each incorrect causal sheet/endpoint convention has a sizable witness.
    wrong_partner = literal(values, transport, children, selected, values.conj())
    wrong_no_transpose = literal(values, transport, children, selected, values)
    assert np.max(abs(expected - wrong_partner)) > .1
    assert np.max(abs(expected - wrong_no_transpose)) > .1


def test_action_on_vectors_and_causal_frequency_scalar_remains_unconjugated(mesh):
    values, transport, children = plant()
    selected = np.array([1, 2, 3])
    scalar = complex(.3, -.8)
    actual = transport_sphere_response(face(values * scalar, mesh), transport, children,
        mesh=mesh, child_rows=selected,
        transposed_parent_same_z=face(values.swapaxes(-1, -2) * scalar, mesh))
    expected = literal(values, transport, children, selected, values.swapaxes(-1, -2))
    host = small_host(actual)
    np.testing.assert_allclose(host, scalar * expected, rtol=3e-13, atol=3e-13)
    vector = np.array([.3 + .2j, -.8 + .5j, .4 - .7j, 0.])
    for out, child in enumerate(selected):
        count = children.ngk[child]
        transformed = np.zeros(4, complex)
        for i in range(count):
            for j in range(count):
                transformed[i] += scalar * expected[out, i, j] * vector[j]
        np.testing.assert_allclose(host[out] @ vector, transformed,
                                   rtol=3e-13, atol=3e-13)
    assert np.max(abs(scalar * expected - scalar.conjugate() * expected)) > .1


def test_unitary_rows_do_not_require_a_partner(mesh):
    values, transport, children = plant()
    selected = np.array([3, 4, 0])
    actual = transport_sphere_response(face(values, mesh), transport, children,
        mesh=mesh, child_rows=selected)
    np.testing.assert_allclose(small_host(actual), literal(values, transport, children, selected, None),
                               rtol=3e-13, atol=3e-13)


def test_antiunitary_row_requires_explicit_same_site_partner(mesh):
    values, transport, children = plant()
    with pytest.raises(ValueError, match='transposed_parent_same_z'):
        transport_sphere_response(face(values, mesh), transport, children,
            mesh=mesh, child_rows=np.array([1]))


@pytest.mark.parametrize('rows', [np.array([], int), np.array([0, 0]),
    np.array([-1]), np.array([5]), np.array([0.]), np.array([True])])
def test_bad_child_selection_is_refused(mesh, rows):
    values, transport, children = plant()
    with pytest.raises(ValueError, match='unique bounded integer'):
        transport_sphere_response(face(values, mesh), transport, children,
            mesh=mesh, child_rows=rows)


@pytest.mark.parametrize('fault', ['source', 'integer_image', 'count', 'phase',
                                  'ghost', 'nonfinite', 'spin', 'dtype', 'partner_shape'])
def test_typed_image_and_input_guards_refuse_physical_faults(mesh, fault):
    values, transport, children = plant()
    partner = values.swapaxes(-1, -2).copy()
    if fault == 'source':
        src = transport.src.copy(); src[0, 1] = src[0, 0]
        transport = replace(transport, src=src)
    elif fault == 'integer_image':
        g = children.gvecs.copy(); g[0, 0, 0] += 1
        children = replace(children, gvecs=g)
    elif fault == 'count':
        counts = children.ngk.copy(); counts[0] -= 1
        children = replace(children, ngk=counts)
    elif fault == 'phase':
        phase = transport.phase.copy(); phase[0, 0] *= 1.01
        transport = replace(transport, phase=phase)
    elif fault == 'ghost':
        values[0, 3, 0] = 1e-30
    elif fault == 'nonfinite':
        values[0, 0, 0] = np.nan
    elif fault == 'spin':
        transport = replace(transport, spin=np.tile(np.eye(2), (5, 1, 1)))
    elif fault == 'dtype':
        values = values.astype(np.complex64)
    elif fault == 'partner_shape':
        partner = np.pad(partner, ((0, 0), (0, 2), (0, 2)))
    with pytest.raises(ValueError):
        transport_sphere_response(face(values, mesh, dtype=values.dtype), transport,
            children, mesh=mesh, child_rows=np.array([0, 1]),
            transposed_parent_same_z=face(partner, mesh))

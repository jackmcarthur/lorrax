"""Runtime one-dimensional density masks and compact abstract lowering."""
import re

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from gw.plane_wave_lehmann import _density_face_guard, OrderedLehmannPair


@pytest.fixture
def mesh():
    return Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ('x', 'y'))


def array(value, mesh, spec):
    return jax.device_put(np.asarray(value), NamedSharding(mesh, spec))


def guarded(density, pairs, endpoints, mesh):
    return np.asarray(_density_face_guard(mesh)(
        array(density, mesh, P(None, 'x', 'y')),
        array(pairs, mesh, P()), array(endpoints, mesh, P())))


def test_changing_pair_and_endpoint_masks_are_runtime_inputs(mesh):
    density = np.zeros((1, 4, 6), np.complex128)
    density[0, 2, 3] = 2. + 1j
    pairs = np.ones(4, bool); endpoints = np.ones(6, bool)
    kernel = _density_face_guard(mesh)
    np.testing.assert_array_equal(guarded(density, pairs, endpoints, mesh), [0., 1.])
    pairs[2] = False
    np.testing.assert_allclose(guarded(density, pairs, endpoints, mesh), [np.sqrt(5.), 1.], rtol=3e-15, atol=0.)
    pairs[2] = True; endpoints[3] = False
    np.testing.assert_allclose(guarded(density, pairs, endpoints, mesh), [np.sqrt(5.), 1.], rtol=3e-15, atol=0.)
    endpoints[3] = True
    np.testing.assert_array_equal(guarded(density, pairs, endpoints, mesh), [0., 1.])
    assert _density_face_guard(mesh) is kernel


@pytest.mark.parametrize('where', ['physical_nan', 'ghost_nan', 'physical_inf', 'tiny_ghost'])
def test_nonfinite_and_nonzero_ghost_guards_match_independent_mask(mesh, where):
    density = np.zeros((1, 4, 6), np.complex128)
    pairs = np.asarray([True, True, False, True])
    endpoints = np.asarray([True, False, True, True, False, True])
    i, j = (1, 2) if where.startswith('physical') else (2, 1)
    density[0, i, j] = {'physical_nan': np.nan, 'ghost_nan': np.nan,
        'physical_inf': np.inf, 'tiny_ghost': 1e-30}[where]
    actual = guarded(density, pairs, endpoints, mesh)
    expected = np.max(abs(density[0][~(pairs[:, None] & endpoints[None, :])]))
    if np.isnan(expected):
        assert np.isnan(actual[0])
    else:
        np.testing.assert_allclose(actual[0], expected, rtol=3e-15, atol=0.)
    assert bool(actual[1]) == np.isfinite(density).all()


@pytest.mark.parametrize('change', ['pairs', 'endpoints'])
def test_public_factory_refuses_changed_masks_after_previous_valid_call(mesh, change):
    rng = np.random.default_rng(9121776)
    density = (rng.normal(size=(1, 4, 6)) + 1j*rng.normal(size=(1, 4, 6))) * .1
    pairs = np.ones(4, bool); endpoints = np.ones(6, bool)
    de = np.asarray([-.7, -1.3, .9, 1.7]); df = np.asarray([.5, .9, -.3, -.2])
    def build(value):
        return OrderedLehmannPair.from_density_face(
            array(value, mesh, P(None, 'x', 'y')), de, df, pairs,
            mesh=mesh, physical_prefactor=1/np.sqrt(512.), endpoint_valid=endpoints,
            normalization='raw centroid 1/sqrtNk', panel_bytes=4096)
    first = build(density)
    if change == 'pairs': pairs[2] = False
    else: endpoints[3] = False
    with pytest.raises(ValueError, match='ghost'):
        build(density)
    cleaned = density.copy()
    cleaned[:, ~pairs] = 0.; cleaned[..., ~endpoints] = 0.
    second = build(cleaned)
    z = np.asarray([.7+.25j])
    expected = np.zeros((1, 1, 6, 6), complex)
    for t in np.flatnonzero(pairs):
        for i in range(6):
            for j in range(6):
                expected[0, 0, i, j] += (cleaned[0, t, i] * cleaned[0, t, j].conj()
                    * df[t] / (z[0]+de[t]) / np.sqrt(512.))
    np.testing.assert_allclose(second.evaluate(z), expected, rtol=3e-12, atol=3e-13)
    assert first.receipt['physical_pair_count'] == 4
    assert second.receipt['physical_pair_count'] == int(pairs.sum())


def test_large_abstract_face_lowering_has_dynamic_1d_masks_and_no_outer_literal(mesh):
    T, M = 1_536_000, 1776
    density = jax.ShapeDtypeStruct((1, T, M), np.complex128,
        sharding=NamedSharding(mesh, P(None, 'x', 'y')))
    pairs = jax.ShapeDtypeStruct((T,), np.bool_, sharding=NamedSharding(mesh, P()))
    endpoints = jax.ShapeDtypeStruct((M,), np.bool_, sharding=NamedSharding(mesh, P()))
    # Lower abstract inputs only: no T*M host mask or device face is allocated.
    ir = str(_density_face_guard(mesh).lower(density, pairs, endpoints).compiler_ir(dialect='stablehlo'))
    assert re.search(r'%arg\d+: tensor<1536000xi1>', ir)
    assert re.search(r'%arg\d+: tensor<1776xi1>', ir)
    constants = [line for line in ir.splitlines() if 'stablehlo.constant' in line]
    assert constants and max(map(len, constants)) < 512
    assert len(ir) < 32768

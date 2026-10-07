"""Independent runtime-map controls for cached all-P response movers."""
import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.collectives import gather_to_host
from gw.mixed_basis_pair_convolution import _sphere_response_programs


@pytest.fixture
def mesh():
    devices = np.asarray(jax.devices())
    shape = (2, 2) if jax.process_count() == 4 and devices.size >= 4 else (1, 1)
    return Mesh(devices[:np.prod(shape)].reshape(shape), ('x', 'y'))


@pytest.mark.parametrize('axis,program_index', [(1, 4), (2, 5)])
def test_cached_mover_keeps_changing_per_parent_maps_dynamic(mesh, axis, program_index):
    values = (np.arange(128).reshape(2, 8, 8) * (1. + .25j)).astype(np.complex128)
    value = jax.device_put(values, NamedSharding(mesh, P(None, 'x', 'y')))
    programs = _sphere_response_programs(mesh)
    mover = programs[program_index]
    assert _sphere_response_programs(mesh) is programs
    assert _sphere_response_programs(mesh)[program_index] is mover
    for maps in (np.tile(np.arange(8), (2, 1)),
                 np.array([[7, 0, 6, 1, 5, 2, 4, 3], [1, 3, 5, 7, 0, 2, 4, 6]]),
                 np.tile(np.arange(7, -1, -1), (2, 1))):
        source = jax.device_put(maps.astype(np.int32), NamedSharding(mesh, P()))
        candidate = mover(value, source)
        assert candidate.sharding == value.sharding
        # An explicitly bounded 2KiB toy oracle may be gathered.
        assert candidate.size * candidate.dtype.itemsize <= 2048
        indices = maps[:, :, None] if axis == 1 else maps[:, None, :]
        expected = np.take_along_axis(values, indices, axis=axis)
        np.testing.assert_array_equal(gather_to_host(candidate), expected)


def test_cached_mover_lowering_takes_maps_as_inputs(mesh):
    face = NamedSharding(mesh, P(None, 'x', 'y'))
    replicated = NamedSharding(mesh, P())
    value = jax.ShapeDtypeStruct((2, 8, 8), np.complex128, sharding=face)
    source = jax.ShapeDtypeStruct((2, 8), np.int32, sharding=replicated)
    for mover in _sphere_response_programs(mesh)[4:]:
        ir = str(mover.lower(value, source).compiler_ir(dialect='stablehlo'))
        assert 'tensor<2x8xi32>' in ir.split('func.func public @main', 1)[1].split('{', 1)[0]
        assert len(ir) < 65536

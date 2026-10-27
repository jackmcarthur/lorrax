"""ELPA tail swaps move exactly one owned line through the shared reducer."""
import math
import numpy as np
import pytest

jax = pytest.importorskip('jax')
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from distrib_la._elpa import _swap


@pytest.mark.parametrize('dtype', [np.float64, np.complex128])
@pytest.mark.parametrize('axis', [0, 1])
def test_dynamic_owned_tail_swap(dtype, axis):
    p = math.isqrt(jax.device_count())
    if p < 3 or p*p != jax.device_count():
        pytest.skip('requires a square virtual mesh with at least three ranks per axis')
    mesh = Mesh(np.asarray(jax.devices()).reshape(p, p), ('x', 'y'))
    n = 13*p
    ij = np.arange(n*n).reshape(n, n)
    value = np.sin(.17*(ij+1))
    if np.dtype(dtype).kind == 'c':
        value = value+1j*np.cos(.13*(ij+3))
    value = value.astype(dtype)
    face, rep = NamedSharding(mesh, P('x','y')), NamedSharding(mesh, P())
    operand = jax.device_put(value, face)
    fn = jax.jit(lambda a, j: _swap(a, j, axis, mesh=mesh),
                 in_shardings=(face, rep), out_shardings=face)
    # Runtime indices cross owners and include the no-op case, in one program.
    for index in (0, p+1, n-2, n-1):
        got = fn(operand, jax.device_put(np.int32(index), rep))
        want = value.copy()
        if axis == 0:
            want[[index,n-1],:] = want[[n-1,index],:]
        else:
            want[:,[index,n-1]] = want[:,[n-1,index]]
        np.testing.assert_array_equal(np.asarray(got), want)
        assert got.sharding.is_equivalent_to(face, 2)
    text = fn.lower(operand, jax.device_put(np.int32(0), rep)).compile().as_text()
    # Exact one-owner line transport has no native floating all-reduce. Integer
    # backend bookkeeping can still use its ordinary exact reduction.
    floating = [line for line in text.splitlines()
                if (' all-reduce(' in line or ' all-reduce-start(' in line)
                and ('f64[' in line or 'c128[' in line)]
    assert not floating, floating


def test_local_swap_retains_previous_contract():
    a = jnp.arange(35, dtype=jnp.float64).reshape(5, 7)
    for axis, index in ((0, 1), (1, 3)):
        want = np.asarray(a).copy()
        if axis == 0:
            want[[index,-1],:] = want[[-1,index],:]
        else:
            want[:,[index,-1]] = want[:,[-1,index]]
        np.testing.assert_array_equal(np.asarray(_swap(a, index, axis)), want)

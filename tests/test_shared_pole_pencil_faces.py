"""Shared-pole pencil blocks stay on the x/y face in both routes.

Eager concatenation and a + a^H of face-sharded operands return replicated arrays, so a
[b, R, R] pencil would occupy 16 R^2 bytes on every rank. The distrib_la face blocks run the
same elementwise program with face output shardings: results must be P(None,'x','y') and
bitwise equal to the eager arithmetic. Checked for the helpers themselves and for the TRS
even and the ordered pencil assemblies on a 2x2 host mesh.
"""
from functools import partial

import numpy as np
import pytest


def _mesh():
    import jax
    from jax.sharding import Mesh
    devices = jax.devices()
    if len(devices) < 4:
        pytest.skip("needs 4 host devices")
    return Mesh(np.asarray(devices[:4]).reshape(2, 2), ("x", "y"))


def _face(mesh, host):
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    return jax.device_put(host, NamedSharding(mesh, P(None, "x", "y")))


def _crand(rng, *shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def _on_face(array, mesh):
    from jax.sharding import NamedSharding, PartitionSpec as P
    return array.sharding.is_equivalent_to(NamedSharding(mesh, P(None, "x", "y")), array.ndim)


def test_face_blocks_are_face_sharded_and_bitwise_eager():
    import jax.numpy as jnp
    import distrib_la as D

    mesh, rng = _mesh(), np.random.default_rng(5)
    a, off, corner = _crand(rng, 2, 4, 4), _crand(rng, 2, 2, 4), _crand(rng, 2, 2, 2)
    panel, extra = _crand(rng, 2, 4, 4), _crand(rng, 2, 4, 2)
    adjoint = lambda x: np.conj(np.swapaxes(x, -1, -2))
    cases = (
        (D.hermitian_part(_face(mesh, a)), (a + adjoint(a)) * 0.5),
        (D.hermitian_block(_face(mesh, a), _face(mesh, off), _face(mesh, corner)),
         np.block([[a, adjoint(off)], [off, corner]])),
        (D.join_columns(_face(mesh, panel), _face(mesh, extra)), np.concatenate((panel, extra), axis=-1)),
        (D.diagonal_like(jnp.asarray(rng.normal(size=(2, 4))), _face(mesh, a)), None),
    )
    for got, want in cases:
        assert _on_face(got, mesh)
        if want is not None:
            assert np.asarray(got).tobytes() == want.tobytes()
    # Unsharded and traced operands take the plain function.
    assert np.asarray(D.hermitian_part(jnp.asarray(a))).tobytes() == ((a + adjoint(a)) * 0.5).tobytes()


def _states(mesh, rng, nodes, n=4, r=2):
    return [(node, _face(mesh, _crand(rng, 1, n, r)), _face(mesh, _crand(rng, 1, n, r)),
             _face(mesh, _crand(rng, 1, n, r))) for node in nodes]


def test_even_and_ordered_pencils_come_out_on_the_face():
    import distrib_la as D
    from gw.shared_pole_constructor import assemble_ordered_shared_pole_pencil, assemble_shared_pole_pencil

    mesh, rng = _mesh(), np.random.default_rng(11)
    matmul = partial(D.matmul, mesh=mesh, backend="off")
    panels = lambda k: tuple(_face(mesh, _crand(rng, 1, 4, 2)) for _ in range(k))
    g, h, output = assemble_shared_pole_pencil(_states(mesh, rng, (0.3, 1.1)), panels(3), matmul=matmul)
    assert g.shape == h.shape == (1, 6, 6) and output.shape == (1, 4, 6)
    assert all(_on_face(x, mesh) for x in (g, h, output))
    nodes = (0.4 + 0.2j, 1.3 + 0.1j)
    g, h, output, _ = assemble_ordered_shared_pole_pencil(
        _states(mesh, rng, nodes + tuple(-z for z in nodes)), panels(5), matmul=matmul)
    assert g.shape == h.shape == (1, 12, 12) and output.shape == (1, 4, 12)
    assert all(_on_face(x, mesh) for x in (g, h, output))

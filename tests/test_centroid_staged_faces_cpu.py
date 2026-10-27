"""Literal faces and data dependence of the single staged centroid chain."""
from functools import reduce
from operator import mul
import re
from types import SimpleNamespace

import numpy as np
import pytest


def _owners(shape, ns, mask=None, mu=13):
    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from common import wfn_transforms as W
    world = int(np.prod(shape))
    if len(jax.devices()) < world:
        pytest.skip(f'needs {world} virtual CPU devices')
    mesh = Mesh(np.asarray(jax.devices()[:world]).reshape(shape), ('x', 'y'))
    carrier = (mu + world - 1) // world * world
    n_mu = mu if mask is None else carrier
    sy = NamedSharding(mesh, P(None, None, None, 'y'))
    sx = NamedSharding(mesh, P(None, 'x', None, None))
    sty = NamedSharding(mesh, P(None, 'y', None, None))
    stx = NamedSharding(mesh, P(None, 'x', None, None))
    band = NamedSharding(mesh, W.band_sphere_spec())
    meta = SimpleNamespace(mu_basis=None, b_id_4_user=world + 3)
    face, _ = W._centroid_face_kernels(0, meta, mask, n_mu, carrier,
                                      world + 3, sx, sy, stx, sty)
    zero, _, insert, _ = W._centroid_stream_kernels(
        face, np.zeros((n_mu, 3), np.int32), 1, SimpleNamespace(nkpts=1),
        mesh, meta, carrier, world + 3, 3, ns, sx, sy)
    return mesh, carrier, band, sy, sx, face, zero, insert


@pytest.mark.parametrize('shape', ((2, 2), (3, 3), (8, 10)))
@pytest.mark.parametrize('ns', (1, 2, 4))
def test_staged_faces_literal_mask_spin_and_donated_terminal_updates(shape, ns):
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    world = int(np.prod(shape))
    carrier = (13 + world - 1) // world * world
    # Holes exercise the packed basis; the other spins exercise ordinary pads.
    mask = None
    if ns == 2:
        mask = np.zeros(carrier, bool)
        mask[np.linspace(0, carrier - 1, 13, dtype=int)] = True
        assert mask.sum() == 13
    _, carrier, band, _, _, _, zero, insert = _owners(shape, ns, mask)
    n_mu = 13 if mask is None else carrier
    expected = np.zeros((3, world + 3, ns, carrier), np.complex128)
    y, x = zero()
    for number, (k0, b0, valid_b) in enumerate(((1, 0, world), (1, world, 3),
                                                (2, 0, world))):
        k, b, s, m = np.indices((2, world, ns, n_mu))
        host = (10000*k + 100*b + 10*s + m + number*.125
                + 1j*(-17*k + 3*b - 7*s + .03125*m)).astype(np.complex128)
        host[1] = np.nan + 1j*np.nan
        host[0, valid_b:] = np.nan + 1j*np.nan
        if mask is not None:
            host[..., ~mask] = np.nan + 1j*np.nan
        tile = jax.make_array_from_callback(host.shape, band, lambda index: host[index])
        previous = y, x
        y, x = insert(y, x, tile, jnp.int32(k0), jnp.int32(b0),
                      valid_k=1, valid_b=valid_b)
        jax.block_until_ready((y, x))
        assert all(value.is_deleted() for value in previous)
        live = host[0, :valid_b].copy()
        if mask is not None:
            live[..., ~mask] = 0
        expected[k0, b0:b0 + valid_b, :, :n_mu] = live
        np.testing.assert_array_equal(np.asarray(y), expected)
        np.testing.assert_array_equal(np.asarray(x), expected.conj().transpose(0, 3, 1, 2))


def _assert_staged_dependency(text, shape):
    """Follow actual ENTRY operands, including fused slice dependencies."""
    entry = text.rsplit('\nENTRY ', 1)[1]
    definitions = {}
    root = None
    for line in entry.splitlines():
        match = re.match(r'\s*(ROOT )?%(\S+) = (.*)', line)
        if match:
            definitions[match[2]] = match[3]
            if match[1]:
                root = match[2]
    assert root is not None
    def ancestors(name):
        found, pending = set(), [name]
        while pending:
            current = pending.pop()
            if current not in found and current in definitions:
                found.add(current)
                pending.extend(re.findall(r'%([\w.\-]+)', definitions[current]))
        return found
    root_operands = re.search(r' tuple\(([^)]*)\)', definitions[root])
    assert root_operands, definitions[root]
    outputs = re.findall(r'%([\w.\-]+)', root_operands[1])
    assert len(outputs) == 2
    gathers = {name for name, rhs in definitions.items() if ' all-gather(' in rhs}
    permutations = {name for name, rhs in definitions.items() if ' collective-permute(' in rhs}
    a2a = {name for name, rhs in definitions.items() if ' all-to-all(' in rhs}
    assert len(gathers) == 2 and len(permutations) == 1 and len(a2a) == 2
    permute = next(iter(permutations))
    y_ancestors, x_ancestors = map(ancestors, outputs)
    # Returned Y's gather must complete before the product tile can move.
    before = gathers & y_ancestors & ancestors(permute)
    assert len(before) == 1
    after = gathers - before
    assert len(after) == 1
    assert permute in ancestors(next(iter(after)))
    assert next(iter(after)) in x_ancestors
    assert a2a <= ancestors(next(iter(before)))
    px, py = shape
    pairs = re.search(r'source_target_pairs=(\{\{.*?\}\})', definitions[permute])
    assert pairs, definitions[permute]
    actual = {(int(a), int(b)) for a, b in re.findall(r'\{(\d+),(\d+)\}', pairs[1])}
    expected = {(x*py+y, y*px+x) for x in range(px) for y in range(py)
                if x*py+y != y*px+x}
    assert actual == expected


@pytest.mark.parametrize('shape', ((2, 2), (3, 3), (8, 10)))
def test_staged_dependency_survives_face_and_terminal_insert_lowering(shape):
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    jax.config.update('jax_enable_x64', True)
    mesh, carrier, band, sy, sx, face, _, insert = _owners(shape, 2)
    world = int(np.prod(shape))
    tile = jax.ShapeDtypeStruct((2, world, 2, 13), jnp.complex128, sharding=band)
    face_exe = face.lower(tile).compile()
    _assert_staged_dependency(face_exe.as_text(), shape)
    max_face = 2 * world * 2 * carrier // min(shape)
    for line in face_exe.as_text().splitlines():
        match = re.match(r'\s*(?:ROOT )?%\S+ = c128\[([\d,]*)\]', line)
        if match:
            count = reduce(mul, map(int, filter(None, match[1].split(','))), 1)
            assert count <= max_face, line
    scalar = jax.ShapeDtypeStruct((), jnp.int32, sharding=NamedSharding(mesh, P()))
    operands = (jax.ShapeDtypeStruct((3, world + 3, 2, carrier), jnp.complex128, sharding=sy),
                jax.ShapeDtypeStruct((3, carrier, world + 3, 2), jnp.complex128, sharding=sx),
                tile, scalar, scalar)
    insert_exe = insert.lower(*operands, valid_k=1, valid_b=3).compile()
    _assert_staged_dependency(insert_exe.as_text(), shape)
    carry_bytes = 3 * (world + 3) * 2 * carrier * 16
    assert insert_exe.memory_analysis().alias_size_in_bytes == (
        carry_bytes // shape[0] + carry_bytes // shape[1])


def test_rectangular_bulk_chain_has_the_same_dependency_and_owned_face_bound():
    import jax
    import jax.numpy as jnp
    from common import wfn_transforms as W
    jax.config.update('jax_enable_x64', True)
    mesh, _, band, sy, sx, _, _, _ = _owners((8, 10), 2)
    from jax.sharding import NamedSharding, PartitionSpec as P
    face, _ = W._centroid_face_kernels(
        0, SimpleNamespace(b_id_4_user=160), None, 1998, 2000, 160,
        sx, sy, NamedSharding(mesh, P(None, 'x', None, None)),
        NamedSharding(mesh, P(None, 'y', None, None)))
    tile = jax.ShapeDtypeStruct((144, 160, 2, 1998), jnp.complex128, sharding=band)
    exe = face.lower(tile).compile()
    _assert_staged_dependency(exe.as_text(), (8, 10))
    assert tuple(info.shape for info in exe.out_info) == (
        (144, 160, 2, 2000), (144, 2000, 160, 2))
    for line in exe.as_text().splitlines():
        match = re.match(r'\s*(?:ROOT )?%\S+ = c128\[([\d,]*)\]', line)
        if match:
            count = reduce(mul, map(int, filter(None, match[1].split(','))), 1)
            assert count <= 144 * 160 * 2 * (2000 // 8), line

"""Literal streamed-face updates and ownership of the legal μ carrier."""
from functools import reduce
from operator import mul
import re
from types import SimpleNamespace

import numpy as np
import pytest


def _owners(side, mu, ns, nb_total, band_tile, mask=None):
    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from common import wfn_transforms as W
    world = side * side
    if len(jax.devices()) < world:
        pytest.skip(f"needs {world} virtual CPU devices")
    mesh = Mesh(np.asarray(jax.devices()[:world]).reshape(side, side), ('x', 'y'))
    carrier = (mu + world - 1) // world * world
    n_mu_in = mu if mask is None else carrier
    y = NamedSharding(mesh, P(None, None, None, 'y'))
    x = NamedSharding(mesh, P(None, 'x', None, None))
    by = NamedSharding(mesh, P(None, 'y', None, None))
    bx = NamedSharding(mesh, P(None, 'x', None, None))
    band = NamedSharding(mesh, W.band_sphere_spec())
    meta = SimpleNamespace(mu_basis=None, b_id_4_user=nb_total)
    face, _ = W._centroid_face_kernels(0, meta, mask, n_mu_in, carrier,
                                     nb_total, x, y, bx, by)
    zero, _, insert, _ = W._centroid_stream_kernels(face,
        np.zeros((n_mu_in, 3), np.int32), 1, SimpleNamespace(nkpts=1), mesh,
        meta, carrier, nb_total, 3, ns, x, y)
    return mesh, carrier, band, y, x, zero, insert


@pytest.mark.parametrize('side', (2, 3))
def test_streamed_faces_keep_spin_conjugation_and_terminal_updates(side):
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    world = side * side
    for mu, ns, packed in ((10, 2, False), (13, 1, False),
                           (13, 2, True), (13, 4, False)):
        carrier = (mu + world - 1) // world * world
        mask = None
        if packed:
            mask = np.ones(carrier, bool)
            mask[np.linspace(1, carrier - 2, carrier - mu, dtype=int)] = False
            assert mask.sum() == mu
        nb_total = world + 3
        mesh, carrier, band, _, _, zero, insert = _owners(
            side, mu, ns, nb_total, world, mask)
        n_mu_in = mu if mask is None else carrier
        ref = np.zeros((3, nb_total, ns, carrier), np.complex128)
        ay, ax = zero()
        # The terminal band tile must not clamp backwards over the first one.
        for call, (k0, b0, valid_b) in enumerate(((1, 0, world), (1, world, 3), (2, 0, world))):
            b, s, m = np.indices((world, ns, n_mu_in))
            host = (call + .1*b + .03*s + .007*m
                    + 1j*(.2 + .02*b - .01*s + .003*m))[None]
            if mask is not None:
                host[..., ~mask] = np.nan + 1j*np.nan
            tile = jax.make_array_from_callback(host.shape, band, lambda index: host[index])
            previous = ay, ax
            ay, ax = insert(ay, ax, tile, jnp.int32(k0), jnp.int32(b0),
                            valid_k=1, valid_b=valid_b)
            jax.block_until_ready((ay, ax))
            assert all(x.is_deleted() for x in previous), 'donated carries stayed live'
            physical = host[0, :valid_b].copy()
            if mask is not None:
                physical[..., ~mask] = 0
            ref[k0, b0:b0+valid_b, :, :n_mu_in] = physical
            np.testing.assert_array_equal(np.asarray(ay), ref)
            np.testing.assert_array_equal(np.asarray(ax), ref.conj().transpose(0, 3, 1, 2))


@pytest.mark.parametrize('side', (2, 3))
def test_abstract_insert_pads_on_band_owner_without_mu_halos_or_full_carry_copy(side):
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    jax.config.update('jax_enable_x64', True)
    mu, ns, nb_total = 2049, 4, 8191
    world = side * side
    band_tile = (8192 + world - 1) // world * world
    mesh, carrier, band, y, x, _, insert = _owners(
        side, mu, ns, nb_total, band_tile)
    scalar = jax.ShapeDtypeStruct((), jnp.int32, sharding=NamedSharding(mesh, P()))
    operands = (jax.ShapeDtypeStruct((3, nb_total, ns, carrier), jnp.complex128, sharding=y),
                jax.ShapeDtypeStruct((3, carrier, nb_total, ns), jnp.complex128, sharding=x),
                jax.ShapeDtypeStruct((1, band_tile, ns, mu), jnp.complex128, sharding=band),
                scalar, scalar)
    exe = insert.lower(*operands, valid_k=1, valid_b=nb_total).compile()
    text = exe.as_text()
    halos = [line for line in text.splitlines() if 'collective-permute' in line
             and 'source_target_pairs=' in line and '/pad' in line]
    assert not halos, halos
    stats = exe.memory_analysis()
    local_carry_bytes = 3 * nb_total * ns * (carrier // side) * 16
    assert stats.alias_size_in_bytes == 2 * local_carry_bytes
    # Exclude the carry parameters and their in-place updates, then bound
    # every remaining complex tensor by one μ-face tile (1/sqrt(P)).
    max_tile_elements = band_tile * ns * (carrier // side)
    for line in text.splitlines():
        if 'parameter(' in line or 'dynamic-update-slice(' in line or 'dynamic_update_slice' in line:
            continue
        match = re.match(r'\s*(?:ROOT )?%\S+ = c128\[([\d,]*)\]', line)
        if match:
            elements = reduce(mul, map(int, filter(None, match[1].split(','))), 1)
            assert elements <= max_tile_elements, line


def test_rectangular_bulk_faces_do_not_rematerialize_the_shared_carrier():
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from common import wfn_transforms as W
    if len(jax.devices()) < 80:
        pytest.skip('needs the historical 8x10 virtual-device geometry')
    jax.config.update('jax_enable_x64', True)
    mesh = Mesh(np.asarray(jax.devices()[:80]).reshape(8, 10), ('x', 'y'))
    sy = NamedSharding(mesh, P(None, None, None, 'y'))
    sx = NamedSharding(mesh, P(None, 'x', None, None))
    sty = NamedSharding(mesh, P(None, 'y', None, None))
    stx = NamedSharding(mesh, P(None, 'x', None, None))
    band = NamedSharding(mesh, W.band_sphere_spec())
    fn, _ = W._centroid_face_kernels(0, SimpleNamespace(b_id_4_user=160),
                                    None, 1998, 2000, 160, sx, sy, stx, sty)
    value = jax.ShapeDtypeStruct((144, 160, 2, 1998), jnp.complex128, sharding=band)
    exe = fn.lower(value).compile()
    expected = ((144, 160, 2, 2000), (144, 2000, 160, 2))
    assert tuple(x.shape for x in exe.out_info) == expected
    max_face_elements = 144 * 160 * 2 * (2000 // 8)
    for line in exe.as_text().splitlines():
        match = re.match(r'\s*(?:ROOT )?%\S+ = c128\[([\d,]*)\]', line)
        if match:
            elements = reduce(mul, map(int, filter(None, match[1].split(','))), 1)
            assert elements <= max_face_elements, line
        assert not ('collective-permute' in line and 'source_target_pairs=' in line and '/pad' in line), line

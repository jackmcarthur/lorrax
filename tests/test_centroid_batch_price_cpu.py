"""The candidate-sketch k batch is priced on everything the loader holds for it.

``centroid.sketch_select._batch_bytes_per_k`` sizes a k batch against the fixed
tile; the batch is one k tile of the shared centroid loader, whose own price of
that tile is ``common.wfn_transforms._centroid_resident_bytes``.  For a large
G sphere on few candidates (the regime where the faces are a small part of
the tile) and for the fixture shapes, the sketch's price of a batch covers the
loader's, and exceeds the faces alone.

CPU only; the P4 case needs four virtual devices
(``XLA_FLAGS=--xla_force_host_platform_device_count=4``).
"""
from types import SimpleNamespace

import numpy as np
import pytest


def _mesh(shape):
    import jax
    from jax.sharding import Mesh
    world = int(np.prod(shape))
    if len(jax.devices()) < world:
        pytest.skip(f"needs {world} virtual CPU devices")
    return Mesh(np.asarray(jax.devices()[:world]).reshape(shape), ("x", "y"))


@pytest.mark.parametrize("shape", [(1, 1), (2, 2)])
def test_sketch_batch_price_covers_the_loader_tile(shape):
    from centroid.candidates import candidate_band_carrier
    from centroid.sketch_select import _batch_bytes_per_k
    from common.wfn_transforms import _centroid_resident_bytes
    from runtime.padding import padded_axis, padded_mu_extent
    mesh = _mesh(shape)
    p = int(mesh.size)
    for ngkmax, M, width, ns in ((30000, 3000, 790, 2), (6355, 8997, 790, 2),
                                 (284, 37, 9, 2), (555, 120, 14, 1)):
        m_pad = padded_mu_extent(M, mesh)
        left, right = (0, min(4, width)), (0, width)
        per_k = _batch_bytes_per_k(mesh, ns=ns, m_pad=m_pad, left=left, right=right,
                                   ngkmax=ngkmax)
        faces = 16.0 * ns * m_pad * candidate_band_carrier(width, mesh, 64) / p
        band_tile = padded_axis(min(width, 64), p, name="tile").carrier
        for k_batch in (1, 7, 64):
            loader = SimpleNamespace(ngkmax=ngkmax, nspinor=ns, nkpts=k_batch)
            _, _, held = _centroid_resident_bytes(
                band_tile, False, k_batch, loader, 0, mesh, M, m_pad, 45 ** 3,
                width, width, k_batch, k_batch, ns, p, False, None, False, True,
                cand_faces=True)
            assert per_k * k_batch >= held, (shape, ngkmax, M, k_batch, per_k * k_batch, held)
        assert per_k > faces, (per_k, faces)

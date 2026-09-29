"""panel_matmul's per-row contraction interval skips the columns outside it.

Columns outside each row's [lo, hi) are NaN in both operands: any flop spent
on them poisons the tile, so a finite result equal to the interval product
shows the gathered local product never touched them.  Runs on the emulated
four-device CPU mesh and on a four-process CUDA mesh (the cuBLAS handler).
"""
from __future__ import annotations

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from lxkit.testing import require_devices

from common.collectives import device_put_process_local
from distrib_la import panel_matmul


def test_panel_matmul_bounds_skip_outside_columns():
    platform = jax.default_backend()
    require_devices(4, platform)
    mesh = Mesh(np.asarray(jax.devices(platform)[:4]).reshape(2, 2), ("x", "y"))
    rng = np.random.default_rng(7)
    q, m, k, n = 4, 6, 8, 10
    a = rng.standard_normal((q, m, k)) + 1j * rng.standard_normal((q, m, k))
    b = rng.standard_normal((q, k, n)) + 1j * rng.standard_normal((q, k, n))
    bounds = np.asarray([[0, 8], [2, 5], [3, 3], [5, 8]], np.int32)
    want = np.zeros((q, m, n), complex)
    for i, (lo, hi) in enumerate(bounds):
        want[i] = a[i, :, lo:hi] @ b[i, lo:hi, :]
        a[i, :, :lo] = a[i, :, hi:] = np.nan
        b[i, :lo, :] = b[i, hi:, :] = np.nan
    face = NamedSharding(mesh, P(None, "x", "y"))
    out = panel_matmul(device_put_process_local(a, face), device_put_process_local(b, face),
                       mesh=mesh, panel_bytes=1 << 30,
                       bounds=device_put_process_local(bounds, NamedSharding(mesh, P())))
    for shard in out.addressable_shards:
        np.testing.assert_allclose(np.asarray(shard.data), want[shard.index],
                                   rtol=1e-12, atol=1e-12)


def test_panel_matmul_stacked_weights_share_one_exchange():
    """(s,q,k) weights return s products a·diag(w_s)·b over each row's interval from one
    panel exchange; the NaN outside every interval shows no dead column is contracted."""
    platform = jax.default_backend()
    require_devices(4, platform)
    mesh = Mesh(np.asarray(jax.devices(platform)[:4]).reshape(2, 2), ("x", "y"))
    rng = np.random.default_rng(11)
    q, m, k, n = 3, 6, 8, 10
    a = rng.standard_normal((q, m, k)) + 1j * rng.standard_normal((q, m, k))
    b = rng.standard_normal((q, k, n)) + 1j * rng.standard_normal((q, k, n))
    w = rng.standard_normal((2, q, k)) + 1j * rng.standard_normal((2, q, k))
    rows = np.asarray([[1, 7], [0, 8], [4, 5]], np.int32)
    want = np.zeros((2, q, m, n), complex)
    for s in range(2):
        for i, (lo, hi) in enumerate(rows):
            want[s, i] = (a[i, :, lo:hi] * w[s, i, lo:hi]) @ b[i, lo:hi, :]
    for i, (lo, hi) in enumerate(rows):
        a[i, :, :lo] = a[i, :, hi:] = np.nan
        b[i, :lo, :] = b[i, hi:, :] = np.nan
    face = NamedSharding(mesh, P(None, "x", "y"))
    rep = NamedSharding(mesh, P())
    out = panel_matmul(device_put_process_local(a, face), device_put_process_local(b, face),
                       mesh=mesh, panel_bytes=1 << 30,
                       bounds=device_put_process_local(np.stack([rows, rows]), rep),
                       weights=device_put_process_local(w, rep))
    assert len(out) == 2
    for s, tile in enumerate(out):
        for shard in tile.addressable_shards:
            np.testing.assert_allclose(np.asarray(shard.data), want[s][shard.index],
                                       rtol=1e-12, atol=1e-12)

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
from distrib_la import gather_panels, gathered_matmul, panel_matmul


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


def test_gathered_matmul_is_panel_matmul_on_held_panels():
    """Panels gathered once and weighted per call give panel_matmul's product, with and without bounds.

    Bitwise on CUDA, where the handler multiplies a materialized weighted panel;
    to round-off on the CPU mesh, where XLA fuses the weight into the dot."""
    platform = jax.default_backend()
    require_devices(4, platform)
    mesh = Mesh(np.asarray(jax.devices(platform)[:4]).reshape(2, 2), ("x", "y"))
    rng = np.random.default_rng(11)
    q, m, k, n = 4, 6, 8, 10
    a = rng.standard_normal((q, m, k)) + 1j * rng.standard_normal((q, m, k))
    b = rng.standard_normal((q, k, n)) + 1j * rng.standard_normal((q, k, n))
    w = rng.standard_normal((q, k)) + 1j * rng.standard_normal((q, k))
    bounds = np.asarray([[0, 8], [2, 5], [3, 3], [5, 8]], np.int32)
    face, rep = NamedSharding(mesh, P(None, "x", "y")), NamedSharding(mesh, P())
    A, B = device_put_process_local(a, face), device_put_process_local(b, face)
    left, right = gather_panels(A, B, mesh=mesh)
    live = (np.arange(k)[None] >= bounds[:, :1]) & (np.arange(k)[None] < bounds[:, 1:])
    for weight, rows in ((w, None), (np.where(live, w, 0), bounds)):
        W = device_put_process_local(weight, rep)
        R = None if rows is None else device_put_process_local(rows, rep)
        want = panel_matmul(A * W[:, None, :], B, mesh=mesh, panel_bytes=1 << 30, bounds=R)
        got = gathered_matmul(left, right, mesh=mesh, weights=W, bounds=R)
        for g, h in zip(got.addressable_shards, want.addressable_shards):
            assert g.index == h.index
            if platform == "cpu":
                np.testing.assert_allclose(np.asarray(g.data), np.asarray(h.data),
                                           rtol=1e-13, atol=1e-13)
            else:
                np.testing.assert_array_equal(np.asarray(g.data), np.asarray(h.data))

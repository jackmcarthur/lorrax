"""P4 parity: PsiGStore's transform-once sources against ``iter_rchunk_bandwise``.

The Galerkin fit reads candidate/selected rows through ``gather_state_rows`` +
``iter_rows_rchunks`` and streams every state once through
``iter_bandchunks_rchunks``.  Both must reproduce, slot for slot, the incumbent
per-r-chunk stream on the same host tiles (same transform, other batching).
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
    import jax as _jax_boot

    _visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    _kwargs = {"local_device_ids": [0]} if _visible and "," not in _visible else {}
    _jax_boot.distributed.initialize(**_kwargs)

from types import SimpleNamespace

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.psi_G_store import build_psi_G_store


def _mesh() -> Mesh:
    devices = jax.devices()
    count = 4 if len(devices) >= 4 else 1
    side = int(np.sqrt(count))
    return Mesh(np.asarray(devices[:count]).reshape(side, side), ("x", "y"))


def _host(value) -> np.ndarray:
    if jax.process_count() == 1:
        return np.asarray(jax.device_get(value))
    from jax.experimental import multihost_utils
    return np.asarray(multihost_utils.process_allgather(value, tiled=True))


class _Loader:
    """The loader surface PsiGStore consumes, on seeded synthetic ψ(G)."""

    def __init__(self, mesh, *, nk, nbands, ns, ngkmax, fft_grid):
        rng = np.random.default_rng(11)
        n_r = int(np.prod(fft_grid))
        self.mesh, self.nkpts, self.nbands, self.ngkmax = mesh, nk, nbands, ngkmax
        self.psi = (rng.normal(size=(nk, nbands, ns, ngkmax))
                    + 1j * rng.normal(size=(nk, nbands, ns, ngkmax)))
        # The per-k sphere index (common.gvec_fft_box.build_sphere_box_index):
        # the flat box cell of each slot, n_r + g on a pad slot.
        g_index = np.tile(n_r + np.arange(ngkmax, dtype=np.int32), (nk, 1))
        for k in range(nk):
            ngk = ngkmax - k % 3          # ragged spheres, as in a real WFN
            g_index[k, :ngk] = rng.choice(n_r, size=ngk, replace=False)
            self.psi[k, :, :, ngk:] = 0.0
        self.g_index = g_index
        self.k = rng.uniform(-0.5, 0.5, size=(nk, 3))

    def load(self, *, bands, k, sharding, bispinor, bispinor_lift):
        del k, bispinor, bispinor_lift
        data = self.psi[:, bands[0]:bands[1]]
        return jax.make_array_from_callback(
            data.shape, NamedSharding(self.mesh, sharding),
            lambda index: data[index])

    def box_index_dev(self, *, k, mesh):
        del k
        return jax.device_put(self.g_index, NamedSharding(mesh, P()))

    def kvecs(self, *, k):
        del k
        return self.k


@pytest.mark.mesh(4)
def test_transform_once_sources_match_the_rchunk_stream():
    mesh = _mesh()
    p = int(mesh.size)
    nk, nbands, ns, ngkmax, grid = 4, 12, 2, 23, (4, 5, 6)
    n_r = int(np.prod(grid))
    meta = SimpleNamespace(nk_tot=nk, nspinor=ns, fft_grid=grid, n_rtot=n_r,
                           b_id_4=nbands, b_id_4_user=nbands)
    loader = _Loader(mesh, nk=nk, nbands=nbands, ns=ns, ngkmax=ngkmax,
                     fft_grid=grid)
    carrier = 2 * p
    chunks = tuple((b, min(b + carrier, nbands))
                   for b in range(0, nbands, carrier))
    step = 7 * p                         # ragged terminal r chunk
    ranges = tuple((r, min(r + step, n_r)) for r in range(0, n_r, step))
    spec = P(None, None, None, ("y", "x"))

    with build_psi_G_store(wfn=loader, mesh_xy=mesh, meta=meta,
                           band_chunk_ranges=chunks,
                           band_pad_to=carrier) as store:
        ref = np.zeros((nk, nbands, ns, n_r), dtype=np.complex128)
        for r0, r1 in ranges:
            for (b0, b1), psi in store.iter_rchunk_bandwise(
                    r0, r1, product_r_spec=spec):
                ref[:, b0:b1, :, r0:r1] = _host(psi)[:, :b1 - b0, :, :r1 - r0]

        # Loop-inverted stream: one transform per band chunk, k tiled.
        inv = np.zeros_like(ref)
        for (b0, b1), r_idx, psi in store.iter_bandchunks_rchunks(
                ranges, product_r_spec=spec, k_tile=2):
            r0, r1 = ranges[r_idx]
            host = _host(psi)
            assert psi.sharding.spec == spec
            assert np.all(host[..., r1 - r0:] == 0)
            inv[:, b0:b1, :, r0:r1] = host[:, :b1 - b0, :, :r1 - r0]
        np.testing.assert_allclose(inv, ref, rtol=0, atol=1e-13)

        # Owner-local rows: arbitrary states, transformed once, same values.
        states = np.random.default_rng(5).permutation(nk * nbands)[:9]
        rows, row_k, slots = store.gather_state_rows(
            states, band_start=0, band_count=nbands, row_multiple=2)
        assert rows.shape[0] % (2 * p) == 0 and np.sum(slots >= 0) == 9
        assert sorted(slots[slots >= 0].tolist()) == sorted(states.tolist())
        got = np.zeros((rows.shape[0], ns, n_r), dtype=np.complex128)
        for r_idx, slab in store.iter_rows_rchunks(
                rows, row_k, ranges, product_r_spec=spec, fft_rows=2):
            r0, r1 = ranges[r_idx]
            assert slab.sharding.spec == spec
            got[:, :, r0:r1] = _host(slab)[0, :, :, :r1 - r0]
        for slot, state in enumerate(slots):
            if state < 0:
                assert np.all(got[slot] == 0)
            else:
                np.testing.assert_allclose(
                    got[slot], ref[state // nbands, state % nbands],
                    rtol=0, atol=1e-13)

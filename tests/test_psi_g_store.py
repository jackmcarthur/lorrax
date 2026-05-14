import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.psi_G_store import _zero_user_band_pad_in_shard, PsiGStore


def test_zero_user_band_pad_in_shard_zeros_only_requested_pad_slots():
    data = np.ones((2, 4, 1, 1, 1, 1), dtype=np.complex128)
    data[:, :, 0, 0, 0, 0] *= np.arange(4)[None, :] + 1

    out = _zero_user_band_pad_in_shard(
        data,
        bc_range=(128, 192),
        shard_band_slice=slice(20, 24),
        user_band_stop=150,
    )

    assert out is not data
    np.testing.assert_array_equal(out[:, :2], data[:, :2])
    assert np.all(out[:, 2:] == 0.0)
    np.testing.assert_array_equal(data[:, :, 0, 0, 0, 0], [[1, 2, 3, 4], [1, 2, 3, 4]])


def test_zero_user_band_pad_in_shard_returns_original_when_no_pad_owned():
    data = np.ones((1, 3, 1, 1, 1, 1), dtype=np.complex128)

    out = _zero_user_band_pad_in_shard(
        data,
        bc_range=(64, 128),
        shard_band_slice=slice(0, 3),
        user_band_stop=150,
    )

    assert out is data


def test_zero_user_band_pad_in_shard_rejects_strided_band_slice():
    data = np.ones((1, 3, 1, 1, 1, 1), dtype=np.complex128)

    with pytest.raises(ValueError, match="contiguous"):
        _zero_user_band_pad_in_shard(
            data,
            bc_range=(0, 64),
            shard_band_slice=slice(0, 6, 2),
            user_band_stop=4,
        )


# ---------------------------------------------------------------------------
# psi_G_device_full lazy property — Path D integration consumer entry point.
# Built without going through HostPsiGStore so the test stays free of WfnLoader
# / Meta plumbing: subclasses PsiGStore, populates _host_tiles directly with a
# random ψ(G) tile, and verifies the round-trip through the io_callback
# shard_map.
# ---------------------------------------------------------------------------


class _FakePsiGStore(PsiGStore):
    """PsiGStore stub: skips loader/meta plumbing for unit tests.

    Caller hands in a single per-rank tile (shape ``(nk, nb_local, ns,
    ngkmax)``) — installed into ``_host_tiles`` for every (x, y) cell in
    the mesh. ``begin_rchunk`` / ``end_rchunk`` are no-ops; ``end_rchunk``
    here triggers ``_clear_tiles`` so the cache-invalidation path is
    exercised by the test.
    """

    def __init__(self, *, mesh_xy, tile, ngkmax):
        # Bypass the parent __init__ — we don't have a loader/meta. Set the
        # subset of fields ``psi_G_device_full`` reads.  Single bc per the
        # whole local axis: bc_band_offsets = [0, nb_local].
        self.mesh = mesh_xy
        self._dtype = tile.dtype
        from common.psi_G_store import _mesh_device_coords
        self._coords = _mesh_device_coords(mesh_xy)
        self._per_rank_shape = tuple(int(s) for s in tile.shape)
        self._host_tiles = {(x, y): tile.copy() for (x, y) in self._coords.values()}
        self._g_index_dev = None
        self._kvecs_frac_dev = None
        self._psi_G_device_full = None
        # One synthetic band-chunk spanning the whole local axis — matches the
        # production layout when all bands fit in a single bc.
        self.band_chunk_ranges = ((0, self._per_rank_shape[1]),)
        self._bc_band_offsets = (0, self._per_rank_shape[1])

    def end_rchunk(self) -> None:
        self._clear_tiles()


def test_psi_G_device_full_round_trips_host_tile():
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1),
                 axis_names=('x', 'y'))
    nk, nb_local, ns, ngkmax = 3, 4, 2, 7
    rng = np.random.default_rng(11)
    tile = (rng.standard_normal((nk, nb_local, ns, ngkmax))
            + 1j * rng.standard_normal((nk, nb_local, ns, ngkmax))).astype(
        np.complex128)

    store = _FakePsiGStore(mesh_xy=mesh, tile=tile, ngkmax=ngkmax)
    out = store.psi_G_device_full
    assert out.shape == tile.shape
    np.testing.assert_array_equal(np.asarray(out), tile)

    # Lazy + cached: second access returns the same jax.Array (no re-pull).
    out2 = store.psi_G_device_full
    assert out2 is out


def test_psi_G_device_full_invalidated_by_clear_tiles():
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1),
                 axis_names=('x', 'y'))
    nk, nb_local, ns, ngkmax = 2, 3, 1, 5
    rng = np.random.default_rng(12)
    tile = (rng.standard_normal((nk, nb_local, ns, ngkmax))
            + 1j * rng.standard_normal((nk, nb_local, ns, ngkmax))).astype(
        np.complex128)
    store = _FakePsiGStore(mesh_xy=mesh, tile=tile, ngkmax=ngkmax)

    out1 = store.psi_G_device_full
    np.testing.assert_array_equal(np.asarray(out1), tile)

    # Repopulate the host tile with new data, clear the device cache, then
    # the next access must re-pull and reflect the new tile.
    new_tile = (rng.standard_normal((nk, nb_local, ns, ngkmax))
                + 1j * rng.standard_normal((nk, nb_local, ns, ngkmax))).astype(
        np.complex128)
    for k in store._host_tiles:
        store._host_tiles[k] = new_tile.copy()
    store.end_rchunk()                                  # triggers _clear_tiles
    assert store._psi_G_device_full is None             # invalidated
    # Re-populate and re-pull.
    for k in list(store._coords.values()):
        store._host_tiles[k] = new_tile.copy()
    out2 = store.psi_G_device_full
    np.testing.assert_array_equal(np.asarray(out2), new_tile)
    assert out2 is not out1


def test_psi_G_device_full_raises_when_tiles_empty():
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1),
                 axis_names=('x', 'y'))
    nk, nb_local, ns, ngkmax = 1, 1, 1, 3
    tile = np.zeros((nk, nb_local, ns, ngkmax), dtype=np.complex128)
    store = _FakePsiGStore(mesh_xy=mesh, tile=tile, ngkmax=ngkmax)
    store._host_tiles.clear()                           # simulate before begin_rchunk
    store._psi_G_device_full = None
    with pytest.raises(RuntimeError, match="begin_rchunk"):
        _ = store.psi_G_device_full

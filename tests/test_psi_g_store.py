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
# _slice_local_tile_bc — per-iter host-tile slicer used by the io_callback
# inside the new c_q_from_psi_sm / z_q_from_psi_sm scan body (Round 6).
# Built without going through HostPsiGStore so the test stays free of
# WfnLoader / Meta plumbing: subclasses PsiGStore, populates _host_tiles
# directly with a random multi-bc ψ(G) tile, and exercises the slicer.
# ---------------------------------------------------------------------------


class _FakePsiGStore(PsiGStore):
    """PsiGStore stub: skips loader/meta plumbing for unit tests.

    Caller hands in a single per-rank tile (shape ``(nk, nb_local, ns,
    ngkmax)``) — installed into ``_host_tiles`` for every (x, y) cell in
    the mesh — plus a ``band_chunk_ranges`` tuple that defines the
    bc-stacked band layout.  bc-band offsets and ``_bpd_max`` are
    derived from ``band_chunk_ranges`` exactly as production code does.
    """

    def __init__(self, *, mesh_xy, tile, band_chunk_ranges):
        self.mesh = mesh_xy
        self._dtype = tile.dtype
        from common.psi_G_store import _mesh_device_coords
        self._coords = _mesh_device_coords(mesh_xy)
        self._per_rank_shape = tuple(int(s) for s in tile.shape)
        self._host_tiles = {(x, y): tile.copy() for (x, y) in self._coords.values()}
        self._g_index_dev = None
        self._kvecs_frac_dev = None
        self.band_chunk_ranges = tuple(tuple(bc) for bc in band_chunk_ranges)
        # Mesh size = 1 here; bpd_per_bc[i] = (b_hi - b_lo) // 1 = bc_size.
        bpd_per_bc = [(b_hi - b_lo) for (b_lo, b_hi) in self.band_chunk_ranges]
        self._bpd_per_bc = tuple(bpd_per_bc)
        self._bpd_max = max(bpd_per_bc) if bpd_per_bc else 0
        offsets = [0]
        for bpd in bpd_per_bc:
            offsets.append(offsets[-1] + bpd)
        self._bc_band_offsets = tuple(offsets)

    def end_rchunk(self) -> None:
        self._clear_tiles()


def test_slice_local_tile_bc_returns_correct_band_window():
    """Slicer returns each bc's bands of the host tile, padded to bpd_max."""
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1),
                 axis_names=('x', 'y'))
    nk, ns, ngkmax = 3, 2, 5
    band_chunks = ((0, 4), (4, 8), (8, 10))             # last bc short (bpd=2 vs 4)
    bpd_max = 4
    nb_local = 4 + 4 + 2
    rng = np.random.default_rng(11)
    tile = (rng.standard_normal((nk, nb_local, ns, ngkmax))
            + 1j * rng.standard_normal((nk, nb_local, ns, ngkmax))).astype(
        np.complex128)

    store = _FakePsiGStore(mesh_xy=mesh, tile=tile, band_chunk_ranges=band_chunks)
    assert store._bpd_max == bpd_max

    # bc 0: bands [0, 4) — full
    out0 = store._slice_local_tile_bc(0, 0, 0)
    assert out0.shape == (nk, bpd_max, ns, ngkmax)
    np.testing.assert_array_equal(out0[:, :4, :, :], tile[:, 0:4, :, :])

    # bc 1: bands [4, 8) — full
    out1 = store._slice_local_tile_bc(0, 0, 1)
    np.testing.assert_array_equal(out1[:, :4, :, :], tile[:, 4:8, :, :])

    # bc 2: bands [8, 10) — short; pad rows must be exactly zero (math-neutral)
    out2 = store._slice_local_tile_bc(0, 0, 2)
    assert out2.shape == (nk, bpd_max, ns, ngkmax)
    np.testing.assert_array_equal(out2[:, :2, :, :], tile[:, 8:10, :, :])
    assert np.all(out2[:, 2:, :, :] == 0.0)             # pad-rows EXACTLY zero
    assert out2[:, 2:, :, :].dtype == np.complex128


def test_slice_local_tile_bc_traced_bc_idx_inputs():
    """Slicer accepts traced int32 scalars (the io_callback contract); resolved
    to Python ints inside the host fn."""
    import jax.numpy as jnp_local                       # noqa
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1),
                 axis_names=('x', 'y'))
    nk, ns, ngkmax = 2, 1, 3
    band_chunks = ((0, 2), (2, 4))
    tile = np.arange(nk * 4 * ns * ngkmax, dtype=np.complex128).reshape(
        nk, 4, ns, ngkmax)

    store = _FakePsiGStore(mesh_xy=mesh, tile=tile, band_chunk_ranges=band_chunks)
    # Traced int32 scalars (what io_callback would pass).
    x_idx = np.int32(0)
    y_idx = np.int32(0)
    bc_idx = np.int32(1)
    out = store._slice_local_tile_bc(x_idx, y_idx, bc_idx)
    np.testing.assert_array_equal(out[:, :2, :, :], tile[:, 2:4, :, :])


def test_slice_local_tile_bc_rejects_out_of_range_bc():
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1),
                 axis_names=('x', 'y'))
    nk, ns, ngkmax = 2, 1, 3
    tile = np.zeros((nk, 4, ns, ngkmax), dtype=np.complex128)
    store = _FakePsiGStore(mesh_xy=mesh, tile=tile,
                            band_chunk_ranges=((0, 2), (2, 4)))
    with pytest.raises(ValueError, match=r"bc_idx=2 not in"):
        store._slice_local_tile_bc(0, 0, 2)


def test_psi_G_device_full_property_removed():
    """Round 6 deleted the buggy lazy property + field.  Confirm absence."""
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1),
                 axis_names=('x', 'y'))
    tile = np.zeros((1, 1, 1, 1), dtype=np.complex128)
    store = _FakePsiGStore(mesh_xy=mesh, tile=tile, band_chunk_ranges=((0, 1),))
    assert not hasattr(store, "psi_G_device_full")
    assert not hasattr(store, "_psi_G_device_full")

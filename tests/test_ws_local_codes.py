"""Addressable WS codes agree with independent nearest-image distances."""
import jax
import numpy as np

from common.ws_images import pair_ws_images, pair_ws_local_codes

jax.config.update("jax_enable_x64", True)


def test_rectangular_ws_codes_match_nearest_images_with_ties_and_padding(monkeypatch):
    from runtime import tiles
    from common import ws_images as ws

    # A small tile forces several displacement chunks, including a short tail.
    monkeypatch.setattr(tiles, "TILE_BYTES", 1 << 13)
    nodes = np.array([[0., 0., 0.], [.5, 0., 0.], [0., .5, 0.],
                      [.25, .25, .5]])
    targets = np.array([[.125, .25, .2], [.5, .5, .1], [.875, .75, .9],
                        [1.e-9, 0., 0.], [1.e-8, 0., 0.]])
    pos = np.concatenate([nodes, targets])
    lattice = np.array([[6., 0., 0.], [1.2, 5., 0.], [.3, .1, 18.]])
    asarray, put = ws.jnp.asarray, jax.device_put
    common_ids, uploads = set(), []

    def rank_local_geometry(value):
        return (isinstance(value, np.ndarray) and value.ndim == 2
                and value.shape[-1] == 3 and id(value) not in common_ids)

    def no_eager_geometry(value, *args, **kw):
        if rank_local_geometry(value):
            raise AssertionError("rank-local WS displacement used eager jnp.asarray")
        return asarray(value, *args, **kw)

    def local_upload(value, device=None, **kw):
        if rank_local_geometry(value):
            assert device == jax.local_devices()[0]
            assert value.dtype == np.float64
            uploads.append(value.shape)
        return put(value, device, **kw)

    monkeypatch.setattr(ws.jnp, "asarray", no_eager_geometry)
    monkeypatch.setattr(jax, "device_put", local_upload)
    for grid, dim in [((4, 4, 1), 2), ((3, 2, 2), 3)]:
        images = pair_ws_images(grid, lattice, pos, dim)
        common_ids = {id(images[k]) for k in ("a", "b", "valid")}
        # Separate face tiles can encounter different tie dictionaries.
        blocks = [(np.array([0, 1, 2, 3, -1, 0]), np.array([0, 2, 3, 0])),
                  # Two nearly equal displacements straddle the physical
                  # tie tolerance, although an integer displacement key
                  # could quantize them to the same representative.
                  (np.array([4, 5, 6, 7, 8, -1]), np.array([0, 1, 2, 3, -1]))]
        codes, patterns = pair_ws_local_codes(images, blocks)
        bits = np.left_shift(np.int64(1), np.arange(images["img"].shape[1]))
        for (rows, cols), code in zip(blocks, codes):
            displacement = pos[np.maximum(rows, 0), None] - pos[None, np.maximum(cols, 0)]
            # Independent physical definition, with the Cartesian pair distance.
            cart = (images["img"][:, None, None] + displacement[None, :, :, None]) @ lattice
            distance = np.einsum("...a,...a->...", cart, cart)
            distance = np.where(images["valid"][:, None, None], distance, np.inf)
            tied = distance <= distance.min(axis=-1, keepdims=True) + images["tol"]
            expected = np.sum(np.where(tied, bits, 0), axis=-1)
            np.testing.assert_array_equal(patterns[code], expected)
        empty, empty_patterns = pair_ws_local_codes(
            images, [(np.array([], dtype=np.int64), np.array([0, 0]))])
        assert empty[0].shape == (int(np.prod(grid)), 0, 2)
        assert empty_patterns.size == 0
    # Actual P64 target-padding stripes: one node tile and three target
    # tiles visit widths 64, 48 and 8 after their common precompile stage.
    monkeypatch.setattr(tiles, "TILE_BYTES", 1 << 30)
    images = pair_ws_images((1, 1, 1), np.eye(3), np.zeros((64 + 13, 3)), 3)
    common_ids = {id(images[k]) for k in ("a", "b", "valid")}
    start = len(uploads)
    codes, patterns = pair_ws_local_codes(images, [
        (np.arange(8), np.arange(8)),
        (np.r_[np.arange(8, 13), -1], np.arange(8)),
        (np.array([-1]), np.arange(8))])
    assert {shape[0] for shape in uploads[start:]} == {64, 48, 8}
    for code in codes:
        np.testing.assert_array_equal(patterns[code], np.ones_like(code))
    assert len(set(uploads)) > 1, "control needs distinct local displacement widths"

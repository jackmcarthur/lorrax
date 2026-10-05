"""The default cuSOLVERMp eigh block (CPU, instant).

Block-cyclic only with a divisor of n/p in (128, 256]; otherwise one tile per
rank, so a prime or small-cofactor n/p never runs at block 1-8 (a prime n/p
ran the face reduction 12x slower at block 1).  Ladder extents
(``runtime.padding.ladder_extent``) always keep their block.
"""
from ffi import _services

_services.ensure_on_path()


def test_block_falls_back_to_one_tile_per_rank():
    from distrib_la._cusolvermp import _block_size, retry_block
    from runtime.padding import ladder_extent

    assert _block_size(778, 2) == 389            # n/p prime: one tile, not block 1
    assert _block_size(4208, 4) == 1052          # 4 * 263: one tile, not block 4
    assert _block_size(4000, 2) == 250
    assert _block_size(200, 2) == 100            # n/p <= 128: one tile, as before
    assert retry_block(4000, 2) == 2000
    for n in range(129, 6000):
        b = _block_size(ladder_extent(n) * 2, 2)
        assert 128 < b <= 256 or b == ladder_extent(n), n

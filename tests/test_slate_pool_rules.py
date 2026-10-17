"""SLATE pool seeds never exceed SLATE's measured high-water (owner rule: a rank's device
peak must not rise above unseeded main).

distrib_la._slate sizes each SLATE handler's u8 workspace; cpp/slate seeds the matrices'
pools from it with the same rules.  A seeded block SLATE never touches is a byte main
never allocated, so every rule must sit at or below the pool's high-water.  The tables
are the measured minimum over the four ranks of blocks in use (pool_seed.h PoolWatch,
read under SLATE's slate_memory lock; MI250X 2x2, Frontier, dlascale lane job hwc,
``[slate.pool_hw]`` lines).  CPU-only: the rules are shape arithmetic.
"""
import pytest

pytest.importorskip("jax")

from distrib_la import _slate  # noqa: E402

P = Q = 2

#: (n, itemsize, min blocks in use: A, X) -- solve_lu, nb = 256, nrhs = q.
LU_HIGH_WATER = [(4096, 8, 75, 9), (4096, 16, 74, 8), (8192, 8, 284, 17), (8192, 16, 275, 17), (16384, 8, 1086, 33), (16384, 16, 1086, 33), (20480, 8, 1687, 41), (20480, 16, 1677, 41), (30720, 8, 3832, 61), (30720, 16, 3774, 61)]

#: (n, itemsize, min blocks in use: A, Z, Z1d) -- eigh, nb = 128.
EIGH_HIGH_WATER = [(4096, 8, 240, 256, 256), (4096, 16, 240, 256, 256), (8192, 8, 992, 1024, 1024), (8192, 16, 992, 1024, 1024), (16384, 8, 4032, 4096, 4096), (16384, 16, 4032, 4096, 4096), (20480, 8, 6320, 6400, 6400), (20480, 16, 6320, 6400, 6400), (30720, 8, 14280, 14400, 14400)]


@pytest.mark.parametrize("n,itemsize,hw_a,hw_x", LU_HIGH_WATER)
def test_lu_seeds_at_or_below_high_water(n, itemsize, hw_a, hw_x):
    nb = 256
    mt = nt = n // nb
    a = (n // P // nb) * (n // Q // nb) + _slate._lu_extra(mt, nt, P, Q)
    x = n // P // nb
    assert a <= hw_a and x <= hw_x
    # the workspace is exactly those two pools (no slack beside the seeds)
    want = _slate._pool_bytes((a, nb * nb * itemsize), (x, nb * (Q // Q) * itemsize))
    assert _slate.lu_workspace_bytes(n, Q, nb, P, Q, itemsize) == want


@pytest.mark.parametrize("n,itemsize,hw_a,hw_z,hw_z1d", EIGH_HIGH_WATER)
def test_eigh_seeds_at_or_below_high_water(n, itemsize, hw_a, hw_z, hw_z1d):
    nb = 128
    mt = nt = n // nb
    a = _slate._eigh_a_blocks(mt, nt, P, Q)
    z = _slate._max_local_tiles(mt, nt, P, Q)
    z1d = -(-nt // (P * Q)) * mt
    assert a <= hw_a and z <= hw_z and z1d <= hw_z1d
    # A's rule is the measured minimum itself (every n), not a margin below it
    assert a == hw_a


def test_potrf_seeds_only_past_the_outside_allowance():
    # one tile per rank on 2x2: SLATE reserves one block per local lower tile
    for n, itemsize in [(4096, 16), (16384, 16), (20480, 16), (30720, 8)]:
        assert _slate.potrf_workspace_bytes(n, n // P, P, Q, itemsize) == _slate._POOL_ALIGN
    tile = (30720 // P) ** 2 * 16                      # 3.77 GB > the 3 GB allowance
    assert _slate.potrf_workspace_bytes(30720, 30720 // P, P, Q, 16) == _slate._pool_bytes((1, tile))


def test_allowance_is_the_memory_models():
    runtime = pytest.importorskip("runtime")
    assert _slate._SLATE_OUTSIDE_ALLOWANCE_BYTES == runtime.DISTRIBUTED_LINALG_OUTSIDE_BYTES

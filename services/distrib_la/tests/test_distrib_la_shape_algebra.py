"""Layer L-a: shape algebra, on a laptop, in milliseconds.

Hostile geometry is mandatory for every mesh-touching service (charter),
and it has three layers.  This is the first: pure arithmetic over the tile
decomposition, no mesh, no devices, no ``.so``.  Layers L-b (four emulated
devices, real arrays) and L-c (four real processes under ``srun``) are the
service test suite's own step; what is here is the part that must never
need a machine.

The one algebraic contract ``native2d`` has is its tile decomposition, and
it has THREE constraints at once — ``n % b == 0`` and ``J % Px == 0`` and
``J % Py == 0`` with ``J = n / b``.  Two of the three hold by accident for
most inputs, which is exactly why the third is worth a test.
"""

from __future__ import annotations

import math

import pytest
from lxkit.testing import hostile_extents

from distrib_la._native2d import block_size_for


def _mesh_shapes():
    """The mesh shapes this service is actually run on, plus the awkward
    ones.  4x4 is the shape the hostile-geometry table was measured at
    (Perlmutter job 56389339); 2x4 and 3x2 are non-square and coprime-ish,
    which is where lcm(Px, Py) stops being max(Px, Py)."""
    return [(1, 1), (2, 2), (4, 1), (1, 4), (2, 4), (4, 4), (3, 2)]


@pytest.mark.parametrize("mesh_shape", _mesh_shapes())
def test_the_block_size_satisfies_all_three_constraints(mesh_shape):
    """Whatever it returns must actually tile: this is the contract, and it
    is checked by RE-DERIVING it, not by re-running the search."""
    Px, Py = mesh_shape
    n = 12 * math.lcm(Px, Py)
    b, J = block_size_for(n, Px, Py)
    assert b > 0 and J > 0
    assert n % b == 0, (n, b)
    assert J == n // b
    assert J % Px == 0 and J % Py == 0, (J, Px, Py)


@pytest.mark.parametrize("mesh_shape", _mesh_shapes())
def test_hostile_extents_are_refused_or_tiled_honestly(mesh_shape):
    """Non-dividing extents either tile or REFUSE — never silently round.

    The five families come from :func:`lxkit.testing.hostile_extents`,
    which generalizes the table measured end-to-end on a 4x4 mesh (job
    56389339): prime extents on both axes, a tighter prime pair,
    non-divisible on one axis only, fewer slices than ranks, and empty
    tiles on one axis.  A prime n on a 2x2 mesh has NO valid decomposition,
    and the honest answer is the raise — silently padding would change the
    matrix.
    """
    Px, Py = mesh_shape
    for case in hostile_extents(mesh_shape):
        for n in (case.logical[0], case.padded[0]):
            try:
                b, J = block_size_for(n, Px, Py)
            except ValueError as exc:
                assert str(math.lcm(Px, Py)) in str(exc), (
                    f"a refusal must name the divisor it wanted: {exc}")
                continue
            assert n % b == 0 and (n // b) % Px == 0 and (n // b) % Py == 0, (
                f"block_size_for({n}, {Px}, {Py}) = {(b, J)} does not tile")


def test_a_prime_extent_on_a_real_mesh_refuses():
    """The anti-tautology cell: something in here has to actually raise, or
    the test above is a loop over cases that all succeed."""
    with pytest.raises(ValueError, match=r"no valid block size"):
        block_size_for(17, 2, 2)


def test_the_refusal_names_the_divisor_and_the_mesh():
    exc = pytest.raises(ValueError, block_size_for, 17, 2, 2).value
    assert "17" in str(exc) and "2x2" in str(exc) and "lcm(2,2)=2" in str(exc)


def test_dense_to_tiles_round_trips_on_the_lower_triangle():
    """L-a's other half: the layout conversion the facade hides.

    ``dense_to_tiles`` zeroes the strictly-upper TILES (``i < j`` in TILE
    indices), not the strictly-upper ELEMENTS — inside a DIAGONAL tile the
    upper triangle survives, because that is what ``jnp.linalg.cholesky``
    is handed and it reads the lower triangle itself.  So the round trip is
    the identity on ``tril`` and NOT on the full matrix, and the zero half
    is BLOCK-triangular, not triangular.

    Both halves are asserted, and asserting the wrong one is not
    hypothetical: the first draft of this cell claimed elementwise
    ``triu(back, 1) == 0`` and failed at ``b = 2`` on the diagonal tiles.
    A round-trip check that only compares ``tril`` cannot see the
    difference, which is how such a check becomes a no-op.
    """
    np = pytest.importorskip("numpy")
    jnp = pytest.importorskip("jax.numpy")
    from distrib_la._native2d import dense_to_tiles, tiles_to_dense

    rng = np.random.default_rng(4)
    A = jnp.asarray(rng.standard_normal((3, 12, 12)))
    idx = np.arange(12)
    for b in (1, 2, 3, 4, 6, 12):
        tiles = dense_to_tiles(A, b)
        assert tiles.shape == (3, 12 // b, 12 // b, b, b)
        back = tiles_to_dense(tiles, b)
        assert back.shape == A.shape
        assert bool(jnp.array_equal(jnp.tril(back), jnp.tril(A))), b
        above = jnp.asarray((idx[:, None] // b) < (idx[None, :] // b))
        assert bool(jnp.all(jnp.where(above, back, 0) == 0)), b
        # ...and the diagonal tiles are UNTOUCHED, which is the half a
        # tril-only comparison cannot see.
        on_diag = jnp.asarray((idx[:, None] // b) == (idx[None, :] // b))
        assert bool(jnp.array_equal(jnp.where(on_diag, back, 0),
                                    jnp.where(on_diag, A, 0))), b

    with pytest.raises(ValueError, match="divisible"):
        dense_to_tiles(A, 5)

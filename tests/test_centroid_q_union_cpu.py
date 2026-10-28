"""The q-union prune's q shells and k+q partner rows (no WFN)."""
import numpy as np

from centroid.pivoted_cholesky import prune_q_union


class _Grid:
    """The loader attributes ``prune_q_union`` reads, for a plain k grid."""

    def __init__(self, kgrid, avec, shift=(0.0, 0.0, 0.0)):
        self.kgrid = np.asarray(kgrid)
        self.avec = np.asarray(avec, dtype=float)
        self.bvec = np.linalg.inv(self.avec).T
        self.blat = 1.0
        idx = np.stack(np.meshgrid(*[np.arange(n) for n in kgrid],
                                   indexing="ij"), -1).reshape(-1, 3)
        self._k = (idx + np.asarray(shift)) / np.asarray(kgrid)
        # A full-BZ table need not be in grid order.
        self._k = self._k[np.random.default_rng(1).permutation(len(self._k))]

    def kvecs(self, *, k="full_bz"):
        return self._k


def _check_partners(grid, q, partner):
    for qi, rows in zip(q, partner):
        assert sorted(rows.tolist()) == list(range(len(grid._k)))
        d = grid._k[rows] - grid._k - qi
        assert np.allclose(d - np.rint(d), 0.0)


def test_hexagonal_shells_run_shortest_to_longest():
    avec = [[1.0, 0.0, 0.0], [-0.5, np.sqrt(3) / 2, 0.0], [0.0, 0.0, 5.0]]
    grid = _Grid((8, 8, 1), avec)
    q, partner = prune_q_union(grid, None, 3)
    b = grid.blat * grid.bvec
    lengths = np.linalg.norm(q @ b, axis=1)
    assert q.shape == (3, 3) and np.all(np.diff(lengths) > 0)
    assert np.isclose(lengths[0], np.linalg.norm(b[0]) / 8)
    # The longest shell on an 8x8 hexagonal grid is the one nearest K.
    images = np.array([[i, j, 0] for i in range(8) for j in range(8)]) / 8.0
    shortest = [min(np.linalg.norm((g + np.array([m, n, 0])) @ b)
                    for m in (-1, 0, 1) for n in (-1, 0, 1)) for g in images]
    assert np.isclose(lengths[-1], max(shortest))
    _check_partners(grid, q, partner)


def test_shifted_grid_partners_and_shell_cap():
    grid = _Grid((3, 3, 1), np.diag([1.0, 1.3, 6.0]), shift=(0.5, 0.5, 0.0))
    q, partner = prune_q_union(grid, None, 10)
    # A 3x3 rectangular grid has three distinct nonzero |q|.
    assert q.shape == (3, 3)
    _check_partners(grid, q, partner)

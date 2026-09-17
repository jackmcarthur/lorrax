"""The −q partner of a raw parent: representative, unitary operation, and the antiunitary refusal.

Tables come from ``find_irreducible_bz_points`` on 3x3x1 and 4x4x1 grids with the groups a
two-dimensional magnet can keep: C3 about z (no q ↔ −q relation inside the group), and
{E, C2z} (a unitary q → −q). The time-reversal-augmented rows (−S, typed antiunitary) are
the rows a ferromagnet does not have. Hand values are written out; nothing re-derives the
function under test.
"""
from __future__ import annotations

import numpy as np
import pytest

from symmetry_maps import find_irreducible_bz_points, minus_q_parent_partners, q_negation_index

GRID = (3, 3, 1)
# C3 about z in crystal coordinates of a triangular lattice (a1 -> a2 - a1, a2 -> -a1).
M3 = np.array([[-1, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int64)
C2 = np.diag([-1, -1, 1]).astype(np.int64)


def _k_action(mtrx):
    """sym_mats_k = mtrx^T for BGW's G-vector matrix mtrx (r-action mtrx^-1)."""
    return np.stack([m.T for m in mtrx])


def _tables(mats_k, grid=GRID):
    coords = np.stack(np.unravel_index(np.arange(int(np.prod(grid))), grid), axis=1)
    irr, sym, reps = find_irreducible_bz_points(coords, mats_k)
    full = np.array([int(np.ravel_multi_index(tuple(r), grid)) for r in reps])
    return full, irr, sym, coords


def _check_images(full, partner, row, mats_k, coords, grid=GRID):
    neg = q_negation_index(grid)
    for p, q in enumerate(full):
        image = (mats_k[row[p]] @ coords[full[partner[p]]]) % np.asarray(grid)
        assert image.tolist() == coords[neg[q]].tolist(), (p, image, coords[neg[q]])


def test_c3_magnet_pairs_minus_q_with_another_parent_through_a_rotation():
    """4x4 grid under C3: parents (1,2) and (2,1) carry each other's -q through the two rotations.

    Hand check: S_2 = M^T = [[-1,1],[-1,0]] sends (2,1) to (-1,-2) = (3,2) = -(1,2); S_1 = (M^2)^T =
    [[0,-1],[1,-1]] sends (1,2) to (-2,-1) = (2,3) = -(2,1).
    """
    grid = (4, 4, 1)
    mtrx = [np.eye(3, dtype=np.int64), M3 @ M3, M3]
    mats_k = _k_action(mtrx)
    full, irr, sym, coords = _tables(mats_k, grid)
    anti = np.zeros(3, bool)
    partner, row = minus_q_parent_partners(full, irr, sym, kgrid=grid, sym_mats_k=mats_k,
                                           antiunitary=anti, authorized_rows=np.arange(3))
    _check_images(full, partner, row, mats_k, coords, grid)
    assert full.tolist() == [0, 1, 2, 3, 6, 9]
    assert partner.tolist() == [0, 3, 2, 1, 5, 4]
    assert row.tolist() == [0, 0, 0, 0, 2, 1]
    # RED TWIN: the full-grid spelling (index of -q among the parents) has no parent for q = (1,2).
    neg = q_negation_index(grid)
    assert int(neg[6]) not in full.tolist()


def test_antiunitary_only_route_refuses_by_name():
    mtrx = [np.eye(3, dtype=np.int64), M3 @ M3, M3]
    mats_k = np.concatenate([_k_action(mtrx), -_k_action(mtrx)])
    full, irr, sym, coords = _tables(mats_k)
    anti = np.array([False] * 3 + [True] * 3)
    with pytest.raises(ValueError, match="GATE minus_q_partner.*antiunitary row"):
        minus_q_parent_partners(full, irr, sym, kgrid=GRID, sym_mats_k=mats_k,
                                antiunitary=anti, authorized_rows=np.arange(6))


def test_a_unitary_route_is_taken_when_the_table_names_an_antiunitary_row():
    mats_k = np.stack([np.eye(3, dtype=np.int64), C2, -np.eye(3, dtype=np.int64), -C2])
    anti = np.array([False, False, True, True])
    full, irr, sym, coords = _tables(mats_k)
    neg = q_negation_index(GRID)
    edited = sym.copy()
    for q in range(9):
        if int(neg[q]) != q and irr[q] == irr[neg[q]] and q not in full.tolist() and sym[q] == 1:
            edited[q] = 2      # -E maps the parent to -q as well: a legal table, antiunitary
    assert np.any(edited != sym)
    partner, row = minus_q_parent_partners(full, irr, edited, kgrid=GRID, sym_mats_k=mats_k,
                                           antiunitary=anti, authorized_rows=np.arange(4))
    _check_images(full, partner, row, mats_k, coords)
    assert not np.any(anti[row])
    changed = [p for p, q in enumerate(full) if edited[neg[q]] == 2]
    assert changed and all(row[p] == 1 for p in changed)


def test_a_table_built_with_another_q_action_refuses():
    mtrx = [np.eye(3, dtype=np.int64), M3 @ M3, M3]
    mats_k = _k_action(mtrx)
    full, irr, sym, _ = _tables(np.stack([m.T for m in mats_k]))
    with pytest.raises(ValueError, match="GATE minus_q_partner.*not sym_mats_k"):
        minus_q_parent_partners(full, irr, sym, kgrid=GRID, sym_mats_k=mats_k,
                                antiunitary=np.zeros(3, bool), authorized_rows=np.arange(3))


def test_full_grid_parents_reduce_to_the_negation_index():
    """No reduction: every q is a parent and −q's parent is −q itself, with the identity row."""
    mats_k = np.eye(3, dtype=np.int64)[None]
    full, irr, sym, _ = _tables(mats_k)
    partner, row = minus_q_parent_partners(full, irr, sym, kgrid=GRID, sym_mats_k=mats_k,
                                           antiunitary=np.zeros(1, bool), authorized_rows=[0])
    assert full.tolist() == list(range(9))
    assert partner.tolist() == [0, 2, 1, 6, 8, 7, 3, 5, 4]
    assert row.tolist() == [0] * 9

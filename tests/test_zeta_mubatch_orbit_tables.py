"""Whole-orbit μ batches and orbit-closed rank r blocks for the μ-batch ζ fit.

Host-only contracts of ``gw.centroid_k_unfold.orbit_mu_batches`` and
``orbit_r_blocks`` (docs/architecture/zeta_fit_mubatch.md, "Symmetry: parent
k"), on the committed A-cubic fixture (diamond-H2, 48 operations, time
reversal, 3 raw parents of 8 k) and a layered synthetic group.  The endpoint
tables are checked against an evaluation that shares no code with their
builder: the Seitz action applied to fractional coordinates directly.  Red
twins construct a split orbit and a permuted table and show the refusal.
The device parity against the r-chunk path is
``tests/test_zeta_mubatch_sym_parity.py``.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gw.centroid_k_unfold import (
    build_centroid_k_unfold_plan,
    build_real_grid_orbit_tiles,
    mu_batch_tables,
    orbit_mu_batches,
    orbit_r_blocks,
    r_block_tables,
)

_ACUBIC = Path(__file__).resolve().parent / "core" / "fixtures" / "A-cubic"
# The GW mesh is square; the builders only read its shape.
_MESH_2X2 = SimpleNamespace(shape={"x": 2, "y": 2})


def _acubic_plan():
    from file_io import WfnLoader

    with WfnLoader(_ACUBIC / "WFN.h5", backend="eager",
                   qe_schema=_ACUBIC / "data-file-schema.xml") as loader:
        sym = loader.symmetry()
        fft_grid = np.asarray(loader.fft_grid, dtype=np.int64)
        k_parent = loader.kvecs(k=sym.parent_k_domain)
    frac = np.loadtxt(_ACUBIC / "centroids_frac_48.txt")
    idx = (np.rint(frac * fft_grid) % fft_grid).astype(np.int32)
    plan = build_centroid_k_unfold_plan(
        sym, idx, fft_grid, _MESH_2X2, nspinor=1, parent_k_frac=k_parent)
    return plan, idx


@pytest.fixture(scope="module")
def acubic():
    return _acubic_plan()


def _seitz_source(plan, fft_grid, flat, row):
    """Source point and wrap of ``row`` for flat grid targets, from the Seitz
    rows directly: ``y = mtrx·(x − τ) = x_src + L`` (BGW convention)."""
    fg = np.asarray(fft_grid, dtype=np.int64)
    n_sym = int(plan.n_sym_spatial)
    s = int(row) % n_sym                      # time reversal fixes r
    x = np.stack([flat // (fg[1] * fg[2]), (flat // fg[2]) % fg[1],
                  flat % fg[2]], axis=-1) / fg
    tau = np.asarray(plan.translations, dtype=np.float64)[s] / (2 * np.pi)
    y = (np.asarray(plan.spatial_ops, dtype=np.float64)[s] @ (x - tau).T).T
    L = np.floor(y + 1e-9)
    src = np.rint((y - L) * fg).astype(np.int64) % fg
    return src[:, 0] * fg[1] * fg[2] + src[:, 1] * fg[2] + src[:, 2], L


# ---------------------------------------------------------------------------
# μ batches
# ---------------------------------------------------------------------------

def test_the_fixture_exercises_parent_k(acubic):
    """Route receipt: parents < full k and non-identity rows are used.

    A-cubic has inversion, so every child is reached by a spatial row; the
    antiunitary rows are exercised by the glide cases of the parity test.
    """
    plan, _ = acubic
    rows = np.unique(plan.sym_idx)
    assert (plan.n_parent, plan.n_full, plan.n_sym_spatial) == (3, 8, 48)
    assert rows.size > 1 and np.all(rows < plan.n_sym_spatial), rows
    # The 2x2 grouped layout pads: the builder must skip those slots.
    assert plan.n_centroid_packed > plan.n_centroid_logical


@pytest.mark.parametrize("n_ranks,b_target", [(4, 16), (4, 24), (4, 64), (2, 30), (3, 7)])
def test_mu_batches_are_whole_orbits_with_plan_tables(acubic, n_ranks, b_target):
    plan, _ = acubic
    mb = orbit_mu_batches(plan, plan.n_centroid_packed, n_ranks, b_target=b_target)
    assert mb.b % n_ranks == 0 and mb.c * n_ranks == mb.b
    active = np.flatnonzero(plan.layout.axis.active_mask)
    got = mb.mu[mb.mu >= 0]
    assert np.array_equal(np.sort(got), active), "every active centroid once"
    rows = np.unique(plan.sym_idx)
    assert np.array_equal(mb.rows, rows)
    for beta in range(mb.n_batch):
        slots = np.flatnonzero(mb.mu[beta] >= 0)
        members = mb.mu[beta, slots]
        for row in range(plan.sym_perm.shape[0]):
            if row not in rows:
                assert np.all(mb.left_perm[beta, row] == -1)
                continue
            src = mb.left_perm[beta, row, slots]
            # batch-local source slot ↔ the plan's packed source map
            assert np.array_equal(mb.mu[beta, src], plan.sym_perm[row, members])
            assert np.array_equal(mb.left_L[beta, row, slots],
                                  plan.L_table[row, members])
            pads = np.flatnonzero(mb.mu[beta] < 0)
            assert np.array_equal(mb.left_perm[beta, row, pads], pads)
            assert not mb.left_L[beta, row, pads].any()
    # Balance: per-rank rows of one batch differ by at most one.
    per_rank = (mb.rank_mu >= 0).sum(-1)
    assert np.all(per_rank.max(1) - per_rank.min(1) <= 1), per_rank
    # packed_to_slot inverts mu and marks the layout pads.
    p2s = mb.packed_to_slot(plan.n_centroid_packed)
    assert np.array_equal(mb.mu.reshape(-1)[p2s[active]], active)
    assert np.all(p2s[~plan.layout.axis.active_mask] == -1)


def test_mu_batch_width_is_raised_to_one_orbit(acubic):
    """A target below the largest orbit widens the batch (reported via b)."""
    plan, _ = acubic
    from symmetry_maps import permutation_orbit_labels
    lab = permutation_orbit_labels(plan.sym_perm[np.unique(plan.sym_idx)])
    largest = np.bincount(lab[plan.layout.axis.active_mask]).max()
    mb = orbit_mu_batches(plan, plan.n_centroid_packed, 4, b_target=1)
    assert mb.b == -(-largest // 4) * 4


def test_mu_batch_count_is_minimal_and_balanced(acubic):
    """Equal 24-centroid orbits: the batch count is the capacity bound and
    the realized width shrinks to the largest load."""
    plan, _ = acubic
    n_act = int(plan.layout.axis.active_mask.sum())
    for b_target, n_batch, b in ((24, 2, 24), (40, 2, 24), (48, 1, 48), (96, 1, 48)):
        mb = orbit_mu_batches(plan, plan.n_centroid_packed, 4, b_target=b_target)
        loads = (mb.mu >= 0).sum(1)
        assert (mb.n_batch, mb.b) == (n_batch, b), (b_target, mb.n_batch, mb.b)
        assert loads.sum() == n_act and loads.max() - loads.min() <= 24, loads


def test_split_orbit_refuses_by_name(acubic):
    """Red twin: swap one centroid between two batches of distinct orbits."""
    plan, _ = acubic
    mb = orbit_mu_batches(plan, plan.n_centroid_packed, 4, b_target=24)
    assert mb.n_batch >= 2
    bad = mb.mu.copy()
    a = int(np.flatnonzero(bad[0] >= 0)[0])
    b = int(np.flatnonzero(bad[1] >= 0)[0])
    bad[0, a], bad[1, b] = bad[1, b], bad[0, a]
    with pytest.raises(ValueError, match="not a union of whole orbits"):
        mu_batch_tables(plan, bad)
    # A duplicated or missing centroid refuses too.
    dup = mb.mu.copy()
    dup[1] = dup[0]
    with pytest.raises(ValueError, match="exactly one slot"):
        mu_batch_tables(plan, dup)


# ---------------------------------------------------------------------------
# r blocks
# ---------------------------------------------------------------------------

def _check_blocks(plan, rb, fft_grid):
    """Partition, closure, table semantics (independent Seitz evaluation)."""
    n_rtot = int(np.prod(fft_grid))
    act = rb.points[rb.points >= 0]
    assert np.array_equal(np.sort(act), np.arange(n_rtot)), "a partition"
    rows = rb.local_perm.shape[2]
    assert rows == 2 * plan.n_sym_spatial
    for p in range(rb.n_ranks):
        for s in range(rb.n_sub):
            pts = rb.points[p, s]
            live = np.flatnonzero(pts >= 0)
            assert np.all(pts[live[-1] + 1:] < 0) if live.size else True, \
                "pads trail within a block"
            for row in range(rows):
                src, L = _seitz_source(plan, fft_grid, pts[live], row)
                off = rb.local_perm[p, s, row]
                assert np.array_equal(pts[off[live]], src), (p, s, row)
                assert np.array_equal(rb.wraps[p, s, row, live], L.astype(int))
                pad = np.flatnonzero(pts < 0)
                assert np.array_equal(off[pad], pad)
            pl = rb.planes[p, s]
            coord = np.stack([pts[live] // (fft_grid[1] * fft_grid[2]),
                              (pts[live] // fft_grid[2]) % fft_grid[1],
                              pts[live] % fft_grid[2]])[rb.plane_axis]
            assert np.array_equal(pl[pl >= 0], np.unique(coord))


@pytest.mark.parametrize("route", ["cache", "planes"])
@pytest.mark.parametrize("n_ranks,r_s", [(4, 96), (4, 216), (3, 100), (2, 500)])
def test_r_blocks_partition_the_grid_into_closed_blocks(acubic, route, n_ranks, r_s):
    plan, _ = acubic
    fg = tuple(int(v) for v in plan.fft_grid)
    rb = orbit_r_blocks(plan, fg, n_ranks, r_s_target=r_s, route=route)
    assert rb.route == route and rb.points.shape[0] == n_ranks
    assert rb.r_s <= max(r_s, 48), (rb.r_s, r_s)     # 48 = the largest orbit
    _check_blocks(plan, rb, np.asarray(fg))


def test_orbit_split_across_ranks_refuses(acubic):
    """Red twin: exchange one point between two ranks' blocks of one sub-block."""
    plan, _ = acubic
    fg = tuple(int(v) for v in plan.fft_grid)
    rb = orbit_r_blocks(plan, fg, 4, r_s_target=216, route="cache")
    pts = rb.points.copy()
    from symmetry_maps import real_space_orbit_labels
    lab = real_space_orbit_labels(plan.spatial_ops, plan.translations, fg)
    a, b = pts[0, 0, 0], pts[1, 0, 0]
    assert lab[a] != lab[b]
    pts[0, 0, 0], pts[1, 0, 0] = b, a
    with pytest.raises(ValueError, match="not a union of whole orbits"):
        r_block_tables(plan, fg, pts)
    # Across sub-blocks the tile itself is no longer closed: refused too.
    pts = rb.points.copy()
    a, b = pts[0, 0, 0], pts[0, 1, 0]
    pts[0, 0, 0], pts[0, 1, 0] = b, a
    with pytest.raises(ValueError, match="not a union of whole orbits"):
        r_block_tables(plan, fg, pts)


def test_owner_contiguous_blocks_stay_on_their_own_planes():
    """Layered group (C4 x sigma_h): the 'planes' route keeps each rank on a
    contiguous run of stacking planes (plus mirror partners), while every
    block remains orbit closed."""
    c4 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int64)
    sh = np.diag([1, 1, -1]).astype(np.int64)
    rots = [np.linalg.matrix_power(c4, i) for i in range(4)]
    ops = np.array(rots + [sh @ r for r in rots])
    fg = np.asarray((8, 8, 16))
    plan = SimpleNamespace(spatial_ops=ops, translations=np.zeros((8, 3)),
                           fft_grid=fg, n_sym_spatial=8)
    rb = orbit_r_blocks(plan, fg, 4, r_s_target=64, route="planes")
    assert rb.plane_axis == 2
    _check_blocks(plan, rb, fg)
    # Orbits pair plane z with -z; runs of the plane order meet only at
    # run boundaries, each adding at most one shared (z, -z) pair.
    per_rank = [set(rb.planes[p][rb.planes[p] >= 0].tolist()) for p in range(4)]
    assert sum(len(s) for s in per_rank) <= fg[2] + 2 * (4 - 1), per_rank
    # a block of 64 points on 8x8 planes: at most two (z, -z) pairs
    assert (rb.planes >= 0).sum(-1).max() <= 4
    # the r-chunk fill on the same group deals every rank onto the tile's
    # planes: the per-rank plane sets then overlap on every plane
    tiles = build_real_grid_orbit_tiles(ops, np.zeros((8, 3)), fg, n_y=4,
                                        target_width=256, fill="least_loaded")
    owners = tiles.owner_planes()
    total = sum(len(set(owners[t, o][owners[t, o] >= 0].tolist()))
                for t in range(tiles.n_tiles) for o in range(4))
    assert total > sum(len(s) for s in per_rank), total


def test_fill_is_a_named_selection():
    with pytest.raises(TypeError):
        build_real_grid_orbit_tiles(np.eye(3, dtype=int)[None], np.zeros((1, 3)),
                                    (4, 4, 4), n_y=2, target_width=16)
    with pytest.raises(ValueError, match="fill must be"):
        build_real_grid_orbit_tiles(np.eye(3, dtype=int)[None], np.zeros((1, 3)),
                                    (4, 4, 4), n_y=2, target_width=16, fill="lpt")

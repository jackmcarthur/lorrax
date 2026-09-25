"""Whole-orbit μ batches for the route-G μ-batch ζ fit (the owners' bins).

Host-only contracts of ``gw.centroid_k_unfold.orbit_mu_batches``
(docs/architecture/zeta_fit_mubatch.md), on the committed A-cubic fixture
(diamond-H2, 48 operations, time reversal, 3 raw parents of 8 k).  The
endpoint tables are checked against an evaluation that shares no code with
their builder: the Seitz action applied to fractional coordinates directly.
A red twin constructs a split orbit and shows the refusal.  The device
parity is ``tests/multi_device/zeta_mubatch_p4.py``.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gw.centroid_k_unfold import (
    build_centroid_k_unfold_plan,
    mu_batch_tables,
    orbit_mu_batches,
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


def test_route_g_plans_the_whole_tile_tier_under_linalg_distributed():
    """`linalg = distributed` sets the other stages; route G's ζ tier stays
    the planner's whole-tile choice, stated once in the plan receipt."""
    from gw.gw_config import resolve_linalg
    from gw.gw_init import _plan_route_g_for_channel
    from gw.wavefunction_bundle import BandSlices
    prof = resolve_linalg({"linalg": "distributed"})
    assert (prof.w_dyson_solver, prof.eigh_backend, prof.distributed_lu) == (
        "distributed", "distributed", "distributed")
    meta = SimpleNamespace(nk_tot=64, nspinor=2, n_rmu=1446, n_rmu_padded=1448,
                           n_rtot=60 * 60 * 180, fft_grid=(60, 60, 180))
    cfg = SimpleNamespace(
        zeta_nband=144,
        memory=SimpleNamespace(
            per_device_gb=33.9, chunk_target_utilization=0.0,
            low_mem_bands=False),
        backend=SimpleNamespace(
            distributed_zeta_solve=prof.distributed_zeta_solve,
            charge_zeta_solve=prof.charge_zeta_solve))
    mesh = SimpleNamespace(shape={'x': 2, 'y': 2},
                           devices=np.empty(4, dtype=object))
    lines = []
    chunks = _plan_route_g_for_channel(
        meta=meta, cfg=cfg,
        band_slices=BandSlices.from_band_edges(0, 0, 130, 144, 144),
        mesh_xy=mesh, n_q_selected=10, n_parent=10,
        print_fn=lines.append, zeta_ngkmax=8000, psi_ngkmax=12000)
    assert chunks["mubatch"].zeta_tier == "local"
    receipt = "\n".join(lines).splitlines()
    assert sum("ζ tier" in line for line in receipt) == 1


def test_best_owner_batching_never_packs_worse_than_the_planned_bin(acubic):
    """The chosen bins are no wider than planned and cost no more padded work
    than packing at the planned width."""
    from isdf.zeta_mubatch import best_owner_orbit_batches, owner_orbit_batches
    plan, _ = acubic
    mu_pad = int(plan.n_centroid_packed)
    for c_max in (3, 5, 7, 11):
        got = best_owner_orbit_batches(plan, mu_pad, 4, c_max=c_max)
        at_plan = owner_orbit_batches(plan, mu_pad, 4, c_target=c_max)
        assert got.c <= max(c_max, at_plan.c)
        assert got.n_batch * (got.c + 1) <= at_plan.n_batch * (at_plan.c + 1)
        live = got.mu[got.mu >= 0]
        assert np.array_equal(np.sort(live), np.unique(live))   # each centroid once

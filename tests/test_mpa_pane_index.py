"""The pane as an index set: same panes, and the gate that proves it.

WHAT THIS FILE IS FOR.  The MPA width clause partitions a non-crossing
branch into ~218 panes on the audited field, and each pane used to say
which modes it held with a full-size ``(n_q, N_mu, N_mu)`` boolean --
81.4 MB apiece, **17.8 GB** for the partition, which the 2026-08-09
memory row measured as the whole of one pass's excess over the two-point
path.  The panes are now index sets, and the panes are the ONLY thing
that must not have changed: same membership, same order, same rules, same
nodes.  These cells are the executable form of that "must not".

WHAT THEY DO NOT CLAIM.  Nothing here is a cost gate.  The mask-dependent
stage of a tau node is 4.6 ms of a 165 ms node at production shapes
(measured on an idle A100), so a pane at 1/218 occupancy costs what a
full pane costs and no representation of its membership changes that.
The pane COUNT is the cost, and it is the owner's registered row.
"""

from __future__ import annotations

import numpy as np
import pytest

from gw.mpa import sigma_pass as SP


def _field(n_q=6, n_mu=24, seed=11):
    """A pole field with the audited geometry's character, miniaturised.

    ``Gamma = ratio * a`` with ``ratio <= 1`` is the fitter's fourth
    guard, ``a`` spans two decades, and ``e_lo + min(a) < omega_max`` --
    the same relation that floored the old predicate -- so the planner
    takes the aligned path and returns a real handful of panes.
    """
    rng = np.random.default_rng(seed)
    shape = (n_q, n_mu, n_mu)
    a = np.exp(rng.uniform(np.log(0.02), np.log(2.0), size=shape))
    g = np.exp(rng.uniform(np.log(1e-4), np.log(1.0), size=shape)) * a
    B = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    E_A = np.linspace(0.0255, 4.1424, 16)
    return dict(
        a_ry=a, gamma_ry=g, B=B,
        live_mask=np.ones(shape, dtype=bool),
        E_A_host=E_A, base_mask_A_host=np.ones(E_A.size, dtype=bool),
        omega_nonneg_ry=np.linspace(0.0, 0.147, 5),
        xi_ry=0.476 / 13.6056980659, edge_factor=4.0)


def _plan(fld, *, space="val", neg=False):
    return SP.plan_branch_groups(
        a_ry=fld["a_ry"], gamma_ry=fld["gamma_ry"],
        live_mask=fld["live_mask"], E_A_host=fld["E_A_host"],
        base_mask_A_host=fld["base_mask_A_host"],
        omega_nonneg_ry=fld["omega_nonneg_ry"], space=space,
        neg_omega_half=neg, xi_ry=fld["xi_ry"],
        edge_factor=fld["edge_factor"], b_abs=np.abs(fld["B"]),
        print_fn=lambda *a, **k: None)


def _W_from_mask(mask, B, omega_operand, e_ref_b, t):
    """The tau kernel's W(t) built the way it was before -- dense mask."""
    ph = np.exp(-1j * (omega_operand - e_ref_b) * t)
    return np.where(mask, B * ph, np.asarray(0.0 + 0.0j))


def _W_from_index(idx, shape, B, omega_operand, e_ref_b, t):
    """The same W(t) built by gathering the pane's live modes."""
    W = np.zeros(shape, dtype=np.complex128)
    flat = W.reshape(-1)
    om = np.asarray(omega_operand).reshape(-1)[idx]
    flat[idx] = np.asarray(B).reshape(-1)[idx] * np.exp(-1j * (om - e_ref_b) * t)
    return W


def _pass_sigma(groups, fld, *, w_of_group):
    """A whole pass's accumulated Sigma, in the pinned group order.

    Not the production contraction -- a fixed linear functional of the
    same W(t) operands, summed over every group and every certified tau
    node in the order the pass loop walks them.  That is exactly the
    property under test: the operands and the order, not the kernel.
    """
    rng = np.random.default_rng(4)
    G = (rng.standard_normal(fld["a_ry"].shape)
         + 1j * rng.standard_normal(fld["a_ry"].shape))
    total = 0.0 + 0.0j
    for grp in groups:
        for win in grp.windows:
            t_host = np.asarray(win.nodes.t, dtype=np.complex128)
            alpha = np.asarray(win.nodes.alpha, dtype=np.complex128)
            aeff = alpha * np.exp(-1j * (win.E_ref_A + win.E_ref_B) * t_host)
            for j in range(t_host.size):
                W = w_of_group(grp, win, t_host[j])
                total += aeff[j] * np.sum(W * G)
    return total


def test_the_panes_partition_the_live_set_exactly():
    """Every live mode in exactly one pane -- the failure this guards is
    a mode summed twice or never, which returns a finite, smooth Sigma of
    the right shape and tens of meV of error."""
    fld = _field()
    groups, stats = _plan(fld)
    assert len(groups) > 3, [g.name for g in groups]
    tally = np.zeros(fld["a_ry"].size, dtype=np.int64)
    for grp in groups:
        assert grp.idx_B.ndim == 1
        assert np.all(np.diff(grp.idx_B) > 0), (
            f"{grp.name}: the index set must be strictly ascending, which "
            f"is what makes it the same selector a dense mask was")
        tally[np.asarray(grp.idx_B, dtype=np.int64)] += 1
    assert np.array_equal(tally.reshape(fld["a_ry"].shape),
                          fld["live_mask"].astype(np.int64))
    assert sum(int(g.n_modes) for g in groups) == int(fld["live_mask"].sum())
    assert stats["n_narrow"] + stats["n_wide"] == fld["live_mask"].size


def test_the_dense_mask_a_group_stands_for_is_the_one_the_kernel_gets():
    """``dense_mask_B()`` is the boolean, not a boolean: the tau loop is
    handed exactly the selector the index set names, so nothing below the
    planner can tell the two representations apart."""
    fld = _field()
    groups, _ = _plan(fld)
    for grp in groups:
        m = grp.dense_mask_B()
        assert m.shape == fld["a_ry"].shape and m.dtype == bool
        assert int(m.sum()) == int(grp.n_modes)
        assert np.array_equal(np.flatnonzero(m.reshape(-1)),
                              np.asarray(grp.idx_B, dtype=np.int64))


def test_the_W_operand_is_bit_identical_per_pane():
    """PER-PANE EQUALITY.  W(t) built by masking the whole field and W(t)
    built by gathering the pane's live modes are the SAME BITS, because
    the expression per element is the same expression -- no summation is
    re-associated by the change, so there is no roundoff to report."""
    fld = _field()
    groups, _ = _plan(fld)
    n_checked = 0
    for grp in groups:
        mask = grp.dense_mask_B()
        for win in grp.windows:
            for t in np.asarray(win.nodes.t, dtype=np.complex128)[:3]:
                dense = _W_from_mask(mask, fld["B"], grp.omega_operand,
                                     win.E_ref_B, t)
                comp = _W_from_index(np.asarray(grp.idx_B, dtype=np.int64),
                                     fld["a_ry"].shape, fld["B"],
                                     grp.omega_operand, win.E_ref_B, t)
                assert np.array_equal(dense, comp), grp.name
                assert np.isfinite(dense).all()
                n_checked += 1
    assert n_checked > 10


def test_the_accumulated_sigma_over_a_whole_pass_is_bit_identical():
    """FULL-PASS EQUALITY on a small deck: every group, every certified
    node, accumulated in the pinned order.  Bit-identical, and the number
    is reported rather than asserted away."""
    fld = _field()
    groups, _ = _plan(fld)
    s_dense = _pass_sigma(
        groups, fld,
        w_of_group=lambda g, w, t: _W_from_mask(
            g.dense_mask_B(), fld["B"], g.omega_operand, w.E_ref_B, t))
    s_comp = _pass_sigma(
        groups, fld,
        w_of_group=lambda g, w, t: _W_from_index(
            np.asarray(g.idx_B, dtype=np.int64), fld["a_ry"].shape, fld["B"],
            g.omega_operand, w.E_ref_B, t))
    assert s_dense == s_comp, (s_dense, s_comp, abs(s_dense - s_comp))


def test_one_dropped_index_fails_the_equality_gate():
    """THE RED TWIN.  A pane that quietly loses ONE of its eighty-odd
    thousand modes is the failure this whole representation could
    introduce, and it produces a finite, smooth, entirely plausible
    Sigma.  The gate above must see it -- if it does not, the gate is
    decoration."""
    fld = _field()
    groups, _ = _plan(fld)
    victim = max(range(len(groups)), key=lambda i: groups[i].n_modes)
    good = np.asarray(groups[victim].idx_B, dtype=np.int64)
    assert good.size > 1
    wrong = {id(groups[victim]): np.delete(good, good.size // 2)}

    def w_wrong(g, w, t):
        idx = wrong.get(id(g), np.asarray(g.idx_B, dtype=np.int64))
        return _W_from_index(idx, fld["a_ry"].shape, fld["B"],
                             g.omega_operand, w.E_ref_B, t)

    s_good = _pass_sigma(
        groups, fld,
        w_of_group=lambda g, w, t: _W_from_mask(
            g.dense_mask_B(), fld["B"], g.omega_operand, w.E_ref_B, t))
    s_bad = _pass_sigma(groups, fld, w_of_group=w_wrong)
    assert s_good != s_bad, (
        "dropping a mode from a pane left the accumulated Sigma unchanged, "
        "which means this file's equality cells cannot see a coverage error")
    assert abs(s_good - s_bad) > 0.0


def test_the_partition_costs_one_index_per_live_mode_not_one_mask_per_pane():
    """THE MEMORY ROW, as an inequality this repository can check.

    The panes partition the live set, so their index sets together hold
    one index per live mode however many panes there are; dense masks
    held one field per pane.  On the production deck that is 0.32 GB
    against 17.8 GB, and the ratio here must already be the same shape.
    """
    fld = _field()
    groups, _ = _plan(fld)
    idx_bytes = sum(int(np.asarray(g.idx_B).nbytes) for g in groups)
    mask_bytes = len(groups) * int(fld["live_mask"].size)   # bool = 1 byte
    live = int(fld["live_mask"].sum())
    assert idx_bytes <= live * 8, (idx_bytes, live)
    assert idx_bytes < mask_bytes
    # and the index really is the narrow one on any field this size
    assert all(np.asarray(g.idx_B).dtype == np.int32 for g in groups)


@pytest.mark.parametrize("n_flat,want", [
    (10, np.int32), ((1 << 31) - 1, np.int32), (1 << 31, np.int64),
    (81_432_576, np.int32),
])
def test_the_index_width_refuses_to_wrap(n_flat, want):
    """4 bytes per live mode on every field this code has met, 8 on one
    it has not -- and never a silent wrap, because a wrapped index is a
    pane holding the wrong modes."""
    assert SP.flat_index_dtype(n_flat) is want

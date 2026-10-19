"""Strict orbit-mode selection delivers the certified whole-orbit prefix (CPU, toy Gram).

The cr80 Gamma deck (25.5k-point pool, 2026-10-19) refused at N_c 16500 and
17000 with the same 4 dependent points: the 2nd and 3rd C3 images of two
orbits whose first image was taken just above the floor, mid-order.  Lowering
N to the reported rank cannot remove them; cutting back to the start of the
first orbit that holds a sub-floor pivot does.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from centroid.pivoted_cholesky import (
    certified_group_prefix,
    refuse_unless_select_certified,
    select_mode,
)
from common.pivoted_cholesky import group_block_pivoted_cholesky_select

jax.config.update("jax_enable_x64", True)


def _toy():
    """Nine 3-point orbits; orbit 5 repeats one row three times, mid-order."""
    rng = np.random.default_rng(7)
    q, _ = np.linalg.qr(rng.standard_normal((40, 40)))
    rows, d = [], 0
    for g in range(9):
        if g == 5:
            v = 3.0 * q[d]; d += 1
            rows += [v, v, v]                 # dependent siblings
        else:
            scale = 10.0 if g < 5 else 1.0
            for _ in range(3):
                rows.append(scale * q[d]); d += 1
    rows = np.asarray(rows)
    groups = np.repeat(np.arange(9), 3)
    return rows @ rows.T, groups


def test_strict_prefix_cuts_at_the_orbit_holding_the_first_dependent_pivot():
    G, groups = _toy()
    piv, _, rank, d_final, d_taken, _, _ = group_block_pivoted_cholesky_select(
        jnp.asarray(G), 27, jnp.asarray(groups), n_groups=9)
    piv = np.asarray(piv)
    d_taken = np.asarray(d_taken)
    floor = np.sqrt(np.finfo(np.float64).eps) * np.diag(G).max()
    assert int(rank) == 25                    # 27 delivered, 2 dependent
    assert piv.size == 27 and (piv >= 0).all()
    below = np.flatnonzero(d_taken[:27] <= floor)
    assert below.size == 2 and below[-1] < 26  # mid-order, independents follow
    cut = certified_group_prefix(piv, groups, d_taken, floor)
    assert cut == 15                          # orbits 0-4, before orbit 5
    assert (d_taken[:cut] > floor).all()
    kept = groups[piv[:cut]]
    assert all(np.count_nonzero(kept == g) == 3 for g in np.unique(kept))
    full = certified_group_prefix(piv[:15], groups, d_taken[:15], floor)
    assert full == 15                         # a clean prefix is returned whole


def test_select_mode_refuses_a_misspelled_mode(monkeypatch):
    monkeypatch.setenv("LORRAX_CENTROID_SELECT", "strickt")
    with pytest.raises(ValueError, match="not one of"):
        select_mode()
    monkeypatch.setenv("LORRAX_CENTROID_SELECT", "strict")
    assert select_mode() == "strict"


def test_point_mode_refusal_names_the_orbit_mode_prefix(monkeypatch):
    G, groups = _toy()
    piv, _, rank, d_final, d_taken, _, psd = group_block_pivoted_cholesky_select(
        jnp.asarray(G), 27, jnp.asarray(groups), n_groups=9)
    psd_host = (float(np.asarray(psd[0])), int(np.asarray(psd[1])),
                int(np.asarray(psd[2])))
    monkeypatch.setenv("LORRAX_CENTROID_SELECT", "strict")
    with pytest.raises(RuntimeError) as err:
        refuse_unless_select_certified(
            np.asarray(piv), int(rank), psd_host, n_keep=27, M=27, M_pad=27,
            orbit_id=None, d0max=float(np.diag(G).max()), tol_rel=None)
    msg = str(err.value)
    assert "longest whole-orbit prefix" in msg and "only in point mode" in msg
    assert "LOWER N" not in msg

"""kmeans names what its pivoted Cholesky measured (CPU, toy Grams).

The Fe 4³ CD deck (claim 3291): request 350 kept 312 points and the header
read "achieved numerical rank=312" while the pair set ranks above 1304.  The
point budget, not the pool, stopped that select.  The warning keys on the
largest residual left on an unpicked candidate: above the floor, the pool
still held directions; at or below it, the pool is spent.  Orbits of
dependent members make rank < written without a spent pool, which is why
``rank == written`` is not the test.
"""
import jax
import jax.numpy as jnp
import numpy as np

from centroid.pivoted_cholesky import _unpicked_over_floor
from centroid.production_output import (
    format_centroid_header,
    pool_rank_warning,
    rank_law_estimate,
)
from common.pivoted_cholesky import group_block_pivoted_cholesky_select

jax.config.update("jax_enable_x64", True)


def _select(rows, groups, budget):
    G = rows @ rows.T
    _, _, rank, d_final, *_ = group_block_pivoted_cholesky_select(
        jnp.asarray(G), budget, jnp.asarray(groups),
        n_groups=int(groups.max()) + 1)
    return int(rank), _unpicked_over_floor(d_final, np.diag(G).max(), None)


def _header(*, written, rank, unpicked, rank_law=None):
    return format_centroid_header(
        feature_fit="toy", source_wfn="WFN.h5", weight_label="toy",
        num_electrons=16.0, occupied_boundary=18, fft_grid=(25, 25, 25),
        kgrid=(4, 4, 4), shift=(0.0, 0.0, 0.0), seed=42, rho_power=1.0,
        requested=written, candidates=12, written=written,
        pruning="pivoted Cholesky", prune_rank=rank, prune_left=(0, 35),
        prune_right=(0, 35), prune_label="explicit feature pair",
        orbit_aware=True, n_sym=2, density_mode="scalar", pool=12,
        unpicked=unpicked, rank_law=rank_law)


def test_budget_stop_warns_and_spent_pool_does_not():
    rng = np.random.default_rng(3)
    orbits = np.repeat(np.arange(6), 2)      # six orbits of two equal points
    law = rank_law_estimate(left=(0, 35), right=(0, 35), nspinor=2,
                            kgrid=(4, 4, 4), ng=10**5, ecutwfc=40.0,
                            ecutrho=160.0)
    rich = np.repeat(rng.standard_normal((6, 9)), 2, axis=0)
    rank, unpicked = _select(rich, orbits, 6)
    assert rank == 3 and unpicked > 1.0      # rank < written, pool unspent
    warning = pool_rank_warning(pool=12, written=6, unpicked=unpicked,
                                rank_law=law)
    assert warning is not None and "rank not established" in warning
    assert "--oversample" in warning and "written/estimate = 0.01" in warning
    head = _header(written=6, rank=rank, unpicked=unpicked, rank_law=law)
    assert "pool rank=3 of 6 kept pivots on a 12-point pool" in head
    assert "stop: point budget" in head and "= 760 at L=17.5" in head
    assert "pair-density windows left=(0, 35), right=(0, 35)" in head

    poor = np.repeat(rng.standard_normal((6, 3)) @ rng.standard_normal(
        (3, 9)), 2, axis=0)                  # the pool spans 3 directions
    rank, unpicked = _select(poor, orbits, 10)
    assert rank == 3 and unpicked <= 1.0
    assert pool_rank_warning(pool=12, written=10, unpicked=unpicked) is None
    assert "stop: pool spent" in _header(written=10, rank=rank,
                                         unpicked=unpicked)


def test_rank_law_reproduces_the_documented_fe_point():
    # docs/theory/isdf-exchange-accuracy.md#rank-law: Fe 20³ spinor,
    # L = 14 and B = 60 Kramers pairs, law 2206; the sphere does not bind.
    n_mu, L, B, terms = rank_law_estimate(
        left=(0, 28), right=(0, 120), nspinor=2, kgrid=(20, 20, 20),
        ng=200_000, ecutwfc=80.0, ecutrho=320.0)
    assert (L, B) == (14.0, 60.0) and abs(terms[1] - 2206) < 1
    assert n_mu == 0.5 * terms[1]

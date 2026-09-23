"""BandLadder.log_moment: column-chunked (threaded) evaluation is bit-identical to one shot."""
import numpy as np

import gw.band_extrapolation as bx


def test_chunked_log_moment_equals_one_shot(monkeypatch):
    rng = np.random.default_rng(11)
    nk, n_dft = 3, 40
    enk_ry = np.sort(rng.uniform(-1.0, 4.0, (nk, n_dft)), axis=1)
    lad = bx.build_band_ladder(enk_ry=enk_ry, kweights=None, n_target=400, b0=0)
    beta = rng.uniform(0.2, 4.0, (7, 13))
    shells = ((5, 20), (20, 40), (30, lad.n_target))
    want = [lad.log_moment(lo, hi, beta) for lo, hi in shells]
    monkeypatch.setattr(bx, "_LOG_MOMENT_CHUNK_BYTES", 8 * 5)   # a few columns per chunk
    got = [lad.log_moment(lo, hi, beta) for lo, hi in shells]
    for g, w in zip(got, want):
        assert g.shape == beta.shape
        np.testing.assert_array_equal(g, w)


def test_row_major_moment_matches_the_column_formula_to_roundoff():
    rng = np.random.default_rng(12)
    enk_ry = np.sort(rng.uniform(-1.0, 4.0, (3, 40)), axis=1)
    lad = bx.build_band_ladder(enk_ry=enk_ry, kweights=None, n_target=400, b0=0)
    beta = rng.uniform(0.2, 4.0, 17)
    lo, hi = 30, lad.n_target
    # The pre-2026-09-23 column-major log-sum-exp, written out.
    x_d = ((lad.e_dft_ev[lo:lad.n_dft] - lad.e0_ev) / lad.estar_ev).reshape(-1)
    w_d = np.tile(np.log(lad.w_k), lad.n_dft - lo)
    x_w = (lad.e_weyl_ev[: hi - lad.n_dft] - lad.e0_ev) / lad.estar_ev
    lg = np.concatenate([-np.log(x_d)[:, None] * beta + w_d[:, None],
                         -np.log(x_w)[:, None] * beta])
    m = lg.max(axis=0)
    want = m + np.log(np.exp(lg - m).sum(axis=0))
    np.testing.assert_allclose(lad.log_moment(lo, hi, beta), want, rtol=1e-14, atol=0)

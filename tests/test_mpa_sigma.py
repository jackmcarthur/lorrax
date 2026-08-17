from types import SimpleNamespace

import numpy as np

from gw.mpa.sigma import _batch_rows, _branches


def _row(poles):
    return SimpleNamespace(
        pole_indices=np.asarray(poles, np.int32),
        bounds=np.zeros((len(poles), 6)),
        phase_real=np.zeros(len(poles), bool))


def test_batch_rows_relocalize_pole_selection_per_batch():
    shared = _row(range(8))
    low_only = _row((0, 1))
    np.testing.assert_array_equal(
        _batch_rows(shared, range(4, 8))[0], np.arange(4, dtype=np.int32))
    assert _batch_rows(low_only, range(4, 8)) is None


def test_small_gap_branching_follows_occupation_not_energy_sign():
    wfns = SimpleNamespace(
        enk=np.asarray([[-0.1, -0.01, 0.02, 0.3]]),
        occ=np.asarray([[1.0, 0.0, 1.0, 0.0]]),
        # ``sigma_sum`` and DELIBERATELY NOT ``full``.  Since the chi/Sigma
        # split those are different windows -- ``full`` is the LOADED extent
        # max(chi, sigma), ``sigma_sum`` is the band sum these branches run
        # over -- and the causal branching is a statement about the SIGMA
        # sum.  Omitting ``full`` makes this cell fail loudly if the
        # production code ever reaches back for the larger consumer's count.
        slices=SimpleNamespace(sigma_sum=slice(None)))
    branches = _branches(wfns, np.asarray([-0.2, 0.4]), 0.0)
    pos_cond = next(b for b in branches
                    if b.space == "cond" and not b.neg_omega_half)
    neg_val = next(b for b in branches
                   if b.space == "val" and b.neg_omega_half)
    assert pos_cond.base_mask_A.tolist() == [[False, True, False, True]]
    assert neg_val.base_mask_A.tolist() == [[True, False, True, False]]
    assert float(pos_cond.E_A[0, 1]) < 0.0
    assert float(neg_val.E_A[0, 2]) < 0.0

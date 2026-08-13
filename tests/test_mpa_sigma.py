from types import SimpleNamespace

import numpy as np

from gw.mpa.sigma import _batch_rows, _branches, execution_census


def _row(poles, n_tau):
    return SimpleNamespace(
        pole_indices=np.asarray(poles, np.int32),
        bounds=np.zeros((len(poles), 6)),
        phase_real=np.zeros(len(poles), bool),
        window=SimpleNamespace(n_tau=n_tau))


def test_memory_batches_change_dispatches_not_pole_selection():
    shared = _row(range(8), 11)
    low_only = _row((0, 1), 7)
    assert execution_census((shared, low_only), 8, 4) == {
        "n_sweeps": 3, "n_tau": 29}
    np.testing.assert_array_equal(
        _batch_rows(shared, range(4, 8))[0], np.arange(4, dtype=np.int32))
    assert _batch_rows(low_only, range(4, 8)) is None


def test_small_gap_branching_follows_occupation_not_energy_sign():
    wfns = SimpleNamespace(
        enk=np.asarray([[-0.1, -0.01, 0.02, 0.3]]),
        occ=np.asarray([[1.0, 0.0, 1.0, 0.0]]),
        slices=SimpleNamespace(full=slice(None)))
    branches = _branches(wfns, np.asarray([-0.2, 0.4]), 0.0)
    pos_cond = next(b for b in branches
                    if b.space == "cond" and not b.neg_omega_half)
    neg_val = next(b for b in branches
                   if b.space == "val" and b.neg_omega_half)
    assert pos_cond.base_mask_A.tolist() == [[False, True, False, True]]
    assert neg_val.base_mask_A.tolist() == [[True, False, True, False]]
    assert float(pos_cond.E_A[0, 1]) < 0.0
    assert float(neg_val.E_A[0, 2]) < 0.0

from types import SimpleNamespace

import numpy as np

from gw.mpa.sigma import _batch_rows, execution_census


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

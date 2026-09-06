"""Falsify the mini's completion and value gates without starting GPU work."""
from types import SimpleNamespace

import numpy as np
import pytest

from tests.mini.checks import check_results, require_p4


def _rows():
    sigma = np.zeros((27, 7))
    sigma[:, 0] = np.repeat(np.arange(9), 3)
    sigma[:, 1] = np.tile(np.arange(3), 9)
    sigma[:, 2:5] = [-2.0, 0.5, -1.5]
    return np.linspace(-3, 3, 27), sigma


def test_one_process_with_four_devices_cannot_satisfy_p4():
    valid = dict(process_count=4, n_devices=4, n_local_devices=1, mesh_shape=(2, 2))
    require_p4(SimpleNamespace(**valid))
    for changes in (dict(process_count=1, n_local_devices=4),
                    dict(n_local_devices=4), dict(mesh_shape=(1, 4))):
        with pytest.raises(AssertionError):
            require_p4(SimpleNamespace(**(valid | changes)))


@pytest.mark.parametrize("fault", ["nan", "missing", "duplicate", "sum", "drift"])
def test_invalid_results_cannot_satisfy_the_same_gate_used_by_drivers(fault):
    eqp, sigma = _rows()
    reference = eqp.copy(), sigma.copy()
    check_results(eqp, sigma, reference=reference)
    if fault == "nan":
        eqp[-1] = np.nan
    elif fault == "missing":
        sigma = sigma[:-1]
    elif fault == "duplicate":
        sigma[-1, :2] = sigma[0, :2]
    elif fault == "sum":
        sigma[-1, 4] += 0.01
    else:
        eqp[-1] += 0.001
    with pytest.raises(AssertionError):
        check_results(eqp, sigma, reference=reference)

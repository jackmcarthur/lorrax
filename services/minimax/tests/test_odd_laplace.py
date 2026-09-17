"""The shared odd resolvent channel meets its kernel contract."""
import numpy as np
import pytest

import minimax


def test_odd_laplace_preserves_even_nodes_and_resolves_independent_grid():
    even_times = np.geomspace(.02, 5., 8)
    times, weights, added, sampled = minimax.augment_odd_laplace(
        even_times, 1., 20., 1., tolerance=1e-4,
        max_extra=4, grid_size=256, candidates=20)
    assert added >= 1
    np.testing.assert_array_equal(times[:even_times.size], even_times)
    x = np.geomspace(1., 20., 10001)
    error = np.max(np.abs(np.exp(-x[:, None] * times) @ weights
                          - 1 / (x*x + 1)))
    assert error < 1e-4
    assert sampled < 1e-4


def test_odd_laplace_refuses_an_unmet_gate():
    with pytest.raises(RuntimeError, match="GATE odd_kernel_representation"):
        minimax.augment_odd_laplace([1.], 1., 20., 1.,
                                   tolerance=1e-12, max_extra=0,
                                   grid_size=32, candidates=4)

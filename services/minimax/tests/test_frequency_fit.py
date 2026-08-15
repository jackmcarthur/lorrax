import numpy as np

from minimax import fit_damped_reciprocal


def test_damped_reciprocal_fit_shares_one_rule_across_widths():
    cells = np.asarray([
        [0.5, 4.0, 0.2, 0.5],
        [0.5, 4.0, 1.0, 2.0],
    ])
    fit = fit_damped_reciprocal(
        cells, target_error=3.0e-3, max_rank=48,
        training_points=7, validation_points=18, contour_count=3)
    assert fit.nodes.size <= 48
    assert fit.sampled_max_error <= 3.0e-3
    assert np.isfinite(fit.amplification)
    assert np.all(np.imag(fit.nodes) >= 0.0)

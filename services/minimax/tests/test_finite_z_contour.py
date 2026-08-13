import numpy as np

from minimax import build_finite_z_contour


def test_one_sided_union_meets_sampled_kernel_target():
    interval = np.asarray((0.625, 4.0))
    z = np.asarray((1.0 + 1.0j, 3.0 + 1.0j, 0.0 + 10.0j))
    candidate = build_finite_z_contour(
        interval, z, 1.0e-5, design_points=257,
        validation_points=2049, angle_step=0.05)

    assert candidate.resonant.frequency_sign == 1
    assert candidate.antiresonant.frequency_sign == -1
    assert np.all((candidate.resonant.weights / 1j).real > 0.0)
    assert np.all(np.real(
        candidate.antiresonant.weights / candidate.antiresonant.contour)
        > 0.0)
    assert candidate.executed_nodes < 100
    assert candidate.heldout_combined_scaled_error <= 1.0e-5

import numpy as np

from minimax.reciprocal_fit import ReciprocalMeasureProblem
from minimax.windowed_fit import fit_phase_bounded_candidates


def test_phase_bounded_crossing_fit_uses_nonnegative_coefficients():
    problem = ReciprocalMeasureProblem(
        frequencies=np.linspace(-1.0, 1.0, 17),
        internal_sums=np.array([-0.7 - 0.5j, 0.0 - 0.4j, 0.8 - 0.6j]),
        cell_masses=np.array([1.0, 2.0, 1.0]),
    )
    candidates = np.linspace(0.02, 30.0, 120)
    fit = fit_phase_bounded_candidates(
        problem, candidates, target_error=3.0e-2, max_rank=96)

    assert fit.target_met
    assert fit.sampled_relative_residual <= 3.0e-2
    assert np.all(fit.coefficients >= 0.0)
    assert np.max(np.abs(fit.weights.real)) <= 1.0e-14
    assert np.all(fit.weights.imag <= 0.0)
    assert fit.amplification_p99 < 10.0


def test_miss_returns_actual_best_rule_and_metrics():
    problem = ReciprocalMeasureProblem(
        frequencies=np.linspace(-1.0, 1.0, 9),
        internal_sums=np.array([-0.5 - 0.2j, 0.5 - 0.2j]),
        cell_masses=np.ones(2),
    )
    fit = fit_phase_bounded_candidates(
        problem, np.linspace(0.1, 2.0, 8),
        target_error=1.0e-8, max_rank=4, lawson_iterations=2)

    assert not fit.target_met
    assert np.isfinite(fit.sampled_relative_residual)
    assert np.isfinite(fit.delivered_error)
    assert np.isfinite(fit.amplification_p99)
    assert fit.node_count <= 4

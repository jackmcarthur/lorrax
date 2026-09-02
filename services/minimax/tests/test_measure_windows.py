"""The tail-refined lattice is the service's one measure compressor."""
import numpy as np
import pytest

from minimax.measure_windows import tail_refined_lattice_measure


def test_tail_refined_lattice_conserves_mass_and_refines():
    rng = np.random.default_rng(11)
    support = rng.normal(2.0, 1.5, 400) - 1.0j * rng.gamma(2.0, 0.1, 400)
    masses = rng.random(400) * np.exp(-rng.uniform(0.0, 8.0, 400))

    cells, cell_mass, refined, refined_mass = tail_refined_lattice_measure(
        support, masses, bins_per_axis=25)

    assert cell_mass.sum() == pytest.approx(masses.sum(), rel=1e-12)
    assert refined_mass.sum() == pytest.approx(masses.sum(), rel=1e-12)
    assert refined.size > cells.size
    assert np.all(cell_mass > 0.0) and np.all(refined_mass > 0.0)
    # the compressed support stays inside the raw bounding box
    for lattice in (cells, refined):
        assert lattice.real.min() >= support.real.min() - 1e-12
        assert lattice.real.max() <= support.real.max() + 1e-12


def test_degenerate_one_value_axis_conserves_mass_exactly():
    support = np.full(7, 3.0) - 0.25j  # both axes degenerate
    masses = np.arange(1.0, 8.0)
    cells, cell_mass, refined, refined_mass = tail_refined_lattice_measure(
        support, masses, bins_per_axis=25)
    assert cell_mass.sum() == masses.sum()
    assert refined_mass.sum() == masses.sum()
    np.testing.assert_allclose(cells, support[0])


def test_rejects_empty_or_nonfinite_measures():
    with pytest.raises(ValueError):
        tail_refined_lattice_measure(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        tail_refined_lattice_measure(
            np.array([1.0 + 0.0j, np.nan + 0.0j]), np.ones(2))

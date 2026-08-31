"""Focused contracts for unit-weight centroid fitting and spatial closure."""

from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_nonsymmorphic_grid_average_uses_the_seitz_translation():
    from centroid.charge_density import symmetrize_on_grid

    field = np.arange(8.0).reshape(4, 2, 1)
    identity = np.eye(3, dtype=np.int32)
    operations = np.stack((identity, identity))
    translations = np.asarray(((0.0, 0.0, 0.0), (0.5, 0.0, 0.0)))

    got = symmetrize_on_grid(field, operations, translations)
    expected = 0.5 * (field + np.roll(field, shift=2, axis=0))
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(
        np.roll(got, shift=2, axis=0), got,
        err_msg="the projected weight is not invariant under the translation")


def test_off_grid_seitz_translation_refuses_instead_of_rounding():
    from centroid.charge_density import symmetrize_on_grid

    with pytest.raises(RuntimeError, match="not commensurate"):
        symmetrize_on_grid(
            np.arange(8.0).reshape(2, 2, 2),
            np.eye(3, dtype=np.int32)[None, ...],
            np.asarray(((0.25, 0.0, 0.0),)),
        )


def test_kmeans_cli_has_no_occupation_weight_fit_option():
    source = (ROOT / "src/centroid/kmeans_cli.py").read_text()
    assert 'choices=("charge_density", "band_range")' not in source
    assert "n_occ =" not in source

    # Red twin: both retired policy tokens are detected.
    old = 'choices=("charge_density", "band_range")\nn_occ = int(wfn.nelec)'
    assert 'choices=("charge_density", "band_range")' in old
    assert "n_occ =" in old

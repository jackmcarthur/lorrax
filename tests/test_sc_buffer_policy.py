"""Named-window versus execution-buffer separation in QSGW."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from gw.band_partition import BandPartition
from gw.sc_iteration import (
    _apply_sc_buffer_partition, _carry_sc_buffer_diagonal, _sc_buffer_mask)
from gw.wavefunction_bundle import BandSlices


def _inputs(mode="diagonal"):
    return SimpleNamespace(
        config=SimpleNamespace(
            nval=4, ncond=4,
            sc=SimpleNamespace(buffer_nbands=4, buffer_mode=mode)),
        meta=SimpleNamespace(nelec=26),
        band_slices=BandSlices.from_band_edges(0, 18, 26, 34, 80),
    )


def test_symmetric_buffer_surrounds_named_window_and_is_diagonal_only():
    inputs = _inputs()
    buffer = _sc_buffer_mask(inputs)
    expected = np.zeros(34, dtype=bool)
    expected[18:22] = True
    expected[30:34] = True
    np.testing.assert_array_equal(buffer, expected)

    part = _apply_sc_buffer_partition(BandPartition.all_protected(34), inputs)
    np.testing.assert_array_equal(np.asarray(part.in_range_mask), True)
    np.testing.assert_array_equal(
        np.asarray(part.protected_mask), ~expected)


def test_one_sided_buffer_keeps_cross_edge_couplings():
    inputs = _inputs("one_sided")
    original = BandPartition.all_protected(34)
    assert _apply_sc_buffer_partition(original, inputs) is original


def test_carried_buffer_replaces_only_selected_diagonals():
    H_new = jnp.diag(jnp.asarray([1.0, 2.0, 3.0]))[None]
    H_input = jnp.diag(jnp.asarray([10.0, 20.0, 30.0]))[None]
    out = np.asarray(_carry_sc_buffer_diagonal(
        H_new, H_input, jnp.asarray([False, True, False])))
    np.testing.assert_allclose(np.diagonal(out, axis1=-2, axis2=-1),
                               [[1.0, 20.0, 3.0]])

"""A QP block may cross uncorrected DFT guards without losing f(H) support."""
import numpy as np
import jax.numpy as jnp

from bandstructure.fh_interp import _f_params_from_energies, f_transform_eigs


def test_partial_qp_bank_keeps_returned_states_visible():
    # First two rows are returned; row 2 is a corrected guard, row 3 an
    # uncorrected DFT guard. Their row order is deliberately not energy order.
    energies = jnp.asarray([[-3., -2.], [4., 5.], [8., 9.], [2., 3.]])
    f_eps, _, _, shift = f_transform_eigs(energies, a_band_index=0)
    assert shift == 9.
    assert np.all(np.asarray(f_eps)[:2] < 0.)
    # The highest guard still supplies the null endpoint, not an invented
    # offset above the whole spectrum that would evade the shoulder check.
    assert float(f_eps[2, 1]) == 0.


def test_sorted_bank_retains_original_endpoint_and_width():
    energies = jnp.asarray([[-3., -2.], [2., 3.], [4., 5.], [8., 9.]])
    a, n, shift = _f_params_from_energies(energies, 3, a_band_index=0)
    assert shift == float(np.max(np.asarray(energies)[-1]))
    assert a == 4. * (1. + 1e-14)
    assert n == 3.


def test_endpoint_excludes_rows_outside_requested_window():
    energies = jnp.asarray([[-3., -2.], [8., 9.], [2., 3.], [100., 101.]])
    _, _, shift = _f_params_from_energies(energies, 2, a_band_index=0)
    assert shift == 9.

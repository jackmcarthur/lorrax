"""The SC partition is fixed at map 0; a frozen state leaving the grid refuses."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from gw import sc_iteration


def _partition(nb, lo, hi):
    m = np.zeros(nb, dtype=bool); m[lo:hi] = True
    return SimpleNamespace(protected_mask=m, in_range_mask=m)


def test_inside_states_return_the_smallest_margin(capsys):
    e = np.array([[-3.0, 1.0, 12.0], [-2.0, 2.0, 15.9]])       # (nk=2, nb=3)
    margin = sc_iteration._refuse_frozen_partition_escape(
        _partition(3, 0, 3), e, band_offset=4, omega_min_abs_ev=-15.0,
        omega_max_abs_ev=18.0, tolerance_ev=0.136, omega_step_ev=0.25,
        iteration=3, print_fn=print)
    assert margin == pytest.approx(18.0 - 15.9)
    out = capsys.readouterr().out
    assert "frozen from map 0 (5-7, 3 of 3)" in out


def test_escaping_state_refuses_with_band_k_and_window_value():
    e = np.array([[-3.0, 1.0, 12.0], [-2.0, 2.0, 18.9]])
    with pytest.raises(ValueError) as err:
        sc_iteration._refuse_frozen_partition_escape(
            _partition(3, 0, 3), e, band_offset=4, omega_min_abs_ev=-15.0,
            omega_max_abs_ev=18.0, tolerance_ev=0.136, omega_step_ev=0.25,
            iteration=3, print_fn=lambda _s: None)
    text = str(err.value)
    assert "band 7 left the Sigma grid at the top edge" in text
    assert "E(k=1, band 7) = 18.900 eV" in text
    assert "widen the grid by 3.00 eV (sigma_omega_max_ev moved up" in text


def test_scissored_bands_do_not_count():
    e = np.array([[-3.0, 1.0, 40.0]])
    margin = sc_iteration._refuse_frozen_partition_escape(
        _partition(3, 0, 2), e, band_offset=0, omega_min_abs_ev=-15.0,
        omega_max_abs_ev=18.0, tolerance_ev=0.136, omega_step_ev=0.25,
        iteration=1, print_fn=lambda _s: None)
    assert margin == pytest.approx(12.0)

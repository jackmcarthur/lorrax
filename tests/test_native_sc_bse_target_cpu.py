"""Cheap target-report tests; these do not certify the live BSE callback."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('native_bse_probe',
    Path(__file__).parent/'tdagw/cri3_full_microscopic/native_bse_probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_target_requires_brightness_subgap_and_resolved_residual():
    result = probe.bright_subgap_summary([1., 1.2, 1.4, 2.1, 1.5],
        [0., 1.e-5, 1.e-5, 1.e-5, .01],
        [[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [5., 0., 0.], [9., 0., 0.]],
        gap_ev=2., nk=64)
    assert result['bright_subgap_root_indices'] == [1, 2]
    assert result['target_root_index'] == 2
    assert result['target_energy_ev'] == 1.4
    assert result['strength_xyz_bohr2_cell'][2][0] == 4./64


def test_no_bright_bound_state_is_not_a_pulse_target():
    result = probe.bright_subgap_summary([1., 2.], [0., 0.],
        [[0., 0., 1.], [1., 0., 0.]], gap_ev=2., nk=4)
    assert result['target_root_index'] is None
    assert result['target_energy_ev'] is None


def test_negative_excitation_is_not_admitted():
    result = probe.bright_subgap_summary([-.1], [0.], [[1., 0., 0.]], gap_ev=2., nk=4)
    assert not result['bright_subgap_root_indices']


@pytest.mark.parametrize('residual,dipole', [([float('nan')], [[1., 0., 0.]]),
    ([0.], [[1., 0.]]), ([], [[1., 0., 0.]])])
def test_malformed_root_tables_refused(residual, dipole):
    with pytest.raises(ValueError):
        probe.bright_subgap_summary([1.], residual, dipole, gap_ev=2., nk=4)

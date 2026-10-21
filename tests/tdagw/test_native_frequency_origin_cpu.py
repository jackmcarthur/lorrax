"""Numerical frequency-origin covariance, not a physical chemical-potential change."""
import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from common.units import RYD_TO_EV
from gw.head_correction import HeadSigmaDiag
from gw.ppm_sigma import ordered_window_request
from tdagw import snapshot_runtime as owner


@pytest.fixture(autouse=True)
def fp64_metadata():
    import jax
    with jax.enable_x64(True):
        yield


def case():
    energy = np.array([[-1.1, -.7, -.2, .4, 1.2], [-1., -.6, -.1, .5, 1.3]])
    f = np.broadcast_to([1., 1., 1., 0., 0.], energy.shape).copy()
    return energy, f


def test_accepted_logical_midgap_does_not_mutate_source_tables_or_mu():
    energy, f = case()
    old_e, old_f = energy.copy(), f.copy()
    origin, receipt = owner._native_frequency_origin(energy, f, -.9)
    assert origin == pytest.approx(.15)
    assert receipt['source_reference_ry'] == -.9
    assert receipt['numerical_origin_ry'] == origin
    assert receipt['policy'] == 'accepted_binary_midgap'
    assert not receipt['physical_census_mu_changed']
    np.testing.assert_array_equal(energy, old_e)
    np.testing.assert_array_equal(f, old_f)


@pytest.mark.parametrize('mode', ['fractional', 'all_filled', 'inverted'])
def test_nonbinary_or_nongapped_reference_retains_explicit_original(mode):
    energy, f = case()
    if mode == 'fractional':
        f[0, 1] = .8
    elif mode == 'all_filled':
        f[:] = 1.
    else:
        energy[:, 3] = -3.
    origin, receipt = owner._native_frequency_origin(energy, f, -.9)
    assert origin == -.9
    assert receipt['policy'] == 'explicit_source_reference_fallback'


def test_common_origin_preserves_denominators_absolute_support_and_full_band_energies():
    energy, f = case()
    old = -.9
    new, _ = owner._native_frequency_origin(energy, f, old)
    active = energy[:, 1:4]
    poles = np.array([.7-.03j, -.8-.05j])
    old_delta = (active-old)[:, :, None, None]-(energy-old)[:, None, :, None]-poles
    new_delta = (active-new)[:, :, None, None]-(energy-new)[:, None, :, None]-poles
    np.testing.assert_allclose(new_delta, old_delta, rtol=2e-15, atol=2e-15)
    old_request, _ = ordered_window_request(active, old)
    new_request, _ = ordered_window_request(active, new)
    np.testing.assert_allclose(old_request.tables_ry+old, new_request.tables_ry+new,
                               rtol=2e-15, atol=2e-15)
    plan = dict(omega_range_ev=[-.25, .25], out_of_grid='cover')
    old_reads = owner._native_window_support_reads(active, old, plan)
    new_reads = owner._native_window_support_reads(active, new, plan)
    # Round trips through relative eV may introduce harmless last-bit duplicates.
    for reads in (old_reads, new_reads):
        assert np.max(np.min(abs(reads[:, None]-active.ravel()[None]*RYD_TO_EV),axis=1)) < 2e-14


def test_exact_head_value_and_derivative_covariance_for_row_and_column_shells():
    energy, f = case()
    old = -.9
    new, _ = owner._native_frequency_origin(energy, f, old)
    head = HeadSigmaDiag.build(omega_grid_ry=np.array([-2., 2.]), enk_ry=energy,
        occupations=f, efermi_ry=old, poles_ry=np.array([.7-.02j, 1.1-.04j]),
        residues_ry=np.array([.012, .018]), cell_volume=8., nk_tot=2)
    shifted = replace(head, efermi_ry=new)
    for i in range(energy.shape[1]):
        for absolute in (energy, np.repeat(energy[:, i:i+1], energy.shape[1], axis=1)):
            value, derivative = head.on_shell(absolute-old)
            actual, actual_derivative = shifted.on_shell(absolute-new)
            np.testing.assert_allclose(actual, value, rtol=3e-14, atol=3e-15)
            np.testing.assert_allclose(actual_derivative, derivative, rtol=3e-14, atol=3e-15)
    np.testing.assert_array_equal(shifted.enk_ry, energy)
    np.testing.assert_array_equal(shifted.poles_ry, head.poles_ry)


def test_native_factory_binds_one_origin_to_supports_and_model_handle():
    source = Path(owner.__file__).read_text()
    tree = ast.parse(source)
    factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name == 'make_native_snapshot_evaluator')
    evaluator = next(n for n in factory.body if isinstance(n, ast.FunctionDef) and n.name == 'evaluate')
    calls = [n for n in ast.walk(evaluator) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    support = next(n for n in calls if n.func.id == '_native_window_support_reads')
    handle = next(n for n in calls if n.func.id == 'shared_pole_window_source')
    assert isinstance(support.args[1], ast.Name) and support.args[1].id == 'frequency_origin'
    reference = next(k.value for k in handle.keywords if k.arg == 'efermi_ry')
    assert isinstance(reference, ast.Name) and reference.id == 'frequency_origin'
    assert '_native_frequency_origin' not in [n.func.id for n in calls]

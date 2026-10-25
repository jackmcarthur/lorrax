"""Compact Σ state support: literal selectors/fold and forbidden all-window masks."""
from functools import partial
from types import SimpleNamespace

import runtime
from runtime.source_closure import ensure_source_closure
ensure_source_closure()
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update('jax_enable_x64', True)


def _mesh(side):
    if jax.device_count() < side * side:
        pytest.skip('requires nine virtual CPU devices')
    return Mesh(np.array(jax.devices()[:side*side]).reshape(side, side), ('x', 'y'))


def _plan():
    from gw.ppm_windows import _SigmaBranch
    from gw.mpa.sigma_windows import summarize_tangential_poles
    from gw.sigma_box_plan import plan_sigma_windows
    # Exact ties at the ordinary state edge .55 and its adjacent float;
    # a signed MP weight, a dead state, and nonfinite energy sentinels.
    energy = np.tile([0., .15, .55, np.nextafter(.55, np.inf), 1.,
                      .2, np.nan, np.inf, -np.inf], (6, 1))
    base = np.tile([1, 1, 1, 1, 1, 0, 1, 1, 1], (6, 1)).astype(bool)
    weight = np.tile([.1, -.2, .7, 1., 2., 3., .3, .4, .5], (6, 1))
    omega = np.array([.05, .15, .3, .4])
    branch = _SigmaBranch('positive cond', energy, base, 'cond', False,
                          omega, np.arange(4), weight)
    summary = summarize_tangential_poles(np.array([[.1, .55, .56, 2.]])**2,
        np.array([4]), [branch], regularization_width_ry=.1, edge_factor=1.5)
    rows, geometry = plan_sigma_windows(summary, [branch], omega, .1,
        eps=1e-3, scope=None, print_fn=lambda *_: None,
        omega_group=np.array([-1, 0, 1, 0]), omega_eta_ry=np.full(4, .1),
        group_fixed=(True, True))
    return rows, geometry, energy, base, weight, omega


def test_parent_selection_keeps_host_payload_off_devices(monkeypatch):
    from gw import centroid_k_unfold as owner
    irr = np.array([2, 0, 1, 2, 1, 0], np.int32)
    values = np.arange(6*9).reshape(6, 9)
    expected = values[[5, 4, 3, 0]]  # last child per raw parent; unused parent →row0
    owner._PARENT_SOURCE.clear()
    with monkeypatch.context() as patch:
        patch.setattr(owner.jnp, 'asarray', lambda *_a, **_kw:
            (_ for _ in ()).throw(AssertionError('host parent selection placed a payload')))
        got = owner.parent_rows(irr, 4, values)
    assert isinstance(got, np.ndarray)
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(owner.parent_rows(irr, 4, jnp.asarray(values)), expected)
    np.testing.assert_array_equal(owner.parent_rows(irr, 4, values.T, axis=1), expected.T)
    with pytest.raises(ValueError, match='full-k axis'):
        owner.parent_rows(irr, 4, values[:5])


@pytest.mark.parametrize('side', [2, 3])
def test_compact_planner_selectors_and_window_fold_match_literal(side):
    from gw.centroid_k_unfold import parent_rows
    from gw.greens_function_kernel import _weighted_tau_phases
    from gw.mpa.sigma import _sigma_window_table, _window_parent_inputs, _window_selector
    from gw.ppm_accumulators import DeviceOmegaAccumulator
    from common.collectives import device_put_process_local
    mesh = _mesh(side);rep = NamedSharding(mesh, P())
    rows, geometry, energy, base, weight, omega = _plan()
    assert len(rows) > 3  # coarse support really split; no inferred omega interval
    assert len({id(row.window.mask_A) for row in rows}) == 1
    assert any(np.any(np.diff(row.omega_idx) > 1) for row in rows)
    bounds = {(row.window.E_min, row.window.E_max) for row in rows}
    assert (-np.inf, .55) in bounds and (.55, np.inf) in bounds
    assert geometry['n_windows'] == len(rows)
    plan = SimpleNamespace(parent_rows=partial(parent_rows,
        np.array([2, 0, 1, 2, 1, 0], np.int32), 4))
    source_rows = [5, 4, 3, 0]
    rng = np.random.default_rng(651)
    psi = rng.normal(size=(4, 9, 6)) + 1j*rng.normal(size=(4, 9, 6))
    psi_d = device_put_process_local(psi, rep)
    acc = DeviceOmegaAccumulator.grid(device_put_process_local(omega, rep),
        shape=(4, 4, 6, 6), sharding=NamedSharding(mesh, P(None, None, 'x', 'y')),
        omega_axis=0)
    table = _sigma_window_table(acc, SimpleNamespace(window_intervals=None), rows,
                                max(len(row.window.nodes.t) for row in rows))
    assert 'x_mask' not in table.fields and table.fields['x_state_bounds'].shape == (len(rows), 2)
    counter = acc.window_counter();expected = np.zeros((4, 4, 6, 6), complex)

    @jax.jit
    def tau(e, selector, psi, ref, t, _active):
        phases = _weighted_tau_phases(e, 1j*t, band_weight=selector, e_ref=ref)
        return jnp.einsum('knm,kn,knl->kml', psi, phases, psi.conj())

    for row in rows:
        t, coeff, n_active, operands, counter = acc.window_pick(table, counter)
        e, support, w = _window_parent_inputs(row, plan, rep)
        selector = _window_selector(True, True)(support, e, w, operands['state_bounds'])
        literal = np.zeros_like(weight)
        for k in range(6):
            for n in range(9):
                if (base[k, n] and np.isfinite(energy[k, n])
                        and np.isfinite(weight[k, n]) and abs(weight[k, n]) > 0
                        and row.window.E_min < energy[k, n] <= row.window.E_max):
                    literal[k, n] = weight[k, n]
        np.testing.assert_array_equal(np.asarray(selector), literal[source_rows])
        acc.run_window(tau, (e, selector, psi_d, operands['e_ref_a']),
            t, coeff, n_active, operands['active'])
        win = row.window
        for ti, alpha in zip(win.nodes.t, win.nodes.alpha):
            sigma = np.zeros((4, 6, 6), complex)
            for k in range(4):
                for n in range(9):
                    selected = literal[source_rows[k], n]
                    if selected:
                        phase = selected*np.exp(-1j*(energy[source_rows[k], n]-win.E_ref_A)*ti)
                        sigma[k] += phase*np.outer(psi[k, n], psi[k, n].conj())
            for index, value in zip(row.omega_idx, row.omega_abs):
                expected[index] += (win.prefactor*alpha
                    *np.exp(-1j*(win.E_ref_A+win.E_ref_B-win.omega_sign*value)*ti))*sigma
    np.testing.assert_allclose(np.asarray(acc.finalize()), expected, rtol=2e-12, atol=2e-12)
    explicit = np.array([[True, False, True]])
    # An existing explicit one-window selector is not reinterpreted by a
    # synthetic interval, even at nonfinite sentinel energies.
    np.testing.assert_array_equal(_window_selector(False, False)(
        explicit, np.array([[-np.inf, 0., np.nan]]), None, np.array([-np.inf, np.inf])), explicit)


def test_window_table_never_reads_full_k_summed_band_masks(monkeypatch):
    from gw.mpa.sigma import _sigma_window_table
    from gw.ppm_accumulators import DeviceOmegaAccumulator
    from common.collectives import device_put_process_local
    mesh = _mesh(2);rep = NamedSharding(mesh, P())
    # Shape-only 20^3/20k support: a one-byte broadcast view, never allocated.
    support = np.broadcast_to(np.array(True), (8000, 20000))
    row = SimpleNamespace(window=SimpleNamespace(mask_A=support, E_min=-np.inf,
        E_max=np.inf, E_ref_A=0., E_ref_B=0., omega_sign=1, prefactor=1.,
        nodes=SimpleNamespace(t=np.array([.2+.1j]), alpha=np.array([.5+.2j]))),
        pole_indices=np.array([0]), bounds=np.zeros((1, 6)), omega_idx=np.array([0, 2]),
        omega_abs=np.array([.1, .3]))
    acc = DeviceOmegaAccumulator.grid(device_put_process_local(np.array([.1, .2, .3]), rep),
        shape=(3, 1, 2, 2), sharding=NamedSharding(mesh, P(None, None, 'x', 'y')),
        omega_axis=0)
    original = np.asarray
    def guarded(value, *args, **kwargs):
        if value is support:
            raise AssertionError('all-window table inspected a full-k summed-band mask')
        return original(value, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(np, 'asarray', guarded)
        table = _sigma_window_table(acc, SimpleNamespace(window_intervals=None), [row]*14, 1)
    assert table.nbytes < 4096
    assert all(value.size < 200 for value in table.fields.values())

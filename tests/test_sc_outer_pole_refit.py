import numpy as np
import pytest
from types import SimpleNamespace

from gw.gw_config import SCConfig
from gw.head_correction import HeadSample
from gw.minimax_screening import refit_gn_ppm_residues_at_fixed_poles
from gw.mpa.pade_fit import (
    refit_mpa_residues_at_fixed_poles,
    synthesize_w_samples,
)
from gw.ppm_pipeline import _fit_head_correction
from gw.ppm_sigma import fit_ppm


def test_mpa_fixed_pole_refit_recovers_changed_residues():
    omega = np.array([0.9 - 0.03j, 2.1 - 0.08j], dtype=np.complex128)
    residues = np.array([0.22 + 0.04j, -0.13 + 0.02j])
    z = np.array(
        [0.15 + 0.7j, 0.4 + 1.1j, 1.3 + 2.0j, 2.4 + 3.2j],
        dtype=np.complex128,
    )
    samples = synthesize_w_samples(omega, residues, z)

    fitted, diagnostics = refit_mpa_residues_at_fixed_poles(
        samples, z, omega, np.ones(2, dtype=bool))

    np.testing.assert_allclose(fitted, residues, rtol=2e-12, atol=2e-12)
    assert float(diagnostics["rel_rms_residual"]) < 2e-12


def test_mpa_fixed_pole_refit_reports_inadequate_positions():
    omega = np.array([0.9 - 0.03j, 2.1 - 0.08j], dtype=np.complex128)
    moved = np.array([1.15 - 0.03j, 2.45 - 0.08j], dtype=np.complex128)
    residues = np.array([0.22 + 0.04j, -0.13 + 0.02j])
    z = np.array(
        [0.15 + 0.7j, 0.4 + 1.1j, 1.3 + 2.0j, 2.4 + 3.2j],
        dtype=np.complex128,
    )
    samples = synthesize_w_samples(moved, residues, z)

    _, diagnostics = refit_mpa_residues_at_fixed_poles(
        samples, z, omega, np.ones(2, dtype=bool))

    assert float(diagnostics["rel_rms_residual"]) > 1e-4


def test_gn_fixed_frequency_refits_strength_and_checks_probe():
    omega = np.array([[[0.8, 1.2], [1.2, 1.7]]], dtype=np.float64)
    wc0 = np.array(
        [[[-0.5, -0.2], [-0.2, -0.1]]], dtype=np.complex128)
    z = 2.0j
    residue = -0.5 * wc0 * omega
    probe = 2.0 * omega * residue / (z * z - omega * omega)

    fitted, odd, residual = refit_gn_ppm_residues_at_fixed_poles(
        wc0, probe, omega, z)

    assert odd is None
    np.testing.assert_allclose(fitted, residue, rtol=2e-14, atol=2e-14)
    assert residual < 2e-14

    _, _, inadequate = refit_gn_ppm_residues_at_fixed_poles(
        wc0, 1.05 * probe, omega, z)
    assert inadequate > 1e-3


class _HeadPair:
    def __init__(self, *, vc0, wc0, wc_probe, probe):
        self.vc0 = float(vc0)
        self.wc0 = complex(wc0)
        self.wc_probe = complex(wc_probe)
        self.probe = complex(probe)

    def at(self, omega):
        value = self.wc0 if complex(omega) == 0.0j else self.wc_probe
        return HeadSample(
            vc0=complex(self.vc0), wcoul0=complex(value),
            source="unit", omega=complex(omega))


@pytest.mark.parametrize("model", ["gn", "hl"])
def test_dynamic_head_keeps_frequency_when_fixed_refit_is_adequate(model):
    probe = 2.0j if model == "gn" else 3.0
    config = SimpleNamespace(
        ppm=SimpleNamespace(head_omega_h_ry=None),
        compute_mode=SimpleNamespace(ppm_model=model),
    )
    meta = SimpleNamespace(nelec=4.0, cell_volume=16.0 * np.pi)
    session = {"policy": "fixed_poles"}
    messages = []
    first_source = _HeadPair(
        vc0=10.0, wc0=6.0, wc_probe=8.0, probe=probe)
    first = _fit_head_correction(
        first_source, config=config, meta=meta, probe_omega=probe,
        print_fn=messages.append, outer_refit_session=session)

    # Scale Wc(0) and its probe value together.  Residues must move while
    # the exact same head pole still reconstructs both samples.
    second_source = _HeadPair(
        vc0=10.0,
        wc0=10.0 + 0.75 * (first.wc_head_0),
        wc_probe=10.0 + 0.75 * (
            first.B_h / (complex(probe) ** 2 - first.omega_h_sq)),
        probe=probe,
    )
    second = _fit_head_correction(
        second_source, config=config, meta=meta, probe_omega=probe,
        print_fn=messages.append, outer_refit_session=session)

    assert second.omega_h == first.omega_h
    assert second.B_h == pytest.approx(0.75 * first.B_h)
    assert any("verdict=ADEQUATE" in message for message in messages)


def test_frozen_dynamic_head_reuses_model_without_resolving_samples():
    sentinel = object()
    source = SimpleNamespace(
        at=lambda _omega: pytest.fail("frozen head touched its source"))
    config = SimpleNamespace(
        ppm=SimpleNamespace(head_omega_h_ry=None),
        compute_mode=SimpleNamespace(ppm_model="gn"),
    )
    session = {"policy": "fixed_poles", "head": {"current": sentinel}}

    got = _fit_head_correction(
        source, config=config, meta=SimpleNamespace(), probe_omega=2.0j,
        print_fn=lambda _message: None, outer_refit_session=session,
        frozen=True)

    assert got is sentinel


def test_frozen_ppm_body_reuses_model_before_allocating_wc_tensors():
    sentinel = object()
    session = {"policy": "fixed_poles", "current": sentinel}

    got = fit_ppm(
        object(), object(), object(), 2.0j, object(),
        n_mu_logical=1, outer_refit_session=session, frozen=True,
        print_fn=lambda _message: None)

    assert got is sentinel


def _sc_config(**overrides):
    values = dict(
        max_iter=4,
        tol_ev=1e-4,
        accelerator="rcrop",
        history_depth=5,
        mixing=1.0,
        dump_dir=None,
    )
    values.update(overrides)
    return SCConfig(**values)


def test_sc_outer_refit_policy_is_typed_and_defaults_fixed():
    assert _sc_config().outer_refit_policy == "fixed_poles"
    assert _sc_config(outer_refit_policy="full").outer_refit_policy == "full"
    with pytest.raises(ValueError, match="sc_outer_refit_policy"):
        _sc_config(outer_refit_policy="freeze_forever")

"""Prepared deck/config only; no W, SC convergence or material qualification."""
import configparser
import hashlib
from pathlib import Path


INPUTS = Path(__file__).with_name("inputs")
BASE = INPUTS/"corrected_protected160_750_q4_from_accepted_q8.in"
FEJER = INPUTS/"corrected_protected160_750_q4_fejer_from_accepted_q8.in"


def test_fejer_prepared_deck_changes_only_explicit_window_from_actual_sharp_input():
    assert hashlib.sha256(BASE.read_bytes()).hexdigest() == "1712d907f1468a518afe8c8a4fdd17b7d7909392fbd97e22c6e7d11996b9c299"
    def section(path):
        cfg = configparser.ConfigParser(interpolation=None)
        cfg.read(path)
        return dict(cfg["cohsex"])
    sharp, fejer = section(BASE), section(FEJER)
    assert fejer.pop("qgrid_response_window") == "fejer"
    assert sharp == fejer
    assert "nval" not in fejer and "ncond" not in fejer


def test_native_config_preserves_proper_SC_full750_tail_and_protected160():
    from gw.gw_config import LorraxConfig
    cfg = LorraxConfig.from_input_file(str(FEJER), resolve_hardware=False, print_fn=lambda *a: None)
    assert cfg.qgrid_response_window == "fejer" and cfg.qgrid == (4, 4, 1)
    assert cfg.nband == cfg.zeta_nband == cfg.bands.chi == cfg.bands.sigma == 750
    assert cfg.number_bands_protected == 160 and cfg.density_self_consistent
    assert cfg.sc.max_iter == 30 and cfg.sc.tol_ev == .001
    assert cfg.sc.semicore == "dft" and cfg.sc.head_update == "dft_velocity"
    assert cfg.sigma.w_model == "shared_pole" and cfg.sigma.w_accuracy == "production"
    assert cfg.sigma.regularization_ev == .25 and cfg.sigma.quadrature_eps == 1.e-4
    assert cfg.sigma.band_extrapolation and cfg.sigma.band_extrapolation_estimator == "spectral_shell"

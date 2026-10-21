"""CPU pins of the contour-deformation deck gates (``gw.gw_config._resolve_cd_inputs``) read from a deck.

GATE cd_self_consistent_unimplemented (owner 2026-10-20): CD is the one-shot reference route, so a deck whose
``qp_solver`` resolves to self_consistent -- by name, or ``auto`` with the deprecated ``self_consistent =
true`` (``GWConfig.qp_solver``) -- refuses at parse time; ``one_shot_dft`` and a bare ``auto`` parse.
"""
import warnings

import pytest

from gw.gw_config import read_lorrax_input

CD_DECK = """[cohsex]
sys_dim = 3
compute_mode = mpa
sigma_frequency_integration = contour_deformation
head_correction = off
linalg = local
use_band_extrapolation = false
"""


def _read(tmp_path, extra):
    deck = tmp_path / "cd.in"
    deck.write_text(CD_DECK + extra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)      # the self_consistent key is deprecated
        return read_lorrax_input(str(deck))


@pytest.mark.parametrize("extra", ["qp_solver = self_consistent\n",
                                   "self_consistent = true\n",
                                   "qp_solver = auto\nself_consistent = true\n"])
def test_cd_refuses_self_consistent(tmp_path, extra):
    with pytest.raises(NotImplementedError, match="GATE cd_self_consistent_unimplemented") as refusal:
        _read(tmp_path, extra)
    text = str(refusal.value)
    assert "got:" in text and "want:" in text and "why:" in text, text


@pytest.mark.parametrize("extra", ["", "qp_solver = auto\n", "qp_solver = one_shot_dft\n",
                                   "qp_solver = auto\nself_consistent = false\n"])
def test_cd_one_shot_parses(tmp_path, extra):
    params = _read(tmp_path, extra)
    assert params["sigma_frequency_integration"] == "contour_deformation"


def test_self_consistent_pole_model_is_untouched(tmp_path):
    deck = tmp_path / "pole.in"
    deck.write_text("[cohsex]\nsys_dim = 3\ncompute_mode = mpa\nqp_solver = self_consistent\n")
    assert read_lorrax_input(str(deck))["sigma_frequency_integration"] == "pole_model"

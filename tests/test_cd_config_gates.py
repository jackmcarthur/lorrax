"""CPU pin of GATE cd_self_consistent_unimplemented (``gw.gw_config._resolve_cd_inputs``), read from a deck.

Owner 2026-10-20: CD is the one-shot reference route, so a deck whose ``qp_solver`` resolves to self_consistent --
by name, or ``auto`` with the deprecated ``self_consistent = true`` (``GWConfig.qp_solver``) -- refuses at parse
time; ``one_shot_dft`` and a bare ``auto`` parse, and the pole route is untouched.
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


def test_cd_refuses_self_consistent_and_keeps_one_shot(tmp_path):
    deck = tmp_path / "cd.in"

    def read(text):
        deck.write_text(text)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)      # the self_consistent key is deprecated
            return read_lorrax_input(str(deck))
    for extra in ("qp_solver = self_consistent\n", "qp_solver = auto\nself_consistent = true\n"):
        with pytest.raises(NotImplementedError, match="GATE cd_self_consistent_unimplemented") as refusal:
            read(CD_DECK + extra)
        assert all(part in str(refusal.value) for part in ("got:", "want:", "why:")), refusal.value
    for extra in ("", "qp_solver = auto\n", "qp_solver = one_shot_dft\n"):
        assert read(CD_DECK + extra)["sigma_frequency_integration"] == "contour_deformation"
    pole = read("[cohsex]\nsys_dim = 3\ncompute_mode = mpa\nqp_solver = self_consistent\n")
    assert pole["sigma_frequency_integration"] == "pole_model"

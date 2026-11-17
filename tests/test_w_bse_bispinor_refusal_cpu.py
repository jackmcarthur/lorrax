"""``GATE w_bse_bispinor_uncertified``: the ladder W refuses a bispinor reference by name.

The ladder (``screening_diagrams = w_bse``) is tested and gated on scalar and
two-component references.  With ``bispinor = true`` nothing refused it and
nothing certified it, so the parse gate names it.  ``w_rpa`` is untouched and
the check mode ``w_rpa_resolvent`` stays open (its W is the Dyson one).
"""
from dataclasses import replace

import pytest

from gw.gw_config import (LorraxConfig, ScreeningDiagrams,
                          refuse_unsupported_screening_diagrams)


def _config(tmp_path, *, diagrams="w_rpa", mode="cohsex", extra=""):
    path = tmp_path / "gw.in"
    path.write_text(f"[cohsex]\nsys_dim = 3\ncompute_mode = {mode}\n"
                    f"screening_diagrams = {diagrams}\nhead_correction = off\n"
                    "number_bands = 30\nnval = 4\nncond = 4\n" + extra)
    return LorraxConfig.from_input_file(str(path), resolve_hardware=False,
                                        runtime_platform="cpu", print_fn=lambda *_a: None)


def test_ladder_refuses_a_bispinor_reference_by_name(tmp_path):
    ladder = _config(tmp_path, diagrams="w_bse")
    assert ladder.screening.diagrams is ScreeningDiagrams.W_BSE and not ladder.bispinor
    refuse_unsupported_screening_diagrams(ladder)           # two-component: served
    with pytest.raises(ValueError, match="GATE w_bse_bispinor_uncertified") as err:
        refuse_unsupported_screening_diagrams(replace(ladder, bispinor=True))
    text = str(err.value)
    for part in ("got:", "want:", "fix:", "why:", "bispinor = true",
                 "screening_diagrams = w_rpa"):
        assert part in text


def test_other_diagram_sets_keep_their_bispinor_behaviour(tmp_path):
    for diagrams in ("w_rpa", "w_rpa_resolvent"):
        config = _config(tmp_path, diagrams=diagrams)
        refuse_unsupported_screening_diagrams(replace(config, bispinor=True))


def test_an_input_file_with_both_keys_refuses_at_parse(tmp_path):
    for mode in ("cohsex", "gn_ppm", "tangential"):
        with pytest.raises(ValueError, match="GATE w_bse_bispinor_uncertified"):
            _config(tmp_path, diagrams="w_bse", mode=mode, extra="bispinor = true\n")

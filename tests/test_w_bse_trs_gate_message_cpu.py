"""``GATE w_bse_requires_measured_trs`` says why the verdict is off, and the fix for that reason.

Time reversal is off for a magnetic reference and also for a two-component
WFN staged without its QE ``data-file-schema.xml`` (no evidence either way).
The second is a staging mistake on a nonmagnetic system, so the refusal names
it and says which files to put beside the WFN instead of sending the user to
``w_rpa``.
"""
from types import SimpleNamespace as NS

import pytest

from gw.gw_config import ScreeningDiagrams
from gw.screening import refuse_w_bse_without_trs


def _config(diagrams=ScreeningDiagrams.W_BSE):
    return NS(screening=NS(diagrams=diagrams))


def _refusal(sym) -> str:
    with pytest.raises(RuntimeError, match="GATE w_bse_requires_measured_trs") as err:
        refuse_w_bse_without_trs(_config(), sym)
    return str(err.value)


def test_missing_schema_is_named_with_its_fix():
    text = _refusal(NS(trs_allowed=False, qe_symmetry_binding=None,
                       qe_symmetry_diagnostic="no nearby data-file-schema.xml candidate was found"))
    assert "no evidence either way" in text
    assert "two-component WFN has no matching QE data-file-schema.xml" in text
    assert "no nearby data-file-schema.xml candidate was found" in text
    fix = text.split("fix:")[1].split("why:")[0]
    assert "stage the data-file-schema.xml" in fix and ".save" in fix
    assert "w_rpa" not in fix


def test_schema_without_a_matching_scf_is_named_with_its_fix():
    binding = NS(n_antiunitary=0, scf_absolute_magnetization=None)
    text = _refusal(NS(trs_allowed=False, qe_symmetry_binding=binding))
    assert "no evidence either way" in text and "no SCF schema" in text
    assert "stage the data-file-schema.xml" in text.split("fix:")[1].split("why:")[0]


@pytest.mark.parametrize("binding", [
    NS(n_antiunitary=2, scf_absolute_magnetization=None),      # QE types t_rev rows
    NS(n_antiunitary=0, scf_absolute_magnetization=3.2),       # a magnetic SCF
    NS(n_antiunitary=0, scf_absolute_magnetization=0.0),       # the Kramers check failed
])
def test_a_reference_that_breaks_time_reversal_is_sent_to_w_rpa(binding):
    text = _refusal(NS(trs_allowed=False, qe_symmetry_binding=binding))
    assert "breaks time reversal" in text and "no evidence" not in text
    assert "screening_diagrams = w_rpa" in text.split("fix:")[1].split("why:")[0]


def test_gate_is_silent_with_time_reversal_and_for_other_diagrams():
    refuse_w_bse_without_trs(_config(), NS(trs_allowed=True, qe_symmetry_binding=None))
    off = NS(trs_allowed=False, qe_symmetry_binding=None)
    for diagrams in (ScreeningDiagrams.W_RPA, ScreeningDiagrams.W_RPA_RESOLVENT):
        refuse_w_bse_without_trs(_config(diagrams), off)

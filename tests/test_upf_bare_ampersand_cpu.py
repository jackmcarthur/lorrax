"""A UPF whose PP_INFO carries ld1.x's unescaped ``&INPUT`` namelist loads.

ld1.x copies its input into PP_INFO without escaping ``&``; strict XML refuses
the file ("not well-formed (invalid token)") and the run then had no
pseudopotentials, while QE reads it.  The reader escapes bare ``&`` only:
entities and character references stay what they were.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

pytest.importorskip("xsdata")

from psp.upf.load_upf import (detect_version, load_upf, load_upf_pswfc,  # noqa: E402
                              parse_upf_xml)

FIXTURE = Path(__file__).parent / "hsuite/fixture/H.upf"
LD1_INPUT = (" &INPUT\n   title='H',\n   zed=1.,\n   config='1s1',\n   iswitch=3,\n /\n"
             " &INPUTP\n   pseudotype=2,\n /\n Fe &amp; Co &#38; &lt;r&gt; R&D\n")


def test_ld1_input_namelist_in_pp_info(tmp_path):
    text = FIXTURE.read_text()
    upf = tmp_path / "H_ld1.upf"
    upf.write_text(text.replace("<PP_INPUTFILE>\n", "<PP_INPUTFILE>\n" + LD1_INPUT, 1))
    with pytest.raises(ET.ParseError, match="not well-formed"):
        ET.parse(str(upf))
    assert detect_version(upf) == detect_version(FIXTURE) == "2.0.1"
    info = parse_upf_xml(upf).find("PP_INFO/PP_INPUTFILE").text
    assert "&INPUT" in info and "&INPUTP" in info and "Fe & Co & <r> R&D" in info
    got, want = load_upf_pswfc(upf), load_upf_pswfc(FIXTURE)
    assert np.array_equal(got["r"], want["r"])
    assert [c["label"] for c in got["chi"]] == [c["label"] for c in want["chi"]]
    pseudo = load_upf(upf)
    assert pseudo.pp_header.z_valence == load_upf(FIXTURE).pp_header.z_valence

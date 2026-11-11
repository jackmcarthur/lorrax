"""GATE gaunt_upf_j_averaged: a gaunt_pp-corrected UPF refuses a j-averaged V_NL.

gaunt_pp stores the frozen-core Gaunt exchange as a full 2x2 dD per (l, j)
channel in PP_DIJ and stamps PP_INFO.  ``vnl_ops.build_vnl_setup`` (the one
owner of the soc verdict) must refuse the j-averaged operator, which reads only
diag(D), for a stamped file, and announce the variant when j-resolved.  The
fixture is the hsuite's fully-relativistic H.upf, stamped in tmp_path.
"""
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from psp import vnl_ops
from psp.pseudos import gaunt_stamp, load_pseudopotentials

FIXTURE = Path(__file__).parent / "hsuite/fixture/H.upf"
STAMP = "Modified by gaunt_pp on 2026-11-10: variant 'gaunt_core_so'.\n"


def _pseudos(tmp_path, stamped):
    d = tmp_path / ("stamped" if stamped else "plain")
    d.mkdir()
    shutil.copy(FIXTURE, d / "H.upf")
    if stamped:
        text = (d / "H.upf").read_text()
        assert text.count("<PP_INFO>\n") == 1
        (d / "H.upf").write_text(text.replace("<PP_INFO>\n", "<PP_INFO>\n" + STAMP))
    return load_pseudopotentials(str(d))


def _wfn(*, spinorbit, nspinor):
    a = 10.0
    return SimpleNamespace(
        spinorbit=spinorbit, nspinor=nspinor, blat=2 * np.pi / a,
        bvec=np.eye(3), cell_volume=a ** 3, ecutwfc=4.0,
        atom_types=np.array([1]), atom_crys=np.zeros((1, 3)))


def test_gaunt_upf_gate(tmp_path):
    stamped = _pseudos(tmp_path, stamped=True)
    plain = _pseudos(tmp_path, stamped=False)

    # Detector: the variant from the retained PP_INFO, and from the file when
    # the parsed object holds no PP_INFO; None for an unstamped file.
    assert gaunt_stamp(stamped["H"]) == "gaunt_core_so"
    assert gaunt_stamp(plain["H"]) is None
    bare = SimpleNamespace(pp_info=None, _source_path=stamped["H"]._source_path)
    assert gaunt_stamp(bare) == "gaunt_core_so"

    # The real predicate fires: a j-averaged verdict from QE <spinorbit>=false
    # and one forced by nspinor=1, both through build_vnl_setup.
    for wfn in (_wfn(spinorbit=False, nspinor=2), _wfn(spinorbit=None, nspinor=1)):
        with pytest.raises(ValueError) as err:
            vnl_ops.build_vnl_setup(wfn, pseudos=stamped, q_max=2.0,
                                    print_fn=lambda s: None)
        text = str(err.value)
        assert text.startswith("GATE gaunt_upf_j_averaged:")
        for field in ("got:", "want:", "why:", "fix:", "gaunt_core_so"):
            assert field in text

    # j-resolved: passes, announces the variant, and stamps the provenance.
    said = []
    setup = vnl_ops.build_vnl_setup(_wfn(spinorbit=True, nspinor=2),
                                    pseudos=stamped, q_max=2.0,
                                    print_fn=said.append)
    assert setup.soc
    assert any("Gaunt" in s and "gaunt_core_so" in s for s in said)
    assert "gaunt_core_so" in setup.soc_provenance

    # An unstamped file passes both verdicts and announces nothing.
    for spinorbit in (True, False):
        said = []
        setup = vnl_ops.build_vnl_setup(_wfn(spinorbit=spinorbit, nspinor=2),
                                        pseudos=plain, q_max=2.0,
                                        print_fn=said.append)
        assert setup.soc is spinorbit
        assert not any("Gaunt" in s for s in said)

    # A species absent from the cell is not checked.
    assert vnl_ops.gaunt_upf_gate(
        stamped, SimpleNamespace(atom_types=np.array([26]), spinorbit=False),
        soc=False, nspinor=2) == {}

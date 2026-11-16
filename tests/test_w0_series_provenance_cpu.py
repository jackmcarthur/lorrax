"""Which W a stored ISDF tensor file carries, and who may read it.

Under ``screening_diagrams = w_bse`` the GW run's Sigma uses the ladder W,
while the file stores the RPA W(0) that the ladder used as its direct rung
(``gw.screening_bse.prepare_ladder_restart``).  The file must say both, and a
consumer that needs one series must refuse another
(``file_io.restart_bundle.require_w0_series``).
"""
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from file_io.restart_bundle import (describe_w0_provenance, read_w0_provenance,
                                    require_w0_series)
from gw.gw_config import ComputeMode, ScreeningDiagrams
from gw.gw_output import _stamp_screening_diagrams, stamp_w0_route


def _bundle(path, *, w0=None, head=None, lr=None, gw=None):
    with h5py.File(path, "w") as f:
        d = f.create_dataset("W0_qmunu", data=np.zeros((1, 2, 2), np.complex128))
        d.attrs["W0_ready"] = True
        if w0 is not None:
            d.attrs["screening_diagrams"] = w0
        h = f.create_dataset("whead", data=np.zeros(1, np.complex128))
        if head is not None:
            h.attrs["screening_diagrams"] = head
        if lr is not None:
            f.create_group("W0_lr").attrs["screening_diagrams"] = lr
        if gw is not None:
            f.attrs["gw_screening_diagrams"] = gw
    return str(path)


def _config(diagrams, mode=ComputeMode.COHSEX):
    return SimpleNamespace(compute_mode=mode,
                           screening=SimpleNamespace(diagrams=diagrams))


def test_writers_stamp_body_head_and_sigma_series(tmp_path):
    """A w_bse run: W0/whead stamped w_rpa (the rung), the file stamped w_bse (Sigma)."""
    path = _bundle(tmp_path / "isdf.h5")
    with h5py.File(path, "a") as f:
        f.create_group("W0_lr")          # a stale group this persist did not write
    _stamp_screening_diagrams(path, _config(ScreeningDiagrams.W_RPA),
                              members=("W0_qmunu", "whead"))
    stamp_w0_route(path, _config(ScreeningDiagrams.W_BSE))
    prov = read_w0_provenance(path)
    assert prov["W0_qmunu"] == "w_rpa" and prov["whead"] == "w_rpa"
    assert prov["W0_lr"] is None          # not written now, so not relabelled
    assert prov["gw"] == "w_bse" and prov["gw_compute_mode"] == "cohsex"
    line = describe_w0_provenance(prov, consumer="BSE")
    assert "RPA W(0)" in line and "screening_diagrams = w_bse" in line and "NOT stored" in line
    # The ladder rung and an RPA consumer accept it and log the line.
    seen = []
    require_w0_series(path, "w_rpa", consumer="rung", print_fn=seen.append)
    assert seen and "NOT stored" in seen[0]
    # x_only builds no W: no Sigma series is claimed.
    bare = _bundle(tmp_path / "bare.h5")
    stamp_w0_route(bare, _config(ScreeningDiagrams.W_RPA, ComputeMode.X_ONLY))
    assert read_w0_provenance(bare)["gw"] is None


def test_rpa_run_states_no_difference(tmp_path):
    path = _bundle(tmp_path / "rpa.h5", w0="w_rpa", head="w_rpa", gw="w_rpa")
    line = describe_w0_provenance(read_w0_provenance(path), consumer="BSE")
    assert "NOT stored" not in line and "screening_diagrams = w_rpa" in line


def test_consumer_refuses_other_series(tmp_path):
    ladder = _bundle(tmp_path / "ladder.h5", w0="w_bse", head="w_bse", gw="w_bse")
    with pytest.raises(ValueError, match="GATE w0_series:"):
        require_w0_series(ladder, "w_rpa", consumer="the ladder's direct rung W_R")
    require_w0_series(ladder, "w_bse", consumer="a ladder-W consumer", print_fn=lambda *_: None)
    rpa = _bundle(tmp_path / "rpa.h5", w0="w_rpa", gw="w_bse")
    with pytest.raises(ValueError, match="GATE w0_series:.*needs the ladder-corrected W"):
        require_w0_series(rpa, "w_bse", consumer="a ladder-W consumer")


def test_torn_and_unstamped_files(tmp_path):
    torn = _bundle(tmp_path / "torn.h5", w0="w_rpa", head="w_bse")
    with pytest.raises(ValueError, match="GATE w0_series_torn"):
        require_w0_series(torn, "w_rpa", consumer="BSE")
    torn_lr = _bundle(tmp_path / "torn_lr.h5", w0="w_rpa", head="w_rpa", lr="w_bse")
    with pytest.raises(ValueError, match="GATE w0_series_torn"):
        require_w0_series(torn_lr, "w_rpa", consumer="BSE")
    # Written before the stamp: read as RPA, and said so.
    old = _bundle(tmp_path / "old.h5")
    seen = []
    prov = require_w0_series(old, "w_rpa", consumer="BSE", print_fn=seen.append)
    assert prov["W0_qmunu"] is None and "unstamped" in seen[0]
    with pytest.raises(ValueError, match="GATE w0_series:"):
        require_w0_series(old, "w_bse", consumer="a ladder-W consumer")

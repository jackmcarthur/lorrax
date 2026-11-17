"""The bsemat export says which W it reads, and reads the RPA ``W0_qmunu`` only.

The export loads through the shared BSE loader, which follows
``bse_screening_diagrams``; its own head split reads the file's RPA head
tensor and wings.  So the export names the series itself: it prints the
``W0 provenance:`` line, pins the loader to ``w_rpa``, and refuses the ladder
key (``GATE bsemat_ladder_w0``) instead of writing a ladder body on an RPA
head.  Unit cells: exempt from the four-GPU rule (file attributes only).
"""
import inspect

import h5py
import numpy as np
import pytest

from bse import bsemat_export
from bse.bsemat_export import export_w0_provenance
from file_io.restart_bundle import LADDER_S_CART, LADDER_W0, LADDER_WHEAD


def _file(path, *, w0="w_rpa", gw=None, ladder=False):
    with h5py.File(path, "w") as f:
        d = f.create_dataset("W0_qmunu", data=np.zeros((1, 2, 2), np.complex128))
        d.attrs["W0_ready"] = True
        if w0:
            d.attrs["screening_diagrams"] = w0
        f.create_dataset("whead", data=np.asarray([1.0 + 0j]))
        if gw:
            f.attrs["gw_screening_diagrams"] = gw
        if ladder:
            d = f.create_dataset(LADDER_W0, data=np.ones((1, 2, 2), np.complex128))
            d.attrs["W0_ready"] = True
            d.attrs["screening_diagrams"] = "w_bse"
            f.create_dataset(LADDER_WHEAD, data=np.asarray([2.0 + 0j]))
            f.create_dataset(LADDER_S_CART, data=np.eye(3, dtype=np.complex128))
    return str(path)


def test_export_prints_its_provenance_line(tmp_path):
    lines = []
    export_w0_provenance(_file(tmp_path / "rpa.h5"), {}, print_fn=lines.append)
    assert len(lines) == 1 and "W0 provenance:" in lines[0]
    assert "bsemat export" in lines[0] and "RPA W(0)" in lines[0]

    lines = []
    prov = export_w0_provenance(_file(tmp_path / "ladder.h5", gw="w_bse", ladder=True),
                                {"bse_screening_diagrams": "w_rpa"}, print_fn=lines.append)
    assert prov["gw"] == "w_bse" and prov["ladder_stored"]
    # After a w_bse GW run the line says that Sigma used the ladder W and that
    # this consumer reads the RPA one.
    assert "screening_diagrams = w_bse" in lines[0] and "this consumer reads the RPA" in lines[0]

    lines = []
    export_w0_provenance(_file(tmp_path / "old.h5", w0=None), {}, print_fn=lines.append)
    assert "unstamped" in lines[0]


def test_export_refuses_the_ladder_key_and_a_non_rpa_w0(tmp_path):
    path = _file(tmp_path / "ladder.h5", gw="w_bse", ladder=True)
    with pytest.raises(ValueError, match="GATE bsemat_ladder_w0"):
        export_w0_provenance(path, {"bse_screening_diagrams": "w_bse"}, print_fn=lambda _l: None)
    with pytest.raises(ValueError, match="GATE w0_series"):
        export_w0_provenance(_file(tmp_path / "bad.h5", w0="w_bse"), {}, print_fn=lambda _l: None)


def test_export_context_names_the_series_before_and_at_the_load():
    source = inspect.getsource(bsemat_export._export_context)
    gate = source.index("export_w0_provenance(restart, params)")
    load = source.index("load_bse_data_from_restart_sharded(\n")
    assert gate < load
    call = source[load:source.index(")", load)]
    assert 'w_series="w_rpa"' in call

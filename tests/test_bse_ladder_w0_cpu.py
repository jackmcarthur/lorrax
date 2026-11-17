"""``bse_screening_diagrams``: the BSE reads the ladder W(0) only when asked, and only if stored.

A ``screening_diagrams = w_bse`` GW run stores the RPA rung as ``W0_qmunu``
and the ladder W(0) its Sigma used as ``W0_ladder_qmunu`` with ``whead_ladder``
(``gw.gw_output.persist_ladder_w0``).  The loader resolves the series once
(``restart_bundle.resolve_bse_w_series``) and threads it to both readers; the
ladder's own reload always asks for the rung.
"""
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from file_io import restart_bundle
from file_io.restart_bundle import (LADDER_S_CART, LADDER_W0, LADDER_WHEAD,
                                    describe_w0_provenance, read_w0_provenance,
                                    require_ladder_w0, resolve_bse_w_series)


def _file(path, *, ladder=True, ready=True, stamp="w_bse", head=True):
    with h5py.File(path, "w") as f:
        d = f.create_dataset("W0_qmunu", data=np.zeros((1, 2, 2), np.complex128))
        d.attrs["W0_ready"] = True
        d.attrs["screening_diagrams"] = "w_rpa"
        f.create_dataset("whead", data=np.asarray([1.0 + 0j]))
        f.attrs["gw_screening_diagrams"] = "w_bse"
        if ladder:
            d = f.create_dataset(LADDER_W0, data=np.ones((1, 2, 2), np.complex128))
            d.attrs["W0_ready"] = ready
            if stamp:
                d.attrs["screening_diagrams"] = stamp
            if head:
                f.create_dataset(LADDER_WHEAD, data=np.asarray([2.0 + 0j]))
                f.create_dataset(LADDER_S_CART, data=np.eye(3, dtype=np.complex128))
    return str(path)


def test_key_resolves_and_refuses_unknown_values():
    assert resolve_bse_w_series(None) == "w_rpa"
    assert resolve_bse_w_series({}) == "w_rpa"
    assert resolve_bse_w_series({"bse_screening_diagrams": " W_BSE "}) == "w_bse"
    with pytest.raises(ValueError, match="bse_screening_diagrams"):
        resolve_bse_w_series({"bse_screening_diagrams": "w_rpa_resolvent"})


def test_deck_default_is_rpa(tmp_path):
    from gw.gw_config import read_lorrax_input
    deck = tmp_path / "gw.in"
    deck.write_text("[cohsex]\nsys_dim = 3\nnumber_bands = 8\n")
    assert resolve_bse_w_series(read_lorrax_input(str(deck))) == "w_rpa"
    deck.write_text("[cohsex]\nsys_dim = 3\nnumber_bands = 8\nbse_screening_diagrams = w_bse\n")
    assert resolve_bse_w_series(read_lorrax_input(str(deck))) == "w_bse"


def test_ladder_w0_is_required_complete_and_stamped(tmp_path):
    require_ladder_w0(_file(tmp_path / "ok.h5"))
    for name, kwargs in (("absent", dict(ladder=False)), ("placeholder", dict(ready=False)),
                         ("unstamped", dict(stamp=None)), ("rpa_stamp", dict(stamp="w_rpa")),
                         ("headless", dict(head=False))):
        with pytest.raises(ValueError, match="GATE bse_ladder_w0"):
            require_ladder_w0(_file(tmp_path / f"{name}.h5", **kwargs))


def test_provenance_says_the_ladder_is_stored_beside_the_rung(tmp_path):
    line = describe_w0_provenance(read_w0_provenance(_file(tmp_path / "ok.h5")), consumer="BSE")
    assert "stored beside it as W0_ladder_qmunu" in line and "NOT stored" not in line
    line = describe_w0_provenance(
        read_w0_provenance(_file(tmp_path / "no.h5", ladder=False)), consumer="BSE")
    assert "NOT stored" in line


def test_metadata_and_payload_follow_the_series(tmp_path, monkeypatch):
    path = _file(tmp_path / "ok.h5")
    monkeypatch.setattr(restart_bundle, "_require_current", lambda _f: None)
    monkeypatch.setattr(restart_bundle, "_metadata_band_count", lambda _f: 0)
    with h5py.File(path, "a") as f:
        f.create_dataset("n_rmu_logical", data=2)
    rpa = restart_bundle.read_metadata(path)
    ladder = restart_bundle.read_metadata(path, w_series="w_bse")
    assert rpa["screened_head"][0] == 1.0 and rpa["head_cartesian"] is None
    assert ladder["screened_head"][0] == 2.0 and ladder["head_cartesian"].shape == (3, 3)
    assert rpa["screened_ready"] and ladder["screened_ready"]

    # read_bse_payload reads the dataset of the series it was given, never a rebuilt W.
    kinds = []
    monkeypatch.setattr(restart_bundle, "read_metadata",
                        lambda _p, **k: dict(screened_ready=True, grid=(1, 1, 1)))
    monkeypatch.setattr(restart_bundle, "read_interaction_gamma", lambda *a, **k: "V0")
    monkeypatch.setattr(restart_bundle, "read_interaction_qgrid",
                        lambda _f, kind, *a, **k: kinds.append((kind, k.get("nohead"))) or kind)
    out = restart_bundle.read_bse_payload(path, None, None, [0], [1], read_psi=False,
                                          w_series="w_bse")
    assert out[3] == "screened_ladder" and kinds == [("screened_ladder", False)]
    out = restart_bundle.read_bse_payload(path, None, None, [0], [1], read_psi=False)
    assert out[3] == "screened"
    with pytest.raises(ValueError, match="GATE bse_ladder_w0"):
        restart_bundle.read_bse_payload(path, None, None, [0], [1], read_psi=False,
                                        w_series="w_bse", screened_w=object())
    with pytest.raises(ValueError, match="GATE bse_ladder_w0"):
        restart_bundle.read_bse_payload(_file(tmp_path / "no.h5", ladder=False), None, None,
                                        [0], [1], read_psi=False, w_series="w_bse")


def test_report_row_names_both_stored_w(tmp_path):
    from gw.gw_config import LorraxConfig
    from gw.production_report import stored_w0_row

    def config(mode):
        path = tmp_path / f"{mode}.in"
        path.write_text(f"[cohsex]\nsys_dim = 3\ncompute_mode = {mode}\n"
                        "screening_diagrams = w_bse\nhead_correction = off\n"
                        "number_bands = 30\nnval = 4\nncond = 4\n")
        return LorraxConfig.from_input_file(str(path), resolve_hardware=False,
                                           runtime_platform="cpu", print_fn=lambda *_a: None)
    assert "stored beside it (W0_ladder_qmunu)" in stored_w0_row(config("cohsex"))
    assert "not stored on this route" in stored_w0_row(config("tangential"))

"""Standalone htransform's guarded output-window contract."""
from __future__ import annotations

import inspect

import pytest


def test_local_vbm_index_uses_the_absolute_window_start():
    from bandstructure.htransform import resolve_local_vbm_index

    assert resolve_local_vbm_index(nelec=10, band_start=0,
                                   n_return_bands=18) == 9
    assert resolve_local_vbm_index(nelec=10, band_start=6,
                                   n_return_bands=12) == 3
    for start, count in ((10, 8), (0, 9)):
        with pytest.raises(ValueError, match="does not contain the VBM"):
            resolve_local_vbm_index(10, start, count)


def test_standalone_main_buys_guards_and_returns_only_the_deck_window():
    from bandstructure import htransform as ht

    src = inspect.getsource(ht.main)
    assert '"--guard-bands", type=int, default=4' in src
    assert "n_guard_bands=args.guard_bands" in src
    assert 'n_return_bands = int(params["nval"]) + int(params["ncond"])' in src
    assert "n_return_bands=n_return_bands" in src
    assert 'band_start=int(wfn.nelec) - int(params["nval"])' in src


def test_standalone_htransform_reuses_the_f_shoulder_gate():
    from bandstructure import htransform as ht

    src = inspect.getsource(ht.h_transform)
    assert "from .bse_setup import _f_shoulder_gate" in src
    assert "f_eps, 0, nb_keep" in src
    assert 'where="htransform"' in src
    assert "nb_keep = states if n_return_bands is None" in src


def test_output_writer_stamps_fit_and_guard_windows(tmp_path):
    pytest.importorskip("jax")
    import numpy as np
    from bandstructure.htransform import write_bands_to_file
    from common.units import RYD_TO_EV

    path = tmp_path / "bandstructure.dat"
    write_bands_to_file(
        str(path), np.asarray([[1.0, 2.0]]),
        np.asarray([[0.0, 0.0, 0.0]]), np.asarray([0.0]),
        band_start=6, nb_fit=6)
    text = path.read_text()
    assert "absolute_band_window=[6,8)" in text
    assert "fit_bands=6 guard_bands=4" in text
    assert "energy_eV" in text.splitlines()[0]
    assert f"{2.0 * RYD_TO_EV: .8f}" in text

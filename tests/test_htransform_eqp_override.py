"""``bandstructure.htransform.read_eqp_energies`` — the ``--eqp-file`` leg.

The override chain is: ``--eqp-file`` (htransform CLI, argparse at
htransform.py:2510) -> 4th positional of ``initialize_wfns`` (:2623) ->
``enk_sigma = read_eqp_energies(...)`` replacing the DFT energies, with
FATAL on a missing or unparseable file (:1986-2007) -> ``enk_sigma``
returned (:2034-2035) and consumed by ``h_transform`` (:2673).  This file
covers the one leg of that chain that is testable without a deck: the
reader itself, on a synthetic wedge file and a minimal ``SymMaps`` stub.

Nothing here is a writer/reader round-trip tautology: the assertions
compare against a HAND-BUILT expectation of where each wedge value must
land in the full BZ (including the duplicated row for the symmetry-mapped
k), in the module's own Ry unit, and the four refusal arms each construct
the FALSE case and assert it refuses.
"""
import os
import types

import numpy as np
import pytest


def _imports():
    """Inside a function, not at module scope: importing
    ``bandstructure.htransform`` sets ``HDF5_USE_FILE_LOCKING``, and the
    harness refuses env mutation at collection time."""
    from bandstructure.htransform import read_eqp_energies
    from common.units import RYD_TO_EV
    from gw.eqp_bgw import write_bgw_eqp
    return read_eqp_energies, RYD_TO_EV, write_bgw_eqp


def _stub_sym():
    """4-point full BZ over a 3-point file wedge.

    Full-BZ k index 2 is the symmetry image of wedge row 1, so the unfold
    must DUPLICATE that row — the one thing a positional pass-through
    could not do.  ``sym_mats_k`` has 4 rows so ``star_tables_of`` derives
    ``n_sym_spatial = 2``; ``sym_idx_k`` stays below it, so no row is
    conjugated (energies are real either way).
    """
    full_k = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0],
                       [0.0, 0.5, 0.0], [0.5, 0.5, 0.0]])
    return types.SimpleNamespace(
        unfolded_kpts=full_k,
        kirr_fullids=np.array([0, 1, 3]),
        nk_tot=4, nk_red=3,
        irr_idx_k=np.array([0, 1, 1, 2], dtype=np.int32),
        sym_idx_k=np.array([0, 0, 1, 0], dtype=np.int32),
        sym_mats_k=np.zeros((4, 3, 3)),
    )


def _write_wedge_eqp(write_bgw_eqp, path, sym, *, band_offset=2, nb=4):
    """Distinct value per (wedge k, band): 100 + 10*k + band (eV)."""
    wedge_k = np.asarray(sym.unfolded_kpts)[np.asarray(sym.kirr_fullids)]
    bands = np.arange(band_offset, band_offset + nb, dtype=np.float64)
    e_qp = 100.0 + 10.0 * np.arange(len(wedge_k))[:, None] + bands[None, :]
    write_bgw_eqp(path, wedge_k, e_qp - 1.0, e_qp, band_offset=band_offset)
    return e_qp


def test_wedge_values_land_on_the_right_full_bz_k(tmp_path):
    read_eqp_energies, RYD_TO_EV, write_bgw_eqp = _imports()
    sym = _stub_sym()
    path = os.path.join(tmp_path, "eqp1.dat")
    e_qp_ev = _write_wedge_eqp(write_bgw_eqp, path, sym, band_offset=2, nb=4)

    # Absolute window [3, 5) -> file columns 1:3.
    out = np.asarray(read_eqp_energies(path, sym, (3, 5))) * RYD_TO_EV

    assert out.shape == (2, 4), out.shape
    # Hand-built expectation: full-BZ k order is (wedge0, wedge1, wedge1
    # again — the symmetry image — then wedge2), bands transposed to rows.
    expected = e_qp_ev[[0, 1, 1, 2], 1:3].T
    np.testing.assert_allclose(out, expected, atol=1e-8)


def test_a_wrong_sized_wedge_file_refuses(tmp_path):
    read_eqp_energies, _, write_bgw_eqp = _imports()
    sym = _stub_sym()
    path = os.path.join(tmp_path, "eqp1.dat")
    # 2 blocks where the deck's wedge has 3.
    wedge_k = np.asarray(sym.unfolded_kpts)[np.asarray(sym.kirr_fullids)]
    e = np.array([[1.0, 2.0], [3.0, 4.0]])
    write_bgw_eqp(path, wedge_k[:2], e, e, band_offset=0)
    with pytest.raises(ValueError, match="k-blocks"):
        read_eqp_energies(path, sym, (0, 2))


def test_a_foreign_decks_kpoints_refuse(tmp_path):
    read_eqp_energies, _, write_bgw_eqp = _imports()
    sym = _stub_sym()
    path = os.path.join(tmp_path, "eqp1.dat")
    # Right block count, wrong coordinates (shifted off the wedge, by more
    # than a lattice vector ambiguity can absorb).
    wedge_k = np.asarray(sym.unfolded_kpts)[np.asarray(sym.kirr_fullids)]
    e = np.ones((3, 2))
    write_bgw_eqp(path, wedge_k + 0.25, e, e, band_offset=0)
    with pytest.raises(ValueError, match="does not belong"):
        read_eqp_energies(path, sym, (0, 2))


def test_a_window_outside_the_files_bands_refuses(tmp_path):
    read_eqp_energies, _, write_bgw_eqp = _imports()
    sym = _stub_sym()
    path = os.path.join(tmp_path, "eqp1.dat")
    _write_wedge_eqp(write_bgw_eqp, path, sym, band_offset=2, nb=4)  # absolute bands [2, 6)
    with pytest.raises(ValueError, match="does not span"):
        read_eqp_energies(path, sym, (0, 4))


def test_a_short_block_inside_the_window_refuses(tmp_path):
    read_eqp_energies, _, _ = _imports()
    sym = _stub_sym()
    path = os.path.join(tmp_path, "eqp1.dat")
    # Ragged file: block 2 carries one band fewer, so read_bgw_eqp NaN-pads
    # it; a window that touches the padded column must refuse, not pad.
    wedge_k = np.asarray(sym.unfolded_kpts)[np.asarray(sym.kirr_fullids)]
    lines = []
    for ik, k in enumerate(wedge_k):
        nb = 2 if ik < 2 else 1
        lines.append(f"{k[0]:13.9f}{k[1]:13.9f}{k[2]:13.9f}{nb:8d}")
        for ib in range(nb):
            lines.append(f"{1:8d}{ib + 1:8d}{1.0:15.9f}{2.0:15.9f}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="missing"):
        read_eqp_energies(path, sym, (0, 2))

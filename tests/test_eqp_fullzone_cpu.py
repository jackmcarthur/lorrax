"""BSE eqp reader: a full-zone file gives every k its own block; star-asymmetric shifts refuse.

``file_io.restart_bundle.apply_eqp_corrections`` reads a BerkeleyGW eqp file
either as the WFN's file wedge (unfolded through the symmetry service, as
before) or, when the file has a block for every full-BZ k, block by block,
which is how BerkeleyGW reads it.  Both refuse QP shifts E_QP - E_DFT that
differ across a symmetry star by more than 1 meV (``GATE eqp_star_spread``).
htransform's ``read_eqp_energies`` places its file through the same helper.

Host-side file reader, no device code: a CPU unit cell, exempt from the
four-GPU rule under its unit-and-CPU-cell clause.  The stars are the bcc Na
hsuite fixture's own (3x3x3: 27 full k, 4 stored); the spectra are synthetic.
"""
import os
from types import SimpleNamespace

import numpy as np
import pytest

from file_io.restart_bundle import (
    EQP_STAR_SPREAD_TOL_EV, _eqp_on_full_bz, apply_eqp_corrections,
    read_bgw_eqp)
from gw.eqp_bgw import write_bgw_eqp

HERE = os.path.dirname(os.path.abspath(__file__))
WFN = os.path.join(HERE, "hsuite", "fixture_na", "WFN.h5")
RY = 13.6056980659          # apply_eqp_corrections' default ry_to_ev
NB_DFT, NB_EQP = 8, 6       # the file covers fewer bands than the spectrum


@pytest.fixture(scope="module")
def case():
    from ffi import _services
    _services.ensure_on_path()
    from symmetry_maps import (reduce_full_bz_to_file_wedge,
                               unfold_file_wedge_to_full_bz)
    from wfn_loader import WfnLoader

    sym = WfnLoader(WFN).symmetry()
    assert int(sym.nk_red) < int(sym.nk_tot)
    rng = np.random.default_rng(20261020)
    e_w = np.sort(rng.uniform(-5.0, 10.0, (int(sym.nk_red), NB_DFT)), axis=1)
    q_w = e_w[:, :NB_EQP] + rng.uniform(-0.5, 0.5, (int(sym.nk_red), NB_EQP))
    k_full = np.asarray(sym.unfolded_kpts, dtype=np.float64)
    return SimpleNamespace(
        sym=sym, rng=rng, k_full=k_full,
        k_wedge=np.asarray(reduce_full_bz_to_file_wedge(sym, k_full)),
        e_wedge=e_w, q_wedge=q_w,
        e_full=np.asarray(unfold_file_wedge_to_full_bz(sym, e_w)),
        q_full=np.asarray(unfold_file_wedge_to_full_bz(sym, q_w)),
        unfold=unfold_file_wedge_to_full_bz)


def _deck(tmp_path):
    path = tmp_path / "deck.in"
    path.write_text(f"wfn_file = {WFN}\n")
    return str(path)


def _write(path, kpts, e_dft, e_qp):
    write_bgw_eqp(str(path), kpts, e_dft[:, :NB_EQP], e_qp, band_offset=0)
    return str(path)


def _pre_fix_rule(case, sym, enk_ry, eqp_file):
    """The reader before this change: every k takes its star parent's block."""
    _k, _d, q, _off = read_bgw_eqp(eqp_file)
    block = np.asarray(case.unfold(sym, q))[:, :NB_EQP]
    out = enk_ry.copy()
    out[:, :NB_EQP] = np.where(~np.isnan(block), block / RY, out[:, :NB_EQP])
    return out


def _k_off_its_parent(sym):
    """A full-BZ k whose star parent is another k: the pre-fix rule drops its block."""
    irr = _full_zone_wfn(sym).irr_idx_k
    return int(np.flatnonzero(irr != np.arange(irr.size))[0])


def _full_zone_wfn(sym):
    """``sym`` as a WFN storing every full-BZ k would carry it (Si 8³ eqpA case).

    The file wedge is the full BZ, and each k's parent row is its star
    representative's own row, so the wedge unfold reads one block per star.
    """
    kf = np.asarray(sym.kirr_fullids, dtype=np.int32)
    return SimpleNamespace(
        unfolded_kpts=np.asarray(sym.unfolded_kpts),
        irr_idx_k=kf[np.asarray(sym.irr_idx_k)],
        sym_idx_k=np.asarray(sym.sym_idx_k),
        sym_mats_k=np.asarray(sym.sym_mats_k),
        nk_tot=int(sym.nk_tot), nk_red=int(sym.nk_tot),
        kirr_fullids=np.arange(int(sym.nk_tot), dtype=np.int32))


def test_readers_agree_wedge_unfold_and_full_zone_blocks(case, tmp_path):
    """The routes that must agree on which QP block lands on which k.

    A wedge file is applied exactly as the pre-fix rule did (bit for bit); a
    symmetric full-zone file, shuffled and shifted by G, equals the unfold of
    the wedge file; and a full-zone file gives each k its own block, where
    the pre-fix wedge unfold would have read the star representative's block.
    """
    deck = _deck(tmp_path)
    enk = case.e_full / RY
    f_wedge = _write(tmp_path / "eqp_wedge.dat", case.k_wedge, case.e_wedge,
                     case.q_wedge)

    # Wedge file unchanged bit for bit.
    unfolded = apply_eqp_corrections(enk, f_wedge, deck)
    assert np.array_equal(unfolded, _pre_fix_rule(case, case.sym, enk, f_wedge))
    assert np.array_equal(unfolded[:, NB_EQP:], enk[:, NB_EQP:])

    # Symmetric full-zone file equals the unfold.
    nk = case.k_full.shape[0]
    perm = case.rng.permutation(nk)
    shift_g = case.rng.integers(-1, 2, (nk, 3))
    f_full = _write(tmp_path / "eqp_full.dat", case.k_full[perm] + shift_g,
                    case.e_full[perm], case.q_full[perm])
    per_k = apply_eqp_corrections(enk, f_full, deck)
    assert float(np.max(np.abs(per_k - unfolded))) <= 1.0e-12

    # Each k gets its own block (control: the pre-fix wedge unfold drops it).
    fz = _full_zone_wfn(case.sym)
    k_star = _k_off_its_parent(case.sym)
    q = case.q_full.copy()
    q[k_star, 2] += 0.4 * EQP_STAR_SPREAD_TOL_EV       # under the gate
    f = _write(tmp_path / "eqp_own.dat", case.k_full, case.e_full, q)
    _k, _d, q_file, _off = read_bgw_eqp(f)
    got = _eqp_on_full_bz(fz, f, case.e_full)
    assert np.array_equal(got, q_file)
    old = np.asarray(case.unfold(fz, q_file))
    assert abs(old[k_star, 2] - got[k_star, 2]) > 0.3 * EQP_STAR_SPREAD_TOL_EV



def test_eqp_writer_numeric_blocks_precede_metadata_and_roundtrip(tmp_path):
    """BGW's numeric-first reader reaches all nine k blocks before metadata."""
    import itertools

    k = np.array(list(itertools.product(range(3), range(3), range(1)))) / [3, 3, 1]
    dft = np.arange(9 * 4, dtype=float).reshape(9, 4) / 64 - 2
    qp = dft + np.arange(9)[:, None] / 16 + .125
    path = tmp_path / "eqp.dat"
    write_bgw_eqp(str(path), k, dft, qp, band_offset=20,
                  comments=("full-zone diagnostic",))
    records = iter(path.read_text().splitlines())
    # Like Common/eqpcor.f90: a numeric read, with no comment-skipping rule.
    for ik in range(9):
        header = next(records).split()
        np.testing.assert_allclose(np.asarray(header[:3], float), k[ik], atol=5e-10)
        assert int(header[3]) == 4
        for ib in range(4):
            row = next(records)
            assert int(row[:8]) == 1 and int(row[8:16]) == 21 + ib
            assert float(row[16:31]) == dft[ik, ib]
            assert float(row[31:46]) == qp[ik, ib]
    footer = list(records)
    assert footer[0].startswith("# Generated by LORRAX ")
    assert footer[-1] == "# full-zone diagnostic"
    assert all(line.startswith("#") for line in footer)
    k_read, dft_read, qp_read, offset = read_bgw_eqp(str(path))
    np.testing.assert_allclose(k_read, k, atol=5e-10)
    np.testing.assert_array_equal(dft_read, dft)
    np.testing.assert_array_equal(qp_read, qp)
    assert offset == 20

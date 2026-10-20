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


def test_wedge_file_is_unchanged_bit_for_bit(case, tmp_path):
    enk = case.e_full / RY
    f = _write(tmp_path / "eqp_wedge.dat", case.k_wedge, case.e_wedge,
               case.q_wedge)
    got = apply_eqp_corrections(enk, f, _deck(tmp_path))
    assert np.array_equal(got, _pre_fix_rule(case, case.sym, enk, f))
    assert np.array_equal(got[:, NB_EQP:], enk[:, NB_EQP:])


def test_symmetric_full_zone_file_equals_the_unfold(case, tmp_path):
    enk = case.e_full / RY
    nk = case.k_full.shape[0]
    perm = case.rng.permutation(nk)
    shift_g = case.rng.integers(-1, 2, (nk, 3))
    f_full = _write(tmp_path / "eqp_full.dat", case.k_full[perm] + shift_g,
                    case.e_full[perm], case.q_full[perm])
    f_wedge = _write(tmp_path / "eqp_wedge.dat", case.k_wedge, case.e_wedge,
                     case.q_wedge)
    deck = _deck(tmp_path)
    per_k = apply_eqp_corrections(enk, f_full, deck)
    unfolded = apply_eqp_corrections(enk, f_wedge, deck)
    assert float(np.max(np.abs(per_k - unfolded))) <= 1.0e-12


def test_full_zone_file_gives_each_k_its_own_block(case, tmp_path):
    fz = _full_zone_wfn(case.sym)
    k_star = _k_off_its_parent(case.sym)
    q = case.q_full.copy()
    q[k_star, 2] += 0.4 * EQP_STAR_SPREAD_TOL_EV       # under the gate
    f = _write(tmp_path / "eqp_full.dat", case.k_full, case.e_full, q)
    _k, _d, q_file, _off = read_bgw_eqp(f)
    got = _eqp_on_full_bz(fz, f, case.e_full)
    assert np.array_equal(got, q_file)
    # Control variant: the pre-fix wedge unfold reads the star
    # representative's block at k_star and drops k_star's own.
    old = np.asarray(case.unfold(fz, q_file))
    assert abs(old[k_star, 2] - got[k_star, 2]) > 0.3 * EQP_STAR_SPREAD_TOL_EV


def test_star_asymmetric_full_zone_file_refuses(case, tmp_path):
    k_star = _k_off_its_parent(case.sym)
    q = case.q_full.copy()
    q[k_star, 3] += 5.0e-3
    f = _write(tmp_path / "eqp_asym.dat", case.k_full, case.e_full, q)
    with pytest.raises(ValueError, match=r"GATE eqp_star_spread: .*eqp_asym\.dat"
                       r": the QP shift E_QP - E_DFT of band 4 differs by "
                       r"5\.00\d meV") as err:
        apply_eqp_corrections(case.e_full / RY, f, _deck(tmp_path))
    assert f"full-BZ k {k_star} " in str(err.value)


def test_htransform_reader_keeps_full_zone_blocks(case, tmp_path):
    """``read_eqp_energies``: a full-zone file keeps its per-k blocks; a wedge file reads as before.

    The first assertion fails under the pre-fix reader, which unfolded the
    file from star parents on a full-zone WFN.
    """
    import jax.numpy as jnp
    from common.units import RYD_TO_EV
    from file_io.restart_bundle import read_eqp_energies

    window = (1, 5)                                     # absolute bands, offset 0
    fz = _full_zone_wfn(case.sym)
    k_star = _k_off_its_parent(case.sym)
    q = case.q_full.copy()
    q[k_star, 2] += 0.4 * EQP_STAR_SPREAD_TOL_EV       # under the gate
    f = _write(tmp_path / "eqp_full.dat", case.k_full, case.e_full, q)
    _k, _d, q_file, _off = read_bgw_eqp(f)

    def ry(a_ev):
        return np.asarray(jnp.asarray(a_ev.T / RYD_TO_EV, dtype=jnp.float64))

    got = np.asarray(read_eqp_energies(f, fz, window))  # (nb, nk) Ry
    assert np.array_equal(got, ry(q_file[:, 1:5]))
    # Control variant: the pre-fix unfold reads k_star's star parent.
    old = np.asarray(case.unfold(fz, q_file[:, 1:5]))
    assert abs(old[k_star, 1] - q_file[k_star, 2]) > 0.3 * EQP_STAR_SPREAD_TOL_EV

    fw = _write(tmp_path / "eqp_wedge.dat", case.k_wedge, case.e_wedge,
                case.q_wedge)
    _k, _d, q_w, _off = read_bgw_eqp(fw)
    assert np.array_equal(np.asarray(read_eqp_energies(fw, case.sym, window)),
                          ry(np.asarray(case.unfold(case.sym, q_w[:, 1:5]))))


def test_htransform_reader_refuses_star_asymmetric_file(case, tmp_path):
    """``read_eqp_energies`` gates against the file's own E_DFT column."""
    from file_io.restart_bundle import read_eqp_energies

    fz = _full_zone_wfn(case.sym)
    k_star = _k_off_its_parent(case.sym)
    q = case.q_full.copy()
    q[k_star, 3] += 5.0e-3
    f = _write(tmp_path / "eqp_asym.dat", case.k_full, case.e_full, q)
    with pytest.raises(ValueError, match=r"GATE eqp_star_spread: .*eqp_asym\.dat"
                       r": the QP shift E_QP - E_DFT of band 4 differs by "
                       r"5\.00\d meV"):
        read_eqp_energies(f, fz, (1, 5))


def test_superset_full_zone_file_refuses(case, tmp_path):
    """More blocks than the full BZ: the block map would not be a bijection."""
    extra = case.k_full[:3] + np.array([0.01, 0.0, 0.0])
    kpts = np.concatenate([case.k_full, extra])
    e = np.concatenate([case.e_full, case.e_full[:3]])
    q = np.concatenate([case.q_full, case.q_full[:3]])
    f = _write(tmp_path / "eqp_super.dat", kpts, e, q)
    with pytest.raises(ValueError, match=r"GATE eqp_full_zone_count: eqp_super"
                       r"\.dat holds 30 k-blocks but this deck's full BZ has "
                       r"27 k, so 3 blocks would go unused"):
        apply_eqp_corrections(case.e_full / RY, f, _deck(tmp_path))


def test_duplicate_block_refuses(case, tmp_path):
    kpts = case.k_full.copy()
    kpts[6] = kpts[5] + np.array([1.0, 0.0, -1.0])     # block 6 repeats k 5
    f = _write(tmp_path / "eqp_dup.dat", kpts, case.e_full, case.q_full)
    with pytest.raises(ValueError, match=r"full-BZ k 5 .* matches 2 of them"):
        apply_eqp_corrections(case.e_full / RY, f, _deck(tmp_path))


def _degenerate_pair(case, lo):
    """DFT spectrum with bands ``lo``, ``lo + 1`` 0.2 meV apart (one multiplet)."""
    e_w = case.e_wedge.copy()
    e_w[:, lo + 1] = e_w[:, lo] + 2.0e-4
    assert np.all(np.diff(e_w, axis=1) > 0)
    return np.asarray(case.unfold(case.sym, e_w))


def test_gauge_split_inside_a_multiplet_passes_and_is_applied(case, tmp_path,
                                                              capsys):
    """The gate reads multiplet means; per-band values are applied as given.

    Bands 2 and 3 form one DFT multiplet whose two shifts differ by 10 meV
    and are swapped at one star member: a per-band gate refuses this file.
    """
    e_full = _degenerate_pair(case, 2)
    shift = np.asarray(case.q_full) - np.asarray(case.e_full)[:, :NB_EQP]
    shift[:, 2], shift[:, 3] = 0.10, 0.11
    k_star = _k_off_its_parent(case.sym)
    shift[k_star, 2], shift[k_star, 3] = 0.11, 0.10
    q = e_full[:, :NB_EQP] + shift
    f = _write(tmp_path / "eqp_gauge.dat", case.k_full, e_full, q)
    _k, _d, q_file, _off = read_bgw_eqp(f)
    got = apply_eqp_corrections(e_full / RY, f, _deck(tmp_path))
    assert np.array_equal(got[:, :NB_EQP], q_file / RY)
    out = capsys.readouterr().out
    assert "10.0000 meV band by band" in out
    assert "largest spread inside one multiplet 10.0000 meV" in out


def test_multiplet_cut_by_the_file_edge_is_reported_not_gated(case, tmp_path,
                                                              capsys):
    """The file's top band (6) is degenerate with DFT band 7, outside the file."""
    e_full = _degenerate_pair(case, NB_EQP - 1)
    q = e_full[:, :NB_EQP] + (np.asarray(case.q_full)
                              - np.asarray(case.e_full)[:, :NB_EQP])
    q[_k_off_its_parent(case.sym), NB_EQP - 1] += 5.0e-3
    f = _write(tmp_path / "eqp_cut.dat", case.k_full, e_full, q)
    apply_eqp_corrections(e_full / RY, f, _deck(tmp_path))
    out = capsys.readouterr().out
    assert "WARNING eqp_cut.dat: band 6 ends inside a DFT multiplet at 27 k" in out
    assert "5.0000 meV, is reported and not gated" in out


def test_full_zone_sized_file_off_the_grid_refuses(case, tmp_path):
    kpts = case.k_full.copy()
    kpts[5] += np.array([0.01, 0.0, 0.0])
    f = _write(tmp_path / "eqp_off.dat", kpts, case.e_full, case.q_full)
    with pytest.raises(ValueError, match=r"full-BZ k 5 .* matches 0 of them"):
        apply_eqp_corrections(case.e_full / RY, f, _deck(tmp_path))

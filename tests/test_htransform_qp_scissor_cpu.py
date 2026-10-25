"""htransform's band window: the QP scissor above the eqp file, and QE crystal_b counts.

``fh_interp.qp_window_energies`` takes QP energies from the eqp file where it
has them and, when the htransform window reaches above the file's top band,
gives the bands above it their DFT energies plus the mean QP-DFT shift of the
file's conduction bands.  ``generate_kpath_from_qe_segments`` reads QE's
``K_POINTS crystal_b``: each corner's count is the number of steps to the
next corner.

Host-side readers, no device code: a CPU unit cell, exempt from the four-GPU
rule under its unit-and-CPU-cell clause.  The stars are the bcc Na hsuite
fixture's (3x3x3: 27 full k, 4 stored); the spectra are synthetic.
"""
import os

import numpy as np
import pytest

from file_io.restart_bundle import read_eqp_energies
from gw.eqp_bgw import write_bgw_eqp

HERE = os.path.dirname(os.path.abspath(__file__))
WFN = os.path.join(HERE, "hsuite", "fixture_na", "WFN.h5")
RY = 13.6056980659
NB_DFT, NB_EQP, NELEC = 8, 6, 2


@pytest.fixture(scope="module")
def case():
    from ffi import _services
    _services.ensure_on_path()
    from symmetry_maps import (reduce_full_bz_to_file_wedge,
                               unfold_file_wedge_to_full_bz)
    from wfn_loader import WfnLoader

    sym = WfnLoader(WFN).symmetry()
    rng = np.random.default_rng(20261024)
    e_w = np.sort(rng.uniform(-5.0, 10.0, (int(sym.nk_red), NB_DFT)), axis=1)
    q_w = e_w[:, :NB_EQP] + rng.uniform(-0.5, 0.5, (int(sym.nk_red), NB_EQP))
    k_full = np.asarray(sym.unfolded_kpts, dtype=np.float64)
    k_wedge = np.asarray(reduce_full_bz_to_file_wedge(sym, k_full))
    e_full = np.asarray(unfold_file_wedge_to_full_bz(sym, e_w))
    q_full = np.asarray(unfold_file_wedge_to_full_bz(sym, q_w))
    return sym, k_wedge, e_w, q_w, e_full, q_full


def _eqp(path, case, nb):
    _sym, k_wedge, e_w, q_w, _e, _q = case
    write_bgw_eqp(str(path), k_wedge, e_w[:, :nb], q_w[:, :nb], band_offset=0)
    return str(path)


def test_bands_above_the_eqp_file_get_the_mean_conduction_shift(
        case, tmp_path, caplog):
    from bandstructure.fh_interp import qp_window_energies
    sym, _k, _ew, _qw, e_full, q_full = case
    eqp = _eqp(tmp_path / "eqp1.dat", case, NB_EQP)
    with caplog.at_level("INFO", logger="lorrax.bandstructure.fh_interp"):
        got = np.asarray(qp_window_energies(
            eqp, sym, (0, NB_DFT), e_full.T / RY, NELEC))
    lines = [r.getMessage() for r in caplog.records if "QP scissor" in r.getMessage()]
    shift = float(np.mean((q_full - e_full[:, :NB_EQP])[:, NELEC:]))
    assert np.allclose(got[:NB_EQP], q_full.T / RY, atol=1e-12)
    assert np.allclose(got[NB_EQP:], (e_full[:, NB_EQP:].T + shift) / RY,
                       atol=1e-12)
    assert len(lines) == 1 and f"{shift:+.4f} eV" in lines[0]
    assert f"[{NB_EQP}, {NB_DFT})" in lines[0]


def test_window_inside_the_eqp_file_is_unchanged(case, tmp_path, caplog):
    from bandstructure.fh_interp import qp_window_energies
    sym, _k, _ew, _qw, e_full, _q = case
    eqp = _eqp(tmp_path / "eqp1.dat", case, NB_EQP)
    with caplog.at_level("INFO", logger="lorrax.bandstructure.fh_interp"):
        got = qp_window_energies(eqp, sym, (0, NB_EQP),
                                 e_full[:, :NB_EQP].T / RY, NELEC)
    assert np.array_equal(np.asarray(got),
                          np.asarray(read_eqp_energies(eqp, sym, (0, NB_EQP))))
    assert not any("QP scissor" in r.getMessage() for r in caplog.records)


def test_eqp_file_without_conduction_bands_refuses(case, tmp_path):
    from bandstructure.fh_interp import qp_window_energies
    sym, _k, _ew, _qw, e_full, _q = case
    eqp = _eqp(tmp_path / "eqp1.dat", case, NELEC)
    with pytest.raises(ValueError, match="below the first conduction band"):
        qp_window_energies(eqp, sym, (0, NB_DFT), e_full.T / RY, NELEC)


def _si_path(tmp_path):
    """Si's qe/04-bands path (L-G-X-U|K-G), QE's own counts."""
    from gw.gw_config import read_lorrax_input
    deck = tmp_path / "ht.in"
    deck.write_text(
        "[cohsex]\nsys_dim = 3\nnval = 8\nncond = 16\n\n"
        "K_POINTS crystal_b\n6\n"
        "0.5 0.5 0.5 50  ! L\n0.0 0.0 0.0 50  ! G\n0.0 0.5 0.5 50  ! X\n"
        "0.25 0.625 0.625 1  ! U\n0.375 0.75 0.375 50  ! K\n"
        "0.0 0.0 0.0 50  ! G\n")
    return read_lorrax_input(str(deck))


def test_crystal_b_counts_are_steps_to_the_next_corner(tmp_path):
    """The path QE writes has 202 points; read from the end corner it had 153."""
    from bandstructure.fh_interp import generate_kpath_from_qe_segments
    kpts, nodes, _labels = generate_kpath_from_qe_segments(
        _si_path(tmp_path), None)
    kpts = np.asarray(kpts)
    assert kpts.shape == (202, 3)
    assert nodes.tolist() == [0, 50, 100, 150, 151, 201]
    assert np.allclose(kpts[1], [0.49, 0.49, 0.49])
    assert np.allclose(kpts[151], [0.375, 0.75, 0.375])


def test_exciton_q_floor_reads_the_same_counts(tmp_path):
    try:
        from bse.exciton_bands import apply_q_per_segment
    except RuntimeError as exc:          # the host FFI library is not built
        pytest.skip(str(exc).splitlines()[0])
    params = _si_path(tmp_path)
    assert apply_q_per_segment(params, 1) == 202
    assert apply_q_per_segment(params, 2) == 203   # only U|K is raised

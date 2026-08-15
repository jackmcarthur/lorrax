"""Which k each energy row belongs to, pinned in the files themselves.

Sigma is extracted on the irreducible wedge, and ``eqp{0,1}.dat`` are
written there — one block per ``wfn.kpoints`` entry.  ``sigma_diag.dat``
and ``eqp_g0w0.dat`` stay on the full BZ, because
``bandstructure.htransform.read_eqp_energies`` takes the ``sigX=`` column
of the former as its ``--eqp-file`` and refuses any row count other than
``sym.nk_tot`` (``htransform.py:1200-1206``).

So the tree holds BOTH bases at once, in files that used to look
identical: a ``k-point N:`` block label is a POSITION, and nothing said
which k it was.  Three separate downstream comparisons then paired
LORRAX rows against BerkeleyGW's IBZ blocks by position.  On Si 4x4x4
the map is ``[0, 1, 2, 5, 6, 7, 10, 27]`` — positions 0, 1 and 2
coincide and only diverge from position 3 on, which is exactly why the
error survived casual checking every time.  The worst instance reported
a 291 meV disagreement where the true figure was 28 meV; an earlier one
manufactured a 600 meV "non-symmorphic phase bug" and cost a session.

These cells pin the two properties that make that mistake impossible to
repeat: every k-block states its crystal coordinate, and the coordinate
a block states is the one whose energies the block carries.
"""
from __future__ import annotations

import re

import numpy as np
import pytest

from file_io.sigma_output import write_eqp_g0w0, write_sigma_to_file
from gw.eqp_bgw import write_bgw_eqp


#: THE Si 4x4x4 IBZ -> full-BZ map, not a stand-in: this is the deck
#: family that produced the same bug three times.  Its first three
#: entries are the identity and the fourth is not, which is the whole
#: reason a positional join looked right.
SI444_IRR_TO_FULL = np.array([0, 1, 2, 5, 6, 7, 10, 27], dtype=np.int64)
NB = 4


def _full_bz_444() -> np.ndarray:
    """The 64 points of a 4x4x4 grid in crystal coordinates.

    Only two things matter here and both hold for any ordering: the
    points are distinct, and the wedge is a SUBSET taken through
    :data:`SI444_IRR_TO_FULL`.  The cells below never assume a
    Monkhorst-Pack convention.
    """
    g = np.arange(4) / 4.0
    return np.array([[x, y, z] for x in g for y in g for z in g],
                    dtype=np.float64)


def _wedge_444() -> np.ndarray:
    """The 8 irreducible k, as ``wfn.kpoints`` would hold them."""
    return _full_bz_444()[SI444_IRR_TO_FULL]


def _k_tagged_energies(nk: int) -> tuple[np.ndarray, np.ndarray]:
    """(E_DFT, E_QP) whose VALUE encodes the k row it belongs to.

    ``E_DFT[i, b] == 100*i + b``, so a row that ends up under the wrong
    coordinate is not merely different — it names the k it came from.
    That is what lets the pairing cells below assert *which* k drifted
    rather than only that something moved.
    """
    e_dft = (100.0 * np.arange(nk)[:, None]
             + np.arange(NB)[None, :]).astype(np.float64)
    return e_dft, e_dft + 0.5


def _read_bgw_blocks(path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a BGW ``eqp*.dat`` into (kpts (nk,3), E_DFT (nk,nb)).

    Deliberately NOT ``bse.bse_io.read_bgw_eqp``: this file is pinning
    the bytes the writer produces, so it reads them itself rather than
    inheriting a reader's tolerance for them.
    """
    kpts, energies = [], []
    lines = [ln for ln in path.read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]
    i = 0
    while i < len(lines):
        head = lines[i].split()
        assert len(head) == 4, f"not a k header: {lines[i]!r}"
        kpts.append([float(x) for x in head[:3]])
        nrow = int(head[3])
        block = []
        for j in range(1, nrow + 1):
            cols = lines[i + j].split()
            assert len(cols) == 4, f"not a band row: {lines[i + j]!r}"
            block.append(float(cols[2]))
        energies.append(block)
        i += nrow + 1
    return np.asarray(kpts), np.asarray(energies)


def _read_kcrys_blocks(path) -> tuple[np.ndarray, list[int]]:
    """Parse ``# kcrys`` coordinates and the ``k-point N:`` labels."""
    kpts, labels = [], []
    for ln in path.read_text().splitlines():
        m = re.search(r"k-point\s+(\d+)\s*:", ln)
        if m:
            labels.append(int(m.group(1)))
        elif ln.startswith("# kcrys"):
            kpts.append([float(x) for x in ln.split()[2:5]])
    return np.asarray(kpts), labels


# ===========================================================================
#  eqp{0,1}.dat — the wedge, and only the wedge
# ===========================================================================

def test_eqp_writer_emits_the_wedge_it_was_given_not_the_full_bz(tmp_path):
    """8 blocks on Si 4x4x4, carrying ``wfn.kpoints`` — never 64.

    This is the cell that goes red if the unfold is ever reintroduced on
    the eqp write path (e.g. by handing the writer ``sym.unfolded_kpts``
    and the full-BZ Sigma rows to match).
    """
    wedge, full = _wedge_444(), _full_bz_444()
    e_dft, e_qp = _k_tagged_energies(wedge.shape[0])
    out = tmp_path / "eqp0.dat"
    write_bgw_eqp(str(out), wedge, e_dft, e_qp, band_offset=0)

    kpts, _ = _read_bgw_blocks(out)
    assert kpts.shape == (8, 3), (
        f"eqp0.dat holds {kpts.shape[0]} k-blocks; Sigma was extracted on "
        f"the 8-point wedge, so anything else means the write path "
        f"unfolded to the full BZ again")
    assert kpts.shape[0] != full.shape[0]
    np.testing.assert_allclose(kpts, wedge, atol=1e-9)


def test_the_wedge_and_the_full_bz_agree_for_exactly_three_positions():
    """The reason a positional join survived casual checking, made a cell.

    If this ever goes green-by-coincidence — i.e. the two lists agree
    everywhere — the fixture has stopped modelling the deck that produced
    the bug, and the cell above would no longer be testing anything.
    """
    wedge, full = _wedge_444(), _full_bz_444()
    np.testing.assert_allclose(wedge[:3], full[:3], atol=1e-9)
    assert not np.allclose(wedge[3], full[3]), (
        "wedge and full BZ agree at position 3 — the fixture no longer "
        "reproduces the near-miss that made the positional pairing look "
        "correct, so these cells would pass on a broken writer")


def test_eqp_coordinates_and_energies_stay_in_step_under_a_permutation(
        tmp_path):
    """Row i's coordinate and row i's energies come from the SAME k.

    The energies are k-tagged, so a writer that permuted one axis and not
    the other is caught by value, not merely by shape.
    """
    wedge = _wedge_444()
    e_dft, e_qp = _k_tagged_energies(wedge.shape[0])
    perm = np.array([5, 0, 7, 2, 6, 1, 4, 3], dtype=np.int64)
    out = tmp_path / "eqp1.dat"
    write_bgw_eqp(str(out), wedge[perm], e_dft[perm], e_qp[perm],
                  band_offset=0)

    kpts, energies = _read_bgw_blocks(out)
    np.testing.assert_allclose(kpts, wedge[perm], atol=1e-9)
    # The tag says which wedge row each block came from; it must be the
    # row whose coordinate the block printed.
    tag = np.rint(energies[:, 0] / 100.0).astype(np.int64)
    np.testing.assert_array_equal(tag, perm)
    for i, k_from_tag in enumerate(tag):
        np.testing.assert_allclose(
            kpts[i], wedge[k_from_tag], atol=1e-9,
            err_msg=(f"block {i} prints coordinate {kpts[i]} but carries "
                     f"the energies of wedge k {k_from_tag} "
                     f"({wedge[k_from_tag]}) — coordinates and energies "
                     f"are out of step"))


def test_eqp_writer_refuses_a_full_bz_kpoint_list(tmp_path):
    """The structural guard: 64 coordinates cannot label 8 rows of Sigma.

    This is what makes "emit the full BZ" impossible to do by accident
    rather than merely discouraged — the mismatch is refused at the
    write call, not discovered in a comparison months later.
    """
    e_dft, e_qp = _k_tagged_energies(8)
    out = tmp_path / "eqp0.dat"
    with pytest.raises(ValueError, match="kpoints shape"):
        write_bgw_eqp(str(out), _full_bz_444(), e_dft, e_qp, band_offset=0)


# ===========================================================================
#  sigma_diag.dat / eqp_g0w0.dat — full BZ, but no longer anonymous
# ===========================================================================

def _diag(vals: np.ndarray) -> np.ndarray:
    out = np.zeros(vals.shape + (vals.shape[-1],), dtype=np.complex128)
    idx = np.arange(vals.shape[-1])
    out[:, idx, idx] = vals
    return out


def test_sigma_diag_states_a_coordinate_for_every_block(tmp_path):
    """One ``# kcrys`` line per ``k-point N:`` block, in the same order.

    Without it the file cannot say whether it is the wedge or the full
    BZ, which is the ambiguity every one of the three incidents turned
    into a wrong number.
    """
    full = _full_bz_444()
    e_dft, _ = _k_tagged_energies(full.shape[0])
    out = tmp_path / "sigma_diag.dat"
    write_sigma_to_file(_diag(e_dft), filename=str(out), kpoints_crys=full,
                        energies_dft_ev=e_dft)

    kpts, labels = _read_kcrys_blocks(out)
    assert labels == list(range(full.shape[0]))
    assert kpts.shape == full.shape, (
        f"{kpts.shape[0]} coordinate lines for {len(labels)} k-blocks — a "
        f"block without a coordinate is a block a consumer must pair by "
        f"position")
    np.testing.assert_allclose(kpts, full, atol=1e-9)


def test_eqp_g0w0_states_a_coordinate_for_every_block(tmp_path):
    full = _full_bz_444()
    e_dft, e_qp = _k_tagged_energies(full.shape[0])
    out = tmp_path / "eqp_g0w0.dat"
    write_eqp_g0w0(str(out), e_dft, e_qp.astype(np.complex128),
                   kpoints_crys=full)

    kpts, labels = _read_kcrys_blocks(out)
    assert labels == list(range(full.shape[0]))
    np.testing.assert_allclose(kpts, full, atol=1e-9)


@pytest.mark.parametrize("n_kpts, n_sigma", [(8, 64), (64, 8)])
def test_sigma_diag_refuses_a_k_list_from_the_other_basis(
        tmp_path, n_kpts, n_sigma):
    """Wedge coordinates against full-BZ Sigma (and the reverse) is refused.

    Both directions, because both are reachable: handing the writer
    ``wfn.kpoints`` while it still has the unfolded Sigma, or truncating
    Sigma and forgetting the k-list.
    """
    grid = _full_bz_444()
    kpts = grid[SI444_IRR_TO_FULL] if n_kpts == 8 else grid
    e_dft, _ = _k_tagged_energies(n_sigma)
    out = tmp_path / "sigma_diag.dat"
    with pytest.raises(ValueError, match="different k-bases"):
        write_sigma_to_file(_diag(e_dft), filename=str(out),
                            kpoints_crys=kpts, energies_dft_ev=e_dft)


# ===========================================================================
#  write_results — the seam where the two bases are actually chosen
# ===========================================================================

def test_write_results_puts_eqp_on_the_wedge_and_the_rest_on_the_full_bz(
        tmp_path):
    """THE cell: the driver's own writer, on a deck where 8 != 64.

    Everything above tests a writer in isolation, which cannot see the
    mistake that actually happened — the CALLER handing a full-BZ k-list
    and full-BZ Sigma to the eqp writer.  This drives
    ``gw_output.write_results`` end to end with ``nk_full=64`` and
    ``nk_irr=8`` and checks all three files at once, by coordinate AND
    by value: ``E_DFT`` is k-tagged, so "it took the first 8 rows" and
    "it took the right 8 rows" are distinguishable.
    """
    from common.units import RYD_TO_EV
    from gw.gw_output import GWResults, write_results

    full, wedge = _full_bz_444(), _wedge_444()
    nk_full, nb = full.shape[0], NB
    # k-tagged in Ry: row i band b carries i + b/100, so any row that
    # lands in the wrong file names the k it came from.  Kept O(10 Ry)
    # because the BGW column is ``%15.9f`` — a 6-digit integer part runs
    # into the band-index field and the writer's own verifier says so.
    e_dft_ry = (np.arange(nk_full)[:, None]
                + np.arange(nb)[None, :] / 100.0).astype(np.float64)
    zeros = np.zeros((nk_full, nb, nb), dtype=np.complex128)
    eye = np.broadcast_to(np.eye(nb), (nk_full, nb, nb)).copy()

    results = GWResults(
        sig_sx=zeros.copy(), sig_coh=zeros.copy(), sig_h=zeros.copy(),
        sig_x=zeros.copy(), E_qp_ry=e_dft_ry.copy(), U_qp=eye,
        E_dft_ry=e_dft_ry, kin_ion_ry=zeros.copy(),
        band_start=0, band_stop=nb,
        sigma_xc_at_dft_ev=np.zeros((nk_full, nb), dtype=np.float64),
    )
    write_results(
        results,
        sigma_diag_file=str(tmp_path / "sigma_diag.dat"),
        eqp0_file=str(tmp_path / "eqp0.dat"),
        eqp1_file=str(tmp_path / "eqp1.dat"),
        input_dir=str(tmp_path),
        kpoints_crys=full,
        kgrid=(4, 4, 4),
        kpoints_irr_frac=wedge,
        kirr_to_kfull=SI444_IRR_TO_FULL,
        write_qp_rotations=False,
        print_fn=lambda *a, **k: None,
    )

    # --- eqp0 / eqp1: the wedge, by coordinate and by which rows ---
    for name in ("eqp0.dat", "eqp1.dat"):
        kpts, energies = _read_bgw_blocks(tmp_path / name)
        assert kpts.shape == (8, 3), (
            f"{name} has {kpts.shape[0]} k-blocks on a deck whose wedge is "
            f"8 of 64 — write_results unfolded the eqp path again")
        np.testing.assert_allclose(kpts, wedge, atol=1e-9)
        np.testing.assert_allclose(
            energies, e_dft_ry[SI444_IRR_TO_FULL] * RYD_TO_EV, atol=1e-6,
            err_msg=(f"{name} carries the wrong 8 rows: the tags say it "
                     f"subset by position, not through kirr_to_kfull "
                     f"({SI444_IRR_TO_FULL.tolist()})"))

    # --- sigma_diag / eqp_g0w0: the full BZ, each block naming its k ---
    for name in ("sigma_diag.dat", "eqp_g0w0.dat"):
        kpts, labels = _read_kcrys_blocks(tmp_path / name)
        assert labels == list(range(nk_full)), (
            f"{name} has {len(labels)} k-blocks; htransform requires all "
            f"{nk_full} (htransform.py:1200-1206)")
        np.testing.assert_allclose(kpts, full, atol=1e-9)


def test_the_coordinate_line_is_invisible_to_the_block_parsers(tmp_path):
    """The coordinate must not land on the header or on a data row.

    ``htransform.read_eqp_energies`` anchors its header regex with ``$``
    (``htransform.py:1153``) and finds band rows by searching for an
    ``n=<digits>`` token, and six parsers across the tree discriminate
    header-from-body by "exactly four whitespace tokens".  Appending the
    coordinate to ``k-point N:`` or to a band row breaks those silently
    or fatally.  A separate ``#`` line is invisible to all of them —
    this cell is what stops someone "tidying" it onto the header.
    """
    full = _full_bz_444()
    e_dft, _ = _k_tagged_energies(full.shape[0])
    out = tmp_path / "sigma_diag.dat"
    write_sigma_to_file(_diag(e_dft), filename=str(out), kpoints_crys=full,
                        energies_dft_ev=e_dft)

    # The literal regexes htransform compiles, so this pin tracks it.
    ht_header = re.compile(r"^\s*k-point\s+(\d+)\s*:\s*$")
    ht_band = re.compile(r"n\s*=\s*(\d+)")

    kcrys = [ln for ln in out.read_text().splitlines()
             if ln.startswith("# kcrys")]
    assert kcrys, "no coordinate lines were written"
    for ln in kcrys:
        assert ht_band.search(ln) is None, (
            f"the coordinate line {ln!r} contains an 'n=<digits>' token, so "
            f"htransform would read it as a band row")
        assert not ln.startswith("n="), ln
        assert len(ln.split()) != 4, (
            f"the coordinate line {ln!r} has exactly four tokens, the shape "
            f"six parsers use to mean 'k header' or 'band row'")

    # ...and every k-point header still matches htransform's ANCHORED
    # regex, i.e. the coordinate did not get appended to it.
    headers = [ln for ln in out.read_text().splitlines()
               if "k-point" in ln]
    assert headers
    for ln in headers:
        assert ht_header.match(ln) is not None, (
            f"header {ln!r} no longer matches htransform's anchored regex — "
            f"the --eqp-file hop from QSGW is broken")

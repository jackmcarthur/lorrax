"""eigenvalues*_noeh.dat in BerkeleyGW's format (``BSE/absp_io.f90``
``write_eigenvalues_noeh``).

The header lines and two rows are those of a BerkeleyGW 4.0 Si 4^3 singlet run
(``eigenvalues_b1_noeh.dat``, 64 k x 4 c x 4 v, ``use_momentum``).  Checked: the
header byte for byte, the ``(4i6, 6e16.8)`` row layout, the loop order
(k, c, v), the valence axis counted down from the VBM, and the file names.
"""

import numpy as np

from bse.absorption_common import bgw_noeh_lines, write_bgw_eigenvalues_noeh

BGW_HEADER = [
    "# neig  =     1024",
    "# vol   =  0.172868583E+05",
    "# nspin, nspinor, nkpt, ncb, nvb =        1       1      64       4       4",
    "#   ik    ic    iv    is         ec (eV)         ev (eV)        eig (eV)"
    "   abs(dipole)^2      Re(dipole)      Im(dipole)",
]
#: (ik, ic, iv, is, ec, ev, eig, |d|^2, Re d, Im d) of rows 1 and 4.
BGW_ROWS = [
    (1, 1, 1, 1, 8.8203133, 6.2797338, 2.5405794, 8.9128003, 2.4463907, 1.7111320),
    (1, 1, 4, 1, 8.8203133, -5.6985719, 14.518885, 0.063200583, -0.050306884, 0.24631240),
]


def _parse(row):
    assert len(row) == 4 * 6 + 6 * 16
    ints = [int(row[6 * i:6 * i + 6]) for i in range(4)]
    return ints + [float(row[24 + 16 * i:40 + 16 * i]) for i in range(6)]


def test_noeh_header_and_rows_match_berkeleygw(tmp_path):
    nk, nc, nv = 64, 4, 4
    ec = np.zeros((nk, nc))
    ev = np.zeros((nk, nv))
    d = np.zeros((nk, nc, nv), complex)
    ec[0, 0] = 8.8203133
    ev[0, nv - 1], ev[0, 0] = 6.2797338, -5.6985719          # LORRAX: valence ascending
    d[0, 0, nv - 1] = 2.4463907 + 1.7111320j
    d[0, 0, 0] = -0.050306884 + 0.24631240j
    lines = bgw_noeh_lines(ec, ev, d, n_spin=1, n_spinor=1, vol_supercell=17286.8583)
    assert lines[:4] == BGW_HEADER
    assert len(lines) == 4 + nk * nc * nv
    for want, row in zip(BGW_ROWS, (lines[4], lines[7])):
        got = _parse(row)
        assert got[:4] == list(want[:4])
        np.testing.assert_allclose(got[4:], want[4:], rtol=2e-7, atol=0)
    assert _parse(lines[8])[:3] == [1, 2, 1] and _parse(lines[4 + nc * nv])[:3] == [2, 1, 1]

    d3 = np.stack([d, 2 * d, 0 * d])
    names = write_bgw_eigenvalues_noeh(tmp_path, ec, ev, d3, n_spin=1, n_spinor=1,
                                       vol_supercell=17286.8583)
    assert [p.split("/")[-1] for p in names] == [
        "eigenvalues_b1_noeh.dat", "eigenvalues_b2_noeh.dat", "eigenvalues_b3_noeh.dat"]
    b2 = open(names[1]).read().splitlines()
    np.testing.assert_allclose(_parse(b2[4])[8:], [2 * 2.4463907, 2 * 1.7111320], rtol=1e-7)
    one = write_bgw_eigenvalues_noeh(tmp_path, ec, ev, d3, n_spin=1, n_spinor=1,
                                     vol_supercell=1.0, directions=np.array([[1.0, 0, 0]]),
                                     labels=("e",))
    assert [p.split("/")[-1] for p in one] == ["eigenvalues_noeh.dat"]

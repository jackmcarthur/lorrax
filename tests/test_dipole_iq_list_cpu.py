"""``psp.get_dipole_mtxels --with-finite-q``: the default ``--iq-list`` is the stored q.

The finite-q loop indexes ``sym.kq_map[:, iq]``, one column per stored
(reduced-BZ) k. The default once ran to the full-zone count and raised
IndexError on a symmetry-reduced WFN (H2-: 9 stored of 25). The P4 script
``tests/test_finite_q_k_tiles.py`` runs the default end to end.
"""
import pytest


def test_default_is_every_stored_q_and_past_them_refuses():
    gd = pytest.importorskip("psp.get_dipole_mtxels")
    assert gd.finite_q_iq_list(None, 9) == list(range(9))
    assert gd.finite_q_iq_list([0, 4, 8], 9) == [0, 4, 8]
    for bad in ([9], [-1], [3, 25]):
        with pytest.raises(ValueError, match="dipole_finite_q_iq_list"):
            gd.finite_q_iq_list(bad, 9)

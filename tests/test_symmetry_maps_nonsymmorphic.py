import numpy as np

from src.common.symmetry_maps import SymMaps


class _FakeWFN:
    def __init__(self):
        self.kpoints = np.asarray([[0.25, 0.0, 0.0]], dtype=np.float64)
        self.translations = np.asarray([[np.pi, 0.0, 0.0]], dtype=np.float64)
        self.ntran = 1
        self.nkpts = 1
        self.ngk = np.asarray([3], dtype=np.int32)
        self._gvecs = np.asarray(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
            ],
            dtype=np.int32,
        )
        self._coeffs = np.asarray(
            [
                [
                    [1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j],
                    [10.0 + 0.0j, 20.0 + 0.0j, 30.0 + 0.0j],
                ],
                [
                    [4.0 + 0.0j, 5.0 + 0.0j, 6.0 + 0.0j],
                    [40.0 + 0.0j, 50.0 + 0.0j, 60.0 + 0.0j],
                ],
            ],
            dtype=np.complex128,
        )

    def get_gvec_nk(self, ik):
        assert ik == 0
        return self._gvecs

    def get_cnk(self, ik, ib):
        assert ik == 0
        return self._coeffs[ib]

    def get_cnk_batch(self, ik, band_indices):
        assert ik == 0
        return self._coeffs[np.asarray(band_indices, dtype=np.int64)]


def _make_symmaps():
    sym = SymMaps.__new__(SymMaps)
    c2z = np.asarray(
        [
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1],
        ],
        dtype=np.int32,
    )
    sym.irk_sym_map = np.asarray([0], dtype=np.int32)
    sym.irk_to_k_map = np.asarray([0], dtype=np.int32)
    sym.sym_matrices = c2z[None, :, :]
    sym.sym_mats_k = c2z[None, :, :]
    sym.unfolded_kpts = np.asarray([[0.75, 0.0, 0.0]], dtype=np.float64)
    sym.U_spinor = np.eye(2, dtype=np.complex128)[None, :, :]
    return sym


def test_get_gvecs_kfull_uses_bgw_umklapp_convention():
    wfn = _FakeWFN()
    sym = _make_symmaps()

    gvecs_full = sym.get_gvecs_kfull(wfn, 0)

    np.testing.assert_array_equal(
        gvecs_full,
        np.asarray(
            [
                [-1, 0, 0],
                [-2, 0, 0],
                [-1, -1, 0],
            ],
            dtype=np.int32,
        ),
    )


def test_get_cnk_fullzone_batch_applies_nonsymmorphic_phase():
    wfn = _FakeWFN()
    sym = _make_symmaps()

    cnk_full = sym.get_cnk_fullzone_batch(wfn, np.asarray([0, 1]), 0)

    expected_phase = np.asarray([1.0, -1.0, 1.0], dtype=np.complex128)
    expected = wfn.get_cnk_batch(0, [0, 1]) * expected_phase[None, None, :]
    np.testing.assert_allclose(cnk_full, expected, rtol=0.0, atol=1e-12)


def test_get_cnk_fullzone_matches_batch_path():
    wfn = _FakeWFN()
    sym = _make_symmaps()

    scalar = sym.get_cnk_fullzone(wfn, 1, 0)
    batch = sym.get_cnk_fullzone_batch(wfn, np.asarray([1]), 0)[0]
    np.testing.assert_allclose(scalar, batch, rtol=0.0, atol=1e-12)

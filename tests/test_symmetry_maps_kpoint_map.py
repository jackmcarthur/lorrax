import numpy as np

from src.common.symmetry_maps import SymMaps


class _FakeWFN2x2x2:
    def __init__(self):
        self.kgrid = np.asarray([2, 2, 2], dtype=np.int32)
        self.shift = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
        self.kpoints = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5],
                [0.0, 0.5, 0.5],
                [0.5, 0.5, 0.5],
            ],
            dtype=np.float64,
        )
        self.nkpts = self.kpoints.shape[0]

        # Identity in real-space metric is enough for the spinor helper path.
        self.bvec = np.eye(3, dtype=np.float64)
        self.translations = np.zeros((16, 3), dtype=np.float64)

        identity = np.eye(3, dtype=np.int32)
        swap_xz = np.asarray([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.int32)
        swap_yz = np.asarray([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.int32)
        cycle_xyz = np.asarray([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.int32)

        # Place the useful symmetries at indices >= nk_full to reproduce the
        # original bug where symmetry ids were later misread as k-point ids.
        self.sym_matrices = np.stack(
            [
                identity,
                identity,
                identity,
                identity,
                identity,
                identity,
                identity,
                identity,
                swap_xz,
                swap_yz,
                cycle_xyz,
                cycle_xyz,
                cycle_xyz,
                identity,
                identity,
                identity,
            ],
            axis=0,
        )
        self.ntran = self.sym_matrices.shape[0]


def test_kpoint_map_stores_irreducible_k_indices_not_symmetry_ids():
    wfn = _FakeWFN2x2x2()
    sym = SymMaps(wfn)

    np.testing.assert_array_equal(
        sym.kpoint_map_ibz_ids,
        np.asarray([0, 1, 1, 2, 1, 2, 2, 3], dtype=np.int32),
    )
    assert np.max(sym.kpoint_map_ibz_ids) < wfn.nkpts

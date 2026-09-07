"""A torn current-family pair refuses before any tensor transport."""
import h5py
import numpy as np
import pytest

from file_io.restart_bundle import read_restart_state_from_h5


@pytest.mark.parametrize("low_mem_bands", [False, True])
@pytest.mark.parametrize("missing", ["psi_parent_y_transverse", "psi_parent_y_transverse_mun"])
def test_torn_current_faces_refuse_both_layouts(tmp_path, low_mem_bands, missing):
    path = tmp_path / "torn.h5"
    with h5py.File(path, "w") as f:
        f["psi_parent_y"] = np.zeros((1, 2, 4, 4), complex)
        f["psi_parent_y_mun"] = np.zeros((1, 4, 4, 2), complex)
        f["psi_parent_k_rows"] = [0]
        f["band_window"] = [0, 0, 1, 2, 2]
        f["band_window_schema"] = 2
        f["enk_full"] = np.zeros((1, 2))
        f["kgrid"] = [1, 1, 1]
        for name, shape in (("psi_parent_y_transverse", (1, 2, 4, 4)),
                            ("psi_parent_y_transverse_mun", (1, 4, 4, 2))):
            if name != missing:
                f[name] = np.zeros(shape, complex)
    with pytest.raises(ValueError, match="torn transverse parent faces"):
        read_restart_state_from_h5(path, None, low_mem_bands=low_mem_bands)

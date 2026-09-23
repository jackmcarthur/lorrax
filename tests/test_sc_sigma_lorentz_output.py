"""A stopped SC run retains its Lorentz matrices and QP basis."""

def test_sc_lorentz_map_roundtrip(tmp_path):
    import h5py
    import numpy as np

    from file_io.sigma_output import (read_sc_sigma_lorentz_h5,
                                      write_sc_sigma_lorentz_h5)

    parts = np.zeros((3, 2, 3, 3), dtype=np.complex128)
    parts[1, 1, 0, 2] = 0.012 + 0.003j
    rotation = np.broadcast_to(np.eye(3), (2, 3, 3)).copy()
    kpoints = np.array([[0., 0., 0.], [.5, .5, 0.]])
    path = tmp_path / "sigma_lorentz_iter0001.h5"

    write_sc_sigma_lorentz_h5(
        path, parts, rotation, kpoints, call_index=1, role="trial",
        kset="star_wedge", band_start_1based=9)

    got_parts, got_rotation, got_kpoints = read_sc_sigma_lorentz_h5(path)
    np.testing.assert_array_equal(got_parts, parts)
    np.testing.assert_array_equal(got_rotation, rotation)
    np.testing.assert_array_equal(got_kpoints, kpoints)
    with h5py.File(path) as h5:
        assert h5.attrs["sectors"].tolist() == ["CC", "CT+TC", "TT"]
        assert h5.attrs["basis"] == "input_qp"
        assert h5.attrs["kset"] == "star_wedge"
        assert int(h5.attrs["band_start_1based"]) == 9
        assert int(h5["lorrax_io_committed"][0]) == 1

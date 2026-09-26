"""Physical reciprocal distance for a QE crystal_b path."""

from types import SimpleNamespace

import numpy as np

def test_bcc_gamma_h_distance_uses_one_reciprocal_scale():
    from bandstructure.fh_interp import initialize_kpath

    alat_bohr = 5.42
    wfn = SimpleNamespace(
        bvec=np.array([[1.0, 0.0, 1.0], [-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]]),
        blat=2.0 * np.pi / alat_bohr,
    )
    params = {"kpoints_crystal_b": {"segments": [
        {"k": [0.0, 0.0, 0.0], "n": 1, "label": "Gamma"},
        {"k": [0.5, 0.5, -0.5], "n": 1, "label": "H"},
    ]}}

    kpath, s, nodes, labels, gamma = initialize_kpath(wfn, params)

    np.testing.assert_allclose(np.asarray(kpath)[nodes],
                               [[0.0, 0.0, 0.0], [0.5, 0.5, -0.5]])
    np.testing.assert_allclose(s[nodes], [0.0, 2.0 * np.pi / alat_bohr])
    assert labels == ["Γ", "H"]
    assert gamma == [0]

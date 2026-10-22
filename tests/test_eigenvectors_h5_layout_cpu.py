"""eigenvectors.h5 in BerkeleyGW's layout (``Common/evecs.f90``), as its reader
and a post-processing script read it.

A 2x1x1 grid, 2 conduction x 3 valence bands with one mesh-pad row on each band
axis, two states.  Checked: the group/dataset tree (no eps_header/bse_header),
int32 scalars, ``kpts`` (nk, 3), eigenvectors (nQ, nevecs, nk, nc, nv, ns, 2)
with the valence axis counted down from the VBM, eigenvalues in eV, the four
eigenvector datasets under the TDA and the (X, Y) -> right/left split of a
full-BSE pair, ``spin_kernel``, and the WFN's ``mf_header`` (k list replaced on
a kgrid_fine grid).
"""

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("jax")

from bse.bse_window import write_eigenvectors_stream  # noqa: E402

NC, NV, NK, NW = 2, 3, 2, 2
RY = 13.6056980659


def _wfn(path):
    with h5py.File(path, "w") as f:
        f["mf_header/versionnumber"] = np.int32(1)
        f["mf_header/crystal/celvol"] = 270.1
        k = f.create_group("mf_header/kpoints")
        k["nrk"], k["rk"] = np.int32(2), np.array([[0.0, 0, 0], [0.5, 0, 0]])
        k["el"], k["kgrid"] = np.zeros((1, 2, 8)), np.array([2, 1, 1], np.int32)
        f["wfns/coeffs"] = np.zeros(4)                       # not copied


def _amplitudes(rng, n_comp):
    """(NW, n_comp, NC+1, NV+1, NK): the internal layout, valence ascending,
    the pad row/column (top of each band axis) exactly zero."""
    a = np.zeros((NW, n_comp, NC + 1, NV + 1, NK), complex)
    a[:, :, :NC, :NV, :] = (rng.standard_normal((NW, n_comp, NC, NV, NK))
                            + 1j * rng.standard_normal((NW, n_comp, NC, NV, NK)))
    return a


def test_eigenvectors_h5_is_berkeleygw_layout(tmp_path):
    rng = np.random.default_rng(0)
    wfn = str(tmp_path / "WFN.h5")
    _wfn(wfn)
    evals_ry = np.array([0.20, 0.21])

    tda = _amplitudes(rng, 1)
    out = str(tmp_path / "tda.h5")
    write_eigenvectors_stream(out, evals_ry, tda, NV, NC, 2, 1, 1, NW, use_tda=True,
                              spin_kernel=1, wfn_path=wfn, wfn_grid=True)
    with h5py.File(out, "r") as f:
        assert sorted(f) == ["exciton_data", "exciton_header", "mf_header"]
        assert "wfns" not in f and f["mf_header/kpoints/el"].shape == (1, 2, 8)
        ints = ["exciton_header/versionnumber", "exciton_header/flavor",
                "exciton_header/kpoints/nk", "exciton_header/kpoints/nQ"] + [
            f"exciton_header/params/{n}" for n in (
                "bse_hamiltonian_size", "evec_sz", "nevecs", "ns", "nc", "nv",
                "spin_kernel", "use_tda")]
        assert all(f[n].dtype == np.int32 and f[n].shape == () for n in ints)
        assert [int(f[n][()]) for n in ints[4:]] == [NK * NC * NV, NK * NC * NV, NW,
                                                     1, NC, NV, 1, 1]
        assert f["exciton_header/kpoints/kpts"].shape == (NK, 3)
        assert np.allclose(f["exciton_header/kpoints/kpts"][1], [0.5, 0, 0])
        assert f["exciton_header/kpoints/exciton_Q_shifts"].shape == (1, 3)
        assert np.allclose(f["exciton_data/eigenvalues"][()], evals_ry * RY)
        shape = (1, NW, NK, NC, NV, 1, 2)
        names = ("eigenvectors", "eigenvectors_left", "eigenvectors_deexcitation",
                 "eigenvectors_deexcitation_left")
        assert all(f[f"exciton_data/{n}"].shape == shape and
                   f[f"exciton_data/{n}"].dtype == np.float64 for n in names)
        A = f["exciton_data/eigenvectors"][()]
        assert not np.any(f["exciton_data/eigenvectors_left"][()])
    A = A[..., 0] + 1j * A[..., 1]
    # BerkeleyGW's v = 1 is the VBM: internal valence index NV-1.
    want = np.transpose(tda[:, 0, :NC, :NV, :], (0, 3, 1, 2))[:, :, :, ::-1]
    assert np.array_equal(A[0, :, :, :, :, 0], want)

    full = _amplitudes(rng, 2)
    out = str(tmp_path / "full.h5")
    write_eigenvectors_stream(out, evals_ry, full, NV, NC, 2, 1, 1, NW, use_tda=False,
                              spin_kernel=0, wfn_path=wfn, wfn_grid=False,
                              k_shift=(0.5, 0.0, 0.0))
    with h5py.File(out, "r") as f:
        d = {n: f[f"exciton_data/{n}"][()] for n in names}
        assert int(f["exciton_header/params/evec_sz"][()]) == 2 * NK * NC * NV
        assert int(f["exciton_header/params/spin_kernel"][()]) == 0
        kp = f["mf_header/kpoints"]
        assert "el" not in kp and int(kp["nrk"][()]) == NK
        assert np.allclose(kp["rk"][()], f["exciton_header/kpoints/kpts"][()])
        assert np.allclose(kp["rk"][()][:, 0], [0.25, 0.75])
    assert np.array_equal(d["eigenvectors"], d["eigenvectors_left"])
    assert np.array_equal(d["eigenvectors_deexcitation"], -d["eigenvectors_deexcitation_left"])
    Y = d["eigenvectors_deexcitation"][..., 0] + 1j * d["eigenvectors_deexcitation"][..., 1]
    want = np.transpose(full[:, 1, :NC, :NV, :], (0, 3, 1, 2))[:, :, :, ::-1]
    assert np.array_equal(Y[0, :, :, :, :, 0], want)

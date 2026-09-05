"""Authentication and cheap public-door checks on the core fixture family."""
from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from symmetry_maps import verify_centroid_orbit_closure
from wfn_loader import WfnLoader
from zeta_loader import ZetaLoader, probe_zeta_file

from .fixtures.stamp_references import expected


@pytest.mark.parametrize("label", ("A", "A-prime", "A-cubic", "B"))
def test_fixture_bytes_match_the_portable_stamp(core_fixtures, label):
    stamp = core_fixtures / label / "PROVENANCE.json"
    assert json.loads(stamp.read_text(encoding="utf-8")) == expected(label)


@pytest.mark.parametrize(
    "label,nk,nb,ns,kgrid,ntran,trs",
    (
        ("A", 5, 8, 1, (3, 3, 1), 1, True),
        ("A-prime", 9, 7, 2, (3, 3, 1), 1, False),
        ("A-cubic", 3, 8, 1, (2, 2, 2), 48, True),
        ("B", 1, 8, 1, (1, 1, 1), 2, True),
    ),
)
def test_wfn_loader_and_typed_symmetry_match_the_stamped_shape(
        core_fixtures, label, nk, nb, ns, kgrid, ntran, trs):
    root = core_fixtures / label
    with WfnLoader(
            root / "WFN.h5", backend="eager",
            qe_schema=root / "data-file-schema.xml") as loader:
        assert (loader.nkpts, loader.nbands, loader.nspinor) == (nk, nb, ns)
        assert tuple(loader.kgrid) == kgrid
        assert int(loader.ntran) == ntran
        sym = loader.symmetry()
        assert sym.operation_typing_source == "qe-schema"
        assert sym.trs_allowed is trs
        psi = np.asarray(loader.load(bands=(0, min(3, nb)), k="ibz"))
        assert psi.shape[:3] == (nk, min(3, nb), ns)
        ngk = loader.ngk_valid(k="ibz")
        for ik, nvalid in enumerate(ngk):
            assert not psi[ik, :, :, int(nvalid):].any()


@pytest.mark.parametrize(
    "label,centroids,ntran,closed",
    (
        ("A", "centroids_frac_21.txt", 1, True),
        ("A-prime", "centroids_frac_21_current.txt", 1, True),
        ("A-cubic", "centroids_frac_48.txt", 48, True),
        ("A-cubic", "centroids_frac_23_literal.txt", 48, False),
        ("B", "centroids_frac_13.txt", 2, True),
    ),
)
def test_orbit_and_literal_centroid_verdicts(
        core_fixtures, label, centroids, ntran, closed):
    root = core_fixtures / label
    with WfnLoader(root / "WFN.h5", backend="eager") as loader:
        points = np.loadtxt(root / centroids)
        verdict = verify_centroid_orbit_closure(
            points, loader.sym_matrices[:ntran],
            tnp=loader.translations[:ntran], fft_grid=loader.fft_grid,
            tol=1.1e-5,
        )
    assert verdict.closed is closed
    assert verdict.n_centroids == len(points)
    if closed:
        assert verdict.worst_residual <= 1.1e-5
    else:
        assert verdict.worst_residual > 1e-2


@pytest.mark.parametrize(
    "label,nmu,nq,q_layout",
    (("A", 21, 5, "ibz"), ("B", 13, 1, "full_bz")),
)
def test_zeta_reference_exposes_rank_truncation_and_qirr_slice(
        core_fixtures, label, nmu, nq, q_layout):
    path = core_fixtures / label / "tmp" / "zeta_q.h5"
    probe = probe_zeta_file(path)
    assert probe.readable and probe.zeta_done
    assert (probe.dataset_name, probe.mu_extent) == ("zeta_q_G", nmu)
    with ZetaLoader(path) as loader:
        assert loader.n_rmu == nmu
        assert loader.q_layout == q_layout
        qslice = np.asarray(loader.read_zeta_G_local(slice(0, nq)))
        assert qslice.shape[:2] == (nq, nmu)
        provenance = json.loads(loader.fit_provenance)
        assert provenance["charge_zeta_solve"] == "rank_truncate"
        assert provenance["write_ibz_only"] is True
        if nq > 1:
            with pytest.raises(NotImplementedError, match="unfold_v_q"):
                loader.load(q="full_bz")


def test_cached_production_carriers_are_complete(core_fixtures):
    """Authenticate the restart, W-sample, MPA-fit and Sigma carriers."""
    restart = core_fixtures / "A" / "tmp" / "isdf_tensors_21.h5"
    with h5py.File(restart, "r") as h5:
        assert h5["V_qmunu"].shape == h5["W0_qmunu"].shape == (5, 21, 21)
        assert h5["psi_full_y"].shape == (9, 7, 1, 21)
        assert int(h5["n_rmu_logical"][()]) == 21
        assert int(h5["restart_format_version"][()]) >= 1

    mpa = core_fixtures / "B" / "tmp" / "mpa"
    with h5py.File(mpa / "mpa_samples_oneshot.h5", "r") as h5:
        assert h5["Wc_qmunu_z"].shape == (4, 1, 13, 13)
        assert np.asarray(h5["Wc_qmunu_z__mpa/data_ready"]).all()
    with h5py.File(mpa / "mpa_fit_oneshot.h5", "r") as h5:
        assert h5["Omega_p"].shape == h5["B_p"].shape == (2, 1, 13, 13)
        assert bool(h5.attrs["mpa_fit_complete"])
        assert np.asarray(h5["__mpafit/blocks_done"]).all()
    with h5py.File(core_fixtures / "B" / "mpa_sigma.h5", "r") as h5:
        assert h5["sigma_total_kij_ev"].shape == (25, 1, 3, 3)


def test_corrupt_zeta_file_is_a_nonraising_probe_refusal(tmp_path):
    path = tmp_path / "corrupt-zeta.h5"
    path.write_bytes(b"not an HDF5 file")
    probe = probe_zeta_file(path)
    assert probe.exists and not probe.readable
    assert probe.error and "OSError" in probe.error

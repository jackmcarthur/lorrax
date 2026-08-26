"""Contract for htransform's rank-zero scientific report."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from bandstructure.production_report import HTransformProductionReport


def _runtime():
    return SimpleNamespace(process_index=0, facts={
        "process_count": 4, "n_devices": 4, "n_local_devices": 1,
        "backend": "gpu", "device_kind": "A100", "mesh_shape": (2, 2),
        "threads": {"affinity": 16, "OMP_NUM_THREADS": "8"},
        "jax_version": "0.9.1", "jaxlib_version": "0.9.1", "x64": True,
    })


def test_htransform_report_names_spaces_path_progress_and_files(tmp_path):
    output = []
    path = tmp_path / "htransform.out"
    report = HTransformProductionReport(
        str(path), runtime=_runtime(), debug=False, stdout=output.append)
    report.begin(input_file="ht.in", output_file="bandstructure.dat",
                 energy_source="DFT eigenvalues from the WFN")
    report.architecture()
    wfn = SimpleNamespace(
        backend="phdf5", num_electrons=8.0, nelec=4,
        density_symmetry=None, kgrid=np.array([2, 2, 2]),
        shift=np.zeros(3), kpoints=np.array([[0.0, 0.0, 0.0]]),
        kweights=np.ones(1),
    )
    sym = SimpleNamespace(
        Rinv_grid=np.eye(3, dtype=int)[None], translations=np.zeros((1, 3)),
        trs_allowed=True, nk_tot=8, nk_red=1,
    )
    params = {"eigh_backend": "auto", "htransform_rank_multiplier": 0.0}
    report.environment(
        params=params, wfn=wfn, eigh_backend="auto", batched_route="auto",
        diagnostics_policy="auto", diagnostics_enabled=False)
    centroids = SimpleNamespace(closure=SimpleNamespace(
        closed=False, n_sym=2, n_violating=1, n_centroids=20,
        tol=1.0e-5, worst_residual=0.25, worst_op=1))
    report.sampling(wfn=wfn, sym=sym, centroids=centroids)
    result = {
        "band_start": 2, "nb_keep": 4, "nb_fit": 6,
        "n_guard_bands": 2, "nk_total": 8,
        "path_range": (-0.2, 0.4),
        "kpath_data": (
            np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]), None,
            np.array([0, 1]), ["Γ", "X"], [0]),
    }
    report.interpolation_space(
        params=params, wfn=wfn,
        meta=SimpleNamespace(n_rmu=20, n_rmu_padded=20), result=result,
        enk_sigma_ry=np.arange(48, dtype=float).reshape(6, 8) / 100.0,
        ctilde=np.zeros((8, 6, 12)), centroid_file="centroids_frac.txt",
        energy_source="DFT eigenvalues from the WFN")
    report.path_summary(result=result)
    report.progress("Started Hamiltonian interpolation at 12:00:00.")
    report.timings([
        {"name": "initialize_wfns", "inclusive": 2.0},
        {"name": "ht.build_fH_R", "inclusive": 1.0},
        {"name": "ht.kpath_loop", "inclusive": 3.0},
    ], wall=10.0)
    report.files([
        ("interpolated bands", "written", str(tmp_path / "bandstructure.dat")),
        ("calculation report", "written", str(path)),
    ])
    report.finish()

    text = path.read_text(encoding="utf-8")
    assert text == "\n".join(output) + "\n"
    assert "Returned bands : 3-6" in text
    assert "Fitted bands   : 3-8 (2 upper guard bands)" in text
    assert "Centroid sites : 20 logical; 20 mesh-padded" in text
    assert "Centroid orbit: NOT CLOSED : 1/2 spatial operations" in text
    assert "worst residual 2.50000e-01 at S02" in text
    assert "Galerkin basis : rank 12" in text
    assert "N01    Γ  k=( 0.00000  0.00000  0.00000)" in text
    assert "Gram eigensolve: auto (other choices:" in text
    assert "-> auto" not in text
    assert "auto (other choices:" in text
    assert "Started Hamiltonian interpolation" in text
    assert "OUTPUT FILES AND INPUTS" in text
    assert "calculation report" not in text
    assert "HDF5" not in text and "h5py" not in text


def test_htransform_runtime_startup_uses_the_one_debug_stream():
    source = (Path(__file__).parents[1] / "src" / "bandstructure" /
              "htransform.py").read_text(encoding="utf8")
    assert "initialize_communicator_stack(print_fn=debug_print)" in source
    assert "RUNTIME = initialize_communicator_stack()" not in source

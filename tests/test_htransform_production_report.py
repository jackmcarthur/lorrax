"""Contract for htransform's rank-zero scientific report."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from bandstructure.production_report import HTransformProductionReport
from common import rank_criterion, spectral_closure


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
        params=params, wfn=wfn,
        gram_plan=SimpleNamespace(
            requested="auto", backend="native", is_native=True, n=48,
            mesh=SimpleNamespace(shape={"x": 2, "y": 2})),
        fine_eigh_backend="auto", fine_enabled=False, batched_route="auto",
        diagnostics_policy="auto", diagnostics_enabled=False)
    centroids = SimpleNamespace(closure=SimpleNamespace(
        closed=False, n_sym=2, n_violating=1, n_centroids=20,
        tol=1.0e-5, worst_residual=0.25, worst_op=1))
    report.sampling(wfn=wfn, sym=sym, centroids=centroids)
    result = {
        "band_start": 2, "nb_keep": 4, "nb_fit": 6,
        "n_guard_bands": 2, "nk_total": 8,
        "path_range": (-0.2, 0.4),
        "gamma_energy_checkpoint": {
            "max_abs_ev": 1.2e-3, "rms_ev": 4.0e-4, "n_bands": 4,
        },
        "f_transform": {
            "a_ry": 0.4, "n": 3.0, "shift_ry": 0.47,
            "scale_band_local": 5, "shoulder_band_local": 5,
        },
        "kpath_data": (
            np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]), None,
            np.array([0, 1]), ["Γ", "X"], [0]),
    }
    report.interpolation_space(
        params=params, wfn=wfn,
        meta=SimpleNamespace(n_rmu=20, n_rmu_padded=20), result=result,
        enk_sigma_ry=np.arange(48, dtype=float).reshape(6, 8) / 100.0,
        ctilde=np.zeros((8, 6, 12)), centroid_file="centroids_frac.txt",
        energy_source="DFT eigenvalues from the WFN",
        centroids=SimpleNamespace(n_rmu=20, source_n_rmu=20))
    spectrum = [1.0, 0.5, 0.1, 0.01]
    numerical_report = rank_criterion.rank_report(
        spectrum, 0.05, rank_used=3, rank_ceiling=4)
    report.spectral_compression({
        "stacked_states": 48, "site_spin_columns": 40,
        "structural_ceiling": 40, "numerical_rank": 3,
        "retained_rank": 3, "carried_rank": 4, "null_padding": 1,
        "rank_multiplier": 0.0, "numerical_report": numerical_report,
        "compression": rank_criterion.singular_value_compression(spectrum, 3),
        "numerical_closure": spectral_closure.cluster_at_cut(spectrum, 3),
        "model_closure": None,
    })
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
    assert "Centroid sites : 20 requested from the full table; 20 mesh-padded" in text
    assert "Centroid orbit: NOT CLOSED : 1/2 spatial operations" in text
    assert "worst residual 2.50000e-01 at S02" in text
    assert "Galerkin basis : rank 12" in text
    assert "Galerkin basis : 48 stacked states x 40 site-spin samples" in text
    assert "Model order     : full numerical span; retained rank 3" in text
    assert "Sampled-psi tail:" in text
    assert "not a band-energy error bound" in text
    assert "rank 4 = 3 physical + 1 exact-null mesh pad" in text
    assert "Guard buffer   :" in text
    assert "f-transform    : a=0.40000 Ry (4 x bandwidth of band 8); n=3.00" in text
    assert "Zero shoulder  : max E(band 8) = 0.47000 Ry" in text
    assert "N01    Γ  k=( 0.00000  0.00000  0.00000)" in text
    assert "L01  Γ -> X: 1 intervals; 2 endpoint-inclusive points" in text
    assert "Gram eigensolve: auto (other choices:" in text
    assert "-> native; replicated native JAX; n=48" in text
    assert "Fine-k eigensolve: not used" in text
    assert "Batched LA     : not used" in text
    assert "Energy checkpoint: Gamma max |Delta E|=1.20000e-03 eV" in text
    assert "not a global path-error bound" in text
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

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
        fine_plan=SimpleNamespace(
            requested="distributed", backend="cusolvermp",
            requested_batched_route="auto", batched_route="scan"),
        fine_enabled=True)
    centroids = SimpleNamespace(closure=SimpleNamespace(
        closed=False, n_sym=2, n_violating=1, n_centroids=20,
        tol=1.0e-5, worst_residual=0.25, worst_op=1))
    report.sampling(wfn=wfn, sym=sym, centroids=centroids)
    result = {
        "band_start": 2, "nb_keep": 4, "nb_fit": 6,
        "n_guard_bands": 2, "nk_total": 8,
        "path_range": (-0.2, 0.4),
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
        "method": "whole_state_randomized_qrcp",
        "stacked_states": 48, "state_dimension": 40,
        "candidate_count": 12, "search_rank": 8, "raw_rank": 3,
        "retained_rank": 3, "carried_rank": 4, "null_padding": 1,
        "rank_multiplier": 20.0, "qr_eps": 1.0e-3, "qrcp_seed": 0,
        "qrcp_rng_version": "test-v1", "candidate_hash": "abc",
        "pivot_hash": "def", "min_cholesky_diagonal": 0.2,
        "coefficient_orthogonality_error": 2.0e-7,
        "max_missing_state_norm_squared": 3.0e-4,
        "relative_frobenius_residual": 4.0e-3,
        "selected_orientation_error": 5.0e-12,
        "selected_orientation_tolerance": 2.0e-8,
    })
    report.htransform_quality({
        "row_isometry_max": 2.0e-7,
        "row_isometry_cap": 1.0e-6,
        "outer_shell_l2_fraction": 0.051,
        "outer_shell_max_over_r0": 0.012,
        "locality_wall_seconds": 0.08,
    })
    report.path_summary(result=result)
    report.progress("Started Hamiltonian interpolation at 12:00:00.")
    report.timings([
        {"name": "htransform.runtime_stack.jax_import", "inclusive": 1.5},
        {"name": "htransform.imports", "inclusive": 0.5},
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
    assert "Centroid sites : 20 requested from the full table" in text
    assert "mesh-padded" not in text
    assert "Centroid orbit: NOT CLOSED : 1/2 spatial operations" in text
    assert "worst residual 2.50000e-01 at S02" in text
    assert "Galerkin basis : rank 12" in text
    assert "Galerkin basis : 48 stacked states x 40 full-Bloch components" in text
    assert "QRCP rank      : qr_eps=1.00000e-03; raw rank 3" in text
    assert "relative Frobenius residual=4.00000e-03" in text
    assert "not a fine-k band-energy error bound" in text
    assert "rank 4 = 3 physical + 1 exact-null mesh pad" in text
    assert "Guard buffer   :" in text
    assert "Guard bands    : 7-8; E range" in text
    assert "f-transform    : a=5.44228 eV (4 x bandwidth of band 8); n=3.00" in text
    assert "Zero shoulder  : max E(band 8) = 6.39468 eV" in text
    assert "N01    Γ  k=( 0.00000  0.00000  0.00000)" in text
    assert "L01  Γ -> X: 1 intervals; 2 endpoint-inclusive points" in text
    assert "whole-state randomized QRCP" in text
    assert "no Gram eigensolve" in text
    assert "Fine-k eigensolve: distributed -> cusolvermp" in text
    assert "Batched LA     : auto (other choices: batch_reshard) -> scan" in text
    assert "Row isometry   : max |C C^H - I|=2.00000e-07" in text
    assert "On-grid screen" not in text
    assert "f(H)_R locality: outer-shell ||f(H)_R||_F / total = 5.10%" in text
    assert "Fine-k error   : unavailable without an independent" in text
    assert "Energy checkpoint" not in text
    assert "Processor mesh : 2 x 2\n" in text
    assert " Ry" not in text
    assert "auto (other choices:" in text
    assert "Started Hamiltonian interpolation" in text
    assert "runtime bring-up" in text
    assert "pre-main + imports" in text
    assert "other driver work" in text
    assert "OUTPUT FILES AND INPUTS" in text
    assert "calculation report" not in text
    assert "HDF5" not in text and "h5py" not in text


def test_htransform_runtime_startup_uses_the_one_debug_stream():
    source = (Path(__file__).parents[1] / "src" / "bandstructure" /
              "htransform.py").read_text(encoding="utf8")
    assert "initialize_communicator_stack(print_fn=debug_print)" in source
    assert "RUNTIME = initialize_communicator_stack()" not in source


def test_basis_input_emits_exactly_one_canonical_rank_receipt(
        tmp_path, monkeypatch):
    import bandstructure.htransform as htransform
    from isdf.galerkin import GalerkinBasis

    selected = (0, 1)
    basis = GalerkinBasis(
        ctilde=np.eye(2, dtype=np.complex128)[None],
        basis_at_nodes=np.ones((2, 1, 1), dtype=np.complex128),
        rank_physical=2,
        band_range=(3, 5),
        selected_state_indices=selected,
        selection_factor=np.eye(2, dtype=np.complex128),
        qrcp_seed=7,
        qrcp_eps=1.0e-3,
        qrcp_raw_rank=2,
        qrcp_search_rank=2,
        candidate_hash="a" * 64,
        pivot_hash="b" * 64,
    )
    basis_path = tmp_path / "basis.h5"
    basis_path.touch()
    monkeypatch.setattr(
        htransform, "read_galerkin_basis", lambda *args, **kwargs: basis)
    records = []
    restored = htransform.streaming_galerkin_solve(
        object(), object(), SimpleNamespace(nspinor=1, n_rtot=2),
        np.zeros((1, 3), dtype=np.int32), object(), (3, 5),
        basis_input=str(basis_path), rank_record_fn=records.append,
        rank_multiplier=20.0, qr_eps=1.0e-3, qrcp_seed=7,
    )

    assert restored is basis
    assert len(records) == 1
    record = records[0]
    assert (record["method"], record["stacked_states"],
            record["state_dimension"]) == (
                "whole_state_randomized_qrcp", 2, 2)
    assert (record["raw_rank"], record["retained_rank"],
            record["carried_rank"], record["null_padding"]) == (2, 2, 2, 0)
    assert (record["candidate_hash"], record["pivot_hash"]) == (
        "a" * 64, "b" * 64)
    assert record["min_cholesky_diagonal"] == 1.0
    assert record["coefficient_orthogonality_error"] == 0.0
    assert record["relative_frobenius_residual"] == 0.0
    assert "singular_values" not in record


def test_outer_r_shell_mask_handles_even_odd_and_singleton_axes():
    from bandstructure.htransform import build_R_grid_np, outer_r_shell_mask

    grid = (4, 3, 1)
    r_grid = np.asarray(build_R_grid_np(grid), dtype=int)
    mask = outer_r_shell_mask(grid)
    expected = ((np.abs(r_grid[:, 0]) == 2)
                | (np.abs(r_grid[:, 1]) == 1))
    assert mask.dtype == np.bool_
    assert np.array_equal(mask, expected)
    assert not mask[np.flatnonzero(np.all(r_grid == 0, axis=1))[0]]

"""Contract tests for the rank-zero human-readable GW output."""

from types import SimpleNamespace

import numpy as np

from common.units import RYD_TO_EV
from gw.production_report import (
    EQP0_FILE_ROLE,
    EQP1_FILE_ROLE,
    QP_ROTATIONS_FILE_ROLE,
    QP_WFN_FILE_ROLE,
    GWProductionReport,
)


def _config():
    return SimpleNamespace(
        compute_mode=SimpleNamespace(value="gn_ppm", is_dynamic=True),
        qp_solver=SimpleNamespace(value="one_shot_dft"),
        sys_dim=2,
        do_screened=True,
        bispinor=False,
        no_degen_averaging=False,
        degen_avg_tol_ry=1.0e-6,
        restart=False,
        backend=SimpleNamespace(
            linalg="local",
            linalg_provenance="default",
            charge_zeta_solve="rank_truncate",
            distributed_zeta_solve="per_q",
            w_dyson_solver="distributed",
            distributed_lu="cusolvermp",
            eigh_backend="native",
            distrib_la_batched_route="batch_reshard",
        ),
        memory=SimpleNamespace(
            per_device_gb=40.0, low_mem_bands=False, band_chunk_size=16),
        raw_input_keys=frozenset(),
        screening=SimpleNamespace(
            diagrams=SimpleNamespace(value="w_rpa"), method="minimax"),
        head=SimpleNamespace(
            correction=SimpleNamespace(value="full"),
            wcoul0_source="s_tensor"),
        sigma=SimpleNamespace(
            omega_min_ev=-5.0,
            omega_max_ev=5.0,
            omega_step_ev=0.5,
            fermi_reference="midgap",
            band_extrapolation=True,
            band_extrapolation_estimator="spectral_shell",
            band_extrapolation_bracket_scheme="total_fractions",
        ),
    )


def _runtime():
    facts = {
        "process_count": 4,
        "n_devices": 4,
        "n_local_devices": 1,
        "backend": "gpu",
        "device_kind": "NVIDIA A100-SXM4-40GB",
        "mesh_shape": (2, 2),
        "threads": {"affinity": 16, "OMP_NUM_THREADS": "8"},
        "jax_version": "0.test",
        "jaxlib_version": "0.testlib",
        "x64": True,
        "ffi_dials": [
            {"env": "LORRAX_FFT_FFI", "enabled": True,
             "platforms": ("CUDA",), "target": "lorrax_cufft_flat_k"},
            {"env": "LORRAX_FFT_FFI_FUSED", "enabled": False,
             "platforms": ("CUDA",), "off_label": "three FFT chain"},
            # CPU-only: production output must state the active GPU-native
            # lowering, not describe a skipped platform-policy gate.
            {"env": "LORRAX_BANDS_GEMM_FFI", "enabled": True,
             "platforms": ("cpu",), "off_label": "native XLA dot lowering"},
            {"env": "LORRAX_CONV_KPAIR_FFI", "enabled": True,
             "platforms": ("CUDA",), "target": "lorrax_cufft_conv_kpair"},
            # No production Sigma caller: this capability is not a run control.
            {"env": "LORRAX_CONV_KLEAD_FFI", "enabled": False,
             "platforms": ("CUDA",), "off_label": "XLA"},
        ],
    }
    return SimpleNamespace(process_index=0, facts=facts)


def test_report_is_scientific_rank_zero_output(tmp_path):
    output = []
    report_path = tmp_path / "gwjax.out"
    report = GWProductionReport(
        str(report_path), runtime=_runtime(), debug=False, stdout=output.append)
    config = _config()
    report.begin(input_file="gw.in", config=config)
    report.architecture()
    report.method(config=config)

    wfn = SimpleNamespace(
        backend="phdf5", kweights=np.array([0.25, 0.75]),
        kpoints=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        kgrid=np.array([2, 1, 1]), shift=np.zeros(3),
        num_electrons=4.0, efermi=0.0,
    )
    sym = SimpleNamespace(
        Rinv_grid=np.eye(3, dtype=int)[None],
        translations=np.zeros((1, 3)), trs_allowed=True,
        active_symmetry_rows=np.array([0], dtype=np.int32),
        nk_tot=2, nk_red=2, kirr_fullids=np.array([0, 1]),
    )
    bands = SimpleNamespace(
        b0=0, b1=1, b2=2, b3=3, b4=4, b4_chi=4, b4_sigma=4)
    energies = np.array([
        [-2.0, -0.2, 0.3],
        [-1.8, -0.1, 0.4],
    ]) / RYD_TO_EV

    report.environment(config=config, wfn=wfn)
    centroids = SimpleNamespace(closure=SimpleNamespace(
        closed=True, n_sym=1, n_violating=0, n_centroids=399,
        tol=1.0e-5, worst_residual=0.0, worst_op=0))
    report.sampling(wfn=wfn, sym=sym, centroids=centroids)
    report.bands(config=config, wfn=wfn, band_slices=bands,
                 zeta_ranges=((0, 3), (1, 4)))
    sigma_result = SimpleNamespace(
        efermi_dft_ev=1.0, omega_reference_provenance="fixed-N mu",
        omega_grid_ev=np.linspace(-5.0, 5.0, 21),
        band_extrapolation_counts=(36, 41, 46),
        band_extrapolation_estimator="spectral_shell",
        band_extrapolation_scheme="total_fractions",
    )
    report.sigma_coverage(
        config=config, band_slices=bands, enk_dft_ry=energies,
        sigma_result=sigma_result)
    report.legacy_print("[rank 1] harmless backend implementation chatter")
    report.legacy_print("Started sigma[correlation] at 12:00:00.")
    report.legacy_print(
        "[ 12:00:01 | █████░░░░░ |  50% ] tau node 5 / 10 · ETA 1 s")
    report.legacy_print(
        "  SC fixed quadrature: iteration=2, initialized=False, "
        "rebuilds_this_iteration=0, rebuilds_total=0, pair_cost=61")
    report.legacy_print(
        "    SC fixed window: c:bulk: n_tau=8, nodes=0123456789abcdef, "
        "cache=hit:sc-fixed, padded_box=(-13.6, 27.2, 1.36, 2.72) eV")
    report.legacy_print("WARNING: protected state leaves the omega grid")
    report.qp_gap(band_slices=bands, e_dft_ry=energies,
                  e_qp_ry=energies + 0.1 / RYD_TO_EV)
    report.timings([
        {"name": "gw_jax.runtime_stack.jax_import", "path":
         ("gw_jax.runtime_stack.jax_import",), "inclusive": 2.0},
        {"name": "gw_jax.imports", "path":
         ("gw_jax.imports",), "inclusive": 1.0},
        {"name": "gw_jax.startup", "path":
         ("gw_jax.startup",), "inclusive": 1.5},
        {"name": "gw_jax.isdf", "path":
         ("gw_jax.isdf",), "inclusive": 4.5},
        {"name": "gw_jax.zeta_fit_chunked", "path":
         ("gw_jax.isdf", "gw_jax.zeta_fit_chunked"), "inclusive": 1.0},
        {"name": "gw_jax.V_q_compute", "path":
         ("gw_jax.isdf", "gw_jax.V_q_compute"), "inclusive": 2.0},
        {"name": "gw_jax.restart_load", "path":
         ("gw_jax.isdf", "gw_jax.restart_load"), "inclusive": 0.5},
        {"name": "gw_jax.minimax_quadrature", "path":
         ("gw_jax.minimax_quadrature",), "inclusive": 0.5},
        {"name": "gw_jax.screening", "path":
         ("gw_jax.screening",), "inclusive": 8.0},
        {"name": "chi.exec", "path":
         ("gw_jax.screening", "chi.exec"), "inclusive": 3.0},
        {"name": "W.exec", "path":
         ("gw_jax.screening", "W.exec"), "inclusive": 4.0},
        {"name": "gw_jax.persist_w0", "path":
         ("gw_jax.persist_w0",), "inclusive": 0.25},
        {"name": "gw_jax.static_head", "path":
         ("gw_jax.static_head",), "inclusive": 0.25},
        {"name": "gw_jax.sigma", "path": ("gw_jax.sigma",),
         "inclusive": 5.0},
        {"name": "gw_jax.kin_ion_load", "path":
         ("gw_jax.kin_ion_load",), "inclusive": 0.25},
        {"name": "gw_jax.solve_qp", "path":
         ("gw_jax.solve_qp",), "inclusive": 0.5},
        {"name": "gw_jax.qp_eigh", "path":
         ("gw_jax.qp_eigh",), "inclusive": 0.25},
        {"name": "gw_jax.output", "path":
         ("gw_jax.output",), "inclusive": 0.5},
    ], wall=26.5)
    report.finish()

    text = report_path.read_text(encoding="utf-8")
    assert text == "\n".join(output) + "\n"
    assert "MPI ranks      : 4" in text
    assert "JAX/JAXLIB     : 0.test / 0.testlib" in text
    assert "Godby-Needs plasmon-pole GW" in text
    assert "one-shot full-matrix effective Hamiltonian" in text
    assert "fixed-DFT-state diagonal G0W0" in text
    assert "textbook G0W0" not in text and "standard G0W0" not in text
    assert ("QP consistency  : one_shot_dft | other options: fixed_point "
            "(diagonal on-shell), self_consistent (rebuild G/W/Sigma)") in text
    assert "EQP2 treatment  : off (set write_eqp2=true" in text
    assert "RPA Dyson series; minimax imaginary-axis quadrature" in text
    assert "Degenerate sets: averaged at 1.36057e-05 eV" in text
    assert " Ry" not in text and "Rydberg" not in text
    assert "Coulomb system : 2D slab; Hartree=live G-space" in text
    assert "Full BZ grid   : 2 k points" in text
    assert "Stored IBZ     : 2 k points" in text
    assert ("Centroid orbit: CLOSED : 1/1 spatial operations preserve 399 "
            "sites (tol=1.00000e-05)") in text
    assert "QP valence     : 2-2" in text
    assert "indices [" not in text
    assert "Energy origin   : E_F = +1.00000 eV (fixed-N mu)" in text
    assert "Absolute window" not in text
    assert "Coverage status : COMPLETE" in text
    assert "Grid margins    : 3.80000 eV below; 5.60000 eV above" in text
    assert "Tail calculations: N1=36 / N2=41 / N3=46 cumulative bands" in text
    assert "DFT gap        : 0.40000 eV" in text
    assert "Full-matrix effective-H gap: 0.40000 eV (insulating)" in text
    assert "Gap correction : +0.00000 eV relative to DFT" in text
    assert "Quasiparticle energies" not in text
    assert "cuFFT flat-k FFI" in text
    assert "cuFFT pair-convolution FFI" in text
    assert "mklfft" not in text
    gemm_line = next(line for line in text.splitlines()
                     if "LORRAX_BANDS_GEMM_FFI" in line)
    assert "= NATIVE" in gemm_line
    assert "LORRAX_CONV_KLEAD_FFI" not in text
    assert "harmless backend implementation chatter" not in text
    assert "Started sigma[correlation]" in text
    assert "tau node 5 / 10" in text
    assert "SC fixed quadrature: iteration=2" in text
    assert "SC fixed window: c:bulk" in text
    assert "nodes=0123456789abcdef" in text
    assert "protected state leaves the omega grid" in text
    zeta_line = next(line for line in text.splitlines()
                     if line.startswith("  zeta"))
    sigma_line = next(line for line in text.splitlines()
                      if line.startswith("  Sigma"))
    assert "      1.00" in zeta_line
    assert "      5.00" in sigma_line
    assert "runtime bring-up             2.00" in text
    assert "pre-main + imports           1.00" in text
    assert "input + run setup            1.50" in text
    assert "restart load                 0.50" in text
    assert "ISDF setup + I/O             1.00" in text
    assert "screening support            1.00" in text
    assert "W persist + q0 head          0.50" in text
    assert "QP solve + diagonalize       0.75" in text
    assert "result writes                0.50" in text
    assert "other driver work            2.00" in text
    assert "HDF5" not in text and "h5py" not in text


def test_report_spells_out_enabled_eqp2_and_convergence(tmp_path):
    path = tmp_path / "gwjax.out"
    report = GWProductionReport(
        str(path), runtime=_runtime(), debug=False, stdout=lambda line: None)
    config = _config()
    config.eqp2 = SimpleNamespace(
        enabled=True, tol_ev=1.0e-3, max_iter=20,
        accelerator="rcrop", history_depth=5)
    report.method(config=config)
    bands = SimpleNamespace(b0=0, b2=1)
    energies = np.array([[-1.0, 1.0], [-0.8, 0.9]]) / RYD_TO_EV
    report.eqp2_summary(
        band_slices=bands, e_eqp2_ry=energies,
        iterations=4, residual_ev=0.0007, tol_ev=0.001)
    report.finish()
    text = path.read_text(encoding="utf-8")
    assert "fixed-Sigma eigenvalue self-consistency" in text
    assert ("max|dE| cutoff=1.000 meV (non-scissored); "
            "accelerator=rcrop(depth=5); "
            "max_iter=20; screening unchanged") in text
    assert "max|dE|=0.700000 meV <= 1.000000 meV" in text
    assert "final post-rotation map verified" in text
    assert "semicore preserves E-E_F" in text
    assert "Held fixed      : screening, W, and all self-energy diagrams" in text
    assert "EQP2 eigenvectors: internal to this consistency loop; not serialized" in text
    assert "QP WFN/rotations: remain the ordinary one-shot" in text
    assert "eqp2.dat contains the final EQP2 eigenvalue spectrum only" in text


def test_closing_file_roles_do_not_conflate_full_matrix_and_diagonal_outputs():
    assert EQP0_FILE_ROLE == "fixed-DFT-state diagonal zeroth-order energies"
    assert EQP1_FILE_ROLE == "fixed-DFT-state diagonal Z-linearized energies"
    assert QP_ROTATIONS_FILE_ROLE == "full-matrix effective-H rotations"
    assert QP_WFN_FILE_ROLE == "matched full-matrix QP wavefunctions"


def test_incomplete_sigma_coverage_is_an_actionable_final_warning(tmp_path):
    path = tmp_path / "gwjax.out"
    report = GWProductionReport(
        str(path), runtime=_runtime(), debug=False, stdout=lambda line: None)
    config = _config()
    config.sigma.omega_min_ev = -0.15
    config.sigma.omega_max_ev = 0.25
    config.sigma.omega_step_ev = 0.05
    bands = SimpleNamespace(
        b0=0, b1=1, b2=2, b3=3, b4=4, b4_chi=4, b4_sigma=4)
    energies = np.array([
        [-2.0, -0.2, 0.3],
        [-1.8, -0.1, 0.4],
    ]) / RYD_TO_EV
    coverage = SimpleNamespace(
        mask_kn=np.array([[False, True, False], [False, True, False]]),
        n_uncovered=4, policy="clamp")
    sigma_result = SimpleNamespace(
        efermi_dft_ev=0.0, omega_reference_provenance="midgap",
        omega_grid_ev=np.linspace(-0.15, 0.25, 9),
        omega_coverage=coverage,
        band_extrapolation_counts=None,
        band_extrapolation_estimator="spectral_shell",
        band_extrapolation_scheme="total_fractions",
    )

    report.sigma_coverage(
        config=config, band_slices=bands, enk_dft_ry=energies,
        sigma_result=sigma_result)
    report.finish()

    text = path.read_text(encoding="utf-8")
    assert "Coverage status : INCOMPLETE" in text
    assert "Grid shortfall  : 0.05000 eV below; 0.15000 eV above" in text
    assert "WARNINGS" in text
    assert "dynamic Sigma grid is incomplete for protected DFT states" in text
    assert "Sigma(E_DFT) has 4/6 out-of-grid cells" in text
    assert "out-of-range policy=clamp" in text
    assert "sigma_omega_min_ev / sigma_omega_max_ev" in text
    assert "sigma_omega_patches_ev" in text
    assert "LORRAX_OMEGA_OUT_OF_RANGE=refuse" in text


def test_timing_report_hides_sub_display_precision_residuals(tmp_path):
    path = tmp_path / "gwjax.out"
    report = GWProductionReport(
        str(path), runtime=_runtime(), debug=False, stdout=lambda line: None)
    report.timings([
        {"name": "gw_jax.isdf", "path": ("gw_jax.isdf",),
         "inclusive": 1.000001},
        {"name": "gw_jax.restart_load",
         "path": ("gw_jax.isdf", "gw_jax.restart_load"),
         "inclusive": 1.0},
    ], wall=1.5)
    report.finish()

    text = path.read_text(encoding="utf-8")
    assert "restart load                 1.00" in text
    assert "ISDF setup + I/O" not in text
    assert "other driver work            0.50" in text


def test_transverse_timing_counts_outer_wall_once(tmp_path):
    """Overlapping current workers contribute their enclosing wall interval once."""
    path = tmp_path / "gwjax.out"
    report = GWProductionReport(
        str(path), runtime=_runtime(), debug=False, stdout=lambda line: None)
    report.timings([
        {"name": "gw_jax.isdf", "path": ("gw_jax.isdf",), "inclusive": 8.0},
        {"name": "gw_jax.zeta_fit_chunked",
         "path": ("gw_jax.isdf", "gw_jax.zeta_fit_chunked"), "inclusive": 1.0},
        {"name": "gw_jax.zeta_fit_transverse",
         "path": ("gw_jax.isdf", "gw_jax.zeta_fit_transverse"), "inclusive": 4.0},
        {"name": "gw_jax.zeta_fit_chunked_mu1",
         "path": ("gw_jax.zeta_fit_chunked_mu1",), "inclusive": 3.0},
        {"name": "gw_jax.zeta_fit_chunked_mu2",
         "path": ("gw_jax.zeta_fit_chunked_mu2",), "inclusive": 3.0},
        {"name": "gw_jax.zeta_fit_chunked_mu3",
         "path": ("gw_jax.zeta_fit_chunked_mu3",), "inclusive": 3.0},
        {"name": "gw_jax.V_q_compute",
         "path": ("gw_jax.isdf", "gw_jax.V_q_compute"), "inclusive": 1.0},
    ], wall=10.0)
    report.finish()
    text = path.read_text()
    assert "zeta                         1.00" in text
    assert "zeta transverse              4.00" in text
    assert "ISDF setup + I/O             2.00" in text
    assert "other driver work            2.00" in text


def test_report_collapses_minimax_catalog_receipts(tmp_path):
    path = tmp_path / "gwjax.out"
    report = GWProductionReport(
        str(path), runtime=_runtime(), debug=False, stdout=lambda line: None)
    report.legacy_print(
        "RuntimeWarning: minimax: served noncrossing/inverse R=10 "
        "target 1e-06 -> 8 nodes, max_err 1.85e-07 | shipped a.npz "
        "sha256:aaaa gen unrecorded backend unrecorded UNCERTIFIED")
    report.legacy_print(
        "RuntimeWarning: minimax: served crossing/hgl A_dim=24 "
        "target 1e-06 -> 48 nodes, max_err 3.2e-07 | shipped b.npz "
        "sha256:bbbb gen unrecorded backend unrecorded UNCERTIFIED")
    report.finish()

    text = path.read_text(encoding="utf-8")
    assert "2 catalog entries used (8-48 nodes)" in text
    assert "requested tolerance 1.00000e-06" in text
    assert "worst reported error 3.20000e-07" in text
    assert text.count("UNCERTIFIED") == 1
    assert "sha256" not in text


def test_nonzero_rank_never_creates_report(tmp_path):
    runtime = _runtime()
    runtime.process_index = 2
    path = tmp_path / "gwjax.out"
    report = GWProductionReport(
        str(path), runtime=runtime, debug=False,
        stdout=lambda line: (_ for _ in ()).throw(AssertionError(line)))
    report.emit("must not be emitted")
    report.finish()
    assert not path.exists()


def test_local_linalg_warning_has_dense_matrix_numbers(tmp_path):
    path = tmp_path / "gwjax.out"
    report = GWProductionReport(
        str(path), runtime=_runtime(), debug=False, stdout=lambda line: None)
    report.layout_dials(config=_config(), n_mu=640, n_q_irr=9, processes=4)
    report.finish()

    text = path.read_text(encoding="utf-8")
    assert "[config provenance] linalg = local (default)" in text
    assert (
        "linalg = local: fastest, but each task holds ceil(N_q_irr/P) "
        "complete N_mu x N_mu dense matrices (here ceil(9/4) x 640^2 x "
        "16 B = 18.8 MiB per task, complex128) plus their factor workspace; "
        "on large systems where that is a large fraction of memory per "
        "task, set linalg = distributed" in text)
    assert "[config provenance] low_mem_bands = false (default)" in text


def test_parent_carrier_description_has_automatic_chunk_number(tmp_path):
    path = tmp_path / "gwjax.out"
    config = _config()
    config.memory.low_mem_bands = True
    config.memory.band_chunk_size = 24
    config.raw_input_keys = frozenset({"low_mem_bands"})
    report = GWProductionReport(
        str(path), runtime=_runtime(), debug=False, stdout=lambda line: None)
    report.layout_dials(
        config=config, n_mu=7000, n_q_irr=9, processes=4)
    report.finish()

    text = path.read_text(encoding="utf-8")
    assert "[config provenance] low_mem_bands = true (deck)" in text
    assert (
        "low_mem_bands = true: the two-face wavefunction carrier (band "
        "chunks of 24); required for the raw-parent (k_irr) route" in text)
    assert "set false" not in text


def test_distributed_linalg_reports_2d_layout(tmp_path):
    path = tmp_path / "gwjax.out"
    config = _config()
    config.backend.linalg = "distributed"
    config.backend.linalg_provenance = "deck"
    report = GWProductionReport(
        str(path), runtime=_runtime(), debug=False, stdout=lambda line: None)
    report.layout_dials(
        config=config, n_mu=9000, n_q_irr=9, processes=4)
    report.finish()

    text = path.read_text(encoding="utf-8")
    assert "[config provenance] linalg = distributed (deck)" in text
    assert "2D distributed across the process mesh (N_mu = 9000)" in text
    assert "linalg = local: fastest" not in text


def test_mpa_trs_route_is_part_of_the_scientific_run_record(tmp_path):
    path = tmp_path / "gwjax.out"
    report = GWProductionReport(
        str(path), runtime=_runtime(), debug=False, stdout=lambda line: None)
    config = _config()
    config.compute_mode = SimpleNamespace(value="mpa", is_dynamic=True)

    report.trs_pathways(
        config=config, sym=SimpleNamespace(trs_allowed=False),
        material_class="insulator")
    report.finish()

    text = path.read_text(encoding="utf-8")
    assert ("TRS pathways   : automatic from SymMaps.trs_allowed=false; "
            "no input override") in text
    assert "global time reversal MEASURED BROKEN" in text
    assert "one contour sweep plus the q-negated conjugate partner" in text

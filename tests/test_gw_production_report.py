"""Contract tests for the rank-zero human-readable GW output."""

from types import SimpleNamespace

import numpy as np

from common.units import RYD_TO_EV
from gw.production_report import GWProductionReport


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
        hartree_source="auto",
        backend=SimpleNamespace(
            charge_zeta_solve="rank_truncate",
            distributed_zeta_solve="per_q",
            w_dyson_solver="distributed",
            distributed_lu="cusolvermp",
            eigh_backend="native",
        ),
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
        nk_tot=2, nk_red=2, kirr_fullids=np.array([0, 1]),
    )
    bands = SimpleNamespace(
        b0=0, b1=1, b2=2, b3=3, b4=4, b4_chi=4, b4_sigma=4)
    energies = np.array([
        [-2.0, -0.2, 0.3],
        [-1.8, -0.1, 0.4],
    ]) / RYD_TO_EV

    report.environment(config=config, wfn=wfn)
    report.sampling(wfn=wfn, sym=sym)
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
    report.legacy_print("WARNING: protected state leaves the omega grid")
    report.qp_energies(wfn=wfn, sym=sym, band_slices=bands,
                       e_dft_ry=energies, e_qp_ry=energies + 0.1 / RYD_TO_EV)
    report.timings([
        {"name": "gw_jax.zeta_fit_chunked", "path":
         ("gw_jax.isdf", "gw_jax.zeta_fit_chunked"), "inclusive": 1.0},
        {"name": "gw_jax.V_q_compute", "path":
         ("gw_jax.isdf", "gw_jax.V_q_compute"), "inclusive": 2.0},
        {"name": "chi.exec", "path":
         ("gw_jax.screening", "chi.exec"), "inclusive": 3.0},
        {"name": "W.exec", "path":
         ("gw_jax.screening", "W.exec"), "inclusive": 4.0},
        {"name": "gw_jax.sigma", "path": ("gw_jax.sigma",),
         "inclusive": 5.0},
    ], wall=20.0)
    report.finish()

    text = report_path.read_text(encoding="utf-8")
    assert text == "\n".join(output) + "\n"
    assert "MPI ranks      : 4" in text
    assert "Godby-Needs plasmon-pole GW" in text
    assert "RPA Dyson series; minimax imaginary-axis quadrature" in text
    assert "Coulomb system : 2D slab; Hartree source=auto" in text
    assert "Full BZ grid   : 2 k points" in text
    assert "Stored IBZ     : 2 k points" in text
    assert "QP valence     : 2-2  (indices [1,2))" in text
    assert "Energy origin   : E_F = +1.000000 eV (fixed-N mu)" in text
    assert "Omega margins  : +3.800 eV below; +5.600 eV above" in text
    assert "Tail calculations: N1=36 / N2=41 / N3=46 cumulative bands" in text
    gemm_line = next(line for line in text.splitlines()
                     if "LORRAX_BANDS_GEMM_FFI" in line)
    assert "= NATIVE" in gemm_line
    assert "LORRAX_CONV_KLEAD_FFI" not in text
    assert "harmless backend implementation chatter" not in text
    assert "protected state leaves the omega grid" in text
    zeta_line = next(line for line in text.splitlines()
                     if line.startswith("  zeta"))
    sigma_line = next(line for line in text.splitlines()
                      if line.startswith("  Sigma"))
    assert "1.000" in zeta_line
    assert "5.000" in sigma_line
    assert "HDF5" not in text and "h5py" not in text


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

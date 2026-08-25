"""Contract for the shared dipole / kin_ion scientific report."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from common.preprocessing_output import PreprocessingProductionReport


def _runtime(rank=0):
    return SimpleNamespace(process_index=rank, facts={
        "process_count": 4, "n_devices": 4, "n_local_devices": 1,
        "backend": "gpu", "device_kind": "A100", "mesh_shape": (2, 2),
        "threads": {"affinity": 16, "OMP_NUM_THREADS": "8"},
        "jax_version": "0.9.1", "jaxlib_version": "0.9.1", "x64": True,
    })


def test_preprocessing_report_is_human_readable_and_shared(tmp_path):
    output = []
    path = tmp_path / "dipole.out"
    report = PreprocessingProductionReport(
        str(path), runtime=_runtime(), debug=False, stdout=output.append,
        driver_name="psp.get_dipole_mtxels",
        calculation_name="dipole and velocity preprocessing")
    report.begin(input_file="cohsex.in")
    report.architecture(mesh_role="band-matrix axes X x Y")
    wfn = SimpleNamespace(
        backend="phdf5", density_symmetry=None,
        kgrid=np.array([2, 1, 1]), shift=np.zeros(3),
        kpoints=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        kweights=np.array([0.25, 0.75]),
    )
    sym = SimpleNamespace(
        Rinv_grid=np.eye(3, dtype=int)[None],
        translations=np.array([[np.pi, 0.0, 0.0]]),
        trs_allowed=True, nk_tot=2, nk_red=2,
    )
    report.environment(wfn=wfn, lines=("Output writer  : bounded owner gather",))
    report.pathways(("Velocity       : p + i[r, V_NL]",))
    report.system(natoms=2, species=("Si.upf",), fft_grid=(8, 8, 8))
    report.sampling(wfn=wfn, sym=sym)
    report.bands(("Matrix written : 1-8",))
    report.progress("Started q=0 velocity matrix construction at 12:00:00.")
    report.legacy_print("HDF5 library rank-local implementation chatter")
    report.legacy_print("WARNING: retained physical warning")
    report.timings((("q=0 velocity", 1.25),), wall=2.0)
    report.files((("dipole matrices", "written", "dipole.h5"),))
    report.finish()

    text = path.read_text(encoding="utf-8")
    assert text == "\n".join(output) + "\n"
    assert "MPI ranks      : 4" in text
    assert "JAX/JAXLIB     : 0.9.1 / 0.9.1" in text
    assert "Spatial group   : 1 operations; 1 with fractional translations" in text
    assert "tau=( 0.500  0.000  0.000)" in text
    assert "fractional tau" not in text
    assert "  2   0.50000   0.00000   0.00000   0.75000" in text
    assert "          1.25" in text
    assert "HDF5 library rank-local" not in text
    assert "retained physical warning" in text
    assert "OUTPUT FILES AND INPUTS" in text


def test_preprocessing_report_is_rank_zero_only(tmp_path):
    path = tmp_path / "kin_ion.out"
    report = PreprocessingProductionReport(
        str(path), runtime=_runtime(rank=2), debug=False,
        stdout=lambda line: (_ for _ in ()).throw(AssertionError(line)),
        driver_name="gw.kin_ion_io",
        calculation_name="kinetic and ionic preprocessing")
    report.emit("must remain silent")
    report.finish()
    assert not path.exists()


def test_dipole_cli_has_no_second_debug_print_control():
    source = (Path(__file__).parents[1] / "src" / "psp" /
              "get_dipole_mtxels.py").read_text(encoding="utf-8")
    assert '"--debug"' not in source
    assert '"--debug-kindex"' not in source
    assert '"--divide-energy"' not in source
    assert "LORRAX_DEBUG_PRINT" in source


def test_bse_random_demo_is_named_as_a_self_test_not_a_debug_switch():
    source = (Path(__file__).parents[1] / "src" / "bse" /
              "bse_jax.py").read_text(encoding="utf-8")
    assert '"--debug-parallelism"' not in source
    assert '"--parallelism-self-test"' in source
    assert "ScientificProductionReport" in source
    assert '"--report-file"' in source
    assert "report.sampling(wfn=wfn, sym=sym)" in source
    assert 'barrier("bse.report_written")' in source


def test_exciton_bands_uses_shared_report_and_paths_last_contract():
    source = (Path(__file__).parents[1] / "src" / "bse" /
              "exciton_bands.py").read_text(encoding="utf-8")
    assert "ScientificProductionReport" in source
    assert "ProductionStdout" in source
    assert '"--report-file"' in source
    assert "report.sampling(wfn=wfn, sym=sym)" in source
    assert "stage_progress = LoopProgress(" in source
    assert "report.files(file_rows)" in source
    assert source.index("report.files(file_rows)") < source.index("report.finish()")

"""Golden text of the shared main-log blocks (``common.report_blocks``) and of the
``ProductionReport`` stream that prints them.

Fixed facts in, exact lines out: the primitives, the header, the Devices block, the
outputs table and the closing line, then one report written end to end (header,
Devices, files, ERROR and COMPLETED closings) with the report's own path left out
of its files table.  No device, no JAX: the runtime is a namespace of facts.
"""
import re
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pytest

from common import report_blocks as rb

RULE = "=" * 100
PDT = timezone(timedelta(hours=-7), "PDT")
STARTED = datetime(2026, 10, 19, 19, 16, 11, tzinfo=PDT)
ENDED = datetime(2026, 10, 19, 19, 16, 55, tzinfo=PDT)

GPU_FACTS = {
    "process_count": 4, "n_nodes": 1, "n_devices": 4, "n_local_devices": 1,
    "device_kind": "NVIDIA A100-SXM4-80GB", "backend": "gpu", "mesh_shape": (2, 2),
    "threads": {"affinity": 32, "OPENBLAS_NUM_THREADS": 16, "OMP_NUM_THREADS": "8",
                "MKL_NUM_THREADS": None},
    "device_memory": {"total": 85.09e9, "reservation": 77.89e9, "headroom": 7.20e9,
                      "budget": 66.35e9, "outside_max": 1.47e9},
    "jax_version": "0.9.1", "jaxlib_version": "0.9.1", "x64": True,
}
CPU_FACTS = {
    "process_count": 2, "n_devices": 4, "n_local_devices": 2, "device_kind": "cpu",
    "backend": "cpu", "mesh_shape": (2, 2), "threads": {"affinity": None},
    "jax_version": "0.9.1", "jaxlib_version": "0.9.1", "x64": True,
    "collectives": {"impl": "gloo"},
}


# ── primitives ───────────────────────────────────────────────────────────────
def test_heading_and_kv_rows():
    assert rb.heading("Output files and inputs") == [
        "", "OUTPUT FILES AND INPUTS", "-----------------------"]
    assert rb.kv_lines([("MPI ranks", "4 on 1 node"), ("a label of seventeen", "v")]) == [
        "  MPI ranks        4 on 1 node",
        "  a label of seventeen v"]
    assert rb.kv_lines([("Host", "nid200257")], indent=0) == ["Host             nid200257"]


def test_table_right_align_and_wrap():
    assert rb.table_lines(("op", "used as", "note"), [("S01", "unitary", "identity"),
                                                       ("S03", "with TR", "")],
                          widths=(4, 8, 0), align="rll",
                          legend=[("with TR", "composed with time reversal")]) == [
        "    op used as  note",
        "   S01 unitary  identity",
        "   S03 with TR",
        "    with TR          composed with time reversal"]
    line = rb.right_align("[1/4] ISDF basis and Coulomb metric", "19:16:12")
    assert line == "[1/4] ISDF basis and Coulomb metric" + " " * 57 + "19:16:12"
    assert len(line) == 100
    assert rb.right_align("x" * 99, "[id]") == "x" * 99 + "  [id]"     # runs over, 2 spaces
    assert rb.right_align("left", "") == "left"
    assert rb.wrap_value("aa bb cc dd", first_col=3, width=9) == ["aa bb", "   cc dd"]
    assert rb.wrap_value("a " + "x" * 20, first_col=3, width=9) == ["a", "   " + "x" * 20]


@pytest.mark.parametrize("seconds, text", [
    (0.0, "0.0 s"), (78.53, "78.5 s"), (99.94, "99.9 s"), (99.96, "1 min 40 s"),
    (1119.0, "18 min 39 s"), (3599.4, "59 min 59 s"), (3720.0, "1 h 02 min"),
    (36000.0, "10 h 00 min"), (-1.0, "0.0 s")])
def test_fmt_wall(seconds, text):
    assert rb.fmt_wall(seconds) == text


def test_display_path():
    assert rb.display_path("/scratch/run/tmp/isdf_tensors_6.h5", "/scratch/run") == \
        "tmp/isdf_tensors_6.h5"
    assert rb.display_path("/scratch/run2/WFN.h5", "/scratch/run") == "/scratch/run2/WFN.h5"
    assert rb.display_path("", "/scratch/run") == "-"
    assert rb.display_path("-", "/scratch/run") == "-"


# ── header ───────────────────────────────────────────────────────────────────
def test_header_lines():
    run = rb.RunFacts(
        driver="gw", purpose="GW quasiparticles",
        argv=("python", "-u", "-m", "gw.gw_jax", "-i", "gnppm.in"),
        directory="/pscratch/sd/j/jackm/run", started=STARTED, host="nid200257",
        job="Slurm job 60021274", logs=("gnppm.out",))
    assert rb.header_lines(run) == [
        RULE,
        "LORRAX gw · GW quasiparticles",
        RULE,
        "Command          python -u -m gw.gw_jax -i gnppm.in",
        "Directory        /pscratch/sd/j/jackm/run",
        "Started          2026-10-19 19:16:11 PDT",
        "Host             nid200257, Slurm job 60021274",
        "Logs             gnppm.out",
    ]


def test_header_optional_rows_and_command_wrap():
    argv = ("python", "-m", "bse.bse_jax", "-i", "bse.in") + tuple(
        f"--flag-{i}" for i in range(12))
    run = rb.RunFacts(driver="bse", purpose="Bethe-Salpeter eigensolve", argv=argv,
                      directory="/run", started=STARTED, host="login01", debug=True,
                      version="LORRAX main, clean", native_bundle="cloud bundle")
    assert rb.header_lines(run) == [
        RULE,
        "LORRAX bse · Bethe-Salpeter eigensolve",
        RULE,
        "Version          LORRAX main, clean",
        "Native bundle    cloud bundle",
        "Command          python -m bse.bse_jax -i bse.in --flag-0 --flag-1 --flag-2 --flag-3 --flag-4",
        "                 --flag-5 --flag-6 --flag-7 --flag-8 --flag-9 --flag-10 --flag-11",
        "Directory        /run",
        "Started          2026-10-19 19:16:11 PDT",
        "Host             login01",
        "Debug            on (LORRAX_DEBUG_PRINT=1)",
    ]


# ── devices ──────────────────────────────────────────────────────────────────
def test_devices_gpu():
    assert rb.devices_lines(GPU_FACTS, mesh_role="centroid axes X x Y",
                            extras=[("dense linalg", "local layout")]) == [
        "  MPI ranks        4 on 1 node",
        "  GPUs             4 x NVIDIA A100-SXM4-80GB, 1 per rank",
        "  mesh             2 x 2 (centroid axes X x Y)",
        "  CPU              32 cores per rank",
        "  host threads     OpenBLAS 16, OMP 8",
        "  device memory    card 85.09 GB, XLA pool 77.89 GB, 7.20 GB outside the pool",
        "  memory budget    66.35 GB per device",
        "  after warm-up    1.47 GB outside the pool (max over ranks)",
        "  JAX              0.9.1 / jaxlib 0.9.1, FP64 / complex128, NCCL collectives",
        "  dense linalg     local layout",
    ]


def test_devices_memory_warning():
    facts = dict(GPU_FACTS, device_memory=dict(GPU_FACTS["device_memory"], outside_max=6.0e9))
    lines = rb.devices_lines(facts, mesh_role=None)
    assert lines[6:10] == [
        "  after warm-up    6.00 GB outside the pool (max over ranks)",
        "  WARNING  memory outside the XLA pool is high"
        "                          [device_memory_outside_pool]",
        "           6.00 GB is already outside the pool after warm-up, above 0.75 of the "
        "7.20 GB outside the",
        "           reservation (runtime.pool_reservation_bytes)",
    ]
    assert not any(line.startswith("  mesh ") for line in lines)


def test_devices_cpu_without_launcher_facts():
    assert rb.devices_lines(CPU_FACTS, mesh_role="k axis over every device") == [
        "  MPI ranks        2",
        "  GPUs             none; 4 CPU devices, 2 per rank",
        "  mesh             2 x 2 (k axis over every device)",
        "  CPU              unknown cores per rank",
        "  host threads     runtime defaults",
        "  JAX              0.9.1 / jaxlib 0.9.1, FP64 / complex128, GLOO collectives",
    ]


# ── symmetry, time reversal, k points ────────────────────────────────────────
import numpy as np

_ROT = np.array([np.eye(3), np.diag([-1, 1, -1]), -np.eye(3), np.diag([1, -1, 1])], dtype=int)
_TAU = 2 * np.pi * np.array([[0, 0, 0], [0, 0.5, 0], [0, 0.5, 0], [0, 0, 0]])


def _sym(*, anti=(False,) * 4, active=(0, 1, 2, 3), trs=False, nk_red=9, qe=True):
    binding = NS(antiunitary=np.array(anti)) if qe else None
    return NS(Rinv_grid=_ROT, translations=_TAU, qe_symmetry_binding=binding,
              qe_operation_antiunitary=np.array(anti), active_symmetry_rows=np.array(active),
              trs_allowed=trs, nk_tot=25, nk_red=nk_red,
              qe_symmetry_diagnostic="no data-file-schema.xml beside WFN.h5")


def _magnetic_receipt(**kw):
    return NS(**dict(dict(method="occupied-density-subspace", trs_basis="qe-magnetic",
                          trs_holds=False, conclusive=False, subspace_residual=None,
                          trs_coverage=0.0, evidence_counts=()), **kw))


def test_symmetry_block_with_broken_time_reversal():
    sym = _sym()
    tr, note = rb.time_reversal_lines(_magnetic_receipt(), sym,
                                      consequence="GN probe: ordered orientations with the "
                                                  "time-odd channel, in its own pass")
    assert tr == [
        "  time reversal    not a symmetry alone: k and -k are sampled apart",
        "                   GN probe: ordered orientations with the time-odd channel, in its own pass"]
    assert note.kind == "NOTE" and note.id == "time_reversal_broken"
    closure = NS(n_sym=4, n_violating=0, n_centroids=6, closed=True)
    lines = rb.symmetry_lines(sym, centroids=NS(closure=closure),
                              tr_lines=tr + rb.message_lines(note, indent=2))
    assert lines == [
        "  operations       4, 2 with fractional translations; inversion S03",
        "  QE types         4 unitary, 0 composed with time reversal; all 4 used",
        "  centroid orbit   closed: 4/4 operations map the 6 sites onto themselves",
        "  time reversal    not a symmetry alone: k and -k are sampled apart",
        "                   GN probe: ordered orientations with the time-odd channel, in its own pass",
        rb.right_align("  NOTE     the DFT reference breaks time reversal (magnetic)",
                       "[time_reversal_broken]"),
        "           2-component density check: basis qe-magnetic, residual n/a, coverage "
        "0.00%, no",
        "           independent evidence",
        "           k and -k are not paired by time reversal",
        "  op    QE-TR  used as   R^-1                             tau      (r' = R^-1 r + tau)",
        "  S01   no     unitary   [ 1  0  0;  0  1  0;  0  0  1]   ( 0.000  0.000  0.000)",
        "  S02   no     unitary   [-1  0  0;  0  1  0;  0  0 -1]   ( 0.000  0.500  0.000)",
        "  S03   no     unitary   [-1  0  0;  0 -1  0;  0  0 -1]   ( 0.000  0.500  0.000)",
        "  S04   no     unitary   [ 1  0  0;  0 -1  0;  0  0  1]   ( 0.000  0.000  0.000)"]
    assert all(len(line) <= 100 for line in lines)


def test_time_reversal_variants_and_op_usage():
    magnetic = _sym(anti=(False, True, False, True), active=(0, 2, 5, 6, 7))
    tr, note = rb.time_reversal_lines(_magnetic_receipt(), magnetic)
    assert tr == ["  time reversal    not a symmetry alone: used only inside the 2 t_rev = 1 "
                  "operations"]
    assert note.text == "QE marks 2 of the 4 operations t_rev = 1 (magnetic)"
    assert note.lines[-1] == "k and -k pair only through those 2 operations"
    table = rb.symmetry_lines(magnetic)[2:]
    assert [row[8:24].split() for row in table[1:]] == [
        ["no", "unitary"], ["yes", "with", "TR"], ["no", "both"], ["yes", "with", "TR"]]
    scalar = NS(method="occupied-density-subspace", trs_basis="not-2c", nspin=1, nspinor=1)
    assert rb.time_reversal_lines(scalar, _sym(trs=True)) == (
        ["  time reversal    used (nspin 1, nspinor 1: no 2-component check)"], None)
    holds = _magnetic_receipt(trs_holds=True, conclusive=True)
    assert rb.time_reversal_lines(holds, _sym(trs=True))[0] == ["  time reversal    used"]
    assert rb.time_reversal_lines(None, _sym())[0] == [
        "  time reversal    not used (no density check retained)"]
    no_qe = rb.symmetry_lines(_sym(qe=False, active=(0, 1)))
    assert no_qe[1] == ("  QE types         unknown (no data-file-schema.xml beside WFN.h5); "
                        "2 of 4 used")
    assert no_qe[3].startswith("  S01   -      unitary") and no_qe[5].startswith("  S03   -      unused")


def test_kpoint_lines():
    wfn = NS(kgrid=(5, 5, 1), shift=(0.0, 0.0, 0.0), kweights=np.array([1.0, 2.0, 2.0]),
             kpoints=np.array([[0, 0, 0], [0, 0.2, 0], [-1e-9, 0.4, -0.5]]))
    assert rb.kpoint_lines(wfn, _sym(nk_red=3)) == [
        "  k grid           5 x 5 x 1, unshifted: 25 k in the BZ, 3 in the IBZ",
        "     ik         kx         ky         kz     weight",
        "      1    0.00000    0.00000    0.00000    0.20000",
        "      2    0.00000    0.20000    0.00000    0.40000",
        "      3    0.00000    0.40000   -0.50000    0.40000"]
    assert rb.kpoint_lines(wfn, _sym(nk_red=3), extra={"ngk": [284, 290, 301]})[1:3] == [
        "     ik         kx         ky         kz     weight   ngk",
        "      1    0.00000    0.00000    0.00000    0.20000   284"]
    shifted = NS(kgrid=(9, 9, 1), shift=(0.5, 0.5, 0.0))
    assert rb.kpoint_lines(shifted, NS(nk_tot=81, nk_red=81)) == [
        "  k grid           9 x 9 x 1, shift (0.500, 0.500, 0.000): 81 k, all stored "
        "(no IBZ reduction)"]
    many = NS(kgrid=(8, 8, 8), shift=(0, 0, 0), kweights=np.ones(150),
              kpoints=np.zeros((150, 3)))
    lines = rb.kpoint_lines(many, NS(nk_tot=512, nk_red=150))
    assert len(lines) == 2 + 20 + 1 and lines[-1] == "       [... 130 more k points]"


def test_report_sampling_counts_the_note_once(tmp_path, monkeypatch):
    report, shown = _report(tmp_path, monkeypatch)
    report.legacy_print("RuntimeWarning: SymMaps: the two-component DFT reference says GLOBAL "
                        "TIME-REVERSAL SYMMETRY IS BROKEN.")
    wfn = NS(trs_reference=_magnetic_receipt(), kgrid=(5, 5, 1), shift=(0, 0, 0),
             kweights=np.ones(9), kpoints=np.zeros((9, 3)))
    report.sampling(wfn=wfn, sym=_sym())
    report.close()
    assert shown[:3] == ["", "SYMMETRY", "--------"]
    assert "K POINTS" in shown
    assert [m.id for m in report.messages()] == ["time_reversal_broken"]
    assert report.messages()[0].text == "the DFT reference breaks time reversal (magnetic)"


# ── timing and memory ────────────────────────────────────────────────────────
def test_timing_lines_device_column_and_memory_summary():
    T = rb.TimingRow
    rows = [T("setup", 35.91), T("runtime bring-up", 18.33, 1), T("pre-main + imports", 17.58, 1),
            T("ISDF basis and Coulomb metric", 9.39, 0, 0.09), T("zeta", 4.56, 1, 0.09),
            T("V(q)", 1.48, 1, 0.02), T("other driver work", 33.23)]
    lines = rb.timing_lines(rows, wall=78.53, notes=["stage walls: host-attributed " * 5],
                            budget_gb=66.35, device_peak=(0.09, "zeta"),
                            host_peak=(2.38, 2.35), n_ranks=4)
    assert lines[:9] == [
        "  stage                                  wall (s)   fraction   device GB",
        "  setup                                     35.91     45.73%",
        "    runtime bring-up                        18.33     23.34%",
        "    pre-main + imports                      17.58     22.39%",
        "  ISDF basis and Coulomb metric              9.39     11.96%        0.09",
        "    zeta                                     4.56      5.81%        0.09",
        "    V(q)                                     1.48      1.88%        0.02",
        "  other driver work                         33.23     42.32%",
        "  total run                                 78.53    100.00%        0.09"]
    assert lines[9].startswith("  stage walls: host-attributed") and lines[10].startswith("    ")
    assert lines[-3:] == [
        "  budget           66.35 GB per device (memory_per_device_gb)",
        "  device peak      0.09 GB in zeta, max over 4 ranks (within the budget)",
        "  host peak        2.38 GB per rank, max over 4 ranks (min 2.35 GB)"]
    assert all(len(line) <= 100 for line in lines)
    bare = rb.timing_lines([T("runtime + imports", 1.0), T("stage", 1.0)], wall=2.0)
    assert bare == ["  stage                                  wall (s)   fraction",
                    "  runtime + imports                          1.00     50.00%",
                    "  stage                                      1.00     50.00%",
                    "  total run                                  2.00    100.00%"]


def test_report_timings_take_device_peaks_from_stage_rows(tmp_path, monkeypatch):
    report, shown = _report(tmp_path, monkeypatch)
    report.timings([("runtime + imports", 30.0), ("BSE eigensolve", 1.17, 0.42e9),
                    ("eigenvector write", 0.0, None)], wall=34.6)
    report.close()
    table = shown[shown.index("MAJOR-STAGE TIMING") + 2:]
    assert table[0].endswith("device GB") and table[2].endswith("0.42")
    assert "  device peak      0.42 GB in BSE eigensolve (within the budget)" in table
    assert "  budget           66.35 GB per device (memory_per_device_gb)" in table


def test_gw_stage_grouping():
    from gw.production_report import _group_stages
    names = ["runtime bring-up", "pre-main + imports", "input + run setup", "zeta", "V(q)",
             "minimax quadrature", "chi0", "W", "bank other", "spole rank synchronization",
             "Sigma rule plan", "Sigma tau setup", "SC Sigma rule refit", "mean-field load",
             "QP solve + diagonalize", "other driver work"]
    groups = _group_stages([(n, 1.0) for n in names])
    assert [(stage, len(rows)) for stage, rows in groups] == [
        ("setup", 3), ("ISDF basis and Coulomb metric", 2), ("Screening", 5),
        ("Self-energy", 3), ("Quasiparticle bands", 2), (None, 1)]


# ── settings ─────────────────────────────────────────────────────────────────
def test_settings_lines():
    S = rb.Setting
    rows = [S("compute_mode", "gn_ppm", "deck", "Godby-Needs plasmon pole, one pole per element"),
            S("--davidson-m-max", "1120", "derived", "10 x n_eig: subspace before restart"),
            S("W-av shells", "first on, second off", "derived", "neighbour shells"),
            S("memory_per_device_gb", "66.35", "derived",
              "card rule, A100-SXM4-80GB, linalg = local, " + "and more words " * 12)]
    lines = rb.settings_lines(rows[:2] + rows[3:])
    assert lines[:3] == [
        "  key                      value           source   effect",
        "  compute_mode             gn_ppm          deck     Godby-Needs plasmon pole, one pole per element",
        "  --davidson-m-max         1120            derived  10 x n_eig: subspace before restart"]
    assert lines[3].startswith("  memory_per_device_gb     66.35           derived  card rule,")
    assert len(lines) == 5 and lines[4].startswith(" " * 52 + "more") and lines[4].endswith("...")
    assert all(len(line) <= 100 for line in lines)
    wide = rb.settings_lines(rows)              # a 20-column value moves the effects to 57
    assert "  W-av shells              first on, second off derived  neighbour shells" in wide
    assert {line.index(effect) for line in wide for effect in ("Godby", "10 x", "neighbour")
            if effect in line} == {57}
    assert all(line[:57].isspace() and line[57] != " " for line in wide if line.startswith("   "))
    with pytest.raises(ValueError):
        rb.settings_lines([S("k", "v", "provenance", "e")])


def test_flag_source_and_soc_setting():
    from common.preprocessing_output import soc_setting
    from common.production_report import ProductionReport
    argv = ["-i", "bse.in", "--tda", "--solver=davidson"]
    assert ProductionReport.flag_source("--tda", argv=argv) == "cli"
    assert ProductionReport.flag_source("--solver", argv=argv) == "cli"
    assert ProductionReport.flag_source("--eqp", "--qp-rotations", argv=argv) == "default"
    assert soc_setting("j-RESOLVED, from QE <spinorbit>") == rb.Setting(
        "spin-orbit V_NL", "j-resolved", "QE", "from QE <spinorbit>")
    assert soc_setting("none (no projector setup)").value == "none"


def _gw_config(**over):
    from gw.gw_config import ComputeMode, HeadCorrection, QPSolver, ScreeningDiagrams
    base = dict(
        raw_input_keys=frozenset({"compute_mode", "qp_solver", "sys_dim", "restart"}),
        compute_mode=ComputeMode("gn_ppm"), compute_mode_raw="gn_ppm",
        qp_solver=QPSolver("one_shot_dft"), qp_solver_raw="one_shot_dft",
        screening=NS(diagrams=ScreeningDiagrams("w_rpa"), method="minimax"), sys_dim=3,
        head=NS(correction=HeadCorrection("full"), wcoul0_source="s_tensor"), bispinor=False,
        density_self_consistent=False, sc=NS(head_update="off"), no_degen_averaging=False,
        degen_avg_tol_ry=1e-6, restart=False,
        backend=NS(linalg="local", linalg_provenance="deck"),
        memory=NS(band_chunk_size=16, per_device_gb=0.0),
        mpa=NS(sampling_alpha=1, sampling_alpha_provenance="default for nonmetal"))
    base.update(over)
    return NS(**base)


def test_gw_settings_rows():
    from gw.production_report import resolved_notes, settings_rows
    rows = settings_rows(_gw_config(), n_mu=6, n_q_irr=9, processes=1, budget_gb=4.01,
                         device="NVIDIA A100-SXM4-80GB")
    assert rb.settings_lines(rows) == [
        "  key                      value           source   effect",
        "  compute_mode             gn_ppm          deck     Godby-Needs plasmon pole, one pole per element",
        "  qp_solver                one_shot_dft    deck     Sigma at E_DFT; H(QP) diagonalized once",
        "  screening_diagrams       w_rpa           default  RPA Dyson series, minimax chi0 time rules",
        "  sys_dim                  3               deck     3D bulk: no Coulomb truncation",
        "  head_correction          full            default  q->0 head with local fields (s_tensor)",
        "  degen_avg_tol_ry         1e-06           default  reported diagonals averaged within 1.4e-05 eV",
        "  restart                  false           deck     fresh ISDF fit; no restart bundle is read",
        "  linalg                   local           deck     9 whole N_mu^2 matrices per rank (0.00494 MiB)",
        "  memory_per_device_gb     4.01            derived  card rule on NVIDIA A100-SXM4-80GB",
        "  band_chunk_size          16              default  band panels gathered in chunks of at least this"]
    from gw.gw_config import QPSolver
    sc = settings_rows(_gw_config(qp_solver=QPSolver("self_consistent"),
                                  density_self_consistent=True,
                                  sc=NS(head_update="parallel_transport")),
                       n_mu=56, n_q_irr=4, processes=1)
    keys = {row.key: row for row in sc}
    assert keys["density_self_consistent"].source == "derived"
    assert keys["sc_head_update"].value == "parallel_transport"
    assert "degen_avg_tol_ry" not in keys and "memory_per_device_gb" not in keys
    assert resolved_notes([
        "  [config provenance] sc_head_update was not named: parallel_transport (x exists)",
        "  [config provenance] this deck names 1 LEGACY key: do_G0"]) == [
        "this deck names 1 LEGACY key: do_G0"]


# ── outputs ──────────────────────────────────────────────────────────────────
def test_output_file_lines_inputs_first_relative_paths():
    rows = [
        ("ISDF restart tensors", "written", "/scratch/run/tmp/isdf_tensors_6.h5"),
        ("input deck", "read", "/scratch/run/gnppm.in"),
        ("diagonal QP energies eqp1", "written", "/scratch/run/gnppm_eqp1.dat"),
        ("DFT wavefunctions", "read", "/global/cfs/WFN.h5"),
        ("restart bundle", "off", "-"),
    ]
    assert rb.output_file_lines(rows, directory="/scratch/run",
                                next_step="set centroids_file = c6.txt in the GW deck") == [
        "  role                      state       path",
        "  input deck                read        gnppm.in",
        "  DFT wavefunctions         read        /global/cfs/WFN.h5",
        "  ISDF restart tensors      written     tmp/isdf_tensors_6.h5",
        "  diagonal QP energies eqp1 written     gnppm_eqp1.dat",
        "  restart bundle            off         -",
        "  next             set centroids_file = c6.txt in the GW deck",
    ]


def test_output_long_state_and_gw_roles_stay_aligned():
    from gw.production_report import EQP0_FILE_ROLE, EQP1_FILE_ROLE, QP_WFN_FILE_ROLE
    rows = [("Galerkin basis", "read or published", "/r/galerkin_dft.h5"),
            (EQP0_FILE_ROLE, "written", "/r/gnppm_eqp0.dat"),
            (EQP1_FILE_ROLE, "written", "/r/gnppm_eqp1.dat"),
            (QP_WFN_FILE_ROLE, "absent", "/r/WFN_qp.h5")]
    lines = rb.output_file_lines(rows, directory="/r")
    assert {line.index(path) for line, path in zip(lines, (
        "path", "galerkin_dft.h5", "gnppm_eqp0.dat", "gnppm_eqp1.dat", "WFN_qp.h5"))} == {54}     # 2 + role 32 + 1 + state 18 + 1
    assert all(len(role) <= 40 for role, _, _ in rows)


def test_devices_thread_note_drops_the_library_path():
    facts = dict(CPU_FACTS, threads={"affinity": 6, "OPENBLAS_NUM_THREADS": (
        "6 (auto: physical cores of the affinity mask; applied to libscipy_openblas64_.so)")})
    assert rb.devices_lines(facts, mesh_role=None)[3] == \
        "  host threads     OpenBLAS 6 (auto: physical cores of the affinity mask)"


def test_output_role_width_bounds():
    short = rb.output_file_lines([("x", "written", "/r/a")], directory="/r")
    assert short == ["  role                     state       path",
                     "  x                        written     a"]
    long_role = "r" * 50
    lines = rb.output_file_lines([(long_role, "written", "/r/a")], directory="/r")
    assert lines[0] == "  role" + " " * 41 + "state       path"            # 44 wide, capped
    assert lines[1] == f"  {long_role} written     a"


# ── closing ──────────────────────────────────────────────────────────────────
def test_closing_lines():
    assert rb.closing_lines("gw", "COMPLETED", ended=ENDED, wall_s=78.53, n_warnings=1) == [
        RULE, "LORRAX gw COMPLETED 2026-10-19 19:16:55 PDT · 78.5 s · 1 warning"]
    assert rb.closing_lines("gw", "ERROR", ended=ENDED, wall_s=1119.0, n_warnings=2)[1] == \
        "LORRAX gw ERROR 2026-10-19 19:16:55 PDT · 18 min 39 s · 2 warnings"
    assert rb.closing_lines("dipole", "COMPLETED", ended=ENDED, wall_s=3720.0,
                            n_warnings=0)[1] == \
        "LORRAX dipole COMPLETED 2026-10-19 19:16:55 PDT · 1 h 02 min"
    with pytest.raises(ValueError):
        rb.closing_lines("gw", "REFUSED", ended=ENDED, wall_s=1.0, n_warnings=0)


# ── the report stream ────────────────────────────────────────────────────────
def _report(tmp_path, monkeypatch, cls=None):
    from common.preprocessing_output import ScientificProductionReport
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["bse.bse_jax", "-i", "bse.in"])
    monkeypatch.setenv("SLURM_JOB_ID", "60021274")
    monkeypatch.setenv("SLURM_STEP_ID", "6")
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "1")
    shown = []
    cls = cls or ScientificProductionReport
    report = cls(str(tmp_path / "bse.out"), runtime=NS(process_index=0, facts=GPU_FACTS),
                 debug=False, stdout=shown.append, driver="bse",
                 purpose="Bethe-Salpeter eigensolve")
    return report, shown


def test_report_writes_header_devices_files_and_closing(tmp_path, monkeypatch):
    report, shown = _report(tmp_path, monkeypatch)
    report.begin(input_file="bse.in")
    report.devices(mesh_role="BSE transition axes c x v")
    report.legacy_print("WARNING: one retained warning")
    report.legacy_print("component chatter")
    report.warnings()
    report.files([("input deck", "read", str(tmp_path / "bse.in")),
                  ("run report", "written", str(tmp_path / "bse.out")),
                  ("exciton energies", "written", str(tmp_path / "eigenvalues.dat"))])
    report.finish(wall=78.53)
    text = (tmp_path / "bse.out").read_text().splitlines()
    assert text == shown
    assert text[:4] == [RULE, "LORRAX bse · Bethe-Salpeter eigensolve", RULE,
                        "Command          python -m bse.bse_jax -i bse.in"]
    assert text[4] == f"Directory        {tmp_path}"
    assert re.fullmatch(r"Started          \d{4}-\d\d-\d\d \d\d:\d\d:\d\d \S+", text[5])
    assert text[6].startswith("Host             ") and \
        text[6].endswith(", Slurm job 60021274, step 6")
    assert text[7] == "Logs             bse.out"
    assert text[8:12] == ["", "DEVICES", "-------", "  MPI ranks        4 on 1 node"]
    files = text.index("OUTPUT FILES AND INPUTS")
    assert text[files + 2:files + 5] == [
        "  role                     state       path",
        "  input deck               read        bse.in",
        "  exciton energies         written     eigenvalues.dat"]
    assert rb.right_align("WARNING  one retained warning", "[unclassified]") in text
    warnings = text.index("WARNINGS")
    assert text[warnings + 2:warnings + 4] == [
        "  1 warning; each was printed in full where it happened",
        "  WARNING [unclassified]              setup   x1  one retained warning"]
    assert warnings < files                     # the summary, then the files table
    assert not any("chatter" in line for line in text)
    assert text[-2] == RULE
    assert re.fullmatch(r"LORRAX bse COMPLETED \d{4}-\d\d-\d\d \d\d:\d\d:\d\d \S+ · 78\.5 s"
                        r" · 1 warning", text[-1])
    assert all(len(line) <= 100 for line in text if not line.startswith("Directory"))
    assert report._stream is None                                   # closed


def test_report_error_closing_names_the_id_not_gate(tmp_path, monkeypatch):
    report, _ = _report(tmp_path, monkeypatch)
    report.finish(status="ERROR", wall=1119.0, error=ValueError(
        "GATE dipole_band_window: dipole.h5 was built for another band window; got: bands "
        "1-24 (nval 4, ncond 20); want: bands 1-34 (nval 4, ncond 30); fix: rerun dipole "
        "with this deck: python -m psp.get_dipole_mtxels -i deck.in; why: the head reads "
        "every band of the window"))
    text = (tmp_path / "bse.out").read_text().splitlines()
    assert text[-11:-2] == [
        "",
        "WARNINGS  none",
        "",
        "ERROR    dipole.h5 was built for another band window" + " " * 28
        + "[dipole_band_window]",
        "         found: bands 1-24 (nval 4, ncond 20)    this run needs: bands 1-34 "
        "(nval 4, ncond 30)",
        "         why: the head reads every band of the window",
        "         ValueError; the full message is on stderr",
        "         fix: rerun dipole with this deck: python -m psp.get_dipole_mtxels -i deck.in",
        ""]
    assert len(text[-8]) == 100
    assert re.fullmatch(r"LORRAX bse ERROR \S+ \S+ \S+ · 18 min 39 s", text[-1])
    assert not any("GATE" in line for line in text)


def test_error_without_a_gate_or_clauses():
    msg = rb.message_from_exception(RuntimeError("the solver diverged"))
    assert (msg.kind, msg.id, msg.text, msg.found, msg.fix) == (
        "ERROR", "unclassified", "the solver diverged", None, None)
    assert rb.message_from_exception(ValueError("GATE x_y: got: 1")).text == \
        "the run stopped with ValueError"


# ── messages and the WARNINGS summary ────────────────────────────────────────
def test_message_lines_three_kinds_one_shape():
    error = rb.Message("ERROR", "dipole_band_window", "dipole.h5 was built for another band window",
                       found=("dipole.h5", "bands 1-24 (nval 4, ncond 20)"),
                       needs="bands 1-34 (nval 4, ncond 30)",
                       fix="rerun dipole with this deck: python -m psp.get_dipole_mtxels -i deck.in")
    assert rb.message_lines(error) == [
        "ERROR    dipole.h5 was built for another band window"
        "                            [dipole_band_window]",
        "         dipole.h5: bands 1-24 (nval 4, ncond 20)    this run needs: bands 1-34 "
        "(nval 4, ncond 30)",
        "         fix: rerun dipole with this deck: python -m psp.get_dipole_mtxels -i deck.in"]
    warning = rb.Message("WARNING", "slab_single_stripe",
                         "the bundle file has one Lustre stripe (speed only)",
                         fix="lfs setstripe -c 8 tmp/ before the GW run writes the bundle")
    assert rb.message_lines(warning, indent=6) == [
        "      WARNING  the bundle file has one Lustre stripe (speed only)"
        "               [slab_single_stripe]",
        "               fix: lfs setstripe -c 8 tmp/ before the GW run writes the bundle"]
    note = rb.Message("NOTE", "soc_j_averaged",
                      "Si.upf is fully relativistic; QE ran without spin-orbit",
                      lines=("V_NL is j-averaged, as in QE (average_pp): dD = 0.148261 Ry "
                             "(2.0172 eV) dropped",))
    assert rb.message_lines(note, indent=2) == [
        "  NOTE     Si.upf is fully relativistic; QE ran without spin-orbit"
        "                  [soc_j_averaged]",
        "           V_NL is j-averaged, as in QE (average_pp): dD = 0.148261 Ry (2.0172 eV) "
        "dropped"]
    assert all(len(line) == 100 for line in (rb.message_lines(error)[0],
                                              rb.message_lines(warning, indent=6)[0]))
    with pytest.raises(ValueError):
        rb.message_lines(rb.Message("REFUSED", "x", "y"))


def test_message_long_text_and_found_wrap():
    msg = rb.Message("WARNING", "an_id", "word " * 30, found=("f", "a" * 60), needs="b" * 60)
    lines = rb.message_lines(msg)
    assert lines[0].endswith("  [an_id]") and len(lines[0]) <= 100
    assert all(line.startswith(" " * 9) for line in lines[1:])
    assert lines[-2:] == ["         f: " + "a" * 60, "         this run needs: " + "b" * 60]
    assert all(len(line) <= 100 for line in lines)


def test_warnings_summary_lines():
    msgs = [rb.Message("WARNING", "centroid_window_short", "centroids not placed for the zeta fit"),
            rb.Message("NOTE", "time_reversal_broken", "magnetic QE run: k and -k sampled apart"),
            rb.Message("NOTE", "minimax_runtime_solve", "13-node chi0 rule solved in this run",
                       stage="[2/4]"),
            rb.Message("NOTE", "minimax_runtime_solve", "13-node chi0 rule solved in this run",
                       stage="[3/4]")]
    assert rb.warnings_summary_lines(msgs) == [
        "", "WARNINGS", "--------",
        "  1 warning, 2 notes; each was printed in full where it happened",
        "  WARNING [centroid_window_short]     setup   x1  centroids not placed for the zeta fit",
        "  NOTE    [time_reversal_broken]      setup   x1  magnetic QE run: k and -k sampled apart",
        "  NOTE    [minimax_runtime_solve]     [2/4]   x2  13-node chi0 rule solved in this run"]
    assert rb.warnings_summary_lines([]) == ["", "WARNINGS  none"]
    assert rb.warning_count(msgs) == 1
    long = rb.warnings_summary_lines([rb.Message("WARNING", "unclassified", "x " * 80),
                                      rb.Message("WARNING", "unclassified", "other text")])
    assert long[3] == "  2 warnings; each was printed in full where it happened"
    assert len(long[4]) == 100 and long[4].endswith("...")


def test_legacy_warnings_get_ids_and_print_at_the_site(tmp_path, monkeypatch):
    report, shown = _report(tmp_path, monkeypatch)
    report.legacy_print(
        "RuntimeWarning: centroid file /run/centroids_frac_6.txt: selection window left=(0, 3) "
        "ends below the ζ left leg's top 5. The ζ fit still trains on left=(0, 5) x right=(0, "
        "7); for best placement regenerate the centroids with centroid.kmeans_cli -i <deck>.")
    report.progress("[stage 12:10:58] -> screening (chi0 -> W)")
    for _ in range(2):
        report.legacy_print(
            "RuntimeWarning: minimax: SOLVED noncrossing/inverse R=119.051 target 5e-09 "
            "n_max=64 -> 13 nodes, max_err 1.123e-09, sum|w| 11.92, kappa0 1 | runtime solve "
            "UNCERTIFIED. Solved here at run time.")
    report.legacy_print(
        "RuntimeWarning: SymMaps: the two-component DFT reference says GLOBAL TIME-REVERSAL "
        "SYMMETRY IS BROKEN. The authenticated QE schema still authorizes 0 rows.")
    report.legacy_print("WARNING: GATE pt_head_window_hybridized: the window edge cuts")
    report.close()
    start = shown.index(rb.right_align("WARNING  centroids not placed for the zeta fit",
                                       "[centroid_window_short]"))
    assert shown[start + 1:start + 4] == [
        "         centroids_frac_6.txt: selected on left 1-3    this run needs: left window up "
        "to band 5",
        "         the zeta fit still trains on left 1-5 x right 1-7",
        "         fix: python -m centroid.kmeans_cli -i <deck>"]
    assert rb.right_align("NOTE     13-node chi0 time rule solved in this run",
                          "[minimax_runtime_solve]") in shown
    assert not any("TIME-REVERSAL" in line for line in shown)      # quiet: SYMMETRY prints it
    assert rb.right_align("WARNING  the window edge cuts", "[pt_head_window_hybridized]") in shown
    assert not any("GATE" in line for line in shown)
    summary = rb.warnings_summary_lines(report.messages())
    assert summary[3] == "  2 warnings, 2 notes; each was printed in full where it happened"
    # id column 28 (the longest id + 3); the stage is its name without the detail
    assert summary[6] == ("  NOTE    [minimax_runtime_solve]      screening   x2  "
                          "13-node chi0 time rule solved in this run")
    assert summary[4] == ("  WARNING [centroid_window_short]      setup       x1  "
                          "centroids not placed for the zeta fit")
    assert summary[7].startswith("  NOTE    [time_reversal_broken]       screening   x1  ")


def test_device_memory_warning_is_counted_once(tmp_path, monkeypatch):
    report, shown = _report(tmp_path, monkeypatch)
    report.runtime.facts = dict(GPU_FACTS, device_memory=dict(GPU_FACTS["device_memory"],
                                                              outside_max=6.0e9))
    report.devices(mesh_role=None)
    report.finish(wall=1.0)
    assert sum("[device_memory_outside_pool]" in line for line in shown) == 2   # site + summary
    assert shown[-1].endswith(" · 1 warning")


def test_gw_report_keeps_its_receipts_and_warning_words(tmp_path, monkeypatch):
    from gw.production_report import GWProductionReport
    monkeypatch.chdir(tmp_path)
    shown = []
    report = GWProductionReport(str(tmp_path / "gnppm.out"),
                                runtime=NS(process_index=0, facts=GPU_FACTS),
                                debug=False, stdout=shown.append)
    report.legacy_print("  Z store: host, 2 tiles")
    report.legacy_print("[WARN] dropped by gw")
    report.legacy_print("WARNING: kept")
    report.finish(wall=2.0)
    assert shown[:2] == ["  Z store: host, 2 tiles",
                         rb.right_align("WARNING  kept", "[unclassified]")]
    assert shown[shown.index("WARNINGS") + 3] == \
        "  WARNING [unclassified]              setup   x1  kept"
    assert re.fullmatch(r"LORRAX gw COMPLETED \S+ \S+ \S+ · 2\.0 s · 1 warning", shown[-1])


def test_gw_devices_extras(tmp_path, monkeypatch):
    from gw.production_report import GWProductionReport
    monkeypatch.chdir(tmp_path)
    facts = dict(GPU_FACTS, ffi_platform="CUDA", ffi_dials=[
        {"env": "LORRAX_BANDS_GEMM_FFI", "platforms": ("CUDA",), "enabled": True,
         "target": "cuBLAS batched GEMM FFI"},
        {"env": "LORRAX_OTHER_FFI", "platforms": ("CUDA",), "enabled": True}])
    shown = []
    report = GWProductionReport(str(tmp_path / "gnppm.out"),
                                runtime=NS(process_index=0, facts=facts),
                                debug=False, stdout=shown.append)
    report.devices(config=NS(backend=NS(linalg="local")))
    report.close()
    assert shown[:4] == ["", "DEVICES", "-------", "  MPI ranks        4 on 1 node"]
    assert shown[-3:] == [
        "  dense linalg     local layout",
        "  k-convolutions   CUDA, nvidia-mathdx fused cuFFTDx kernels (zeta fit and Sigma)",
        "  G(tau) GEMM      cuBLAS batched GEMM FFI (LORRAX_BANDS_GEMM_FFI = ON)"]


def test_htransform_header_keeps_the_energy_source(tmp_path, monkeypatch):
    from bandstructure.production_report import HTransformProductionReport
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["bandstructure.htransform", "-i", "h.in"])
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    shown = []
    report = HTransformProductionReport(str(tmp_path / "htransform.out"),
                                        runtime=NS(process_index=0, facts=GPU_FACTS),
                                        debug=False, stdout=shown.append)
    report.begin(input_file="h.in", output_file="bands.dat",
                 energy_source="the WFN's eigenvalues")
    report.close()
    assert shown[1] == "LORRAX htransform · Hamiltonian interpolation"
    assert shown[3] == "Command          python -m bandstructure.htransform -i h.in"
    assert shown[-2:] == ["Logs             htransform.out",
                          "Energy source    the WFN's eigenvalues"]

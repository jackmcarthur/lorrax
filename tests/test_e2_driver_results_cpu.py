"""Golden RESULTS blocks of the BSE and preprocessing drivers (slice E2).

Every block is a pure function from host values to lines, so these tests pin
the text without running a calculation: kin-ion, dipole, BSE, exciton bands,
htransform and kmeans, plus the one-row htransform receipts and the stage
sequence the long drivers open their ``[i/N]`` banners through.
"""
import logging
import re
from types import SimpleNamespace as NS

import numpy as np
import pytest

from bandstructure.production_report import (band_table_lines, guard_rows,
                                             htransform_results_lines, path_text)
from bse.report_lines import (bse_results_lines, exciton_band_results_lines,
                              exciton_energy_lines, path_node_lines,
                              window_direct_gap)
from centroid.production_output import kmeans_results_lines, kmeans_settings_rows
from common.preprocessing_output import (StageSequence, degeneracy_check_text,
                                         dipole_result_rows, galerkin_rows,
                                         kin_ion_result_rows, kv_block_lines,
                                         locality_rows, parallel_transport_text,
                                         results_lines, system_lines)

#: The harness's BSE eigenvalue pattern (tests/hsuite/chain.py, "bse").
HARNESS_S_ROW = re.compile(r"^\s*S\d+\s+([0-9.+-]+)\s*$")


def _fits(lines):
    assert all(len(line) <= 100 for line in lines), [l for l in lines if len(l) > 100]
    return lines


# ── shared rows ──────────────────────────────────────────────────────────────
def test_kv_block_wraps_under_its_value_and_keeps_line_breaks():
    assert kv_block_lines([("k path", "3 points\nN01  G")]) == [
        "  k path           3 points", "                   N01  G"]
    wrapped = kv_block_lines([("x", "word " * 30)])
    assert wrapped[0].startswith("  x                word")
    assert all(line.startswith(" " * 19) for line in wrapped[1:])
    assert results_lines([]) == ["", "RESULTS", "-------"]


def test_system_block():
    assert _fits(system_lines(
        natoms=2, species=["H"], fft_grid=(15, 18, 20),
        pseudopotentials=[("H", "H.upf, fully-relativistic, z_val 1, NLCC no "
                                "(ONCVPSP-3.3.0)")],
        rows=[("spin", "nspin 1, 2-component spinors")])) == [
        "  atoms            2 (H)",
        "  pseudopotential  H.upf, fully-relativistic, z_val 1, NLCC no (ONCVPSP-3.3.0)",
        "  FFT grid         15 x 18 x 20",
        "  spin             nspin 1, 2-component spinors"]


# ── kin-ion and dipole ───────────────────────────────────────────────────────
def test_kin_ion_results_golden():
    lines = results_lines(kin_ion_result_rows(
        out_name="kin_ion_bisp.h5", bands="1-5", nk_stored=9, nk_full=25,
        soc="j-resolved, from QE <spinorbit>",
        degeneracy=dict(n_manifolds=12, n_split=0, max_split_ry=3.0e-9, clean=True)))
    assert _fits(lines) == [
        "", "RESULTS", "-------",
        "  kin_ion_bisp.h5  T + V_loc + V_NL on bands 1-5, 9 star-wedge k (the 25 "
        "full-BZ k are rebuilt on",
        "                   read)",
        "  V_H              not stored; gw_jax builds it live in G-space",
        "  V_NL             j-resolved, from QE <spinorbit>",
        "  degeneracy test  consistent: 12 degenerate manifolds, max split 0.0000 meV"]


def test_degeneracy_verdicts():
    assert degeneracy_check_text(None) == "not run"
    assert degeneracy_check_text(dict(n_manifolds=8, n_split=2, max_split_ry=1e-4,
                                      clean=False)) == (
        "SPLIT: 2 of 8 degenerate manifolds, worst 1.3606 meV (see the WARNING above)")


def test_dipole_results_with_the_parallel_transport_verdict():
    gauge = parallel_transport_text(requested=True, written=True, lines=[
        "PARALLEL-TRANSPORT VALIDATION",
        "Covariant DFT velocity: above rtol (diagnostic); relative L2 on the head's "
        "elements (transitions, Fermi-surface diagonal)=1.6371e-01 (the SC head's "
        "link error); max abs=6.37431e-01"])
    assert gauge == ("parallel_transport.h5 written; covariant velocity above rtol "
                     "(diagnostic), head relative L2 1.6371e-01")
    lines = results_lines(dipole_result_rows(
        out_name="dipole.h5", bands="1-13", nk=27,
        velocity="p + i[V_NL, r] = i[H, r]; analytic V_NL derivative",
        vnl="j-averaged, from QE <spinorbit>", parallel_transport=gauge))
    assert _fits(lines)[3:] == [
        "  dipole.h5        q = 0 velocity <mk|v|nk> on bands 1-13, 27 k (full BZ), "
        "with the band energies",
        "  velocity         p + i[V_NL, r] = i[H, r]; analytic V_NL derivative",
        "  V_NL             j-averaged, from QE <spinorbit>",
        "  parallel gauge   parallel_transport.h5 written; covariant velocity above "
        "rtol (diagnostic), head",
        "                   relative L2 1.6371e-01"]


def test_parallel_transport_outcomes():
    assert parallel_transport_text(requested=False, written=False) is None
    assert parallel_transport_text(
        requested=True, written=False, dropped="pt_collapsed_axis_cut_density") == (
        "parallel_transport.h5 not written [pt_collapsed_axis_cut_density]; see the "
        "NOTE above")
    assert parallel_transport_text(
        requested=True, written=False,
        dropped="the k grid (5, 5, 1) has no link stencil on (2,)") == (
        "parallel_transport.h5 not written: the k grid (5, 5, 1) has no link "
        "stencil on (2,)")
    assert parallel_transport_text(requested=True, written=True,
                                   velocity_only=True).endswith("no links")


# ── BSE ──────────────────────────────────────────────────────────────────────
def test_bse_results_golden_keeps_the_harness_rows():
    eps_v = np.array([[-0.10, -9.0]] * 25)          # one physical band + a pad
    eps_c = np.array([[0.05, 0.20]] * 25)
    eps_c[4, 0] = 0.02
    gap_ry, ik = window_direct_gap(eps_v, eps_c, n_val=1, n_cond=2)
    assert (round(gap_ry, 12), ik) == (0.12, 5)
    lines = bse_results_lines(np.array([-0.94399, -0.78215]),
                              gap=(gap_ry * 13.6056980659, ik, "DFT"))
    assert _fits(lines) == [
        "", "RESULTS", "-------",
        "  state       energy (eV)",
        "  S001         -0.94399",
        "  S002         -0.78215",
        "",
        "  DFT gap          1.63268 eV, the lowest direct transition in the window (k 5)",
        "  binding (S1)     2.57667 eV below the DFT gap",
        "  eigenvectors     not written (--write-eigs)"]
    found = [float(m.group(1)) for m in map(HARNESS_S_ROW.match, lines) if m]
    assert found == [-0.94399, -0.78215]


def test_bse_oscillator_rows_never_read_as_energy_rows():
    lines = bse_results_lines(
        np.array([-0.94399, -0.78215]), dipoles=np.array([[1e-2, 2e-2j, 0], [0, 1e-3, 0]]),
        n_k=25, eigenvectors="eigenvectors.h5, 2 states with <0|r|S>",
        extra_rows=(("eigensolve", "stopped after 12 of 40 iterations"),))
    assert "  S001      4.0000e-06    1.6000e-05    0.0000e+00" in lines
    assert [l for l in lines if HARNESS_S_ROW.match(l)] == exciton_energy_lines(
        [-0.94399, -0.78215])[1:]
    assert lines[-1] == "  eigensolve       stopped after 12 of 40 iterations"


# ── exciton bands ────────────────────────────────────────────────────────────
Q3 = np.array([[0, 0, 0], [0.4, 0, 0], [0.4, 0.4, 0]])


def test_exciton_band_results_golden():
    lines = exciton_band_results_lines(
        np.array([[-0.94397], [-1.11099], [-0.86284]]), Q3, [0, 1, 2], ["G", "X", "M"],
        hermiticity="PASS (worst relative 2.89e-12, tol 1e-09)",
        certificate="not applicable to this exchange route",
        files="exciton.dat (all levels), exciton.png")
    assert _fits(lines) == [
        "", "RESULTS", "-------",
        "    iQ             Qx       Qy       Qz          E_1   (eV)",
        "     1    G   0.00000  0.00000  0.00000     -0.94397",
        "     2    X   0.40000  0.00000  0.00000     -1.11099",
        "     3    M   0.40000  0.40000  0.00000     -0.86284",
        "",
        "  levels           1 per Q at 3 path points",
        "  lowest E_S       -1.11099 eV at X (iQ 2)",
        "  energy range     -1.11099 .. -0.86284 eV",
        "  Hermiticity      PASS (worst relative 2.89e-12, tol 1e-09)",
        "  refit check      not applicable to this exchange route",
        "  written          exciton.dat (all levels), exciton.png"]
    assert path_node_lines(Q3, [0, 1, 2], ["G", "X", "M"])[1] == (
        "N02    X  Q = ( 0.40000  0.00000  0.00000)  path index 2")


def test_long_exciton_path_lists_its_nodes_only():
    levels = np.linspace(0.0, 1.0, 50 * 6).reshape(50, 6)
    path = np.zeros((50, 3))
    lines = exciton_band_results_lines(
        levels, path, [0, 49], ["G", "X"], n_extra=2, hermiticity="PASS",
        certificate="-", files="-")
    assert lines[3].endswith("E_4   ...   (eV)")
    assert lines[6] == "  [labelled nodes only; all 50 path points are in the .dat file]"
    assert "(+2 extra Q, in the .dat file)" in lines[8]


# ── htransform ───────────────────────────────────────────────────────────────
def test_htransform_results_golden():
    energies = np.array([[-2.33, -1.0, -0.5, 0.0, 1.2], [-2.0, -0.9, -0.3, -0.1, 1.5],
                         [-1.9, -0.8, -0.4, -0.05, 5.01035]])
    enk = np.zeros((6, 25))
    enk[4] = np.linspace(0.0, 0.03, 25)
    enk[5] = np.linspace(0.0127, 0.0779, 25)
    lines = htransform_results_lines(
        energies_ev=energies, band_start=0, n_val=3, n_keep=5, n_fit=6,
        on_grid=(3, 1.2e-5, 8e-6), guard=guard_rows(enk, start=0, keep=5, fit=6),
        energy_source="the WFN's eigenvalues (QP for a WFN_qp.h5)",
        written="htransform.dat (band, k, energy in eV on the path)")
    assert _fits(lines) == [
        "", "RESULTS", "-------",
        "    band    min (eV)    max (eV)  width (eV)   (on the path, relative to the VBM)",
        "       1    -2.33000    -1.90000     0.43000",
        "       2    -1.00000    -0.80000     0.20000",
        "       3    -0.50000    -0.30000     0.20000  VBM",
        "       4    -0.10000    +0.00000     0.10000  CBM",
        "       5    +1.20000    +5.01035     3.81035",
        "",
        "  on-grid error    max |dE| 0.0120 meV, RMS 0.0080 meV over 3 path points on "
        "the coarse grid",
        "  bands            1-5 returned, 1-6 fitted (1 guard)",
        "  guard bands      6-6: 0.17279 .. 1.05988 eV on the coarse grid; at least "
        "0.17279 eV above band 5",
        "                   at the same k; the ranges overlap by 0.23538 eV",
        "  energy range     -2.33000 .. +5.01035 eV relative to the VBM",
        "  energy source    the WFN's eigenvalues (QP for a WFN_qp.h5)",
        "  written          htransform.dat (band, k, energy in eV on the path)"]


def test_htransform_without_on_grid_points_and_a_wide_window():
    lines = htransform_results_lines(
        energies_ev=np.zeros((2, 0)), band_start=0, n_val=1, n_keep=0, n_fit=0,
        on_grid=None, energy_source="-", written="-")
    assert lines[3] == "  on-grid error    no path point lies on the coarse grid"
    table = band_table_lines(np.zeros((2, 40)), band_start=10, n_val=20)
    assert len(table) == 1 + 12 + 1 and table[-1].startswith("  [6 bands on each side")
    assert path_text(None, [], []).startswith("none")


# ── kmeans ───────────────────────────────────────────────────────────────────
def test_kmeans_results_golden():
    lines = kmeans_results_lines(
        requested=6, candidates=9, pool=9, written=6, out_file="centroids_frac_6.txt",
        orbit_aware=True, n_sym=4, windows=((0, 3), (0, 7)),
        window_label="explicit feature pair", density_mode="scalar", prune_rank=6,
        unpicked=2.01e4, rank_law=(66.0, 1.5, 3.5, (131.2, 696, 2118)))
    assert _fits(lines) == [
        "", "RESULTS", "-------",
        "  centroids        6 written to centroids_frac_6.txt (6 requested)",
        "  selection        6 requested -> 9 k-means candidates -> 9-point pool -> 6 written",
        "  orbit closure    whole orbits of 4 spatial operations (decorated atoms); "
        "snapped to the FFT grid",
        "  sampling metric  sqrt(diag q=0 charge Gram) on left 1-3 x right 1-7 "
        "(explicit feature pair), unit",
        "                   band weights",
        "  pair-space rank  NOT established: the point budget stopped the select at 6 "
        "points of the 9-point",
        "                   pool (largest unpicked residual 2.01e+04 x the floor); 6 "
        "kept pivots",
        "  rank law         N_mu(1 meV RMS Sigma_x) ~ 66 at L=1.5, B=3.5; "
        "written/estimate 0.09"]


def test_kmeans_settings_rows():
    args = NS(N_c=6, oversample=1.5, density_mode="scalar", seed=42, rho_power=1.0,
              fit_window="0:3,0:7", prune_window="v_x_vc", no_orbit=False, orbit=True)
    rows = kmeans_settings_rows(args, source=lambda *flags: "cli")
    assert [(r.key, r.value, r.source) for r in rows] == [
        ("N_c", "6", "cli"), ("oversample", "1.5", "cli"),
        ("density_mode", "scalar", "cli"), ("seed", "42", "cli"),
        ("fit_window", "0:3,0:7", "cli"), ("symmetry closure", "orbits", "cli")]
    assert all(len(r.effect) <= 49 for r in rows)
    assert rows[1].effect == "9 k-means candidates, pruned by pivoted Cholesky"


# ── htransform receipts, one row each ────────────────────────────────────────
def test_galerkin_and_locality_rows():
    assert galerkin_rows(dict(
        method="kmeans_nodes", retained_rank=12, nodes=6, centroids=6, orbits=2,
        stacked_states=150, metric_eps=0.8646, zero_weight_nodes=0)) == [
        ("Galerkin basis", "nodal rank 12 on 6 orbit-closed nodes (2 orbits, the ISDF "
                           "centroids); 150 states fitted"),
        ("node metric", "eps 8.65e-01 (max_k ||Phi^H W Phi - 1||_2), 0 zero-weight nodes")]
    assert locality_rows(dict(row_isometry_max=5.1e-15, outer_shell_l2_fraction=0.0514,
                              outer_shell_max_over_r0=0.0188)) == [
        ("locality", "outer shell 5.14% of ||f(H)_R||_F, largest shell 1.88% of R=0; "
                     "row isometry 5.1e-15")]


# ── the stage sequence ───────────────────────────────────────────────────────
def test_stage_sequence_opens_banners_and_results_in_order(caplog):
    with caplog.at_level(logging.INFO, logger="lorrax.stage"):
        with StageSequence(2) as stages:
            stages.begin(1, "First stage", name="e2.test.first")
            stages.result("one thing")
            stages.begin(2, "Second stage", name="e2.test.second")
    lines = [r.getMessage() for r in caplog.records if r.name == "lorrax.stage"]
    banners = [l for l in lines if re.match(r"^\[\d/2\] ", l)]
    results = [l for l in lines if l.lstrip().startswith("=>")]
    assert [b.split("  ")[0] for b in banners] == ["[1/2] First stage", "[2/2] Second stage"]
    assert results[0].startswith("      => one thing") and results[1].startswith("      => done")
    assert all(len(l) == 100 for l in banners + results)


def test_stage_sequence_never_swallows_an_exception():
    with pytest.raises(RuntimeError):
        with StageSequence(1) as stages:
            stages.begin(1, "Failing stage", name="e2.test.fail")
            raise RuntimeError("boom")

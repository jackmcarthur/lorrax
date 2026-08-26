"""Shared scientific-output vocabulary used by multiple drivers."""

from types import SimpleNamespace

import numpy as np

from common.scientific_output import (
    architecture_lines,
    band_range,
    centroid_orbit_line,
    policy,
    symmetry_sampling_lines,
)


def test_band_ranges_and_auto_policies_are_human_facing():
    assert band_range(0, 8) == "1-8"
    assert band_range(8, 8) == "none"
    assert policy("auto", ("auto", "local", "distributed")) == (
        "auto (other choices: local, distributed)")
    assert policy("distributed", ("auto", "distributed")) == "distributed"


def test_architecture_is_shared_and_compact():
    runtime = SimpleNamespace(facts={
        "process_count": 4,
        "n_devices": 4,
        "n_local_devices": 1,
        "backend": "gpu",
        "device_kind": "A100",
        "mesh_shape": (2, 2),
        "threads": {"affinity": 16, "OMP_NUM_THREADS": "8"},
    })
    text = "\n".join(architecture_lines(runtime, mesh_role="matrix axes"))
    assert "MPI ranks      : 4" in text
    assert "Processor mesh : 2 x 2  (matrix axes)" in text
    assert "OMP_NUM_THREADS=8" in text


def test_centroid_orbit_receipt_is_compact_and_names_open_set_residual():
    closed = SimpleNamespace(closure=SimpleNamespace(
        closed=True, n_sym=48, n_violating=0, n_centroids=960,
        tol=1.0e-5, worst_residual=1.0e-6, worst_op=7))
    opened = SimpleNamespace(closure=SimpleNamespace(
        closed=False, n_sym=48, n_violating=47, n_centroids=960,
        tol=1.0e-5, worst_residual=0.1318, worst_op=12))

    assert centroid_orbit_line(closed) == (
        "Centroid orbit: CLOSED : 48/48 spatial operations preserve 960 "
        "sites (tol=1.00000e-05)")
    open_line = centroid_orbit_line(opened)
    assert "NOT CLOSED : 47/48 spatial operations" in open_line
    assert "worst residual 1.31800e-01 at S13" in open_line
    assert "full BZ" not in open_line and "IBZ" not in open_line


def test_symmetry_receipt_fractional_ops_and_ibz_use_five_decimals():
    receipt = SimpleNamespace(
        trs_holds=True, trs_basis="measured", m_rel=1.42e-13,
        trs_coverage=1.0, trs_implied_by_mesh=True,
        spatial_residual=np.array([0.0, 3.0e-14]),
    )
    wfn = SimpleNamespace(
        density_symmetry=receipt,
        kgrid=np.array([2, 2, 2]), shift=np.array([-1.0e-8, 0.0, 0.0]),
        kpoints=np.array([[-1.0e-8, 0.0, 0.0], [0.25, 0.25, 0.25]]),
        kweights=np.array([1.0, 3.0]),
    )
    sym = SimpleNamespace(
        Rinv_grid=np.array([np.eye(3, dtype=int), np.eye(3, dtype=int)]),
        translations=np.array([[0.0, 0.0, 0.0],
                               [np.pi, np.pi, np.pi]]),
        trs_allowed=True, nk_tot=8, nk_red=2,
    )
    text = "\n".join(symmetry_sampling_lines(wfn, sym))
    assert "1 with fractional translations" in text
    assert "fractional tau" not in text
    assert "tau=(" in text
    assert "Time reversal  : HOLDS (measured" in text
    assert "coverage=100.00%" in text
    assert "TRS unfolding  : enabled from the measured density verdict" in text
    assert " 0.25000   0.25000   0.25000   0.75000" in text
    assert ".250000" not in text
    assert "-0.00000" not in text

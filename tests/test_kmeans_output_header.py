"""Contract tests for the scientific provenance on centroid coordinate files."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from centroid.pivoted_cholesky import _gram_meta_band_counts
from centroid.production_output import (
    format_centroid_header,
    format_kmeans_report,
    prune_band_ranges,
)


def test_centroid_header_names_density_band_space_and_selection(tmp_path):
    source = tmp_path / "WFN.h5"
    header = format_centroid_header(
        feature_fit="bands 1-40 (indices [0,40)): sum_n sum_k w_k |psi_nk(r)|^2",
        source_wfn=str(source),
        weight_label="band-range density",
        num_electrons=16.0,
        occupied_boundary=8,
        fft_grid=(12, 12, 36),
        kgrid=(3, 3, 1),
        shift=(0.0, 0.0, 0.0),
        seed=42,
        rho_power=0.5,
        requested=400,
        candidates=800,
        written=399,
        pruning="pivoted Cholesky",
        prune_rank=399,
        prune_left=(0, 8),
        prune_right=(0, 40),
        prune_label="valence x (valence + conduction)",
        orbit_aware=True,
        n_sym=12,
        density_mode="scalar",
    )

    assert header.startswith("LORRAX ISDF centroid coordinates")
    assert "feature fit: bands 1-40 (indices [0,40))" in header
    assert f"source wavefunctions: {source}" in header
    assert "electrons: 16; occupied-band boundary: 8" in header
    assert "requested=400; candidates=800; written=399" in header
    assert "achieved numerical rank=399" in header
    assert "left=(0, 8), right=(0, 40)" in header
    assert "spatial operations=12" in header


def test_prune_header_and_executor_share_one_window_resolver():
    args = SimpleNamespace(prune_window="v_x_vc")
    assert prune_band_ranges(args, 8, 32) == (
        (0, 8), (0, 40), "valence x (valence + conduction)")


def test_explicit_fit_window_is_independent_of_physical_occupancy():
    args = SimpleNamespace(
        prune_window="v_x_vc", fit_window="0:16,0:28")
    assert prune_band_ranges(args, 8, 20) == (
        (0, 16), (0, 28), "explicit feature pair")
    assert _gram_meta_band_counts(8, 28, None, None) == (8, 20)


def test_explicit_fit_window_refuses_ambiguous_or_out_of_wfn_ranges():
    with pytest.raises(ValueError, match="non-default"):
        prune_band_ranges(SimpleNamespace(
            prune_window="v_x_c", fit_window="0:16,0:28"), 8, 20)
    with pytest.raises(ValueError, match="lie in"):
        prune_band_ranges(SimpleNamespace(
            prune_window="v_x_vc", fit_window="0:16,0:29"), 8, 20)


def test_kmeans_report_is_compact_scientific_output(tmp_path):
    runtime = SimpleNamespace(facts={
        "process_count": 4,
        "n_devices": 4,
        "backend": "gpu",
        "device_kind": "NVIDIA A100-SXM4-40GB",
        "mesh_shape": (2, 2),
        "jax_version": "0.9.1",
        "jaxlib_version": "0.9.1",
        "x64": True,
    })
    text = format_kmeans_report(
        header="feature fit: bands 1-40\nselection: weighted k-means",
        source_wfn=str(tmp_path / "WFN.h5"),
        centroid_file=str(tmp_path / "centroids_frac_20.txt"),
        report_file=str(tmp_path / "kmeans.out"),
        wfn_backend="phdf5", elapsed_s=12.5, runtime=runtime,
        timing_records=(
            {"name": "setup.wfn_io", "path": ("setup.wfn_io",),
             "inclusive": 1.0},
            {"name": "setup.weight", "path": ("setup.weight",),
             "inclusive": 4.0},
            {"name": "kmeans", "path": ("kmeans",), "inclusive": 2.0},
            {"name": "init", "path": ("kmeans", "init"),
             "inclusive": 0.2},
            {"name": "lloyd", "path": ("kmeans", "lloyd"),
             "inclusive": 1.5},
            {"name": "assign_labels",
             "path": ("kmeans", "assign_labels"), "inclusive": 0.3},
            {"name": "snap_unfold", "path": ("snap_unfold",),
             "inclusive": 0.5},
            {"name": "release_before_prune",
             "path": ("release_before_prune",), "inclusive": 1.0},
            {"name": "prune", "path": ("prune",), "inclusive": 3.0},
            # Nested diagnostics: already included in the prune parent.
            {"name": "prune.gram", "path": ("prune", "prune.gram"),
             "inclusive": 2.0},
            {"name": "prune.select",
             "path": ("prune", "prune.select"), "inclusive": 1.0},
        ),
        warnings=("RuntimeWarning: density symmetry residual is large",))

    assert "MPI ranks      : 4" in text
    assert "Wavefunctions  : phdf5 reader" in text
    assert "JAX/JAXLIB     : 0.9.1 / 0.9.1" in text
    assert "feature fit: bands 1-40" in text
    assert "TIMING (POST-STARTUP)" in text
    timing_lines = {line.split(":", 1)[0].rstrip(): line
                    for line in text.splitlines() if " s" in line}
    assert timing_lines["Feature metric [setup.weight]"].endswith("4.000 s")
    assert timing_lines["Orbit snap/unfold [snap_unfold]"].endswith("0.500 s")
    assert timing_lines["  Gram build [prune/prune.gram]"].endswith("2.000 s")
    assert timing_lines["  Block selection [prune/prune.select]"].endswith(
        "1.000 s")
    assert timing_lines["Other selection work"].endswith("1.000 s")
    assert "Selection wall : 12.500 s" in text
    assert "WARNINGS" in text
    assert "density symmetry residual is large" in text
    assert "per-rank" not in text and "HDF5" not in text and "h5py" not in text


def test_kmeans_phase_waits_for_every_returned_lloyd_value():
    """The k-means phase must include asynchronous device completion.

    ``np.asarray(centroids)`` happens after the timing context.  Without a
    watcher the context measures dispatch while the following host conversion
    pays the device work, making the phase table systematically too small.
    """
    source = (Path(__file__).parents[1] / "src" / "centroid" /
              "kmeans_cli.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    watched = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.With) or len(node.items) != 1:
            continue
        item = node.items[0]
        call = item.context_expr
        if not (isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "section"
                and call.args
                and isinstance(call.args[0], ast.Constant)
                and call.args[0].value == "kmeans"
                and isinstance(item.optional_vars, ast.Name)):
            continue
        section_name = item.optional_vars.id
        for child in ast.walk(node):
            if (isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and isinstance(child.func.value, ast.Name)
                    and child.func.value.id == section_name
                    and child.func.attr == "watch"):
                watched = {
                    arg.id for arg in child.args if isinstance(arg, ast.Name)
                }
    assert watched == {
        "labels", "centroids", "_lloyd_steps", "_lloyd_move_sq",
    }

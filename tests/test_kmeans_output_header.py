"""Contract tests for the scientific provenance on centroid coordinate files."""

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
        warnings=("RuntimeWarning: density symmetry residual is large",))

    assert "MPI ranks      : 4" in text
    assert "Wavefunctions  : phdf5 reader" in text
    assert "JAX/JAXLIB     : 0.9.1 / 0.9.1" in text
    assert "feature fit: bands 1-40" in text
    assert "Selection wall : 12.500 s" in text
    assert "WARNINGS" in text
    assert "density symmetry residual is large" in text
    assert "per-rank" not in text and "HDF5" not in text and "h5py" not in text

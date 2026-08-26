"""The shared centroid loader returns the symmetry service's closure fact."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


_REPO = Path(__file__).resolve().parents[1]


def test_this_suite_uses_the_selected_checkout_and_jax09():
    """A green result from another checkout or JAX generation is no result."""
    import file_io.centroids as centroids_module
    import jax

    assert Path(centroids_module.__file__).resolve() == (
        _REPO / "src" / "file_io" / "centroids.py")
    assert jax.__version__.split(".")[:2] == ["0", "9"]


def _sym(*, translation=(0.0, 0.0, 0.0)):
    """Identity + inversion, with BGW's stored ``tnp = 2*pi*tau``."""
    matrices = np.asarray([
        np.eye(3, dtype=np.int64),
        -np.eye(3, dtype=np.int64),
    ])
    tau = np.asarray([
        np.zeros(3),
        np.asarray(translation, dtype=np.float64),
    ])
    return SimpleNamespace(sym_matrices=matrices,
                           translations=2.0 * np.pi * tau)


def test_checked_loader_returns_the_canonical_closed_verdict(tmp_path):
    from file_io import load_centroid_basis

    path = tmp_path / "closed.txt"
    np.savetxt(path, np.asarray([
        [0.0, 0.0, 0.0],
        [0.25, 0.0, 0.0],
        [0.75, 0.0, 0.0],
    ]))
    loaded = load_centroid_basis(path, (4, 4, 4), sym=_sym())

    assert loaded.path == str(path.resolve())
    assert loaded.n_rmu == 3
    assert loaded.centroid_indices.dtype == np.int64
    assert np.array_equal(loaded.centroid_indices[:, 0], [0, 1, 3])
    assert loaded.orbit_closed
    assert loaded.closure.closed
    assert loaded.closure.n_sym == 2
    assert loaded.closure.n_centroids == 3


def test_nonclosed_set_is_a_structured_fact_not_a_loader_refusal(tmp_path):
    from file_io import load_centroid_basis
    from symmetry_maps import resolve_qgrid_symmetry

    path = tmp_path / "open.txt"
    np.savetxt(path, np.asarray([
        [0.0, 0.0, 0.0],
        [0.25, 0.0, 0.0],
    ]))
    sym = _sym()
    loaded = load_centroid_basis(path, (4, 4, 4), sym=sym)

    assert not loaded.orbit_closed
    assert loaded.closure.violating_ops == (1,)
    assert loaded.closure.worst_residual == pytest.approx(0.25)

    # GW's q-grid resolver must see exactly the same fact.  This is a parity
    # check against the production door, not a second reference formula.
    qgrid = resolve_qgrid_symmetry(
        loaded.centroid_indices,
        sym.sym_matrices,
        tnp=sym.translations,
        fft_grid=np.asarray((4, 4, 4)),
        context="unit parity",
    )
    assert qgrid.verdict.centroid_hash == loaded.closure.centroid_hash
    assert qgrid.verdict.violating_ops == loaded.closure.violating_ops
    assert np.array_equal(qgrid.verdict.residual_by_op,
                          loaded.closure.residual_by_op)


def test_fractional_translation_uses_symmaps_tnp_convention(tmp_path):
    from file_io import load_centroid_basis

    path = tmp_path / "translated.txt"
    np.savetxt(path, np.asarray([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
    ]))
    # This affine operation's half-cell translation permutes these points.
    loaded = load_centroid_basis(
        path, (4, 4, 4), sym=_sym(translation=(0.5, 0.0, 0.0)))
    assert loaded.orbit_closed


def test_selection_is_measured_after_slicing_without_losing_parent_count(
        tmp_path):
    from file_io import load_centroid_basis

    path = tmp_path / "parent.txt"
    np.savetxt(path, np.asarray([
        [0.0, 0.0, 0.0],
        [0.25, 0.0, 0.0],
        [0.75, 0.0, 0.0],
    ]))
    loaded = load_centroid_basis(
        path, (4, 4, 4), sym=_sym(), selection=np.asarray([0]))

    assert loaded.source_n_rmu == 3
    assert loaded.n_rmu == 1
    assert np.array_equal(loaded.centroid_indices, [[0, 0, 0]])
    assert loaded.closure.n_centroids == 1
    assert loaded.orbit_closed


def test_loader_calls_the_service_verifier_and_never_rebuilds_symmaps():
    """The shared loader is a consumer of the symmetry authority, not one."""
    path = _REPO / "src" / "file_io" / "centroids.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        calls.append(fn.id if isinstance(fn, ast.Name)
                     else fn.attr if isinstance(fn, ast.Attribute) else "")
    assert calls.count("verify_centroid_orbit_closure") == 1
    assert "SymMaps" not in calls


def test_gw_and_htransform_both_retain_the_checked_record():
    gw = (_REPO / "src" / "gw" / "gw_jax.py").read_text(encoding="utf-8")
    ht = (_REPO / "src" / "bandstructure" / "htransform.py").read_text(
        encoding="utf-8")

    assert "centroid_basis = load_centroid_basis(" in gw
    assert "centroids=centroid_basis" in gw
    assert "centroid_basis = load_centroid_basis(" in ht
    assert "centroid_record_fn(centroid_basis)" in ht
    assert "def _load_centroids(" not in ht


def test_single_row_and_periodic_wrap_match_the_legacy_tuple_surface(tmp_path):
    from file_io import load_centroids

    path = tmp_path / "one.txt"
    np.savetxt(path, np.asarray([[0.999999, -0.25, 1.25]]))
    frac, indices, n_rmu = load_centroids(path, (4, 4, 4))

    assert frac.shape == (1, 3)
    assert n_rmu == 1
    assert np.array_equal(indices, np.asarray([[0, 3, 1]]))


@pytest.mark.parametrize("rows", [
    np.empty((0, 3)),
    np.ones((2, 2)),
    np.asarray([[0.0, np.nan, 0.0]]),
])
def test_malformed_centroid_tables_refuse_at_the_loading_door(tmp_path, rows):
    from file_io import load_centroids

    path = tmp_path / "bad.txt"
    np.savetxt(path, rows)
    with pytest.raises(ValueError):
        load_centroids(path, (4, 4, 4))

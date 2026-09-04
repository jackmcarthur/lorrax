"""Focused contracts for unit-weight centroid fitting and spatial closure."""

import ast
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_nonsymmorphic_grid_average_uses_the_seitz_translation():
    from centroid.charge_density import symmetrize_on_grid

    field = np.arange(8.0).reshape(4, 2, 1)
    identity = np.eye(3, dtype=np.int32)
    operations = np.stack((identity, identity))
    translations = np.asarray(((0.0, 0.0, 0.0), (0.5, 0.0, 0.0)))

    got = symmetrize_on_grid(field, operations, translations)
    expected = 0.5 * (field + np.roll(field, shift=2, axis=0))
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(
        np.roll(got, shift=2, axis=0), got,
        err_msg="the projected weight is not invariant under the translation")


def test_off_grid_seitz_translation_refuses_instead_of_rounding():
    from centroid.charge_density import symmetrize_on_grid

    with pytest.raises(RuntimeError, match="not commensurate"):
        symmetrize_on_grid(
            np.arange(8.0).reshape(2, 2, 2),
            np.eye(3, dtype=np.int32)[None, ...],
            np.asarray(((0.25, 0.0, 0.0),)),
        )


def test_kmeans_cli_has_no_occupation_weight_fit_option():
    source = (ROOT / "src/centroid/kmeans_cli.py").read_text()
    assert 'choices=("charge_density", "band_range")' not in source
    assert "n_occ =" not in source
    assert 'add_argument("--weight-bands"' not in source
    assert 'add_argument("--centroid-weight"' not in source
    assert "metric_diagonal = build_feature_metric_diagonal(" in source
    assert "weight = np.sqrt(metric_diagonal)" in source
    assert "left_range, right_range, range_label = prune_band_ranges(" in source

    # Red twin: both retired policy tokens are detected.
    old = ('choices=("charge_density", "band_range")\n'
           'p.add_argument("--weight-bands")\n'
           'n_occ = int(wfn.nelec)')
    assert 'choices=("charge_density", "band_range")' in old
    assert 'add_argument("--weight-bands"' in old
    assert "n_occ =" in old


def _report_file_assignment():
    """The `report_file = ...` node in ``kmeans_cli.main``, via AST.

    Matched on the parse tree, not on the source text: a comment or docstring
    mentioning the filename is not a fact about the code (TASTE.md row 17),
    and this module's other source test is string-based precisely because it
    is asserting ABSENCE, where a stray comment can only make it stricter.
    """
    tree = ast.parse((ROOT / "src/centroid/kmeans_cli.py").read_text())
    found = [node.value for node in ast.walk(tree)
             if isinstance(node, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "report_file"
                     for t in node.targets)]
    assert len(found) == 1, f"expected one report_file assignment, got {len(found)}"
    return found[0]


def test_kmeans_report_carries_the_density_mode_suffix():
    """A charge and a current selection in one directory must not collide.

    The deck names both tables, so both runs happen in the centroid
    directory.  The table was already suffixed ('' / '_current'); the report
    was a bare `kmeans.out`, so the second run silently destroyed the first
    one's provenance -- its band window, seed, candidate pool and achieved
    rank.  Measured on the MoS2 3x3 deck 2026-09-01: the charge report was
    gone after the current run.
    """
    value = _report_file_assignment()
    assert isinstance(value, ast.JoinedStr), (
        "report_file must interpolate the suffix, not be a constant; got "
        f"{type(value).__name__}")
    interpolated = {node.value.id for node in ast.walk(value)
                    if isinstance(node, ast.FormattedValue)
                    and isinstance(node.value, ast.Name)}
    assert interpolated == {"out_suffix"}, interpolated

    # The suffix is the one the table uses, so the two names track together.
    source = (ROOT / "src/centroid/kmeans_cli.py").read_text()
    assert 'out_file = f"centroids_frac_{n_unique}{out_suffix}.txt"' in source

    # Red twin: the pre-fix constant is exactly what this test rejects.
    old = ast.parse('report_file = "kmeans.out"').body[0].value
    assert not isinstance(old, ast.JoinedStr)


@pytest.mark.parametrize(
    ("change", "message"),
    (({"no_orbit": True}, "orbit closure"),
     ({"rho_power": 0.5}, "feature-row norm"),
     ({"oversample": 1.0}, "transverse-Gram pruning")),
)
def test_current_mode_refuses_metric_bypasses(change, message):
    from centroid.production_output import validate_mode_policy

    values = dict(
        density_mode="current", no_orbit=False, rho_power=1.0,
        oversample=1.5)
    values.update(change)
    with pytest.raises(ValueError, match=message):
        validate_mode_policy(SimpleNamespace(**values))


def test_scalar_mode_retains_explicit_experiment_switches():
    from centroid.production_output import validate_mode_policy

    validate_mode_policy(SimpleNamespace(
        density_mode="scalar", no_orbit=True, rho_power=0.5,
        oversample=1.0))


def _load_prune_stage_without_driver_startup():
    """Compile the real policy-stage functions without bootstrapping JAX."""
    source = (ROOT / "src/centroid/kmeans_cli.py").read_text()
    tree = ast.parse(source)
    wanted = {"_resolve_sigma_window", "_prune"}
    definitions = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in wanted
    ]
    assert {node.name for node in definitions} == wanted
    namespace = {
        "__package__": "centroid",
        "np": np,
        "print0": lambda *args, **kwargs: None,
        "rank0_print": lambda *args, **kwargs: None,
        "debug_print_enabled": lambda: False,
        "process_rank": lambda: 0,
        "timing": SimpleNamespace(section=lambda _name: nullcontext()),
        "_DEFAULT_PRUNE_TIME_BUDGET_SECONDS": 900.0,
    }
    exec(compile(ast.Module(body=definitions, type_ignores=[]),
                 str(ROOT / "src/centroid/kmeans_cli.py"), "exec"), namespace)
    return namespace["_prune"]


def test_current_prune_routes_exact_transverse_metric_through_group_blocks(
        monkeypatch):
    """Current mode spends a point budget and pivots complete atom orbits."""
    from centroid.production_output import prune_band_ranges

    calls = []
    weights = np.asarray([0.25, 0.75])

    prune_module = ModuleType("centroid.pivoted_cholesky")

    def fake_prune(**kwargs):
        calls.append(kwargs)
        return (np.asarray(kwargs["cand_idx"])[:4], 4, None, None,
                None, None, None)

    prune_module.prune_candidates_by_pivoted_cholesky = fake_prune
    metric_module = ModuleType("centroid.sampling_metric")
    metric_module.full_k_quadrature_weights = lambda _wfn, _sym: weights
    monkeypatch.setitem(sys.modules, "centroid.pivoted_cholesky", prune_module)
    monkeypatch.setitem(sys.modules, "centroid.sampling_metric", metric_module)

    prune = _load_prune_stage_without_driver_startup()
    prune.__globals__["prune_band_ranges"] = prune_band_ranges
    args = SimpleNamespace(
        density_mode="current", prune_n_val=None, prune_n_cond=None,
        prune_window="v_x_vc", fit_window=None,
    )
    wfn = SimpleNamespace(nelec=2, nbands=5)
    candidates = np.arange(18, dtype=np.int64).reshape(6, 3)
    orbit_id = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int32)
    selected, rank = prune(
        args, wfn, object(), object(), candidates, orbit_id, 6, 4)

    assert rank == 4 and selected.shape == (4, 3)
    assert len(calls) == 1
    call = calls[0]
    assert call["group_block"] is True
    assert call["n_keep"] == call["n_point_budget"] == 4
    assert call["select_time_budget_s"] == 900.0
    assert call["bispinor"] is True
    assert call["gamma_mode"] == "transverse"
    assert "band_norms" not in call
    np.testing.assert_array_equal(call["orbit_id"], orbit_id)
    np.testing.assert_array_equal(call["k_weights"], weights)
    assert call["band_range_left"] == (0, 2)
    assert call["band_range_right"] == (0, 5)


def test_scalar_prune_keeps_the_same_ungrouped_charge_route(monkeypatch):
    """The current hardening does not redirect scalar point selection."""
    from centroid.production_output import prune_band_ranges

    calls = []
    prune_module = ModuleType("centroid.pivoted_cholesky")

    def fake_prune(**kwargs):
        calls.append(kwargs)
        return (np.asarray(kwargs["cand_idx"])[:3], 3, None, None,
                None, None, None)

    prune_module.prune_candidates_by_pivoted_cholesky = fake_prune
    metric_module = ModuleType("centroid.sampling_metric")
    metric_module.full_k_quadrature_weights = lambda _wfn, _sym: np.ones(1)
    monkeypatch.setitem(sys.modules, "centroid.pivoted_cholesky", prune_module)
    monkeypatch.setitem(sys.modules, "centroid.sampling_metric", metric_module)

    prune = _load_prune_stage_without_driver_startup()
    prune.__globals__["prune_band_ranges"] = prune_band_ranges
    args = SimpleNamespace(
        density_mode="scalar", prune_n_val=None, prune_n_cond=None,
        prune_window="v_x_vc", fit_window=None,
    )
    selected, rank = prune(
        args, SimpleNamespace(nelec=2, nbands=5), object(), object(),
        np.arange(15, dtype=np.int64).reshape(5, 3), None, 5, 3)

    assert rank == 3 and selected.shape == (3, 3)
    call = calls[0]
    assert call["group_block"] is False
    assert call["n_point_budget"] is None
    assert call["select_time_budget_s"] == 900.0
    assert call["bispinor"] is False
    assert call["gamma_mode"] == "charge"

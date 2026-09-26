"""ISDF pair-density band sets: every occupied band on both legs.

Owner rule (reports/audits_2026-09-25/PAIRS.md, 2026-09-26): the left leg of
every fitted or selected pair density is all occupied states plus the
protected Σ conduction window, the right leg every band in the sums, and the
lowest valence states are never skipped.  Each refusal has a red twin.
"""
from __future__ import annotations

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _slices(b0=0, b1=0, b2=8, b3=60, b4=60):
    from gw.wavefunction_bundle import BandSlices
    return BandSlices.from_band_edges(b0, b1, b2, b3, b4)


def _gw_init():
    pytest.importorskip("jax")
    from gw import gw_init
    return gw_init


# ---------------------------------------------------------------------------
# ζ fit
# ---------------------------------------------------------------------------

def test_zeta_right_leg_starts_at_band_zero_when_nval_below_nocc():
    """nval < nocc: b1 > 0, and the right leg still holds [0, b1)."""
    gi = _gw_init()
    bs = _slices(b1=14, b2=18, b3=26, b4=36)
    left, right = gi.zeta_fit_band_ranges(bs, None, log=lambda *_: None)
    assert left == (0, 26) and right == (0, 36)
    gi.assert_zeta_fit_keeps_occupied(bs, left, right)


def test_zeta_ranges_unchanged_when_nval_equals_nocc():
    """nval == nocc: b1 == b0, so (b1, b4) and (b0, b4) are one range."""
    gi = _gw_init()
    bs = _slices(b1=0, b2=130, b3=144, b4=144)
    left, right = gi.zeta_fit_band_ranges(bs, None, log=lambda *_: None)
    assert left == (bs.b0, bs.b3) and right == (bs.b1, bs.b4)


def test_zeta_narrowed_right_leg_starts_at_band_zero():
    gi = _gw_init()
    bs = _slices(b1=4, b2=8, b3=40, b4=64)
    left, right = gi.zeta_fit_band_ranges(bs, 52, log=lambda *_: None)
    assert left == (0, 40) and right == (0, 52)


def test_zeta_fit_window_dropping_occupied_bands_refuses_by_name():
    gi = _gw_init()
    bs = _slices(b1=14, b2=18, b3=26, b4=36)
    # Red twin: the pre-2026-09-26 right leg (b1, b4).
    with pytest.raises(gi.ZetaFitWindowDropsOccupiedError,
                       match="ZetaFitWindowDropsOccupiedError"):
        gi.assert_zeta_fit_keeps_occupied(bs, (0, 26), (14, 36))
    # A left leg that stops inside the occupied manifold.
    with pytest.raises(gi.ZetaFitWindowDropsOccupiedError):
        gi.assert_zeta_fit_keeps_occupied(bs, (0, 16), (0, 36))
    gi.assert_zeta_fit_keeps_occupied(bs, (0, 26), (0, 36))


def test_zeta_contract_calls_the_refusal_after_the_resolver():
    src = (ROOT / "src/gw/gw_init.py").read_text(encoding="utf8")
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef)
              and n.name == "_resolve_zeta_fit_contract")
    code = ast.unparse(fn)
    assert code.index("zeta_fit_band_ranges(") < code.index(
        "assert_zeta_fit_keeps_occupied(")
    assert "_check_centroid_selection_windows(" in code


# ---------------------------------------------------------------------------
# kmeans selection windows
# ---------------------------------------------------------------------------

def test_kmeans_default_left_leg_is_occupied_plus_sigma_conduction():
    from centroid.production_output import prune_band_ranges
    args = SimpleNamespace(prune_window="v_x_vc", fit_window=None,
                           sigma_ncond=8)
    left, right, label = prune_band_ranges(args, 18, 17)
    assert left == (0, 26) and right == (0, 35)
    assert "Sigma conduction" in label


def test_kmeans_without_a_deck_falls_back_to_the_full_square():
    from centroid.production_output import prune_band_ranges
    args = SimpleNamespace(prune_window="v_x_vc", fit_window=None,
                           sigma_ncond=None)
    left, right, label = prune_band_ranges(args, 18, 17)
    assert left == right == (0, 35)
    assert "Sigma window unknown" in label


def test_kmeans_fit_window_dropping_occupied_bands_refuses_by_name():
    from centroid.production_output import prune_band_ranges
    from file_io.centroids import CentroidWindowDropsOccupiedError
    for window in ("2:26,0:35", "0:26,4:35", "0:12,0:35"):
        with pytest.raises(CentroidWindowDropsOccupiedError,
                           match="CentroidWindowDropsOccupiedError"):
            prune_band_ranges(SimpleNamespace(
                prune_window="v_x_vc", fit_window=window), 18, 17)
    # Red twin: the full occupied manifold on both legs passes.
    assert prune_band_ranges(SimpleNamespace(
        prune_window="v_x_vc", fit_window="0:26,0:35"), 18, 17)[:2] == (
            (0, 26), (0, 35))


def _sigma_window_resolver():
    """The real ``_resolve_sigma_window`` without the driver's startup."""
    source = (ROOT / "src/centroid/kmeans_cli.py").read_text()
    fn = next(node for node in ast.parse(source).body
              if isinstance(node, ast.FunctionDef)
              and node.name == "_resolve_sigma_window")
    namespace = {"__package__": "centroid", "np": np,
                 "timing": SimpleNamespace(section=lambda _n: nullcontext())}
    exec(compile(ast.Module(body=[fn], type_ignores=[]),
                 str(ROOT / "src/centroid/kmeans_cli.py"), "exec"), namespace)
    return namespace["_resolve_sigma_window"]


def test_kmeans_prune_n_val_below_nelec_refuses_by_name():
    from file_io.centroids import CentroidWindowDropsOccupiedError
    resolve = _sigma_window_resolver()
    wfn = SimpleNamespace(nelec=18, nbands=35)
    with pytest.raises(CentroidWindowDropsOccupiedError,
                       match="prune-n-val 4"):
        resolve(SimpleNamespace(prune_n_val=4, prune_n_cond=None), wfn)
    # Red twins: the default and a value at nelec pass.
    assert resolve(SimpleNamespace(prune_n_val=None, prune_n_cond=None),
                   wfn) == (18, 17)
    assert resolve(SimpleNamespace(prune_n_val=18, prune_n_cond=None),
                   wfn) == (18, 17)


# ---------------------------------------------------------------------------
# gw_jax reading the centroid header
# ---------------------------------------------------------------------------

def _table(tmp_path, left, right):
    from centroid.production_output import format_centroid_header
    header = format_centroid_header(
        feature_fit="test", source_wfn="WFN.h5", weight_label="test",
        num_electrons=18.0, occupied_boundary=18, fft_grid=(4, 4, 4),
        kgrid=(1, 1, 1), shift=(0.0, 0.0, 0.0), seed=0, rho_power=1.0,
        requested=2, candidates=3, written=2, pruning="pivoted Cholesky",
        prune_rank=2, prune_left=left, prune_right=right, prune_label="t",
        orbit_aware=False, n_sym=1, density_mode="scalar")
    path = tmp_path / "centroids_frac_2.txt"
    np.savetxt(path, np.zeros((2, 3)), header=header, fmt="%.6f",
               comments="# ")
    return str(path)


def test_centroid_header_windows_round_trip(tmp_path):
    from file_io.centroids import read_centroid_pair_windows
    path = _table(tmp_path, (0, 26), (0, 35))
    assert read_centroid_pair_windows(path) == ((0, 26), (0, 35))


def test_centroid_header_covering_the_zeta_legs_is_silent(tmp_path):
    from file_io.centroids import check_centroid_pair_windows
    path = _table(tmp_path, (0, 35), (0, 35))
    assert check_centroid_pair_windows(path, 18, (0, 26), (0, 36)) is not None
    assert check_centroid_pair_windows(path, 18, (0, 26), (0, 35)) is None


def test_old_v_x_vc_header_warns_and_does_not_refuse(tmp_path):
    """The pre-2026-09-26 default left leg (0, nocc) lacks the Σ conduction."""
    from file_io.centroids import check_centroid_pair_windows
    path = _table(tmp_path, (0, 18), (0, 35))
    warning = check_centroid_pair_windows(path, 18, (0, 26), (0, 35))
    assert warning is not None and "left=(0, 18)" in warning


def test_centroid_header_dropping_occupied_bands_refuses_by_name(tmp_path):
    from file_io.centroids import (CentroidWindowDropsOccupiedError,
                                   check_centroid_pair_windows)
    for left, right in (((0, 18), (18, 35)),     # legacy v_x_c
                        ((4, 26), (0, 35)),
                        ((0, 12), (0, 35))):
        path = _table(tmp_path, left, right)
        with pytest.raises(CentroidWindowDropsOccupiedError):
            check_centroid_pair_windows(path, 18, (0, 26), (0, 35))


def test_centroid_file_without_header_warns(tmp_path):
    from file_io.centroids import check_centroid_pair_windows
    path = tmp_path / "bare.txt"
    np.savetxt(path, np.zeros((2, 3)))
    warning = check_centroid_pair_windows(str(path), 18, (0, 26), (0, 35))
    assert warning is not None and "no 'pair-density" in warning


def test_gw_warns_once_per_centroid_table(tmp_path):
    gi = _gw_init()
    path = _table(tmp_path, (0, 18), (0, 35))
    cfg = SimpleNamespace(bispinor=False, paths=SimpleNamespace(
        centroids_file=path, centroids_file_current=None))
    bs = _slices(b1=14, b2=18, b3=26, b4=36)
    gi._CENTROID_WINDOW_WARNED.discard(path)
    with pytest.warns(RuntimeWarning) as record:
        for _ in range(2):
            gi._check_centroid_selection_windows(cfg, bs, (0, 26), (0, 36))
    said = [str(w.message) for w in record if "centroid file" in str(w.message)]
    assert len(said) == 1 and "left=(0, 18)" in said[0]


# ---------------------------------------------------------------------------
# BSE on stored V/W
# ---------------------------------------------------------------------------

def test_bse_window_inside_the_stamped_zeta_legs_passes():
    pytest.importorskip("jax")
    from bse.bse_window import assert_bse_window_in_zeta_training
    header = {"zeta_fit_windows": np.asarray([0, 26, 0, 36]),
              "band_window": np.asarray([0, 14, 18, 26, 36])}
    assert_bse_window_in_zeta_training(header, 2, 26)


def test_bse_window_outside_the_zeta_training_refuses_by_name():
    pytest.importorskip("jax")
    from bse.bse_window import (BseWindowOutsideZetaTrainingError,
                                assert_bse_window_in_zeta_training)
    stamped = {"zeta_fit_windows": np.asarray([0, 26, 0, 36]),
               "band_window": np.asarray([0, 14, 18, 26, 36])}
    with pytest.raises(BseWindowOutsideZetaTrainingError,
                       match="BseWindowOutsideZetaTrainingError"):
        assert_bse_window_in_zeta_training(stamped, 10, 28)
    # A legacy bundle (no stamp) trained on [b1, b3) only.
    legacy = {"zeta_fit_windows": None,
              "band_window": np.asarray([0, 14, 18, 26, 36])}
    with pytest.raises(BseWindowOutsideZetaTrainingError, match="legacy"):
        assert_bse_window_in_zeta_training(legacy, 10, 22)
    assert_bse_window_in_zeta_training(legacy, 14, 26)

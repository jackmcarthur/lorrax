"""Published whole-state randomized-QRCP htransform contract."""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest


def test_rank_multiplier_vocabulary_and_default():
    pytest.importorskip("jax")
    from isdf.galerkin import validate_rank_multiplier
    from gw.gw_config import _DEFAULTS

    assert _DEFAULTS["htransform_rank_multiplier"] == 20.0
    assert validate_rank_multiplier(0, name="htransform_rank_multiplier") == 20.0
    assert validate_rank_multiplier(
        "10", name="htransform_rank_multiplier") == 10.0
    for bad in (-1, 0.5, float("nan"), "not-a-number"):
        with pytest.raises(ValueError, match="htransform_rank_multiplier"):
            validate_rank_multiplier(bad, name="htransform_rank_multiplier")


def test_downfold_centroid_subset_is_ordered_strict_and_checked():
    from bandstructure.htransform import validate_centroid_subset_idx

    got = validate_centroid_subset_idx(np.asarray([7, 1, 9, 3]), 10)
    assert np.array_equal(got, [7, 1, 9, 3])
    for bad in ([], [1, 1], [-1, 2], [1, 10], [1.0, 2.0], [[1, 2]]):
        with pytest.raises(ValueError, match="centroid subset"):
            validate_centroid_subset_idx(np.asarray(bad), 10)


def test_newton_inverse_reports_the_archived_residual_contract():
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from bandstructure.htransform import (
        NEWTON_RESIDUAL_MAX,
        fun,
        newton_inv,
        require_newton_converged,
    )

    a, n, shift = 0.8, 3.0, 1.25
    expected = jnp.asarray([-1.6, -0.4, 0.2, 1.0], dtype=jnp.float64)
    recovered, residual = newton_inv(
        a, n, shift, fun(a, n, shift, expected))
    assert np.max(np.abs(np.asarray(recovered) - np.asarray(expected))) < 1e-11
    assert float(residual) <= NEWTON_RESIDUAL_MAX
    require_newton_converged(float(residual), where="unit receipt")
    with pytest.raises(ValueError, match="did not converge"):
        require_newton_converged(
            2.0 * NEWTON_RESIDUAL_MAX, where="red receipt")


def test_standalone_htransform_refuses_an_occupied_band_cut(monkeypatch):
    pytest.importorskip("jax")
    from types import SimpleNamespace
    import file_io.centroids
    from bandstructure import htransform

    monkeypatch.setattr(
        htransform, "setup_wfn_and_sym",
        lambda *args, **kwargs: (SimpleNamespace(nelec=12), object()))

    def _centroids_were_reached(*args, **kwargs):
        raise RuntimeError("centroid stage reached")

    monkeypatch.setattr(
        file_io.centroids, "load_centroids",
        _centroids_were_reached)
    params = {"wfn_file": "unused.h5", "nval": 11, "ncond": 2, "nband": 13}
    with pytest.raises(ValueError, match="requires every occupied band"):
        htransform.initialize_wfns(
            "unused.in", params, lambda *args: None, mesh_xy=object(),
            require_all_occupied=True)

    # Internal BSE windows deliberately keep their explicit partial-window
    # contract, so the same setup reaches the next stage when the standalone
    # gate is absent.
    with pytest.raises(RuntimeError, match="centroid stage reached"):
        htransform.initialize_wfns(
            "unused.in", params, lambda *args: None, mesh_xy=object())


def test_refit_consumes_the_compact_whole_state_factor():
    pytest.importorskip("jax")
    from isdf import galerkin
    root = Path(__file__).resolve().parents[1] / "src"
    src = (root / "bandstructure" / "htransform.py").read_text()
    fit_src = inspect.getsource(galerkin.fit_galerkin_basis)
    assert "return_full_proj" not in src
    assert "include_projector" not in fit_src
    assert "selector_projector" not in fit_src
    assert "rank_multiplier=params.get" in src
    assert "selected_state_indices" in fit_src
    assert "selection_factor=L" in fit_src


def test_bse_consumers_forward_the_q_chunk_key():
    """The local-batch route is useful only if the documented width arrives."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "bse"
    for name in ("bse_densify.py", "exciton_bands.py"):
        src = (root / name).read_text()
        assert 'batch_size=int(params.get("wfn_fi_q_chunk", 0))' in src, name
        assert "centroid_subset_idx=_fit_subset" in src, name
        assert "_fit_subset = keep" in src, name
        assert "_output_keep = None" in src, name

    refit_src = (root / "vq_interp.py").read_text()
    assert "centroid_subset_idx=keep_idx" in refit_src
    assert "B_at_mu = B_at_mu[:, :," not in refit_src


def test_exciton_a_band_reaches_both_htransform_calls():
    """The densified stored leg and shifted-Q leg must share one shoulder."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "bse"
    main_src = (root / "exciton_bands.py").read_text()
    load_src = (root / "bse_loading.py").read_text()
    dense_src = (root / "bse_densify.py").read_text()
    assert "htransform_a_band=args.a_band" in main_src
    assert "a_band_index=args.a_band" in main_src
    assert "htransform_a_band=htransform_a_band" in load_src
    assert "a_band_index=htransform_a_band" in dense_src

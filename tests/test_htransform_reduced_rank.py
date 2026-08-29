"""Published whole-state randomized-QRCP htransform contract."""
from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def test_whole_state_workspace_must_fit_the_bfc_reserve(monkeypatch):
    """An aggregate-safe FFT still refuses when its arena cannot be placed."""
    pytest.importorskip("jax")
    from isdf import galerkin

    def _ledger(**kwargs):
        del kwargs
        return {
            "HWM": 700.0,
            "WFN_RCHUNK_TRANSFORM": 650.0,
            "WFN_RCHUNK_COMPILED": 450.0,
            "WFN_CUFFT_WORKSPACE": 200.0,
            "Q_TILE_LOCAL": 10.0,
            "r_chunk_carrier": 16.0,
        }

    monkeypatch.setattr(galerkin, "_whole_state_memory_ledger", _ledger)
    mesh = SimpleNamespace(size=16, shape={"x": 4, "y": 4})
    kwargs = dict(
        meta=object(), mesh_xy=mesh, nk=2, nspinor=2, ngkmax=8,
        band_carrier=64, state_count=32, search_rank=4,
        candidate_carrier=8, requested_q_tile_budget=512,
        device_pool_limit=1000.0, log_fn=lambda *args: None)

    # spinor width 2 owns the public 0.85 target: HWM 700 < 850, but the
    # independently allocated workspace 200 is larger than its 150 reserve.
    with pytest.raises(MemoryError, match="contiguous BFC reserve"):
        galerkin._resolve_whole_state_stream_budget(**kwargs)

    safe = dict(kwargs, device_pool_limit=1400.0)
    _, ledger = galerkin._resolve_whole_state_stream_budget(**safe)
    assert ledger["WFN_CUFFT_WORKSPACE"] == 200.0


def test_wfn_rchunk_integer_peak_api_is_the_cached_breakdown_view(monkeypatch):
    """Existing callers retain the integer-total API without another compile."""
    pytest.importorskip("jax")
    from common import wfn_transforms

    planning_src = inspect.getsource(
        wfn_transforms.gflat_to_rchunk_aot_memory)
    assert "not memory.cufft_measured" in planning_src
    assert "known-low memory preflight" in planning_src

    calls = []

    def _memory(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(total=1234)

    monkeypatch.setattr(
        wfn_transforms, "gflat_to_rchunk_aot_memory", _memory)
    got = wfn_transforms.gflat_to_rchunk_aot_peak_bytes(
        mesh=object(), nk=1, band_carrier=1, nspinor=1, ngkmax=1,
        fft_grid=(1, 1, 1), r_carrier=1, norm="ortho")
    assert got == 1234
    assert len(calls) == 1


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

    inverse_src = inspect.getsource(newton_inv)
    assert "lax.while_loop" in inverse_src
    assert "lax.fori_loop" not in inverse_src


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
        assert "centroid_keep_idx" not in src, name

    refit_src = (root / "vq_interp.py").read_text()
    assert "centroid_subset_idx=keep_idx" in refit_src
    assert "B_at_mu = B_at_mu[:, :," not in refit_src


def test_kpath_inverts_only_physical_states_and_publishes_return_window():
    """The rank-space null carrier is not a physical band window."""
    root = Path(__file__).resolve().parents[1] / "src"
    src = (root / "bandstructure" / "htransform.py").read_text()
    fft_src = (root / "common" / "fft_helpers.py").read_text()

    assert "[:nq, :states]" in src
    assert "energies_on_path = energies_sorted_jax" in src
    assert "jax.vmap(\n                lambda row: newton_inv" not in src
    assert "FLAT_K_FFT_VALUE_RTOL = 1.0e-12" in fft_src
    assert "fft_rel > FLAT_K_FFT_VALUE_RTOL" in src


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

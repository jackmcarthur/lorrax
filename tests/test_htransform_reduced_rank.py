"""Published whole-state randomized-QRCP htransform contract."""
from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def test_whole_state_workspace_must_fit_the_bfc_reserve(monkeypatch):
    """An aggregate-safe FFT still refuses when its arena cannot be placed,
    and the planner sizes every stream from the budget (no tile knob)."""
    pytest.importorskip("jax")
    from isdf import galerkin

    monkeypatch.setattr(
        galerkin, "gflat_to_rchunk_aot_memory",
        lambda **kwargs: SimpleNamespace(total=100.0, cufft_scratch=200.0))
    kwargs = dict(
        meta=SimpleNamespace(n_rtot=64, fft_grid=(4, 4, 4)),
        mesh_xy=SimpleNamespace(size=16), nk=2, nspinor=2, ngkmax=8,
        band_divisor=16, band_range=(0, 32))

    # spinor width 2 owns the public 0.85 target: the stage fits the target,
    # but the independently allocated workspace 200 exceeds its 150 reserve.
    with pytest.raises(MemoryError, match="contiguous BFC reserve"):
        galerkin._whole_state_geometry(device_pool_limit=1000.0, **kwargs)

    geom, capacity, memory = galerkin._whole_state_geometry(
        device_pool_limit=1.0e9, **kwargs)
    assert memory.cufft_scratch == 200.0 and geom["row_fft"] == 100.0
    groups, fft, plan, _ = galerkin._plan_rows_pass(
        geom, rows=3, omega_rows=2, resident=0.0, capacity=capacity,
        name="test")
    assert (groups, fft, len(plan.r_chunk_ranges)) == (1, 3, 1)
    with pytest.raises(MemoryError, match="one state per device"):
        galerkin._plan_rows_pass(geom, rows=3, omega_rows=2, resident=0.0,
                                 capacity=1.0, name="test")
    assert "LORRAX_GALERKIN_CHUNK_GIB" not in inspect.getsource(galerkin)


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
        lambda *args, **kwargs: (
            SimpleNamespace(nelec=12, fft_grid=(1, 1, 1)), object()))

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

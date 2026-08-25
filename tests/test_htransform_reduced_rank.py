"""Opt-in reduced cross-k Galerkin model-order contract.

The historical htransform carries the full numerical rank and remains the
default.  These cells pin the distinct approximation requested by the input
key: rank proportional to bands at one k, followed by a per-k polar factor so
``build_fH_R`` keeps its row-isometry invariant.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest


def test_rank_multiplier_vocabulary_and_default():
    pytest.importorskip("jax")
    from isdf.galerkin import validate_rank_multiplier
    from gw.gw_config import _DEFAULTS

    assert _DEFAULTS["htransform_rank_multiplier"] == 0.0
    assert validate_rank_multiplier(0, name="htransform_rank_multiplier") == 0.0
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


def test_lowdin_restores_every_k_row_isometry_and_red_twin_is_nonorthogonal():
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from isdf.galerkin import _lowdin_orthonormalize_band_rows

    rng = np.random.default_rng(20260820)
    nk, nb, rank = 5, 6, 40
    c = (rng.standard_normal((nk, nb, rank))
         + 1j * rng.standard_normal((nk, nb, rank)))
    # Plant a well-conditioned but visibly non-isometric shared-span block.
    c[:, 0] *= 0.35
    gram_before = np.einsum("kna,kma->knm", c, np.conj(c), optimize=True)
    before = float(np.max(np.abs(gram_before - np.eye(nb)[None])))
    assert before > 1.0, "RED TWIN DID NOT GO RED: input was accidentally isometric"

    out, lmin, lmax, before_dev, after_dev, move = (
        _lowdin_orthonormalize_band_rows(jnp.asarray(c)))
    out = np.asarray(out)
    gram_after = np.einsum("kna,kma->knm", out, np.conj(out), optimize=True)
    assert np.max(np.abs(gram_after - np.eye(nb)[None])) < 2.0e-13
    assert float(after_dev) < 2.0e-13
    assert float(before_dev) == pytest.approx(before, rel=2.0e-13)
    assert 0.0 < float(lmin) <= float(lmax)
    assert float(move) > 0.01


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


def test_exact_rank_report_excludes_left_gram_null_tail_and_counts_pad():
    """The carried report owns the selected block plus exact-null padding."""
    pytest.importorskip("jax")
    from isdf import galerkin
    from common import rank_criterion

    src = inspect.getsource(galerkin.fit_galerkin_basis)
    assert "s_host[:rank_phys], rtol" in src

    # Toy of the production failure plus the closure corner: the raw left
    # Gram has three above-rtol null-space values beyond a five-direction
    # physical block.  A mesh-aligned carrier of eight must be reported as
    # +3 exact-null pads, never as a -3 device-grid truncation.
    raw_left_gram = np.asarray([1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.09, 0.08])
    rank_phys = 5
    carried = 8
    assert rank_criterion.select_rank(raw_left_gram, 1.0e-8) > rank_phys
    report = rank_criterion.rank_report(
        raw_left_gram[:rank_phys], 1.0e-8, rank_used=carried)
    assert report.rank_criterion == rank_phys
    assert report.n_padded_alignment == carried - rank_phys
    assert report.n_dropped_alignment == 0
    assert not report.violations(), report.violations()


def test_bse_consumers_forward_the_q_chunk_key():
    """The local-batch route is useful only if the documented width arrives."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "bse"
    for name in ("bse_densify.py", "exciton_bands.py"):
        src = (root / name).read_text()
        assert 'batch_size=int(params.get("wfn_fi_q_chunk", 0))' in src, name
        assert "centroid_subset_idx=_fit_subset" in src, name
        assert "_output_keep = None if _fit_subset is not None" in src, name


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

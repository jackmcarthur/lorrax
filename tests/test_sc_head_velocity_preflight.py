"""D3 preflight refusals on ``sc_head_update``'s velocity source.

``reports/metal_head_pt_pipelines_2026-08-23/PLAN.md`` pipeline step 3:
three refusals, checked at ``gw.sc_iteration.load_head_velocity_source``
(config-resolution / driver-entry altitude), before the expensive
per-iteration head machinery ever sees the loaded source.

(c) per-axis link stencil support -- ``parallel_transport`` only.
(b) independent multiplet/TRIM degeneracy at the active window's top edge
    -- both metal head modes, pure DFT energies.
(a) link singular-value hybridization at the active window's top edge --
    ``parallel_transport`` only.

Every cell carries BOTH sides: the red twin that must refuse, and the clean
window that must pass in silence -- the same discipline
``tests/test_band_degeneracy.py`` documents for the primitive D3(b) reuses.
Pure numpy -- no SlabIO, no WFN, no FFI; these three functions take only
plain arrays, which is deliberate (see their docstrings in
``gw/sc_iteration.py``): the artifact/WFN I/O is the loader's job, not the
preflight's.
"""
from __future__ import annotations

import numpy as np
import pytest

from common.band_degeneracy import BandWindowDegeneracyError
from gw.sc_iteration import (
    _LINK_HYBRIDIZATION_FLOOR,
    _refuse_degenerate_window_edge,
    _refuse_hybridized_window_edge,
    _refuse_unsupported_link_stencil,
)


# ---------------------------------------------------------------------------
# D3(c): per-axis stencil support
# ---------------------------------------------------------------------------
def test_stencil_check_passes_a_well_resolved_3d_mesh():
    _refuse_unsupported_link_stencil((8, 8, 8), where="test")   # no raise


def test_stencil_check_refuses_a_collapsed_2d_slab_axis_by_name():
    with pytest.raises(ValueError) as excinfo:
        _refuse_unsupported_link_stencil((9, 9, 1), where="sc_head_update=parallel_transport")
    msg = str(excinfo.value)
    assert "GATE pt_head_stencil_unsupported" in msg
    assert "undersampled axes z" in msg
    assert "dft_velocity" in msg          # names the actual escape hatch


def test_stencil_check_refuses_an_undersampled_periodic_axis_by_name():
    with pytest.raises(ValueError) as excinfo:
        _refuse_unsupported_link_stencil((4, 6, 6), where="test")
    assert "undersampled axes x" in str(excinfo.value)


# ---------------------------------------------------------------------------
# D3(b): independent multiplet/TRIM degeneracy at the window's top edge
# ---------------------------------------------------------------------------
def _kramers_spectrum(nk=4, n_pairs=8, seed=3):
    """(nk, 2*n_pairs) energies whose bands come in EXACT Kramers pairs.

    Same construction as ``tests/test_band_degeneracy.py``'s fixture: every
    eigenvalue doubled, so an ODD active-window band count always splits a
    pair and an EVEN one never does.
    """
    rng = np.random.default_rng(seed)
    e_pair = np.cumsum(0.05 + 0.05 * rng.random((nk, n_pairs)), axis=1)
    return np.repeat(e_pair, 2, axis=1)


def test_degenerate_window_edge_refuses_a_split_kramers_pair():
    """ODD nb_logical cuts the pair at the window top -- refused, by name."""
    e = _kramers_spectrum(n_pairs=8)          # 16 bands, pairs at (0,1)(2,3)...
    with pytest.raises(BandWindowDegeneracyError) as excinfo:
        _refuse_degenerate_window_edge(
            e, 11, where="sc_head_update=parallel_transport",
            trs_measured=True)
    msg = str(excinfo.value)
    assert "cuts a degenerate multiplet" in msg
    assert "Kramers pair is exactly degenerate" in msg
    assert "TRS measured to hold" in msg


def test_degenerate_window_edge_passes_a_clean_even_cut():
    e = _kramers_spectrum(n_pairs=8)
    _refuse_degenerate_window_edge(
        e, 12, where="test", trs_measured=True)     # no raise: clean boundary


def test_degenerate_window_edge_is_a_no_op_with_no_bands_beyond_the_window():
    """Honest scope limit: a window with nothing above it cannot see its own
    cut (band_degeneracy's own ``boundary_min_gaps`` docstring) -- this must
    NOT be mistaken for "clean" and must not raise either; it is a skip."""
    e = _kramers_spectrum(n_pairs=8)[:, :12]        # exactly nb_logical wide
    _refuse_degenerate_window_edge(e, 12, where="test")   # no raise: no-op


def test_degenerate_window_edge_names_trs_not_measured_to_hold():
    e = _kramers_spectrum(n_pairs=8)
    with pytest.raises(BandWindowDegeneracyError) as excinfo:
        _refuse_degenerate_window_edge(
            e, 11, where="test", trs_measured=False)
    assert "TRS measured NOT to hold" in str(excinfo.value)


def test_degenerate_window_edge_names_trs_not_measured_at_all():
    e = _kramers_spectrum(n_pairs=8)
    with pytest.raises(BandWindowDegeneracyError) as excinfo:
        _refuse_degenerate_window_edge(e, 11, where="test")
    assert "TRS not measured" in str(excinfo.value)


# ---------------------------------------------------------------------------
# D3(a): link singular-value hybridization at the window's top edge
# ---------------------------------------------------------------------------
def _clean_singular_values(n_source=6, nb=12, seed=7):
    """Descending per-(k,direction) singular values, all well above the
    hybridization floor -- a healthy, non-hybridized link."""
    rng = np.random.default_rng(seed)
    base = np.sort(0.9 + 0.09 * rng.random((n_source, 3, nb)), axis=-1)[
        ..., ::-1]
    return base


def test_hybridized_window_edge_passes_a_clean_window():
    sv = _clean_singular_values(nb=12)
    _refuse_hybridized_window_edge(sv, 12, where="test")   # no raise


def test_hybridized_window_edge_refuses_a_collapsed_retained_rank():
    """A single collapsed singular value INSIDE the retained window (rank <
    nb_logical) refuses, naming its source-k row, direction and rank."""
    sv = _clean_singular_values(n_source=6, nb=12)
    sv[3, 1, 9] = 4.476e-8          # source-k row 3, direction 'y', rank 10
    with pytest.raises(ValueError) as excinfo:
        _refuse_hybridized_window_edge(sv, 12, where="test")
    msg = str(excinfo.value)
    assert "GATE pt_head_window_hybridized" in msg
    assert "source-k row 3" in msg
    assert "direction 'y'" in msg
    assert "retained rank 10 of 12" in msg
    assert "4.476" in msg


def test_hybridized_window_edge_ignores_a_collapse_outside_the_window():
    """A collapsed singular value PAST nb_logical (never retained) does not
    refuse -- only the KEPT ranks matter (proposal_1's "minimum RETAINED
    singular value")."""
    sv = _clean_singular_values(n_source=6, nb=12)
    sv[3, 1, 11] = 4.476e-8         # rank 12 of 12 -- excluded when nb=10
    _refuse_hybridized_window_edge(sv, 10, where="test")   # no raise


def test_hybridization_floor_is_the_documented_round_number():
    assert _LINK_HYBRIDIZATION_FLOOR == pytest.approx(0.5)

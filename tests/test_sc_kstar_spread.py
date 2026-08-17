"""The k-star spread of Σ+V_H is a refusal, not a log line.

``KStarMap.spread_rel`` is documented as the only check that catches a
gauge mismatch introduced upstream — hermiticity, the norm and the
electron count all survive one — and under ``sc_on_ibz`` it is computed
every iteration on the seam where the full-BZ Σ meets the IBZ carry.  It
used to be formatted into a ``k-star:`` line and discarded.

Tolerance provenance is in ``sc_iteration._KSTAR_SPREAD_TOL``'s comment;
this file pins that the threshold is enforced, that it refuses loudly
rather than warning, and that NaN does not slip through the comparison.

``KStarMap`` needs a symmetry deck to build, so the map is duck-typed
here on the four members ``_check_kstar_spread`` touches — the same
device ``tests/test_scissor_weights.py`` uses for ``k_star_weights``.

Requires jax (``sc_iteration`` imports it at module scope), so this runs
in the container, not on a login node.
"""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

jax = pytest.importorskip("jax")

from gw import sc_iteration                                    # noqa: E402
from gw.sc_iteration import _KSTAR_SPREAD_TOL                  # noqa: E402


class _FakeKStar:
    """Duck type of the four ``KStarMap`` members the check reads."""

    def __init__(self, spread, nk_full=16, nk_irr=10):
        self._spread = spread
        self.nk_full = nk_full
        self.nk_irr = nk_irr
        self.reduction = nk_full / nk_irr

    def spread_rel(self, A_full):
        self.seen = np.asarray(A_full)
        return self._spread


def _check(spread):
    lines = []
    ks = _FakeKStar(spread)
    val = sc_iteration._check_kstar_spread(
        ks, np.zeros((4, 2, 2), dtype=np.complex128), print_fn=lines.append)
    return val, lines


# ---------------------------------------------------------------------------
# The tolerance itself
# ---------------------------------------------------------------------------

def test_threshold_has_headroom_over_every_measured_value():
    """1.178e-10 is the largest ``k-star:`` residual on record (job 7889590).

    Pinned so that a later tightening toward the observed maximum — the
    thing that turns a gate into a flake generator — has to change this
    line and say why.
    """
    largest_measured = 1.178e-10
    assert _KSTAR_SPREAD_TOL >= 1e3 * largest_measured
    # ... and still far below a gauge mismatch, which is O(1e-2) relative.
    assert _KSTAR_SPREAD_TOL <= 1e-4


# ---------------------------------------------------------------------------
# Accept
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spread", [0.0, 1.585e-12, 1.178e-10])
def test_healthy_spreads_pass_and_are_reported(spread):
    val, lines = _check(spread)
    assert val == spread
    assert any("k-star:" in ln and "16->10 k" in ln for ln in lines)


# ---------------------------------------------------------------------------
# Refuse
# ---------------------------------------------------------------------------

def test_spread_above_the_threshold_refuses():
    with pytest.raises(ValueError, match="k-star spread"):
        _check(1e-5)


def test_a_gauge_mismatch_sized_spread_refuses():
    """The failure the check exists for: an O(1) relative residual."""
    with pytest.raises(ValueError, match="different gauges"):
        _check(3.2e-2)


def test_nan_spread_refuses():
    """``not (x <= tol)`` rather than ``x > tol`` — NaN must not pass."""
    with pytest.raises(ValueError, match="k-star spread"):
        _check(float("nan"))


def test_the_line_is_printed_before_the_refusal():
    """The measured value has to reach the log, or the refusal is unreadable."""
    lines = []
    with pytest.raises(ValueError):
        sc_iteration._check_kstar_spread(
            _FakeKStar(1e-3), np.zeros((2, 2, 2), dtype=np.complex128),
            print_fn=lines.append)
    assert any("1.000e-03" in ln for ln in lines)


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

def test_gw_iteration_map_uses_the_checked_form():
    """No surviving bare ``spread_rel`` print on the iteration path.

    MIGRATED, not deleted, 2026-08-16.  This cell used to assert the literal
    ``_check_kstar_spread(ks, delta_h_qp`` — the check's OLD position, on the
    raw Sigma+V_H before ``apply_band_partition``.  The partition is the last
    operation that can break the star relation (a protected mask whose edge
    fell inside a degenerate multiplet gave one member off-diagonal Sigma and
    the other a scalar scissor), so checking before it ran certified an object
    the loop then rewrote; the call moved after it and its operand became the
    unfolded ``H_qp_dft_new``.

    The GUARDED PROPERTY IS UNCHANGED and is what is asserted here: the
    enforcing call is on the iteration path, and no bare ``spread_rel`` print
    came back beside it.  Pinning the operand SPELLING is what made this cell
    break on a move that preserved its intent, so it pins the call and the
    absence, not the argument text.
    """
    src = pathlib.Path(sc_iteration.__file__).read_text()
    body = src[src.index("def gw_iteration_map("):
               src.index("def _scissor_E_qp_for_outofrange(")]
    assert "_check_kstar_spread(" in body, (
        "the enforcing call left gw_iteration_map; a printed spread is not a "
        "gate, which is the whole point of this file")
    assert "ks.spread_rel(" not in body, (
        "a bare spread_rel came back on the iteration path — it must be "
        "reached only through _check_kstar_spread, which refuses")

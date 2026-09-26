"""The door: its surface, its announcements, and the runtime solve.

1. **Door reachability.**  Everything a consumer needs is a top-level
   name, and the solver half is on the door LAZILY.  A cell that only
   checked ``hasattr`` would be satisfied by an eager import, so the lazy
   half is checked for deferral as well as for presence.
2. **Announced once.**  Every rule served says where it came from, once
   per distinct request — not once per call, because a quadrature request
   repeats per q-block per SCF iteration per rank and an announcement
   nobody can read is the same as no announcement.
3. **Every rule is computed at run time.**  ``serve`` solves in process,
   refuses a retired selector by name, and refuses a family with no
   in-process solver.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

import minimax as M


# ---------------------------------------------------------------------------
#  1.  The door
# ---------------------------------------------------------------------------

def test_every_name_on_the_door_resolves():
    for name in M.__all__:
        assert hasattr(M, name), name


def test_the_solver_half_is_deferred_until_it_is_named():
    """The lazy door, from inside a process that has already imported the
    package.  ``minimax.solver`` must not be in ``sys.modules`` merely
    because ``minimax`` is — the in-process half of the claim the
    isolation suite measures in a child."""
    pytest.importorskip("scipy")
    import subprocess                              # noqa: PLC0415
    import sys                                     # noqa: PLC0415
    src = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "src")
    probe = (
        "import sys; sys.path.insert(0, %r)\n"
        "import minimax\n"
        "assert 'minimax.solver' not in sys.modules, 'eager'\n"
        "minimax.G_hgl\n"
        "assert 'minimax.solver' in sys.modules, 'never loaded'\n"
        "print('OK')\n" % (src,))
    out = subprocess.run([sys.executable, "-c", probe],
                         capture_output=True, text=True)
    assert out.stdout.strip().endswith("OK"), (out.stdout, out.stderr)


def test_the_lazy_door_refuses_a_name_it_does_not_have():
    """``__getattr__`` must not become a hole that answers anything."""
    with pytest.raises(AttributeError):
        M.not_a_solver_name           # noqa: B018


# ---------------------------------------------------------------------------
#  2.  The announcement
# ---------------------------------------------------------------------------

def _solved_lines(caught):
    return [str(w.message) for w in caught if "SOLVED" in str(w.message)]


def test_a_solve_announces_its_origin_once(isolated_cache):
    """The first serve of a request announces; the second of the SAME
    request does not."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        M.serve(family="noncrossing", target="inverse", range_value=10.0,
                error_bound=1.0e-6, n_max=64)
        M.serve(family="noncrossing", target="inverse", range_value=10.0,
                error_bound=1.0e-6, n_max=64)
    lines = _solved_lines(caught)
    assert len(lines) == 1, lines
    assert "noncrossing/inverse R=10" in lines[0]
    assert "sha256:" in lines[0] or "runtime solve" in lines[0]


def test_a_different_request_announces_separately(isolated_cache):
    """RED TWIN for the once-only rule.  Announce-once must be keyed on the
    REQUEST; a global "announced already" flag would silence the second
    rule entirely, which makes a log look clean while two different rules
    are in play."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        M.serve(family="noncrossing", target="inverse", range_value=10.0,
                error_bound=1.0e-6, n_max=64)
        M.serve(family="noncrossing", target="inverse", range_value=1000.0,
                error_bound=1.0e-6, n_max=64)
    lines = _solved_lines(caught)
    assert len(lines) == 2, lines
    assert lines[0] != lines[1]


def test_the_announcement_reset_is_not_a_no_op(isolated_cache):
    """RED TWIN for the conftest fixture.

    Every announcement cell in this suite depends on the autouse reset
    actually clearing state.  If it silently did nothing, the cells would
    still pass whenever they happened to run first — so the reset itself
    is measured.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        M.serve(family="noncrossing", target="inverse", range_value=10.0,
                error_bound=1.0e-6, n_max=64)
        M.reset_announcements()
        M.serve(family="noncrossing", target="inverse", range_value=10.0,
                error_bound=1.0e-6, n_max=64)
    assert len(_solved_lines(caught)) == 2


# ---------------------------------------------------------------------------
#  3.  Every rule is computed at run time
# ---------------------------------------------------------------------------

def test_serve_solves_in_process(isolated_cache):
    """The rule comes from a runtime solve (or its disk cache), never from
    a precomputed artifact."""
    q = M.serve(family="noncrossing", target="inverse", range_value=10.0,
                error_bound=1.0e-6, n_max=64)
    assert q.provenance.source in ("runtime-uncertified", "cache")
    assert q.provenance.certified is False
    assert q.max_error <= 1.0e-6


def test_a_retired_use_shipped_selector_refuses_rather_than_being_ignored(
        isolated_cache):
    """An un-updated caller must hear that the selector is gone; silently
    dropping it is the parsed-but-ignored-key defect (TASTE 13)."""
    with pytest.raises(M.UnknownTarget) as excinfo:
        M.serve(family="noncrossing", target="inverse", range_value=10.0,
                error_bound=1.0e-6, n_max=64, use_shipped=False)
    assert "use_shipped" in str(excinfo.value)


def test_the_solve_announces_itself_once_with_its_numbers(isolated_cache):
    """The loudest line in the service: the request, the achieved error,
    the measured Σ|w| and κ₀.

    A_dim = 20 because the point of this cell is the ANNOUNCEMENT, and a
    small bandwidth solves in about a second.
    """
    pytest.importorskip("scipy")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        q = M.serve(family="crossing", target="hgl", range_value=20.0,
                    error_bound=1.0e-6, n_max=60, eps_q=1.0e-3)
    assert q.provenance.source in ("runtime-uncertified", "cache")
    assert q.kappa0 is not None
    lines = [str(w.message) for w in caught if "SOLVED" in str(w.message)]
    assert len(lines) == 1, lines
    line = lines[0]
    assert "crossing/hgl A_dim=20" in line
    assert "sum|w|" in line and "kappa0" in line
    assert "Solved here at run time" in line
    assert ("met its target" in line) == (q.max_error <= q.error_bound)


def test_the_solved_announcement_says_when_the_target_was_missed():
    """A rule whose measured error exceeds the request is announced as a miss."""
    from minimax import door
    from minimax.records import Quadrature, runtime_provenance
    q = Quadrature(
        nodes=np.ones(3), weights=np.ones(3), family="crossing",
        target="hgl", range_param="A_dim", range_value=20.0,
        error_bound=1.0e-6, max_error=3.0e-6, kappa0=None, kappa1=None,
        provenance=runtime_provenance("x", "numpy"))
    door._SERVE_ANNOUNCED.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        door._announce_solved(q, sum_abs_w=3.0, n_max=3)
    line = str(caught[-1].message)
    assert "MISSED its target" in line and "met its target" not in line


def test_a_family_with_no_in_process_solver_refuses_rather_than_hanging():
    """``complex_laplace`` has no in-process solver, so it refuses by name."""
    with pytest.raises(M.UncertifiedSolveRefused) as excinfo:
        M.serve(family="complex_laplace", target="complex_laplace",
                range_value=21.544346900318832, error_bound=1.0e-6, n_max=64)
    assert "complex_laplace" in str(excinfo.value)
    assert "no in-process solver" in str(excinfo.value)
    assert M.FAMILIES["complex_laplace"].wired is False

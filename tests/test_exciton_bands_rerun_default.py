"""Gate: the exciton-bands warm re-run check is OFF by default, ON on request.

The warm re-run at ``exciton_bands.solve_path`` re-solves the ENTIRE Q scan a
second time and asserts the two tables agree to 1e-10.  It buys no physics.
``PROFILE_htransform_exciton.md`` measured it at **37.7 % of driver wall** at
P=4 / 41 Q, confirming the 38.1 % seen at P=64 / 91 Q in job 7882533 — the
largest single row in the driver's stage table, in both geometries.

On 2026-08-08 the owner approved flipping the default: skipping is what you get
unless you ask, and the diagnostic moved behind an explicit ``--rerun-check``.

A default flip is exactly the kind of change that rots into a lie, in two
different directions, and this file pins both:

1. **The default really is off, and the opt-in really turns it back on.**
   Tested through the driver's own parser and the driver's own predicate, not
   through a restatement of either.  A test that only checked
   ``rerun_check_enabled(Namespace(rerun_check=True))`` would stay green if the
   ``--rerun-check`` flag were deleted from the CLI tomorrow.

2. **The predicate is actually wired to the diagnostic.** Everything in (1) is
   theatre if ``solve_path`` still branches on something else — the predicate
   would be a green, well-tested, entirely unread function while the re-run
   went on costing 37.7 % of every wall.  The AST gate below reads the real
   ``if`` that guards the real re-run and insists its test is a call to
   ``rerun_check_enabled``.  That is the assertion that makes this a red twin
   for the flip rather than for a helper.

Source-level plus argparse only: no solve, no fixture, no GPU.  The behaviour
under a real solve is covered by ``test_exciton_bands.py``, which is
fixture-gated and costs minutes.
"""
from __future__ import annotations

import ast
import os

import pytest

SRC = os.path.join(os.path.dirname(__file__), "..", "src", "bse",
                   "exciton_bands.py")

BASE_ARGV = ["-i", "cohsex.in"]


# ---------------------------------------------------------------------------
# (1) the CLI contract, through the driver's own parser and predicate
# ---------------------------------------------------------------------------

def _driver():
    pytest.importorskip("jax")
    from bse import exciton_bands
    return exciton_bands


def _decide(argv):
    """What the driver would decide for this command line."""
    exciton_bands = _driver()
    args = exciton_bands.build_parser().parse_args(BASE_ARGV + list(argv))
    return exciton_bands.rerun_check_enabled(args)


def test_rerun_check_is_off_by_default():
    """Bare command line ⇒ no warm re-run.  This is the flip itself."""
    assert _decide([]) is False, (
        "the exciton-bands warm re-run is back on by default; that is 37.7% "
        "of driver wall spent on a reproducibility assert nobody asked for")


def test_rerun_check_flag_turns_the_diagnostic_back_on():
    """``--rerun-check`` ⇒ warm re-run.  The opt-in has to actually opt in."""
    assert _decide(["--rerun-check"]) is True, (
        "--rerun-check did not re-enable the warm re-run; the old behaviour "
        "is now unreachable from the CLI")


def test_skip_rerun_check_still_parses_and_still_skips():
    """The pre-flip flag is a no-op, not an error.

    Campaign harnesses and the archived launch recipes pass
    ``--skip-rerun-check``.  Removing it would turn every one of those into an
    argparse SystemExit, which on a batch node reads as a crashed run.
    """
    assert _decide(["--skip-rerun-check"]) is False


def test_skip_wins_over_rerun_check_when_both_are_passed():
    """A flag whose name says "skip" must never switch the re-run on."""
    assert _decide(["--rerun-check", "--skip-rerun-check"]) is False
    assert _decide(["--skip-rerun-check", "--rerun-check"]) is False


def test_help_text_states_the_default():
    """The flip has to be discoverable from ``--help``, not just from git."""
    exciton_bands = _driver()
    actions = {a.dest: a for a in exciton_bands.build_parser()._actions}
    assert "rerun_check" in actions, "--rerun-check is gone from the CLI"
    assert actions["rerun_check"].default is False
    assert actions["skip_rerun_check"].default is False
    rerun_help = (actions["rerun_check"].help or "").lower()
    skip_help = (actions["skip_rerun_check"].help or "").lower()
    assert "default" in rerun_help or "off by default" in rerun_help, (
        "--rerun-check's help does not say the diagnostic is off by default")
    assert "default" in skip_help, (
        "--skip-rerun-check's help does not say it now names the default")


# ---------------------------------------------------------------------------
# (2) the wiring: the real re-run is guarded by the real predicate
# ---------------------------------------------------------------------------

def _source_tree():
    with open(SRC, encoding="utf-8") as fh:
        return ast.parse(fh.read(), filename=SRC)


def _rerun_guard():
    """The ``if`` statement that guards the warm re-run, found by its body.

    Located by the ``solve_scan_warm`` timing tick rather than by line number
    or by the guard's own text, so this test cannot be satisfied by moving the
    branch or by renaming the predicate into place somewhere harmless.
    """
    hits = []
    for node in ast.walk(_source_tree()):
        if not isinstance(node, ast.If):
            continue
        marks = [
            n for n in ast.walk(node)
            if isinstance(n, ast.Constant) and n.value == "solve_scan_warm"
        ]
        if marks:
            hits.append(node)
    assert hits, ("no `if` in exciton_bands.py contains the solve_scan_warm "
                  "tick — the warm re-run is unguarded, or it moved")
    # innermost enclosing `if` wins if the site is nested
    return min(hits, key=lambda n: (n.end_lineno - n.lineno))


def test_the_warm_rerun_is_guarded_by_the_predicate():
    """The branch the driver takes is the one the tests above exercise."""
    guard = _rerun_guard()
    test = guard.test
    assert isinstance(test, ast.Call) and isinstance(test.func, ast.Name) \
        and test.func.id == "rerun_check_enabled", (
            "the warm re-run is guarded by "
            f"`{ast.unparse(test)}`, not by rerun_check_enabled(args); the "
            "default-flip tests in this file are then testing a function the "
            "driver does not consult")
    assert len(test.args) == 1 and isinstance(test.args[0], ast.Name) \
        and test.args[0].id == "args", (
            f"rerun_check_enabled called as `{ast.unparse(test)}` — it must be "
            "handed the parsed args namespace")


def test_the_guard_is_positive_not_negated():
    """``if rerun_check_enabled(args)``, never ``if not ...``.

    A negation here would invert the whole flip while leaving every other test
    in this file green: the predicate would be correct, the wiring would be
    present, and the re-run would fire on exactly the runs that did not ask.
    """
    guard = _rerun_guard()
    assert not isinstance(guard.test, ast.UnaryOp), (
        f"the warm-re-run guard is `{ast.unparse(guard.test)}` — a negation "
        "inverts the default flip")


def test_the_skip_branch_still_reports_itself():
    """The else-branch must say the check was skipped.

    Silence is how a default flip becomes invisible: a log that stops
    mentioning the re-run leaves every reader of an old log unable to tell
    whether this run checked reproducibility or not.
    """
    guard = _rerun_guard()
    assert guard.orelse, "the warm-re-run guard has no else branch"
    text = "\n".join(ast.unparse(n) for n in guard.orelse)
    assert "SKIPPED" in text, (
        "the skip path no longer logs SKIPPED; a reader cannot tell from the "
        "log whether the reproducibility check ran")
    assert "--rerun-check" in text, (
        "the skip path does not name --rerun-check; a reader who wants the "
        "check back has nothing to search for")

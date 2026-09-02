"""The documented quickstart, run as documented, on the artifacts we ship.

WHY THIS FILE EXISTS
--------------------
On 2026-09-01 the command the front page tells a new user to run —

    python -m gw.gw_jax -i tests/regression/cohsex_debug/cohsex_test.in

— **refused on ``origin/main``**, twice over, and no gate saw it:

* under the deck's default ``head_correction = full`` it stopped at
  ``GATE dft_head_dipole_provenance``, because the shipped ``dipole.h5``
  predated the provenance stamps and could not authenticate;
* under ``head_correction = no_local_fields`` it stopped instead in
  ``file_io/kin_ion.py``, because the shipped ``kin_ion.h5`` carried no
  bispinor-representation stamp.

``test_gw_jax_regression.py::test_gw_jax_matches_reference[cohsex]`` runs the
same deck, so it was red too — but it is a NUMBERS gate, and a numbers gate
that cannot start reads as "the physics moved" rather than "the shipped
fixture no longer authenticates".  Those are different repairs.  This file is
the door check: does the documented command, on the artifacts in the tree,
still get to the end?

WHAT THIS FILE CHECKS
---------------------
1. ``test_the_documented_quickstart_deck_completes_on_the_shipped_fixture``
   runs the deck end to end on a COPY of the fixture and asserts the driver
   reached its completion line and wrote all three eqp tables.  It asserts
   NO numbers.
2. ``test_the_head_policy_the_quickstart_runs_under_is_the_default`` pins the
   thing the 2026-09-01 refusal was really about: the deck must not carry a
   ``head_correction`` key, so the run above exercises the SHIPPING default.
   Pointing the quickstart at ``head_correction = off`` would make cell 1
   green and mean nothing — ``off`` is debug-only by the owner ruling of
   2026-09-01.
3. ``test_the_docs_still_name_the_deck_this_file_runs`` is a cheap
   documentation-consistency cell: the two pages that publish the command
   must still name a deck that exists.  It matches the literal command
   string; it does not read policy out of prose.

WHAT THIS FILE DOES **NOT** CHECK — read this before citing it
--------------------------------------------------------------
* **No numbers.**  Not one energy is compared.  A run that completes with
  wrong physics passes here.  ``eqp_ref.dat`` is the numbers gate and it
  lives in ``test_gw_jax_regression.py``; this cell is upstream of it.
* **Not the literal repo-root invocation.**  ``gw_jax`` resolves every path
  in the deck — including ``tmp/`` and the eqp writers — against the DECK's
  directory, not the CWD, so the verbatim in-tree command writes INTO
  ``tests/regression/cohsex_debug/``.  ``harness.protect_fixtures()`` keeps
  that directory read-only from the first moment of any pytest session, so
  running it literally in-tree fails on ``EACCES`` and would say nothing
  about the code.  The cell therefore copies the fixture and runs the same
  command shape against ``<copy>/cohsex_test.in`` from OUTSIDE the deck
  directory, which is what exercises the deck-relative resolution.  It does
  not prove the repo-root form works on a fresh clone.
* **One rank, one GPU.**  It says nothing about P=4 placement, sharding or
  collectives.
* **Not the other decks.**  ``cohsex_debug`` is the only regression deck that
  ships its own pseudopotentials, so it is the only one whose artifacts can
  be regenerated from a clean checkout; nothing here covers ``gnppm``,
  ``bispinor``, Si or hBN.
"""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import (          # noqa: E402
    REG,
    REPO_ROOT,
    copy_fixture,
    run_gw_jax,
    skip_unless_gpu,
)

CASE_DIR = REG / "cohsex_debug"
DECK = "cohsex_test.in"

#: The command the docs publish, verbatim, minus the runner prefix (``uv run``
#: / ``lx run``) that differs per page.  Matched as a literal string; nothing
#: is parsed out of the surrounding prose.
DOCUMENTED_INVOCATION = (
    "python -m gw.gw_jax -i tests/regression/cohsex_debug/cohsex_test.in")

#: The line the driver prints last on a successful run (``gw_output``).
COMPLETION_MARKER = "LORRAX GW calculation completed"

#: Every table the quickstart is advertised to produce.
EXPECTED_OUTPUTS = ("eqp_test.dat", "eqp0_test.dat", "eqp1_test.dat")


@pytest.mark.regression
def test_the_documented_quickstart_deck_completes_on_the_shipped_fixture(
        tmp_path):
    """The shipped artifacts still authenticate under the shipped default.

    SCOPE: completion and file presence only — see this module's docstring.
    """
    skip_unless_gpu(pytest)
    # Run from OUTSIDE the deck directory, addressing the deck by path, which
    # is the form the docs publish and the form that exercises the
    # deck-relative resolution of every input and output.
    run_dir = copy_fixture(CASE_DIR, tmp_path / "cohsex_debug")
    result = run_gw_jax(tmp_path, f"{run_dir.name}/{DECK}")

    # Check the ARTIFACTS first and the exit code second: rc is not evidence
    # on this platform (rc=134 has been observed on runs whose outputs were
    # written correctly), but a missing eqp table always is.
    missing = [n for n in EXPECTED_OUTPUTS if not (run_dir / n).is_file()]
    assert not missing, (
        f"the documented quickstart did not write {missing}.\n"
        f"stdout tail:\n{result.stdout[-4000:]}\n"
        f"stderr tail:\n{result.stderr[-2000:]}")
    assert COMPLETION_MARKER in result.stdout, (
        "the run wrote its tables but never reached "
        f"{COMPLETION_MARKER!r}.\nstdout tail:\n{result.stdout[-4000:]}")
    assert result.returncode == 0, (
        f"tables were written but the driver exited {result.returncode}.\n"
        f"stderr tail:\n{result.stderr[-2000:]}")
    for name in EXPECTED_OUTPUTS:
        assert (run_dir / name).stat().st_size > 0, f"{name} is empty"


def test_the_head_policy_the_quickstart_runs_under_is_the_default():
    """The quickstart must exercise the SHIPPING head, not a debug override.

    ``head_correction = off`` would make the cell above green while removing
    the q→0 completion the refusal was about.  The deck is required to stay
    silent on the key so the run resolves the default (``full``).
    """
    deck = (CASE_DIR / DECK).read_text()
    body = "\n".join(
        line for line in deck.splitlines() if not line.lstrip().startswith("#"))
    assert "head_correction" not in body, (
        "cohsex_test.in now sets head_correction; the quickstart gate above "
        "then stops testing the shipping default.  If the key is genuinely "
        "needed, this assertion is the place to record why.")


@pytest.mark.parametrize("page", ["docs/index.md", "docs/quickstart.md"])
def test_the_docs_still_name_the_deck_this_file_runs(page):
    """The published command must name a deck that exists.

    A literal-string check on purpose: reading policy out of prose is how
    ``test_env_registry`` disabled itself with the sentence that documented
    it.  This asserts only that the page still contains the command and that
    the path inside the command resolves.
    """
    text = (REPO_ROOT / page).read_text()
    assert DOCUMENTED_INVOCATION in text, (
        f"{page} no longer publishes {DOCUMENTED_INVOCATION!r}; either the "
        "docs moved the quickstart or this gate is now testing a command "
        "nobody is told to run.")
    deck_path = re.search(r"-i (\S+cohsex_test\.in)", DOCUMENTED_INVOCATION)
    assert deck_path, "the invocation constant lost its -i argument"
    assert (REPO_ROOT / deck_path.group(1)).is_file(), (
        f"{page} points at {deck_path.group(1)}, which does not exist")

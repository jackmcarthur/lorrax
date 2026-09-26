"""The refusals, each with the case where it returns FALSE.

``SERVICE_FORM.md:46`` — *every new check ships with the case where it
returns FALSE, no exceptions*.  A cell asserting "the door refuses X" is
worth very little on its own: a door that refused everything would pass it.
The paired cell is what makes it evidence.

===  ==========================  =========================================
 #   refusal                     the FALSE case shipped beside it
===  ==========================  =========================================
 F3  ``UnknownTarget``           every vocabulary member resolves
 F5  ``UncertifiedSolveRefused`` ``test_minimax_door.py``: a family with a
                                 solver is served
 F6  ``SamplingUnsupported``     the three live cells resolve
===  ==========================  =========================================
"""

from __future__ import annotations

import pytest

import minimax as M


# ---------------------------------------------------------------------------
#  F3 — unknown target / unknown family / unknown selector
# ---------------------------------------------------------------------------

def test_f3_an_undeclared_target_refuses_and_lists_the_vocabulary():
    with pytest.raises(M.UnknownTarget) as excinfo:
        M.serve(family="noncrossing", target="quadratic",
                range_value=10.0, error_bound=1.0e-6, n_max=64)
    assert "Declared targets" in str(excinfo.value)


def test_f3_a_target_the_family_cannot_serve_refuses():
    """'hgl' is a real target and 'noncrossing' is a real family; the PAIR
    is the error.  Accepting it would serve a 1/x rule to a caller who
    asked for a sign-regularization, which is a wrong answer rather than a
    missing one."""
    with pytest.raises(M.UnknownTarget) as excinfo:
        M.serve(family="noncrossing", target="hgl",
                range_value=10.0, error_bound=1.0e-6, n_max=64)
    assert "cannot serve" in str(excinfo.value)


def test_f3_an_unknown_selector_refuses_rather_than_being_ignored():
    """A silently-dropped selector is how you serve a rule fitted to a
    different function."""
    with pytest.raises(M.UnknownTarget) as excinfo:
        M.serve(family="noncrossing", target="inverse", range_value=10.0,
                error_bound=1.0e-6, n_max=64, beta=3.0)
    assert "beta" in str(excinfo.value)


def test_f3_false_case_every_declared_vocabulary_member_resolves():
    """The FALSE case, and it is also the anti-rot check on the tables:
    a target declared in :data:`minimax.TARGETS` with no family, or a
    family naming a target that is not declared, would make the vocabulary
    a lie that nothing catches."""
    for name, spec in M.FAMILIES.items():
        assert spec.target in M.TARGETS, (name, spec.target)
        assert spec.character in M.CHARACTERS, (name, spec.character)
    # every target is reachable from some family, except the one the
    # design registers as a hole on purpose
    served = {f.target for f in M.FAMILIES.values()} | {"hgl", "fermi"}
    assert set(M.TARGETS) - served == set(), set(M.TARGETS) - served


# ---------------------------------------------------------------------------
#  F6 — the 2x2's empty cell
# ---------------------------------------------------------------------------

def test_f6_the_strip_cell_refuses_by_name():
    """Both parts of z nonzero is where the MPA fit stage lives, and no
    quadrature family serves it.  The refusal happens before any physics
    runs, from declarative data, rather than somewhere inside a kernel.
    """
    with pytest.raises(M.SamplingUnsupported) as excinfo:
        M.family_for_character("strip")
    text = str(excinfo.value)
    assert "damped_line" in text
    assert "strip" in text, text


def test_f6_false_case_the_three_live_cells_resolve():
    """The FALSE case, and the 2×2 read out loud."""
    assert M.family_for_character("static") == "noncrossing"
    assert M.family_for_character("real") == "crossing"
    assert M.family_for_character("imag") == "noncrossing_imag"


def test_f6_a_character_that_is_not_one_of_the_four_refuses():
    with pytest.raises(M.SamplingUnsupported) as excinfo:
        M.family_for_character("diagonal")
    assert "not an analytic character" in str(excinfo.value)

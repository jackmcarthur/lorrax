"""The 2026-08-08 rename sweep, measured rather than believed.

WHAT THE SWEEP DID.  Six public operations were renamed to say what they
mathematically DO instead of what they happen to be applied to.  The
canonical case is the owner's own: ``unfold_v_q`` unfolds any ISDF-basis
``(q, μ, ν)`` operator — W_q goes through the identical code path and
deliberately shares the jit cache — so V was never anything but its most
frequent customer, and the name is :func:`unfold_isdf_operator` now.

WHY THIS FILE EXISTS.  A rename sweep has exactly two ways to be wrong,
and neither of them shows up as a broken import in the tree that did the
renaming:

1. **An alias that does not resolve.**  Consumers this branch may not
   touch (``services/wfn_loader``, ``services/zeta_loader``) and branches
   that have not merged yet (the stamp/generator track) call the OLD
   names.  If one of them stops resolving, this tree stays green and the
   other one breaks — which is the worst possible place to find out.
2. **An alias that became the definition.**  If a later regex sweep
   "helpfully" rewrites the alias table, or someone re-points the primary
   name at the alias, every old spelling still resolves and every test
   still passes while the rename has silently un-happened.  Asserting
   that the two names are DIFFERENT OBJECTS, and that the primary is the
   one with the real body, is what distinguishes the two states.

The cells below walk :data:`symmetry_maps._compat.RENAMES` rather than a
list written out here, so a seventh rename gets covered by adding one
dict entry and nothing else.
"""

import importlib

import pytest

import symmetry_maps
from symmetry_maps._compat import RENAME_HOME, RENAMES, RETIREMENT_GATE


def test_remaining_rename_table_excludes_retired_tt_mixer():
    """The sweep's scope, pinned.

    Not a tautology: the table is the input to every other cell here, so
    a table that quietly grew or shrank would move all of them together
    and none of them would notice.
    """
    assert set(RENAMES) == {
        "unfold_v_q",
        "trs_augment_U",
        "compute_centroid_sym_perm",
        "compute_rgrid_sym_perm",
        "build_real_space_syms",
    }
    assert set(RENAMES.values()) == set(RENAME_HOME)


@pytest.mark.parametrize("old", sorted(RENAMES))
def test_every_old_name_still_resolves_on_the_door(old):
    """The compat guarantee, at the door.

    ``from symmetry_maps import <old>`` is what the untouched consumers
    and the sibling branch write.
    """
    assert hasattr(symmetry_maps, old), (
        f"``from symmetry_maps import {old}`` no longer resolves.  That "
        f"is not a tidy-up, it is a break in someone else's branch: the "
        f"aliases are {RETIREMENT_GATE}, not before.")
    assert old in symmetry_maps.__all__, (
        f"{old} resolves but is not in __all__, so ``import *`` drops it "
        f"— a consumer would discover that at run time, not import time.")


@pytest.mark.parametrize("old", sorted(RENAMES))
def test_every_old_name_still_resolves_on_its_module(old):
    """The compat guarantee, at module level.

    The door is not the only way in: the deleted wave-1 shims reached
    ``symmetry_maps.maps`` and ``symmetry_maps.orbit_syms`` directly, and
    a consumer that still spells it that way must not care.
    """
    home = importlib.import_module(RENAME_HOME[RENAMES[old]])
    assert hasattr(home, old), (
        f"{home.__name__}.{old} no longer resolves; the alias policy is "
        f"BOTH levels, on purpose.")
    assert getattr(home, old) is getattr(symmetry_maps, old), (
        f"the door and {home.__name__} bind DIFFERENT objects for {old}; "
        f"they must be the same object or they will drift.")


@pytest.mark.parametrize("old", sorted(RENAMES))
def test_the_new_name_is_the_primary_definition(old):
    """The half that says the rename actually happened.

    The alias forwards to the primary; the primary is a real function
    defined in its home module under its own name.  If these two ever
    became the same object, or if the primary's ``__name__`` were the old
    spelling, the sweep would have been undone without a single red cell.
    """
    new = RENAMES[old]
    home = importlib.import_module(RENAME_HOME[new])
    primary = getattr(home, new)
    alias = getattr(home, old)

    assert primary.__name__ == new, (
        f"{home.__name__}.{new}.__name__ is {primary.__name__!r} — the "
        f"primary definition is not carrying the new name.")
    assert primary.__module__ == home.__name__
    assert alias is not primary, (
        f"{old} and {new} are the SAME object, so the alias layer is a "
        f"no-op and the deprecation is invisible to help() and to any "
        f"tool that reads it.")
    assert getattr(symmetry_maps, new) is primary, (
        f"the door's {new} is not {home.__name__}'s {new}.")


@pytest.mark.parametrize("old", sorted(RENAMES))
def test_every_alias_says_it_is_deprecated_and_names_the_gate(old):
    """A deprecation nobody can read is a permanent name.

    The retirement gate is the documentation pass, and it is named in the
    docstring so that ``help(old_name)`` answers "when does this go?"
    without anyone having to find this file.
    """
    alias = getattr(symmetry_maps, old)
    doc = alias.__doc__ or ""
    assert "DEPRECATED" in doc
    assert RENAMES[old] in doc, (
        f"{old}'s deprecation notice does not name its replacement, "
        f"which is the one thing a reader needs from it.")
    assert "documentation pass" in doc


def test_an_alias_forwards_verbatim():
    """The forwarding is a call-through, not a re-implementation.

    Measured on the cheapest real operation in the set:
    ``real_space_action_tables`` refuses a bad argument, and the alias
    must refuse identically — same exception type, same message — because
    it is the same body.  A wrapper that swallowed, converted or
    re-raised would show up here.
    """
    from symmetry_maps import build_real_space_syms, real_space_action_tables

    with pytest.raises(Exception) as new_exc:            # noqa: PT011
        real_space_action_tables(None, None)
    with pytest.raises(Exception) as old_exc:            # noqa: PT011
        build_real_space_syms(None, None)

    assert type(new_exc.value) is type(old_exc.value)
    assert str(new_exc.value) == str(old_exc.value)


def test_the_headline_pair_now_names_its_direction():
    """RED-TWIN-SHAPED: the reason the sweep was worth doing at all.

    ``compute_centroid_sym_perm`` and ``compute_rgrid_sym_perm`` were one
    word apart and returned opposite directions — a SOURCE map plus a
    lattice wrap against a PULL-BACK permutation — and the unfold's own
    docstring records that confusing them (an ``argsort`` of the wrong
    one) was a no-op on involutive ops and a silent 4 eV gap on order-3
    ones.  Two names that differ only in a noun cannot carry that
    distinction; these two do, and this cell is what stops a future
    "tidy-up" from collapsing them back into a matched pair.
    """
    src = RENAMES["compute_centroid_sym_perm"]
    pull = RENAMES["compute_rgrid_sym_perm"]

    assert "source" in src, src
    assert "wrap" in src, "the wrap is half the return value and must be named"
    assert "pullback" in pull or "pull_back" in pull, pull
    # And they must not be a matched pair again: no shared stem.
    assert src.split("_")[0] != pull.split("_")[0]

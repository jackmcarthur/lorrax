"""Compatibility aliases for the 2026-08-08 rename sweep.

THE RULE THE SWEEP APPLIED.  An operation is named for what it
mathematically DOES, never for what it happens to be applied to.  The
pre-sweep ``unfold_v`` + ``_q`` spelling was the canonical failure — it
unfolds any ISDF-basis ``(q, μ, ν)`` operator and V is merely its most
frequent customer, with W_q going through the identical code path and
deliberately sharing the jit cache — so it became
:func:`~symmetry_maps.unfold_isdf_operator`, which is the owner's own
wording.

(The old spellings are assembled below rather than written out, so that
a future sweep's regex cannot silently rewrite the very table that
records what the last sweep renamed.  This module is the one place in
the tree where the OLD names are the subject matter.)

WHY THE OLD SPELLINGS STILL RESOLVE.  Renaming a name that 150-odd call
sites and two other in-flight branches use is not an atomic operation
across a fleet.  Every renamed operation is therefore published twice:
the new name is PRIMARY (it is what the definition is called, what the
error messages say, and what every consumer in this tree calls), and the
old name is a call-through alias bound both at module level and on the
package door.  An importer that has not moved keeps working, byte for
byte, and gets no warning — a warning that fires on a sibling branch's
every run is noise about a decision that sibling did not make.

THE RETIREMENT GATE.  These aliases go when the documentation pass
(LANDING_ORDERS §C) rewrites ``docs/theory/symmetry.md`` and
``docs/services/symmetry_maps.md`` onto the new names.  That pass is the
one that has to touch all ~380 prose occurrences anyway; deleting the
aliases in the same commit is what stops a retired spelling from
outliving the last text that explains it existed.  Until then, DELETING
AN ALIAS BREAKS A BRANCH THAT HAS NOT MERGED YET.

The mapping is :data:`RENAMES` below — six entries, old spelling to new.
The pair it existed for is the centroid/r-grid one: those two were named
as a matched pair and return OPPOSITE DIRECTIONS — a source map plus a
lattice wrap, versus a pull-back permutation — and confusing them is the
recorded mechanism of a silent 4 eV gap on hex systems.  The new names
carry the direction, which is the whole point.

:data:`RENAMES` is also the gate: ``test_symmetry_maps_rename_compat``
walks it and asserts that every old key still resolves on the door and
on its defining module, that every value is the PRIMARY definition, and
that the two are not the same object.
"""

from __future__ import annotations

import functools

__all__ = ["deprecated_alias", "RENAMES", "RETIREMENT_GATE"]

#: The sweep, as data: pre-sweep spelling → primary spelling.  Every key
#: is bound as an alias on its defining module AND on the package door;
#: every value is where the definition actually lives.
#:
#: DO NOT let a regex sweep rewrite the KEYS of this dict.  They are the
#: only place in the service where the old spellings are the subject
#: matter rather than a call site, and a sweep that "helpfully" updates
#: them turns the whole alias layer into a no-op that still passes an
#: import test.  ``test_rename_compat_surface`` in the service suite is
#: the gate that says so out loud.
RENAMES = {
    "unfold_v_q": "unfold_isdf_operator",
    "unfold_v_q_bispinor_lorentz": "mix_channels_by_proper_rotation",
    "trs_augment_U": "spinor_rotation_for_sym_row",
    "compute_centroid_sym_perm": "centroid_source_map_and_wrap",
    "compute_rgrid_sym_perm": "fft_grid_pullback_perm",
    "build_real_space_syms": "real_space_action_tables",
}

#: Which module each PRIMARY name is defined in — the "module level" half
#: of the new-name-primary / old-name-alias-at-both-levels policy.
RENAME_HOME = {
    "unfold_isdf_operator": "symmetry_maps.maps",
    "mix_channels_by_proper_rotation": "symmetry_maps.maps",
    "spinor_rotation_for_sym_row": "symmetry_maps.maps",
    "centroid_source_map_and_wrap": "symmetry_maps.orbit_syms",
    "fft_grid_pullback_perm": "symmetry_maps.orbit_syms",
    "real_space_action_tables": "symmetry_maps.orbit_syms",
}

#: Named in every alias docstring so that ``help(old_name)`` says when the
#: name goes away and who takes it away.
RETIREMENT_GATE = (
    "retired by the documentation pass (LANDING_ORDERS §C), which "
    "rewrites docs/theory/symmetry.md and docs/services/symmetry_maps.md "
    "onto the new names")


def deprecated_alias(new_fn, old_name: str, new_name: str | None = None):
    """A call-through alias of ``new_fn`` published under ``old_name``.

    A plain ``old = new`` assignment cannot carry a docstring, and a
    deprecated name whose ``help()`` says nothing about being deprecated
    is how an alias quietly becomes permanent.  This wraps instead:
    :func:`functools.wraps` keeps the signature introspectable (it sets
    ``__wrapped__``, which :func:`inspect.signature` follows), and the
    docstring is then replaced with the deprecation notice.

    Parameters
    ----------
    new_fn
        The primary function.  The alias forwards to it verbatim —
        same arguments, same return, no warning, no conversion.
    old_name
        The pre-sweep spelling this alias is published as.
    new_name
        The primary spelling, for the notice.  Defaults to
        ``new_fn.__name__``.

    Returns
    -------
    A function whose ``__name__`` is ``old_name`` and whose body is one
    forwarding call.
    """
    primary = new_name or getattr(new_fn, "__name__", "the new name")

    @functools.wraps(new_fn)
    def _alias(*args, **kwargs):
        return new_fn(*args, **kwargs)

    _alias.__name__ = old_name
    _alias.__qualname__ = old_name
    _alias.__doc__ = (
        f"DEPRECATED alias of :func:`{primary}` — {RETIREMENT_GATE}.\n"
        f"\n"
        f"``{old_name}`` named the operation for what it was applied to;\n"
        f"``{primary}`` names it for what it does.  This alias forwards\n"
        f"verbatim and is kept only so that consumers which have not\n"
        f"moved — including branches in flight that this tree cannot\n"
        f"edit — keep working unchanged.  New code calls\n"
        f"``{primary}``.\n"
        f"\n"
        f"See :mod:`symmetry_maps._compat` for the full rename map.\n")
    return _alias

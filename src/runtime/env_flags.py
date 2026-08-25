"""THE boolean-environment grammar.  One parse, one announcement, one home.

WHY THIS MODULE EXISTS.  A boolean env knob has exactly two interesting
failure modes and this project has met both, repeatedly:

* the **presence test** — ``if os.environ.get(NAME):`` — under which
  ``NAME=0`` turns the knob **ON**.  ``LORRAX_MEM_DEBUG`` shipped that way
  at four sites, ``LORRAX_EXIT_AFTER_ZETA`` ended production runs with
  ``SystemExit(0)`` on ``=0``, and ``LORRAX_PHDF5_TIME`` carried it in C++
  until 2026-08-06;
* the **silent swallow** — an unrecognised token resolving to one of the
  two answers with nothing said.  ``LORRAX_MALLOC_TRIM=OFF`` left the trim
  hook on while its documented sibling ``LORRAX_MALLOC_TUNE=OFF``
  correctly turned off, because one parse was case-sensitive.

``gw.gw_config.env_bool`` fixed both for the four files IT owns.  It could
not fix them anywhere else: it is L1, and the parsers that still swallowed
live in ``file_io`` and ``runtime``, which are L3 and may not import
uphill.  So the grammar moved DOWN here — L3, no jax, no config, importable
by anything — and ``gw_config.env_bool`` is now a re-export.  The defect
register's own prescription, verbatim: *"one announcing helper in
``common/``, imported by all five (gw_config is the wrong home for a
file_io/ffi dependency)"*.

THE GRAMMAR, and it is deliberately small:

===============================  ==========================================
unset, or empty/whitespace       the caller's stated default
``1 true yes on``  (any case)    ``True``
``0 false no off`` (any case)    ``False``
anything else                    ``False``, **announced once per (name,
                                 value)** with the ``*** LORRAX SANITY``
                                 marker
===============================  ==========================================

The last row is the one worth arguing about, and the answer is inherited
rather than re-decided: ``tests/test_env_grammar.py`` pins unrecognised ->
``False``, and ``TASTE.md`` rule 13 says a typo must never REFUSE.  So an
unrecognised token resolves, in a stated direction, and says so — which is
what makes the direction safe to depend on.

THE C++ TWIN.  ``src/ffi/cpp/phdf5/ctx.h::env_flag`` implements the same
table for the native layer, which cannot call Python.  Two transcriptions
of one grammar; ``tests/test_env_grammar.py`` is what holds them in step.
"""
from __future__ import annotations

import os

__all__ = [
    "env_bool",
    "env_falsy",
    "reset_env_announce_state",
    "ANNOUNCED",
    "ENV_TRUE",
    "ENV_FALSE",
]

#: Accepted spellings, lower-cased.  Kept as module constants because two
#: refusal messages and one C++ transcription quote them.
ENV_TRUE = ("1", "true", "yes", "on")
ENV_FALSE = ("0", "false", "no", "off")

#: ``(name, raw)`` pairs already announced.  Exported as ``ANNOUNCED``
#: because ``gw_config.env_float`` — the numeric twin of this grammar,
#: which stays there — shares the memo, so one knob spelled wrong in both
#: a boolean and a numeric read does not print two lines and one
#: ``reset_env_announce_state()`` clears both.  Once per distinct value, not
#: once per read: a knob read at four sites must not print four lines, and
#: a knob whose value CHANGES mid-process (tests do this) must still get a
#: line for the new one.
_ANNOUNCED: set = set()

#: Public alias for the memo above (see ``gw_config.env_float``).
ANNOUNCED = _ANNOUNCED


def reset_env_announce_state() -> None:
    """Forget which grammar errors have been announced (tests only)."""
    _ANNOUNCED.clear()


def env_bool(name: str, default: bool, *, print_fn=print) -> bool:
    """``name`` as a boolean, under the grammar in the module docstring.

    Parameters
    ----------
    name
        Environment variable, e.g. ``"LORRAX_DEBUG_PRINT"``.
    default
        Value when the variable is unset or blank.  This is the knob's
        DOCUMENTED default, not a guess — ``docs/dev/env_vars.md`` is the
        table it has to agree with.
    print_fn
        Where the announcement goes.  Defaults to ``print`` so a caller
        that has no logger still gets the line.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    tok = raw.strip().lower()
    if tok in ENV_TRUE:
        return True
    if tok in ENV_FALSE:
        return False
    key = (name, raw)
    if key not in _ANNOUNCED:
        _ANNOUNCED.add(key)
        print_fn(
            f"  *** LORRAX SANITY: {name}={raw!r} is not a recognised "
            f"boolean.  Accepted: {'/'.join(ENV_TRUE)} (on), "
            f"{'/'.join(ENV_FALSE)} (off), unset/blank (default="
            f"{'on' if default else 'off'}).  Treating it as OFF. ***")
    return False


def env_falsy(name: str, default: str = "1", *, print_fn=print) -> bool:
    """True when knob ``name`` reads as OFF.  The inverse spelling.

    ``runtime`` has several knobs whose call sites read naturally as *"is
    this turned off"* (``LORRAX_MALLOC_TUNE``, ``LORRAX_CPU_SKIP_GPU_PLUGINS``)
    and inverting at each site is how a reader loses track of which
    direction the default points.  This is one negation over
    :func:`env_bool`, not a second grammar — which is the whole point:
    before 2026-08-22 it was a second grammar, and an unrecognised token
    left those knobs silently ON while every other parser in the tree
    resolved the same token to OFF.

    ``default`` is a STRING for backward compatibility with the call sites
    (``_env_falsy(name, "0")``); it is parsed through the same table.
    """
    return not env_bool(
        name, str(default).strip().lower() not in ENV_FALSE,
        print_fn=print_fn)

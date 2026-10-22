"""Renamed names: the one table of old spellings and what replaced them.

A rename changes a deck key, a deck value, a command-line flag or a name
written to disk.  Every row maps the OLD spelling to the NEW one.

Decks and command lines are NOT aliased (owner, 2026-10-21): an old deck key,
deck value or flag refuses with one line, "`old` is now `new`" (or "`old` was
removed: ..."), and the user edits the input.  The tables below are the hint
table for those refusals.  Names on disk are different: an HDF5 dataset, a
schema string or a run-directory suffix inside an existing restart is data, not
input, so :data:`DISK_NAMES` keeps the old spelling readable, with one NOTE.

A row is either the new name (a string) or a :class:`Rename`, which adds a note
to the refusal.  ``Rename("")`` means the name was removed; its note says why
or what to use.  A :data:`CLI_FLAGS` row whose new name does not start with
``--`` names a deck key: the setting moved from the command line into the deck.

This module lives in ``runtime`` (the bottom layer, stdlib only) so that the
argv hook every driver runs at start-up (``runtime.read_verbose_switches``)
can refuse old flags before any parser sees them; every layer may import it.

The source tree is renamed with ``tools/rename_sweep.sh``; docs/dev/renaming.md has
the procedure.  A row may be dropped once nobody can still be carrying the old
spelling.  This file is never swept, so the old spellings below survive every
sweep.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("lorrax.runtime.renames")


@dataclass(frozen=True)
class Rename:
    """One renamed name.

    ``new``: the new name; empty when the name was removed.  ``note``: appended
    to the refusal (what else changed, or why a removed name has no successor).
    """

    new: str
    note: str = ""


#: Deck keys: {(deck, old key): new key or Rename}.  ``deck`` is ``"gw"`` (the
#: GW deck's ``[cohsex]`` section, also read by the drivers that share it) or
#: ``"td"`` (the TD-aGW TOML deck, dotted keys such as ``lorrax.collision``).
DECK_KEYS: dict[tuple[str, str], str | Rename] = {}

#: Deck values: {(deck, key, old value): new value or Rename}.  ``key`` is the NEW key.
DECK_VALUES: dict[tuple[str, str, str], str | Rename] = {
    # 2026-10-21: the shared-pole W model is the tangential W model.
    ("gw", "sigma_w_model", "shared_pole"): "tangential",
    ("gw", "bispinor_gw", "full_shared_pole"): "full_tangential",
    ("td", "lorrax.collision_screening", "shared_pole"): "tangential",
}

#: Command-line flags: {(program, old flag): new flag, deck key or Rename}.
#: ``program`` is the driver's module (``gw.gw_jax``, ``bse.bse_jax``) or
#: ``"*"`` for every driver.  Long flags only; ``--old``, ``--old=v`` and
#: ``--old v`` are all refused before the driver's parser runs.
CLI_FLAGS: dict[tuple[str, str], str | Rename] = {}

#: Names on disk: {old: new}.  HDF5 dataset names, header schema strings and
#: run-directory suffixes.  A reader probes the new name, then the old one.
DISK_NAMES: dict[str, str] = {
    # 2026-10-21, tangential W model.  Restart bundle member (restart.h5):
    "shared_pole_member": "tangential_member",
    # model.h5, bank.h5, sectors.json and receipt schemas:
    "lorrax.shared-real-pole.v1": "lorrax.tangential.v1",
    "lorrax.shared-real-pole-bank.v3": "lorrax.tangential-bank.v3",
    "lorrax.shared-real-pole-sectors.v1": "lorrax.tangential-sectors.v1",
    "lorrax.shared-real-pole.receipt.v1": "lorrax.tangential.receipt.v1",
    # the per-map store directory, <run_dir>/<label>_tangential/:
    "_shared_pole": "_tangential",
}

#: Spellings kept ON PURPOSE: {spelling: why}.  Not renamed, by the sweep or by
#: hand; the procedure's check skips them.
FROZEN: dict[str, str] = {
    "shared_real_pole_v1_r4": "RECIPE_VERSION: a restart compares the stored tag, so a new "
                              "spelling is a recipe bump that refuses every stored model",
    "shared_real_pole_gates_v1_r3b": "GATE_VERSION, same reason; the table variables keep the tag's name",
    "shared_real_pole_gates_ordered_v1": "ORDERED_GATE_VERSION, same reason",
    "shared_real_pole_charge4_v1": "CHARGE4_GATE_VERSION, same reason",
    "new shared-pole objects": "text inside the hashed gate table (GATE_HASH); editing it "
                               "changes the hash and refuses every stored model",
    "lorrax.shared-real-pole-bank.v1": "retired bank schema, refused by name",
    "lorrax.shared-real-pole-bank.v2": "retired bank schema, refused by name",
}

_noted: set[str] = set()


def _row(value) -> Rename:
    return value if isinstance(value, Rename) else Rename(str(value))


def refusal(old: str, value, *, flag: bool = False) -> str:
    """The one-line refusal for ``old`` (a deck key, ``key = value`` or a flag)."""
    row = _row(value)
    if not row.new:
        text = f"`{old}` was removed"
        return text + (f": {row.note}" if row.note else "")
    new = row.new
    if flag and not new.startswith("--"):
        new = f"deck key `{new}`"
    else:
        new = f"`{new}`"
    return f"`{old}` is now {new}" + (f" ({row.note})" if row.note else "")


def refuse_old_deck_keys(deck: str, section) -> None:
    """Refuse every old key of ``section`` (a dict or a configparser section)."""
    lines = [refusal(old, value) for (family, old), value in DECK_KEYS.items()
             if family == deck and old in section]
    if lines:
        raise ValueError("\n".join(lines))


def check_deck_value(deck: str, key: str, value):
    """``value`` unchanged, or a refusal when it is an old spelling of ``key``'s value."""
    if not isinstance(value, str):
        return value
    hit = DECK_VALUES.get((deck, key, value.strip().lower()))
    if hit is None:
        return value
    row = _row(hit)
    raise ValueError(refusal(f"{key} = {value.strip()}",
                             Rename(f"{key} = {row.new}" if row.new else "", row.note)))


def refuse_old_deck_values(deck: str, params: dict) -> None:
    """Refuse every old value in ``params`` (keyed by current names)."""
    for (family, key, _old) in DECK_VALUES:
        if family == deck and key in params:
            check_deck_value(deck, key, params[key])


def main_program() -> str:
    """The running driver's module (``gw.gw_jax``), or the script's base name."""
    spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    if spec is not None and spec.name:
        return spec.name
    return os.path.splitext(os.path.basename(sys.argv[0] if sys.argv else ""))[0]


def old_cli_flags(argv: list[str], program: str | None = None) -> list[str]:
    """One refusal line per old flag of :data:`CLI_FLAGS` in ``argv`` (no program name)."""
    if not CLI_FLAGS:
        return []
    program = main_program() if program is None else program
    rows = {old: v for (prog, old), v in CLI_FLAGS.items() if prog in ("*", program)}
    lines = []
    for arg in argv:
        flag = arg.partition("=")[0]
        if arg.startswith("--") and flag in rows:
            lines.append(refusal(flag, rows[flag], flag=True))
    return lines


def refuse_old_cli_flags(argv: list[str], program: str | None = None) -> None:
    """Exit with the refusal lines when ``argv`` names an old flag."""
    lines = old_cli_flags(argv, program)
    if lines:
        raise SystemExit("\n".join(f"lorrax: {line}" for line in lines))


def _note(old: str, new: str, where: str) -> None:
    """One NOTE per old on-disk spelling and process (``common.log`` message extras)."""
    if old in _noted:
        return
    _noted.add(old)
    log.warning(f"renamed: {where} {old!r} is now {new!r}; the old spelling is read",
                extra={"lorrax": {"id": "renamed_name", "found": (where, old),
                                  "fix": f"new files carry {new}", "kind": "NOTE"}})


def disk_name(name: str) -> str:
    """The current spelling of a name read from disk (old or new); an old one is noted."""
    new = DISK_NAMES.get(name) if isinstance(name, str) else None
    if new is None:
        return name
    _note(name, new, "name on disk")
    return new


def disk_names(name: str) -> tuple[str, ...]:
    """``name`` (current) first, then each old spelling that maps to it."""
    return (name, *(old for old, new in DISK_NAMES.items() if new == name))


def run_dir_entry(run_dir, name: str) -> Path:
    """``run_dir / name``, or its old spelling when only that one exists.

    ``name`` ends in a renamed suffix of :data:`DISK_NAMES` (``oneshot_tangential``).
    An old run directory keeps its old store in place: restart members name it by
    relative path, so it is read where it is and never moved.
    """
    path = Path(run_dir) / name
    if path.exists():
        return path
    for old, new in DISK_NAMES.items():
        if new.startswith("_") and name.endswith(new):
            legacy = path.with_name(name[: -len(new)] + old)
            if legacy.exists():
                _note(legacy.name, name, "run directory")
                return legacy
    return path


def suffix_pattern(new_suffix: str) -> str:
    """A regex alternation of ``new_suffix`` and its old spellings (for scratch sweeps)."""
    return "(?:" + "|".join(re.escape(s) for s in disk_names(new_suffix)) + ")"

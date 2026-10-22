"""Renamed names: the one table of old spellings LORRAX still accepts.

A rename changes a deck key, a deck value, a command-line flag or a name
written to disk.  Every row maps the OLD spelling to the NEW one.  Readers go
through the helpers below, so an old deck, an old command line or an old file
keeps working: the old name is accepted, one NOTE per name and process says
what to write instead, and the run continues under the new name.  Writers only
ever write the new name.

A row is either the new name (a string) or a :class:`Rename`, which adds a note
for the NOTE line and, when the meaning moved as well as the name, a converter
from the old value to the new one.  Example (hypothetical)::

    ("gw", "nval"): Rename("number_val_bands", note="counts from the top of the
                           valence, as BerkeleyGW", convert=lambda v: str(int(v) + 1))

This module lives in ``runtime`` (the bottom layer, stdlib only) so that the
argv hook every driver runs at start-up (``runtime.read_verbose_switches``)
can rename flags before any parser sees them; every layer may import it.

The source tree is renamed with ``tools/rename_sweep.sh``; docs/dev/renaming.md has
the procedure.  Drop a row only once no deck, command line or file still in use can
carry the old spelling.  This file is never swept, so the old spellings below
survive every sweep.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("lorrax.runtime.renames")


@dataclass(frozen=True)
class Rename:
    """One renamed name.

    ``new``: the new name.  ``note``: what else changed, appended to the NOTE.
    ``convert``: old value -> new value, for a rename whose meaning moved too.
    A deck-key converter may return a dict ``{new key: value}`` to fill several
    keys from one old key.  A flag converter takes and returns the argument
    string.
    """

    new: str
    note: str = ""
    convert: Callable[[Any], Any] | None = None


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

#: Command-line flags: {(program, old flag): new flag or Rename}.  ``program``
#: is the driver's module (``gw.gw_jax``, ``centroid.kmeans_cli``) or ``"*"``
#: for every driver.  Long flags only; ``--old``, ``--old=v`` and ``--old v``
#: are all rewritten before the driver's parser runs.
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
#: hand; RENAMING.md's check skips them.
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


def _note(old: str, row: Rename, where: str) -> None:
    """One NOTE per old spelling and process (``common.log`` message extras)."""
    if old in _noted:
        return
    _noted.add(old)
    text = f"renamed: {where} {old!r} is now {row.new!r}; the old spelling is accepted"
    if row.note:
        text += f" ({row.note})"
    log.warning(text, extra={"lorrax": {"id": "renamed_name", "found": (where, old),
                                        "fix": f"write {row.new}", "kind": "NOTE"}})


def rename_deck_keys(deck: str, section) -> None:
    """Move each old key of ``section`` (a dict or a configparser section) to its new key."""
    for (family, old), value in DECK_KEYS.items():
        if family != deck or old not in section:
            continue
        row = _row(value)
        moved = row.convert(section[old]) if row.convert else section[old]
        targets = moved if isinstance(moved, dict) else {row.new: moved}
        for new in targets:
            if new in section:
                raise ValueError(f"GATE renamed_key_twice: the deck names both {old} (old) and "
                                 f"{new} (new); keep only {new}")
        del section[old]
        for new, val in targets.items():
            section[new] = val
        _note(old, row, "deck key")


def deck_value(deck: str, key: str, value):
    """The new spelling of ``value`` for ``key``; any other value comes back unchanged."""
    if not isinstance(value, str):
        return value
    hit = DECK_VALUES.get((deck, key, value.strip().lower()))
    if hit is None:
        return value
    row = _row(hit)
    _note(value.strip(), row, f"{key} =")
    return row.convert(value) if row.convert else row.new


def rename_deck_values(deck: str, params: dict) -> None:
    """Rewrite every old value in ``params`` (new keys) to its new spelling, in place."""
    for (family, key, _old) in DECK_VALUES:
        if family == deck and key in params:
            params[key] = deck_value(deck, key, params[key])


def main_program() -> str:
    """The running driver's module (``gw.gw_jax``), or the script's base name."""
    spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    if spec is not None and spec.name:
        return spec.name
    return os.path.splitext(os.path.basename(sys.argv[0] if sys.argv else ""))[0]


def rename_cli_argv(argv: list[str], program: str | None = None) -> list[str]:
    """``argv`` (without the program name) with every old flag of :data:`CLI_FLAGS` renamed."""
    if not CLI_FLAGS:
        return list(argv)
    program = main_program() if program is None else program
    rows = {old: _row(v) for (prog, old), v in CLI_FLAGS.items() if prog in ("*", program)}
    out, i = [], 0
    while i < len(argv):
        arg = argv[i]
        flag, eq, val = arg.partition("=")
        row = rows.get(flag) if arg.startswith("--") else None
        if row is None:
            out.append(arg)
            i += 1
            continue
        _note(flag, row, "command-line flag")
        if eq:
            out.append(f"{row.new}={row.convert(val) if row.convert else val}")
        elif row.convert and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            out += [row.new, row.convert(argv[i + 1])]
            i += 1
        else:
            out.append(row.new)
        i += 1
    return out


def disk_name(name: str) -> str:
    """The current spelling of a name read from disk (old or new); an old one is noted."""
    new = DISK_NAMES.get(name) if isinstance(name, str) else None
    if new is None:
        return name
    _note(name, Rename(new), "name on disk")
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
                _note(legacy.name, Rename(name), "run directory")
                return legacy
    return path


def suffix_pattern(new_suffix: str) -> str:
    """A regex alternation of ``new_suffix`` and its old spellings (for scratch sweeps)."""
    return "(?:" + "|".join(re.escape(s) for s in disk_names(new_suffix)) + ")"

"""Renamed names: the one table of old spellings LORRAX still accepts.

A rename changes a deck key, a deck value or a name written to disk.  Every row
maps the OLD spelling to the NEW one.  Readers go through the helpers below, so
an old deck or an old file keeps working: the old name is accepted, one NOTE per
name and process says what to write instead, and the run continues under the
new name.  Writers only ever write the new name.

The source tree itself is renamed with ``tools/rename_sweep.sh``; RENAMING.md
has the procedure.  To rename a deck key or value: add its row here, run the
sweep, and add an UPGRADE_NOTES.md entry.  Drop a row only once no deck or file
still in use can carry the old spelling.  This file is never swept, so the old
spellings below survive every sweep.
"""

from __future__ import annotations

import re
from pathlib import Path

from common.log import get_logger, message

log = get_logger(__name__)

#: Deck keys: {(deck, old key): new key}.  ``deck`` is ``"gw"`` (the GW deck's
#: ``[cohsex]`` section, also read by the drivers that share it) or ``"td"``
#: (the TD-aGW TOML deck, dotted keys such as ``lorrax.collision``).
DECK_KEYS: dict[tuple[str, str], str] = {}

#: Deck values: {(deck, key, old value): new value}.  ``key`` is the NEW key.
DECK_VALUES: dict[tuple[str, str, str], str] = {
    # 2026-10-21: the shared-pole W model is the tangential W model.
    ("gw", "sigma_w_model", "shared_pole"): "tangential",
    ("gw", "bispinor_gw", "full_shared_pole"): "full_tangential",
    ("td", "lorrax.collision_screening", "shared_pole"): "tangential",
}

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


def _note(old: str, new: str, where: str) -> None:
    """One NOTE per old spelling and process."""
    if old in _noted:
        return
    _noted.add(old)
    log.warning(f"renamed: {where} {old!r} is now {new!r}; the old spelling is accepted",
                extra=message("renamed_name", found=(where, old), fix=f"write {new}",
                              kind="NOTE"))


def rename_deck_keys(deck: str, section) -> None:
    """Move each old key of ``section`` (a dict or a configparser section) to its new key."""
    for (family, old), new in DECK_KEYS.items():
        if family != deck or old not in section:
            continue
        if new in section:
            raise ValueError(f"GATE renamed_key_twice: the deck names both {old} (old) and "
                             f"{new} (new); keep only {new}")
        section[new] = section[old]
        del section[old]
        _note(old, new, "deck key")


def deck_value(deck: str, key: str, value):
    """The new spelling of ``value`` for ``key``; any other value comes back unchanged."""
    if not isinstance(value, str):
        return value
    new = DECK_VALUES.get((deck, key, value.strip().lower()))
    if new is None:
        return value
    _note(value.strip(), new, f"{key} =")
    return new


def rename_deck_values(deck: str, params: dict) -> None:
    """Rewrite every old value in ``params`` (new keys) to its new spelling, in place."""
    for (family, key, _old) in DECK_VALUES:
        if family == deck and key in params:
            params[key] = deck_value(deck, key, params[key])


def disk_name(name: str) -> str:
    """The current spelling of a name read from disk (old or new)."""
    return DISK_NAMES.get(name, name)


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

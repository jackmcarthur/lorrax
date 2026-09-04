"""Executable reader for the exact-name known-failure ledger.

The ledger lives in ``tests/KNOWN_FAILURES.md`` so the explanation and the
pytest disposition cannot drift.  JSON Lines is used inside the Markdown
document because parametrized node ids may contain punctuation or newlines.
"""

from __future__ import annotations

import json
from pathlib import Path


START = "<!-- executable-xfails:start -->"
END = "<!-- executable-xfails:end -->"


def read_known_xfails(path: str | Path) -> dict[str, dict[str, str]]:
    """Return exact node-id entries, refusing malformed or duplicate rows."""
    text = Path(path).read_text(encoding="utf-8")
    if text.count(START) != 1 or text.count(END) != 1:
        raise ValueError(
            f"{path}: expected exactly one {START!r} / {END!r} pair")
    body = text.split(START, 1)[1].split(END, 1)[0]
    entries: dict[str, dict[str, str]] = {}
    for lineno, raw in enumerate(body.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("```") or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: malformed ledger row {lineno}: {exc}") from exc
        if (not isinstance(row, dict)
                or set(row) != {"nodeid", "reason", "owner"}):
            raise ValueError(
                f"{path}: ledger row {lineno} must contain exactly "
                "nodeid, reason, owner")
        if not all(isinstance(row[key], str) and row[key].strip() for key in row):
            raise ValueError(f"{path}: ledger row {lineno} has an empty field")
        nodeid = row["nodeid"]
        if nodeid in entries:
            raise ValueError(f"{path}: duplicate ledger node id {nodeid!r}")
        entries[nodeid] = {"reason": row["reason"], "owner": row["owner"]}
    return entries


def is_complete_tests_selection(args, tests_root: str | Path) -> bool:
    """Whether pytest was asked to collect the complete ``tests`` tree."""
    selected = [Path(arg).resolve() for arg in (args or ())]
    return selected == [Path(tests_root).resolve()]


def missing_nodeids(entries, collected_nodeids) -> list[str]:
    """Ledger node ids absent from a completed collection, sorted."""
    return sorted(set(entries) - set(collected_nodeids))

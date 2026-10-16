#!/usr/bin/env python3
"""Keep docs/input_reference.md in step with the deck parsers.

The page's meanings and prose are written by hand; its key set and its
default column come from the parsers.  From the repo root::

    python3 tools/gen_input_reference.py          # rewrite the default cells
    python3 tools/gen_input_reference.py --check  # exit 1 if the page differs

The rewrite sets the head of every default cell from the parser that
``TABLES`` names for its section (``gw_config._DEFAULTS`` for the GW deck,
then ``downfold_config.DOWNFOLD_DEFAULTS``, ``tdagw.td_config.TD_DEFAULTS``,
whose keys are dotted TOML paths, and ``epsmat_config.EPSMAT_DEFAULTS`` under
their headings) and keeps the hand-written note after it.  A new input file is
one ``TABLES`` row plus its section on the page.  Both modes refuse, writing nothing,
when a key has no row, when a key has two rows, or when a row names a key
the parser no longer accepts; a row whose default cell is ``retired`` is
allowed only if ``gw_config.py`` still refuses that key by name.
``tests/test_env_registry.py`` runs the check, so gate0 fails on drift.

Default-cell grammar: ``HEAD``, ``HEAD (note)`` or ``HEAD — note``.  HEAD is
the rendered code default: ``unset`` for None, `` `true` `` / `` `false` ``,
`` `text` `` / `` `""` ``, `` `number` `` (floats at or below 1e-3 as
``1e-4``).  Defaults are read by AST (no import, no jax); a default that is a
constant imported from a sibling module (``from .band_extrapolation import
BAND_EXTRAPOLATION_ESTIMATOR_DEFAULT``) is read from that module.
"""

from __future__ import annotations

import ast
import difflib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "src" / "gw" / "gw_config.py"
OUT = REPO / "docs" / "input_reference.md"

#: THE TABLE REGISTRY, in page order: (table, the heading that opens its
#: section, the parser module, its defaults dict).  A section runs from its
#: heading to the next registered heading; the GW deck is the page's first
#: section and has no heading of its own.
TABLES = (
    ("gw", None, CONFIG, "_DEFAULTS"),
    ("downfold", "## Downfold", REPO / "src" / "gw" / "downfold_config.py",
     "DOWNFOLD_DEFAULTS"),
    ("td", "## TD-aGW", REPO / "src" / "tdagw" / "td_config.py", "TD_DEFAULTS"),
    ("epsmat", "## Epsmat", REPO / "src" / "gw" / "epsmat_config.py",
     "EPSMAT_DEFAULTS"),
)


def _module_consts(path):
    """Module-level ``NAME = <literal>`` / ``NAME: T = <literal>``."""
    consts = {}
    for node in ast.parse(path.read_text()).body:
        target = value = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        if isinstance(target, ast.Name):
            try:
                consts[target.id] = ast.literal_eval(value)
            except ValueError:
                pass
    return consts


def harvest_defaults(path, dict_name):
    """Read the ``dict_name`` literal out of ``path`` without importing it."""
    tree = ast.parse(path.read_text())
    consts = _module_consts(path)
    for node in tree.body:
        # ``from .band_extrapolation import X_DEFAULT`` or ``from
        # gw.downfold import DEFAULT_RCOND``: read the constant there.
        if (isinstance(node, ast.ImportFrom) and node.level in (0, 1)
                and node.module):
            base = path.parent if node.level == 1 else REPO / "src"
            sibling = base.joinpath(*node.module.split(".")).with_suffix(".py")
            if sibling.is_file():
                found = _module_consts(sibling)
                for alias in node.names:
                    if alias.name in found:
                        consts[alias.asname or alias.name] = found[alias.name]
    for node in tree.body:
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == dict_name
                        for t in node.targets)):
            if not isinstance(node.value, ast.Dict):
                raise SystemExit(f"{dict_name} in {path} is not a dict literal")
            out = {}
            for k_node, v_node in zip(node.value.keys, node.value.values):
                key = ast.literal_eval(k_node)
                if isinstance(v_node, ast.Name) and v_node.id in consts:
                    out[key] = consts[v_node.id]
                    continue
                try:
                    out[key] = ast.literal_eval(v_node)
                except ValueError as exc:
                    raise SystemExit(
                        f"gen_input_reference: cannot evaluate the default for "
                        f"'{key}' in {path} (line {v_node.lineno}): it is not a "
                        f"literal, a module-level constant, or a constant "
                        f"imported from a sibling module.") from exc
            return out
    raise SystemExit(f"{dict_name} dict not found in {path}")


def harvest_tables(tables=TABLES):
    """``{table: defaults}`` for every ``TABLES`` row, read by AST."""
    return {name: harvest_defaults(path, const)
            for name, _heading, path, const in tables}


def render_default(v):
    if v is None:
        return "unset"
    if isinstance(v, bool):
        return "`true`" if v else "`false`"
    if isinstance(v, str):
        return f"`{v}`" if v else '`""`'
    if isinstance(v, float) and v != 0.0 and abs(v) <= 1e-3:
        mant, exp = f"{v:e}".split("e")
        mant = mant.rstrip("0").rstrip(".")
        return f"`{mant}e{int(exp)}`"
    return f"`{v!r}`"


_CELL_SPLIT = re.compile(r"(?<!\\)\|")
_HEAD = re.compile(r"^(.*?)((?:\s+\(|\s+—).*)?$")


def sync(text, defaults, config_source, tables=TABLES):
    """Return (page with generated default heads, list of refusals).

    ``defaults`` maps ``TABLES`` names to their parser's defaults; a table it
    leaves out is empty, so every row in that section refuses as unknown.
    ``config_source`` is ``gw_config.py``'s text (the ``retired`` rule).
    """
    names = [row[0] for row in tables]
    stray = sorted(set(defaults) - set(names))
    if stray:
        raise ValueError(f"sync: tables {stray} are not in the registry {names}")
    defaults = {name: defaults.get(name, {}) for name in names}
    headings = [(heading, name) for name, heading, _path, _const in tables
                if heading]
    errors = []
    seen = {name: {} for name in names}
    default_col = None
    out = []
    table = names[0]
    for lineno, line in enumerate(text.split("\n"), 1):
        for heading, name in headings:
            if line.startswith(heading):
                table = name
                break
        cells = _CELL_SPLIT.split(line)
        if len(cells) > 2 and cells[1].strip() == "key":
            heads = [c.strip() for c in cells]
            default_col = heads.index("default") if "default" in heads else None
        m = re.match(r"^\| `([A-Za-z0-9_.]+)` \|", line)
        if not m or default_col is None:
            out.append(line)
            continue
        key = m.group(1)
        if key in seen[table]:
            errors.append(f"line {lineno}: `{key}` has a second row "
                          f"(first at line {seen[table][key]})")
        seen[table][key] = lineno
        cell = cells[default_col].strip()
        if key not in defaults[table]:
            if not (table == "gw" and cell == "retired"
                    and f'"{key}"' in config_source):
                errors.append(
                    f"line {lineno}: `{key}` is documented but not in the "
                    f"parser's defaults; delete the row, or mark it `retired` "
                    f"if gw_config still refuses it by name")
            out.append(line)
            continue
        note = _HEAD.match(cell).group(2) or ""
        cells[default_col] = f" {render_default(defaults[table][key])}{note} "
        out.append("|".join(cells))
    for table, table_defaults in defaults.items():
        for key in table_defaults:
            if key not in seen[table]:
                errors.append(f"`{key}` ({table} deck) has no row; add one by "
                              f"hand in its section")
    return "\n".join(out), errors


def problems():
    """Every way the committed page differs from the parsers (empty = in step)."""
    text = OUT.read_text()
    new, errors = sync(text, harvest_tables(), CONFIG.read_text())
    if not errors and new != text:
        diff = difflib.unified_diff(text.split("\n"), new.split("\n"),
                                    "committed", "generated", lineterm="", n=0)
        errors.append("default cells differ from the parser; run "
                      "python3 tools/gen_input_reference.py\n" + "\n".join(diff))
    return errors, new


def main(argv):
    check = "--check" in argv
    errors, new = problems()
    refusals = [e for e in errors if not e.startswith("default cells differ")]
    if refusals or (check and errors):
        print("gen_input_reference: docs/input_reference.md is out of step:")
        for e in errors:
            print("  " + e)
        return 1
    if not check and new != OUT.read_text():
        OUT.write_text(new)
        print(f"wrote {OUT}")
    else:
        print(f"{OUT} is in step with the parsers")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

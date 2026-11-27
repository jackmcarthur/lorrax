"""Census: every reader of a stored screened W(0) says which W it reads.

Each entry is a function that reads ``W0_qmunu`` (or the ladder W(0)) values
from an ISDF tensor file for a consumer.  Its body must print the
``W0 provenance:`` line itself or call a helper that does
(``restart_bundle.require_w0_series``, ``w_ladder.require_rpa_rung``,
``bsemat_export.export_w0_provenance``).  Source-level, so no driver is
imported.  A new reader goes in this list with its line.
"""
import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
ANNOUNCERS = ("W0 provenance:", "require_w0_series(", "require_rpa_rung(",
              "export_w0_provenance(")

#: (file under src/, function, the consumer it serves)
READERS = (
    ("file_io/restart_bundle.py", "read_bse_payload",
     "BSE loader: bse_jax, kpm, Haydock, exciton bands, TD-aGW, the fine-grid model"),
    ("file_io/restart_bundle.py", "read_downfold_geometry", "downfold (cRPA U)"),
    ("file_io/restart_bundle.py", "require_w0_series", "the shared refusing announcer"),
    ("bse/w_ladder.py", "require_rpa_rung", "the ladder's direct rung"),
    ("bse/w_ladder.py", "prepare_ladder_head", "the ladder head's rung reload"),
    ("bse/w_ladder.py", "compute_wc_qwedge", "the ladder resolvent's rung reload"),
    ("bse/bse_w_exact.py", "main", "the resolvent cross-check"),
    ("bse/bsemat_export.py", "export_w0_provenance", "the bsemat export's announcer"),
    ("bse/bsemat_export.py", "_export_context", "the bsemat export"),
    ("gw/epsmat_export.py", "open_export_source", "the epsmat export"),
    ("gw/screening.py", "read_imported_screening", "Sigma on BerkeleyGW's projected W"),
)


def _function_source(rel, name):
    text = (SRC / rel).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(text, node)
    raise AssertionError(f"{rel}: no function {name}")


@pytest.mark.parametrize("rel,name,consumer", READERS, ids=[f"{r}:{n}" for r, n, _ in READERS])
def test_reader_announces_the_w_it_reads(rel, name, consumer):
    source = _function_source(rel, name)
    assert any(token in source for token in ANNOUNCERS), (
        f"{rel}:{name} ({consumer}) reads a stored W(0) and prints no W0 provenance line")


def test_no_unlisted_function_reads_the_stored_screened_body():
    """The low-level screened reads are reached only from listed readers."""
    listed = {(rel, name) for rel, name, _ in READERS}
    low_level = {"read_interaction", "read_interaction_qgrid", "read_interaction_gamma"}
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        text = path.read_text(encoding="utf-8")
        if "screened" not in text:
            continue
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for call in ast.walk(node):
                if not (isinstance(call, ast.Call) and len(call.args) >= 2):
                    continue
                func = call.func
                called = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
                kind = call.args[1]
                if (called in low_level and isinstance(kind, ast.Constant)
                        and str(kind.value).startswith("screened")
                        and (rel, node.name) not in listed):
                    offenders.append(f"{rel}:{node.name}")
    # read_coarse_interactions loads W0 for the V/W interpolation payload, whose
    # only consumer (bse.vq_interp) drops it unread.
    assert sorted(set(offenders)) == ["file_io/restart_bundle.py:read_coarse_interactions"]

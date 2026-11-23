"""Every serial HDF5 open of an existing file goes through ``file_io.commit_state.open_h5_settled``.

A file another process (or this rank's collective MPI-IO handle) wrote moments
before can read back shorter than its own superblock's end of file on a Lustre
client (Frontier 2026-10, Perlmutter scratch 2026-11: ``truncated file: eof <
stored_eof`` at an exact stripe boundary; the data is complete).  The settled
open waits that view out and otherwise refuses by name, so it must be the one
route: this scan of ``src/`` and ``services/*/src`` (bench drivers excepted)
finds no ``h5py.File`` / ``h5.File`` call whose mode is ``'r'`` (the default),
``'a'`` or ``'r+'``, and no call with a non-literal mode, outside the opener.
"""
import ast
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_OPENER = "src/file_io/commit_state.py"


def _sources():
    yield from sorted((_REPO / "src").rglob("*.py"))
    yield from sorted((_REPO / "services").glob("*/src/**/*.py"))


def _mode(call):
    node = (call.args[1] if len(call.args) > 1
            else next((k.value for k in call.keywords if k.arg == "mode"), None))
    if node is None:
        return "r"
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def test_every_serial_open_of_an_existing_file_is_settled():
    offenders = []
    for path in _sources():
        rel = path.relative_to(_REPO).as_posix()
        if rel == _OPENER or "/bench/" in rel:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "File" and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in ("h5py", "h5")):
                mode = _mode(node)
                if mode is None or mode in ("r", "a", "r+"):
                    offenders.append(f"{rel}:{node.lineno} mode={mode!r}")
    assert not offenders, (
        f"{len(offenders)} serial HDF5 open(s) of an existing file bypass "
        "file_io.commit_state.open_h5_settled:\n  " + "\n  ".join(offenders))

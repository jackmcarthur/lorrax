"""Every serial HDF5 open of an existing file goes through ``file_io.commit_state.open_h5_settled``.

A file another process (or this rank's collective MPI-IO handle) wrote moments
before can read back shorter than its own superblock's end of file on a Lustre
client (Frontier 2026-10, Perlmutter scratch 2026-11: ``truncated file: eof <
stored_eof`` at an exact stripe boundary; the data is complete).  The settled
open waits that view out and otherwise refuses by name, so it must be the one
route: this scan of ``src/`` and ``services/*/src`` (bench drivers excepted)
finds no ``h5py.File`` call whose mode is ``'r'`` (the default), ``'a'`` or
``'r+'``, and no call with a non-literal mode, outside the openers.  It
follows the spellings ``h5py.File``, ``import h5py as X; X.File`` and
``from h5py import File [as Y]; Y(...)``; it does not follow ``getattr``,
``functools.partial`` or a ``File`` object passed around as a value, and it
does not scan ``tools/``, ``scripts/`` or ``tests/``.

The second opener is ``zeta_loader.format._open_settled``: the never-raising
ζ probe promises no jax and no LORRAX, so it carries its own copy of the
retry at the guard bound.
"""
import ast
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_OPENERS = {"src/file_io/commit_state.py",
            "services/zeta_loader/src/zeta_loader/format.py"}


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


def _offending_lines(tree):
    """Line numbers of serial HDF5 opens of an existing file in ``tree``."""
    modules, files = {"h5py", "h5"}, set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "h5py":
                    modules.add(a.asname or "h5py")
        elif isinstance(node, ast.ImportFrom) and node.module == "h5py":
            for a in node.names:
                if a.name == "File":
                    files.add(a.asname or "File")
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        hit = ((isinstance(f, ast.Attribute) and f.attr == "File"
                and isinstance(f.value, ast.Name) and f.value.id in modules)
               or (isinstance(f, ast.Name) and f.id in files))
        if hit:
            mode = _mode(node)
            if mode is None or mode in ("r", "a", "r+"):
                out.append((node.lineno, mode))
    return out


def test_every_serial_open_of_an_existing_file_is_settled():
    offenders = []
    for path in _sources():
        rel = path.relative_to(_REPO).as_posix()
        if rel in _OPENERS or "/bench/" in rel:
            continue
        for line, mode in _offending_lines(ast.parse(path.read_text(encoding="utf-8"))):
            offenders.append(f"{rel}:{line} mode={mode!r}")
    assert not offenders, (
        f"{len(offenders)} serial HDF5 open(s) of an existing file bypass "
        "file_io.commit_state.open_h5_settled:\n  " + "\n  ".join(offenders))


def test_scan_follows_aliases_and_from_imports():
    src = ("import h5py as hp\nfrom h5py import File\nfrom h5py import File as F5\n"
           "def f(p, m):\n"
           "    hp.File(p)\n    File(p, 'a')\n    F5(p, mode=m)\n"
           "    hp.File(p, 'w')\n    File(p, 'w-')\n    F5(p, 'x')\n")
    assert sorted(line for line, _ in _offending_lines(ast.parse(src))) == [5, 6, 7]

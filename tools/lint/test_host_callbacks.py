"""No raw host callback outside ``lxkit.callbacks`` (host functions see NumPy, not jax.Array).

JAX hands a host callback ``jax.Array`` arguments; a slice or reshape of one
dispatches to the device that is running the caller's loop, and on a busy CPU
pool the two wait on each other for ever (the CPU Davidson stall, 4aba67e0d).
Every ``jax.pure_callback`` / ``io_callback`` / ``jax.debug.callback`` under
``src/`` and ``services/*/src`` must therefore go through
``lxkit.host_pure_callback`` / ``host_io_callback`` / ``host_debug_callback``,
which hand the host function ``np.asarray`` views.  The runtime regression is
``tests/test_host_callbacks_cpu.py``.
"""
import ast
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ("pure_callback", "io_callback")
OWNER = ROOT / "services" / "lxkit" / "src" / "lxkit" / "callbacks.py"


def _sources():
    yield from sorted((ROOT / "src").rglob("*.py"))
    for svc in sorted((ROOT / "services").glob("*/src")):
        yield from sorted(svc.rglob("*.py"))


def _is_debug(node) -> bool:
    return (isinstance(node, ast.Name) and node.id == "debug") or \
           (isinstance(node, ast.Attribute) and node.attr == "debug")


def _raw_callbacks(path: Path):
    text = path.read_text()
    lines = text.splitlines()
    for node in ast.walk(ast.parse(text, filename=str(path))):
        hit = isinstance(node, ast.Attribute) and (
            node.attr in RAW or (node.attr == "callback" and _is_debug(node.value)))
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("jax"):
            hit = any(a.name in RAW or (node.module == "jax.debug" and a.name == "callback")
                      for a in node.names)
        if hit:
            yield f"{path.relative_to(ROOT)}:{node.lineno}: {lines[node.lineno - 1].strip()}"


def test_no_raw_host_callbacks():
    files = [p for p in _sources() if p != OWNER]
    assert len(files) > 200, f"scanned only {len(files)} files: is ROOT right ({ROOT})?"
    bad = sorted(hit for p in files for hit in _raw_callbacks(p))
    assert not bad, (
        "raw host callback outside lxkit.callbacks: use lxkit.host_pure_callback / "
        "host_io_callback / host_debug_callback, whose host functions get NumPy views:\n  "
        + "\n  ".join(bad))


def test_lint_sees_a_raw_callback():
    tmp_path = Path(tempfile.mkdtemp())
    src = tmp_path / "m.py"
    src.write_text("import jax\n"
                   "from jax.experimental import io_callback\n"
                   "from lxkit import host_pure_callback\n"
                   "def f(x, s):\n"
                   "    a = jax.pure_callback(g, s, x)\n"
                   "    jax.debug.callback(print, x)\n"
                   "    b = host_pure_callback(g, s, x)\n"
                   "    jax.debug.print('{}', x)\n"
                   "    return a, b\n")
    global ROOT
    root, ROOT = ROOT, tmp_path
    try:
        hits = sorted(_raw_callbacks(src), key=lambda h: int(h.split(":")[1]))
    finally:
        ROOT = root
    assert [h.split(":")[1] for h in hits] == ["2", "5", "6"], hits


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok  ", name)

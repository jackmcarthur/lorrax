"""No raw floating sum over a mesh axis outside lxkit (placement-independent bits).

A library all-reduce / reduce-scatter adds the per-rank terms in an order its rings
take from the device topology, so the same run on another GPU set rounds differently
(det lane 2026-10-16: Fe V2 sigma_c 4e-7 eV, eqp print flips).  Every
``lax.psum`` / ``lax.psum_scatter`` / ``lax.pmean`` under ``src/`` and
``services/*/src`` must therefore go through ``lxkit.ordered_psum`` /
``lxkit.ordered_psum_scatter`` (exact and free at two ranks, rank-ordered above),
or carry ``# exact-reduce: <reason>`` on a line of the call itself when the sum is exact at any
rank count: an integer operand, one owner per element with zeros elsewhere, a
warm-up of zeros.  ``pmax`` / ``pmin`` are order-independent and are not checked.
The runtime twin is ``LORRAX_REDUCTION_GUARD=raise`` (``lxkit.install_reduction_guard``).
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKED = ("psum", "psum_scatter", "pmean")
MARKER = "exact-reduce:"
OWNER = ROOT / "services" / "lxkit" / "src" / "lxkit" / "collectives.py"


def _sources():
    yield from sorted((ROOT / "src").rglob("*.py"))
    for svc in sorted((ROOT / "services").glob("*/src")):
        yield from sorted(svc.rglob("*.py"))


def _is_lax(node) -> bool:
    return (isinstance(node, ast.Name) and node.id == "lax") or \
           (isinstance(node, ast.Attribute) and node.attr == "lax")


def _raw_reductions(path: Path):
    text = path.read_text()
    tree = ast.parse(text, filename=str(path))
    lines = text.splitlines()
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    for node in ast.walk(tree):
        hit = isinstance(node, ast.Attribute) and node.attr in CHECKED and _is_lax(node.value)
        if isinstance(node, ast.ImportFrom) and node.module in ("jax.lax",):
            hit = any(a.name in CHECKED for a in node.names)
        if not hit:
            continue
        span = node
        parent = parents.get(node)
        if isinstance(parent, ast.Call) and parent.func is node:
            span = parent
        lo, hi = span.lineno, getattr(span, "end_lineno", span.lineno)
        if not any(MARKER in lines[i - 1] for i in range(lo, hi + 1)):
            yield f"{path.relative_to(ROOT)}:{node.lineno}: {lines[node.lineno - 1].strip()}"


def test_no_unordered_floating_reductions():
    files = [p for p in _sources() if p != OWNER]
    assert len(files) > 200, f"scanned only {len(files)} files: is ROOT right ({ROOT})?"
    bad = sorted(hit for p in files for hit in _raw_reductions(p))
    assert not bad, (
        "raw lax.psum/psum_scatter/pmean outside lxkit: use lxkit.ordered_psum / "
        "lxkit.ordered_psum_scatter, or mark the call '# exact-reduce: <reason>' when "
        "the sum is exact at any rank count:\n  " + "\n  ".join(bad))


def test_lint_sees_a_raw_call(tmp_path):
    src = tmp_path / "m.py"
    src.write_text("from jax import lax\n"
                   "def f(x):\n"
                   "    a = lax.psum(x, 'x')\n"
                   "    b = lax.psum(x.astype(int), 'x')  # exact-reduce: int\n"
                   "    c = jax.lax.psum_scatter(\n"
                   "        x, 'x', tiled=True)\n"
                   "    g = lax.psum\n"
                   "    return a, b, c, g\n")
    global ROOT
    root, ROOT = ROOT, tmp_path
    try:
        hits = sorted(_raw_reductions(src), key=lambda h: int(h.split(":")[1]))
    finally:
        ROOT = root
    assert [h.split(":")[1] for h in hits] == ["3", "5", "7"], hits

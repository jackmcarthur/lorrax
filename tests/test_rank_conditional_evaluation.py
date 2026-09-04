"""INVARIANTS row 21: rank gates must not guard JAX evaluation.

A Python rank branch may choose who formats or writes an already-materialized
host value.  It may not choose who asks JAX for that value: a seemingly cheap
``float(x)`` or ``np.asarray(x)`` can lower a collective when ``x`` is globally
sharded, leaving the other ranks past the matching rendezvous.

This is deliberately a provenance-aware source check: it reports values that
can be proved JAX-backed from local syntax or an explicit fixture seed, and
does not guess about unannotated external return values.  A reviewed false
positive is named by exact source location below, with the property that makes
the value safe.  That makes the exception count ratchet instead of turning an
entire file or spelling into an exemption.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


# Exact old blocks from the two fixes whose corrected descendants are present
# at ``d294ac52^`` and ``3111aa71^``.  The bad statements are copied verbatim
# from the immediate pre-fix snapshots ``7a9c290a^`` and ``75ba9d82^``;
# respectively, those are ancestors of the two requested audit points.  Both
# lived in ``prune_candidates_by_pivoted_cholesky``; the surrounding
# thousand-line selector is irrelevant to the lexical invariant, so the
# fixtures retain the function name, parameters, and bad statements verbatim.
_GRAM_DIAGNOSTIC_BEFORE_7A9C290A = """
def prune_candidates_by_pivoted_cholesky(G, M, verbose):
    if verbose:
        # Report the LOGICAL diagonal (pads are zero and would drag the min to
        # 0.0, changing a diagnostic the gates compare across P).
        diag = jnp.real(jnp.diag(G))[:M]
        print(f"[pivoted_cholesky] G built, shape=({M}, {M}), "
              f"diag range [{float(diag.min()):.3e}, {float(diag.max()):.3e}]")
"""

_RESIDUAL_DIAGNOSTIC_BEFORE_75BA9D82 = """
def prune_candidates_by_pivoted_cholesky(d_taken, trR_over_trG, n_keep,
                                          verbose):
    if verbose:
        print(f"[pivoted_cholesky] picked-pivot residuals: "
              f"first={float(d_taken[0]):.3e}, "
              f"mid={float(d_taken[n_keep // 2]):.3e}, "
              f"last={float(d_taken[-1]):.3e}")
        print(f"[pivoted_cholesky] tr(R_k)/tr(G): "
              f"first={float(trR_over_trG[1]):.3e}, "
              f"mid={float(trR_over_trG[n_keep // 2 + 1]):.3e}, "
              f"last={float(trR_over_trG[n_keep]):.3e}")
"""


@dataclass(frozen=True, order=True)
class Hit:
    """One possible JAX evaluation below a rank-dependent branch."""

    path: str
    line: int
    column: int
    kind: str

    @property
    def key(self) -> str:
        return f"{self.path}:{self.line}:{self.column}:{self.kind}"


# A source-line exception, never a file/function exemption.  Values listed
# here must already be replicated or have had their rank-symmetric collective
# completed before the branch.  Keep the reason on the same row so a reviewer
# can decide whether a future edit invalidates it.
_ALLOWLIST: dict[str, str] = {
    "src/centroid/pivoted_cholesky.py:112:27:float(jax-value)": (
        "diag_min is a replicated scalar whose JIT and readiness wait run on every rank"
    ),
    "src/centroid/pivoted_cholesky.py:112:50:float(jax-value)": (
        "diag_max is a replicated scalar whose JIT and readiness wait run on every rank"
    ),
}


_HOST = "host"
_JAX = "jax"
_UNKNOWN = "unknown"

_DIRECT_RANK_NAMES = {
    "is_rank0", "rank0", "process_rank", "process_index", "verbose",
}
_FORCING_BUILTINS = {"bool", "float", "int"}
_JNP_BRANCH_REDUCTIONS = {"all", "any", "max", "min", "sum"}
_JAX_HOST_QUERIES = {
    "device_count", "devices", "local_device_count", "local_devices",
    "process_count", "process_index",
}


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _assigned_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Attribute):
        name = _call_name(node)
        return {name} if name else set()
    if isinstance(node, (ast.Tuple, ast.List)):
        return set().union(*(_assigned_names(item) for item in node.elts))
    return set()


def _rank_mentions(node: ast.AST, derived: set[str]) -> bool:
    """Whether an expression depends on rank/debug-print selection."""
    if node is None:
        return False
    for item in ast.walk(node):
        if isinstance(item, ast.Name):
            name = item.id
            if (name in derived or name in _DIRECT_RANK_NAMES
                    or name.endswith("_rank0")):
                return True
        elif isinstance(item, ast.Attribute):
            name = _call_name(item)
            if name in derived or name.endswith(".rank"):
                return True
        elif isinstance(item, ast.Call):
            name = _call_name(item.func)
            if name.rsplit(".", 1)[-1] in {"process_rank", "process_index"}:
                return True
        elif (isinstance(item, ast.Constant)
              and item.value == "LORRAX_DEBUG_PRINT"):
            return True
        elif isinstance(item, ast.Compare):
            operands = [item.left, *item.comparators]
            has_zero = any(
                isinstance(operand, ast.Constant) and operand.value == 0
                for operand in operands)
            has_rank = any(
                _call_name(operand).rsplit(".", 1)[-1] == "rank"
                for operand in operands)
            if has_zero and has_rank:
                return True
    return False


def _rank_derived_names(tree: ast.AST) -> set[str]:
    """Follow simple assignments such as ``rank0 = process_rank() == 0``."""
    def _simple_rank_value(value, derived):
        if isinstance(value, (ast.Name, ast.Attribute, ast.Compare,
                              ast.BoolOp, ast.UnaryOp, ast.IfExp)):
            return _rank_mentions(value, derived)
        if isinstance(value, ast.Call):
            leaf = _call_name(value.func).rsplit(".", 1)[-1]
            if leaf in {"process_rank", "process_index"}:
                return True
            if leaf in {"bool", "int"} and value.args:
                return _simple_rank_value(value.args[0], derived)
            return any(
                isinstance(item, ast.Constant)
                and item.value == "LORRAX_DEBUG_PRINT"
                for item in ast.walk(value))
        return False

    derived: set[str] = set()
    assignments = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr))
    ]
    changed = True
    while changed:
        changed = False
        for node in assignments:
            value = node.value
            targets = (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            if value is not None and _simple_rank_value(value, derived):
                for target in targets:
                    for name in _assigned_names(target):
                        if name not in derived:
                            derived.add(name)
                            changed = True
    return derived


def _jax_annotation(node: ast.AST | None) -> bool:
    if node is None:
        return False
    text = _call_name(node)
    return text in {"jax.Array", "jnp.ndarray"} or "jax.Array" in ast.unparse(node)


def _expression_origin(node: ast.AST, origins: dict[str, str]) -> str:
    """Classify an expression as host, JAX, or statically unknown."""
    if isinstance(node, ast.Constant):
        return _HOST
    if isinstance(node, ast.Name):
        return origins.get(node.id, _UNKNOWN)
    if isinstance(node, ast.Attribute):
        return origins.get(_call_name(node), _expression_origin(node.value, origins))
    if isinstance(node, ast.Subscript):
        return _expression_origin(node.value, origins)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        values = [_expression_origin(item, origins) for item in node.elts]
    elif isinstance(node, ast.Dict):
        values = [_expression_origin(item, origins) for item in node.values]
    elif isinstance(node, ast.UnaryOp):
        return _expression_origin(node.operand, origins)
    elif isinstance(node, ast.BinOp):
        values = [_expression_origin(node.left, origins),
                  _expression_origin(node.right, origins)]
    elif isinstance(node, ast.BoolOp):
        values = [_expression_origin(item, origins) for item in node.values]
    elif isinstance(node, ast.Compare):
        values = [_expression_origin(node.left, origins)] + [
            _expression_origin(item, origins) for item in node.comparators]
    elif isinstance(node, ast.IfExp):
        values = [_expression_origin(node.body, origins),
                  _expression_origin(node.orelse, origins)]
    elif isinstance(node, ast.Call):
        name = _call_name(node.func)
        if name.startswith("jax.debug."):
            return _HOST
        if name.startswith("jax.") and name.rsplit(".", 1)[-1] in _JAX_HOST_QUERIES:
            return _HOST
        if name.startswith(("jnp.", "jax.")):
            return _JAX
        if name.startswith(("np.", "numpy.")):
            return _HOST
        if isinstance(node.func, ast.Attribute):
            receiver = _expression_origin(node.func.value, origins)
            if receiver == _JAX:
                return _JAX
        return _UNKNOWN
    else:
        return _UNKNOWN
    if _JAX in values:
        return _JAX
    if _UNKNOWN in values:
        return _UNKNOWN
    return _HOST


def _jax_function_names(tree: ast.AST) -> set[str]:
    """Names of local functions whose source declares a JAX result."""
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        decorators = " ".join(ast.unparse(item) for item in node.decorator_list)
        if "jax.jit" in decorators or _jax_annotation(node.returns):
            names.add(node.name)
    return names


def _value_origins(tree: ast.AST, known_jax_names: set[str],
                   jax_function_names: set[str]) -> dict[str, str]:
    """Collect simple, conservative value provenance for forcing calls."""
    origins: dict[str, str] = {name: _JAX for name in known_jax_names}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in (*node.args.posonlyargs, *node.args.args,
                        *node.args.kwonlyargs):
                origins.setdefault(
                    arg.arg,
                    _JAX if _jax_annotation(arg.annotation) else _UNKNOWN)
    assignments = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr))
    ]
    # Revisit because an assignment can precede its producer in ast.walk's
    # cross-scope order.  Only monotone upgrades toward known JAX/host values
    # are made; disagreement remains unknown.
    for _ in range(len(assignments) + 1):
        changed = False
        for node in assignments:
            value = node.value
            targets = (
                node.targets if isinstance(node, ast.Assign) else [node.target]
            )
            if value is None:
                continue
            if (isinstance(value, ast.Call)
                    and _call_name(value.func).rsplit(".", 1)[-1]
                    in jax_function_names):
                origin = _JAX
            else:
                origin = _expression_origin(value, origins)
            for target in targets:
                for name in _assigned_names(target):
                    old = origins.get(name)
                    new = origin if old in (None, origin) else _UNKNOWN
                    if old != new:
                        origins[name] = new
                        changed = True
        if not changed:
            break
    return origins


def _inside_nodes(branch: ast.If):
    """Walk a branch but not a nested function/class body's later work."""
    class _LexicalBranchWalker(ast.NodeVisitor):
        def __init__(self):
            self.nodes = []

        def generic_visit(self, node):
            self.nodes.append(node)
            super().generic_visit(node)

        def visit_FunctionDef(self, node):
            self.nodes.append(node)

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_ClassDef = visit_FunctionDef
        visit_Lambda = visit_FunctionDef

    walker = _LexicalBranchWalker()
    for statement in (*branch.body, *branch.orelse):
        walker.visit(statement)
    yield from walker.nodes


def _forcing_kind(call: ast.Call, origins: dict[str, str],
                  branch_tests: set[int]) -> str | None:
    name = _call_name(call.func)
    if name.startswith("jax.debug."):
        return None
    leaf = name.rsplit(".", 1)[-1]
    if leaf == "block_until_ready":
        return "block_until_ready"
    if name == "jax.device_get":
        return "jax.device_get"
    if leaf == "item" and isinstance(call.func, ast.Attribute):
        if _expression_origin(call.func.value, origins) == _JAX:
            return ".item"
        return None
    if name in {"np.asarray", "numpy.asarray", "np.array", "numpy.array"}:
        if call.args and _expression_origin(call.args[0], origins) == _JAX:
            return name
        return None
    if name in _FORCING_BUILTINS:
        if call.args and _expression_origin(call.args[0], origins) == _JAX:
            return f"{name}(jax-value)"
        return None
    if (name.startswith(("jnp.", "jax.numpy."))
            and leaf in _JNP_BRANCH_REDUCTIONS
            and id(call) in branch_tests):
        return f"{name}(python-branch)"
    return None


def scan_source(source: str, path: str = "<fixture>", *,
                known_jax_names: set[str] | None = None) -> list[Hit]:
    """Return rank-conditional JAX-evaluation candidates in Python source."""
    tree = ast.parse(source, filename=path)
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }

    def _scope(node):
        parent = parents.get(node)
        while parent is not None:
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef,
                                   ast.Lambda)):
                return parent
            parent = parents.get(parent)
        return tree

    rank_cache = {}
    origin_cache = {}
    file_rank_attrs = {
        name for name in _rank_derived_names(tree) if "." in name}
    jax_function_names = _jax_function_names(tree)
    branch_tests = {
        id(item)
        for branch in ast.walk(tree) if isinstance(branch, ast.If)
        for item in ast.walk(branch.test)
    }
    hits: dict[tuple[int, int, str], Hit] = {}
    for branch in ast.walk(tree):
        if not isinstance(branch, ast.If):
            continue
        scope = _scope(branch)
        rank_derived = rank_cache.setdefault(
            id(scope), _rank_derived_names(scope) | file_rank_attrs)
        if not _rank_mentions(branch.test, rank_derived):
            continue
        origins = origin_cache.setdefault(
            id(scope), _value_origins(
                scope, known_jax_names or set(), jax_function_names))
        for node in _inside_nodes(branch):
            if not isinstance(node, ast.Call):
                continue
            kind = _forcing_kind(node, origins, branch_tests)
            if kind is not None:
                key = (node.lineno, node.col_offset, kind)
                hits[key] = Hit(path, node.lineno, node.col_offset, kind)
    return sorted(hits.values())


def _production_sources() -> list[Path]:
    roots = [ROOT / "src"]
    roots.extend(
        service / "src"
        for service in (ROOT / "services").iterdir()
        if service.is_dir() and (service / "src").is_dir()
    )
    return sorted(
        path
        for root in roots
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def test_the_two_production_deadlock_shapes_are_detected():
    gram = scan_source(_GRAM_DIAGNOSTIC_BEFORE_7A9C290A, "d294ac52^ fixture")
    residual = scan_source(
        _RESIDUAL_DIAGNOSTIC_BEFORE_75BA9D82, "3111aa71^ fixture",
        # These are the tuple outputs of the sharded selector immediately
        # above the copied block in the historical function.  ``scan_source``
        # accepts this seed for values whose external return annotation cannot
        # be recovered from one file's AST.
        known_jax_names={"d_taken", "trR_over_trG"})

    assert [hit.kind for hit in gram] == [
        "float(jax-value)", "float(jax-value)"]
    assert [hit.kind for hit in residual] == [
        "float(jax-value)", "float(jax-value)", "float(jax-value)",
        "float(jax-value)", "float(jax-value)", "float(jax-value)",
    ]


def test_jnp_reduction_used_as_a_python_branch_is_detected():
    hits = scan_source("""
def broken(x, rank):
    if rank == 0:
        if jnp.any(x):
            print("bad")
""")
    assert [hit.kind for hit in hits] == ["jnp.any(python-branch)"]


def test_rank_derived_assignments_are_followed_and_jax_debug_is_safe():
    hits = scan_source("""
def broken(x):
    owner = process_rank() == 0
    writer = owner
    if writer:
        jax.debug.print("x={x}", x=x)
        x.block_until_ready()
""")
    assert [hit.kind for hit in hits] == ["block_until_ready"]


def test_current_tree_has_only_reviewed_rank_conditional_evaluations():
    hits = []
    for path in _production_sources():
        rel = path.relative_to(ROOT).as_posix()
        hits.extend(scan_source(path.read_text(encoding="utf-8"), rel))

    unreviewed = [hit.key for hit in hits if hit.key not in _ALLOWLIST]
    stale = sorted(set(_ALLOWLIST) - {hit.key for hit in hits})
    assert not unreviewed, (
        "rank-conditional code may evaluate a sharded JAX value; move the "
        "evaluation/materialization above the rank branch, leaving only "
        f"formatting/writing inside it:\n" + "\n".join(unreviewed))
    assert not stale, "stale rank-evaluation allowlist rows:\n" + "\n".join(stale)
    assert all(reason.strip() for reason in _ALLOWLIST.values())

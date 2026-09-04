"""One-owner gate for mesh-padded extent arithmetic.

The exceptions below are modular *routing* or a physics refusal, not carrier
arithmetic.  Keeping the registry explicit makes a new ``% px`` impossible to
smuggle in under the ring precedent.
"""
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OWNER = Path("src/runtime/padding.py")

MESH_DIVISOR_NAMES = {
    "p_x", "p_y", "px", "py", "p_xy", "p_prod", "P_total",
    "n_dev", "ndev", "n_devices", "device_count", "total_devices",
    "world_size", "proc",
}

# key: (repository-relative path, nested function, normalized expression)
# Every entry carries both the reason and its disposition/follow-up.
MODULO_EXCEPTIONS = {
    ("src/bse/bse_ring_comm.py", "_ring_sum_valence/step",
     "(axis_index_y - jnp.asarray(i, dtype=jnp.int32)) % py"): {
        "reason": "cyclic source-rank address in a y-axis ring",
        "follow_up": "none; this selects a torus neighbour, not an extent",
    },
    ("src/bse/bse_ring_comm.py", "_ring_sum_conduction/step",
     "(axis_index_x - jnp.asarray(i, dtype=jnp.int32)) % px"): {
        "reason": "cyclic source-rank address in an x-axis ring",
        "follow_up": "none; this selects a torus neighbour, not an extent",
    },
    ("src/bse/bse_ring_comm.py", "_ring_sum_B_encode/step",
     "(axis_index_y - jnp.asarray(i, dtype=jnp.int32)) % py"): {
        "reason": "cyclic source-rank address in a y-axis ring",
        "follow_up": "none; this selects a torus neighbour, not an extent",
    },
    ("src/bse/bse_ring_comm.py", "apply_V_ring/step_y",
     "(axis_index_y - jnp.asarray(i, dtype=jnp.int32)) % py"): {
        "reason": "cyclic source-rank address in a y-axis ring",
        "follow_up": "none; this selects a torus neighbour, not an extent",
    },
    ("src/bse/bse_ring_comm.py",
     "build_density_snapshot_operator/_map/step_y",
     "(axis_index_y - jnp.asarray(i, dtype=jnp.int32)) % py"): {
        "reason": "cyclic source-rank address in a y-axis ring",
        "follow_up": "none; this selects a torus neighbour, not an extent",
    },
    ("src/gw/qsgw_head.py", "_head_wing_kernel_legacy", "(i + 1) % px"): {
        "reason": "cyclic x-axis ppermute destination",
        "follow_up": "retire with the legacy head-wing kernel",
    },
    ("src/gw/qsgw_head.py", "_head_wing_kernel_legacy", "(i + 1) % py"): {
        "reason": "cyclic y-axis ppermute destination",
        "follow_up": "retire with the legacy head-wing kernel",
    },
}

DIVISIBILITY_REFUSAL_EXCEPTIONS = {
    ("src/bse/bse_ring_comm.py", "create_mesh_xy"): {
        "reason": "square-mesh backend topology, not a padded array axis",
        "follow_up": "distrib_la owns rectangular-backend enablement",
    },
    ("src/isdf/core.py", "_resolve_solver_kind_transverse"): {
        "reason": (
            "padding the indefinite near-null transverse LU is not inert; "
            "the owner tag is used to detect the requested carrier"
        ),
        "follow_up": "rank_truncate is the padded distributed alternative",
    },
    ("src/runtime/__init__.py", "reshape"): {
        "reason": "square runtime mesh topology, not a padded array axis",
        "follow_up": "retire when the runtime admits rectangular meshes",
    },
}


class _PaddingCensus(ast.NodeVisitor):
    def __init__(self, path: Path):
        self.path = path
        self.stack: list[str] = []
        self.modulo: set[tuple[str, str, str]] = set()
        self.refusals: set[tuple[str, str]] = set()
        self.forbidden: list[str] = []

    @property
    def function(self) -> str:
        return "/".join(self.stack) or "<module>"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in {"round_up", "mesh_padded", "padded_extent"}:
            self.forbidden.append(
                f"{self.path}:{node.lineno}: forbidden symbol {node.id!r}")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in {"round_up", "mesh_padded", "padded_extent"}:
            self.forbidden.append(
                f"{self.path}:{node.lineno}: forbidden attribute {node.attr!r}")
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Mod):
            rhs_names = {
                item.id for item in ast.walk(node.right)
                if isinstance(item, ast.Name)
            }
            if rhs_names & MESH_DIVISOR_NAMES:
                self.modulo.add(
                    (str(self.path), self.function, ast.unparse(node)))
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        text = " ".join(
            item.value for item in ast.walk(node)
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        ).lower()
        if ("divisib" in text
                and any(word in text for word in ("mesh", "p_x", "p_y", "px", "py"))):
            self.refusals.add((str(self.path), self.function))
        self.generic_visit(node)


def test_runtime_padding_is_the_only_mesh_extent_arithmetic_owner():
    modulo = set()
    refusals = set()
    forbidden = []
    for absolute in sorted((ROOT / "src").rglob("*.py")):
        relative = absolute.relative_to(ROOT)
        if relative == OWNER:
            continue
        census = _PaddingCensus(relative)
        census.visit(ast.parse(absolute.read_text(), filename=str(relative)))
        modulo.update(census.modulo)
        refusals.update(census.refusals)
        forbidden.extend(census.forbidden)

    assert not forbidden, "\n".join(forbidden)
    assert modulo == set(MODULO_EXCEPTIONS), (
        f"mesh-modulo census changed: added={modulo - set(MODULO_EXCEPTIONS)}, "
        f"retired={set(MODULO_EXCEPTIONS) - modulo}")
    assert refusals == set(DIVISIBILITY_REFUSAL_EXCEPTIONS), (
        "mesh-divisibility refusal census changed: "
        f"added={refusals - set(DIVISIBILITY_REFUSAL_EXCEPTIONS)}, "
        f"retired={set(DIVISIBILITY_REFUSAL_EXCEPTIONS) - refusals}")

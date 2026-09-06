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


}


def _registered(entries, *, reason, follow_up):
    """Attach a reviewable reason and disposition to every exact census item."""
    return {
        entry: {"reason": reason, "follow_up": follow_up}
        for entry in entries
    }


MODULO_EXCEPTIONS.update(_registered({
    ("services/distrib_la/src/distrib_la/_batch_reshard.py",
     "validate_batch_reshard_operands", "n % px"),
    ("services/distrib_la/src/distrib_la/_batch_reshard.py",
     "validate_batch_reshard_operands", "n % py"),
    ("services/distrib_la/src/distrib_la/_batch_reshard.py",
     "validate_batch_reshard_operands", "nrhs % py"),
    ("services/distrib_la/src/distrib_la/matmul.py", "matmul",
     "int(x.shape[-1]) % py"),
    ("services/distrib_la/src/distrib_la/matmul.py", "matmul",
     "int(x.shape[-2]) % px"),
    ("services/distrib_la/src/distrib_la/matmul.py", "matmul", "m_out % px"),
    ("services/distrib_la/src/distrib_la/matmul.py", "matmul", "n_out % py"),
    ("services/distrib_la/src/distrib_la/polar.py", "_mesh_contract", "n % px"),
    ("services/distrib_la/src/distrib_la/polar.py", "_mesh_contract", "n % py"),
    ("services/distrib_la/src/distrib_la/resolve.py", "resolve_backend",
     "int(n) % px"),
    ("services/distrib_la/src/distrib_la/resolve.py", "resolve_backend",
     "int(n) % py"),
}, reason="backend authenticates an already-produced distributed carrier",
   follow_up="caller must obtain the carrier from runtime.padding"))

MODULO_EXCEPTIONS.update(_registered({
    ("services/symmetry_maps/src/symmetry_maps/maps.py",
     "unfold_spin_centroid_operator", "n_left * ns % px"),
    ("services/symmetry_maps/src/symmetry_maps/maps.py",
     "unfold_spin_centroid_operator", "perm_ms % (n_left * ns // px)"),
}, reason="symmetry transport authenticates an already-padded package",
   follow_up="accept the runtime receipt when the service API next changes"))

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

DIVISIBILITY_REFUSAL_EXCEPTIONS.update(_registered({
    ("services/distrib_la/src/distrib_la/_batch_reshard.py",
     "validate_batch_reshard_operands"),
    ("services/distrib_la/src/distrib_la/_cusolvermp.py",
     "batched_distributed_cholesky"),
    ("services/distrib_la/src/distrib_la/_cusolvermp.py",
     "batched_distributed_getrf"),
    ("services/distrib_la/src/distrib_la/_cusolvermp.py",
     "batched_distributed_getrs"),
    ("services/distrib_la/src/distrib_la/_cusolvermp.py",
     "batched_distributed_potrs"),
    ("services/distrib_la/src/distrib_la/_cusolvermp.py",
     "batched_distributed_solve_lu"),
    ("services/distrib_la/src/distrib_la/_cusolvermp.py", "distributed_eigh"),
    ("services/distrib_la/src/distrib_la/_native2d.py", "block_size_for"),
    ("services/distrib_la/src/distrib_la/_scalapack.py",
     "_validate_lu_geometry"),
    ("services/distrib_la/src/distrib_la/_scalapack.py",
     "batched_distributed_eigh"),
    ("services/distrib_la/src/distrib_la/_scalapack.py",
     "batched_distributed_getrs"),
    ("services/distrib_la/src/distrib_la/_scalapack.py",
     "batched_distributed_solve_lu"),
    ("services/distrib_la/src/distrib_la/_slate.py",
     "batched_distributed_cholesky"),
    ("services/distrib_la/src/distrib_la/_slate.py", "batched_distributed_trsm"),
    ("services/distrib_la/src/distrib_la/_slate.py", "distributed_cholesky"),
    ("services/distrib_la/src/distrib_la/_slate.py", "distributed_eigh"),
    ("services/distrib_la/src/distrib_la/_slate.py", "distributed_trsm"),
    ("services/distrib_la/src/distrib_la/polar.py", "_mesh_contract"),
    ("services/distrib_la/src/distrib_la/resolve.py", "resolve_backend"),
}, reason="backend cannot consume a ragged carrier",
   follow_up="retain as provider authentication; runtime.padding owns production"))

DIVISIBILITY_REFUSAL_EXCEPTIONS.update(_registered({
    ("services/symmetry_maps/src/symmetry_maps/maps.py",
     "_get_unfold_isdf_operator_jit"),
}, reason="collective authenticates the already-padded symmetry carrier",
   follow_up="accept the runtime receipt when the service API next changes"))


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
    sources = list((ROOT / "src").rglob("*.py"))
    sources.extend((ROOT / "services").glob("*/src/**/*.py"))
    for absolute in sorted(sources):
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

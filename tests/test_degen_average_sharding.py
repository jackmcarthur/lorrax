"""Full-operator diagonal averaging must not re-enter the GW pipeline."""
from pathlib import Path
import ast


def test_matrix_averaging_entry_points_are_absent():
    root = Path(__file__).resolve().parents[1] / "src" / "gw"
    retired = {"average_sigma_components", "average_matrix_diagonal",
               "apply_to_matrix_diagonals"}
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.name not in retired, (path, node.lineno)
            elif isinstance(node, ast.Name):
                assert node.id not in retired, (path, node.lineno)
            elif isinstance(node, ast.Attribute):
                assert node.attr not in retired, (path, node.lineno)

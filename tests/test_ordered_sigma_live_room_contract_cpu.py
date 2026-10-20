"""Execute the real dispatch statements with host spies; NOT MPI admission."""
import ast
from pathlib import Path
from types import SimpleNamespace


SOURCE = Path(__file__).resolve().parents[1]/"src/gw/ppm_sigma.py"


def test_ordered_executor_binds_fresh_live_room_before_any_synthesis():
    tree = ast.parse(SOURCE.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_ordered_window_lines")
    index = next(i for i, node in enumerate(function.body) if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "ordered_shared_pole_sigma")
    statements = function.body[index-2:index+1]
    assert isinstance(statements[0], ast.Assign)
    assert statements[0].targets[0].id == "meta_sigma"
    assert statements[0].value.func.id == "fresh_shared_pole_capacity"
    assert isinstance(statements[1], ast.Expr)
    assert ast.unparse(statements[1].value) == "meta_sigma.shared_pole_capacity.at_phase()"
    assert statements[2].value.args[2].id == "meta_sigma"
    events = []
    original_meta, mesh, wfns, model = (object() for _ in range(4))
    result, receipt = object(), object()
    fresh = SimpleNamespace(shared_pole_capacity=SimpleNamespace(
        at_phase=lambda: events.append("all-rank-live-room")))
    def create(meta, xy):
        assert meta is original_meta and xy is mesh
        events.append("fresh-ledger")
        return fresh
    def execute(wavefunctions, handle, meta, xy, **kw):
        assert events == ["fresh-ledger", "all-rank-live-room"]
        assert wavefunctions is wfns and handle is model and xy is mesh
        assert meta is fresh
        assert kw["projection_bands"] == (2, 6)
        assert kw["regularization_width_ry"] == .25/13.
        assert kw["quadrature_eps"] == 1.e-4
        assert kw["request"] is request and kw["sigma_branches"] is branches
        assert kw["band_brackets"] is brackets and kw["band_counts"] is counts
        assert kw["band_matrix_counts"] is matrix_counts
        events.append("native-ordered-executor")
        return result, receipt
    request, branches, counts, matrix_counts = (object() for _ in range(4))
    brackets = ((0, 6),)
    scope = dict(fresh_shared_pole_capacity=create, ordered_shared_pole_sigma=execute,
        meta_t=original_meta, mesh_xy=mesh, wfns_t=wfns, source=dict(model=model),
        request=request, branches=branches, brackets=brackets, band_counts=counts,
        matrix_counts=matrix_counts, b_lo=2, hi=6, print_fn=print,
        cfg=SimpleNamespace(regularization_ev=.25, quadrature_eps=1.e-4), RYD_TO_EV=13.)
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(SOURCE), "exec"), scope)
    assert scope["res"] is result and scope["receipt"] is receipt
    assert events == ["fresh-ledger", "all-rank-live-room", "native-ordered-executor"]


def test_live_room_binding_is_call_site_only_not_a_new_fresh_ledger_default():
    tree = ast.parse(SOURCE.read_text())
    fresh = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "fresh_shared_pole_capacity")
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr == "at_phase" for node in ast.walk(fresh))

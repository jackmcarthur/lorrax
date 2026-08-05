"""The SC loop must not hold two ω-cubes at once.

``SigmaResult.sigma_c_omega_kij_ry`` is ``(nω, nk, nb, nb)`` and, at the
default ``sigma_omega_layout = "replicated"``, it is replicated on every
rank: ``gw_config.py`` prices it at 2751 MB/rank at nb=512, and that
figure does not shrink with P.  Both accelerator drivers used to keep
iteration *i−1*'s ``SigmaResult`` alive for the whole of iteration *i* —
rCROP held it in a closure cell AND passed it into the input state,
linear mixing passed the loop carry itself — so the peak was two of
them, a P-independent doubling of the largest object on the surface.

Nothing in the loop reads it.  ``SCState``'s own docstring says so
("purely for the final output writer … it does not feed the next
iteration"); the only consumers are ``dump_sigma_omega_h5_final`` and
``run_sc_driver``'s finalize, and both want the LAST one, which still
survives.

That "nothing reads it" is the load-bearing claim, so it is the
assertion: ``gw_iteration_map`` may touch ``state.iteration`` and
``state.H_qp_dft`` and nothing else.  A later change that made the map
depend on the previous Σ would silently make the drop wrong, and would
fail here first.

AST plus source text, no jax, so it runs anywhere.
"""
import ast
import pathlib

import pytest


_PATH = (pathlib.Path(__file__).resolve().parents[1]
         / "src" / "gw" / "sc_iteration.py")
_SRC = _PATH.read_text()
_TREE = ast.parse(_SRC)


def _func(name):
    for node in ast.walk(_TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {_PATH}")


def _block(name):
    """Source text of a top-level ``def``, up to the next top-level one."""
    start = _SRC.index(f"\ndef {name}(") + 1
    nxt = _SRC.find("\ndef ", start + 1)
    return _SRC[start: nxt if nxt != -1 else len(_SRC)]


def _state_attrs(fn, var="state"):
    return {n.attr for n in ast.walk(fn)
            if isinstance(n, ast.Attribute)
            and isinstance(n.value, ast.Name) and n.value.id == var}


def test_gw_iteration_map_reads_only_the_carry_and_the_counter():
    assert _state_attrs(_func("gw_iteration_map")) == {
        "iteration", "H_qp_dft"}


@pytest.mark.parametrize("driver", ["_run_rcrop", "_run_linear_mixing"])
def test_no_driver_feeds_a_stale_sigma_result_into_the_map(driver):
    """Every ``SCState(...)`` built as a map ARGUMENT carries H and i only.

    The finalize state at the end of ``_run_rcrop`` legitimately carries
    the last Σ, so the check is on the constructions whose keywords are
    exactly the two fields the map reads — there must be at least one —
    and on every other construction still naming ``last_sigma_result``.
    """
    fn = _func(driver)
    inputs, finals = [], []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "SCState":
            keys = {kw.arg for kw in node.keywords}
            (inputs if keys == {"H_qp_dft", "iteration"} else finals).append(
                keys)
    assert inputs, f"{driver} builds no bare input SCState"
    for keys in finals:
        assert "last_sigma_result" in keys, (
            f"{driver} builds an SCState with {sorted(keys)} — an argument "
            f"to gw_iteration_map must carry H_qp_dft and iteration only")


def test_rcrop_clears_the_capture_cells_before_the_map_call():
    """The cell must be emptied BEFORE ``gw_iteration_map``, not after.

    Assigning after the call is what the previous code did, and that is
    the whole defect: the reference stays live for the entire build of
    the next Σ.
    """
    body = _block("_run_rcrop")
    clear = body.index("_last_sigma[0] = None")
    call = body.index("gw_iteration_map(")
    assert clear < call


def test_linear_mixing_rebuilds_the_carry_before_the_map_call():
    body = _block("_run_linear_mixing")
    rebuild = body.index("state = SCState(H_qp_dft=state.H_qp_dft")
    call = body.index("gw_iteration_map(")
    assert rebuild < call


def test_the_omega_cube_is_in_the_residency_census():
    """The saving has to be measurable, or the claim is unverifiable."""
    body = _block("gw_iteration_map")
    census = body[body.index("_residency_census("):]
    assert '"sigma_c_omega_kij_ry"' in census

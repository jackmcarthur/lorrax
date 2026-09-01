"""Source contract for minimum-memory IBZ TT Lorentz persistence."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WRITER = ROOT / "src" / "gw" / "v_q_bispinor.py"


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    matches = [node for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(matches) == 1, (name, len(matches))
    return matches[0]


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def test_ibz_tt_writer_mixes_persists_and_deletes_one_output_per_loop():
    """The writer cannot regress to synthesising/storing all TT outputs."""
    source = WRITER.read_text()
    fn = _function(
        ast.parse(source), "compute_V_q_bispinor_g_flat_to_h5")

    mix_loops = []
    for loop in (node for node in ast.walk(fn) if isinstance(node, ast.For)):
        calls = [call for call in ast.walk(loop) if isinstance(call, ast.Call)]
        if any(_call_name(call) == "mix_one_channel_by_proper_rotation"
               for call in calls):
            mix_loops.append(loop)
    assert len(mix_loops) == 1
    loop = mix_loops[0]
    assert isinstance(loop.iter, ast.Name) and loop.iter.id == "UNIQUE_TILES"
    assert any(
        isinstance(node, ast.Delete)
        and any(isinstance(target, ast.Name) and target.id == "V_mix"
                for target in node.targets)
        for node in ast.walk(loop)
    ), "each persisted output must be explicitly released in its loop"

    names = {node.id for node in ast.walk(fn) if isinstance(node, ast.Name)}
    assert "tt_full_in" not in names
    assert "tt_mixed" not in names
    assert not any(
        isinstance(call, ast.Call)
        and _call_name(call) == "mix_channels_by_proper_rotation"
        for call in ast.walk(fn)
    ), "the all-output compatibility mixer is forbidden in the writer"


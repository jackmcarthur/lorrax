"""Header path identity across checkout aliases, without JAX or a device client."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def _owner():
    # Run the actual stdlib-only owner; importing the FFT router would load
    # JAX just to test filesystem metadata on a login host.
    source = Path(__file__).parents[1] / "src/ffi/fft.py"
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "mathdx_root")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["mathdx_root"]


def test_mathdx_header_root_aliases_and_distinct_wheels(tmp_path, monkeypatch):
    roots = [tmp_path / name for name in ("wheel_a", "wheel_b")]
    for root in roots:
        (root / "include").mkdir(parents=True)
        (root / "include/cufftdx.hpp").write_text("same header contents\n")
    aliases = [tmp_path / "checkout_a", tmp_path / "checkout_b"]
    for alias in aliases:
        alias.symlink_to(roots[0], target_is_directory=True)
    observed = []
    for location in (*aliases, roots[1]):
        monkeypatch.setattr(importlib.util, "find_spec", lambda name, p=location:
                            SimpleNamespace(submodule_search_locations=[str(p)]))
        observed.append(_owner()())
    assert observed[0] == observed[1] == str(roots[0].resolve())
    # Equal contents do not merge genuinely different physical wheel roots.
    assert observed[2] == str(roots[1].resolve()) != observed[0]
    monkeypatch.setattr(importlib.util, "find_spec", lambda name:
                        SimpleNamespace(submodule_search_locations=[str(tmp_path / "absent")]))
    with pytest.raises(RuntimeError, match="GATE mathdx-headers"):
        _owner()()

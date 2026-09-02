"""Static-photon stream policy uses the canonical configured memory budget."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gw import w_isdf


ROOT = Path(__file__).resolve().parents[1]


def _function(name: str) -> ast.FunctionDef:
    tree = ast.parse((ROOT / "src/gw/w_isdf.py").read_text())
    matches = [node for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(matches) == 1
    return matches[0]


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _keyword_forwarded(caller: str, callee: str) -> ast.expr:
    fn = _function(caller)
    calls = [node for node in ast.walk(fn)
             if isinstance(node, ast.Call) and _call_name(node) == callee]
    assert len(calls) == 1, (caller, callee, len(calls))
    values = [kw.value for kw in calls[0].keywords
              if kw.arg == "memory_per_device_gb"]
    assert len(values) == 1, (caller, callee)
    return values[0]


def test_run272_explicit_116gb_selects_full_cc_ct_without_autodetect(
        monkeypatch):
    import common.gpu_utils as gpu_utils

    def refuse_autodetect():
        raise AssertionError("explicit production budget was ignored")

    monkeypatch.setattr(gpu_utils, "get_device_memory_gb", refuse_autodetect)
    mesh = SimpleNamespace(
        devices=np.empty((8, 8), dtype=object))
    # Exact run272 P64 face extents: nq=81, padded C=6016, padded T=2048.
    charge = (81, 500, 6016, 4)
    transverse = (81, 500, 2048, 4)

    for process_index in (0, 63):
        monkeypatch.setattr(
            w_isdf.jax, "process_index",
            lambda process_index=process_index: process_index)
        assert not w_isdf._resolve_vertex_spin_pair_stream(
            mesh, charge, charge, None, memory_per_device_gb=116.0)
        assert not w_isdf._resolve_vertex_spin_pair_stream(
            mesh, charge, transverse, None, memory_per_device_gb=116.0)


def test_stream_policy_is_rank_invariant_for_one_configured_budget(
        monkeypatch, capsys):
    import common.gpu_utils as gpu_utils

    def refuse_autodetect():
        raise AssertionError("explicit production budget was ignored")

    monkeypatch.setattr(gpu_utils, "get_device_memory_gb", refuse_autodetect)
    mesh = SimpleNamespace(devices=np.empty((8, 8), dtype=object))
    charge = (81, 500, 6016, 4)

    decisions = []
    for process_index in (0, 1, 31, 63):
        monkeypatch.setattr(
            w_isdf.jax, "process_index",
            lambda process_index=process_index: process_index)
        decisions.append(w_isdf._resolve_vertex_spin_pair_stream(
            mesh, charge, charge, None, memory_per_device_gb=16.0))
    assert decisions == [True, True, True, True]
    assert capsys.readouterr().out.count("spin-pair stream selected") == 1


def test_production_threads_configured_budget_to_stream_resolver():
    static_to_packed = _keyword_forwarded(
        "compute_static_photon_response",
        "compute_experimental_no_pair_photon_chi0")
    assert isinstance(static_to_packed, ast.IfExp)
    assert isinstance(static_to_packed.orelse, ast.Attribute)
    assert static_to_packed.orelse.attr == "per_device_gb"
    memory = static_to_packed.orelse.value
    assert isinstance(memory, ast.Attribute) and memory.attr == "memory"
    assert isinstance(memory.value, ast.Name) and memory.value.id == "config"

    # Production fusion and the retained per-block oracle both terminate at
    # the same resolver with the same canonical budget.
    for caller, callee in (
            ("compute_experimental_no_pair_photon_chi0",
             "_resolve_vertex_spin_pair_stream"),
            ("compute_no_pair_dirac_current_block",
             "_get_chi_minimax_kernel"),
            ("_get_chi_minimax_kernel",
             "_resolve_vertex_spin_pair_stream")):
        value = _keyword_forwarded(caller, callee)
        assert isinstance(value, ast.Name)
        assert value.id == "memory_per_device_gb"


def test_only_diagnostic_none_budget_falls_back_to_autodetect(monkeypatch):
    import common.gpu_utils as gpu_utils

    calls = []
    monkeypatch.setattr(
        gpu_utils, "get_device_memory_gb",
        lambda: calls.append("autodetect") or 116.0)
    mesh = SimpleNamespace(devices=np.empty((8, 8), dtype=object))
    charge = (81, 500, 6016, 4)

    assert not w_isdf._resolve_vertex_spin_pair_stream(
        mesh, charge, charge, None)
    assert calls == ["autodetect"]

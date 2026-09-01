"""Static-photon dead-state and coarse completion-cadence contracts."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def _function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    matches = [node for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(matches) == 1, (path, name, len(matches))
    return matches[0]


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _config(*, static, screening=True, head="full"):
    return SimpleNamespace(
        static=static,
        compute_mode=SimpleNamespace(needs_screening=screening),
        head=SimpleNamespace(correction=head),
    )


def test_scalar_head_bundle_is_suppressed_only_for_static_photon_modes():
    fn = _function(ROOT / "src/gw/gw_init.py",
                   "_needs_scalar_head_wavefunctions")
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "uses_static_photon_response": lambda cfg: cfg.static,
        "ISOMETRIC_KINETIC_BALANCE_CHARGE_REPRESENTATION": "ikb",
    }
    exec(compile(module, "gw_init_policy", "exec"), namespace)
    needs = namespace["_needs_scalar_head_wavefunctions"]
    representation = SimpleNamespace(
        charge_representation="ikb")

    assert not needs(representation, _config(static=True))

    # The pre-existing non-photon FULL-head predicate remains true.
    assert needs(representation, _config(static=False))
    assert not needs(representation, _config(static=False, screening=False))
    assert not needs(representation, _config(static=False, head="off"))
    assert not needs(
        SimpleNamespace(charge_representation="pauli"),
        _config(static=False))


def test_scalar_head_load_and_both_bundle_builds_share_the_consumer_guard():
    """A false policy result must bypass load and both layout constructors."""
    fn = _function(ROOT / "src/gw/gw_init.py",
                   "prepare_isdf_and_wavefunctions")
    guarded = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        if any(isinstance(call, ast.Call)
               and _call_name(call) == "_needs_scalar_head_wavefunctions"
               for call in ast.walk(node.test)):
            guarded.append(node)
    assert len(guarded) == 1
    body_calls = {
        _call_name(call) for stmt in guarded[0].body
        for call in ast.walk(stmt) if isinstance(call, ast.Call)
    }
    assert {
        "load_centroids_band_chunked",
        "wavefunctions_face_from_restart",
        "build_wavefunction_bundle",
    } <= body_calls


class _ReadyValue:
    def __init__(self, events):
        self.events = events
        self.shape = (1, 1, 1)

    def block_until_ready(self):
        self.events.append("ready")
        return self


def test_completion_receipt_blocks_every_rank_before_rank0_emits(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "isolated_common_timing", ROOT / "src/common/timing.py")
    assert spec is not None and spec.loader is not None
    timing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(timing)
    events = []
    monkeypatch.setattr(timing, "_rank0", lambda: True)
    monkeypatch.setattr(timing, "_trace", lambda message: events.append(message))
    monkeypatch.setattr(timing.time, "perf_counter", lambda: 12.5)

    elapsed = timing.completion_receipt(
        "packed photon test", _ReadyValue(events), started_at=10.0)
    assert events == ["ready", "packed photon test complete  2.5 s"]
    assert elapsed == 2.5

    events.clear()
    monkeypatch.setattr(timing, "_rank0", lambda: False)
    timing.completion_receipt("peer test", _ReadyValue(events))
    assert events == ["ready"], "non-owner ranks must synchronize but stay silent"


def test_chi_packer_has_one_post_insert_receipt_in_row_major_4x4_loop():
    fn = _function(ROOT / "src/gw/photon_layout.py",
                   "pack_photon_operator")
    outer = [node for node in ast.walk(fn)
             if isinstance(node, ast.For)
             and isinstance(node.target, ast.Name) and node.target.id == "A"]
    assert len(outer) == 1
    inner = [node for node in ast.walk(outer[0])
             if isinstance(node, ast.For)
             and isinstance(node.target, ast.Name) and node.target.id == "B"]
    assert len(inner) == 1
    for loop in (outer[0], inner[0]):
        assert isinstance(loop.iter, ast.Call)
        assert _call_name(loop.iter) == "range"
        assert (len(loop.iter.args) == 1
                and isinstance(loop.iter.args[0], ast.Name)
                and loop.iter.args[0].id == "N_LORENTZ")
    calls = [call for call in ast.walk(inner[0]) if isinstance(call, ast.Call)]
    gets = [call for call in calls if _call_name(call) == "get_block"]
    inserts = [call for call in calls if _call_name(call) == "_insert"]
    receipts = [call for call in calls
                if _call_name(call) == "completion_receipt"]
    assert len(gets) == len(inserts) == len(receipts) == 1
    assert gets[0].lineno < inserts[0].lineno < receipts[0].lineno
    assert isinstance(receipts[0].args[1], ast.Name)
    assert receipts[0].args[1].id == "packed"


def test_dyson_and_sigma_receipts_follow_their_completed_outputs():
    w_fn = _function(ROOT / "src/gw/w_isdf.py",
                     "compute_static_photon_response")
    w_receipts = [call for call in ast.walk(w_fn)
                  if isinstance(call, ast.Call)
                  and _call_name(call) == "completion_receipt"]
    assert len(w_receipts) == 1
    assert (isinstance(w_receipts[0].args[0], ast.Constant)
            and w_receipts[0].args[0].value == "packed photon Dyson")
    assert isinstance(w_receipts[0].args[1], ast.Name)
    assert w_receipts[0].args[1].id == "W_packed"

    sigma_fn = _function(ROOT / "src/gw/photon_sigma.py",
                         "compute_static_photon_sigma")
    outer = [node for node in ast.walk(sigma_fn)
             if isinstance(node, ast.For)
             and isinstance(node.target, ast.Name) and node.target.id == "A"]
    assert len(outer) == 1
    inner = [node for node in ast.walk(outer[0])
             if isinstance(node, ast.For)
             and isinstance(node.target, ast.Name) and node.target.id == "B"]
    assert len(inner) == 1
    receipts = [call for call in ast.walk(inner[0])
                if isinstance(call, ast.Call)
                and _call_name(call) == "completion_receipt"]
    contractions = [call for call in ast.walk(inner[0])
                    if isinstance(call, ast.Call)
                    and _call_name(call) == "contract_block"]
    assert len(receipts) == 1
    assert len(contractions) == 3
    assert isinstance(receipts[0].args[1], ast.Name)
    assert receipts[0].args[1].id == "sig_coh"
    assert receipts[0].lineno > max(call.lineno for call in contractions)


def test_no_completion_receipt_is_inside_a_tau_kernel():
    fn = _function(ROOT / "src/gw/w_isdf.py",
                   "_get_chi_minimax_kernel_face")
    assert not any(isinstance(call, ast.Call)
                   and _call_name(call) == "completion_receipt"
                   for call in ast.walk(fn))

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


def test_static_photon_driver_releases_scalar_v_before_packed_response():
    fn = _function(ROOT / "src/gw/gw_jax.py", "main")
    photon_arms = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        test_calls = [_call_name(call) for call in ast.walk(node.test)
                      if isinstance(call, ast.Call)]
        body_calls = [_call_name(call) for stmt in node.body
                      for call in ast.walk(stmt)
                      if isinstance(call, ast.Call)]
        if ("uses_static_photon_response" in test_calls
                and "compute_static_photon_response" in body_calls):
            photon_arms.append(node)
    assert len(photon_arms) == 1
    arm = photon_arms[0]
    response_call = next(
        call for call in ast.walk(arm)
        if isinstance(call, ast.Call)
        and _call_name(call) == "compute_static_photon_response")
    none_assignments = []
    for node in ast.walk(arm):
        if not isinstance(node, ast.Assign):
            continue
        if (not isinstance(node.value, ast.Constant)
                or node.value.value is not None):
            continue
        none_assignments.extend((target, node.lineno) for target in node.targets)
    names = {
        target.id: lineno for target, lineno in none_assignments
        if isinstance(target, ast.Name)
    }
    attributes = {
        (target.value.id, target.attr): lineno
        for target, lineno in none_assignments
        if (isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name))
    }
    assert names["V_q"] < response_call.lineno
    assert attributes[("isdf", "V_qmunu")] < response_call.lineno

    # There must be no second local alias keeping the same device buffer live.
    local_vqmunu = [node for node in ast.walk(fn)
                    if isinstance(node, ast.Name)
                    and node.id == "V_qmunu"]
    assert not local_vqmunu


def test_sigma_dispatch_enforces_disjoint_scalar_and_photon_v_owners():
    fn = _function(ROOT / "src/gw/sigma_dispatch.py", "compute_sigma_xc")
    photon_arms = [
        node for node in ast.walk(fn)
        if (isinstance(node, ast.If)
            and any(isinstance(call, ast.Call)
                    and _call_name(call) == "uses_static_photon_response"
                    for call in ast.walk(node.test)))
    ]
    assert len(photon_arms) == 1
    arm = photon_arms[0]

    first = arm.body[0]
    assert isinstance(first, ast.If)
    assert isinstance(first.test, ast.Compare)
    assert (isinstance(first.test.left, ast.Name)
            and first.test.left.id == "V_q")
    assert isinstance(first.test.ops[0], ast.IsNot)
    assert (isinstance(first.test.comparators[0], ast.Constant)
            and first.test.comparators[0].value is None)
    assert any(isinstance(node, ast.Raise) for node in ast.walk(first))

    # Scalar W fallback and scalar kernels live only in the non-photon arm,
    # after its reciprocal V_q-is-None refusal.
    assert arm.orelse and isinstance(arm.orelse[0], ast.If)
    scalar_guard = arm.orelse[0]
    assert isinstance(scalar_guard.test, ast.Compare)
    assert (isinstance(scalar_guard.test.left, ast.Name)
            and scalar_guard.test.left.id == "V_q")
    assert isinstance(scalar_guard.test.ops[0], ast.Is)
    assert any(isinstance(node, ast.Raise)
               for node in ast.walk(scalar_guard.body[0]))
    scalar_calls = {
        _call_name(call) for node in arm.orelse[1:]
        for call in ast.walk(node) if isinstance(call, ast.Call)
    }
    assert {"get", "compute_cohsex_sigma", "compute_sigma_x"} <= scalar_calls
    assert not any(_call_name(call) in {
        "get", "compute_cohsex_sigma", "compute_sigma_x"
    } for node in arm.body for call in ast.walk(node)
      if isinstance(call, ast.Call))


def test_solve_w_default_copies_rhs_and_consume_contract_does_not():
    public = _function(ROOT / "src/gw/w_isdf.py", "solve_w")
    defaults = {
        arg.arg: default for arg, default in zip(
            public.args.kwonlyargs, public.args.kw_defaults)
        if default is not None
    }
    consume_default = defaults["consume_v"]
    assert (isinstance(consume_default, ast.Constant)
            and consume_default.value is False)

    selector = _function(ROOT / "src/gw/w_isdf.py", "_select_w_rhs")
    module = ast.Module(body=[selector], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, "w_rhs_policy", "exec"), namespace)
    choose = namespace["_select_w_rhs"]
    source = object()
    copied = object()
    calls = []

    def copy_fn(value):
        calls.append(value)
        return copied

    assert choose(source, copy_fn, consume_v=False) is copied
    assert calls == [source], "the incumbent default must retain its copy"
    calls.clear()
    assert choose(source, copy_fn, consume_v=True) is source
    assert calls == [], "the consuming path must pass V directly as B"


def test_consuming_residual_keeps_only_a_bounded_pre_donation_snapshot():
    fn = _function(ROOT / "src/gw/w_isdf.py", "_solve_w_dist")
    snapshot_calls = [
        call for call in ast.walk(fn)
        if isinstance(call, ast.Call) and _call_name(call) == "_snapshot"
    ]
    assert len(snapshot_calls) == 1
    snapshot_arg = snapshot_calls[0].args[0]
    assert isinstance(snapshot_arg, ast.Subscript)
    assert (isinstance(snapshot_arg.value, ast.Name)
            and snapshot_arg.value.id == "V_flat")
    assert isinstance(snapshot_arg.slice, ast.Slice)
    assert (isinstance(snapshot_arg.slice.upper, ast.Name)
            and snapshot_arg.slice.upper.id == "ns")

    plan_calls = [
        call for call in ast.walk(fn)
        if isinstance(call, ast.Call) and _call_name(call) == "batched"
    ]
    reports = [
        call for call in ast.walk(fn)
        if isinstance(call, ast.Call)
        and _call_name(call) == "_w_residual_report"
    ]
    assert len(plan_calls) == len(reports) == 1
    assert snapshot_calls[0].lineno < plan_calls[0].lineno < reports[0].lineno
    assert (isinstance(reports[0].args[0], ast.Name)
            and reports[0].args[0].id == "residual_v")


def test_one_reader_packer_helper_owns_both_photon_v_loads():
    helper = _function(ROOT / "src/gw/w_isdf.py",
                       "_load_packed_static_photon_v")
    helper_calls = [_call_name(call) for call in ast.walk(helper)
                    if isinstance(call, ast.Call)]
    assert helper_calls.count("BispinorVqReader") == 1
    assert helper_calls.count("pack_photon_operator") == 1

    response = _function(ROOT / "src/gw/w_isdf.py",
                         "compute_static_photon_response")
    calls = [call for call in ast.walk(response) if isinstance(call, ast.Call)]
    loads = sorted(
        (call for call in calls
         if _call_name(call) == "_load_packed_static_photon_v"),
        key=lambda call: call.lineno)
    assert len(loads) == 2
    assert not any(_call_name(call) in {
        "BispinorVqReader", "pack_photon_operator"
    } for call in calls)
    assert not any(keyword.arg == "expected_layout"
                   for keyword in loads[0].keywords)
    assert any(keyword.arg == "expected_layout"
               and isinstance(keyword.value, ast.Name)
               and keyword.value.id == "layout"
               for keyword in loads[1].keywords)


def test_consumed_photon_v_is_invalid_until_w_completion_and_reload():
    fn = _function(ROOT / "src/gw/w_isdf.py",
                   "compute_static_photon_response")
    calls = [call for call in ast.walk(fn) if isinstance(call, ast.Call)]
    solve = next(call for call in calls if _call_name(call) == "solve_w")
    consume = next(keyword.value for keyword in solve.keywords
                   if keyword.arg == "consume_v")
    assert isinstance(consume, ast.Constant) and consume.value is True

    invalidations = [
        node for node in ast.walk(fn)
        if (isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name)
                    and target.id == "V_packed" for target in node.targets)
            and isinstance(node.value, ast.Constant)
            and node.value.value is None)
    ]
    receipts = [call for call in calls
                if _call_name(call) == "completion_receipt"]
    reloads = [call for call in calls
               if (_call_name(call) == "_load_packed_static_photon_v"
                   and any(keyword.arg == "expected_layout"
                           for keyword in call.keywords))]
    assert len(invalidations) == len(receipts) == len(reloads) == 1
    invalid = invalidations[0]
    receipt = receipts[0]
    reload = reloads[0]
    assert solve.lineno < invalid.lineno < receipt.lineno < reload.lineno

    stale_reads = [
        node for node in ast.walk(fn)
        if (isinstance(node, ast.Name)
            and node.id == "V_packed"
            and isinstance(node.ctx, ast.Load)
            and invalid.lineno < node.lineno < reload.lineno)
    ]
    assert not stale_reads

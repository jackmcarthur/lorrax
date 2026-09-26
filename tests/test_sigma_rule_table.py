"""The run-independent Σ rule table: a memo of the builder, never a policy.

A hit must return the bytes a cold build returned, so a warm plan is the
cold plan bit for bit; an entry the reader cannot authenticate is a named
miss; concurrent writers publish exactly one entry. Every cell here has its
own table (``tests/conftest.py::_private_sigma_rule_table``).
"""

import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gw.sigma_box_plan import (
    _rule_cache_lookup,
    _rule_digest,
    _rule_table_key,
    _rule_table_lookup,
    _rule_table_path,
    _rule_table_root,
    _rule_table_store,
    plan_sigma_windows,
)
from minimax import UniformRule, build_uniform_rule

from test_sigma_box_plan import _branch, _fake_rule, _frozen_digests, _summaries


_BOX = (-0.2, 0.1, 0.15, 0.15)
_QUIET = dict(print_fn=lambda *_args, **_kwargs: None)


def _plan(cache_dir, **kwargs):
    return plan_sigma_windows(
        _summaries(), [_branch()], np.asarray([0.2, 0.5]), 0.1,
        eps=1.0e-4, cache_dir=cache_dir, **kwargs)


def _drifting(calls):
    """A builder whose answer changes on every call: only a memo can make a
    warm plan equal the cold one, so the negative control is built in."""
    def build(box, eps, **kwargs):
        calls.append(tuple(box))
        rule = _fake_rule(box, eps, **kwargs)
        return replace(rule, times=rule.times + 1.0e-3j * len(calls))
    return build


def _nodes(plan):
    return [(np.asarray(row.window.nodes.t), np.asarray(row.window.nodes.alpha))
            for row in plan]


def test_warm_plan_is_the_cold_plan_bit_for_bit_without_a_builder_call(
        monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("gw.sigma_box_plan.build_uniform_rule", _drifting(calls))
    cold_plan, cold = _plan(str(tmp_path / "run_a"), **_QUIET)
    assert len(calls) == 3
    assert cold["rule_table_lookups"] == {"hit": 0, "built": 3}
    assert cold["rule_table_dir"] == os.environ["LORRAX_SIGMA_RULE_TABLE_TEST_DIR"]

    calls.clear()
    warm_plan, warm = _plan(str(tmp_path / "run_b"), **_QUIET)
    assert calls == []
    assert warm["rule_table_lookups"] == {"hit": 3, "built": 0}
    rows = warm["branches"][0]["windows"]
    assert [row["rule_table"] for row in rows] == ["hit"] * 3
    # The request scope is fresh, so its status is what a cold run reports.
    assert [row["cache_status"] for row in rows] == ["miss"] * 3
    for (t_cold, a_cold), (t_warm, a_warm) in zip(_nodes(cold_plan), _nodes(warm_plan)):
        assert t_cold.tobytes() == t_warm.tobytes()
        assert a_cold.tobytes() == a_warm.tobytes()

    # Negative control: with caching off the drifting builder runs again and
    # the plan moves, so the equality above was the table's doing.
    calls.clear()
    off_plan, off = _plan(None, **_QUIET)
    assert len(calls) == 3 and off["rule_table_dir"] is None
    assert [row["rule_table"] for row in off["branches"][0]["windows"]] == ["none"] * 3
    assert any(t.tobytes() != t_off.tobytes()
               for (t, _), (t_off, _) in zip(_nodes(cold_plan), _nodes(off_plan)))


def test_warm_sc_freeze_is_the_cold_freeze(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("gw.sigma_box_plan.build_uniform_rule", _drifting(calls))
    digests = []
    for run in ("run_a", "run_b"):
        session = {}
        _, geometry = _plan(str(tmp_path / run), fixed_rule_session=session, **_QUIET)
        digests.append(_frozen_digests(session, geometry))
        if run == "run_a":
            built = len(calls)
            calls.clear()
    assert built > 0 and calls == []
    assert digests[0] == digests[1]


def test_caching_off_writes_no_table(monkeypatch):
    monkeypatch.setattr("gw.sigma_box_plan.build_uniform_rule", _fake_rule)
    assert _rule_table_root(None) is None
    _plan(None, **_QUIET)
    assert not Path(os.environ["LORRAX_SIGMA_RULE_TABLE_TEST_DIR"]).exists()


def test_default_table_sits_beside_the_compile_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("LORRAX_SIGMA_RULE_TABLE_TEST_DIR")
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    assert _rule_table_root("any") == str(
        tmp_path / ".cache" / "lorrax" / "sigma_box_rules")


def test_fresh_request_scope_is_an_empty_cache_not_a_failure(tmp_path):
    assert _rule_cache_lookup(
        str(tmp_path / "request_fresh"), _BOX, 1.0e-4, False,
        noise_amplification_cap=1.0e9) == (None, ())


# ------------------------------------------------------------- red twins
def _stored_entry(root, box=_BOX):
    key = _rule_table_key(box, 1.0e-4, False, None)
    rule = _fake_rule(box, 1.0e-4)
    assert _rule_table_store(root, key, rule, 1.0) is None
    path, _blob = _rule_table_path(root, key)
    return key, rule, Path(path)


def _rewrite(path, **changes):
    with np.load(path) as data:
        values = {name: np.asarray(data[name]) for name in data.files}
    values.update(changes)
    with open(path, "wb") as handle:
        np.savez(handle, **values)


def test_a_valid_entry_is_served(tmp_path):
    """The positive control for the red twins below."""
    key, rule, _path = _stored_entry(str(tmp_path))
    entry, why = _rule_table_lookup(str(tmp_path), key)
    assert why is None and entry[2] == _rule_digest(rule, 1.0)
    assert entry[0].times.tobytes() == rule.times.tobytes()


@pytest.mark.parametrize("changes,named", [
    (dict(table_format="sigma-box-table-v0"), "schema mismatch"),
    (dict(schema="sigma-box-ry-v4"), "schema mismatch"),
    (dict(box=np.asarray([-0.3, 0.1, 0.15, 0.15])), "key mismatch"),
    (dict(key="{}"), "key mismatch"),
    (dict(weights=np.asarray([0.6 - 0.1j, 0.31 + 0.05j])), "digest mismatch"),
    (dict(sup_error=np.asarray(2.0e-4)), "digest mismatch"),
])
def test_an_unauthenticated_entry_is_a_named_miss_and_is_replaced(
        monkeypatch, tmp_path, changes, named):
    root = os.environ["LORRAX_SIGMA_RULE_TABLE_TEST_DIR"]
    calls = []
    monkeypatch.setattr("gw.sigma_box_plan.build_uniform_rule", _drifting(calls))
    _plan(str(tmp_path / "run_a"), **_QUIET)
    paths = sorted(Path(root).rglob("rule_*.npz"))
    assert len(paths) == 3
    _rewrite(paths[0], **changes)

    calls.clear()
    lines = []
    _plan(str(tmp_path / "run_b"), print_fn=lines.append)
    warnings = [line for line in lines if "rule table entry not served" in line]
    assert len(calls) == 1 and len(warnings) == 1
    assert named in warnings[0] and str(paths[0]) in warnings[0]
    assert "replaced by this run's build" in " ".join(lines)

    calls.clear()
    _, geometry = _plan(str(tmp_path / "run_c"), **_QUIET)
    assert calls == [] and geometry["rule_table_lookups"]["hit"] == 3


def test_another_solver_identity_opens_another_namespace(monkeypatch, tmp_path):
    key, _rule, _path = _stored_entry(str(tmp_path))
    monkeypatch.setattr(
        "gw.sigma_box_plan.uniform_rule_solver_identity",
        lambda: dict(key["solver"], minimax_sources="edited builder"))
    other = _rule_table_key(_BOX, 1.0e-4, False, None)
    assert _rule_table_path(str(tmp_path), other)[0] != _rule_table_path(
        str(tmp_path), key)[0]
    assert _rule_table_lookup(str(tmp_path), other) == (None, None)


def test_a_one_ulp_request_is_another_key(tmp_path):
    """Exact keys only: a table never serves a neighbouring cell."""
    key, _rule, _path = _stored_entry(str(tmp_path))
    near = _rule_table_key((_BOX[0], np.nextafter(_BOX[1], 1.0), *_BOX[2:]),
                           1.0e-4, False, None)
    assert _rule_table_lookup(str(tmp_path), near) == (None, None)


# ------------------------------------------------------ concurrent writers
_CHILD = r"""
import json, sys, time
from pathlib import Path
go, root, box, variant, mode = sys.argv[1], sys.argv[2], json.loads(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
from runtime.source_closure import ensure_source_closure
ensure_source_closure(print_fn=lambda _line: None)
import numpy as np
from common.collectives import process_rank
from minimax import UniformRule
from gw.sigma_box_plan import (_rule_digest, _rule_table_key,
                               _rule_table_lookup, _rule_table_store)
process_rank()
rule = UniformRule(
    times=np.asarray([0.2 + 0.03j, 0.4 + 0.02j + 1.0e-3j * variant]),
    weights=np.asarray([0.6 - 0.1j, 0.3 + 0.05j]), box=tuple(box), eps=1.0e-4,
    relative=False, theta_deg=5.0, rank=3, sup_error=5.0e-5, kappa_max=1.2,
    seconds=0.0)
key = _rule_table_key(box, 1.0e-4, False, None)
Path(go + ".ready." + sys.argv[6]).touch()
while not Path(go).exists():
    time.sleep(0.0005)
if mode == "write":
    print(json.dumps({"digest": _rule_digest(rule, 1.0),
                      "warning": _rule_table_store(root, key, rule, 1.0)}))
else:
    problems, hits, end = [], 0, time.time() + 1.5
    while time.time() < end:
        entry, why = _rule_table_lookup(root, key)
        hits += entry is not None
        if why is not None:
            problems.append(why)
    print(json.dumps({"problems": problems, "hits": hits}))
"""


def _race(tmp_path, box, variants, readers=2):
    root = str(tmp_path / "table")
    go = str(tmp_path / "go")
    env = dict(os.environ, JAX_PLATFORMS="cpu",
               PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    jobs = [("write", variant) for variant in variants] + [("read", 0)] * readers
    procs = [subprocess.Popen(
        [sys.executable, "-c", _CHILD, go, root, json.dumps(list(box)),
         str(variant), mode, str(index)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        for index, (mode, variant) in enumerate(jobs)]
    deadline = time.time() + 180.0
    while len(list(tmp_path.glob("go.ready.*"))) < len(jobs):
        assert time.time() < deadline, "writers did not start"
        assert all(proc.poll() is None for proc in procs), [
            proc.stderr.read() for proc in procs if proc.poll() is not None]
        time.sleep(0.05)
    Path(go).touch()
    results = []
    for (mode, _variant), proc in zip(jobs, procs):
        out, err = proc.communicate(timeout=180)
        assert proc.returncode == 0, err
        results.append((mode, json.loads(out.strip().splitlines()[-1])))
    return root, results


def _entries(root):
    return (sorted(Path(root).rglob("rule_*.npz")),
            sorted(Path(root).rglob("*.tmp")))


def test_concurrent_writers_of_one_rule_publish_one_entry(tmp_path):
    root, results = _race(tmp_path, _BOX, [0] * 8)
    writes = [row for mode, row in results if mode == "write"]
    reads = [row for mode, row in results if mode == "read"]
    assert all(row["warning"] is None for row in writes)
    npz, temporaries = _entries(root)
    assert len(npz) == 1 and temporaries == []
    entry, why = _rule_table_lookup(root, _rule_table_key(_BOX, 1.0e-4, False, None))
    assert why is None and entry[2] == writes[0]["digest"]
    assert all(row["problems"] == [] for row in reads)


def test_concurrent_writers_of_different_rules_keep_the_first(tmp_path):
    """A builder that is not a function of its key: exactly one entry
    survives, it never changes once published, and every other writer
    says DETERMINISM."""
    root, results = _race(tmp_path, _BOX, list(range(8)))
    writes = [row for mode, row in results if mode == "write"]
    reads = [row for mode, row in results if mode == "read"]
    npz, temporaries = _entries(root)
    assert len(npz) == 1 and temporaries == []
    entry, why = _rule_table_lookup(root, _rule_table_key(_BOX, 1.0e-4, False, None))
    assert why is None
    winners = [row for row in writes if row["warning"] is None]
    assert len(winners) == 1 and winners[0]["digest"] == entry[2]
    assert all("DETERMINISM" in row["warning"] for row in writes
               if row is not winners[0])
    assert all(row["problems"] == [] for row in reads)


# ------------------------------------------------------ the premise itself
@pytest.mark.parametrize("box,kwargs", [
    ((0.05, 3.0, 0.05, 0.05), dict(kappa_cap=100.0)),
    ((-0.6, 0.4, 0.1, 0.1), {}),
])
def test_the_builder_is_a_function_of_its_key(box, kwargs):
    """What makes a memo exact (claim 2737): two builds of one key on one
    machine return the same bytes."""
    first = build_uniform_rule(box, 1.0e-4, **kwargs)
    second = build_uniform_rule(box, 1.0e-4, **kwargs)
    assert first.times.tobytes() == second.times.tobytes()
    assert first.weights.tobytes() == second.weights.tobytes()
    assert _rule_digest(first, 1.0) == _rule_digest(second, 1.0)

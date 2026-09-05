"""The executable known-failure ledger refuses silent coverage loss."""

from pathlib import Path

import pytest

import known_failure_ledger as ledger


def _write(tmp_path, rows):
    path = tmp_path / "KNOWN_FAILURES.md"
    path.write_text(
        "before\n" + ledger.START + "\n```jsonl\n" + "\n".join(rows)
        + "\n```\n" + ledger.END + "\nafter\n",
        encoding="utf-8")
    return path


def test_ledger_reads_exact_node_reason_and_owner(tmp_path):
    path = _write(tmp_path, [
        '{"nodeid":"tests/test_a.py::test_x[p|q]",'
        '"reason":"measured mismatch","owner":"GW"}'
    ])
    assert ledger.read_known_xfails(path) == {
        "tests/test_a.py::test_x[p|q]": {
            "reason": "measured mismatch", "owner": "GW"}}


@pytest.mark.parametrize("rows,match", [
    (["not json"], "malformed ledger row"),
    ([('{"nodeid":"tests/test_a.py::test_x","reason":"r",'
       '"owner":"o","extra":1}')], "must contain exactly"),
    ([('{"nodeid":"tests/test_a.py::test_x","reason":"r","owner":"o"}'),
      ('{"nodeid":"tests/test_a.py::test_x","reason":"r","owner":"o"}')],
     "duplicate ledger node id"),
])
def test_ledger_refuses_malformed_or_duplicate_rows(tmp_path, rows, match):
    with pytest.raises(ValueError, match=match):
        ledger.read_known_xfails(_write(tmp_path, rows))


def test_complete_tests_selection_is_exact(tmp_path):
    root = tmp_path / "tests"
    root.mkdir()
    assert ledger.is_complete_tests_selection([str(root)], root)
    assert not ledger.is_complete_tests_selection([], root)
    assert not ledger.is_complete_tests_selection(
        [str(root / "test_one.py")], root)


def test_stale_ledger_name_is_a_collection_error_input():
    entries = {
        "tests/test_live.py::test_live": {},
        "tests/test_removed.py::test_gone": {},
    }
    assert ledger.missing_nodeids(
        entries, ["tests/test_live.py::test_live"]) == [
            "tests/test_removed.py::test_gone"]

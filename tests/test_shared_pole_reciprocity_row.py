"""The model-reciprocity receipt row must never record PASS for a comparison it did not make.

``shared_pole_reciprocity`` is a CONDITIONAL predicate: it compares the model's
transpose symmetry against the held reference's only where that reference has
the symmetry (``applicable``), and returns ``passed=True`` on a record it did
not evaluate, which is correct for its own contract and is asserted by
``tests/test_shared_pole_local_checks.py``. The constructor then wrote
``passed=True`` as a LITERAL into the receipt, so a run whose reference was
never symmetric enough to compare recorded a PASS having compared nothing.

Measured on the Si reference deck (SCGRAM-A, 2026-09-16): at q=0 the held
reference's own transpose defect is ~5e-08 against a ``reference_relative_max``
of 1e-12, so the row was unevaluated on every map of both accelerators while
reading as PASS. INVARIANTS 23: an absent measurement is never PASS.

Pure host Python -- no JAX, no mesh, no device. Run it directly or under pytest.
"""
import os
import sys

if __name__ == "__main__" or "gw" not in sys.modules:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "src"))

from gw.shared_pole_recipe import (  # noqa: E402
    gate_receipt, reciprocity_row_verdict, shared_real_pole_gates_v1_r3b,
)


def _row(measured):
    """The receipt row the constructor now writes for this predicate."""
    return gate_receipt("model_reciprocity", value=measured,
                        passed=reciprocity_row_verdict(measured),
                        reason="unit test")


def test_nothing_evaluated_is_not_measured():
    # Every record inapplicable: the predicate compared nothing.
    measured = {"passed": [True, True, True, True],
                "applicable": [False, False, False, False],
                "reference_relative": [5.4e-08, 4.1e-08, 9.8e-09, 9.1e-09],
                "model_relative": [1.0e-12, 1.0e-12, 1.0e-12, 1.0e-12]}
    assert reciprocity_row_verdict(measured) is None
    assert _row(measured)["status"] == "NOT_MEASURED"

    # NEGATIVE CONTROL: the literal the constructor used to write would have
    # recorded a PASS for this very same input. If this assertion ever fails,
    # the regression it guards has come back.
    assert gate_receipt("model_reciprocity", value=measured, passed=True,
                        reason="the old literal")["status"] == "PASS"


def test_evaluated_and_passing_is_pass():
    measured = {"passed": [True, True], "applicable": [True, True],
                "reference_relative": [1e-13, 1e-14], "model_relative": [1e-12, 1e-13]}
    assert reciprocity_row_verdict(measured) is True
    assert _row(measured)["status"] == "PASS"


def test_an_evaluated_failure_still_fails():
    # A real failure must not be hidden by the new None branch.
    measured = {"passed": [True, False], "applicable": [True, True],
                "reference_relative": [1e-13, 1e-14], "model_relative": [1e-12, 1e-3]}
    assert reciprocity_row_verdict(measured) is False
    assert _row(measured)["status"] == "FAIL"


def test_a_failure_on_an_inapplicable_record_is_ignored():
    # Only evaluated records carry the verdict.
    measured = {"passed": [True, False], "applicable": [True, False],
                "reference_relative": [1e-13, 5e-08], "model_relative": [1e-12, 1e-3]}
    assert reciprocity_row_verdict(measured) is True


def test_nested_per_sample_and_per_field_records():
    # The batched path hands back [field][sample] nested lists.
    measured = {"passed": [[True, True], [True, True]],
                "applicable": [[False, False], [False, False]],
                "reference_relative": [[5e-08, 4e-08], [1e-07, 2e-07]],
                "model_relative": [[1e-12, 1e-12], [1e-12, 1e-12]]}
    assert reciprocity_row_verdict(measured) is None
    measured["applicable"] = [[True, True], [True, False]]
    assert reciprocity_row_verdict(measured) is True


def test_degenerate_inputs_are_not_measured():
    assert reciprocity_row_verdict({}) is None
    assert reciprocity_row_verdict({"passed": [], "applicable": []}) is None
    # Mismatched records cannot be adjudicated.
    assert reciprocity_row_verdict({"passed": [True], "applicable": [True, True]}) is None


def test_the_ordered_route_row_is_unchanged():
    # Time-reversal-broken data carry no transpose symmetry; that row already
    # recorded NOT_MEASURED and must stay that way.
    row = gate_receipt("model_reciprocity", value=None, passed=None,
                       reason="not applicable: time-reversal-broken samples")
    assert row["status"] == "NOT_MEASURED"
    assert shared_real_pole_gates_v1_r3b["model_reciprocity"]["threshold"][
        "reference_relative_max"] == 1.0e-12


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc!r}")
    print(f"\n{'ALL PASS' if not failures else str(failures) + ' FAILURES'}")
    raise SystemExit(1 if failures else 0)

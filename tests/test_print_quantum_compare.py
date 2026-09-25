"""``harness.print_quantum_diff``: one printed digit passes, a pad defect fails.

The bispinor μ-pad cell compares ``sigma_diag`` at one unit in the last
printed digit per token (owner, 2026-09-25).  These cells are its negative
controls: they build the failure the gate exists for and confirm it fires.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import print_quantum_diff  # noqa: E402

_REF = """\
k-point 0:
n=0   sigX=  -37.083116              sigC=    1.777920+  0.007256i  sigTT=    0.000000              Z=    1.000000  Z_status=OK
n=1   sigX=  -30.569827              sigC=    1.900644+  0.010451i  sigTT=    0.004195              Z=    0.912345  Z_status=OK
# kcrys 0.000000000 0.333333333 0.000000000
"""


def _swap(text, old, new):
    assert old in text
    return text.replace(old, new, 1)


def test_identical_text_scores_zero():
    d = print_quantum_diff(_REF, _REF)
    assert d["skeleton_equal"] and d["max_quanta"] == 0.0
    assert d["moved_tokens"] == 0 and d["n_tokens"] > 0


def test_one_last_digit_flip_is_one_quantum():
    out = _swap(_REF, "1.777920+", "1.777921+")
    out = _swap(out, "0.912345", "0.912344")
    d = print_quantum_diff(_REF, out)
    assert d["skeleton_equal"]
    assert d["moved_tokens"] == 2
    assert 0.99 < d["max_quanta"] < 1.01


def test_a_sign_entering_a_zero_field_is_a_number_not_structure():
    d = print_quantum_diff(_REF, _swap(_REF, "    0.000000  ", "   -0.000000  "))
    assert d["skeleton_equal"] and d["max_quanta"] == 0.0


def test_two_quanta_fail_the_gate_bound():
    d = print_quantum_diff(_REF, _swap(_REF, "-37.083116", "-37.083118"))
    assert 1.99 < d["max_quanta"] < 2.01 and d["max_quanta"] > 1.5


def test_a_whole_row_pad_defect_fails():
    # The class the μ-pad cell pins: a pad-extent fault moves every value of
    # a row (MoS2 668->672 once moved a Σ^B tile by ~118 eV).
    bad = _REF.replace(
        "n=1   sigX=  -30.569827              sigC=    1.900644+  0.010451i  "
        "sigTT=    0.004195",
        "n=1   sigX= -148.469827              sigC=    3.100644+  0.910451i  "
        "sigTT=    0.504195")
    assert bad != _REF
    d = print_quantum_diff(_REF, bad)
    assert d["skeleton_equal"]
    assert d["moved_tokens"] == 4
    assert d["max_quanta"] > 1e6
    assert "n=1" in d["worst_line"]


def test_an_index_change_is_never_absorbed():
    d = print_quantum_diff(_REF, _swap(_REF, "n=1 ", "n=2 "))
    assert d["max_quanta"] == float("inf")


def test_structure_changes_fail_the_skeleton():
    assert not print_quantum_diff(
        _REF, _swap(_REF, "Z_status=OK", "Z_status=PATHOLOGICAL_EQP1_FALLBACK")
    )["skeleton_equal"]
    dropped = "\n".join(_REF.splitlines()[:-2]) + "\n"
    assert not print_quantum_diff(_REF, dropped)["skeleton_equal"]

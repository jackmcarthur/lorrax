"""Construction receipt verdicts are computed from measurements, never asserted.

Two rows used to write ``passed=True`` as a literal: ``representation`` (so a deck that
violated the row still recorded PASS) and, on the ordered route, ``retained_subspace_moments``
(so the spurious-pole case that the row exists to catch -- m3 own-norm 0.632 from 100-1000 Ry
poles -- would have recorded PASS on a production run). Both verdicts now come from the
measured values against the table.
"""
import pytest

from gw.shared_pole_recipe import (representation_row_passed, retained_moment_row_passed,
                                   shared_real_pole_gates_ordered_v1 as ORDERED,
                                   shared_real_pole_gates_v1_r3b as TRS)


@pytest.mark.parametrize("table,measured,expected", [
    # The stored operator is the spin-traced mu x mu charge response, so one and two
    # component decks both pass; anything else does not.
    (TRS, {"nspinor": 1, "trs_allowed": True}, True),
    (TRS, {"nspinor": 2, "trs_allowed": True}, True),
    (TRS, {"nspinor": 4, "trs_allowed": True}, False),
    (TRS, {"nspinor": 1, "trs_allowed": False}, False),
    (ORDERED, {"nspinor": 2, "trs_allowed": False, "ordered": True}, True),
    (ORDERED, {"nspinor": 1, "trs_allowed": True, "ordered": True}, False),
    (ORDERED, {"nspinor": 1, "trs_allowed": False, "ordered": False}, False),
])
def test_representation_verdict_compares_against_the_threshold(table, measured, expected):
    assert representation_row_passed(measured, table["representation"]["threshold"]) is expected


def test_representation_threshold_admits_the_two_component_charge_operator():
    for table in (TRS, ORDERED):
        assert table["representation"]["threshold"]["nspinor"] == (1, 2)


def test_ordered_retained_moment_verdict_follows_the_measured_defects():
    row = ORDERED["retained_subspace_moments"]
    band = row["calibration_range"]
    assert row["diagnostic"] is True and band[0] < band[1]
    # Measured CrI3 q=1 construction (claim 2357 comparison).
    assert retained_moment_row_passed(
        {"m0": 4.8e-9, "m1": 6.8e-11, "m2": 1.2e-5, "m3": 1.3e-7}, row) is True
    # The hand-caught spurious-pole case this row exists for.
    assert retained_moment_row_passed(
        {"m0": 1e-9, "m1": 1e-11, "m2": 1e-6, "m3": 0.632}, row) is False
    # Nothing measured is not a pass.
    assert retained_moment_row_passed({}, row) is None


def test_trs_retained_moment_verdict_uses_the_exact_identity_threshold():
    row = TRS["retained_subspace_moments"]
    assert "calibration_range" not in row
    assert retained_moment_row_passed({"m1": [1e-12], "m3": [2e-12]}, row) is True
    assert retained_moment_row_passed({"m1": [1e-12], "m3": [5e-9]}, row) is False

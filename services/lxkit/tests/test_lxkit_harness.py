"""``lxkit.testing`` — the harness, tested by the harness.

The autouse ``gate_state`` fixture, the ABSENT/BROKEN arm, the device skip,
the hostile-geometry table, and the two deliberately-unimplemented
skip-honesty entry points.
"""

from __future__ import annotations

import pytest

from lxkit.gate import announce_once
from lxkit.testing import (
    ABSENT, BROKEN, OK, absent_or_broken, hostile_extents, require_devices,
)


# ---------------------------------------------------------------------------
# gate_state — the fix for survey surprise #1
# ---------------------------------------------------------------------------
# `_ANNOUNCED` is process-global and `reset_gate_state()` had ZERO callers at
# 96a6399, so every announcement-asserting cell in the tree was
# order-dependent: whichever ran first burned the key and the rest passed by
# observing silence.  These two cells assert the SAME key and both must see
# the announcement.  Delete the autouse fixture and exactly one of them
# fails -- whichever pytest happens to run second.

def test_an_announcement_is_visible_in_this_cell(capsys):
    assert announce_once(("shared-key",), "the receipt") is True
    assert "the receipt" in capsys.readouterr().out


def test_the_same_announcement_is_visible_in_the_next_cell_too(capsys):
    assert announce_once(("shared-key",), "the receipt") is True
    assert "the receipt" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# ABSENT is a skip.  BUILT-AND-BROKEN is a failure.
# ---------------------------------------------------------------------------

def test_ok_leaves_the_cell_alone():
    def cell():
        return "ran"
    assert absent_or_broken(OK, "")(cell) is cell


def test_absent_marks_the_cell_skipped():
    def cell():
        raise AssertionError("must not run")
    marks = absent_or_broken(ABSENT, "not built here")(cell).pytestmark
    assert any(m.name == "skip" and m.kwargs["reason"] == "not built here"
               for m in marks)


def test_broken_FAILS_the_cell_quoting_the_loader_reason():
    """The 2026-08-06 loss: 19 cells reported "skipped" for a .so that was
    sitting right there and would not dlopen.  A built library is an
    explicit request; an explicit request that cannot be honored refuses."""
    reason = "liblorrax_ffi_host.so: libfftw3.so.mpi31.3: not found"

    @absent_or_broken(BROKEN, reason)
    def cell():
        raise AssertionError("must not run")

    with pytest.raises(pytest.fail.Exception) as ei:
        cell()
    assert reason in str(ei.value)


def test_broken_does_not_quietly_become_a_skip():
    """The red twin of the arm above: BROKEN must NOT carry a skip mark."""
    decorated = absent_or_broken(BROKEN, "why")(lambda: None)
    assert not any(m.name == "skip"
                   for m in getattr(decorated, "pytestmark", ()))


def test_an_unrecognized_state_refuses_at_decoration_time():
    """A typo'd state that quietly meant OK is the same hole in a different
    wall."""
    with pytest.raises(ValueError, match="not one of"):
        absent_or_broken("aboslutely fine", "typo")


# ---------------------------------------------------------------------------
# require_devices — SKIPS, never asserts
# ---------------------------------------------------------------------------

def test_require_devices_skips_rather_than_failing():
    """KNOWN_FAILURES.md records 11 cells (test_contract_bands 9 +
    test_projection_lgemm 2) that FAILED the bare 1-device leg purely
    because they wrote `assert n_dev >= 4`."""
    with pytest.raises(pytest.skip.Exception) as ei:
        require_devices(1_000_000)
    assert "xla_force_host_platform_device_count" in str(ei.value)


def test_require_devices_returns_the_count_when_it_is_enough():
    pytest.importorskip("jax")
    assert require_devices(1) >= 1


# ---------------------------------------------------------------------------
# Hostile geometry
# ---------------------------------------------------------------------------

def test_the_4x4_table_is_the_one_measured_on_job_56389339():
    """The generalization is checkable precisely because it reproduces the
    measured rows: 4 nodes / 16 ranks, 4x4 mesh, each case written from a
    padded sharded operand and read back against a single-rank reference."""
    got = {(g.logical, g.padded) for g in hostile_extents((4, 4))}
    assert got == {((17, 23), (20, 24)),
                   ((13, 17), (16, 20)),
                   ((17, 16), (20, 16)),
                   ((1, 1), (4, 4)),
                   ((2, 16), (4, 16))}


@pytest.mark.parametrize("mesh_shape", [(1, 1), (2, 2), (4, 4), (2, 8),
                                        (4, 1), (3, 5)])
def test_every_padded_shape_divides_the_mesh_and_covers_its_logical(
        mesh_shape):
    for g in hostile_extents(mesh_shape):
        assert len(g.logical) == len(g.padded) == 2
        for lg, pd, m in zip(g.logical, g.padded, mesh_shape):
            assert pd % m == 0, (g.name, mesh_shape)
            assert pd >= lg and pd - lg < m, (g.name, mesh_shape)


@pytest.mark.parametrize("mesh_shape", [(2, 2), (4, 4), (2, 8), (3, 5)])
def test_the_geometry_is_actually_hostile(mesh_shape):
    """A 'hostile' table whose extents all divide the mesh would be the
    tautology this helper exists to prevent."""
    rows = hostile_extents(mesh_shape)
    assert any(g.logical != g.padded for g in rows)
    assert any(g.logical[0] < mesh_shape[0] for g in rows), "empty tiles"


def test_hostile_extents_refuses_a_mesh_it_cannot_describe():
    for bad in [(4,), (2, 2, 2), (0, 4), (4, -1)]:
        with pytest.raises(ValueError, match="2-axis mesh"):
            hostile_extents(bad)


def test_the_rows_are_named_so_a_failure_says_which_geometry():
    names = [g.name for g in hostile_extents((4, 4))]
    assert len(set(names)) == len(names)
    assert "fewer-slices-than-ranks" in names


# ---------------------------------------------------------------------------
# Skip-honesty lives in its own file
# ---------------------------------------------------------------------------
# The stubs this file used to pin (``machine_profile`` /
# ``assert_skips_match_profile`` raising NotImplementedError, with "a
# permissive default would turn the gate green while measuring nothing" as
# the reason) are IMPLEMENTED as of distrib_la step 2, from real Perlmutter
# probe results.  Their tests -- including both red twins and the
# session-level wiring -- are test_lxkit_machine_profile.py.

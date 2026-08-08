"""The skip-honesty gate, symmetry_maps' half.

The charter: each machine declares a profile, and one gate per service
asserts that reality matches it.  An unexpected skip is a FAILURE, because
a skip reads as "not applicable on this machine" and this tree has the
receipt for what that costs — 2026-08-06, nineteen ScaLAPACK/SLATE contract
cells reported as "19 skipped" beside "0 failed", suite green, coverage
gone.

The mechanism is :mod:`lxkit.testing`; what lives HERE is the part only
this service can supply.  For distrib_la that was an FFI probe.  This
service has no ``.so`` at all, so its MUST rows name CAPABILITIES — the
four in-tree WFN decks, ``cohsex_debug/WFNsmall.h5`` specifically, h5py,
and four FORCEABLE host devices — and ``_profiles.probe_capability`` is
the probe.  The vocabulary is the service's; the policy stays lxkit's.

WHY THESE FOUR AND NOT A LONGER LIST.  Each one, if absent, silently
deletes a whole tier while the suite stays green:

===========================  ================================================
row                          what evaporates when it is missing
===========================  ================================================
``decks:all-four``           the L-a+ tier, INCLUDING the bit-equality
                             acceptance gate on ``(irr_idx_k, sym_idx_k)``
                             — the extraction's whole criterion and the
                             §8.1 op-selection tripwire
``file:cohsex/WFNsmall.h5``  the I5/I6 discriminator (the only in-tree deck
                             with a TRS-first star) AND lorrax's own four
                             ``test_density_symmetry_check`` fixture cells
``module:h5py``              every deck cell at once, as an importorskip
``devices:4-forceable``      the L-b emulated-mesh tier:
``unfold_isdf_operator``'s
                             two ``all_to_all`` branches are unreachable at
                             1x1, so G10 and G6 have nowhere else to run
===========================  ================================================

MEASURED on the dev box, 2026-08-07::

    WSL_DISTRO_NAME set, /proc/version contains 'microsoft'
    detect_machine() -> 'unknown'  ->  profile_for_this_machine() -> wsl
    profile wsl               asserts_nothing=False  min_collected=40
    decks             decks:all-four              -> available
    density_fixture   file:cohsex_debug/WFNsmall.h5 -> available
    h5py              module:h5py                 -> available
    emulated_devices  devices:4-forceable         -> available (4 cpu devices)

THE NEGATIVE HALF — every skip this session emitted, diffed against the
allowlist, plus the minimum-collected floor — is armed for the whole suite
in ``conftest.py`` and runs at ``sessionfinish``, so it needs no cell here.
What DOES need a cell is the allowlist's own shape: the rows are what
decide whether a skip is coverage or an excuse, and two of them
(``no lorrax file_io``, ``h5py is not importable``) are broad enough to
swallow a real regression if they were ever widened.  Those are pinned
below, in both directions.
"""

from __future__ import annotations

import os

import pytest
from lxkit.testing import (AllowedSkip, MachineProfile, MustRow,
                           ObservedSkip, assert_must_probe,
                           assert_skips_match_profile, machine_profile)

from _deck_stub import DECKS, deck_available, deck_path
from _profiles import (allowed_skips_for, is_wsl, probe_capability,
                       profile_for_this_machine)

#: The probes this service can answer for.  A row whose service is not in
#: here comes back as UNASSERTED, which is the point of the mapping.
_PROBES = {"symmetry_maps": probe_capability}

_TESTS = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# THE POSITIVE HALF
# ---------------------------------------------------------------------------

def test_this_machines_declared_capabilities_are_actually_there():
    """THE GATE.  On a machine that promises decks and devices, they exist.

    A failure quotes the probe's own reason, because the four rungs have
    four different fixes: a missing blob is a checkout made without the
    fixtures, a missing h5py is an environment, a machine that cannot force
    four host devices is one the L-b tier has never run on, and an
    unrecognised target is a typo in the table.  "unavailable" names none
    of them.
    """
    profile = profile_for_this_machine()
    if profile.asserts_nothing:
        pytest.skip(
            f"machine {profile.machine!r} declares no profile, so there is "
            f"nothing to assert — covered by the wsl leg (this dev box, "
            f"detected from /proc/version) and by the Perlmutter leg, where "
            f"NERSC_HOST selects the row that carries these same four "
            f"capability MUST rows on top of lxkit's backend ones")
    gaps = assert_must_probe(profile, _PROBES, service="symmetry_maps")
    assert gaps == [], (
        f"service='symmetry_maps' filtered the rows, so nothing should have "
        f"been unassertable; got {gaps}")


def test_the_wsl_profile_is_the_one_this_dev_box_selects():
    """Detection, asserted rather than assumed.

    ``detect_machine()`` answers ``unknown`` here — WSL sets no
    ``LX_MACHINE`` and no ``NERSC_HOST`` — so the whole gate would be the
    asserts-nothing row if ``profile_for_this_machine`` did not fall
    through to the kernel string.  That fall-through is the difference
    between this suite promising four things and promising none, which is
    worth one assertion.
    """
    if not is_wsl():
        pytest.skip("not a WSL kernel; the wsl row is covered by the dev-box "
                    "leg, and this machine's own row is asserted above")
    prof = profile_for_this_machine()
    assert prof.machine == "wsl" and not prof.asserts_nothing
    assert {r.name for r in prof.must} == {
        "decks", "density_fixture", "h5py", "emulated_devices"}


def test_the_perlmutter_row_keeps_lxkits_backends_and_gains_ours():
    """lxkit is read-only this wave, so the rows are ADDED, not edited.

    Both halves matter.  Dropping lxkit's four would quietly stop
    Perlmutter promising ScaLAPACK; failing to add ours would make the
    Perlmutter leg promise less than the dev box, which is backwards.
    ``must_for`` filtering is what keeps the two services out of each
    other's failures.
    """
    prof = profile_for_this_machine("perlmutter")
    names = {r.name for r in prof.must}
    assert names >= {"scalapack", "slate", "cusolvermp", "phdf5"}
    assert names >= {"decks", "density_fixture", "h5py", "emulated_devices"}
    assert {r.name for r in prof.must_for("symmetry_maps")} == {
        "decks", "density_fixture", "h5py", "emulated_devices"}
    # distrib_la's own gate reads lxkit's table directly and must be
    # untouched by what we added on top of our copy.
    assert {r.name for r in machine_profile("perlmutter").must_for(
        "distrib_la")} == {"scalapack", "slate", "cusolvermp"}


def test_the_capability_gate_can_fail():
    """RED TWIN, with THIS service's real probe.

    lxkit's own twin uses a stub, which proves the reporting.  This proves
    the WIRING: a MUST row naming a capability the probe has never heard of
    must make the real ``probe_capability`` say no, and that no must reach
    the failure with its reason intact.  Without this cell a mis-plumbed
    ``_PROBES`` mapping would make the gate above pass on any machine by
    asserting nothing at all.
    """
    fake = MachineProfile(
        machine="wsl",
        must=(MustRow("ghost", "symmetry_maps", "decks:no-such-thing", "cpu"),),
        allowed_skips=(), min_collected=1)
    # ``pytest.fail`` raises ``Failed``, which derives from BaseException
    # and NOT from Exception — ``pytest.raises(Exception)`` does not catch
    # it, and a twin written that way ERRORS instead of passing.
    with pytest.raises(pytest.fail.Exception) as ei:
        assert_must_probe(fake, _PROBES, service="symmetry_maps")
    msg = str(ei.value)
    assert "decks:no-such-thing" in msg and "DEFECT" in msg
    assert "not a capability" in msg, msg


def test_each_probe_rung_says_something_different():
    """Four capabilities, four reasons — checked one rung at a time.

    A probe whose failure text was the same sentence four times would pass
    the twin above and still be useless: the reason is the only part of the
    message a person acts on, and "missing" does not say whether to fetch
    a fixture, pip-install, or find another machine.
    """
    ok, why = probe_capability("decks:all-four", "in-tree")
    assert ok, why
    ok, why = probe_capability("file:cohsex_debug/no_such_file.h5", "in-tree")
    assert not ok and "test_density_symmetry_check" in why
    ok, why = probe_capability("module:no_such_module_xyz", "any")
    assert not ok and "not a dependency of the symmetry_maps PACKAGE" in why
    ok, why = probe_capability("devices:4-forceable", "cpu")
    assert ok, why


# ---------------------------------------------------------------------------
# THE MUST-RUN ROWS — a skip HERE would be lost coverage
# ---------------------------------------------------------------------------

def test_the_deck_cells_have_no_excuse_to_skip_on_this_machine():
    """MUST-RUN, deck tier.  The fixtures are IN-TREE, so nothing gates them.

    ``_deck._deck()`` skips on two conditions and both are supposed to be
    impossible in this checkout.  Asserted against the allowlist as well as
    against the filesystem: the row that would EXCUSE a fixture-absent skip
    must not exist, or a future "make the suite green on CI" edit could add
    one and delete this tier without turning anything red.
    """
    prof = profile_for_this_machine()
    if prof.asserts_nothing:
        pytest.skip(f"{prof.machine!r} promises nothing; covered by the wsl "
                    f"and perlmutter legs")
    for deck in DECKS:
        assert deck_available(deck), (
            f"{deck_path(deck)} is absent; the deck tier would skip and the "
            f"bit-equality acceptance gate with it")
    allowed = tuple(prof.allowed_skips) + tuple(allowed_skips_for(prof))
    fixture_absent = ObservedSkip(
        "services/symmetry_maps/tests/test_symmetry_maps_deck_tables.py"
        "::test_the_deck_tables_are_bit_identical_to_the_committed_derivation",
        "no cohsex_debug/WFNsmall.h5 in this checkout (fixture blobs absent)")
    assert_skips_match_profile.__doc__          # (documents the mechanism)
    with pytest.raises(AssertionError, match="skip-honesty"):
        assert_skips_match_profile([fixture_absent], prof, collected=999,
                                   extra_allowed=allowed)


def test_the_wfn_loader_parity_arm_must_run_when_lorrax_is_present():
    """MUST-RUN, parity arm.  The allowlist row is for STANDALONE only.

    ``AllowedSkip("deck_tables", "no lorrax file_io", ...)`` exists because
    a standalone install of this service has no lorrax to import — that is
    the whole point of the quarantine.  It must not become an excuse in the
    monorepo run, where the arm is the ONE cell connecting the service's
    stub tier to the loader the tree actually feeds ``SymMaps``.  So: with
    a lorrax ``src/`` next door, ``from file_io import WfnLoader`` has to
    work, and this asserts it directly rather than trusting that the arm
    did not skip.
    """
    repo = os.path.dirname(os.path.dirname(os.path.dirname(_TESTS)))
    if not os.path.isdir(os.path.join(repo, "src", "file_io")):
        pytest.skip("no lorrax src/ next to this service (standalone "
                    "install); the parity arm is inapplicable by "
                    "construction and the stub arm is the standalone claim")
    from file_io import WfnLoader                              # noqa: F401


def test_the_emulated_mesh_tier_is_runnable_on_this_machine():
    """MUST-RUN, L-b.  Not "did it run" — "can it run anywhere here".

    The L-b cells legitimately skip in the full monorepo run (jax is
    already imported, the XLA flag is inert, ``require_devices`` skips).
    What must NOT be true is that they skip in EVERY invocation, because
    then the universal ``needs >= 4 devices`` allowlist row would be
    pointing at a covering leg that does not exist on this machine.  The
    probe answers that in a child process, so the claim holds whichever leg
    is asking.
    """
    ok, why = probe_capability("devices:4-forceable", "cpu")
    assert ok, why


# ---------------------------------------------------------------------------
# THE NEGATIVE HALF — the allowlist's own shape, both directions
# ---------------------------------------------------------------------------

def test_every_allowed_skip_names_the_leg_that_covers_it():
    """``covered_by`` is required, not decorative.

    A skip nobody can point at a covering run for is evaporated coverage
    wearing a green dot, and the allowlist is where that claim is written
    down.  A row with an empty or one-word ``covered_by`` is the shape that
    turns the gate into a rubber stamp.
    """
    rows = allowed_skips_for(profile_for_this_machine("wsl"))
    assert rows, "the wsl profile allows nothing; that cannot be right"
    for row in rows:
        assert isinstance(row, AllowedSkip)
        assert len(row.covered_by) > 20, (
            f"AllowedSkip({row.where!r}, {row.why!r}) does not name a leg: "
            f"{row.covered_by!r}")


def test_an_unlisted_skip_fails_the_session():
    """NEGATIVE HALF, shown firing.

    The reason string is one a real regression would produce — the deck
    tier going quiet because ``SymMaps`` grew a twelfth attribute the stub
    does not supply.  Nothing in the allowlist covers it, and nothing
    should.
    """
    prof = profile_for_this_machine("wsl")
    rogue = ObservedSkip(
        "services/symmetry_maps/tests/test_symmetry_maps_deck_tables.py"
        "::test_the_stub_supplies_exactly_what_symmaps_reads",
        "stub is missing an attribute SymMaps now reads")
    with pytest.raises(AssertionError, match="unexpected skip is a FAILURE"):
        assert_skips_match_profile([rogue], prof, collected=999,
                                   extra_allowed=allowed_skips_for(prof))


def test_a_listed_skip_passes_the_session():
    """The other direction: the two rows that ARE legitimate.

    Without this the cell above would be satisfied by an allowlist that
    matched nothing at all, and every honest standalone skip would fail the
    session — which is the failure mode that gets a gate disarmed.
    """
    prof = profile_for_this_machine("wsl")
    legit = [
        ObservedSkip("test_symmetry_maps_deck_tables.py::test_the_stub_and_"
                     "the_production_loader_build_the_same_tables[gnppm_debug]",
                     "no lorrax file_io on this path (ModuleNotFoundError); "
                     "the stub arm above is the standalone claim"),
        ObservedSkip("test_symmetry_maps_algebra.py::test_the_cartesian_"
                     "metric_convention_is_discriminated_by_bdot",
                     "h5py is not importable"),
        # VERBATIM from lxkit.testing.require_devices(4, platform="cpu") —
        # the reason the L-b tier emits in the full monorepo run, where the
        # XLA flag is inert by design.  Typed out rather than paraphrased
        # because the allowlist matches on a SUBSTRING and a paraphrase
        # would pass here while the real skip failed the session.
        ObservedSkip("test_symmetry_maps_multiproc.py::test_the_check_body_"
                     "passes_on_an_emulated_2x2[star_sharding]",
                     "needs >= 4 devices on platform 'cpu', have 1.  Set "
                     "XLA_FLAGS=--xla_force_host_platform_device_count=4 "
                     "BEFORE the first jax import, or run the real "
                     "multi-process leg"),
    ]
    assert_skips_match_profile(legit, prof, collected=999,
                               extra_allowed=allowed_skips_for(prof))


def test_the_device_skip_reason_is_quoted_from_the_source_not_paraphrased(
        monkeypatch):
    """RED TWIN for the cell above, and the reason it types the string out.

    The allowlist matches a SUBSTRING of the reason, so a test that invents
    a plausible-looking reason can pass while the real one fails the
    session.  That is not hypothetical: on 2026-08-07 the ``wsl`` row was
    built as ``_ALLOWED`` alone, inheriting none of lxkit's
    ``_UNIVERSAL_SKIPS``, and the eight emulated-mesh cells' own skips
    would have failed the gate in the full monorepo run.  Nothing caught it
    until the string came from ``require_devices`` itself.

    ``jax.devices`` is monkeypatched to report one device rather than
    branching on what this process has: the cell must produce the SAME
    evidence on the four-device standalone leg and the one-device monorepo
    leg, and a cell that skipped on the machine where the flag worked would
    be the exact failure this file is about.
    """
    import jax
    from lxkit.testing import require_devices

    monkeypatch.setattr(jax, "devices", lambda *a, **k: [object()])
    # ``pytest.skip`` raises ``Skipped``, which derives from BaseException
    # and NOT from Exception — ``pytest.raises(Exception)`` lets it through
    # and the cell SKIPS instead of passing, which is both wrong and, in a
    # file about unexplained skips, ironic.  Measured writing this cell:
    # the escaped Skipped was itself an unlisted skip and failed the gate.
    with pytest.raises(pytest.skip.Exception) as ei:
        require_devices(4, platform="cpu")
    reason = str(getattr(ei.value, "msg", ei.value))
    assert "needs >= 4 devices" in reason, reason
    prof = profile_for_this_machine("wsl")
    rows = tuple(prof.allowed_skips) + tuple(allowed_skips_for(prof))
    assert any(r.why and r.why in reason for r in rows), (
        f"nothing in the wsl allowlist matches the reason require_devices "
        f"actually emits: {reason!r}.  Allowed: "
        f"{[(r.where or '*', r.why or '*') for r in rows]}")


def test_the_minimum_collected_floor_is_armed():
    """THE FLOOR, both directions.

    Both halves above are vacuous in a session that collected nothing: a
    38-byte junitxml parses as zero tests and zero failures, and pytest
    rc=5 is not a pass.  The floor is checked FIRST inside
    ``assert_skips_match_profile`` for exactly that reason, and it is set
    just under the L-a tier's size so a collection error that dropped a
    module trips it instead of reading as a smaller suite.
    """
    prof = profile_for_this_machine("wsl")
    assert prof.min_collected >= 40
    with pytest.raises(AssertionError, match="below the floor"):
        assert_skips_match_profile([], prof, collected=3)
    assert_skips_match_profile([], prof, collected=prof.min_collected)


def test_the_unknown_machine_asserts_nothing_and_says_so():
    """The explicit nothing-promised row, kept explicit.

    A machine nobody has measured must not be handed the wsl promises by
    accident — a fall-through that silently adopted another machine's
    profile would fail legs for reasons that have nothing to do with them.
    ``asserts_nothing`` is a decision in the table, and this is the cell
    that keeps it one.
    """
    prof = profile_for_this_machine("a-machine-that-does-not-exist")
    assert prof.asserts_nothing and prof.must == ()
    assert allowed_skips_for(prof) == ()
    # ...and it still enforces the FLOOR, which is not a promise about the
    # machine but about the session having happened at all.
    with pytest.raises(AssertionError, match="below the floor"):
        assert_skips_match_profile([], prof, collected=0)

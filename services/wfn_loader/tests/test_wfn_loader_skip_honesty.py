"""The skip-honesty gate, wfn_loader's half — SERVICE-LOCAL by adjudication.

The charter: each machine declares an expected-backend profile, and one
gate per service asserts that reality matches it.  An unexpected skip is a
FAILURE, because a skip reads as "not applicable on this machine" and this
tree has the receipt for what that costs — 2026-08-06, nineteen
ScaLAPACK/SLATE contract cells reported as "19 skipped" beside "0 failed",
suite green, coverage gone.

WHY THIS FILE IS NOT ``arm_skip_honesty(...)`` (DESIGN.md DECISION 4,
step-1a adjudication (ii), VERIFIED at ``lxkit/testing.py:801``):
``arm_skip_honesty`` writes a PROCESS-GLOBAL ``_ARMED`` dict, so a second
caller does not add a second gate — it REPLACES the first one's scope.  In
a full-suite run the conftests load in directory order, so a call from
``services/wfn_loader/tests/conftest.py`` would land after
``services/distrib_la/tests/conftest.py`` and silently disarm the gate
that found the LORRAX_FFI_HOST_SO restore defect on 2026-08-07.  lxkit is
FROZEN this wave (ruling 3), so per-scope arming is REGISTERED to the main
Fable and this service carries its own gate instead.

WHAT IS BORROWED AND WHAT IS LOCAL.  lxkit's machine-profile VOCABULARY is
consumed READ-ONLY: :class:`~lxkit.testing.MachineProfile`,
:class:`~lxkit.testing.MustRow`, :class:`~lxkit.testing.AllowedSkip`,
:func:`~lxkit.testing.machine_profile`,
:func:`~lxkit.testing.assert_must_probe` and
:func:`~lxkit.testing.assert_skips_match_profile` are pure data and pure
functions — calling them mutates nothing.  What is LOCAL is the
observation: ``conftest.pytest_runtest_logreport`` is THIS directory's own
hook writing THIS directory's own dict, so it neither double-counts
lxkit's recorder nor judges anybody else's skips.

THE ROW THIS SERVICE CLOSES.  Perlmutter's profile declares four MUST
backends and the fourth — ``phdf5``/``lorrax_phdf5_read`` — belongs to
``slab_io``.  distrib_la's gate reports it UNASSERTED, by name, because
distrib_la has no library for it (see
``services/distrib_la/tests/test_distrib_la_skip_honesty.py``).
wfn_loader is a CLIENT of that door — the whole point of DECISION 1's fold
is that only slab_io sees phdf5 — so it can answer for the row, and
:func:`test_this_service_closes_the_gap_distrib_la_reports` is where the
gap is written down as CLOSED rather than merely moved.

MEASURED, WSL 2026-08-07 (``LX_MACHINE`` unset, ``NERSC_HOST`` unset)::

    detect_machine() -> unknown       asserts_nothing=True
    probe lorrax_phdf5_read cpu       -> ABSENT ("could not be loaded")
    probe lorrax_phdf5_read_kchunk_union CUDA -> ABSENT (same)

so both halves are inert here and say so, which is the honest answer for a
laptop with no FFI build.  The Perlmutter leg is where they assert.
"""

from __future__ import annotations

import os

import pytest
from lxkit.testing import (AllowedSkip, MachineProfile, MustRow, ObservedSkip,
                           assert_must_probe, assert_skips_match_profile,
                           machine_profile)

_TESTS = os.path.dirname(os.path.abspath(__file__))

#: The read door's own target — the one ``WfnLoader._auto_pick_backend``
#: dispatches through ``file_io.slab_io.probe_read_availability`` after
#: the 2026-08-07 fold.  Distinct from the profile's ``lorrax_phdf5_read``
#: row, which names the single-slab handler: they ship in the SAME library
#: and the machine's promise is about the library, so both are asserted
#: and a machine where only one answers is reported as the partial build
#: it is.
UNION_TARGET = "lorrax_phdf5_read_kchunk_union"

# --- the three-way probe taxonomy, split into a two-way policy ------------
OK, ABSENT, BROKEN, GRAMMAR = "ok", "absent", "broken", "grammar"


def classify_probe(ok: bool, reason: str) -> str:
    """Map ``probe_target``'s three-way reason onto the ABSENT/BROKEN rule.

    The rule (charter): **ABSENT is an honest skip, BUILT-AND-BROKEN is a
    FAILURE.**  A library that is not on this machine says nothing about
    the code; a library that loaded and does not export the handler is a
    partial build, and skipping on it is how a broken build ships.

    ``probe_target`` already distinguishes the three cases and this maps
    them:

      * ``could not be loaded`` -> ABSENT   (missing .so, bad
        LD_LIBRARY_PATH, glibc mismatch — the handler may be perfectly
        well compiled, which is why this is not a build defect)
      * ``does not export``     -> BROKEN   (the genuine partial build)
      * ``unknown target`` / ``unknown FFI platform`` -> GRAMMAR (a typo
        in the caller, never a machine property)

    An unrecognized reason returns BROKEN, NOT ABSENT.  That default is
    the whole design: a reason nobody has classified must not quietly
    become "not applicable here", because that is the failure mode the
    gate exists to remove.  :func:`test_the_absent_versus_broken_split_can_fail`
    pins it.
    """
    if ok:
        return OK
    r = (reason or "").lower()
    if "unknown target" in r or "unknown ffi platform" in r:
        return GRAMMAR
    if "could not be loaded" in r or "could not locate" in r:
        return ABSENT
    if "does not export" in r:
        return BROKEN
    return BROKEN


def probe(target: str, platform: str) -> tuple:
    """``(ok, reason)`` for an FFI target, through lorrax's own loader.

    Deliberately the raw ``probe_target`` rather than
    ``slab_io.probe_read_availability``: the profile's MUST row names its
    own target, and a probe that ignored it and asked about a different
    handler would report PASS for a row nobody checked.
    """
    from ffi.common.ffi_loader import probe_target
    return probe_target(target, platform)


#: The probes this service can answer for.  ``slab_io`` is the door
#: wfn_loader reads through; a row whose service is not in here comes back
#: UNASSERTED from ``assert_must_probe``, which is the point of the map.
_PROBES = {"slab_io": probe}


# ---------------------------------------------------------------------------
# The allowlist — every row names the leg that runs the cell instead
# ---------------------------------------------------------------------------
#: Skips THIS SERVICE is allowed to emit on a machine that declares a
#: profile, on top of lxkit's universal rows (device counts, real
#: multi-process, no-lorrax, no-jax).  A skip nobody can point at a
#: covering run for is evaporated coverage, so ``covered_by`` is required
#: rather than decorative.
_ALLOWED = (
    # NOTE the wording: NOT "no monorepo", which is an lxkit UNIVERSAL row
    # present in every profile and matched by substring — a reason worded
    # that way would be allowed on Perlmutter whatever this list says.
    # See conftest.NO_DECK, which carries the same warning.
    AllowedSkip("", "checked-in deck absent",
                "the monorepo run: a standalone install has no "
                "tests/regression/, and every deck-dependent cell here is "
                "about a checked-in WFN.h5 (this row is REMOVED by the "
                "perlmutter profile — see _allowed_for)"),
    AllowedSkip("", "not built on this machine",
                "leg L-c with the BUILD_NOTES.md .so pins; ABSENT is the "
                "honest skip and BUILT-AND-BROKEN is a FAILURE, which is "
                "the split classify_probe makes and this row does not "
                "weaken (this row is REMOVED by the perlmutter profile)"),
    AllowedSkip("", "requires 2-spinor WFN",
                "any deck in conftest.DECKS — all five are nspinor=2, so "
                "this fires only on a hypothetical scalar deck"),
)


def _allowed_for(profile: MachineProfile):
    """Rows a machine that PROMISES the fixtures and the .so does not get.

    Both removed rows are fine on a laptop and defects on Perlmutter: the
    fixture tree is CHECKED IN, so a missing-deck skip there means the
    suite ran outside the monorepo; and BUILD_NOTES.md pins the library,
    so a not-built skip there means the pin was not passed and the tier
    evaporated.  Filtering here rather than writing two lists keeps the
    reason for the difference next to the rows.
    """
    if profile.machine == "perlmutter":
        return tuple(r for r in _ALLOWED
                     if r.why not in ("checked-in deck absent",
                                      "not built on this machine"))
    return _ALLOWED


#: Deck-dependent cells that MUST have run on a machine with the fixtures.
#: Substrings of a nodeid, because the nodeid's prefix differs between the
#: monorepo run and the standalone one (pytest picks a different rootdir
#: when the argument is the service directory — see arm_skip_honesty's
#: docstring for the same hazard).
_MUST_RUN = (
    "test_the_deck_table_is_true_of_the_file_on_disk[gnppm]",
    "test_the_gnppm_deck_loads_and_its_pad_slots_are_the_sentinel",
    "test_load_process_local_serves_a_different_k_per_call",
    "test_the_twin_agrees_with_the_door",
    "test_the_negative_control_runs_without_a_cluster",
    "test_sentinel_mask_conjunction_eager_arm_at_1x1",
    "test_load_process_local_per_rank_windows_at_1x1",
)


# ===========================================================================
#  POSITIVE HALF — the backends this machine PROMISES are really there
# ===========================================================================

def test_this_machines_declared_phdf5_row_is_actually_available():
    """THE GATE.  On a machine that promises phdf5, it is there.

    On Perlmutter that is the fourth MUST row, ``lorrax_phdf5_read`` on
    ``cpu`` — the row distrib_la's gate reports as a GAP.  Anywhere else
    the profile is ``unknown``, which asserts nothing, on purpose: a
    laptop has no parallel-HDF5 FFI build and a gate that demanded one
    would train everybody to ignore it.

    A failure quotes the probe's own three-way reason (unknown target /
    library would not load / loaded but does not export the symbol),
    because those have three different fixes and "unavailable" names none
    of them.
    """
    profile = machine_profile()
    if profile.asserts_nothing:
        pytest.skip(
            f"machine {profile.machine!r} declares no expected-backend "
            f"profile, so there is nothing to assert — covered by the "
            f"Perlmutter leg, where NERSC_HOST selects the profile that "
            f"names phdf5 (slab_io, lorrax_phdf5_read, cpu) as a MUST row")
    gaps = assert_must_probe(profile, _PROBES, service="slab_io")
    assert gaps == [], (
        f"service='slab_io' filtered the rows and this service supplies "
        f"the probe, so nothing should have been unassertable; got {gaps}")


def test_the_union_read_target_this_service_dispatches_is_there_too():
    """The row's LIBRARY is the promise; this is the handler we call.

    The profile names ``lorrax_phdf5_read`` (single slab).  After the
    2026-08-07 fold ``WfnLoader`` dispatches
    ``lorrax_phdf5_read_kchunk_union`` through ``SlabIO.read_slabs``, and
    the two ship in the same ``.so``.  Asserting only the profile's row
    would pass on a library built before the union handler existed —
    which is precisely the partial build the BROKEN rung is for.
    """
    profile = machine_profile()
    if profile.asserts_nothing:
        pytest.skip(
            f"machine {profile.machine!r} declares no expected-backend "
            f"profile — covered by the Perlmutter leg, where the same "
            f"library serves both phdf5 read handlers")
    fails = []
    for platform in ("cpu", "CUDA"):
        ok, reason = probe(UNION_TARGET, platform)
        if not ok and classify_probe(ok, reason) != ABSENT:
            fails.append(f"{platform}: {reason}")
    assert not fails, (
        f"{UNION_TARGET} is present-but-unusable on a machine that "
        f"promises the phdf5 library:\n  " + "\n  ".join(fails))


def test_the_gate_can_fail():
    """RED TWIN, with THIS service's real probe.

    lxkit's own twin uses a stub, which proves the reporting.  This one
    proves the WIRING: a MUST row naming a target lorrax's FFI loader has
    never heard of must make the real ``probe`` say no, and that no must
    reach the failure with its reason intact.  Without this cell a
    mis-plumbed ``_PROBES`` mapping would make the gate above pass on any
    machine by asserting nothing.
    """
    fake = MachineProfile(
        machine="perlmutter",
        must=(MustRow("ghost", "slab_io", "lorrax_no_such_target", "cpu"),),
        allowed_skips=(), min_collected=1)
    # ``pytest.fail`` raises ``Failed``, which derives from BaseException
    # and NOT from Exception — ``pytest.raises(Exception)`` does not catch
    # it and a twin written that way ERRORS instead of passing.
    with pytest.raises(pytest.fail.Exception) as ei:
        assert_must_probe(fake, _PROBES, service="slab_io")
    msg = str(ei.value)
    assert "lorrax_no_such_target" in msg
    assert "DEFECT" in msg
    assert "not a target" in msg or "unknown" in msg.lower(), msg


def test_this_service_closes_the_gap_distrib_la_reports():
    """The phdf5 row: UNASSERTED over there, ASSERTED here.

    distrib_la's gate has a cell saying "phdf5 is a machine promise
    distrib_la has no library for; it comes back as a GAP and the slab_io
    retrofit supplies the probe".  This service IS a client of that door,
    so the inversion belongs here: with wfn_loader's probe in the map the
    row is no longer a gap.  Stub probes on both sides, because the claim
    is about WHICH ROWS come back unasserted — running the real probe
    would make the cell also depend on whether this machine has the .so,
    i.e. measure two things and report one.
    """
    profile = machine_profile("perlmutter")
    stub = lambda t, p: (True, "")                            # noqa: E731
    # ...as distrib_la sees it: phdf5 is a gap.
    gaps = assert_must_probe(profile, {"distrib_la": stub})
    assert [r.name for r in gaps] == ["phdf5"]
    assert gaps[0].service == "slab_io"
    assert gaps[0].target == "lorrax_phdf5_read"
    # ...and as THIS service sees it: no gaps left unassertable except
    # the three distrib_la owns.
    gaps2 = assert_must_probe(profile, {"slab_io": stub})
    assert [r.name for r in gaps2] == ["scalapack", "slate", "cusolvermp"]
    assert all(r.service == "distrib_la" for r in gaps2)


# ===========================================================================
#  THE ABSENT / BUILT-AND-BROKEN SPLIT
# ===========================================================================

def test_the_read_door_is_absent_not_broken_on_this_machine():
    """A library that will not LOAD is a skip; one that loaded and does
    not EXPORT the handler is a FAILURE.

    This is the split the charter names, applied to the door this service
    reads through.  On a laptop both platforms come back ABSENT (no build)
    and the cell passes having established exactly that.  On a machine
    with a build that is missing the union handler it goes RED, which is
    the outcome a skip would have hidden.
    """
    verdicts = {}
    for platform in ("cpu", "CUDA"):
        for target in ("lorrax_phdf5_read", UNION_TARGET):
            ok, reason = probe(target, platform)
            verdicts[(platform, target)] = (classify_probe(ok, reason),
                                            reason)
    broken = {k: v[1] for k, v in verdicts.items() if v[0] == BROKEN}
    assert not broken, (
        "the FFI library LOADED on this machine and does not export the "
        "phdf5 read handler(s) below.  That is a partial build, not a "
        "platform this suite does not apply to, and a skip here would "
        "hide it:\n  "
        + "\n  ".join(f"{p}/{t}: {r}" for (p, t), r in broken.items()))
    grammar = {k: v[1] for k, v in verdicts.items() if v[0] == GRAMMAR}
    assert not grammar, (
        "a target name in this file is not a target of the FFI library — "
        "a typo here, not a machine property:\n  "
        + "\n  ".join(f"{p}/{t}: {r}" for (p, t), r in grammar.items()))


def test_the_absent_versus_broken_split_can_fail():
    """RED TWIN for :func:`classify_probe`, including its DEFAULT.

    Four rungs and the default, because the default is the design: an
    unrecognized reason must classify BROKEN.  A classifier that fell
    through to ABSENT would turn every reason nobody had thought about
    into an honest-looking skip — which is the exact shape of the failure
    the gate exists to remove, rebuilt inside the gate.
    """
    assert classify_probe(True, "") == OK
    assert classify_probe(
        False, "the cpu FFI library could not be loaded: OSError: ...") \
        == ABSENT
    assert classify_probe(
        False, "loaded but does not export lorrax_phdf5_read_kchunk_union") \
        == BROKEN
    assert classify_probe(
        False, "unknown target: 'nope' is not a target of the cpu FFI "
               "library") == GRAMMAR
    # THE DEFAULT.  A reason from the future must be a defect, not a skip.
    assert classify_probe(False, "something nobody has classified") == BROKEN, (
        "an unrecognized probe reason classified as ABSENT, so any new "
        "failure mode becomes an honest-looking skip")


# ===========================================================================
#  NEGATIVE HALF — what this session actually DID, diffed against the profile
# ===========================================================================

def _observed_skips(outcomes: dict):
    return tuple(ObservedSkip(nodeid, reason)
                 for nodeid, (outcome, reason) in outcomes.items()
                 if outcome == "skipped")


def test_the_deck_dependent_cells_ran_on_a_machine_that_has_the_decks(
        observed_outcomes, collected_nodeids, deck_resolver):
    """THE NEGATIVE HALF.  Three claims, checked in this order.

    1. THE FLOOR, first.  A session that collected nothing from this
       directory skips nothing and fails nothing, so every check below
       would pass vacuously — that is the 38-byte-junitxml failure mode,
       and checking it last would reproduce it.
    2. THE DECK CELLS WERE COLLECTED, and the deck they depend on
       RESOLVES.  Collection is order-free and so is the deck lookup, so
       this half of the claim holds under ``-n 4`` as well as a serial
       run.  On a machine whose profile promises the fixture tree, a deck
       that does not resolve is the defect; the cells would then skip
       with ``conftest.NO_DECK``, and the allowlist (``_allowed_for``)
       removes that row on Perlmutter so the skip would be a FAILURE.
    3. EVERY SKIP OBSERVED SO FAR in this directory is on the allowlist.
       Order-dependent by nature — this file sorts last in the directory,
       so in a serial run "so far" is "all of them" — and the observed
       count is REPORTED rather than assumed, because an assertion over
       an empty observation set is not an assertion.
    """
    profile = machine_profile()
    mine = {n for n in collected_nodeids}

    # 1. THE FLOOR
    assert len(mine) >= max(1, profile.min_collected), (
        f"this session collected {len(mine)} cell(s) from {_TESTS}, below "
        f"the floor of {max(1, profile.min_collected)}.  A session that "
        f"collected nothing measures nothing; pytest rc=5 is not a pass.")

    if profile.asserts_nothing:
        pytest.skip(
            f"machine {profile.machine!r} declares no profile, so the "
            f"negative half asserts nothing here (it observed "
            f"{len(_observed_skips(observed_outcomes))} skip(s) across "
            f"{len(mine)} collected cell(s)) — covered by the Perlmutter "
            f"leg, whose profile removes the missing-deck and not-built "
            f"rows from the allowlist")

    # 2. COLLECTED, and the deck really is there
    missing = [name for name in _MUST_RUN
               if not any(name in n for n in mine)]
    assert not missing, (
        f"cells that must run on a machine with the checked-in decks were "
        f"not collected: {missing}.  Either the suite was deselected "
        f"(--no-services / -k) or a module failed to import, and both look "
        f"like a green run from the outside.")
    assert deck_resolver("gnppm") is not None, (
        f"machine profile {profile.machine!r} promises the monorepo "
        f"fixture tree and tests/regression/gnppm_debug/WFN.h5 does not "
        f"resolve; every deck-dependent cell in this service just became "
        f"a skip")

    # 3. THE SKIP DIFF
    skips = _observed_skips(observed_outcomes)
    assert_skips_match_profile(
        skips, profile, collected=len(mine),
        extra_allowed=_allowed_for(profile),
        min_collected=max(1, profile.min_collected))


def test_the_negative_half_can_fail(no_deck_reason):
    """RED TWIN for the skip diff AND for the floor.

    Both arms, because they are different assertions with different
    failure texts and a twin that only exercised one would leave the other
    unproven.  Pure calls into lxkit's ``assert_skips_match_profile``, so
    the twin needs no fixture gymnastics — which is exactly why that
    function is pure.
    """
    perl = machine_profile("perlmutter")

    # (a) a skip no row allows must be a FAILURE.
    with pytest.raises(AssertionError) as ei:
        assert_skips_match_profile(
            [ObservedSkip("services/wfn_loader/tests/x.py::t",
                          "because I felt like it")],
            perl, collected=50, extra_allowed=_allowed_for(perl))
    assert "does not allow" in str(ei.value)
    assert "because I felt like it" in str(ei.value)

    # (b) ...and one that IS allowed passes, so (a) is about the row and
    #     not about the function refusing everything.
    assert_skips_match_profile(
        [ObservedSkip("services/wfn_loader/tests/x.py::t",
                      "real multi-process leg: this is a single-process "
                      "pytest")],
        perl, collected=50, extra_allowed=_allowed_for(perl))

    # (c) THE FLOOR: an empty session must not pass quietly.
    with pytest.raises(AssertionError, match="below the floor"):
        assert_skips_match_profile([], perl, collected=0,
                                   extra_allowed=_allowed_for(perl),
                                   min_collected=1)

    # (d) the perlmutter filter really removes the two rows, so a
    #     missing-deck skip there is red rather than allowed.  MEASURED
    #     writing this cell: the first wording of conftest.NO_DECK said
    #     "no monorepo fixture tree", and lxkit's UNIVERSAL "no monorepo"
    #     row matched it by substring on EVERY profile — so this arm went
    #     green-that-should-be-red and the one machine that must have the
    #     decks was the one machine where their absence was invisible.
    real_reason = no_deck_reason
    with pytest.raises(AssertionError, match="does not allow"):
        assert_skips_match_profile(
            [ObservedSkip("x::t", real_reason)],
            perl, collected=50, extra_allowed=_allowed_for(perl))
    # ...and the SAME reason on an unmeasured machine is fine, which is
    # what makes (d) a statement about Perlmutter rather than about the
    # string.
    assert_skips_match_profile(
        [ObservedSkip("x::t", real_reason)],
        machine_profile("a-laptop"), collected=50,
        extra_allowed=_allowed_for(machine_profile("a-laptop")))
    # ...and the not-built row behaves the same way.
    with pytest.raises(AssertionError, match="does not allow"):
        assert_skips_match_profile(
            [ObservedSkip("x::t", "not built on this machine — no FFI "
                                  "library serves the collective WFN read")],
            perl, collected=50, extra_allowed=_allowed_for(perl))


def test_arming_is_per_scope_and_cannot_disarm_another_service():
    """STRUCTURAL, for adjudication (ii): arming is per SCOPE, not one slot.

    ``lxkit.testing._ARMED`` used to be a single dict that
    ``arm_skip_honesty`` overwrote, so a second calling service did not add a
    second gate — it replaced the first one's scope and allowlist.  MEASURED
    at the merged head on a full ``services/`` run: three services arm
    unconditionally, conftests load in directory order, and the surviving
    scope was VCOUL's, with distrib_la's and symmetry_maps' gates silently
    inert.  That is why this file has a service-local collector at all.

    The registry is now keyed by directory, so this cell asserts the property
    CONSTRUCTIVELY rather than observing whoever happens to hold a slot: arm
    a scope of our own and require that every row already present survives.
    The red twin is the single-slot behaviour itself — under it ``before`` is
    not a subset of ``after``, because arming replaced the dict.
    """
    from lxkit import testing as lxt

    before = {r.scope: r for r in lxt.armed_scopes()}
    probe = os.path.join(_TESTS, "_arming_probe_dir")
    key = os.path.realpath(probe)
    assert key not in before, "the probe scope must start unarmed"
    lxt.arm_skip_honesty(machine_profile("a-laptop"), scope=probe)
    try:
        after = {r.scope: r for r in lxt.armed_scopes()}
        assert key in after, (
            "arm_skip_honesty did not register the scope it was given")
        missing = set(before) - set(after)
        assert not missing, (
            f"arming {probe!r} REMOVED {sorted(missing)} from the registry. "
            f"That is the single-slot defect: a second service's arm call "
            f"takes the first service's gate down, and nothing turns red.")
        for scope, row in before.items():
            assert after[scope] == row, (
                f"arming {probe!r} MUTATED the row for {scope!r} "
                f"({before[scope]} -> {after[scope]})")
    finally:
        lxt._ARMED.pop(key, None)


def test_this_suites_local_collector_is_still_the_one_ruling_on_it():
    """This file's gate is service-local (DESIGN.md DECISION 4) and stays so.

    Per-scope arming makes lxkit's gate safe to use, but this suite's gate
    rules on OUTCOMES (what every cell in this directory actually did), not
    only on skips, so it keeps its own collector.  What must remain true is
    that this suite did not quietly arm lxkit's registry under its own
    directory as well, which would judge the same skips twice.
    """
    from lxkit import testing as lxt
    ours = os.path.realpath(_TESTS)
    assert ours not in {r.scope for r in lxt.armed_scopes()}, (
        f"something in this suite armed lxkit's skip-honesty gate on "
        f"{_TESTS!r}; this service is ruled on by its own outcome collector "
        f"and a second gate over the same directory double-judges it")


def test_the_profile_names_the_machine_it_is_talking_about():
    """A failure that says "some backend is missing" without saying where
    sends the reader to guess.  Cheap to assert, and it is the part of the
    message a person acts on first."""
    assert machine_profile("perlmutter").machine == "perlmutter"
    assert machine_profile("a-laptop").machine == "a-laptop"
    assert machine_profile("a-laptop").asserts_nothing is True
    assert machine_profile("perlmutter").asserts_nothing is False

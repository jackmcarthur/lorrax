"""``compute_mode = mpa`` — the mode that used to exist in order to refuse.

WHAT IS BEING PINNED, AND WHY IT IS WORTH A FILE OF ITS OWN.  The MPA
("FF") self-energy landed in two halves.  First the safety skeleton: the
value on the axis, a parser that accepts it, a refusal that named it, a
row in the channel-availability table, and — the part that is actually
load-bearing — an explicit branch at every site in the tree that
dispatches on ``compute_mode``.  Then, on 2026-08-09, the driver seam:
``gw.mpa_pipeline``, the q→0 head in ``gw.mpa.sigma_head``, a branch in
``sigma_dispatch``, a screening plan, and the deck key ``mpa_fit_file``
that says which staged pole store to read.

The failure this file forecloses is not "MPA runs badly".  It is "MPA
runs as GN-PPM and nobody can tell".  The Σ dispatch reached its
plasmon-pole pipeline by ELSE: anything that was neither X_ONLY nor
COHSEX ran it.  Under that shape a multipole run would have fitted a GN
pole to two W samples, called the result Σ_c(ω), and produced a
complete, finite, plausible set of QP energies for the wrong ansatz.

THE CELLS THAT SAID "IT REFUSES" NOW SAY WHAT IT DOES INSTEAD, which is
what the ``UNIMPLEMENTED_MODES`` comment asked of whoever removed the
row: rewrite, do not delete.  The refusals did not go away — they MOVED,
from an axis of modes to the two files a multipole run reads.  A store
with no ``__mpahead`` head axis, a store on the symmetry wedge, and a
deck naming no store at all are each still refused BY NAME, and the
cells below now pin those instead.  They are exercised end-to-end in
``tests/test_mpa_sigma_head.py``; what is pinned here is that the
DISPATCH reaches them rather than reaching the plasmon-pole pipeline.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from gw.gw_config import (
    MODE_SIGMA_CHANNELS,
    UNIMPLEMENTED_MODES,
    ComputeMode,
    SigmaChannel,
    coerce_compute_mode,
    explain_missing_channels,
    mode_builds_channels,
    refuse_unimplemented_compute_mode,
    sigma_channels_for,
)

_REPO = pathlib.Path(__file__).resolve().parents[1]
_GW = _REPO / "src" / "gw"

#: The modes whose Σ stage exists today — ALL FIVE, since the multipole
#: seam landed.  Every "nothing moved" assertion below runs over exactly
#: this set, so it grew by one line rather than by a rewrite, which is the
#: property it was written for.
_RUNNABLE = tuple(m for m in ComputeMode if m not in UNIMPLEMENTED_MODES)

#: The four modes that were runnable BEFORE the multipole seam.  Kept
#: separately from :data:`_RUNNABLE` because "MPA did not disturb the
#: modes that already worked" is a claim about these four specifically,
#: and folding MPA into the set that is supposed to be unmoved would make
#: the claim about itself.
_RUNNABLE_BEFORE_MPA = tuple(m for m in _RUNNABLE if m is not ComputeMode.MPA)


# ---------------------------------------------------------------------------
# 1. The value, and how it is spelled
# ---------------------------------------------------------------------------

def test_the_axis_carries_mpa_and_spells_it_mpa():
    """The deck value is ``mpa``; ``full_freq`` is not a spelling of it.

    The naming decision is argued in the ``ComputeMode`` docstring — every
    value on this axis names an ansatz, and "full frequency" names a
    family of them, so it would have to be qualified by a second key.
    Pinning the rejection here means a later "helpful" alias cannot
    reintroduce the ambiguity without this cell going red.
    """
    assert ComputeMode.MPA.value == "mpa"
    assert "full_freq" not in {m.value for m in ComputeMode}


def test_the_naming_decision_and_its_rejected_alternative_are_written_down():
    """The docstring has to say WHY, not just WHAT.

    A reader who wants to add ``full_freq`` later needs to find the
    reasoning at the enum, not in a commit message.
    """
    doc = ComputeMode.__doc__
    assert "full_freq" in doc
    assert "mpa" in doc


def test_a_deck_can_select_it_and_the_parser_resolves_the_enum():
    """Parsing is not permitting: the config layer answers, and the
    driver is what refuses.  Config-only consumers (the deck echo, a
    layering test, an operator reading a config back) must be able to
    name the mode without tripping over it."""
    assert coerce_compute_mode("mpa") is ComputeMode.MPA
    assert coerce_compute_mode("  MPA ") is ComputeMode.MPA
    assert coerce_compute_mode(ComputeMode.MPA) is ComputeMode.MPA


def test_a_typo_gets_the_unknown_mode_error_not_the_refusal():
    """RED TWIN of the refusal: two different operator mistakes, two
    different words.

    ``mpaa`` is "no such mode" — a ValueError naming the legal set.
    ``mpa`` is "that mode, not yet" — a NotImplementedError naming the
    mode.  Collapsing them would tell someone with a typo to go read the
    MPA theory doc.
    """
    with pytest.raises(ValueError) as exc:
        coerce_compute_mode("mpaa")
    msg = str(exc.value)
    assert "mpaa" in msg
    for mode in ComputeMode:
        assert mode.value in msg
    assert not isinstance(exc.value, NotImplementedError)


def test_auto_never_infers_mpa_from_a_legacy_deck():
    """``compute_mode = auto`` reads legacy flags that all predate MPA, so
    no combination of them may reach it.

    THE REASON CHANGED WHEN THE MODE LANDED, and the cell is worth more
    now than it was.  While ``mpa`` refused, an ``auto`` route to it would
    have given an existing deck a refusal it never asked for — annoying,
    and loud.  Now that it runs, the same route would hand an existing
    deck a DIFFERENT SELF-ENERGY, silently, and then refuse only because
    it happens to name no ``mpa_fit_file``.  A mode that reads its whole
    screening off disk must be asked for by name.

    Every unimplemented mode is checked the same way, so the day another
    joins the axis this cell covers it without an edit.
    """
    source = (_GW / "gw_config.py").read_text()
    tree = ast.parse(source)
    resolver = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "compute_mode")
    returned = {
        ast.unparse(node.value) for node in ast.walk(resolver)
        if isinstance(node, ast.Return) and node.value is not None
    }
    for mode in (ComputeMode.MPA,) + tuple(UNIMPLEMENTED_MODES):
        assert not any(f"ComputeMode.{mode.name}" in r for r in returned), (
            f"the auto branch can return {mode.value}, which no legacy deck "
            f"asked for")


# ---------------------------------------------------------------------------
# 2. What the refusal became
# ---------------------------------------------------------------------------

def test_selecting_it_no_longer_refuses_on_the_mode_axis():
    """THE LANDING GESTURE, PINNED.

    ``UNIMPLEMENTED_MODES``' own comment says removing a row is how a mode
    is turned on, and that the suite pinning the refusal must fail loudly
    until it is rewritten.  This is the rewrite: ``mpa`` is not on that
    axis any more, and selecting it is no longer an error in itself.

    The dict is checked EMPTY rather than only for the absence of ``mpa``,
    because a mode quietly re-added to it would be a mode turned off
    again, and the mechanism is worth a cell of its own even while unused.
    """
    assert ComputeMode.MPA not in UNIMPLEMENTED_MODES
    assert UNIMPLEMENTED_MODES == {}
    assert refuse_unimplemented_compute_mode(
        ComputeMode.MPA) is ComputeMode.MPA


def test_what_it_refuses_now_is_a_deck_that_names_no_fit_store():
    """The refusal MOVED from the mode to the file, and this is where.

    ``compute_mode = mpa`` reads its entire screening from a staged pole
    store, and ``screening_requests_for(mpa)`` deliberately asks the
    screening stage for nothing — so a deck that selects the mode and
    names no store has requested a screened self-energy and supplied no
    screening.  There is nothing to default to: falling back to the bare
    Coulomb would return an unscreened Σ_c under the multipole name.

    The two OTHER refusals the old message named — a wedge-shaped pole
    axis and a headless store — are properties of the file rather than of
    the deck, and are exercised through the pipeline in
    ``tests/test_mpa_sigma_head.py``.
    """
    import types

    from gw.gw_config import refuse_missing_mpa_fit_store

    cfg = types.SimpleNamespace(
        compute_mode=ComputeMode.MPA,
        paths=types.SimpleNamespace(mpa_fit_file=None))
    with pytest.raises(ValueError) as exc:
        refuse_missing_mpa_fit_store(cfg, context="the LORRAX GW driver")
    msg = str(exc.value)
    assert "mpa_fit_file" in msg
    assert "the LORRAX GW driver" in msg
    assert "__mpahead" in msg and "full-BZ" in msg
    # RED TWIN: with a store named, it returns the path and does not raise.
    cfg.paths = types.SimpleNamespace(mpa_fit_file="/tmp/fit.h5")
    assert refuse_missing_mpa_fit_store(cfg) == "/tmp/fit.h5"


@pytest.mark.parametrize("mode", _RUNNABLE_BEFORE_MPA, ids=lambda m: m.value)
def test_no_other_mode_acquired_the_new_refusal(mode):
    """RED TWIN of the cell above: the new guard sits on the driver's fast
    path beside the old one, so it must be invisible to every mode whose
    screening comes from the screening stage."""
    import types

    from gw.gw_config import refuse_missing_mpa_fit_store

    cfg = types.SimpleNamespace(
        compute_mode=mode,
        paths=types.SimpleNamespace(mpa_fit_file=None))
    assert refuse_missing_mpa_fit_store(cfg) is None


def test_the_document_the_landing_points_at_is_in_the_tree():
    """A pointer that dangles is half a document."""
    doc = _REPO / "docs" / "theory" / "THEORY_mpa_implementation.md"
    assert doc.is_file()
    text = doc.read_text()
    assert "compute_mode = mpa" in text
    assert "UNIMPLEMENTED_MODES" in text


@pytest.mark.parametrize("mode", _RUNNABLE, ids=lambda m: m.value)
def test_the_refusal_is_a_no_op_for_every_mode_that_has_a_sigma_stage(mode):
    """RED TWIN: the guard sits on the driver's fast path, so it must be
    invisible to the five modes that work — which is now all of them."""
    assert refuse_unimplemented_compute_mode(mode) is mode


def test_the_driver_refuses_at_entry_before_it_spends_anything():
    """WHERE the refusal sits is the whole value of it.

    Refusing after the ζ fit is a refusal that costs an allocation.  Read
    from the source rather than by running the driver (which needs a deck,
    a WFN and a mesh): the call must appear before the WFN loader.
    """
    source = (_GW / "gw_jax.py").read_text()
    guard = source.index("refuse_unimplemented_compute_mode(mode")
    assert guard < source.index("WfnLoader(config.paths.wfn_file")
    assert guard < source.index("isdf = prepare_isdf_and_wavefunctions(")


# ---------------------------------------------------------------------------
# 3. Exhaustive dispatch — no site inherits an else-branch
# ---------------------------------------------------------------------------

def test_the_screening_planner_asks_for_nothing_and_that_is_the_answer():
    """MPA's W is sampled on the double-parallel grid, not at {0, probe},
    and it is sampled by the FIT DRIVER before the run rather than by this
    stage during it.

    So the plan is the empty list, and the two facts that make that a
    decision rather than a hole are pinned here.  First, it is not a
    fall-through: an unknown mode still raises, and the MPA branch is
    reached by an explicit ``is`` test.  Second, the static channels do
    not secretly need a W either — ``mpa`` builds ``{Σ_X, Σ_c(ω)}``, so
    ``sigma_dispatch`` routes it to the V-only kernel
    (``compute_v_h_sigma_x``), which touches W nowhere.  Returning the PPM
    pair instead would be a wrong answer no downstream stage could detect,
    because the shapes and roles would all be right.
    """
    from gw.screening import screening_requests_for

    assert screening_requests_for(ComputeMode.MPA, config=None) == []
    # The static screened channels are what a `static` W would be FOR, and
    # this mode builds neither of them.
    assert not mode_builds_channels(
        ComputeMode.MPA, SigmaChannel.SX, SigmaChannel.COH)
    # RED TWIN: the empty list is this branch's answer, not the planner
    # having no opinion — an unrecognised mode still raises.
    with pytest.raises(ValueError, match="unknown compute mode"):
        screening_requests_for("not-a-mode", config=None)


def test_the_v_only_kernel_is_what_serves_mpas_static_channels():
    """The claim above, read off the dispatch rather than asserted.

    ``builds_static_screened`` is a lookup in ``MODE_SIGMA_CHANNELS``, and
    the branch it drives is what decides whether the run touches a
    screened kernel at all.  If a future edit gave MPA the SX/COH row, the
    empty screening plan would become a ``KeyError`` on ``W_by_role`` at
    the worst possible moment; this cell is what fails first instead.
    """
    source = (_GW / "sigma_dispatch.py").read_text()
    tree = ast.parse(source)
    fn = next(node for node in ast.walk(tree)
              if isinstance(node, ast.FunctionDef)
              and node.name == "compute_sigma_xc")
    body = ast.unparse(fn)
    assert "builds_static_screened = mode_builds_channels(" in body
    assert "compute_v_h_sigma_x(" in body
    assert sigma_channels_for(ComputeMode.MPA) == frozenset(
        {SigmaChannel.X, SigmaChannel.C_OMEGA})


def test_the_screening_planner_is_unmoved_for_the_modes_that_work():
    """RED TWIN / bit-identity: the four existing plans, unchanged.

    ``x_only`` asks for nothing, ``cohsex`` for the static W alone, and
    the two PPMs for static plus a probe — imaginary for GN, real for HL,
    which is the difference the two-point fit is built on.
    """
    import types

    from gw.screening import screening_requests_for

    cfg = types.SimpleNamespace(ppm=types.SimpleNamespace(omega_p=1.5))
    assert screening_requests_for(ComputeMode.X_ONLY, cfg) == []

    cohsex = screening_requests_for(ComputeMode.COHSEX, cfg)
    assert [r.role for r in cohsex] == ["static"]
    assert cohsex[0].omega_ry == 0.0

    gn = screening_requests_for(ComputeMode.GN_PPM, cfg)
    assert [r.role for r in gn] == ["static", "probe"]
    assert gn[1].omega_ry == 1.5j

    hl = screening_requests_for(ComputeMode.HL_PPM, cfg)
    assert [r.role for r in hl] == ["static", "probe"]
    assert hl[1].omega_ry == complex(1.5, 0.0)


def test_the_sigma_dispatch_no_longer_reaches_the_ppm_pipeline_by_else():
    """THE CENTRAL CELL OF THIS FILE.

    ``compute_sigma_xc`` handles X_ONLY and COHSEX by name and then ran
    the two-point plasmon-pole pipeline for everything else.  The guard
    that replaced that ELSE asks ``ppm_model``, which is 'gn' or 'hl' for
    exactly the two PPM modes and None for every other member of the enum,
    present or future.  Read from source: reaching the real function needs
    a mesh, a wavefunction bundle and a screened W.
    """
    source = (_GW / "sigma_dispatch.py").read_text()
    tree = ast.parse(source)
    fn = next(node for node in ast.walk(tree)
              if isinstance(node, ast.FunctionDef)
              and node.name == "compute_sigma_xc")
    body = ast.unparse(fn)

    guard = body.index("mode.ppm_model is None")
    assert guard < body.index("compute_ppm_sigma_pipeline("), (
        "the pole-model guard must come BEFORE the PPM pipeline call, or "
        "it is not guarding it")
    assert "refuse_unimplemented_compute_mode(mode" in body
    # AND MPA IS SERVED ABOVE THE GUARD, not by widening it.  The guard's
    # job is to turn away a mode with no Σ kernel; a mode that HAS one is
    # dispatched before it is asked.
    assert body.index("compute_mpa_sigma_pipeline(") < guard


def test_the_ppm_pipeline_turns_away_a_mode_with_no_pole_model():
    """The pipeline reads the mode as "HL, or else GN" in three places.

    One entry check is the honest way to hold that: the alternative is
    three separate refusals, or (as before) three silent GN defaults.
    """
    source = (_GW / "ppm_pipeline.py").read_text()
    tree = ast.parse(source)
    fn = next(node for node in ast.walk(tree)
              if isinstance(node, ast.FunctionDef)
              and node.name == "compute_ppm_sigma_pipeline")
    body = ast.unparse(fn)
    guard = body.index("ppm_model is None")
    assert guard < body.index("is_hl = ppm_model")
    assert "NotImplementedError" in body


def test_the_head_persistence_refuses_to_invent_a_probe_frequency():
    """The stored ω grid is the probe set, so the POLE MODEL decides it.

    A dynamic-else-GN read would stamp {0, iω_p} onto a restart file whose
    W was never evaluated there — an artifact that lies quietly to the
    next run.
    """
    source = (_GW / "gw_output.py").read_text()
    tree = ast.parse(source)
    fn = next(node for node in ast.walk(tree)
              if isinstance(node, ast.FunctionDef)
              and node.name == "persist_w0_and_head")
    body = ast.unparse(fn)
    assert "ppm_model is not None" in body
    assert "NotImplementedError" in body


def test_ppm_model_separates_the_two_questions_the_tree_was_conflating():
    """``is_dynamic`` ("has an ω axis") and ``ppm_model`` ("which pole
    fit") agreed for every mode that existed, which is why sites could
    use the first to mean the second.  MPA is the member where they
    part."""
    assert ComputeMode.MPA.is_dynamic is True
    assert ComputeMode.MPA.ppm_model is None
    assert ComputeMode.MPA.needs_screening is True
    # Unmoved for the four:
    assert ComputeMode.GN_PPM.ppm_model == "gn"
    assert ComputeMode.HL_PPM.ppm_model == "hl"
    assert ComputeMode.COHSEX.ppm_model is None
    assert ComputeMode.X_ONLY.ppm_model is None
    assert [m.is_dynamic for m in _RUNNABLE_BEFORE_MPA] == [
        False, False, True, True]


def test_no_module_dispatches_on_the_mode_through_a_bare_else():
    """THE AUDIT, AS A TEST.

    Any site that compares against a specific ``ComputeMode`` member and
    then falls into a plain ``else`` is a site that will serve the next
    mode whatever the last one got.  The scan is deliberately crude — it
    reads source text, not behaviour — because its job is to make a
    reviewer look at a new one, not to prove the existing ones correct.
    """
    offenders = []
    for path in sorted(_GW.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.If) or not node.orelse:
                continue
            # ``elif`` chains are not an else-branch; only a plain else is.
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                continue
            test_src = ast.unparse(node.test)
            if "ComputeMode." not in test_src:
                continue
            offenders.append(f"{path.name}:{node.lineno} — if {test_src}")
    assert offenders == [], (
        "a compute-mode comparison with a plain else-branch: the next mode "
        "added to the enum silently takes that branch.  Name it, or route "
        "the decision through gw_config.MODE_SIGMA_CHANNELS.\n  "
        + "\n  ".join(offenders))


# ---------------------------------------------------------------------------
# 4. The channel-availability table
# ---------------------------------------------------------------------------

def test_every_mode_has_a_row_and_adding_one_without_a_row_fails_here():
    """THE RATCHET.  This is the cell a future mode trips over.

    An enum member with no row is not a mode with no channels; it is a
    mode nobody has thought about, and the writers would read it as "this
    run built nothing", which is a legible wrong answer.
    """
    assert set(MODE_SIGMA_CHANNELS) == set(ComputeMode), (
        "every ComputeMode member needs a row in MODE_SIGMA_CHANNELS "
        f"(missing: {sorted(m.value for m in set(ComputeMode) - set(MODE_SIGMA_CHANNELS))})")


def test_the_table_records_what_the_four_working_modes_actually_build():
    """Bit-identity, stated as channels.

    Read off the shipped kernels: ``x_only`` returns Σ_x alone; ``cohsex``
    returns Σ_SX + Σ_COH beside its bare Σ_x; the two PPMs return Σ_x plus
    a Σ_c(ω) cube and never touch the static screened channels
    (``sigma_dispatch`` routes them to ``compute_v_h_sigma_x``).
    """
    assert sigma_channels_for(ComputeMode.X_ONLY) == frozenset(
        {SigmaChannel.X})
    assert sigma_channels_for(ComputeMode.COHSEX) == frozenset(
        {SigmaChannel.X, SigmaChannel.SX, SigmaChannel.COH})
    for ppm in (ComputeMode.GN_PPM, ComputeMode.HL_PPM):
        assert sigma_channels_for(ppm) == frozenset(
            {SigmaChannel.X, SigmaChannel.C_OMEGA})


def test_mpas_row_says_what_its_sigma_stage_will_build():
    """Same two channels as the PPMs, different producer — which is
    exactly why the table cannot be the only safety net, and why the mode
    also refuses to run."""
    assert sigma_channels_for(ComputeMode.MPA) == frozenset(
        {SigmaChannel.X, SigmaChannel.C_OMEGA})


def test_the_table_takes_the_spellings_its_callers_actually_hold():
    """Writers reach the table holding a resolved enum, a bare string, or
    a stand-in object carrying ``.value``.  One normalisation, so the
    table is the single answer rather than the next hand-check."""
    import types

    for spelling in ("cohsex", ComputeMode.COHSEX,
                     types.SimpleNamespace(value="cohsex")):
        assert mode_builds_channels(spelling, SigmaChannel.SX,
                                    SigmaChannel.COH)


def test_a_channel_asked_of_a_mode_that_does_not_build_it_is_named():
    """The omission clause, which is what a writer prints."""
    text = explain_missing_channels(ComputeMode.GN_PPM,
                                    SigmaChannel.SX, SigmaChannel.COH)
    # The physics spelling, which is what the QSGW appendix's line has
    # always printed — the lowercase enum values are for keying tables.
    assert "Σ_SX" in text and "Σ_COH" in text
    assert "gn_ppm" in text
    assert "not built" in text
    # RED TWIN: asking for a channel the mode DOES build says so plainly
    # rather than producing an empty, printable-looking clause.
    assert "nothing missing" in explain_missing_channels(
        ComputeMode.COHSEX, SigmaChannel.SX)


def test_a_mode_with_no_row_refuses_rather_than_reading_as_empty():
    """RED TWIN of the ratchet, at the accessor.

    The ratchet above fails in CI; this is what happens at runtime in the
    window before someone runs CI.
    """
    saved = MODE_SIGMA_CHANNELS.pop(ComputeMode.HL_PPM)
    try:
        with pytest.raises(KeyError) as exc:
            sigma_channels_for(ComputeMode.HL_PPM)
        assert "MODE_SIGMA_CHANNELS" in str(exc.value)
    finally:
        MODE_SIGMA_CHANNELS[ComputeMode.HL_PPM] = saved


def test_the_qsgw_writers_cohsex_omission_now_goes_through_the_table():
    """The ad-hoc rule the QSGW appendix landed with, formalised.

    That writer asked ``results.use_ppm`` — a proxy that was right for the
    four modes that existed.  The question it was really asking is one row
    of the table, and asking the table means a future mode gets the right
    answer with no edit to the writer.
    """
    source = (_GW / "gw_output.py").read_text()
    tree = ast.parse(source)
    fn = next(node for node in ast.walk(tree)
              if isinstance(node, ast.FunctionDef)
              and node.name == "write_qsgw_qp_ladders")
    body = ast.unparse(fn)
    assert "mode_builds_channels(config.compute_mode" in body
    assert "explain_missing_channels(config.compute_mode" in body
    # And the decision it replaced is gone from that arm, not merely
    # supplemented: two authorities on one question is the state this
    # change exists to leave.
    assert "results.use_ppm" not in body


def test_the_advice_the_writers_print_picks_up_a_mode_when_it_lands():
    """The advice is derived, so a landing changes it with no edit there.

    While ``mpa`` refused, telling an operator to "use mpa" would have
    sent them into the entry refusal, so the advice subtracted
    ``UNIMPLEMENTED_MODES``.  It still does; the dict is simply empty now,
    and the visible consequence is that ``mpa`` appears in the list of
    modes that build Σ_c(ω) — which is the behaviour the derivation was
    written for and is checked here rather than assumed.
    """
    from gw.gw_output import _runnable_modes_building

    advice = _runnable_modes_building(SigmaChannel.C_OMEGA)
    assert "gn_ppm" in advice and "hl_ppm" in advice
    assert "mpa" in advice
    # RED TWIN: the subtraction is still wired, so a mode put back on the
    # unimplemented axis drops straight back out of the advice.
    UNIMPLEMENTED_MODES[ComputeMode.MPA] = "synthetic, for this cell only"
    try:
        assert "mpa" not in _runnable_modes_building(SigmaChannel.C_OMEGA)
    finally:
        del UNIMPLEMENTED_MODES[ComputeMode.MPA]


# ---------------------------------------------------------------------------
# 5. The deck-facing surface
# ---------------------------------------------------------------------------

def test_the_documented_set_in_the_config_lists_the_new_value():
    """The comment block beside the ``compute_mode`` default is what a
    deck author reads; a value missing from it is a value nobody finds."""
    source = (_GW / "gw_config.py").read_text()
    block = source[source.index("# ``compute_mode`` is the single axis"):
                   source.index('"compute_mode": "auto"')]
    for mode in ComputeMode:
        assert f'"{mode.value}"' in block


@pytest.mark.parametrize("doc", ["input_reference.md", "drivers.md"])
def test_the_reference_docs_carry_the_mode_and_say_what_it_needs(doc):
    """A documented mode whose entry still says "REFUSES" is worse than an
    undocumented one, because it tells an operator not to try something
    that works.

    What the row has to say now is the PRECONDITION, since that is the
    part an operator can get wrong: ``mpa`` runs, and it needs a fit
    store named by ``mpa_fit_file``.
    """
    text = (_REPO / "docs" / doc).read_text()
    row = next(ln for ln in text.splitlines() if "`compute_mode`" in ln)
    assert "mpa" in row
    assert "REFUSES" not in row.upper()
    assert "mpa_fit_file" in row


@pytest.mark.parametrize("doc", ["input_reference.md", "drivers.md"])
def test_the_new_deck_key_is_documented_where_a_deck_author_looks(doc):
    """A key that is only findable by grepping the source is a key nobody
    sets — and this one is the difference between a multipole run and a
    refusal."""
    text = (_REPO / "docs" / doc).read_text()
    assert "`mpa_fit_file`" in text

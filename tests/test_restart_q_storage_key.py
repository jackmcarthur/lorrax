"""``restart_q_storage`` — the deck key, the seam, and the ONE resolution.

WHAT IS MEASURED HERE.  Three things, and they are separated on purpose
because they fail for different reasons and get fixed by different people:

1. THE KEY.  Parsed, normalised, validated at PARSE time (the shape
   ``hartree_source`` uses), reaching the frozen dataclass the driver reads
   as a ``_raw`` field — raw because ``auto`` cannot be resolved at parse
   time: its answer depends on the run's centroid set, which does not exist
   until the ISDF stage.

2. THE SEAM.  ``closure_for_restart`` is the ONE function the restart writer
   asks the closure question through, so the owner's stamp ruling
   (DESIGN_symmetry_restart_followup.md, "The stamp architecture", third
   point) lands as a one-line edit.  An AST cell pins that the writer path
   does not route around it.

3. THE RESOLUTION.  A truth table over (request x closed x use_ibz), with
   the two refusal arms named separately, because "your centroid set is not
   closed" and "your q path did not reduce" need different fixes and a
   single refusal would send half the operators to the wrong one.

WHY THE FAKES ARE FAKES.  The resolution consumes a
``QgridSymmetryResolution`` and reads exactly three things off it —
``verdict``, ``use_ibz``, ``reason``.  Building a real one needs a WFN, a
centroid set and the service's geometric measurement; that is covered where
it belongs (``services/symmetry_maps/tests/test_symmetry_maps_closure.py``
measures the REAL verdicts, including the adopted orbit-closed 480 set).
What is NOT covered anywhere else is the decision this module takes on those
three values, so that is what these cells vary, one axis at a time.
"""

from __future__ import annotations

import ast
import dataclasses
import pathlib

import numpy as np
import pytest

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
_STORAGE_MOD = _SRC / "gw" / "restart_q_storage.py"


# ---------------------------------------------------------------------------
# Fakes: the three fields the resolution reads, and nothing else
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class _FakeVerdict:
    """The ``CentroidClosureVerdict`` surface the resolution touches.

    ``describe()`` is included because the refusal message quotes it and a
    refusal that cannot format is a refusal nobody reads.
    """

    closed: bool
    tol: float = 1e-6
    n_sym: int = 48
    n_centroids: int = 480
    worst_residual: float = 0.0
    worst_op: int = 0
    violating_ops: tuple = ()

    @property
    def n_violating(self) -> int:
        return len(self.violating_ops)

    def describe(self) -> str:
        return (f"centroid orbit closure: "
                f"{'CLOSED' if self.closed else 'NOT CLOSED'} — "
                f"{self.n_violating}/{self.n_sym} ops violating")


@dataclasses.dataclass(frozen=True)
class _FakeResolution:
    verdict: _FakeVerdict
    mode: str
    reason: str = ""

    @property
    def use_ibz(self) -> bool:
        return self.mode == "ibz"


def _closed_and_reduced():
    return _FakeResolution(_FakeVerdict(closed=True, worst_residual=1e-9),
                           mode="ibz")


def _open_set():
    return _FakeResolution(
        _FakeVerdict(closed=False, worst_residual=1.318e-1,
                     violating_ops=tuple(range(1, 48))),
        mode="full_bz", reason="centroid set is not closed")


def _closed_but_unreduced():
    return _FakeResolution(_FakeVerdict(closed=True), mode="full_bz",
                           reason="the q-grid admits no reduction")


# ---------------------------------------------------------------------------
# 1. The key
# ---------------------------------------------------------------------------

def test_the_key_defaults_to_auto_because_storage_follows_the_wfn_file():
    """AUTO IS THE DEFAULT AGAIN — and this cell has now held both values.

    That history is the point, so read it before moving it a third time.

    It first asserted ``auto``, citing a design-doc line that did not exist.
    It was moved to ``full`` because the 2026-08-08 landing census priced
    the difference: with ``auto`` in ``_DEFAULTS`` both orbit-closed in-tree
    decks silently began writing the q wedge and **nine cells went red
    across the GW and BSE restart paths, because neither reader unfolded.**
    That clause is the whole reason, and it stopped being true:

    * the GW restart reader has unfolded since ``536cbac9``
      (``file_io.tagged_arrays._unfold_wedge``, applied at ``:1098``/``:1197``)
    * ``bse._MunuSlabPlan``'s refusal was lifted 2026-08-15 and
      ``bse_loading.py:707-711`` unfolds through the SAME function
    * the cost argument behind that refusal was measured and did not
      survive: 57.4 GiB/s unfold against 2.919 GiB/s disk, 6-17x at every
      size tested, with mu^2 cancelling out of the comparison entirely

    So ``auto`` is now what the owner's 2026-08-08 ~13:20 ruling asked for
    in the first place: "symmetries should not need an auto mode — if
    symmetries are not to be used, the wavefunction file should've been
    generated with no symmetries."  ``auto`` IS "follow the file"; on a
    non-closed centroid set it is byte-for-byte ``full``.

    WHAT WOULD JUSTIFY MOVING IT BACK: a reader that does not unfold.  Not
    a red cell that pins the old bytes — that is a cell to update — and not
    a preference for the smaller diff.
    """
    from gw.gw_config import _DEFAULTS

    assert _DEFAULTS["restart_q_storage"] == "auto"


def test_a_deck_that_never_heard_of_the_key_gets_auto(tmp_path):
    """Every archived deck keeps parsing, and now follows its own WFN.

    The bytes-preservation half moved deliberately: a deck that never named
    the key gets ``auto``, which reduces IFF its centroid set is orbit-closed
    AND its q path reduced.  On a non-closed set that is byte-for-byte the
    old file, so "standing still changes nothing" still holds for every deck
    that cannot reduce; for a deck that CAN, the file it writes is the one
    its own symmetry says it should have been writing.  ``full`` is the
    escape hatch for a deck that must be pinned to the old bytes.
    """
    from gw.gw_config import read_lorrax_input

    deck = tmp_path / "cohsex.in"
    deck.write_text("[LORRAX]\nnval = 4\nncond = 4\nnband = 8\n")
    assert read_lorrax_input(str(deck))["restart_q_storage"] == "auto"


def test_the_hand_built_params_fallback_agrees_with_the_registered_default(
        tmp_path, monkeypatch):
    """The ``or`` in the parse site is a SECOND spelling of the default.

    ``LorraxConfig`` resolves ``_g("restart_q_storage") or <fallback>``, and
    that fallback is reached by any caller that assembles the params dict
    itself and omits the key.  Two spellings of one default is how they
    drift, and a drift here is a caller silently storing a different q-set,
    so the two are asserted equal rather than eyeballed.
    """
    monkeypatch.chdir(tmp_path)
    from gw.gw_config import LorraxConfig, _DEFAULTS

    deck = tmp_path / "cohsex.in"
    deck.write_text("[LORRAX]\nnval = 4\nncond = 4\nnband = 8\n")
    cfg = LorraxConfig.from_input_file(str(deck),
                                       print_fn=lambda *a, **k: None)
    assert cfg.restart_q_storage_raw == _DEFAULTS["restart_q_storage"]


@pytest.mark.parametrize("spelling", ["ibz", "IBZ", "  Ibz  ", "FULL",
                                      "Auto"])
def test_the_key_normalises_case_and_whitespace(tmp_path, spelling):
    """``_NORMALIZE_STR`` membership, measured rather than asserted by eye.

    Every other enumerated string key in the deck is case-insensitive, and
    a key that is the one exception is a key an operator types in the
    obvious way and then cannot explain the behaviour of.
    """
    from gw.gw_config import read_lorrax_input

    deck = tmp_path / "cohsex.in"
    deck.write_text(f"[LORRAX]\nnval = 4\nncond = 4\nnband = 8\n"
                    f"restart_q_storage = {spelling}\n")
    got = read_lorrax_input(str(deck))["restart_q_storage"]
    assert got == spelling.strip().lower()


def test_the_key_is_not_an_unknown_deck_key(tmp_path):
    """``strict_keys = true`` must ACCEPT it — the ``_DEFAULTS`` membership."""
    from gw.gw_config import read_lorrax_input

    deck = tmp_path / "cohsex.in"
    deck.write_text("[LORRAX]\nnval = 4\nncond = 4\nnband = 8\n"
                    "strict_keys = true\nrestart_q_storage = full\n")
    assert read_lorrax_input(str(deck))["restart_q_storage"] == "full"


def test_the_key_reaches_the_dataclass_as_a_RAW_field(tmp_path, monkeypatch):
    """Through ``LorraxConfig.from_input_file``, and UNRESOLVED.

    The ``_raw`` suffix is the contract: ``auto`` is still ``auto`` on the
    frozen config, because resolving it needs the closure answer and the
    centroid set does not exist at parse time.  A field that arrived here
    already resolved would have had to guess.
    """
    monkeypatch.chdir(tmp_path)
    from gw.gw_config import LorraxConfig

    deck = tmp_path / "cohsex.in"
    deck.write_text("[LORRAX]\nnval = 4\nncond = 4\nnband = 8\n"
                    "restart_q_storage = auto\n")
    cfg = LorraxConfig.from_input_file(str(deck),
                                       print_fn=lambda *a, **k: None)
    assert cfg.restart_q_storage_raw == "auto"
    assert cfg.write_restart_tensors is True, "independent axes"


def test_a_typo_dies_at_parse_time_not_after_the_compute(tmp_path,
                                                         monkeypatch):
    """RED TWIN of the parse-time validation.

    ``hartree_source``'s comment says why this shape exists: the alternative
    is a refusal 20 minutes into a 40-node run, after the ISDF stage that
    the restart write is the tail of.  The refusal names the legal set.
    """
    monkeypatch.chdir(tmp_path)
    from gw.gw_config import LorraxConfig

    deck = tmp_path / "cohsex.in"
    deck.write_text("[LORRAX]\nnval = 4\nncond = 4\nnband = 8\n"
                    "restart_q_storage = wedge\n")
    with pytest.raises(ValueError, match=r"restart_q_storage='wedge'"):
        LorraxConfig.from_input_file(str(deck),
                                     print_fn=lambda *a, **k: None)


# ---------------------------------------------------------------------------
# 2. The seam
# ---------------------------------------------------------------------------

def test_the_seam_reads_the_verdict_and_does_not_remeasure():
    """``closure_for_restart`` RETURNS the resolution's own verdict object.

    Identity, not equality: a seam that recomputed an equal verdict would
    pass an equality check and would be a SECOND measurement of one
    question — the thing the resolution point was consolidated to remove.
    """
    from gw.restart_q_storage import closure_for_restart

    res = _closed_and_reduced()
    assert closure_for_restart(res) is res.verdict


def test_the_seam_is_one_line_and_the_stamp_swap_is_marked():
    """The stamp ruling lands as ONE edit — measured on the source.

    The design doc's third stamp point is that there is ONE predicate and
    that phase 3's writer gates on the STAMP rather than on runtime
    geometry.  Phase 2.5 has not landed, so what this branch can do is
    guarantee the swap is a single line in a single function — and say
    where.  A seam whose body grew a second statement is a seam the swap
    would have to re-derive.
    """
    tree = ast.parse(_STORAGE_MOD.read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef)
              and n.name == "closure_for_restart")
    body = [s for s in fn.body if not isinstance(s, ast.Expr)]
    assert len(body) == 1 and isinstance(body[0], ast.Return), (
        f"closure_for_restart must stay a single return statement so the "
        f"stamp swap is one line; found {len(body)} statements")
    src = _STORAGE_MOD.read_text()
    assert "THE ONE LINE THE STAMP SWAP REPLACES" in src, (
        "the swap point must stay marked in the source, not only in a "
        "commit message nobody greps")


def test_the_resolution_asks_the_seam_and_not_the_verdict_directly():
    """AST RATCHET: ``resolve_restart_q_storage`` reads closure ONLY via the seam.

    The bypass this forbids is one character wide — ``resolution.verdict``
    instead of ``closure_for_restart(resolution)`` — and it would leave the
    stamp swap silently ineffective on the arm that matters.  Note the
    matcher looks for the ATTRIBUTE access, so it also fires if a helper
    is added that reaches past the seam.
    """
    tree = ast.parse(_STORAGE_MOD.read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef)
              and n.name == "resolve_restart_q_storage")
    direct = [n for n in ast.walk(fn)
              if isinstance(n, ast.Attribute) and n.attr == "verdict"]
    assert not direct, (
        "resolve_restart_q_storage reads .verdict directly; it must go "
        "through closure_for_restart or the stamp swap misses it")
    calls = [n for n in ast.walk(fn)
             if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "closure_for_restart"]
    assert len(calls) == 1, (
        f"expected exactly one seam call, found {len(calls)}")


# ---------------------------------------------------------------------------
# 3. The resolution truth table
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("req,res_fn,want", [
    # auto follows the answer, both ways.
    ("auto", _closed_and_reduced, "ibz"),
    ("auto", _open_set, "full"),
    ("auto", _closed_but_unreduced, "full"),
    # full NEVER moves, whatever the answer is.
    ("full", _closed_and_reduced, "full"),
    ("full", _open_set, "full"),
    ("full", _closed_but_unreduced, "full"),
    # ibz passes only on the one arm where the wedge exists.
    ("ibz", _closed_and_reduced, "ibz"),
])
def test_the_truth_table(req, res_fn, want):
    from gw.restart_q_storage import resolve_restart_q_storage

    got = resolve_restart_q_storage(req, res_fn(), context="gate")
    assert got.mode == want
    assert got.requested == req
    assert got.store_wedge is (want == "ibz")
    assert got.describe(), "a decision with no reason is a decision nobody audits"


def test_full_does_not_even_ask_the_question():
    """THE CONTROL ARM CANNOT BE MOVED BY THE ANSWER.

    ``full`` is what the Perlmutter A/B measures against, so an arm that
    consulted the closure verdict would not be a control.  Measured by
    handing it a resolution object that RAISES if anything is read off it.
    """
    from gw.restart_q_storage import resolve_restart_q_storage

    class _Landmine:
        def __getattr__(self, name):            # pragma: no cover - the point
            raise AssertionError(
                f"restart_q_storage=full read {name!r} off the resolution")

    got = resolve_restart_q_storage("full", _Landmine(), context="gate")
    assert got.mode == "full"
    assert got.resolution is None


def test_ibz_refuses_a_non_closed_set_and_names_the_ops():
    """RED TWIN 1: the refusal production hits, with the residuals in it.

    Spec §6 gate 1: name the offending ops and residuals, plural, because
    the production failure is 47 of 48 and a message naming only the first
    reads as one bad op.
    """
    from gw.restart_q_storage import resolve_restart_q_storage

    with pytest.raises(ValueError, match=r"NOT CLOSED"):
        resolve_restart_q_storage("ibz", _open_set(), context="V_q restart")


def test_ibz_refuses_an_unreduced_q_path_DIFFERENTLY():
    """RED TWIN 2, and the reason it is a separate cell.

    A closed set on a q-grid the group does not reduce is a completely
    different operational state from a non-closed set: nothing is wrong and
    nothing needs regenerating, there is simply no wedge.  A single refusal
    covering both would send an operator to regenerate a centroid set that
    is already perfect.  The two messages are asserted DISJOINT here.
    """
    from gw.restart_q_storage import resolve_restart_q_storage

    with pytest.raises(ValueError) as exc:
        resolve_restart_q_storage("ibz", _closed_but_unreduced(),
                                  context="V_q restart")
    msg = str(exc.value)
    assert "IS \nnot" not in msg
    assert "did not reduce" in msg
    assert "Regenerate" not in msg, (
        "the closed-but-unreduced arm must not advise regeneration")


def test_ibz_refuses_when_no_resolution_was_taken_at_all():
    """RED TWIN 3: no symmetry information is not a quiet fallback for ``ibz``.

    ``auto`` degrades to ``full`` here (there is nothing to be closed
    ABOUT), but ``ibz`` is an assertion the deck made and it is refused —
    otherwise the one value whose job is to be loud is the one value that
    silently does nothing.
    """
    from gw.restart_q_storage import resolve_restart_q_storage

    with pytest.raises(ValueError, match=r"no centroid permutation"):
        resolve_restart_q_storage("ibz", None, context="V_q restart")
    assert resolve_restart_q_storage(
        "auto", None, context="V_q restart").mode == "full"


def test_an_unrecognised_mode_refuses_at_the_resolution_too():
    """Belt behind the parse-time brace.

    The parser is the gate; this is the assertion that the gate was passed.
    A resolver that quietly treated an unknown string as ``auto`` would
    make the parse-time validation the only thing standing between a typo
    and a changed on-disk format.
    """
    from gw.restart_q_storage import resolve_restart_q_storage

    with pytest.raises(ValueError, match=r"not one of"):
        resolve_restart_q_storage("wedge", _closed_and_reduced(),
                                  context="gate")


def test_the_reason_distinguishes_the_two_ways_auto_falls_back():
    """AUTO'S FALLBACK IS AUDITABLE, which is the whole point of the record.

    The frontier's §9e measurement exists because somebody could tell an
    8x-too-large restart file from an intended one.  ``auto`` resolving
    ``full`` is the normal path on production decks; the reason string is
    what says WHICH normal path.
    """
    from gw.restart_q_storage import resolve_restart_q_storage

    open_reason = resolve_restart_q_storage(
        "auto", _open_set(), context="gate").reason
    unred_reason = resolve_restart_q_storage(
        "auto", _closed_but_unreduced(), context="gate").reason
    assert "NOT orbit-closed" in open_reason
    assert "did not reduce" in unred_reason
    assert open_reason != unred_reason


def test_the_ibz_decision_carries_the_tables_forward():
    """The resolution object is HELD, not distilled to a bool.

    The writer needs the unfold tables (``sym_perm`` / ``L_table`` /
    ``n_sym_spatial``) and the verdict's centroid hash for the stamp.  A
    decision that returned only a mode would force the writer to re-derive
    both — and a re-derived table is a table that can disagree with the one
    the run actually unfolded with.
    """
    from gw.restart_q_storage import resolve_restart_q_storage

    res = _closed_and_reduced()
    got = resolve_restart_q_storage("auto", res, context="gate")
    assert got.resolution is res
    # ...and the full arm deliberately carries nothing, so a caller cannot
    # accidentally reach tables that describe a wedge it did not store.
    assert resolve_restart_q_storage(
        "auto", _open_set(), context="gate").resolution is None


def test_the_legal_set_is_the_one_the_parser_validates_against():
    """ONE tuple, two consumers.  A second list is a second answer."""
    from gw.gw_config import _DEFAULTS
    from gw.restart_q_storage import RESTART_Q_STORAGE

    assert RESTART_Q_STORAGE == ("auto", "full", "ibz")
    assert _DEFAULTS["restart_q_storage"] in RESTART_Q_STORAGE
    src = (_SRC / "gw" / "gw_config.py").read_text()
    assert "RESTART_Q_STORAGE" in src, (
        "gw_config must validate against the module's tuple, not a copy")
    assert np.all(np.array([m == m.strip().lower()
                            for m in RESTART_Q_STORAGE])), (
        "the legal values must already be in normalised form")


# ---------------------------------------------------------------------------
# 5. THE WFN-DERIVED DECISION (owner ruling 2026-08-08 ~13:20)
#
# "if symmetries are not to be used, the wavefunction file should've been
# generated with no symmetries."  The ruling's consequence for the writer is
# that a WFN with no symmetry gets full storage AUTOMATICALLY — not because a
# key says so, but because there is no wedge.
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class _FakeCapture:
    """The three fields ``with_capture``'s demotion reads.

    Deliberately not a ``PreUnfoldCapture``: the demotion must depend on the
    q COUNTS and nothing else, and a fake that carried a tensor would let a
    future implementation start reading one without this cell noticing.
    """

    irr_idx_q: np.ndarray
    q_irr_frac: np.ndarray
    X_ibz: object = None


def _capture(n_q_ibz, n_q_full):
    return _FakeCapture(irr_idx_q=np.zeros(n_q_full, dtype=np.int32),
                        q_irr_frac=np.zeros((n_q_ibz, 3)))


def test_a_wedge_that_is_the_whole_bz_is_demoted_to_full():
    """THE NO-SYMMETRY CASE, AND IT IS ARITHMETIC RATHER THAN A FLAG.

    A WFN with ``ntran = 1`` takes ``SymMaps``' trivial branch: ``irr_idx_q``
    is the identity and the q axis does not reduce.  Such a deck's centroid
    set is still orbit-closed — trivially, under the identity op — so the
    closure question answers yes and the q-grid resolution reaches
    ``use_ibz``.  Nothing there is wrong; there is simply no wedge.

    Storing "the wedge" anyway would write the full BZ under a q_irr stamp
    and a table group, and the format's own table validation would label
    that file ``"full"`` from its SHAPE — so the writer and the reader
    would describe one file two ways.  The demotion applies the READER's
    own test at the writer, which is what makes them agree by construction.
    """
    from gw.restart_q_storage import resolve_restart_q_storage

    d = resolve_restart_q_storage("auto", _closed_and_reduced(),
                                  context="gate")
    assert d.mode == "ibz" and d.store_wedge

    # 5 of 9: a real wedge survives.
    kept = d.with_capture(_capture(5, 9))
    assert kept.mode == "ibz" and kept.store_wedge
    assert kept.capture is not None

    # 9 of 9: no reduction, so no wedge — demoted, and it says why.
    flat = d.with_capture(_capture(9, 9))
    assert not flat.store_wedge, (
        "a q axis that does not reduce has no wedge to store; storing one "
        "would stamp the full BZ as a q_irr file")
    assert flat.mode == "full"
    assert flat.capture is not None, (
        "the capture stays bound — the demotion is about the SHAPE, not "
        "about the hand-off having failed")
    assert "does not reduce" in flat.reason
    assert "9" in flat.reason, "the reason must name the counts it measured"


def test_the_demotion_leaves_the_full_arm_and_the_empty_capture_alone():
    """The two arms the demotion must NOT touch, for two different reasons.

    A ``full`` decision has nothing to demote.  A missing capture is a
    PLUMBING failure — the resolution said wedge and the producer deposited
    nothing — and the writer refuses that by name; silently rewriting it to
    ``full`` would turn a broken hand-off into a quietly larger file.
    """
    from gw.restart_q_storage import resolve_restart_q_storage

    full = resolve_restart_q_storage("full", None, context="gate")
    assert full.with_capture(_capture(9, 9)).mode == "full"
    assert full.with_capture(None).capture is None

    ibz = resolve_restart_q_storage("auto", _closed_and_reduced(),
                                    context="gate")
    none_bound = ibz.with_capture(None)
    assert none_bound.store_wedge, (
        "a missing capture must stay 'ibz' so the writer's named refusal "
        "still fires; demoting here would hide a broken hand-off")
    assert none_bound.capture is None


def test_a_nosym_wfn_really_does_produce_an_unreduced_q_axis():
    """THE FIXTURE BEHIND THE CASE ABOVE, measured rather than assumed.

    ``hbn_cohsex_debug`` is a genuine ``nosym`` WFN (``ntran = 1``, from its
    ``qe/nscf.in``), and it is the sharpest available test vector precisely
    because its CENTROID set is orbit-closed anyway — ``centroid.kmeans_cli``
    recovered a 12-op symmorphic point group from the charge density while
    the WFN stores one op.  A design that keyed on closure would store a
    wedge for this deck; a design that follows the WFN must not.  So the two
    answers differ on a fixture already in the tree, and this cell is what
    proves the premise is real: ntran is 1, and the q axis does not reduce.
    """
    import h5py
    from ffi import _services
    _services.ensure_on_path()
    import symmetry_maps

    deck = (pathlib.Path(__file__).resolve().parents[1]
            / "tests" / "regression" / "hbn_cohsex_debug" / "WFN.h5")
    if not deck.exists():                                # pragma: no cover
        pytest.skip(f"nosym fixture absent: {deck}")
    with h5py.File(deck, "r") as f:
        ntran = int(f["mf_header"]["symmetry"]["ntran"][()])
    assert ntran == 1, (
        f"{deck} is the tree's no-symmetry fixture; it now reports "
        f"ntran={ntran}, so this cell and the ruling's example need "
        f"re-checking against a different deck")

    from file_io.mf_header import read_mf_header
    hdr = read_mf_header(str(deck))
    sym = symmetry_maps.SymMaps(hdr)
    n_q_full = int(np.asarray(sym.irr_idx_q).shape[0])
    n_q_ibz = int(len(set(np.asarray(sym.irr_idx_q).tolist())))
    assert n_q_ibz == n_q_full, (
        f"a nosym WFN's q axis must not reduce; got {n_q_ibz} of "
        f"{n_q_full}.  If this changed, the demotion in with_capture is no "
        f"longer what makes a nosym deck store the full BZ.")


# ---------------------------------------------------------------------------
# 6. THE KEY IS DEPRECATED, AND SAYS SO TO THE DECKS THAT PIN IT
# ---------------------------------------------------------------------------

def test_the_parser_records_which_keys_the_deck_itself_named(tmp_path):
    """A pinned key and an inherited default must be distinguishable.

    They are not distinguishable from the resolved value: a deck pinning
    ``full`` and a deck that never mentions the key both arrive as
    ``"full"``.  The parser knows the difference — it reads ``None`` from the
    section for an absent key — and now records it, because a deprecation
    that cannot tell those apart must either shout at every deck in the tree
    or at none of them.
    """
    from gw.gw_config import read_lorrax_input

    bare = tmp_path / "bare.in"
    bare.write_text("[LORRAX]\nnval = 4\nncond = 4\n")
    pinned = tmp_path / "pinned.in"
    pinned.write_text(
        "[LORRAX]\nnval = 4\nncond = 4\nrestart_q_storage = full\n")

    bare_keys = read_lorrax_input(str(bare))["_deck_named_keys"]
    pin_keys = read_lorrax_input(str(pinned))["_deck_named_keys"]

    assert "restart_q_storage" not in bare_keys
    assert "restart_q_storage" in pin_keys
    assert "nval" in bare_keys and "nval" in pin_keys, (
        "the record must cover every named key, not only the deprecated one")
    # It is not a deck key and must never be mistaken for one.
    from gw.gw_config import _DEFAULTS
    assert "_deck_named_keys" not in _DEFAULTS


def test_the_deprecation_speaks_only_to_a_deck_that_named_the_key():
    """OWNER RULING 2026-08-08 ~13:20: this key is deleted, not defaulted.

    So a deck that pins it is building on something with an end date and is
    told once; a deck that never mentions it hears nothing.  Warning on the
    default path would print for every deck in the tree, which is how a
    deprecation notice becomes noise nobody reads — and the DEFAULT is not
    what is deprecated, the KEY is.
    """
    import types
    from gw.restart_q_storage import _deck_named_the_key

    assert _deck_named_the_key(
        types.SimpleNamespace(raw_input_keys=frozenset({"restart_q_storage"})))
    assert not _deck_named_the_key(
        types.SimpleNamespace(raw_input_keys=frozenset({"nval"})))
    # A config with no record at all — a hand-built params dict, or an older
    # caller — must answer "did not ask" rather than raising or shouting.
    assert not _deck_named_the_key(types.SimpleNamespace())
    assert not _deck_named_the_key(
        types.SimpleNamespace(raw_input_keys=None))


def test_the_key_still_works_because_removal_is_the_owners_step():
    """DEPRECATED-BUT-FUNCTIONAL.  A pinned deck's behaviour is untouched.

    The ruling schedules the deletion; it does not ask this branch to take
    it.  ``si_bse_debug`` pins ``full`` deliberately and its README schedules
    that pin to be deleted WITH the key rather than before it, so the key
    must keep resolving exactly as it did.
    """
    from gw.restart_q_storage import (RESTART_Q_STORAGE,
                                      resolve_restart_q_storage)
    from gw.gw_config import _DEFAULTS

    assert RESTART_Q_STORAGE == ("auto", "full", "ibz"), (
        "the key is still functional; its legal set does not shrink before "
        "the owner-reviewed deletion")
    assert _DEFAULTS["restart_q_storage"] == "auto"
    assert resolve_restart_q_storage(
        "full", _closed_and_reduced(), context="gate").mode == "full"
    assert resolve_restart_q_storage(
        "auto", _closed_and_reduced(), context="gate").mode == "ibz"

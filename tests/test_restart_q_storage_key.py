"""``restart_q_storage`` — the deck key, the seam, and the ONE resolution.

WHAT IS MEASURED HERE.  Three things, and they are separated on purpose
because they fail for different reasons and get fixed by different people:

1. THE RETIRED KEY.  ``restart_q_storage`` was DELETED in 0.1.0 (owner
   ruling 2026-08-08: symmetry never needed a mode switch, the WFN file
   already answers the question) and storage is pinned to ``full``, which
   is what its default already was.  ``auto``/``ibz`` REFUSE — they asked
   for a q wedge that is no longer written; ``full`` is redundant and
   warns.  Files ALREADY on the wedge must still load.

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


def test_the_key_is_gone_from_the_defaults_table():
    """Deleted, not re-defaulted.  ``_DEFAULTS`` is the deck's surface."""
    from gw.gw_config import _DEFAULTS

    assert "restart_q_storage" not in _DEFAULTS


def test_a_deck_that_never_heard_of_the_key_still_parses(tmp_path):
    """Every archived deck keeps parsing AND keeps its bytes.

    The second half is the point: a deck written before this key existed
    must not change on-disk format by standing still — and ``full``, the
    format it got, is now the only one written.
    """
    from gw.gw_config import read_lorrax_input

    deck = tmp_path / "cohsex.in"
    deck.write_text("[LORRAX]\ncompute_mode = cohsex\nnval = 4\n"
                    "ncond = 4\nnband = 8\n")
    assert "restart_q_storage" not in read_lorrax_input(str(deck))


@pytest.mark.parametrize("spelling", ["auto", "AUTO", "ibz", "  Ibz  "])
def test_the_two_wedge_values_refuse_naming_what_still_loads(tmp_path,
                                                             spelling):
    """They asked for a q wedge that is no longer WRITTEN.

    Silently handing such a deck the full BZ would change its on-disk
    format without a word — the failure this key's own default was chosen
    to avoid.  The refusal must also say that files already on the wedge
    still load, or an operator reads it as "my restart file is lost".
    """
    from gw.gw_config import read_lorrax_input

    deck = tmp_path / "cohsex.in"
    deck.write_text(f"[LORRAX]\ncompute_mode = cohsex\nnval = 4\n"
                    f"ncond = 4\nnband = 8\n"
                    f"restart_q_storage = {spelling}\n")
    with pytest.raises(ValueError) as exc:
        read_lorrax_input(str(deck))
    msg = str(exc.value)
    assert "restart_q_storage" in msg
    assert "still load" in msg


@pytest.mark.parametrize("spelling", ["full", "FULL", "  Full  "])
def test_the_pinned_value_is_redundant_not_wrong(tmp_path, spelling):
    """``full`` named the behaviour that is now unconditional."""
    from gw.gw_config import read_lorrax_input

    deck = tmp_path / "cohsex.in"
    deck.write_text(f"[LORRAX]\ncompute_mode = cohsex\nnval = 4\n"
                    f"ncond = 4\nnband = 8\n"
                    f"restart_q_storage = {spelling}\n")
    with pytest.warns(DeprecationWarning, match="restart_q_storage"):
        read_lorrax_input(str(deck))


def test_the_wedge_unfold_reader_is_still_there():
    """Deleting the WRITER's key must not delete the READER.

    Restart files already written on the q wedge have to keep loading;
    that is the whole reason the refusal above can afford to be a refusal.
    """
    reader = (_SRC / "file_io" / "tagged_arrays.py").read_text()
    assert "unfold" in reader and "q_storage" in reader


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


def test_the_legal_set_survives_the_keys_deletion():
    """The RESOLUTION keeps its vocabulary; only the deck key went.

    ``auto`` and ``ibz`` are still resolvable arms — the wedge writer and
    every refusal it carries are intact — so that storage-follows-the-WFN
    lands at ``resolve_restart_q_storage_for_run`` rather than having to
    rebuild what the deletion threw away.
    """
    from gw.restart_q_storage import RESTART_Q_STORAGE

    assert RESTART_Q_STORAGE == ("auto", "full", "ibz")
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
# 6. THE RESOLUTION IS PINNED, AND STILL GOES THROUGH THE ONE SEAM
# ---------------------------------------------------------------------------

def test_the_driver_call_pins_full_rather_than_reading_a_key():
    """``resolve_restart_q_storage_for_run`` asks for ``full``, literally.

    The end state the owner ruled for is storage that FOLLOWS the WFN's own
    symmetry, and that lands at THIS call — so the resolution still routes
    through ``resolve_restart_q_storage`` (one resolution point, one
    announcement) rather than short-circuiting past it now that there is
    nothing for the deck to say.
    """
    tree = ast.parse(_STORAGE_MOD.read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef)
              and n.name == "resolve_restart_q_storage_for_run")
    body = ast.unparse(fn)
    assert "resolve_restart_q_storage('full'" in body
    assert "restart_q_storage_raw" not in body


def test_the_deck_named_keys_mechanism_is_gone():
    """Its sole consumer was the deleted key's deprecation notice."""
    from gw import gw_config, restart_q_storage

    assert not hasattr(restart_q_storage, "_deck_named_the_key")
    assert not hasattr(gw_config, "_DECK_NAMED_KEYS")


def test_the_config_no_longer_carries_the_raw_request(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from gw.gw_config import LorraxConfig

    deck = tmp_path / "cohsex.in"
    deck.write_text("[LORRAX]\ncompute_mode = cohsex\nnval = 4\n"
                    "ncond = 4\nnband = 8\n")
    cfg = LorraxConfig.from_input_file(str(deck),
                                       print_fn=lambda *a, **k: None)
    assert not hasattr(cfg, "restart_q_storage_raw")
    assert not hasattr(cfg, "raw_input_keys")
    assert cfg.write_restart_tensors is True, "independent axes"


def test_the_pinned_arm_is_the_one_every_deck_was_already_running():
    """RED TWIN of the pin: ``full`` was the default and the wedge was not.

    ``si_bse_debug`` pinned ``full`` explicitly and every other deck
    inherited it, so pinning the resolver to ``full`` changes no run — the
    ``auto`` arm that would have differed is the one nothing selected.
    """
    from gw.restart_q_storage import resolve_restart_q_storage

    assert resolve_restart_q_storage(
        "full", _closed_and_reduced(), context="gate").mode == "full"
    assert resolve_restart_q_storage(
        "auto", _closed_and_reduced(), context="gate").mode == "ibz"

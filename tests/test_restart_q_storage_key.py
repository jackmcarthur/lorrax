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

def test_the_key_defaults_to_auto():
    """AUTO IS THE DESIGNED DEFAULT AND IT MOVES BYTES.

    Named in the design doc (phase-3 deliverable 1: "Default auto") and
    owner-visible, because on a deck whose centroid set is orbit-closed —
    ``si_bse_debug`` since fb046e0c — this default changes the on-disk
    restart FORMAT.  A cell here is what makes a silent flip of that default
    impossible.
    """
    from gw.gw_config import _DEFAULTS

    assert _DEFAULTS["restart_q_storage"] == "auto"


def test_a_deck_that_never_heard_of_the_key_gets_auto(tmp_path):
    """Every archived deck keeps parsing; the key lives in ``_DEFAULTS``."""
    from gw.gw_config import read_lorrax_input

    deck = tmp_path / "cohsex.in"
    deck.write_text("[LORRAX]\nnval = 4\nncond = 4\nnband = 8\n")
    assert read_lorrax_input(str(deck))["restart_q_storage"] == "auto"


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

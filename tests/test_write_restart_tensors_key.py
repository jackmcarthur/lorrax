"""``write_restart_tensors`` — the deck key that skips the artifact.

WHAT IT IS FOR, in the owner's framing (SPEC_qirr_restart_tensors.md §7,
carried into DESIGN_symmetry_restart_followup.md).  Nothing in ``gw_jax``
reads ``W0_qmunu`` or ``V_qmunu`` back: they exist for BSE and for future
Σ-builders.  A GW run with no BSE downstream therefore spends measured 4.5 s
of a 19.4–22.6 s warm Si wall (~21%) and 2.01 GB of disk on bytes nobody
opens.  This key lets such a run say so.

IT IS A COMPLEMENT, NEVER AN ALTERNATIVE, TO q_irr STORAGE, and the
distinction is the whole reason the key is framed rather than just added:
it helps only the runs that DISCARD the tensor, and does nothing at all for
the runs that KEEP it — which is exactly the population the q_irr format
work exists for.  Reading it as "we can skip the write instead of doing the
format work" optimises the cheap case and leaves the expensive one alone.

THE DEFAULT IS TRUE and that is deliberate: the default (write or not for
non-BSE runs) is the owner's call, and this branch ships today's behaviour
preserved until it is made.

WHAT GUARDS THE DOWNSTREAM: nothing new.  Suppressing the write makes the
file ABSENT, which is the loudest state the existing BSE refusals
distinguish — ``bse_io._find_restart_file`` raises FileNotFoundError naming
``isdf_tensors_*.h5``, and every W0 consumer already gates on the
``W0_ready`` attr rather than on the dataset's presence
(``tests/test_bse_w0_ready_gate.py``, the April all-zero-screening
mechanism).  The attr arm was already covered; the MISSING-FILE arm was
not, and is covered here.

WHY THE CELLS ARE SHAPED THIS WAY.  The end-to-end perturbation — flip the
key on a fixture run, watch ``tmp/`` stay empty — needs the phdf5 FFI
library, which this checkout's dev box does not have (every restart-writer
test in the tree is red here for that reason).  So the skip is measured at
the exact call sites instead: the ONE decision function, the W0 leg with
the production writer stubbed, and an ``ast`` gate over ``gw_init``'s three
guarded writes.  Same choice, for the same reason, as
``test_bse_w0_ready_gate.py``'s source cell.
"""

from __future__ import annotations

import ast
import os
import pathlib
import types

import pytest

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
_GW_INIT = _SRC / "gw" / "gw_init.py"


@pytest.fixture()
def clean_announcements():
    """Forget memoized announcements before and after; see ffi.gate."""
    from ffi.gate import reset_gate_state

    reset_gate_state()
    yield
    reset_gate_state()


# ---------------------------------------------------------------------------
# The key itself
# ---------------------------------------------------------------------------

def test_the_key_defaults_to_todays_behaviour():
    """DEFAULT PRESERVES TODAY'S BEHAVIOUR — the owner's standing rule.

    The default (write or not for non-BSE runs) is the owner's decision,
    made on numbers that are not in yet.  A branch that shipped the key
    defaulting to ``false`` would have taken that decision by landing.
    """
    from gw.gw_config import _DEFAULTS

    assert _DEFAULTS["write_restart_tensors"] is True


def test_a_deck_that_never_heard_of_the_key_still_writes(tmp_path):
    """Every archived deck keeps working, unchanged.

    ``_DEFAULTS`` is the fallback for a key the deck omits, so this is
    really a test that the key was added to ``_DEFAULTS`` and not only to
    the dataclass — which would raise on every existing input file.
    """
    from gw.gw_config import read_lorrax_input

    deck = tmp_path / "cohsex.in"
    deck.write_text("[LORRAX]\ncompute_mode = cohsex\nnval = 4\nncond = 4\nnband = 8\n")
    assert read_lorrax_input(str(deck))["write_restart_tensors"] is True


@pytest.mark.parametrize("spelling,want",
                         [("false", False), ("true", True),
                          ("no", False), ("yes", True),
                          ("0", False), ("1", True)])
def test_the_key_parses_the_decks_boolean_grammar(tmp_path, spelling, want):
    """The same grammar every other boolean deck key uses.

    ``configparser.getboolean`` is what ``restart`` / ``do_G0`` /
    ``bispinor`` are read with, so a hand-typed ``no`` or ``yes`` means
    what it says here too.  A key with its own private grammar is worse
    than one with a single wrong grammar, because then the behaviour
    depends on which spelling the operator reached for.
    """
    from gw.gw_config import read_lorrax_input

    deck = tmp_path / "cohsex.in"
    deck.write_text(
        f"[LORRAX]\ncompute_mode = cohsex\nnval = 4\nncond = 4\nnband = 8\n"
        f"write_restart_tensors = {spelling}\n")
    got = read_lorrax_input(str(deck))["write_restart_tensors"]
    assert got is want


def test_the_key_is_not_an_unknown_deck_key(tmp_path, capsys):
    """``strict_keys = true`` must ACCEPT it.

    The deck-hygiene check refuses anything that is not in ``_DEFAULTS``.
    A key wired into the dataclass but not the defaults table would parse
    on a lenient deck and refuse on a strict one — which is the worst
    place to find out.
    """
    from gw.gw_config import read_lorrax_input

    deck = tmp_path / "cohsex.in"
    deck.write_text(
        "[LORRAX]\ncompute_mode = cohsex\nnval = 4\nncond = 4\nnband = 8\n"
        "strict_keys = true\nwrite_restart_tensors = false\n")
    assert read_lorrax_input(str(deck))["write_restart_tensors"] is False


def test_the_key_reaches_the_config_object_the_driver_holds(tmp_path,
                                                            monkeypatch):
    """Through ``LorraxConfig.from_input_file``, not just ``_DEFAULTS``.

    The parse has three layers — the defaults table, the params dict, and
    the frozen dataclass the driver actually reads — and a key wired into
    two of them reads as its default forever with nothing to show for it.
    ``gw_init`` and ``gw_output`` consult the DATACLASS, so that is what
    this asserts.
    """
    monkeypatch.chdir(tmp_path)
    from gw.gw_config import LorraxConfig

    deck = tmp_path / "cohsex.in"
    deck.write_text("[LORRAX]\ncompute_mode = cohsex\nnval = 4\nncond = 4\nnband = 8\n"
                    "write_restart_tensors = false\n")
    cfg = LorraxConfig.from_input_file(str(deck),
                                       print_fn=lambda *a, **k: None)
    assert cfg.write_restart_tensors is False
    assert cfg.restart is True, "the two keys are independent axes"


# ---------------------------------------------------------------------------
# The decision point, and its announcement
# ---------------------------------------------------------------------------

def _cfg(**kw):
    base = dict(write_restart_tensors=True)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_the_decision_announces_once_and_only_when_it_is_off(
        capsys, clean_announcements):
    """One line per run, and none at all on the default path.

    Four calls because the key gates four writes (the V_q/G0/E_nk flush,
    the ψ append, the centroid stamp, and the W0 + head write).  Asking
    the config at each site would be four chances to disagree; announcing
    at each would be four lines saying one thing.
    """
    from gw.gw_output import restart_tensor_writes_enabled

    path = "/nonexistent/tmp/isdf_tensors_960.h5"
    off = _cfg(write_restart_tensors=False)
    assert [restart_tensor_writes_enabled(off, path) for _ in range(4)] \
        == [False] * 4
    out = capsys.readouterr().out
    assert out.count("write_restart_tensors = false") == 1, out
    assert "isdf_tensors_960.h5" in out
    assert "SKIPPING" in out
    # It names what is lost, dataset by dataset — an operator reading this
    # in a log six months later cannot be expected to know the list.
    for ds in ("V_qmunu", "G0_mu_nu", "enk_full", "psi_full_y", "W0_qmunu"):
        assert ds in out, out
    # And it names the downstream consequence, because the person who set
    # the key is often not the person who later runs BSE here.
    assert "BSE" in out and "refuse" in out


def test_the_default_path_is_silent_and_writes(capsys, clean_announcements):
    """RED TWIN.  Nothing is announced when nothing is suppressed."""
    from gw.gw_output import restart_tensor_writes_enabled

    path = "/nonexistent/tmp/isdf_tensors_960.h5"
    assert restart_tensor_writes_enabled(_cfg(), path) is True
    assert capsys.readouterr().out.strip() == ""


def test_a_config_object_predating_the_key_still_writes():
    """``getattr(..., True)``: absence means the old behaviour.

    Several tools in this tree hand ``persist_w0_and_head`` a stub config
    rather than a parsed ``LorraxConfig``.  A missing attribute must mean
    "write", never "skip" — the failure direction of the other choice is a
    silently absent artifact.
    """
    from gw.gw_output import restart_tensor_writes_enabled

    bare = types.SimpleNamespace()
    assert restart_tensor_writes_enabled(bare, "/tmp/x.h5") is True


# ---------------------------------------------------------------------------
# The W0 leg: the write is skipped, and the file is not created
# ---------------------------------------------------------------------------

class _Head:
    wcoul0 = complex(1.0, 0.0)
    vc0 = complex(2.0, 0.0)


class _Resolver:
    def at(self, omega):
        return _Head()


def _persist(tmp_path, monkeypatch, *, key, precreate):
    """Run ``persist_w0_and_head`` with the production writers stubbed.

    Returns ``(calls, path)``.  The stubs stand in for
    ``file_io.write_w0_qmunu_to_h5`` / ``write_head_scalars_to_h5``, which
    go through the phdf5 FFI and cannot run on a box without the library —
    so what is measured here is whether they are REACHED, which is the
    thing the key controls.
    """
    import file_io
    from gw.gw_config import ComputeMode
    from gw.gw_output import persist_w0_and_head

    calls = []
    monkeypatch.setattr(file_io, "write_w0_qmunu_to_h5",
                        lambda *a, **k: calls.append("w0"), raising=True)
    monkeypatch.setattr(file_io, "write_head_scalars_to_h5",
                        lambda *a, **k: calls.append("head"), raising=True)
    path = tmp_path / "isdf_tensors_144.h5"
    if precreate:
        path.write_bytes(b"")
    # THE REAL ENUM, not a stand-in carrying one attribute.  This function's
    # own control arm (below) exists to catch a guard that returns early for
    # an unrelated reason, and a stub config missing an attribute is the
    # example it names — so the mode here is the actual COHSEX member, which
    # answers ``is_dynamic``, ``ppm_model`` and everything else the writer
    # asks it exactly as a run would.
    cfg = types.SimpleNamespace(
        write_restart_tensors=key, compute_mode=ComputeMode.COHSEX)
    persist_w0_and_head(
        object(), tensors_filename=str(path), head_resolver=_Resolver(),
        config=cfg, meta=types.SimpleNamespace(n_rmu=4), mesh_xy=None,
        print_fn=lambda *a, **k: None)
    return calls, path


def test_w0_is_written_on_the_default_path(tmp_path, monkeypatch,
                                           clean_announcements):
    """The control arm.  Without it, the cell below proves nothing.

    A guard that returned early for an unrelated reason — a stub config
    missing an attribute, say — would make the "skipped" assertion pass
    while the key did nothing.
    """
    calls, _ = _persist(tmp_path, monkeypatch, key=True, precreate=True)
    assert calls == ["w0", "head"]


def test_w0_is_skipped_and_no_file_appears_when_the_key_is_off(
        tmp_path, monkeypatch, capsys, clean_announcements):
    """THE PERTURBATION: same call, key flipped, write skipped, no file.

    Both halves are asserted.  "The writer was not called" is the
    behaviour; "the path does not exist" is what an operator would check,
    and the two can come apart if some other line in the function touches
    the file first.
    """
    calls, path = _persist(tmp_path, monkeypatch, key=False, precreate=False)
    assert calls == []
    assert not path.exists(), f"{path} was created on the skip path"
    out = capsys.readouterr().out
    assert out.count("write_restart_tensors = false") == 1, out


def test_the_skip_survives_a_pre_existing_file(tmp_path, monkeypatch,
                                               clean_announcements):
    """``restart = true`` + key off: the old file is left ALONE.

    A restart run reuses tensors it did not write.  Re-stamping a fresh
    W0 into a file the deck said to leave alone would be the key doing the
    opposite of what it says on exactly the runs most likely to set it.
    """
    calls, path = _persist(tmp_path, monkeypatch, key=False, precreate=True)
    assert calls == []
    assert path.exists() and path.stat().st_size == 0


# ---------------------------------------------------------------------------
# The gw_init leg: an AST gate, because the e2e run needs the FFI
# ---------------------------------------------------------------------------

def _prepare_isdf_tree():
    with open(_GW_INIT, encoding="utf-8") as fh:
        mod = ast.parse(fh.read(), filename=str(_GW_INIT))
    for node in ast.walk(mod):
        if isinstance(node, ast.FunctionDef) and \
                node.name == "prepare_isdf_and_wavefunctions":
            return node
    raise AssertionError("gw_init.prepare_isdf_and_wavefunctions not found")


def _guarded_by_write_restart(fn, node):
    """Is ``node`` inside an ``if _write_restart:`` within ``fn``?"""
    for cand in ast.walk(fn):
        if not isinstance(cand, ast.If):
            continue
        t = cand.test
        if not (isinstance(t, ast.Name) and t.id == "_write_restart"):
            continue
        for sub in ast.walk(cand):
            if sub is node:
                return True
    return False


def test_every_restart_write_in_gw_init_is_behind_the_one_decision():
    """No unguarded write may survive, and there must be writes to guard.

    A partially-written restart file is worse than none: the band-window
    attrs, the n_rmu attr and the ``W0_ready`` flag would all pass on a
    file missing ``psi_full_y``, and the BSE loader would trip over the
    absence one dataset later, several stages downstream of the decision
    that caused it.  So the key gates the flush, the ψ append and the
    centroid-hash stamp together, off one local taken once.
    """
    fn = _prepare_isdf_tree()
    writes, guarded = 0, 0
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        nm = (f.id if isinstance(f, ast.Name)
              else f.attr if isinstance(f, ast.Attribute) else "")
        if nm != "write_restart_state_to_h5":
            continue
        writes += 1
        guarded += bool(_guarded_by_write_restart(fn, node))
    assert writes >= 2, (
        f"expected the two write_restart_state_to_h5 calls (the V_q/G0/enk "
        f"flush and the psi append); found {writes}")
    assert guarded == writes, (
        f"{writes - guarded} of {writes} restart writes in "
        f"prepare_isdf_and_wavefunctions are not behind "
        f"``if _write_restart:`` — write_restart_tensors = false would "
        f"leave a partially written file, which every downstream guard "
        f"accepts")


def test_the_decision_is_taken_once_not_per_write():
    """One ``_write_restart = restart_tensor_writes_enabled(...)``.

    Two calls would be two chances to disagree (and, before the dedup,
    two announcements); re-reading ``cfg.write_restart_tensors`` inline at
    each site would be three.
    """
    fn = _prepare_isdf_tree()
    assigns = [n for n in ast.walk(fn)
               if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "_write_restart"
                       for t in n.targets)]
    assert len(assigns) == 1, (
        f"_write_restart is assigned {len(assigns)} times; the decision is "
        f"taken once")
    call = assigns[0].value
    assert isinstance(call, ast.Call)
    nm = (call.func.id if isinstance(call.func, ast.Name)
          else getattr(call.func, "attr", ""))
    assert nm == "restart_tensor_writes_enabled", (
        f"_write_restart is assigned from {nm!r}, not from the one decision "
        f"function — an inline ``cfg.write_restart_tensors`` here would skip "
        f"the announcement")


# ---------------------------------------------------------------------------
# The downstream guard: the MISSING-FILE arm the W0_ready gate does not cover
# ---------------------------------------------------------------------------

def test_bse_refuses_by_name_when_the_restart_file_is_absent(tmp_path):
    """The other half of ``tests/test_bse_w0_ready_gate.py``.

    That file pins the ``W0_ready`` arm — a restart file that is PRESENT
    and carries the zero placeholder.  ``write_restart_tensors = false``
    produces the state it does not cover: no file at all.  The refusal
    already existed; nothing measured it, because every caller of
    ``_find_restart_file`` in the suite points at a fixture that has one.

    The message must name the glob, not just "not found": the file is
    namespaced by centroid count (``isdf_tensors_<n_rmu>.h5``) and an
    operator who set this key needs to be told which artifact is missing
    and from where.
    """
    from bse import bse_io

    deck = tmp_path / "bse.in"
    deck.write_text("[LORRAX]\ncompute_mode = cohsex\nnval = 4\n")
    (tmp_path / "tmp").mkdir()
    with pytest.raises(FileNotFoundError) as exc:
        bse_io._find_restart_file(str(deck))
    msg = str(exc.value)
    assert "isdf_tensors_*.h5" in msg, msg
    assert str(tmp_path) in msg or os.path.basename(str(tmp_path)) in msg, msg


def test_the_missing_file_arm_is_distinct_from_the_ambiguous_arm(tmp_path):
    """RED TWIN: a directory that HAS files takes the other branch.

    ``_find_restart_file`` has two refusals — none found, and more than
    one found with different centroid counts.  A cell that only ever saw
    an empty directory could not tell a broken glob from a working one.
    """
    from bse import bse_io

    deck = tmp_path / "bse.in"
    deck.write_text("[LORRAX]\ncompute_mode = cohsex\nnval = 4\n")
    tmpdir = tmp_path / "tmp"
    tmpdir.mkdir()
    (tmpdir / "isdf_tensors_144.h5").write_bytes(b"")
    found = bse_io._find_restart_file(str(deck))
    assert found.endswith("isdf_tensors_144.h5")

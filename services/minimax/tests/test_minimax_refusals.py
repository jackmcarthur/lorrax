"""The six refusals, each with the case where it returns FALSE.

``SERVICE_FORM.md:46`` — *every new check ships with the case where it
returns FALSE, no exceptions* — and this file is where that rule does the
most work, because every one of these refusals replaces something that
used to be silent.  A cell asserting "the door refuses X" is worth very
little on its own: a door that refused everything would pass it.  The
paired cell is what makes it evidence.

===  ==========================  =========================================
 #   refusal                     the FALSE case shipped beside it
===  ==========================  =========================================
 F1  ``NoCertifiedTable``        a request inside the catalog returns
 F2  ``AmplificationCap``        a κ₀ = 1.19 complex_laplace entry is clean
 F3  ``UnknownTarget``           every vocabulary member resolves
 F4  ``CatalogUnavailable`` /    a healthy bundle loads
     ``TableUnreadable`` /
     ``CatalogCorrupt``
 F5  ``UncertifiedSolveRefused`` with the hatch set, it solves and announces
 F6  ``SamplingUnsupported``     the three live cells resolve
===  ==========================  =========================================
"""

from __future__ import annotations

import numpy as np
import pytest

import minimax as M
from minimax import _catalog as C


# ---------------------------------------------------------------------------
#  F1 — no certified table
# ---------------------------------------------------------------------------

def test_f1_a_request_outside_the_catalog_refuses_and_names_both_levers():
    """A_dim = 83 is the G2 gate's own request, and the catalog stops at 60.

    The message shape is the ruling, not a nicety: a refusal that does not
    tell you how to make it go away is a crash with better manners.  It
    must carry the nearest certified artifact (so you know where the edge
    is), the PHYSICS lever (so a deck can move inside it) and the
    GENERATOR lever (so the catalog can be extended to cover it).
    """
    with pytest.raises(M.NoCertifiedTable) as excinfo:
        M.lookup(family="crossing", target="hgl", range_value=83.0,
                 error_bound=1.0e-6, n_max=500, eps_q=1.0e-3)
    text = str(excinfo.value)
    assert "A_dim=83" in text
    assert "nearest certified below: A_dim=60" in text, text
    assert "omega_max" in text and "xi" in text, text
    assert "generate_minimax_assets.py" in text, text


def test_f1_false_case_a_request_inside_the_catalog_returns_with_provenance():
    """The FALSE case.  R = 10 at the 1e-6 tier is the census's most-loaded
    table (47 of the measured requests) and it must come back, with its
    origin attached."""
    q = M.lookup(family="noncrossing", target="inverse", range_value=10.0,
                 error_bound=1.0e-6, n_max=64)
    assert q.node_count == 7
    assert q.provenance.source == "shipped"
    assert q.provenance.catalog_entry == \
        "noncrossing/noncrossing_R_10p000000_eps_1p0em06.npz"
    assert q.provenance.table_hash.startswith("sha256:")


def test_f1_names_the_structural_hole_when_a_family_ships_nothing():
    """``noncrossing_imag`` is not a sparse catalog — it is an empty one.

    This is R1's finding, and the refusal has to say which kind of miss it
    is: "your request is past the edge" and "there is no edge because
    there is nothing" call for different actions, and the design's whole
    staging argument turns on the difference.
    """
    with pytest.raises(M.NoCertifiedTable) as excinfo:
        M.lookup(family="noncrossing_imag", target="inverse_imag",
                 range_value=42.68, error_bound=1.0e-6, n_max=64)
    text = str(excinfo.value)
    assert "ZERO shipped entries" in text, text
    assert "structural hole" in text, text
    assert "nearest certified below: none" in text, text


def test_f1_the_selection_rule_rounds_the_range_up_and_the_error_down():
    """The conservative convention, carried verbatim across the boundary.

    A request between two tabulated ranges takes the LARGER range (the
    requested interval is then a subset of the tabulated one, which is the
    safe direction) and the LOOSEST bound still at least as strict as
    asked.  The census's sub-10 requests all round up to the R = 10 table
    this way, and 51 of its 54 measured requests depend on this exact
    behaviour being unchanged.
    """
    q = M.lookup(family="noncrossing", target="inverse", range_value=15.0,
                 error_bound=1.0e-6, n_max=64)
    assert q.provenance.catalog_entry == \
        "noncrossing/noncrossing_R_21p544347_eps_1p0em06.npz"
    # ... and the error tier is the loosest acceptable, not the strictest
    # available: the 2e-7 table at the same R exists and is NOT chosen.
    assert q.error_bound == 1.0e-6


def test_f1_a_node_budget_that_no_table_meets_refuses():
    """``n_max`` is a hard filter, not a preference."""
    with pytest.raises(M.NoCertifiedTable):
        M.lookup(family="noncrossing", target="inverse", range_value=1.0e5,
                 error_bound=1.0e-6, n_max=4)


# ---------------------------------------------------------------------------
#  F2 — amplification cap
# ---------------------------------------------------------------------------

def _cl_view():
    """The v2 catalog, which is the only shipped one carrying κ₀."""
    return C.catalog_view("catalog_complex_laplace.json")


def test_f2_a_table_above_the_declared_cap_refuses():
    """The cap is DATA, and the refusal reads it out of the artifact.

    κ₀ ≤ 2 ships normally, 2–4 is a versioned exception, above 4 is
    rejected — the theory plan owns those numbers and the generator
    stamps them into ``shipping_rule``.  A service-side constant would be
    a second opinion, so the check is against the catalog's own threshold
    and this cell forges an entry past it.
    """
    view = _cl_view()
    assert view.shipping_rule["rejected_above"] == 4.0
    bad = C.parse_entry(
        dict(view.entries[0].raw, kappa0=9.5), 0,
        catalog_name="forged")
    with pytest.raises(M.AmplificationCap) as excinfo:
        from minimax.door import _check_amplification   # noqa: PLC0415
        _check_amplification(bad, view)
    assert "9.5" in str(excinfo.value)
    assert "rejection threshold of 4" in str(excinfo.value)


def test_f2_false_case_a_well_conditioned_table_is_clean():
    """The FALSE case.  Every shipped complex_laplace entry measures κ₀
    between 1.08 and 1.35 — the cap never binds on the real bundle, which
    is the finding the imag-table campaign landed and the reason F2 is a
    guard rather than a routine event."""
    from minimax.door import _check_amplification       # noqa: PLC0415
    view = _cl_view()
    for entry in view.entries:
        assert entry.kappa0 is not None
        assert entry.kappa0 < 2.0, (entry.file, entry.kappa0)
        _check_amplification(entry, view)               # must not raise


def test_f2_is_silent_where_the_artifact_declares_nothing():
    """The v1 catalog records no κ₀ at all, so there is nothing to check.

    The door says so in the PROVENANCE rather than inventing a number,
    and this cell pins that: a missing certification record must not be
    read as a passing one.
    """
    q = M.lookup(family="noncrossing", target="inverse", range_value=10.0,
                 error_bound=1.0e-6, n_max=64)
    assert q.kappa0 is None
    assert q.provenance.certified is False
    assert "UNCERTIFIED" in q.one_line()


# ---------------------------------------------------------------------------
#  F3 — unknown target / unknown family / unknown selector
# ---------------------------------------------------------------------------

def test_f3_an_undeclared_target_refuses_and_lists_the_vocabulary():
    with pytest.raises(M.UnknownTarget) as excinfo:
        M.lookup(family="noncrossing", target="quadratic",
                 range_value=10.0, error_bound=1.0e-6, n_max=64)
    assert "Declared targets" in str(excinfo.value)


def test_f3_a_target_the_family_cannot_serve_refuses():
    """'hgl' is a real target and 'noncrossing' is a real family; the PAIR
    is the error.  Accepting it would serve a 1/x table to a caller who
    asked for a sign-regularization, which is a wrong answer rather than a
    missing one."""
    with pytest.raises(M.UnknownTarget) as excinfo:
        M.lookup(family="noncrossing", target="hgl",
                 range_value=10.0, error_bound=1.0e-6, n_max=64)
    assert "cannot serve" in str(excinfo.value)


def test_f3_an_unknown_selector_refuses_rather_than_being_ignored():
    """A silently-dropped selector is how you serve a table fitted to a
    different function — which is precisely the hazard that keeps the
    eighteen staged complex_laplace entries unwired."""
    with pytest.raises(M.UnknownTarget) as excinfo:
        M.lookup(family="noncrossing", target="inverse", range_value=10.0,
                 error_bound=1.0e-6, n_max=64, beta=3.0)
    assert "beta" in str(excinfo.value)


def test_f3_false_case_every_declared_vocabulary_member_resolves():
    """The FALSE case, and it is also the anti-rot check on the tables:
    a target declared in :data:`minimax.TARGETS` with no family, or a
    family naming a target that is not declared, would make the vocabulary
    a lie that nothing catches."""
    for name, spec in M.FAMILIES.items():
        assert spec.target in M.TARGETS, (name, spec.target)
        assert spec.character in M.CHARACTERS, (name, spec.character)
    # every target is reachable from some family, except the one the
    # design registers as a hole on purpose
    served = {f.target for f in M.FAMILIES.values()} | {"hgl", "fermi"}
    assert set(M.TARGETS) - served == set(), set(M.TARGETS) - served


# ---------------------------------------------------------------------------
#  F4 — the bundle itself.  R2's first four rows.
# ---------------------------------------------------------------------------

def test_f4a_a_missing_catalog_refuses_and_names_the_resolved_path():
    """Was ``except Exception: return None``, twice.

    A missing bundle and a healthy one used to be the same event to every
    caller, and the caller's reading of that event was "solve it yourself,
    for four minutes, uncertified".
    """
    with pytest.raises(M.CatalogUnavailable) as excinfo:
        M.load_catalog_dict("catalog_that_is_not_there.json")
    assert "not in the bundle" in str(excinfo.value)
    assert "catalog_that_is_not_there.json" in str(excinfo.value)


def test_f4a_invalid_json_refuses_differently_from_a_missing_file(tmp_path,
                                                                  monkeypatch):
    """"absent" and "present but corrupt" are different defects and now
    produce different messages, which is the whole content of R2's first
    row."""
    root = tmp_path / "minimax_assets"
    root.mkdir()
    (root / "catalog.json").write_text("{not json at all")
    monkeypatch.setattr(C, "_asset_root", lambda: root)
    C.clear_caches()
    with pytest.raises(M.CatalogUnavailable) as excinfo:
        M.load_catalog_dict("catalog.json")
    assert "not valid JSON" in str(excinfo.value)


def test_f4b_an_unreadable_payload_names_the_file_and_the_keys_found(
        tmp_path, monkeypatch):
    """Was ``except Exception: return None``.

    "unreadable" and "readable but missing ``alpha``" are different
    defects with the same old symptom, so the message carries the npz key
    set that was actually there.
    """
    root = tmp_path / "minimax_assets"
    (root / "noncrossing").mkdir(parents=True)
    np.savez(root / "noncrossing" / "bad.npz",
             tau=np.array([1.0]), max_error=np.array(1e-6))
    monkeypatch.setattr(C, "_asset_root", lambda: root)
    C.clear_caches()
    entry = C.parse_entry(
        {"family": "noncrossing", "range_max": 10.0, "error_bound": 1e-6,
         "node_count": 1, "file": "noncrossing/bad.npz"}, 0,
        catalog_name="forged")
    with pytest.raises(M.TableUnreadable) as excinfo:
        C.load_table(entry)
    text = str(excinfo.value)
    assert "bad.npz" in text and "'alpha'" in text
    assert "tau" in text and "max_error" in text, text


def test_f4c_a_malformed_entry_refuses_the_catalog_rather_than_skipping_it():
    """THE ROW THAT CHANGED MEANING, not just volume.

    A malformed entry used to ``continue`` — i.e. be treated as ABSENT —
    so the selection rule went on to pick something else, or found nothing
    and fell through to an uncertified solve.  A malformed entry means the
    ARTIFACT IS CORRUPT.  The refusal names the index and the field.
    """
    catalog = {"tables": [
        {"family": "noncrossing", "range_max": 10.0, "error_bound": 1.0e-6,
         "node_count": 7, "file": "a.npz"},
        {"family": "noncrossing", "range_max": "not a number",
         "error_bound": 1.0e-6, "node_count": 7, "file": "b.npz"},
    ]}
    with pytest.raises(M.CatalogCorrupt) as excinfo:
        M.parse_catalog(catalog, catalog_name="forged")
    text = str(excinfo.value)
    assert "entry 1" in text and "range_max" in text


def test_f4c_a_missing_required_field_refuses_and_lists_what_was_there():
    catalog = {"tables": [
        {"family": "noncrossing", "range_max": 10.0, "error_bound": 1.0e-6,
         "file": "a.npz"},
    ]}
    with pytest.raises(M.CatalogCorrupt) as excinfo:
        M.parse_catalog(catalog, catalog_name="forged")
    text = str(excinfo.value)
    assert "node_count" in text and "Entry keys present" in text


def test_f4c_a_present_but_unparseable_eps_q_refuses():
    """The fourth handler: ``except Exception: continue`` around the
    ``eps_q`` comparison.  ``float(None)`` on an entry that has no eps_q
    was legitimate absence; ``float('wide')`` on one that declares a bad
    eps_q is corruption, and the two were indistinguishable."""
    with pytest.raises(M.CatalogCorrupt) as excinfo:
        M.parse_catalog({"tables": [
            {"family": "crossing", "target_kind": "hgl", "range_max": 20.0,
             "error_bound": 1.0e-6, "node_count": 26, "eps_q": "wide",
             "file": "c.npz"}]}, catalog_name="forged")
    assert "eps_q" in str(excinfo.value)


def test_f4c_absence_of_an_optional_field_is_not_corruption():
    """The distinction, from the other side.

    A ``noncrossing`` entry carries no ``eps_q`` and never did.  Parsing
    must accept that and the SELECTION rule must then treat it as
    non-matching for a request that specifies one — absence stays a skip,
    which is where the old behaviour was right.
    """
    entries = M.parse_catalog({"tables": [
        {"family": "noncrossing", "range_max": 10.0, "error_bound": 1.0e-6,
         "node_count": 7, "file": "a.npz"}]}, catalog_name="forged")
    assert entries[0].eps_q is None and entries[0].target_kind is None
    assert M.select_entry(entries, "noncrossing", range_value=5.0,
                          target_error=1.0e-6, max_nodes=64,
                          eps_q=1.0e-3) is None
    assert M.select_entry(entries, "noncrossing", range_value=5.0,
                          target_error=1.0e-6, max_nodes=64) is entries[0]


def test_f4_false_case_the_shipped_bundle_loads_and_every_entry_parses():
    """The FALSE case for all of F4: the real artifact is healthy.

    31 entries in ``catalog.json``, 54 in the complex_laplace one and 29
    in the damped_line one, every payload resolvable and every field
    typed.  If this cell ever goes red the bundle is broken, which is
    exactly the event the four refusals above exist to report instead of
    swallow.

    THE COMPLEX_LAPLACE COUNT MOVED 18 -> 54 AND THE PIN IS SUPPOSED TO
    NOTICE.  The first campaign swept a beta grid taken from the request
    census's three-decimal DISPLAY; the full-precision campaign regenerated
    at the decks' own omega-hats and added the 1e-12 fit-stage tier, which
    is 36 more entries.  A count pin that did not have to be touched for
    that would not be pinning anything.
    """
    view = M.catalog()
    assert len(view) == 31
    assert view.schema_version == 1
    for entry in view.entries:
        tau, alpha, err, _k, h = C.load_table(entry)
        assert tau.shape == alpha.shape == (entry.node_count,)
        assert np.isfinite(err) and err > 0.0
        assert h.startswith("sha256:")
    cl = _cl_view()
    assert len(cl) == 54 and cl.schema_version == 2


# ---------------------------------------------------------------------------
#  F6 — the 2x2's empty cell
# ---------------------------------------------------------------------------

def test_f6_the_strip_cell_refuses_by_name():
    """``gw/screening.py:527-531``'s refusal, moved somewhere that can say
    what is missing and what would create it.

    Both parts of z nonzero is where the whole MPA fit stage lives, and
    the family that serves it does not exist.  Refusing here means the
    refusal happens before any physics runs, from declarative data,
    rather than somewhere inside a kernel.
    """
    with pytest.raises(M.SamplingUnsupported) as excinfo:
        M.family_for_character("strip")
    text = str(excinfo.value)
    assert "damped_line" in text
    assert "WP9" in text or "campaign" in text, text


def test_f6_false_case_the_three_live_cells_resolve():
    """The FALSE case, and the 2×2 read out loud."""
    assert M.family_for_character("static") == "noncrossing"
    assert M.family_for_character("real") == "crossing"
    assert M.family_for_character("imag") == "noncrossing_imag"


def test_f6_a_character_that_is_not_one_of_the_four_refuses():
    with pytest.raises(M.SamplingUnsupported) as excinfo:
        M.family_for_character("diagonal")
    assert "not an analytic character" in str(excinfo.value)


def test_the_wired_beta_family_still_refuses_a_beta_blind_lookup():
    """``complex_laplace`` is WIRED now, and a β-blind lookup still cannot
    reach one of its tables.

    This cell replaces the one that asserted ``wired is False``, and it
    is deliberately the same property under a rule that can also say yes.
    The reason the family stayed unwired was never "no rule": it was that
    the only rule available matched on three axes that round safely, and
    β rounds neither way — ``1/(u - iβ)`` is a different function at every
    β — so a β-blind match serves a table fitted to something else.
    ``minimax.beta_selector`` is that axis, and the door routes this
    family to it.  What must remain impossible is asking for one of these
    eighteen entries WITHOUT saying which β and which clause you meant, so
    that is what is asserted: not a refusal because nothing can serve it,
    but a refusal because the request is under-specified.  The clause
    matters as much as the number — the two clauses of the envelope
    overlap around β ≈ 0.6, so neither can be inferred from the other.
    """
    assert M.FAMILIES["complex_laplace"].shipped is True
    assert M.FAMILIES["complex_laplace"].wired is True

    with pytest.raises(M.UnknownTarget) as no_beta:
        M.lookup(family="complex_laplace", target="complex_laplace",
                 range_value=21.544346900318832, error_bound=1.0e-6,
                 n_max=64)
    assert "no beta" in str(no_beta.value)
    assert "does not round" in str(no_beta.value)

    with pytest.raises(M.UnknownTarget) as no_clause:
        M.lookup(family="complex_laplace", target="complex_laplace",
                 range_value=21.544346900318832, error_bound=1.0e-6,
                 n_max=64, beta=0.5)
    assert "no beta_clause" in str(no_clause.value)

    # The FALSE case: fully specified, the same request is served.  A
    # refusal cell whose positive twin is missing cannot tell "correctly
    # strict" from "broken".
    quad = M.lookup(family="complex_laplace", target="complex_laplace",
                    range_value=21.544346900318832, error_bound=1.0e-6,
                    n_max=64, beta=5.836,
                    beta_clause=M.beta_selector.HEIGHT)
    assert quad.family == "complex_laplace"
    assert quad.provenance.source == "shipped"
    assert quad.node_count > 0


# ---------------------------------------------------------------------------
#  nearest_certified — the one function contracted never to raise
# ---------------------------------------------------------------------------

def test_nearest_certified_never_raises_even_on_nonsense():
    """Its only caller is a refusal message.  An exception raised while
    building the explanation for another exception replaces a useful
    refusal with a confusing one, so the contract is absolute."""
    assert M.nearest_certified(family="not_a_family", target="inverse",
                               range_value=1.0, error_bound=1.0,
                               n_max=1) is None
    assert M.nearest_certified(family="noncrossing", target="inverse",
                               range_value=1.0, error_bound=1.0e-6,
                               n_max=64) is None


def test_nearest_certified_finds_the_edge_of_the_certified_region():
    q = M.nearest_certified(family="crossing", target="hgl",
                            range_value=83.0, error_bound=1.0e-6,
                            n_max=500, eps_q=1.0e-3)
    assert q is not None and q.range_value == 60.0
    assert q.provenance.source == "shipped"


def test_the_refusal_text_survives_a_catalog_with_no_nearest_entry():
    """A refusal must still be a message when there is nothing to offer."""
    from minimax.refusals import no_certified_table_text  # noqa: PLC0415
    text = no_certified_table_text(
        family="damped_line", target="damped_line", range_param="A_dim",
        range_value=200.0, error_bound=1.0e-6, n_max=64,
        nearest_below=None, range_lever="no lever",
        generator_hint="nothing to run yet")
    assert "nearest certified below: none" in text
    assert "no lever" in text

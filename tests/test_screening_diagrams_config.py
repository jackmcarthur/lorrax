"""``screening_diagrams = w_rpa | w_bse`` — the axis, its refusals, its fork.

WHAT IS BEING PINNED.  A new deck key that changes which DIAGRAMS the
screened Coulomb sums.  Everything about it that can be wrong before any
resolvent is solved lives here: the value parses and normalises, a typo
refuses naming the legal set, each unsupported combination refuses AT
PARSE TIME with a rule id an operator can grep for, the key actually
reaches a different screening model, and — the half that protects
everything already in the tree — a deck that does not mention the key
takes exactly the path it took before the key existed.

WHY THE "IT ACTUALLY CHANGES SOMETHING" CELL IS NOT PARANOIA.  There is
no structural gate in this tree that a registered configuration value has
a live consumer (KNOWN_LORRAX_ISSUES.md:22), and the last key to go
unread was ``screening_method``, which parsed ``ctsp``, normalised it, and
ran minimax for months while the deck, the run log and the provenance
record all agreed on a method the code never had
(docs/architecture/decisions.md, 2026-08-06).  So the fork is asserted
behaviourally, from both sides: ``w_bse`` reaches the ladder helper and
never the RPA executor, ``w_rpa`` reaches the RPA executor and never so
much as imports the ladder helper.
"""

from __future__ import annotations

import ast
import os
import pathlib
import sys

import pytest

from gw.gw_config import (
    ComputeMode,
    HeadCorrection,
    LorraxConfig,
    ScreeningConfig,
    ScreeningDiagrams,
    coerce_screening_diagrams,
    refuse_unsupported_screening_diagrams,
)

_REPO = pathlib.Path(__file__).resolve().parents[1]
_GW = _REPO / "src" / "gw"

_BASE = """\
[cohsex]
nval = 2
ncond = 2
nband = 10
memory_per_device_gb = 4.0
"""

_METAL_KEYS = """\
mpa_material_class = metal
occ_smearing_family = mp1
occ_smearing_width_ry = 0.02
fermi_reference = mp1_fixed_n
sigma_omega_layout = sharded
"""


def _config(tmp_path, extra="", name="screening_diagrams.in"):
    path = tmp_path / name
    path.write_text(_BASE + extra)
    return LorraxConfig.from_input_file(
        str(path), print_fn=lambda *a, **k: None)


# ---------------------------------------------------------------------------
# 1. Parse -> normalise -> enum
# ---------------------------------------------------------------------------

def test_a_deck_that_never_mentions_the_key_resolves_to_w_rpa(tmp_path):
    """THE FROZEN-PIN CELL.  Silence means the random-phase series.

    Every deck in this repo predates the key.  If the default were
    anything else, or if the field could arrive unset, the Si pins and the
    0.644 meV BGW MAE cell would move without a deck changing.
    """
    config = _config(tmp_path)
    assert config.screening.diagrams is ScreeningDiagrams.W_RPA
    assert config.screening.diagrams.value == "w_rpa"


def test_head_correction_defaults_full_and_legacy_do_g0_maps_to_off(tmp_path):
    assert _config(tmp_path).head.correction is HeadCorrection.FULL
    legacy = _config(tmp_path, "do_G0 = false\n", name="legacy_off.in")
    assert legacy.head.correction is HeadCorrection.OFF
    assert legacy.do_G0 is False


@pytest.mark.parametrize(
    "spelling, expected",
    [("full", HeadCorrection.FULL),
     ("NO_LOCAL_FIELDS", HeadCorrection.NO_LOCAL_FIELDS),
     ("off", HeadCorrection.OFF)],
)
def test_public_head_policy_is_parsed_and_controls_the_legacy_mirror(
        tmp_path, spelling, expected):
    config = _config(
        tmp_path, f"head_correction = {spelling}\n",
        name=f"head_{spelling.lower()}.in")
    assert config.head.correction is expected
    assert config.do_G0 is (expected is not HeadCorrection.OFF)


def test_contradictory_legacy_and_named_head_flags_refuse(tmp_path):
    with pytest.raises(ValueError, match="contradictory deck settings"):
        _config(
            tmp_path, "do_G0 = false\nhead_correction = full\n",
            name="contradictory_head.in")


def test_the_value_normalises_like_every_other_string_key(tmp_path):
    """Case and whitespace are the parser's problem, not the deck's."""
    config = _config(tmp_path, "screening_diagrams =   W_BSE  \n")
    assert config.screening.diagrams is ScreeningDiagrams.W_BSE


def test_the_key_is_registered_so_it_is_not_an_unknown_key(tmp_path):
    """``strict_keys = true`` must ACCEPT it — the registry test.

    A key absent from ``_DEFAULTS`` is warned about and ignored, which for
    a physics-selecting key is the worst of the three possible outcomes.
    """
    config = _config(
        tmp_path, "strict_keys = true\nscreening_diagrams = w_rpa\n")
    assert config.screening.diagrams is ScreeningDiagrams.W_RPA


@pytest.mark.parametrize("typo", ["w_ladder", "bse", "ladder", "wrpa", ""])
def test_an_unknown_value_refuses_naming_the_legal_set(tmp_path, typo):
    """A typo never resolves to the default; it names both members."""
    with pytest.raises(ValueError) as exc:
        _config(tmp_path, f"screening_diagrams = {typo}\n")
    message = str(exc.value)
    assert "screening_diagrams" in message
    assert "w_rpa" in message and "w_bse" in message


def test_coerce_accepts_the_enum_its_value_and_a_bare_string():
    """One normaliser, the ``coerce_compute_mode`` shape."""
    for spelling in (ScreeningDiagrams.W_BSE, "w_bse", "  W_BSE "):
        assert coerce_screening_diagrams(spelling) is ScreeningDiagrams.W_BSE
    with pytest.raises(ValueError, match="w_rpa, w_bse"):
        coerce_screening_diagrams("w_gw")


def test_the_dataclass_itself_refuses_a_value_off_the_axis():
    """The guard for every constructor that is NOT the parser.

    Tools and test stubs build this record directly; the parser's coercion
    does not protect them, and the axis must not acquire a third value by
    assignment.
    """
    with pytest.raises(ValueError) as exc:
        ScreeningConfig(
            method="minimax", occ_broadening_ev=0.0,
            minimax_target_error=1e-6, minimax_max_nodes=64,
            regenerate_minimax_tables=False, minimax_energy_reference="midgap",
            diagrams="w_bse")          # a STRING, not the member
    message = str(exc.value)
    assert "w_rpa" in message and "w_bse" in message
    assert "orthogonal" in message.lower()


def test_the_default_is_spelled_the_same_in_both_places():
    """``_DEFAULTS`` and the dataclass default must agree.

    A fallback that disagrees with the registered default makes a
    hand-built config take a different decision from a parsed one — the
    defect the ``restart_q_storage`` note in ``gw_config`` describes.
    """
    from gw.gw_config import _DEFAULTS
    assert _DEFAULTS["screening_diagrams"] == ScreeningDiagrams.W_RPA.value
    field = ScreeningConfig.__dataclass_fields__["diagrams"]
    assert field.default is ScreeningDiagrams.W_RPA


# ---------------------------------------------------------------------------
# 2. The refusal matrix — every cell refuses at PARSE time, by rule id
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rule_id, extra", [
    ("w_bse_needs_a_screened_mode", "compute_mode = x_only\n"),
    ("w_bse_hl_ppm_broadening_unimplemented", "compute_mode = hl_ppm\n"),
    ("w_bse_self_consistency_unimplemented",
     "compute_mode = cohsex\nqp_solver = self_consistent\n"),
    ("w_bse_head_placement_unimplemented",
     "compute_mode = cohsex\nmc_average_placement = bgw\n"),
    ("w_bse_insulators_only",
     "compute_mode = mpa\n" + _METAL_KEYS),
])
def test_each_unsupported_combination_refuses_at_parse_time(
        tmp_path, rule_id, extra):
    """AT PARSE TIME, not at the stage that would have run it.

    Every one of these is knowable from the deck alone, and the run they
    would produce costs a ζ fit and a χ₀ build before reaching the code
    that could notice.  The message carries got / want / fix / why / doc so
    the operator does not have to come to this file.
    """
    with pytest.raises(ValueError) as exc:
        _config(tmp_path, "screening_diagrams = w_bse\n" + extra)
    message = str(exc.value)
    assert rule_id in message
    for part in ("got:", "want:", "fix:", "why:", "doc:"):
        assert part in message, f"{rule_id} refusal is missing '{part}'"


def test_the_hl_ppm_refusal_cites_the_known_broadening_discrepancy(tmp_path):
    """It refuses BECAUSE improvising a third ξ convention is the hazard.

    GN-PPM and MPA already evaluate Σ at silently different broadenings on
    one deck (KNOWN_LORRAX_ISSUES.md:131 / :134).  A ladder that invented
    its own real-axis broadening would make that a three-way disagreement,
    which is why the refusal names the existing one instead of quietly
    picking a number.
    """
    with pytest.raises(ValueError) as exc:
        _config(tmp_path,
                "screening_diagrams = w_bse\ncompute_mode = hl_ppm\n")
    message = str(exc.value)
    assert "KNOWN_LORRAX_ISSUES" in message
    assert "broadening" in message


@pytest.mark.parametrize("extra", [
    "compute_mode = cohsex\n",
    "compute_mode = gn_ppm\n",
    "compute_mode = mpa\n",
    "compute_mode = gn_ppm\nqp_solver = one_shot_dft\n",
    "compute_mode = gn_ppm\nqp_solver = fixed_point\n",
])
def test_the_supported_combinations_parse(tmp_path, extra):
    """The other half of a refusal matrix: what it does NOT refuse.

    ``mpa`` is on this list deliberately: its insulating path is live, and
    this axis supports the ladder W at the same sample-plan frequencies.
    """
    config = _config(tmp_path, "screening_diagrams = w_bse\n" + extra)
    assert config.screening.diagrams is ScreeningDiagrams.W_BSE


def test_w_rpa_decks_are_untouched_by_every_rule(tmp_path):
    """Not one of the refusal-table combinations fires under the default.

    The refusal function returns before resolving ``compute_mode`` or
    ``qp_solver`` on a ``w_rpa`` deck, so a default deck cannot acquire a
    NEW parse-time resolution — and therefore a new possible error — from
    this feature existing.
    """
    for extra in ("compute_mode = x_only\n",
                  "compute_mode = hl_ppm\n",
                  "compute_mode = cohsex\nqp_solver = self_consistent\n",
                  "compute_mode = cohsex\nmc_average_placement = bgw\n",
                  "compute_mode = mpa\n" + _METAL_KEYS):
        config = _config(tmp_path, extra, name="rpa_arm.in")
        assert config.screening.diagrams is ScreeningDiagrams.W_RPA
        refuse_unsupported_screening_diagrams(config)   # must not raise


def test_a_metal_deck_still_parses_and_runs_under_w_rpa(tmp_path):
    """THE INSULATOR GATE IS SCOPED TO ``w_bse`` AND MUST STAY THERE.

    ``mpa_material_class = metal`` is a legal, live setting on the RPA
    branch: the metal merge supplies fractional-occupation chi, W and Sigma.
    A screening-diagram axis that started refusing metal decks in general
    would break that production path while claiming to guard only the
    distinct ladder operator.
    """
    config = _config(tmp_path, "compute_mode = mpa\n" + _METAL_KEYS)
    assert config.screening.diagrams is ScreeningDiagrams.W_RPA
    assert config.mpa.material_class == "metal"
    refuse_unsupported_screening_diagrams(config)       # must not raise


def test_the_insulator_refusal_names_the_live_metal_alternative(tmp_path):
    """The refusal points a metal deck to the live RPA implementation.

    The metal merge makes MPA+RPA production-capable; it does not change the
    ladder's insulator-only derivation.  The diagnostic must state both facts
    so the new metal capability and the older ladder refusal cannot be read
    as contradictory global claims.
    """
    with pytest.raises(ValueError) as exc:
        _config(tmp_path,
                "screening_diagrams = w_bse\ncompute_mode = mpa\n"
                + _METAL_KEYS)
    message = str(exc.value)
    assert "screening_diagrams = w_rpa" in message
    assert "metallic MPA screening/Sigma pipeline is live" in message
    assert "insulator" in message


def test_every_rule_has_all_five_parts_and_a_unique_id():
    """THE RATCHET on the table itself.

    A rule added without a ``fix`` or without a ``why`` is a refusal that
    stops a run without telling anyone what to do about it, which is the
    class of message this table's shape exists to make impossible.
    """
    from gw.gw_config import _W_BSE_REFUSALS
    ids = [row[0] for row in _W_BSE_REFUSALS]
    assert len(ids) == len(set(ids)), f"duplicate rule id in {ids}"
    for row in _W_BSE_REFUSALS:
        rule_id, predicate, got, want, fix, doc = row
        assert callable(predicate) and callable(got)
        for text, part in ((rule_id, "id"), (want, "want"), (fix, "fix"),
                           (doc, "why")):
            assert isinstance(text, str) and len(text) > 8, (
                f"{rule_id}: {part} is missing or too short to be advice")


# ---------------------------------------------------------------------------
# 2b. The runtime half of ``w_bse_insulators_only`` — the occupations
#     themselves, for the metallic WFN no deck key declares
# ---------------------------------------------------------------------------

def _occ(nspin=1, nk=3, nb=8, n_occ=4):
    """An insulating ``mf_header/kpoints/occ``: ``(nspin, nk, nb)`` in {0,1}."""
    import numpy as np

    occ = np.zeros((nspin, nk, nb), dtype=np.float64)
    occ[:, :, :n_occ] = 1.0
    return occ


def test_the_runtime_gate_passes_an_insulating_occupation_array():
    """The green case, and the number it returns is the margin.

    A gapped mean field writes exact 0.0 / 1.0, so the worst
    distance-from-integer is 0.0 — not merely under tolerance.  Reporting
    the margin rather than a bool is what lets the run log say how much
    room the gate had.
    """
    from gw.screening_bse import refuse_fractional_occupations

    worst = refuse_fractional_occupations(
        _occ(), band_lo=0, band_hi=8, source="unit.h5",
        print_fn=lambda *a, **k: None)
    assert worst == 0.0


def test_the_runtime_gate_refuses_a_fractional_occupation_in_the_active_window():
    """THE CASE NO DECK KEY CAN DECLARE: a metallic WFN on a silent deck.

    ``mpa_material_class`` is the only key in the parser that says "metal",
    and nothing at all in a deck describes the WFN's occupations — so a
    smeared mean field reaches the stage looking exactly like an insulating
    one until something reads ``occ``.  The refusal carries the SAME rule
    id as its parse-time twin (one grep for an operator) and the same five
    parts, and it names the offending (spin, k, band) so the reader can go
    look at it.
    """
    from gw.screening_bse import refuse_fractional_occupations

    occ = _occ()
    occ[0, 2, 3] = 0.62                      # a partially filled band at E_F
    with pytest.raises(NotImplementedError) as exc:
        refuse_fractional_occupations(
            occ, band_lo=0, band_hi=8, source="metal.h5",
            print_fn=lambda *a, **k: None)
    message = str(exc.value)
    assert "w_bse_insulators_only" in message
    for part in ("got:", "want:", "fix:", "why:", "doc:"):
        assert part in message, f"runtime refusal is missing '{part}'"
    assert "spin 0, k-point 2, band 3" in message
    assert "metallic MPA screening/Sigma pipeline remains available" in message


def test_the_runtime_gate_tolerance_separates_noise_from_a_partial_filling():
    """1e-6 sits in an EMPTY band, which is why it is a safe threshold.

    Below it there is only float/HDF5 representation noise; above it there
    is only physics (an occupation within 1e-6 of a step value is >13 kT
    from E_F at any smearing a GW deck uses).  Both ends are pinned here so
    that a future edit to the constant has to argue with both.
    """
    import numpy as np

    from gw.screening_bse import _OCC_INTEGER_TOL, refuse_fractional_occupations

    assert _OCC_INTEGER_TOL == 1e-6
    noisy = _occ()
    noisy[0, 0, 0] = 1.0 - 1e-9
    assert refuse_fractional_occupations(
        noisy, band_lo=0, band_hi=8, source="noise.h5",
        print_fn=lambda *a, **k: None) == pytest.approx(1e-9)
    partial = _occ()
    partial[0, 0, 0] = 1.0 - 1e-3
    with pytest.raises(NotImplementedError, match="w_bse_insulators_only"):
        refuse_fractional_occupations(
            partial, band_lo=0, band_hi=8, source="smeared.h5",
            print_fn=lambda *a, **k: None)
    # An occupation of 2.0 is an INTEGER: some WFN spin conventions write
    # the two-electron value, and that is not a partial filling.
    two = _occ()
    two[:, :, :4] = 2.0
    assert refuse_fractional_occupations(
        two, band_lo=0, band_hi=8, source="two.h5",
        print_fn=lambda *a, **k: None) == 0.0
    assert np.isfinite(_OCC_INTEGER_TOL)


def test_the_runtime_gate_is_scoped_to_the_run_s_own_band_window():
    """It asks about the bands the ladder actually builds from.

    ``[b0, b4)`` is the pair basis and the resolvent's pole set.  The
    scoping is not a loophole — the window straddles E_F by construction
    (``b2 = nelec`` is inside it), so a partially filled band cannot sit
    above ``b4`` — and a window that does not intersect the file's bands is
    a caller error, refused rather than silently empty.
    """
    from gw.screening_bse import refuse_fractional_occupations

    occ = _occ(nb=12)
    occ[0, 1, 11] = 0.5                      # outside a [0, 8) window
    assert refuse_fractional_occupations(
        occ, band_lo=0, band_hi=8, source="wide.h5",
        print_fn=lambda *a, **k: None) == 0.0
    with pytest.raises(NotImplementedError, match="w_bse_insulators_only"):
        refuse_fractional_occupations(
            occ, band_lo=0, band_hi=12, source="wide.h5",
            print_fn=lambda *a, **k: None)
    with pytest.raises(ValueError, match="does not intersect"):
        refuse_fractional_occupations(
            occ, band_lo=99, band_hi=120, source="wide.h5",
            print_fn=lambda *a, **k: None)


def test_the_stage_reads_the_wfn_occupations_and_not_the_step_built_bundle():
    """``wfns.occ`` CANNOT fire this gate, so the gate must not read it.

    ``wavefunction_bundle._build_occ`` builds the bundle's occupations from
    a band-index cut or one E_F comparison — {0, 1} by construction.  A
    gate reading it would be a green light with a name.  The mean field's
    own ``mf_header/kpoints/occ`` is the only array in this pipeline where
    a smeared occupation survives, and it is read through the same
    ``file_io.mf_header`` reader ``WfnLoader`` uses.
    """
    import inspect

    from gw import screening_bse

    src = inspect.getsource(screening_bse._refuse_metallic_mean_field)
    assert "read_mf_header" in src and ".occs" in src
    assert "wfns.occ" not in src
    assert "meta.band_edges" in src


def test_the_occupation_gate_runs_before_any_compute_in_the_stage():
    """Cost order, and it is the reason the check is placed where it is.

    Everything the gate needs is metadata the run already has; learning it
    after the chi0 build would spend the expensive part of the run to be
    told something the WFN said at load.  Pinned as an ORDER in the
    production entry point, the same way the closure gate's composition is.
    """
    import inspect

    from gw import screening_bse

    src = inspect.getsource(screening_bse.prepare_ladder_restart)
    order = [src.index(name) for name in (
        "_refuse_metallic_mean_field(", "_refuse_unusable_restart(",
        "compute_static_w(")]
    assert order == sorted(order), (
        "prepare_ladder_restart no longer checks the occupations and the "
        "restart handoff BEFORE the RPA static leg")


# ---------------------------------------------------------------------------
# 3. THE KEY ACTUALLY CHANGES THE DISPATCHED SCREENING MODEL
# ---------------------------------------------------------------------------

class _Reached(Exception):
    """Raised by the stand-ins below; reaching them IS the assertion."""


def _stub_kwargs(config, tmp_path):
    return dict(
        quad=None, e_ref=0.0, sym=None, centroid_indices=None,
        config=config, meta=None, mesh_xy=None,
        run_dir=str(tmp_path / "mpa"), label="unit",
        material_class="insulator",
        tensors_filename=str(tmp_path / "isdf_tensors_4.h5"),
        print_fn=lambda *a, **k: None)


def test_w_bse_reaches_the_ladder_helper_and_not_the_rpa_executor(
        tmp_path, monkeypatch):
    """The consumer check KNOWN_LORRAX_ISSUES.md:22 says nothing enforces.

    Monkeypatching the facade side and asserting it is REACHED is the only
    way to tell "the key selects the ladder" from "the key parses".
    """
    from gw import screening, screening_bse

    config = _config(tmp_path, "screening_diagrams = w_bse\n"
                               "compute_mode = cohsex\n")
    monkeypatch.setattr(
        screening_bse, "compute_screening_ladder",
        lambda *a, **k: (_ for _ in ()).throw(_Reached("ladder")))
    monkeypatch.setattr(
        screening, "compute_screening",
        lambda *a, **k: pytest.fail("w_bse took the RPA executor"))

    with pytest.raises(_Reached, match="ladder"):
        screening.compute_screening_model(
            ComputeMode.COHSEX, None, None, **_stub_kwargs(config, tmp_path))


def test_w_rpa_reaches_the_rpa_executor_and_never_touches_the_helper(
        tmp_path, monkeypatch):
    """The control arm.  A default deck must not even reach the ladder.

    This is the cell that would fail if the fork were written as "call the
    helper, which decides" — a shape that keeps ``w_rpa`` bit-identical
    only for as long as the helper's early return stays perfect.
    """
    from gw import screening, screening_bse

    config = _config(tmp_path)
    assert config.screening.diagrams is ScreeningDiagrams.W_RPA
    monkeypatch.setattr(
        screening_bse, "compute_screening_ladder",
        lambda *a, **k: pytest.fail("w_rpa reached the ladder helper"))
    monkeypatch.setattr(
        screening_bse, "make_ladder_wc_source",
        lambda *a, **k: pytest.fail("w_rpa built a ladder wc_source"))
    monkeypatch.setattr(
        screening, "compute_screening",
        lambda *a, **k: (_ for _ in ()).throw(_Reached("rpa")))

    with pytest.raises(_Reached, match="rpa"):
        screening.compute_screening_model(
            ComputeMode.COHSEX, None, None, **_stub_kwargs(config, tmp_path))


def test_mpa_gets_the_ladder_through_the_wc_source_seam_only(
        tmp_path, monkeypatch):
    """MPA's shared frequency walk keeps its own shape on both branches.

    The ladder does NOT replace ``build_mpa_fit``; it replaces the one
    call inside it that produces Wc(z).  Anything else would fork the
    sample plan, the fit and the store lifecycle as well.
    """
    from gw import screening, screening_bse
    from gw.mpa import model

    seen = {}
    monkeypatch.setattr(
        screening_bse, "make_ladder_wc_source",
        lambda *a, **k: "LADDER_SOURCE")
    monkeypatch.setattr(
        model, "build_mpa_fit",
        lambda *a, **k: (
            seen.setdefault("wc_source", k.get("wc_source")), None))

    for extra, expect in (("screening_diagrams = w_bse\n", "LADDER_SOURCE"),
                          ("", None)):
        seen.clear()
        config = _config(tmp_path, extra + "compute_mode = mpa\n",
                         name="mpa_arm.in")
        screening.compute_screening_model(
            ComputeMode.MPA, None, None,
            head_resolver=object(), **_stub_kwargs(config, tmp_path))
        assert seen["wc_source"] == expect


def test_the_default_wc_source_is_the_rpa_dyson_solve():
    """``wc_source = None`` means ``_solve_wc`` — stated, not inferred.

    The RPA arm's bit-identity rests on this default, so it is pinned
    rather than left to inference from a generic callable expression.
    """
    import inspect
    from gw.mpa import model

    src = inspect.getsource(model.build_mpa_fit)
    assert "if wc_source is None:" in src
    assert "iteration_head = _solve_wc(" in src
    assert "iteration_head = wc_source(" in src
    assert inspect.signature(model.build_mpa_fit).parameters[
        "wc_source"].default is None


# ---------------------------------------------------------------------------
# 4. Frozen-pin immunity — the RPA path did not move
# ---------------------------------------------------------------------------

def test_no_rpa_screening_code_moved_out_of_gw_screening():
    """The cheapest structural statement that the fork moved nothing.

    Every object the RPA path executes is still DEFINED in
    ``gw.screening``.  A refactor that hoisted any of them into a shared
    helper "so both branches can use it" would fail here — and that is the
    refactor that silently re-times, re-orders or re-donates the buffers
    the frozen Si pins were measured on.
    """
    from gw import screening

    for name in ("screening_requests_for", "compute_static_w",
                 "compute_screening", "compute_screening_model", "_gate_w"):
        obj = getattr(screening, name)
        assert obj.__module__ == "gw.screening", (
            f"{name} now lives in {obj.__module__}; the RPA path moved")


def test_the_rpa_tail_of_the_fork_is_the_pre_fork_body():
    """The last two statements of ``compute_screening_model`` are unchanged.

    Read out of the AST rather than diffed against a golden string, so the
    cell says what it means: after the diagram check, the RPA arm still
    plans its requests and hands them to ``compute_screening``, with no
    statement in between.
    """
    from gw import screening

    tree = ast.parse(pathlib.Path(screening.__file__).read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef)
              and n.name == "compute_screening_model")
    tail = [ast.unparse(s) for s in fn.body[-3:]]
    assert tail[0] == "requests = screening_requests_for(mode, config)"
    assert tail[1].startswith("if static_only:")
    assert tail[2].startswith("return compute_screening(")


def test_the_ladder_helper_is_never_imported_at_module_scope():
    """``gw.screening`` must not import ``gw.screening_bse`` eagerly.

    Two reasons, and they point the same way: importing it would pull
    ``bse`` into every GW run that never asked for a ladder, and a
    module-scope ``bse`` import from ``gw`` is the Python cycle the
    function-level imports in ``screening_bse`` exist to avoid.
    """
    from gw import screening

    tree = ast.parse(pathlib.Path(screening.__file__).read_text())
    for node in tree.body:                      # MODULE SCOPE only
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            text = ast.unparse(node)
            assert "screening_bse" not in text, text


def test_the_driver_hands_its_w0_flush_to_one_predicate():
    """``driver_persists_w0`` answers for both reasons the flush is skipped.

    MPA (no ``{0, probe}`` head grid to stamp) and w_bse (the stage helper
    already persisted, and ``W_by_role["static"]`` is now the LADDER W).
    Re-persisting under w_bse would put the ladder W under ``W0_qmunu``,
    which every downstream consumer reads as the RPA static W — the
    provenance failure this predicate exists to stop.
    """
    from gw.screening import driver_persists_w0

    class _Cfg:
        def __init__(self, diagrams):
            self.screening = type("S", (), {"diagrams": diagrams})()

    rpa, bse = _Cfg(ScreeningDiagrams.W_RPA), _Cfg(ScreeningDiagrams.W_BSE)
    assert driver_persists_w0(ComputeMode.COHSEX, rpa) is True
    assert driver_persists_w0(ComputeMode.GN_PPM, rpa) is True
    assert driver_persists_w0(ComputeMode.MPA, rpa) is False
    assert driver_persists_w0(ComputeMode.COHSEX, bse) is False
    assert driver_persists_w0(ComputeMode.MPA, bse) is False


def test_the_driver_no_longer_spells_the_mpa_check_itself():
    """gw_jax asks the predicate; it does not restate the mode test.

    Two opinions about "should this run persist W0" is exactly the shadow
    accounting that would let the w_bse branch double-write.
    """
    src = (_GW / "gw_jax.py").read_text()
    assert "driver_persists_w0(mode, config)" in src
    assert "if mode is not ComputeMode.MPA:" not in src


# ---------------------------------------------------------------------------
# 5. The dispatch ratchet, twinned onto the new axis
# ---------------------------------------------------------------------------

def _bare_else_offenders(source: str, marker: str, filename="<src>"):
    """Sites comparing against ``marker`` that fall into a plain ``else``.

    Deliberately the same crude text-level scan
    ``test_ff_compute_mode.py::test_no_module_dispatches_on_the_mode_through_a_bare_else``
    runs on the ``ComputeMode`` axis, for the same reason: its job is to
    make a reviewer look at a NEW dispatch site, not to prove the existing
    ones correct.
    """
    offenders = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.If) or not node.orelse:
            continue
        if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
            continue                                    # an elif chain
        if marker not in ast.unparse(node.test):
            continue
        offenders.append(f"{filename}:{node.lineno}")
    return offenders


def test_no_module_dispatches_on_the_diagram_axis_through_a_bare_else():
    """THE TWIN RATCHET.  A third diagram set must not inherit a branch.

    ``ScreeningDiagrams`` has two members today and the whole reason the
    ``compute_mode`` audit (16e0c9c0) was worth its commit message is that
    an axis with two members is exactly when the else-branch looks safe.
    """
    offenders = []
    for path in sorted(_GW.rglob("*.py")):
        offenders += _bare_else_offenders(
            path.read_text(), "ScreeningDiagrams.", path.name)
    assert offenders == [], (
        "a screening-diagram comparison with a plain else-branch: the next "
        "member added to the enum silently takes that branch.  Name it.\n  "
        + "\n  ".join(offenders))


def test_the_bare_else_scanner_detects_the_shape_it_forbids():
    """THE RED TWIN.  A scanner that cannot fail certifies nothing."""
    bad = (
        "if diagrams is ScreeningDiagrams.W_BSE:\n"
        "    x = 1\n"
        "else:\n"
        "    x = 2\n"
    )
    good = (
        "if diagrams is ScreeningDiagrams.W_BSE:\n"
        "    x = 1\n"
        "elif diagrams is ScreeningDiagrams.W_RPA:\n"
        "    x = 2\n"
    )
    assert _bare_else_offenders(bad, "ScreeningDiagrams.") != []
    assert _bare_else_offenders(good, "ScreeningDiagrams.") == []


def test_the_fork_names_both_members_and_refuses_a_third():
    """``compute_screening_model`` refuses an unhandled member by name.

    Falling through to the RPA arm would produce a complete, plausible W
    under another diagram set's name — the failure the ``compute_mode``
    dispatch audit closed on its own axis.
    """
    from gw import screening

    src = ast.unparse(next(
        n for n in ast.walk(ast.parse(
            pathlib.Path(screening.__file__).read_text()))
        if isinstance(n, ast.FunctionDef)
        and n.name == "compute_screening_model"))
    assert "ScreeningDiagrams.W_BSE" in src
    assert "ScreeningDiagrams.W_RPA" in src
    assert "has no branch here" in src


# ---------------------------------------------------------------------------
# 6. The stage helper's shape (what the GPU closure gate rides on)
# ---------------------------------------------------------------------------

def test_the_ladder_stage_composes_exactly_the_steps_the_closure_gate_drives():
    """The GPU closure gate drives the helpers; this pins that they ARE the path.

    ``tests/test_w_bse_wiring_closure.py`` measures
    ``_ladder_wedge`` -> ``_assert_wedge_matches_run`` ->
    ``_assemble_full_bz_w`` rather than re-running the RPA static leg it
    already has on disk.  That is only a gate on the production path if the
    production entry point composes the same three calls, in that order,
    and nothing else between them — so it is read out of the AST here,
    cheaply, instead of being assumed there, expensively.
    """
    import inspect
    from gw import screening_bse

    src = inspect.getsource(screening_bse.compute_screening_ladder)
    order = [src.index(name) for name in (
        "prepare_ladder_restart(", "_ladder_wedge(",
        "_assert_wedge_matches_run(", "_assemble_full_bz_w(",
        "_gate_w_or_refuse(")]
    assert order == sorted(order), (
        "compute_screening_ladder no longer calls prepare -> wedge -> "
        "wedge-check -> assemble -> gate in that order; the closure gate "
        "drives the middle three directly and would stop covering the "
        "production composition")


@pytest.mark.parametrize("level", [None, "0", "1"])
def test_w_bse_w_gate_is_a_five_part_refusal_regardless_of_sanity_default(
        monkeypatch, level):
    """A broken ladder W cannot inherit sanity's warn/off global policy.

    The three-point q grid makes q=1 and q=2 a +/- pair.  Their unequal
    real values pass finiteness and q=0 hermiticity, then fail exactly the
    production reciprocity observable at 5e-1.  ``got`` must preserve that
    value so the refusal names what failed rather than only saying "bad W".
    """
    import numpy as np
    from gw import screening_bse
    from gw.screening import ScreeningRequest

    if level is None:
        monkeypatch.delenv("LORRAX_SANITY", raising=False)
    else:
        monkeypatch.setenv("LORRAX_SANITY", level)
    W = np.asarray([[[1.0]], [[1.0]], [[2.0]]], dtype=np.complex128)
    with pytest.raises(ValueError) as exc:
        screening_bse._gate_w_or_refuse(
            W, ScreeningRequest(0.0 + 0.0j, "static"),
            stage="assembled ladder W[static]", kgrid=(3, 1, 1),
            print_fn=lambda *a, **k: None)
    message = str(exc.value)
    assert "w_bse_w_stage_invariants" in message
    for part in ("got:", "want:", "fix:", "why:", "doc:"):
        assert part in message
    assert "q<->-q conjugate reciprocity" in message
    assert "5.000e-01 > 1.0e-05" in message
    assert os.environ.get("LORRAX_SANITY") == level


def test_w_bse_w_gate_restores_sanity_after_a_healthy_stage(monkeypatch):
    """The stage-local strict pin must not change later global gate policy."""
    import numpy as np
    from gw import screening_bse
    from gw.screening import ScreeningRequest

    monkeypatch.setenv("LORRAX_SANITY", "off")
    W = np.ones((3, 1, 1), dtype=np.complex128)
    screening_bse._gate_w_or_refuse(
        W, ScreeningRequest(0.0 + 0.0j, "static"),
        stage="healthy scalar ladder W[static]", kgrid=(3, 1, 1),
        print_fn=lambda *a, **k: None)
    assert os.environ["LORRAX_SANITY"] == "off"


def test_the_z_list_comes_from_the_role_plan_and_not_a_second_table():
    """One source of truth for "which W does this Sigma need".

    A ladder that carried its own {0, i*omega_p} table would drift from
    ``screening_requests_for`` the first time a role's frequency moved, and
    the drift would be invisible: both sides return the right shapes with
    the right role labels.
    """
    import inspect
    from gw import screening_bse

    src = inspect.getsource(screening_bse.compute_screening_ladder)
    assert "screening_requests_for(mode, config)" in src
    assert "z_list = [complex(r.omega_ry) for r in requests]" in src


def test_the_stage_helper_imports_bse_only_inside_functions():
    """The gw<->bse Python cycle stays broken, and the driver stays clean.

    ``bse`` imports ``gw`` helpers at module scope, so a module-scope
    ``import bse`` here would make the two mutually importable and the
    winner would depend on which driver started.  L1->L1 is legal by LEVEL
    (docs/architecture/layers.md); this is about Python, and the module
    docstring says so.
    """
    import inspect
    from gw import screening_bse

    tree = ast.parse(pathlib.Path(screening_bse.__file__).read_text())
    for node in tree.body:                       # MODULE SCOPE only
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            assert "bse" not in ast.unparse(node), ast.unparse(node)
    assert "from bse.w_ladder import compute_wc_qwedge" in inspect.getsource(
        screening_bse._ladder_wedge)


def test_no_new_scan_site_omits_its_unroll():
    """Owner rule 2026-08-05, checked on the files this feature adds."""
    for path in (_GW / "screening_bse.py", _GW / "screening.py"):
        src = path.read_text()
        for i, line in enumerate(src.splitlines(), 1):
            if "lax.scan(" in line:
                window = "\n".join(src.splitlines()[i - 1:i + 8])
                assert "unroll=" in window, f"{path.name}:{i}"


def test_the_wedge_residual_gate_refuses_an_absent_residual():
    """"No residuals" and "all residuals fine" must not look the same.

    The facade contract makes the CALLER gate on per-column residuals
    rather than on a return code, so a wedge object that carries none is a
    contract break, and treating it as convergence is exactly the
    non-discriminating observable QUALITY_PATTERNS' addendum names.
    """
    from gw import screening_bse

    class _NoResiduals:
        wc = None
        q_irr_kgrid_int = ()

    assert screening_bse._wedge_field(
        _NoResiduals(), screening_bse._RESIDUAL_FIELDS, float) is None
    for name in screening_bse._RESIDUAL_FIELDS:
        holder = type("W", (), {name: [1.0, 2.0]})()
        got = screening_bse._wedge_field(
            holder, screening_bse._RESIDUAL_FIELDS, float)
        assert got is not None and float(got.max()) == 2.0


def test_the_production_cap_accepts_the_measured_nb20_krylov_depth(monkeypatch):
    """The wide-window scalar-Si solve needs 242 iterations, not a looser tol.

    The former 200 cap returned a finite tile with a 9.17e-3 true residual.
    At identical operands/tolerance the same columns converge normally at
    iterations 240--242.  This cell pins the production caller's headroom and
    its independent residual gate: a 242-iteration, 9e-7 result must pass.
    """
    import numpy as np
    from bse import w_ladder
    from gw import screening_bse

    class _MeasuredWideWindow:
        gmres_resid = np.asarray([9.0e-7], dtype=np.float64)
        gmres_iters = np.asarray([242], dtype=np.int64)

    def _fake(*_args, gmres_max_iter, **_kwargs):
        assert gmres_max_iter >= 300
        return _MeasuredWideWindow()

    monkeypatch.setattr(w_ladder, "compute_wc_qwedge", _fake)
    got = screening_bse._ladder_wedge(
        "restart.h5", [0.0 + 0.0j], None, input_file="deck.in",
        print_fn=lambda *_a, **_k: None)
    assert got.gmres_iters[0] == 242


# ---------------------------------------------------------------------------
# 7. Provenance (QUALITY_PATTERNS #10)
# ---------------------------------------------------------------------------

def test_the_w0_writer_stamps_which_diagram_set_made_it():
    """``W0_qmunu`` is the most-reused artifact this driver writes.

    BSE, downfold and the next restart all read it, and none of them has a
    Coulomb config of its own to check it against.  RPA and ladder W are
    the same shape with the same flags.
    """
    import inspect
    from gw import gw_output

    src = inspect.getsource(gw_output._stamp_screening_diagrams)
    assert 'attrs["screening_diagrams"]' in src
    assert "_stamp_screening_diagrams(tensors_filename, config)" in \
        inspect.getsource(gw_output.persist_w0_and_head)


def test_the_mpa_fit_store_carries_the_tag_and_the_consumer_asserts_it():
    """Written by ``build_mpa_fit``, enforced by ``validate_fit_store``.

    An UNSTAMPED store is refused rather than assumed RPA: "written before
    the axis existed" and "written by the RPA path" are different facts.
    """
    import inspect
    from file_io import mpa_store
    from gw.mpa import model, sigma

    assert '"screening_diagrams": diagrams' in inspect.getsource(
        model.build_mpa_fit)
    validate = inspect.getsource(mpa_store.validate_fit_store)
    assert "expected_screening_diagrams" in validate
    assert "carries no screening_diagrams stamp" in validate
    assert "expected_screening_diagrams=expected_screening_diagrams" in \
        inspect.getsource(sigma.compute_sigma_c_mpa_omega_grid)
    assert "expected_screening_diagrams=config.screening.diagrams" in \
        (_GW / "sigma_dispatch.py").read_text()


# ---------------------------------------------------------------------------
# 8. Ladder q/-q unfold gauge -- MOVED
# ---------------------------------------------------------------------------
# The q-axis time-reversal policy (pair-coherent rows, the fixed-q Theta
# projector, and the MEASURED verdict that gates both) is owned by
# ``symmetry_maps.qgrid_trs`` and consumed by bare V, RPA W and ladder W
# alike through ``gw.qgrid_symmetry.qgrid_trs_policy_for``.  Its cells --
# including the magnetic, nonmagnetic-scalar and spinor/Kramers red twins --
# live with the owner, in ``tests/test_qgrid_trs_policy.py``.  They were here
# only because the ladder shipped the policy first.


# ---------------------------------------------------------------------------
# 9. Docs
# ---------------------------------------------------------------------------

def test_the_hand_written_reference_row_says_orthogonal_and_names_the_rules():
    """docs/input_reference.md is hand-maintained; the row must be there.

    The generator is a DRIFT CHECKER on this page (its header says so and
    says why), so nothing regenerates this row — it is asserted instead.
    """
    page = (_REPO / "docs" / "input_reference.md").read_text()
    row = next(line for line in page.splitlines()
               if line.startswith("| `screening_diagrams`"))
    assert "`\"w_rpa\"`" in row
    assert "ORTHOGONAL" in row
    assert "`screening_method`" in row and "`compute_mode`" in row
    from gw.gw_config import _W_BSE_REFUSALS
    for rule_id, *_ in _W_BSE_REFUSALS:
        assert rule_id in row, f"{rule_id} is not named in the docs row"


def test_the_generator_knows_the_key_so_the_drift_check_accepts_it():
    """A new deck key cannot land without a row in the generator's table."""
    sys.path.insert(0, str(_REPO / "tools"))
    try:
        import gen_input_reference
    finally:
        sys.path.pop(0)
    group, meaning = gen_input_reference.KEYS["screening_diagrams"]
    assert group == "Screening"
    assert "w_rpa" in meaning and "w_bse" in meaning

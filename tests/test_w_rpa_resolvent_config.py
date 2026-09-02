"""``screening_diagrams = w_rpa_resolvent`` — the axis' third member.

WHAT IS BEING PINNED.  ``w_rpa_resolvent`` reaches the SAME ladder facade
as ``w_bse`` (``gw.screening_bse.compute_screening_ladder``) with
``include_w=False``: one matvec builder (``bse.bse_ring_comm.
build_bse_ring_matvec_full``), kernel-switched, not a second one.  This
file pins the config-level half of that: the value parses and normalises,
each audited refusal fires at PARSE TIME with its own rule id, the
combinations the ``w_bse`` audit did NOT carry over (MPA) refuse by name
rather than silently falling back to a different diagram set under this
one's label, the key reaches ``compute_screening_ladder`` with
``include_w=False`` and never the RPA Dyson executor, and — the half that
protects everything already in the tree — neither ``w_rpa`` nor ``w_bse``
decks acquire a new parse-time resolution from this member existing.

See ``tests/test_screening_diagrams_config.py`` for the twin coverage of
``w_rpa`` / ``w_bse``; this file does not repeat cells that axis file
already owns (the bare-else dispatch scanner, the driver's
``driver_persists_w0`` MPA case, ``w_rpa``'s own untouched-ness) except
where the third member changes what they must additionally prove.
"""

from __future__ import annotations

import pathlib

import pytest

from gw.gw_config import (
    ComputeMode,
    LorraxConfig,
    ScreeningDiagrams,
    coerce_screening_diagrams,
    refuse_unsupported_screening_diagrams,
)

_REPO = pathlib.Path(__file__).resolve().parents[1]

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


def _config(tmp_path, extra="", name="w_rpa_resolvent.in"):
    path = tmp_path / name
    path.write_text(_BASE + extra)
    return LorraxConfig.from_input_file(
        str(path), print_fn=lambda *a, **k: None)


# ---------------------------------------------------------------------------
# 1. Parse -> normalise -> enum
# ---------------------------------------------------------------------------

def test_the_spelling_parses_and_normalises(tmp_path):
    config = _config(tmp_path, "screening_diagrams = w_rpa_resolvent\n"
                               "compute_mode = cohsex\n")
    assert config.screening.diagrams is ScreeningDiagrams.W_RPA_RESOLVENT
    assert config.screening.diagrams.value == "w_rpa_resolvent"


def test_coercion_accepts_the_member_its_value_and_a_padded_string():
    for spelling in (ScreeningDiagrams.W_RPA_RESOLVENT, "w_rpa_resolvent",
                     "  W_RPA_RESOLVENT  "):
        assert (coerce_screening_diagrams(spelling)
                is ScreeningDiagrams.W_RPA_RESOLVENT)


def test_a_typo_still_names_all_three_legal_values(tmp_path):
    with pytest.raises(ValueError) as exc:
        _config(tmp_path, "screening_diagrams = w_rpa_resolvant\n")
    message = str(exc.value)
    for legal in ("w_rpa", "w_bse", "w_rpa_resolvent"):
        assert legal in message


# ---------------------------------------------------------------------------
# 2. The refusal matrix — audited, not copied from _W_BSE_REFUSALS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rule_id, extra", [
    ("w_rpa_resolvent_needs_a_screened_mode", "compute_mode = x_only\n"),
    ("w_rpa_resolvent_mpa_unimplemented", "compute_mode = mpa\n"),
    ("w_rpa_resolvent_hl_ppm_broadening_unimplemented",
     "compute_mode = hl_ppm\n"),
    ("w_rpa_resolvent_self_consistency_unimplemented",
     "compute_mode = cohsex\nqp_solver = self_consistent\n"),
    ("w_rpa_resolvent_head_placement_unimplemented",
     "compute_mode = cohsex\nmc_average_placement = bgw\n"),
])
def test_each_unsupported_combination_refuses_at_parse_time(
        tmp_path, rule_id, extra):
    """AT PARSE TIME, with the SAME five-part shape ``_W_BSE_REFUSALS`` uses.

    ``w_rpa_resolvent_mpa_unimplemented`` has no ``w_bse`` counterpart: MPA
    is SUPPORTED there (``make_ladder_wc_source``) and refused here,
    because that seam hard-codes ``include_w=True`` and was not extended
    or gated for this arm this session.  There is no parametrized case for
    ``w_rpa_resolvent_insulators_only`` — see
    ``test_the_parse_time_insulators_only_row_would_be_dead_code`` for why
    a parse-time cell for it does not exist, and the runtime section below
    for its actual (live) enforcement.
    """
    with pytest.raises(ValueError) as exc:
        _config(tmp_path, "screening_diagrams = w_rpa_resolvent\n" + extra)
    message = str(exc.value)
    assert rule_id in message
    for part in ("got:", "want:", "fix:", "why:", "doc:"):
        assert part in message, f"{rule_id} refusal is missing '{part}'"


def test_the_parse_time_insulators_only_row_would_be_dead_code(tmp_path):
    """WHY THERE IS NO ``w_rpa_resolvent_insulators_only`` PARSE-TIME ROW.

    ``mpa_material_class = metal`` is refused outside ``compute_mode =
    mpa`` by an EARLIER, unrelated gate
    (``metal_material_class_requires_mpa``) regardless of
    ``screening_diagrams``.  So a deck-key predicate mirroring
    ``w_bse_insulators_only`` could only ever fire on a
    ``compute_mode = mpa`` deck — and ``w_rpa_resolvent_mpa_
    unimplemented`` already refuses EVERY such deck, unconditionally,
    earlier in the same table.  A parallel row would be the exact
    "narrowed row whose predicate is a strict subset of an earlier row's"
    shape ``gw_config._LOW_MEM_BANDS_REFUSALS`` deletes on sight rather
    than ships unreachable.  This test is the measurement backing that
    argument, not merely the argument: the SAME deck this file's ``w_bse``
    sibling uses to reach ``w_bse_insulators_only`` reaches a DIFFERENT,
    earlier rule id here.
    """
    with pytest.raises(ValueError) as exc:
        _config(tmp_path,
                "screening_diagrams = w_rpa_resolvent\ncompute_mode = mpa\n"
                + _METAL_KEYS)
    message = str(exc.value)
    assert "w_rpa_resolvent_mpa_unimplemented" in message
    assert "insulators_only" not in message


@pytest.mark.parametrize("extra", [
    "compute_mode = cohsex\n",
    "compute_mode = gn_ppm\n",
    "compute_mode = gn_ppm\nqp_solver = one_shot_dft\n",
    "compute_mode = gn_ppm\nqp_solver = fixed_point\n",
])
def test_the_supported_combinations_parse(tmp_path, extra):
    """The other half of the matrix: what it does NOT refuse.

    ``mpa`` is deliberately ABSENT from this list — see the parametrized
    refusal above; that is the one row where ``w_rpa_resolvent`` is
    narrower than its ``w_bse`` sibling.
    """
    config = _config(tmp_path, "screening_diagrams = w_rpa_resolvent\n" + extra)
    assert config.screening.diagrams is ScreeningDiagrams.W_RPA_RESOLVENT


def test_mpa_is_supported_under_w_bse_but_not_under_w_rpa_resolvent(tmp_path):
    """States the asymmetry explicitly so it cannot be read as an oversight."""
    bse = _config(tmp_path, "screening_diagrams = w_bse\ncompute_mode = mpa\n",
                 name="bse_mpa.in")
    assert bse.screening.diagrams is ScreeningDiagrams.W_BSE
    with pytest.raises(ValueError, match="w_rpa_resolvent_mpa_unimplemented"):
        _config(tmp_path,
                "screening_diagrams = w_rpa_resolvent\ncompute_mode = mpa\n",
                name="resolvent_mpa.in")


def test_w_rpa_and_w_bse_decks_are_untouched_by_the_new_table(tmp_path):
    """Adding a third table must not change what the first two refuse.

    In particular ``compute_mode = mpa`` must still parse cleanly under
    BOTH ``w_rpa`` (always supported) and ``w_bse`` (audited-supported) —
    the new ``w_rpa_resolvent``-only MPA refusal must not leak onto either.
    """
    for extra, name in (
        ("compute_mode = mpa\n", "rpa_mpa.in"),
        ("screening_diagrams = w_bse\ncompute_mode = mpa\n", "bse_mpa2.in"),
    ):
        config = _config(tmp_path, extra, name=name)
        refuse_unsupported_screening_diagrams(config)   # must not raise


def test_every_rule_has_all_five_parts_and_a_unique_id():
    from gw.gw_config import _W_RPA_RESOLVENT_REFUSALS

    ids = [row[0] for row in _W_RPA_RESOLVENT_REFUSALS]
    assert len(ids) == len(set(ids)), f"duplicate rule id in {ids}"
    assert all(rid.startswith("w_rpa_resolvent_") for rid in ids)
    for row in _W_RPA_RESOLVENT_REFUSALS:
        rule_id, predicate, got, want, fix, doc = row
        assert callable(predicate) and callable(got)
        for text, part in ((rule_id, "id"), (want, "want"), (fix, "fix"),
                           (doc, "why")):
            assert isinstance(text, str) and len(text) > 8, (
                f"{rule_id}: {part} is missing or too short to be advice")


# ---------------------------------------------------------------------------
# 3. The runtime occupation gate carries the right rule id
# ---------------------------------------------------------------------------

def test_the_runtime_gate_labels_itself_by_diagram_name():
    import numpy as np

    from gw.screening_bse import refuse_fractional_occupations

    occ = np.zeros((1, 3, 8), dtype=np.float64)
    occ[:, :, :4] = 1.0
    occ[0, 1, 2] = 0.5
    with pytest.raises(NotImplementedError) as exc:
        refuse_fractional_occupations(
            occ, band_lo=0, band_hi=8, source="metal.h5",
            print_fn=lambda *a, **k: None,
            diagram_name="w_rpa_resolvent")
    message = str(exc.value)
    assert "w_rpa_resolvent_insulators_only" in message
    assert "w_bse_insulators_only" not in message
    for part in ("got:", "want:", "fix:", "why:", "doc:"):
        assert part in message


def test_the_runtime_gate_default_is_unchanged_for_existing_callers():
    """No ``diagram_name`` kwarg -> byte-identical ``w_bse`` behaviour."""
    import numpy as np

    from gw.screening_bse import refuse_fractional_occupations

    occ = np.zeros((1, 3, 8), dtype=np.float64)
    occ[:, :, :4] = 1.0
    occ[0, 1, 2] = 0.5
    with pytest.raises(NotImplementedError, match="w_bse_insulators_only"):
        refuse_fractional_occupations(
            occ, band_lo=0, band_hi=8, source="metal.h5",
            print_fn=lambda *a, **k: None)


# ---------------------------------------------------------------------------
# 4. THE KEY ACTUALLY REACHES include_w=False, and never the RPA executor
# ---------------------------------------------------------------------------

class _Reached(Exception):
    pass


def _stub_kwargs(config, tmp_path):
    return dict(
        quad=None, e_ref=0.0, sym=None, centroid_indices=None,
        config=config, meta=None, mesh_xy=None,
        run_dir=str(tmp_path / "mpa"), label="unit",
        material_class="insulator",
        tensors_filename=str(tmp_path / "isdf_tensors_4.h5"),
        print_fn=lambda *a, **k: None)


def test_w_rpa_resolvent_reaches_the_ladder_helper_with_include_w_false(
        tmp_path, monkeypatch):
    from gw import screening, screening_bse

    config = _config(tmp_path, "screening_diagrams = w_rpa_resolvent\n"
                               "compute_mode = cohsex\n")
    seen = {}

    def _stub(*a, **k):
        seen["include_w"] = k.get("include_w")
        raise _Reached("ladder")

    monkeypatch.setattr(screening_bse, "compute_screening_ladder", _stub)
    monkeypatch.setattr(
        screening, "compute_screening",
        lambda *a, **k: pytest.fail("w_rpa_resolvent took the RPA executor"))

    with pytest.raises(_Reached, match="ladder"):
        screening.compute_screening_model(
            ComputeMode.COHSEX, None, None, **_stub_kwargs(config, tmp_path))
    assert seen["include_w"] is False


def test_w_bse_still_reaches_include_w_true_after_the_third_member(
        tmp_path, monkeypatch):
    """The control arm: the generalized dispatch must not flip w_bse's own."""
    from gw import screening, screening_bse

    config = _config(tmp_path, "screening_diagrams = w_bse\n"
                               "compute_mode = cohsex\n", name="bse_ctl.in")
    seen = {}

    def _stub(*a, **k):
        seen["include_w"] = k.get("include_w")
        raise _Reached("ladder")

    monkeypatch.setattr(screening_bse, "compute_screening_ladder", _stub)
    with pytest.raises(_Reached, match="ladder"):
        screening.compute_screening_model(
            ComputeMode.COHSEX, None, None, **_stub_kwargs(config, tmp_path))
    assert seen["include_w"] is True


# ---------------------------------------------------------------------------
# 5. The driver's W0-flush predicate treats both resolvent diagrams alike
# ---------------------------------------------------------------------------

def test_the_driver_skips_its_own_w0_flush_under_w_rpa_resolvent_too():
    from gw.screening import driver_persists_w0

    class _Cfg:
        def __init__(self, diagrams):
            self.screening = type("S", (), {"diagrams": diagrams})()

    resolvent = _Cfg(ScreeningDiagrams.W_RPA_RESOLVENT)
    assert driver_persists_w0(ComputeMode.COHSEX, resolvent) is False
    assert driver_persists_w0(ComputeMode.GN_PPM, resolvent) is False

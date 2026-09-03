"""Input-controlled MPA sample geometry, independent of metal physics."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gw.gw_config import LorraxConfig, _DEFAULTS, validate_material_inputs
from gw.mpa import model, sample_plan, sampling


_BASE = """\
[cohsex]
nval = 2
ncond = 2
nband = 10
memory_per_device_gb = 4.0
"""


def _config(tmp_path, extra=""):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "mpa_sampling.in"
    path.write_text(_BASE + extra)
    return LorraxConfig.from_input_file(
        str(path), print_fn=lambda *args, **kwargs: None)


_METAL_KEYS = (
    "compute_mode = mpa\n"
    "occ_smearing_family = mp1\n"
    "occ_smearing_width_ry = 0.02\n"
    "fermi_reference = mp1_fixed_n\n"
)


def test_metal_sampling_flags_build_the_configured_grid(tmp_path):
    config = _config(
        tmp_path,
        "mpa_n_poles = 4\n"
        + _METAL_KEYS +
        "mpa_sampling_alpha = 2\n"
        "mpa_varpi_near_ry = 0.15\n"
        "mpa_varpi_far_ry = 1.5\n",
    )
    plan = config.mpa.sample_plan(8.0, material_class="metal")
    np.testing.assert_array_equal(
        sample_plan.plan_z(plan),
        np.asarray([
            2.0e-5j, 0.5 + 0.15j, 2.0 + 0.15j, 8.0 + 0.15j,
            1.5j, 0.5 + 1.5j, 2.0 + 1.5j, 8.0 + 1.5j,
        ], dtype=np.complex128),
    )


def test_leon_schedule_and_companion_solver_are_deck_selectable(tmp_path):
    config = _config(
        tmp_path,
        "mpa_n_poles = 15\n" + _METAL_KEYS
        + "mpa_sampling_alpha = 2\n"
        + "mpa_sampling_schedule = leon\n"
        + "mpa_pole_solver = companion\n")
    assert config.mpa.sampling_schedule == "leon"
    assert config.mpa.pole_solver == "companion"
    z = sample_plan.plan_z(
        config.mpa.sample_plan(8.0, material_class="metal"))
    expected_real = np.asarray(
        [float(f) ** 2 * 8.0
         for f in sampling.leon_partition_fractions(15)])
    np.testing.assert_array_equal(z[:15].real, expected_real)


def test_thiele_solver_is_deck_selectable(tmp_path):
    config = _config(tmp_path, "mpa_pole_solver = thiele\n")
    assert config.mpa.pole_solver == "thiele"


def test_explicit_mpa_fit_reuse_path_resolves_beside_deck(tmp_path):
    config = _config(
        tmp_path,
        "compute_mode = mpa\n"
        "mpa_fit_reuse_file = ../parent/mpa_fit_oneshot.h5\n",
    )
    assert os.path.normpath(config.mpa.fit_reuse_file) == os.path.normpath(
        tmp_path / "../parent/mpa_fit_oneshot.h5")


def test_mpa_fit_reuse_refuses_a_non_mpa_mode(tmp_path):
    with pytest.raises(ValueError, match="mpa_fit_reuse_file.*compute_mode"):
        _config(tmp_path, "mpa_fit_reuse_file = poles.h5\n")


def test_explicit_mpa_fit_reuse_gates_the_fresh_head_allocation():
    source = (Path(__file__).parents[1] / "src/gw/gw_jax.py").read_text()
    plan_start = source.index("oneshot_mpa_plan = None")
    screening_start = source.index("# SC solves W inside each map", plan_start)
    head_setup = source[plan_start:screening_start]
    assert "config.mpa.fit_reuse_file is not None" in head_setup
    assert "if oneshot_omegas.size and not reused_mpa_fit_owns_head" in head_setup


def test_completed_artifact_overwrite_is_an_explicit_false_by_default(tmp_path):
    default = _config(tmp_path / "default")
    enabled = _config(
        tmp_path / "enabled",
        "mpa_overwrite_completed_artifacts = true\n")
    assert default.mpa.overwrite_completed_artifacts is False
    assert enabled.mpa.overwrite_completed_artifacts is True


@pytest.mark.parametrize(
    "trs_allowed,expected_names",
    [
        (True, ("chi_qmunu_z", "Wc_qmunu_z")),
        (False, (
            "chi_qmunu_z", "Wc_qmunu_z",
            "chi_qmunu_minus_conj_z", "Wc_qmunu_minus_z")),
    ],
)
def test_build_checks_both_managed_artifacts_before_allocation(
        tmp_path, monkeypatch, trs_allowed, expected_names):
    """The format guard runs before the first inode-owning writer."""
    import common.collectives as collectives

    config = _config(tmp_path / "deck")
    sample_path = str(tmp_path / "mpa_samples_oneshot.h5")
    fit_path = str(tmp_path / "mpa_fit_oneshot.h5")
    calls = []

    monkeypatch.setattr(collectives, "process_rank", lambda: 0)
    monkeypatch.setattr(collectives, "barrier", lambda *args, **kwargs: None)
    monkeypatch.setattr(model, "make_mpa_plan", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        sample_plan, "plan_z",
        lambda _plan: np.asarray([0.0 + 0.1j, 1.0 + 0.1j]))
    monkeypatch.setattr(sample_plan, "refuse_unsupported", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        model, "_q_wedge",
        lambda *args, **kwargs: (np.asarray([0]), object(), object()))
    monkeypatch.setattr(
        model, "iteration_artifact_paths",
        lambda *args, **kwargs: (sample_path, fit_path))

    class Protected(RuntimeError):
        pass

    def guard(samples, fit, *, sample_names, overwrite_completed):
        calls.append((samples, fit, tuple(sample_names), overwrite_completed))
        raise Protected("stop before allocation")

    monkeypatch.setattr(
        model.mpa_store, "refuse_completed_artifact_replacement", guard)
    monkeypatch.setattr(
        model.mpa_store, "allocate_w_omega_collective",
        lambda *args, **kwargs: pytest.fail("sample inode touched before guard"))

    with pytest.raises(Protected, match="stop before allocation"):
        model.build_mpa_fit(
            tmp_path, "oneshot", wfns=None, V_q=None,
            quad=SimpleNamespace(x_max=1.0),
            sym=SimpleNamespace(trs_allowed=trs_allowed),
            centroid_indices=None, head_resolver=None, config=config,
            meta=None, mesh_xy=None, material_class="insulator")

    assert calls == [(
        sample_path,
        fit_path,
        expected_names,
        False,
    )]


@pytest.mark.parametrize(
    "line,value",
    [("mpa_sampling_schedule = improvised\n", "mpa_sampling_schedule"),
     ("mpa_pole_solver = regularized\n", "mpa_pole_solver")])
def test_unknown_leon_recipe_selector_refuses(tmp_path, line, value):
    with pytest.raises(ValueError, match=value):
        _config(tmp_path, line)


def test_material_class_is_not_a_deck_key():
    assert "mpa_material_class" not in _DEFAULTS


def test_metal_evaluator_refuses_before_creating_output(tmp_path):
    # The blanket capability gate is discharged; the entry now refuses a
    # metal plan without an OccupationState, still before any inode exists.
    config = _config(tmp_path, _METAL_KEYS)
    run_dir = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="mpa_metal_needs_occupations"):
        model.build_mpa_fit(
            run_dir, "metal", wfns=None, V_q=None, quad=None, sym=None,
            centroid_indices=None, head_resolver=None, config=config,
            meta=None, mesh_xy=None, material_class="metal")
    assert not run_dir.exists()


# --- Metal deck-key cross-validation (W4) ---------------------------------
# A parsed key either has a consumer or refuses by name; the metal pair is
# required together, and both are off-dials under an insulator.


def test_metal_without_the_smearing_pair_refuses_by_name(tmp_path):
    config = _config(tmp_path, "compute_mode = mpa\nfermi_reference = mp1_fixed_n\n")
    with pytest.raises(ValueError, match="occ_smearing_family"):
        validate_material_inputs(config, "metal")


def test_metal_with_a_gap_fermi_reference_refuses_by_name(tmp_path):
    config = _config(
        tmp_path,
        "compute_mode = mpa\n"
        "occ_smearing_family = mp1\n"
        "occ_smearing_width_ry = 0.02\n")
    with pytest.raises(ValueError, match="mp1_fixed_n"):
        validate_material_inputs(config, "metal")


def test_sigma_layout_is_not_a_deck_key():
    assert "sigma_omega_layout" not in _DEFAULTS


def test_insulator_with_smearing_keys_refuses_the_off_dial(tmp_path):
    config = _config(
        tmp_path,
        "occ_smearing_family = mp1\nocc_smearing_width_ry = 0.02\n")
    with pytest.raises(ValueError, match="integer WFN occupations"):
        validate_material_inputs(config, "insulator")


def test_insulator_with_mp1_fermi_reference_refuses(tmp_path):
    config = _config(tmp_path, "fermi_reference = mp1_fixed_n\n")
    with pytest.raises(ValueError, match="integer WFN occupations"):
        validate_material_inputs(config, "insulator")


def test_uncertified_smearing_family_refuses_by_name(tmp_path):
    with pytest.raises(ValueError, match="only 'mp1'"):
        _config(
            tmp_path,
            _METAL_KEYS.replace(
                "occ_smearing_family = mp1",
                "occ_smearing_family = fermi_dirac"))


def test_a_legal_metal_deck_parses_and_carries_the_pair(tmp_path):
    config = _config(tmp_path, _METAL_KEYS)
    validate_material_inputs(config, "metal")
    assert config.occ_smearing_family == "mp1"
    assert config.occ_smearing_width_ry == 0.02
    assert config.sigma.fermi_reference == "mp1_fixed_n"


@pytest.mark.parametrize("mode", ("x_only", "cohsex", "gn_ppm", "hl_ppm"))
def test_a_metal_deck_refuses_every_mode_without_an_occupation_aware_head(
        tmp_path, mode):
    with pytest.raises(ValueError) as excinfo:
        config = _config(
            tmp_path,
            _METAL_KEYS.replace(
                "compute_mode = mpa", f"compute_mode = {mode}"))
        validate_material_inputs(config, "metal")

    message = str(excinfo.value)
    assert "GATE fractional_occupations_require_mpa" in message
    assert f"compute_mode={mode}" in message
    assert "compute_mode=mpa" in message


def test_a_metal_deck_refuses_auto_after_naming_its_resolved_mode(tmp_path):
    with pytest.raises(ValueError) as excinfo:
        config = _config(
            tmp_path, _METAL_KEYS.replace("compute_mode = mpa\n", ""))
        validate_material_inputs(config, "metal")

    message = str(excinfo.value)
    assert "compute_mode=cohsex" in message
    assert "compute_mode=mpa" in message


def test_an_insulating_deck_carries_no_smearing_pair(tmp_path):
    config = _config(tmp_path)
    assert config.occ_smearing_family is None
    assert config.occ_smearing_width_ry is None


# --- Occupation provenance stamps (W4) ------------------------------------
# Written only when a state is supplied (insulating stores byte-identical);
# asserted at reuse sites; refuse an unstamped store under a metallic reuse.


def _occ_state(**over):
    from types import SimpleNamespace

    base = dict(
        f_kn=None, mu_ry=0.379, smearing_family="mp1",
        smearing_width_ry=0.02, n_electrons=9.0, occ_hash="abc123def4567890")
    base.update(over)
    return SimpleNamespace(**base)


def _tiny_fit_store(tmp_path, occupation_state):
    from file_io import mpa_store as MS

    dest = tmp_path / "fit.h5"
    MS.allocate_fit_store(
        dest, n_q=1, n_mu=2, n_p=1, energy_unit="Ry",
        occupation_state=occupation_state)
    return dest


def test_stamps_round_trip_through_the_fit_store(tmp_path):
    from file_io import mpa_store as MS

    dest = _tiny_fit_store(tmp_path, _occ_state())
    stamps = MS.read_occupation_stamps(dest)
    assert stamps["occ_hash"] == "abc123def4567890"
    assert stamps["smearing_family"] == "mp1"
    assert stamps["smearing_width_ry"] == 0.02
    assert stamps["occ_nelec"] == 9.0
    MS.assert_occupation_stamps(dest, _occ_state())


def test_a_mismatched_state_is_refused_with_the_field_named(tmp_path):
    from file_io import mpa_store as MS

    dest = _tiny_fit_store(tmp_path, _occ_state())
    with pytest.raises(ValueError, match="occ_hash"):
        MS.assert_occupation_stamps(dest, _occ_state(occ_hash="deadbeef"))


def test_a_legacy_zero_pad_hash_is_accepted_only_when_reproduced(tmp_path):
    from file_io import mpa_store as MS

    legacy = _occ_state(occ_hash="legacy-p36")
    dest = _tiny_fit_store(tmp_path, legacy)
    live = _occ_state(occ_hash="logical")
    assert MS.assert_occupation_stamps(
        dest, live, compatible_occ_hashes={"legacy-p36"}) == "legacy_zero_pad"
    with pytest.raises(ValueError, match="occ_hash"):
        MS.assert_occupation_stamps(
            dest, live, compatible_occ_hashes={"some-other-hash"})


def test_an_unstamped_store_is_refused_at_a_metallic_reuse(tmp_path):
    from file_io import mpa_store as MS

    dest = _tiny_fit_store(tmp_path, None)
    with pytest.raises(ValueError, match="no occupation stamps"):
        MS.assert_occupation_stamps(dest, _occ_state())


def test_an_insulating_store_carries_no_occ_attrs(tmp_path):
    import h5py

    from file_io import mpa_store as MS

    dest = _tiny_fit_store(tmp_path, None)
    assert MS.read_occupation_stamps(dest) is None
    banned = {"mpa_" + key for key in MS._OCC_STAMP_ORDER}
    with h5py.File(dest, "r") as f:

        def _no_occ(name, obj):
            hit = banned & set(map(str, obj.attrs))
            assert not hit, (name, hit)

        _no_occ("/", f["/"])
        f.visititems(_no_occ)


# --- One width, one convention (2026-08-15) --------------------------------
# ``occ_smearing_width_ry`` had no runtime consumer while every MP1 solve read
# ``occ_broadening``, and the two keys were documented in DIFFERENT
# conventions (QE degauss vs BerkeleyGW half-width).  The decision, recorded
# in ``_validate_occupation_smearing`` and ``LorraxConfig.occ_broadening_ry``:
# BOTH keys carry BerkeleyGW's ``occ_broadening`` — MP1 argument
# ``(E-mu)/(2*width)`` — so both are HALF the QE degauss, the Ry-valued key is
# what the solve consumes, and a disagreement is refused at parse.

_SC_KEYS = (
    "qp_solver = self_consistent\n"
    "self_consistent = true\n"
    "sc_accelerator = linear\n"
    "sc_head_update = parallel_transport\n"
)

#: The staged sodium deck: QE degauss 0.02 Ry, therefore half-width 0.01 Ry,
#: therefore occ_broadening = 0.01 * 13.605693122994 eV/Ry.
_NA_WIDTH_RY = 0.01
_NA_BROADENING_EV = 0.13605693122994


def test_disagreeing_width_keys_refuse_by_name_with_the_conversion(tmp_path):
    """The factor-of-two trap: someone types the QE degauss into the Ry key.

    ``_METAL_KEYS`` carries ``occ_smearing_width_ry = 0.02``, which is the
    sodium deck's QE ``degauss``; beside it ``occ_broadening =
    0.13605693122994`` eV is the same deck's half-width, 0.01 Ry.  Exactly
    2x apart, and only one of them can be the width the MP1 solve takes.
    """
    with pytest.raises(ValueError, match="occ_smearing_width_ry") as excinfo:
        _config(
            tmp_path,
            _METAL_KEYS + _SC_KEYS
            + f"occ_broadening = {_NA_BROADENING_EV}\n")
    message = str(excinfo.value)
    # The refusal has to carry the conversion, or it just says "no".
    assert "occ_broadening" in message
    assert "degauss" in message
    assert "(E-mu)/(2*width)" in message


def test_agreeing_width_keys_parse_and_the_ry_key_is_the_exact_one(tmp_path):
    """The staged sodium deck's own pair, and what the solve then consumes."""
    config = _config(
        tmp_path,
        _METAL_KEYS.replace(
            "occ_smearing_width_ry = 0.02",
            f"occ_smearing_width_ry = {_NA_WIDTH_RY}")
        + _SC_KEYS
        + f"occ_broadening = {_NA_BROADENING_EV}\n")
    # The two keys agree to 3.6e-7 relative -- not exactly, because the deck's
    # eV value was written with CODATA 13.605693122994 while the code converts
    # with common.units.RYD_TO_EV.  That is the whole reason the tolerance is
    # relative and the Ry key wins: it is the one that made no round trip.
    from common.units import RYD_TO_EV

    round_trip = config.screening.occ_broadening_ev / RYD_TO_EV
    assert abs(round_trip - _NA_WIDTH_RY) < 1.0e-6 * _NA_WIDTH_RY
    assert round_trip != _NA_WIDTH_RY
    assert config.occ_broadening_ry == _NA_WIDTH_RY


def test_the_width_the_solve_consumes_comes_from_the_new_key(tmp_path):
    """``occ_broadening_ry`` prefers ``occ_smearing_width_ry``, and the QE
    degauss is twice it -- the one arithmetic statement every metal deck
    depends on."""
    import dataclasses

    config = _config(tmp_path, _METAL_KEYS)          # width key = 0.02 Ry
    assert config.occ_broadening_ry == 0.02
    assert 2.0 * config.occ_broadening_ry == 0.04    # the QE degauss

    # Break the tie the parse-time refusal normally forbids: the property
    # must read the Ry key, not the eV one.  (Constructed, not parsed --
    # a deck like this cannot exist.)
    moved = dataclasses.replace(config, occ_smearing_width_ry=0.031)
    assert moved.occ_broadening_ry == 0.031


def test_an_insulating_deck_still_resolves_its_width_from_occ_broadening(
        tmp_path):
    """No metal key => the historical path, bit-for-bit."""
    from common.units import RYD_TO_EV

    config = _config(
        tmp_path, _SC_KEYS + f"occ_broadening = {_NA_BROADENING_EV}\n")
    assert config.occ_smearing_width_ry is None
    assert (config.occ_broadening_ry
            == float(_NA_BROADENING_EV) / RYD_TO_EV)


def test_the_step_occupation_control_arm_still_parses(tmp_path):
    """``occ_broadening = 0`` is the DIAL, not a width, so a metal deck may
    carry a live smearing pair beside it (the b24/b40 stepocc arms do)."""
    config = _config(tmp_path, _METAL_KEYS + "occ_broadening = 0.0\n")
    assert config.screening.occ_broadening_ev == 0.0
    assert config.occ_broadening_ry == 0.02


# --- The metal origin shift, as a deck key (2026-08-15) --------------------
# ``sampling._METAL_ORIGIN_SHIFT`` was a module-private constant with no deck
# key, no parameter and no env override -- the one sampling-geometry quantity
# that could not be laddered, and exactly the one the shifted-origin
# conditioning risk turns on (KNOWN_LORRAX_ISSUES, gw/mpa/sampling row).  The
# key mirrors ``mpa_varpi_near_ry``'s shape: unset = the published constant,
# bit-for-bit; set = refused unless it is a metal deck and 0 < shift < near.
#
# THE UNIT.  The key is Ry, like every deck key; the papers and the owner's
# fallback ladder are Ha; Ry = 2*Ha.  The ladder 1e-4 / 3e-4 / 1e-3 Ha is
# typed as 2e-4 / 6e-4 / 2e-3 Ry, and both columns are written out below
# because halving or doubling this quantity is invisible in the output.

#: The published default, in both columns.
_SHIFT_DEFAULT_HA = 1.0e-5
_SHIFT_DEFAULT_RY = 2.0e-5

#: The owner's R4 fallback ladder, Ha as quoted -> Ry as typed into a deck.
_LADDER_HA_TO_RY = ((1.0e-4, 2.0e-4), (3.0e-4, 6.0e-4), (1.0e-3, 2.0e-3))


def test_the_shift_default_and_ladder_are_a_factor_of_two_apart():
    """The Ha<->Ry doubling, stated as arithmetic rather than as prose."""
    from gw.mpa import sampling

    assert sampling._METAL_ORIGIN_SHIFT["Ha"] == _SHIFT_DEFAULT_HA
    assert sampling._METAL_ORIGIN_SHIFT["Ry"] == _SHIFT_DEFAULT_RY
    assert (sampling._METAL_ORIGIN_SHIFT["Ry"]
            == 2.0 * sampling._METAL_ORIGIN_SHIFT["Ha"])
    for ha, ry in _LADDER_HA_TO_RY:
        assert ry == pytest.approx(2.0 * ha, rel=1e-15)


# (1) REFUSAL: an insulating deck must refuse the key by name, and an
#     out-of-range shift must refuse by name whichever side it falls on.


def test_insulator_with_the_origin_shift_refuses_the_off_dial(tmp_path):
    config = _config(tmp_path, "mpa_metal_origin_shift_ry = 2e-4\n")
    with pytest.raises(ValueError, match="mpa_metal_origin_shift_ry"):
        validate_material_inputs(config, "insulator")


def test_insulator_refusal_says_it_is_metal_only(tmp_path):
    config = _config(tmp_path, "mpa_metal_origin_shift_ry = 2e-4\n")
    with pytest.raises(ValueError, match="metal-only"):
        validate_material_inputs(config, "insulator")


@pytest.mark.parametrize("bad", ["0.0", "-2e-4", "0.2", "0.5"])
def test_a_shift_outside_zero_to_varpi_near_refuses_by_name(tmp_path, bad):
    """0 and varpi_near are both EXCLUDED: at 0 the sample is the insulating
    origin the metal protocol exists to avoid, and at varpi_near it has
    climbed onto the near line and collides with its own first partition
    point when omega_1 = 0."""
    with pytest.raises(ValueError, match="mpa_metal_origin_shift_ry"):
        _config(tmp_path, _METAL_KEYS + f"mpa_metal_origin_shift_ry = {bad}\n")


def test_the_range_refusal_carries_the_unit_warning(tmp_path):
    with pytest.raises(ValueError) as excinfo:
        _config(tmp_path, _METAL_KEYS + "mpa_metal_origin_shift_ry = 0.5\n")
    message = str(excinfo.value)
    assert "mpa_varpi_near_ry" in message
    assert "1e-5 Ha = 2e-5 Ry" in message


def test_the_sampler_refuses_an_origin_shift_under_an_insulator():
    """The library door has the same off-dial, so a direct caller cannot
    pass a shift that would be silently dropped at ``near[0] = 0``."""
    from gw.mpa import sampling

    with pytest.raises(ValueError, match="GATE origin_shift_metal_only"):
        sampling.double_parallel_grid(4, 4.0, origin_shift=1.0e-4)
    with pytest.raises(ValueError, match="GATE origin_shift_ordering"):
        sampling.double_parallel_grid(
            4, 4.0, material_class="metal", origin_shift=1.0)


# (2) DEFAULT-IDENTITY: a deck that does not set the key builds the grid it
#     built before the key existed, bit-for-bit, and so does one that types
#     the published default in by hand.


def test_an_unset_key_is_the_published_constant_bit_for_bit(tmp_path):
    config = _config(tmp_path, "mpa_n_poles = 4\n" + _METAL_KEYS)
    assert config.mpa.metal_origin_shift_ry is None

    from gw.mpa import sampling

    z = sample_plan.plan_z(
        config.mpa.sample_plan(8.0, material_class="metal"))
    reference = sampling.double_parallel_grid(
        4, 8.0, material_class="metal", varpi_near=0.2, varpi_far=2.0,
        energy_unit="Ry")
    np.testing.assert_array_equal(z, reference)
    assert z[0] == 1j * _SHIFT_DEFAULT_RY


def test_typing_the_default_in_by_hand_changes_no_bit(tmp_path):
    base = _config(tmp_path / "a", "mpa_n_poles = 4\n" + _METAL_KEYS)
    typed = _config(
        tmp_path / "b",
        "mpa_n_poles = 4\n" + _METAL_KEYS
        + f"mpa_metal_origin_shift_ry = {_SHIFT_DEFAULT_RY}\n")
    np.testing.assert_array_equal(
        sample_plan.plan_z(base.mpa.sample_plan(
            8.0, material_class="metal")),
        sample_plan.plan_z(typed.mpa.sample_plan(
            8.0, material_class="metal")))


def test_an_insulating_grid_is_untouched_by_the_new_parameter(tmp_path):
    """The other half of default-identity: the key cannot reach an
    insulating plan at all, so z = 0 stays exactly 0."""
    config = _config(tmp_path, "mpa_n_poles = 4\n")
    z = sample_plan.plan_z(
        config.mpa.sample_plan(8.0, material_class="insulator"))
    assert z[0] == 0.0 + 0.0j


@pytest.mark.parametrize("shift_ha,shift_ry", _LADDER_HA_TO_RY)
def test_each_ladder_rung_lands_where_the_owner_quoted_it(
        tmp_path, shift_ha, shift_ry):
    """The whole point of the key: the R4 fallback ladder is runnable from a
    deck.  The assertion is against the HARTREE number the owner quoted, so
    a unit slip in the plumbing fails here rather than in a fit diagnostic."""
    config = _config(
        tmp_path,
        "mpa_n_poles = 4\n" + _METAL_KEYS
        + f"mpa_metal_origin_shift_ry = {shift_ry}\n")
    assert config.mpa.metal_origin_shift_ry == shift_ry
    z = sample_plan.plan_z(
        config.mpa.sample_plan(8.0, material_class="metal"))
    assert z[0] == 1j * shift_ry
    assert z[0].imag == pytest.approx(2.0 * shift_ha, rel=1e-15)
    # Nothing else on either line moved.
    reference = _config(tmp_path / "ref", "mpa_n_poles = 4\n" + _METAL_KEYS)
    np.testing.assert_array_equal(
        z[1:], sample_plan.plan_z(reference.mpa.sample_plan(
            8.0, material_class="metal"))[1:])


# (3) STAMPED ROUND-TRIP: the resolved shift reaches the sample store as an
#     ADDITIVE attr, and a deck that leaves it unset writes the same bytes.


def _sampling_record(config, omega_max):
    """Exactly ``model.build_mpa_fit``'s record, without the compute."""
    record = {"protocol": "double_parallel", "varpi": np.array([0.2, 2.0]),
              "n_p": int(config.mpa.n_poles), "alpha": 1,
              "omega_max": float(omega_max)}
    if config.mpa.metal_origin_shift_ry is not None:
        record["metal_origin_shift_ry"] = float(
            config.mpa.metal_origin_shift_ry)
    return record


def _stamped_store(path, record):
    from file_io import mpa_store as MS

    from tests._mpa_test_geometry import geometry

    tables, verdict, n_mu = geometry([[1, 2, 3], [4, 5, 6]])
    MS.allocate_w_omega(
        path, "chi_qmunu_z", n_omega=2, n_q_on_disk=3, n_mu=n_mu,
        tables=tables, omega=np.array([2.0e-5j, 0.5 + 0.2j]),
        omega_line=np.array([0, 1], dtype=np.int32), sampling=record,
        closure_verdict=verdict, energy_unit="Ry")
    return path


def test_a_declared_shift_is_stamped_on_the_sample_store(tmp_path):
    import h5py

    config = _config(
        tmp_path, _METAL_KEYS + "mpa_metal_origin_shift_ry = 6e-4\n")
    path = _stamped_store(
        str(tmp_path / "samples.h5"), _sampling_record(config, 8.0))
    with h5py.File(path, "r") as f:
        attrs = f["chi_qmunu_z"].attrs
        assert attrs["mpa_prov_metal_origin_shift_ry"] == 6.0e-4


def test_an_undeclared_shift_leaves_the_store_byte_identical(tmp_path):
    """ADDITIVE means additive.  Holding the ω grid fixed isolates the ONE
    thing the key changes in the store: an attr that appears when the deck
    declares it and is absent otherwise.  So a deck that leaves the key
    unset hands ``allocate_w_omega`` the same record it handed it before the
    key existed -- same attr set, same digest, same bytes."""
    import h5py

    plain = _config(tmp_path, _METAL_KEYS)
    shifted = _config(
        tmp_path / "b", _METAL_KEYS + "mpa_metal_origin_shift_ry = 6e-4\n")
    p0 = _stamped_store(
        str(tmp_path / "plain.h5"), _sampling_record(plain, 8.0))
    p1 = _stamped_store(
        str(tmp_path / "shifted.h5"), _sampling_record(shifted, 8.0))

    with h5py.File(p0, "r") as f0, h5py.File(p1, "r") as f1:
        a0 = dict(f0["chi_qmunu_z"].attrs)
        a1 = dict(f1["chi_qmunu_z"].attrs)
        assert "mpa_prov_metal_origin_shift_ry" not in a0
        # The ONLY difference is the additive attr.
        assert set(a1) - set(a0) == {"mpa_prov_metal_origin_shift_ry"}
        assert not set(a0) - set(a1)
        # And the digest -- the store's identity -- is untouched by it.
        assert a0["mpa_grid_hash"] == a1["mpa_grid_hash"]


def test_the_stamp_is_outside_the_omega_grid_digest():
    """Source-level pin of the same fact: ``_SAMPLING_ORDER`` is what the
    digest hashes, and the shift is deliberately not in it."""
    from file_io import mpa_store as MS

    assert "metal_origin_shift_ry" not in MS._SAMPLING_ORDER
    omega = np.array([2.0e-5j, 0.5 + 0.2j])
    line = np.array([0, 1], dtype=np.int32)
    base = {"protocol": "double_parallel", "varpi": np.array([0.2, 2.0]),
            "n_p": 4, "alpha": 1, "omega_max": 8.0}
    assert (MS.omega_grid_digest(omega, line, base)
            == MS.omega_grid_digest(
                omega, line, dict(base, metal_origin_shift_ry=6.0e-4)))


def test_no_mp1_solve_converts_occ_broadening_behind_the_property():
    """Source pin: the four sc_iteration width reads go through the one
    resolver.  A new ``occ_broadening_ev / RYD_TO_EV`` anywhere in the SC
    driver is a second width source and reopens the gap this closed."""
    import inspect

    from gw import sc_iteration

    source = inspect.getsource(sc_iteration)
    assert "occ_broadening_ev) / RYD_TO_EV" not in source
    assert "occ_broadening_ev / RYD_TO_EV" not in source
    assert source.count("config.occ_broadening_ry") >= 3

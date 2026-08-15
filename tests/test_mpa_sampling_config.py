"""Input-controlled MPA sample geometry, independent of metal physics."""

from __future__ import annotations

import numpy as np
import pytest

from gw.gw_config import LorraxConfig
from gw.mpa import model, sample_plan


_BASE = """\
[cohsex]
nval = 2
ncond = 2
nband = 10
memory_per_device_gb = 4.0
"""


def _config(tmp_path, extra=""):
    path = tmp_path / "mpa_sampling.in"
    path.write_text(_BASE + extra)
    return LorraxConfig.from_input_file(
        str(path), print_fn=lambda *args, **kwargs: None)


_METAL_KEYS = (
    "mpa_material_class = metal\n"
    "occ_smearing_family = mp1\n"
    "occ_smearing_width_ry = 0.02\n"
    "fermi_reference = mp1_fixed_n\n"
    "sigma_omega_layout = sharded\n"
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

    plan = config.mpa.sample_plan(8.0)
    np.testing.assert_array_equal(
        sample_plan.plan_z(plan),
        np.asarray([
            2.0e-5j, 0.5 + 0.15j, 2.0 + 0.15j, 8.0 + 0.15j,
            1.5j, 0.5 + 1.5j, 2.0 + 1.5j, 8.0 + 1.5j,
        ], dtype=np.complex128),
    )


def test_unknown_material_class_is_a_parse_error(tmp_path):
    with pytest.raises(ValueError, match="mpa_material_class"):
        _config(tmp_path, "mpa_material_class = semimetal\n")


def test_metal_evaluator_refuses_before_creating_output(tmp_path):
    # The blanket capability gate is discharged; the entry now refuses a
    # metal plan without an OccupationState, still before any inode exists.
    config = _config(tmp_path, _METAL_KEYS)
    run_dir = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="mpa_metal_needs_occupations"):
        model.build_mpa_fit(
            run_dir, "metal", wfns=None, V_q=None, quad=None, sym=None,
            centroid_indices=None, head_resolver=None, config=config,
            meta=None, mesh_xy=None)
    assert not run_dir.exists()


# --- Metal deck-key cross-validation (W4) ---------------------------------
# A parsed key either has a consumer or refuses by name; the metal pair is
# required together, and both are off-dials under an insulator.


def test_metal_without_the_smearing_pair_refuses_by_name(tmp_path):
    with pytest.raises(ValueError, match="occ_smearing_family"):
        _config(
            tmp_path,
            "mpa_material_class = metal\n"
            "fermi_reference = mp1_fixed_n\n"
            "sigma_omega_layout = sharded\n")


def test_metal_with_a_gap_fermi_reference_refuses_by_name(tmp_path):
    with pytest.raises(ValueError, match="mp1_fixed_n"):
        _config(
            tmp_path,
            "mpa_material_class = metal\n"
            "occ_smearing_family = mp1\n"
            "occ_smearing_width_ry = 0.02\n"
            "sigma_omega_layout = sharded\n")


def test_metal_with_the_replicated_cube_refuses_by_name(tmp_path):
    with pytest.raises(ValueError, match="sigma_omega_layout"):
        _config(
            tmp_path,
            "mpa_material_class = metal\n"
            "occ_smearing_family = mp1\n"
            "occ_smearing_width_ry = 0.02\n"
            "fermi_reference = mp1_fixed_n\n")


def test_insulator_with_smearing_keys_refuses_the_off_dial(tmp_path):
    with pytest.raises(ValueError, match="metal-only"):
        _config(tmp_path, "occ_smearing_family = mp1\n")


def test_insulator_with_mp1_fermi_reference_refuses(tmp_path):
    with pytest.raises(ValueError, match="mpa_material_class = metal"):
        _config(tmp_path, "fermi_reference = mp1_fixed_n\n")


def test_uncertified_smearing_family_refuses_by_name(tmp_path):
    with pytest.raises(ValueError, match="only 'mp1'"):
        _config(
            tmp_path,
            _METAL_KEYS.replace(
                "occ_smearing_family = mp1",
                "occ_smearing_family = fermi_dirac"))


def test_a_legal_metal_deck_parses_and_carries_the_pair(tmp_path):
    config = _config(tmp_path, _METAL_KEYS)
    assert config.occ_smearing_family == "mp1"
    assert config.occ_smearing_width_ry == 0.02
    assert config.sigma.fermi_reference == "mp1_fixed_n"


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

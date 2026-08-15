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
    config = _config(tmp_path, _METAL_KEYS)
    run_dir = tmp_path / "must_not_exist"
    with pytest.raises(
            NotImplementedError, match="mpa_metal_evaluator_unavailable"):
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

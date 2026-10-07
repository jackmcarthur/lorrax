"""Representation/interaction separation for normalized RKB Coulomb GW.

These small host tests cover deck admission and the shared runtime selection.
The P4 driver evidence separately covers fitting, heads, Hartree and restart.
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from common.bispinor_init import NORMALIZED_RKB_LIFT
from common.four_current_model import resolve_four_current_representation
from gw.gw_config import (
    BispinorGWMode, HeadCorrection, LorraxConfig,
    mpa_sigma_runs_scalar_executor, packed_photon_replaces_charge_sigma,
    refuse_unsupported_bispinor_gw, uses_bare_tt_gamma_head,
    uses_coupled_photon_head, uses_dynamic_packed_photon_route,
    uses_full_bispinor_shared_pole, uses_static_photon_response,
    uses_transverse_interaction,
)


def config(tmp_path, body):
    deck = tmp_path / "cohsex.in"
    deck.write_text("[cohsex]\nnval = 1\nncond = 2\nnumber_bands = 7\n"
                    "sys_dim = 3\nqp_solver = one_shot_dft\nlinalg = local\n" + body)
    return LorraxConfig.from_input_file(
        str(deck), resolve_hardware=False, print_fn=lambda *_: None)


@pytest.mark.parametrize("mode", ["x_only", "cohsex", "gn_ppm", "hl_ppm", "mpa"])
def test_coulomb_only_keeps_four_charge_and_scalar_consumers(tmp_path, mode):
    cfg = config(tmp_path, "bispinor = true\nbispinor_gw = coulomb_only\n"
                 f"compute_mode = {mode}\n")
    rep = resolve_four_current_representation(cfg.bispinor, cfg.bispinor_gw)
    assert rep.charge_bispinor and rep.scalar_head_bispinor
    assert rep.charge_lift == NORMALIZED_RKB_LIFT
    assert cfg.head.correction is HeadCorrection.FULL
    assert not cfg.paths.centroids_file_current
    assert not uses_transverse_interaction(cfg)
    assert not uses_static_photon_response(cfg)
    assert not packed_photon_replaces_charge_sigma(cfg)
    assert not uses_dynamic_packed_photon_route(cfg)
    assert not uses_coupled_photon_head(cfg)
    assert not uses_bare_tt_gamma_head(cfg)
    assert not uses_full_bispinor_shared_pole(cfg)
    assert mpa_sigma_runs_scalar_executor(cfg)


def test_coulomb_only_accepts_scalar_head_diagnostic(tmp_path):
    cfg = config(tmp_path, "bispinor = true\nbispinor_gw = coulomb_only\n"
                 "compute_mode = mpa\nsigma_w_model = shared_pole\n"
                 "head_correction = no_local_fields\n")
    assert cfg.head.correction is HeadCorrection.NO_LOCAL_FIELDS
    refuse_unsupported_bispinor_gw(cfg)


def test_existing_bare_transverse_default_keeps_current(tmp_path):
    cfg = config(tmp_path, "bispinor = true\ncompute_mode = x_only\n")
    assert cfg.bispinor_gw is BispinorGWMode.BARE_TRANSVERSE
    assert uses_transverse_interaction(cfg)
    assert uses_bare_tt_gamma_head(cfg)
    assert resolve_four_current_representation(True, cfg.bispinor_gw) == (
        resolve_four_current_representation(True, BispinorGWMode.COULOMB_ONLY))


@pytest.mark.parametrize("mode", list(BispinorGWMode))
def test_representation_does_not_choose_interaction(mode):
    rep = resolve_four_current_representation(True, mode)
    assert rep.charge_lift == NORMALIZED_RKB_LIFT
    assert rep.current_lift == NORMALIZED_RKB_LIFT
    cfg = SimpleNamespace(bispinor=True, bispinor_gw=mode)
    assert uses_transverse_interaction(cfg) == (mode is not BispinorGWMode.COULOMB_ONLY)


def test_coulomb_only_refuses_scalar_representation(tmp_path):
    with pytest.raises(ValueError, match="coulomb_only_requires_bispinor"):
        config(tmp_path, "bispinor = false\nbispinor_gw = coulomb_only\n"
               "compute_mode = x_only\n")


def test_coulomb_only_refuses_transverse_head_in_programmatic_config(tmp_path):
    cfg = config(tmp_path, "bispinor = true\nbispinor_gw = coulomb_only\n"
                 "compute_mode = x_only\n")
    cfg = replace(cfg, head=replace(cfg.head, bispinor_tt_head_correction=True))
    with pytest.raises(ValueError, match="coulomb_only_transverse_head"):
        refuse_unsupported_bispinor_gw(cfg)


def test_unknown_interaction_refuses(tmp_path):
    with pytest.raises(ValueError, match="not a known mode"):
        config(tmp_path, "bispinor = true\nbispinor_gw = coulomb_oly\n")


@pytest.mark.parametrize("changes", [
    {"bispinor": False},
    {"bispinor_gw": BispinorGWMode.BARE_TRANSVERSE},
    {"bispinor_gw": BispinorGWMode.FULL_SHARED_POLE},
    {"sys_dim": 2},
    {"sys_dim": 0},
])
@pytest.mark.parametrize("restart", [False, True])
def test_augmentation_domain_checked_for_programmatic_configs(tmp_path, changes, restart):
    cfg = config(tmp_path, "bispinor = true\nbispinor_gw = coulomb_only\n"
                 "compute_mode = x_only\n")
    cfg = replace(cfg, paths=replace(cfg.paths, atomic_reconstruction_dir="atomic-data"),
                  restart=restart, **changes)
    with pytest.raises(ValueError, match="GATE atomic_augmentation_domain"):
        refuse_unsupported_bispinor_gw(cfg)


@pytest.mark.parametrize("restart", [False, True])
def test_augmentation_domain_keeps_coulomb_only_supported(tmp_path, restart):
    cfg = config(tmp_path, "bispinor = true\nbispinor_gw = coulomb_only\n"
                 "compute_mode = x_only\natomic_reconstruction_dir = atomic-data\n")
    refuse_unsupported_bispinor_gw(replace(cfg, restart=restart))

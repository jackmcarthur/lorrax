"""First-run refusals: a deck without sys_dim, and the band-extrapolation floor at startup.

Both used to fail late or not at all: an omitted ``sys_dim`` silently ran the
slab truncation on any cell, and ``number_bands_sigma < 2 n_occ`` under the
default ``use_band_extrapolation`` refused only at the Sigma stage, after the
zeta fit and the W build.
"""
from __future__ import annotations

import types

import pytest

from gw import gw_config
from gw.band_extrapolation import (BandExtrapolationRefused,
                                   require_extrapolation_band_floor)
from gw.gw_init import check_band_extrapolation_floor


def _deck(tmp_path, body):
    p = tmp_path / "deck.in"
    p.write_text("[cohsex]\nnval = 4\nncond = 4\nnumber_bands = 20\n" + body)
    return str(p)


def test_deck_without_sys_dim_refuses_by_name(tmp_path):
    with pytest.raises(ValueError, match="GATE sys_dim_required") as exc:
        gw_config.read_lorrax_input(_deck(tmp_path, ""))
    assert "sys_dim = 3 for bulk, 2 for a slab" in str(exc.value)


@pytest.mark.parametrize("value", [2, 3])
def test_deck_with_sys_dim_parses(tmp_path, value):
    params = gw_config.read_lorrax_input(_deck(tmp_path, f"sys_dim = {value}\n"))
    assert params["sys_dim"] == value


def test_extrapolation_floor_rule():
    require_extrapolation_band_floor(8, 16)          # n_cond == n_occ passes
    with pytest.raises(BandExtrapolationRefused, match="2\\*n_occ = 16"):
        require_extrapolation_band_floor(8, 15)


def _cfg(*, enabled, mode):
    return types.SimpleNamespace(
        sigma=types.SimpleNamespace(band_extrapolation=enabled),
        sc=types.SimpleNamespace(stages=()),
        compute_mode=mode)


_SLICES = types.SimpleNamespace(b0=0, b2=8, b4=16)
_META = types.SimpleNamespace(b_id_4_sigma_user=12)


def test_startup_floor_refuses_a_ppm_run_before_the_zeta_fit():
    with pytest.raises(BandExtrapolationRefused, match="n_cond = 4"):
        check_band_extrapolation_floor(
            _cfg(enabled=True, mode="gn_ppm"), _SLICES, _META)


@pytest.mark.parametrize("enabled,mode", [(False, "gn_ppm"), (True, "cohsex"),
                                          (True, "mpa")])
def test_startup_floor_skips_runs_that_do_not_consume_the_key(enabled, mode):
    check_band_extrapolation_floor(_cfg(enabled=enabled, mode=mode),
                                   _SLICES, _META)

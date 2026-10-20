"""Cheap configuration checks for separate electronic and bosonic grids."""

import dataclasses

import numpy as np
import pytest

from common.meta import Meta
from gw.gw_config import LorraxConfig, parse_qgrid


def _meta(qgrid=None):
    return Meta(
        rank=0, n_proc=1, b_id_0=0, b_id_1=0, b_id_2=2, b_id_3=4,
        b_id_4=8, fft_grid=(8, 8, 8), cell_volume=10.0,
        n_rtot=512, n_rmu=4, npol=1, nfreq=1, nspin=1,
        nspinor=2, nspinor_wfnfile=2, nkx=24, nky=24, nkz=1,
        nk_tot=576, qgrid=qgrid)


@pytest.mark.parametrize("value", [None, "", "  "])
def test_empty_grid_preserves_electronic_domain(value):
    assert parse_qgrid(value) is None
    meta = _meta(parse_qgrid(value))
    assert meta.qgrid == meta.kgrid == (24, 24, 1)
    assert meta.nq_tot == meta.nk_tot == 576


@pytest.mark.parametrize("value", ["6 6 1", "6,6,1", (6, 6, 1), [6, 6, 1]])
def test_bosonic_grid_does_not_relabel_electronic_grid(value):
    meta = _meta(parse_qgrid(value))
    assert meta.qgrid == (6, 6, 1)
    assert meta.nq_tot == 36
    assert meta.kgrid == (24, 24, 1)
    assert meta.nk_tot == 576
    np.testing.assert_array_equal(meta.qgrid_np, [6, 6, 1])
    assert meta.band_edges == (0, 0, 2, 4, 8)


@pytest.mark.parametrize("value", ["6 6", "6 6 1 1", "0 6 1", "-6 6 1",
                                  "6.0 6 1", "6x6x1", "none", (True, 6, 1)])
def test_bad_grid_spelling_refuses(value):
    with pytest.raises(ValueError, match="qgrid"):
        parse_qgrid(value)


@pytest.mark.parametrize("grid", [(5, 6, 1), (48, 6, 1), (6, 6, 2), (6, 0, 1),
                                 (6, 6), (6.0, 6, 1)])
def test_finer_or_incommensurate_grid_refuses(grid):
    with pytest.raises(ValueError, match="qgrid"):
        _meta(grid)


def test_public_key_assembles_and_round_trips(tmp_path):
    deck = tmp_path / "grid.in"
    deck.write_text("[cohsex]\nsys_dim = 2\nqgrid = 6,6,1\n")
    config = LorraxConfig.from_input_file(
        str(deck), resolve_hardware=False, print_fn=lambda *args: None)
    assert config.qgrid == (6, 6, 1)
    assert "qgrid" in config.raw_input_keys
    assert dataclasses.replace(config, qgrid=None).qgrid is None

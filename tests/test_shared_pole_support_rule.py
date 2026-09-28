"""The production line ladder is a fixed lattice (SPCOST2, claim 2906).

Sites sit 4 eta apart from 4 eta to the first lattice point at or above the top
(the requested window plus 2 eV); a growing top only appends sites, and a top
within the SC envelope's outward snap of a lattice point keeps the ladder.
RED TWIN: a count taken as a plain ceil adds a site at a snapped 12 eV top.
"""
import math

import numpy as np

from gw.sigma_box_plan import snap_outward


def test_lattice_sites_and_extension():
    from gw.shared_pole_recipe import line_ladder_ev
    np.testing.assert_array_equal(line_ladder_ev(12.0, 1.0), np.arange(1.0, 13.0))
    np.testing.assert_array_equal(line_ladder_ev(12.3, 1.0), np.arange(1.0, 14.0))
    np.testing.assert_array_equal(line_ladder_ev(12.3, 1.0)[:12], line_ladder_ev(12.0, 1.0))
    assert line_ladder_ev(0.1, 1.0).tolist() == [1.0, 2.0]


def test_snapped_top_keeps_the_ladder():
    from gw.shared_pole_recipe import line_ladder_ev
    snapped = snap_outward(12.0, 1.0, +1)
    assert snapped > 12.0
    assert line_ladder_ev(snapped, 1.0).size == 12
    assert math.ceil(snapped / 1.0) == 13

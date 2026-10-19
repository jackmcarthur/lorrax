"""Shared-pole round carriers: a state panel's or the infinity block's carrier is its recipe
width, or the array's own width when a multiplet closure made it wider, never narrower (CPU)."""
from types import SimpleNamespace

import pytest
import jax.numpy as jnp


def test_recipe_panel_widths_are_known_before_round_one():
    """Panels narrower than their recipe width are padded to it; a wider one keeps its own."""
    from gw.shared_pole_local import recipe_panel_widths
    recipe = {"imaginary_width": 100, "line_direction_cap": 25}
    roles = [{"role": "imaginary:0"}, {"role": "imaginary:0", "conjugate": True},
             {"role": "line:1"}, {"role": "line:1", "conjugate": True},
             {"role": "imaginary:0", "mirror": True}, {"role": "line:1", "mirror": True}]
    panel = lambda w: (0j, SimpleNamespace(shape=(4, 432, w)))
    states = [panel(64), panel(64), panel(20), panel(20), panel(64), panel(112)]
    widths = recipe_panel_widths(roles, states, recipe, column_extent=lambda w: -(-w // 8) * 8, logical_n=432)
    assert widths == [104, 104, 32, 32, 104, 112]
    # No line cap: the logical extent bounds the line panels.
    widths = recipe_panel_widths(roles[2:3], states[2:3], {"imaginary_width": 100}, column_extent=int, logical_n=432)
    assert widths == [432]


def test_infinity_block_wider_than_the_recipe_keeps_its_own_carrier():
    """An M1 selection closed over a multiplet past the recipe width is never cut (Na, a metal)."""
    from gw.shared_pole_local import recipe_infinity_width, pad_states
    recipe = {"infinity_width": 54}
    extent = lambda w: -(-w // 8) * 8
    narrow = (SimpleNamespace(shape=(4, 432, 40)),)
    wide = (SimpleNamespace(shape=(4, 432, 72)),)
    assert recipe_infinity_width(narrow, recipe, column_extent=extent, logical_n=432) == 56
    assert recipe_infinity_width(wide, recipe, column_extent=extent, logical_n=432) == 72
    assert recipe_infinity_width(wide, recipe, column_extent=extent, logical_n=60) == 72
    # A carrier below a panel is refused by name, never a negative pad.
    block = (jnp.ones((1, 4, 72)),)
    with pytest.raises(ValueError, match="GATE shared_pole_carrier"):
        pad_states([], [], block, 56)
    assert pad_states([], [], block, 72)[1][0].shape == (1, 4, 72)

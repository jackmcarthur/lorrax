"""Frozen scissor fit contract, retained pending the Na freeze A/B gate."""
from types import SimpleNamespace

from gw import sc_iteration


def test_scissor_fits_are_frozen_from_map_zero():
    fits = ("active-fit", "tail-fit")
    assert sc_iteration._frozen_scissor_fits(
        SimpleNamespace(iteration=0, frozen_scissor_fits=fits)) == (None, None)
    assert sc_iteration._frozen_scissor_fits(
        SimpleNamespace(iteration=3, frozen_scissor_fits=None)) == (None, None)
    assert sc_iteration._frozen_scissor_fits(
        SimpleNamespace(iteration=3, frozen_scissor_fits=fits)) == fits
    outputs = SimpleNamespace(scissor_fit="a", tail_scissor_fit=None)
    assert sc_iteration._capture_frozen_scissor_fits(outputs) == ("a", None)
    assert sc_iteration._capture_frozen_scissor_fits(None) is None

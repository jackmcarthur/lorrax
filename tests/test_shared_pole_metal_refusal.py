"""A time-reversal-broken metal is refused on the shared-pole route, by the MPA route's owner.

screening.compute_screening_model dispatches to screen_shared_poles before any MPA gate runs, so
GATE mpa_ordered_metal never fired on this route: a metal with global time reversal measured broken
would fit one residue with no odd channel, from pair and stream samples in the transposed
orientation. The refusal is the first statement of the entry, ahead of every other check, and it is
the same owner (gw.mpa.model._require_metal_time_reversal) with the same message.
"""
from types import SimpleNamespace as NS

import pytest


def _call(material_class, trs_allowed):
    from gw.shared_pole_screening import screen_shared_poles

    def unreachable(*_args, **_kwargs):
        raise AssertionError("screening proceeded past the refusal")

    return screen_shared_poles(
        None, None, NS(shared_pole_recipe=None, shared_pole_capacity=None),
        NS(write_poles=False, debug=NS(write_w=False)),
        mesh_xy=None, sym=NS(trs_allowed=trs_allowed), centroid_indices=None, run_dir=None,
        label="oneshot", wfn=None, wfn_fingerprint_binding=None, tensors_filename=None,
        occupation_state=None, print_fn=unreachable, material_class=material_class)


def test_time_reversal_broken_metal_is_refused_on_the_shared_pole_route():
    with pytest.raises(ValueError, match="GATE mpa_ordered_metal"):
        _call("metal", False)


@pytest.mark.parametrize("material_class,trs_allowed", [
    ("metal", True),        # a time-reversal-symmetric metal keeps the route
    ("insulator", False),   # the ordered route is exactly what this series adds
    ("insulator", True),
])
def test_every_other_combination_passes_the_gate(material_class, trs_allowed):
    # Past the gate the stub bundle fails on its own terms, never on GATE mpa_ordered_metal.
    with pytest.raises(Exception) as excinfo:
        _call(material_class, trs_allowed)
    assert "GATE mpa_ordered_metal" not in str(excinfo.value)

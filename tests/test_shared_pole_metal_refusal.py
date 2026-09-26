"""A time-reversal-broken metal keeps the shared-pole route (METAL ruling 2026-09-16, f4b832e9c).

The ordered bank keeps both particle-hole orientations with the odd channel, so
screening.compute_screening_model's shared-pole entry announces a TR-broken metal and proceeds.
GATE mpa_ordered_metal stays on the MPA route (gw.mpa.model._require_metal_time_reversal), which
fits one residue with no odd channel; no combination is refused by it here.
"""
from types import SimpleNamespace as NS

import pytest


def _call(material_class, trs_allowed, print_fn):
    from gw.shared_pole_screening import screen_shared_poles

    return screen_shared_poles(
        None, None, NS(shared_pole_recipe=None, shared_pole_capacity=None),
        NS(write_poles=False, debug=NS(write_w=False), bispinor=False),
        mesh_xy=None, sym=NS(trs_allowed=trs_allowed), centroid_indices=None, run_dir=None,
        label="oneshot", wfn=None, wfn_fingerprint_binding=None, tensors_filename=None,
        occupation_state=None, print_fn=print_fn, material_class=material_class)


@pytest.mark.parametrize("material_class,trs_allowed", [
    ("metal", False),       # the ordered bank carries the odd channel
    ("metal", True),
    ("insulator", False),
    ("insulator", True),
])
def test_every_combination_passes_the_gate(material_class, trs_allowed):
    # Past the gate the stub bundle fails on its own terms, never on GATE mpa_ordered_metal.
    printed = []
    with pytest.raises(Exception) as excinfo:
        _call(material_class, trs_allowed, printed.append)
    assert "GATE mpa_ordered_metal" not in str(excinfo.value)
    announced = any("time-reversal-broken METAL" in str(line) for line in printed)
    assert announced == (material_class == "metal" and not trs_allowed)

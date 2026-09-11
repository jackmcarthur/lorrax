"""Current-map frequency selection; no W construction or SC physics claim."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from gw.gw_config import ComputeMode
from gw.sc_iteration import _sc_head_frequency_plan


@pytest.mark.parametrize('static_head', [False, True])
def test_shared_sc_uses_current_fit_coordinates(monkeypatch, static_head):
    from gw.mpa import model

    def legacy_plan(*args, **kwargs):
        pytest.fail('shared-pole SC must not construct an elementwise MPA plan')

    monkeypatch.setattr(model, 'make_mpa_plan', legacy_plan)
    config = NS(compute_mode=ComputeMode.MPA, sigma=NS(w_model='shared_pole'),
                do_G0=static_head)
    recipe = dict(z_ry=np.array([1j, 2+1j, 1j, 1+1j]),
                  distinct_id=np.array([0, 1, 0, 2]),
                  held=np.array([False, False, False, True]))
    requests, plan, z = _sc_head_frequency_plan(
        config, None, material_class='metal', shared_pole_recipe=recipe)
    assert requests == [] and plan is None
    assert z == [1j, 2+1j] + ([0j] if static_head else [])
    # A new map supplies changed coordinates; no cached/frozen plan survives.
    recipe['z_ry'] = recipe['z_ry'] + .03
    _, _, changed = _sc_head_frequency_plan(
        config, None, material_class='metal', shared_pole_recipe=recipe)
    assert changed == [.03+1j, 2.03+1j] + ([0j] if static_head else [])


def test_shared_sc_missing_recipe_refuses():
    config = NS(compute_mode=ComputeMode.MPA, sigma=NS(w_model='shared_pole'))
    with pytest.raises(ValueError, match='current-map recipe is missing'):
        _sc_head_frequency_plan(config, None, material_class='metal')


def test_incumbent_sc_plan_still_uses_owner(monkeypatch):
    from gw.mpa import model, sample_plan

    sentinel = object()
    monkeypatch.setattr(model, 'make_mpa_plan', lambda *a, **k: sentinel)
    monkeypatch.setattr(sample_plan, 'plan_z', lambda plan: [1j] if plan is sentinel else [])
    config = NS(compute_mode=ComputeMode.MPA, sigma=NS(w_model='mpa'), do_G0=False)
    requests, plan, z = _sc_head_frequency_plan(config, object(), material_class='metal')
    assert requests == [] and plan is sentinel and z == [1j]


def test_shared_full_head_uses_mpa_owner_at_current_energy_span(monkeypatch):
    from gw.gw_config import HeadCorrection
    from gw.mpa import model, sample_plan
    seen = []
    def owner(config, quad, **kw):
        seen.append(quad.x_max)
        return quad.x_max
    monkeypatch.setattr(model, 'make_mpa_plan', owner)
    monkeypatch.setattr(sample_plan, 'plan_z', lambda plan: [plan+1j])
    config = NS(compute_mode=ComputeMode.MPA, sigma=NS(w_model='shared_pole'),
                head=NS(correction=HeadCorrection.FULL), do_G0=True)
    for span in (3., 3.2):
        _, plan, z = _sc_head_frequency_plan(config, None, material_class='insulator',
            shared_pole_recipe={'census': {'energy_span_ry': span}})
        assert plan == span and z == [span+1j, 0j]
    assert seen == [3., 3.2]

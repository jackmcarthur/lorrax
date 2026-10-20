"""Deck-to-native numerical partition wiring; no microscopic run or GPU."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from tdagw.td_config import TD_DEFAULTS, inactive_keys


@pytest.mark.parametrize('mode', ['joint', 'joint_frozen'])
def test_native_stage_forwards_response_partition_without_changing_the_window(monkeypatch, mode):
    from tdagw import stages, snapshot_runtime
    values = dict(TD_DEFAULTS)
    values.update({'lorrax.collision': 'raw', 'lorrax.collision_screening': mode,
                   'lorrax.collision_spm_chi0_active': [88, 144]})
    deck = NS(values=values, trials=(None,),
              given={'lorrax.collision_spm_chi0_active': [88, 144]})
    assert 'lorrax.collision_spm_chi0_active' not in inactive_keys(deck)
    source = {'config': NS(sigma=NS(regularization_ev=.1))}
    system = NS(ops=object(), hdr={'kpts': np.zeros((1, 3))}, bands=(129, 132))
    captured = {}

    class Captured(Exception):
        pass

    def stop_after_native_options(source_arg, ops, **kwargs):
        captured.update(kwargs)
        raise Captured

    monkeypatch.setattr(snapshot_runtime, 'snapshot_collision', stop_after_native_options)
    with pytest.raises(Captured):
        stages._attach_collision(deck, system, object(), NS(method='cfm4'), source,
                                 None, say=lambda _: None, log=None)
    assert captured['b_lo'] == 128
    assert captured['options']['spm_chi0_active'] == [88, 144]
    assert captured['options']['screening'] == mode


def test_partition_is_still_inactive_without_shared_pole_response():
    values = dict(TD_DEFAULTS)
    values.update({'lorrax.collision': 'raw', 'lorrax.collision_screening': 'frozen'})
    deck = NS(values=values, given={'lorrax.collision_spm_chi0_active': [88, 144]})
    assert 'lorrax.collision_spm_chi0_active' in inactive_keys(deck)

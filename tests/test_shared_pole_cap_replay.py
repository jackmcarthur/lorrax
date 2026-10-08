"""Host metadata cap replay: new model identity, immutable sampled objective.

These tests do not construct or sample a response/model. Each negative
control corrupts an input field a physical-bank replay must authenticate.
"""
import copy
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from gw.shared_pole_recipe import (RECIPE_HASH, RECIPE_VERSION, GATE_HASH,
                                  GATE_VERSION, authenticate_cap_only_replay)


def resolved(cap):
    # Independently specified public resolver contract; no copied reducer.
    digest = hashlib.sha256(
        f'{RECIPE_HASH}|fixed-body-pole-budget-v1|{cap}'.encode()).hexdigest()
    recipe = dict(recipe_version=RECIPE_VERSION+'+pole_budget', recipe_hash=digest,
        gate_version=GATE_VERSION, gate_hash=GATE_HASH, accuracy='production',
        support_sites_override='', pole_budget=cap, pole_budget_override=cap,
        pole_budget_policy='fixed-body-pole-budget-v1', automatic_pole_budget=9720,
        n=5400, eta_ev=.25, height_ev=2.6, direction_cutoff=.001,
        infinity_width=675, imaginary_width=1350, line_direction_cap=338,
        z_ry=np.array([1+.2j, 2+.2j, 1.5+.2j], np.complex128),
        role=np.array([0, 0, 3], np.int8), distinct_id=np.array([0, 1, 2], np.int64),
        held=np.array([False, False, True]), fit_ids=[0, 1], held_ids=[2],
        support_pair=[[0, 1], [0, 1], [0, 1]], role_codes={'line':0,'held_line':3},
        census=dict(trs_allowed=True, valid_kn_sha256='a'*64, energy_sha256='b'*64))
    identity=dict(iteration_id='oneshot_native_fixture', hamiltonian='h', energies='e',
        occupations='f', wavefunctions='psi', centroids='mu', recipe_hash=digest,
        gate_hash=GATE_HASH)
    return recipe, identity


@pytest.mark.parametrize('cap', [5500, 8000])
def test_cap_replay_identity_split_and_no_mutation(cap):
    old, bank = resolved(5500)
    new, model = resolved(cap)
    from file_io.shared_pole_store import _json
    before = _json((old, bank, new, model))
    result = authenticate_cap_only_replay(old, new, bank, model)
    assert result['original_pole_budget'] == 5500
    assert result['model_pole_budget'] == cap
    assert result['bank_identity'] == bank and result['model_identity'] == model
    assert _json((old, bank, new, model)) == before
    assert (result['original_recipe_hash'] != result['model_recipe_hash']) == (cap != 5500)


@pytest.mark.parametrize('field,value', [
    ('eta_ev', .3), ('height_ev', 1.), ('direction_cutoff', .01),
    ('infinity_width', 674), ('held', [False, True, True]),
    ('fit_ids', [0]), ('z_ry', np.array([1+.2j, 2.001+.2j, 1.5+.2j])),
    ('census', dict(trs_allowed=True, valid_kn_sha256='c'*64, energy_sha256='b'*64)),
    ('gate_hash', 'f'*64), ('recipe_hash', 'd'*64), ('pole_budget_override', 7999)])
def test_physical_recipe_site_mask_and_hash_mutations_refuse(field, value):
    old, bank = resolved(5500)
    new, model = resolved(8000)
    new[field] = value
    with pytest.raises(ValueError, match='shared_pole_cap_replay'):
        authenticate_cap_only_replay(old, new, bank, model)


@pytest.mark.parametrize('field', ['energies', 'occupations', 'wavefunctions', 'centroids', 'iteration_id'])
def test_physical_model_identity_mutation_refuses(field):
    old, bank = resolved(5500)
    new, model = resolved(8000)
    model[field] += '_changed'
    with pytest.raises(ValueError, match='shared_pole_cap_replay'):
        authenticate_cap_only_replay(old, new, bank, model)


def test_changed_canonical_geometry_refuses(monkeypatch):
    import file_io.shared_pole_store as store
    geometry=dict(representation='scalar-trs-even-s', parent_convention='raw-parent',
        n_q_irr=8, n_q_full=64, n_mu_logical=5400, nspinor=1, centroid_digest='mu',
        grid=[4,4,4], fft_grid=[30,30,30], q_order='canonical-full-flat',
        q_shift=[0.,0.,0.], q_irr_full_idx=list(range(8)),
        qirr={'irr_idx_q':list(range(64))}, operations={'rows':[0]})
    monkeypatch.setattr(store, '_metadata', lambda *a, **k: geometry)
    header=dict(geometry, recipe={}, identity={})
    store.authenticate_bank_geometry(header, meta=None, tables=None)
    for key, value in [('grid',[8,8,8]), ('q_irr_full_idx',[7,1,2,3,4,5,6,0]),
                       ('centroid_digest','other'), ('qirr',{'irr_idx_q':[0]*64})]:
        mutated=copy.deepcopy(header); mutated[key]=value
        with pytest.raises(ValueError, match='geometry differs'):
            store.authenticate_bank_geometry(mutated, meta=None, tables=None)


@pytest.mark.parametrize('field,value', [('iteration_id','sc_001'),
    ('wavefunctions','qp_rotation_unreceipted'), ('authentication','NON-AUTHENTICATING'),
    ('hamiltonian','sc_map_001')])
def test_equal_unauthenticated_markers_in_both_states_refuse(field, value):
    old, bank = resolved(5500)
    new, model = resolved(8000)
    bank[field] = model[field] = value
    with pytest.raises(ValueError, match='shared_pole_sc_restart'):
        authenticate_cap_only_replay(old, new, bank, model)


def test_default_constructor_still_refuses_changed_cap_hash(monkeypatch):
    import file_io.shared_pole_store as store
    from gw.shared_pole_constructor import construct_shared_poles
    old, bank_identity = resolved(5500)
    current, _ = resolved(8000)
    header={'recipe': old}
    monkeypatch.setattr(store, 'validate_shared_pole_bank', lambda *a, **k: header)
    meta=SimpleNamespace(nspinor=1, n_rmu_padded=5424, shared_pole_recipe=current,
                         shared_pole_capacity=SimpleNamespace(live_stages=()))
    config=SimpleNamespace(backend=SimpleNamespace(linalg='distributed'))
    bank=dict(path='unused', identity=bank_identity,
              tables={'sym':SimpleNamespace(trs_allowed=True)})
    with pytest.raises(ValueError, match='stale recipe_hash'):
        construct_shared_poles(bank, bank, meta, config, mesh_xy=None, output='unused')

"""Physical-current lifecycle accounting and immutable operator handoff."""
from types import SimpleNamespace
import hashlib
import json
import numpy as np


def test_current_contraction_prices_only_existing_final_banks():
    from gw.isdf_augmentation import _current_contract_workspace_bytes
    geometry=dict(qpad=16,mu=4272,atoms=8,harmonics=64,radial_points=256,g_tile=4096,shards=16,
                  solved_half_banks=True)
    faithful=_current_contract_workspace_bytes(**geometry)
    physical=_current_contract_workspace_bytes(**geometry,physical_three_banks=True)
    # A complete three-current radial bank follows the actual owner shape;
    # all factor/matrix/Fourier scratch must remain in the physical route.
    bank=16*(16/16)*4272*8*64*256
    assert faithful['total']-physical['total']==3*bank
    assert faithful['solve_scratch']==physical['solve_scratch']==0
    for key in physical:
        if key not in ('paired_operands','total'):
            assert physical[key]==faithful[key]
    assert physical['total']<66.3630109186e9
    oversized=_current_contract_workspace_bytes(**dict(geometry,g_tile=40448),physical_three_banks=True)
    assert oversized['total']>66.3630109186e9


def test_actual_operator_contract_is_canonically_digested_in_generation_policy(monkeypatch):
    from gw import gw_init
    parent=dict(head='point',generation_policy='same')
    monkeypatch.setattr(gw_init,'_bispinor_v_policy',lambda cfg,meta:dict(parent))
    contract=dict(schema='lorrax.current.physical_Slab_mixed_operator.v1',
                  endpoint_source_binding=dict(full150_frame_sha256='actual-frame',physical_bands=200),
                  current_geometry_binding=dict(logical_mu=4262,centroid_hash='closed-chart'))
    policy=gw_init._bispinor_policy_with_current_contract(None,None,
        SimpleNamespace(physical_current_operator_contract=contract))
    raw=json.dumps(contract,sort_keys=True,separators=(',',':'),allow_nan=False)
    assert policy['physical_current_operator_identity']==hashlib.sha256(raw.encode()).hexdigest()
    assert policy['physical_current_operator_contract']==contract
    assert gw_init._bispinor_policy_with_current_contract(None,None,None)==parent
    changed=dict(contract,endpoint_source_binding=dict(full150_frame_sha256='changed-frame',physical_bands=200))
    assert gw_init._bispinor_policy_with_current_contract(None,None,
        SimpleNamespace(physical_current_operator_contract=changed))['physical_current_operator_identity']!=policy['physical_current_operator_identity']

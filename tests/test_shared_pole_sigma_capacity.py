"""Sigma panel admission against the canonical map ledger (scalar metadata)."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
from gw.shared_pole_recipe import CapacityLedger
from gw.mpa.sigma import _shared_pole_memory_schedule


def fixture(caller_fraction=0.0):
    mesh = NS(shape={'x': 2, 'y': 2})
    from common.grouped_layout import identity_square_grouped_shard_layout
    layout = identity_square_grouped_shard_layout(16,16,(2,2))
    meta = NS(nk_tot=64,nspinor=1,n_rmu=16,
              mu_basis=NS(n_packed=16,layout=layout,active_mask=layout.axis.active_mask))
    meta.shared_pole_capacity = CapacityLedger(meta, mesh_xy=mesh)
    ledger = meta.shared_pole_capacity
    ledger.reserve('sigma.inputs', resident_bytes_per_rank=int(caller_fraction*ledger.U_bytes_per_rank),
                   workspace_bytes_per_rank=0)
    ledger.reserve('sigma.spatial', resident_bytes_per_rank=0, workspace_bytes_per_rank=0)
    ledger.live_stages = ('sigma.inputs', 'sigma.spatial')
    header = dict(n_q_full=64, n_q_irr=4, n_mu_logical=16, nspinor=1, Kmax=50,
                  grid=(4,4,4),representation='planted-capacity-only',
                  q_irr_full_idx=[0,16,32,48],operations=dict(authorized_rows=[0]),
                  qirr=dict(irr_idx_q=np.repeat(np.arange(4),16), sym_perm=np.arange(16)[None,:],
                            L_table=np.zeros((1,16,3),np.int32), sym_idx_q=np.zeros(64,np.int32),
                            q_irr_frac=np.zeros((4,3)), n_sym_spatial=1))
    return meta, header, mesh


def test_actual_panels_and_concurrency():
    loose, h, mesh = fixture()
    tight, _, _ = fixture(1.0)
    a = _shared_pole_memory_schedule(loose, h, mesh_xy=mesh)
    b = _shared_pole_memory_schedule(tight, h, mesh_xy=mesh)
    assert a['column_capacity'] > 0 and b['column_capacity'] > 0
    assert b['parent_capacity']*b['column_capacity'] < a['parent_capacity']*a['column_capacity']
    assert b['peak_in_U'] <= 3
    assert b['capacity_receipt']['concurrent_with'] == ['sigma.inputs', 'sigma.spatial']
    assert b['compiled_peak_status'] == 'NOT_MEASURED'


def test_minimum_refusal_is_ledger_row():
    meta, h, mesh = fixture(2.9)
    with pytest.raises(MemoryError, match='sigma.synthesis'):
        _shared_pole_memory_schedule(meta, h, mesh_xy=mesh)
    assert meta.shared_pole_capacity.entries[-1]['status'] == 'FAIL'
    assert meta.shared_pole_capacity.entries[-1]['stage'] == 'sigma.synthesis'


def test_missing_caller_and_geometry_refuse():
    meta, h, mesh = fixture()
    del meta.shared_pole_capacity
    with pytest.raises(ValueError, match='CapacityLedger'):
        _shared_pole_memory_schedule(meta, h, mesh_xy=mesh)
    meta.shared_pole_capacity = CapacityLedger(meta, mesh_xy=mesh)
    with pytest.raises(ValueError, match='unbound caller lifetimes'):
        _shared_pole_memory_schedule(meta, h, mesh_xy=mesh)
    meta, h, mesh = fixture()
    h['n_mu_logical'] += 1
    with pytest.raises(ValueError, match='geometry mismatch'):
        _shared_pole_memory_schedule(meta, h, mesh_xy=mesh)

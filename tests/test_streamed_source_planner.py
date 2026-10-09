"""Source batches may grow independently of the resident plane row stage."""
from types import SimpleNamespace


def _case(monkeypatch,*,budget=33.966):
    from gw import gflat_memory_model as model
    monkeypatch.setattr(model,'_host_bytes_per_rank',lambda:64e9)
    meta=SimpleNamespace(nk_tot=216,nspinor=4,n_rmu=2304,n_rmu_padded=2304,fft_grid=(50,50,50))
    mesh=SimpleNamespace(shape={'x':4,'y':4})
    return model,dict(meta=meta,mesh_xy=mesh,n_q_selected=16,ngkmax=5642,psi_ngkmax=5648,
        fit_nb=120,n_col=464,n_s=24,budget_gb=budget,n_parent=16,orbit_width=24)


def test_source_growth_keeps_the_canonical_stream_and_budget(monkeypatch):
    model,kwargs=_case(monkeypatch)
    plan=model.plan_zeta_route_g(**kwargs)
    assert plan.b>16*24
    assert plan.c_out<plan.b//16
    assert plan.working_set(plan.b,plan.r_sub,plan.b//16,1)>plan.target_bytes
    assert plan.working_set(plan.b,plan.r_sub,plan.c_out,plan.n_blk)<=plan.target_bytes
    assert plan.hwm_bytes<=plan.target_bytes
    assert plan.n_batch<6


def test_actual_orbit_batches_govern_the_receipt(monkeypatch):
    model,kwargs=_case(monkeypatch)
    from isdf import zeta_mubatch
    calls=[]
    def packed(plan,mu,ranks,*,c_max,build_tables):
        assert build_tables is False
        calls.append(c_max)
        return SimpleNamespace(b=ranks*c_max,n_batch=(mu+ranks*c_max-1)//(ranks*c_max)+1)
    monkeypatch.setattr(zeta_mubatch,'best_owner_orbit_batches',packed)
    plan=model.plan_zeta_route_g(**kwargs,k_unfold_plan=object())
    assert calls==[24,48,72,96,120,144]
    assert plan.n_batch==(2304+plan.b-1)//plan.b+1
    assert plan.hwm_bytes<=plan.target_bytes


def test_too_small_budget_preserves_the_minimum_refusal_geometry(monkeypatch):
    model,kwargs=_case(monkeypatch,budget=.0001)
    plan=model.plan_zeta_route_g(**kwargs)
    assert plan.b==16*24
    assert plan.hwm_bytes>plan.target_bytes
    assert plan.min_c==24


def test_preview_uses_identical_packing_and_cannot_unfold(monkeypatch):
    import numpy as np
    import pytest
    from gw import centroid_k_unfold as owner
    from isdf.zeta_mubatch import owner_orbit_batches,best_owner_orbit_batches
    perm=np.stack((np.arange(12),np.r_[np.arange(8)[::-1],np.arange(8,12)]))
    plan=SimpleNamespace(n_centroid_packed=12,sym_idx=np.array([0,1]),sym_perm=perm,
        L_table=np.arange(2*12*3).reshape(2,12,3),layout=SimpleNamespace(
            axis=SimpleNamespace(active_mask=np.arange(12)<8)))
    original=owner.mu_batch_tables
    calls=[]
    def materialize(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(owner,'mu_batch_tables',materialize)
    preview=owner.orbit_mu_batches(plan,12,1,b_target=3,build_tables=False)
    assert calls==[] and preview.left_perm is None
    full=owner.orbit_mu_batches(plan,12,1,b_target=3)
    np.testing.assert_array_equal(preview.mu,full.mu)
    calls.clear()
    unfolding=[]
    original_groups=owner.unfold_orbits
    def groups(p):
        unfolding.append(1)
        return original_groups(p)
    monkeypatch.setattr(owner,'unfold_orbits',groups)
    best=best_owner_orbit_batches(plan,12,4,c_max=8)
    assert len(calls)==1 and len(unfolding)==1
    calls.clear()
    cheap=best_owner_orbit_batches(plan,12,4,c_max=8,build_tables=False)
    assert calls==[]
    np.testing.assert_array_equal(cheap.mu,best.mu)
    np.testing.assert_array_equal(cheap.slot_of_packed,best.slot_of_packed)
    with pytest.raises(ValueError,match='transport preview'):
        cheap.transport(0)
    expected=owner_orbit_batches(plan,12,4,c_target=best.c)
    for a,b in zip(best.transport(0),expected.transport(0)):
        np.testing.assert_array_equal(a,b)

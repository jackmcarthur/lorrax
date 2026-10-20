"""Pure tiny-host canonical parent scheduling; no material matrices or GPU."""
import copy
import numpy as np
import pytest

from gw import shared_pole_constructor as owner
from tests.test_shared_pole_general_cpu import _case


def test_metadata_batch_bound_and_single_parent_fallback_without_pole_cut():
    shapes=[(100,100)]*8
    broad=owner.general_pencil_batch_plan(shapes,max_order=90,processes=4,max_workspace_bytes=10**9)
    assert broad['batch_size']==4 and not broad['full_microscopic_host_payload']
    single=owner.general_pencil_batch_plan(shapes,max_order=90,processes=4,
        max_workspace_bytes=owner.general_pencil_batch_plan(shapes[:1],max_order=90,processes=4)['host_workspace_bytes'])
    assert single['batch_size']==1
    with pytest.raises(ValueError,match='shared_general_capacity'):
        owner.general_pencil_batch_plan(shapes,max_order=90,processes=4,max_workspace_bytes=1)


def test_fixed_space_cannot_silently_exceed_priced_order():
    _,_,e,a,_,_=_case()
    initial=owner.reduce_general_pencil(e,a,max_order=4)
    with pytest.raises(ValueError,match='shared_general_basis'):
        owner.reduce_general_pencil(e,a,max_order=1,fixed_spaces=initial['spaces'])


def test_partitioned_models_are_canonical_and_match_root_serial_transfer(monkeypatch):
    import common.collectives as cc
    import jax
    jax.config.update('jax_enable_x64',True)
    world=3;state=dict(rank=0,calls=[],shares=[])
    monkeypatch.setattr(cc,'process_count',lambda:world)
    monkeypatch.setattr(cc,'process_rank',lambda:state['rank'])
    systems=[_case(seed=9+i) for i in range(world)]
    pure=owner.reduce_general_pencil
    old=[pure(s[2],s[3],max_order=4) for s in systems]
    shifts=[.001*(q+1) for q in range(world)]
    pencils=[(s[2],s[3]+shift*s[2]) for s,shift in zip(systems,shifts)]
    canonical=[pure(e,a,max_order=4,fixed_spaces=old[q]['spaces']) for q,(e,a) in enumerate(pencils)]
    serial_models=copy.deepcopy(canonical)
    # Independent owners may choose DIFFERENT modal phase gauges. The
    # canonical result of each parent must still be identical on all ranks.
    for i,model in enumerate(canonical):
        phase=np.exp(1j*(i+.37)*np.arange(1,len(model['poles_ry'])+1))
        model['output_map']*=phase;model['input_map']*=phase
    def reduce(e,a,**kw):
        assigned=state['rank'];state['calls'].append(assigned)
        assert e is pencils[assigned][0] and a is pencils[assigned][1]
        assert kw['fixed_spaces'] is old[assigned]['spaces']
        return copy.deepcopy(canonical[assigned])
    def share(model,error,*,owner_rank,price,shape):
        assert state['calls'][-1]==state['rank'] # all assigned work BEFORE firsthandoff
        assert (model is not None)==(state['rank']==owner_rank)
        assert error is None
        state['shares'].append((state['rank'],owner_rank))
        return copy.deepcopy(canonical[owner_rank])
    monkeypatch.setattr(owner,'reduce_general_pencil',reduce)
    monkeypatch.setattr(owner,'_share_general_pencil',share)
    all_results=[]
    for rank in range(world):
        state['rank']=rank
        fixed=[None]*world;fixed[rank]=old[rank]['spaces']
        all_results.append(owner.collective_general_pencil_batch(pencils,max_order=4,fixed_spaces=fixed))
    assert state['calls']==[0,1,2]
    assert state['shares']==[(rank,i) for rank in range(world) for i in range(world)]
    for q,system in enumerate(systems):
        poles,exact,e,a,b,c=system
        from tests._full_microscopic_oracle import retarded_reference_pairs
        rng=np.random.default_rng(729+q)
        vertices=.1*(rng.normal(size=(3,2,2,5))+1j*rng.normal(size=(3,2,2,5)))
        electronic=np.array([[-.2,.3],[-.1,.4],[-.3,.2]])
        rho=np.broadcast_to(np.array([[.6,.13+.07j],[.13-.07j,.3]]),(3,2,2))
        def sigma(model):
            left=c@model['output_map'];right=b.conj().T@model['input_map']
            residues=np.stack([np.outer(left[:,i],right[:,i].conjugate()) for i in range(left.shape[1])])
            return retarded_reference_pairs(vertices,electronic,rho,electronic,
                model['poles_ry'],residues,.3)
        sigma_serial=sigma(serial_models[q])
        for rank in range(world):
            model=all_results[rank][q]
            for key in ('poles_ry','output_map','input_map'):
                np.testing.assert_array_equal(np.ascontiguousarray(model[key]).view(np.uint64),
                    np.ascontiguousarray(canonical[q][key]).view(np.uint64))
            for z in [-.3+.2j,.35+.11j]:
                got=((c@model['output_map'])/(z-model['poles_ry']))@(b.conj().T@model['input_map']).conj().T
                np.testing.assert_allclose(got,exact(z-shifts[q]),rtol=4e-12,atol=4e-12)
            for actual,expected in zip(sigma(model),sigma_serial):
                np.testing.assert_allclose(actual,expected,rtol=4e-12,atol=4e-12)


def test_nonroot_owner_raw_bits_and_error_protocol(monkeypatch):
    import common.collectives as cc
    from jax.experimental import multihost_utils as mh
    state=dict(rank=1,g=0,b=0,records=[],bits=[])
    monkeypatch.setattr(cc,'process_rank',lambda:state['rank'])
    def gather(a):
        i=state['g'];state['g']+=1
        if state['rank']==1:state['records'].append(np.asarray(a).copy())
        root=state['records'][i]
        return np.stack((np.zeros_like(root),root))
    def broadcast(a,*,is_source):
        assert is_source==(state['rank']==1) and a.dtype==np.uint64
        i=state['b'];state['b']+=1
        if is_source:state['bits'].append(a.copy())
        return state['bits'][i].copy()
    monkeypatch.setattr(cc,'all_gather_processes',gather)
    monkeypatch.setattr(mh,'broadcast_one_to_all',broadcast)
    _,_,e,a,_,_=_case();model=owner.reduce_general_pencil(e,a,max_order=4)
    plan=owner.general_pencil_batch_plan([e.shape]*2,max_order=4,processes=2)
    root=owner._share_general_pencil(model,None,owner_rank=1,price=plan,shape=e.shape)
    state.update(rank=0,g=0,b=0)
    worker=owner._share_general_pencil(None,None,owner_rank=1,price=plan,shape=e.shape)
    assert root['receipt']==worker['receipt']
    for key in ('poles_ry','output_map','input_map'):
        np.testing.assert_array_equal(root[key].view(np.uint64),worker[key].view(np.uint64))
    state.update(rank=1,g=0,b=0,records=[],bits=[])
    with pytest.raises(ValueError,match='owner_refusal'):
        owner._share_general_pencil(None,ValueError('owner_refusal'),owner_rank=1,price=plan,shape=e.shape)
    state.update(rank=0,g=0,b=0)
    with pytest.raises(ValueError,match='owner_refusal'):
        owner._share_general_pencil(None,None,owner_rank=1,price=plan,shape=e.shape)
    assert not state['bits']

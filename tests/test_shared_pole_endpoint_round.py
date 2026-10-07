"""Independent finite-endpoint laws and production/selected-round contracts.

CPU P1 mathematical fixtures; MPI4 callers can run the same tiny physical
algebra tests. Large native state/cache/constructor pricing is a separate leg.
"""
from dataclasses import replace
from types import SimpleNamespace as NS

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.collectives import gather_to_host
from gw.response_bank import (physical_charge_response_algebra,
    physical_charge_coulomb_roots, PhysicalChargeRoundProvider)
from gw.shared_pole_capacity import ConstructorCapacity, shared_pole_byte_terms
from gw.shared_pole_recipe import (CapacityLedger, table_hash,
                                   shared_real_pole_gates_v1_r3b as GATES)
from gw.shared_pole_round import (PoleEndpointGeometry, SharedPoleRoundPlan,
                                  construct_shared_pole_round)


@pytest.fixture
def mesh():
    if jax.process_count() == 4:
        return Mesh(np.asarray(jax.devices()).reshape(2, 2), ('x','y'))
    assert len(jax.devices()) == 1, 'Run algebra fixture on P1 or true MPI4'
    return Mesh(np.asarray(jax.devices()).reshape(1,1), ('x','y'))


def face(value, mesh):
    a=np.asarray(value, np.complex128)
    return jax.device_put(a, NamedSharding(mesh,P(*((None,)*(a.ndim-2)),'x','y')))


def host(a):
    assert a.nbytes <= 16384
    return np.asarray(gather_to_host(a))


def noncommuting_inputs():
    h=np.diag([1.3,.7,0.,0.]).astype(complex)
    a=np.zeros((4,4),complex);b=a.copy()
    a[:2,:2]=[[.12,.02+.025j],[.02-.025j,.07]]
    b[:2,:2]=[[.03,-.006+.002j],[-.006-.002j,.09]]
    return h,a,b


@pytest.mark.parametrize('nk', [1,512])
def test_physical_dyson_and_slope_have_no_extra_k_or_volume_factor(mesh,nk):
    h,a,b=noncommuting_inputs()
    chi=(.3-.2j)*a+(.1+.4j)*b;dchi=(-.07+.11j)*a+(.08-.02j)*b
    dyson,slope,_,receipt=physical_charge_response_algebra(mesh_xy=mesh,
        packed_endpoints=4,physical_k_count=nk,linalg='local')
    hv,cv,dv=map(lambda x:face(x[None],mesh),(h,chi,dchi))
    wc,ds=dyson.pair('face')(dyson.place(hv),cv,dv)
    v=h@h;w=h@np.linalg.solve(np.eye(4)-h@chi@h,h)
    np.testing.assert_allclose(host(wc)[0],w-v,rtol=3e-13,atol=4e-16)
    np.testing.assert_allclose(host(ds)[0],w@dchi@w,rtol=3e-13,atol=4e-16)
    np.testing.assert_allclose(host(slope(hv,wc,dv)),host(ds),rtol=3e-13,atol=4e-16)
    assert receipt['prefactor']==1 and receipt['prefactor_q_count']==nk
    assert np.linalg.norm(w@dchi@w-w.conj().T@dchi@w)>1e-5


@pytest.mark.parametrize('ordered', [False,True])
def test_exact_noncommuting_screened_moments_match_series_recurrence(mesh,ordered):
    h,a,b=noncommuting_inputs();o0=.15j*(a-b);o1=.17*(a+2*b)
    _,_,moments,_=physical_charge_response_algebra(mesh_xy=mesh,
        packed_endpoints=4,physical_k_count=512,linalg='local',ordered=ordered)
    args=(h,a,b,o0,o1) if ordered else (h,a,b)
    got=moments(*(face(x[None],mesh) for x in args))
    # Independent noncommutative W=V+V chi W series, not the implementation's
    # whitened X polynomial. Coefficients include exact odd projection terms.
    v=h@h; coefficients=(o0,a,o1,b) if ordered else (np.zeros_like(a),a,np.zeros_like(a),b)
    c=[]
    for k,x in enumerate(coefficients):
        value=v@x@v
        for i in range(k): value+=v@coefficients[i]@c[k-i-1]
        c.append(value)
    expected=tuple(x/2 for x in c) if ordered else (c[1]/2,c[3]/2)
    for x,y in zip(got,expected): np.testing.assert_allclose(host(x)[0],y,rtol=3e-13,atol=4e-16)
    if ordered: assert np.linalg.norm(expected[0])>0 and np.linalg.norm(expected[2])>0


def test_supported_null_and_interleaved_virtual_coulomb_modes(mesh):
    v=np.diag([2.,0.,3.,0.]).astype(complex);valid=np.array([True,True,True,False])
    root,inv,ranks=physical_charge_coulomb_roots(face(v[None],mesh),mesh_xy=mesh,
        logical_endpoints=3,endpoint_valid=valid,linalg='local')
    np.testing.assert_allclose(host(root)[0],np.diag(np.sqrt([2.,0.,3.,0.])),atol=3e-15)
    np.testing.assert_allclose(host(inv)[0],np.diag([1/np.sqrt(2),0.,1/np.sqrt(3),0.]),atol=3e-15)
    np.testing.assert_array_equal(host(ranks),[2])
    assert not host(inv)[0,1].any() and not host(inv)[0,3].any()


@pytest.mark.parametrize('bad', ['negative','ghost','nonfinite','mask','dtype'])
def test_physical_coulomb_refuses_bad_inputs_without_repairs(mesh,bad):
    v=np.diag([2.,0.,3.,0.]).astype(complex);valid=np.array([True,True,True,False])
    if bad=='negative':v[0,0]=-1.
    elif bad=='ghost':v[3,0]=1e-15
    elif bad=='nonfinite':v[0,0]=np.nan
    elif bad=='mask':valid=np.ones(4,bool)
    value=face(v[None],mesh)
    if bad=='dtype':value=value.real
    with pytest.raises(ValueError):
        physical_charge_coulomb_roots(value,mesh_xy=mesh,logical_endpoints=3,
                                      endpoint_valid=valid,linalg='local')


@pytest.mark.parametrize('change', [dict(logical_size=5),dict(carrier_size=0),
    dict(physical_k_count=1),dict(k_grid=(8,8,True)),dict(nspinor=3),dict(kind='centroid-shaped-G')])
def test_endpoint_identity_does_not_relabel_mesh_or_coordinates(change):
    args=dict(kind='gamma-real-charge',logical_size=3,carrier_size=4,nspinor=1,
              k_grid=(8,8,8),physical_k_count=512);args.update(change)
    with pytest.raises(ValueError):PoleEndpointGeometry(**args)


@pytest.mark.parametrize('phase', ['selection','reduction','model'])
def test_capacity_default_and_equal_explicit_endpoints_are_literal_identical(mesh,phase):
    from gw.gw_config import linalg_resolution
    resolution=linalg_resolution({'linalg':'local'});meta=NS(n_rmu_padded=8)
    kwargs=dict(mesh_xy=mesh,resolution=resolution,pencil_side=16,parent_batch=1,
                sample_batch=2,phase=phase)
    assert shared_pole_byte_terms(meta,**kwargs)==shared_pole_byte_terms(None,packed_endpoints=8,**kwargs)
    got=shared_pole_byte_terms(None,packed_endpoints=4,**kwargs)
    assert got!=shared_pole_byte_terms(meta,**kwargs)
    ledger=NS(live_stages=())
    a=ConstructorCapacity(meta,resolution,mesh_xy=mesh,ledger=ledger,upstream=())
    b=ConstructorCapacity(None,resolution,mesh_xy=mesh,ledger=ledger,upstream=(),packed_endpoints=8)
    assert a.resident_quote(16,phase=phase,sample_batch=2)==b.resident_quote(16,phase=phase,sample_batch=2)


@pytest.mark.parametrize('count', [True,4.,0,-4])
def test_capacity_explicit_count_refuses_bad_types(mesh,count):
    from gw.gw_config import linalg_resolution
    with pytest.raises(ValueError):
        shared_pole_byte_terms(None,mesh_xy=mesh,resolution=linalg_resolution({'linalg':'local'}),
            pencil_side=8,parent_batch=1,sample_batch=1,packed_endpoints=count)


def toy(mesh):
    from gw.gw_config import linalg_resolution
    from gw.shared_pole_directions import port_extent
    endpoint=PoleEndpointGeometry('gamma-real-charge',3,4,1,(8,8,8),512)
    identity=dict(endpoint_kind=endpoint.kind,logical_endpoints=3,packed_endpoints=4,
        physical_k_count=512,k_grid=[8,8,8],source_parent_ids=[0],
        normalization='physical spin/(Nk*Omega)',moment_convention='M_k=C_(k+1)/2',
        basis_sha256='ab'*32,native_state_sha256='bc'*32,recipe_hash='cd'*32,coulomb_sha256='de'*32)
    z=np.array([.7j,.4+.3j,.9j,.8+.2j]);lam=np.array([.9,2.4])
    b=np.array([[.2,.06],[.03,.22],[0.,0.],[0.,0.]])
    def value(z):return (b/(z*z-lam))@b.T
    def ds(z):return (-b/(z*z-lam)**2)@b.T
    m={'M1':(b@b.T)/2,'M3':((b*lam)@b.T)/2}
    def samples(ids,*,sample_ids):
        return {key:face(np.repeat(np.stack([fn(z[j]) for j in sample_ids])[None],len(ids),axis=0),mesh)
                for key,fn in [('Wc',value),('dWc_ds',ds)]}
    def moments(ids,*,fields):return {key:face(np.repeat(m[key][None],len(ids),axis=0),mesh) for key in fields}
    def coulomb(span):return face(np.diag([1/np.sqrt(2),1/np.sqrt(3),0.,0.])[None],mesh),dict(coulomb_identity={'sha256':'de'*32},support_ranks=[2])
    provider=PhysicalChargeRoundProvider(endpoint,identity,np.array([True,True,True,False]),
        sample_reader=samples,moment_reader=moments,coulomb_reader=coulomb,mesh_xy=mesh)
    recipe=dict(gate_hash=table_hash(GATES),recipe_hash=identity['recipe_hash'],fit_ids=[0,1],held_ids=[2,3],
        distinct_id=[0,1,2,3],role=[1,0,4,3],held=[False,False,True,True],z_ry=z,
        infinity_width=2,imaginary_width=2,line_direction_cap=2,direction_cutoff=1e-10,
        multiplet_relative_tolerance=1e-8,eta_ev=.25,pole_budget=None)
    # Ledger uses the actual incumbent metadata scale, separate from the
    # explicitly smaller endpoint allocation count. No endpoint Meta is forged.
    ledger=CapacityLedger(NS(nk_tot=512,nspinor=1,n_rmu=912),mesh_xy=mesh,device_budget_bytes=1<<30)
    budget=ConstructorCapacity(None,linalg_resolution({'linalg':'local'}),mesh_xy=mesh,
        ledger=ledger,upstream=(),execution='local',packed_endpoints=4)
    plan=SharedPoleRoundPlan(endpoint,recipe,GATES,mesh,'local',False,False,False,False,
        identity,ledger,(),24,6,port_extent(mesh),budget.eigenplan(4),(0,1),(0,0),('M1','M3'),{})
    return provider,plan,budget,value,ds


def test_selected_round_matches_independent_poles_and_held_operators(mesh):
    if mesh.size!=1:pytest.skip('local one-round law uses P1; P4 algebra/metadata have separate plants')
    provider,plan,budget,value,ds=toy(mesh)
    out=construct_shared_pole_round(provider,plan=plan,q_ids=[0],real_rows=1,budget=budget)
    assert out.q_ids==(0,) and out.counts.tolist()==[2]
    b=host(out.factor)[0];lam=out.poles2[0];active=np.arange(len(lam))<out.counts[0]
    for z in [.2+.15j,.75j,1.2+.11j]:
        w=(b[:,active]/(z*z-lam[active]))@b[:,active].conj().T
        d=(-b[:,active]/(z*z-lam[active])**2)@b[:,active].conj().T
        np.testing.assert_allclose(w,value(z),atol=2e-13,rtol=3e-11)
        np.testing.assert_allclose(d,ds(z),atol=2e-13,rtol=3e-11)
    assert out.receipts[0]['identity']==provider.identity
    assert budget.retained_panels==(out.factor,)
    assert out.capacity_entry_end==len(plan.ledger.entries)


@pytest.mark.parametrize('poison', ['identity','recipe','parent','tails','held','fieldghost','fieldnan','layout'])
def test_round_and_provider_refuse_unmatched_premises(mesh,poison):
    provider,plan,budget,_,_=toy(mesh)
    ids=[0];real=1
    if poison=='identity':plan=replace(plan,identity={'other':'state'})
    elif poison=='recipe':plan=replace(plan,recipe=dict(plan.recipe,recipe_hash='ff'*32))
    elif poison=='parent':ids=[1]
    elif poison=='tails':ids=[0,1]
    elif poison=='held':plan=replace(plan,recipe=dict(plan.recipe,held_ids=[]))
    else:
        original=provider._moments
        def bad(ids,*,fields):
            d=original(ids,fields=fields)
            a=host(d[fields[0]])
            if poison=='fieldghost':a[:,3,0]=1e-15
            elif poison=='fieldnan':a[:,0,0]=np.nan
            d[fields[0]]=face(a,mesh) if poison!='layout' else jax.device_put(a)
            return d
        provider._moments=bad
    with pytest.raises(ValueError):
        construct_shared_pole_round(provider,plan=plan,q_ids=ids,real_rows=real,budget=budget)


def test_default_store_constructor_reuses_selected_round_math(mesh,monkeypatch):
    """Full default orchestration vs physical provider using identical toy data.

    Only store/metadata/route doors are simulated; selection, Gram reduction,
    Ritz, moment/passivity/held checks, restore and full writer stack are real.
    """
    if mesh.size!=1:pytest.skip('P1 full local orchestration regression')
    from contextlib import nullcontext
    import file_io.shared_pole_store as store
    import gw.shared_pole_constructor as constructor
    import gw.shared_pole_execution as execution
    import gw.w_isdf as w_isdf
    import gw.shared_pole_round as round_owner
    provider,plan,budget,_,_=toy(mesh)
    reference=construct_shared_pole_round(provider,plan=plan,q_ids=[0],real_rows=1,budget=budget)
    header=dict(recipe=plan.recipe,ordered=False,odd_moments=False,
                bank_shape={'nq':1},line_panels={'sample_span':(0,0)})
    meta=NS(shared_pole_recipe=plan.recipe,shared_pole_capacity=plan.ledger,n_rmu=3,
            n_rmu_padded=4,nspinor=1,nkx=8,nky=8,nkz=8,nk_tot=512)
    config=NS(backend=NS(linalg='local'),debug=NS(sigma_freq_debug_output=False))
    bank=dict(path='samples',identity=provider.identity,tables={'sym':NS(trs_allowed=True)})
    monkeypatch.setattr(store,'charge_representation',lambda meta:True)
    monkeypatch.setattr(store,'validate_shared_pole_bank',lambda *a,**kw:header)
    monkeypatch.setattr(store,'open_shared_pole_bank',lambda path,**kw:nullcontext(path))
    def read(io,*,q_ids,fields,sample_ids=None,sample_span=None,**kw):
        if io=='moments':return provider.moments(q_ids,fields=fields)
        if sample_ids is not None:return provider._sample_rows(q_ids,sample_ids)
        return provider.samples(q_ids,sample_span=sample_span)
    monkeypatch.setattr(store,'read_shared_pole_bank',read)
    monkeypatch.setattr(constructor,'constructor_route',lambda *a,**kw:('local',{},plan.column_extent,6,('M1','M3')))
    monkeypatch.setattr(execution,'sector_round_schedule',lambda *a,**kw:[([0],1,np.array([0]))])
    def coulomb(*a,q_span,**kw):
        inverse,receipt=provider.coulomb_inverse(q_span)
        return inverse,inverse,receipt
    monkeypatch.setattr(w_isdf,'response_coulomb_powers',coulomb)
    captured={}
    def write(output,b,p,c,**kw):captured.update(b=b,p=host(p),c=np.asarray(c),kwargs=kw);return {'test':'strict writer door'}
    monkeypatch.setattr(store,'write_shared_pole_model',write)
    calls=[];original=round_owner.construct_shared_pole_round
    def observed(*a,**kw):calls.append(kw['plan'].endpoint.kind);return original(*a,**kw)
    monkeypatch.setattr(round_owner,'construct_shared_pole_round',observed)
    result=constructor.construct_shared_poles(bank,{'path':'moments'},meta,config,mesh_xy=mesh,output='new-model')
    assert calls==['packed-centroid-charge'] and captured['kwargs']['q_span']==(0,1)
    np.testing.assert_array_equal(captured['c'],reference.counts)
    np.testing.assert_allclose(captured['p'],reference.poles2,atol=2e-13,rtol=3e-12)
    np.testing.assert_allclose(host(captured['b'])@host(captured['b']).conj().swapaxes(-1,-2),
        host(reference.factor)@host(reference.factor).conj().swapaxes(-1,-2),atol=3e-13,rtol=3e-12)
    assert result['model_header']=={'test':'strict writer door'}


def test_four_cpu_geometry_subset_and_explicit_carrier_refusals():
    """Independent process uses all four devices; no giant physical allocation."""
    import os,subprocess,sys
    code=r'''
from runtime import bootstrap
bootstrap(platform='cpu')
import jax,numpy as np
from jax.sharding import Mesh
from types import SimpleNamespace as NS
from gw.response_bank import physical_charge_response_algebra
from gw.shared_pole_capacity import shared_pole_byte_terms
from gw.gw_config import linalg_resolution
full=Mesh(np.asarray(jax.devices()).reshape(2,2),('x','y'))
small=Mesh(np.asarray(jax.devices()[:1]).reshape(1,1),('x','y'))
for mesh,n in [(small,4),(full,3)]:
 try:physical_charge_response_algebra(mesh_xy=mesh,packed_endpoints=n,physical_k_count=512,linalg='local')
 except ValueError:pass
 else:raise AssertionError('incomplete mesh/undivisible endpoint admitted')
try:shared_pole_byte_terms(None,mesh_xy=full,resolution=linalg_resolution({'linalg':'local'}),pencil_side=8,parent_batch=1,sample_batch=1,packed_endpoints=5)
except ValueError:pass
else:raise AssertionError('undivisible face carrier admitted')
print('three active geometry refusals')
'''
    env=dict(os.environ,JAX_PLATFORMS='cpu',XLA_FLAGS='--xla_force_host_platform_device_count=4')
    run=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True,timeout=120)
    assert run.returncode==0 and 'three active geometry refusals' in run.stdout,run.stderr[-3000:]

"""Held numerical transition-space gauge transport; no physical state freeze."""
import numpy as np
import pytest
import jax
from gw import transition_galerkin as owner
from tests.test_native_transition_galerkin_bridge_cpu import _case,_mesh,_bridge

jax.config.update('jax_enable_x64',True)


def test_transport_preserves_full_port_map_K_Gram_and_crossing(monkeypatch):
    from common import collectives
    place=collectives.device_put_process_local;placements=[]
    def process_local(a,sharding):
        placements.append((np.asarray(a).shape,sharding))
        return place(a,sharding)
    monkeypatch.setattr(collectives,'device_put_process_local',process_local)
    grid,psi,energy,rho,partners=_case();mesh=_mesh();nk,nb=energy.shape
    rng=np.random.default_rng(442);active=slice(1,3)
    ur=np.stack([np.linalg.qr(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))[0] for _ in range(nk)])
    uc=np.stack([np.linalg.qr(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))[0] for _ in range(nk)])
    s=ur.conj().swapaxes(-1,-2)@uc
    ref=psi.copy();cur=psi.copy()
    ref[:,1:3]=np.einsum('kasu,kai->kisu',psi[:,1:3],ur)
    cur[:,1:3]=np.einsum('kasu,kai->kisu',psi[:,1:3],uc)
    rnew=rho.copy();rnew[:,1:3,1:3]=s.conj().swapaxes(-1,-2)@rho[:,1:3,1:3]@s
    brid,cat,tref=_bridge(mesh,ref,energy,rho,partners[2])
    _,other,tnew=_bridge(mesh,cur,energy,rnew,partners[2])
    np.testing.assert_array_equal(cat['b'],other['b'])
    raw=(rng.normal(size=(5,*cat['valid'].shape))+1j*rng.normal(size=(5,*cat['valid'].shape)))*cat['valid'][None]
    q=jax.device_put(raw,brid['state_sharding'])
    def move(value,which=cat,reference=ur,current=uc):
        return owner.transport_transition_basis(brid,which,value,reference,current,
            active_slice=active,source_resident_bytes_per_rank=100000,
            max_device_bytes=10**9,max_block_size=3)[0]
    moved=move(q)
    assert len(placements)==5  # two small rotations plus b/d/valid, never Q or mu
    np.testing.assert_allclose(brid['to_port'](tnew,moved),brid['to_port'](tref,q),rtol=3e-13,atol=3e-13)
    np.testing.assert_allclose(brid['gram'](moved,moved),brid['gram'](q,q),rtol=3e-13,atol=3e-13)
    args=[cat[k] for k in ('kmq','b','d','valid')]
    np.testing.assert_allclose(brid['K'](moved,rnew,*args),move(brid['K'](q,rho,*args)),rtol=3e-13,atol=3e-13)
    # Antiunitary q/-q crossing commutes with transport in their exact catalogues.
    inv=np.argsort(cat['kmq'])
    opposite=owner.transition_catalogue(energy,rho,inv,coherent_slice=active,pair_tile=4)
    cross=owner.transition_crossing_map(cat,opposite)
    qcross=brid['cross'](q,cross,opposite['valid'])
    np.testing.assert_allclose(move(qcross,opposite),brid['cross'](moved,cross,opposite['valid']),rtol=3e-13,atol=3e-13)
    if int(mesh.size)>1:
        assert all(sh.data.shape[1]==nk//int(mesh.size) for sh in moved.addressable_shards)
    # Transport never binds an energy table: D still reads CURRENT energies.
    d0=brid['D'](moved,energy,*args,.04)
    d1=brid['D'](moved,energy+np.arange(nb)[None]*.03,*args,.04)
    assert np.max(abs(np.asarray(d1)-np.asarray(d0)))>.01


def test_transport_refuses_unreserved_or_corrupt_catalogue_and_bad_units():
    _,psi,energy,rho,partners=_case();mesh=_mesh()
    brid,cat,_=_bridge(mesh,psi,energy,rho,partners[0])
    q=jax.device_put(np.zeros((2,*cat['valid'].shape),complex),brid['state_sharding'])
    u=np.broadcast_to(np.eye(2,dtype=complex),(len(energy),2,2)).copy()
    def run(c=cat,a=u,budget=10**9):
        return owner.transport_transition_basis(brid,c,q,u,a,active_slice=slice(1,3),
            source_resident_bytes_per_rank=1,max_device_bytes=budget)
    bad=dict(cat,receipt=dict(cat['receipt'],coherent_reservation=None))
    with pytest.raises(ValueError,match='reservation'):run(bad)
    bad=dict(cat,valid=cat['valid'].copy());bad['valid'][0,0]=False
    # Remove a guaranteed active coordinate rather than a possibly frozen row.
    i=np.flatnonzero(cat['valid'][0]&((cat['b'][0]==1)|(cat['d'][0]==1)))[0]
    bad['valid'][0,i]=False
    with pytest.raises(ValueError,match='coordinate omitted'):run(bad)
    with pytest.raises(ValueError,match='unitary'):run(a=u*1.01)
    with pytest.raises(ValueError,match='budget exceeded'):run(budget=1)

"""Tiny native transition-action bridge checks, NOT a production GW route.

No material state, frequency fit, spatial truncation, or trajectory is used.
"""
from types import SimpleNamespace
import numpy as np
import pytest
import jax
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P

from gw import transition_galerkin as owner
from tests._full_microscopic_oracle import ordered_pr_time,ordered_pr,density_vertices,retarded_reference_pairs
from tests.test_full_density_transition_galerkin_cpu import _transition_maps,_column_sigma_from_feedback

jax.config.update("jax_enable_x64", True)


def _mesh():
    devices=jax.devices("cpu")
    n=4 if len(devices)>=4 else 1
    return Mesh(np.asarray(devices[:n]).reshape((2,2) if n==4 else (1,1)),("x","y"))


def _case(ns=2):
    rng=np.random.default_rng(809+ns);grid=(4,2,1);nk,nb,nmu=8,4,4
    psi=.27*(rng.normal(size=(nk,nb,ns,nmu))+1j*rng.normal(size=(nk,nb,ns,nmu)))
    energy=np.array([[-.9,-.24,.41,1.13]])+np.arange(nk)[:,None]*np.array([.011,-.007,.009,-.004])
    rho=np.zeros((nk,nb,nb),complex);rho[:,0,0]=1
    for k in range(nk):
        u=np.linalg.qr(rng.normal(size=(2,2))+1j*rng.normal(size=(2,2)))[0]
        rho[k,1:3,1:3]=u@np.diag([.12,.87])@u.conj().T
    coords=np.stack(np.unravel_index(np.arange(nk),grid),axis=-1)
    partners=[]
    for q in coords:
        p=(coords-q)%np.array(grid)
        partners.append(np.ravel_multi_index(tuple(p.T),grid))
    return grid,psi,energy,rho,np.asarray(partners)


def _bridge(mesh,psi,energy,rho,partner,*,nspin=1,pair_tile=4):
    catalogue=owner.transition_catalogue(energy,rho,partner,
        coherent_slice=slice(1,3),pair_tile=pair_tile)
    factor=owner.physical_transition_normalization(len(energy),nspin,psi.shape[2])
    bridge=owner.make_transition_bridge(mesh,psi.shape,catalogue["b"].shape[1],
        normalization=factor,pair_tile=pair_tile)
    t=bridge["vertex"](psi,partner,catalogue["b"],catalogue["d"],catalogue["valid"])
    return bridge,catalogue,t


def _literal_t(psi,partner,catalogue,factor):
    nk,width=catalogue["b"].shape;out=np.zeros((nk,width,psi.shape[-1]),complex)
    for k in range(nk):
        for p in np.flatnonzero(catalogue["valid"][k]):
            b,d=catalogue["b"][k,p],catalogue["d"][k,p]
            for mu in range(psi.shape[-1]):
                for spin in range(psi.shape[2]):
                    out[k,p,mu]+=factor*psi[k,b,spin,mu]*psi[partner[k],d,spin,mu].conjugate()
    return out


def _unpack(x,catalogue,nb):
    x=np.asarray(x);out=np.zeros((x.shape[0],len(catalogue["b"]),nb,nb),complex)
    for k in range(out.shape[1]):
        keep=catalogue["valid"][k]
        out[:,k,catalogue["b"][k,keep],catalogue["d"][k,keep]]=x[:,k,keep]
    return out


def _pack(x,catalogue):
    return np.stack([x[:,k,catalogue["b"][k],catalogue["d"][k]]
                     *catalogue["valid"][k] for k in range(len(catalogue["b"]))],axis=1)


@pytest.mark.parametrize("ns,nspin",[(1,1),(2,1),(1,2)])
def test_native_transition_vertex_matches_both_literal_and_native_FFT_fullcoherent_PR(ns,nspin):
    from ffi import fft
    from gw.greens_function_kernel import density_left_endpoint
    from gw.w_isdf import full_density_response_stream,_w_solve_pref_scalar
    grid,psi,energy,rho,partners=_case(ns);mesh=_mesh();nk,nb,_,nmu=psi.shape
    times=np.array([.07+.13j,-.11+.19j,.04-.23j])
    pm=psi.transpose(0,2,3,1)
    endpoint=density_left_endpoint(mesh,pm.shape)(pm,rho)
    meta=SimpleNamespace(nk_tot=nk,nspin=nspin,nspinor_wfnfile=ns)
    pref=_w_solve_pref_scalar(meta);spin=2/(nspin*ns)
    with fft.xla_reference():
        stream=full_density_response_stream(mesh,grid,(nk,nb,nmu,ns),q_ids=[0,2,6],n_outputs=len(times))
        native=np.asarray(stream(times,np.eye(len(times),dtype=complex),pm,psi,endpoint,energy))*pref
    for iq,q in enumerate([0,2,6]):
        bridge,cat,t=_bridge(mesh,psi,energy,rho,partners[q],nspin=nspin)
        factor=owner.physical_transition_normalization(nk,nspin,ns)
        np.testing.assert_allclose(t,_literal_t(psi,partners[q],cat,factor),rtol=2e-14,atol=2e-15)
        seed=bridge["to_state"](t,np.eye(nmu,dtype=complex))
        delta=(np.take_along_axis(energy,cat["b"],axis=1)
               -np.take_along_axis(energy[partners[q]],cat["d"],axis=1))
        response=[]
        for time in times:
            # rho belongs to the OUTPUT side: K AFTER the right-index phase.
            phased=seed*np.exp(-delta*time)[None]
            weighted=bridge["K"](phased,rho,partners[q],cat["b"],cat["d"],cat["valid"])
            response.append(np.asarray(bridge["to_port"](t,weighted)))
        literal=ordered_pr_time(psi,energy,rho,partners[q],times,spin_weight=spin)
        np.testing.assert_allclose(response,literal,rtol=4e-12,atol=4e-14)
        np.testing.assert_allclose(response,native[iq],rtol=4e-12,atol=4e-14)
        if int(mesh.size)==4:
            assert all(s.data.shape[0]==nk//4 for s in t.addressable_shards)
            assert all(s.data.shape[1]==nk//4 for s in seed.addressable_shards)


def test_native_bridge_K_D_adjoint_and_changed_current_inputs_match_literal_actions():
    _,psi,energy,rho,partners=_case();mesh=_mesh();partner=partners[2]
    bridge,cat,t=_bridge(mesh,psi,energy,rho,partner);rng=np.random.default_rng(812)
    u=rng.normal(size=(4,3))+1j*rng.normal(size=(4,3))
    x=rng.normal(size=(3,*cat["valid"].shape))+1j*rng.normal(size=(3,*cat["valid"].shape))
    tx=np.asarray(bridge["to_port"](t,x));htu=np.asarray(bridge["to_state"](t,u))
    np.testing.assert_allclose(htu,np.einsum("kpm,mr->rkp",np.asarray(t).conj(),u),rtol=3e-14,atol=3e-14)
    np.testing.assert_allclose(tx,np.einsum("kpm,rkp->mr",np.asarray(t),x),rtol=3e-14,atol=3e-14)
    np.testing.assert_allclose(np.vdot(tx,u),np.vdot(x,htu),rtol=3e-14,atol=3e-14)
    full=_unpack(x,cat,4)
    for current in [rho,rho*.73]:
        exact=np.einsum("rkbd,kde->rkbe",full,current[partner])-np.einsum("kab,rkbd->rkad",current,full)
        actual=bridge["K"](x,current,partner,cat["b"],cat["d"],cat["valid"])
        np.testing.assert_allclose(actual,_pack(exact,cat),rtol=3e-14,atol=3e-14)
    # Same compiled factory must read changed CURRENT energies, not a closure.
    for current in [energy,energy+np.array([.03,-.01,.04,-.02])]:
        delta=(np.take_along_axis(current,cat["b"],axis=1)
               -np.take_along_axis(current[partner],cat["d"],axis=1))
        actual=bridge["D"](x,current,partner,cat["b"],cat["d"],cat["valid"],.047)
        np.testing.assert_allclose(actual,x*(delta-.047j)[None]*cat["valid"][None],rtol=3e-14,atol=3e-14)


def test_live_catalogue_has_exact_full160_count_and_never_pinches_coherent_columns():
    nk,nb=36,160;energy=np.broadcast_to(np.arange(nb,dtype=float),(nk,nb)).copy()
    rho=np.zeros((nk,nb,nb),complex);rho[:,:130,:130]=np.eye(130)
    rho[:,128:132,128:132]=np.array([[.8,.03j,.04,0],[-.03j,.6,.02j,.01],
                                   [.04,-.02j,.3,.02],[0,.01,.02,.1]])
    cat=owner.transition_catalogue(energy,rho,np.roll(np.arange(nk),1),coherent_slice=slice(128,132))
    assert cat["receipt"]["live_transitions"]==303552
    assert np.all(np.asarray(cat["receipt"]["per_k_live"])==8432)
    assert cat["receipt"]["population_threshold"] is None
    # Even a currently empty physical active block retains the same catalogue.
    rho[:,128:132,128:132]=0
    empty=owner.transition_catalogue(energy,rho,np.roll(np.arange(nk),1),coherent_slice=slice(128,132))
    np.testing.assert_array_equal(empty["valid"],cat["valid"])
    # Without a held coherent reservation, full off-window coherences MUST
    # create exact live columns instead of being silently discarded.
    small=np.zeros((1,4,4),complex);small[:,0,0]=1
    base=owner.transition_catalogue(np.arange(4.)[None],small,np.array([0]),pair_tile=1)
    small[:,0,3]=.03j;small[:,3,0]=-.03j
    coherent=owner.transition_catalogue(np.arange(4.)[None],small,np.array([0]),pair_tile=1)
    assert coherent["receipt"]["live_transitions"]>base["receipt"]["live_transitions"]


def test_q_crossing_mapping_preserves_literal_vertex_K_and_signed_D():
    _,psi,energy,rho,partners=_case();mesh=_mesh()
    plus,cp,tp=_bridge(mesh,psi,energy,rho,partners[2])
    minus,cm,tm=_bridge(mesh,psi,energy,rho,partners[6])
    mapping=owner.transition_crossing_map(cp,cm)
    mapped=np.asarray(tp).reshape(-1,4)[mapping]
    np.testing.assert_allclose(tm,mapped.conj()*cm["valid"][:,:,None],atol=2e-15)
    rng=np.random.default_rng(813)
    x=(rng.normal(size=(2,*cp["valid"].shape))+1j*rng.normal(size=(2,*cp["valid"].shape)))*cp["valid"][None]
    reverse=lambda value: np.asarray(value).reshape(2,-1)[:,mapping].conj()*cm["valid"][None]
    xp=reverse(x)
    kp=plus["K"](x,rho,partners[2],cp["b"],cp["d"],cp["valid"])
    km=minus["K"](xp,rho,partners[6],cm["b"],cm["d"],cm["valid"])
    np.testing.assert_allclose(km,-reverse(kp),rtol=3e-14,atol=3e-14)
    dp=plus["D"](x,energy,partners[2],cp["b"],cp["d"],cp["valid"],.04)
    dm=minus["D"](xp,energy,partners[6],cm["b"],cm["d"],cm["valid"],.04)
    np.testing.assert_allclose(dm,-reverse(dp),rtol=3e-14,atol=3e-14)
    # A shared covariant TRANSITION subspace gives q/-q crossing of the
    # actual native full-port modal factors. Odd count tests explicit padding.
    initial=(rng.normal(size=(3,*cp["valid"].shape))+1j*rng.normal(size=(3,*cp["valid"].shape)))*cp["valid"][None]
    q=np.linalg.qr(initial.reshape(3,-1).T)[0].T.reshape(initial.shape)
    qp=jax.device_put(q,plus["state_sharding"])
    qm=minus["cross"](qp,mapping,cm["valid"])
    vv=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4));v=.03*vv@vv.conj().T
    vf=jax.device_put(v,NamedSharding(mesh,P("x","y")))
    vfm=jax.device_put(v.conj(),NamedSharding(mesh,P("x","y")))
    mp=owner.transition_parent_model(plus,tp,cp,energy,rho,vf,qp,.04,max_block_size=2)
    mm=owner.transition_parent_model(minus,tm,cm,energy,rho,vfm,qm,.04,max_block_size=2)
    assert mp["count"]==mm["count"]==3
    expected_carrier=4 if int(mesh.size)==4 else 3
    assert mp["left"].shape[-1]==mm["left"].shape[-1]==expected_carrier
    def response(model,z):
        n=model["count"];l=np.asarray(model["left"])[0,:,:n];r=np.asarray(model["right"])[0,:,:n]
        return (l/(z-model["poles_ry"][:n]))@r.conj().T
    for z in [.2+.3j,-.4+.2j]:
        np.testing.assert_allclose(response(mm,z),response(mp,-z.conjugate()).conj(),rtol=5e-11,atol=3e-13)
    with pytest.raises(ValueError,match="transition_orthogonality"):
        owner.transition_parent_model(plus,tp,cp,energy,rho,vf,qp*.9,.04,max_block_size=2)


def test_held_catalogue_refuses_new_offwindow_coherence_but_reads_active_current_density():
    _,psi,energy,rho,partners=_case();_,cat,_=_bridge(_mesh(),psi,energy,rho,partners[0])
    assert owner.validate_transition_catalogue(cat,rho)
    changed=rho.copy();changed[:,1:3,1:3]*=.8
    assert owner.validate_transition_catalogue(cat,changed)
    changed[:,0,3]=.02j;changed[:,3,0]=-.02j
    with pytest.raises(ValueError,match="transition_catalogue_changed"):
        owner.validate_transition_catalogue(cat,changed)


def _reduced_from_native(bridge,cat,t,basis,energy,rho,v,gamma):
    weighted=bridge["K"](basis,rho,cat["kmq"],cat["b"],cat["d"],cat["valid"])
    ar=np.asarray(bridge["to_port"](t,weighted));br=np.asarray(bridge["to_port"](t,basis))
    image=bridge["D"](basis,energy,cat["kmq"],cat["b"],cat["d"],cat["valid"],gamma)
    q=np.asarray(basis).reshape(basis.shape[0],-1)
    dr=q.conj()@np.asarray(image).reshape(basis.shape[0],-1).T
    return ar,br,dr,dr+br.conj().T@v@ar


def test_native_Dyson_seeds_TSQR_CGS_preserve_allport_actions_and_dual_derivative():
    _,psi,energy,rho,partners=_case();bridge,cat,t=_bridge(_mesh(),psi,energy,rho,partners[2])
    rng=np.random.default_rng(814);z=.21+.34j;gamma=.037
    vv=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4));v=.03*vv@vv.conj().T
    exact_p=ordered_pr(psi,energy,rho,partners[2],[z],gamma)[0]
    w=np.linalg.solve(np.eye(4)-v@exact_p,v)
    u=rng.normal(size=(4,2))+1j*rng.normal(size=(4,2))
    test=rng.normal(size=(4,2))+1j*rng.normal(size=(4,2))
    x,y=owner.dyson_transition_seeds(bridge,t,cat,energy,rho,z,w@u,w.conj().T@test,gamma)
    delta=(np.take_along_axis(energy,cat["b"],axis=1)
           -np.take_along_axis(energy[partners[2]],cat["d"],axis=1))
    raw_right=np.einsum("kpm,mr->rkp",np.asarray(t).conj(),w@u)
    raw_left=np.einsum("kpm,mr->rkp",np.asarray(t).conj(),w.conj().T@test)
    full=_unpack(raw_left,cat,4)
    weighted=np.einsum("rkbd,kde->rkbe",full,rho[partners[2]])-np.einsum("kab,rkbd->rkad",rho,full)
    np.testing.assert_allclose(x,raw_right/(z-delta+1j*gamma)[None]*cat["valid"][None],rtol=3e-13,atol=3e-14)
    np.testing.assert_allclose(y,_pack(weighted,cat)/(z.conjugate()-delta-1j*gamma)[None]
                               *cat["valid"][None],rtol=3e-13,atol=3e-14)
    # Repeated and near-dependent states must not add arbitrary QR completion.
    basis,receipt=owner.transition_basis(bridge,[x,x*(1+1e-14),y],cat,rho,
        capacity=12,rank_tolerance=1e-10,max_block_size=2)
    assert receipt["order"]==4
    assert not receipt["production_default"]
    assert not receipt["price"]["all_parent_retention_admitted"]
    q=np.asarray(basis).reshape(4,-1)
    np.testing.assert_allclose(q.conj()@q.T,np.eye(4),atol=3e-13)
    ar,br,dr,mr=_reduced_from_native(bridge,cat,t,basis,energy,rho,v,gamma)
    rr=np.linalg.solve(z*np.eye(4)-mr,np.eye(4))
    wr=v+v@ar@rr@br.conj().T@v
    np.testing.assert_allclose(wr@u,w@u,rtol=2e-11,atol=2e-13)
    np.testing.assert_allclose(test.conj().T@wr,test.conj().T@w,rtol=2e-11,atol=2e-13)
    # Independent full-transition analytic derivative, using literal indices.
    a,b,dt,_=_transition_maps(psi,energy,rho,partners[2])
    pp=-(a/(z-dt+1j*gamma)**2)@b.conj().T
    derivative=w@pp@w
    reduced_derivative=-v@ar@rr@rr@br.conj().T@v
    np.testing.assert_allclose(test.conj().T@reduced_derivative@u,
                               test.conj().T@derivative@u,rtol=3e-11,atol=2e-13)
    # Actual native parent factors use the same existing canonical modal
    # owner, not the independent host projection above.
    vf=jax.device_put(v,NamedSharding(bridge["mesh"],P("x","y")))
    model=owner.transition_parent_model(bridge,t,cat,energy,rho,vf,basis,gamma,max_block_size=2)
    count=model["count"];poles=model["poles_ry"][:count]
    left=np.asarray(model["left"])[0,:,:count];right=np.asarray(model["right"])[0,:,:count]
    np.testing.assert_allclose(model["projected_generator"],mr,rtol=3e-12,atol=3e-13)
    np.testing.assert_allclose((left/(z-poles))@right.conj().T,wr-v,rtol=3e-11,atol=3e-13)
    residues=np.stack([np.outer(left[:,i],right[:,i].conjugate()) for i in range(count)])
    vertices=density_vertices(psi,partners[2],[1,2]);ep,ev=np.linalg.eig(mr)
    actual=retarded_reference_pairs(vertices,energy[partners[2]],rho[partners[2]],energy[:,1:3],
                                    poles,residues,.4)
    expected=_column_sigma_from_feedback(vertices,energy[partners[2]],rho[partners[2]],energy[:,1:3],
        mr,v@ar,br.conj().T@v,ep,ev,.4)
    for table,reference in zip(actual,expected):
        np.testing.assert_allclose(table,reference,rtol=3e-10,atol=3e-13)
        assert np.linalg.norm(reference.imag)>1e-5
    assert not model["receipt"]["production_default"]
    assert not model["receipt"]["all_parent_retention_admitted"]
    with pytest.raises(ValueError,match="transition_capacity"):
        owner.transition_basis(bridge,[x],cat,rho,capacity=12,rank_tolerance=1e-10,
            max_block_size=2,max_device_bytes=1)
    with pytest.raises(ValueError,match="transition_capacity"):
        owner.transition_basis(bridge,[x,y],cat,rho,capacity=1,rank_tolerance=1e-10,max_block_size=1)


def test_native_stationary_sector_basis_and_selfcrossing_closure_are_structural():
    _,psi,energy,rho,partners=_case();rho[:]=np.diag([1.,1.,0.,0.]);mesh=_mesh()
    cat=owner.transition_catalogue(energy,rho,partners[0],pair_tile=4)
    bridge=owner.make_transition_bridge(mesh,psi.shape,cat["b"].shape[1],
        normalization=owner.physical_transition_normalization(8,1,2),pair_tile=4)
    t=bridge["vertex"](psi,cat["kmq"],cat["b"],cat["d"],cat["valid"])
    rng=np.random.default_rng(815);seed=rng.normal(size=(2,*cat["valid"].shape))+1j*rng.normal(size=(2,*cat["valid"].shape))
    crossing=owner.transition_crossing_map(cat,cat)
    basis,receipt=owner.transition_basis(bridge,[seed],cat,rho,capacity=16,
        rank_tolerance=1e-10,max_block_size=2,self_crossing=crossing)
    assert receipt["stationary_binary_sectors"]
    q=np.asarray(basis).reshape(basis.shape[0],-1)
    mirror=q.conj()[:,crossing.reshape(-1)]
    coefficients=q.conj()@mirror.T
    np.testing.assert_allclose(coefficients.T@q,mirror,rtol=3e-12,atol=3e-13)
    v=np.eye(4)*.4;gamma=.04
    ar,br,dr,mr=_reduced_from_native(bridge,cat,t,basis,energy,rho,v,gamma)
    kq=np.asarray(bridge["K"](basis,rho,cat["kmq"],cat["b"],cat["d"],cat["valid"])).reshape(q.shape)
    j=q.conj()@kq.T
    np.testing.assert_allclose(j,np.diag(j.diagonal()),atol=3e-13)
    np.testing.assert_allclose(abs(j.diagonal()),1,atol=3e-13)
    np.testing.assert_allclose(ar,br@j,rtol=3e-12,atol=3e-13)
    positive=j@(mr+1j*gamma*np.eye(mr.shape[0]))
    np.testing.assert_allclose(positive,positive.conj().T,atol=3e-12)
    assert np.linalg.eigvalsh(positive).min()>.1
    np.testing.assert_allclose(np.linalg.eigvals(mr).imag,-gamma,atol=3e-12)


def test_binary_reserved_catalogue_refuses_without_silently_removing_Kzero_rows():
    _,psi,energy,rho,partners=_case();rho[:]=np.diag([1.,1.,0.,0.])
    bridge,cat,_=_bridge(_mesh(),psi,energy,rho,partners[0])
    # Reservation retains same-filled active coordinates for a later
    # coherent state. The internal stationary initializer does not yet
    # support those K=0 rows and must not silently shrink the held catalogue.
    assert cat["receipt"]["coherent_reservation"]==[1,3]
    original=cat["valid"].copy()
    seed=np.ones((1,*cat["valid"].shape),complex)
    with pytest.raises(ValueError,match="transition_sectors.*exact live catalogue"):
        owner.transition_basis(bridge,[seed],cat,rho,capacity=4,
            rank_tolerance=1e-10,max_block_size=1)
    np.testing.assert_array_equal(cat["valid"],original)


def test_native_parent_model_retains_true_inverted_growth_without_pole_projection():
    mesh=_mesh();nk,nb,ns,nmu=8,2,1,2
    psi=np.ones((nk,nb,ns,nmu),complex)
    energy=np.broadcast_to(np.array([-.5,.5]),(nk,nb)).copy()
    rho=np.broadcast_to(np.diag([0.,1.]),(nk,nb,nb)).astype(complex).copy()
    cat=owner.transition_catalogue(energy,rho,np.arange(nk),pair_tile=2)
    bridge=owner.make_transition_bridge(mesh,psi.shape,cat["b"].shape[1],
        normalization=owner.physical_transition_normalization(nk,1,1),pair_tile=2)
    t=bridge["vertex"](psi,cat["kmq"],cat["b"],cat["d"],cat["valid"])
    nt=cat["valid"].size
    basis=jax.device_put(np.eye(nt,dtype=complex).reshape(nt,*cat["valid"].shape),bridge["state_sharding"])
    v=jax.device_put(.8*np.eye(nmu,dtype=complex),NamedSharding(mesh,P("x","y")))
    gamma=.04
    model=owner.transition_parent_model(bridge,t,cat,energy,rho,v,basis,gamma,max_block_size=2)
    growth=model["poles_ry"][:model["count"]].imag.max()
    np.testing.assert_allclose(growth,np.sqrt(5.4)-gamma,rtol=3e-12,atol=3e-13)
    assert growth>2.2
    pole=model["poles_ry"][:model["count"]][np.argmax(model["poles_ry"][:model["count"]].imag)]
    literal_p=ordered_pr(psi,energy,rho,np.arange(nk),[pole],gamma,spin_weight=2.)[0]
    epsilon=np.eye(nmu)-np.asarray(v)@literal_p
    assert np.linalg.svd(epsilon,compute_uv=False)[-1]<3e-13

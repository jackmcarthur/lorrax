"""Support-owner refusals and polarized same-Z hybrid identity."""
from types import SimpleNamespace
import copy
import numpy as np
import pytest


def supported_fixture(bvec=None):
    from tests.test_streamed_current_coefficients import fixture
    from common.coulomb_sphere import compute_per_q_bare_coulomb_components
    from symmetry_maps import bgw_integer_q_to_fractional
    from vcoul import CoulombGeometry
    from isdf.atomic_breit import static_breit_body_head_slots
    from isdf.zeta_mubatch import ZStore
    from runtime.padding import PaddedAxis, padded_axis
    f = fixture()
    grid = np.asarray([16, 16, 16], np.int32)
    B = np.eye(3) if bvec is None else np.asarray(bvec, float)
    kgrid = np.asarray([3, 3, 1], np.int32)
    integer_k = np.indices(tuple(kgrid)).reshape(3, -1).T
    q = bgw_integer_q_to_fractional(integer_k[f.zetas[0].fit_q_full_indices], kgrid)
    # A poison Nyquist pad row is not a physical source reciprocal vector.
    source_G = np.asarray([[[0, 0, 0], [1, 0, 0], [8, 8, 8]]]*3, np.int32)
    source_n = np.asarray([2, 2, 2], np.int32)
    k = np.asarray([[0., 0., 0.], [0., 1/3, 0.], [1/3, 0., 0.]])
    wfn = SimpleNamespace(blat=1., bvec=B, cell_volume=(2*np.pi)**3/np.linalg.det(B),nbands=200,
        fft_grid=grid, kgrid=kgrid, gvecs=lambda **kw:source_G,
        ngk_valid=lambda **kw:source_n, kvecs=lambda **kw:k, alat=1.,
        avec=2*np.pi*np.linalg.inv(B).T,
        atom_crys=np.asarray([[.2,.3,.5],[.6,.3,.52]]))
    plan = SimpleNamespace(sym=SimpleNamespace(parent_k_domain='ibz',kvecs_asints=integer_k),
                           k_parent_frac=k, mesh_xy=f.mesh)
    package = compute_per_q_bare_coulomb_components(tuple(grid), B, q, 8.)
    G, counts = package['gvec_components_padded'], package['ngk_per_q']
    for z in f.zetas:
        z.ngk_per_q = counts.copy();z.gvec_components = G.copy()
        z.store = ZStore(mesh=f.mesh,q_axis=PaddedAxis('q',f.Q,f.Qp,1),
            mu_pad=f.mu,g_axis=padded_axis(G.shape[-1],4,name='G'),b=f.mu,placement='host')
    policy = static_breit_body_head_slots(q_frac=q,gvec_components=G,ngk_per_q=counts,
        geometry=CoulombGeometry.from_wfn(wfn),vcoul_cutoff_ry=8.,sys_dim=2)
    return f, wfn, plan, policy, source_G


def test_actual_support_three_bank_store_matches_signed_direct_solve():
    from isdf.current_support import certify_smooth_current_support
    from isdf.current_coefficients import CurrentCoefficientStream
    f,wfn,plan,policy,_ = supported_fixture()
    proof = certify_smooth_current_support(wfn,plan,f.zetas,policy)
    stream = CurrentCoefficientStream(f.zetas,atoms=f.na,harmonics=f.nh,
        radial_points=f.nr,radial_packet=4,identity='same-field',
        angular_identity='same-angular-support',operand_kind='delta',smooth_support=proof)
    for first in (4,0):
        for c in range(3):
            stream.append(c,0,f.put(f.rhs[c][0][...,first:first+4].reshape(f.Qp,f.mu,-1)),first)
    banks = stream.finish();actual = banks.authenticate(f.zetas)
    assert len(actual)==3 and all(len(pair)==1 for pair in actual)
    with pytest.raises(AttributeError):banks.operand_kind='delta_and_ps'
    with pytest.raises(AttributeError):banks.smooth_support=True
    assert stream.memory_receipt['final_half_banks']==3*16*np.prod(banks.shape)
    for c in range(3):
        expected=np.stack([np.linalg.solve(f.C[c][q],f.rhs[c][0][q].reshape(f.mu,-1))
                           for q in range(f.Q)])
        np.testing.assert_allclose(np.asarray(actual[c][0])[:f.Q],expected,atol=3e-15,rtol=3e-13)
        assert not np.asarray(actual[c][0])[f.Q:].any()
    banks.close()
    assert proof._wfn is None and not proof._factors
    with pytest.raises(ValueError,match='unchanged actual OWN'):
        proof.authenticate(f.zetas,policy)


@pytest.mark.parametrize('mutation',['cutoff','FFT','q','missing_G','source_growth','negative_q_alias'])
def test_support_refuses_incomplete_or_changed_actual_geometry(mutation):
    from isdf.current_support import certify_smooth_current_support
    f,wfn,plan,policy,source = supported_fixture()
    policy=copy.deepcopy(policy)
    if mutation=='cutoff':policy['cutoff_ry']=1.
    elif mutation=='FFT':wfn.fft_grid=np.asarray([4,4,4],np.int32)
    elif mutation=='q':policy['q_frac'][1,0]+=.1
    elif mutation=='missing_G':policy['ngk_per_q'][0]-=1
    elif mutation=='source_growth':source[0,1,0]=4
    else:
        for z in f.zetas:z.fit_q_full_indices=np.asarray([0,-4,1])
    with pytest.raises(ValueError):certify_smooth_current_support(wfn,plan,f.zetas,policy)


def test_three_bank_refuses_user_bool_instead_of_support_owner():
    from isdf.current_coefficients import CurrentCoefficientStream
    f,_,_,_,_=supported_fixture()
    with pytest.raises(ValueError,match='actual same-Z support'):
        CurrentCoefficientStream(f.zetas,atoms=f.na,harmonics=f.nh,
            radial_points=f.nr,radial_packet=4,identity='same-field',
            angular_identity='same-angular-support',operand_kind='delta',smooth_support=True)


def test_support_nonorthogonal_cell_nonGamma_complete_sphere_and_source_ghosts():
    from isdf.current_support import certify_smooth_current_support
    B=np.asarray([[1.,.12,0],[.08,1.02,0],[0,0,.95]])
    f,wfn,plan,policy,_=supported_fixture(B)
    proof=certify_smooth_current_support(wfn,plan,f.zetas,policy)
    d=proof.descriptor
    assert 0<d['source_max_K2_Ry']<2.
    assert d['pair_bandwidth_bound_Ry']<d['body_cutoff_Ry']
    assert min(d['unique_FFT_margin'])>0
    assert np.any(np.asarray(policy['q_frac'])!=0.)
    # The same physical body count is admitted with a larger inert carrier.
    padded=np.pad(policy['gvec_components'],((0,0),(0,0),(0,3)),constant_values=8)
    policy=copy.deepcopy(policy);policy['gvec_components']=padded
    policy['q_plus_G_cart']=(padded.transpose(0,2,1)+policy['q_frac'][:,None])@B
    from tests.test_static_breit_body_heads import rehash
    rehash(policy)
    for z in f.zetas:z.gvec_components=padded.copy()
    certify_smooth_current_support(wfn,plan,f.zetas,policy)


def test_sealed_support_refuses_actual_source_cell_change():
    from isdf.current_support import certify_smooth_current_support
    f,wfn,plan,policy,_=supported_fixture()
    proof=certify_smooth_current_support(wfn,plan,f.zetas,policy)
    wfn.bvec=wfn.bvec*2
    with pytest.raises(ValueError,match='actual WFN reciprocal support changed'):
        proof.authenticate(f.zetas,policy)


def test_sealed_support_refuses_mutated_live_q_chart():
    from isdf.current_support import certify_smooth_current_support
    f,wfn,plan,policy,_=supported_fixture()
    proof=certify_smooth_current_support(wfn,plan,f.zetas,policy)
    plan.sym.kvecs_asints[5,0]+=1
    with pytest.raises(ValueError,match='actual WFN reciprocal support changed'):
        proof.authenticate(f.zetas,policy)


def physical_fixture():
    from isdf.current_support import certify_smooth_current_support
    f,wfn,plan,policy,_=supported_fixture()
    centers=(wfn.atom_crys%1.)@(wfn.alat*wfn.avec)
    radii=np.asarray([.2,.3])
    proof=certify_smooth_current_support(wfn,plan,f.zetas,policy,
        centers_cart=centers,support_radius=radii)
    return f,wfn,plan,policy,centers,radii,proof


def test_physical_layer_actual_atoms_all_images_and_unitary_point_kernel():
    from isdf.atomic_breit import _current_point_body_tile
    f,wfn,_,policy,centers,radii,proof=physical_fixture()
    assert proof.physical_mixed
    layer=proof.descriptor['physical_Slab_layer']
    Lz=2*np.pi
    # Independent enumeration includes both half-period planes and remote
    # images; the same-atom diameter test alone would miss cross-layer overlap.
    expected=np.asarray([[min(abs(a[2]-b[2]-(m+.5)*Lz) for m in range(-3,4))-ra-rb
        for b,rb in zip(centers,radii)] for a,ra in zip(centers,radii)])
    np.testing.assert_allclose(layer['all_pair_clearance_margin'],expected,rtol=0,atol=1e-15)
    assert min(np.asarray(layer['all_pair_clearance_margin']).ravel())>0
    with pytest.raises(AttributeError):proof.physical_mixed=False
    copied=proof.descriptor;copied['physical_Slab_layer']['support_radii'][0]=10.
    assert proof.descriptor['physical_Slab_layer']['support_radii']==radii.tolist()
    U=np.asarray([[1,1j,0],[1,-1j,0],[0,0,np.sqrt(2)]],complex)/np.sqrt(2)
    pairs=tuple((a,b) for a in range(3) for b in range(3))
    ng=policy['gvec_components'].shape[-1];pref=-8*np.pi/wfn.cell_volume
    tables=_current_point_body_tile(policy,np.arange(f.Q),0,ng,pairs,U,pref,sys_dim=2)
    proof.authenticate_point_kernel(pairs,tuple(tables[:,i] for i in range(9)),U,pref)
    proof.authenticate(f.zetas,policy,centers_cart=centers,support_radius=radii)


@pytest.mark.parametrize('mutation',[None,'source','kernel','layer'])
def test_early_physical_preflight_has_no_OWN_dependency(mutation):
    from isdf.atomic_breit import _current_point_body_tile
    from isdf.current_support import preflight_physical_current_support
    f,wfn,plan,policy,source=supported_fixture()
    centers=(wfn.atom_crys%1.)@(wfn.alat*wfn.avec)
    pair=((0,0),);pref=-8*np.pi/wfn.cell_volume
    table=_current_point_body_tile(policy,np.arange(f.Q),0,policy['gvec_components'].shape[-1],
                                 pair,np.eye(3),pref,sys_dim=2)[:,0]
    rows=f.zetas[0].fit_q_full_indices.copy()
    for z in f.zetas:z.L_q=None;z.lu_piv=None
    if mutation=='source':source[0,1,0]=5
    elif mutation=='kernel':table[1,0]+=.001
    elif mutation=='layer':
        wfn.atom_crys[1,2]=wfn.atom_crys[0,2]+.5
        centers=(wfn.atom_crys%1.)@(wfn.alat*wfn.avec)
    call=lambda:preflight_physical_current_support(wfn,plan,rows,policy,
        centers_cart=centers,support_radius=.2,pairs=pair,v_tables=(table,),
        current_basis_rows=np.eye(3),prefactor=pref)
    if mutation is None:
        result=call()
        assert isinstance(result,dict) and result['physical_source_bands']==200
        assert result['physical_Slab_layer']['minimum_clearance_margin']>0
    else:
        with pytest.raises(ValueError):call()


@pytest.mark.parametrize('mutation',['cross_layer','centers','radius','qz','tilted','Gamma'])
def test_physical_layer_refuses_unsupported_geometry(mutation):
    from isdf.current_support import _slab_layer_geometry
    f,wfn,_,policy,centers,radii,_=physical_fixture()
    policy=copy.deepcopy(policy);q=np.asarray(policy['q_frac']).copy()
    if mutation=='cross_layer':
        wfn.atom_crys[1,2]=wfn.atom_crys[0,2]+.5
        centers=(wfn.atom_crys%1.)@(wfn.alat*wfn.avec)
        # Each individual sphere is smaller than zc/2, but the pair fails.
        assert np.all(2*radii<policy['slab_half_height'])
    elif mutation=='centers':centers[0,0]+=.001
    elif mutation=='radius':radii[0]=np.pi
    elif mutation=='qz':q[1,2]=1/3
    elif mutation=='tilted':policy['bvec'][2,0]=.1
    else:policy['gamma_head_cartesian'][0,0,0]=1.
    with pytest.raises(ValueError):_slab_layer_geometry(wfn,policy,q,centers,radii)


@pytest.mark.parametrize('mutation',['actual_atoms','prepared_radius','policy'])
def test_physical_layer_sealed_actual_binding_refuses_changes(mutation):
    f,wfn,_,policy,centers,radii,proof=physical_fixture()
    if mutation=='actual_atoms':
        wfn.atom_crys[0,0]+=.001
        with pytest.raises(ValueError,match='actual atom geometry changed'):proof.authenticate(f.zetas)
    elif mutation=='prepared_radius':
        with pytest.raises(ValueError,match='prepared geometry changed'):
            proof.authenticate(f.zetas,policy,centers_cart=centers,support_radius=radii*.9)
    else:
        policy['v_avg'][1]*=2
        with pytest.raises(ValueError,match='body/head identity changed'):proof.authenticate(f.zetas)


@pytest.mark.parametrize('mutation',['finite_head','Gamma','scale'])
def test_physical_layer_refuses_actual_nonpoint_kernel_tables(mutation):
    from isdf.atomic_breit import _current_point_body_tile
    f,wfn,_,policy,_,_,proof=physical_fixture()
    pair=((0,0),);pref=-8*np.pi/wfn.cell_volume
    ng=policy['gvec_components'].shape[-1]
    table=_current_point_body_tile(policy,np.arange(f.Q),0,ng,pair,np.eye(3),pref,sys_dim=2)[:,0]
    if mutation=='finite_head':
        selected=policy['sel'][1,policy['mask'][1]==1]
        table[1,selected]+=.001
    elif mutation=='Gamma':
        zero=np.sum(policy['q_plus_G_cart'][0]**2,axis=-1)<1e-12
        table[0,zero]=.001
    else:table*=2
    with pytest.raises(ValueError,match='actual point Slab kernel'):
        proof.authenticate_point_kernel(pair,(table,),np.eye(3),pref)


def test_polarized_physical_mixed_tile_retains_both_adjoints_and_compensation():
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding,PartitionSpec as P
    from tests.test_streamed_current_coefficients import fixture
    from isdf.zeta_mubatch import _pair_tile_kernel
    f=fixture();rng=np.random.default_rng(67131)
    random=lambda shape:rng.normal(size=shape)+1j*rng.normal(size=shape)
    Q,mu,ng=f.Qp,f.mu,4
    Z,delta,g=([random((Q,mu,ng)) for _ in range(3)] for _ in range(3))
    pairs=tuple((a,b) for a in range(3) for b in range(3))
    target=random((len(pairs),Q,ng))
    put=lambda a,s:jax.device_put(np.asarray(a),NamedSharding(f.mesh,s))
    qs=P(('x','y'),None,None);acc=P(None,'x','y')
    fn=_pair_tile_kernel(f.mesh,ng,pairs,use_local_augmentation=True)
    values=fn(tuple(put(v,qs) for v in Z),put(target,P(None,('x','y'),None)),
        put(np.full(Q,ng,np.int32),P(('x','y'))),put(np.zeros((Q,1),np.int32),P(('x','y'),None)),
        jnp.int32(0),tuple(put(np.zeros((Q,mu,mu),complex),acc) for _ in pairs),
        tuple(put(np.zeros((Q,mu,1),complex),acc) for _ in range(3)),
        (tuple(put(v,qs) for v in delta),tuple(put(v,qs) for v in g)))[0]
    errors=[];omissions=[[],[],[],[]]
    for i,(a,b) in enumerate(pairs):
        inner=lambda l,r:np.einsum('qmg,qg,qng->qmn',l.conj(),target[i],r)
        terms=(inner(Z[a],Z[b]),inner(Z[a],delta[b]),inner(delta[a],Z[b]),inner(g[a],g[b]))
        expected=sum(terms)
        got=np.asarray(values[i]);errors.append(np.max(abs(got-expected)))
        for t,term in enumerate(terms):omissions[t].append(np.max(abs(got-term-expected)))
    tolerance=2e-11
    assert max(errors)<tolerance
    assert all(max(error)>tolerance for error in omissions)

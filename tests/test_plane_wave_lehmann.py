"""Independent finite-G ordered Γ response and full-W derivative controls."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from gw.lehmann_response import lehmann_pair_weights
from gw.plane_wave_lehmann import gamma_transition_vertices, GammaLehmannResponse
from gw.plane_wave_screening import SphereScreening
from gw.response_bank import screened_interaction_slope


@pytest.fixture
def mesh():
    return Mesh(np.asarray(jax.devices()[:1]).reshape(1,1),("x","y"))


def plant(mesh):
    rng=np.random.default_rng(88733)
    g=np.asarray([[0,0,0],[1,0,0],[-1,0,0]],np.int32)
    ea=np.asarray([[-2.,-1.1],[-1.9,-1.05]])
    eb=np.asarray([[.3,.8,1.7,1e100],[.4,1e100,1.8,-1e100]])
    valid=np.asarray([[True,True,True,False],[True,False,True,False]])
    vertices=rng.normal(size=(4,2,2,4))+1j*rng.normal(size=(4,2,2,4))
    vertices=np.where(valid.T[:,:,None,None],vertices,0.)
    vertices[...,3]=0.
    placed=jax.device_put(vertices,NamedSharding(mesh,P(("x","y"),None,None,None)))
    bank=GammaLehmannResponse(placed,ea,eb,valid,gvecs=g,mesh=mesh,
        cell_volume=11.,n_spin=1,n_spinor=1,panel_bytes=4096)
    return bank,vertices,ea,eb,valid,g


def independent_response(vertices,ea,eb,valid,z):
    """Both directed transitions, conventional M and independent spin/(ΩNk)."""
    out=np.zeros((len(z),1,4,4),complex);slope=np.zeros_like(out)
    negative=np.asarray([0,2,1,3])
    for k in range(2):
        for b in np.flatnonzero(valid[k]):
            for a in range(2):
                forward=vertices[b,k,a]
                reverse=forward[negative].conj()
                delta=ea[k,a]-eb[k,b]
                for de,df,m in ((delta,1.,forward),(-delta,-1.,reverse)):
                    denominator=de+z
                    outer=np.outer(m,m.conj())
                    out[:,0]+=df/denominator[:,None,None]*outer
                    slope[:,0]+=-df/(2*z*denominator**2)[:,None,None]*outer
    return out*(2./(11.*2)),slope*(2./(11.*2))


def test_ordered_complex_g_vertices_and_slope_match_independent_sum(mesh):
    bank,vertices,ea,eb,valid,g=plant(mesh)
    z=np.asarray([.7+.25j,1.4+.5j,.2+1.3j])
    expected,ds=independent_response(vertices,ea,eb,valid,z)
    assert np.max(np.abs(expected-expected.swapaxes(-1,-2)))>.01
    assert np.max(np.abs(expected-expected.swapaxes(-1,-2).conj()))>.01
    value,derivative=bank.evaluate(z,with_derivative=True)
    np.testing.assert_allclose(value,expected,rtol=3e-12,atol=3e-12)
    np.testing.assert_allclose(derivative,ds,rtol=3e-12,atol=3e-12)
    np.testing.assert_array_equal(bank.evaluate(z),value)
    assert bank.receipt["physical_pair_count"]==10
    assert np.max(np.abs(np.asarray(value)[...,3,:]))==0.
    assert np.max(np.abs(np.asarray(value)[...,:,3]))==0.


def test_ordered_response_derivative_chain_in_complex_squared_frequency(mesh):
    bank,*_=plant(mesh);z=.7+.25j
    value,derivative=bank.evaluate(np.asarray([z]),with_derivative=True)
    s=z*z
    for h in (1e-5,5e-6):
        sites=np.asarray([np.sqrt(s+h),np.sqrt(s-h)])
        assert np.all(sites.imag>0)
        samples=np.asarray(bank.evaluate(sites))
        difference=(samples[0]-samples[1])/(2*h)
        np.testing.assert_allclose(difference,np.asarray(derivative)[0],rtol=2e-9,atol=2e-9)


@pytest.mark.parametrize("ghost",["band","g_slot"])
def test_nonzero_native_or_g_padding_vertices_refuse(mesh,ghost):
    bank,v,ea,eb,valid,g=plant(mesh)
    if ghost=="band":v[3,0,0,0]=1e-20
    else:v[0,0,0,3]=1e-20
    placed=jax.device_put(v,NamedSharding(mesh,P(("x","y"),None,None,None)))
    with pytest.raises(ValueError,match="nonzero native-band or retained-G ghost"):
        GammaLehmannResponse(placed,ea,eb,valid,gvecs=g,mesh=mesh,cell_volume=11.)


@pytest.mark.parametrize("control",["missing_negative","duplicate_g","bad_validity","overlap"])
def test_unauthenticated_gamma_transition_census_refuses(mesh,control):
    bank,v,ea,eb,valid,g=plant(mesh)
    if control=="missing_negative":g[2]=[2,0,0]
    elif control=="duplicate_g":g[2]=g[1]
    elif control=="bad_validity":valid=valid.astype(np.int32)
    else:eb[0,0]=-2.1
    placed=jax.device_put(v,NamedSharding(mesh,P(("x","y"),None,None,None)))
    with pytest.raises(ValueError):
        GammaLehmannResponse(placed,ea,eb,valid,gvecs=g,mesh=mesh,cell_volume=11.)


@pytest.mark.parametrize("z",[[],[0.],[.7-.1j],[np.inf+.1j]])
def test_nonfinite_or_nonretarded_frequency_sites_refuse(mesh,z):
    bank,*_=plant(mesh)
    with pytest.raises(ValueError,match="finite, nonempty, upper-half-plane"):
        bank.evaluate(np.asarray(z))


def test_transition_fft_matches_independent_spinor_coefficient_convolution(mesh):
    rng=np.random.default_rng(8122);n=8
    support=np.asarray([[-1,0,0],[0,0,0],[1,0,0]],int)
    ca=rng.normal(size=(2,2,3))+1j*rng.normal(size=(2,2,3))
    cb=rng.normal(size=(4,2,3))+1j*rng.normal(size=(4,2,3))
    ca/=np.sqrt(np.sum(np.abs(ca)**2,axis=(1,2)))[:,None,None]
    cb/=np.sqrt(np.sum(np.abs(cb)**2,axis=(1,2)))[:,None,None]
    r=np.stack(np.meshgrid(*(np.arange(n)/n for _ in range(3)),indexing="ij"),axis=-1)
    phase=np.exp(2j*np.pi*np.einsum("gd,xyzd->gxyz",support,r))/np.sqrt(n**3)
    a=np.einsum("asg,gxyz->asxyz",ca,phase)[None]
    b=np.einsum("bsg,gxyz->bsxyz",cb,phase)[None]
    a=jax.device_put(a,NamedSharding(mesh,P(None,None,None,None,None,None)))
    b=jax.device_put(b,NamedSharding(mesh,P(None,("x","y"),None,None,None,None)))
    retained=np.asarray([[0,0,0],[1,0,0],[-1,0,0]],int)
    indices=np.ravel_multi_index(tuple((retained%n).T),(n,n,n))[None]
    actual=gamma_transition_vertices(a,b,indices,mesh=mesh,
        occupied_replication_bound_bytes=a.size*a.dtype.itemsize)
    expected=np.zeros((4,2,3),complex)
    for bi in range(4):
        for ai in range(2):
            for gi,G in enumerate(retained):
                for l,Gl in enumerate(support):
                    for m,Gm in enumerate(support):
                        if np.array_equal(Gm-Gl,G):
                            expected[bi,ai,gi]+=np.vdot(ca[ai,:,l],cb[bi,:,m])
    np.testing.assert_allclose(actual,expected,atol=2e-13,rtol=2e-13)
    with pytest.raises(ValueError,match="replication bound"):
        gamma_transition_vertices(a,b,indices,mesh=mesh,occupied_replication_bound_bytes=1)


@pytest.mark.parametrize("with_derivative",[False,True])
def test_pure_lehmann_physical_pair_mask_and_derivative(with_derivative):
    de=np.asarray([[-2.,-1.],[1.,2.]])
    df=np.asarray([[1.,.25],[-.25,-1.]])
    mask=np.asarray([[True,False],[True,True]])
    z=np.asarray([.7+.3j,.4+1.2j]);den=de[None]+z[:,None,None]
    expected=df[None]/den
    if with_derivative:expected=np.concatenate((expected,-df[None]/(2*z[:,None,None]*den**2)))
    expected=np.where(mask[None],expected,0.)
    np.testing.assert_allclose(lehmann_pair_weights(de,df,z,pair_mask=mask,
        with_derivative=with_derivative),expected,atol=1e-14,rtol=1e-14)


def test_full_noncommuting_dyson_slope_without_transpose_or_adjoint(mesh):
    rng=np.random.default_rng(5883)
    a=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4));a=a@a.conj().T*.02
    b=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4));b=b@b.conj().T*.03
    v=np.diag([0.,2.,3.,4.]);s=(.7+.25j)**2
    def exact(t):
        chi=a/(t-2.)+b/(t-5.)
        return np.linalg.solve(np.eye(4)-v@chi,v)
    chi=a/(s-2.)+b/(s-5.);dchi=-a/(s-2.)**2-b/(s-5.)**2
    w=exact(s);expected=w@dchi@w
    assert np.linalg.norm(w@dchi@w.conj().T-expected)>1e-3
    assert np.linalg.norm(w@dchi.T@w-expected)>1e-3
    face=NamedSharding(mesh,P(None,"x","y"))
    device_w=jax.device_put(w[None],face);device_dchi=jax.device_put(dchi[None],face)
    actual=screened_interaction_slope(device_w,device_dchi,mesh_xy=mesh,backend="off")
    np.testing.assert_allclose(actual[0],expected,atol=2e-12,rtol=2e-12)
    for h in (1e-5,5e-6):
        np.testing.assert_allclose((exact(s+h)-exact(s-h))/(2*h),expected,atol=2e-9,rtol=2e-9)
    # Test the canonical value+slope composition with a planted bare operator.
    solver=SphereScreening.__new__(SphereScreening)
    solver.mesh=mesh;solver.linalg="local";solver.batched_route="auto";solver.axis=None
    solver._V=jax.device_put(v[None].astype(complex),face)
    value,slope=solver.solve_pair(jax.device_put(chi[None],face),device_dchi)
    np.testing.assert_allclose(value[0],w,atol=2e-12,rtol=2e-12)
    np.testing.assert_allclose(slope[0],expected,atol=2e-12,rtol=2e-12)

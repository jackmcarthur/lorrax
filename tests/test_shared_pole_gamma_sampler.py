"""Independent complex pole sums and same-time endpoint derivative plants."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from gw.shared_pole_head import realized_gamma_correlation_sampler,_realized_gamma_body


def identity(a,t):return a,t
def transpose_projection(a,t):return .5*(a+t),t


@pytest.fixture
def mesh():
    # CPU/unit or one-process GPU unit geometry; actual P4 plant is separate.
    return Mesh(np.asarray(jax.devices()[:1]).reshape(1,1),("x","y"))


def data(mesh):
    rng=np.random.default_rng(8402026)
    b=rng.normal(size=(1,4,4))+1j*rng.normal(size=(1,4,4));b[...,2:]*=1e20
    poles=np.array([[.2,1.3,10.,-.4]]);counts=np.array([2],np.int32)
    face=NamedSharding(mesh,P(None,"x","y"));rep=NamedSharding(mesh,P())
    return b,poles,counts,jax.device_put(b,face),jax.device_put(poles,rep),jax.device_put(counts,rep)


def literal(b,poles,s):
    value=np.zeros((1,4,4),complex);slope=value.copy()
    for p in range(2):
        residue=np.outer(b[0,:,p],b[0,:,p].conj())
        value[0]+=residue/(s-poles[0,p]);slope[0]-=residue/(s-poles[0,p])**2
    return value,slope


@pytest.mark.parametrize("s",[-.01+0j,.7+.2j,2.+.4j,-.4+0j])
@pytest.mark.parametrize("realize",[identity,transpose_projection])
def test_value_slope_literal_poles_and_legacy_value(mesh,s,realize):
    b,p,c,db,dp,dc=data(mesh);face=NamedSharding(mesh,P(None,"x","y"))
    evaluate=realized_gamma_correlation_sampler(mesh,realize,route=("auto","auto"),representation="scalar-trs-even-s")
    value,slope=evaluate(jnp.asarray(s,jnp.complex128),db,dp,dc)
    expected,derivative=literal(b,p,s)
    if realize is transpose_projection:
        expected=.5*(expected+expected.swapaxes(-1,-2));derivative=.5*(derivative+derivative.swapaxes(-1,-2))
    np.testing.assert_allclose(value,expected,rtol=3e-13,atol=3e-13)
    np.testing.assert_allclose(slope,derivative,rtol=3e-13,atol=3e-13)
    legacy=_realized_gamma_body(mesh,realize,("auto","auto"))(jnp.asarray(s),db,dp,dc,jax.device_put(np.zeros((1,4,4),complex),face))
    np.testing.assert_array_equal(value,legacy)
    assert value.sharding==face and slope.sharding==face
    if s.imag:
        # Conjugating a causal scalar coefficient is an observable wrong map.
        wrong=expected.conj()
        assert np.max(abs(wrong-expected))>.1


def test_exact_slope_agrees_with_refined_finite_difference(mesh):
    _,_,_,b,p,c=data(mesh);s=.7+.2j
    evaluate=realized_gamma_correlation_sampler(mesh,identity,route=("auto","auto"),representation="scalar-trs-even-s")
    _,slope=evaluate(jnp.asarray(s),b,p,c);errors=[]
    for h in [1e-3,5e-4,1e-4]:
        plus,_=evaluate(jnp.asarray(s+h),b,p,c);minus,_=evaluate(jnp.asarray(s-h),b,p,c)
        errors.append(float(np.max(abs(np.asarray((plus-minus)/(2*h)-slope)))))
    assert errors[1]<.3*errors[0] and errors[2]<.05*errors[1]
    assert errors[-1]/np.max(abs(np.asarray(slope)))<1e-7


def test_equivalent_mesh_cache_and_changed_factor_operands(mesh):
    _,_,_,b,p,c=data(mesh)
    another=Mesh(np.asarray(mesh.devices),mesh.axis_names)
    one=realized_gamma_correlation_sampler(mesh,identity,route=("auto","auto"),representation="scalar-trs-even-s")
    two=realized_gamma_correlation_sampler(another,identity,route=("auto","auto"),representation="scalar-trs-even-s")
    assert one is two
    old=one(jnp.asarray(.7+.2j),b,p,c);new=two(jnp.asarray(.7+.2j),2*b,p,c)
    for a,z in zip(old,new):np.testing.assert_allclose(z,4*a,rtol=3e-13,atol=3e-13)


@pytest.mark.parametrize("representation",["scalar-ordered-ph","unknown",None])
def test_signed_z_models_never_enter_squared_frequency_sampler(mesh,representation):
    with pytest.raises(ValueError,match="ordered z models"):
        realized_gamma_correlation_sampler(mesh,identity,representation=representation)


def test_multi_parent_or_real_frequency_request_refuses(mesh):
    _,_,_,b,p,c=data(mesh)
    evaluate=realized_gamma_correlation_sampler(mesh,identity,route=("auto","auto"),representation="scalar-trs-even-s")
    with pytest.raises(ValueError,match="scalar complex"):
        evaluate(jnp.asarray(.7),b,p,c)
    with pytest.raises(ValueError,match="one Γ"):
        evaluate(jnp.asarray(.7+.2j),jnp.concatenate((b,b)),jnp.concatenate((p,p)),jnp.concatenate((c,c)))

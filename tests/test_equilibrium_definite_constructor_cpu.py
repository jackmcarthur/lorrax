"""Opt-in tiny EQ frequency reduction only; no production/material selection."""
import numpy as np
import pytest

from gw.shared_pole_constructor import reduce_general_pencil


def case(seed=77, redundant=False):
    rng=np.random.default_rng(seed)
    j=np.diag([1.,1.,-1.,-1.])
    a=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4))
    h=a.conj().T@a+np.eye(4)
    c=rng.normal(size=(3,4))+1j*rng.normal(size=(3,4))
    x=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4))
    if redundant:
        x=np.concatenate((x,x[:,:2]),axis=1)
    g=x.conj().T@j@x
    definite=x.conj().T@h@x
    o=c@x
    exact=lambda z,gamma:(c@np.linalg.solve((z+1j*gamma)*j-h,c.conj().T))
    return g,definite,o,o.conj().T@o,exact,j,h,c


def value(model,o,z):
    l=o@model['output_map'];r=o@model['input_map']
    return (l/(z-model['poles_ry']))@r.conj().T


def test_complex_signed_eq_and_current_held_H_congruence_nonzero_C1():
    g,h,o,n,exact,j,latent_h,c=case()
    gamma=.037
    kw=dict(max_order=4,rank_tolerance=1e-10,
            structure=dict(kind='equilibrium_definite',gamma_ry=gamma),output_gram=n)
    model=reduce_general_pencil(g,h,**kw)
    assert np.any(model['poles_ry'].real<0) and np.any(model['poles_ry'].real>0)
    np.testing.assert_array_equal(model['poles_ry'].imag,np.full(4,-gamma))
    np.testing.assert_array_equal(model['spaces']['left'],model['spaces']['right'])
    for z in [.21+.03j,-.4+.19j,1.3+.7j]:
        np.testing.assert_allclose(value(model,o,z),exact(z,gamma),rtol=4e-11,atol=4e-11)
    left=o@model['output_map'];right=o@model['input_map']
    c1=c@j@c.conj().T
    assert np.linalg.norm(c1)>.1
    np.testing.assert_allclose(left@right.conj().T,c1,rtol=3e-11,atol=3e-11)
    c2=c@j@latent_h@j@c.conj().T-1j*gamma*c1
    np.testing.assert_allclose((left*model['poles_ry'][None])@right.conj().T,c2,rtol=3e-11,atol=3e-11)
    # Change CURRENT definite member and output while retaining one common
    # frequency span. No current-H rank cut or old whitening may be reused.
    for shift in [.13,.29]:
        current_h=h+shift*np.eye(4)
        current=reduce_general_pencil(g,current_h,fixed_spaces=model['spaces'],**kw)
        for z in [.31+.17j,-.6+.3j]:
            target=o@np.linalg.solve((z+1j*gamma)*g-current_h,o.conj().T)
            np.testing.assert_allclose(value(current,o,z),target,rtol=5e-11,atol=5e-11)


def test_exact_H_dependencies_need_current_G_and_output_null_witnesses():
    g,h,o,n,exact,*_=case(redundant=True)
    model=reduce_general_pencil(g,h,max_order=4,rank_tolerance=1e-9,
        structure=dict(kind='equilibrium_definite',gamma_ry=.02),output_gram=n)
    assert model['receipt']['held_state_rank']==4
    np.testing.assert_allclose(value(model,o,.4+.2j),exact(.4+.2j,.02),rtol=1e-10,atol=1e-10)


@pytest.mark.parametrize('g,h,n,match',[
    (np.eye(2),np.diag([1.,-.1]),np.eye(2),'negative H'),
    (np.diag([1.,0.]),np.diag([1.,0.]),np.eye(2),'live H-null'),
    (np.eye(2),np.diag([1.,0.]),np.diag([1.,0.]),'live H-null'),
    (np.diag([1.,0.]),np.eye(2),np.eye(2),'constant feedthrough'),
])
def test_live_or_inconsistent_nulls_refuse_without_repairs(g,h,n,match):
    with pytest.raises(ValueError,match=match):
        reduce_general_pencil(g,h,max_order=2,structure=dict(kind='equilibrium_definite',gamma_ry=.03),output_gram=n)


def test_zero_mu_dark_output_is_not_a_pole_and_held_state_is_retained():
    model=reduce_general_pencil(np.diag([1.,0.]),np.eye(2),max_order=2,
        structure=dict(kind='equilibrium_definite',gamma_ry=.03),output_gram=np.diag([1.,0.]))
    assert len(model['poles_ry'])==1 and model['spaces']['right'].shape==(2,2)


def test_capacity_and_missing_gram_refuse_before_eigh(monkeypatch):
    import scipy.linalg as la
    monkeypatch.setattr(la,'eigh',lambda *a,**k:pytest.fail('unadmitted eigensolve'))
    kw=dict(max_order=2,structure=dict(kind='equilibrium_definite',gamma_ry=.03))
    with pytest.raises(ValueError,match='explicit output Gram'):
        reduce_general_pencil(np.eye(2),np.eye(2),**kw)
    with pytest.raises(ValueError,match='before copies/eigh'):
        reduce_general_pencil(np.eye(2),np.eye(2),output_gram=np.eye(2),max_workspace_bytes=1,**kw)


def test_general_default_arrays_unchanged_and_genuine_gain_not_projected():
    e=np.eye(3,dtype=complex);a=np.diag([-.8-.02j,.4+.17j,1.3-.03j])
    old=reduce_general_pencil(e,a,max_order=3)
    explicit=reduce_general_pencil(e,a,max_order=3,structure=None,output_gram=None)
    for key in ('poles_ry','output_map','input_map'):
        np.testing.assert_array_equal(np.ascontiguousarray(old[key]).view(np.uint64),
                                      np.ascontiguousarray(explicit[key]).view(np.uint64))
    assert max(old['poles_ry'].imag)>.1


def test_optin_uses_existing_canonical_rawbit_protocol(monkeypatch):
    import jax
    import gw.shared_pole_constructor as owner
    from tests.test_shared_pole_general_cpu import _fake_canonical_handoff
    state,select=_fake_canonical_handoff(monkeypatch)
    g,h,o,n,exact,*_=case()
    with jax.enable_x64(True):
        kw=dict(max_order=4,structure=dict(kind='equilibrium_definite',gamma_ry=.03),output_gram=n)
        select(0);root=owner.collective_general_pencil(g,h,**kw)
        select(1);worker=owner.collective_general_pencil(g,h,**kw)
    for key in ('poles_ry','output_map','input_map'):
        np.testing.assert_array_equal(root[key].view(np.uint64),worker[key].view(np.uint64))
    assert len(state['bits'])==5
    np.testing.assert_allclose(value(worker,o,.2+.3j),exact(.2+.3j,.03),rtol=3e-11,atol=3e-11)

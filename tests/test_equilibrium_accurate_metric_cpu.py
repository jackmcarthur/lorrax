"""Tiny host-only accurate metric contracts, NOT material/GPU qualification."""
import numpy as np
import pytest

from gw.shared_pole_constructor import (reduce_general_pencil,
    equilibrium_metric_workspace_bytes,general_pencil_workspace_bytes)


POLICY='original-columns-extended-newton-v1'


def fixture():
    rng=np.random.default_rng(634)
    a=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4))
    latent=a.conj().T@a+np.eye(4)
    j=np.diag([1.,-1.,1.,-1.])
    c=rng.normal(size=(3,4))+1j*rng.normal(size=(3,4))
    x=np.diag([1e-4,2e-2,3.,1e3]).astype(complex)
    g=x.conj().T@j@x;h=x.conj().T@latent@x;o=c@x
    structure=dict(kind='equilibrium_definite',gamma_ry=.031,
        state_selection='principal-rrqr-v1',metric_solver=POLICY,
        state_units=1/np.sqrt(h.diagonal().real),frequency_scale_ry=1.)
    return g,h,o,o.conj().T@o,structure


def transfer(model,o,z):
    return ((o@model['output_map'])/(z-model['poles_ry']))@(o@model['input_map']).conj().T


def test_all_selected_accurate_core_original_selector_and_current_held_H():
    g,h,o,n,structure=fixture()
    model=reduce_general_pencil(g,h,max_order=4,rank_tolerance=1e-10,
        structure=structure,output_gram=n)
    assert model['receipt']['metric_solver']==POLICY
    assert model['receipt']['fixed_congruence_count']==2
    assert model['receipt']['computed_scaled_H_min_lower']>0
    assert max(model['receipt']['original_H_extended_whitening_errors'])<1e-9
    p=model['spaces']['right']
    np.testing.assert_array_equal(p,model['spaces']['left'])
    np.testing.assert_array_equal(np.sum(abs(p),axis=0),np.ones(4))
    assert set(np.unique(p))<={0,1}
    for shift in (0.,.2):
        current=h+shift*np.diag(h.diagonal().real)
        m=model if shift==0 else reduce_general_pencil(g,current,max_order=4,
            rank_tolerance=1e-10,structure=structure,output_gram=n,fixed_spaces=model['spaces'])
        np.testing.assert_array_equal(m['spaces']['right'],p)
        for z in (.2+.3j,-.4+.17j):
            expected=o@np.linalg.solve((z+.031j)*g-current,o.conj().T)
            np.testing.assert_allclose(transfer(m,o,z),expected,rtol=2e-10,atol=2e-10)


def test_legacy_dense_held_gauge_is_not_reinterpreted_as_selector():
    g,h,o,n,structure=fixture()
    dense=np.eye(4,dtype=complex)*2
    with pytest.raises(ValueError,match='original-column P_I'):
        reduce_general_pencil(g,h,max_order=4,structure=structure,output_gram=n,
            fixed_spaces=dict(right=dense,left=dense))


def test_resolved_indefinite_H_still_refuses_with_full_rank_selector():
    structure=dict(kind='equilibrium_definite',gamma_ry=.01,state_selection='principal-rrqr-v1',
        metric_solver=POLICY,state_units=np.ones(2),frequency_scale_ry=1.)
    with pytest.raises(ValueError,match='negative H'):
        reduce_general_pencil(np.eye(2),np.diag([1.,-.1]),max_order=2,
            structure=structure,output_gram=np.eye(2))


def test_workspace_extended_envelope_precedes_copies_and_fits_batch_price(monkeypatch):
    import gw.shared_pole_constructor as owner
    for n in (2,10,2084):
        for m in (1,n):
            assert equilibrium_metric_workspace_bytes(n,m)<=general_pencil_workspace_bytes((n,n))
    monkeypatch.setattr(owner,'_equilibrium_extended_product',lambda *a:pytest.fail('unadmitted extended copy'))
    with pytest.raises(ValueError,match='before copies'):
        owner._equilibrium_accurate_metric(np.eye(2),np.eye(2),full_order=2,
            max_workspace_bytes=1,rounding=1e-13)


def test_accurate_policy_requires_principal_original_state_contract():
    with pytest.raises(ValueError,match='original-column principal'):
        reduce_general_pencil(np.eye(2),np.eye(2),max_order=2,output_gram=np.eye(2),
            structure=dict(kind='equilibrium_definite',gamma_ry=.01,metric_solver=POLICY))


def test_zero_mu_live_span_cannot_hide_behind_another_modes_units():
    # Raw v†Nv / ||v||² can hide the live third mode under the large, dark
    # second mode. Original-coordinate QR of the WHOLE null span cannot.
    with pytest.raises(ValueError,match='constant feedthrough'):
        reduce_general_pencil(np.diag([1.,0.,0.]),np.diag([1.,1e-3,1e3]),max_order=3,
            rank_tolerance=1e-10,output_gram=np.diag([1.,0.,1e-8]),
            structure=dict(kind='equilibrium_definite',gamma_ry=.01))


def test_original_null_span_is_unit_invariant_including_infinitesimal_scale():
    from gw.shared_pole_constructor import _equilibrium_original_null_span
    v=np.eye(3,dtype=complex)[:,1:]
    projector=_equilibrium_original_null_span(v)
    scaled=_equilibrium_original_null_span(v*np.array([1e-30,1e30])[None])
    np.testing.assert_allclose(projector@projector.conj().T,scaled@scaled.conj().T,atol=1e-15)


def test_nonfinite_metric_eigenpairs_fail_closed(monkeypatch):
    import scipy.linalg as la
    g,h,o,n,structure=fixture()
    monkeypatch.setattr(la,'eigh',lambda a,**k:(np.full(len(a),np.nan),np.eye(len(a))))
    with pytest.raises(ValueError,match='nonfinite metric eigenpairs'):
        reduce_general_pencil(g,h,max_order=4,structure=structure,output_gram=n,
            fixed_spaces=dict(right=np.eye(4),left=np.eye(4)))

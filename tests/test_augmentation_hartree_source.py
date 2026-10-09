"""Physical occupation and source/frame binding guards before source capture."""
from types import SimpleNamespace

import numpy as np
import pytest


def physical_fixture():
    occ = np.asarray([[1., .4, 0., 0.], [1., .6, 0., 0.]])
    parents = np.asarray([0, 1, 1, 0])
    wfn = SimpleNamespace(nbands=4, physical_density_band_stop=2,
        occupation_state_capacity=1., cell_volume=70.)
    wfn.physical_density_occupations = lambda *, k, unit_as_none: (
        occ[:, :2] if k == 'file' else occ[parents, :2])
    plan = SimpleNamespace(n_parent=2,n_full=4,nspinor=4,irr_idx=parents,
                           sym=SimpleNamespace(parent_k_domain='ibz'))
    request = dict(occupations=occ.copy(),full_kweights=np.full(4,.25),spin_degeneracy=1.)
    return request,wfn,plan


def test_fractional_physical_occupations_are_not_the_fitting_loss():
    from gw.isdf_augmentation import _hartree_source_request
    request,wfn,plan = physical_fixture()
    source = _hartree_source_request(request,wfn=wfn,plan=plan,public_range=(0,4))
    np.testing.assert_array_equal(source['occupations'],request['occupations'])
    np.testing.assert_array_equal(source['parent_kweights'],[.5,.5])
    assert source['electron_count'] == 1.5
    assert source['spin_degeneracy'] == 1.
    assert len(source['occupations_sha256']) == 64


def test_exact_insulator_none_uses_the_same_WFN_occupation_owner():
    from gw.isdf_augmentation import _hartree_source_request
    request,wfn,plan = physical_fixture()
    request['occupations'] = request['full_kweights'] = None
    wfn.physical_density_occupations = lambda **kw: None
    source = _hartree_source_request(request,wfn=wfn,plan=plan,public_range=(0,4))
    np.testing.assert_array_equal(source['occupations'],[[1,1,0,0],[1,1,0,0]])
    assert source['electron_count'] == 2.


def test_physical_frame_identity_ignores_transport_padding_but_detects_physical_change():
    from gw.isdf_augmentation import _physical_full_wfn_frame_binding
    rng=np.random.default_rng(812)
    physical=rng.normal(size=(2,3,3))+1j*rng.normal(size=(2,3,3))
    short=np.pad(physical,((0,0),(0,1),(0,1)))
    wide=np.pad(physical,((0,0),(0,5),(0,5)))
    # Ghost transport values are deliberately different; neither changes
    # the physical frame's payload, band domain or shape.
    wide[:,3:,:]=7.+3j
    wide[:,:,3:]=-2.+5j
    short_binding=_physical_full_wfn_frame_binding(short,3)
    wide_binding=_physical_full_wfn_frame_binding(wide,3)
    assert short_binding==wide_binding
    assert wide_binding['physical_frame_shape']==[2,3,3]
    assert wide_binding['physical_frame_band_domain']==[0,3]
    wide[0,1,2]+=1e-6j
    assert _physical_full_wfn_frame_binding(wide,3)!=short_binding


@pytest.mark.parametrize('change',('fit_loss','wrong_fraction','signed','nonfinite',
                                   'weights','outside_window','spin','wrong_unfold'))
def test_hartree_capture_refuses_changed_physical_source(change):
    from gw.isdf_augmentation import _hartree_source_request
    request,wfn,plan = physical_fixture()
    window=(0,4)
    if change == 'fit_loss': request['occupations'][:,0]=4.
    elif change == 'wrong_fraction': request['occupations'][0,1]=.5
    elif change == 'signed': request['occupations'][0,1]=-.4
    elif change == 'nonfinite': request['occupations'][0,1]=np.nan
    elif change == 'weights': request['full_kweights']=np.asarray([.1,.4,.4,.1])
    elif change == 'outside_window': window=(0,1)
    elif change == 'spin': request['spin_degeneracy']=True
    else: plan.irr_idx=np.asarray([0,0,1,1])
    with pytest.raises(ValueError,match='Hartree source'):
        _hartree_source_request(request,wfn=wfn,plan=plan,public_range=window)

"""Common vector coordinates versus independent Cartesian curl fields."""
import numpy as np


def test_common_vector_coordinates_preserve_smooth_neutral_cross():
    from isdf.augmentation_breit import (
        static_transverse_geometry,build_two_moment_compensation,
        static_transverse_block_factors,current_radial_moments,
        curl_poisson_field_tile,compensation_curl_field_tile)
    lm=np.asarray([(l,m) for l in range(4) for m in range(-l,l+1)])
    radius=(np.arange(1,11)/10.)**3
    geometry=static_transverse_geometry(radius,lm,support_radius=1.)
    compensation=build_two_moment_compensation(lm[:,0],support_radius=1.)
    blocks=static_transverse_block_factors(geometry,compensation)
    rng=np.random.default_rng(683191)
    random=lambda shape:rng.normal(size=shape)+1j*rng.normal(size=shape)
    delta=random((4,3,len(lm),len(radius)))
    smooth=random(delta.shape)
    moments=current_radial_moments(delta,geometry)
    radial=geometry['radial'];weight=radial['quadrature_weights_dr']*radial['quadrature_radius']**2
    whole=slice(0,len(weight))
    fd=curl_poisson_field_tile(delta,geometry,quadrature_slice=whole)
    fs=curl_poisson_field_tile(smooth,geometry,quadrature_slice=whole)
    fg=compensation_curl_field_tile(moments,compensation,geometry,quadrature_slice=whole)
    form=lambda a,b:np.einsum('mqpo,q,nqpo->mn',a.conj(),weight,b)
    expected=form(fd,fd)-form(fg,fg)+form(fs,fd-fg)+form(fd-fg,fs)
    U=blocks['basis']['transform']
    transform=lambda v:np.einsum('bk,mbr->mkr',U.conj(),v.reshape(4,-1,v.shape[-1]))
    d,s,g=map(transform,(delta,smooth,moments))
    actual=np.zeros_like(expected);self_form=np.zeros_like(expected)
    for sector in blocks['sectors']:
        assert sector['delta_factor'].shape[0]==sector['compensation_factor'].shape[0]
        for index in sector['indices_by_M']:
            project=lambda f,v:np.einsum('fr,mr->mf',f,v[:,index].reshape(4,-1))
            D=project(sector['delta_factor'],d)
            S=project(sector['delta_factor'],s)
            G=project(sector['compensation_factor'],g)
            inner=lambda a,b:a.conj()@b.T
            same=inner(D,D)-inner(G,G)
            self_form+=same
            actual+=same+inner(S,D-G)+inner(D-G,S)
    scale=max(float(np.max(abs(expected))),1.)
    assert np.max(abs(actual-expected))<3e-13*scale
    assert np.max(abs(expected-self_form))>1e-3
    assert np.max(abs(actual-actual.conj().T))<3e-13*scale
    assert blocks['retained_columns']==3*len(lm)*len(radius)
    assert blocks['retained_compensation_columns']==3*len(lm)*2
    assert blocks['no_cutoff']

"""Independent radial differentiation and occupied tail-bound checks."""
import numpy as np
import pytest


def test_positive_laplacian_map_matches_independent_radial_polynomial():
    from isdf.augmentation_breit import build_two_moment_compensation
    from isdf.current_tail import laplacian3_positive_map
    c=build_two_moment_compensation(np.arange(8),support_radius=2.55)
    t=laplacian3_positive_map(c);R=c['support_radius'];p=c['power']
    x,w=np.polynomial.legendre.leggauss(t['GL_order']);r=(x+1)*R/2;weight=w*R/2*r*r
    window=np.polynomial.polynomial.polypow(np.asarray([1.,0.,-1/R**2]),p)
    for row,l in enumerate(c['degrees']):
        columns=[]
        for d in range(2):
            poly=np.zeros(l+2+len(window))
            for j in range(2):
                poly[l+2*j:l+2*j+len(window)]+=c['profile_coefficients'][row,j,d]*window/R**(2*l+3+2*j)
            for _ in range(3):
                # f''+2f'/r-l(l+1)f/r^2, independently differentiated inr.
                second=np.polynomial.polynomial.polyder(poly,2)
                first=2*np.polynomial.polynomial.polyder(poly)[1:]
                centrifugal=-l*(l+1)*poly[2:]
                poly=np.pad(second,(0,max(0,len(first)-len(second))))+first+centrifugal
            columns.append(np.polynomial.polynomial.polyval(r,poly)*np.sqrt(weight))
        np.testing.assert_allclose(t['map'][row],columns,rtol=2e-8,atol=2e-6)
    changed=dict(c,power=5)
    with pytest.raises(ValueError,match='C5'):laplacian3_positive_map(changed)


def test_actual_covector_triangle_bound_contains_complex_direct_contraction():
    from isdf.current_tail import occupied_exchange_entry_bound
    rng=np.random.default_rng(9);F=rng.normal(size=(4,7,3,11))+1j*rng.normal(size=(4,7,3,11))
    g=rng.normal(size=(3,11,19))+1j*rng.normal(size=(3,11,19))
    R=np.sum(abs(g)**2,axis=-1);weights=np.asarray([.3,.2,.1,.4,.7,.2,.8])
    actual=np.einsum('atcm,cmg->atcg',F,g)
    matrix=np.einsum('atcg,t,btcg->ab',actual.conj(),weights,actual)*2*.017/320
    bound=occupied_exchange_entry_bound(F,R,weights,prefactor_Ry=-.017,cutoff_Ry=320)
    assert np.max(abs(matrix)-bound)<=0
    with pytest.raises(ValueError,match='nonnegative'):
        occupied_exchange_entry_bound(F,R,-weights,prefactor_Ry=-.017,cutoff_Ry=320)


def test_bounded_observer_matches_full_positive_map_and_tail_partial_tile():
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
    from isdf.augmentation_breit import build_two_moment_compensation
    from isdf.current_tail import laplacian3_positive_map,make_sobolev_norm_kernel,per_centroid_tail_norm_upper
    mesh=Mesh(np.asarray(jax.devices()[:1]).reshape(1,1),('x','y'));mu=140;na=2
    labels=np.asarray([(l,m) for l in range(3) for m in range(-l,l+1)])
    c=build_two_moment_compensation(labels[:,0],support_radius=2.55);t=laplacian3_positive_map(c)
    rng=np.random.default_rng(7);values=rng.normal(size=(2,3,mu,na,len(labels),2))+1j*rng.normal(size=(2,3,mu,na,len(labels),2))
    values[:,:,-3:]=0
    kernel,L,A=make_sobolev_norm_kernel(mesh,mu=mu,atoms=na,lm=labels,positive_map=t,fft_points=40,volume=30.)
    moment=tuple(jax.device_put(values[:,i].reshape(2,mu,-1),NamedSharding(mesh,P(('x','y'),None,None))) for i in range(3))
    actual=np.stack([np.asarray(v) for v in kernel(moment,jax.device_put(L,NamedSharding(mesh,P())),jax.device_put(A,NamedSharding(mesh,P())))],axis=1)
    expected=per_centroid_tail_norm_upper(values,labels,t,fft_points=40,volume=30.,cutoff_Ry=320)*320**6
    np.testing.assert_allclose(actual,expected,rtol=4e-15,atol=1e-8)
    assert not actual[:,:,-3:].any()

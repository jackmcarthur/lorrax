"""Independent periodic plane-wave limits of fitting-produced Hartree."""
from types import SimpleNamespace

import numpy as np
import pytest


def periodic_fixture():
    grid=(8,8,8);volume=1000.;blat=2*np.pi/10
    wfn=SimpleNamespace(fft_grid=grid,cell_volume=volume,bdot=blat**2*np.eye(3),
                        bvec=np.eye(3),blat=blat)
    x=np.arange(8)[:,None,None]
    charges=np.asarray([3.,5.]);amplitudes=np.asarray([.2,-.15])
    rho=np.stack([np.broadcast_to(q/volume*(1+a*np.cos(2*np.pi*x/8)),grid)
                  for q,a in zip(charges,amplitudes)])
    local=np.zeros((2,1,1,8),complex)
    options=dict(radius=(np.arange(1,9)/8)**3,weights_dr=np.ones(8)/8,
        lm=np.asarray([[0,0]]),centers_cart=np.zeros((1,3)),support_radius=1.,
        minimum_atom_image_distance=10.,electron_count=charges)
    boxes=np.zeros((1,2,2,*grid),complex)
    boxes[0,0,0]=1/np.sqrt(np.prod(grid))
    boxes[0,1,0]=np.broadcast_to(np.exp(2j*np.pi*x/8),grid)/np.sqrt(np.prod(grid))
    pair=np.zeros((1,2,2,1,1,8),complex);m0=np.zeros((1,2,2,1),complex)
    expected=np.zeros((2,1,2,2),complex)
    expected[:,0,0,1]=expected[:,0,1,0]=4*np.pi*charges*amplitudes/(volume*blat**2)
    return wfn,rho,local,options,boxes,pair,m0,expected


def test_unaugmented_periodic_cosine_has_analytic_complex_band_matrix():
    from isdf.atomic_hartree import prepare_charge_hartree,make_charge_hartree_tile
    wfn,rho,local,options,boxes,pair,m0,expected=periodic_fixture()
    operand=prepare_charge_hartree(wfn,rho,local,local,np.zeros((2,1)),**options)
    contract=make_charge_hartree_tile(operand)
    matrix,body,terms,mean,charge=map(np.asarray,contract(boxes,boxes,pair,pair,m0))
    np.testing.assert_allclose(matrix,expected,rtol=2e-13,atol=2e-14)
    np.testing.assert_array_equal(body,matrix)
    assert np.max(abs(terms))==0 and np.max(abs(mean))==0
    np.testing.assert_allclose(charge,np.eye(2)[None],atol=2e-14)
    np.testing.assert_allclose(operand['source_charge'],options['electron_count'],atol=2e-14)
    # Independent complex gauge and rectangular receiving-window controls.
    phases=np.exp(1j*np.asarray([.31,-.83]))
    phased=boxes*phases[None,:,None,None,None,None]
    actual=np.asarray(contract(phased,phased,pair,pair,m0)[0])
    rotated=expected*phases.conj()[None,None,:,None]*phases[None,None,None,:]
    np.testing.assert_allclose(actual,rotated,rtol=2e-13,atol=2e-14)
    rectangular=np.asarray(contract(phased[:,:1],phased,pair[:,:1],pair[:,:1],m0[:,:1])[0])
    np.testing.assert_allclose(rectangular,rotated[:,:,:1],rtol=2e-13,atol=2e-14)


@pytest.mark.parametrize('change',('charge','local_shape','overlap','complex_density','nonfinite',
                                  'complex_harmonic','complex_monopole','metric','complex_count'))
def test_occupation_trace_refuses_inconsistent_physical_source(change):
    from isdf.atomic_hartree import prepare_charge_hartree
    wfn,rho,local,options,_,_,_,_=periodic_fixture()
    m0=np.zeros((2,1))
    if change=='charge':options['electron_count']=np.asarray([36.,36.])
    elif change=='local_shape':local=local[:,:,:,:4]
    elif change=='overlap':options['minimum_atom_image_distance']=2.
    elif change=='complex_density':rho=rho.astype(complex)+1e-3j
    elif change=='nonfinite':rho[0,0,0,0]=np.nan
    elif change=='complex_harmonic':local[0,0,0,0]=1j
    elif change=='complex_monopole':m0=m0.astype(complex)+1j
    elif change=='metric':wfn.bdot=np.eye(3)
    else:options['electron_count']=np.asarray([3.,5.])+1j
    with pytest.raises(ValueError,match='Hartree'):
        prepare_charge_hartree(wfn,rho,local,local,m0,**options)


def test_hartree_operator_refuses_exchange_body_or_wrong_receiving_grid():
    from isdf.atomic_hartree import prepare_charge_hartree,make_charge_hartree_tile
    wfn,rho,local,options,boxes,pair,m0,_=periodic_fixture()
    operand=prepare_charge_hartree(wfn,rho,local,local,np.zeros((2,1)),**options)
    with pytest.raises(ValueError,match='own full-FFT'):
        make_charge_hartree_tile(dict(operand,operator='exchange_body80'))
    with pytest.raises(ValueError,match='receiving vertices'):
        make_charge_hartree_tile(operand)(boxes[...,:7],boxes[...,:7],pair,pair,m0)


def local_functional_fixture():
    """Independent complex linear probes with activated M0 and mean terms."""
    from isdf.atomic_coulomb import atomic_radial_metrics
    from isdf.atomic_hartree import prepare_charge_hartree
    wfn,rho,_,options,boxes,_,_,_=periodic_fixture()
    lm=np.asarray([[0,0],[1,-1],[1,0],[1,1]])
    options['lm']=lm
    rng=np.random.default_rng(78193)
    envelope=np.maximum(1-options['radius']**2,0.)**6
    def real_source(scale):
        a=(rng.normal(size=(2,1,4,8))+1j*rng.normal(size=(2,1,4,8)))*scale*envelope
        a[:,:,0]=a[:,:,0].real;a[:,:,2]=a[:,:,2].real
        a[:,:,1]=-a[:,:,3].conj()
        return a
    sp,sd=real_source(.005),real_source(.002)
    tables=atomic_radial_metrics(options['radius'],options['weights_dr'],lm[:,0],
        support_radius=1.,fft_points=1,cell_volume=1,interpolation_degree=5,quadrature_order=16)
    exact=np.einsum('uar,r->ua',sd[:,:,0],tables['moments'][0])+np.asarray([[2e-5],[-3e-5]])
    options['electron_count']=np.asarray([3.,5.])+np.sqrt(4*np.pi)*exact[:,0].real
    operand=prepare_charge_hartree(wfn,rho,sp,sd,exact,**options)
    tp=(rng.normal(size=(1,2,2,1,4,8))+1j*rng.normal(size=(1,2,2,1,4,8)))*.01
    td=(rng.normal(size=tp.shape)+1j*rng.normal(size=tp.shape))*.005
    target_m0=np.einsum('vijar,r->vija',td[:,:,:,:,0],tables['moments'][0])
    target_m0=target_m0+(.003+.002j)*rng.normal(size=target_m0.shape)
    raw=np.concatenate((td.reshape(4,-1),tp.reshape(4,-1),target_m0.reshape(4,-1)),axis=-1)
    raw=raw*float(wfn.cell_volume)/np.prod(wfn.fft_grid)
    return operand,boxes,tp,td,target_m0,raw


def test_linear_functional_retains_complex_adjoints_m0_and_both_means():
    from isdf.atomic_hartree import charge_hartree_functional,make_charge_hartree_tile
    operand,boxes,tp,td,m0,raw=local_functional_fixture()
    reference,_,local,mean,_=map(np.asarray,make_charge_hartree_tile(operand)(boxes,boxes,tp,td,m0))
    functional=charge_hartree_functional(operand)
    smooth=np.einsum('visxyz,uxyz,vjsxyz->uvij',boxes.conj(),np.asarray(functional['smooth_potential']),boxes)
    response=np.asarray(functional['local_response'])
    local_action=(raw@response.T).T.reshape(2,1,2,2)
    np.testing.assert_allclose(smooth+local_action,reference,rtol=2e-13,atol=2e-13)
    assert np.max(abs(local[-1]))>1e-7 and np.max(abs(mean))>1e-5
    wrong=(raw@response.conj().T).T.reshape(2,1,2,2)
    assert np.max(abs(wrong-local_action))>1e-4
    with pytest.raises(ValueError,match='real occupied-source neutral mean'):
        charge_hartree_functional(dict(operand,source_phi=operand['source_phi']+.001j))


def check_local_functional_rhs(mesh):
    from isdf.atomic_hartree import charge_hartree_functional,make_charge_hartree_rhs_contractor
    from common.collectives import device_put_process_local,gather_to_host
    from jax.sharding import NamedSharding,PartitionSpec as P
    operand,_,_,_,_,raw=local_functional_fixture()
    functional=charge_hartree_functional(operand)
    qpad=int(np.prod(list(mesh.shape.values())))
    slot=qpad-1
    rhs=np.full((qpad,4,raw.shape[-1]),np.nan+1j*np.nan)
    rhs[slot]=raw
    placed=device_put_process_local(rhs,NamedSharding(mesh,P(('x','y'),None,None)))
    result=np.asarray(gather_to_host(make_charge_hartree_rhs_contractor(
        functional,mesh,q0_slot=slot)(placed)))
    expected=raw@np.asarray(functional['local_response']).T
    np.testing.assert_allclose(result[slot],expected,rtol=2e-13,atol=2e-13)
    assert np.isfinite(result).all() and np.max(abs(np.delete(result,slot,axis=0)),initial=0.)==0
    return dict(max_local_functional_error_Ry=float(np.max(abs(result[slot]-expected))),
        nonGamma_NaN_inert=True,q0_owner_slot=slot,local_to_grid=functional['local_to_grid'],
        scope='Complex local linear functional and q-owner arithmetic, not actual ISDF rank accuracy')


def test_local_functional_uses_exact_q_owner_and_no_receiving_conjugation():
    from tests.test_hartree_point_trace import cpu_mesh
    check_local_functional_rhs(cpu_mesh())

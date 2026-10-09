"""Occupation arithmetic and ownership controls; no atomic-WFN physics claim."""
from types import SimpleNamespace

import numpy as np
import pytest


def trace_fixture(mesh):
    from gw.centroid_k_unfold import CentroidKUnfoldPlan,OperationClasses
    from common.collectives import device_put_process_local
    from jax.sharding import NamedSharding,PartitionSpec as P

    npoint=12;shard=npoint//int(mesh.shape['y'])
    active=np.tile([True,True,True,True,False,False],2)
    perm=np.stack((np.arange(npoint),np.tile([3,2,1,0,4,5],2)+np.repeat([0,6],6)))
    irr=np.asarray([0,1,1,0]);cls=np.asarray([0,0,1,1])
    counts=np.zeros((2,3));np.add.at(counts,(cls,irr),1.)
    classes=OperationClasses(np.asarray([0,1]),np.asarray([False,True]),counts,cls,
                            perm%shard,np.asarray([0,1,0]))
    spin=np.broadcast_to(np.eye(4,dtype=complex),(4,4,4)).copy()
    spin[2:]=np.diag([1j,-1j,-1j,1j])
    plan=SimpleNamespace(n_parent=3,n_full=4,nspinor=4,fft_grid=(4,4,4),
        mesh_xy=mesh,layout=SimpleNamespace(axis=SimpleNamespace(active_mask=active)),
        spin_action_full=spin,sym_perm=perm,irr_idx=irr,n_centroid_packed=npoint,
        operation_classes=lambda:classes,transport_classes=CentroidKUnfoldPlan.transport_classes)
    occ=np.asarray([[1.,.5,.25,0,0,0,0,0],[.7,.3,.1,0,0,0,0,0],np.ones(8)])
    weights=np.asarray([.1,.2,.3,.4])
    rng=np.random.default_rng(17921)
    clean=(rng.normal(size=(3,8,4,npoint))+1j*rng.normal(size=(3,8,4,npoint)))*.1
    expected=np.zeros(npoint)
    # Literal full-child spin, antiunitary and point transformation.
    for k,parent in enumerate(irr):
        child=np.take(clean[parent],perm[cls[k]],axis=-1)
        if classes.antiunitary[cls[k]]:child=child.conj()
        child=np.einsum('st,ntm->nsm',spin[k],child)
        child*=np.exp(1j*(.31+k)*np.arange(npoint))[None,None]
        expected+=weights[k]*np.einsum('nsm,n->m',abs(child)**2,occ[parent])
    expected=np.where(active,expected*64/100,0.)
    poisoned=clean.copy()
    poisoned[:2,3:]=np.nan+1j*np.nan
    poisoned[:,:, :,~active]=np.inf+1j*np.inf
    poisoned[2]=1e308+1e308j
    face=device_put_process_local(poisoned,NamedSharding(mesh,P(None,'x',None,'y')))
    return plan,occ,weights,face,expected


def check_point_trace(mesh):
    from isdf.atomic_hartree import make_occupied_point_trace
    from common.collectives import gather_to_host

    plan,occ,weights,face,expected=trace_fixture(mesh)
    trace=make_occupied_point_trace(plan,occ,weights,cell_volume=100.,spin_degeneracy=1.)
    actual=np.asarray(gather_to_host(trace(face)))
    error=float(np.max(abs(actual-expected)))
    np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=2e-14)
    assert np.isfinite(actual).all() and np.max(abs(actual[~plan.layout.axis.active_mask]))==0
    return dict(max_physical_density_error=error,source_band_psum_axis='x',point_axis='y',
        nonuniform_full_kweights=True,scalar_spatial_transport=True,
        spin_and_Bloch_norm_invariance=True,zero_occupation_NaN_inert=True,
        unused_parent_overflow_inert=True,inactive_point_infinity_inert=True,
        scope='Adversarial occupation/ownership arithmetic; no actual atomic wavefunction accuracy claim')


def cpu_mesh():
    import jax
    from jax.sharding import Mesh
    return Mesh(np.asarray(jax.devices('cpu')[:1]).reshape(1,1),('x','y'))


def test_physical_point_trace_matches_literal_full_star():
    check_point_trace(cpu_mesh())


@pytest.mark.parametrize('change',('loss_weight','kweights','double_spin','spin','ghost'))
def test_trace_refuses_nonphysical_occupation_or_transport(change):
    from isdf.atomic_hartree import make_occupied_point_trace
    plan,occ,weights,_,_=trace_fixture(cpu_mesh());fspin=1.
    if change=='loss_weight':occ[0,0]=4.
    elif change=='kweights':weights*=2
    elif change=='double_spin':fspin=2.
    elif change=='spin':plan.spin_action_full[0,0,0]=1.01
    else:plan.sym_perm[1,0]=4
    with pytest.raises(ValueError,match='Hartree'):
        make_occupied_point_trace(plan,occ,weights,cell_volume=100.,spin_degeneracy=fspin)


def check_source_projection(mesh):
    from isdf.atomic_hartree import make_occupied_density_projection
    from common.collectives import device_put_process_local,gather_to_host
    from jax.sharding import NamedSharding,PartitionSpec as P
    plan,_,_,_,physical=trace_fixture(mesh)
    active=plan.layout.axis.active_mask
    fields=np.stack((physical,.3*physical))
    rng=np.random.default_rng(19774)
    weights=(rng.normal(size=(2,3,12))+1j*rng.normal(size=(2,3,12)))*active[None,None]
    expected=np.einsum('sp,hfp->sfh',fields,weights)
    py=int(mesh.shape['y']);shard=12//py
    indices=np.broadcast_to(np.arange(shard),(py,3,shard)).copy()
    buckets=weights.reshape(2,3,py,shard).transpose(2,0,1,3)
    poisoned=np.where(active[None],fields,np.inf)
    density=device_put_process_local(poisoned,NamedSharding(mesh,P(None,'y')))
    project=make_occupied_density_projection(mesh,indices,buckets,output_shape=(3,2))
    result=np.asarray(gather_to_host(project(density)))
    np.testing.assert_allclose(result,expected,rtol=2e-13,atol=2e-14)
    assert np.isfinite(result).all()
    return dict(max_source_projection_error=float(np.max(abs(result-expected))),
        zero_weight_ghost_infinity_inert=True,physical_fields=2,
        reduced_point_axis='y',no_second_band_axis_reduction=True,
        scope='Canonical bucket projection arithmetic, not actual atomic density accuracy')


def test_scalar_source_projection_preserves_canonical_rows_and_no_extra_x_sum():
    check_source_projection(cpu_mesh())


@pytest.mark.parametrize('change',('complex_weights_nan','shape','bounds','index_type'))
def test_source_projection_refuses_inconsistent_bucket_tables(change):
    from isdf.atomic_hartree import make_occupied_density_projection
    from common.collectives import device_put_process_local
    from jax.sharding import NamedSharding,PartitionSpec as P
    mesh=cpu_mesh();indices=np.zeros((1,2,1),dtype=int)
    weights=np.ones((1,3,2,1),complex);shape=(2,3)
    if change=='complex_weights_nan':weights[0,0,0,0]=np.nan
    elif change=='shape':shape=(3,3)
    elif change=='bounds':indices[0,0,0]=12
    else:indices=indices.astype(float)
    with pytest.raises(ValueError,match='Hartree'):
        project=make_occupied_density_projection(mesh,indices,weights,output_shape=shape)
        density=device_put_process_local(np.ones((2,12)),NamedSharding(mesh,P(None,'y')))
        project(density)

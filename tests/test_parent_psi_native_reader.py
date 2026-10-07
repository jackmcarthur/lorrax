"""Independent native coefficient-input/transport guards; P1 unit scope."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P

from common.psi_G_store import (load_parent_psi_G,native_parent_reader_counts,
                               validate_native_parent_band_tile)
from common.wfn_layout import band_sphere_spec


@pytest.fixture(scope='module')
def mesh():
    return Mesh(np.asarray(jax.devices()[:1]).reshape(1,1),('x','y'))


def plant(mesh, *, g_counts=(3,2), band_range=(3,6), pad=4):
    g_counts=np.asarray(g_counts,np.int64);counts=2*g_counts
    indices=band_range[0]+np.arange(pad)
    valid=(indices[None]<counts[:,None])&(indices[None]<band_range[1])
    value=np.zeros((2,pad,2,3),np.complex128)
    for k,b,s,g in np.ndindex(value.shape):
        if valid[k,b] and g<g_counts[k]:value[k,b,s,g]=(1+k+2*b+3*s+g)*(.1+.03j)
    psi=jax.device_put(value,NamedSharding(mesh,band_sphere_spec()))
    kwargs=dict(mesh_xy=mesh,n_parent=2,band_range=band_range,pad_to=pad,
                nspinor=2,ngkmax=3,physical_band_counts=counts,physical_g_counts=g_counts)
    return psi,valid,kwargs


@pytest.mark.parametrize('counts',[[True,True],[3.,4.],[0,4],[-1,4],[[3],[4]],[3],
                                  np.asarray([2**63,4],np.uint64)])
def test_native_counts_refuse_nonphysical_metadata(counts):
    with pytest.raises(ValueError,match='native_parent_reader'):
        native_parent_reader_counts(counts,2)


def test_native_counts_preserve_exact_integer_values():
    np.testing.assert_array_equal(native_parent_reader_counts(np.asarray([6,4],np.int32),2),[6,4])
    with pytest.raises(ValueError,match='integer'):native_parent_reader_counts([6],True)


def test_ragged_terminal_and_requested_pad_are_distinct_exact_zero_masks(mesh):
    psi,valid,k=plant(mesh)
    np.testing.assert_array_equal(valid,[[True,True,True,False],[True,False,False,False]])
    assert validate_native_parent_band_tile(psi,valid,**k) is psi
    # A suffix of the requested range is invalid even at the larger parent.
    bad=valid.copy();bad[0,3]=True
    with pytest.raises(ValueError,match='mask differs'):
        validate_native_parent_band_tile(psi,bad,**k)


@pytest.mark.parametrize('slot',[(0,3,0,0),(1,1,0,0),(1,0,0,2)])
def test_independent_requested_native_and_G_ghosts_refuse_nonzero(mesh,slot):
    psi,valid,k=plant(mesh);a=np.asarray(psi).copy();a[slot]=1e-100+2e-100j
    bad=jax.device_put(a,psi.sharding)
    with pytest.raises(ValueError,match='tail is nonzero'):
        validate_native_parent_band_tile(bad,valid,**k)


@pytest.mark.parametrize('value',[np.nan,np.inf])
def test_nonfinite_physical_coefficients_refused(mesh,value):
    psi,valid,k=plant(mesh);a=np.asarray(psi).copy();a[0,0,0,0]=value
    with pytest.raises(ValueError,match='nonfinite'):
        validate_native_parent_band_tile(jax.device_put(a,psi.sharding),valid,**k)


def test_coefficient_dtype_layout_and_mask_dtype_guards(mesh):
    psi,valid,k=plant(mesh)
    for bad in (psi.astype(jnp.complex64),jax.device_put(np.asarray(psi),NamedSharding(mesh,P())),psi[:,:,:,:2]):
        with pytest.raises(ValueError,match='shape/layout'):
            validate_native_parent_band_tile(bad,valid,**k)
    with pytest.raises(ValueError,match='boolean native mask'):
        validate_native_parent_band_tile(psi,valid.astype(np.int64),**k)
    with pytest.raises(ValueError,match='spinor times'):
        validate_native_parent_band_tile(psi,valid,**{**k,'physical_band_counts':[6,5]})


def test_changing_runtime_masks_do_not_reuse_stale_literal(mesh):
    psi,valid,k=plant(mesh)
    validate_native_parent_band_tile(psi,valid,**k)
    newer=np.asarray(psi).copy();newer[0,2]=0
    changed=valid.copy();changed[0,2]=False
    args={**k,'band_range':(3,5)}
    value=jax.device_put(newer,psi.sharding)
    validate_native_parent_band_tile(value,changed,**args)
    with pytest.raises(ValueError,match='mask differs'):
        validate_native_parent_band_tile(value,valid,**args)


class Loader:
    ngkmax=3;nkpts=2;nspinor=2
    def __init__(self,g_counts):self.counts=np.asarray(g_counts,np.int64)
    def ngk_valid(self,*,k):return self.counts
    def kvecs(self,*,k):return np.asarray([[0.,0.,0.],[.25,-.125,0.]])
    def box_index(self,*,k):
        return np.asarray([[0,9,18],[0,3,27]],np.int32) if self.counts[1]==2 else np.asarray([[0,9,18],[0,3,6]],np.int32)


def reader(mesh,counts):
    calls=[]
    def read(*,band_range,pad_to,k_domain):
        calls.append((band_range,pad_to,k_domain))
        lo,hi=band_range;valid=(lo+np.arange(pad_to)[None]<2*np.asarray(counts)[:,None])&(lo+np.arange(pad_to)[None]<hi)
        a=np.zeros((2,pad_to,2,3),np.complex128)
        for k,b,s,g in np.ndindex(a.shape):
            if valid[k,b] and g<counts[k]:a[k,b,s,g]=(1+k+lo+b+2*s+g)*(.07+.13j)
        return jax.device_put(a,NamedSharding(mesh,band_sphere_spec())),valid
    return read,calls


@pytest.mark.parametrize('ragged',[False,True])
def test_original_Gslot_and_centroid_owner_with_independent_dense_DFT(mesh,monkeypatch,ragged):
    from common import wfn_transforms
    counts=(3,2) if ragged else (3,3);wfn=Loader(counts);read,calls=reader(mesh,counts)
    meta=SimpleNamespace(fft_grid=(3,3,3),nspinor=2,nk_tot=2,b_id_4_user=6,memory_per_device_gb=1.)
    points=np.asarray([[0,0,0],[1,2,0],[2,1,1]],np.int32)
    kwargs=dict(wfn=wfn,mesh_xy=mesh,meta=meta,band_range=(0,6),band_chunk=4,
                centroid_indices=points,print_fn=lambda _:None)
    native=load_parent_psi_G(**kwargs,band_reader=read,native_parent_band_counts=2*np.asarray(counts))
    assert calls==[((0,4),4,'ibz'),((4,6),2,'ibz')]
    full,_=read(band_range=(0,6),pad_to=6,k_domain='ibz');a=np.asarray(full)
    np.testing.assert_array_equal(np.asarray(native.psi_G),a)
    g=np.asarray([[[0,0,0],[1,0,0],[-1,0,0]],[[0,0,0],[0,1,0],[0,-1,0]]])
    phase=np.exp(2j*np.pi*np.einsum('kgd,md->kgm',(g+wfn.kvecs(k='ibz')[:,None])/3.,points))/np.sqrt(27.)
    expected=np.einsum('kbsg,kgm->kbsm',a,phase)
    np.testing.assert_allclose(np.asarray(native.faces[0]),expected,rtol=3e-13,atol=3e-13)
    np.testing.assert_allclose(np.asarray(native.faces[1]),expected.conj().transpose(0,3,1,2),rtol=3e-13,atol=3e-13)
    if not ragged:
        def ordinary(loader,band_range,**kw):
            return read(band_range=band_range,pad_to=kw['pad_to'],k_domain=kw['k'])[0]
        monkeypatch.setattr(wfn_transforms,'load_psi_gflat_padded',ordinary)
        original=load_parent_psi_G(**kwargs)
        np.testing.assert_array_equal(np.asarray(original.psi_G),np.asarray(native.psi_G))
        for a,b in zip(original.faces,native.faces):np.testing.assert_array_equal(np.asarray(a),np.asarray(b))


def test_callback_census_required_and_bispinor_refused_before_read(mesh):
    wfn=Loader((3,2));meta=SimpleNamespace(fft_grid=(3,3,3),nspinor=2,nk_tot=2)
    args=dict(wfn=wfn,mesh_xy=mesh,meta=meta,band_range=(0,6),band_chunk=4,print_fn=lambda _:None)
    with pytest.raises(ValueError,match='counts require'):
        load_parent_psi_G(**args,native_parent_band_counts=[6,4])
    with pytest.raises(ValueError,match='callable ordinary'):
        load_parent_psi_G(**args,band_reader=lambda **kw:None,native_parent_band_counts=[6,4],bispinor=True)


def test_odd_native_extent_preserves_all_P_face_carrier_and_zero_tails():
    """A genuine 2x2 addressable mesh exercises the incumbent face finish."""
    if len(jax.devices()) < 4 or jax.process_count() != 1:
        pytest.skip('requires one process with four CPU devices')
    mesh4=Mesh(np.asarray(jax.devices()[:4]).reshape(2,2),('x','y'))

    class ScalarLoader:
        ngkmax=5;nkpts=1;nspinor=1
        def ngk_valid(self,*,k):return np.asarray([5],np.int64)
        def kvecs(self,*,k):return np.zeros((1,3),np.float64)
        def box_index(self,*,k):return np.asarray([[0,16,48,4,12]],np.int32)

    coefficients=np.zeros((1,8,1,5),np.complex128)
    for b,g in np.ndindex(5,5):
        coefficients[0,b,0,g]=(1+2*b+g)*(.07+.11j)
    valid=np.arange(8)[None]<5
    calls=[]
    def read(*,band_range,pad_to,k_domain):
        calls.append((band_range,pad_to,k_domain))
        assert band_range==(0,5) and pad_to==8 and k_domain=='ibz'
        return jax.device_put(coefficients,NamedSharding(mesh4,band_sphere_spec())),valid
    points=np.asarray([[0,0,0],[1,2,0],[2,1,1],[3,0,2]],np.int32)
    meta=SimpleNamespace(fft_grid=(4,4,4),nspinor=1,nk_tot=1,
                         b_id_4_user=5,memory_per_device_gb=1.)
    result=load_parent_psi_G(wfn=ScalarLoader(),mesh_xy=mesh4,meta=meta,
        band_range=(0,5),band_chunk=8,centroid_indices=points,
        band_reader=read,native_parent_band_counts=np.asarray([5]),
        print_fn=lambda _:None)
    assert calls==[((0,5),8,'ibz')]
    assert result.psi_G.shape==(1,8,1,8)
    expected_G=np.pad(coefficients,((0,0),(0,0),(0,0),(0,3)))
    np.testing.assert_array_equal(np.asarray(result.psi_G),expected_G)
    g=np.asarray([[0,0,0],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0]])
    phase=np.exp(2j*np.pi*np.einsum('gd,md->gm',g/4.,points))/8.
    expected=np.einsum('kbsg,gm->kbsm',coefficients,phase)
    # Each persistent face shards bands over one size-2 mesh axis; the
    # Gslot store shards over the product size4. These carriers differ.
    assert result.faces[0].shape==(1,6,1,4)
    assert result.faces[1].shape==(1,4,6,1)
    np.testing.assert_allclose(np.asarray(result.faces[0]),expected[:,:6],rtol=3e-13,atol=3e-13)
    np.testing.assert_allclose(np.asarray(result.faces[1]),expected[:,:6].conj().transpose(0,3,1,2),rtol=3e-13,atol=3e-13)
    for face in result.faces:
        assert len(face.sharding.device_set)==4
        assert len(face.addressable_shards)==4
    np.testing.assert_array_equal(np.asarray(result.faces[0])[:,5:],0.)
    np.testing.assert_array_equal(np.asarray(result.faces[1])[:,:,5:],0.)

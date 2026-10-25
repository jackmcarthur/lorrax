"""Selected SlabIO windows and device optical algebra against literal sums."""
from pathlib import Path
from unittest.mock import patch

import runtime
from runtime.source_closure import ensure_source_closure
ensure_source_closure()
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from file_io.restart_bundle import load_dipole_h5, read_dipole_cv_block
from file_io.slab_io import SlabIO
from file_io.commit_state import set_commit_state

jax.config.update("jax_enable_x64", True)


def write_dipole_fixture(path, *, legacy=False):
    """A valid full file; selected carrier ghosts lie inside other bands."""
    nk, nb = 2, 23
    rng = np.random.default_rng(714)
    a = rng.normal(size=(3,nk,nb,nb)) + 1j*rng.normal(size=(3,nk,nb,nb))
    v = a + a.swapaxes(-1,-2).conj()
    energies = np.stack([np.linspace(-4,5,nb) + .03*k for k in range(nk)])
    with h5py.File(path,"w") as f:
        f.attrs.update(format_version=2, band_min=0, band_max=nb, nbands=nb, nk=nk, basis="dft", prov_wfn_sha256="fixture")
        f["dipole_cart"] = v
        if legacy:
            f["deltaE"] = energies[:,:,None]-energies[:,None,:]
        else:
            f["band_energies"] = energies
        set_commit_state(f, True)
    return v, energies


def mesh_for(side):
    if jax.device_count() < side*side:
        pytest.skip("requires nine virtual CPU devices")
    return Mesh(np.array(jax.devices()[:side*side]).reshape(side,side),("x","y"))


def no_global_host(a,*args,**kwargs):
    if isinstance(a,jax.Array) and isinstance(a.sharding,NamedSharding) and a.ndim>=3:
        raise AssertionError("optical field converted to a global host tensor")
    return _ASARRAY(a,*args,**kwargs)


_ASARRAY = np.asarray


@pytest.mark.parametrize("side,legacy",[(2,False),(3,False),(3,True)])
def test_real_window_reader_padding_and_optical_outputs(tmp_path,side,legacy):
    from bse import absorption_common as AC
    from bse.absorption_common import (build_dipole_vector_bse, _transition_spectrum,
        project_dipoles, write_bgw_eigenvalues_noeh, bgw_noeh_lines)
    from bse.head_resolvent import build_head_dipole, build_head_rhs
    from bse.bse_ring_comm import make_bse_shardings
    from common.chi_from_dipole import compute_S_omega_cv
    mesh = mesh_for(side)
    path = tmp_path/"dipole.h5"
    v,e = write_dipole_fixture(path,legacy=legacy)
    wfn = tmp_path/"WFN.h5"
    with h5py.File(wfn,"w"):
        pass
    rows,cols=(11,16),(4,11)
    rp,cp=-(-5//side)*side,-(-7//side)*side
    rep=NamedSharding(mesh,P())
    data=dict(psi_c_X=jax.device_put(np.zeros((2,rp,1,side*side),complex),
        NamedSharding(mesh,P(None,None,None,("x","y")))),
        n_cond=5,n_val=7,n_cond_pad=rp,n_val_pad=cp,nkx=2,nky=1,nkz=1,
        eps_c=jax.device_put(np.pad(e[:,11:16],((0,0),(0,rp-5)),constant_values=1000),rep),
        eps_v=jax.device_put(np.pad(e[:,4:11],((0,0),(0,cp-7)),constant_values=-1000),rep))
    seen=[]
    real_read=SlabIO.read_slab
    def read(self,name,**kw):
        seen.append((name,kw.copy()))
        return real_read(self,name,**kw)
    with patch.object(SlabIO,"read_slab",read),patch.object(np,"asarray",side_effect=no_global_host):
        velocity,delta,attrs=load_dipole_h5(path,wfn_path=wfn,mesh_xy=mesh,
            band_rows=rows,band_cols=cols)
        with patch.object(AC,"_deck_wfn_fingerprint",return_value="fixture"):
            d,public_delta=AC.window_dipoles(data,path,n_occ=11,wfn_path=wfn)
        operator_delta=AC.window_transition_energies(data)
        ec,ev=AC.window_band_energies(data)
        pair=build_head_dipole(velocity,delta,n_cond_pad=rp,n_val_pad=cp)
        rhs=build_head_rhs(pair,make_bse_shardings(mesh),physical_branches=True)
        directions=np.array([[1.,0.,0.],[0.,.6,.8]])
        pol=project_dipoles(d,directions)
        omega=np.linspace(.1,4.,9)
        spectra={kind:_transition_spectrum(pol,delta,omega,.13,kind)
                 for kind in ("lorentzian","gaussian")}
        text=write_bgw_eigenvalues_noeh(tmp_path,e[:,11:16],e[:,4:11],d,
            n_spin=1,n_spinor=2,vol_supercell=28.,directions=directions,labels=("a","b"))
    assert attrs["band_rows"]==rows and attrs["band_cols"]==cols
    request=next(kw for name,kw in seen if name=="dipole_cart")
    assert request["partition_spec"]==P(None,None,"x","y")
    assert request["shape"]==(3,2,rp,cp) and request["valid_shape"]==(3,2,5,7)
    de=e[:,11:16,None]-e[:,None,4:11]
    dv=v[:,:,11:16,4:11]/de[None]
    vp=np.pad(v[:,:,11:16,4:11],((0,0),(0,0),(0,rp-5),(0,cp-7)))
    np.testing.assert_array_equal(np.asarray(velocity),vp)
    np.testing.assert_allclose(np.asarray(delta)[:,:5,:7],de,rtol=0,atol=0)
    assert np.isinf(np.asarray(delta)[:,5:]).all() and np.isinf(np.asarray(delta)[:,:,7:]).all()
    np.testing.assert_array_equal(np.asarray(public_delta),np.asarray(delta))
    np.testing.assert_array_equal(np.asarray(operator_delta),np.asarray(delta))
    np.testing.assert_array_equal(ec,e[:,11:16])
    np.testing.assert_array_equal(ev,e[:,4:11])
    expected=np.pad(dv.transpose(0,2,3,1),((0,0),(0,rp-5),(0,cp-7),(0,0)))
    np.testing.assert_allclose(np.asarray(pair),expected,rtol=3e-15,atol=3e-15)
    np.testing.assert_allclose(np.asarray(rhs),np.stack([expected,expected.conj()]),rtol=3e-15,atol=3e-15)
    assert not velocity.sharding.is_fully_replicated and not pair.sharding.is_fully_replicated
    assert not pol.sharding.is_fully_replicated
    assert max(x.data.nbytes for x in pol.addressable_shards)==pol.nbytes//(side*side)
    assert max(x.data.nbytes for x in velocity.addressable_shards)==velocity.nbytes//(side*side)
    dp=np.einsum("pa,akcv->pkcv",directions,dv)
    for kind,got in spectra.items():
        distance=omega[:,None,None,None]-de[None]
        kernel=(.13/(np.pi*(distance**2+.13**2)) if kind=="lorentzian" else
                np.exp(-.5*(distance/.13)**2)/(np.sqrt(2*np.pi)*.13))
        want=np.column_stack([np.einsum("wkcv,pkcv->wp",kernel,np.abs(dp)**2),kernel.sum(axis=(1,2,3))])
        np.testing.assert_allclose(got,want,rtol=3e-14,atol=3e-14)
    for file,dpol in zip(text,dp):
        expected_text="\n".join(bgw_noeh_lines(e[:,11:16],e[:,4:11],dpol,
            n_spin=1,n_spinor=2,vol_supercell=28.))+"\n"
        assert Path(file).read_text()==expected_text
    with patch.object(np,"asarray",side_effect=no_global_host):
        all_v,all_de=read_dipole_cv_block(path,nelec=11,mesh=mesh)
        tensor=compute_S_omega_cv(all_v,all_de,14.,2,1,2,jnp.array([.2+.3j]))
    de_all=e[:,11:,None]-e[:,None,:11]
    weight=4/(14*2*1*2)/(de_all*((.2+.3j)**2-de_all**2))
    reference=np.einsum("akcv,kcv,bkcv->ab",v[:,:,11:,:11].conj(),weight,v[:,:,11:,:11])
    np.testing.assert_allclose(np.asarray(tensor)[0],reference,rtol=3e-14,atol=3e-14)


@pytest.mark.parametrize("side",[2,3])
def test_qp_full_active_square_includes_same_class_commutator(tmp_path,side):
    from bse.bse_qp_rotation import qp_window_dipoles
    mesh=mesh_for(side)
    path=tmp_path/"dipole.h5"
    v,e=write_dipole_fixture(path)
    q0,q1,nocc=2,9,4
    n=7
    rng=np.random.default_rng(602+side)
    u=np.stack([np.linalg.qr(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))[0] for _ in range(2)])
    eqp=np.tile(np.array([-2.,-1.,.6,1.2,2.,2.7,3.]),(2,1))
    qp=dict(band_range=(q0,q1),n_occ=nocc,n_val=2,n_cond=3,
        U_val=u[:,:,:2],U_cond=u[:,:,2:5],eps_v=eqp[:,:2],eps_c=eqp[:,2:5])
    shape=(-(-3//side)*side,-(-2//side)*side)
    with patch.object(np,"asarray",side_effect=no_global_host):
        velocity,de,_=load_dipole_h5(path,wfn_path=None,mesh_xy=mesh,
            band_rows=(q0,q1),band_cols=(q0,q1))
        got,delta=qp_window_dipoles(velocity,de,qp,mesh_xy=mesh,output_shape=shape)
    active=v[:,:,q0:q1,q0:q1]
    energies=e[:,q0:q1]
    cross=(np.arange(n)[:,None]<2)!=(np.arange(n)[None,:]<2)
    denom=energies[:,:,None]-energies[:,None,:]
    w=np.zeros_like(active)
    np.divide(active,denom[None],out=w,where=cross[None,None])
    dh=np.einsum("kij,kj,klj->kil",u,eqp,u.conj())-np.array([np.diag(row) for row in energies])
    corrected=active+dh[None]@w-w@dh[None]
    uc=u[:,:,2:5].conj().swapaxes(-1,-2)[None]
    uv=u[:,:,:2][None]
    gap=eqp[:,2:5,None]-eqp[:,None,:2]
    reference=(uc@corrected@uv)/gap[None]
    bare_rotation=uc@w@uv
    assert np.max(np.abs(reference-bare_rotation))>1e-3
    np.testing.assert_allclose(np.asarray(got)[:,:,:3,:2],reference,rtol=5e-13,atol=5e-13)
    assert np.isfinite(np.asarray(got)).all()
    assert np.count_nonzero(np.asarray(got)[:,:,3:])==0
    assert np.count_nonzero(np.asarray(got)[:,:,:,2:])==0
    np.testing.assert_array_equal(np.asarray(delta)[:,:3,:2],gap)
    assert not got.sharding.is_fully_replicated


def test_window_admission_refuses_before_payload_io(tmp_path):
    mesh=mesh_for(2)
    path=tmp_path/"dipole.h5"
    write_dipole_fixture(path)
    wfn=tmp_path/"WFN.h5"
    with h5py.File(wfn,"w"):
        pass
    def request(**kw):
        return load_dipole_h5(path,wfn_path=wfn,mesh_xy=mesh,
            band_rows=(11,16),band_cols=(4,11),**kw)
    with patch.object(SlabIO,"read_slab",side_effect=AssertionError("invalid window reached I/O")):
        with pytest.raises(ValueError,match="band_shape"):
            request(band_shape=(4,8))
        for key,value,gate in (("band_min",1,"dipole_band_window"),
                               ("basis","qp","dipole_basis"),
                               ("format_version",1,"dipole_format_version")):
            with h5py.File(path,"a") as f:
                old=f.attrs[key]
                f.attrs[key]=value
            with pytest.raises(ValueError,match=gate):
                request()
            with h5py.File(path,"a") as f:
                f.attrs[key]=old


def test_text_and_spectrum_programs_keep_only_owned_input_tiles():
    """Large abstract CV banks expose implicit all-gathers hidden by value tests."""
    import math
    import re
    from bse.absorption_common import _noeh_block_program, _transition_spectrum_program
    mesh=mesh_for(2)
    shape=(3,2,514,518)
    sh=NamedSharding(mesh,P(None,None,"x","y"))
    de_sh=NamedSharding(mesh,P(None,"x","y"))
    field=jax.ShapeDtypeStruct(shape,np.complex128,sharding=sh)
    delta=jax.ShapeDtypeStruct(shape[1:],np.float64,sharding=de_sh)
    programs=[_noeh_block_program(mesh,513,517).lower(field,np.arange(1024,dtype=np.int32)).compile(),
              _transition_spectrum_program(mesh,"lorentzian").lower(
                  field,delta,np.linspace(.1,2.,7),np.asarray(.1)).compile()]
    local_elements=math.prod(shape)//4
    for executable in programs:
        hlo=executable.as_text()
        complex_shapes=re.findall(r'c128\[([0-9,]+)\]',hlo)
        assert complex_shapes, "expected actual optimized complex HLO"
        assert max(math.prod(map(int,dims.split(','))) for dims in complex_shapes)<=local_elements

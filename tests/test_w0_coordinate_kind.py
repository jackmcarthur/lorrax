"""Real W0 persistence on declared fractional/FFT q parents and minimal meta."""
from types import SimpleNamespace

import h5py
import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.centroid_basis import PackedCentroidBasis
from file_io import write_restart_state_to_h5
from gw.gw_config import BispinorGWMode, ComputeMode, HeadCorrection, ScreeningDiagrams
from gw.gw_output import persist_w0_and_head
from gw.restart_q_storage import (
    capture_scope, deposit_pre_unfold, resolve_restart_q_storage_for_run,
    take_pre_unfold,
)
from symmetry_maps import QirrOperator, read_tensor as read_qirr_tensor


def _cpu_mesh():
    return Mesh(np.asarray(jax.devices('cpu')[:1]).reshape(1, 1), ('x', 'y'))


def _config():
    return SimpleNamespace(do_screened=True, write_restart_tensors=True,
        compute_mode=ComputeMode.COHSEX, bispinor=True,
        bispinor_gw=BispinorGWMode.COULOMB_ONLY,
        screening=SimpleNamespace(diagrams=ScreeningDiagrams.W_RPA),
        head=SimpleNamespace(correction=HeadCorrection.FULL),
        qp_solver='one_shot_dft', restart_q_storage_raw='auto')


def _head():
    return SimpleNamespace(omega=0j, vc0=1.25+0j, wcoul0=.5+0j,
        S_cart=np.eye(3)*.2, source='test_static_physical_units', response_kind='test')


def _complex_parents(nq, nmu):
    # Distinct complex Hermitian blocks expose row/centroid/adjoint mistakes.
    x=np.arange(nq*nmu*nmu, dtype=np.float64).reshape(nq,nmu,nmu)
    a=(x+1)/37+1j*(x[::-1]+2)/53
    return a+np.swapaxes(a.conj(),-1,-2)+np.eye(nmu)[None]


def _deposit(name, values, resolution, basis):
    perm, wraps = basis.pack_tables(resolution.sym_perm, resolution.L_table)
    deposit_pre_unfold(name, values, n_rmu_logical=basis.n_logical,
        q_irr_frac=np.asarray([[0.,0.,0.],[1/3,0.,0.]]),
        irr_idx_q=np.asarray([0,1,1],np.int32),
        sym_idx_q=np.asarray([0,0,1],np.int32),
        sym_perm=perm, L_table=wraps, n_sym_spatial=2, mu_basis=basis)
    return QirrOperator(values=values, irr_idx=np.asarray([0,1,1],np.int32),
        sym_idx=np.asarray([0,0,1],np.int32), sym_perm=perm, L_table=wraps,
        q_irr_frac=np.asarray([[0.,0.,0.],[1/3,0.,0.]]),
        n_sym_spatial=2, full_rows=np.asarray([0,1],np.int32))


@pytest.mark.parametrize('coordinate_kind', ['fractional','fft_indices'])
def test_declared_centroid_kind_preserves_native_w0_parents(tmp_path, coordinate_kind):
    mesh=_cpu_mesh();cfg=_config();grid=(10,10,10)
    sym=SimpleNamespace(sym_matrices=np.asarray([np.eye(3),-np.eye(3)],np.int32),
                        translations=np.zeros((2,3)),sym_idx_q=np.asarray([0,0,1],np.int32))
    points=(np.asarray([[0.,0.,0.],[.173,.217,.391],[.827,.783,.609]])
            if coordinate_kind=='fractional' else
            np.asarray([[0,0,0],[1,2,3],[9,8,7]],np.int32))
    basis=PackedCentroidBasis.build(points,sym,grid,mesh,coordinate_kind=coordinate_kind)
    meta=SimpleNamespace(n_rmu=len(points),fft_grid=grid,mu_basis=basis)
    if coordinate_kind=='fractional':
        assert np.max(np.abs(points*np.asarray(grid)-np.rint(points*np.asarray(grid))))>.1
        # Demonstrate the real failure: the same coordinates interpreted as
        # integer indices do not preserve this inversion-closed point cloud.
        wrong=resolve_restart_q_storage_for_run(cfg,sym=sym,centroid_indices=points,
            fft_grid=grid,coordinate_kind='fft_indices',print_fn=lambda *_:None)
        assert not wrong.store_wedge
    decision=resolve_restart_q_storage_for_run(cfg,sym=sym,centroid_indices=points,
        fft_grid=grid,coordinate_kind=coordinate_kind,print_fn=lambda *_:None)
    assert decision.store_wedge and decision.resolution.verdict.closed
    put=lambda x:jax.device_put(x,NamedSharding(mesh,P(None,'x','y')))
    vcanonical=_complex_parents(2,len(points))
    wcanonical=_complex_parents(2,len(points))*0.31
    v=basis.pack_operator(put(vcanonical));w=basis.pack_operator(put(wcanonical))
    path=tmp_path/(coordinate_kind+'.h5')
    with capture_scope():
        _deposit('V_qmunu',v,decision.resolution,basis)
        vcapture=decision.with_capture(take_pre_unfold('V_qmunu'))
        write_restart_state_to_h5(str(path),V_qmunu=v,
            n_rmu_logical=len(points),mesh=mesh,init_W0=True,
            qirr=vcapture,kgrid=(3,1,1))
        with h5py.File(path,'r') as f:
            assert f['V_qmunu'].shape==f['W0_qmunu'].shape==(2,len(points),len(points))
        wop=_deposit('W0_qmunu',w,decision.resolution,basis)
        persist_w0_and_head(wop,tensors_filename=str(path),head_resolver=SimpleNamespace(at=lambda _: _head()),
            config=cfg,meta=meta,mesh_xy=mesh,sym=sym,centroid_indices=points,print_fn=lambda *_:None)
        assert take_pre_unfold('W0_qmunu') is None  # Exactly one producer capture consumed.
    vread,vhdr=read_qirr_tensor(str(path),'V_qmunu',mesh_xy=mesh,unfold=False)
    wread,whdr=read_qirr_tensor(str(path),'W0_qmunu',mesh_xy=mesh,unfold=False)
    np.testing.assert_array_equal(np.asarray(vread),vcanonical)
    np.testing.assert_array_equal(np.asarray(wread),wcanonical)
    assert vhdr.q_storage==whdr.q_storage=='ibz' and whdr.data_ready
    with h5py.File(path,'r') as f:
        assert f['W0_qmunu'].attrs['W0_ready']
        np.testing.assert_array_equal(f['whead'][()],np.asarray([.5+0j]))
        assert str(f['W0_qmunu'].attrs['screening_diagrams'])=='w_rpa'
    # Read actual public full-zone actions, including nontrivial complex wrap
    # phases. A missing centroid/q permutation cannot pass native equality alone.
    full,_=read_qirr_tensor(str(path),'W0_qmunu',mesh_xy=mesh,unfold=True)
    expected=basis.unpack_operator(wop.unfold(mesh))
    np.testing.assert_array_equal(np.asarray(full),np.asarray(expected))
    assert np.max(np.abs(np.asarray(full)[2]-wcanonical[1]))>1.e-3


def test_legacy_integer_meta_without_basis_keeps_parent_domain(tmp_path):
    # The production default before fractional support carried no typed basis.
    # Declared integer points must retain that fallback without an AttributeError.
    mesh=_cpu_mesh();cfg=_config();grid=(10,10,10)
    sym=SimpleNamespace(sym_matrices=np.asarray([np.eye(3),-np.eye(3)],np.int32),translations=np.zeros((2,3)),sym_idx_q=np.asarray([0,0,1],np.int32))
    points=np.asarray([[0,0,0],[1,2,3],[9,8,7]],np.int32)
    basis=PackedCentroidBasis.build(points,sym,grid,mesh)
    decision=resolve_restart_q_storage_for_run(cfg,sym=sym,centroid_indices=points,fft_grid=grid,print_fn=lambda *_:None)
    put=lambda x:jax.device_put(x,NamedSharding(mesh,P(None,'x','y')))
    native=_complex_parents(2,3);packed=basis.pack_operator(put(native))
    path=tmp_path/'legacy_integer.h5'
    with capture_scope():
        _deposit('V_qmunu',packed,decision.resolution,basis)
        write_restart_state_to_h5(str(path),V_qmunu=packed,n_rmu_logical=3,mesh=mesh,init_W0=True,
            qirr=decision.with_capture(take_pre_unfold('V_qmunu')),kgrid=(3,1,1))
        op=_deposit('W0_qmunu',packed,decision.resolution,basis)
        persist_w0_and_head(op,tensors_filename=str(path),head_resolver=SimpleNamespace(at=lambda _: _head()),
            config=cfg,meta=SimpleNamespace(n_rmu=3,fft_grid=grid),mesh_xy=mesh,
            sym=sym,centroid_indices=points,print_fn=lambda *_:None)
    actual,hdr=read_qirr_tensor(str(path),'W0_qmunu',mesh_xy=mesh,unfold=False)
    np.testing.assert_array_equal(np.asarray(actual),native)
    assert hdr.q_storage=='ibz'


def test_minimal_meta_without_fft_or_basis_keeps_full_zone(tmp_path):
    mesh=_cpu_mesh();cfg=_config()
    native=_complex_parents(3,2)
    value=jax.device_put(native,NamedSharding(mesh,P(None,'x','y')))
    path=tmp_path/'minimal_full.h5'
    write_restart_state_to_h5(str(path),V_qmunu=value,n_rmu_logical=2,mesh=mesh,init_W0=True,kgrid=(3,1,1))
    persist_w0_and_head(value,tensors_filename=str(path),head_resolver=SimpleNamespace(at=lambda _: _head()),
        config=cfg,meta=SimpleNamespace(n_rmu=2),mesh_xy=mesh,print_fn=lambda *_:None)
    actual,hdr=read_qirr_tensor(str(path),'W0_qmunu',mesh_xy=mesh,unfold=False)
    np.testing.assert_array_equal(np.asarray(actual),native)
    assert hdr.q_storage=='full'

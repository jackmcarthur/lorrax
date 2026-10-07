"""Fail closed before a partial or mismatched augmented-current fit starts."""
from types import SimpleNamespace
import numpy as np
import pytest


def operands():
    from jax.sharding import Mesh
    import jax
    mesh=Mesh(np.asarray(jax.devices('cpu')[:1]).reshape(1,1),('x','y'))
    nmu=SimpleNamespace(shape=(1,4,4,2));mun=SimpleNamespace(shape=(1,4,2,4))
    plan=SimpleNamespace(n_parent=1,n_centroid_packed=2,coordinate_kind='fft_indices')
    parent=SimpleNamespace(psi_G=SimpleNamespace(is_deleted=lambda:False))
    meta=SimpleNamespace(n_rmu=2,n_rmu_padded=2,mu_solve_extent=2,mu_basis=None,
        n_rtot=8,nk_tot=3,kgrid=(3,1,1),fft_grid=(2,2,2))
    state=dict(meta=meta,plan=plan,parent_psi=parent,parent_faces=(nmu,mun),
        unit_endpoint_loss=True,current_basis_rows=None,band_range_left=(0,2),band_range_right=(1,4),
        source_frame_policy='same_actual_served_four_spinor_full_WFN_Lowdin',shared_physical_bands=150,
        q_full_indices=np.arange(3))
    args=dict(wfn=SimpleNamespace(nbands=150),sym=SimpleNamespace(kvecs_asints=np.asarray([[0,0,0],[1,0,0],[2,0,0]])),
        meta=meta,centroid_indices=np.zeros((2,3),np.int32),mesh_xy=mesh,
        output_files={1:'unused1',2:'unused2',3:'unused3'},k_unfold_plan=plan,
        psi_nmu_parent=nmu,psi_mun_parent=mun,mubatch_plan=object(),parent_psi=parent,
        use_augmented_samples=True,current_augmentation=state,write_zeta_file=False,
        write_ibz_only=False,band_range_left=(0,2),band_range_right=(1,4))
    return state,args


@pytest.mark.parametrize('change',('partial','writer','unaugmented','weights','meta','source','faces','frame','bands'))
def test_current_fit_requires_one_complete_protected_unit_loss_family(change):
    from gw.isdf_fitting import fit_zeta_to_h5
    state,args=operands()
    if change=='partial':args['output_files']={1:'unused1',2:'unused2'}
    elif change=='writer':args['write_zeta_file']=True
    elif change=='unaugmented':args['use_augmented_samples']=False
    elif change=='weights':args['charge_fit_weights']={'occupied_stop':2,'occupied_weight':4.}
    elif change=='meta':state['meta']=object()
    elif change=='source':state['parent_psi']=object()
    elif change=='faces':state['parent_faces']=(object(),args['psi_mun_parent'])
    elif change=='frame':state['source_frame_policy']='native_Pauli_overlap'
    else:state['shared_physical_bands']=120
    with pytest.raises(ValueError,match='complete no-file OWN3'):
        fit_zeta_to_h5(**args)


def test_current_fit_refuses_charge_donated_source():
    from gw.isdf_fitting import fit_zeta_to_h5
    state,args=operands();args['parent_psi'].psi_G.is_deleted=lambda:True
    with pytest.raises(ValueError,match='donated by the preceding charge fit'):
        fit_zeta_to_h5(**args)


def test_current_fit_refuses_changed_basis_and_windows():
    from gw.isdf_fitting import fit_zeta_to_h5
    state,args=operands();args['current_basis_rows']=np.eye(3)
    with pytest.raises(ValueError,match='canonical fitting basis'):
        fit_zeta_to_h5(**args)
    args['current_basis_rows']=None;args['band_range_right']=(1,3)
    with pytest.raises(ValueError,match='band windows differ'):
        fit_zeta_to_h5(**args)


def test_current_fit_refuses_reordered_q_rows_before_sphere_and_large_buffers():
    from gw.isdf_fitting import fit_zeta_to_h5
    state,args=operands();state['q_full_indices']=np.asarray([2,1,0])
    with pytest.raises(ValueError,match='canonical q rows differ'):
        fit_zeta_to_h5(**args)


def test_current_fit_prices_raw_operands_beside_native_planner_before_geometry(monkeypatch):
    from gw.isdf_fitting import fit_zeta_to_h5
    from common import gpu_utils
    state,args=operands();args['mubatch_plan']=SimpleNamespace(hwm_bytes=8.)
    state.update(fit_resident_extra_bytes_per_rank=3.,phase_prices={})
    monkeypatch.setattr(gpu_utils,'device_budget_bytes',lambda:10.)
    monkeypatch.setattr(gpu_utils,'warn_over_budget',lambda *args:None)
    with pytest.raises(ValueError,match='shared-resident device memory budget'):
        fit_zeta_to_h5(**args)

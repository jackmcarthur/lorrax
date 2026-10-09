"""Protected canonical current operands and typed V metadata fail closed."""
from types import SimpleNamespace
import numpy as np
import pytest


def test_live_bispinor_preflight_keeps_exact_fractional_basis(monkeypatch):
    from gw import v_q_g_flat,v_q_bispinor
    points=np.asarray([[.1234567,.271,.31],[.6234567,.729,.69]])
    basis=SimpleNamespace(coordinate_kind='fractional')
    seen={}
    def resolver(**kwargs):
        seen.update(kwargs)
        return (None,)*6+(True,)
    monkeypatch.setattr(v_q_g_flat,'_resolve_ibz_q_list',resolver)
    assert v_q_bispinor._bispinor_ibz(object(),points,(2,2,1),(8,8,8),
        'fractional witness',mu_basis=basis)
    assert seen['mu_basis'] is basis and seen['centroid_indices'] is points
    assert seen['coordinate_kind'] is None  # Existing resolver consumes basis.kind.


def test_file_bispinor_preflight_refuses_fractional_reinterpretation():
    from gw.v_q_bispinor import compute_V_q_bispinor_g_flat_to_h5
    loader=SimpleNamespace(coordinate_kind='fractional')
    with pytest.raises(ValueError,match='coordinate kind disagrees'):
        compute_V_q_bispinor_g_flat_to_h5(
            zeta_C_loader=loader,zeta_T_loaders=(loader,)*3,
            output_h5_path='never-created.h5',mesh_xy=None,kgrid=(1,1,1),
            fft_grid=(8,8,8),bvec=np.eye(3),cell_volume=1.,sys_dim=3,
            n_rmu_C=2,n_rmu_T=2,sym=object(),use_ibz=True,
            coordinate_kind_C='fft_indices',coordinate_kind_T='fractional',
            mc_average_vcoul_body=False)


@pytest.mark.parametrize('file_read', [False,True])
def test_group_writer_refuses_scalar_and_file_only_families_before_geometry(file_read):
    from gw.v_q_g_flat import _compute_V_q_g_flat_tiles
    fields={} if file_read else dict(contract_v=lambda:None)
    loader=SimpleNamespace(zeta_layout='G_flat',**fields)
    with pytest.raises(ValueError,match='complete live three-current'):
        _compute_V_q_g_flat_tiles(
            [dict(L=loader,R=None,timing_label='negative',is_charge_cc=True)],
            kgrid=(1,1,1),fft_grid=(8,8,8),mesh_xy=None,g_chunk=None,
            sym=None,centroid_indices=None,verbose=False,zeta_ios=(object(),))


def test_augmented_current_fit_consumes_protected_faces_without_raw_reload(monkeypatch):
    from gw import gw_init,isdf_fitting,isdf_augmentation
    from common import psi_G_store
    protected=(object(),object());source=object();plan=object()
    meta=SimpleNamespace(current_basis_rows=None)
    state=dict(parent_faces=protected,parent_psi=source,plan=plan,meta=meta)
    seen={};channels={v:SimpleNamespace(close=lambda:None) for v in (1,2,3)}
    def forbidden(*args,**kwargs):raise AssertionError('raw source reloaded after reconstruction')
    monkeypatch.setattr(psi_G_store,'load_parent_psi_G',forbidden)
    carrier=SimpleNamespace(plan=plan,psi_nmu=object(),psi_mun=object())
    def sample(*args,**kwargs):
        assert kwargs['faces'] is protected and kwargs['plan'] is plan
        return dict(green_parent=carrier)
    monkeypatch.setattr(gw_init,'_transverse_wfn_data',sample)
    def fit(**kwargs):
        assert kwargs['parent_psi'] is source
        assert kwargs['psi_nmu_parent'] is protected[0]
        assert kwargs['psi_mun_parent'] is protected[1]
        assert kwargs['current_augmentation'] is state
        assert kwargs['k_unfold_plan'] is plan and kwargs['meta'] is meta
        assert kwargs['write_zeta_file'] is False and kwargs['use_augmented_samples'] is True
        assert kwargs.get('charge_fit_weights') is None
        seen['fit']=True
        return 0,channels
    monkeypatch.setattr(isdf_fitting,'fit_zeta_to_h5',fit)
    monkeypatch.setattr(gw_init,'_current_body_contract',lambda *args,**kwargs:{'witness':True})
    def attach(zetas,current,*,body_contract):
        assert zetas is channels and current is state and body_contract=={'witness':True}
        seen['attach']=True
    monkeypatch.setattr(isdf_augmentation,'attach_current_augmentation',attach)
    def tiles(zetas,**kwargs):
        assert list(zetas)==list(channels.values()) and kwargs['zeta_ios'] is None
        return 'parked complete current metric'
    monkeypatch.setattr(gw_init,'_bispinor_current_tiles',tiles)
    monkeypatch.setattr(gw_init,'_gate_fresh_zeta_rank_findings',lambda *args,**kwargs:None)
    monkeypatch.setattr(gw_init,'barrier',lambda *args:None)
    cfg=SimpleNamespace(paths=SimpleNamespace(centroids_file_current='synthetic'),
        backend=SimpleNamespace(distrib_la_batched_route='batch_reshard'),write_restart_tensors=False)
    chunk=dict(mubatch=SimpleNamespace(band_chunk=4),band_chunk=4,centroid_k_chunk=1,k_unfold_plan=plan)
    result,tiles=gw_init._fit_transverse_zeta_channels(
        None,object(),chunk,meta,lambda mu:'unused',(False,)*3,(),True,
        {mu:f'unused{mu}' for mu in (1,2,3)},2.,(0,4),(0,8),object(),cfg,
        object(),lambda line:None,SimpleNamespace(current_lift='normalized_rkb'),
        object(),object(),object(),current_augmentation=state)
    assert seen==dict(fit=True,attach=True) and result['green_parent'] is carrier
    assert tiles=='parked complete current metric'


def test_canonical_prepared_faces_are_not_conjugated_or_transposed_again(monkeypatch):
    from gw import gw_init,wavefunction_bundle
    from common import wfn_transforms
    faces=(object(),object());plan=object();seen={}
    def forbidden(*args,**kwargs):raise AssertionError('prepared canonical faces treated as raw loader faces')
    monkeypatch.setattr(wavefunction_bundle,'parent_faces',forbidden)
    monkeypatch.setattr(wfn_transforms,'load_centroids_band_chunked',forbidden)
    monkeypatch.setattr(wfn_transforms,'get_enk_bandrange',lambda *args,**kwargs:(object(),None))
    monkeypatch.setattr(wavefunction_bundle,'wavefunctions_face_from_restart',lambda *args,**kwargs:object())
    def carrier(wfns,nmu,mun,**kwargs):
        assert nmu is faces[0] and mun is faces[1] and kwargs['plan'] is plan
        seen['canonical']=True
        return 'canonical current carrier'
    monkeypatch.setattr(wavefunction_bundle,'build_packed_parent_green_carrier',carrier)
    result=gw_init._transverse_wfn_data(object(),object(),object(),object(),
        SimpleNamespace(bispinor=True,bispinor_gw=None),object(),
        SimpleNamespace(full_range=(0,8),b1=4,b3=8),4,faces=faces,plan=plan)
    assert seen==dict(canonical=True) and result['green_parent']=='canonical current carrier'


def test_augmented_current_refuses_partial_family_before_loading():
    from gw.gw_init import _fit_transverse_zeta_channels
    cfg=SimpleNamespace(paths=SimpleNamespace(centroids_file_current='synthetic'))
    with pytest.raises(ValueError,match='fresh complete OWN3'):
        _fit_transverse_zeta_channels(None,None,{},None,lambda mu:'unused',
            (False,True,False),(),True,{1:'unused',2:'unused',3:'unused'},
            2.,(0,4),(0,8),None,cfg,None,lambda line:None,None,None,None,None,
            current_augmentation={})

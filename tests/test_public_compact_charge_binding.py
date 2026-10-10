"""Public compact scalar binding: source, frame, file and restart admission.

Synthetic metadata isolate provenance contracts. Numerical C/D/B/A,
reconstructed Hartree and fitted Coulomb errors retain their own oracles.
"""
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _public_config():
    from gw.gw_config import (BispinorGWMode, ComputeMode, HeadCorrection,
        QPSolver, ScreeningDiagrams)
    return SimpleNamespace(paths=SimpleNamespace(atomic_reconstruction_dir='bound'),
        bispinor=True,bispinor_gw=BispinorGWMode.COULOMB_ONLY,sys_dim=2,
        compute_mode=ComputeMode.MPA,qp_solver=QPSolver.ONE_SHOT_DFT,
        density_self_consistent=False,head=SimpleNamespace(correction=HeadCorrection.OFF),
        occ_smearing_width_ry=None,screening=SimpleNamespace(occ_broadening_ev=0.,
            diagrams=ScreeningDiagrams.W_RPA),
        restart=False,sigma=SimpleNamespace(w_model='shared_pole'))


def test_public_scalar_slab_accepts_existing_dynamic_models_without_new_equations():
    from gw.gw_config import (ComputeMode,refuse_unsupported_bispinor_gw,
        refuse_unsupported_compact_charge)
    cfg=_public_config()
    for model in (ComputeMode.X_ONLY,ComputeMode.COHSEX,ComputeMode.MPA):
        cfg.compute_mode=model
        refuse_unsupported_bispinor_gw(cfg)
        refuse_unsupported_compact_charge(cfg,{'compact_target':object()})


def test_compact_request_cannot_inherit_legacy_bulk_current_admission():
    from gw.gw_config import (BispinorGWMode,ComputeMode,
        refuse_unsupported_bispinor_gw,refuse_unsupported_compact_charge)
    cfg=_public_config();cfg.sys_dim=3
    cfg.bispinor_gw=BispinorGWMode.BARE_TRANSVERSE;cfg.compute_mode=ComputeMode.X_ONLY
    refuse_unsupported_bispinor_gw(cfg)
    with pytest.raises(ValueError,match='public_compact_charge_domain'):
        refuse_unsupported_compact_charge(cfg,{'compact_target':object()})
    refuse_unsupported_compact_charge(cfg,{'compact_target':None})


@pytest.mark.parametrize('change',['current','head','SC','density','smearing','broadening','dimension'])
def test_public_scalar_slab_envelope_refuses_unproved_cases(change):
    from gw.gw_config import BispinorGWMode,HeadCorrection,QPSolver,refuse_unsupported_bispinor_gw
    cfg=_public_config()
    if change=='current':cfg.bispinor_gw=BispinorGWMode.BARE_TRANSVERSE
    elif change=='head':cfg.head.correction=HeadCorrection.FULL
    elif change=='SC':cfg.qp_solver=QPSolver.SELF_CONSISTENT
    elif change=='density':cfg.density_self_consistent=True
    elif change=='smearing':cfg.occ_smearing_width_ry=.01
    elif change=='broadening':cfg.screening.occ_broadening_ev=.01
    else:cfg.sys_dim=1
    with pytest.raises(ValueError,match='atomic_augmentation_domain'):
        refuse_unsupported_bispinor_gw(cfg)


@pytest.fixture
def bound_source(monkeypatch):
    from common import parallel_transport
    from gw import augmentation_hartree as owner,isdf_augmentation as stage
    from isdf.atomic_hartree import charge_hartree_operator_contract
    wfn=SimpleNamespace(nbands=5,fft_grid=(8,8,12),cell_volume=1280.,alat=1.,
        avec=np.diag([8.,8.,20.]),blat=1.,bvec=np.diag(2*np.pi/np.array([8.,8.,20.])),
        atom_crys=np.zeros((1,3)),atom_types=np.array([1]),
        kvecs=lambda **_:np.array([[0.,0.,0.],[.5,0.,0.]]))
    sym=SimpleNamespace(R_cart=np.eye(3)[None],kirr_fullids=np.array([0,1]),
        parent_k_domain='ibz',nk_red=2)
    plan=SimpleNamespace(sym=sym,nspinor=4,n_parent=2,parent_full_rows=sym.kirr_fullids,
        k_parent_frac=wfn.kvecs(),irr_idx=np.arange(2))
    fp='3'*64;token=object()
    def fingerprint(binding,actual_wfn):
        assert binding is token and actual_wfn is wfn
        return fp
    monkeypatch.setattr(parallel_transport,'fingerprint_from_binding',fingerprint)
    physical=dict(occupations_sha256='4'*64,full_kweights_sha256='5'*64,spin_degeneracy=1.)
    monkeypatch.setattr(stage,'_hartree_source_request',lambda *a,**k:physical)
    common=dict(model='compact_native_pauli_common_frame_v1',carrier='normalized_rkb',
        physical_bands=5,complete_all_FILE_parents=True,wfn_sha256='2'*64,
        field_policy='unwindowed_U_of_compact_native_pauli',
        source_frame_policy='compact_native_pauli_common_A_before_U',
        normalization='one compact target A; no represented-field renormalization',
        frame_file_sha256='6'*64)
    A=np.broadcast_to(np.eye(5,dtype=complex),(2,5,5)).copy()
    public=dict(wfn_sha256='2'*64,
        wfn_fingerprint_scheme=parallel_transport.WFN_FINGERPRINT_SCHEME,wfn_fingerprint=fp)
    artifact=dict(identity='7'*64,radial=dict(kind='log_simpson',points=8,r_min=1e-7,
        r_max=.7,support_radius=.7,interpolation_degree=5,quadrature_order=16),
        angular=dict(lebedev_order=5,lmax=1,orthogonality_tolerance=2e-12),
        compact_target=dict(binding=common,frame=dict(arrays=dict(inverse_sqrt=A))),
        public_source_identity=public)
    source=dict(schema='lorrax.augmentation_occupied_source.v1',augmentation_identity=artifact['identity'],
        prepared_raw_binding=common,compact_target_binding=common,physical_bands=5,
        public_band_range=[0,5],**stage._physical_full_wfn_frame_binding(A,5),**physical,
        fft_grid=list(wfn.fft_grid),cell_volume=wfn.cell_volume,
        source_frame_policy=common['source_frame_policy'],sys_dim=2,
        hartree_kernel=charge_hartree_operator_contract(wfn,sys_dim=2)['kernel'],**public)
    def provenance(source=source,band_range=(0,5)):
        binding=owner._operator_binding(wfn=wfn,sym=sym,artifact=artifact,
            source_identity=owner._identity(source),band_range=band_range,sys_dim=2)
        return dict(schema='lorrax.resident_charge_hartree.v1',source_identity=owner._identity(source),
            source_binding=source,operator_identity=owner._identity(binding),operator_binding=binding,
            band_range=list(band_range),parent_full_rows=binding['parent_full_rows'],
            parent_k_frac=binding['parent_k_frac'],units='Ry',k_domain='file_wedge',trs_rule='conj')
    return SimpleNamespace(wfn=wfn,sym=sym,plan=plan,token=token,artifact=artifact,source=source,
        provenance=provenance,owner=owner)


def _authenticate(f,record=None,artifact=None,band_range=(0,5)):
    return f.owner.require_resident_hartree_source(f.provenance() if record is None else record,
        wfn=f.wfn,sym=f.sym,plan=f.plan,artifact=f.artifact if artifact is None else artifact,
        wfn_fingerprint_binding=f.token,band_range=band_range)


def test_exact_compact_full_band_source_serves_with_only_thin_context(bound_source):
    f=bound_source
    assert _authenticate(f)['source_binding']==f.source
    bound=f.owner.bind_resident_hartree(dict(parent_kij_ry=object(),provenance=f.provenance()),
        wfn=f.wfn,sym=f.sym,plan=f.plan,artifact=f.artifact,
        wfn_fingerprint_binding=f.token,band_range=(0,5))
    thin=bound['source_context']['artifact']
    assert set(thin)=={'identity','radial','angular','compact_target_binding',
                       'compact_frame_sha256','public_source_identity'}
    assert _authenticate(f,artifact=thin)['source_binding']==f.source
    assert not any(key in thin for key in ('state','frame','compact_target','caches','inverse_sqrt','parent_psi'))


@pytest.mark.parametrize('change',['frame','paired_field','carrier','fingerprint','whole_WFN','window','kernel'])
def test_changed_physical_compact_source_refuses_even_after_rehash(bound_source,change):
    f=bound_source;source=copy.deepcopy(f.source)
    if change=='frame':source['full150_frame_sha256']='8'*64
    elif change=='paired_field':source['compact_target_binding']['field_policy']='other'
    elif change=='carrier':source['compact_target_binding']['carrier']='pauli2embed4'
    elif change=='fingerprint':source['wfn_fingerprint']='9'*64
    elif change=='whole_WFN':source['wfn_sha256']='9'*64
    elif change=='window':source['physical_frame_band_domain']=[1,5]
    else:source['hartree_kernel']['truncation_half_height_bohr']=9.
    with pytest.raises(ValueError,match='resident_hartree_source'):
        _authenticate(f,record=f.provenance(source))


def test_edge_only_private_matrix_cannot_cover_full_public_GW_window(bound_source):
    f=bound_source
    with pytest.raises(ValueError,match='resident_hartree_source'):
        _authenticate(f,record=f.provenance(band_range=(3,5)))


def test_public_radial_identity_uses_same_log_decoder_as_fitting(bound_source):
    from gw.isdf_augmentation import _radial_grid
    f=bound_source;explicit=copy.deepcopy(f.artifact)
    radius,weights,support=_radial_grid(explicit['radial'])
    explicit['radial']=dict(radius=radius,weights_dr=weights,support_radius=support,
        interpolation_degree=5,quadrature_order=16)
    kwargs=dict(wfn=f.wfn,sym=f.sym,source_identity='1'*64,band_range=(0,5),sys_dim=2)
    assert f.owner._operator_binding(artifact=f.artifact,**kwargs)==f.owner._operator_binding(artifact=explicit,**kwargs)


def test_public_manifest_uses_existing_target_loader_and_one_actual_WFN_join(monkeypatch,tmp_path):
    import hashlib
    from common import parallel_transport
    from gw import isdf_augmentation as stage
    wfn=SimpleNamespace(path=str(tmp_path/'WFN.h5'));sym=object();token=object()
    Path(wfn.path).write_bytes(b'fixed synthetic source bytes')
    whole_sha=hashlib.sha256(Path(wfn.path).read_bytes()).hexdigest()
    request=dict(frame_file='frame.npz',frame_sha256='1'*64,wfn_sha256=whole_sha,
        carrier='normalized_rkb',species_fields={'1':dict(file='fields.npz',
            file_sha256='3'*64,common_spectrum_sha256='4'*64)})
    original=dict(directory=str(tmp_path),carrier='normalized_rkb',compact_target_request=request)
    monkeypatch.setattr(stage,'read_augmentation_manifest',lambda _:original)
    seen=[]
    def bind(artifact,resolved,*,wfn,sym):
        assert artifact is original
        assert resolved['frame_file']==str(tmp_path/'frame.npz')
        assert resolved['species_fields']['1']['file']==str(tmp_path/'fields.npz')
        seen.append((wfn,sym))
        return dict(artifact,identity='5'*64,compact_target=dict(frame=dict(metadata=dict(
            source_input_files_sha256={}))))
    monkeypatch.setattr(stage,'bind_compact_target_artifact',bind)
    monkeypatch.setattr(parallel_transport,'fingerprint_from_binding',
        lambda binding,actual:'6'*64 if binding is token and actual is wfn else pytest.fail('wrong FP owner'))
    result=stage.read_bound_augmentation_manifest(tmp_path,wfn=wfn,sym=sym,wfn_fingerprint_binding=token)
    assert seen==[(wfn,sym)] and result['identity']=='5'*64
    assert result['public_source_identity']['wfn_fingerprint']=='6'*64
    assert result['public_source_identity']['wfn_sha256']==whole_sha
    stage.require_public_compact_wfn_source(result,wfn)
    Path(wfn.path).write_bytes(b'changed source bytes after coefficient preparation')
    with pytest.raises(ValueError,match='changed during coefficient preparation'):
        stage.require_public_compact_wfn_source(result,wfn)
    Path(wfn.path).write_bytes(b'fixed synthetic source bytes')
    request['carrier']='pauli2embed4'
    with pytest.raises(ValueError,match='public_compact_charge_carrier'):
        stage.read_bound_augmentation_manifest(tmp_path,wfn=wfn,sym=sym,wfn_fingerprint_binding=token)
    original['carrier']='pauli2embed4'
    result=stage.read_bound_augmentation_manifest(tmp_path,wfn=wfn,sym=sym,wfn_fingerprint_binding=token)
    assert result['carrier']=='pauli2embed4' and len(seen)==2
    original['carrier']='normalized_rkb'
    request['carrier']='normalized_rkb';request['wfn_sha256']='7'*64
    with pytest.raises(ValueError,match='currently loaded WFN'):
        stage.read_bound_augmentation_manifest(tmp_path,wfn=wfn,sym=sym,wfn_fingerprint_binding=token)


def test_public_restart_authenticates_same_bound_identity_before_tensor_reads(monkeypatch):
    from gw import gw_init
    from file_io import restart_bundle
    class ReachedAtomicIdentity(Exception):pass
    def capture(path,identity):
        assert path=='closed_public.h5' and identity=='bound_target_identity'
        raise ReachedAtomicIdentity
    monkeypatch.setattr(restart_bundle,'require_atomic_augmentation_match',capture)
    cfg=SimpleNamespace(paths=SimpleNamespace(atomic_reconstruction_dir='bound'))
    with pytest.raises(ReachedAtomicIdentity):
        gw_init._read_authenticated_restart(None,None,cfg,None,None,None,None,
            'closed_public.h5',charge_fit_context={},
            augmentation_artifact={'identity':'bound_target_identity'})
    with pytest.raises(AssertionError,match='SAME authenticated bound artifact'):
        gw_init._read_authenticated_restart(None,None,cfg,None,None,None,None,
            'closed_public.h5',charge_fit_context={})


def test_declared_pauli_source_embedding_preserves_upper_components_and_zero_ghosts():
    import jax
    from jax.sharding import Mesh,NamedSharding
    from common.four_current_model import PAULI_ZERO_SMALL_CARRIER
    from common.wfn_layout import band_sphere_spec
    from common.wfn_transforms import load_psi_gflat_padded
    mesh=Mesh(np.array(jax.devices()[:1]).reshape(1,1),('x','y'))
    physical=(np.arange(24).reshape(1,3,2,4)+1j*np.arange(24,48).reshape(1,3,2,4)).astype(complex)
    calls=[]
    class Loader:
        nbands=3;nspinor=2
        def load(self,**kwargs):
            calls.append(kwargs)
            return jax.device_put(physical.copy(),NamedSharding(mesh,band_sphere_spec()))
    with mesh:
        result=load_psi_gflat_padded(Loader(),(0,4),mesh_xy=mesh,bispinor=True,
            bispinor_lift=PAULI_ZERO_SMALL_CARRIER)
        result.block_until_ready()
    actual=np.asarray(result)
    assert actual.shape==(1,4,4,4) and len(calls)==1
    assert calls[0]['bispinor'] is False and calls[0]['bispinor_lift']=='raw'
    np.testing.assert_array_equal(actual[:,:3,:2],physical)
    assert not np.any(actual[:,:,2:]) and not np.any(actual[:,3:])


def test_pauli_control_has_distinct_charge_receipt_and_refuses_current(monkeypatch):
    from common import parallel_transport
    from common.four_current_model import (PAULI_ZERO_SMALL_CARRIER,
        PAULI_ZERO_SMALL_PROVENANCE,resolve_four_current_representation)
    from file_io.wfn_basis import WavefunctionBasisReceipt
    from gw.gw_config import BispinorGWMode
    monkeypatch.setattr(parallel_transport,'fingerprint_from_binding',lambda *a:'a'*64)
    rep=resolve_four_current_representation(True,BispinorGWMode.COULOMB_ONLY,
        charge_carrier=PAULI_ZERO_SMALL_CARRIER)
    assert not rep.current_bispinor and not rep.scalar_head_bispinor
    kwargs=dict(wfn=SimpleNamespace(nbands=5,nspinor=2),wfn_fingerprint_binding=object(),
        role='charge',bispinor=True,bispinor_lift=rep.charge_lift,band_interval=(0,5),
        fft_grid=(4,4,4),centroid_fft_idx=np.array([[0,0,0],[1,1,1]]),
        n_rmu_logical=2,n_rmu_padded=2)
    pauli=WavefunctionBasisReceipt.from_bound_source(**kwargs)
    rkb=WavefunctionBasisReceipt.from_bound_source(**dict(kwargs,bispinor_lift='normalized_rkb'))
    assert pauli.nspinor_sampled==4 and pauli.bispinor_lift_provenance==PAULI_ZERO_SMALL_PROVENANCE
    with pytest.raises(ValueError):pauli.assert_same_carrier(rkb,where='paired source')
    with pytest.raises(ValueError,match='unknown sampled-spinor'):
        WavefunctionBasisReceipt.from_bound_source(**dict(kwargs,role='transverse'))
    with pytest.raises(ValueError,match='Coulomb-only'):
        resolve_four_current_representation(True,BispinorGWMode.BARE_TRANSVERSE,
            charge_carrier=PAULI_ZERO_SMALL_CARRIER)


@pytest.mark.parametrize('stored_carrier',['pauli2embed4','normalized_rkb'])
def test_public_restart_keeps_exact_declared_pauli_carrier(monkeypatch,stored_carrier):
    from common.four_current_model import resolve_four_current_representation
    from file_io import restart_bundle
    from gw import gw_init
    from gw.gw_config import BispinorGWMode
    points=np.array([[0,0,0],[1,1,1]])
    rep=resolve_four_current_representation(True,BispinorGWMode.COULOMB_ONLY,
        charge_carrier='pauli2embed4')
    stored=resolve_four_current_representation(True,BispinorGWMode.COULOMB_ONLY,
        charge_carrier=stored_carrier)
    monkeypatch.setattr(restart_bundle,'read_metadata',lambda _:dict(
        centroid_hashes={'charge':gw_init._centroid_table_md5(points),'current':None},
        charge_representation=stored.charge_representation,bispinor_gw='coulomb_only'))
    class Receipt:
        @staticmethod
        def from_bound_source(**kwargs):return kwargs
    meta=SimpleNamespace(nspinor=4,fft_grid=(4,4,4),mu_basis=SimpleNamespace(
        coordinate_kind='fft_indices'),n_rmu=2,n_rmu_padded=2)
    def invoke():return gw_init._restart_charge_basis(Receipt,(0,5),True,object(),
        points,None,meta,lambda *a:None,'metadata-only',object(),
        bispinor_gw=BispinorGWMode.COULOMB_ONLY,representation=rep)
    if stored_carrier=='normalized_rkb':
        with pytest.raises(ValueError,match='restart_bispinor_charge_carrier'):invoke()
    else:
        assert invoke()[1]['bispinor_lift']=='pauli2embed4'

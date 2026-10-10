"""Independent compact-target Gram/charge controls; no CrI3 or P4 certificate."""
import hashlib
import json

import numpy as np
import pytest

from psp.reconstruction_overlap import (COMPACT_PAULI_FRAME_MODEL,
    COMPACT_PAULI_FRAME_SCHEMA, COMPACT_PAULI_SUPPORT_POLICY,
    native_hermite_delta_gram, compact_frame_factor, load_compact_pauli_frame)
from psp.augmentation_cache import _payload_hash, load_paired_native_cache


def _native():
    r = np.array([.02,.17,.43,.81,1.])
    u = (r-r[0])*(1-r)**2*(1+.3j)
    du = ((1-r)**2-2*(r-r[0])*(1-r))*(1+.3j)
    return dict(r=r, weights_dr=np.full(len(r),.2), delta_u=u[:,None],
        delta_du_dr=du[:,None], l=np.array([0]), kappa=np.array([-1]),
        metadata=dict(source_sha256='a'*64,payload_sha256='b'*64))


def _physical_fixture():
    """Construct actual complex orbitals; the reference never expands C/D/B."""
    rng = np.random.default_rng(927)
    psi = (rng.normal(size=(2,3,2,8))+1j*rng.normal(size=(2,3,2,8)))/5
    radial = rng.normal(size=8); radial /= np.linalg.norm(radial)
    B = native_hermite_delta_gram(_native())
    delta = np.zeros((2,2,8),complex)
    delta[0,0] = np.sqrt(B[0,0].real)*radial
    delta[1,1] = np.sqrt(B[1,1].real)*radial
    dual = (rng.normal(size=(2,2,8))+1j*rng.normal(size=(2,2,8)))/7
    C = np.einsum('isg,pnsg->pni',dual.conj(),psi)
    D = np.einsum('isg,pnsg->pni',delta.conj(),psi)
    corrected = psi+np.einsum('pni,isg->pnsg',C,delta)
    S0 = np.einsum('pnsg,pmsg->pnm',psi.conj(),psi)
    G = np.einsum('pnsg,pmsg->pnm',corrected.conj(),corrected)
    eigenvalues, vectors = np.linalg.eigh(G)
    A = (vectors*eigenvalues[:,None]**-.5)@vectors.conj().swapaxes(-1,-2)
    arrays = dict(source_gram=S0,target_gram=G,inverse_sqrt=A,
        target_gram_GL8_control=G.copy(),atom_C_000=C,atom_D_000=D,
        species_B_1=B,species_labels_1=np.array([[0,-1],[0,1]]),
        atom_types=np.array([1]),centers_cart=np.array([[0.,0.,0.]]),
        parent_FILE_rows=np.arange(2),parent_k_frac=np.array([[0.,0.,0.],[1/3,0.,0.]]),
        ngk_valid_parent=np.full(2,8),source_pauli_sha256_by_parent=np.zeros((2,32),np.uint8),
        gvec_parent_000=np.column_stack((np.arange(8),np.zeros((8,2),int))),
        gvec_parent_001=np.column_stack((np.arange(8),np.zeros((8,2),int))))
    return psi,delta,C,D,corrected,dict(arrays=arrays)


def test_native_compact_hermite_gram_integrates_complex_degree_six_product():
    data = _native()
    # Independent analytic polynomial integral, including the first positive
    # endpoint. The native weighted table deliberately is not that integral.
    p = np.polynomial.Polynomial([-data['r'][0],1])*(np.polynomial.Polynomial([1,-1])**2)
    integral = (p*p).integ()
    exact = 1.09*(integral(1)-integral(data['r'][0]))
    B4 = native_hermite_delta_gram(data)
    np.testing.assert_allclose(B4,exact*np.eye(2),rtol=2e-14,atol=2e-16)
    np.testing.assert_allclose(B4,native_hermite_delta_gram(data,quadrature_order=8),rtol=2e-14,atol=2e-16)
    native_weighted = np.sum(data['weights_dr']*abs(data['delta_u'][:,0])**2)
    assert abs(native_weighted-exact)>1e-4


def test_common_A_complex_orientation_and_full_before_crop():
    psi,delta,C,D,corrected,frame = _physical_fixture()
    data = frame['arrays']
    packet = np.pad(data['source_gram'],((0,0),(0,1),(0,1)))
    receipt = compact_frame_factor(packet,[np.pad(C,((0,0),(0,1),(0,0)))],frame,
        parent_start=0,physical_bands=3)
    A = receipt['inverse_sqrt'][:,:3,:3]
    # Direct orbital action independently checks the complex column convention.
    normalized = np.einsum('pmn,pm...->pn...',A,corrected)
    Q = np.einsum('pnsg,pmsg->pnm',normalized.conj(),normalized)
    np.testing.assert_allclose(Q,np.broadcast_to(np.eye(3),Q.shape),atol=3e-14)
    np.testing.assert_array_equal(receipt['inverse_sqrt'][:,-1,-1],1.)
    wrong = np.einsum('pmn,pm...->pn...',A.conj(),corrected)
    assert np.max(abs(np.einsum('pnsg,pmsg->pnm',wrong.conj(),wrong)-np.eye(3)))>.01
    cropped = np.einsum('pmn,pm...->pn...',A[:,:2,:2],corrected[:,:2])
    assert np.max(abs(np.einsum('pnsg,pmsg->pnm',cropped.conj(),cropped)-np.eye(2)))>1e-4


def test_target_monopole_keeps_charge_without_renormalizing_served_fields():
    from isdf.atomic_moments import (exact_pair_moments, auxiliary_charge_geometry,
        evaluate_auxiliary_charge)
    _,delta,C,D,corrected,frame = _physical_fixture(); data=frame['arrays']; A=data['inverse_sqrt']
    c = np.einsum('pmn,pmi->pni',A,C); d=np.einsum('pmn,pmi->pni',A,D)
    raw = corrected-np.einsum('pni,isg->pnsg',C,delta)
    smooth = np.einsum('pmn,pm...->pn...',A,raw)
    whole = np.einsum('pmn,pm...->pn...',A,corrected)
    Q0 = np.einsum('pnsg,pmsg->pnm',smooth.conj(),smooth)
    target = np.stack([exact_pair_moments(ci,di,ci,di,data['species_B_1']) for ci,di in zip(c,d)])
    direct = np.einsum('pnsg,pmsg->pnm',whole.conj(),whole)-Q0
    np.testing.assert_allclose(target,direct,atol=4e-15)
    aux=auxiliary_charge_geometry(dict(upper=dict(ell=np.array([0]),kappa=np.array([-1])),
        labels=data['species_labels_1'],B=data['species_B_1']))
    plus,minus=evaluate_auxiliary_charge(c,d,aux,grid_sample_scale=1.)
    signed=np.einsum('knst,kmst,t->knm',plus.conj(),plus,aux['integration_weights'])
    signed-=np.einsum('knst,kmst,t->knm',minus.conj(),minus,aux['integration_weights'])
    np.testing.assert_allclose(signed,target,atol=3e-14)
    represented=smooth+1.03*np.einsum('pni,isg->pnsg',c,delta)
    served_Q=np.einsum('pnsg,pmsg->pnm',represented.conj(),represented)-Q0
    epsilon=(target-served_Q)/np.sqrt(4*np.pi)
    np.testing.assert_allclose(served_Q+np.sqrt(4*np.pi)*epsilon,target,atol=2e-16)
    assert np.max(abs(Q0+served_Q-np.eye(3)))>1e-4
    np.testing.assert_allclose(Q0+target,np.broadcast_to(np.eye(3),Q0.shape),atol=3e-14)


def test_exact_U_preserves_common_target_and_commutes_with_band_A():
    from psp.augmentation_spinors import _lift_cartesian
    _,_,_,_,corrected,frame=_physical_fixture(); A=frame['arrays']['inverse_sqrt']
    K=np.array([[0,0,0],[1,.4,0],[-2,.3,1],[4,-1,2],[10,3,-1],[50,0,0],[2,5,-7],[150,20,-4.]])
    normalized=np.einsum('pmn,pm...->pn...',A,corrected)
    four=np.stack([_lift_cartesian(psi,K) for psi in normalized])
    separate=np.stack([_lift_cartesian(psi,K) for psi in corrected])
    np.testing.assert_allclose(four,np.einsum('pmn,pm...->pn...',A,separate),atol=5e-16)
    np.testing.assert_allclose(np.einsum('pnsg,pmsg->pnm',four.conj(),four),
        np.broadcast_to(np.eye(3),(2,3,3)),atol=3e-14)


def test_bounded_graph_measurement_accepts_both_arms_and_exposes_wrong_lower():
    from common.collectives import single_device_mesh
    from jax.sharding import PartitionSpec as P
    from psp.augmentation_spinors import _lift_cartesian
    from gw.isdf_augmentation import _compact_pauli_gram_kernel,_put
    psi,_,_,_,_,frame=_physical_fixture()
    K=np.broadcast_to(np.array([[i,.3*i,-.2*i] for i in range(8)]),(2,8,3)).copy()
    mesh=single_device_mesh()
    momentum=_put(K,mesh,P(None,('x','y'),None))
    for carrier in ('pauli2embed4','normalized_rkb'):
        four=(np.concatenate((psi,np.zeros_like(psi)),axis=2) if carrier=='pauli2embed4'
            else np.stack([_lift_cartesian(row,k) for row,k in zip(psi,K)]))
        measure=_compact_pauli_gram_kernel(mesh,carrier)
        gram,error,_=measure(_put(four,mesh,P(None,None,None,('x','y'))),momentum)
        np.testing.assert_allclose(gram,frame['arrays']['source_gram'],atol=3e-14)
        assert float(error)<2e-15
        damaged=four.copy();damaged[:,:,2,3]+=.002j
        _,error,_=measure(_put(damaged,mesh,P(None,None,None,('x','y'))),momentum)
        assert float(error)>.001


@pytest.mark.parametrize('change,message',[
    ('source','source Gram'),('C','pseudo-dual C'),('A','Lowdin/isometry')])
def test_common_frame_rejects_live_source_phase_or_factor_mismatch(change,message):
    _,_,C,_,_,frame=_physical_fixture()
    source=frame['arrays']['source_gram'].copy(); coefficients=C.copy()
    if change=='source':source[:,0,0]+=.001
    elif change=='C':coefficients[:,0]*=np.exp(.03j)
    else:frame['arrays']['inverse_sqrt'][:,0,0]+=.001
    with pytest.raises(ValueError,match=message):
        compact_frame_factor(source,[coefficients],frame,parent_start=0,physical_bands=3)


def _write_frame(tmp_path,monkeypatch,change=None):
    _,_,_,_,_,frame=_physical_fixture(); arrays=frame['arrays']; native=_native()
    upf=tmp_path/'source.upf';upf.write_text('independent test source')
    bank=tmp_path/'native.npz';bank.write_bytes(b'opaque native owner fixture')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    native['metadata']['source_sha256']=sha(upf)
    species={'1':dict(native=str(bank),native_sha256=sha(bank),upf=str(upf),upf_sha256=sha(upf))}
    metadata=dict(schema=COMPACT_PAULI_FRAME_SCHEMA,model=COMPACT_PAULI_FRAME_MODEL,
        source_frame='original WFN Pauli band labels',complete_all_FILE_parents=True,physical_bands=3,
        wfn_sha256='c'*64,target_support_policy=COMPACT_PAULI_SUPPORT_POLICY,
        fourier_controls=dict(momentum_max=16.,momentum_points=4097,
            relative_tolerance=1e-10,absolute_tolerance=1e-12,validation_points=64),
        source_input_files_sha256={str(upf):sha(upf),str(bank):sha(bank)},atomic_species_inputs=species,
        native_target_radial_interval_by_species_bohr={'1':[.02,1.]})
    if change=='partial':metadata['complete_all_FILE_parents']=False
    elif change=='source':metadata['wfn_sha256']='d'*64
    elif change=='labels':arrays['species_labels_1']=arrays['species_labels_1'][::-1]
    elif change=='rows':arrays['parent_FILE_rows']=np.array([1,0])
    elif change=='B':arrays['species_B_1']=arrays['species_B_1']*1.01
    metadata['payload_sha256']=_payload_hash(arrays)
    path=tmp_path/'frame.npz';np.savez(path,**arrays,metadata_json=np.asarray(json.dumps(metadata)))
    monkeypatch.setattr('psp.atomic_reconstruction.load_atomic_reconstruction',lambda *args:native)
    return path,sha(path),native,arrays


@pytest.mark.parametrize('change',[None,'partial','source','labels','rows','B'])
def test_frame_file_refuses_incomplete_wrong_source_labels_or_non_target_B(tmp_path,monkeypatch,change):
    path,digest,native,data=_write_frame(tmp_path,monkeypatch,change)
    kwargs=dict(expected_file_sha256=digest,expected_wfn_sha256='c'*64,tables={1:native},
        parent_k_frac=np.array([[0,0,0],[1/3,0,0]]),gvecs=np.stack([data[f'gvec_parent_{i:03d}'] for i in range(2)]),
        ngk_valid=np.full(2,8),atom_types=np.array([1]),centers_cart=np.zeros((1,3)),physical_bands=3)
    if change is None:
        result=load_compact_pauli_frame(path,**kwargs)
        assert result['metadata']['model']==COMPACT_PAULI_FRAME_MODEL
    else:
        with pytest.raises(ValueError):load_compact_pauli_frame(path,**kwargs)


def test_paired_field_loader_refuses_legacy_compact_after_U_before_using_arrays(tmp_path):
    path=tmp_path/'legacy.npz'
    np.savez(path,field_model=np.asarray('compact_upper_hermite_sigma_gradient'),
        metadata_json=np.asarray(json.dumps(dict(schema='lorrax.normalized_augmentation_cache.v2'))))
    with pytest.raises(ValueError,match='policy/source/spectrum/payload'):
        load_paired_native_cache(path,_native(),expected_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            common_spectrum_sha256='e'*64,carrier='normalized_rkb',support_radius=2.55)


@pytest.mark.parametrize('change',['wrong_policy','legacy_cache','legacy_served','legacy_target'])
def test_unbound_paired_manifest_refuses_unknown_policy_or_legacy_mix(tmp_path,change):
    from gw.isdf_augmentation import read_augmentation_manifest
    from psp.augmentation_cache import PAIRED_COMPACT_PAULI_FIELD_POLICY
    manifest=dict(schema='lorrax.isdf_augmentation.v1',carrier='normalized_rkb',
        frozen_core_policy='reconstruct_valence_only',field_policy=PAIRED_COMPACT_PAULI_FIELD_POLICY,
        overlap=dict(mode='full_wfn_lowdin'),charge_metric=dict(moment_enrichment='served_monopole'),cache={})
    if change=='wrong_policy':manifest['field_policy']='compact_after_U'
    elif change=='legacy_cache':manifest['cache']['species_files']={}
    elif change=='legacy_served':manifest['served_moments']={}
    else:manifest['cache']['target']={}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='forbids legacy'):
        read_augmentation_manifest(tmp_path)


def test_unbound_paired_stage_refuses_before_any_scientific_owner(monkeypatch):
    from gw.isdf_augmentation import prepare_augmentation
    from psp.augmentation_cache import PAIRED_COMPACT_PAULI_FIELD_POLICY
    # None sentinels cannot support any source/header/scientific operation.
    with pytest.raises(ValueError,match='unbound paired artifact'):
        prepare_augmentation(wfn=None,sym=None,meta=None,cfg=None,mesh_xy=None,plan=None,
            centroid_indices=None,parent_psi=None,parent_faces=None,
            band_range_left=None,band_range_right=None,
            artifact=dict(field_policy=PAIRED_COMPACT_PAULI_FIELD_POLICY))


@pytest.mark.parametrize('change',[None,'drop_Cr_f','invent_Cr_selection','I_missing_channel'])
def test_per_species_policy_keeps_complete_Cr_and_authenticated_selected_I(tmp_path,monkeypatch,change):
    from gw.isdf_augmentation import read_augmentation_manifest
    from psp.augmentation_cache import PAIRED_COMPACT_PAULI_FIELD_POLICY
    channels=[[0,-1],[1,-2],[1,1],[2,-3],[2,2],[3,-4],[3,3]]
    controls={'24':dict(mode='complete_unselected_sidecar',channels=channels),
        '53':dict(mode='native_oncv_reference_channels',lmax=2)}
    banks,sources={},{}
    for z,rows in ((24,channels),(53,channels[:5])):
        source=dict(source_sha256='a'*64,operator_sha256='b'*64,generator_input_sha256='c'*64,
            frozen_configuration_sha256='d'*64,pseudo_type='NC',relativistic='full',has_so='T',
            atomic_number=z,spin_channels=[dict(lll=l,jjj=-k-.5 if k<0 else k-.5) for l,k in rows[:5]])
        meta=dict(operator_comparison=dict(source=source),atomic_number=z,payload_sha256='e'*64)
        if z==53:
            meta['radial_channel_selection']=dict(mode='native_oncv_reference_channels',lmax=2,
                retained_channels=[dict(l=l,kappa=k) for l,k in rows],
                parent_payload_sha256='f'*64,parent_metadata_sha256='0'*64)
        banks[z]=dict(l=np.array([r[0] for r in rows]),kappa=np.array([r[1] for r in rows]),metadata=meta)
        sources[z]=source
    if change=='drop_Cr_f':controls['24']['channels']=channels[:5]
    elif change=='invent_Cr_selection':banks[24]['metadata']['radial_channel_selection']={}
    elif change=='I_missing_channel':banks[53]['l']=banks[53]['l'][:-1];banks[53]['kappa']=banks[53]['kappa'][:-1]
    monkeypatch.setattr('psp.atomic_reconstruction.load_atomic_reconstruction',
        lambda path,upf:banks[int(path.stem)])
    monkeypatch.setattr('psp.atomic_reconstruction.upf_identity',lambda path:sources[int(path.stem)])
    manifest=dict(schema='lorrax.isdf_augmentation.v1',carrier='normalized_rkb',
        frozen_core_policy='reconstruct_valence_only',field_policy=PAIRED_COMPACT_PAULI_FIELD_POLICY,
        overlap=dict(mode='full_wfn_lowdin'),partial_wave_channels=dict(species=controls),
        species={str(z):dict(source_upf=f'{z}.upf',reconstruction=f'{z}.npz') for z in banks},
        radial=dict(interpolation_degree=5),angular={},cache={},runtime={},
        charge_metric=dict(smooth_neutral_cross='onsite',moment_enrichment='served_monopole'))
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    if change is None:
        artifact=read_augmentation_manifest(tmp_path)
        assert set(zip(artifact['tables'][24]['l'],artifact['tables'][24]['kappa']))==set(map(tuple,channels))
        assert artifact['normalized_caches'] is None and artifact['raw_parent_moments'] is None
    else:
        with pytest.raises(ValueError):read_augmentation_manifest(tmp_path)

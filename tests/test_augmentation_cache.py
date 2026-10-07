"""Immutable artifact checks; normalized carrier mathematics has its own tests."""
import json
import numpy as np
import pytest

from runtime import bootstrap
bootstrap()

from psp.augmentation_cache import write_normalized_cache, load_normalized_cache
from psp.augmentation_spinors import COMPACT_GRAPH_FIELD_MODEL
from common.bispinor_init import HALFALPHA


def fixture_cache():
    control = dict(momentum_max=30., momentum_points=24, momentum_quadrature="gauss_legendre",
                   radius_kind="log", radius_min=1e-7, radius_max=1.8, radius_points=32,
                   tail_relative_tolerance=1., taper_start=1.)
    radius = np.concatenate(([0.], np.geomspace(1e-7, 1.8, 31)))
    data = dict(r=np.asarray((.1, .9)), l=np.asarray((0, 1)), kappa=np.asarray((-1, 1)),
                metadata=dict(source_sha256="a"*64, payload_sha256="b"*64,
                              operator_comparison={"authenticated": True}, phase_branch_validated=True))
    value = np.asarray(np.exp(-radius[:, None])*np.asarray((1., 0.7))[None], dtype=np.complex128)
    cache = dict(radius=radius, ell=data["l"], kappa=data["kappa"], large_R=value,
                 dlarge_R_dr=-value, small_R=0.1j*value, dsmall_R_dr=-0.1j*value,
                 field_model=np.asarray(COMPACT_GRAPH_FIELD_MODEL), taper_start=np.asarray(1.),
                 support_radius=np.asarray(1.2), half_alpha=np.asarray(float(HALFALPHA)))
    return data, control, cache


def test_cache_roundtrip_refuses_different_atomic_metadata_controls_and_support(tmp_path):
    data, control, cache = fixture_cache()
    path = tmp_path / "normalized.npz"
    write_normalized_cache(path, cache, data, control, support_radius=1.2)
    actual = load_normalized_cache(path, data, dict(control, species_files={"1": "elsewhere"}), support_radius=1.2)
    for key in cache:
        np.testing.assert_array_equal(actual[key], cache[key])
    changed = dict(data, metadata=dict(data["metadata"], source_sha256="c"*64))
    with pytest.raises(ValueError, match="provenance mismatch"):
        load_normalized_cache(path, changed, control, support_radius=1.2)
    changed = dict(data, metadata=dict(data["metadata"], extra_generator_information="changed"))
    with pytest.raises(ValueError, match="provenance mismatch"):
        load_normalized_cache(path, changed, control, support_radius=1.2)
    with pytest.raises(ValueError, match="provenance mismatch"):
        load_normalized_cache(path, data, dict(control, momentum_max=31.), support_radius=1.2)
    with pytest.raises(ValueError, match="provenance mismatch"):
        load_normalized_cache(path, data, control, support_radius=1.3)


def test_cache_refuses_payload_and_metadata_poison_without_rebuild(tmp_path):
    data, control, cache = fixture_cache()
    path = tmp_path / "normalized.npz"
    write_normalized_cache(path, cache, data, control, support_radius=1.2)
    with np.load(path, allow_pickle=False) as source:
        changed = {key: source[key] for key in source.files}
    changed["large_R"] = changed["large_R"] + 0.1
    poisoned = tmp_path / "poisoned.npz"
    np.savez(poisoned, **changed)
    with pytest.raises(ValueError, match="payload checksum"):
        load_normalized_cache(poisoned, data, control, support_radius=1.2)
    metadata = json.loads(str(changed["metadata_json"]))
    metadata["binding"]["carrier"] = "raw"
    changed["metadata_json"] = np.asarray(json.dumps(metadata))
    poisoned_metadata = tmp_path / "poisoned_metadata.npz"
    np.savez(poisoned_metadata, **changed)
    with pytest.raises(ValueError, match="metadata checksum"):
        load_normalized_cache(poisoned_metadata, data, control, support_radius=1.2)
    missing = tmp_path / "missing.npz"
    with pytest.raises(FileNotFoundError):
        load_normalized_cache(missing, data, control, support_radius=1.2)
    assert not missing.exists()


def test_cache_writer_rejects_wrong_grid_labels_dtype_and_nonfinite_values(tmp_path):
    data, control, cache = fixture_cache()
    changes = (("radius", cache["radius"]*1.001), ("ell", np.asarray((0, 2))),
               ("large_R", cache["large_R"].astype(np.complex64)),
               ("small_R", np.full_like(cache["small_R"], np.nan)))
    for index, (name, replacement) in enumerate(changes):
        with pytest.raises(ValueError):
            write_normalized_cache(tmp_path / f"wrong{index}.npz", dict(cache, **{name: replacement}),
                                   data, control, support_radius=1.2)


def test_cache_cannot_be_overwritten(tmp_path):
    data, control, cache = fixture_cache()
    path = tmp_path / "normalized.npz"
    write_normalized_cache(path, cache, data, control, support_radius=1.2)
    with pytest.raises(FileExistsError):
        write_normalized_cache(path, cache, data, control, support_radius=1.2)


def test_cache_requires_explicit_native_preserving_compact_descriptor(tmp_path):
    data, control, cache = fixture_cache()
    with pytest.raises(ValueError, match='explicit taper_start'):
        write_normalized_cache(tmp_path/'missing.npz', cache, data,
            {key: value for key, value in control.items() if key != 'taper_start'}, support_radius=1.2)
    with pytest.raises(ValueError, match='native reconstruction sphere'):
        write_normalized_cache(tmp_path/'core.npz', cache, data, dict(control, taper_start=.8), support_radius=1.2)
    for key, value in (('field_model', np.asarray('hard_mask')), ('half_alpha', np.asarray(2*HALFALPHA)),
                       ('taper_start', np.asarray(.7))):
        with pytest.raises(ValueError, match='descriptor'):
            write_normalized_cache(tmp_path/f'{key}.npz', dict(cache, **{key: value}), data, control, support_radius=1.2)


def target_fixture(tmp_path):
    """Small genuine paired-input files, with unchanged native projection rows."""
    import hashlib
    from psp.atomic_reconstruction import write_atomic_reconstruction, load_atomic_reconstruction
    from psp.augmentation_cache import AE_LARGE_TARGET
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    upf = tmp_path/'source.upf'
    upf.write_text('<UPF><PP_HEADER element="H" pseudo_type="NC" relativistic="full" '
        'has_so="T" functional="PBE" z_valence="1"/><PP_INFO><PP_INPUTFILE>'
        'H 1 0 1\n1 0 1\n</PP_INPUTFILE></PP_INFO><PP_MESH><PP_R>.1 .5</PP_R></PP_MESH><PP_NONLOCAL>'
        '<PP_BETA.1 cutoff_radius="0.6">1 2</PP_BETA.1></PP_NONLOCAL></UPF>')
    r = np.geomspace(1e-5, .9, 65)
    ell, kappa = np.array((0, 1)), np.array((-1, 1))
    ae = r[:,None]**(ell[None]+1)*np.exp(-12*r[:,None]**2)
    dae = ae*((ell[None]+1)/r[:,None]-24*r[:,None])
    arrays = dict(r=r,l=ell,kappa=kappa,ps_u=np.zeros_like(ae),ps_du_dr=np.zeros_like(ae),
        ae_u=ae,ae_du_dr=dae,delta_R=ae/r[:,None],weights_dr=np.ones_like(r),
        channel_l=ell,channel_kappa=kappa,channel_nopf=np.ones(2,dtype=int),
        channel_0_coefficients=np.ones((1,1)),channel_1_coefficients=np.ones((1,1)))
    native = tmp_path/'native.npz'
    metadata = dict(source_sha256=sha(upf),operator_comparison={'authenticated':True},phase_branch_validated=True)
    write_atomic_reconstruction(native,arrays,metadata)
    data = load_atomic_reconstruction(native,upf)
    coeff_sha = hashlib.sha256(np.ones((1,1)).tobytes()).hexdigest()
    qchannels = [dict(l=int(l),kappa=int(k),rank=1,PCA_coefficients_sha256=coeff_sha) for l,k in zip(ell,kappa)]
    pchannels = [dict(ell=int(l),kappa=int(k),rank=1,coefficients_sha256=coeff_sha) for l,k in zip(ell,kappa)]
    qmeta = dict(data['metadata'],dev_matched_dirac_Q_OPF=dict(schema='lorrax.dev.matched_dirac_q_opf.v1',
        native_payload_sha256=data['metadata']['payload_sha256'],source_UPF_sha256=sha(upf),
        raw_bank_sha256='a'*64,channels=qchannels))
    paired = tmp_path/'paired.npz'
    write_atomic_reconstruction(paired,dict(arrays,ae_small_u=0*ae,ae_small_du_dr=0*ae,
        channel_0_training_energies_ps_ha=np.array((.1,)),channel_1_training_energies_ps_ha=np.array((.2,))),qmeta)
    er = np.r_[r,np.linspace(1.,1.6,17)]
    exterior = tmp_path/'exterior.npz'
    pmeta = dict(schema='lorrax.dev.native_ps_exterior.v1',source_UPF_sha256=sha(upf),
        native_payload_sha256=data['metadata']['payload_sha256'],bank_sha256='a'*64,channels=pchannels)
    np.savez(exterior,r=er,ps_u=np.zeros((len(er),2)),ps_du_dr=np.zeros((len(er),2)),
        ell=ell,kappa=kappa,channel_0_training_energies=np.array((.1,)),
        channel_1_training_energies=np.array((.2,)),metadata_json=np.asarray(json.dumps(pmeta)))
    entry = dict(nuclear_charge=1,matched_dirac_file=str(paired),matched_dirac_sha256=sha(paired),
        pseudo_exterior_file=str(exterior),pseudo_exterior_sha256=sha(exterior),
        source_upf_file=str(upf),source_upf_sha256=sha(upf),dirac_window_start=.6,
        dirac_window_stop=.9,completion_start=1.1,completion_stop=1.5)
    control = dict(momentum_max=30.,momentum_points=128,momentum_quadrature='gauss_legendre',
        source_quadrature_order=12,radius_kind='linear',radius_max=1.8,radius_points=128,taper_start=1.,
        target=dict(kind=AE_LARGE_TARGET,species={sha(upf):entry}))
    return data,control,entry


def test_explicit_target_roundtrip_and_semantic_separation(tmp_path):
    from psp.augmentation_cache import build_normalized_cache,normalized_cache_binding,AE_LARGE_TARGET,NATIVE_PAULI_TARGET
    data,control,entry = target_fixture(tmp_path)
    cache = build_normalized_cache(data,control,support_radius=1.2)
    path = tmp_path/'target.npz'
    meta = write_normalized_cache(path,cache,data,control,support_radius=1.2)
    assert meta['binding']['target_kind'] == AE_LARGE_TARGET
    restored = load_normalized_cache(path,data,control,support_radius=1.2)
    assert all(np.array_equal(cache[k],restored[k]) for k in cache)
    default = {k:v for k,v in control.items() if k!='target'}
    assert normalized_cache_binding(data,default,support_radius=1.2)['target_kind'] == NATIVE_PAULI_TARGET
    with pytest.raises(ValueError,match='provenance mismatch'):
        load_normalized_cache(path,data,default,support_radius=1.2)


@pytest.mark.parametrize('mutation',('kind','unknown_key','missing_source','relative_path','charge','window','support','file_sha','native_phase','pca'))
def test_target_refuses_wrong_model_source_native_phase_and_window(tmp_path,mutation):
    import copy,hashlib
    from psp.augmentation_cache import build_normalized_cache
    from psp.atomic_reconstruction import load_atomic_reconstruction,write_atomic_reconstruction
    data,control,entry = target_fixture(tmp_path)
    control = copy.deepcopy(control)
    e = next(iter(control['target']['species'].values()))
    if mutation=='kind': control['target']['kind']='free_dirac_positive_spectral'
    elif mutation=='unknown_key': e['normalization']='renormalize_every_row'
    elif mutation=='missing_source': control['target']['species']={'a'*64:dict(e,source_upf_sha256='a'*64)}
    elif mutation=='relative_path': e['matched_dirac_file']='paired.npz'
    elif mutation=='charge': e['nuclear_charge']=2
    elif mutation=='window': e['dirac_window_start']=.59
    elif mutation=='support': e['completion_stop']=1.7
    elif mutation=='file_sha': e['matched_dirac_sha256']='a'*64
    else:
        q = load_atomic_reconstruction(e['matched_dirac_file'],e['source_upf_file'])
        a = {k:v.copy() for k,v in q.items() if k!='metadata'}
        key = 'ae_u' if mutation=='native_phase' else 'channel_0_coefficients'
        a[key] = -a[key]
        poison = tmp_path/'changed_pair.npz'
        write_atomic_reconstruction(poison,a,q['metadata'])
        e['matched_dirac_file']=str(poison)
        e['matched_dirac_sha256']=hashlib.sha256(poison.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        build_normalized_cache(data,control,support_radius=1.2)


def test_offline_target_lower_phase_matches_independent_radial_gradient():
    from scipy.special import roots_legendre
    from psp.augmentation_cache import _inverse_pauli_spectrum
    K,w = roots_legendre(96);K,w=15*(K+1),15*w
    ell=np.array((0,1,1,2,2));kappa=np.array((-1,1,-2,2,-3))
    spectrum=np.column_stack([K**l*np.exp(-K*K/8) for l in ell])
    r=np.linspace(.03,1.5,37)
    cache=_inverse_pauli_spectrum(spectrum,ell,kappa,K,w,r)
    # The independent angular Dirac identity gives i h(R'+(kappa+1)R/r).
    expected=1j*HALFALPHA*(cache['dlarge_R_dr']+(kappa[None]+1)*cache['large_R']/r[:,None])
    np.testing.assert_allclose(cache['small_R'],expected,atol=2e-16,rtol=2e-12)
    assert np.max(abs(cache['small_R']+expected)) > 1e-3


def test_target_load_refuses_rebound_changed_Q_or_exterior_primitive(tmp_path):
    import copy,hashlib
    from psp.augmentation_cache import build_normalized_cache
    from psp.atomic_reconstruction import load_atomic_reconstruction,write_atomic_reconstruction
    data,control,_ = target_fixture(tmp_path)
    cache=build_normalized_cache(data,control,support_radius=1.2)
    path=tmp_path/'served_target.npz'
    write_normalized_cache(path,cache,data,control,support_radius=1.2)
    changed=copy.deepcopy(control);entry=next(iter(changed['target']['species'].values()))
    q=load_atomic_reconstruction(entry['matched_dirac_file'],entry['source_upf_file'])
    arrays={k:v.copy() for k,v in q.items() if k!='metadata'}
    arrays['ae_small_u']+=.1*arrays['ae_u']
    new=tmp_path/'different_Q.npz'
    write_atomic_reconstruction(new,arrays,q['metadata'])
    entry['matched_dirac_file']=str(new);entry['matched_dirac_sha256']=hashlib.sha256(new.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='provenance mismatch'):
        load_normalized_cache(path,data,changed,support_radius=1.2)
    changed=copy.deepcopy(control);entry=next(iter(changed['target']['species'].values()))
    with np.load(entry['pseudo_exterior_file'],allow_pickle=False) as z:
        arrays={k:z[k].copy() for k in z.files}
    arrays['ps_du_dr'][0,0]=1e-4
    new=tmp_path/'different_exterior.npz';np.savez(new,**arrays)
    entry['pseudo_exterior_file']=str(new);entry['pseudo_exterior_sha256']=hashlib.sha256(new.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='native PS amplitudes or derivatives'):
        load_normalized_cache(path,data,changed,support_radius=1.2)

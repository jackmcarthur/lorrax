import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('pencil_analysis',Path(__file__).with_name('analyze_general_pencil.py'))
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)


def test_weak_descriptor_direction_can_give_unobservable_large_growth_without_proving_material_gain(tmp_path):
    # An EXACTLY dark second coordinate. Union rank sees A, even though E is
    # poorly scaled. This is a possibility witness, not material evidence.
    e=np.diag([1.,1e-11]).astype(complex);a=np.diag([-.1j,12j*1e-11])
    b=np.array([[1.],[0.]],complex);c=b.T.copy()
    sites=np.array([-.2+.3j,.4+.2j]);target=(1/(sites+.1j))[:,None,None]
    capture=tmp_path/'tiny.npz'
    np.savez(capture,E=e,A=a,B=b,C=c,sites_ry=sites,target=target,C1=np.ones((1,1)),C2=np.array([[-.1j]]))
    report=helper.analyze(capture,max_order=2,tolerances=(1e-9,1e-10,1e-12))
    stable,unresolved,growing=report['studies']
    assert stable['selected_order']==1 and unresolved['selected_order']==growing['selected_order']==2
    assert stable['transfer']['maximum_growth_ry']==pytest.approx(-.1)
    # Union selection can expose a descriptor direction that the CURRENT
    # balance correctly refuses. Only a resolving diagnostic tolerance may
    # recover the dark growing pole; production gates are never bypassed.
    assert unresolved['status']=='refused' and 'shared_general_balance' in unresolved['error']
    assert growing['transfer']['maximum_growth_ry']==pytest.approx(12.)
    assert growing['transfer']['largest_growth'][0]['residue_frobenius']==0.
    assert max(p['absolute_target_defect'] for p in growing['transfer']['probes'])<1e-14
    assert growing['projected_E_singular_values']['condition']==pytest.approx(1e11)


def test_no_actions_means_descriptor_only_and_capacity_refuses_before_svd(tmp_path):
    capture=tmp_path/'small.npz';np.savez(capture,E=np.eye(2),A=np.diag([-.1j,-.3j]))
    with pytest.raises(ValueError,match='Bounded analysis refuses'):
        helper.analyze(capture,max_order=2,max_workspace_bytes=1)
    report=helper.analyze(capture,max_order=2,tolerances=(1e-8,))
    assert 'DESCRIPTOR COORDINATES ONLY' in report['transfer_scope']
    assert report['studies'][0]['status']=='reduced-for-diagnostic-only'
    assert 'NOT physical W' in report['studies'][0]['transfer']['scope']


def test_progress_records_completed_ranks_without_claiming_admission(tmp_path):
    capture=tmp_path/'progress.npz'
    np.savez(capture,E=np.eye(2),A=np.diag([-.1j,-.3j]))
    records=[]
    report=helper.analyze(capture,max_order=2,tolerances=(1e-8,),record=records.append)
    assert len(records)==3
    assert records[0]['studies']==[]
    assert records[1]['status']=='RUNNING_DIAGNOSTIC_NOT_ADMITTED'
    assert len(records[1]['studies'])==1
    assert records[-1]['status']==report['status']=='COMPLETE_DIAGNOSTIC_NOT_ADMITTED'


def test_cli_creates_exclusively_and_preserves_completed_report(tmp_path, monkeypatch, capsys):
    import json
    capture=tmp_path/'cli.npz';output=tmp_path/'report.json'
    np.savez(capture,E=np.eye(2),A=np.diag([-.1j,-.3j]))
    monkeypatch.setattr('sys.argv',['analyze_general_pencil.py','--input',str(capture),
        '--output',str(output),'--max-order','2','--tolerances','1e-8'])
    helper.main()
    capsys.readouterr()
    saved=output.read_bytes()
    assert json.loads(saved)['status']=='COMPLETE_DIAGNOSTIC_NOT_ADMITTED'
    assert not list(tmp_path.glob('*.pending'))
    with pytest.raises(FileExistsError):
        helper.main()
    assert output.read_bytes()==saved


def test_frequency_only_grams_and_mixed_actions_equal_literal_complex_full_port_norms(tmp_path):
    from gw.shared_pole_constructor import reduce_general_pencil
    rng=np.random.default_rng(175)
    e=np.eye(3,dtype=complex);a=np.diag([-.4-.1j,.6-.2j,1.3-.3j])
    b=rng.normal(size=(3,7))+1j*rng.normal(size=(3,7))
    c=rng.normal(size=(7,3))+1j*rng.normal(size=(7,3))
    sites=np.array([.1+.5j,.8+.4j]);model=reduce_general_pencil(e,a,max_order=3)
    exact=np.stack([c@la_solve(z*e-a,b) for z in sites])
    # Add a known independent nonzero target error to avoid a cancellation-
    # dominated test of an exactly zero norm expressed as norm²+norm²−2inner.
    target=exact+.03*(rng.normal(size=exact.shape)+1j*rng.normal(size=exact.shape))
    mixed=np.stack([c.conj().T@w@b.conj().T for w in target])
    result=helper.gram_transfer_summary(model,c.conj().T@c,b@b.conj().T,sites,
        probe_sites=sites,mixed=mixed,exact_sq=np.linalg.norm(target,axis=(1,2))**2)
    for n,row in enumerate(result['probes']):
        assert row['transfer_frobenius']==pytest.approx(np.linalg.norm(exact[n]),rel=2e-14)
    for n,row in enumerate(result['exact_probe_defects']):
        assert row['absolute_target_defect']==pytest.approx(np.linalg.norm(exact[n]-target[n]),rel=2e-11)
    capture=tmp_path/'gram.npz'
    np.savez(capture,E=e,A=a,GC=c.conj().T@c,GB=b@b.conj().T,sites_ry=sites,
        probe_sites_ry=sites,probe_mixed_J=mixed,probe_exact_norm_sq=np.linalg.norm(target,axis=(1,2))**2)
    report=helper.analyze(capture,max_order=3,tolerances=(1e-9,))
    assert report['transfer_scope']=='RAW UNSYMMETRIZED full-port norms via frequency-only GC/GB'
    assert 'crossing-mirror interference' in report['bank_scope']
    assert 'NOT physical W' in report['studies'][0]['transfer']['scope']
    assert len(report['studies'][0]['physical_transfer']['exact_probe_defects'])==2


def la_solve(a,b):
    return np.linalg.solve(a,b)


def test_descriptor_svd_is_distinct_accuracy_resolved_basis_without_owner_mutation(tmp_path):
    import hashlib
    from gw.shared_pole_constructor import reduce_general_pencil
    owner_path=Path(reduce_general_pencil.__code__.co_filename)
    before=hashlib.sha256(owner_path.read_bytes()).hexdigest()
    e=np.diag([1.,1e-11]).astype(complex);a=np.diag([-.1j,.5j])
    b=np.array([[1.],[0.]],complex);c=b.T.copy()
    capture=tmp_path/'basis.npz';np.savez(capture,E=e,A=a,B=b,C=c)
    union=helper.analyze(capture,max_order=2,tolerances=(1e-8,),basis='union')
    records=[]
    descriptor=helper.analyze(capture,max_order=2,tolerances=(1e-8,),basis='descriptor',record=records.append)
    assert union['studies'][0]['selected_order']==2
    row=descriptor['studies'][0]
    assert row['selected_order']==1 and row['transfer']['maximum_growth_ry']==pytest.approx(-.1)
    assert row['projected_E_singular_values']['condition']<=1e8
    assert descriptor['basis_method']=='two-sided SVD(E)'
    assert descriptor['left_union_singular_values'] is None
    assert descriptor['production_reducer_sha256']==before==hashlib.sha256(owner_path.read_bytes()).hexdigest()
    assert records[0]['status']=='RUNNING_DIAGNOSTIC_NOT_ADMITTED'
    assert records[-1]['status']=='COMPLETE_DIAGNOSTIC_NOT_ADMITTED'

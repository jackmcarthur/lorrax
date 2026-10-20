"""Cheap campaign-instrumentation tests; no GW evaluation or trajectory."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

import observer


def test_native_observer_captures_accepted_tables_and_refuses_bad_density(tmp_path):
    from tdagw import snapshot_runtime as sr, stages
    nk, na = 2, 2
    eye=np.broadcast_to(np.eye(na),(nk,na,na)).copy().astype(complex)
    basis=dict(U=eye,e=np.zeros((nk,na)),rho=.5*eye)
    pairs={branch:np.stack([eye*(.03-.01j)]*2) for branch in ('cond','val')}
    row=dict(gamma=.02*eye,injection=.01*eye,m_h=.03*eye,
             raw={name:pairs for name in ('retarded_pair','active_branch_pair','branch_pair')})
    ops=SimpleNamespace(w_k=np.full(nk,.5),rho_eq=.5*eye)
    rec=dict(number_residual_rel=[0.],number_residual_per_fs=[0.])
    state=dict(n=0,t_last=0.,basis_last=[basis],snapshot_held=np.zeros((1,nk,na,na)),
               history=[rec],hf_reference_ry=np.zeros((nk,na,na)),one_body_counterterm_ry=np.zeros((nk,na,na)))
    def factory(*args):
        return (lambda b,**kw:row),{}
    def snapshot(*args,**kw):
        evaluate,_=sr.make_native_snapshot_evaluator(None,None,None)
        evaluate(basis)
        return dict(state=state,bundle_of=np.arange(nk),c_eq=np.zeros((1,nk,na,na)),
                    eq=row,refresh=lambda *a,**k:None)
    calls=[]
    import time
    with patch.object(sr,'make_native_snapshot_evaluator',factory),patch.object(sr,'snapshot_collision',snapshot),\
            patch.object(stages,'_write_chunk',lambda *a,**k:calls.append(True)):
        restore=observer.install(tmp_path,start=time.perf_counter(),budget_s=30.,tf_fs=10.,
            refresh_fs=.5,storage_gib=.01,limits=dict(rho_min=-.001,rho_max=1.001,
            trace_error_electrons_per_cell=.001,fresh_trace_relative=1e-8,fresh_trace_per_fs=1e-8))
        try:
            sr.snapshot_collision(None,ops)
            z=np.load(tmp_path/'campaign_diagnostics/snapshot_0000.npz')
            assert np.array_equal(z['U'],basis['U'][None])
            assert np.array_equal(z['M_H_current_ry'],row['m_h'][None])
            assert np.array_equal(z['retarded_pair_cond_ry'],pairs['cond'][None])
            from common.units import RYD_TO_EV
            from tdagw.tdgw_io import HA_IN_EV
            assert np.allclose(z['Gamma_ha'],row['gamma'][None]*RYD_TO_EV/HA_IN_EV)
            system=SimpleNamespace(nk=nk)
            rho=.5*eye[None]
            stages._write_chunk(None,system,0,1,{},lambda:(rho,),0.,None)
            bad=rho.copy();bad[0,0,0,0]=-.01
            with pytest.raises(ValueError,match='campaign_density_bounds'):
                stages._write_chunk(None,system,1,2,{},lambda:(bad,),1.,None)
            assert len(calls)==2
            health=json.loads((tmp_path/'campaign_diagnostics/checkpoint_health.json').read_text())
            assert health[-1]['rho_min'] == -.01
        finally:
            restore()


def test_prepared_native_decks_have_no_prototype_controls(tmp_path, monkeypatch, capsys):
    import prepare
    from tdagw.td_config import read_td_deck
    source=tmp_path/'source.in'
    source.write_text('[cohsex]\nnumber_bands = 160\nsigma_w_model = shared_pole\n'
                      'sigma_regularization_ev = 0.1\n')
    root=tmp_path/'prepared'
    monkeypatch.setattr('sys.argv',['prepare.py','--out',str(root),'--source',str(source),
        '--ref',str(tmp_path/'bundle'),'--source-tree','a'*40])
    prepare.main()
    capsys.readouterr()
    forbidden=('collision_density_rank','collision_pole_order','collision_projection_order',
               'collision_pencil_mb','collision_fit_samples','collision_fit_height_ev')
    decks=list(root.glob('*.toml'))
    assert len(decks)==6
    for path in decks:
        text=path.read_text()
        assert all(key not in text for key in forbidden)
        deck=read_td_deck(str(path))
        assert deck.values['lorrax.collision_hamiltonian']=='hf'
        assert deck.values['lorrax.collision']=='raw'
        assert deck.values['lorrax.collision_statistics']=='retarded_reference'


def test_plot_reader_crops_checkpoint_and_authenticates_native(tmp_path):
    import h5py
    import plot_full_native as plot
    out=tmp_path/'E1e+10_joint'
    diag=out/'campaign_diagnostics';diag.mkdir(parents=True)
    native=dict(construction='full-microscopic-general-shared-pole',n_mu=978,n_q=36,
                screening_bands=160,spatial_projection=False,screening_density='full_matrix')
    (diag/'snapshot_0000.json').write_text(json.dumps(dict(q_receipts=native)))
    (tmp_path/'runs_E1e+10_joint.json').write_text(json.dumps(dict(source_tree='test-tree',
        status='REFUSED_NO_COMPLETION_CLAIM',runs=[dict(tag=out.name,status='REFUSED_PARTIAL_NOT_COMPLETED')])))
    times=np.arange(5)*.05*plot.FS_AU
    with h5py.File(out/'tdgw.h5','w') as f:
        cp=f.create_group('tdgw_data/checkpoint');cp['it_next']=3;cp['tchkp']=.15*plot.FS_AU
        td=f.create_group('tdgw_data/time_domain')
        for name,shape in [('polarization',(5,3)),('Efield',(5,3)),('carrier_density',(5,2)),('rho_full',(5,36,4,4))]:
            g=td.create_group(name);g['times']=times
            data=np.zeros(shape,complex)
            if name=='rho_full':data[:3]=.5*np.eye(4)
            data[3:]=999 # MUST never enter plotted samples or metrics.
            g['data']=data
        h=f.create_group('tdgw_header/wfn_header');h['w_k']=np.ones(36)/36;h['kpts']=np.zeros((36,3))
        b=h.create_group('bands');b['start']=129;b['stop']=133;b['step']=1
    with pytest.raises(ValueError,match='not authenticated complete'):
        plot.read_run(out,out.name,'C',10.,False,'test-tree')
    report,data=plot.read_run(out,out.name,'C',10.,True,'test-tree')
    assert report['status']=='PARTIAL_NOT_COMPLETED'
    assert data['polarization'].shape==(3,3) and report['last_actual_sample_fs']==.1
    assert data['eigen_max'].max()==.5
    native['n_mu']=1
    (diag/'snapshot_0000.json').write_text(json.dumps(dict(q_receipts=native)))
    with pytest.raises(ValueError,match='Nonmatching microscopic'):
        plot.read_run(out,out.name,'C',10.,True,'test-tree')


def test_plot_exact_comparison_refuses_time_or_field_changes():
    import plot_full_native as plot
    item=dict(time_fs=np.array([0.,.05]),kpts=np.zeros((36,3)),weights=np.ones(36)/36,
              field=np.array([[0.,0.,0.],[1.,0.,0.]]))
    assert plot.exact_common([item,item])==2
    changed=dict(item,field=item['field']+.001)
    with pytest.raises(ValueError,match='Drive datasets differ'):
        plot.exact_common([item,changed])
    changed=dict(item,time_fs=np.array([0.,.05000001]))
    with pytest.raises(ValueError,match='time_fs mismatch'):
        plot.exact_common([item,changed])

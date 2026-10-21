"""Bounded spectral replay plumbing; no material source or positive closure."""
import json

import numpy as np
import pytest

from tests.tdagw.cri3_qgrid_750_campaign import same_w_rate_replay as replay


def _argv(tmp_path):
    return [item for name in ('setup-deck','accepted-deck','frames','snapshot','model','model-digest')
            for item in ('--'+name,'unused')] + ['--out',str(tmp_path/'fresh')]


def test_spectral_cli_selects_one_eps_and_preserves_band_order(tmp_path):
    args=replay.parse_args(_argv(tmp_path)+['--spectral-bands','4','25','24','23'])
    assert args.spectral_bands==[4,25,24,23]
    assert args.spectral_k==0 and args.quadrature_eps==[1e-8]
    assert args.snapshot_json is None
    rate=replay.parse_args(_argv(tmp_path)+['--snapshot-json','unused'])
    assert rate.spectral_bands is None and rate.quadrature_eps==[1e-4,1e-6]


@pytest.mark.parametrize('extra',[
    ['--spectral-bands','44'], ['--spectral-bands','-1'],
    ['--spectral-bands','4','4'], ['--spectral-bands','4','--spectral-k','64'],
    ['--spectral-bands','4','--quadrature-eps','1e-6','1e-8'],
    ['--spectral-bands','4','--quadrature-eps','nan'], [],
    ['--snapshot-json','unused','--spectral-k','1'],
])
def test_spectral_cli_refuses_unbounded_or_ambiguous_request(tmp_path,extra):
    with pytest.raises(SystemExit):
        replay.parse_args(_argv(tmp_path)+extra)


def test_spectral_probe_keeps_only_selected_k_full_matrices_and_negative_eigenvalues():
    energy=np.array([[-.8,-.2,.5],[-.7,-.1,.6]])
    bands,k,origin=[2,0],1,-.05
    width=np.array([[[-.4,.02j,0.],[-.02j,.2,.03],[0.,.03,.3]],
                    [[.1,0.,.04j],[0.,-.2,0.],[-.04j,0.,.5]]])
    r=np.full((2,2,3,3),7.+4j)
    r[:,k]=np.eye(3)[None]-.5j*width
    raw=dict(omega_ry=energy[k,bands]-origin,retarded_omega=dict(cond=r,val=2*r))
    payload,report=replay.spectral_probe_matrices(raw,dict(E_ry=energy),bands=bands,k=k,origin=origin)
    for space,factor in [('cond',1.),('val',2.)]:
        np.testing.assert_array_equal(payload['retarded_'+space+'_ry'],factor*r[:,k])
        np.testing.assert_allclose(payload['K_'+space+'_ry'],factor*width,atol=0.)
        assert payload['K_'+space+'_ry'].shape==(2,3,3)
        eig=np.linalg.eigvalsh(factor*width)
        np.testing.assert_allclose(report['branches'][space]['minimum_eigenvalue_ry'],eig[:,0])
        assert np.max(report['branches'][space]['minimum_eigenvalue_ry'])<0.
        np.testing.assert_array_equal(report['branches'][space]['hermiticity_frobenius_ry'],0.)
    np.testing.assert_array_equal(payload['omega_absolute_ry'],energy[k,bands])
    np.testing.assert_array_equal(payload['omega_relative_ry'],energy[k,bands]-origin)
    assert report['bands_active_relative']==bands
    json.dumps(report,allow_nan=False)
    raw['omega_ry']=energy[k,bands]  # Absolute instead of relative is not silently accepted.
    with pytest.raises(ValueError,match='shell frequencies changed'):
        replay.spectral_probe_matrices(raw,dict(E_ry=energy),bands=bands,k=k,origin=origin)


@pytest.mark.parametrize('bad', ['shape','nonfinite'])
def test_spectral_probe_refuses_invalid_branch_output(bad):
    r=np.zeros((1,2,3,3),complex)
    if bad=='shape':
        r=r[:,:,:2,:2]
    else:
        r[0,0,0,0]=np.nan
    raw=dict(omega_ry=np.array([-.8]),retarded_omega=dict(cond=r,val=r))
    with pytest.raises(ValueError,match='full active branch matrices'):
        replay.spectral_probe_matrices(raw,dict(E_ry=np.full((2,3),-.8)),bands=[0],k=0,origin=0.)


def test_spectral_mode_performs_one_grid_call_without_rate_operators(monkeypatch):
    from gw import ppm_sigma as ps
    from gw.gw_config import DynamicSigmaConfig
    import tdagw.collision as collision

    energy=np.linspace(-1.,2.,2*752).reshape(2,752)
    e=energy[:,100:144].copy()+.03
    snapshot=dict(U=np.broadcast_to(np.eye(44),(2,44,44)),E_ry=e,
                  rho_in_basis=np.broadcast_to(np.diag([1.]*30+[0.]*14),(2,44,44)))
    prepared=dict(energy_padded=energy,frequency_origin_ry=-.17,mesh=object())
    source=dict(meta=object(),config=object(),wfns=object(),linalg='local')
    calls=[]
    def grid(wfns,handle,window,plan,**kwargs):
        calls.append((window,plan,kwargs))
        return dict(grid_only=True)
    def forbidden(*args,**kwargs):
        raise AssertionError('spectral mode must not construct rate operators or collision')
    monkeypatch.setattr(ps,'shared_pole_window_source',lambda *a,**k:object())
    sigma=DynamicSigmaConfig(omega_min_ev=-2.,omega_max_ev=2.,omega_step_ev=.1,
        regularization_ev=.25,window_edge_factor=1.5,fermi_reference='midgap',sigma_at_dft_energies=False)
    monkeypatch.setattr(ps,'window_sigma_plan',lambda *a,**k:dict(sigma_cfg=sigma))
    monkeypatch.setattr(ps,'window_sigma_c',grid)
    monkeypatch.setattr(collision,'branch_operators',forbidden)
    monkeypatch.setattr(replay,'collision_statistics',forbidden)
    omega=e[1,[4,25,24,23]]-prepared['frequency_origin_ry']
    result=replay.evaluate_rate_body(source,prepared,snapshot,{},quadrature_eps=1e-8,omega_ry=omega)
    assert result==dict(raw=dict(grid_only=True)) and len(calls)==1
    window,plan,kwargs=calls[0]
    np.testing.assert_array_equal(window['e'],e)
    np.testing.assert_array_equal(window['rho'],snapshot['rho_in_basis'])
    np.testing.assert_array_equal(window['e_frozen'],replay.snapshot_energy(prepared,snapshot))
    np.testing.assert_array_equal(kwargs['omega_ry'],omega)
    assert kwargs['internal']=='active' and kwargs['branches'] and kwargs['pairing']=='vertex'
    assert plan['sigma_cfg'].quadrature_eps==1e-8

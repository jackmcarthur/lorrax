"""Strict shared-pole grammar and metadata geometry; no dense physics fixture."""
import configparser
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from common.units import RYD_TO_EV
from gw.gw_config import LorraxConfig, read_lorrax_input
from gw.shared_pole_recipe import (
    bind_shared_pole_census, resolve_shared_pole_recipe,
    construction_receipt, gate_receipt, shared_real_pole_gates_v1_r3b,
)

BASE = '[cohsex]\nnval=2\nncond=2\nnumber_bands=10\n'


def parse(tmp_path, extra):
    path = tmp_path / 'deck.in'
    path.write_text(BASE + extra)
    return LorraxConfig.from_input_file(str(path), resolve_hardware=False,
                                       runtime_platform='cpu', print_fn=lambda *_: None)


@pytest.mark.parametrize('model', ['', 'sigma_w_model=mpa\n'])
def test_default_mpa(tmp_path, model):
    c = parse(tmp_path, 'compute_mode=mpa\n' + model)
    assert (c.sigma.w_model, c.sigma.w_accuracy) == ('mpa', 'production')


@pytest.mark.parametrize('tier,eps', [('production', 1e-4), ('relaxed', 1e-3)])
@pytest.mark.parametrize('backend', ['local', 'distributed'])
def test_selected_tier(tmp_path, tier, eps, backend):
    c = parse(tmp_path, f'compute_mode=mpa\nsigma_w_model=shared_pole\nsigma_w_accuracy={tier}\nlinalg={backend}\nsigma_quadrature_eps={eps}\n')
    assert c.sigma.w_accuracy == tier
    assert c.sigma.quadrature_eps == eps


@pytest.mark.parametrize('entry,match', [
    ('sigma_w_model=poles', 'GATE shared_pole_enum'),
    ('sigma_w_accuracy=fast', 'GATE shared_pole_enum'),
    ('sigma_regularization_ev=0', 'GATE sigma_regularization'),
    ('sigma_regularization_ev=-1', 'GATE sigma_regularization'),
    ('sigma_regularization_ev=nan', 'GATE sigma_regularization'),
    ('sigma_regularization_ev=inf', 'GATE sigma_regularization'),
    ('linalg=auto', 'linalg'),
    ('sigma_w_height=1', 'unrecognized deck'),
    ('sigma_w_rank=896', 'unrecognized deck'),
    ('sigma_w_damping=0', 'unrecognized deck'),
    ('sigma_w_plasma=6', 'unrecognized deck'),
    ('sigma_w_samples=12', 'unrecognized deck'),
    ('sigma_w_gate=1e-6', 'unrecognized deck'),
])
def test_bad_key_value(tmp_path, entry, match):
    with pytest.raises(ValueError, match=match):
        parse(tmp_path, 'compute_mode=mpa\n' + entry + '\n')


@pytest.mark.parametrize('model', ['mpa', 'shared_pole'])
@pytest.mark.parametrize('mode', ['cohsex', 'gn_ppm', 'hl_ppm'])
def test_model_elsewhere_refuses(tmp_path, model, mode):
    with pytest.raises(ValueError, match='GATE shared_pole_applicability'):
        parse(tmp_path, f'compute_mode={mode}\nsigma_w_model={model}\n')


@pytest.mark.parametrize('mode', ['mpa', 'cohsex'])
def test_accuracy_elsewhere_refuses(tmp_path, mode):
    with pytest.raises(ValueError, match='GATE shared_pole_applicability'):
        parse(tmp_path, f'compute_mode={mode}\nsigma_w_accuracy=production\n')


@pytest.mark.parametrize('key', ['sigma_w_model', 'sigma_w_accuracy'])
def test_duplicates_refuse(tmp_path, key):
    with pytest.raises(configparser.DuplicateOptionError):
        parse(tmp_path, f'{key}=mpa\n{key}=mpa\n')


@pytest.mark.parametrize('entry', [
    'mpa_n_poles=8', 'mpa_sampling_alpha=1', 'mpa_sampling_schedule=nested',
    'mpa_pole_solver=loewner', 'mpa_varpi_near_ry=.2', 'mpa_varpi_far_ry=2',
    'mpa_metal_origin_shift_ry=.00002', 'mpa_pole_batch_size=4',
    'mpa_fit_reuse_file=old.h5', 'mpa_overwrite_completed_artifacts=false',
])
def test_unused_elementwise_inputs(tmp_path, entry):
    with pytest.raises(ValueError, match='GATE shared_pole_unused_inputs'):
        parse(tmp_path, 'compute_mode=mpa\nsigma_w_model=shared_pole\n' + entry + '\n')


def test_epsilon_conflict(tmp_path):
    with pytest.raises(ValueError, match='GATE shared_pole_epsilon_conflict'):
        parse(tmp_path, 'compute_mode=mpa\nsigma_w_model=shared_pole\nsigma_w_accuracy=relaxed\nsigma_quadrature_eps=1e-4\n')


def fixture(*, metal=False, eta=.25, tier='production', top=20.):
    energies = np.array([[-20., -1., 1., 1000.], [-20., 1., 2., -1000.]]) / RYD_TO_EV
    if not metal:
        energies[:, 1] = -1 / RYD_TO_EV
    occ = np.array([[1., .75 if metal else 1., 0., 1.], [1., .25 if metal else 1., 0., 1.]])
    wf = NS(enk=energies, occ=occ, slices=NS(b0=0,b4_logical=3,val=slice(0,2),cond_all_logical=slice(2,3)))
    state = NS(f_kn=occ,mu_ry=0.,smearing_family='fixed') if metal else None
    # Set volume so the known active electron count yields the desired top.
    electrons = 1. if metal else 2.
    volume = 4*np.pi*electrons / (((top-3.5)/RYD_TO_EV/2)**2)
    meta = NS(nspin=1,nspinor=1,n_rmu=17,nk_tot=2,cell_volume=volume)
    bind_shared_pole_census(wf,meta,occupation_state=state,trs_allowed=True,state_capacity=2.,kweights=[.5,.5])
    config = NS(sigma=NS(w_model='shared_pole',w_accuracy=tier,regularization_ev=eta))
    return config,wf,meta


def resolve(args):
    return resolve_shared_pole_recipe(*args,mesh_xy=NS(shape={'x':2,'y':2}),print_fn=lambda *_:None)


def test_geometry_padding_charge_and_holds():
    args=fixture()
    r=resolve(args)
    assert r['n']==17 and r['imaginary_width']==5 and r['infinity_width']==3
    assert r['census']['active_electrons']==2
    assert r['census']['borderline_bands']==[0]
    np.testing.assert_allclose(r['line_ev'],np.r_[np.arange(0,12,.5),np.arange(12,21)])
    assert not set(r['fit_ids']) & set(r['held_ids'])
    assert len(r['role']) == r['unique_evaluations']
    assert r['imaginary_count']==3
    assert r['accuracy_status']=='NOT_MEASURED'
    args[1].enk[:,-1] *= 10
    np.testing.assert_array_equal(resolve(args)['line_ev'],r['line_ev'])


def test_metal_and_eta_scaling():
    r=resolve(fixture(metal=True,eta=.1,top=10))
    assert r['height_ev']==.4 and r['low_step_ev']==.2
    assert r['census']['partial_at_mu']
    assert r['line_ev'][-1]==pytest.approx(10)
    r=resolve(fixture(eta=.1,top=10))
    assert r['low_step_ev']==.2 and not r['census']['partial_at_mu']


def test_relaxed():
    r=resolve(fixture(tier='relaxed',eta=.1))
    assert r['line_count']==8 and r['imaginary_count']==2
    assert r['held_count']==3  # duplicate imaginary held roles share one call
    assert r['imaginary_width']==3 and r['infinity_width']==2


def test_inverted_interval():
    with pytest.raises(ValueError,match='GATE shared_pole_interval'):
        resolve(fixture(eta=5,top=20))


def test_stale_census():
    args=fixture();args[1].enk[0,0]+=.001
    with pytest.raises(ValueError,match='stale energies'):
        resolve(args)


def test_receipt_absence_and_nonfinite():
    rows=construction_receipt()['gates']
    assert len(rows)==len(shared_real_pole_gates_v1_r3b)==13
    assert all(r['status']=='NOT_MEASURED' for r in rows)
    assert gate_receipt('capacity',passed=True,reason='absent')['status']=='NOT_MEASURED'
    assert gate_receipt('capacity',4,passed=False,reason='4U')['status']=='FAIL'
    with pytest.raises(ValueError):
        gate_receipt('capacity',float('nan'),passed=True,reason='invalid')
    json.dumps(rows,allow_nan=False)


def test_flat_role_serialization_and_deduplication():
    from gw.shared_pole_recipe import ROLE_CODES
    r=resolve(fixture(metal=True,top=10))
    assert r['z_ry'].shape==r['role'].shape==r['held'].shape==r['distinct_id'].shape
    assert r['z_ry'].dtype==np.complex128
    assert r['role'].dtype==np.int8 and r['distinct_id'].dtype==np.int64
    assert r['held'].dtype==np.bool_
    assert r['role_codes']==ROLE_CODES==dict(line=0,imaginary=1,infinity=2,held_line=3,held_imaginary=4)
    assert 2 not in r['role']  # no fake infinity bank call
    assert r['distinct_id'][0]==r['distinct_id'][r['line_count']]
    assert len(set(r['distinct_id']))==r['unique_evaluations']
    for i in set(r['distinct_id']):
        assert np.unique(r['z_ry'][r['distinct_id']==i]).size==1
    assert not np.any(r['held'][np.isin(r['distinct_id'],r['fit_ids'])])


def test_nested_missing_measurements_and_warning_diagnostics():
    assert gate_receipt('passivity',{'minimum':None},passed=True,reason='missing')['status']=='NOT_MEASURED'
    r=gate_receipt('full_m1_defect',3e-4,passed=False,reason='outside calibrated band')
    assert r['status']=='WARN' and r['version']=='cd8_58061895.50'
    assert r['threshold']==2e-4
    assert gate_receipt('full_m3_defect',1e-3,passed=True,reason='measured')['status']=='PASS'


def test_band_top_charge_nonuniform_weights_and_unclipped_tail():
    c,w,m=fixture(metal=True)
    f=np.array([[1.,1.05,0.,0.],[1.,-.05,0.,0.]])
    state=NS(f_kn=f,mu_ry=0.,smearing_family='mp1')
    bind_shared_pole_census(w,m,occupation_state=state,trs_allowed=True,
                           state_capacity=2.,kweights=[.75,.25])
    assert m.shared_pole_census['active_electrons']==pytest.approx(1.55)
    assert m.shared_pole_census['active_bands']==[1,2]
    assert m.shared_pole_census['borderline_bands']==[0]
    # Whole-band selection turns on at exactly mu-15 eV; one k row suffices.
    w.enk[1,0]=-15/RYD_TO_EV
    bind_shared_pole_census(w,m,occupation_state=state,trs_allowed=True,
                           state_capacity=2.,kweights=[.75,.25])
    assert m.shared_pole_census['active_electrons']==pytest.approx(3.55)
    assert m.shared_pole_census['borderline_bands']==[]


@pytest.mark.parametrize('weights', [[.5,.4],[-.1,1.1],[float('nan'),.5],[1.]])
def test_weight_refusal(weights):
    c,w,m=fixture()
    with pytest.raises(ValueError,match='GATE shared_pole_kweights'):
        bind_shared_pole_census(w,m,occupation_state=None,trs_allowed=True,
                               state_capacity=2.,kweights=weights)


@pytest.mark.parametrize('spin,trs',[(2,True),(1,False)])
def test_representation_refusal(spin,trs):
    c,w,m=fixture();m.nspinor=spin
    with pytest.raises(ValueError,match='GATE shared_pole_representation'):
        bind_shared_pole_census(w,m,occupation_state=None,trs_allowed=trs,
                               state_capacity=2.,kweights=[.5,.5])


def test_current_map_rebind_at_30mev():
    c,w,m=fixture();before=resolve((c,w,m))
    w.enk[:,:3]+=.03/RYD_TO_EV
    bind_shared_pole_census(w,m,occupation_state=None,trs_allowed=True,
                           state_capacity=2.,kweights=[.5,.5])
    after=resolve((c,w,m))
    assert before['census']['energy_sha256']!=after['census']['energy_sha256']
    np.testing.assert_allclose(after['z_ry'],before['z_ry'],rtol=1e-14)
    assert after['census']['mu_ry']-before['census']['mu_ry']==pytest.approx(.03/RYD_TO_EV)


def test_exact_applicability_message(tmp_path):
    with pytest.raises(ValueError) as exc:
        parse(tmp_path,'compute_mode=cohsex\nsigma_w_model=mpa\n')
    assert str(exc.value)==("GATE shared_pole_applicability: sigma_w_model got: 'mpa' "
        "with compute_mode='cohsex'; want: compute_mode=mpa; "
        "why: this key selects the MPA Sigma W representation")


def test_minimax_tolerance_does_not_override_bank(tmp_path):
    c=parse(tmp_path,'compute_mode=mpa\nsigma_w_model=shared_pole\nminimax_target_error=1e-3\n')
    assert c.screening.minimax_target_error==1e-3
    _,w,m=fixture()
    assert resolve((c,w,m))['bank_rule_tolerance']==1e-8

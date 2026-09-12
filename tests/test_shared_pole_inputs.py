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


def parse(tmp_path, extra, *, head='off'):
    path = tmp_path / 'deck.in'
    path.write_text(BASE + ('' if head is None else f'head_correction={head}\n') + extra)
    return LorraxConfig.from_input_file(str(path), resolve_hardware=False,
                                       runtime_platform='cpu', print_fn=lambda *_: None)


@pytest.mark.parametrize('model', ['', 'sigma_w_model=mpa\n'])
def test_default_mpa(tmp_path, model):
    c = parse(tmp_path, 'compute_mode=mpa\n' + model)
    assert (c.sigma.w_model, c.sigma.w_accuracy) == ('mpa', 'production')
    assert not c.debug.write_w and not c.write_poles


def _output_flag(config, key):
    """``write_w`` is the DEBUG sibling and lives in the ``debug`` group."""
    return config.debug.write_w if key == 'write_w' else config.write_poles


@pytest.mark.parametrize('key', ['write_w', 'write_poles'])
def test_shared_output_applicability(tmp_path, key):
    with pytest.raises(ValueError, match=key + '=true requires'):
        parse(tmp_path, f'compute_mode=mpa\n{key}=true\n')
    config = parse(tmp_path, f'compute_mode=mpa\nsigma_w_model=shared_pole\n{key}=true\n')
    assert _output_flag(config, key)


def test_write_w_is_a_debug_key_announced_at_parse_time(tmp_path, capsys):
    """Owner ruling 2026-09-11: the frequency-bank dump is a debug feature.

    Two halves, both load-bearing.  It is carried in ``DebugConfig`` -- the
    existing home for debug-only flags, the one ``write_qsgw_datasets``
    names when it says it is NOT a debug flag -- so there is no top-level
    ``config.write_w`` to mistake for a production output.  And the notice
    fires at PARSE time, through the same rank-0 deck reporter the
    retired-key report uses, not deep inside the Sigma stage: a run that
    asked for tens of GiB of debug bytes learns so before it spends them.
    ``write_poles`` stays a production export and says nothing.
    """
    quiet = parse(tmp_path, 'compute_mode=mpa\nsigma_w_model=shared_pole\n'
                            'write_poles=true\nwfn_file=WFN.h5\n')
    assert quiet.write_poles and not quiet.debug.write_w
    assert 'WARNING -- DEBUG' not in capsys.readouterr().out
    assert not hasattr(quiet, 'write_w')

    loud = parse(tmp_path, 'compute_mode=mpa\nsigma_w_model=shared_pole\n'
                           'write_w=true\nwfn_file=WFN.h5\n')
    assert loud.debug.write_w
    report = capsys.readouterr().out
    assert 'WARNING -- DEBUG' in report and 'write_w = true' in report
    # The documented reason, not just a scary box.
    assert 'NOT needed for BSE' in report and 'write_poles' in report


@pytest.mark.parametrize('head', [None, 'full', 'no_local_fields'])
@pytest.mark.parametrize('tier', ['production', 'relaxed'])
def test_shared_enabled_head_uses_scalar_mpa(tmp_path, head, tier):
    c = parse(tmp_path, f'compute_mode=mpa\nsigma_w_model=shared_pole\nsigma_w_accuracy={tier}\nmpa_n_poles=6\n',
              head=head)
    assert c.head.correction.value == ('full' if head is None else head)
    assert c.mpa.n_poles == 6


@pytest.mark.parametrize('head', [None, 'full', 'no_local_fields'])
def test_incumbent_enabled_head_remains_valid(tmp_path, head):
    c = parse(tmp_path, 'compute_mode=mpa\nsigma_w_model=mpa\n', head=head)
    assert c.head.correction.value == ('full' if head is None else head)


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


def fixture(*, metal=False, eta=.25, tier='production', plasma_seed=10., box=(-5., 5.),
            patches=()):
    """Two k, three logical bands: a shallow frontier band and one at depth 20 eV.

    ``plasma_seed`` is omega_p of the SEED band alone (the shallowest occupied
    one), in eV; the volume is set from it. The deep band's membership is then
    an OUTPUT of the active-set fixed point, not a fixture assumption: it joins
    when its 20 eV depth is at or under omega_p of the enlarged set. That is
    the knob these tests turn.
    """
    energies = np.array([[-20., -1., 1., 1000.], [-20., 1., 2., -1000.]]) / RYD_TO_EV
    if not metal:
        energies[:, 1] = -1 / RYD_TO_EV
    occ = np.array([[1., .75 if metal else 1., 0., 1.], [1., .25 if metal else 1., 0., 1.]])
    wf = NS(enk=energies, occ=occ, slices=NS(b0=0,b4_logical=3,val=slice(0,2),cond_all_logical=slice(2,3)))
    state = NS(f_kn=occ,mu_ry=0.,smearing_family='fixed') if metal else None
    seed_electrons = 1. if metal else 2.
    volume = 4*np.pi*seed_electrons / ((plasma_seed/RYD_TO_EV/2)**2)
    # b_id_4_chi_user is what response_weights masks the screening bands by;
    # the resolver now reaches the bank's own window partitioner through it.
    meta = NS(nspin=1,nspinor=1,n_rmu=17,nk_tot=2,cell_volume=volume,b_id_4_chi_user=3)
    bind_shared_pole_census(wf,meta,occupation_state=state,trs_allowed=True,state_capacity=2.,kweights=[.5,.5])
    # Resolver inputs carry the already-resolved positive device budget (R24).
    config = NS(sigma=NS(w_model='shared_pole',w_accuracy=tier,regularization_ev=eta,
                         omega_min_ev=box[0],omega_max_ev=box[1],
                         parsed_omega_patches_ev=lambda p=tuple(patches): list(p)),
                memory=NS(per_device_gb=30.))
    return config,wf,meta


def resolve(args):
    return resolve_shared_pole_recipe(*args,mesh_xy=NS(shape={'x':2,'y':2}),print_fn=lambda *_:None)


def test_geometry_padding_charge_and_holds():
    args=fixture()
    r=resolve(args)
    assert r['n']==17 and r['imaginary_width']==5 and r['infinity_width']==3
    # omega_p(seed) = 10 eV; the 20 eV band would only lift it to 14.14, so it
    # stays out and lands in the 2*threshold diagnostic band instead.
    assert r['census']['active_electrons']==2
    assert r['census']['active_bands']==[1] and r['census']['borderline_bands']==[0]
    assert r['plasma_ev']==pytest.approx(10) and r['active_depth_ev']==pytest.approx(1)
    assert r['omega_fine_ev']==pytest.approx(10)      # max(10, gap 2 + depth 1)
    assert r['sigma_window_ev']==pytest.approx(6)     # box 5 + active depth 1
    assert r['top_bound_by']=='plasmon'               # 2.25*10 beats 1.25*6
    assert r['top_ev']==pytest.approx(22.5)
    # Uniform 2*eta to omega_fine, then a step that grows by 1.25 per interval.
    np.testing.assert_allclose(r['line_ev'][:21],np.arange(0,10.5,.5))
    steps=np.diff(r['line_ev'][20:-1])
    np.testing.assert_allclose(steps[1:]/steps[:-1],1.25,rtol=1e-12)
    assert r['line_ev'][-1]==pytest.approx(22.5)
    assert not set(r['fit_ids']) & set(r['held_ids'])
    assert len(r['role']) == r['unique_evaluations']
    assert r['imaginary_count']==3 and r['u_max_ev']==pytest.approx(25)
    # One held point per spacing law, each the adjacent-support midpoint
    # nearest its own target: 0.5*omega_fine = 5 in the uniform region,
    # sqrt(omega_fine*top) = 15 in the geometric one.
    mids=.5*(r['line_ev'][:-1]+r['line_ev'][1:])
    np.testing.assert_allclose(r['held_line_ev'],
                               [mids[np.argmin(abs(mids-5.))],
                                mids[np.argmin(abs(mids-np.sqrt(10*22.5)))]],rtol=0)
    assert r['held_line_ev'][0]==pytest.approx(4.75)
    assert abs(r['held_line_ev'][1]-15.) < 1.4   # inside one local step of 15
    assert r['accuracy_status']=='NOT_MEASURED'
    # An empty band far above mu carries no charge and cannot move the geometry.
    args[1].enk[:,-1] *= 10
    np.testing.assert_array_equal(resolve(args)['line_ev'],r['line_ev'])


def test_active_set_fixed_point_admits_a_deep_band():
    """The 20 eV band joins exactly when omega_p of the set CONTAINING it clears it."""
    # seed 10 eV -> trial omega_p = 10*sqrt(2) = 14.14 < 20: excluded.
    out=resolve(fixture(plasma_seed=10.))
    assert out['census']['active_electrons']==2 and out['active_depth_ev']==pytest.approx(1)
    # seed 16.5 eV -> trial 23.33 >= 20: the deep band screens after all, and
    # omega_fine follows it (max(23.33, gap 2 + depth 20)).
    deep=resolve(fixture(plasma_seed=16.5))
    assert deep['census']['active_electrons']==4
    assert deep['census']['active_bands']==[0,1]
    assert deep['plasma_ev']==pytest.approx(16.5*np.sqrt(2))
    assert deep['active_depth_ev']==pytest.approx(20)
    assert deep['omega_fine_ev']==pytest.approx(16.5*np.sqrt(2))
    assert deep['census']['borderline_bands']==[]


def test_remote_domain_cap_limits_every_emitted_sample():
    """A deck with a remote Laplace cell caps BOTH the line top and u_max.

    The bank expands its remote cells about the lowest transition edge and
    refuses a sample outside that Taylor domain (`response_laplace_rule`), so
    the recipe -- which is what emits samples -- asks the bank's own
    partitioner where the edge is and stays inside it. Without the cap this
    deck would emit u_max = 58.3 eV against an edge at 37 eV and the bank
    would refuse the run, which is what happened on Si P4 (job 58217047.5).
    """
    import minimax
    # One occupied band at -36 eV, past response_windows' -35 eV near edge, so
    # a remote cell forms with delta_lo = (lowest empty) - (deepest occupied).
    energies = np.array([[-36., -20., -1., 1.], [-36., -20., -1., 2.]])/RYD_TO_EV
    occ = np.array([[1., 1., 1., 0.]]*2)
    wf = NS(enk=energies, occ=occ,
            slices=NS(b0=0, b4_logical=4, val=slice(0, 3), cond_all_logical=slice(3, 4)))
    volume = 4*np.pi*2/((16.5/RYD_TO_EV/2)**2)
    meta = NS(nspin=1, nspinor=1, n_rmu=17, nk_tot=2, cell_volume=volume,
              b_id_4_chi_user=4)
    bind_shared_pole_census(wf, meta, occupation_state=None, trs_allowed=True,
                            state_capacity=2., kweights=[.5, .5])
    config = NS(sigma=NS(w_model='shared_pole', w_accuracy='production',
                         regularization_ev=.25, omega_min_ev=-5., omega_max_ev=5.,
                         parsed_omega_patches_ev=list),
                memory=NS(per_device_gb=30.))
    r = resolve((config, wf, meta))
    edge = r['remote_delta_lo_ev']
    assert edge == pytest.approx(37.0, abs=1e-9)      # 1 - (-36)
    # The ceiling is the bank's, not a local constant: same call, same answer.
    cap = minimax.response_remote_max_abs_z(edge/RYD_TO_EV, r['height_ry'],
                                            r['bank_rule_tolerance'])*RYD_TO_EV
    assert r['remote_cap_ev'] == pytest.approx(cap)
    assert r['top_bound_by'] == 'remote_cap' and r['u_max_bound_by'] == 'remote_cap'
    # u_max is the radius itself; the line top is its real part at height h.
    assert r['u_max_ev'] == pytest.approx(cap)
    assert r['top_ev'] == pytest.approx(np.sqrt(cap**2 - r['height_ev']**2))
    # Both uncapped rules wanted more, and every emitted sample is now inside.
    assert r['top_uncapped_ev'] > cap and r['u_max_uncapped_ev'] > cap
    assert np.all(np.abs(r['z_ry'])*RYD_TO_EV <= cap*(1+1e-12))


def test_sigma_window_can_set_the_top():
    """A patch over a deep state raises the support, rather than being clamped."""
    plain=resolve(fixture(plasma_seed=10.))
    assert plain['top_bound_by']=='plasmon' and plain['top_ev']==pytest.approx(22.5)
    wide=resolve(fixture(plasma_seed=10.,patches=[(-40.,-30.),(-5.,5.)]))
    assert wide['sigma_extent_ev']==pytest.approx(40)
    assert wide['sigma_window_ev']==pytest.approx(41)   # + active depth 1
    assert wide['top_bound_by']=='sigma_window'
    assert wide['top_ev']==pytest.approx(51.25)         # 1.25 * 41
    # The cost of reaching 51 eV is logarithmic, not linear, in the extent.
    assert wide['line_count'] - plain['line_count'] <= 5
    assert wide['omega_fine_ev']==plain['omega_fine_ev']  # structure scale unmoved


def test_metal_and_eta_scaling():
    r=resolve(fixture(metal=True,eta=.1,plasma_seed=10.))
    assert r['height_ev']==pytest.approx(.4) and r['line_step_ev']==pytest.approx(.2)
    assert r['census']['partial_at_mu']
    assert r['top_ev']==pytest.approx(22.5)   # 2.25*10, the metal has no gap
    r=resolve(fixture(eta=.1,plasma_seed=10.))
    assert r['line_step_ev']==pytest.approx(.2) and not r['census']['partial_at_mu']


def test_relaxed():
    r=resolve(fixture(tier='relaxed',eta=.1))
    # Same shape, coarser dials: step 4*eta and a 1.5 growth ratio.
    assert r['line_step_ev']==pytest.approx(.4) and r['line_growth_fraction']==.5
    np.testing.assert_allclose(r['line_ev'][:26],np.arange(0,10.4,.4),atol=1e-12)
    assert r['line_ev'][-1]==pytest.approx(22.5) and r['imaginary_count']==2
    assert r['held_count']==3  # duplicate imaginary held roles share one call
    assert r['imaginary_width']==3 and r['infinity_width']==2


def test_inverted_interval():
    with pytest.raises(ValueError,match='GATE shared_pole_interval'):
        resolve(fixture(eta=7,plasma_seed=10.))  # h=28 eV above u_max=25


def test_stale_census():
    args=fixture();args[1].enk[0,0]+=.001
    with pytest.raises(ValueError,match='stale energies'):
        resolve(args)


def test_receipt_absence_and_nonfinite():
    rows=construction_receipt()['gates']
    assert len(rows)==len(shared_real_pole_gates_v1_r3b)==16
    assert all(r['status']=='NOT_MEASURED' for r in rows)
    assert gate_receipt('capacity',passed=True,reason='absent')['status']=='NOT_MEASURED'
    assert gate_receipt('capacity',4,passed=False,reason='4U')['status']=='FAIL'
    with pytest.raises(ValueError):
        gate_receipt('capacity',float('nan'),passed=True,reason='invalid')
    json.dumps(rows,allow_nan=False)


def test_flat_role_serialization_and_deduplication():
    from gw.shared_pole_recipe import ROLE_CODES
    r=resolve(fixture(metal=True,plasma_seed=10.))
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
    c,w,m=fixture(metal=True,plasma_seed=10.)
    f=np.array([[1.,1.05,0.,0.],[1.,-.05,0.,0.]])
    state=NS(f_kn=f,mu_ry=0.,smearing_family='mp1')
    bind_shared_pole_census(w,m,occupation_state=state,trs_allowed=True,
                           state_capacity=2.,kweights=[.75,.25])
    # Unclipped tail charge: 2*(.75*1.05 + .25*(-.05)) = 1.55, and the empty
    # band carries none, so it is not a screening band at all.
    assert m.shared_pole_census['active_electrons']==pytest.approx(1.55)
    assert m.shared_pole_census['active_bands']==[1]
    assert m.shared_pole_census['borderline_bands']==[0]
    # Whole-band selection turns on at the band TOP: raising one k row of the
    # deep band to -10 eV brings its depth under omega_p of the enlarged set
    # (1.55 + 2 = 3.55 electrons, omega_p = 10*sqrt(3.55) = 18.8 eV > 10).
    w.enk[1,0]=-10/RYD_TO_EV
    bind_shared_pole_census(w,m,occupation_state=state,trs_allowed=True,
                           state_capacity=2.,kweights=[.75,.25])
    assert m.shared_pole_census['active_electrons']==pytest.approx(3.55)
    assert m.shared_pole_census['active_bands']==[0,1]
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
    c=parse(tmp_path,'compute_mode=mpa\nsigma_w_model=shared_pole\nminimax_target_error=1e-3\nmemory_per_device_gb=30\n')
    assert c.screening.minimax_target_error==1e-3
    _,w,m=fixture()
    assert resolve((c,w,m))['bank_rule_tolerance']==1e-8

@pytest.mark.parametrize('tier', ['production', 'relaxed'])
def test_shared_mp1_refuses_positive_measure(tmp_path, tier):
    with pytest.raises(ValueError) as caught:
        parse(tmp_path, 'compute_mode=mpa\nsigma_w_model=shared_pole\n'
              f'sigma_w_accuracy={tier}\nocc_smearing_family=mp1\n'
              'occ_smearing_width_ry=.01\nfermi_reference=mp1_fixed_n\n')
    assert str(caught.value) == (
        'shared_pole needs a positive spectral measure: MP1 occupations '
        'are non-monotonic; use occ_smearing_family = fd')


@pytest.mark.parametrize('model,family', [('shared_pole', 'fd'), ('mpa', 'fd'), ('mpa', 'mp1')])
def test_supported_metal_family(tmp_path, model, family):
    c = parse(tmp_path, f'compute_mode=mpa\nsigma_w_model={model}\n'
              f'occ_smearing_family={family}\nocc_smearing_width_ry=.01\n'
              'fermi_reference=mp1_fixed_n\n')
    assert c.occ_smearing_family == family
    assert c.occ_broadening_ry == .01


@pytest.mark.parametrize('entry', ['occ_smearing_family=fermi_dirac\nocc_smearing_width_ry=.01',
                                  'occ_smearing_family=fd',
                                  'occ_smearing_family=fd\nocc_smearing_width_ry=0'])
def test_fd_grammar_refusal_twins(tmp_path, entry):
    with pytest.raises(ValueError):
        parse(tmp_path, 'compute_mode=mpa\n' + entry + '\n')

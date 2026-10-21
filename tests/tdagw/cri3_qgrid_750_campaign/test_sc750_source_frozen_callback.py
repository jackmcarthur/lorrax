"""Host routing/label contracts, not a material or MPI admission proof."""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


frozen = load('sc750_source_frozen_callback')
common = load('sc750_td_callback')
driven = load('sc750_driven_callback')


def test_explicit_source_frozen_deck_changes_only_frozen_screening():
    from tdagw.td_config import read_td_deck
    inputs = Path(__file__).with_name('inputs')
    base = read_td_deck(str(inputs/'sc750_zero_joint_44band_0p6fs.toml'))
    deck = read_td_deck(str(inputs/'sc750_source_frozen_zero44_0p6fs.toml'))
    assert common.validate_probe_deck(deck) == pytest.approx(.6)
    assert (common.probe_active_window(deck).start, common.probe_active_window(deck).stop) == (100,144)
    assert {k for k in deck.values if deck.values[k] != base.values[k]} == {'lorrax.collision_screening'}
    assert deck.values['lorrax.collision_screening'] == 'joint_frozen'
    assert deck.measurements == base.measurements


def test_diagnostic_role_cannot_masquerade_as_matching_zero():
    source = dict(status='SHORT_ZERO_FIELD_PROBE_COMPLETE_NOT_DRIVEN_ADMISSION', zero_gate=True,
                  response_gamma_ev=.05)
    assert common.diagnostic_record(source) is source
    tagged = common.diagnostic_record(source, frozen.ROLE)
    assert tagged['diagnostic_role'] == tagged['screening_control'] == frozen.ROLE
    assert tagged['zero_gate'] is False and tagged['matched_BC_admission'] is False
    assert tagged['refreshed_screening'] is False
    assert tagged['configured_response_gamma_ev'] == .05
    assert tagged['response_gamma_ev'] == tagged['response_lag_gamma_applied_to_W_ev'] == 0.
    assert tagged['source_W_unshifted'] is True
    assert tagged['status'] == 'SOURCE_FROZEN_DIAGNOSTIC_COMPLETE_NOT_BC_ADMISSION'
    with pytest.raises(ValueError, match='unknown diagnostic'):
        common.diagnostic_record({}, 'unrecognized')
    with pytest.raises(ValueError, match='cannot admit matched B/C'):
        driven.require_matching_zero_result(tagged, slice(100,144), {})


def test_raw_zero_validator_rejects_role_before_reading_any_native_data(tmp_path):
    (tmp_path/'complete.json').write_text(json.dumps(dict(diagnostic_role=frozen.ROLE)))
    (tmp_path/'reference.json').write_text('{}')
    with pytest.raises(ValueError, match='cannot admit matched B/C'):
        driven.validate_raw_zero_prerequisite(dict(zero_directory=str(tmp_path)), {})


def test_production_factory_captures_only_source_provider_and_is_restored(monkeypatch):
    from gw import shared_pole_screening as screening
    original = screening.screen_general_shared_poles
    source, prepared, meta = object(), dict(nb=750), object()
    options = dict(screening='frozen', b_lo=100, n_active=44, untouched=object())
    model = dict(source_frozen_receipt=dict(source_digest='authenticated-positive-digest'))
    cache = object()
    def original_factory(s, p, o):
        assert s is source and p is prepared and o is options
        provider = screening.screen_general_shared_poles
        def evaluate(basis, **kwargs):
            return dict(provider=provider(), same_current_basis=basis, same_kwargs=kwargs)
        return evaluate, cache
    evaluate, returned = frozen.bind_source_frozen_evaluator(source, prepared, options,
        model=model, meta=meta, original_factory=original_factory)
    assert screening.screen_general_shared_poles is original and returned is cache
    for basis in [dict(U='full-complex-U1',e='current-E1',rho='coherent-rho1'),
                  dict(U='full-complex-U2',e='current-E2',rho='coherent-rho2')]:
        row = evaluate(basis, equilibrium=True, fit_bases={})
        assert row['same_current_basis'] is basis
        result = row['provider']
        assert result['shared_pole'] is model and result['meta'] is meta
        assert result['receipt']['diagnostic_role'] == frozen.ROLE
        assert result['receipt']['current_U_E_fullrho_lines'] is True
        assert result['receipt']['refreshed_screening'] is False


def test_provider_restores_on_factory_exception():
    from gw import shared_pole_screening as screening
    original = screening.screen_general_shared_poles
    def refused(*a, **kw):
        raise RuntimeError('actual factory refusal')
    with pytest.raises(RuntimeError, match='actual factory refusal'):
        frozen.bind_source_frozen_evaluator({}, dict(nb=750),
            dict(screening='frozen',b_lo=100,n_active=44),
            model=dict(source_frozen_receipt={}),meta=object(),original_factory=refused)
    assert screening.screen_general_shared_poles is original


@pytest.mark.parametrize('failure', [None, RuntimeError('unrelated solver refusal'),
                                   ValueError('unexpected owner failure'), 'returned'])
def test_one_site_uses_actual_snapshot_fresh_ledger_and_only_named_stop(monkeypatch, tmp_path, failure):
    from gw import shared_pole_screening as screening, ppm_sigma
    events = []
    current_meta, isolated_meta, mesh = object(), object(), object()
    U, E, rho = object(), object(), object()
    snapshot = dict(U_active=U, E_active_ry=E, rho_active=rho, support_reads_ev=[1.])
    def fresh(meta, actual_mesh):
        assert meta is current_meta and actual_mesh is mesh
        events.append('fresh_ledger')
        return isolated_meta
    monkeypatch.setattr(ppm_sigma, 'fresh_shared_pole_capacity', fresh)
    def original_provider(*args, **kwargs):
        assert args[2] is isolated_meta
        assert kwargs['snapshot'] is not snapshot
        assert all(kwargs['snapshot'][k] is snapshot[k] for k in ('U_active','E_active_ry','rho_active'))
        assert kwargs['snapshot']['direct_dielectric_probe'] == frozen.DIRECT_REQUEST
        assert 'direct_dielectric_probe' not in snapshot
        assert kwargs['label'] == 'one_site_epsilon_suspect'
        events.append('direct_probe')
        if failure == 'returned':
            return {}
        if failure is not None:
            raise failure
        raise RuntimeError(frozen.DIRECT_STOP+' UNADMITTED q=0 one-site')
    monkeypatch.setattr(screening, 'screen_general_shared_poles', original_provider)
    ordered = dict(source_frozen_receipt={})
    def model():
        events.append('adapter')
        return ordered
    def verify():
        events.append('verified_both_artifacts')
        return dict(inverse_error='unresolved is preserved')
    def factory(*args):
        return screening.screen_general_shared_poles, {}
    provider, _ = frozen.bind_source_frozen_evaluator({}, dict(nb=750,mesh=mesh),
        dict(screening='frozen',b_lo=100,n_active=44), model=model,meta=current_meta,
        original_factory=factory,direct_probe=frozen.DIRECT_REQUEST,
        probe_root=tmp_path/'one_site_epsilon_suspect_general_shared_pole', verify_probe=verify)
    assert screening.screen_general_shared_poles is original_provider
    if failure is not None:
        expected = ValueError if failure == 'returned' else type(failure)
        with pytest.raises(expected):
            provider(None,None,current_meta,None,snapshot=snapshot)
        assert events == ['fresh_ledger','direct_probe']
    else:
        for _ in range(2):
            result = provider(None,None,current_meta,None,snapshot=snapshot)
            assert result['shared_pole'] is ordered
            assert result['receipt']['one_site_dielectric']['inverse_error'] == 'unresolved is preserved'
        assert events == ['fresh_ledger','direct_probe','verified_both_artifacts','adapter']


def test_one_site_requires_both_exact_durable_records_without_resolution_claim(tmp_path):
    row = dict(q=0,z_ry=frozen.DIRECT_REQUEST['z_ry'],logical_dimension=3000,
               admitted=False,sigma_computed=False,inverse_error='checked LU refused')
    for kind in ('basic','inverse'):
        (tmp_path/f'q_000_dielectric_00_{kind}.json').write_text(json.dumps(row))
    witness = frozen.read_one_site_witness(tmp_path,frozen.DIRECT_REQUEST,3000)
    assert witness['physical_root_claim'] is False
    assert witness['artifacts']['inverse']['inverse_error'] == 'checked LU refused'
    (tmp_path/'q_000_dielectric_00_inverse.json').unlink()
    with pytest.raises(FileNotFoundError):
        frozen.read_one_site_witness(tmp_path,frozen.DIRECT_REQUEST,3000)


@pytest.mark.parametrize('change', [dict(screening='joint'),dict(b_lo=128),dict(n_active=4)])
def test_provider_refuses_implicit_model_or_window_substitution(change):
    options = dict(screening='frozen', b_lo=100, n_active=44)
    options.update(change)
    with pytest.raises(ValueError, match='active44/full750'):
        frozen.bind_source_frozen_evaluator({}, dict(nb=750), options,
            model={},meta=object(),original_factory=lambda *a:None)


def test_verification_has_one_native_adapter_and_no_second_sigma_equation():
    tree = ast.parse(Path(frozen.__file__).read_text())
    calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
    assert calls.count('source_frozen_ordered_model') == 1
    assert calls.count('original_factory') == 1
    assert calls.count('common.run_td_probe') == 1
    assert not any('window_sigma' in c or 'branch_operators' in c for c in calls)
    assert 'fit_band_extrapolation' not in Path(frozen.__file__).read_text()

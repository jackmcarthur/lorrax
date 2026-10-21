"""Metadata-only t0 wrapper lifecycle tests; no native/GPU claim."""
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


def fixture(monkeypatch, tmp_path, *, failure=None, write_capture=True, call_snapshot=True):
    path = Path(__file__).parent/'tdagw/cri3_qgrid_750_campaign/sc750_t0_only_callback.py'
    spec = importlib.util.spec_from_file_location('t0_only_test_owner', path)
    wrapper = importlib.util.module_from_spec(spec); spec.loader.exec_module(wrapper)
    events = []
    def snapshot():
        events.append('native_snapshot')
        if failure is not None:
            raise failure
        return {'native': True}
    sr = SimpleNamespace(snapshot_collision=snapshot)
    def propagate():
        if call_snapshot:
            sr.snapshot_collision()
        events.append('integration_would_start')
    stages = SimpleNamespace(propagate=propagate)
    tdagw = ModuleType('tdagw'); tdagw.stages=stages; tdagw.snapshot_runtime=sr
    collective = ModuleType('common.collectives')
    collective.rank0_transaction = lambda path, *, stage, write: write()
    monkeypatch.setitem(sys.modules, 'tdagw', tdagw)
    monkeypatch.setitem(sys.modules, 'common.collectives', collective)
    out = tmp_path/'new_output'
    def ordinary(source, **kwargs):
        out.mkdir()
        old = sr.snapshot_collision
        def observed():
            value = old()
            if write_capture:
                (out/'reference.json').write_text(json.dumps(dict(actual_active=[100,144],virtual_MH_bands=750)))
                (out/'snapshot_0000.npz').write_bytes(b'tiny-mocked-operator-payload')
                for name in ('snapshot_0000.json', 'snapshot_0000_boundary.json'):
                    (out/name).write_text('{}')
                events.append('committed_capture')
            return value
        sr.snapshot_collision=observed
        try:
            stages.propagate()
        finally:
            sr.snapshot_collision=old
    monkeypatch.setattr(wrapper, '_ordinary_callback', lambda: SimpleNamespace(run_td_probe=ordinary))
    return wrapper, sr, stages, snapshot, propagate, events, out


def test_stops_after_committed_ordinary_capture_and_restores(monkeypatch, tmp_path):
    w,sr,stages,snapshot,propagate,events,out=fixture(monkeypatch,tmp_path)
    w.run_td_probe({'frame':{'frame':7}},out=out,deck_path='unchanged.toml')
    assert events == ['native_snapshot','committed_capture']
    assert sr.snapshot_collision is snapshot and stages.propagate is propagate
    record=json.loads((out/'T0_ONLY_DIAGNOSTIC.json').read_text())
    assert record['purpose']=='t0_only' and record['time_steps_executed']==0
    assert not record['zero_gate'] and not record['matched_BC_admission']
    assert len(record['artifact_sha256'])==4 and not (out/'complete.json').exists()


def test_unexpected_scientific_failure_propagates_and_restores(monkeypatch,tmp_path):
    failure=RuntimeError('actual native W refusal')
    w,sr,stages,snapshot,propagate,events,out=fixture(monkeypatch,tmp_path,failure=failure)
    with pytest.raises(RuntimeError,match='actual native W refusal') as caught:
        w.run_td_probe({'frame':{'frame':7}},out=out,deck_path='unchanged.toml')
    assert caught.value is failure
    assert sr.snapshot_collision is snapshot and stages.propagate is propagate
    assert not (out/'T0_ONLY_DIAGNOSTIC.json').exists()


@pytest.mark.parametrize('options,message',[
    ({'write_capture':False},'missing committed t0 artifact'),
    ({'call_snapshot':False},'snapshot capture was not reached'),
])
def test_missing_capture_is_not_admitted(monkeypatch,tmp_path,options,message):
    w,sr,stages,snapshot,propagate,events,out=fixture(monkeypatch,tmp_path,**options)
    with pytest.raises(ValueError,match=message):
        w.run_td_probe({'frame':{'frame':7}},out=out,deck_path='unchanged.toml')
    assert sr.snapshot_collision is snapshot and stages.propagate is propagate
    assert not (out/'T0_ONLY_DIAGNOSTIC.json').exists()

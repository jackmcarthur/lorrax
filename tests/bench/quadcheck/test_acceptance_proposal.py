"""Proposed policy regression: expected red on current production SC policy.

Run with pytest explicitly; this diagnostic directory is outside the default
gate. A future policy change must make all four paths refuse an over-budget
certificate even when the rule's separate roundoff test passes.
"""
import dataclasses
import runpy
from pathlib import Path

import numpy as np
import pytest

helpers=runpy.run_path(str(Path(__file__).resolve().parents[2]/'test_sigma_box_plan.py'))


@pytest.mark.parametrize('path',['one_shot','sc_initial','sc_rebuild','sc_cache'])
def test_over_budget_rule_is_never_accepted(monkeypatch,tmp_path,path):
    """An approximation-error failure cannot be excused by a noise pass."""
    from gw.sigma_box_plan import plan_sigma_windows
    fake=helpers['_fake_rule']
    session=None if path=='one_shot' else {}
    cache=str(tmp_path) if path=='sc_cache' else None
    def plan():
        return plan_sigma_windows(
            helpers['_summaries'](),[helpers['_branch']()],np.array([.2,.5]),.1,
            eps=1e-4,reduction_seconds=12,cache_dir=cache,
            fixed_rule_session=session,print_fn=lambda *a,**k:None)
    monkeypatch.setattr('gw.sigma_box_plan.build_uniform_rule',fake)
    if path in ('sc_rebuild','sc_cache'):
        plan()
        if path=='sc_rebuild':
            session['rules'].clear()  # an absent product requires a rebuild
        else:
            session.clear()
            for file in tmp_path.glob('*.npz'):
                with np.load(file) as f:
                    data={k:f[k] for k in f.files}
                data['sup_error']=.04111167770862537
                np.savez(file,**data)
    def bad(box,eps,**kwargs):
        return dataclasses.replace(fake(box,eps,**kwargs),sup_error=.04111167770862537)
    monkeypatch.setattr('gw.sigma_box_plan.build_uniform_rule',bad)
    with pytest.raises(RuntimeError,match='rule sup error'):
        plan()

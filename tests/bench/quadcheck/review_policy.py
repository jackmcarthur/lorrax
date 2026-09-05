"""CPU pinning probe of 6035f72f's finite-data and retry receipts.

Load the named commit as an isolated module; source worktree stays unchanged.
The controlled builder injects invalid results at the consumer boundary.
"""
from pathlib import Path
import subprocess
import types
import json
import os
import runtime
runtime.bootstrap(platform='cpu')
import numpy as np
from minimax import UniformRule
root=Path(__file__).resolve().parents[3]
code=subprocess.check_output(['git','show','6035f72f:src/gw/sigma_box_plan.py'],cwd=root,text=True)
m=types.ModuleType('gw.quadcheck_review_6035f72f')
exec(compile(code,'6035f72f:src/gw/sigma_box_plan.py','exec'),m.__dict__)
spec=m.make_sigma_box_spec(name='probe',frequencies=np.array([.2,.5]),states=np.array([.1,.2]),pole_stats=[(1.,1.,.08,.08)],pole_sign=1.,eta_ry=.1)
result=dict(jid=os.environ.get('SLURM_JOB_ID'),step=os.environ.get('SLURM_STEP_ID'),commit='6035f72f',cases={})
for case in ('nan_weight','nan_time','nan_sup','failed_retry_receipt'):
    budgets=[]
    def builder(box,eps,**kwargs):
        budgets.append(kwargs['time_budget'])
        t=np.array([.2+0j]); w=np.array([.1+0j]); sup=eps/2
        if case=='nan_weight': w[:]=np.nan
        if case=='nan_time': t[:]=np.nan
        if case=='nan_sup': sup=np.nan
        if case=='failed_retry_receipt': sup=eps*(4 if len(budgets)==1 else 2)
        return UniformRule(t,w,box,eps,True,0.,1,sup,1.,0.)
    m.build_uniform_rule=builder
    try:
        fit=m._fit_rule(spec,1e-4,12.,None,.1)
        item=dict(accepted=True,noise_bound=str(fit['noise_bound']),factor_growth=[str(x) for x in fit['factor_growth']])
    except Exception as error:
        item=dict(accepted=False,error=str(error))
    item['budgets']=budgets
    result['cases'][case]=item
out=Path('/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/Na/14_quadcheck_2026-09-05')
(out/'policy_review.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
assert result['cases']['nan_weight']['accepted']
assert result['cases']['nan_time']['accepted']
assert result['cases']['nan_sup']['budgets']==[12.]
assert '0.0004 exceeds' in result['cases']['failed_retry_receipt']['error']

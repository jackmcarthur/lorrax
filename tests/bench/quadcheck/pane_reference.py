"""Independent pane-family control for E4's sign-definite pole tail.

Pane service fits 1/(x-i gamma), x>0. The target here is its negative:
1/(-x+i gamma) = -sum w exp(-(x-i gamma)n), hence t=-i*n, weight=-w.
"""
from pathlib import Path
import json
import os
import runtime
runtime.bootstrap(platform='cpu')
import numpy as np
import minimax
from rule_audit import measure
root=Path('/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/Na/14_quadcheck_2026-09-05/rule_audit')
audit=json.loads((root/'rule_audit.json').read_text())
box=tuple(audit['physical_box_ry'])
rect=np.array([[-box[1],-box[0],box[2],box[3]]])
fit=minimax.fit_damped_reciprocal(rect,target_error=1e-8,max_rank=128)
t=-1j*np.asarray(fit.nodes); w=-np.asarray(fit.weights)
check=measure(t,w,box,relative=True)
assert check['sampled_sup'] < 1e-7, check
np.savez(root/'pane_reference.npz',times=t,weights=w,box=box,relative=True,eps=1e-7,sup_error=check['sampled_sup'])
result=dict(jid=os.environ.get('SLURM_JOB_ID'),step=os.environ.get('SLURM_STEP_ID'),nodes=len(t),validation=check,fit=str(fit))
(root/'pane_reference.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)

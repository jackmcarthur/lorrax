"""Dense composite GL time-contour control, independent of fitted box rules.

For d=x+i*y, x<0,y>0, deform the causal integral into t=s*(1-i)/sqrt(2):
1/d=-i*c integral_0^infinity exp(i*d*c*s) ds, c=(1-i)/sqrt(2).
The relative truncation error is exp(-(y-x)*T/sqrt(2)). Geometric finite
panels resolve short-time boundary layers. Three GL orders check discretization.
"""
from pathlib import Path
import json
import os
import runtime
runtime.bootstrap(platform='cpu')
import numpy as np
from common.gauss_legendre import gauss_legendre_interval
from rule_audit import measure
root=Path('/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/Na/14_quadcheck_2026-09-05/rule_audit')
box=tuple(json.loads((root/'rule_audit.json').read_text())['physical_box_ry'])
assert box[1]<0 and box[2]>0
c=(1-1j)/np.sqrt(2)
decay=(box[2]-box[1])/np.sqrt(2)
T=np.log(1e12)/decay
start=1/max(abs(complex(x,y)) for x in box[:2] for y in box[2:])
edges=[0.,start]
while edges[-1]<T:
    edges.append(min(T,2*edges[-1]))
result=dict(jid=os.environ.get('SLURM_JOB_ID'),step=os.environ.get('SLURM_STEP_ID'),
            box_ry=box,T=T,relative_truncation_bound=float(np.exp(-decay*T)),panels=len(edges)-1,orders={})
for order in (16,24,32):
    times=[]; weights=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        x,w=gauss_legendre_interval(order,lo,hi)
        times.extend(c*x); weights.extend(-1j*c*w)
    t=np.asarray(times); w=np.asarray(weights)
    check=measure(t,w,box,relative=True)
    result['orders'][str(order)]=dict(nodes=len(t),validation=check)
    print(order,json.dumps(result['orders'][str(order)]),flush=True)
    np.savez(root/f'contour_gl{order}.npz',times=t,weights=w,box=box,relative=True,
             eps=1e-8,sup_error=check['sampled_sup'])
assert result['orders']['24']['validation']['sampled_sup']<1e-8
assert result['orders']['32']['validation']['sampled_sup']<1e-8
(root/'contour_reference.json').write_text(json.dumps(result,indent=2)+'\n')

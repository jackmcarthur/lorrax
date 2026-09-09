"""Observe complete service spectra while replaying the unchanged constructor.

The Run339 replay owns input authentication, layout, construction and writing.
This adapter observes its spectral service seam; no W or direction is changed.
"""
from pathlib import Path
import hashlib
import json
import os
import sys

ROWS=[]
BASE=0
FINITE=False
PENDING={}


def install(owner):
    import distrib_la
    import distrib_la.polar as polar
    import jax
    import jax.numpy as jnp
    import numpy as np
    original_retained=polar._retained_columns
    original_states=owner._direction_states
    reference=json.loads(Path(sys.argv[1]).read_text())
    constructor=reference.get('constructor',reference)
    recipe=constructor['model_header']['recipe']
    roles=[role for role in owner._fit_roles(recipe) if not role['held']]
    events=[]
    for sample in recipe['fit_ids']:
        events.extend([role for role in roles if role['sample_id']==sample])
    cursor=0

    def observe_matrix(W, kind):
        global PENDING
        nonlocal cursor
        if FINITE:
            role=events[cursor];cursor+=1
            assert role['role'].split(':')[0]==kind
        else:
            role=dict(role='infinity',sample_id=None)
        # Direct denominators are measured BEFORE any decomposition.
        traces=np.asarray(jnp.trace(W,axis1=-2,axis2=-1))
        norms=np.asarray(jnp.sum(jnp.abs(W)**2,axis=(-2,-1)))
        PENDING=dict(kind=kind,role=role,traces=np.atleast_1d(traces),norms=np.atleast_1d(norms))

    def retain(Q, values, count, **kw):
        block=np.atleast_2d(values)
        counts=count if isinstance(count,tuple) else (count,)
        for i,(row,n) in enumerate(zip(block,counts)):
            direct=PENDING['traces'][i]
            ROWS.append(dict(parent=BASE+i,kind=PENDING['kind'],role=PENDING['role'],
                values=row.tolist(),retained=int(n),direct_trace_real=float(direct.real),
                direct_trace_imag=float(direct.imag),direct_frobenius_squared=float(PENDING['norms'][i]),
                job_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']))
        return original_retained(Q,values,count,**kw)

    right=distrib_la.right_singular_vectors
    leading=distrib_la.leading_eigenvectors
    def right_observed(W,*args,**kw):
        observe_matrix(W,'line');return right(W,*args,**kw)
    def leading_observed(W,*args,**kw):
        observe_matrix(W,'imaginary' if FINITE else 'infinity');return leading(W,*args,**kw)
    def states(*args,**kw):
        global BASE,FINITE
        nonlocal cursor
        FINITE=True;cursor=0
        try:result=original_states(*args,**kw)
        finally:FINITE=False
        assert cursor==len(events)
        BASE+=len(result[0])
        output=Path(sys.argv[3]);output.mkdir(parents=True,exist_ok=True)
        if jax.process_index()==0:
            (output/'spectra.json').write_text(json.dumps(dict(status='PARTIAL',rows=ROWS,
                recipe=recipe,reference=str(Path(sys.argv[1]).resolve()),
                reference_sha256=hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest()),indent=2)+'\n')
        return result
    polar._retained_columns=retain
    distrib_la.right_singular_vectors=right_observed
    distrib_la.leading_eigenvectors=leading_observed
    owner._direction_states=states


def main():
    sandbox=Path('/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14')
    replay=sandbox/'runs/frequency_integration_sandbox/339_acon_20260909/replay.py'
    text=replay.read_text()
    needle='    from gw import shared_pole_constructor as owner\n'
    assert text.count(needle)==1 and text.count("for temperature in ('cold', 'warm'):")==1
    text=text.replace(needle,needle+'    install(owner)\n')
    text=text.replace("for temperature in ('cold', 'warm'):","for temperature in ('cold',):")
    scope={'__name__':'arank_replay_adapter','install':install}
    exec(compile(text,str(replay)+':ARANK-observer','exec'),scope)
    scope['main']()
    output=Path(sys.argv[3]);p=output/'spectra.json'
    # Runtime may have finalized here; only rank0 owns the JSON file.
    if os.environ.get('SLURM_PROCID','0')=='0':
        rec=json.loads(p.read_text());rec.update(status='COMPLETE',adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            replay=str(replay),replay_sha256=hashlib.sha256(replay.read_bytes()).hexdigest())
        p.write_text(json.dumps(rec,indent=2)+'\n')


if __name__=='__main__':main()

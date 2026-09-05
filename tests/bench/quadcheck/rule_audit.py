"""Audit a retained Na rule; diagnostic only, no production state mutation.

The reciprocal is 1/d = -i integral_0^infinity exp(i*t*d) dt, Im(d)>0.
All energies are Ry and times inverse Ry. Chunking bounds host matrix memory.
"""
from pathlib import Path
import argparse
import json
import os
import re
import time
import numpy as np
from scipy.special import roots_legendre
import runtime
runtime.bootstrap(platform="cpu")
from minimax import build_uniform_rule
from common.units import RYD_TO_EV


def measure(times, weights, box, *, relative, nx=1601, ny=24):
    """Measure reciprocal error on a rectangular independent validation grid."""
    reals = np.linspace(box[0], box[1], nx)
    imags = np.geomspace(box[2], box[3], ny)
    points = (reals[:, None] + 1j * imags).reshape(-1)
    worst = (0., 0j)
    for i in range(0, len(points), 32):
        d = points[i:i+32]
        q = np.exp(1j*d[:, None]*times[None, :]) @ weights
        error = np.abs(q - 1/d) * (np.abs(d) if relative else box[2])
        j = int(np.argmax(error))
        if error[j] > worst[0]:
            worst = (float(error[j]), complex(d[j]))
    return dict(sampled_sup=worst[0], worst_d_ry=[worst[1].real, worst[1].imag],
                nx=nx, ny=ny, currency='relative' if relative else 'peak-relative')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    log=(args.source/'evidence/armE/gw.rank0000.log').read_text()
    lines=[s for s in log.splitlines() if 'SC fixed window: ω≥E_F cond:pole_tail:' in s]
    # Exactly the fifth map receipt, accepted call 0004.
    receipt=lines[4]
    physical=np.array([float(x) for x in re.search(r' box=\((.*?)\) eV',receipt)[1].split(',')])/RYD_TO_EV
    with np.load(args.source/'tmp/sigma_quadrature_rules/rule_7a4c117704ec3991.npz') as f:
        old={k:f[k] for k in f.files}
    box=tuple(old['box'])
    result=dict(jid=os.environ.get('SLURM_JOB_ID'),step=os.environ.get('SLURM_STEP_ID'),
                map=4,receipt=receipt,physical_box_ry=physical.tolist(),padded_box_ry=list(box),
                retained_nodes=len(old['times']),retained_reported_sup=float(old['sup_error']),
                retained_reported_kappa=float(old['kappa_max']))
    assert len(old['times'])==900
    result['retained_on_physical']=measure(old['times'],old['weights'],physical,relative=False)
    result['retained_on_padded']=measure(old['times'],old['weights'],box,relative=False)
    print(json.dumps(result),flush=True)
    # Independent real-time Gauss-Legendre reference, not a per-state/pole resolvent.
    # The truncation error in peak currency is <= exp(-im_lo*T).
    T=np.log(1e10)/box[2]
    for n in (12000,16000):
        x,w=roots_legendre(n)
        t=(x+1)*T/2
        weight=-1j*w*T/2
        audit=measure(t,weight,box,relative=False,nx=401,ny=12)
        audit.update(nodes=n,truncation_bound=float(np.exp(-box[2]*T)))
        result[f'dense_tau_{n}']=audit
        print('DENSE',json.dumps(audit),flush=True)
        if audit['sampled_sup']<1e-7:
            np.savez(args.output/f'dense_tau_{n}.npz',times=t,weights=weight,box=box,
                     measured_sup=audit['sampled_sup'],eps=1e-7,relative=False)
    # This branch's physical support is sign-definite; retain the exact tuple
    # partition and certify a replacement over its actual support.
    for budget in (12.,120.):
        start=time.monotonic()
        rule=build_uniform_rule(tuple(physical),1e-4,time_budget=budget,backend='numpy')
        audit=measure(rule.times,rule.weights,physical,relative=True)
        audit.update(nodes=rule.node_count,reported_sup=rule.sup_error,
                     kappa=rule.kappa_max,seconds=time.monotonic()-start,budget=budget)
        result[f'physical_{int(budget)}s']=audit
        print('PHYSICAL',json.dumps(audit),flush=True)
        np.savez(args.output/f'physical_{int(budget)}s.npz',times=rule.times,weights=rule.weights,
                 box=rule.box,eps=rule.eps,relative=rule.relative,sup_error=rule.sup_error)
    (args.output/'rule_audit.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    main()

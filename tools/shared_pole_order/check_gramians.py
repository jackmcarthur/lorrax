"""Small complex-state red twins for weighted Gramian and full balancing identities."""
from pathlib import Path
import json
import os
import sys
import numpy as np
from scipy.linalg import cholesky, svd, solve_continuous_lyapunov


def main():
    rng = np.random.default_rng(304)
    b = rng.normal(size=(3, 4))+1j*rng.normal(size=(3, 4))
    om = np.array([.1,.17,.3,.7])
    eta = .025
    a = np.r_[-eta-1j*om,-eta+1j*om]
    A = np.diag(a)
    B = np.tile(b.conj().T, (2,1))
    C = np.c_[-1j*b,1j*b]
    adj = lambda x: x.conj().T
    grid = np.linspace(-1.5,1.5,2001)
    w = np.exp(-grid**2) * (grid[1]-grid[0])/(2*np.pi)
    w[[0,-1]] *= .5
    r = 1/(1j*grid[None,:]-a[:,None])
    direct = (r*w[None,:])@adj(r)
    g = r@w
    fast = -(g[:,None]+g.conj()[None,:])/(a[:,None]+a.conj()[None,:])
    rel = lambda x,y: float(np.linalg.norm(x-y)/np.linalg.norm(y))
    kernel_error = rel(fast,direct)
    p = (B@adj(B))*fast
    q = (adj(C)@C)*fast.T
    qdirect = sum(weight*(adj(C/(1j*t-a)[None,:])@(C/(1j*t-a)[None,:])) for t,weight in zip(grid,w))
    qerror = rel(q,qdirect)
    wrong_q_error = rel((adj(C)@C)*fast,qdirect)
    pu = -(B@adj(B))/(a[:,None]+a.conj()[None,:])
    qu = -(adj(C)@C)/(a.conj()[:,None]+a[None,:])
    lp = cholesky(pu,lower=True)
    lq = cholesky(qu,lower=True)
    u,h,vh = svd(adj(lq)@lp)
    T = (lp@adj(vh))/np.sqrt(h)[None,:]
    Ti = (adj(u)@adj(lq))/np.sqrt(h)[:,None]
    Ar,Br,Cr = Ti@A@T,Ti@B,C@T
    full_errors=[]
    sign_errors=[]
    for z in np.array([0,.07,.21,.8])+.025j:
        s=-1j*z-eta
        W=(b*(2*om/(z*z-om*om))[None,:])@adj(b)
        wr=Cr@np.linalg.solve(s*np.eye(len(a))-Ar,Br)
        full_errors.append(rel(wr,W))
        sign_errors.append(rel(-wr,W))
    receipt=dict(jobid=os.getenv('SLURM_JOB_ID'),stepid=os.getenv('SLURM_STEP_ID'),
                 weighted_partial_fraction_error=kernel_error,observability_error=qerror,
                 wrong_observability_red_twin_error=wrong_q_error,
                 full_balanced_roundtrip_max=max(full_errors),sign_red_twin_min=min(sign_errors),
                 unweighted_P_lyapunov_error=rel(pu,solve_continuous_lyapunov(A,-B@adj(B))),
                 unweighted_Q_lyapunov_error=rel(qu,solve_continuous_lyapunov(adj(A),-adj(C)@C)))
    assert kernel_error<1e-12 and qerror<1e-12 and wrong_q_error>1e-2,receipt
    assert max(full_errors)<1e-10 and min(sign_errors)>1,receipt
    Path(sys.argv[1]).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    main()

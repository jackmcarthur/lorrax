"""Independent cell-quadrature check of the narrow-shift weight integral."""
import json
import os
from pathlib import Path
import numpy as np
from balance import EV,weighted_resolvent_integral,sha
from thresholds import R


def main():
    assert os.getenv('SLURM_JOB_ID')
    from runtime import initialize_communicator_stack
    initialize_communicator_stack()
    import jax.numpy as jnp
    with np.load(R/'01_weight/weight.npz') as f:
        om=f['omega_ry'];w=f['weight_ry_minus2']
    grid=np.r_[-om[:0:-1],om];values=np.r_[w[:0:-1],w]
    nodes,weights=np.polynomial.legendre.leggauss(48)
    dx=np.diff(grid)
    x=grid[:-1,None]+dx[:,None]*(nodes[None,:]+1)/2
    val=values[:-1,None]+np.diff(values)[:,None]*(nodes[None,:]+1)/2
    rows=[]
    for delta in (.25,.0625,.015625):
        a=(-delta+1j*np.array([-.393,-1.001,-7.123,1.001,135.1]))/EV
        direct=np.array([np.sum(val/(1j*x-state)*dx[:,None]*weights[None,:]/2)/(2*np.pi) for state in a])
        got=np.asarray(weighted_resolvent_integral(jnp.asarray(a),jnp.asarray(grid),jnp.asarray(values)))
        error=float(np.linalg.norm(got-direct)/np.linalg.norm(direct))
        assert error<5e-11,(delta,error)
        rows.append(dict(shift_ev=delta,relative_error=error))
    out=Path(os.environ['ORDER_SHIFT_CHECK_OUT']);out.mkdir(parents=True,exist_ok=True)
    (out/'quadrature_check.json').write_text(json.dumps(dict(status='PASS',rows=rows,jobid=os.environ['SLURM_JOB_ID'],stepid=os.environ['SLURM_STEP_ID'],script_sha256=sha(__file__),balance_sha256=sha(Path(__file__).with_name('balance.py'))),indent=2)+'\n')
    print(rows,flush=True)


if __name__=='__main__':main()

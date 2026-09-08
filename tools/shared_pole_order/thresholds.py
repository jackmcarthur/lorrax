"""Re-threshold measured HSV curves without re-reading wavefunction HDF5."""
import hashlib
import json
import socket
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

R = Path('/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/frequency_integration_sandbox/304_order_weighted_bt_20260907')
STAR = np.array([1,12,12,12,6,24,48,48,24,24,6,24,24,12,24,48,8,8,24,12,24,6,24,12,24,2,6,12,1])


def counts_receipt(d):
    """Report signed-state orders and nominal paired storage, not Sigma order."""
    return dict(counts=d.tolist(), q0_rank_from_smallest=1+int(np.sum(d<d[0])),
                nominal_sum_n_K=float(896*np.sum(d)/2), sum_signed_states=int(np.sum(d)))


def main():
    """Apply absolute HSV floors and star-weighted absolute tail budgets."""
    assert STAR.sum()==512
    out=R/'18_absolute_thresholds';out.mkdir(exist_ok=True)
    result=dict(stamp=datetime.now(timezone.utc).isoformat(),host=socket.gethostname(),
                execution='login-node small NumPy; no HDF5',star_sizes=STAR.tolist(),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),weights={})
    for weight,stage,job in [('weighted','04_weighted_curves','58051053.11'),('unweighted','03_unweighted_signfix','58051053.3')]:
        h=[np.sort(np.load(R/stage/f'q{q:02d}/hsv.npy'))[::-1] for q in range(29)]
        top=np.array([v[0] for v in h]);tails=[np.r_[np.cumsum(v[::-1])[::-1],0.] for v in h]
        a=[]
        for label,tau in [('1e-3_q0',1e-3*top[0]),('1e-3_median',1e-3*np.median(top)),('1e-2_q0',1e-2*top[0]),('1e-4_q0',1e-4*top[0])]:
            a.append(dict(label=label,tau=float(tau),**counts_receipt(np.array([np.count_nonzero(v>tau) for v in h]))))
        raw_T=2*tails[0][2647]
        b=[]
        # The brief's raw T lacks star_0/512. Preserve it and also run the
        # corrected center that actually reproduces q0's d=2647.
        for center_name,center in [('requested_unstarred',raw_T),('q0_calibrated',raw_T/512)]:
            for scale in np.logspace(-1.5,1.5,13):
                T=float(center*scale)
                d=np.array([np.searchsorted(-2*STAR[q]/512*tails[q],-T,side='left') for q in range(29)])
                b.append(dict(center=center_name,scale=float(scale),T=T,**counts_receipt(d)))
        result['weights'][weight]=dict(input_job_step=job,summary_sha256=hashlib.sha256((R/stage/'summary.json').read_bytes()).hexdigest(),
                                      hsv_sha256=[hashlib.sha256((R/stage/f'q{q:02d}/hsv.npy').read_bytes()).hexdigest() for q in range(29)],
                                      top=top.tolist(),sums=[float(t[0]) for t in tails],rule_A=a,rule_B=b)
    result['scope']='Surrogate signed-state orders, no meV certification; n*d/2 assumes pairs and excludes general left/right storage.'
    result['budget_correction']='To reproduce q0 d=2647 under (star/512)*2*tail<=T requires T=2*tail_q0(2647)/512, not the unstarred center printed in the experiment brief. Both sweeps retained.'
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'manifest.yaml').write_text('run_id: 304_order_absolute_thresholds\nstatus: complete\ngeometry: login-node small NumPy\n')
    print(json.dumps({k:dict(rule_A=v['rule_A'],rule_B_center=[x for x in v['rule_B'] if x['scale']==1]) for k,v in result['weights'].items()},indent=2))


if __name__=='__main__':
    main()

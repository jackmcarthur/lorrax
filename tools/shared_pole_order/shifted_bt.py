"""Coordinator-requested shifted BT with explicit floor and star-tail budgets."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import balance
import reductions
from thresholds import STAR,R


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--divisor',type=int,choices=(4,16),required=True)
    parser.add_argument('--q',type=int,nargs='+',default=[0,1,14])
    args=parser.parse_args()
    balance.ETA=.25/balance.EV/args.divisor
    original=reductions.reduce_models
    def observe(a,B,C,balancing,context):
        h=np.asarray(balancing[3]);q=context['receipt']['q']
        budget_path=args.out/'star_budgets.json'
        if q==0:
            tail=np.r_[np.cumsum(h[::-1])[::-1],0]
            budgets={f'tail{eps:.0e}':float(2*tail[np.count_nonzero(h>eps*h[0])]/512) for eps in (1e-2,1e-3,1e-4)}
            budget_path.write_text(json.dumps(budgets,indent=2)+'\n')
        else:budgets=json.loads(budget_path.read_text())
        tail=np.r_[np.cumsum(h[::-1])[::-1],0]*2*STAR[q]/512
        targets=[(label,max(1,int(np.searchsorted(-tail,-T,side='left'))),None,None) for label,T in budgets.items()]
        context.update(targets=targets,target_metadata={label:dict(T=T,star=int(STAR[q]),formula='(star/512)*2*neglected_tail <= T') for label,T in budgets.items()},shift_ry=balance.ETA,floor_damping=True)
        original(a,B,C,balancing,context)
    reductions.reduce_models=observe
    sys.argv=[sys.argv[0],'--out',str(args.out),'--weight',str(R/'01_weight/weight.npz'),'--q']+list(map(str,args.q))
    balance.main()


if __name__=='__main__':main()

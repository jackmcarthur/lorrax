"""Select immutable certificates accepted for both recorded source windows."""
import ast
import json
from pathlib import Path
import shutil
import numpy as np
from gw import sigma_box_plan as owner

requests=json.loads(Path('common_rule_requests.json').read_text());results=[]
for pair in requests:
    parent,fixed=Path(pair['parent']),Path(pair['fixed'])
    for i,(p,f) in enumerate(zip(pair['p'],pair['f'])):
        pbox,fbox=ast.literal_eval(p[3]),ast.literal_eval(f[3])
        with np.load(parent/'tmp/sigma_quadrature_rules'/p[2].removeprefix('hit:')) as h:
            eps=float(h['eps']);relative=bool(h['relative'])
        cap=owner._RUNTIME_NOISE_SAFETY*eps/owner._RUNTIME_NOISE_EPSILON
        best=None
        for origin in (parent,fixed):
            for candidate in sorted((origin/'tmp/sigma_quadrature_rules').glob('*.npz')):
                probe=Path('rule_probes')/parent.name/str(i)/candidate.stem
                probe.mkdir(parents=True,exist_ok=True)
                target=probe/candidate.name
                if not target.exists():target.symlink_to(candidate)
                a,_=owner._rule_cache_lookup(probe,pbox,eps,relative,noise_amplification_cap=cap)
                b,_=owner._rule_cache_lookup(probe,fbox,eps,relative,noise_amplification_cap=cap)
                if a is not None and b is not None:
                    if best is None or a[0].node_count<best[0]:best=(a[0].node_count,candidate)
        row={'parent':str(parent),'fixed':str(fixed),'window':i,'pbox':pbox,'fbox':fbox,'selected':None}
        if best:
            dest=Path('common_rules')/parent.name/str(i);dest.mkdir(parents=True,exist_ok=True)
            shutil.copy2(best[1],dest/best[1].name)
            row.update(selected=str((dest/best[1].name).resolve()),nodes=best[0])
        results.append(row);print(row,flush=True)
Path('common_rule_selection.json').write_text(json.dumps(results,indent=2)+'\n')

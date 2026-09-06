"""Diff recorded compiler invocations by module name and source owner."""
from pathlib import Path
from collections import defaultdict
import argparse,json
p=argparse.ArgumentParser();p.add_argument('parent',type=Path);p.add_argument('fixed',type=Path);p.add_argument('output',type=Path);a=p.parse_args()

def reduce(run):
 rows=[json.loads(x) for x in (run/'compile_modules.jsonl').read_text().splitlines()]
 groups=defaultdict(lambda:dict(count=0,seconds=0.,digests=set(),sites=set()))
 for row in rows:
  key=row['name'];g=groups[key];g['count']+=1;g['seconds']+=row['seconds'];g['digests'].add(row['digest'])
  g['sites'].update(f"{x['file'].split('/src/')[-1]}:{x['line']}:{x['function']}" for x in row['stack'] if 'jax_compile_cache' not in x['file'])
 for g in groups.values():g['digests']=sorted(g['digests']);g['sites']=sorted(g['sites'])
 return groups
P,F=reduce(a.parent),reduce(a.fixed)
rows=[]
for name in P.keys()|F.keys():
 pv,fv=P.get(name,{}),F.get(name,{})
 rows.append(dict(name=name,parent=pv,fixed=fv,count_delta=pv.get('count',0)-fv.get('count',0),seconds_delta=pv.get('seconds',0.)-fv.get('seconds',0.)))
rows.sort(key=lambda x:x['seconds_delta'],reverse=True)
result=dict(parent=str(a.parent),fixed=str(a.fixed),parent_count=sum(x['count'] for x in P.values()),fixed_count=sum(x['count'] for x in F.values()),parent_seconds=sum(x['seconds'] for x in P.values()),fixed_seconds=sum(x['seconds'] for x in F.values()),modules=rows)
a.output.write_text(json.dumps(result,indent=2)+'\n')
for x in rows[:25]:print(x['name'],x['parent'].get('count',0),x['fixed'].get('count',0),round(x['seconds_delta'],4))

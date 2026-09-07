"""Extract one occurrence of an XLA module from native Nsight CUDA records."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3

p = argparse.ArgumentParser()
p.add_argument('run', type=Path)
p.add_argument('program', type=int)
p.add_argument('--occurrence', type=int, default=1, help='Zero-based; 1 selects the second call')
a = p.parse_args()
db = sqlite3.connect(f'file:{a.run}/nsys_rank0.sqlite?mode=ro', uri=True)
scopes = db.execute('''SELECT n.start,n.end,n.globalTid,coalesce(n.text,s.value)
 FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id
 WHERE coalesce(n.text,s.value) LIKE ? ORDER BY n.start''',
 (f'XlaModule:%program_id={a.program}#%',)).fetchall()
assert len(scopes)>a.occurrence, (len(scopes), a.occurrence)
start,end,tid,label=scopes[a.occurrence]
kernels=db.execute('''SELECT k.start,k.end,s.value,k.correlationId
 FROM CUPTI_ACTIVITY_KIND_KERNEL k
 JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON k.correlationId=r.correlationId
 JOIN StringIds s ON k.demangledName=s.id
 WHERE r.globalTid=? AND r.start>=? AND r.start<=? ORDER BY k.start''',
 (tid,start,end)).fetchall()
assert kernels, 'No native CUDA kernels correlated to selected module'
thunks=db.execute('''SELECT n.start,n.end,coalesce(n.text,s.value)
 FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id
 WHERE n.globalTid=? AND n.start>=? AND n.end<=?
 AND coalesce(n.text,s.value) LIKE 'Thunk:%' ORDER BY n.start''',
 (tid,start,end)).fetchall()
groups=defaultdict(list)
for ks,ke,name,corr in kernels:
 groups[name].append(ke-ks)
ranges=[]
for ts,te,name in thunks:
 rows=db.execute('''SELECT k.start,k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k
 JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON k.correlationId=r.correlationId
 WHERE r.globalTid=? AND r.start>=? AND r.start<=?''',(tid,ts,te)).fetchall()
 if rows:
  ranges.append(dict(name=name, kernels=len(rows), projected_ms=(max(e for _,e in rows)-min(s for s,_ in rows))/1e6,
                     kernel_sum_ms=sum(e-s for s,e in rows)/1e6))
result=dict(program=a.program, occurrence=a.occurrence, occurrences=len(scopes),
 label=label, host_range_ms=(end-start)/1e6,
 projected_ms=(max(k[1] for k in kernels)-min(k[0] for k in kernels))/1e6,
 kernel_count=len(kernels), kernel_sum_ms=sum(k[1]-k[0] for k in kernels)/1e6,
 kernels=[dict(name=n,count=len(v),sum_ms=sum(v)/1e6) for n,v in groups.items()],
 thunks=ranges, scope='Native launch-correlation projection; nested thunk ranges must not be summed')
(a.run/f'unit_{a.program}_{a.occurrence}.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['kernels','thunks']},indent=2))

"""Count repeated optimized module fingerprints without inferring compiler seconds."""
from pathlib import Path
from collections import defaultdict
import argparse,json,re
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
groups=defaultdict(lambda:defaultdict(list))
for f in (a.run/'xla_dump_rank0').glob('*after_optimizations.txt'):
 text=f.read_text();header=text.splitlines()[0]
 name=re.search(r'HloModule ([^,]+)',header).group(1)
 fingerprint=re.search(r'fingerprint_before_lhs="([^"]+)"',header)
 if fingerprint:
  groups[name][fingerprint.group(1)].append(dict(file=f.name,zero_initializer='"_zeros"' in text,source_functions=text.partition('FunctionNames\n')[2].partition('\n\n')[0]))
rows=[]
for name,values in groups.items():
 rows.append(dict(name=name,modules=sum(map(len,values.values())),fingerprints=len(values),repeated_modules=sum(len(v)-1 for v in values.values()),groups=values))
rows.sort(key=lambda x:x['repeated_modules'],reverse=True)
(a.run/'compile_hlo_repeats.json').write_text(json.dumps(rows,indent=2)+'\n')
for row in rows[:15]:print(row['name'],row['modules'],row['fingerprints'],row['repeated_modules'])

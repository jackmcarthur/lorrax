"""Compare the provided rule checker's findings without changing its allowlist."""
from pathlib import Path
import json,re
r=Path(__file__).resolve().parent.parent
pattern=re.compile(r'^  ([^:]+): (\S+) has (\d+) match',re.M)
def read(path):
 return {rule+' '+file:int(count) for rule,file,count in pattern.findall(path.read_text())}
before=read(r/'31_cpu_baseline_rules/rules.log')
after=read(r/'30_cpu_repository_gate/gate0.log')
result=dict(baseline=before,candidate=after,added=sorted(set(after)-set(before)),
 increased={k:[before.get(k,0),v] for k,v in after.items() if v>before.get(k,0)})
(r/'rule_scope_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
assert len(before)==17 and len(after)==17 and not result['added'] and not result['increased']
print('17 baseline and17 candidate findings; no added or increased rule counts.')

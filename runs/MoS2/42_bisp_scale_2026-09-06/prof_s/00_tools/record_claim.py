"""Record a local lane verdict through the sandbox's atomic claim writer."""
from pathlib import Path
import argparse
import contextlib
import io
import runpy
import sys
p=argparse.ArgumentParser()
p.add_argument('run',type=Path)
p.add_argument('body')
p.add_argument('--jid', default='57982945')
a=p.parse_args()
w=Path(__file__).resolve().parents[5]
s=Path('/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14')
r=a.run.resolve()
step=next(x for x in (r/'driver.1.log').read_text().splitlines() if '[lx] step ' in x and ' exit ' in x)
body=a.body+' On branch perf/bisp-prof-s-2026-09-06, unmerged.'
mod=runpy.run_path(str(s/'tools/claims_append.py'))
mod['main'].__globals__['CLAIMS']=w/'CLAIMS.md'
sys.argv=['claims_append.py',body,'--evidence','JID'+a.jid+'; '+step,'--artifact',str(r.relative_to(w))]
out=io.StringIO()
with contextlib.redirect_stdout(out):
    assert mod['main']()==0
number=int(out.getvalue().strip())
(w/f'claims/{number:04d}.md').write_text(body+'\n\nJID'+a.jid+'; '+step+'\n\nEvidence: '+str(r.relative_to(w))+'\n')
print(number)

"""Verify every archived member against the recorded content digest."""
import hashlib,json,tarfile
from pathlib import Path
r=json.loads(Path('six_archive_receipt.json').read_text());root=Path(r['destination']);records={x['path']:x for x in json.loads((root/'manifest.json').read_text())};seen=set();bad=[]
with tarfile.open(r['archive'],'r|') as archive:
 for member in archive:
  handle=archive.extractfile(member)
  if handle is None:continue
  digest=hashlib.file_digest(handle,'sha256').hexdigest();seen.add(member.name)
  if digest!=records[member.name]['sha256'] or member.size!=records[member.name]['bytes']:bad.append(member.name)
result={'members_verified':len(seen),'missing':sorted(set(records)-seen),'mismatched':bad,'pass':not bad and seen==set(records)}
Path('six_archive_verification.json').write_text(json.dumps(result,indent=2)+'\n');(root/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
raise SystemExit(0 if result['pass'] else 1)

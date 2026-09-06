"""Archive only the bounded six-by-six lane evidence and exact source snapshots."""
from pathlib import Path
import hashlib,json,tarfile
s=Path(__file__).resolve().parents[3];d=Path(__file__).parent;root=s/'runs/MoS2/42_bisp_scale_2026-09-06/prof_zw';report=s/'tmp/worktrees/wt_bisp_prof_zw_codex_20260906/reports/bisp_prof_zw_2026-09-06';destination=Path('/global/cfs/cdirs/m4598/jackm/lorrax_evidence/bisp_prof_zw_6x6_2026-09-06');paths=set()
for run in root.glob('[0-9][0-9]_*'):
 for path in run.iterdir():
  if path.is_file() and path.suffix in {'.log','.txt','.json','.csv','.dat','.yaml','.sh','.py','.in','.md','.nsys-rep','.diff','.sha256'}:paths.add(path)
 summary=run/'xla_dump_rank0/hlo_summary.json'
 if summary.exists():
  paths.add(summary)
  for data in json.loads(summary.read_text())['modules'].values():
   for key in ('hlo','memory'):
    path=Path((data.get(key) or {}).get('path',''))
    if path.is_file() and path.parent.resolve()==summary.parent.resolve():paths.add(path)
 paths.update((run/'tmp/sigma_quadrature_rules').glob('*.npz'))
 if run.name in ('47_P6_fresh_before','52_F6_fresh_control'):paths.update((run/'tmp').glob('zeta_q*.h5'))
for pattern in ('six_*.py','six_*.json','six_*.log','prepare_6x6.py','prepare_streamed_runs.py','prepare_cpu63.py','prepare_fixed_units.py','prepare_zeta_receipts.py','p16_*.py','p16_*.json','p16_*.log','p16_*.txt','gate0_six*.log','chi_class_profile.py','zeta*profile*.py','gate_candidate.py','source_parent_71ae0bde.tar','source_green_9e31a1a7.tar','source_fixed_e1559a07.tar','archive_six_evidence.py','verify_six_archive.py'):
 paths.update(d.glob(pattern))
paths.update((d/'six_analysis').glob('*.json'))
paths.update(p for p in report.iterdir() if p.is_file());paths.update(p for p in (report/'instruments').iterdir() if p.is_file())
for number in json.loads((d/'six_claim_numbers.json').read_text()):paths.add(s/'claims'/f'{number:04d}.md')
for name in ('CLAIMS.md','RUNS_INFLIGHT.md','KNOWN_SANDBOX_ERRORS.md','KNOWN_LORRAX_ISSUES.md','tools/eqp_ab.py','tools/compare_zeta_h5.py','tools/sigma_diag_rows.py','tools/hlo/analyze_hlo_dump.py','tmp/worktrees/wt_psi_irr_perf2_codex_20260905/tools/profile_collective_census.py'):paths.add(s/name)
paths={p for p in paths if p.is_file()};destination.mkdir(parents=True,exist_ok=True);archive=destination/'evidence.tar';assert not archive.exists();records=[]
with tarfile.open(archive,'w') as bundle:
 for p in sorted(paths):
  with p.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
  name=str(p.relative_to(s));records.append({'path':name,'bytes':p.stat().st_size,'sha256':digest});bundle.add(p,arcname=name,recursive=False)
(destination/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')
with archive.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
receipt={'destination':str(destination),'archive':str(archive),'files':len(records),'archive_bytes':archive.stat().st_size,'sha256':digest,'scope':'six-by-six lane step/science/parser receipts, rank-zero optimized HLO and native captures, canonical P47/F52 zeta, pinned source snapshots; excludes full WFN and other-rank HLO'}
(d/'six_archive_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');(destination/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt)

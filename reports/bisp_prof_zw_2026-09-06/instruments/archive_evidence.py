"""Archive bounded lane receipts, native captures and rank-zero optimized HLO to CFS."""
from pathlib import Path
import hashlib
import json
import tarfile

sandbox=Path(__file__).resolve().parents[3]
destination=Path('/global/cfs/cdirs/m4598/jackm/lorrax_evidence/bisp_prof_zw_2026-09-06')
paths=set()
roots=[sandbox/'runs/MoS2/41_bisp_parent_route_2026-09-05/prof_zw',sandbox/'runs/Si/100_bisp_parent_route_2026-09-05/prof_zw']
for root in roots:
    for run in sorted(root.glob('[0-9][0-9]_*')):
        if not run.is_dir():continue
        for path in run.iterdir():
            if path.is_file() and path.suffix in {'.log','.txt','.json','.csv','.dat','.yaml','.sh','.py','.in','.md','.nsys-rep','.diff','.sha256'}:
                paths.add(path)
        summary=run/'xla_dump_rank0/hlo_summary.json'
        if summary.is_file():
            paths.add(summary)
            for data in json.loads(summary.read_text())['modules'].values():
                for key in ('hlo','memory'):
                    path=Path((data.get(key) or {}).get('path',''))
                    if path.is_file() and path.parent.resolve()==summary.parent.resolve():paths.add(path)
        paths.update((run/'tmp/sigma_quadrature_rules').glob('*.npz'))
        paths.update((run/'replay_rules').glob('*/*.npz'))
        paths.update((run/'common_rules').glob('*/*/*.npz'))
        if run.name in ('09_parent_zeta','24_zeta_source'):
            paths.update((run/'tmp').glob('zeta_q*.h5'))
for path in Path(__file__).parent.iterdir():
    if path.is_file() and path.name!='archive_receipt.json' and path.suffix in {'.log','.json','.py','.sh','.patch','.tar','.txt'}:paths.add(path)
paths.add(Path.cwd()/'baseline_analysis/host.json')
report=sandbox/'tmp/worktrees/wt_bisp_prof_zw_codex_20260906/reports/bisp_prof_zw_2026-09-06'
paths.update(path for path in report.iterdir() if path.is_file())
paths.update(path for path in (report/'instruments').iterdir() if path.is_file())
for number in (996,998,1004,1007,1030,1031,1033,1039,1049,1055,1056,1066,1073,1077,1078):
    paths.add(sandbox/'claims'/f'{number:04d}.md')
for name in ('CLAIMS.md','RUNS_INFLIGHT.md','KNOWN_LORRAX_ISSUES.md','KNOWN_SANDBOX_ERRORS.md','tools/eqp_ab.py','tools/sigma_diag_rows.py','tools/compare_zeta_h5.py','tools/hlo/analyze_hlo_dump.py','tmp/worktrees/wt_psi_irr_perf2_codex_20260905/tools/profile_collective_census.py'):
    paths.add(sandbox/name)
paths={p for p in paths if p.is_file()}
destination.mkdir(parents=True,exist_ok=True)
archive=destination/'evidence.tar';records=[]
print(f'Archiving {len(paths)} bounded files',flush=True)
with tarfile.open(archive,'w') as bundle:
    for index,path in enumerate(sorted(paths)):
        relative=path.relative_to(sandbox)
        with path.open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
        records.append({'path':str(relative),'bytes':path.stat().st_size,'sha256':digest})
        bundle.add(path,arcname=str(relative),recursive=False)
        if index%5000==0:print(index,flush=True)
(destination/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')
with archive.open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
receipt={'destination':str(destination),'archive':str(archive),'sha256':digest,'files':len(records),'payload_bytes':sum(row['bytes'] for row in records),'archive_bytes':archive.stat().st_size,'scope':'named lane logs/manifests/scripts/native captures; rank-zero optimized HLO and memory reports; selected zeta/rules; source snapshots and parser owners; no full WFN inputs or other-rank HLO'}
Path('archive_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
(destination/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(receipt,flush=True)

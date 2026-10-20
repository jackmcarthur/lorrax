"""Guarded native B/C campaign; called collectively inside an owned P4 step."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import runtime


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest')
    p.add_argument('--tags',required=True,help='exact comma-separated selected legs; normally ONE')
    a=p.parse_args()
    plan=json.loads(Path(a.manifest).read_text())
    wanted=set(a.tags.split(','))
    rows=[r for r in plan['runs'] if r['tag'] in wanted]
    if not rows or {r['tag'] for r in rows} != wanted:
        raise ValueError('Requested exact tags missing from manifest')
    from tdagw.td_config import read_td_deck
    decks=[(row,read_td_deck(row['deck'])) for row in rows]
    source=Path(plan['source'])
    if hashlib.sha256(source.read_bytes()).hexdigest() != plan['source_sha256']:
        raise ValueError('Source deck changed after manifest preparation')
    if os.environ.get('LEG_SOURCE_TREE') != plan['source_tree']:
        raise ValueError('Immutable payload tree does not match manifest tree')
    for row in rows:
        if Path(row['out']).exists():
            raise FileExistsError('Refusing existing leg directory: '+row['out'])
    stack=runtime.initialize_communicator_stack()
    import jax
    from tdagw import stages,dynsig
    from tdagw.td_log import TdReport,log_path
    from tdagw.tdgw_io import efield_function
    from runtime.run_session import RunSession
    from observer import install
    from file_io.commit_state import agree_io_refusal
    rank0=jax.process_index()==0
    say=print if rank0 else (lambda *_:None)
    root=Path(a.manifest).resolve().parent
    receipt=dict(status='RUNNING_NOT_COMPLETE',source_tree=plan['source_tree'],
        archive_sha256=os.environ.get('LEG_SOURCE_ARCHIVE_SHA256'),manifest=str(Path(a.manifest).resolve()),
        selected_tags=sorted(wanted),physical_scope=plan['hamiltonian'],runs=[],shared_setup={})
    receipt_path=root/('runs_'+'__'.join(sorted(wanted))+'.json')
    if receipt_path.exists():
        raise FileExistsError('Refusing prior run receipt: '+str(receipt_path))
    def save():
        error=None
        if rank0:
            try:
                receipt_path.write_text(json.dumps(receipt,indent=2,default=str)+'\n')
            except OSError as exc:
                error=exc
        agree_io_refusal(error,path=receipt_path,stage='campaign.native.receipt')
    rc=0;begin=time.perf_counter()
    try:
        if jax.process_count()!=4:
            raise ValueError('This priced campaign requires exactly four ranks')
        first=decks[0][1]
        setup=time.perf_counter();system=stages.load_system(first,stack.mesh)
        model=stages.build_self_energy(first,system,stack.mesh)
        receipt['shared_setup']['system_model_s']=time.perf_counter()-setup;save()
        setup=time.perf_counter()
        source_dir=root/('collision_source_'+'__'.join(sorted(wanted)))
        if source_dir.exists():
            raise FileExistsError('Refusing previous shared source directory: '+str(source_dir))
        source_bundle=dynsig.gw_window_source(plan['source'],stack.mesh,
            workdir=str(source_dir),screening=True,say=say)
        receipt['shared_setup']['source_s']=time.perf_counter()-setup;save()
        from jax.experimental import multihost_utils
        error=None;source_bytes=0
        if rank0:
            try:
                source_bytes=sum(p.stat().st_size for p in source_dir.rglob('*') if p.is_file())
            except OSError as exc:
                error=exc
        agree_io_refusal(error,path=source_dir,stage='campaign.native.source_storage_stat')
        import numpy as np
        source_bytes=int(np.asarray(multihost_utils.broadcast_one_to_all(np.int64(source_bytes),is_source=rank0)))
        receipt['shared_setup']['source_bytes']=source_bytes;save()
        if source_bytes > plan['common_source_storage_gib']*2**30:
            raise ValueError('GATE campaign_resources: common source exceeded explicit storage budget')
        setup_elapsed=float(np.asarray(multihost_utils.broadcast_one_to_all(
            np.float64(time.perf_counter()-begin),is_source=rank0)))
        if setup_elapsed > plan['leg_budget_s']:
            raise ValueError('GATE campaign_resources: common setup exhausted first-leg wall budget')
        for index,(row,deck) in enumerate(decks):
            start=begin if index==0 else time.perf_counter()
            leg=dict(tag=row['tag'],status='RUNNING',out=row['out'],walls_start_s=time.perf_counter()-begin)
            receipt['runs'].append(leg);save()
            restore=install(row['out'],start=start,budget_s=plan['leg_budget_s'],tf_fs=plan['tf_fs'],
                refresh_fs=plan['refresh_fs'],storage_gib=plan['storage_gib'],limits=plan['health'])
            try:
                with RunSession(stack,'tdagw',TdReport,log_path(deck),deck=deck) as run:
                    outputs=stages.open_outputs(deck,system)
                    run.report.header(deck,system,model,stack)
                    stats=stages.propagate(deck,system,model,efield_function(deck.trials),outputs,
                        stack.mesh,say=run.report.echo,log=run.report,collision=source_bundle)
                    run.report.summary(deck,stats)
                    run.complete(files=stages.output_files(outputs))
                leg.update(status='COMPLETE',stats=stats,paths=outputs.paths)
            except Exception:
                leg.update(status='REFUSED_PARTIAL_NOT_COMPLETED',error=traceback.format_exc())
                raise
            finally:
                restore();leg['wall_s']=time.perf_counter()-start;save()
        receipt['status']='COMPLETE_SELECTED_LEGS_ONLY'
    except Exception:
        rc=1;receipt.update(status='REFUSED_NO_COMPLETION_CLAIM',error=traceback.format_exc())
        say(receipt['error'])
    finally:
        receipt['total_wall_s']=time.perf_counter()-begin
        receipt['active_gpu_hours']=4*receipt['total_wall_s']/3600
        save()
    runtime.finalize_process(rc)


if __name__=='__main__':
    main()

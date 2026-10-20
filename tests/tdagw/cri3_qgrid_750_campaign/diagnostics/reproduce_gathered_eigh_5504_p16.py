"""P16 service diagnostic ONLY: recorded gathered-eigh shape, no SC/bank.

PREPARED_NOT_RUN. Ops must enforce kill-on-bad-exit and timeout <=600s.
Compile/load, native execution and the original service residual check have
separate durable markers. Synthetic matrix values are NOT the lost bank.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import threading
import time
import traceback

REVISION='a0c1882d1cc579b1670694615bb00f4b53b15b0d'
CACHE_SHA='9860455fa2b871b29194db0ae785ff91841544b2244d47043f3a7b5359a91a8e'
SIDE=5504


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--max-wall-seconds',type=float,default=540.)
    p.add_argument('--device-ceiling-gib',type=float,default=8.)
    return p


def main():
    args=parser().parse_args()
    if not 0<args.max_wall_seconds<=600 or not 0<args.device_ceiling_gib<=8:
        raise ValueError('bounded600s/eightGiB-per-rank diagnostic required')
    if os.environ.get('LEG_HEAD')!=REVISION:
        raise ValueError('launcher LEG_HEAD must bind reviewed a0c1882 source')
    started=time.monotonic()
    timer=threading.Timer(args.max_wall_seconds,lambda:os._exit(124))
    timer.daemon=True;timer.start()
    # Actual driver bootstrap initializes MPI/JAX before any NumPy import.
    import gw.gw_jax as driver
    import runtime
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding,PartitionSpec as P
    from common.collectives import (device_put_process_tiles,gather_to_host,
        rank0_transaction,all_gather_processes)
    from common.gpu_utils import device_budget_bytes
    from common.jax_compile_cache import compile_cache_stats
    from distrib_la.workspace import local_eigh_workspace
    from distrib_la._result_check import eigh_errors,accept
    plan_module=importlib.import_module('distrib_la.plan')
    jax.config.update('jax_enable_x64',True)
    source=Path(args.source_root).resolve()
    if source not in Path(driver.__file__).resolve().parents:
        raise ValueError('driver source-origin mismatch')
    if source not in Path(plan_module.__file__).resolve().parents:
        raise ValueError('distrib_la owner source-origin mismatch')
    mesh=driver.RUNTIME.mesh
    if (jax.process_count()!=16 or jax.local_device_count()!=1
            or tuple(mesh.devices.shape)!=(4,4)
            or any(d.platform!='gpu' for d in jax.devices())):
        raise ValueError('exact16GPU-rank4x4 service fixture required')
    out=Path(args.out).resolve()
    report=dict(status='RUNNING_SERVICE_DIAGNOSTIC',SC_run=False,physical_result=False,
        revision=REVISION,archive_sha256=os.environ.get('LEG_SOURCE_ARCHIVE_SHA256'),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        recorded_cache_sha256=CACHE_SHA,input_shape=[SIDE,SIDE],dtype='complex128',
        input_sharding="P('x','y')",output_shardings=["P()","P('x','y')"],mesh=[4,4],
        exact_scope='recorded shape/layout/owner; deterministic synthetic values, NOT lost bank',
        matrix='nontrivial Hermitian positive diagonal+complex rank-one; last8 exactzero rows',
        runtime_budget_bytes=int(device_budget_bytes()),device_ceiling_bytes=int(args.device_ceiling_gib*2**30),
        max_seconds=args.max_wall_seconds,max_active_GPUh=16*args.max_wall_seconds/3600,
        phases={},unknown_stages_not_run=['normal_eigh batch shape','constructor face-pencil shape'])
    if report['device_ceiling_bytes']>report['runtime_budget_bytes']:
        raise ValueError('declared device ceiling exceeds actual runtime budget')
    def create():
        if out.exists():raise FileExistsError(str(out))
        out.mkdir(parents=True)
    rank0_transaction(out,stage='gathered5504/new_output',write=create)
    def save():
        def write():
            temporary=out/'receipt.tmp.json'
            temporary.write_text(json.dumps(report,indent=2,allow_nan=False,default=str)+'\n')
            os.replace(temporary,out/'receipt.json')
        rank0_transaction(out,stage='gathered5504/receipt',write=write)
    def mark(name,state,**metadata):
        row=dict(marker=name,state=state,elapsed_s=time.monotonic()-started,
                 process=jax.process_index(),**metadata)
        print(json.dumps(row,sort_keys=True),flush=True)
        report['last_marker']=row;save()
    def phase(name,fn):
        elapsed=np.asarray(all_gather_processes(np.asarray(time.monotonic()-started))).max()
        if elapsed>=args.max_wall_seconds:raise TimeoutError('service600s total budget')
        mark(name,'BEGIN');before=time.monotonic()
        result=fn()
        jax.block_until_ready(result)
        seconds=np.asarray(all_gather_processes(np.asarray(time.monotonic()-before))).reshape(-1)
        report['phases'][name]=dict(min_seconds=float(seconds.min()),max_seconds=float(seconds.max()))
        mark(name,'END')
        return result

    # Each process creates only its addressable face shard, ~30.3MiB.
    # This formula is Hermitian at every global index, independent of rank.
    def tile(index):
        rows=np.arange(*index[0].indices(SIDE),dtype=np.int64)
        cols=np.arange(*index[1].indices(SIDE),dtype=np.int64)
        u=np.exp(.017j*rows)/np.sqrt(SIDE)
        v=np.exp(.017j*cols)/np.sqrt(SIDE)
        a=.1*u[:,None]*v[None,:].conj()
        a+=(rows[:,None]==cols[None,:])*(1.+rows[:,None]/SIDE)
        return np.ascontiguousarray(a*((rows[:,None]<SIDE-8)&(cols[None,:]<SIDE-8)),dtype=np.complex128)
    rc=0
    try:
        mark('fixture_scope','RECORDED',cache_input_minor_to_major=[1,0],synthetic_values=True)
        A=phase('place_native_face',lambda:device_put_process_tiles(
            (SIDE,SIDE),NamedSharding(mesh,P('x','y')),tile))
        program=plan_module._gathered_program(mesh)
        # This owner is exactly the cached jit__gathered_eigh, with unchanged
        # native_eigh/zero-row deflation and CUDA jnp.linalg.eigh provider.
        cache_before=compile_cache_stats()
        executable=phase('compile_or_load_exact_gathered_eigh',lambda:program.lower(A).compile())
        cache_after=compile_cache_stats()
        deltas=np.asarray(all_gather_processes(np.asarray([
            cache_after[k]-cache_before[k] for k in ('probes','hits','compiles')],np.int64))).reshape(16,3)
        report['compile_cache']=dict(forced_cold=False,dir=cache_after['dir'],
            columns=['probes','hits','backend_compiles'],rank_deltas=deltas.tolist(),
            scope='owner counters during compile-or-load phase; no forced cold rerun',
            backend_compilation_observed=bool(np.any(deltas[:,2]>0)),
            persistent_hit_observed=bool(np.any(deltas[:,1]>0)))
        mark('compile_cache_state','RECORDED',**report['compile_cache'])
        stats=executable.memory_analysis()
        compiled=int(stats.argument_size_in_bytes+stats.output_size_in_bytes+
                     stats.temp_size_in_bytes-stats.alias_size_in_bytes)
        vendor=phase('native_local_eigh_workspace_query',lambda:
            local_eigh_workspace(mesh,1,SIDE,np.dtype(np.complex128)))
        prices=np.asarray(all_gather_processes(np.asarray([compiled,int(vendor)],np.int64)))
        report['memory']=dict(compiled_bytes_per_rank=int(prices[:,0].max()),
            vendor_workspace_bytes_per_rank=int(prices[:,1].max()),
            aggregate_bound_bytes_per_rank=int(prices.sum(axis=1).max()))
        if report['memory']['aggregate_bound_bytes_per_rank']>report['device_ceiling_bytes']:
            raise MemoryError('compiled+native-query bound exceeds explicit diagnostic ceiling')
        hlo=executable.as_text()
        targets=sorted(set(re.findall(r'custom_call_target="([^"]+)"',hlo)))
        report['custom_call_targets']=targets
        report['compiled_hlo_sha256']=hashlib.sha256(hlo.encode()).hexdigest()
        def save_hlo():
            (out/'compiled_gathered_eigh.hlo.txt').write_text(hlo)
        rank0_transaction(out,stage='gathered5504/hlo',write=save_hlo)
        mark('provider_and_price','RECORDED',custom_call_targets=targets,**report['memory'])
        values,vectors=phase('execute_exact_gathered_eigh',lambda:executable(A))
        # Existing service error equations and acceptance threshold; only
        # their two scalars reach hosts, never A or the eigenvector matrix.
        checker=jax.jit(lambda a,w,v:jnp.stack(eigh_errors(a,w,v,mesh=mesh)),
            out_shardings=NamedSharding(mesh,P()))
        check=phase('compile_service_probe_check',lambda:checker.lower(A,values,vectors).compile())
        errors=phase('execute_service_probe_check',lambda:check(A,values,vectors))
        errors=np.asarray(gather_to_host(errors))
        report['errors']=dict(residual=float(errors[0]),orthogonality=float(errors[1]),
                              unchanged_service_tolerance=float(accept(SIDE,np.complex128)))
        if not np.isfinite(errors).all() or np.max(errors)>accept(SIDE,np.complex128):
            raise ValueError('unchanged distrib_la result check refuses synthetic result')
        report['status']='PASS_RECORDED_SHAPE_SYNTHETIC_VALUES_ONLY'
        mark('diagnostic_complete','PASS',**report['errors'])
    except BaseException as error:
        trace=traceback.format_exc()
        print(f'PYTHON_FAILURE rank={jax.process_index()}\n{trace}',flush=True)
        rc=1;report.update(status='REFUSED_OR_FAILED_SERVICE_DIAGNOSTIC',
            error=repr(error),traceback=trace)
        # C-level crashes do not reach this block. Their last durable BEGIN
        # marker and launcher/core logs remain the authoritative evidence.
    finally:
        report['elapsed_s']=time.monotonic()-started;save();timer.cancel()
    runtime.finalize_process(rc)
    return rc


if __name__=='__main__':raise SystemExit(main())

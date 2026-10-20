"""Prepared P4 CUDA equivalence proof, NOT a material or speedup benchmark."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import traceback

REVISION = '57b30151fba085f6478ea3c1939f7066eb7e57bf'


def compare(mesh, say):
    import jax
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local, gather_to_host
    from common.gpu_utils import shared_device_room
    from gw.shared_pole_directions import port_extent
    from gw.shared_pole_execution import face_reduce_round, face_ritz_carrier
    from gw.shared_pole_local import parent_rounds, round_tables
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1
    from lxkit import memory
    from tests.test_shared_pole_decoupled import _pencil, _pad, _w, _dw, _moment

    # Only this tiny host fixture is replicated. No material data is read.
    states, infinity, _, exact = _pencil(mesh, seed=41, n=12, k=12, width=2, iw=2, q=10)
    states = [(node, q,
        np.stack([_w(*exact[p], node[p]**2) @ q[p] for p in range(10)]),
        np.stack([2*node[p]*_dw(*exact[p], node[p]**2) @ q[p] for p in range(10)]))
        for node, q, _, _ in states]
    qi = infinity[0]
    moments = [np.zeros((10,12,12),complex),
        np.stack([.5*_moment(*m,0) for m in exact]),
        np.zeros((10,12,12),complex), np.stack([.5*_moment(*m,1) for m in exact])]
    infinity = (qi, *(np.einsum('pij,pjk->pik',m,qi) for m in moments))
    extent = port_extent(mesh)
    pw, iw = extent(2), extent(2)
    face = NamedSharding(mesh,P(None,'x','y'))
    put = lambda a: device_put_process_local(np.ascontiguousarray(a),face)
    gram_keep = shared_real_pole_gates_ordered_v1['normalized_gram_keep']['sector_threshold']
    gates = ('gram_valid','retained_metric_positive','orientation_paired','retained_rank')
    phase_rows = []

    def evaluate(width):
        models, diagnostics = [], []
        for number,(ids,real,_) in enumerate(parent_rounds(10,width)):
            start=time.monotonic()
            say(f'BEGIN width={width} round={number} real={real} native_auto_eigh')
            selected=[(s[0][ids],put(_pad(s[1][ids],pw)),
                       put(_pad(s[2][ids],pw)),put(_pad(s[3][ids],pw))) for s in states]
            inf=tuple(put(_pad(a[ids],iw)) for a in infinity)
            table=round_tables(np.full((width,len(states)),2),[pw]*len(states),
                [s[0][0] for s in states],[2]*width,iw,
                column_extent=extent,ordered=True,odd_moments=True)
            # Real provider/compiled program checks stay enabled, not a mock
            # eigenplan or an unpriced alternate mathematical executor.
            resident=sum(a.nbytes for a in jax.tree.leaves((selected,inf)) if isinstance(a,jax.Array))
            beside=memory.Price(xla=memory.Term(0.,float(resident),
                source='actual tiny face input panels'),ranks=mesh.size,label='P4 tiny inputs')
            room=shared_device_room('P4 balanced face proof')
            result=face_reduce_round(selected,inf,table,mesh=mesh,budget=None,
                ordered=True,odd_moments=True,keep_budget=4,eigh_backend='auto',
                room=room,admit=False,beside=beside,gram_keep=gram_keep,
                carrier=face_ritz_carrier(mesh,4))
            host=jax.tree.map(lambda a:np.asarray(gather_to_host(a)),result)
            for i in range(real):
                models.append(tuple(a[i] for a in host[0]))
                diagnostics.append({k:v[i] for k,v in host[3][0].items()})
            seconds=time.monotonic()-start
            phase_rows.append(dict(width=width,round=number,real_parents=real,
                                   seconds=seconds,compiled_admission=True))
            say(f'END width={width} round={number} seconds={seconds:.3f}')
            del selected,inf,result,host
        return models,diagnostics

    padded,pd=evaluate(9)
    balanced,bd=evaluate(5)
    max_pole=max_transfer=0.
    for old,new,a,b in zip(padded,balanced,pd,bd):
        factor,pole,active=old
        fn,pn,an=new
        np.testing.assert_array_equal(active,an)
        np.testing.assert_allclose(pole[active],pn[active],rtol=2e-8,atol=2e-10)
        max_pole=max(max_pole,float(np.max(abs(pole[active]-pn[active]))))
        for z in (-.3+.4j,1.2+.7j):
            w=(factor*(active/(z-pole))[None])@factor.conj().T
            wn=(fn*(an/(z-pn))[None])@fn.conj().T
            np.testing.assert_allclose(w,wn,rtol=3e-7,atol=3e-9)
            max_transfer=max(max_transfer,float(np.max(abs(w-wn))))
        for key in gates:
            np.testing.assert_array_equal(a[key],b[key])
            if not bool(a[key]):raise AssertionError(f'gate failed: {key}')
    return dict(parents=10,widths=[9,5],rounds_each=2,slots=[18,10],
        max_pole_absolute_difference=max_pole,max_complex_transfer_absolute_difference=max_transfer,
        passing_gate_rows=list(gates),phases=phase_rows,
        scope='native_CUDA_MPI_face_round_equivalence_NOT_material_or_speedup',
        admission_scope='actual_eigh_provider_and_compiled_round_NOT_full_constructor_chooser')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--max-wall-seconds',type=float,default=120.)
    a=p.parse_args()
    if not 0<a.max_wall_seconds<=120:raise ValueError('wall cap must be <=120s')
    if os.environ.get('LEG_HEAD')!=REVISION:raise ValueError('launcher source revision mismatch')
    timer=threading.Timer(a.max_wall_seconds,lambda:os._exit(124));timer.daemon=True;timer.start()
    import gw.gw_jax as driver  # Native collective bootstrap BEFORE arrays.
    import runtime
    import jax
    from common.collectives import rank0_transaction
    root=Path(a.source_root).resolve()
    if root not in Path(driver.__file__).resolve().parents:raise ValueError('import origin mismatch')
    mesh=driver.RUNTIME.mesh
    if (jax.process_count()!=4 or jax.local_device_count()!=1 or tuple(mesh.devices.shape)!=(2,2)
            or any(d.platform!='gpu' for d in jax.devices()) or not jax.config.x64_enabled):
        raise ValueError('actual CUDA P4 mesh2x2 and production FP64 required')
    out=Path(a.out).resolve()
    def create():out.mkdir(parents=True,exist_ok=False)
    rank0_transaction(out,stage='balanced_face/new_output',write=create)
    report=dict(status='RUNNING_SOFTWARE_ONLY',revision=REVISION,material_admitted=False,
        source_root=str(root),archive_sha256=os.environ.get('LEG_SOURCE_ARCHIVE_SHA256'),
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        max_wall_seconds=a.max_wall_seconds,max_active_GPUh=4*a.max_wall_seconds/3600)
    say=lambda msg:print(msg,flush=True) if jax.process_index()==0 else None
    rc=0;start=time.monotonic()
    try:
        report['result']=compare(mesh,say)
        report['status']='PASS_SOFTWARE_ONLY_NOT_MATERIAL_OR_SPEEDUP'
    except BaseException as e:
        rc=1;report.update(status='REFUSED_SOFTWARE_PROOF',error=repr(e),traceback=traceback.format_exc())
    finally:
        report['elapsed_s']=time.monotonic()-start
        def save():
            with (out/'receipt.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
        rank0_transaction(out,stage='balanced_face/result',write=save)
        timer.cancel()
    runtime.finalize_process(rc)
    return rc


if __name__=='__main__':raise SystemExit(main())

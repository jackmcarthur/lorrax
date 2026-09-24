"""P4 photon stream admission: compiled workspace, donated carry, hard refusal.

Tiny plant only; does not certify endpoint preparation or native workspace.
"""
from runtime import initialize_communicator_stack, run_main_and_finalize
initialize_communicator_stack()

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from common.collectives import resolve_mesh, gather_to_host
from gw.response_bank import _bank_execution
from gw.shared_pole_recipe import CapacityLedger
from gw.w_isdf import _get_chi_fractional_contour_kernel_face


def main():
    mesh = resolve_mesh()
    assert mesh.size == 4
    root = Path(sys.argv[1])
    nk, nb, ns, n = 8, 8, 4, 16
    rng = np.random.default_rng(477)
    psi = (rng.normal(size=(nk, ns, n, nb))
           + 1j*rng.normal(size=(nk, ns, n, nb))) / 16
    current = psi * np.array([1., -1., 1j, -1j])[None, :, None, None]
    def put(value, spec):
        value = np.asarray(value)
        return jax.make_array_from_callback(value.shape, NamedSharding(mesh, spec),
                                             lambda ix: value[ix])
    mun = tuple(put(v, P(None,None,'x','y')) for v in (psi,current))
    nmu = tuple(put(v.conj().transpose(0,3,1,2), P(None,'x',None,'y'))
                for v in (psi,current))
    energy = np.broadcast_to(np.linspace(-1,2,nb),(nk,nb)).copy()
    f = 1/(1+np.exp(energy/.3))
    args = (put(np.array([0.,.3,1.]),P()),
            put(np.array([[1.,.4j,.2],[.2j,1.,.7j]]),P()),
            mun,nmu,put(energy,P()),put(f,P()),put(1-f,P()),put(np.array(0.),P()))
    carry_spec = P(None,None,'x','y')
    def zero():
        return put(np.zeros((2,3,n,n),complex),carry_spec)
    kernel = _get_chi_fractional_contour_kernel_face(mesh,(2,2,2),2,
        (nk,nb,n,ns),selected_q=(0,3,7),ordered=True,vertex=True,bank_carry=True)
    shape_args = (*args,zero())
    compiled = kernel.lower(*shape_args).compile()
    memory = compiled.memory_analysis()
    assert memory.temp_size_in_bytes > 0 and memory.alias_size_in_bytes > 0
    reference = compiled(*shape_args)
    jax.block_until_ready(reference)
    # All actual stream inputs are priced once. Carry donation means no extra
    # output reservation is required for this selected stream invocation.
    ambient = memory.argument_size_in_bytes
    def setup(limit):
        meta = SimpleNamespace(nk_tot=nk,nspinor=ns,n_rmu=n)
        ledger = CapacityLedger(meta,mesh_xy=mesh,device_budget_bytes=limit)
        ledger.live_stages = ()
        ledger.reserve('inputs',resident_bytes_per_rank=ambient,
                       workspace_bytes_per_rank=0)
        ledger.live_stages = ('inputs',)
        meta.shared_pole_capacity = ledger
        receipt = dict(seconds={},memory=[],compiled=[])
        return meta,receipt
    needed = ambient + memory.temp_size_in_bytes
    meta,receipt = setup(needed)
    execute = _bank_execution(meta,mesh,receipt,{'linalg':'distributed'},photon=True)
    actual = execute(kernel,(*args,zero()),'real_time')
    relative = float(gather_to_host(jnp.linalg.norm(actual-reference)/jnp.linalg.norm(reference)))
    assert relative < 1e-13, relative
    row = receipt['memory'][0]
    assert row['aggregate_bytes_per_rank'] == needed
    assert row['workspace_bytes_per_rank'] == memory.temp_size_in_bytes
    assert receipt['compiled'][0]['stream_temporaries_admitted']
    assert not receipt['compiled'][0]['inherited_stream']
    assert actual.sharding.spec == carry_spec
    refused_meta,refused_receipt = setup(needed-1)
    refused = False
    try:
        _bank_execution(refused_meta,mesh,refused_receipt,{},photon=True)(
            kernel,(*args,zero()),'real_time')
    except MemoryError as exc:
        assert 'GATE shared_pole_capacity' in str(exc)
        refused = True
    assert refused and 'real_time' not in refused_receipt['seconds']
    result = dict(schema='lorrax.response-bank-memory-gate.v1',complete=True,
        job=os.environ.get('SLURM_JOB_ID'),step=os.environ.get('SLURM_STEP_ID'),
        relative=relative,refused_before_execution=refused,
        needed_bytes_per_rank=needed,accepted=receipt,
        refused=refused_meta.shared_pole_capacity.entries[-1],
        scope='P4 planted photon contour stream; no material peak or native workspace proof')
    if jax.process_index() == 0:
        (root/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
        (root/'photon_stream.hlo').write_text(compiled.as_text())
    print('photon stream memory admission PASS',flush=True)


if __name__ == '__main__':
    run_main_and_finalize(main)   # a failure keeps its traceback and a nonzero status

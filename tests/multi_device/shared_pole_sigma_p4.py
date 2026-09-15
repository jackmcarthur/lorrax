"""P4 planted shared-pole synthesis gate; no frozen-Na or Σ accuracy claim.

Both mesh axes exceed one. Tiny dense fixtures are independent algebra
oracles; their test-only host copies are never production factor carriers.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re


def main(runtime):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local
    from common.shard_map import shard_map
    from gw.mpa.sigma import synthesize_shared_pole_parents
    from gw.mpa.sigma_windows import shared_pole_frequencies, shared_pole_intervals
    from gw.ppm_tau_kernel import build_shared_w_tau
    from symmetry_maps import unfold_operator_local

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    mesh = runtime.mesh
    assert jax.process_count() == jax.device_count() == 4
    assert mesh.shape['x'] > 1 and mesh.shape['y'] > 1
    assert all(d.platform == 'gpu' for d in mesh.devices.flat)
    import pytest
    import xml.etree.ElementTree as ET
    junit = args.output / f'metadata_rank{jax.process_index()}.xml'
    assert pytest.main(['-q', 'tests/test_shared_pole_sigma.py',
                        '-p', 'no:cacheprovider', f'--junitxml={junit}']) == 0
    suites = ET.parse(junit).getroot().iter('testsuite')
    assert sum(int(s.get('tests', 0)) for s in suites) == 9

    rng = np.random.default_rng(321)
    n, parents, width = 8, 3, 8
    C = rng.normal(size=(parents,n,1,width)) + 1j*rng.normal(size=(parents,n,1,width))
    C[:,6:] = 0  # logical six centroids, carrier eight
    K = np.asarray([5,7,3], np.int64)
    omega = np.asarray([[1,2,2,3,4,1,1], [1,2,3,4,5,6,7],
                        [1,2,4,1,1,1,1]], dtype=np.float64)
    omega = np.pad(omega, ((0,0),(0,1)), constant_values=1)
    for q,k in enumerate(K):
        C[q,:,:,k:] = 0
    bounds = np.tile([1,4,-np.inf,-np.inf,np.inf,np.inf], (parents,1))
    intervals = shared_pole_intervals(shared_pole_frequencies(omega**2,K),
                                      np.arange(parents),bounds)
    put = lambda a,spec: device_put_process_local(np.asarray(a),NamedSharding(mesh,spec))
    X,Y = put(C,P(None,'x',None,'y')),put(C,P(None,'y',None,'x'))
    lam,iv = put(omega**2,P()),put(intervals,P())
    row_map = np.asarray([0,1,2,0,1,2],np.int32)
    op_map = np.asarray([0,0,0,1,1,1],np.int32)
    qfrac = np.asarray([[.125,0,0],[.25,0,0],[.375,0,0]])
    perm = np.stack([np.arange(n), np.arange(n)^1]).astype(np.int32)
    wraps = np.zeros((2,n,3),np.int32)
    wraps[1,:,0] = np.arange(n)%2
    # Endpoints are demonstrably shard-local for BOTH named mesh axes.
    for axis in ('x','y'):
        extent=n//mesh.shape[axis]
        assert np.all(perm//extent == np.arange(n)[None,:]//extent)
    left=perm%(n//mesh.shape['x']);right=perm%(n//mesh.shape['y'])

    def unfold(a,b):
        return unfold_operator_local(
            a,irr_idx=row_map,sym_idx=op_map,q_irr_frac=qfrac,
            left_local_perm=left,left_L_table=wraps,
            right_local_perm=right,right_L_table=wraps,n_sym_spatial=1,
            trs_rule='pair_transpose',transposed_parent_local=b)
    unfold_sharded=shard_map(unfold,mesh=mesh,
        in_specs=(P(None,'x','y'),P(None,'x','y')),
        out_specs=P(None,'x','y'),check_vma=False)

    from distrib_la import gemm_plan
    gemm = gemm_plan(mesh,m=n,k=width,n=n,nq=parents,dtype=np.complex128)

    @jax.jit
    def synth(x,y,l,r,e,t):
        return synthesize_shared_pole_parents(x,y,l,r,e,t,mesh_xy=mesh,gemm=gemm)

    @jax.jit
    def full(x,y,l,r,e,t):
        a,b=synthesize_shared_pole_parents(x,y,l,r,e,t,mesh_xy=mesh,gemm=gemm)
        return unfold_sharded(a,b)

    results=[]
    for time_node in (.7+.2j, -.7+.2j, -1.2j):
        ref=.6
        e,t=put(np.asarray(ref),P()),put(np.asarray(time_node),P())
        selected=np.zeros_like(omega,dtype=bool)
        for q,(lo,hi) in enumerate(intervals):selected[q,lo:hi]=True
        d=np.where(selected,np.exp(-1j*(omega-ref)*time_node)/(2*omega),0)
        f=C[:,:,0,:]
        a=np.einsum('qik,qk,qjk->qij',f,d,f.conj())
        b=np.einsum('qik,qk,qjk->qij',f.conj(),d,f)
        expected=[]
        for q,op in zip(row_map,op_map):
            phase=np.exp(2j*np.pi*(wraps[op]@qfrac[q]))
            tile=(b if op else a)[q][np.ix_(perm[op],perm[op])]
            if op:phase=phase.conj()
            expected.append(phase[:,None]*tile*phase.conj()[None,:])
        expected=np.asarray(expected)
        operands=(X,Y,lam,iv,e,t)
        compiled=full.lower(*operands).compile()
        got=compiled(*operands)
        oracle=put(expected,P(None,'x','y'))
        error=float(jax.device_get(jnp.max(jnp.abs(got-oracle))))
        assert error < 1e-11,error
        plus,trans=synth(*operands)
        wrong=float(jax.device_get(jnp.max(jnp.abs(trans-jnp.conj(plus)))))
        if time_node.real:
            assert wrong > 1e-3, 'real-time antiunitary red twin is not discriminating'
        else:
            # Laplace weights are real: conjugation and transpose coincide
            # here, so this node cannot diagnose the wrong real-time rule.
            assert wrong < 1e-11, wrong
        # Compare with the incumbent residue-sum owner on this tiny fixture.
        residues=np.einsum('qik,qjk->kqij',f,f.conj())/(2*omega.T[:,:,None,None])
        omega_fields=np.broadcast_to(omega.T[:,:,None,None],residues.shape).copy()
        legacy=jax.jit(build_shared_w_tau)(
            put(residues,P(None,None,'x','y')),
            put(omega_fields,P(None,None,'x','y')),
            put(np.arange(width,dtype=np.int32),P()),
            put(np.tile(bounds[0],(width,1)),P()),
            put(np.ones(width,dtype=bool),P()),e,t)
        carrier_error=float(jax.device_get(jnp.max(jnp.abs(plus-legacy))))
        assert carrier_error < 1e-11,carrier_error
        results.append(dict(time=[time_node.real,time_node.imag],
                            max_abs_W_error=error,carrier_max_abs_W_error=carrier_error,
                            wrong_antiunitary_delta=wrong))
    red_errors = {}
    for name, bad_d in (
        ('missing_2omega', d * (2 * omega)),
        ('double_eta', d * np.exp(-(.25 / 13.605693122994) * abs(time_node))),
    ):
        from gw.mpa.sigma import _shared_pole_contract
        wrong = _shared_pole_contract(X,Y,put(bad_d,P()),gemm=gemm)
        delta = float(jax.device_get(jnp.max(jnp.abs(wrong-plus))))
        assert delta > 1e-3,(name,delta)
        red_errors[name] = delta
    permuted = C.copy()
    for q,k in enumerate(K):permuted[q,:,:,:k] = np.roll(C[q,:,:,:k],1,axis=-1)
    wrong,_ = synth(put(permuted,P(None,'x',None,None)),
                    put(permuted,P(None,'y',None,None)),lam,iv,e,t)
    delta = float(jax.device_get(jnp.max(jnp.abs(wrong-plus))))
    assert delta > 1e-3,delta
    red_errors['permuted_C_without_Lambda'] = delta
    omitted = expected.copy()
    omitted[-1] = 0
    delta = float(np.max(np.abs(omitted-expected)))
    assert delta > 1e-3,delta
    red_errors['missing_q'] = delta
    # The HLO receipt isolates synthesis; spatial/G/projection are out of scope.
    compiled=synth.lower(*operands).compile()
    hlo=compiled.as_text()
    collective_lines=[line for line in hlo.splitlines() if re.search(
        r'\b(all-gather|all-reduce|all-to-all|collective-permute|reduce-scatter)\(',line)]
    assert not any('all-gather(' in line for line in collective_lines),collective_lines
    assert 'lorrax_cublasmp_batched_gemm' in hlo
    # Native GEMM communication is inside the provider, not visible as HLO
    # collectives. The separate reader/packing gate owns its all-to-all proof.
    memory=compiled.memory_analysis()
    report=dict(status='PASS',scope='P4 planted W and local pair-transpose unfold; not frozen Sigma',
        job_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
        results=results,red_errors=red_errors,synthesis_collectives=collective_lines,
        compiled_argument_bytes=memory.argument_size_in_bytes,
        compiled_output_bytes=memory.output_size_in_bytes,
        compiled_temporary_bytes=memory.temp_size_in_bytes,
        aggregate_3U='NOT_MEASURED',native_timeline='NOT_MEASURED',
        frozen_P16='NOT_MEASURED')
    (args.output/f'synthesis_rank{jax.process_index()}.hlo').write_text(hlo)
    (args.output/f'receipt_rank{jax.process_index()}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    from runtime import initialize_communicator_stack, run_main_and_finalize
    runtime=initialize_communicator_stack()
    run_main_and_finalize(lambda:main(runtime))

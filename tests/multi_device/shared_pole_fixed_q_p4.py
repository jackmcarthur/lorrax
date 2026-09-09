"""Ruling23: P4 pair-transpose owner, negative twin and production E call."""
from pathlib import Path
import json
import os
import runpy
import sys


def main(rt):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from symmetry_maps import build_qgrid_trs_policy
    from common.shard_map import shard_map
    from dataclasses import replace

    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    mesh = rt.mesh
    assert jax.process_count() == jax.device_count() == 4
    assert mesh.shape['x'] > 1 and mesh.shape['y'] > 1
    policy = build_qgrid_trs_policy(trs_measured=True, kgrid=(4,1,1),
        irr_idx_q=np.array([0,1,2,1]), sym_idx_q=np.array([0,0,0,1]),
        q_irr_full_idx=np.array([0,1,2]), n_sym_spatial=1)
    rng = np.random.default_rng(23)
    w = rng.normal(size=(3,8,8))+1j*rng.normal(size=(3,8,8))
    wt = w.transpose(0,2,1).copy()
    oracle = w.copy(); oracle[[0,2]] = 0.5*(w[[0,2]]+wt[[0,2]])
    def put(a):
        return jax.make_array_from_callback(a.shape, NamedSharding(mesh,P(None,'x','y')), lambda i:a[i])
    a,b = put(w),put(wt)
    def body(x,y):
        return policy.project_fixed_q(x,np.arange(3),transposed_partner=y,measure=False)[0]
    compiled = jax.jit(shard_map(body,mesh=mesh,in_specs=(P(None,'x','y'),)*2,
        out_specs=P(None,'x','y'),check_vma=False)).lower(a,b).compile()
    good = float(jnp.max(jnp.abs(compiled(a,b)-put(oracle))))
    wrong = float(jnp.max(jnp.abs(compiled(a,a.conj())-put(oracle))))
    assert good < 1e-13 and wrong > 0.1, (good,wrong)
    _,norms = jax.jit(lambda x,y:policy.project_fixed_q(
        x,np.arange(3),transposed_partner=y))(a,b)
    expected = np.linalg.norm(w-wt,axis=(1,2))/np.linalg.norm(w,axis=(1,2)); expected[1]=0
    norm_error=float(jnp.max(jnp.abs(norms-jnp.asarray(expected))))
    assert norm_error < 1e-13
    broken=replace(policy,trs_measured=False)
    unchanged,_=broken.project_fixed_q(a,np.arange(3),transposed_partner=b)
    assert float(jnp.max(jnp.abs(unchanged-a))) == 0
    try:
        policy.project_fixed_q(a,np.arange(3),transposed_partner=b[:2])
    except ValueError: pass
    else: raise AssertionError('mismatched partner accepted')
    hlo=compiled.as_text().lower()
    forbidden=[name for name in ('all-gather','all-to-all','all-reduce','collective-permute') if name in hlo]
    assert not forbidden,forbidden
    receipt=dict(status='PASS',job_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
        owner_max_error=good,wrong_partner_error=wrong,norm_max_error=norm_error,
        per_parent_asymmetry=expected.tolist(),projection_body_collectives=forbidden,
        scope='P4 2x2 owner: Gamma and zone boundary projected; ordinary parent unchanged; TRS-broken identity')
    (out/f'owner_rank{jax.process_index()}.json').write_text(json.dumps(receipt,indent=2)+'\n')
    for name,script in [('store','shared_pole_sigma_store_p4.py'),('nonlocal','shared_pole_sigma_nonlocal_p4.py')]:
        sys.argv=[script,'--output',str(out/name)]
        runpy.run_path('tests/multi_device/'+script)['main'](rt)
    (out/f'complete_rank{jax.process_index()}.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__ == '__main__':
    from runtime import initialize_communicator_stack,run_main_and_finalize
    rt=initialize_communicator_stack()
    run_main_and_finalize(lambda:main(rt))

"""Distributed panel projection parity, tails, and physical mesh ordering.

Run with one process per device through lx run. Tiny independent NumPy
references are replicated only in this test; production operands stay sharded.
"""
import json
import os
from pathlib import Path

import sys
if '--cpu-emulation' in sys.argv:
    root = Path(__file__).resolve().parents[2]
    for service in ('distrib_la', 'lxkit'):
        sys.path.insert(0, str(root / 'services' / service / 'src'))
if "--cpu-emulation" not in sys.argv:
    from runtime import initialize_communicator_stack
    rt = initialize_communicator_stack()
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from distrib_la import band_projection_plan


def put(value, sharding):
    return jax.make_array_from_callback(value.shape, sharding, lambda idx: value[idx])


def main():
    mesh0 = Mesh(np.asarray(jax.devices()).reshape((2, 2)), ('x','y')) if '--cpu-emulation' in sys.argv else rt.mesh
    p = int(mesh0.shape['x'])
    rng = np.random.default_rng(421)
    rows = []
    for reverse in (False, True):
        mesh = Mesh(mesh0.devices.reshape(-1)[::-1].reshape((p, p)), ('x', 'y')) if reverse else mesh0
        for dtype in (np.float64, np.complex128):
            for n, limit in ((p, 1), (65*p, 64), (7*p, 3*p)):
                m, k, nq = 5*p, 9*p, 3
                a = rng.normal(size=(nq,m,k)).astype(dtype)
                b = rng.normal(size=(nq,k,n)).astype(dtype)
                if dtype == np.complex128:
                    a += 1j*rng.normal(size=a.shape)
                    b += 1j*rng.normal(size=b.shape)
                ref = a @ b
                for layout in ('face', 'axis'):
                    for axis in ('x','y'):
                        plan = band_projection_plan(mesh,m=m,k=k,n=n,nq=nq,
                            dtype=dtype,layout=layout,reduction_axis=axis,panel_columns=limit)
                        result = plan(put(a,plan.in_sharding_a),put(b,plan.in_sharding_b))
                        error = max(float(np.max(np.abs(np.asarray(s.data)-ref[s.index]))) for s in result.addressable_shards)
                        assert error < 2e-12, (layout,axis,error)
                        rows.append(dict(reverse=reverse,dtype=np.dtype(dtype).name,n=n,panel_limit=limit,layout=layout,axis=axis,error=error))
    # Static two-sided operator projection with distinct left/right endpoints.
    from common.contract_bands import _face_project_kernel
    nq, nb, ns, ml, mr = 2, 3*p, 2, 5*p, 7*p
    def complex_array(shape):
        return rng.normal(size=shape) + 1j*rng.normal(size=shape)
    left = complex_array((nq,nb,ns,ml))
    right = complex_array((nq,ns,mr,nb))
    operator = complex_array((nq,ns,ml,ns,mr))
    ref = np.einsum('kasu,ksutv,ktvb->kab',left.conj(),operator,right)
    project = _face_project_kernel(mesh0,(nq,nb,ml,ns),('x','y'),right_face_shape=(nq,nb,mr,ns))
    result = jax.jit(project)(put(left,NamedSharding(mesh0,P(None,'x',None,'y'))),
        put(operator,NamedSharding(mesh0,P(None,None,'x',None,'y'))),
        put(right,NamedSharding(mesh0,P(None,None,'x','y'))))
    error = max(float(np.max(np.abs(np.asarray(shard.data)-ref[shard.index]))) for shard in result.addressable_shards)
    assert error < 2e-10, error
    rows.append(dict(case='rectangular_two_sided',error=error))
    dest = Path(os.environ['LORRAX_TEST_OUTPUT'])
    dest.mkdir(parents=True,exist_ok=True)
    (dest/f'rank{jax.process_index()}.json').write_text(json.dumps(dict(job_step=f"{os.getenv('SLURM_JOB_ID')}.{os.getenv('SLURM_STEP_ID')}",ranks=jax.process_count(),rows=rows),indent=2))
    print(f'rank {jax.process_index()}: {len(rows)} panel projection cases passed',flush=True)

if __name__ == '__main__':
    if '--cpu-emulation' in sys.argv:
        main()
    else:
        from runtime import run_main_and_finalize
        run_main_and_finalize(main)

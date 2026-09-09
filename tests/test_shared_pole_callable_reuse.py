"""Changed physical supports and arrays must remain live through cached packing."""
from pathlib import Path
import json
import os
import numpy as np


def check_reuse(mesh):
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from gw.shared_pole_local import pack_parent_panels, _parent_panel_packer
    from gw.shared_pole_constructor import _hermitian_part_kernel, _public_factor_kernel, _stack_model_kernel
    face = NamedSharding(mesh, P(None, 'x', 'y'))
    def put(value):
        return jax.make_array_from_callback(value.shape, face, lambda i: value[i])
    def parents(scale, support):
        result=[]
        for width in (4, 8, 4):
            panel=put(np.full((1, 8, width), scale, np.complex128))
            inf=put(np.full((1, 8, 4), scale*2, np.complex128))
            result.append(([(support,panel,panel*3,panel*5)],(inf,inf*7,inf*11),jnp.ones((1,width+4),bool)))
        return result
    kernel=_parent_panel_packer(mesh)
    assert kernel is _parent_panel_packer(mesh)
    first=pack_parent_panels(parents(1.,2+1j),mesh_xy=mesh)
    jax.block_until_ready(first)
    before=kernel._cache_size()
    second=pack_parent_panels(parents(2.,3+2j),mesh_xy=mesh)
    jax.block_until_ready(second)
    assert kernel._cache_size()==before
    # Supports are runtime values; panel scaling must propagate through all faces.
    assert float(jnp.max(jnp.abs(second[0][0]-first[0][0])))>1
    assert float(jnp.max(jnp.abs(second[0][1]-2*first[0][1])))==0
    assert float(jnp.max(jnp.abs(second[1][0]-2*first[1][0])))==0
    for factory in (_hermitian_part_kernel,_public_factor_kernel,_stack_model_kernel):
        assert factory(mesh) is factory(mesh)
    # Parent identity and spectral counts must remain live without compiling
    # another executable when only those small metadata values change.
    from gw.shared_pole_constructor import _parent_panel_slice
    from distrib_la.polar import _retained_column_kernel
    panels=put(np.stack([np.full((8,8),i+1,np.complex128) for i in range(3)]))
    take=_parent_panel_slice(mesh,4)
    first_parent=take(panels,np.int32(0))
    jax.block_until_ready(first_parent)
    parent_specializations=take._cache_size()
    next_parent=take(panels,np.int32(2))
    jax.block_until_ready(next_parent)
    assert take._cache_size()==parent_specializations
    assert float(jnp.max(jnp.abs(next_parent-3*first_parent)))==0
    select=_retained_column_kernel(mesh,True,4)
    counts=jax.device_put(np.array([2,3,4],np.int64),NamedSharding(mesh,P()))
    first_selection=select(panels,counts)
    jax.block_until_ready(first_selection)
    selection_specializations=select._cache_size()
    changed_counts=jax.device_put(np.array([4,2,3],np.int64),NamedSharding(mesh,P()))
    changed_selection=select(panels,changed_counts)
    jax.block_until_ready(changed_selection)
    assert select._cache_size()==selection_specializations
    assert float(jnp.max(jnp.abs(changed_selection-first_selection)))>0
    for i,count in enumerate((4,2,3)):
        assert float(jnp.max(jnp.abs(changed_selection[i,:,:count]-(i+1))))==0
        if count<4:
            assert float(jnp.max(jnp.abs(changed_selection[i,:,count:])))==0
    return dict(status='PASS',scope='P4 ragged panel packing, current support/array/parent/count reds, identical signature executable reuse',pack_specializations=before,
                parent_specializations=parent_specializations,selection_specializations=selection_specializations)


if __name__=='__main__':
    from runtime import initialize_communicator_stack,run_main_and_finalize
    initialize_communicator_stack(platform='gpu')
    def main():
        import sys,jax
        from common.collectives import resolve_mesh,barrier
        assert jax.process_count()==4
        row=check_reuse(resolve_mesh());row['job_step']=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']
        if jax.process_index()==0:Path(sys.argv[1]).write_text(json.dumps(row,indent=2)+'\n')
        barrier('callable-reuse-gate')
    run_main_and_finalize(main)

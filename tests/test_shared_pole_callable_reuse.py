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
    counts = np.array([[3, 7], [7, 3], [3, 3]], np.int64)
    infinity_counts = np.array([3, 4, 3], np.int64)
    def panels(scale, support, current_counts=counts):
        states = []
        for port in range(2):
            values = scale * (np.arange(3*8*8).reshape(3,8,8) + 1 + 1000*port).astype(np.complex128)
            values *= np.arange(8)[None,None,:] < current_counts[:,port,None,None]
            panel = put(values)
            states.append((support+port, panel, panel*3, panel*5))
        inf = put(np.broadcast_to(np.arange(4)[None,None,:] < infinity_counts[:,None,None],
                                  (3,8,4)).astype(np.complex128)*scale*2)
        return states, (inf, inf*7, inf*11)
    kernel = _parent_panel_packer(mesh, 12, 4, (0, 1), 12)
    assert kernel is _parent_panel_packer(mesh, 12, 4, (0, 1), 12)
    states, infinity = panels(1., 2+1j)
    first, extents = pack_parent_panels(states, infinity, counts, infinity_counts,
                                      mesh_xy=mesh, parent_batch=4, layout="local")
    jax.block_until_ready(first)
    before = kernel._cache_size()
    states2, infinity2 = panels(2., 3+2j)
    second, _ = pack_parent_panels(states2, infinity2, counts, infinity_counts,
                                 mesh_xy=mesh, parent_batch=4, layout="local")
    jax.block_until_ready(second)
    assert kernel._cache_size() == before
    assert extents == ((12,4), (12,4), (8,4))
    # Compare against explicit original per-parent port carriers, including
    # inactive inter-port columns; a sort of only live columns would fail.
    for q, widths in enumerate(((4,8), (8,4), (4,4))):
        expected = jnp.concatenate([state[1][q,:,:width] for state,width in zip(states,widths)], axis=-1)
        assert float(jnp.max(jnp.abs(first[0][1][q,:,:sum(widths)]-expected))) == 0
        if sum(widths) < 12:
            assert float(jnp.max(jnp.abs(first[0][1][q,:,sum(widths):]))) == 0
    assert float(jnp.max(jnp.abs(first[0][1][3]-first[0][1][2]))) == 0
    assert float(jnp.max(jnp.abs(second[0][0]-first[0][0]))) > 1
    assert float(jnp.max(jnp.abs(second[0][1]-2*first[0][1]))) == 0
    assert float(jnp.max(jnp.abs(second[1][0]-2*first[1][0]))) == 0
    changed_counts = counts[:, ::-1].copy()
    changed_states, changed_inf = panels(1., 2+1j, changed_counts)
    changed, _ = pack_parent_panels(changed_states, changed_inf, changed_counts, infinity_counts,
                                   mesh_xy=mesh, parent_batch=4, layout="local")
    jax.block_until_ready(changed)
    assert kernel._cache_size() == before
    assert float(jnp.max(jnp.abs(changed[0][1]-first[0][1]))) > 1
    # A conjugate port shares Q exactly, but its actions and supports differ.
    # Compare the original distributed-layout pack with compact local input
    # expanded after the actual staged movement (including ragged tails).
    from common.staged_reshard import face_to_batch_reshard
    from common.shard_map import shard_map
    paired = [*states, (2-1j, states[0][1], states[0][2]*7, states[0][3]*11)]
    paired_counts = np.column_stack((counts, counts[:, 0]))
    compact, _ = pack_parent_panels(paired, infinity, paired_counts, infinity_counts,
                                    mesh_xy=mesh, parent_batch=4, layout="local")
    full, _ = pack_parent_panels(paired, infinity, paired_counts, infinity_counts,
                                 mesh_xy=mesh, parent_batch=4, layout="distributed")
    assert compact[0][1].shape[-1] == 12 and full[0][1].shape[-1] == 20
    to_batch = face_to_batch_reshard(mesh)
    qspec = P(('x', 'y'))
    def expand(q, columns):
        return jnp.where(columns[:, None, :] >= 0,
                         jnp.take_along_axis(q, jnp.maximum(columns, 0)[:, None, :], axis=-1), 0)
    expand = jax.jit(shard_map(expand, mesh=mesh, in_specs=(qspec, qspec),
                              out_specs=qspec, check_vma=False))
    expanded = expand(to_batch(compact[0][1]), jax.device_put(compact[3], NamedSharding(mesh, qspec)))
    original = to_batch(full[0][1])
    for left, right in zip(expanded.addressable_shards, original.addressable_shards):
        assert np.asarray(left.data).tobytes() == np.asarray(right.data).tobytes()
    for left, right in zip(jax.tree.leaves((compact[0][0], compact[0][2:], compact[1:3])),
                           jax.tree.leaves((full[0][0], full[0][2:], full[1:3]))):
        for a, b in zip(left.addressable_shards, right.addressable_shards):
            assert np.asarray(a.data).tobytes() == np.asarray(b.data).tobytes()
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

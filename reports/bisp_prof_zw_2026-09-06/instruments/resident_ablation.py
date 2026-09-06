"""Isolate family-face residency, including its separate compilation cost."""
from gw import gw_jax, w_isdf
import inspect
import json
import time
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from common.shard_map import shard_map
from common import jax_compile_cache

faces = {}
original_response = w_isdf.compute_experimental_no_pair_photon_chi0
source = inspect.getsource(w_isdf.compute_no_pair_dirac_current_block)
a = source.index('    from .wavefunction_bundle import green_face_kernel_kwargs')
b = source.index('    from common.gamma_matrices import gamma_perm_phase', a)
source = source[:a]+'''    left_face, right_face = resident_faces[id(wfns_left)][0], resident_faces[id(wfns_right)][1]
    from .wavefunction_bundle import green_face_kernel_kwargs
    left_shape = (left_face.shape[0], *green_face_kernel_kwargs(wfns_left)["face_shape"][1:])
    right_shape = (right_face.shape[0], *green_face_kernel_kwargs(wfns_right)["face_shape"][1:])
    kernel = _get_chi_minimax_kernel(
        mesh_xy, kgrid, layout="face", face_shape=left_shape,
        right_face_shape=right_shape, vertex_pair=(A, B), k_unfold_plan=None)
    rows = jnp.asarray(left.plan.irr_idx)
    mask_v = jnp.take(left.plan.parent_rows(wfns_left.band_mask(s.val)), rows, axis=0)
    mask_c = jnp.take(left.plan.parent_rows(wfns_left.band_mask(s.cond)), rows, axis=0)
    energies = jnp.take(left.enk, rows, axis=0)
    args = (nodes, left_face, right_face, mask_v, mask_c,
            energies - jnp.asarray(eref, dtype=energies.dtype),
            jnp.asarray(vmax, dtype=jnp.float64), jnp.asarray(cmin, dtype=jnp.float64))
'''+source[b:]
namespace=dict(w_isdf.__dict__);namespace['resident_faces']=faces
exec(compile(source,'resident_current_block.py','exec'),namespace)
w_isdf.compute_no_pair_dirac_current_block=namespace['compute_no_pair_dirac_current_block']


def response(charge, current, quad, meta, mesh, layout, **kwargs):
    """Unfold each family's endpoint pair once, then release after the response."""
    for bundle in (charge,current):
        parent=bundle.green_parent;plan=parent.plan
        specs=(P(None,None,'x','y'),P(None,'x',None,'y'))
        def child(left,right):
            return (plan.unfold_face(left,spin_axis=1,mu_axis=2,mesh_axis='x'),
                    plan.unfold_face(right,spin_axis=2,mu_axis=3,mesh_axis='y'))
        fn=jax.jit(shard_map(child,mesh=mesh,in_specs=specs,out_specs=specs,check_vma=False))
        state=jax_compile_cache._STATE;before=state.compiles,state.compile_secs
        start=time.perf_counter()
        value=jax.block_until_ready(fn(parent.psi_mun,parent.psi_nmu))
        faces[id(bundle)]=value
        row=dict(label='resident_family_unfold',seconds=time.perf_counter()-start,
                 compiles=state.compiles-before[0],compile_seconds=state.compile_secs-before[1],
                 shapes=[list(v.shape) for v in value],
                 local_bytes=sum(sum(shard.data.nbytes for shard in v.addressable_shards) for v in value))
        print('[resident-ablation] '+json.dumps(row),flush=True)
    try:
        return original_response(charge,current,quad,meta,mesh,layout,**kwargs)
    finally:
        faces.clear()


w_isdf.compute_experimental_no_pair_photon_chi0=response

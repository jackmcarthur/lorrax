"""Execution adapters for shared-pole equations on the complete x/y mesh.

The local adapter lives in shared_pole_local. This module supplies no new
physics: matrix operations enter through distrib_la, matrix results remain
face tiled, and only spectra, masks and small scalar receipts replicate.
"""
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P


def is_face(array):
    spec = tuple(array.sharding.spec)
    return len(spec) == array.ndim and spec[-2:] == ('x', 'y')


def face_program(fn, mesh, *, outputs='matrices'):
    """Compile glue with an explicit matrix/scalar output contract.

    Matrix-only glue preserves every output's trailing two mesh axes,
    including real matrices. Mixed reducer contracts name their matrix
    leaves explicitly; diagnostics and spectra alone replicate.
    """
    compiled = {}
    def call(*args):
        signature = jax.tree.structure(args), tuple((a.shape,a.dtype) for a in jax.tree.leaves(args))
        if signature not in compiled:
            shapes = jax.eval_shape(fn, *args)
            rep=NamedSharding(mesh,P())
            def matrix(v):
                if v.ndim < 3:
                    raise ValueError('constructor matrix output must have explicit parent and two face axes')
                return NamedSharding(mesh,P(*([None]*(v.ndim-2)),'x','y'))
            scalar_tree=lambda tree:jax.tree.map(lambda _:rep,tree)
            model=lambda tree:(matrix(tree[0]),rep,rep)
            if outputs == 'matrices':
                out=jax.tree.map(matrix,shapes)
            elif outputs == 'scalars':
                out=scalar_tree(shapes)
            elif outputs == 'parent':
                out=(model(shapes[0]),model(shapes[1]) if shapes[1] else (),scalar_tree(shapes[2]))
                if len(shapes)==4:out=(*out,matrix(shapes[3]))
            elif outputs == 'cross':
                out=((matrix(shapes[0][0]),matrix(shapes[0][1]),rep,rep),scalar_tree(shapes[1]))
            elif outputs == 'positive_cross':
                out=(tuple(model(m) for m in shapes[0]),scalar_tree(shapes[1]))
            elif outputs == 'compact':
                out=(matrix(shapes[0]),model(shapes[1]))
            else:
                raise ValueError('unknown explicit constructor output contract '+outputs)
            compiled[signature]=jax.jit(fn,out_shardings=out)
        return compiled[signature](*args)
    return call


@lru_cache(maxsize=None)
def face_matmul(mesh):
    from distrib_la import matmul
    return partial(matmul, mesh=mesh, backend='distributed', batched_route='auto')


@lru_cache(maxsize=None)
def face_eigh(mesh, n):
    from distrib_la import plan
    # Explicit auto selects the whole-mesh provider; no capacity-driven local
    # reshard or submesh is allowed inside an oversized parent operation.
    return plan('eigh',mesh,n=int(n),backend='distributed',batched_route='auto')


@lru_cache(maxsize=None)
def face_parent_program(mesh,ordered,odd_moments,keep_budget,retain_span):
    """Retained static-layout executable builder; all state values are operands."""
    from gw.shared_pole_local import solve_parent_pencil
    from gw.shared_pole_gates import sort_shared_pole_columns
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1, shared_real_pole_gates_v1_r3b
    gates=shared_real_pole_gates_ordered_v1 if ordered else shared_real_pole_gates_v1_r3b
    mm=face_matmul(mesh)
    def eigh(a):
        return face_eigh(mesh,a.shape[-1]).batched(a)
    def body(points,order,active,qs,os,ds,infinity):
        def pack(parts):
            return jax.lax.with_sharding_constraint(
                jnp.take(jnp.concatenate(parts,axis=-1),order[0],axis=-1,mode='fill',fill_value=0),
                NamedSharding(mesh,P(None,'x','y')))
        reduced=solve_parent_pencil(points,pack(qs),pack(os),pack(ds),infinity,active,
            eigh=eigh,matmul=mm,gates=gates,ordered=ordered,odd_moments=odd_moments,
            keep_budget=keep_budget,retain_span=retain_span,
            matrix_sharding=NamedSharding(mesh,P(None,"x","y")))
        model,signed,diagnostics=reduced[:3]
        model,permutation=sort_shared_pole_columns(model)
        result=model,signed,(*diagnostics,permutation)
        return (*result,reduced[3]) if retain_span else result
    return face_program(body,mesh,outputs='parent')


def face_reduce_round(states,infinity,tables,*,real,mesh,budget,ordered,odd_moments,
                      keep_budget,retain_span=False):
    """One physical parent, all ranks, with no artificial round zero tails."""
    if real != 1 or len(tables['own']) != 1:
        raise ValueError('distributed constructor requires one physical parent per full-mesh program')
    side=tables['active'].shape[-1]
    budget.plan(side,phase='reduction')
    program=face_parent_program(mesh,ordered,odd_moments,keep_budget,retain_span)
    result=program(jnp.asarray(tables['points']),jnp.asarray(tables['order']),
        jnp.asarray(tables['active']),tuple(s[1] for s in states),
        tuple(s[2] for s in states),tuple(s[3] for s in states),tuple(infinity))
    model,signed,diagnostics=result[:3]
    output=model,signed,model[1:],diagnostics
    return (*output,result[3]) if retain_span else output


def sector_round_schedule(bank,header,meta,config,mesh,partner):
    """Use the existing explicit linalg mode, resolved once before bank reads."""
    from gw.shared_pole_local import parent_rounds
    from gw.gw_config import linalg_resolution
    resolution=linalg_resolution({'linalg':config.backend.linalg})
    if resolution.layout == 'local':
        return [(*row,'local') for row in parent_rounds(header['n_q_irr'],mesh.size,partner)]
    if resolution.layout != 'distributed':
        raise ValueError('unsupported resolved constructor linalg layout')
    # Literal mirrors already contain the same operator; face parents need
    # neither simultaneous partner parents nor artificial rank padding.
    return [([q],1,np.asarray([0],np.int64),'face') for q in range(header['n_q_irr'])]



@lru_cache(maxsize=None)
def cross_parent_program(mesh):
    from gw.shared_pole_sectors import _cross_reduce_equations
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1 as gates
    def eigh(a):
        return face_eigh(mesh,a.shape[-1]).batched(a)
    return face_program(partial(_cross_reduce_equations,mm=face_matmul(mesh),eigh=eigh,gates=gates,
                                matrix_sharding=NamedSharding(mesh,P(None,"x","y"))),
                        mesh,outputs='cross')


@lru_cache(maxsize=None)
def cross_action_program(mesh,sample,mirror,conjugate):
    from gw.shared_pole_sectors import _literal_cross_products
    return face_program(partial(_literal_cross_products,sample=sample,mirror=mirror,
        conjugate=conjugate,mm=face_matmul(mesh)),mesh)


@lru_cache(maxsize=None)
def positive_cross_program(mesh):
    from gw.shared_pole_sectors import _positive_cross_equations
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1 as gates
    return face_program(partial(_positive_cross_equations,gates=gates),mesh,outputs='positive_cross')


@lru_cache(maxsize=None)
def held_program(mesh):
    from gw.shared_pole_sectors import _sector_held_equations
    return face_program(partial(_sector_held_equations,mm=face_matmul(mesh)),mesh,outputs='scalars')


@lru_cache(maxsize=None)
def compact_program(mesh,width):
    from gw.shared_pole_sectors import _compact_sector_equations
    return face_program(partial(_compact_sector_equations,width=width),mesh,outputs='compact')


@lru_cache(maxsize=None)
def cauchy_program(mesh):
    from gw.shared_pole_sectors import sector_cauchy_schwarz
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1 as gates
    def eigh(a):
        return face_eigh(mesh,a.shape[-1]).batched(a)
    def body(c,ct,t):
        return sector_cauchy_schwarz((c,ct,t),eigh_charge=eigh,eigh_current=eigh,
                                    matmul=face_matmul(mesh),gates=gates)
    return face_program(body,mesh,outputs='scalars')

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


def constructor_side_upper_bound(recipe, *, ordered, odd_moments,
                                 logical_n=None,
                                 column_extent=lambda width: width):
    """Conservative recipe-only pencil side before any bank array is read.

    Line supports can contribute conjugate states, and an ordered bank adds a
    mirrored half. Pure-imaginary ordered supports can also contribute the
    independently selected partner direction. Multiplet closure is carried by
    ``column_extent``. This is an admission bound, never a retained-rank rule.
    """
    from gw.shared_pole_recipe import ROLE_CODES

    held = np.asarray(recipe['held'], dtype=bool)
    roles = np.asarray(recipe['role'], dtype=np.int64)
    line = int(np.sum((roles == ROLE_CODES['line']) & ~held))
    imaginary = int(np.sum((roles == ROLE_CODES['imaginary']) & ~held))
    line_cap = recipe.get('line_direction_cap')
    if line_cap is None:
        if logical_n is None:
            raise ValueError('constructor side bound needs logical_n when the line cap is unset')
        line_cap = logical_n
    line_width = column_extent(max(1, int(line_cap)))
    imaginary_width = column_extent(max(1, int(recipe['imaginary_width'])))
    infinity_width = column_extent(max(1, int(recipe['infinity_width'])))
    if ordered:
        finite = 4 * (line * line_width + imaginary * imaginary_width)
        infinity = 2 * infinity_width if odd_moments else 0
    else:
        finite = 2 * line * line_width + imaginary * imaginary_width
        infinity = infinity_width
    return int(finite + infinity)


def constructor_execution(meta, resolution, recipe, *, mesh, ledger, upstream,
                          ordered, odd_moments, sample_fields, moment_fields,
                          parent_count=1, retained_output_families=1,
                          column_extent=lambda width: width):
    """Resolve local or whole-mesh execution once, before a constructor read.

    Explicit distributed service policy selects the face. Otherwise the local
    parent route remains the fast path only when both the complete selection
    stack and the conservative recipe pencil fit the current device budget.
    """
    from gw.shared_pole_capacity import ConstructorCapacity

    side = constructor_side_upper_bound(
        recipe, ordered=ordered, odd_moments=odd_moments,
        logical_n=int(meta.n_rmu),
        column_extent=column_extent)
    fit_ids = [int(i) for i in recipe['fit_ids']]
    fit = max(fit_ids) - min(fit_ids) + 1
    selection_faces = fit * int(sample_fields) + int(moment_fields)
    if resolution.layout == 'distributed':
        return 'face', dict(reason='configured distributed service',
                            conservative_pencil_side=side,
                            selection_face_count=selection_faces)
    if resolution.layout != 'local':
        raise ValueError('unsupported resolved constructor linalg layout')
    local = ConstructorCapacity(meta, resolution, mesh_xy=mesh, ledger=ledger,
                                upstream=upstream, execution='local')
    local.batch_width = int(mesh.size)
    pole_budget = recipe.get('pole_budget')
    if pole_budget is None:
        pole_budget = int(meta.n_rmu)
    output_width = column_extent(max(1, int(pole_budget)))
    retained_outputs = int(np.ceil(
        16 * int(parent_count) * int(retained_output_families)
        * int(meta.n_rmu_padded) * output_width / int(mesh.size)))
    def resident_preview(phase, **kwargs):
        price = local.resident_quote(side, phase=phase, **kwargs)
        row = ledger.preview(
            resident_bytes_per_rank=(price['resident_bytes_per_rank']
                                     + retained_outputs),
            workspace_bytes_per_rank=0, concurrent_with=upstream)
        row['retained_output_upper_bound_bytes_per_rank'] = retained_outputs
        row['native_workspace_query'] = 'NOT_NEEDED_FOR_RESIDENT_LOWER_BOUND'
        return row
    def preview(phase, **kwargs):
        price,native=local.quote(side,phase=phase,**kwargs)
        row=ledger.preview(
            resident_bytes_per_rank=price['resident_bytes_per_rank']+retained_outputs,
            workspace_bytes_per_rank=sum(native.values()),concurrent_with=upstream)
        row['retained_output_upper_bound_bytes_per_rank']=retained_outputs
        return row
    selection_args = dict(sample_batch=fit, selection_faces=selection_faces)
    resident_selection = resident_preview('selection', **selection_args)
    resident_reduction = resident_preview('reduction')
    if any(row['device_budget_status'] != 'PASS'
           for row in (resident_selection, resident_reduction)):
        return 'face', dict(
            reason='local resident lower bound exceeds current device budget',
            conservative_pencil_side=side,
            selection_face_count=selection_faces,
            retained_output_upper_bound_bytes_per_rank=retained_outputs,
            local_selection=resident_selection,
            local_reduction=resident_reduction)
    selection = preview('selection', **selection_args)
    reduction = preview('reduction')
    admitted = all(row['device_budget_status'] == 'PASS'
                   for row in (selection, reduction))
    return ('local' if admitted else 'face'), dict(
        reason=('capacity-admitted local parent' if admitted else
                'local parent exceeds current device budget'),
        conservative_pencil_side=side, selection_face_count=selection_faces,
        retained_output_upper_bound_bytes_per_rank=retained_outputs,
        local_selection=selection, local_reduction=reduction)


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
def face_parent_program(mesh,ordered,odd_moments,keep_budget,retain_span,side,gram_keep=None):
    """Retained static-layout executable builder; all state values are operands."""
    from gw.shared_pole_local import solve_parent_pencil
    from gw.shared_pole_gates import sort_shared_pole_columns
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1, shared_real_pole_gates_v1_r3b
    gates=shared_real_pole_gates_ordered_v1 if ordered else shared_real_pole_gates_v1_r3b
    mm=face_matmul(mesh)
    eigh_plan=face_eigh(mesh,side)
    def body(points,order,active,qs,os,ds,infinity):
        def pack(parts):
            panels = jnp.concatenate((*parts, jnp.zeros_like(parts[0][..., :int(mesh.shape["y"])])), axis=-1)
            from gw.shared_pole_pencil import _matrix_take_columns
            return _matrix_take_columns(panels, order, NamedSharding(mesh,P(None,"x","y")))
        reduced=solve_parent_pencil(points,pack(qs),pack(os),pack(ds),infinity,active,
            eigh=eigh_plan.batched,matmul=mm,gates=gates,ordered=ordered,odd_moments=odd_moments,
            keep_budget=keep_budget,retain_span=retain_span,gram_keep=gram_keep,
            matrix_sharding=NamedSharding(mesh,P(None,"x","y")))
        model,signed,diagnostics=reduced[:3]
        model,permutation=sort_shared_pole_columns(model, matrix_sharding=NamedSharding(mesh,P(None,"x","y")))
        result=model,signed,(*diagnostics,permutation)
        return (*result,reduced[3]) if retain_span else result
    return face_program(body,mesh,outputs='parent')


def face_reduce_round(states,infinity,tables,*,real,mesh,budget,ordered,odd_moments,
                      keep_budget,retain_span=False,admit=True,gram_keep=None):
    """A batch of physical parents with every matrix tiled over all ranks."""
    if real != len(tables['own']):
        raise ValueError('distributed constructor batches contain physical parents only')
    side=tables['active'].shape[-1]
    if admit:
        budget.plan(side,phase='reduction')
    program=face_parent_program(mesh,ordered,odd_moments,keep_budget,retain_span,side,gram_keep)
    result=program(jnp.asarray(tables['points']),jnp.asarray(tables['order']),
        jnp.asarray(tables['active']),tuple(s[1] for s in states),
        tuple(s[2] for s in states),tuple(s[3] for s in states),tuple(infinity))
    model,signed,diagnostics=result[:3]
    output=model,signed,model[1:],diagnostics
    return (*output,result[3]) if retain_span else output


@lru_cache(maxsize=None)
def face_round_check_program(mesh, ordered, n):
    """Whole-mesh adapter for the scalar model's existing gate equations."""
    from gw.shared_pole_local import _round_check_equations
    from gw.shared_pole_recipe import (shared_real_pole_gates_ordered_v1,
                                       shared_real_pole_gates_v1_r3b)
    gates = (shared_real_pole_gates_ordered_v1 if ordered else
             shared_real_pole_gates_v1_r3b)
    eigh_plan = face_eigh(mesh, n)
    return face_program(partial(_round_check_equations, matmul=face_matmul(mesh),
                                eigh=eigh_plan.batched, gates=gates, ordered=ordered),
                        mesh, outputs='scalars')


def sector_round_schedule(bank,header,meta,config,mesh,partner,*,execution=None,batch_width=1):
    """Schedule local parent rounds or bounded batches on the whole mesh."""
    from gw.shared_pole_local import parent_rounds
    from gw.gw_config import linalg_resolution
    resolution=linalg_resolution({'linalg':config.backend.linalg})
    execution = resolution.layout if execution is None else execution
    if execution == 'local':
        return [(*row,'local') for row in parent_rounds(header['n_q_irr'],mesh.size,partner)]
    if execution not in ('distributed', 'face'):
        raise ValueError('unsupported resolved constructor linalg layout')
    # Literal mirrors already contain the same operator; face parents need
    # neither simultaneous partner parents nor artificial rank padding.
    nq = int(header['n_q_irr'])
    return [(list(range(q, min(q + batch_width, nq))), min(batch_width, nq-q),
             np.arange(min(batch_width, nq-q), dtype=np.int64), 'face')
            for q in range(0, nq, batch_width)]


def sector_batch_width(meta, resolution, recipe, routes, *, mesh, ledger, nq):
    """Bound the common CC/TT/CT batch before reading any sample matrix.

    The joint extent covers both retained diagonal spans and the rectangular
    cross pencil. Its dense envelope also covers the surviving diagonal
    arrays; both ordered cross sample stacks are included separately.
    Execution stays face tiled throughout; only the number of parents varies.
    """
    import copy
    from gw.shared_pole_capacity import ConstructorCapacity

    joint = copy.copy(meta)
    joint.n_rmu_padded = sum(row['packed_extent'] for row in routes)
    side = sum(row['conservative_pencil_side'] for row in routes)
    # CT diagonalizes the retained joint span, never the unreduced
    # rectangular C/T pencil. Diagonal sectors still solve their own side.
    eigen_side = max(max(row['conservative_pencil_side'] for row in routes),
                     sum(min(row['signed_side_bound'],row['conservative_pencil_side'])
                         for row in routes))
    fit_ids = recipe['fit_ids']
    fit = int(max(fit_ids) - min(fit_ids) + 1)
    budget = ConstructorCapacity(joint, resolution, mesh_xy=mesh, ledger=ledger,
                                 upstream=ledger.live_stages, execution='face')
    for width in range(int(nq), 0, -1):
        budget.batch_width = width
        # The phase formula covers current pencil/actions, not the two full
        # sample stacks that the caller still holds during cross reduction.
        sample_bytes = int(np.ceil(16 * width * (8 * fit + 8)
                                  * joint.n_rmu_padded**2 / mesh.size))
        resident = budget.resident_quote(side, phase='reduction')
        preview = ledger.preview(
            resident_bytes_per_rank=resident['resident_bytes_per_rank'] + sample_bytes,
            workspace_bytes_per_rank=0, concurrent_with=ledger.live_stages)
        if preview['device_budget_status'] != 'PASS':
            continue
        price, native = budget.quote(side, phase='reduction',eigen_side=eigen_side)
        preview = ledger.preview(
            resident_bytes_per_rank=price['resident_bytes_per_rank'] + sample_bytes,
            workspace_bytes_per_rank=sum(native.values()),
            concurrent_with=ledger.live_stages)
        if preview['device_budget_status'] == 'PASS':
            return width, preview
    raise MemoryError('GATE shared_pole_capacity: distributed sector batch of one '
                      f'parent exceeds the shared device budget before bank read; last price: {preview}')



@lru_cache(maxsize=None)
def cross_parent_program(mesh, side):
    from gw.shared_pole_sectors import _cross_reduce_equations
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1 as gates
    eigh_plan = face_eigh(mesh, side)
    return face_program(partial(_cross_reduce_equations,mm=face_matmul(mesh),eigh=eigh_plan.batched,gates=gates,
                                matrix_sharding=NamedSharding(mesh,P(None,"x","y"))),
                        mesh,outputs='cross')


@lru_cache(maxsize=None)
def cross_action_program(mesh,mirror,conjugate):
    from gw.shared_pole_sectors import _literal_cross_products
    return face_program(partial(_literal_cross_products,mirror=mirror,
        conjugate=conjugate,mm=face_matmul(mesh)),mesh)


@lru_cache(maxsize=None)
def positive_cross_program(mesh):
    from gw.shared_pole_sectors import _positive_cross_equations
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1 as gates
    return face_program(partial(_positive_cross_equations,gates=gates,
        matrix_sharding=NamedSharding(mesh,P(None,"x","y"))),mesh,outputs='positive_cross')


@lru_cache(maxsize=None)
def held_program(mesh):
    from gw.shared_pole_sectors import _sector_held_equations
    return face_program(partial(_sector_held_equations,mm=face_matmul(mesh)),mesh,outputs='scalars')


@lru_cache(maxsize=None)
def compact_program(mesh,width):
    from gw.shared_pole_sectors import _compact_sector_equations
    return face_program(partial(_compact_sector_equations,width=width,
        matrix_sharding=NamedSharding(mesh,P(None,"x","y"))),mesh,outputs='compact')


@lru_cache(maxsize=None)
def cauchy_program(mesh, charge_n, current_n):
    from gw.shared_pole_sectors import sector_cauchy_schwarz
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1 as gates
    charge_eigh = face_eigh(mesh, charge_n)
    current_eigh = face_eigh(mesh, current_n)
    def body(c,ct,t):
        return sector_cauchy_schwarz((c,ct,t),eigh_charge=charge_eigh.batched,
                                    eigh_current=current_eigh.batched,
                                    matmul=face_matmul(mesh),gates=gates)
    return face_program(body,mesh,outputs='scalars')

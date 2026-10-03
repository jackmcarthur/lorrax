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


def line_panel_count(recipe):
    """Fitted line supports off the imaginary axis: the samples the bank stores as panels."""
    from gw.shared_pole_directions import _sample_point
    return sum(1 for sid in recipe['fit_ids'] if _sample_point(recipe, int(sid)).real != 0)


def selection_face_count(recipe, *, n, logical_n, states, rows, dense_fields, moment_fields,
                         column_extent, cross_rows=0):
    """Resident [n,n] face equivalents of one parent's selection inputs.

    ``dense_fields`` faces per dense fitted sample (on the imaginary axis),
    ``moment_fields`` moment faces, and per stored line sample the panel
    [1+2S, rows, r] plus, when ``cross_rows``, the [2S, cross_rows, r] cross
    panel, at r = the carrier of the line cap (``logical_n`` rows when the recipe
    sets none), rounded up to whole faces. An admission bound: a multiplet
    closed past the cap stores its actual width.
    """
    import math
    lines = line_panel_count(recipe)
    dense = len(recipe['fit_ids']) - lines
    cap = recipe.get('line_direction_cap')
    width = int(column_extent(max(1, min(int(logical_n), int(logical_n) if cap is None else int(cap)))))
    panel = lines * ((1 + 2 * int(states)) * int(rows) + 2 * int(states) * int(cross_rows)) * width
    return int(dense_fields) * dense + int(moment_fields) + math.ceil(panel / int(n) ** 2)


def line_selection_price(rows, *, mesh, nq, execution):
    """Per-rank bytes of the producer's selection at one line sample: (resident, workspace).

    Live beside the caller's reservations: the selection copies of every
    endpoint block of W and of dW/ds (16 * 2 * sum_fg n_f n_g per parent;
    whole parents per rank, ceil(nq/P) of them, on the local route; tiles on
    the face) and the largest family's n x n normal matrix W^H W with its
    eigenvectors for every parent of the stack (the local route solves its
    ceil(nq/P) parents in one batched eigh), plus the service's eigh
    workspace. The panels are narrow and ride in the same bound.
    """
    import math
    import distrib_la
    from gw.shared_pole_capacity import constructor_eigenplan
    ranks = int(mesh.size)
    blocks = sum(int(a) * int(b) for a in rows for b in rows)
    largest = max(int(r) for r in rows)
    plan = constructor_eigenplan(mesh, largest, execution)
    if execution == 'local':
        resident = 16 * 2 * math.ceil(int(nq) / ranks) * (blocks + largest ** 2)
        workspace = distrib_la.workspace_bytes_per_rank(plan, "eigh", ((int(nq), largest, largest),), np.complex128)
    else:
        # The whole-mesh kernel forms G = W^H W and its vectors for every
        # parent of the stack at once: 2 nq largest^2 tiles.
        resident = math.ceil(16 * (2 * int(nq) * blocks + 2 * int(nq) * largest ** 2) / ranks)
        workspace = distrib_la.workspace_bytes_per_rank(plan, "eigh", ((1, largest, largest),), np.complex128)
    return int(resident), int(workspace)


def whole_parent_execution(price, *, ledger, carry=0):
    """'local' when whole parents per rank fit the ledger beside its live stages, else 'face'.

    ``price(execution)`` returns the route's ``(resident, workspace)`` bytes per
    rank; ``carry`` is a resident the work runs beside. The face is the full
    mesh, one parent after another, and is never refused. Returns
    ``(execution, resident, workspace)``, the prices of the chosen route.
    """
    for execution in ('local', 'face'):
        resident, workspace = price(execution)
        row = ledger.preview(resident_bytes_per_rank=resident + int(carry), workspace_bytes_per_rank=workspace,
                             concurrent_with=ledger.live_stages)
        if row['device_budget_status'] == 'PASS':
            return execution, resident, workspace
    return 'face', resident, workspace


def line_selection_execution(rows, *, mesh, ledger, nq, carry=0):
    """'local' when the producer's line selection fits with whole parents per rank, else 'face'."""
    return whole_parent_execution(
        lambda execution: line_selection_price(rows, mesh=mesh, nq=nq, execution=execution),
        ledger=ledger, carry=carry)


def constructor_execution(meta, resolution, recipe, *, mesh, ledger, upstream,
                          ordered, odd_moments, selection_faces, sample_batch,
                          parent_count=1, retained_output_families=1,
                          cross_original_sides=None, cross_retained_side=None,
                          column_extent=lambda width: width, ritz_budget=None,
                          retain_span=False, carry=0):
    """Resolve local or whole-mesh execution once, before a constructor read.

    ``selection_faces`` counts the selection's resident [n,n] faces
    (``selection_face_count``) over ``sample_batch`` dense fitted samples.

    Decided once, up front, from the conservative recipe pencil
    (``constructor_side_upper_bound``; decisions.md#no-sub-mesh): local
    parents (one whole parent per rank, R4) when the selection stack and the
    conservative reduction both fit, whatever dense layout the deck names
    (``linalg`` prices the per-matrix service calls, it does not force every
    parent through a whole-mesh eigensolve one at a time); otherwise the face,
    the full mesh, where ``face_batch_width`` sizes the batch. A measured
    pencil is never consulted: a local round could not leave the route once
    its selection is held (CrI3 24x24 at P64, 40 GB: local pencil 64 GB/rank).
    ``ritz_budget`` is the pole budget of a local paired (ordered)
    reduction, priced as that program is (``ConstructorCapacity.ritz_budget``);
    ``retain_span`` adds a sector round's coefficient map. ``carry`` is a
    resident per rank the local round runs beside (an earlier sector's held
    outputs, ``held_sector_bytes``).
    """
    from gw.shared_pole_capacity import ConstructorCapacity

    if (cross_original_sides is None) != (cross_retained_side is None):
        raise ValueError('cross route requires both original sides and retained side')
    side = (sum(map(int, cross_original_sides)) if cross_original_sides is not None
            else constructor_side_upper_bound(
                recipe, ordered=ordered, odd_moments=odd_moments,
                logical_n=int(meta.n_rmu), column_extent=column_extent))
    fit = max(1, int(sample_batch))
    selection_faces = int(selection_faces)
    if resolution.layout not in ('local', 'distributed'):
        raise ValueError('unsupported resolved constructor linalg layout')
    local = ConstructorCapacity(meta, resolution, mesh_xy=mesh, ledger=ledger,
                                upstream=upstream, execution='local')
    local.batch_width = int(mesh.size)
    local.ritz_budget = ritz_budget
    local.retain_span = bool(retain_span)
    pole_budget = recipe.get('pole_budget')
    if pole_budget is None:
        pole_budget = int(meta.n_rmu)
    output_width = column_extent(max(1, int(pole_budget)))
    retained_outputs = int(np.ceil(
        16 * int(parent_count) * int(retained_output_families)
        * int(meta.n_rmu_padded) * output_width / int(mesh.size)))
    carry = int(carry)
    def resident_preview(phase, **kwargs):
        price = local.resident_quote(
            int(cross_retained_side) if phase == 'cross_reduction' else side,
            phase=phase, **kwargs)
        row = ledger.preview(
            resident_bytes_per_rank=(price['resident_bytes_per_rank']
                                     + retained_outputs + carry),
            workspace_bytes_per_rank=0, concurrent_with=upstream)
        row['retained_output_upper_bound_bytes_per_rank'] = retained_outputs
        row['carry_bytes_per_rank'] = carry
        row['native_workspace_query'] = 'NOT_NEEDED_FOR_RESIDENT_LOWER_BOUND'
        return row
    def preview(phase, **kwargs):
        price,native=local.quote(
            int(cross_retained_side) if phase == 'cross_reduction' else side,
            phase=phase,**kwargs)
        row=ledger.preview(
            resident_bytes_per_rank=price['resident_bytes_per_rank']+retained_outputs+carry,
            workspace_bytes_per_rank=sum(native.values()),concurrent_with=upstream)
        row['retained_output_upper_bound_bytes_per_rank']=retained_outputs
        row['carry_bytes_per_rank']=carry
        return row
    selection_args = dict(sample_batch=fit, selection_faces=selection_faces)
    reduction_phase = 'cross_reduction' if cross_original_sides is not None else 'reduction'
    reduction_args = (dict(cross_original_sides=cross_original_sides)
                      if cross_original_sides is not None else {})
    resident_selection = resident_preview('selection', **selection_args)
    resident_reduction = resident_preview(reduction_phase, **reduction_args)
    if any(row['device_budget_status'] != 'PASS'
           for row in (resident_selection, resident_reduction) if row is not None):
        return 'face', dict(
            reason='local resident lower bound exceeds current device budget',
            requested_layout=resolution.layout,
            conservative_pencil_side=side,
            selection_face_count=selection_faces,
            retained_output_upper_bound_bytes_per_rank=retained_outputs,
            local_selection=resident_selection, local_reduction=resident_reduction)
    selection = preview('selection', **selection_args)
    reduction = preview(reduction_phase, **reduction_args)
    admitted = all(row['device_budget_status'] == 'PASS' for row in (selection, reduction))
    return ('local' if admitted else 'face'), dict(
        reason=('capacity-admitted local parent' if admitted else
                'local parent exceeds current device budget'),
        requested_layout=resolution.layout,
        conservative_pencil_side=side, selection_face_count=selection_faces,
        retained_output_upper_bound_bytes_per_rank=retained_outputs,
        local_selection=selection, local_reduction=reduction,
        reduction_admission='conservative recipe bound')


def route_summary(mode, receipt):
    """The constructor route and why, for one report line."""
    rows = ", ".join(
        f"{name} {row['aggregate_bytes_per_rank'] / 1e9:.1f}"
        for name, row in (("selection", receipt.get("local_selection")),
                          ("reduction", receipt.get("local_reduction"))) if row)
    budget = next((row['device_budget_bytes_per_rank'] for row in
                   (receipt.get("local_selection"), receipt.get("local_reduction")) if row), None)
    price = "" if budget is None else f"; local parent GB/rank: {rows} of {budget / 1e9:.1f}"
    batch = receipt.get("face_batch") or (
        {"parent_batch": receipt["parent_batch"]} if "parent_batch" in receipt else None)
    if batch is not None:
        gb = lambda v: "none" if v is None else f"{v / 1e9:.1f}"
        rooms = receipt.get("face_eigh_room_bytes_per_rank") or {}
        if "compiled_program_bytes_per_rank" in batch:
            price += (f"; face program {gb(batch['compiled_program_bytes_per_rank'])} GB/rank compiled, "
                      f"sized in {batch['sizing_seconds']:.1f} s")
        price += (f"; face batch {batch['parent_batch']} parent(s), eigh room GB/rank "
                  + ", ".join(f"{phase} {gb(room)}" for phase, room in rooms.items()))
    return (f"{mode} ({receipt['reason']}, conservative pencil side "
            f"{receipt['conservative_pencil_side']}{price})")


def is_face(array):
    spec = tuple(array.sharding.spec)
    return len(spec) == array.ndim and spec[-2:] == ('x', 'y')


def face_program(fn, mesh, *, outputs='matrices'):
    """Compile glue with an explicit matrix/scalar output contract.

    Matrix-only glue preserves every output's trailing two mesh axes,
    including real matrices. Mixed reducer contracts name their matrix
    leaves explicitly; diagnostics and spectra alone replicate. Its checked
    eighs hold their first attempts; a set flag reruns it whole-chain on the
    whole mesh (``distrib_la.checked_program``), the program
    ``call.lower(*args)`` lowers, for sizing.
    """
    # ponytail: a failed check reruns the whole round on the whole mesh (one
    # cached compile; CrI3 saw 1-10 failures per map). Upgrade path: retry
    # only the failed stack.
    import distrib_la
    compiled = {}
    def program(*args):
        signature = jax.tree.structure(args), tuple((a.shape,a.dtype) for a in jax.tree.leaves(args))
        if signature not in compiled:
            shapes = distrib_la.checked_shapes(fn, *args)
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
            compiled[signature]=distrib_la.checked_program(fn,mesh,out)
        return compiled[signature]
    call=lambda *args: program(*args)(*args)
    call.lower=lambda *args: program(*args).lower(*args)
    return call


#: Agreed sizes by (program, argument shapes); every rank asks the same keys.
_SIZES = {}


def compiled_bytes(program, *args):
    """Compiled new bytes per rank of a ``face_program`` at ``args`` (arrays or shape structs).

    Agreed over ranks (the largest any rank measured); a compile that fails on
    any rank posts distrib_la's sizing-failure size, so every rank rejects it. Cached.
    """
    import sys
    from distrib_la.plan import _SIZING_FAILED
    from runtime.aot_memory import agreed_chunk, compiled_new_bytes
    key = program, jax.tree.structure(args), tuple((a.shape, str(a.dtype)) for a in jax.tree.leaves(args))
    if key in _SIZES:
        return _SIZES[key]
    try:
        local = compiled_new_bytes(program.lower(*args).compile())
    except Exception as exc:        # any failure means "does not fit", on every rank
        print(f"shared-pole face program: sizing failed on process {jax.process_index()} "
              f"({type(exc).__name__}: {exc}); rejected on every rank", file=sys.stderr, flush=True)
        local = _SIZING_FAILED
    _SIZES[key] = -agreed_chunk(-int(local))
    return _SIZES[key]


@lru_cache(maxsize=None)
def face_matmul(mesh):
    from distrib_la import matmul
    return partial(matmul, mesh=mesh, backend='distributed', batched_route='auto')


@lru_cache(maxsize=None)
def face_eigh(mesh, n, room=None):
    """The whole-mesh constructor's n x n eigh plan.

    ``room`` is the caller's device bytes per rank beside its admitted live
    set (``face_eigh_room``), the same on every rank. distrib_la decides each
    stack from it: whole matrices per rank where the program that runs it
    compiles within the room, else the whole mesh. Without a room every stack
    runs on the mesh.
    """
    from distrib_la import plan
    return plan('eigh',mesh,n=int(n),backend='distributed',budget_bytes=int(room or 0))


def face_ritz_carrier(mesh, keep_budget):
    """Columns of a face pencil's kept span: the pole budget, per-rank tile on the extent ladder.

    The local round solves its kept span on at most ``keep_budget`` columns
    (``reduce_round``); the face solves it on this carrier, so its Schur and
    final eigensolves run at the budget's side, not the pencil's. The keep cut
    retains at most ``keep_budget`` directions, so no kept column is ever cut.
    The per-rank tile sits on ``runtime.padding.ladder_extent`` (a multiple of
    a power of two, at most 12.5 % padding): cuSOLVERMp's block edge is the
    largest divisor of n/p up to 256, and the budget rounded only to the mesh
    can give a prime n/p (Fe 4^3 on 2x2: 778/2 = 389, block 1, and the face
    reduction ran 12x slower). CrI3 24x24 on 8x8: 5991 -> 6144, tile 768.
    """
    import math
    from runtime.padding import ladder_extent
    divisor = math.lcm(int(mesh.shape['x']), int(mesh.shape['y']))
    return divisor * ladder_extent(-(-int(keep_budget) // divisor))


@lru_cache(maxsize=None)
def face_parent_program(mesh,ordered,odd_moments,keep_budget,retain_span,side,gram_keep=None,eigh_plan=None,
                        carrier=None):
    """Retained static-layout executable builder; all state values are operands.

    ``eigh_plan`` (``face_eigh`` at the reduction's room) decides the pencil's
    eigh stacks, and the program is keyed on it. ``carrier``
    (``face_ritz_carrier``) solves an ordered pencil's kept span on that many
    columns, None keeping the whole H'_vv side.
    """
    from gw.shared_pole_local import solve_parent_pencil
    from gw.shared_pole_gates import sort_shared_pole_columns
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1, shared_real_pole_gates_v1_r3b
    gates=shared_real_pole_gates_ordered_v1 if ordered else shared_real_pole_gates_v1_r3b
    mm=face_matmul(mesh)
    eigh=(face_eigh(mesh,side) if eigh_plan is None else eigh_plan).batched
    def body(points,order,active,qs,os,ds,infinity):
        def pack(parts):
            panels = jnp.concatenate((*parts, jnp.zeros_like(parts[0][..., :int(mesh.shape["y"])])), axis=-1)
            from gw.shared_pole_pencil import _matrix_take_columns
            return _matrix_take_columns(panels, order, NamedSharding(mesh,P(None,"x","y")))
        reduced=solve_parent_pencil(points,pack(qs),pack(os),pack(ds),infinity,active,
            eigh=eigh,matmul=mm,gates=gates,ordered=ordered,odd_moments=odd_moments,
            keep_budget=keep_budget,retain_span=retain_span,gram_keep=gram_keep,
            matrix_sharding=NamedSharding(mesh,P(None,"x","y")),carrier=carrier)
        model,signed,diagnostics=reduced[:3]
        model,permutation=sort_shared_pole_columns(model, matrix_sharding=NamedSharding(mesh,P(None,"x","y")))
        result=model,signed,(*diagnostics,permutation)
        return (*result,reduced[3]) if retain_span else result
    return face_program(body,mesh,outputs='parent')


def face_reduce_round(states,infinity,tables,*,real,mesh,budget,ordered,odd_moments,
                      keep_budget,retain_span=False,admit=True,gram_keep=None,room=None,carrier=None):
    """A batch of physical parents with every matrix tiled over all ranks.

    ``room`` is the room the pencil's eigh stacks are decided against
    (``face_eigh``), or a function of this round's program size: its
    whole-chain program on the whole mesh, compiled on the arrays it runs,
    which is also its retry. ``carrier`` is the kept-span width of an
    ordered reduction (``face_ritz_carrier``), None solving the whole H'_vv side."""
    if real != len(tables['own']):
        raise ValueError('distributed constructor batches contain physical parents only')
    side=tables['active'].shape[-1]
    if admit:
        budget.plan(side,phase='reduction')
    program=lambda room: face_parent_program(mesh,ordered,odd_moments,keep_budget,retain_span,side,gram_keep,
                                             face_eigh(mesh,side,room),None if carrier is None else int(carrier))
    args=(tables['points'],tables['order'],tables['active'],tuple(s[1] for s in states),
          tuple(s[2] for s in states),tuple(s[3] for s in states),tuple(infinity))
    if callable(room):
        room=room(compiled_bytes(program(None),*args))
    result=program(room)(*args)
    model,signed,diagnostics=result[:3]
    output=model,signed,model[1:],diagnostics
    return (*output,result[3]) if retain_span else output


@lru_cache(maxsize=None)
def face_round_check_program(mesh, ordered, eigh_plan):
    """Whole-mesh adapter for the scalar model's existing gate equations.

    ``eigh_plan`` is the n x n plan (``face_eigh``); distrib_la decides from
    its room whether the passivity eighs run on the mesh or one per rank.
    """
    from gw.shared_pole_local import _round_check_equations
    from gw.shared_pole_recipe import (shared_real_pole_gates_ordered_v1,
                                       shared_real_pole_gates_v1_r3b)
    gates = (shared_real_pole_gates_ordered_v1 if ordered else
             shared_real_pole_gates_v1_r3b)
    return face_program(partial(_round_check_equations, matmul=face_matmul(mesh),
                                eigh=eigh_plan.batched, gates=gates, ordered=ordered),
                        mesh, outputs='scalars')


def sector_round_schedule(bank,header,meta,config,mesh,*,execution=None,batch_width=1):
    """Schedule local parent rounds or bounded batches on the whole mesh."""
    from gw.shared_pole_local import parent_rounds
    from gw.gw_config import linalg_resolution
    resolution=linalg_resolution({'linalg':config.backend.linalg})
    execution = resolution.layout if execution is None else execution
    if execution == 'local':
        return [(*row,'local') for row in parent_rounds(header['n_q_irr'],mesh.size)]
    if execution not in ('distributed', 'face'):
        raise ValueError('unsupported resolved constructor linalg layout')
    # Every parent's minus-q actions are in its own bank panels; face parents
    # need neither simultaneous partner parents nor artificial rank padding.
    nq = int(header['n_q_irr'])
    return [(list(range(q, min(q + batch_width, nq))), min(batch_width, nq-q),
             np.arange(min(batch_width, nq-q), dtype=np.int64), 'face')
            for q in range(0, nq, batch_width)]


def face_reduction_bytes(mesh, width, *, rows, side, infinity_width, infinity_arrays, ordered,
                         odd_moments, keep_budget, carrier):
    """``compiled_bytes`` of ``face_parent_program`` for ``width`` parents at a
    pencil ``side`` whose last ``infinity_width``-wide blocks are the infinity
    columns, every finite column on one state panel of ``rows``, and its eighs
    on the whole mesh."""
    blocks = (2 if odd_moments else 0) if ordered else 1
    finite = int(side) - blocks * int(infinity_width)
    face, rep = NamedSharding(mesh, P(None, 'x', 'y')), NamedSharding(mesh, P())
    spec = lambda shape, dtype, sharding=face: jax.ShapeDtypeStruct(shape, dtype, sharding=sharding)
    panel = (spec((width, rows, finite), jnp.complex128),)
    program = face_parent_program(mesh, ordered, odd_moments, keep_budget, False, int(side), None,
                                  face_eigh(mesh, int(side)), carrier)
    return compiled_bytes(program, spec((width, finite), jnp.complex128, rep), spec((width, finite), jnp.int32, rep),
                          spec((width, int(side)), jnp.bool_, rep), panel, panel, panel,
                          (spec((width, rows, int(infinity_width)), jnp.complex128),) * int(infinity_arrays))


def face_batch_width(meta, resolution, *, mesh, ledger, upstream, side, sample_batch,
                     selection_faces, nq, program_bytes):
    """Largest whole-mesh parent batch whose selection and reduction fit.

    The scalar face route runs a batch of physical parents per round, every
    matrix tiled over all ranks (``face_reduce_round``). Before any bank read,
    at the conservative recipe side, the selection price may reject a width;
    only the reduction program's compiled size at that width
    (``program_bytes(width)``, agreed over ranks) admits
    it. Widths start at one parent per rank, so each eigh stack of a round is
    one route-(c) round, and step down in proportion to the room. The
    constructor still admits every phase at its actual side.
    """
    import time
    from distrib_la.plan import _SIZING_FAILED
    from gw.shared_pole_capacity import ConstructorCapacity

    budget = ConstructorCapacity(meta, resolution, mesh_xy=mesh, ledger=ledger,
                                 upstream=upstream, execution='face')
    width, seconds = min(int(nq), int(mesh.size)), 0.0

    def row(phase, **kwargs):
        price, native = budget.quote(side, phase=phase, **kwargs)
        return ledger.preview(resident_bytes_per_rank=price['resident_bytes_per_rank'],
                              workspace_bytes_per_rank=sum(native.values()), concurrent_with=upstream)
    while True:
        budget.batch_width, budget.program_bytes = width, None
        selection = row('selection', sample_batch=sample_batch, selection_faces=selection_faces)
        if selection['device_budget_status'] != 'PASS' and width > 1:
            width -= 1
            continue
        started = time.perf_counter()
        compiled = program_bytes(width)
        budget.program_bytes, seconds = compiled, seconds + time.perf_counter() - started
        reduction = row('reduction')
        receipt = dict(parent_batch=width, compiled_program_bytes_per_rank=compiled,
                       sizing_seconds=seconds, selection=selection, reduction=reduction)
        if reduction['device_budget_status'] == 'PASS' or width == 1:
            return width, receipt
        room = reduction['available_device_bytes_per_rank'] - reduction['aggregate_bytes_per_rank'] + compiled
        width = (width - 1 if compiled >= _SIZING_FAILED else
                 max(1, min(width - 1, width * max(room, 0) // compiled)))


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
    lines = line_panel_count(recipe)
    dense = len(recipe['fit_ids']) - lines
    # Each family's cross panels: 8 [rows of the other family, line width]
    # blocks per line sample (the four ordered states, output and action).
    charge, current = routes
    cross = lines * 8 * (current['packed_extent'] * charge['line_width']
                         + charge['packed_extent'] * current['line_width'])
    budget = ConstructorCapacity(joint, resolution, mesh_xy=mesh, ledger=ledger,
                                 upstream=ledger.live_stages, execution='face')
    for width in range(int(nq), 0, -1):
        budget.batch_width = width
        # The phase formula covers current pencil/actions, not the dense CT/TC
        # stacks (W and dW/ds at the dense fitted samples), the moments and
        # the cross panels that the caller still holds during cross reduction.
        sample_bytes = int(np.ceil(16 * width * ((4 * dense + 8) * joint.n_rmu_padded**2 + cross)
                                   / mesh.size))
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
    # One parent per round; the constructor's actual-side reservations warn.
    return 1, preview



@lru_cache(maxsize=None)
def cross_parent_program(mesh, eigh_plan):
    """The CT joint reduction on the whole mesh, keyed on its eigh plan (``face_eigh``)."""
    from gw.shared_pole_sectors import _cross_reduce_equations
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1 as gates
    # The joint CT metric carries exact-zero rows (inactive retained columns,
    # held span widths); the service eigh deflates them and checks its result.
    eigh = eigh_plan.batched
    return face_program(partial(_cross_reduce_equations,mm=face_matmul(mesh),eigh=eigh,gates=gates,
                                matrix_sharding=NamedSharding(mesh,P(None,"x","y"))),
                        mesh,outputs='cross')


@lru_cache(maxsize=None)
def cross_action_program(mesh,mirror,imaginary,conjugate):
    from gw.shared_pole_sectors import _cross_products
    return face_program(partial(_cross_products,mirror=mirror,imaginary=imaginary,
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

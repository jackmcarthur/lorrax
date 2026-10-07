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
    batch = receipt.get("face_batch")
    decoupled = receipt.get("decoupled")
    if decoupled is not None:
        price += (f"; decoupled: {decoupled['parents']} parents in sub-batches of {decoupled['sub_batch']}, "
                  f"stacks {decoupled['stacks_bytes_per_rank'] / 1e9:.1f} GB/rank, eigh rooms "
                  + "/".join("none" if r is None else f"{r / 1e9:.1f}" for r in decoupled['eigh_room_bytes_per_rank'])
                  + " GB/rank, "
                  + ("every eigh once over the stack" if decoupled['admitted']
                     else "stacks over budget: face rounds over the stacked panels")
                  + f", max |ZAZ-I|/sqrt(R) {decoupled['keep_residual']:.1e}"
                  + (f" / {decoupled['paired_residual']:.1e}" if 'paired_residual' in decoupled else "")
                  + f" ({decoupled['paired_iterations']} Newton-Schulz iteration(s))")
    if batch is not None:
        gb = lambda v: "none" if v is None else f"{v / 1e9:.1f}"
        rooms = receipt.get("face_eigh_room_bytes_per_rank") or {}
        if "program_bytes_per_rank" in batch:
            price += f"; face program {gb(batch['program_bytes_per_rank'])} GB/rank priced from the shapes"
        price += f"; face batch {batch['parent_batch']} parent(s)" + "".join(
            f", {phase} eigh room {gb(room)} GB/rank" for phase, room in rooms.items())
    return (f"{mode} ({receipt['reason']}, conservative pencil side "
            f"{receipt['conservative_pencil_side']}{price})")


def is_face(array):
    spec = tuple(array.sharding.spec)
    return len(spec) == array.ndim and spec[-2:] == ('x', 'y')


#: Face programs compile with XLA's latency-hiding scheduler, which runs the
#: panel all-gathers of ``panel_matmul`` beside the local GEMMs (claims 2953,
#: 3115: Sigma tau -4 to -10 % on the same option; +0.5 GB per rank, inside
#: ``FACE_PROGRAM_COPIES``). Remat is off globally, so the hazard of claim 2961 is closed.
FACE_COMPILER_OPTIONS = {"xla_gpu_enable_latency_hiding_scheduler": True}

#: Local GEMM depth of one interleaved SUMMA panel: ``panel_matmul`` gathers
#: ``p * FACE_PANEL_DEPTH`` contraction columns per step for the whole batch.
FACE_PANEL_DEPTH = 256


def face_program(fn, mesh, *, outputs='matrices'):
    """Compile glue with an explicit matrix/scalar output contract.

    Matrix-only glue preserves every output's trailing two mesh axes,
    including real matrices. Mixed reducer contracts name their matrix
    leaves explicitly; diagnostics and spectra alone replicate. Its checked
    eighs hold their first attempts; a set flag reruns it whole-chain on the
    whole mesh (``distrib_la.checked_program``).
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
            elif outputs == 'mixed':
                # a stage dict: matrices on the face, per-parent vectors and scalars replicated
                out=jax.tree.map(lambda v:matrix(v) if v.ndim>=3 else rep,shapes)
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
            compiled[signature]=distrib_la.checked_program(fn,mesh,out,compiler_options=FACE_COMPILER_OPTIONS)
        return compiled[signature]
    return lambda *args: program(*args)(*args)


@lru_cache(maxsize=None)
def face_matmul(mesh):
    """The whole-mesh constructor's product: ``distrib_la.panel_matmul``, the batched
    2-D SUMMA of the Green builder (one all_gather per operand per panel for the whole
    stack, a panel prefetched, transposed operands by one grid-transpose exchange), so
    a face program holds no distributed-library GEMM and runs on any backend. The
    panel holds ``p * FACE_PANEL_DEPTH`` contraction columns. A non-square mesh keeps
    the provider product (the slow fallback)."""
    from distrib_la import matmul, panel_matmul
    p = int(mesh.shape['x'])
    if p != int(mesh.shape['y']):
        return partial(matmul, mesh=mesh, backend='distributed', batched_route='auto')

    def product(a, b, *, transa='N', transb='N'):
        q = int(a.shape[0])
        m = int(a.shape[2] if transa != 'N' else a.shape[1])
        n = int(b.shape[1] if transb != 'N' else b.shape[2])
        per_column = a.dtype.itemsize * q * (m // p + n // p)
        return panel_matmul(a, b, mesh=mesh, panel_bytes=per_column * 2 * p * FACE_PANEL_DEPTH,
                            transa=transa, transb=transb)
    return product


@lru_cache(maxsize=None)
def _face_eigh(mesh, n):
    from distrib_la import plan
    return plan('eigh',mesh,n=int(n),backend='distributed',budget_bytes=0)


def face_eigh(mesh, n, room=None):
    """The whole-mesh constructor's n x n eigh plan.

    ``room`` is the caller's device bytes per rank beside its admitted live
    set (``face_eigh_room``), the same on every rank. distrib_la decides each
    stack from it: whole matrices per rank where the program that runs it
    fits the room, else the whole mesh. Without a room every stack runs on
    the mesh. The room rides on the plan as its decision input and is in no
    cache key: plans that differ only by room are equal, so the programs
    built on them (``face_parent_program`` and the others) are shared.
    """
    import dataclasses
    return dataclasses.replace(_face_eigh(mesh, int(n)), budget_bytes=int(room or 0))


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
    No budget (the relaxed tier) gives None: the face solves the whole side.
    """
    import math
    from runtime.padding import ladder_extent
    if keep_budget is None:
        return None
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


def face_reduce_round(states,infinity,tables,*,mesh,budget,ordered,odd_moments,
                      keep_budget,retain_span=False,admit=True,gram_keep=None,room=None,carrier=None):
    """A fixed-width batch of parents with every matrix tiled over all ranks.

    ``room`` is the room the pencil's eigh stacks are decided against
    (``face_eigh``), or a function of this round's program price at its
    actual side (``face_reduction_bytes``). ``carrier`` is the kept-span width of an
    ordered reduction (``face_ritz_carrier``), None solving the whole H'_vv
    side. A short last round's synthetic slots repeat its last parent
    (``parent_rounds``); callers read only the leading real slots."""
    side=tables['active'].shape[-1]
    if admit:
        budget.plan(side,phase='reduction')
    program=lambda room: face_parent_program(mesh,ordered,odd_moments,keep_budget,retain_span,side,gram_keep,
                                             face_eigh(mesh,side,room),None if carrier is None else int(carrier))
    args=(tables['points'],tables['order'],tables['active'],tuple(s[1] for s in states),
          tuple(s[2] for s in states),tuple(s[3] for s in states),tuple(infinity))
    if callable(room):
        room=room(face_reduction_bytes(mesh,int(tables['active'].shape[0]),rows=int(states[0][1].shape[-2]),
                                       side=side,carrier=carrier,retain_span=retain_span))
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
    """Schedule local parent rounds or bounded batches on the whole mesh.

    One schedule for both routes (``parent_rounds``): every round has one
    width, P local slots or ``batch_width`` face parents, and a short last
    round repeats its last real parent, so no round program compiles for a
    ragged tail (CrI3 24x24: 61 parents at width 3 recompiled every CC, TT
    and CT program for the width-1 tail, about 120 s cold per leg)."""
    from gw.shared_pole_local import parent_rounds
    from gw.gw_config import linalg_resolution
    resolution=linalg_resolution({'linalg':config.backend.linalg})
    execution = resolution.layout if execution is None else execution
    if execution not in ('local', 'distributed', 'face'):
        raise ValueError('unsupported resolved constructor linalg layout')
    label, width = ('local', mesh.size) if execution == 'local' else ('face', batch_width)
    return [(*row, label) for row in parent_rounds(header['n_q_irr'], width)]


#: A face program holds each dense operand twice at its peak: the tile the
#: byte model counts ([w, r/Px, r/Py] per rank) and the collective's staged
#: copy of it, in the layout the exchange needs. The optimized HLO of the face
#: parent program (main 0a393dd72, 2x2 mesh, side 1024 and 2048, width 2)
#: holds, beside 230/208 tiles c128[w, r/2, r/2], their regrouped copies
#: c128[w, Px, 1, r/2/Px, r/2] and c128[1, Px, r/2, Py, r/2]: the all-to-all
#: source and destination buffers of the distributed matmul's panel exchange
#: and of the eigh stack's reshard, each the same bytes as the tile it moves,
#: live while the tile itself is. XLA cannot alias a collective's source with
#: its destination, so the second copy is structural, not a fusion accident.
#: Measured against one tiled copy: the probe's temp + output 1.17 (side 1024)
#: and 1.46 (2048); the compiled sizing figures of the P4 Fe 4^3 (0.95, CT
#: 1.09-1.51) and CrI3 6x6 (0.82-1.48) receipts and of the P64 CrI3 24x24
#: receipt (1.57-1.63). Two copies bound every one of them
#: (``tests/test_shared_pole_face_price.py`` replays those receipts).
FACE_PROGRAM_COPIES = 2


def _face_price(mesh, rows, width, side, phase, **terms):
    """Bytes per rank of one face round program for ``width`` parents: an upper bound.

    ``FACE_PROGRAM_COPIES`` of the byte model that admits every local round
    (``shared_pole_byte_terms``), tiled over the mesh (``width / P`` copies
    each). The eigh and matmul workspace is quoted beside it
    (``ConstructorCapacity.quote``), as it was beside the compiled figure.
    Nothing is compiled to be measured, and every rank prices the same
    shapes, so the figure is agreed by construction.
    """
    from types import SimpleNamespace
    from gw.shared_pole_capacity import shared_pole_byte_terms
    price = shared_pole_byte_terms(SimpleNamespace(n_rmu_padded=int(rows)), mesh_xy=mesh,
                                   resolution=SimpleNamespace(layout='distributed'), pencil_side=int(side),
                                   parent_batch=int(width), sample_batch=1, phase=phase, **terms)
    return FACE_PROGRAM_COPIES * int(price['terms_bytes_per_rank']['phase_dense_temporaries'])


def face_reduction_bytes(mesh, width, *, rows, side, carrier, retain_span=False):
    """Price of ``face_parent_program`` for ``width`` parents of ``rows`` at a pencil
    ``side``: an ordered pencil's kept span on ``carrier`` columns
    (``face_ritz_carrier``), None (the relaxed tier) solving the whole side."""
    return _face_price(mesh, rows, width, side, 'reduction', ritz_budget=carrier, retain_span=retain_span)


def face_cross_bytes(mesh, width, rows, sides, spans):
    """Price of ``cross_parent_program`` for ``width`` parents: the C-by-T pencil at the
    sectors' ``sides`` projected on their retained ``spans`` (the joint side is their
    sum) on the joint basis of ``rows``."""
    return _face_price(mesh, rows, width, sum(map(int, spans)), 'cross_reduction',
                       cross_original_sides=tuple(int(s) for s in sides))


def face_check_bytes(mesh, rows, side):
    """Price of ``face_round_check_program`` for one parent's model of ``rows`` x ``side``."""
    return _face_price(mesh, rows, 1, side, 'model')


def face_batch_width(meta, resolution, *, mesh, ledger, upstream, side, nq, program_bytes,
                     selection=None, extra=lambda width: 0, eigen_side=None):
    """Largest whole-mesh parent batch whose selection and reduction fit.

    A face route runs a batch of physical parents per round, every matrix
    tiled over all ranks (``face_reduce_round``, the sector rounds). Before
    any bank read, at the conservative recipe side, the ``selection`` price
    (its ``quote`` arguments) may reject a width; the reduction program's
    price at that width (``program_bytes(width)``: ``shared_pole_byte_terms``
    tiled over the mesh, the same on every rank), beside ``extra(width)``
    resident bytes, admits it. Nothing is compiled to be measured (CrI3 24x24
    at P64 compiled six whole-chain programs that never ran, 187 s of every
    cold map 0). Widths start at every parent, so a deck that fits runs one
    round and compiles each program once per shape (Fe 4^3 bispinor at P4:
    one round of 13 instead of four, whose differing sides recompiled every
    program), and step down in proportion to the room. The constructor still
    admits every phase at its actual side.
    """
    from gw.shared_pole_capacity import ConstructorCapacity

    budget = ConstructorCapacity(meta, resolution, mesh_xy=mesh, ledger=ledger,
                                 upstream=upstream, execution='face')
    width = int(nq)

    def row(phase, extra=0, **kwargs):
        price, native = budget.quote(side, phase=phase, **kwargs)
        return ledger.preview(resident_bytes_per_rank=price['resident_bytes_per_rank'] + extra,
                              workspace_bytes_per_rank=sum(native.values()), concurrent_with=upstream)
    while True:
        budget.batch_width, budget.program_bytes = width, None
        picked = None if selection is None else row('selection', **selection)
        if picked is not None and picked['device_budget_status'] != 'PASS' and width > 1:
            width -= 1
            continue
        price = budget.program_bytes = max(1, int(program_bytes(width)))
        reduction = row('reduction', extra(width), eigen_side=eigen_side)
        receipt = dict(parent_batch=width, program_bytes_per_rank=price, selection=picked, reduction=reduction)
        if reduction['device_budget_status'] == 'PASS' or width == 1:
            return width, receipt
        room = reduction['available_device_bytes_per_rank'] - reduction['aggregate_bytes_per_rank'] + price
        width = max(1, min(width - 1, width * max(room, 0) // price))


def sector_batch_width(meta, resolution, recipe, routes, *, mesh, ledger, nq):
    """The common CC/TT/CT face batch, before reading any sample matrix (``face_batch_width``).

    The joint extent covers both retained diagonal spans and the rectangular
    cross pencil; the dense CT/TC stacks (W and dW/ds at the dense fitted
    samples), the moments and the cross panels that the caller still holds
    during the cross reduction are priced beside it. A width is admitted by
    the largest price of the round's three programs at the conservative
    shapes, CC's and TT's ``face_parent_program`` and CT's
    ``cross_parent_program`` (``sector_program_bytes_per_rank``,
    ``face_reduction_bytes`` and ``face_cross_bytes``). Sizing at the sides
    held after map 0 was measured and declined (CrI3 24x24 P64: batch 3 -> 5,
    a round's cost scales with its parents, -9 % per parent at best).
    """
    import copy

    joint = copy.copy(meta)
    joint.n_rmu_padded = sum(row['packed_extent'] for row in routes)
    side = sum(row['conservative_pencil_side'] for row in routes)
    # CT diagonalizes the retained joint span, never the unreduced
    # rectangular C/T pencil. Diagonal sectors still solve their own side.
    spans = [min(row['signed_side_bound'], row['conservative_pencil_side']) for row in routes]
    eigen_side = max(max(row['conservative_pencil_side'] for row in routes), sum(spans))
    lines = line_panel_count(recipe)
    dense = len(recipe['fit_ids']) - lines
    # Each family's cross panels: 8 [rows of the other family, line width]
    # blocks per line sample (the four ordered states, output and action).
    charge, current = routes
    cross = lines * 8 * (current['packed_extent'] * charge['line_width']
                         + charge['packed_extent'] * current['line_width'])
    sizes = {}

    def program_bytes(width):
        for name, row in zip(('CC', 'TT'), routes):
            sizes[name] = face_reduction_bytes(mesh, width, rows=row['packed_extent'],
                                               side=row['conservative_pencil_side'],
                                               carrier=face_ritz_carrier(mesh, row['pole_budget']), retain_span=True)
        sizes['CT'] = face_cross_bytes(mesh, width, joint.n_rmu_padded,
                                       [row['conservative_pencil_side'] for row in routes], spans)
        return max(sizes.values())
    width, receipt = face_batch_width(
        joint, resolution, mesh=mesh, ledger=ledger, upstream=ledger.live_stages, side=side, nq=nq,
        program_bytes=program_bytes, eigen_side=eigen_side,
        extra=lambda width: int(np.ceil(16 * width * ((4 * dense + 8) * joint.n_rmu_padded**2 + cross)
                                        / mesh.size)))
    return width, dict(receipt, sector_program_bytes_per_rank=sizes)


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


# ---- the decoupled face route: every parent in flight, stage programs over sub-batches ----
#
# The paired reduction is GEMM stages with an eigh between them
# (``shared_pole_reduction``). A face round of w parents runs them as one
# program, so its eigh stacks hold w matrices and the round count sets the
# eigh wall: 21 rounds of 3 at CrI3 24x24 P64 cost 21 serial local eighs
# (SECTFAST). Here each GEMM stage runs as its own program over sub-batches
# of ``width`` parents, its outputs stacked for every parent, and each eigh
# runs once over the whole stack, one matrix per rank (route (c), flat to
# b = P). Nothing between stages is held that the next stage does not read.
# The metric corrections stay Newton-Schulz inside their stage: the paired metric
# Y^H H_r Y is the identity to ~1e-8 by construction, so its bound asks one
# iteration (CrI3 6x6), cheaper than a further eigh of the stack (P64 TT: one
# iteration of 61 parents 10.7 s, the eigh root 25.2 s; claim 3425).


@lru_cache(maxsize=None)
def _stack_parents(mesh, ndim):
    out = NamedSharding(mesh, P(None, 'x', 'y')) if ndim >= 3 else NamedSharding(mesh, P())
    return jax.jit(lambda *parts: jnp.concatenate(parts, axis=0), out_shardings=out)


def _stack(mesh, parts):
    """Concatenate per-sub-batch outputs along the parent axis; matrices stay on the
    face, a 0-d entry (a side, a count) is the same in every part and passes through."""
    def join(*leaves):
        return leaves[0] if jnp.ndim(leaves[0]) == 0 else _stack_parents(mesh, leaves[0].ndim)(*leaves)
    return jax.tree.map(join, *parts)


@lru_cache(maxsize=None)
def _stack_slot(mesh, shape, dtype):
    """The program writing one sub-batch into its rows of a parent stack, in place
    (the stack donated), and the program allocating that stack."""
    sharding = NamedSharding(mesh, P(None, 'x', 'y')) if len(shape) >= 3 else NamedSharding(mesh, P())
    write = jax.jit(lambda full, part, i0: jax.lax.dynamic_update_slice_in_dim(full, part, i0, axis=0),
                    donate_argnums=0, out_shardings=sharding)
    return write, jax.jit(partial(jnp.zeros, shape, dtype), out_shardings=sharding)


def _assemble(mesh, nq, parts):
    """``_stack`` with the stack allocated once and each sub-batch written into it in
    place as it arrives (``parts`` an iterator of (offset, tree)): one stack and one
    sub-batch live, not the sub-batches and their concatenation."""
    stack = None
    for i0, part in parts:
        if stack is None:
            stack = jax.tree.map(lambda a: a if jnp.ndim(a) == 0 else
                                 _stack_slot(mesh, (int(nq), *a.shape[1:]), a.dtype)[1](), part)
        stack = jax.tree.map(lambda full, a: full if jnp.ndim(a) == 0 else
                             _stack_slot(mesh, full.shape, full.dtype)[0](full, a, np.int32(i0)),
                             stack, part)
        del part
    return stack


@lru_cache(maxsize=None)
def _rows_program(sharding):
    """One program taking a parent index set's rows of a stack, in the stack's own
    layout; the index is an operand, so every round and sub-batch of one width shares
    one compile."""
    return jax.jit(lambda x, index: x[index], out_shardings=sharding)


def parent_rows(mesh, tree, index):
    """The ``index`` rows (parents) of every stacked leaf of ``tree`` (matrices on the
    face, other leaves in their own named layout or replicated); 0-d leaves pass through."""
    index = np.asarray(index, np.int32)
    def one(a):
        if jnp.ndim(a) == 0:
            return a
        if isinstance(a, np.ndarray):
            return a[index]
        own = a.sharding if isinstance(a.sharding, NamedSharding) and a.sharding.mesh == mesh else None
        layout = NamedSharding(mesh, P(None, 'x', 'y')) if a.ndim >= 3 else NamedSharding(mesh, P())
        return _rows_program(own or layout)(a, index)
    return jax.tree.map(one, tree)


@lru_cache(maxsize=None)
def _hermitian_stack(mesh):
    """(A + A^H) / 2 of a face stack, on the face (the eigh input of H'_vv, as the round forms it)."""
    from distrib_la import hermitian_part
    return jax.jit(hermitian_part, out_shardings=NamedSharding(mesh, P(None, 'x', 'y')))


def _take(mesh, tree, i0, i1):
    return parent_rows(mesh, tree, np.arange(i0, i1))


@lru_cache(maxsize=None)
def _stage_programs(mesh, ordered, odd_moments, keep_budget, retain_span, gram_keep, carrier):
    """The four stage programs of the decoupled face reduction, each jitted on the face."""
    from gw.shared_pole_pencil import assemble_ordered_shared_pole_pencil, _matrix_take_columns
    from gw.shared_pole_reduction import paired_members, keep_stage, paired_stage, output_stage
    from gw.shared_pole_gates import sort_shared_pole_columns, apply_shared_pole_zero_policy, ordered_moment_identity
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1 as gates
    if not (ordered and odd_moments):
        raise ValueError('the decoupled face route is the ordered sector reduction')
    mm = face_matmul(mesh)
    ms = NamedSharding(mesh, P(None, 'x', 'y'))

    # The stage dicts cross program boundaries, so they carry arrays only; a stage's
    # static extents are read back from the shapes it holds (R = 2 hvv, half from the
    # inverse nodes, the kept width from its mask) before the equations run. A stage
    # returns only what it computes: the keys it passes through unchanged are named
    # (at trace time, ``passthrough``) and the caller keeps its own stacks for them,
    # so no program copies a held stack.
    arrays = lambda d: {k: v for k, v in d.items() if not isinstance(v, int)}
    passthrough = {}

    def new_only(name, before, after):
        same = tuple(sorted(k for k, v in after.items() if k in before and v is before[k]))
        passthrough[name] = same
        return arrays({k: v for k, v in after.items() if k not in same})

    def extents(stage):
        return dict(stage, side=2 * int(stage['scale'].shape[-1]), half=int(stage['inverse'].shape[-1]),
                    **({'width': int(stage['kept'].shape[-1])} if 'kept' in stage else {}))

    def stage1(points, order, active, qs, os_, ds, infinity):
        def pack(parts):
            panels = jnp.concatenate((*parts, jnp.zeros_like(parts[0][..., :int(mesh.shape['y'])])), axis=-1)
            return _matrix_take_columns(panels, order, ms)
        q, o, d = pack(qs), pack(os_), pack(ds)
        pencil = assemble_ordered_shared_pole_pencil([(points, q, o, d)], infinity, matmul=mm, matrix_sharding=ms)
        stage = paired_members(pencil, active, gates=gates, matrix_sharding=ms)
        stage['paired'] = jnp.broadcast_to(stage['paired'], (points.shape[0],))
        return arrays(stage)

    def stage2(stage, gamma, u):
        return new_only('keep', stage, keep_stage(extents(stage), gamma, u, matmul=mm, gates=gates,
            keep_budget=keep_budget, retain_span=retain_span, matrix_sharding=ms, gram_keep=gram_keep, carrier=carrier))

    def stage3(stage, gamma_r, u_r):
        return new_only('paired', stage, paired_stage(extents(stage), gamma_r, u_r, matmul=mm, gates=gates,
                                                      matrix_sharding=ms, gram_keep=gram_keep))

    def stage4(stage, mu, rotation, infinity):
        reduced = output_stage(extents(stage), mu, rotation, matmul=mm, gates=gates, retain_span=retain_span,
                               matrix_sharding=ms)
        model, signed, reduction = reduced[:3]
        retained = ordered_moment_identity(signed, infinity, matmul=mm)
        model, zero = apply_shared_pole_zero_policy(model, gates=gates)
        zero['zero_policy'] = zero['zero_policy'] & reduction['infinite_weight_ok']
        model, permutation = sort_shared_pole_columns(model, matrix_sharding=ms)
        result = model, signed, (reduction, zero, retained, permutation)
        return (*result, reduced[3]) if retain_span else result
    return tuple(face_program(fn, mesh, outputs='mixed' if i < 3 else 'parent')
                 for i, fn in enumerate((stage1, stage2, stage3, stage4))), passthrough


def decoupled_stage_bytes(*, nq, ranks, side, carrier, packed, held, dw_panels, program):
    """Per-rank bytes of the decoupled reduction of ``nq`` parents (complex128).

    ``side`` is the pencil side (2 hvv), ``carrier`` the kept-span columns c (None: hvv),
    ``packed`` the rows, ``held`` the (node, Q, O) and infinity panels per rank,
    ``dw_panels`` the dW Q panels per rank (released after the pencil), ``program`` the
    stage program's bound at the sub-batch width. A stage's run holds its input stack and
    the stack it writes in place: the keep stage the paired members and the restricted
    pencil, the paired stage the restricted pencil and (Y, Y^H G_r Y), the output stage
    (Y, Y^H G_r Y, O_r, span) and the model and coefficient outputs. Returns
    ``(resident, boundaries)``: the stacks row and the three eigh stacks' boundaries
    (H'_vv beside the members and its Hermitian copy, the Schur complement beside the
    restricted pencil, Y^H G_r Y beside the output stage's input), each with ``held``.
    """
    hvv = int(side) // 2
    c = int(carrier or hvv)
    two, packed = 2 * c, int(packed)
    per_rank = lambda b: -(-int(b) * int(nq) // int(ranks))
    members = 16 * (6 * hvv * hvv + 2 * packed * hvv)
    restricted = 16 * (2 * two ** 2 + 2 * c * c + hvv * c + packed * two)
    ritz = 16 * (2 * two ** 2 + hvv * c + packed * two)
    outputs = 16 * (int(side) * two + 2 * packed * two)
    stacks = per_rank(max(members + restricted, restricted + 16 * 2 * two ** 2, ritz + outputs))
    boundaries = tuple(per_rank(b) + int(held) for b in (members + 16 * hvv * hvv, restricted, ritz))
    return stacks + int(held) + int(dw_panels) + int(program), boundaries


def decoupled_cross_bytes(*, nq, ranks, side, rows):
    """Per-rank bytes of the decoupled CT of ``nq`` parents at joint ``side`` K with output
    ``rows`` (n_C, n_T): ``(stacks, boundaries)``, the pencil stack (metric, value, O_C, O_T)
    beside the keep stage's (Y^H V Y, Y), and the two eighs' boundaries (the metric beside the
    pencil, the Ritz step beside (Y^H V Y, Y, O_C, O_T))."""
    per_rank = lambda b: -(-int(b) * int(nq) // int(ranks))
    side, rows = int(side), sum(int(r) for r in rows)
    pencil, keep = 16 * (2 * side * side + rows * side), 16 * 2 * side * side
    return per_rank(pencil + keep), (per_rank(pencil), per_rank(keep + 16 * rows * side))


def _device_free_bytes():
    """This process's free device pool bytes (limit minus in use), or None off a pool (CPU)."""
    stats = jax.local_devices()[0].memory_stats() or {}
    limit, used = stats.get('bytes_limit'), stats.get('bytes_in_use')
    return None if limit is None or used is None else int(limit) - int(used)


def eigh_program_bytes(plan, stack, *, mesh):
    """Per-rank bytes the batched eigh of ``stack`` adds beside it, from the service's decision
    (``stack_route``): route (c)'s program (outputs, temporaries and the local solver's workspace;
    the operand excluded), or on the whole mesh the vectors stack, the values and one solve's
    vendor workspace (``distrib_la.workspace_bytes_per_rank``)."""
    import distrib_la
    route = plan.stack_route(stack.shape, stack.dtype, traced=False)
    if route.route == 'batch_reshard' and route.program_bytes:
        return int(route.program_bytes)
    nq, n = int(stack.shape[0]), int(stack.shape[-1])
    outputs = -(-nq * n * n * stack.dtype.itemsize // int(mesh.size)) + nq * n * 8
    return outputs + int(distrib_la.workspace_bytes_per_rank(plan, 'eigh', ((n, n),), stack.dtype))


def guarded_eigh(plan, stack, *, mesh, label):
    """``plan``'s batched eigh of ``stack``, with a diagnostic on its route-(c) decision.

    Where the ledger admits a stack one whole matrix per rank, the pool's measured free bytes,
    the minimum over processes, are compared with that program; a shortfall gives one warning
    naming it. The route never changes on it: a route read from free memory would move results
    with allocator state (decisions.md, fixed-tile). Honest prices are the protection.
    """
    route = plan.stack_route(stack.shape, stack.dtype, traced=False)
    if route.route == 'batch_reshard' and route.program_bytes:
        import numpy as np
        from common.collectives import all_gather_processes
        free = _device_free_bytes()
        agreed = int(np.min(all_gather_processes(np.asarray([-1 if free is None else free], np.int64))))
        if 0 <= agreed < int(route.program_bytes):
            import warnings
            warnings.warn(f"shared-pole decoupled {label}: the {int(stack.shape[0])} x {int(stack.shape[-1])}^2 eigh "
                          f"stack's program needs {route.program_bytes / 1e9:.1f} GB/rank and the pool has "
                          f"{agreed / 1e9:.1f} GB free on some rank (short by "
                          f"{(route.program_bytes - agreed) / 1e9:.1f} GB); an out-of-memory is possible",
                          RuntimeWarning)
    return plan.batched(stack)


def face_reduce_decoupled(states, infinity, tables, *, mesh, eigh_plans, width, ordered, odd_moments,
                          keep_budget, retain_span=False, gram_keep=None, carrier=None, eigh_rows=None):
    """All ``nq`` parents' ordered reduction: stage programs over ``width`` parents at a
    time, each eigh over the whole stack. ``eigh_plans`` are the three stacks' plans
    (H'_vv, Schur, reduced), each carrying the room beside its own boundary stack
    (``face_eigh``), so distrib_la runs each one whole matrix per rank where that fits.
    ``states`` is the caller's list: the derivative panels (dW Q) enter the pencil only,
    so after the first stage each entry keeps (node, Q, O) and they are released before
    the eighs. ``eigh_rows(k, plan, stack)`` (optional) is the caller's context that prices
    eigh stack k (its ledger row) while it runs. Returns what ``face_reduce_round`` returns,
    for every parent."""
    nq = int(tables['active'].shape[0])
    programs = _stage_programs(mesh, bool(ordered), bool(odd_moments), None if keep_budget is None else int(keep_budget),
                               bool(retain_span), gram_keep, None if carrier is None else int(carrier))
    (stage1, stage2, stage3, stage4), passthrough = programs
    hvv_eigh, schur_eigh, reduced_eigh = (partial(guarded_eigh, plan, mesh=mesh, label=label)
                                          for plan, label in zip(eigh_plans, ('H_vv', 'Schur', 'reduced')))
    cuts = [(i, min(i + int(width), nq)) for i in range(0, nq, int(width))]
    inputs = (tables['points'], tables['order'], tables['active'],
              tuple(s[1] for s in states), tuple(s[2] for s in states), tuple(s[3] for s in states), tuple(infinity))

    def run(program, *stacks):
        return _assemble(mesh, nq, ((i0, program(*_take(mesh, stacks, i0, i1))) for i0, i1 in cuts))

    def advance(name, program, stage, *operands):
        new = run(program, stage, *operands)
        return {**{k: stage[k] for k in passthrough[name]}, **new}
    from common import timing
    from contextlib import nullcontext
    priced = eigh_rows or (lambda k, plan, stack: nullcontext())
    # One timing section per stage and eigh: the report gives each its wall and pool high-water.
    with timing.section('decoupled.members'):
        stage = run(stage1, *inputs)
    infinity = inputs[6]
    del inputs
    states[:] = [s[:3] for s in states]
    with timing.section('decoupled.eigh_hvv'):
        hermitian = _hermitian_stack(mesh)(stage['h_vv'])
        with priced(0, eigh_plans[0], hermitian):
            gamma, u = hvv_eigh(hermitian)
        del hermitian
    with timing.section('decoupled.keep'):
        stage = advance('keep', stage2, stage, gamma, u)
    del gamma, u
    with timing.section('decoupled.eigh_schur'), priced(1, eigh_plans[1], stage['schur']):
        gamma_r, u_r = schur_eigh(stage['schur'])
    with timing.section('decoupled.paired'):
        stage = advance('paired', stage3, stage, gamma_r, u_r)
    del gamma_r, u_r
    with timing.section('decoupled.eigh_reduced'), priced(2, eigh_plans[2], stage['reduced']):
        mu, rotation = reduced_eigh(stage['reduced'])
    with timing.section('decoupled.output'):
        result = run(stage4, stage, mu, rotation, infinity)
    del stage, mu, rotation
    model, signed, diagnostics = result[:3]
    output = model, signed, model[1:], diagnostics
    return (*output, result[3]) if retain_span else output


# ---- the decoupled CT: every parent's joint pencil, each eigh once over the stack ----
#
# A CT round assembles the C-by-T pencil from that round's samples, which stay per round;
# the joint [K, K] pencils (metric, value) and the two output panels are stacked for every
# parent, and the joint reduction's two eighs (the metric's keep cut, the Ritz step) run once
# over the stack, its two GEMM stages over sub-batches, as the diagonal sectors do.


@lru_cache(maxsize=None)
def cross_pencil_program(mesh):
    """The CT joint pencil (metric, value, O_C, O_T) of a face round, no eigh."""
    from gw.shared_pole_sectors import _cross_pencil_equations
    return face_program(partial(_cross_pencil_equations, mm=face_matmul(mesh),
                                matrix_sharding=NamedSharding(mesh, P(None, "x", "y"))), mesh)


@lru_cache(maxsize=None)
def _cross_stage_programs(mesh):
    from gw.shared_pole_sectors import joint_keep_stage, joint_output_stage
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1 as gates
    mm = face_matmul(mesh)
    ms = NamedSharding(mesh, P(None, 'x', 'y'))
    flat = lambda d: {k: v for k, v in d.items() if k != 'diagnostics'} | {'d_' + k: v for k, v in d['diagnostics'].items()}

    # The output panels pass through the keep stage untouched: they stay the caller's stacks.
    def keep(metric, value, oc, ot, gamma, u):
        stage = flat(joint_keep_stage((metric, value, oc, ot), gamma, u, matmul=mm, gates=gates, matrix_sharding=ms))
        return {k: v for k, v in stage.items() if k not in ('oc', 'ot')}

    def output(stage, oc, ot, values, rotation):
        nested = {k: v for k, v in stage.items() if not k.startswith('d_')}
        nested['diagnostics'] = {k[2:]: v for k, v in stage.items() if k.startswith('d_')}
        return joint_output_stage(dict(nested, oc=oc, ot=ot), values, rotation, matmul=mm)
    return face_program(keep, mesh, outputs='mixed'), face_program(output, mesh, outputs='cross')


def face_cross_decoupled(pencil, *, mesh, eigh_plans, width, eigh_rows=None):
    """Every parent's CT joint reduction from the stacked ``pencil`` (metric, value, O_C, O_T),
    the caller's list: the metric and value members are released (set to None) after the keep stage:
    the keep stage and the output stage over ``width`` parents at a time, the metric's and the
    Ritz step's eighs once over the stack (``guarded_eigh``). Returns what the round's
    ``reduce_cross_round`` returns, for every parent. ``eigh_rows(k, plan, stack)`` prices eigh k."""
    from contextlib import nullcontext
    from common import timing
    priced = eigh_rows or (lambda k, plan, stack: nullcontext())
    nq = int(pencil[0].shape[0])
    keep, output = _cross_stage_programs(mesh)
    cuts = [(i, min(i + int(width), nq)) for i in range(0, nq, int(width))]

    def run(program, *stacks):
        return _assemble(mesh, nq, ((i0, program(*_take(mesh, stacks, i0, i1))) for i0, i1 in cuts))
    with timing.section('decoupled.eigh_metric'), priced(0, eigh_plans[0], pencil[0]):
        gamma, u = guarded_eigh(eigh_plans[0], pencil[0], mesh=mesh, label='CT metric')
    with timing.section('decoupled.keep'):
        stage = run(keep, *pencil, gamma, u)
    del gamma, u
    pencil[0] = pencil[1] = None
    reduced = stage.pop('reduced')
    with timing.section('decoupled.eigh_ritz'), priced(1, eigh_plans[1], reduced):
        values, rotation = guarded_eigh(eigh_plans[1], reduced, mesh=mesh, label='CT Ritz')
    del reduced
    with timing.section('decoupled.output'):
        return run(output, stage, pencil[2], pencil[3], values, rotation)


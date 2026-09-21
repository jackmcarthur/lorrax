"""Execution adapters for shared-pole equations on the complete x/y mesh.

The local adapter lives in shared_pole_local. This module supplies no new
physics: matrix operations enter through distrib_la, matrix results remain
face tiled, and only spectra, masks and small scalar receipts replicate.
"""
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np
from common import timing
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
            elif outputs == 'mixed':
                out=(jax.tree.map(matrix,shapes[0]),scalar_tree(shapes[1]))
            elif outputs == 'probe':
                out=(jax.tree.map(matrix,shapes[0]),scalar_tree(shapes[1]),
                     jax.tree.map(matrix,shapes[2]))
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
def face_hermitian_program(mesh):
    from distrib_la import hermitian_part
    return face_program(hermitian_part, mesh)


@lru_cache(maxsize=None)
def face_ordered_prepare_program(mesh, odd_moments):
    """Pack, assemble and normalize one ordered parent on the complete mesh."""
    from gw.shared_pole_pencil import assemble_ordered_shared_pole_pencil
    from gw.shared_pole_reduction import prepare_ordered_shared_pole_reduction

    matrix_sharding = NamedSharding(mesh, P(None, 'x', 'y'))
    mm = face_matmul(mesh)

    def body(points, order, active, qs, os, ds, infinity):
        def pack(parts):
            return jax.lax.with_sharding_constraint(
                jnp.take(jnp.concatenate(parts, axis=-1), order[0], axis=-1,
                         mode='fill', fill_value=0), matrix_sharding)
        finite = [(points, pack(qs), pack(os), pack(ds))]
        pencil = assemble_ordered_shared_pole_pencil(
            finite, infinity if odd_moments else None, matmul=mm,
            matrix_sharding=matrix_sharding)
        return prepare_ordered_shared_pole_reduction(
            pencil, active, matrix_sharding=matrix_sharding)

    return face_program(body, mesh, outputs='mixed')


@lru_cache(maxsize=None)
def face_ordered_probe_program(mesh, odd_moments):
    """Diagnostic canonical prepare retaining its raw H and H_vv."""
    from gw.shared_pole_pencil import assemble_ordered_shared_pole_pencil
    from gw.shared_pole_reduction import prepare_ordered_shared_pole_reduction

    matrix_sharding = NamedSharding(mesh, P(None, 'x', 'y'))
    mm = face_matmul(mesh)

    def body(points, order, active, qs, os, ds, infinity):
        def pack(parts):
            return jax.lax.with_sharding_constraint(
                jnp.take(jnp.concatenate(parts, axis=-1), order[0], axis=-1,
                         mode='fill', fill_value=0), matrix_sharding)
        pencil = assemble_ordered_shared_pole_pencil(
            [(points, pack(qs), pack(os), pack(ds))],
            infinity if odd_moments else None, matmul=mm,
            matrix_sharding=matrix_sharding)
        return prepare_ordered_shared_pole_reduction(
            pencil, active, matrix_sharding=matrix_sharding, return_raw=True)

    return face_program(body, mesh, outputs='probe')


@lru_cache(maxsize=None)
def face_ordered_restrict_program(mesh, keep_budget, retain_span):
    """First metric correction through the restricted H/G/O pencil."""
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1 as gates
    from gw.shared_pole_reduction import restrict_ordered_shared_pole_reduction

    matrix_sharding = NamedSharding(mesh, P(None, 'x', 'y'))
    fn = partial(restrict_ordered_shared_pole_reduction,
                 matmul=face_matmul(mesh), gates=gates,
                 keep_budget=keep_budget, retain_span=retain_span,
                 matrix_sharding=matrix_sharding)
    return face_program(fn, mesh, outputs='mixed')


@lru_cache(maxsize=None)
def face_ordered_ritz_program(mesh, retain_span):
    """Second metric correction through the final signed Ritz matrix."""
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1 as gates
    from gw.shared_pole_reduction import ritz_ordered_shared_pole_reduction

    matrix_sharding = NamedSharding(mesh, P(None, 'x', 'y'))
    fn = partial(ritz_ordered_shared_pole_reduction,
                 matmul=face_matmul(mesh), gates=gates,
                 retain_span=retain_span, matrix_sharding=matrix_sharding)
    return face_program(fn, mesh, outputs='mixed')


@lru_cache(maxsize=None)
def face_ordered_finish_program(mesh, odd_moments, retain_span):
    """Finish the signed model, ordered identities, zero policy and sort."""
    from gw.shared_pole_gates import sort_shared_pole_columns
    from gw.shared_pole_local import finalize_ordered_parent_pencil
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1 as gates
    from gw.shared_pole_reduction import finish_ordered_shared_pole_reduction

    matrix_sharding = NamedSharding(mesh, P(None, 'x', 'y'))
    mm = face_matmul(mesh)

    def body(ritz, mu, rotation, infinity):
        reduced = finish_ordered_shared_pole_reduction(
            ritz, (mu, rotation), matmul=mm, gates=gates,
            retain_span=retain_span, matrix_sharding=matrix_sharding)
        finalized = finalize_ordered_parent_pencil(
            reduced, infinity, matmul=mm, gates=gates,
            odd_moments=odd_moments, retain_span=retain_span)
        model, signed, diagnostics = finalized[:3]
        model, permutation = sort_shared_pole_columns(model)
        result = model, signed, (*diagnostics, permutation)
        return (*result, finalized[3]) if retain_span else result

    return face_program(body, mesh, outputs='parent')


@lru_cache(maxsize=None)
def _probe_vectors(mesh):
    """Replicate the tiny raw-H_vv diagonal used to choose probe columns."""
    return face_program(
        lambda h_vv: jnp.real(jnp.diagonal(h_vv, axis1=-2, axis2=-1)),
        mesh, outputs='scalars')


@lru_cache(maxsize=None)
def _probe_principal(mesh, original, paired):
    """Gather only bounded principal submatrices from all-P raw matrices."""
    original = jnp.asarray(original, dtype=jnp.int32)
    paired = jnp.asarray(paired, dtype=jnp.int32)

    def body(h, h_vv):
        h = jnp.take(jnp.take(h, original, axis=-2), original, axis=-1)
        h_vv = jnp.take(jnp.take(h_vv, paired, axis=-2), paired, axis=-1)
        return h, h_vv

    return face_program(body, mesh, outputs='scalars')


@lru_cache(maxsize=None)
def _probe_panels(mesh, finite_indices, infinity_indices):
    """Replicate selected Q/O/D columns; the full panels remain all-P."""
    finite_indices = jnp.asarray(finite_indices, dtype=jnp.int32)
    infinity_indices = jnp.asarray(infinity_indices, dtype=jnp.int32)

    def body(points, order, active, qs, os, ds, infinity):
        half = points.shape[-1] // 2
        finite = points.shape[-1]
        n_inf = (active.shape[-1] - finite) // 2
        finite_columns = jnp.concatenate((finite_indices,
                                          half + finite_indices))

        def pack(parts):
            packed = jnp.take(jnp.concatenate(parts, axis=-1), order[0],
                              axis=-1, mode='fill', fill_value=0)
            return jnp.take(packed, finite_columns, axis=-1)

        small_points = jnp.take(points, finite_columns, axis=-1)
        small_active = jnp.concatenate((
            jnp.take(active[:, :finite], finite_columns, axis=-1),
            jnp.take(active[:, finite:finite + n_inf], infinity_indices, axis=-1),
            jnp.take(active[:, finite + n_inf:], infinity_indices, axis=-1)), axis=-1)
        small_infinity = tuple(jnp.take(a, infinity_indices, axis=-1)
                               for a in infinity)
        return (small_points, small_active, pack(qs), pack(os), pack(ds),
                small_infinity)

    return face_program(body, mesh, outputs='scalars')


@lru_cache(maxsize=None)
def _probe_small_native(mesh, odd_moments):
    """Run the canonical equations on the bounded subset through face GEMM."""
    from gw.shared_pole_pencil import assemble_ordered_shared_pole_pencil
    from gw.shared_pole_reduction import prepare_ordered_shared_pole_reduction

    matrix_sharding = NamedSharding(mesh, P(None, 'x', 'y'))

    def body(points, active, q, o, d, infinity):
        pencil = assemble_ordered_shared_pole_pencil(
            [(points, q, o, d)], infinity if odd_moments else None,
            matmul=face_matmul(mesh), matrix_sharding=matrix_sharding)
        return prepare_ordered_shared_pole_reduction(
            pencil, active, matrix_sharding=matrix_sharding, return_raw=True)

    return face_program(body, mesh, outputs='probe')


def _probe_small_reference(points, active, q, o, d, infinity, odd_moments):
    """Canonical bounded equations with replicated JAX matmul as reference."""
    from gw.shared_pole_pencil import assemble_ordered_shared_pole_pencil
    from gw.shared_pole_reduction import prepare_ordered_shared_pole_reduction

    def mm(a, b, transa='N', transb='N'):
        def op(x, trans):
            if trans == 'N':
                return x
            x = jnp.swapaxes(x, -1, -2)
            return jnp.conj(x) if trans == 'C' else x
        return jnp.matmul(op(a, transa), op(b, transb))

    pencil = assemble_ordered_shared_pole_pencil(
        [(points, q, o, d)], infinity if odd_moments else None, matmul=mm)
    return prepare_ordered_shared_pole_reduction(
        pencil, active, return_raw=True)


def _choose_probe_indices(diagonal, active, count=8):
    """Choose worst live/padding and healthy controls, then fill deterministically."""
    diagonal, active = np.asarray(diagonal), np.asarray(active, bool)
    chosen = []

    def add(indices):
        for index in indices:
            index = int(index)
            if index not in chosen and len(chosen) < count:
                chosen.append(index)

    bad = np.flatnonzero(active & (~np.isfinite(diagonal) | (diagonal <= 0)))
    add(bad[:3])
    padding = np.flatnonzero(~active)
    add(padding[np.argsort(np.abs(diagonal[padding]))[::-1][:2]])
    live = np.flatnonzero(active & np.isfinite(diagonal))
    add(live[np.argsort(diagonal[live])[:2]])
    add(live[np.argsort(diagonal[live])[::-1]])
    add(np.arange(diagonal.size))
    if len(chosen) != count:
        raise ValueError(f'raw-Hvv probe requires {count} columns, got {len(chosen)}')
    return tuple(chosen)


def _run_ordered_raw_hvv_probe(points, order, active, qs, os, ds, infinity,
                                raw, *, mesh, odd_moments):
    """Print a bounded native/reference receipt and deliberately stop before eig."""
    raw_h, raw_h_vv = raw
    diagonal = np.asarray(_probe_vectors(mesh)(raw_h_vv))[0]
    active_host = np.asarray(active, bool)[0]
    points_host = np.asarray(points)[0]
    finite = int(points.shape[-1])
    half, n_inf = finite // 2, (int(active.shape[-1]) - finite) // 2
    paired_active = np.concatenate((active_host[:half],
                                    active_host[finite:finite + n_inf]))
    finite_indices = _choose_probe_indices(diagonal[:half], paired_active[:half])
    infinity_indices = _choose_probe_indices(diagonal[half:], paired_active[half:])
    original = (*finite_indices, *(half + np.asarray(finite_indices)),
                *(finite + np.asarray(infinity_indices)),
                *(finite + n_inf + np.asarray(infinity_indices)))
    paired = (*finite_indices, *(half + np.asarray(infinity_indices)),)
    full_h, full_h_vv = _probe_principal(mesh, tuple(original), tuple(paired))(
        raw_h, raw_h_vv)
    small = _probe_panels(mesh, finite_indices, infinity_indices)(
        points, order, active, qs, os, ds, infinity)
    small_points, small_active, q, o, d, small_infinity = small
    native = _probe_small_native(mesh, odd_moments)(
        small_points, small_active, q, o, d, small_infinity)
    # Complete every native product before dispatching the reference path;
    # this probe must not recreate the native/XLA overlap under diagnosis.
    jax.block_until_ready(native)
    native_h, native_h_vv = _probe_principal(
        mesh, tuple(range(32)), tuple(range(16)))(*native[2])
    reference = _probe_small_reference(
        small_points, small_active, q, o, d, small_infinity, odd_moments)

    full_h, full_h_vv = np.asarray(full_h)[0], np.asarray(full_h_vv)[0]
    native_h, native_h_vv = np.asarray(native_h)[0], np.asarray(native_h_vv)[0]
    reference_h, reference_h_vv = (np.asarray(a)[0] for a in reference[2])
    q_host = np.asarray(q)[0]

    def error(a, b):
        absolute = float(np.max(np.abs(a - b)))
        relative = absolute / max(float(np.max(np.abs(b))), np.finfo(float).tiny)
        return absolute, relative

    if jax.process_index() == 0:
        print('shared-pole raw Hvv diagnostic: parent=0 paired=16 finite=8 infinity=8', flush=True)
        print('role index active z q_norm_plus q_norm_minus raw_diag Hpp Hpm Hmp Hmm diagonal_sum cross_sum reconstructed', flush=True)
        f = len(finite_indices)
        for local, index in enumerate(finite_indices):
            z = points_host[index]
            hpp, hpm = full_h[local, local], full_h[local, f + local]
            hmp, hmm = full_h[f + local, local], full_h[f + local, f + local]
            diagonal_sum, cross_sum = hpp + hmm, -(hpm + hmp)
            reconstructed = ('PAD0' if not paired_active[index] else
                             ((diagonal_sum + cross_sum) / (4 * abs(z) ** 2)
                              if abs(z) else 'NA'))
            print(f'finite {index} {int(paired_active[index])} {z!r} '
                  f'{np.linalg.norm(q_host[:, local]):.17e} '
                  f'{np.linalg.norm(q_host[:, f + local]):.17e} '
                  f'{full_h_vv[local, local]!r} '
                  f'{hpp!r} {hpm!r} {hmp!r} {hmm!r} '
                  f'{diagonal_sum!r} {cross_sum!r} '
                  f'{reconstructed!r}', flush=True)
        print('role index active z q_norm raw_diag infinity_H_k0k0', flush=True)
        for local, index in enumerate(infinity_indices):
            paired_local = f + local
            original_local = 2 * f + local
            print(f'infinity {index} {int(paired_active[half + index])} 0j '
                  f'{np.linalg.norm(np.asarray(small_infinity[0])[0, :, local]):.17e} '
                  f'{full_h_vv[paired_local, paired_local]!r} '
                  f'{full_h[original_local, original_local]!r}', flush=True)
        print('shared-pole raw Hvv comparison '
              f'full_native_vs_small_jax_H={error(full_h, reference_h)} '
              f'full_native_vs_small_native_H={error(full_h, native_h)} '
              f'small_native_vs_small_jax_H={error(native_h, reference_h)} '
              f'full_native_vs_small_jax_Hvv={error(full_h_vv, reference_h_vv)} '
              f'full_native_vs_small_native_Hvv={error(full_h_vv, native_h_vv)} '
              f'small_native_vs_small_jax_Hvv={error(native_h_vv, reference_h_vv)} '
              f'active_nonpositive={int(np.sum(paired_active & (diagonal <= 0)))} '
              f'active_nonfinite={int(np.sum(paired_active & ~np.isfinite(diagonal)))} '
              f'inactive_nonzero={int(np.sum(~paired_active & (diagonal != 0)))} '
              f'active_raw_min={np.nanmin(diagonal[paired_active])!r} '
              f'inactive_abs_max={np.max(np.abs(diagonal[~paired_active]), initial=0)!r}',
              flush=True)
    raise RuntimeError('DIAGNOSTIC COMPLETE: stopped before shared-pole eigensolves')


def face_ordered_parent(points, order, active, qs, os, ds, infinity, *, mesh,
                        odd_moments, keep_budget, retain_span):
    """Run the three ordered eig equations with explicit eager boundaries."""
    prepare = face_ordered_probe_program(mesh, odd_moments)
    with timing.section(
            'spole.ordered.prepare', announce=True,
            label=f'shared-pole ordered prepare pencil={active.shape[-1]}') as section:
        prepared_face = prepare(points, order, active, qs, os, ds, infinity)
        section.watch(prepared_face)
    prepared = prepared_face[:2]
    _run_ordered_raw_hvv_probe(
        points, order, active, qs, os, ds, infinity, prepared_face[2], mesh=mesh,
        odd_moments=odd_moments)
    with timing.section(
            'spole.ordered.eigh1_input', announce=True,
            label='shared-pole ordered Hvv Hermitian projection') as section:
        eig_input = face_hermitian_program(mesh)(prepared[0][5])
        section.watch(eig_input)
    first_side = int(eig_input.shape[-1])
    with timing.section(
            'spole.ordered.eigh1', announce=True,
            label=f'shared-pole ordered Hvv eig matrix={first_side}') as section:
        gamma, u = face_eigh(mesh, first_side).batched(eig_input)
        section.watch(gamma, u)
    with timing.section(
            'spole.ordered.restrict', announce=True,
            label=f'shared-pole ordered restricted build matrix={first_side}') as section:
        restricted = face_ordered_restrict_program(
            mesh, keep_budget, retain_span)(prepared, (gamma, u))
        section.watch(restricted)
    del prepared_face, prepared, eig_input, gamma, u

    second_input = restricted[0][0]
    second_side = int(second_input.shape[-1])
    with timing.section(
            'spole.ordered.eigh2', announce=True,
            label=f'shared-pole ordered Hr eig matrix={second_side}') as section:
        gamma_r, u_r = face_eigh(mesh, second_side).batched(second_input)
        section.watch(gamma_r, u_r)
    with timing.section(
            'spole.ordered.ritz', announce=True,
            label=f'shared-pole ordered Ritz build matrix={second_side}') as section:
        ritz = face_ordered_ritz_program(mesh, retain_span)(
            restricted, (gamma_r, u_r))
        section.watch(ritz)
    del restricted, second_input, gamma_r, u_r

    third_input = ritz[0][0]
    third_side = int(third_input.shape[-1])
    with timing.section(
            'spole.ordered.eigh3', announce=True,
            label=f'shared-pole ordered signed eig matrix={third_side}') as section:
        mu, rotation = face_eigh(mesh, third_side).batched(third_input)
        section.watch(mu, rotation)
    with timing.section(
            'spole.ordered.finish', announce=True,
            label=f'shared-pole ordered finish matrix={third_side}') as section:
        result = face_ordered_finish_program(
            mesh, odd_moments, retain_span)(ritz, mu, rotation, infinity)
        section.watch(result)
    del ritz, third_input, mu, rotation
    return result


@lru_cache(maxsize=None)
def face_parent_program(mesh,ordered,odd_moments,keep_budget,retain_span,side):
    """Retained static-layout executable builder; all state values are operands."""
    from gw.shared_pole_local import solve_parent_pencil
    from gw.shared_pole_gates import sort_shared_pole_columns
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1, shared_real_pole_gates_v1_r3b
    gates=shared_real_pole_gates_ordered_v1 if ordered else shared_real_pole_gates_v1_r3b
    mm=face_matmul(mesh)
    eigh_plan=face_eigh(mesh,side)
    def body(points,order,active,qs,os,ds,infinity):
        def pack(parts):
            return jax.lax.with_sharding_constraint(
                jnp.take(jnp.concatenate(parts,axis=-1),order[0],axis=-1,mode='fill',fill_value=0),
                NamedSharding(mesh,P(None,'x','y')))
        reduced=solve_parent_pencil(points,pack(qs),pack(os),pack(ds),infinity,active,
            eigh=eigh_plan.batched,matmul=mm,gates=gates,ordered=ordered,odd_moments=odd_moments,
            keep_budget=keep_budget,retain_span=retain_span,
            matrix_sharding=NamedSharding(mesh,P(None,"x","y")))
        model,signed,diagnostics=reduced[:3]
        model,permutation=sort_shared_pole_columns(model)
        result=model,signed,(*diagnostics,permutation)
        return (*result,reduced[3]) if retain_span else result
    return face_program(body,mesh,outputs='parent')


def face_reduce_round(states,infinity,tables,*,real,mesh,budget,ordered,odd_moments,
                      keep_budget,retain_span=False,admit=True):
    """One physical parent, all ranks, with no artificial round zero tails."""
    if real != 1 or len(tables['own']) != 1:
        raise ValueError('distributed constructor requires one physical parent per full-mesh program')
    side=tables['active'].shape[-1]
    if admit:
        budget.plan(side,phase='reduction')
    args=(jnp.asarray(tables['points']),jnp.asarray(tables['order']),
          jnp.asarray(tables['active']),tuple(s[1] for s in states),
          tuple(s[2] for s in states),tuple(s[3] for s in states),tuple(infinity))
    if ordered:
        result=face_ordered_parent(
            *args,mesh=mesh,odd_moments=odd_moments,keep_budget=keep_budget,
            retain_span=retain_span)
    else:
        program=face_parent_program(mesh,ordered,odd_moments,keep_budget,retain_span,side)
        result=program(*args)
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


def sector_round_schedule(bank,header,meta,config,mesh,partner,*,execution=None):
    """Schedule capacity-resolved local rounds or one full-mesh parent."""
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
    return [([q],1,np.asarray([0],np.int64),'face') for q in range(header['n_q_irr'])]



@lru_cache(maxsize=None)
def cross_parent_program(mesh, side):
    from gw.shared_pole_sectors import _cross_reduce_equations
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1 as gates
    eigh_plan = face_eigh(mesh, side)
    return face_program(partial(_cross_reduce_equations,mm=face_matmul(mesh),eigh=eigh_plan.batched,gates=gates,
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

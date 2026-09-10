"""Bounded q-local execution of the shared-pole pencil owners.

Only narrow Q/WQ/dWQ panels enter this boundary. The common staged movement
puts complete parents on distinct mesh ranks; local dense kernels implement
the same Hermite/Ritz equations as the distributed constructor.
"""
from functools import lru_cache


def pack_parent_panels(states, infinity, counts, infinity_counts, *, mesh_xy, parent_batch):
    """Compact batched tangential panels into their original pencil columns.

    ``states`` contains (s,Q,WQ,dWQ), with s scalar Ry² and complex128
    [b,n,r_a] face panels. ``counts`` is host int [b,A] from spectral cuts;
    ``infinity_counts`` is host int [b]. Only O(bR) offsets/masks cross to
    the device. Panels stay face-tiled through the existing y permutation.
    Return the packed finite/infinity/mask bundle and original (R_f,r_inf)
    extents. No physical or native eigensolve dimension changes.
    """
    import numpy as np
    from runtime.padding import padded_axis
    from jax.sharding import PartitionSpec as P

    def extent(width):
        return padded_axis(int(width), mesh_xy, name="shared_pole_port",
                           specs=((P('x', 'y'), 0), (P('x', 'y'), 1))).carrier
    carriers = [[extent(n) for n in row] for row in counts]
    extents = tuple((sum(row), extent(ni))
                    for row, ni in zip(carriers, infinity_counts))
    rf = max(nf for nf, _ in extents)
    ri = infinity[0].shape[-1]
    widths = [state[1].shape[-1] for state in states]
    offsets = np.cumsum([0, *widths[:-1]])
    order = np.empty((len(counts), sum(widths)), np.int32)
    active = np.zeros((len(counts), rf + ri), bool)
    for q, (row, retained) in enumerate(zip(carriers, counts)):
        selected = [int(start)+i for start, width in zip(offsets, row)
                    for i in range(width)]
        tail = np.ones(sum(widths), bool)
        tail[selected] = False
        order[q] = selected + np.flatnonzero(tail).tolist()
        start = 0
        for width, count in zip(row, retained):
            active[q, start:start+int(count)] = True
            start += width
        active[q, rf:rf+int(infinity_counts[q])] = True
    packed = _parent_panel_packer(mesh_xy, rf, parent_batch)(
        tuple(states), infinity, order, active,
        np.asarray([nf for nf, _ in extents], np.int32))
    return packed, extents


@lru_cache(maxsize=None)
def _parent_panel_packer(mesh_xy, finite_width, parent_batch):
    """One batch pack with dynamic column offsets and physical supports."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from gw.shared_pole_constructor import _factor_column_permutation

    face = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    scalar = NamedSharding(mesh_xy, P())
    permute = _factor_column_permutation(mesh_xy)

    @jax.jit
    def pack(states, infinity, order, active, finite_counts):
        b = states[0][1].shape[0]
        valid = jnp.arange(finite_width)[None, :] < finite_counts[:, None]
        points = jnp.concatenate([
            jnp.full((b, s[1].shape[-1]), s[0], jnp.complex128)
            for s in states], axis=-1)
        points = jnp.where(valid, jnp.take_along_axis(points, order, axis=-1)[:, :finite_width], 0)
        panels = tuple(jnp.where(valid[:, None, :], permute(
            jnp.concatenate([s[i] for s in states], axis=-1), order)[:, :, :finite_width], 0)
            for i in (1, 2, 3))
        def pad(value, sharding):
            if parent_batch > b:
                value = jnp.concatenate((value, jnp.repeat(value[-1:], parent_batch-b, axis=0)), axis=0)
            return jax.lax.with_sharding_constraint(value, sharding)
        finite = (pad(points, scalar), *(pad(a, face) for a in panels))
        return finite, tuple(pad(a, face) for a in infinity), pad(active, scalar)
    return pack


def parent_pencil_extents(parent_extents, basis_side):
    """Group pencil carriers in basis-size blocks, keeping counts as masks.

    Only inert columns are appended. Each bucket spans one padded spatial
    basis side, so R/basis_side sets the number of programs without a dial.
    All parents retain the batch's existing infinity carrier.
    """
    ri = max(ni for _, ni in parent_extents)
    return tuple((((nf + ri + basis_side - 1) // basis_side) * basis_side - ri, ri)
                 for nf, _ in parent_extents)


@lru_cache(maxsize=None)
def local_parent_reducer(mesh_xy, native_eigh, parent_extents=None):
    """Fuse assembly, corrected Ritz reduction and moment gates per parent.

    The callable consumes the output of ``pack_parent_panels``. All matrix
    inputs are [b,n,r] face tiles. Factors return as [b,n,R] face tiles;
    poles, masks and gate scalars return replicated. Dense pencil work stays
    q-local and never crosses the host. ``parent_extents`` gives each padded
    q row its finite/infinity carrier widths. Native eigensolves use one
    program per basis-size bucket; original physical counts remain masks.
    Local residency is O(R²+nR) per rank
    for one parent at a time; callers admit the actual packed batch first.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from common.shard_map import shard_map
    from common.staged_reshard import face_to_batch_reshard
    from gw.shared_pole_constructor import (
        assemble_shared_pole_pencil, reduce_shared_pole_pencil,
        apply_shared_pole_zero_policy, retained_moment_identity,
    )
    from gw.shared_pole_recipe import shared_real_pole_gates_v1_r3b as gates

    to_batch = face_to_batch_reshard(mesh_xy)
    qspec = P(('x', 'y'))

    def mm(a, b, *, transa='N', transb='N'):
        # Same local dot primitive used by the resolved service's local plan.
        def transpose(value, trans):
            if trans == 'N':
                return value
            value = jnp.swapaxes(value, -1, -2)
            return value.conj() if trans == 'C' else value
        return jnp.matmul(transpose(a, transa), transpose(b, transb))

    def solve(finite, infinity, active):
        # The physics owners retain their ordinary leading batch dimension.
        finite = tuple(a[None] for a in finite)
        infinity = tuple(a[None] for a in infinity)
        pencil = assemble_shared_pole_pencil([finite], infinity, matmul=mm)
        model, reduction, coefficients = reduce_shared_pole_pencil(
            pencil, active[None], eigh=native_eigh, matmul=mm, gates=gates)
        model, zero = apply_shared_pole_zero_policy(model, gates=gates)
        r, ri = pencil[0].shape[-1], infinity[0].shape[-1]
        selector = (jnp.arange(r)[:, None] == jnp.arange(r-ri, r)[None, :])[None].astype(jnp.complex128)
        retained = retained_moment_identity(pencil, coefficients, model, selector, matmul=mm)
        reduction['pencil_side'] = jnp.full((1,), r, jnp.int64)
        return jax.tree.map(lambda a: a[0], (model, reduction, zero, retained))

    def one(args):
        finite, infinity, active, index = args
        if parent_extents is None:
            return solve(finite, infinity, active)
        # Each branch uses one bucket extent. Physical port padding and
        # bucket padding are inactive columns in the same pencil equations.
        rf, ri = finite[1].shape[-1], infinity[0].shape[-1]
        def branch(nf, ni):
            def run(args):
                f, i, mask = args
                f = (f[0][:nf], *(a[:, :nf] for a in f[1:]))
                i = tuple(a[:, :ni] for a in i)
                mask = jnp.concatenate((mask[:nf], mask[rf:rf+ni]))
                model, reduction, zero, retained = solve(f, i, mask)
                c, poles, active = model
                pad = rf + ri - nf - ni
                model = (jnp.pad(c, ((0, 0), (0, pad))),
                         jnp.pad(poles, (0, pad), constant_values=1),
                         jnp.pad(active, (0, pad)))
                reduction['gram_spectrum_relative'] = jnp.pad(
                    reduction['gram_spectrum_relative'], (0, pad))
                return model, reduction, zero, retained
            return run
        # Padded batch rows reuse the same physical-size program while their
        # panels remain distinct runtime inputs. Do not retrace copied rows.
        sizes = tuple(dict.fromkeys(parent_extents))
        branches = tuple(branch(nf, ni) for nf, ni in sizes)
        if len(sizes) != len(parent_extents):
            dispatch = jnp.asarray([sizes.index(size) for size in parent_extents], jnp.int32)
            index = dispatch[index]
        return jax.lax.switch(index, branches, (finite, infinity, active))

    mapped = shard_map(lambda f, i, a, q: jax.lax.map(one, (f, i, a, q)), mesh=mesh_xy,
                       in_specs=((qspec,)*4, (qspec,)*3, qspec, qspec),
                       out_specs=(qspec, qspec, qspec, qspec), check_vma=False)
    # Explicit inverse movement is needed: a generic batch-to-face reshard
    # can rematerialize every parent on every device.
    px, py = int(mesh_xy.shape['x']), int(mesh_xy.shape['y'])
    def restore(c):
        if py > 1:
            c = jax.lax.all_to_all(c, 'y', split_axis=2, concat_axis=0, tiled=True)
        if px > 1:
            c = jax.lax.all_to_all(c, 'x', split_axis=1, concat_axis=0, tiled=True)
        return c
    restore = shard_map(restore, mesh=mesh_xy, in_specs=qspec,
                        out_specs=P(None, 'x', 'y'), check_vma=False)

    @jax.jit
    def execute(finite, infinity, active):
        if parent_extents is not None:
            old_width = finite[1].shape[-1]
            extra = max(nf for nf, _ in parent_extents) - old_width
            finite = (jnp.pad(finite[0], ((0, 0), (0, extra))),
                      *(jnp.pad(a, ((0, 0), (0, 0), (0, extra))) for a in finite[1:]))
            active = jnp.concatenate((jnp.pad(active[:, :old_width], ((0, 0), (0, extra))),
                                      active[:, old_width:]), axis=-1)
        f = (jax.lax.with_sharding_constraint(finite[0], jax.sharding.NamedSharding(mesh_xy, qspec)),
             *(to_batch(a) for a in finite[1:]))
        i = tuple(to_batch(a) for a in infinity)
        active = jax.lax.with_sharding_constraint(active, jax.sharding.NamedSharding(mesh_xy, qspec))
        indices = jax.lax.with_sharding_constraint(jnp.arange(active.shape[0]),
                                                   jax.sharding.NamedSharding(mesh_xy, qspec))
        model, reduction, zero, retained = mapped(f, i, active, indices)
        c, poles, mask = model
        replicated = jax.sharding.NamedSharding(mesh_xy, P())
        scalars = jax.tree.map(lambda a: jax.lax.with_sharding_constraint(a, replicated),
                               (poles, mask, reduction, zero, retained))
        poles, mask, reduction, zero, retained = scalars
        return (restore(c), poles, mask), reduction, zero, retained
    return execute

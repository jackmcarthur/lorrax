"""Bounded q-local execution of the shared-pole pencil owners.

Only narrow Q/WQ/dWQ panels enter this boundary. The common staged movement
puts complete parents on distinct mesh ranks; local dense kernels implement
the same Hermite/Ritz equations as the distributed constructor.
"""
from functools import lru_cache


@lru_cache(maxsize=16)
def plan_local_parent_reducer(mesh_xy, native_eigh, parent_extents, n):
    """Compile abstract packed panels and price the actual per-rank buffers.

    ``n`` is the padded port dimension; extents are finite/infinity column
    counts for every padded parent. No physical panels are allocated here.
    Supports and band-dependent actions remain executable inputs.
    """
    import jax
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from runtime.aot_memory import aot_kernel_peak_bytes

    b = len(parent_extents)
    rf = max(f for f, _ in parent_extents)
    ri = max(i for _, i in parent_extents)
    face = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    scalar = NamedSharding(mesh_xy, P())
    def abstract(shape, sharding, dtype=np.complex128):
        return jax.ShapeDtypeStruct(shape, dtype, sharding=sharding)
    finite = (abstract((b, rf), scalar),
              *(abstract((b, n, rf), face) for _ in range(3)))
    infinity = tuple(abstract((b, n, ri), face) for _ in range(3))
    active = abstract((b, rf + ri), scalar, np.bool_)
    compiled = local_parent_reducer(mesh_xy, native_eigh, parent_extents).lower(
        finite, infinity, active).compile()
    peak = aot_kernel_peak_bytes(compiled)
    return compiled, dict(total_bytes_per_rank=peak.total,
                          compiled_peak_bytes_per_rank=peak.compiled_peak,
                          resident_increment_bytes_per_rank=peak.resident_increment,
                          cufft_measured=peak.cufft_measured)


def pack_parent_panels(parents, *, mesh_xy):
    """Pack ragged finite/infinity panels without changing physical columns.

    Parameters
    ----------
    parents : sequence
        Per-parent ``(states, infinity, active_columns)``. Each state is
        ``(s,Q,WQ,dWQ)`` with scalar Ry² s and complex128 [1,n,r] face
        panels. Infinity is three [1,n,r_inf] face panels. Masks are
        [1,R] replicated booleans, preserving the selected multiplets.
    mesh_xy : Mesh
        Named x/y mesh. The returned batch tiles both axes on its q axis.

    Returns
    -------
    finite, infinity, active : tuple
        Face-tiled packed panels with only trailing zero padding, replicated
        squared support coordinates [b,R_f] and activity masks [b,R]. The
        batch is padded with copies of its last real parent; callers discard
        those diagnostic/output rows. No full response matrix is retained.
    """
    return _parent_panel_packer(mesh_xy)(*parents)


@lru_cache(maxsize=None)
def _parent_panel_packer(mesh_xy):
    """Reuse panel packing by shape; supports and action arrays stay inputs."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from runtime.padding import padded_axis

    face = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    scalar = NamedSharding(mesh_xy, P())

    @jax.jit
    def pack(*items):
        qtag = padded_axis(len(items), mesh_xy, name="shared_pole_parent",
                           specs=((P(('x', 'y')), 0),))
        finite_width = max(sum(s[1].shape[-1] for s in states)
                           for states, _, _ in items)
        infinity_width = max(inf[0].shape[-1] for _, inf, _ in items)
        finite, infinity, masks = [], [], []
        for states, inf, active in items:
            width = sum(s[1].shape[-1] for s in states)
            ri = inf[0].shape[-1]
            points = jnp.concatenate([jnp.full((1, s[1].shape[-1]), s[0], jnp.complex128)
                                      for s in states], axis=-1)
            panels = tuple(jnp.pad(jnp.concatenate([s[i] for s in states], axis=-1),
                                   ((0, 0), (0, 0), (0, finite_width-width)))
                           for i in (1, 2, 3))
            finite.append((jnp.pad(points, ((0, 0), (0, finite_width-width))), *panels))
            infinity.append(tuple(jnp.pad(a, ((0, 0), (0, 0), (0, infinity_width-ri)))
                                  for a in inf))
            masks.append(jnp.concatenate((
                jnp.pad(active[:, :width], ((0, 0), (0, finite_width-width))),
                jnp.pad(active[:, width:], ((0, 0), (0, infinity_width-ri)))), axis=-1))
        def stack(values, sharding):
            value = jnp.concatenate(values, axis=0)
            if qtag.carrier > qtag.logical:
                value = jnp.concatenate((value, jnp.repeat(value[-1:], qtag.carrier-qtag.logical, axis=0)), axis=0)
            return jax.lax.with_sharding_constraint(value, sharding)
        ff = tuple(stack([f[i] for f in finite], scalar if i == 0 else face)
                   for i in range(4))
        ii = tuple(stack([inf[i] for inf in infinity], face) for i in range(3))
        return ff, ii, stack(masks, scalar)
    return pack


@lru_cache(maxsize=None)
def local_parent_reducer(mesh_xy, native_eigh, parent_extents=None):
    """Fuse assembly, corrected Ritz reduction and moment gates per parent.

    The callable consumes the output of ``pack_parent_panels``. All matrix
    inputs are [b,n,r] face tiles. Factors return as [b,n,R] face tiles;
    poles, masks and gate scalars return replicated. Dense pencil work stays
    q-local and never crosses the host. ``parent_extents`` gives each padded
    q row its original finite/infinity widths; native eigensolves use those
    widths, because padding a dense Gram can defeat solver convergence.
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
        # cuSolver's native eigh can fail on large artificial zero blocks.
        # Each branch solves the original selected pencil, including its
        # original port padding. Only compact outputs acquire q-batch padding.
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
        branches = tuple(branch(nf, ni) for nf, ni in parent_extents)
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

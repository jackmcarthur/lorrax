"""Exact nonzero OWN columns, without changing the current solve domain.

This private descriptor retains the actual factor owners. It is selected before
radial replay; discarded *exact zero* columns are the only removed coordinates.
Physical centroid geometry, padding and public operators remain unchanged.
"""
from __future__ import annotations

import hashlib
import numpy as np


def authenticate(basis, zetas):
    from isdf.current_coefficients import _fit_binding, _factor_leaves
    st=zetas[0].store
    if ((int(st.Q),int(st.Q_pad),int(st.mu_pad)) != (
                basis['q_logical'],basis['q_carrier'],basis['physical_mu'])
            or len(basis['factors'])!=len(zetas)
            or tuple(id(z) for z in zetas) != basis['zeta_ids']
            or tuple(_fit_binding(z) for z in zetas) != basis['bindings']
            or any(len(_factor_leaves(z)) != 1 or _factor_leaves(z)[0] is not factor
                   for z, factor in zip(zetas, basis['factors']))
            or tuple(id(v) for v in basis['thin']) != basis['thin_ids']
            or _selection_digest(basis) != basis['selection_digest']):
        raise ValueError('retained current coordinates changed factor/source/q/logical column binding')
    admission=basis.get('admission')
    if admission is not None and admission.get('admitted',False):
        geometry=admission['geometry_binding'];st=zetas[0].store
        if (int(st.g_tile),int(st.g_axis.carrier),int(st.g_axis.logical)) != (
                geometry['g_tile'],geometry['g_carrier'],geometry['g_logical']):
            raise ValueError('retained current store geometry changed after phase admission')
    return basis['thin']


def _selection_digest(basis):
    digest = hashlib.sha256()
    digest.update(repr((basis['logical_mu'], basis['physical_mu'], basis['rows'],
                        basis['q_logical'], basis['q_carrier'])).encode())
    for ids in basis['column_ids']:
        digest.update(np.ascontiguousarray(ids, dtype=np.int32).tobytes())
    return digest.hexdigest()


def row_kernel(mesh, q_logical, *, project=False, donate=False, logical_mu=None):
    """The two factors of C+ apply; thin B has physical zero-embedded rows."""
    import jax
    import jax.numpy as jnp
    from jax import shard_map
    from jax.sharding import PartitionSpec as P
    qs = P(('x', 'y'), None, None)
    if project and (logical_mu is None or int(logical_mu)<1):
        raise ValueError('current projection requires the original logical solve extent')
    def apply(B, values):
        if project:
            B,values=B[:,:logical_mu],values[:,:logical_mu]
        left = jnp.swapaxes(B.conj(), -1, -2) if project else B
        out = left @ values
        rank = jax.lax.axis_index('x') * int(mesh.shape['y']) + jax.lax.axis_index('y')
        live = rank*out.shape[0] + jnp.arange(out.shape[0]) < q_logical
        return jnp.where(live[:, None, None], out, 0)
    return jax.jit(shard_map(apply, mesh=mesh, in_specs=(qs, qs), out_specs=qs,
                            check_vma=False), donate_argnums=(1,) if donate else ())


def operator_kernel(mesh):
    """Stored [mu,G] Gram orientation: conj(B_a) W_ab B_b.T."""
    import jax
    import jax.numpy as jnp
    from jax import shard_map
    from jax.sharding import PartitionSpec as P
    qs = P(('x', 'y'), None, None)
    return jax.jit(shard_map(lambda a, w, b: a.conj() @ w @ jnp.swapaxes(b, -1, -2),
        mesh=mesh, in_specs=(qs, qs, qs), out_specs=qs, check_vma=False))


def select(zetas):
    """Return collectively admitted exact coordinates, or None before replay.

    Only the small column census reaches host; the factors never leave their
    q owners. Selection and thin-factor outputs are AOT-priced before dispatch.
    Later full-stage admission is caller-owned and precedes RHS consumption.
    """
    import jax
    import jax.numpy as jnp
    from jax import shard_map
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import gather_to_host, all_gather_processes, device_put_process_local
    from common.gpu_utils import device_budget_bytes
    from runtime.aot_memory import compiled_new_bytes
    from isdf.current_coefficients import _fit_binding, _factor_leaves
    if len(zetas) != 3 or any(z.solver_kind != 'positive_rank_truncate' for z in zetas):
        return None
    z0 = zetas[0]
    st, mesh = z0.store, z0.mesh
    n, mu, Q, Qp = int(z0.n_rmu_solve), int(st.mu_pad), int(st.Q), int(st.Q_pad)
    if any(z.n_rmu_solve != n or z.mesh is not mesh or z.mu_basis is not z0.mu_basis
           or (z.store.Q, z.store.Q_pad, z.store.mu_pad) != (Q, Qp, mu) for z in zetas):
        raise ValueError('retained current OWN factors require one logical solve domain')
    factors = tuple(_factor_leaves(z)[0] for z in zetas)
    qs = P(('x', 'y'), None, None)
    def census(B):
        logical = B[:, :n, :n]
        rank = jax.lax.axis_index('x')*int(mesh.shape['y'])+jax.lax.axis_index('y')
        live = rank*B.shape[0]+jnp.arange(B.shape[0]) < Q
        selected = jnp.any(logical != 0, axis=1) & live[:, None]
        valid = jnp.all(jnp.isfinite(logical), axis=(1, 2)) | ~live
        return selected, valid
    count = jax.jit(shard_map(census, mesh=mesh, in_specs=qs,
        out_specs=(P(('x', 'y'), None), P(('x', 'y'))), check_vma=False))
    def live_bytes():
        stats = [device.memory_stats() or {} for device in mesh.local_devices]
        return (all('bytes_in_use' in s for s in stats),
                max((s.get('bytes_in_use', 0) for s in stats), default=0))
    def admitted(extra):
        measured, live = live_bytes()
        rows = np.asarray(all_gather_processes(np.array(
            [int(measured), live+extra, device_budget_bytes()], np.float64)))
        return bool(np.all(rows[:, 0] == 1) and np.all(rows[:, 1] <= rows[:, 2]))
    census_extra = max(float(compiled_new_bytes(count.lower(B).compile())) for B in factors)
    if not admitted(census_extra):
        return None
    masks = []
    for B in factors:
        mask, valid = count(B)
        if not np.all(gather_to_host(valid)[:Q]):
            raise ValueError('retained current logical factor contains nonfinite values')
        masks.append(np.asarray(gather_to_host(mask), bool))
    maximum = max(int(mask[:Q].sum(axis=1).max()) for mask in masks)
    R = ((maximum + mesh.size-1)//mesh.size)*mesh.size
    if R == 0 or R >= mu:
        return None
    indices = []
    for mask in masks:
        ids = np.full((Qp, R), -1, np.int32)
        for q in range(Q):
            columns = np.flatnonzero(mask[q])
            ids[q, :len(columns)] = columns
        ids.setflags(write=False)
        indices.append(ids)
    def thin(B, ids):
        selected = jnp.take_along_axis(B[:, :n, :n],
            jnp.maximum(ids, 0)[:, None, :], axis=2)
        selected = jnp.where(ids[:, None, :] >= 0, selected, 0)
        return jnp.pad(selected, ((0, 0), (0, mu-n), (0, 0)))
    take = jax.jit(shard_map(thin, mesh=mesh, in_specs=(qs, P(('x', 'y'), None)),
        out_specs=qs, check_vma=False))
    abstract_ids = jax.ShapeDtypeStruct((Qp, R), jnp.int32,
        sharding=NamedSharding(mesh, P(('x', 'y'), None)))
    increment = max(float(compiled_new_bytes(take.lower(B, abstract_ids).compile())) for B in factors)
    one = 16*Qp*mu*R/mesh.size
    if not admitted(2*one+increment+4*Qp*R/mesh.size):
        return None
    leaves = []
    for B, ids in zip(factors, indices):
        ids_dev = device_put_process_local(ids, abstract_ids.sharding)
        value = take(B, ids_dev)
        value.block_until_ready()
        leaves.append(value)
    basis = dict(zeta_ids=tuple(id(z) for z in zetas), factors=factors,
        bindings=tuple(_fit_binding(z) for z in zetas), thin=tuple(leaves),
        thin_ids=tuple(id(v) for v in leaves), column_ids=tuple(indices),
        logical_mu=n, physical_mu=mu, rows=R, q_logical=Q, q_carrier=Qp)
    basis['selection_digest'] = _selection_digest(basis)
    return basis

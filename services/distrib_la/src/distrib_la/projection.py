"""Band projection with bounded factor panels and a stationary operator."""
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from distrib_la._shard_map import shard_map
from distrib_la.matmul_plan import (
    _as_extent, _axis_matmul, _check_operand, _mesh_shape, _validate_dtype,
    gemm_plan,
)


@dataclass(frozen=True)
class BandProjectionPlan:
    """Compute A @ B on two-axis faces without communicating A.

    Only the band projection product is exposed: no alpha/beta accumulation,
    active-band selection, or implicit input redistribution. ``panel_columns``
    bounds the gathered column count (at least one column per mesh row).
    """
    m: int
    k: int
    n: int
    nq: int
    dtype: object
    in_sharding_a: NamedSharding
    in_sharding_b: NamedSharding
    out_sharding: NamedSharding
    panel_columns: int
    _run: Callable

    def __call__(self, a, b):
        _check_operand(self, 'A', a, (self.nq, self.m, self.k), self.in_sharding_a)
        _check_operand(self, 'B', b, (self.nq, self.k, self.n), self.in_sharding_b)
        return self._run(a, b)


def band_projection_plan(mesh, *, m, k, n, nq, dtype,
                         layout='face', reduction_axis='y', panel_columns=64):
    """Plan a band projection for distributed or replicated band carriers.

    ``reduction_axis`` controls axis-layout placement only. Face layout
    always reduces its K_Y partial products over Y on a square mesh.

    Face inputs A[M_X,K_Y], B[K_X,N_Y] keep A stationary. For each output
    band chunk, exchange B's small panel across the transpose of the logical
    rank grid, gather its columns over X, multiply locally, and reduce-scatter
    over Y. All k rows participate in each collective together.

    The default gathers at most 64 columns, except meshes wider than 64 where
    one column per row is required. Scratch grows linearly in M and K, not
    as M*K. Tail chunks overlap rather than pad or multiply unused zeros;
    beta accumulation is intentionally absent so repeated writes are safe.
    """
    if layout == 'axis':
        return gemm_plan(mesh, m=m, k=k, n=n, nq=nq, dtype=dtype,
                         layout=layout, reduction_axis=reduction_axis)
    if layout != 'face' or reduction_axis not in ('x', 'y'):
        raise ValueError('band_projection_plan requires face/axis layout and x/y reduction')
    m, k, n, nq, limit = (_as_extent(label, value) for label, value in
        (('m', m), ('k', k), ('n', n), ('nq', nq), ('panel_columns', panel_columns)))
    dtype = jnp.dtype(dtype)
    _validate_dtype(dtype)
    px, py = _mesh_shape(mesh)
    if px != py or any(size % px for size in (m, k, n)):
        raise ValueError('band projection faces require a square mesh and divisible extents')
    spec = P(None, 'x', 'y')
    sharding = NamedSharding(mesh, spec)
    nlocal = n // py
    max_width = max(1, limit // px)
    chunks = (nlocal + max_width - 1) // max_width
    width = (nlocal + chunks - 1) // chunks
    perm = tuple((x * py + y, y * px + x) for x in range(px) for y in range(py))

    def local(a, b):
        def body(index, out):
            start = jnp.minimum(index * width, nlocal - width)
            panel = jax.lax.dynamic_slice_in_dim(b, start, width, axis=2)
            panel = jax.lax.ppermute(panel, ('x', 'y'), perm)
            panel = jax.lax.all_gather(panel, 'x', axis=2, tiled=True)
            value = _axis_matmul(a, panel, alpha=1., beta=0., reduction_axis='y')
            return jax.lax.dynamic_update_slice_in_dim(out, value, start, axis=2)
        return jax.lax.fori_loop(0, chunks, body,
            jnp.zeros((nq, m // px, nlocal), dtype=dtype), unroll=1)

    run = jax.jit(shard_map(local, mesh=mesh, in_specs=(spec, spec),
                           out_specs=spec, check_vma=False))
    return BandProjectionPlan(m, k, n, nq, dtype, sharding, sharding,
                              sharding, px * width, run)

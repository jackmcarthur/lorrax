"""Native face GEMM with bounded broadcasts and a shared-left sample axis."""
from functools import lru_cache, partial
from math import gcd

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import NamedSharding, PartitionSpec as P

from ._shard_map import shard_map


def panel_matmul(a, b, *, mesh, panel_bytes):
    """Multiply face matrices, broadcasting one contraction panel at a time.

    Parameters
    ----------
    a : jax.Array
        Complex/real [q,m,k], sharded P(None,'x','y').
    b : jax.Array
        [q,k,n] with the same face layout, or [q,s,k,n] at
        P(None,None,'x','y'). In the latter case a is shared across s;
        its panel is broadcast once outside the sample loop.
    mesh : jax.sharding.Mesh
        Named x/y processor axes. Matrix extents are already mesh-padded.
    panel_bytes : int
        Caller-admitted bytes per rank for the two live operand panels.
        Output and input faces are accounted for separately by the caller.

    Returns
    -------
    jax.Array
        a @ b, with b's batch/sample axes and an x/y output face. Units
        multiply without any normalization. No full row/column is gathered.
    """
    px, py = int(mesh.shape['x']), int(mesh.shape['y'])
    if a.ndim != 3 or b.ndim not in (3, 4) or a.dtype != b.dtype:
        raise ValueError('panel_matmul requires rank-3 A and rank-3/4 B of one dtype')
    q, m, k = a.shape
    if b.shape[0] != q or b.shape[-2] != k:
        raise ValueError('panel_matmul batch/contraction extents disagree')
    n = b.shape[-1]
    if m % px or k % px or k % py or n % py:
        raise ValueError('panel_matmul requires producer-padded face extents')
    per_column = a.dtype.itemsize * q * (m // px + n // py)
    limit = int(panel_bytes) // per_column
    if limit < 1:
        raise MemoryError('panel_matmul panel budget cannot hold one contraction column')
    common = gcd(k // px, k // py)
    width = min(common, limit)
    while common % width:
        width -= 1
    sample_axis = b.ndim == 4
    return _kernel(mesh, q, m, k, n, width, sample_axis)(a, b)


@lru_cache(maxsize=64)
def _kernel(mesh, q, m, k, n, width, sample_axis):
    px, py = int(mesh.shape['x']), int(mesh.shape['y'])
    mx, ny, kx, ky = m // px, n // py, k // px, k // py
    spec = P(None, None, 'x', 'y') if sample_axis else P(None, 'x', 'y')

    @partial(shard_map, mesh=mesh,
             in_specs=(P(None, 'x', 'y'), spec), out_specs=spec,
             check_vma=False)
    def product(a, b):
        if not sample_axis:
            b = b[:, None]
        ns = b.shape[1]
        x, y = lax.axis_index('x'), lax.axis_index('y')

        def panel(c, ip):
            start = ip * width
            left = lax.dynamic_slice(a, (0, 0, start % ky), (q, mx, width))
            left = lax.psum(jnp.where(y == start // ky, left, 0), 'y')

            def sample(c, isample):
                right = lax.dynamic_slice(b, (0, isample, start % kx, 0),
                                          (q, 1, width, ny))[:, 0]
                right = lax.psum(jnp.where(x == start // kx, right, 0), 'x')
                old = lax.dynamic_slice(c, (0, isample, 0, 0), (q, 1, mx, ny))
                value = old + (left @ right)[:, None]
                return lax.dynamic_update_slice(c, value, (0, isample, 0, 0)), None

            c, _ = lax.scan(sample, c, jnp.arange(ns), unroll=1)
            return c, None

        c = jnp.zeros((q, ns, mx, ny), a.dtype)
        c, _ = lax.scan(panel, c, jnp.arange(k // width), unroll=1)
        return c if sample_axis else c[:, 0]

    face_a = NamedSharding(mesh, P(None, 'x', 'y'))
    face_b = NamedSharding(mesh, spec)
    return jax.jit(product, in_shardings=(face_a, face_b), out_shardings=face_b)

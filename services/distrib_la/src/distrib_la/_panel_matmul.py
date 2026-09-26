"""Native face GEMM with bounded band panels and a shared-left sample axis."""
from functools import lru_cache, partial
from math import gcd

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import NamedSharding, PartitionSpec as P

from ._shard_map import shard_map
from .resolve import mesh_platform


def panel_matmul(a, b, *, mesh, panel_bytes, bounds=None):
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
        A complete contraction panel is exchanged once when it fits this
        budget; otherwise smaller band panels are streamed. Output and
        input faces are accounted for separately by the caller.
    bounds : jax.Array, optional
        Integer (q,2), replicated: per batch row, the half-open contraction
        interval [lo, hi) outside which the caller has already zeroed a (or
        b).  When the complete panel is gathered, the local product runs
        only over that interval (the local active-range GEMM), so the
        dropped columns cost no flops; the result is the same product.  The
        streamed paths contract every column (the zeros make that exact).

    Returns
    -------
    jax.Array
        a @ b, with b's batch/sample axes and an x/y output face. Units
        multiply without any normalization. The output always stays x/y
        tiled; exchanged input panels never exceed ``panel_bytes`` per rank.
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
    sample_axis = b.ndim == 4
    if not sample_axis and limit >= k:
        # One bounded all-gather per operand gives the local GEMM its full K.
        # This avoids p tiny-K GEMMs when the complete panel is already small.
        # The gathered panels hold K in global order, so the caller's band
        # interval selects the same columns of both.
        if bounds is not None:
            bounds = jnp.asarray(bounds, jnp.int32).reshape(q, 2)
            return _kernel(mesh, q, m, k, n, k, False, True)(a, b, bounds)
        width = k
    elif not sample_axis and px == py:
        # Interleaved chunks: every rank contributes `width` of its own K
        # columns to each chunk, so a chunk is ONE all-gather per operand over
        # the whole mesh axis (p times the columns of an owner panel).  Two
        # chunks are live (the one multiplied and the one prefetched).
        per_chunk = limit // (2 * px)
        width = max(d for d in range(1, k // px + 1)
                    if (k // px) % d == 0 and d <= max(per_chunk, 1))
        return _interleaved_kernel(mesh, q, m, k, n, width)(a, b)
    else:
        common = gcd(k // px, k // py)
        width = min(common, limit)
        while common % width:
            width -= 1
    return _kernel(mesh, q, m, k, n, width, sample_axis)(a, b)


def _local_interval_product(mesh):
    """The local active-range GEMM of this mesh's platform: ``(left, right, bounds) -> left @ right`` over each row's interval."""
    one, zero = np.complex128(1.0), np.complex128(0.0)
    if mesh_platform(mesh) == "CUDA":
        from ._active_local_cuda import active_local_cuda, require_active_local_cuda
        require_active_local_cuda()
        contract = active_local_cuda
    else:
        from ._active_local import active_local_matmul as contract

    def product(left, right, bounds):
        weights = jnp.ones(left.shape[::2], left.dtype)
        return contract(left, right, bounds, weights, alpha=one, beta=zero)
    return product


@lru_cache(maxsize=64)
def _kernel(mesh, q, m, k, n, width, sample_axis, active=False):
    px, py = int(mesh.shape['x']), int(mesh.shape['y'])
    mx, ny, kx, ky = m // px, n // py, k // px, k // py
    spec = P(None, None, 'x', 'y') if sample_axis else P(None, 'x', 'y')
    face_a = NamedSharding(mesh, P(None, 'x', 'y'))
    if active:
        interval_product = _local_interval_product(mesh)

        @partial(shard_map, mesh=mesh,
                 in_specs=(P(None, 'x', 'y'), spec, P()), out_specs=spec,
                 check_vma=False)
        def gathered(a, b, bounds):
            left = lax.all_gather(a, 'y', axis=2, tiled=True)
            right = lax.all_gather(b, 'x', axis=1, tiled=True)
            return interval_product(left, right, bounds)

        return jax.jit(gathered, in_shardings=(face_a, face_a, NamedSharding(mesh, P())),
                       out_shardings=face_a)

    @partial(shard_map, mesh=mesh,
             in_specs=(P(None, 'x', 'y'), spec), out_specs=spec,
             check_vma=False)
    def product(a, b):
        if width == k and not sample_axis:
            left = lax.all_gather(a, 'y', axis=2, tiled=True)
            right = lax.all_gather(b, 'x', axis=1, tiled=True)
            return left @ right
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

    face_b = NamedSharding(mesh, spec)
    return jax.jit(product, in_shardings=(face_a, face_b), out_shardings=face_b)


@lru_cache(maxsize=64)
def _interleaved_kernel(mesh, q, m, k, n, width):
    """Stream K in interleaved chunks on a square mesh, prefetching the next.

    Rank ``(x, y)`` holds the K block ``[y·K/p, (y+1)·K/p)`` of A and
    ``[x·K/p, (x+1)·K/p)`` of B.  Chunk ``j`` takes local columns
    ``[j·w, (j+1)·w)`` of every block: the all-gather of A over ``y`` and of
    B over ``x`` then both hold the SAME global K set
    ``{i·K/p + j·w + t}``, in the same order, so the local product of the two
    gathered panels is that chunk's exact contribution to the rank's own
    output tile.  No reduction follows.  Chunk ``j+1`` is gathered before
    chunk ``j`` is multiplied, so the collective overlaps the GEMM.
    """
    p = int(mesh.shape['x'])
    mx, ny, kl = m // p, n // p, k // p
    n_chunk = kl // width

    @partial(shard_map, mesh=mesh, in_specs=(P(None, 'x', 'y'),) * 2,
             out_specs=P(None, 'x', 'y'), check_vma=False)
    def product(a, b):
        def gather(j):
            left = lax.dynamic_slice_in_dim(a, j * width, width, axis=2)
            right = lax.dynamic_slice_in_dim(b, j * width, width, axis=1)
            return (lax.all_gather(left, 'y', axis=2, tiled=True),
                    lax.all_gather(right, 'x', axis=1, tiled=True))

        def step(carry, j):
            c, (left, right) = carry
            ahead = gather(j + 1)
            return (c + left @ right, ahead), None

        c = jnp.zeros((q, mx, ny), a.dtype)
        (c, (left, right)), _ = lax.scan(
            step, (c, gather(0)), jnp.arange(n_chunk - 1), unroll=1)
        return c + left @ right

    face = NamedSharding(mesh, P(None, 'x', 'y'))
    return jax.jit(product, in_shardings=(face, face), out_shardings=face)

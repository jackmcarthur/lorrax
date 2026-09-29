"""Native 2-D face GEMM: batched SUMMA over bounded band panels, and a shared-left sample axis."""
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
    """Multiply face matrices by a batched 2-D SUMMA over bounded contraction panels.

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
    bounds : jax.Array, optional
        Integer (q,2), replicated: per batch row, the half-open contraction
        interval [lo, hi) outside which the caller has already zeroed a (or
        b).  On a square mesh each panel's local product runs only over the
        interval's columns in that panel (the local active-range GEMM), so
        the dropped columns cost no flops; the result is the same product.
        The other streams contract every column (the zeros make that exact).

    Returns
    -------
    jax.Array
        a @ b, with b's batch/sample axes and an x/y output face. Units
        multiply without any normalization. The output always stays x/y
        tiled; no rank ever holds a band-complete panel (on a p x p mesh a
        panel spans at most K/p contraction columns), and exchanged panels
        never exceed ``panel_bytes`` per rank.  Every batch row rides in each
        panel exchange and each local GEMM: one collective per panel, not per q.
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
    if not sample_axis and px == py:
        # SUMMA on interleaved panels: every rank contributes `width` of its
        # own K columns to each panel, so a panel is ONE all-gather per operand
        # over the mesh axis (p·width columns, at most one owner block K/p).
        # Two panels are live (the one multiplied and the one prefetched).
        kl = k // px
        cap = max(1, min(limit // (2 * px), kl // px if px > 1 else kl))
        width = max(d for d in range(1, kl + 1) if kl % d == 0 and d <= cap)
        if bounds is None:
            return _interleaved_kernel(mesh, q, m, k, n, width)(a, b)
        bounds = jnp.asarray(bounds, jnp.int32).reshape(q, 2)
        return _interleaved_kernel(mesh, q, m, k, n, width, True)(a, b, bounds)
    common = gcd(k // px, k // py)
    width = min(common, limit)
    while common % width:
        width -= 1
    return _kernel(mesh, q, m, k, n, width, sample_axis)(a, b)


def _panel_contraction(mesh):
    """``(left, right, bounds, c)``: ``c + left[:, :, lo:hi] @ right[:, lo:hi]`` per row
    (``c=None``: a fresh product), the local active-range GEMM of this mesh's platform.
    The bounds are valid by construction (``_interleaved_kernel``), so no guard runs."""
    one = np.complex128(1.0)
    if mesh_platform(mesh) == "CUDA":
        from ._active_local_cuda import _native, _native_out, require_active_local_cuda
        require_active_local_cuda()

        def contract(left, right, bounds, c):
            if c is None:
                return _native_out(left, right, bounds, alpha=one)
            return _native(left, right, bounds, c, alpha=one, beta=one)
    else:
        from ._active_local import active_local_matmul

        def contract(left, right, bounds, c):
            weights = jnp.ones(left.shape[::2], left.dtype)
            return active_local_matmul(left, right, bounds, weights, c, alpha=one,
                                       beta=np.complex128(0.0) if c is None else one)
    return contract


@lru_cache(maxsize=64)
def _kernel(mesh, q, m, k, n, width, sample_axis):
    px, py = int(mesh.shape['x']), int(mesh.shape['y'])
    mx, ny, kx, ky = m // px, n // py, k // px, k // py
    spec = P(None, None, 'x', 'y') if sample_axis else P(None, 'x', 'y')
    face_a = NamedSharding(mesh, P(None, 'x', 'y'))
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

    face_b = NamedSharding(mesh, spec)
    return jax.jit(product, in_shardings=(face_a, face_b), out_shardings=face_b)


@lru_cache(maxsize=64)
def _interleaved_kernel(mesh, q, m, k, n, width, active=False, depth=1):
    """Batched SUMMA on a square mesh: K streamed in interleaved panels, ``depth`` prefetched.

    Rank ``(x, y)`` holds the K block ``[y·K/p, (y+1)·K/p)`` of A and
    ``[x·K/p, (x+1)·K/p)`` of B.  Panel ``j`` takes local columns
    ``[j·w, (j+1)·w)`` of every block: the all-gather of A over ``y`` and of
    B over ``x`` then both hold the SAME global K set
    ``{i·K/p + j·w + t}``, in the same order, so the local product of the two
    gathered panels is that panel's exact contribution to the rank's own
    output tile.  No reduction follows.  Panels ``j+1 … j+depth`` are gathered
    before panel ``j`` is multiplied, so the collectives overlap the GEMM.
    ``active``: each row's interval ``[lo, hi)`` meets a panel in ONE run of
    panel positions (a suffix of the first live owner's segment, whole
    segments, a prefix of the last), so the local active-range GEMM contracts
    only that run.
    """
    p = int(mesh.shape['x'])
    mx, ny, kl = m // p, n // p, k // p
    n_chunk = kl // width
    depth = max(1, min(int(depth), n_chunk - 1))
    contract = _panel_contraction(mesh) if active else None
    owner = np.arange(p, dtype=np.int32)[None, :]

    def body(a, b, bounds):
        def gather(j):
            left = lax.dynamic_slice_in_dim(a, j * width, width, axis=2)
            right = lax.dynamic_slice_in_dim(b, j * width, width, axis=1)
            return (lax.all_gather(left, 'y', axis=2, tiled=True),
                    lax.all_gather(right, 'x', axis=1, tiled=True))

        def product(c, panel, j):
            left, right = panel
            if not active:
                return left @ right if c is None else c + left @ right
            base = owner * kl + j * width
            start = owner * width + jnp.clip(bounds[:, :1] - base, 0, width)
            stop = owner * width + jnp.clip(bounds[:, 1:] - base, 0, width)
            live = stop > start
            hi = jnp.max(jnp.where(live, stop, 0), axis=1)
            lo = jnp.minimum(jnp.min(jnp.where(live, start, p * width), axis=1), hi)
            return contract(left, right, jnp.stack([lo, hi], axis=1).astype(jnp.int32), c)

        if n_chunk == 1:
            return product(None, gather(0), 0)
        first = gather(0)
        ahead = tuple(gather(j) for j in range(1, depth + 1))
        c = product(None, first, 0)

        def step(carry, j):
            c, ahead = carry
            return (product(c, ahead[0], j), ahead[1:] + (gather(j + depth),)), None

        (c, ahead), _ = lax.scan(step, (c, ahead), jnp.arange(1, n_chunk - depth), unroll=1)
        for i, panel in enumerate(ahead):
            c = product(c, panel, n_chunk - depth + i)
        return c

    face = NamedSharding(mesh, P(None, 'x', 'y'))
    if active:
        kernel = shard_map(body, mesh=mesh, in_specs=(P(None, 'x', 'y'),) * 2 + (P(),),
                           out_specs=P(None, 'x', 'y'), check_vma=False)
        return jax.jit(kernel, in_shardings=(face, face, NamedSharding(mesh, P())),
                       out_shardings=face)
    kernel = shard_map(lambda a, b: body(a, b, None), mesh=mesh,
                       in_specs=(P(None, 'x', 'y'),) * 2, out_specs=P(None, 'x', 'y'),
                       check_vma=False)
    return jax.jit(kernel, in_shardings=(face, face), out_shardings=face)

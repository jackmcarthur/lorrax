"""Placement-independent reductions over mesh axes, for use inside ``shard_map``.

A collective library's all-reduce and reduce-scatter add the P per-rank terms in
an order its rings or trees take from the physical topology of the devices the
job was given, so the same job on another device set, or with ranks paired
differently across packages, differs in the last bit (RCCL on MI250X, det lane
2026-10-16: Fe V2 sigma_c 4e-7 eV and eqp print flips from a 1-ulp psi(r_mu)
reduce-scatter amplified by the ISDF fit).  These functions move the terms with
``all_to_all`` / ``all_gather`` (pure data movement, bitwise whatever the route)
and add them on each device in rank order, so the result depends only on the
rank count.  The reduce-scatter puts the same bytes on the wire as a ring
reduce-scatter; the all-reduce adds the all-gather a ring all-reduce also does.

JAX imports stay inside the functions so importing :mod:`lxkit` stays stdlib-only.
"""

from __future__ import annotations

__all__ = ["ordered_psum_scatter", "ordered_psum"]


def _axis_size(axis_name) -> int:
    from jax import lax
    names = axis_name if isinstance(axis_name, (tuple, list)) else (axis_name,)
    n = 1
    for a in names:
        n *= int(lax.axis_size(a))
    return n


def _rank_order_sum(c):
    """``c[0] + c[1] + ... + c[n-1]``, left to right (the order IS the contract)."""
    acc = c[0]
    for j in range(1, int(c.shape[0])):
        acc = acc + c[j]
    return acc


def ordered_psum_scatter(x, axis_name, *, scatter_dimension: int = 0, tiled: bool = False):
    """``lax.psum_scatter`` with the terms added in rank order on every placement.

    Same arguments and result as ``jax.lax.psum_scatter(x, axis_name,
    scatter_dimension=..., tiled=...)``: device ``p`` (row-major index over
    ``axis_name``) gets the sum over ranks of slice ``p`` of ``x`` along
    ``scatter_dimension``.  One ``all_to_all``, then ``n - 1`` adds.
    """
    import jax.numpy as jnp
    from jax import lax
    n = _axis_size(axis_name)
    d = scatter_dimension % x.ndim
    size = int(x.shape[d])
    if tiled:
        if size % n:
            raise ValueError(f"ordered_psum_scatter: dimension {d} of size {size} is not divisible by {n} devices")
        x = x.reshape(x.shape[:d] + (n, size // n) + x.shape[d + 1:])
    elif size != n:
        raise ValueError(f"ordered_psum_scatter: untiled dimension {d} has size {size}; want the axis size {n}")
    x = jnp.moveaxis(x, d, 0)                                          # (n, ...): slice p for device p
    c = lax.all_to_all(x, axis_name, 0, 0, tiled=True)                 # (n, ...): row j = rank j's slice for me
    out = _rank_order_sum(c)
    return out if tiled else out


def ordered_psum(x, axis_name):
    """``lax.psum`` with the terms added in rank order on every placement: an ordered
    reduce-scatter of the flattened (zero-padded) operand, then an ``all_gather``."""
    import jax.numpy as jnp
    from jax import lax
    n = _axis_size(axis_name)
    if n == 1:
        return x
    shape, flat = x.shape, x.reshape(-1)
    pad = (-flat.shape[0]) % n
    if pad:
        flat = jnp.concatenate([flat, jnp.zeros((pad,), flat.dtype)])
    part = ordered_psum_scatter(flat, axis_name, scatter_dimension=0, tiled=True)
    full = lax.all_gather(part, axis_name, tiled=True)
    return (full[:flat.shape[0] - pad] if pad else full).reshape(shape)

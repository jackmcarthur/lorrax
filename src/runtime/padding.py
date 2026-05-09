"""Shape-padding helpers for sharded arrays.

Single source of truth for the n_rmu / n_band / etc. padding contract:
arrays in memory may be padded to mesh-divisibility (so a JIT boundary
with a product- or single-axis sharding spec doesn't trip
``ValueError: should be divisible by N``), but files on disk store the
logical (unpadded) extent so they can be re-read on any process count.

Three layers, in order of how often callers reach for them:

* :func:`pad_shape_to_mesh` — pure shape arithmetic, no array.  Use
  when you only need to know "what shape would I pad this sharding to?"
  for things like a SlabIO ``shape=`` parameter.
* :class:`PadAxis` + :func:`pad_array_to_mesh` /
  :func:`unpad_array_from_mesh` — array-level helpers with metadata.
  ``pad_array_to_mesh`` zero-pads sharded axes to mesh-product
  divisibility and returns ``(padded_array, pad_meta)``.  ``pad_meta``
  is a tuple of :class:`PadAxis` records — pass it to
  :func:`valid_shape_from_pad_meta` for the SlabIO ``valid_shape=``
  argument.
* :func:`round_up_to_mesh_product` — round n up to ``∏ p_a`` over the
  mesh's axis sizes.  Use for padding decisions on a single integer
  (e.g. ``Meta.n_rmu_padded``) when the call site doesn't yet have
  arrays / shardings in hand.

The pad zone is always zero-filled.  Downstream operators that contract
along a padded axis (e.g. einsums in V_q tile, V·χ in W solve) see no
contribution from the pad rows by construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


# ---------------------------------------------------------------------------
# PadAxis — per-axis padding record that travels alongside a padded array
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PadAxis:
    """Per-axis padding metadata for a sharded array.

    Travels alongside arrays whose extent on ``axis`` was rounded up so
    the shard layout on ``mesh_axes`` divides cleanly.  Use ``logical``
    to recover the unpadded extent (e.g. for ``SlabIO.write_slab``'s
    ``valid_shape=`` argument), and ``padded`` for in-memory shapes.

    Attributes
    ----------
    axis : int
        Position of the padded dim (0-based).
    logical : int
        Original extent before padding.
    padded : int
        Extent after padding.  Always ≥ ``logical``.
    mesh_axes : tuple[str, ...]
        The mesh axes that shard this dim — single-axis (``('x',)``) or
        product (``('x', 'y')``).  Determines the divisor used.
    """
    axis: int
    logical: int
    padded: int
    mesh_axes: tuple[str, ...]

    @property
    def pad_size(self) -> int:
        return self.padded - self.logical

    @property
    def is_padded(self) -> bool:
        return self.pad_size > 0


def pad_shape_to_mesh(shape: Sequence[int], partition_spec, mesh) -> tuple[int, ...]:
    """Round each dim up so the resulting shape divides cleanly into a
    sharded layout on ``mesh`` with the given ``partition_spec``.

    A dim sharded by mesh axis ``a`` (or by a tuple ``(a, b, ...)``)
    must be divisible by ``mesh.shape[a]`` (or the product of mesh
    sizes).  Unsharded dims pass through unchanged.

    Logical → physical example, mesh ``{'x': 4, 'y': 4}``,
    spec ``P(None, None, ('x','y'))``, shape ``(9, 60, 668)``:
    ``668 % 16 = 12`` → 668 → 672.  Returns ``(9, 60, 672)``.

    Used by drivers to know how much zero-padding they need before a
    JIT call whose interior demands product-divisible sharding.
    """
    physical = [int(s) for s in shape]
    for d, axis in enumerate(partition_spec):
        if axis is None:
            continue
        axes = (axis,) if isinstance(axis, str) else tuple(axis)
        prod = 1
        for ax in axes:
            prod *= int(mesh.shape[ax])
        if prod <= 1:
            continue
        rem = physical[d] % prod
        if rem:
            physical[d] += (prod - rem)
    return tuple(physical)


def logical_shape_from_padded(padded: Sequence[int], logical: Sequence[int]) -> tuple[int, ...]:
    """Trivial validator: logical[d] <= padded[d] for every axis.

    Returns ``logical`` unchanged after the assertion.  Mainly a
    documentation hook so callers tagging padding metadata write the
    same dataflow direction every time.
    """
    p = tuple(int(s) for s in padded)
    l = tuple(int(s) for s in logical)
    if len(p) != len(l):
        raise ValueError(
            f"rank mismatch: padded={p} (rank {len(p)}) vs logical={l} (rank {len(l)})"
        )
    for d, (pd, ld) in enumerate(zip(p, l)):
        if ld > pd:
            raise ValueError(
                f"axis {d}: logical {ld} > padded {pd}"
            )
    return l


def round_up_to_mesh_product(n: int, mesh) -> int:
    """Round ``n`` up to ``∏ p_a`` over all mesh axis sizes.

    Use for padding a single integer (centroid count, band count, ...)
    when the call site doesn't yet have array shardings in hand.  The
    mesh-product is the worst-case divisor — any single-axis or
    product-axis ``PartitionSpec`` on this dim will divide it cleanly.

    Returns ``n`` unchanged when already divisible.
    """
    prod = 1
    for axis in mesh.axis_names:
        prod *= int(mesh.shape[axis])
    if prod <= 1:
        return int(n)
    rem = int(n) % prod
    return int(n) + ((prod - rem) if rem else 0)


# ---------------------------------------------------------------------------
# Array-level helpers (jit-safe; no host transfers)
# ---------------------------------------------------------------------------


def _spec_axes(spec_entry):
    """Normalize a PartitionSpec entry to a tuple of mesh axis names or None."""
    if spec_entry is None:
        return None
    if isinstance(spec_entry, str):
        return (spec_entry,)
    return tuple(spec_entry)


def _spec_divisor(spec_entry, mesh) -> int:
    """Effective divisor for a PartitionSpec entry on ``mesh`` (1 for None)."""
    axes = _spec_axes(spec_entry)
    if axes is None:
        return 1
    prod = 1
    for a in axes:
        prod *= int(mesh.shape[a])
    return prod


def pad_array_to_mesh(arr, partition_spec, mesh):
    """Pad sharded axes up to mesh divisibility and apply the sharding constraint.

    Each axis whose ``partition_spec`` entry is non-trivial is rounded up
    so the local shard size is integer.  Pad entries are zero
    (``jnp.pad`` with default ``constant_values=0``).  After padding the
    array is constrained to ``NamedSharding(mesh, partition_spec)`` via
    ``jax.lax.with_sharding_constraint``.

    Parameters
    ----------
    arr : jax.Array | numpy.ndarray
    partition_spec : jax.sharding.PartitionSpec
    mesh : jax.sharding.Mesh

    Returns
    -------
    padded_array : jax.Array
        Same dtype as ``arr``; shape >= ``arr.shape`` element-wise.
        Sharded ``NamedSharding(mesh, partition_spec)``.
    pad_meta : tuple[PadAxis, ...]
        One :class:`PadAxis` per axis that received non-zero padding,
        in axis order.  Empty tuple when no axis needed padding.

    Notes
    -----
    Pure ``jnp`` ops; safe to call inside a ``jax.jit`` boundary as well
    as at the top level.  At the top level the unsharded input does not
    error on uneven extents — sharding is only applied via WSC after the
    pad lands.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding

    arr_shape = tuple(int(s) for s in arr.shape)
    if len(partition_spec) > len(arr_shape):
        raise ValueError(
            f"partition_spec rank {len(partition_spec)} > array rank {len(arr_shape)}")

    pad_widths = [(0, 0)] * len(arr_shape)
    pad_meta: list[PadAxis] = []

    for axis_idx, spec_entry in enumerate(partition_spec):
        divisor = _spec_divisor(spec_entry, mesh)
        if divisor <= 1:
            continue
        logical = arr_shape[axis_idx]
        rem = logical % divisor
        if rem == 0:
            continue
        padded = logical + (divisor - rem)
        pad_widths[axis_idx] = (0, padded - logical)
        pad_meta.append(PadAxis(
            axis=axis_idx, logical=logical, padded=padded,
            mesh_axes=_spec_axes(spec_entry),
        ))

    if any(after for _, after in pad_widths):
        out = jnp.pad(arr, pad_widths)
    else:
        out = arr

    out = jax.lax.with_sharding_constraint(out, NamedSharding(mesh, partition_spec))
    return out, tuple(pad_meta)


def unpad_array_from_mesh(arr_padded, pad_meta, *,
                          target_partition_spec=None, mesh=None):
    """Slice an array back to its logical (unpadded) extent.

    ``pad_meta`` is the tuple returned by :func:`pad_array_to_mesh`.
    Untouched axes pass through unchanged.  When
    ``target_partition_spec`` is given the result is constrained to
    ``NamedSharding(mesh, target_partition_spec)`` via WSC — useful when
    the unpadded layout naturally drops to single-axis (e.g. ``P(None,
    'x', None)``) instead of the padded product layout.

    Parameters
    ----------
    arr_padded : jax.Array
    pad_meta : tuple[PadAxis, ...]
    target_partition_spec : PartitionSpec | None
    mesh : Mesh | None
        Required when ``target_partition_spec`` is given.

    Returns
    -------
    arr : jax.Array
        Logical-shaped, optionally resharded to ``target_partition_spec``.

    Notes
    -----
    Uses ``jax.lax.slice`` (static start/limit) so the output shape is
    statically known at trace time.  Pure jnp ops; jit-safe.
    """
    import jax
    from jax.sharding import NamedSharding

    if not pad_meta:
        if target_partition_spec is not None:
            if mesh is None:
                raise ValueError(
                    "unpad_array_from_mesh: mesh required when "
                    "target_partition_spec is given")
            return jax.lax.with_sharding_constraint(
                arr_padded, NamedSharding(mesh, target_partition_spec))
        return arr_padded

    starts = [0] * arr_padded.ndim
    limits = list(arr_padded.shape)
    for ax in pad_meta:
        if ax.axis < 0 or ax.axis >= arr_padded.ndim:
            raise ValueError(
                f"PadAxis.axis={ax.axis} out of range for rank {arr_padded.ndim}")
        if ax.logical > arr_padded.shape[ax.axis]:
            raise ValueError(
                f"PadAxis.logical={ax.logical} exceeds padded extent "
                f"{arr_padded.shape[ax.axis]} on axis {ax.axis}")
        limits[ax.axis] = ax.logical

    out = jax.lax.slice(arr_padded, starts, limits)

    if target_partition_spec is not None:
        if mesh is None:
            raise ValueError(
                "unpad_array_from_mesh: mesh required when "
                "target_partition_spec is given")
        out = jax.lax.with_sharding_constraint(
            out, NamedSharding(mesh, target_partition_spec))
    return out


def valid_shape_from_pad_meta(padded_shape: Sequence[int],
                               pad_meta: Sequence[PadAxis]) -> tuple[int, ...]:
    """Build a ``valid_shape=`` tuple for SlabIO from a padded shape +
    PadAxis records.

    Pass the returned tuple directly to ``SlabIO.write_slab(...,
    valid_shape=...)`` to clip the on-disk extent to the logical
    prefix, or to ``SlabIO.read_slab(..., shape=padded, valid_shape=...)``
    to read a logical prefix into a padded buffer.  Untouched axes pass
    through unchanged.
    """
    out = list(int(s) for s in padded_shape)
    for ax in pad_meta:
        if ax.axis < 0 or ax.axis >= len(out):
            raise ValueError(
                f"PadAxis.axis={ax.axis} out of range for rank {len(out)}")
        out[ax.axis] = ax.logical
    return tuple(out)


__all__ = [
    "pad_shape_to_mesh",
    "logical_shape_from_padded",
    "PadAxis",
    "round_up_to_mesh_product",
    "pad_array_to_mesh",
    "unpad_array_from_mesh",
    "valid_shape_from_pad_meta",
]

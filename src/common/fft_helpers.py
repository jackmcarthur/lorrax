import math
from typing import Callable

import jax
import jax.numpy as jnp
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map


def compute_block_size_for_2d_cholesky(n_rmu: int, Pr: int, Pc: int) -> tuple[int, int]:
    """
    Compute block size for 2D blocked Cholesky that satisfies distribution constraints.

    Constraints (fundamental to 2D blocked algorithms):
        - n_rmu % block_size == 0  (matrix divides into whole tiles)
        - J % Pr == 0              (tile rows distribute evenly on X-axis)
        - J % Pc == 0              (tile cols distribute evenly on Y-axis)

    Where J = n_rmu / block_size is the number of tiles per dimension.

    The simplest solution: J = lcm(Pr, Pc), giving block_size = n_rmu / J.
    If n_rmu doesn't divide evenly, we try multiples of lcm(Pr, Pc).

    Args:
        n_rmu: Matrix dimension (number of ISDF centroids)
        Pr: Number of devices on X-axis
        Pc: Number of devices on Y-axis

    Returns:
        (block_size, J) tuple

    Raises:
        ValueError: If no valid block size exists (n_rmu incompatible with mesh)
    """
    target_J = math.lcm(Pr, Pc)

    if n_rmu % target_J == 0:
        block_size = n_rmu // target_J
        return block_size, target_J

    # Try multiples of lcm(Pr, Pc)
    for j_mult in range(2, 20):
        J = target_J * j_mult
        if n_rmu % J == 0:
            block_size = n_rmu // J
            return block_size, J

    # Last resort: find any valid block size
    for b in range(n_rmu, 0, -1):
        if n_rmu % b == 0:
            J = n_rmu // b
            if J % Pr == 0 and J % Pc == 0:
                return b, J

    raise ValueError(
        f"No valid block size for n_rmu={n_rmu} with mesh {Pr}×{Pc}. "
        f"n_rmu should be divisible by lcm({Pr},{Pc})={target_J} or a multiple thereof."
    )


# ============================================================================
# shard_map based FFT - runs FFT independently on each device's local data
# See: https://docs.jax.dev/en/latest/notebooks/shard_map.html
# ============================================================================


def _normalize_local_fft_axes(rank: int, axes: tuple[int, ...]) -> tuple[int, ...]:
    normalized = tuple(ax if ax >= 0 else rank + ax for ax in axes)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"FFT axes must be unique, got {axes}.")
    if any(ax < 0 or ax >= rank for ax in normalized):
        raise ValueError(f"FFT axes {axes} out of bounds for rank-{rank} array.")
    return normalized


def _validate_local_fft_specs(in_spec: P, out_spec: P, axes: tuple[int, ...]) -> tuple[int, ...]:
    in_axes = tuple(in_spec)
    out_axes = tuple(out_spec)
    if len(in_axes) != len(out_axes):
        raise ValueError(
            f"Input/output PartitionSpecs must have the same rank, got {in_spec} and {out_spec}."
        )
    fft_axes = _normalize_local_fft_axes(len(in_axes), axes)
    for ax in fft_axes:
        if in_axes[ax] is not None or out_axes[ax] is not None:
            raise ValueError(
                "Jittable local FFT helpers require every transformed axis to be replicated. "
                f"Axis {ax} is sharded in in_spec={in_spec}, out_spec={out_spec}."
            )
    return fft_axes


def _make_jittable_local_fft(
    mesh: Mesh,
    in_spec: P,
    out_spec: P,
    *,
    fft_kind: str,
    norm: str | None,
    axes: tuple[int, ...],
):
    """Return a jit-compatible FFT that preserves sharding on replicated FFT axes."""

    del mesh  # The active mesh is supplied to the partition callback during tracing.
    fft_axes = _validate_local_fft_specs(in_spec, out_spec, axes)
    if fft_kind not in ("ifftn", "fftn"):
        raise ValueError(f"Unsupported fft_kind={fft_kind!r}")

    def _make_axis_wrapper(axis: int):
        def fft_impl(x):
            n_axis = x.shape[axis]
            if fft_kind == "ifftn":
                raw = jnp.conj(jnp.fft.fft(jnp.conj(x), axis=axis))
                if norm in (None, "backward"):
                    return raw / float(n_axis)
                if norm == "ortho":
                    return raw / jnp.sqrt(float(n_axis))
                if norm == "forward":
                    return raw
            else:
                raw = jnp.fft.fft(x, axis=axis)
                if norm in (None, "backward"):
                    return raw
                if norm == "ortho":
                    return raw / jnp.sqrt(float(n_axis))
                if norm == "forward":
                    return raw / float(n_axis)
            raise ValueError(f"Unsupported FFT norm={norm!r}")

        @custom_partitioning
        def _local_fft_axis(x):
            return fft_impl(x)

        def _partition(mesh_arg: Mesh, arg_shapes, result_shape):
            del arg_shapes, result_shape
            return (
                mesh_arg,
                fft_impl,
                NamedSharding(mesh_arg, out_spec),
                (NamedSharding(mesh_arg, in_spec),),
            )

        def _infer(mesh_arg: Mesh, arg_shapes, result_shape):
            del arg_shapes, result_shape
            return NamedSharding(mesh_arg, out_spec)

        _local_fft_axis.def_partition(
            infer_sharding_from_operands=_infer,
            partition=_partition,
            sharding_rule="...i -> ...i",
        )
        return _local_fft_axis

    axis_wrappers = [_make_axis_wrapper(axis) for axis in fft_axes]

    def _local_fft(x):
        for fft_axis in axis_wrappers:
            x = fft_axis(x)
        return x

    return _local_fft


def make_jittable_local_ifftn_3d(
    mesh: Mesh,
    in_spec: P,
    out_spec: P,
    *,
    norm: str | None = None,
    axes: tuple[int, int, int] = (-3, -2, -1),
):
    """Create a jit-compatible local IFFT over replicated FFT axes."""

    return _make_jittable_local_fft(
        mesh,
        in_spec,
        out_spec,
        fft_kind="ifftn",
        norm=norm,
        axes=axes,
    )


def make_jittable_local_fftn_3d(
    mesh: Mesh,
    in_spec: P,
    out_spec: P,
    *,
    norm: str | None = None,
    axes: tuple[int, int, int] = (-3, -2, -1),
):
    """Create a jit-compatible local FFT over replicated FFT axes."""

    return _make_jittable_local_fft(
        mesh,
        in_spec,
        out_spec,
        fft_kind="fftn",
        norm=norm,
        axes=axes,
    )

def make_sharded_ifftn_3d(
	mesh: Mesh,
	in_spec: P,
	out_spec: P,
	*,
	norm: str | None = None,
	axes: tuple[int, int, int] = (-3, -2, -1),
):
    """
    Uses shard_map to run FFT independently on each device's local data.
    The FFT axes (last 3) must NOT be sharded - only batch dims can be sharded.
    Args:
        mesh: The device mesh
        in_spec: PartitionSpec for input (e.g., P(None, ('x','y'), None, None, None, None))
        out_spec: PartitionSpec for output (same as in_spec for FFT)

    Returns:
        A function that performs 3D IFFT on sharded data
    """
    def _local_ifftn(x_local):
        # Each device runs FFT on its local shard independently
        return jnp.fft.ifftn(x_local, axes=axes, norm=norm)

    return shard_map(_local_ifftn, mesh=mesh, in_specs=(in_spec,), out_specs=out_spec)

def make_sharded_fftn_3d(
	mesh: Mesh,
	in_spec: P,
	out_spec: P,
	*,
	norm: str | None = None,
	axes: tuple[int, int, int] = (-3, -2, -1),
):
    """
    shard_map local FFT (forward).

    This is the forward-FFT counterpart to make_sharded_ifftn_3d.
    """
    def _local_fftn(x_local):
        return jnp.fft.fftn(x_local, axes=axes, norm=norm)

    return shard_map(_local_fftn, mesh=mesh, in_specs=(in_spec,), out_specs=out_spec)


# ============================================================================
# Flat-k FFT helpers — callers operate on (nk, *trail) arrays everywhere and
# the k-grid 3D form only exists inside this wrapper, matching the
# "flatten kx/ky/kz except inside the FFT" convention used across the GW
# pipeline (w_isdf chi0, ppm_sigma, gw_jax static COHSEX, isdf_fitting
# CCT/ZCT).  Internally: reshape (nk, *trail) -> (nkx, nky, nkz, *trail),
# pin with `with_sharding_constraint`, run a jittable device-local 3D FFT
# over the leading (0, 1, 2) k-axes, reshape back to (nk, *trail).
# ============================================================================


def make_flat_k_fft(
    mesh: Mesh,
    kgrid: tuple[int, int, int],
    spec: P,
    *,
    kind: str,
    norm: str | None = 'ortho',
    out_spec: P | None = None,
) -> Callable:
    """Return a flat-k FFT: ``(nk, *trail) -> (nk, *trail)``.

    ``spec`` is the ``PartitionSpec`` on the 3-D form
    ``(nkx, nky, nkz, *trail)``.  The three leading k-axes must be
    replicated (``None``) so the inner custom-partitioned FFT sees the
    full FFT axis on every device.  ``out_spec`` defaults to ``spec``;
    pass a different one only if a post-FFT reshard is wanted.

    ``kind='ifftn'`` or ``'fftn'`` selects the direction.  ``norm``
    follows ``jnp.fft.*`` ('ortho', 'forward', 'backward' / None).
    """
    nkx, nky, nkz = (int(v) for v in kgrid)
    nk = nkx * nky * nkz
    in_shard = NamedSharding(mesh, spec)
    if kind == 'ifftn':
        inner = make_jittable_local_ifftn_3d(
            mesh, spec, out_spec if out_spec is not None else spec,
            norm=norm, axes=(0, 1, 2))
    elif kind == 'fftn':
        inner = make_jittable_local_fftn_3d(
            mesh, spec, out_spec if out_spec is not None else spec,
            norm=norm, axes=(0, 1, 2))
    else:
        raise ValueError(f"kind must be 'ifftn' or 'fftn', got {kind!r}")

    def _flat_k_fft(x_flat):
        trail = x_flat.shape[1:]
        x_3d = jax.lax.with_sharding_constraint(
            x_flat.reshape(nkx, nky, nkz, *trail), in_shard)
        return inner(x_3d).reshape(nk, *trail)

    return _flat_k_fft


def make_flat_k_ifftn(
    mesh: Mesh,
    kgrid: tuple[int, int, int],
    spec: P,
    *,
    norm: str | None = 'ortho',
    out_spec: P | None = None,
) -> Callable:
    """Flat-k IFFT ``(nk, *trail) -> (nk, *trail)``.  See :func:`make_flat_k_fft`."""
    return make_flat_k_fft(mesh, kgrid, spec, kind='ifftn',
                           norm=norm, out_spec=out_spec)


def make_flat_k_fftn(
    mesh: Mesh,
    kgrid: tuple[int, int, int],
    spec: P,
    *,
    norm: str | None = 'ortho',
    out_spec: P | None = None,
) -> Callable:
    """Flat-k FFT ``(nk, *trail) -> (nk, *trail)``.  See :func:`make_flat_k_fft`."""
    return make_flat_k_fft(mesh, kgrid, spec, kind='fftn',
                           norm=norm, out_spec=out_spec)

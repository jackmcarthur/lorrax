"""One packed C⊕T1⊕T2⊕T3 layout for four-current operators.

Physics code names logical blocks ``O[A,B]``; the distributed Dyson solve
sees one square matrix.  This module owns the only conversion between those
representations.  It contains no response kernel, propagator formula, or
symmetry operation.

The packed order is mesh-interleaved.  On a square ``p x p`` mesh, packed row
shard ``x`` contains that same shard's local ``C,T1,T2,T3`` row chunks, and
column shard ``y`` contains its analogous chunks.  Row and column therefore
apply the same permutation, so a packed operator is ``P O P^T`` and Dyson
algebra is unchanged.  Pack and block view are local ``shard_map`` slices;
there is no hidden redistribution.  A globally contiguous direct sum under
ordinary ``P(None,'x','y')`` would not have that property.

Each family extent is padded through ``runtime.padding.padded_mu_extent``.
Pack structurally zeros every internal-pad row and column.  Consequently the
distributed Dyson matrix has exact identity at invalid indices and zero RHS;
logical blocks remain the portable on-disk representation.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from runtime.padding import padded_mu_extent


N_LORENTZ = 4
CHARGE = 0
TRANSVERSE = (1, 2, 3)


@dataclass(frozen=True)
class PhotonBasisLayout:
    """Immutable direct-sum ordering and internal-padding invariant."""

    logical_extents: tuple[int, int, int, int]
    padded_extents: tuple[int, int, int, int]
    mesh_side: int
    ordering: str = "mesh_interleaved_direct_sum_v1"
    bare_propagator: str = "instantaneous_coulomb_gauge_v1"

    def __post_init__(self) -> None:
        logical = tuple(int(n) for n in self.logical_extents)
        padded = tuple(int(n) for n in self.padded_extents)
        side = int(self.mesh_side)
        if len(logical) != N_LORENTZ or len(padded) != N_LORENTZ:
            raise ValueError("PhotonBasisLayout requires four Lorentz extents")
        if side < 1 or any(n < 1 for n in logical):
            raise ValueError(
                f"invalid PhotonBasisLayout logical={logical}, mesh_side={side}")
        if any(p < n or p % side for n, p in zip(logical, padded)):
            raise ValueError(
                "padded extents must cover logical extents and divide the "
                f"mesh side: logical={logical}, padded={padded}, side={side}")
        if logical[1:] != (logical[1],) * 3 or padded[1:] != (padded[1],) * 3:
            raise ValueError(
                "one transverse centroid family must serve T1/T2/T3; got "
                f"logical={logical}, padded={padded}")

    @classmethod
    def from_centroid_extents(
        cls, n_charge: int, n_transverse: int, mesh_xy: Mesh,
    ) -> "PhotonBasisLayout":
        px = int(mesh_xy.shape['x'])
        py = int(mesh_xy.shape['y'])
        if px != py:
            raise ValueError(
                "full photon response requires a square ('x','y') mesh so "
                "row/column interleaving is one identical permutation; got "
                f"{px}x{py}")
        divisor = px * py
        p_c = padded_mu_extent(int(n_charge), divisor)
        p_t = padded_mu_extent(int(n_transverse), divisor)
        return cls(
            logical_extents=(int(n_charge), int(n_transverse),
                             int(n_transverse), int(n_transverse)),
            padded_extents=(p_c, p_t, p_t, p_t),
            mesh_side=px)

    @property
    def packed_extent(self) -> int:
        return sum(self.padded_extents)

    def logical_extent(self, channel: int) -> int:
        self._check_channel(channel)
        return self.logical_extents[int(channel)]

    def padded_extent(self, channel: int) -> int:
        self._check_channel(channel)
        return self.padded_extents[int(channel)]

    def local_offset(self, channel: int) -> int:
        """Channel offset inside one packed device shard."""
        self._check_channel(channel)
        return sum(p // self.mesh_side
                   for p in self.padded_extents[:int(channel)])

    def block_shape(self, nq: int, channel_left: int,
                    channel_right: int) -> tuple[int, int, int]:
        return (int(nq), self.padded_extent(channel_left),
                self.padded_extent(channel_right))

    def assert_mesh(self, mesh_xy: Mesh) -> None:
        px = int(mesh_xy.shape['x'])
        py = int(mesh_xy.shape['y'])
        if (px, py) != (self.mesh_side, self.mesh_side):
            raise ValueError(
                f"PhotonBasisLayout is for {self.mesh_side}x{self.mesh_side}, "
                f"got {px}x{py}")

    @staticmethod
    def _check_channel(channel: int) -> None:
        if not 0 <= int(channel) < N_LORENTZ:
            raise ValueError(
                f"Lorentz channel must be in {{0,1,2,3}}; got {channel}")


_zero_cache: dict = {}
_insert_cache: dict = {}
_view_cache: dict = {}
_vector_pack_cache: dict = {}


def _empty(nq, layout, mesh_xy, dtype):
    shape = (int(nq), layout.packed_extent, layout.packed_extent)
    key = (id(mesh_xy), shape, np.dtype(dtype).str)
    if key not in _zero_cache:
        out = NamedSharding(mesh_xy, P(None, 'x', 'y'))

        @partial(jax.jit, out_shardings=out)
        def zeros():
            return jnp.zeros(shape, dtype=dtype)

        _zero_cache[key] = zeros
    return _zero_cache[key]()


def _insert_program(layout, mesh_xy, nq, p_left, p_right):
    """Shape-specialized insert; offsets/valid lengths stay runtime data."""
    key = (id(mesh_xy), int(nq), layout.packed_extent,
           int(p_left), int(p_right))
    if key in _insert_cache:
        return _insert_cache[key]
    from common.shard_map import shard_map

    side = layout.mesh_side
    spec = P(None, 'x', 'y')
    scalar = P()
    nat = NamedSharding(mesh_xy, spec)
    rep0 = NamedSharding(mesh_xy, scalar)

    @partial(shard_map, mesh=mesh_xy,
             in_specs=(spec, spec, scalar, scalar, scalar, scalar),
             out_specs=spec, check_vma=False)
    def insert_local(acc, block, off_left, off_right, n_left, n_right):
        x0 = jax.lax.axis_index('x') * (p_left // side)
        y0 = jax.lax.axis_index('y') * (p_right // side)
        valid_left = x0 + jnp.arange(p_left // side) < n_left
        valid_right = y0 + jnp.arange(p_right // side) < n_right
        block = jnp.where(
            valid_left[None, :, None] & valid_right[None, None, :], block, 0)
        return jax.lax.dynamic_update_slice(
            acc, block, (0, off_left, off_right))

    @partial(jax.jit,
             in_shardings=(nat, nat, rep0, rep0, rep0, rep0),
             out_shardings=nat, donate_argnums=(0,))
    def insert(acc, block, off_left, off_right, n_left, n_right):
        return insert_local(
            acc, block, off_left, off_right, n_left, n_right)

    _insert_cache[key] = insert
    return insert


def _insert(packed, block, layout, A, B, mesh_xy):
    expected = layout.block_shape(int(packed.shape[0]), A, B)
    if tuple(block.shape) != expected:
        raise ValueError(f"photon block ({A},{B}) shape {block.shape} != {expected}")
    scalar = lambda x: jnp.asarray(int(x), dtype=jnp.int32)
    return _insert_program(
        layout, mesh_xy, int(packed.shape[0]),
        layout.padded_extent(A), layout.padded_extent(B))(
            packed, block,
            scalar(layout.local_offset(A)), scalar(layout.local_offset(B)),
            scalar(layout.logical_extent(A)), scalar(layout.logical_extent(B)))


def pack_photon_operator(
    get_block: Callable[[int, int], jax.Array], nq: int,
    layout: PhotonBasisLayout, mesh_xy: Mesh, *, dtype=jnp.complex128,
) -> jax.Array:
    """Stream sixteen blocks into one packed ``P(None,'x','y')`` operator.

    ``get_block(A,B)`` is called only when that block is about to be inserted.
    Each insertion is completed before the next callback so the consumed
    block's device buffer is released before another response block can be
    allocated.  Peak residency is therefore the accumulator plus one block,
    and T1/T2/T3 never imply wavefunction copies.
    """
    layout.assert_mesh(mesh_xy)
    packed = _empty(nq, layout, mesh_xy, dtype)
    for A in range(N_LORENTZ):
        for B in range(N_LORENTZ):
            packed = _insert(
                packed, get_block(A, B), layout, A, B, mesh_xy)
            # JAX dispatch is asynchronous.  The next get_block() is an
            # independent, body-sized response build, so the accumulator
            # dependency alone does not prevent two block outputs from being
            # allocated at once.  This is the explicit one-block lifetime
            # boundary promised by the streaming contract above.
            packed.block_until_ready()
    return packed


def _view_program(layout, mesh_xy, nq, p_left, p_right):
    key = (id(mesh_xy), int(nq), layout.packed_extent,
           int(p_left), int(p_right))
    if key in _view_cache:
        return _view_cache[key]
    from common.shard_map import shard_map

    side = layout.mesh_side
    spec = P(None, 'x', 'y')
    scalar = P()
    nat = NamedSharding(mesh_xy, spec)
    rep0 = NamedSharding(mesh_xy, scalar)

    @partial(shard_map, mesh=mesh_xy,
             in_specs=(spec, scalar, scalar), out_specs=spec,
             check_vma=False)
    def view_local(packed, off_left, off_right):
        return jax.lax.dynamic_slice(
            packed, (0, off_left, off_right),
            (int(nq), p_left // side, p_right // side))

    view = jax.jit(
        view_local, in_shardings=(nat, rep0, rep0), out_shardings=nat)
    _view_cache[key] = view
    return view


def photon_block_view(
    packed: jax.Array, layout: PhotonBasisLayout,
    channel_left: int, channel_right: int, mesh_xy: Mesh,
) -> jax.Array:
    """Unpack padded ``O_AB`` without gathering or redistributing."""
    layout.assert_mesh(mesh_xy)
    expected = (int(packed.shape[0]), layout.packed_extent,
                layout.packed_extent)
    if tuple(packed.shape) != expected:
        raise ValueError(f"packed photon operator shape {packed.shape} != {expected}")
    A, B = int(channel_left), int(channel_right)
    scalar = lambda x: jnp.asarray(int(x), dtype=jnp.int32)
    return _view_program(
        layout, mesh_xy, int(packed.shape[0]),
        layout.padded_extent(A), layout.padded_extent(B))(
            packed, scalar(layout.local_offset(A)),
            scalar(layout.local_offset(B)))


def _vector_pack_program(layout, mesh_xy, nq, dtype, axis_name):
    """One local graph for embedding four channel vectors in packed space."""
    padded = tuple(int(n) for n in layout.padded_extents)
    key = (id(mesh_xy), padded, int(nq), np.dtype(dtype).str, axis_name)
    if key in _vector_pack_cache:
        return _vector_pack_cache[key]
    from common.shard_map import shard_map

    side = layout.mesh_side
    vector_spec = P(None, axis_name)
    packed_spec = P(None, None, axis_name)
    logical_spec = P()
    vector_sharding = NamedSharding(mesh_xy, vector_spec)
    packed_sharding = NamedSharding(mesh_xy, packed_spec)
    logical_sharding = NamedSharding(mesh_xy, logical_spec)
    local_extents = tuple(n // side for n in padded)
    local_packed = sum(local_extents)

    @partial(shard_map, mesh=mesh_xy,
             in_specs=(vector_spec, vector_spec, vector_spec, vector_spec,
                       logical_spec),
             out_specs=packed_spec, check_vma=False)
    def pack_local(v0, v1, v2, v3, logical_extents):
        shard = jax.lax.axis_index(axis_name)
        out = jnp.zeros(
            (int(nq), N_LORENTZ, local_packed), dtype=v0.dtype)
        offset = 0
        for channel, (vector, n_local) in enumerate(
                zip((v0, v1, v2, v3), local_extents)):
            valid = (
                shard * n_local + jnp.arange(n_local)
                < logical_extents[channel])
            vector = jnp.where(valid[None, :], vector, 0)
            out = jax.lax.dynamic_update_slice(
                out, vector[:, None, :], (0, channel, offset))
            offset += n_local
        return out

    @partial(jax.jit,
             in_shardings=(vector_sharding,) * N_LORENTZ
                          + (logical_sharding,),
             out_shardings=packed_sharding)
    def pack(v0, v1, v2, v3, logical_extents):
        return pack_local(v0, v1, v2, v3, logical_extents)

    _vector_pack_cache[key] = pack
    return pack


def pack_photon_channel_vectors(
    vectors_by_channel: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    layout: PhotonBasisLayout,
    mesh_xy: Mesh,
    *,
    axis_name: str,
) -> jax.Array:
    """Embed four native channel vectors in the packed photon basis.

    Input channel ``A`` has shape ``(nq, layout.padded_extent(A))`` and is
    already sharded ``P(None, axis_name)``.  The result has shape
    ``(nq, 4, layout.packed_extent)`` with sharding
    ``P(None, None, axis_name)``; row ``A`` contains the input only in the
    packed ``A`` segment.  Internal channel padding is zeroed here, where the
    mesh-interleaved ordering is owned.  No conjugation or prefactor is
    implicit.
    """
    layout.assert_mesh(mesh_xy)
    if axis_name not in ('x', 'y'):
        raise ValueError(
            f"packed photon-vector axis must be 'x' or 'y'; got {axis_name!r}")
    if len(vectors_by_channel) != N_LORENTZ:
        raise ValueError(
            f"packed photon vectors require four channels; got "
            f"{len(vectors_by_channel)}")

    nq = int(vectors_by_channel[0].shape[0])
    dtype = np.dtype(vectors_by_channel[0].dtype)
    wanted = NamedSharding(mesh_xy, P(None, axis_name))
    for channel, vector in enumerate(vectors_by_channel):
        expected = (nq, layout.padded_extent(channel))
        if tuple(vector.shape) != expected:
            raise ValueError(
                f"photon channel {channel} vector shape {vector.shape} != "
                f"{expected}")
        if np.dtype(vector.dtype) != dtype:
            raise TypeError(
                "all packed photon vectors must have one dtype; got "
                f"{dtype} and {vector.dtype} for channel {channel}")
        sharding = getattr(vector, 'sharding', None)
        if sharding is None or not sharding.is_equivalent_to(wanted, 2):
            raise ValueError(
                f"photon channel {channel} vector must already have sharding "
                f"P(None, {axis_name!r}); got {sharding}")

    logical = jnp.asarray(layout.logical_extents, dtype=jnp.int32)
    return _vector_pack_program(
        layout, mesh_xy, nq, dtype, axis_name)(
            *vectors_by_channel, logical)


__all__ = [
    "CHARGE", "TRANSVERSE", "N_LORENTZ",
    "PhotonBasisLayout", "pack_photon_operator", "photon_block_view",
    "pack_photon_channel_vectors",
]

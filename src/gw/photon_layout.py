"""One packed C⊕T1⊕T2⊕T3 layout for four-current operators.

Physics code names logical blocks ``O[A,B]`` (Lorentz ``A,B in {C, T1, T2,
T3}``, each block a ``(nq, n_mu_A, n_mu_B)`` centroid tile); the distributed
Dyson solve of ``bispinor_gw = full_static_cohsex`` sees one square matrix
of extent ``packed_extent``.  This module owns the only conversion between
those representations, and everything that must know the packed ordering:

* :class:`PhotonBasisLayout` -- the immutable extents/padding/mesh record;
* :func:`pack_photon_operator` / :func:`photon_block_view` -- pack sixteen
  tiles into one ``P(None,'x','y')`` operator and view one block of it
  (local ``shard_map`` slices, no redistribution);
* :func:`pack_photon_channel_vectors` -- pack four per-channel row vectors
  (the literal ``G = 0`` vectors ``g0``, the head wings) into one packed row;
* :func:`add_photon_q0_low_rank` -- the bounded rank-4 update
  ``O[q=0] += L R`` by which ``gw.head_correction.complete_static_slab_photon_q0``
  inserts the bare ``<D>`` and the nine screened Gamma-cell moments; and
* :func:`photon_q0_low_rank_block` -- its diagnostic twin that materializes
  one ``(A,B)`` block of those updates alone, so ``gw.photon_sigma`` can
  attribute Sigma per Lorentz sector without a second packed body.

It contains no response kernel, propagator formula, or symmetry operation.

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
from typing import Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from runtime.padding import PaddedAxis, padded_mu_axis


N_LORENTZ = 4
CHARGE = 0
TRANSVERSE = (1, 2, 3)
MAX_Q0_UPDATE_RANK = 4
PHOTON_BASIS_ORDERING = "mesh_interleaved_direct_sum_v1"
PHOTON_BARE_PROPAGATOR = "instantaneous_coulomb_gauge_v1"


@dataclass(frozen=True)
class PhotonBasisLayout:
    """Immutable direct-sum ordering and internal-padding invariant."""

    channel_axes: tuple[PaddedAxis, PaddedAxis, PaddedAxis, PaddedAxis]
    mesh_side: int
    ordering: str = PHOTON_BASIS_ORDERING
    bare_propagator: str = PHOTON_BARE_PROPAGATOR

    def __post_init__(self) -> None:
        logical = self.logical_extents
        padded = self.carrier_extents
        side = int(self.mesh_side)
        if len(self.channel_axes) != N_LORENTZ:
            raise ValueError("PhotonBasisLayout requires four Lorentz extents")
        if side < 1 or any(n < 1 for n in logical):
            raise ValueError(
                f"invalid PhotonBasisLayout logical={logical}, mesh_side={side}")
        divisors = {int(tag.divisor) for tag in self.channel_axes}
        if len(divisors) != 1:
            raise ValueError(
                f"photon channels carry different divisors {sorted(divisors)}")
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
        p_c = padded_mu_axis(int(n_charge), mesh_xy)
        p_t = padded_mu_axis(int(n_transverse), mesh_xy)
        return cls(
            channel_axes=(p_c, p_t, p_t, p_t),
            mesh_side=px)

    @property
    def logical_extents(self) -> tuple[int, int, int, int]:
        return tuple(int(tag.logical) for tag in self.channel_axes)

    @property
    def carrier_extents(self) -> tuple[int, int, int, int]:
        return tuple(int(tag.carrier) for tag in self.channel_axes)

    @property
    def packed_extent(self) -> int:
        return sum(self.carrier_extents)

    def logical_extent(self, channel: int) -> int:
        self._check_channel(channel)
        return self.logical_extents[int(channel)]

    def carrier_extent(self, channel: int) -> int:
        self._check_channel(channel)
        return self.carrier_extents[int(channel)]

    def channel_axis(self, channel: int) -> PaddedAxis:
        """Logical/carrier receipt for one Lorentz centroid family."""
        self._check_channel(channel)
        return self.channel_axes[int(channel)]

    def local_offset(self, channel: int) -> int:
        """Channel offset inside one packed device shard."""
        self._check_channel(channel)
        return sum(p // self.mesh_side
                   for p in self.carrier_extents[:int(channel)])

    def block_shape(self, nq: int, channel_left: int,
                    channel_right: int) -> tuple[int, int, int]:
        return (int(nq), self.carrier_extent(channel_left),
                self.carrier_extent(channel_right))

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
_q0_update_cache: dict = {}
_q0_block_cache: dict = {}


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
            acc, block, (jnp.asarray(0, dtype=jnp.int32),
                         off_left, off_right))

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
        layout.carrier_extent(A), layout.carrier_extent(B))(
            packed, block,
            scalar(layout.local_offset(A)), scalar(layout.local_offset(B)),
            scalar(layout.logical_extent(A)), scalar(layout.logical_extent(B)))


def pack_photon_operator(
    get_block: Callable[[int, int], jax.Array | None], nq: int,
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
            block = get_block(A, B)
            # Missing physical channels are exact zero blocks.  Keeping this
            # case in the sole packer avoids a second packing graph and a
            # second body-sized zero accumulator.
            if block is None:
                continue
            packed = _insert(packed, block, layout, A, B, mesh_xy)
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
            packed, (jnp.asarray(0, dtype=jnp.int32),
                     off_left, off_right),
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
        layout.carrier_extent(A), layout.carrier_extent(B))(
            packed, scalar(layout.local_offset(A)),
            scalar(layout.local_offset(B)))


def pack_photon_response_tiles(
    tiles: Mapping[tuple[int, int], jax.Array | None],
    nq: int,
    layout: PhotonBasisLayout,
    mesh_xy: Mesh,
    *,
    dtype=jnp.complex128,
) -> jax.Array:
    """Pack present 4x4 response tiles, zero-filling absent channels.

    This mapping adapter delegates all ordering, padding, and insertion to
    :func:`pack_photon_operator`.  A present tile is already in the canonical
    padded block shape with ``P(None,'x','y')`` sharding.  No conjugation,
    symmetry completion, physical prefactor, or alternate JIT is implicit.
    """
    layout.assert_mesh(mesh_xy)
    wanted_dtype = np.dtype(dtype)
    wanted_sharding = NamedSharding(mesh_xy, P(None, "x", "y"))
    normalized = {}
    for key, tile in tiles.items():
        if (not isinstance(key, tuple) or len(key) != 2
                or any(not isinstance(v, (int, np.integer)) for v in key)):
            raise TypeError(
                "photon response tile keys must be integer (A,B) pairs; "
                f"got {key!r}")
        A, B = int(key[0]), int(key[1])
        layout._check_channel(A)
        layout._check_channel(B)
        if tile is not None:
            expected = layout.block_shape(int(nq), A, B)
            if tuple(tile.shape) != expected:
                raise ValueError(
                    f"photon response tile ({A},{B}) shape {tile.shape} != "
                    f"{expected}")
            if np.dtype(tile.dtype) != wanted_dtype:
                raise TypeError(
                    f"photon response tile ({A},{B}) dtype {tile.dtype} != "
                    f"{wanted_dtype}")
            sharding = getattr(tile, "sharding", None)
            if (not isinstance(sharding, NamedSharding)
                    or not sharding.is_equivalent_to(
                        wanted_sharding, tile.ndim)):
                raise ValueError(
                    f"photon response tile ({A},{B}) must already have "
                    f"sharding P(None,'x','y'); got {sharding!r}")
        normalized[(A, B)] = tile
    return pack_photon_operator(
        lambda A, B: normalized.get((A, B)), int(nq), layout, mesh_xy,
        dtype=dtype)


def unpack_photon_response_tiles(
    packed: jax.Array,
    layout: PhotonBasisLayout,
    mesh_xy: Mesh,
) -> tuple[tuple[jax.Array, ...], ...]:
    """Return the padded 4x4 response-tile views of one packed operator.

    Every entry delegates to :func:`photon_block_view`; this helper adds no
    ordering, trimming, redistribution, JIT, or compile-cache surface.
    """
    return tuple(
        tuple(photon_block_view(packed, layout, A, B, mesh_xy)
              for B in range(N_LORENTZ))
        for A in range(N_LORENTZ))


def _vector_pack_program(layout, mesh_xy, nq, dtype, axis_name):
    """One local graph for embedding four channel vectors in packed space."""
    padded = tuple(int(n) for n in layout.carrier_extents)
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

    Input channel ``A`` has shape ``(nq, layout.carrier_extent(A))`` and is
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
        expected = (nq, layout.carrier_extent(channel))
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


def _q0_local_factor_piece(rows, *, axis_name, local_extent, local_offset,
                           logical_extent):
    """Canonical local slice + padding mask for one q=0 factor channel."""
    local_offset = jnp.asarray(local_offset, dtype=jnp.int32)
    zero = jnp.asarray(0, dtype=jnp.int32)
    piece = jax.lax.dynamic_slice(
        rows, (zero, local_offset), (MAX_Q0_UPDATE_RANK, local_extent))
    valid = (
        jax.lax.axis_index(axis_name) * local_extent
        + jnp.arange(local_extent) < logical_extent)
    return jnp.where(valid[None, :], piece, 0)


def _q0_local_outer(left_rows, right_rows):
    """The one local outer-product primitive for every q=0 factor update."""
    return jnp.einsum("ai,aj->ij", left_rows, right_rows)


def _validate_q0_factor_pair(left, right, layout, mesh_xy, *, dtype, label):
    factor_shape = (MAX_Q0_UPDATE_RANK, layout.packed_extent)
    if tuple(left.shape) != factor_shape or tuple(right.shape) != factor_shape:
        raise ValueError(
            f"{label} q=0 factor pair must have two {factor_shape} arrays; "
            f"got {left.shape}/{right.shape}")
    if (np.dtype(left.dtype) != np.dtype(dtype)
            or np.dtype(right.dtype) != np.dtype(dtype)):
        raise TypeError(
            f"{label} q=0 factors must have dtype {np.dtype(dtype)}; got "
            f"{left.dtype}/{right.dtype}")
    for array, name, spec in (
        (left, "left_rows_X", P(None, "x")),
        (right, "right_rows_Y", P(None, "y")),
    ):
        wanted = NamedSharding(mesh_xy, spec)
        sharding = getattr(array, "sharding", None)
        if (sharding is None
                or not sharding.is_equivalent_to(wanted, array.ndim)):
            raise ValueError(
                f"{label} {name} must already have sharding {spec}; got "
                f"{sharding}. Refusing an implicit factor reshard.")


def _q0_update_program(layout, mesh_xy, nq, dtype):
    """One shape-stable local graph per padded q=0 update geometry."""
    key = (id(mesh_xy), tuple(layout.carrier_extents), int(nq),
           np.dtype(dtype).str)
    if key in _q0_update_cache:
        return _q0_update_cache[key]
    from common.shard_map import shard_map

    packed_spec = P(None, 'x', 'y')
    left_spec = P(None, 'x')
    right_spec = P(None, 'y')
    logical_spec = P(None)
    packed_sharding = NamedSharding(mesh_xy, packed_spec)
    left_sharding = NamedSharding(mesh_xy, left_spec)
    right_sharding = NamedSharding(mesh_xy, right_spec)
    logical_sharding = NamedSharding(mesh_xy, logical_spec)

    @partial(shard_map, mesh=mesh_xy,
             in_specs=(packed_spec, left_spec, right_spec, logical_spec),
             out_specs=packed_spec, check_vma=False)
    def add_local(packed, left_rows, right_rows, logical_extents):
        left_pieces = []
        right_pieces = []
        for channel in range(N_LORENTZ):
            local = layout.carrier_extent(channel) // layout.mesh_side
            left_pieces.append(_q0_local_factor_piece(
                left_rows, axis_name="x", local_extent=local,
                local_offset=layout.local_offset(channel),
                logical_extent=logical_extents[channel]))
            right_pieces.append(_q0_local_factor_piece(
                right_rows, axis_name="y", local_extent=local,
                local_offset=layout.local_offset(channel),
                logical_extent=logical_extents[channel]))
        delta_q0 = _q0_local_outer(
            jnp.concatenate(tuple(left_pieces), axis=1),
            jnp.concatenate(tuple(right_pieces), axis=1))
        return packed.at[0, :, :].add(delta_q0)

    @partial(jax.jit,
             in_shardings=(packed_sharding, left_sharding, right_sharding,
                           logical_sharding),
             out_shardings=packed_sharding, donate_argnums=(0,))
    def add_q0(packed, left_rows, right_rows, logical_extents):
        return add_local(packed, left_rows, right_rows, logical_extents)

    _q0_update_cache[key] = add_q0
    return add_q0


def add_photon_q0_low_rank(
    packed: jax.Array,
    layout: PhotonBasisLayout,
    mesh_xy: Mesh,
    *,
    left_rows_X: jax.Array | None = None,
    right_rows_Y: jax.Array | None = None,
) -> jax.Array:
    """Add one bounded low-rank update to the packed q=Gamma operator.

    The mathematical update is ``O[0] += L @ R`` with rank at most four.
    For local 2-D placement, callers supply ``left_rows_X = L.T`` and
    ``right_rows_Y = R``, both stored as fixed-capacity ``(4, N_packed)``
    row arrays.  Their shardings are respectively ``P(None, 'x')`` and
    ``P(None, 'y')``; unused rows are exactly zero.  No conjugation or
    physical prefactor is implicit: directed response factors must already
    contain both.  This convention admits general non-Hermitian bordered
    response updates rather than silently imposing a head model here.

    Internal C/T padding is masked structurally from ``layout``.  The update
    is a local outer-product sum on each device and the donated packed
    accumulator never gathers or reshards.  Calls with both factors absent
    return the identical object without dispatch; supplying only one factor,
    a non-capacity-four factor, or a non-native sharding is refused.
    """
    if left_rows_X is None and right_rows_Y is None:
        return packed
    if left_rows_X is None or right_rows_Y is None:
        raise ValueError(
            "q=0 low-rank update requires both left_rows_X and right_rows_Y")

    layout.assert_mesh(mesh_xy)
    if getattr(packed, "ndim", None) != 3:
        raise ValueError(
            f"packed photon operator must be rank 3; got shape "
            f"{getattr(packed, 'shape', None)}")
    packed_shape = (
        int(packed.shape[0]), layout.packed_extent, layout.packed_extent)
    if int(packed.shape[0]) < 1 or tuple(packed.shape) != packed_shape:
        raise ValueError(
            f"packed photon operator shape {packed.shape} != {packed_shape}")
    _validate_q0_factor_pair(
        left_rows_X, right_rows_Y, layout, mesh_xy, dtype=packed.dtype,
        label="packed update")
    wanted_packed = NamedSharding(mesh_xy, P(None, "x", "y"))
    if (getattr(packed, "sharding", None) is None
            or not packed.sharding.is_equivalent_to(wanted_packed, 3)):
        raise ValueError(
            "packed must already have sharding P(None,'x','y'); got "
            f"{getattr(packed, 'sharding', None)}. Refusing an implicit "
            "packed-body reshard.")

    updated = _q0_update_program(
        layout, mesh_xy, packed_shape[0], packed.dtype)(
            packed, left_rows_X, right_rows_Y,
            jnp.asarray(layout.logical_extents, dtype=jnp.int32))
    # Repeated bounded updates reuse the donated accumulator.  This explicit
    # lifetime boundary prevents independently produced factor pairs from
    # overlapping the next update's live set under asynchronous dispatch.
    updated.block_until_ready()
    return updated


def _q0_block_program(mesh_xy, p_left, p_right, dtype, n_pairs):
    """One q-extent-one graph per padded shape class and factor count."""
    key = (id(mesh_xy), int(p_left), int(p_right), np.dtype(dtype).str,
           int(n_pairs))
    if key in _q0_block_cache:
        return _q0_block_cache[key]
    from common.shard_map import shard_map

    side = int(mesh_xy.shape["x"])
    left_spec = P(None, "x")
    right_spec = P(None, "y")
    out_spec = P(None, "x", "y")
    scalar_spec = P()
    left_sharding = NamedSharding(mesh_xy, left_spec)
    right_sharding = NamedSharding(mesh_xy, right_spec)
    out_sharding = NamedSharding(mesh_xy, out_spec)
    scalar_sharding = NamedSharding(mesh_xy, scalar_spec)

    @partial(
        shard_map,
        mesh=mesh_xy,
        in_specs=(left_spec,) * n_pairs + (right_spec,) * n_pairs
        + (scalar_spec,) * 4,
        out_specs=out_spec,
        check_vma=False,
    )
    def block_local(*args):
        left_rows = args[:n_pairs]
        right_rows = args[n_pairs:2 * n_pairs]
        off_left, off_right, n_left, n_right = args[2 * n_pairs:]
        local_left = p_left // side
        local_right = p_right // side
        delta = jnp.zeros((local_left, local_right), dtype=dtype)
        for left, right in zip(left_rows, right_rows):
            left_piece = _q0_local_factor_piece(
                left, axis_name="x", local_extent=local_left,
                local_offset=off_left, logical_extent=n_left)
            right_piece = _q0_local_factor_piece(
                right, axis_name="y", local_extent=local_right,
                local_offset=off_right, logical_extent=n_right)
            delta = delta + _q0_local_outer(left_piece, right_piece)
        out = jnp.zeros((1, local_left, local_right), dtype=dtype)
        return out.at[0].set(delta)

    @partial(
        jax.jit,
        in_shardings=(left_sharding,) * n_pairs
        + (right_sharding,) * n_pairs + (scalar_sharding,) * 4,
        out_shardings=out_sharding,
    )
    def build(*args):
        return block_local(*args)

    _q0_block_cache[key] = build
    return build


def photon_q0_low_rank_block(
    factor_pairs,
    layout: PhotonBasisLayout,
    channel_left: int,
    channel_right: int,
    mesh_xy: Mesh,
) -> jax.Array:
    """Materialize one final Lorentz block of bounded q=0 factor updates.

    This is the diagnostic twin of :func:`add_photon_q0_low_rank`: it uses
    the same mesh-interleaved channel offsets and internal-padding masks, but
    emits only the requested ``(A,B)`` block with q extent one.  It therefore
    never constructs a second packed ``N_packed x N_packed`` photon body or
    ``nq`` copies of a structurally-zero block.  ``factor_pairs`` is
    a nonempty sequence of the exact ``(left_rows_X, right_rows_Y)`` pairs
    that were inserted into the packed operator.
    """
    layout.assert_mesh(mesh_xy)
    A, B = int(channel_left), int(channel_right)
    layout._check_channel(A)
    layout._check_channel(B)
    pairs = tuple(factor_pairs)
    if not pairs:
        raise ValueError("q=0 Lorentz block requires at least one factor pair")
    lefts = []
    rights = []
    dtype = np.dtype(pairs[0][0].dtype)
    for index, pair in enumerate(pairs):
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError(
                "q=0 factor pairs must be (left_rows_X,right_rows_Y); "
                f"entry {index} is {type(pair).__name__}")
        left, right = pair
        _validate_q0_factor_pair(
            left, right, layout, mesh_xy, dtype=dtype,
            label=f"factor pair {index}")
        lefts.append(left)
        rights.append(right)
    scalar = lambda value: jnp.asarray(int(value), dtype=jnp.int32)
    return _q0_block_program(
        mesh_xy, layout.carrier_extent(A), layout.carrier_extent(B), dtype,
        len(pairs))(
            *lefts, *rights,
            scalar(layout.local_offset(A)), scalar(layout.local_offset(B)),
            scalar(layout.logical_extent(A)), scalar(layout.logical_extent(B)))


__all__ = [
    "CHARGE", "TRANSVERSE", "N_LORENTZ", "MAX_Q0_UPDATE_RANK",
    "PhotonBasisLayout", "pack_photon_operator", "photon_block_view",
    "pack_photon_response_tiles", "unpack_photon_response_tiles",
    "pack_photon_channel_vectors", "add_photon_q0_low_rank",
    "photon_q0_low_rank_block",
]

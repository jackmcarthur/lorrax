"""Deterministic whole-group packing for sharded array axes.

Scientific indices stay in their canonical, process-count-independent order.
This module derives a runtime view in which every labelled group occupies one
contiguous interval on one shard.  It knows no centroid or symmetry physics:
``group_id`` may describe any partition whose members must remain local.

Square meshes use one hierarchical order, never independently optimized X/Y
orders.  It is first divided into ``P=s**2`` equal fine shards.  Coarsening
each ``s`` consecutive fine shards gives the *same* global order and equal
extent for both X and Y.  Thus a group local to a fine shard is automatically
local in the X view, the Y view, and the flat selector view.  Files and
user-visible indices never inherit this runtime order.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import operator
from typing import ClassVar

import numpy as np


_GROUPED_SHARD_LAYOUT_SCHEMA = "lorrax.grouped-shard-layout.v1"


def _readonly(array, dtype=None) -> np.ndarray:
    out = np.array(array, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True, init=False)
class GroupedShardLayout:
    """One reversible, equally sized sharded view of labelled rows.

    ``packed_to_canonical`` contains ``-1`` in pad slots.  Real rows are a
    permutation of ``range(n_logical)``; each dense group is contiguous and
    lies wholly inside the shard named by ``group_owner``.  The group labels
    are dense and ordered by their first canonical member, independent of the
    labels' input spelling.
    """

    n_logical: int
    n_groups: int
    n_shards: int
    shard_size: int
    n_padded: int
    packed_to_canonical: np.ndarray
    canonical_to_packed: np.ndarray
    canonical_group_id: np.ndarray
    packed_group_id: np.ndarray
    active_mask: np.ndarray
    group_owner: np.ndarray
    group_start: np.ndarray
    group_size: np.ndarray
    shard_load: np.ndarray
    __lorrax_grouped_layout_schema__: ClassVar[str] = (
        _GROUPED_SHARD_LAYOUT_SCHEMA)

    def __init__(self, *args, **kwargs):
        del args, kwargs
        raise TypeError("use build_grouped_shard_layout()")

    @property
    def n_pad(self) -> int:
        return self.n_padded - self.n_logical

    @property
    def pad_fraction(self) -> float:
        return self.n_pad / self.n_padded

    def pack_host(self, array, *, axis: int = 0, fill_value=0):
        """Permute one canonical host axis and insert this layout's pads."""
        src = np.asarray(array)
        axis = int(axis) % src.ndim
        if int(src.shape[axis]) != self.n_logical:
            raise ValueError(
                "GroupedShardLayout.pack_host: canonical axis has extent "
                f"{src.shape[axis]}, expected {self.n_logical}.")
        src0 = np.moveaxis(src, axis, 0)
        dst0 = np.full(
            (self.n_padded,) + src0.shape[1:], fill_value, dtype=src.dtype)
        active = self.packed_to_canonical >= 0
        dst0[active] = src0[self.packed_to_canonical[active]]
        return np.moveaxis(dst0, 0, axis)

    def unpack_host(self, array, *, axis: int = 0):
        """Drop pads and restore one packed host axis to canonical order."""
        src = np.asarray(array)
        axis = int(axis) % src.ndim
        if int(src.shape[axis]) != self.n_padded:
            raise ValueError(
                "GroupedShardLayout.unpack_host: packed axis has extent "
                f"{src.shape[axis]}, expected {self.n_padded}.")
        src0 = np.moveaxis(src, axis, 0)
        dst0 = src0[self.canonical_to_packed]
        return np.moveaxis(dst0, 0, axis)

    def pack_permutations_host(self, permutations) -> np.ndarray:
        """Conjugate canonical gather maps into the packed, local view.

        ``permutations[s, i]`` is the source row gathered for target ``i``.
        Pad rows map to themselves.  A map that crosses a shard boundary is
        refused: that means the supplied group partition is not closed under
        the action and a supposedly local symmetry gather would communicate.
        """
        perm = np.asarray(permutations)
        if perm.ndim != 2 or int(perm.shape[1]) != self.n_logical:
            raise ValueError(
                "GroupedShardLayout.pack_permutations_host: expected shape "
                f"(n_map, {self.n_logical}); got {perm.shape}.")
        if perm.dtype.kind not in "iu":
            raise ValueError(
                "GroupedShardLayout.pack_permutations_host: maps must be integer; "
                f"got dtype {perm.dtype}.")
        int64 = np.iinfo(np.int64)
        if perm.size and (int(perm.min()) < int(int64.min)
                          or int(perm.max()) > int(int64.max)):
            raise ValueError(
                "GroupedShardLayout.pack_permutations_host: a map value "
                "cannot be represented as int64.")
        perm = perm.astype(np.int64, copy=False)
        if perm.size and (int(perm.min()) < 0
                          or int(perm.max()) >= self.n_logical):
            raise ValueError(
                "GroupedShardLayout.pack_permutations_host: a map leaves the "
                f"canonical range [0, {self.n_logical}).")
        target = np.arange(self.n_padded, dtype=np.int64)
        out = np.broadcast_to(target, (int(perm.shape[0]), self.n_padded)
                              ).copy()
        active = self.packed_to_canonical >= 0
        canonical_target = self.packed_to_canonical[active]
        out[:, active] = self.canonical_to_packed[
            perm[:, canonical_target]]
        for row_i, row in enumerate(out):
            if np.unique(row).size != self.n_padded:
                raise ValueError(
                    "GroupedShardLayout.pack_permutations_host: canonical map row "
                    f"{row_i} is not a permutation.")
        owner = target // self.shard_size
        crossing = owner[None, :] != owner[out]
        if np.any(crossing):
            row_i, packed_i = np.argwhere(crossing)[0]
            canonical_i = int(self.packed_to_canonical[packed_i])
            raise ValueError(
                "GroupedShardLayout.pack_permutations_host: map row "
                f"{int(row_i)} moves canonical row {canonical_i} from shard "
                f"{int(owner[packed_i])} to {int(owner[out[row_i, packed_i]])}; "
                "the group partition is not closed under the map.")
        return out.astype(np.int32)

    def pack_fine_local_permutations_host(self, permutations) -> np.ndarray:
        """Return gather indices local to this layout's finest shards.

        This is the executable form of :meth:`pack_permutations_host`: global
        packed sources are reduced to offsets inside their owning fine shard.
        The parent method first proves that every source and target share that
        owner, so taking these indices inside a shard needs no communication.
        """
        packed = self.pack_permutations_host(permutations)
        return (packed % self.shard_size).astype(np.int32)


def validate_grouped_shard_layout(layout):
    """Authenticate one host layout without relying on module class identity.

    LORRAX supports imports through both ``common.*`` and ``src.common.*``.
    Python may load those as distinct modules, so a nominal ``isinstance``
    check rejects a valid factory receipt from the other spelling.  The
    stable schema marker plus complete invariant check admits either spelling
    while still refusing independently assembled or mutated metadata.
    """
    who = "validate_grouped_shard_layout"
    if getattr(layout, "__lorrax_grouped_layout_schema__", None) != \
            _GROUPED_SHARD_LAYOUT_SCHEMA:
        raise TypeError(f"{who}: layout must come from build_grouped_shard_layout()")
    scalar_names = (
        "n_logical", "n_groups", "n_shards", "shard_size", "n_padded")
    try:
        n_logical, n_groups, n_shards, shard_size, n_padded = (
            operator.index(getattr(layout, name)) for name in scalar_names)
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError(f"{who}: malformed scalar receipt") from error
    if min(n_logical, n_groups, n_shards, shard_size) < 1:
        raise ValueError(f"{who}: all logical extents must be positive")
    if n_padded != n_shards * shard_size or n_padded < n_logical:
        raise ValueError(f"{who}: padded extent/shard geometry disagrees")

    expected = {
        "packed_to_canonical": ((n_padded,), np.dtype(np.int64)),
        "canonical_to_packed": ((n_logical,), np.dtype(np.int64)),
        "canonical_group_id": ((n_logical,), np.dtype(np.int32)),
        "packed_group_id": ((n_padded,), np.dtype(np.int32)),
        "active_mask": ((n_padded,), np.dtype(bool)),
        "group_owner": ((n_groups,), np.dtype(np.int32)),
        "group_start": ((n_groups,), np.dtype(np.int64)),
        "group_size": ((n_groups,), np.dtype(np.int32)),
        "shard_load": ((n_shards,), np.dtype(np.int64)),
    }
    arrays: dict[str, np.ndarray] = {}
    for name, (shape, dtype) in expected.items():
        value = getattr(layout, name, None)
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{who}: {name} must be a host NumPy array")
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(
                f"{who}: {name} has shape/dtype {value.shape}/{value.dtype}; "
                f"expected {shape}/{dtype}")
        if value.flags.writeable:
            raise ValueError(f"{who}: {name} must be immutable")
        arrays[name] = value

    packed = arrays["packed_to_canonical"]
    inverse = arrays["canonical_to_packed"]
    active = arrays["active_mask"]
    canonical_group = arrays["canonical_group_id"]
    packed_group = arrays["packed_group_id"]
    starts = arrays["group_start"]
    sizes = arrays["group_size"]
    owners = arrays["group_owner"]
    if not np.array_equal(active, packed >= 0):
        raise ValueError(f"{who}: active mask disagrees with pad sentinels")
    if not np.all(packed[~active] == -1):
        raise ValueError(f"{who}: canonical pad sentinels disagree")
    if np.any(sizes <= 0):
        raise ValueError(f"{who}: every dense group must be nonempty")
    active_rows = packed[active]
    if (active_rows.size != n_logical or np.any(active_rows < 0)
            or np.any(active_rows >= n_logical)
            or not np.all(np.bincount(active_rows, minlength=n_logical) == 1)):
        raise ValueError(f"{who}: packed rows are not one canonical permutation")
    if np.any(inverse < 0) or np.any(inverse >= n_padded) or not np.array_equal(
            packed[inverse], np.arange(n_logical)):
        raise ValueError(f"{who}: canonical inverse map disagrees")
    if not np.array_equal(
            inverse[active_rows], np.flatnonzero(active)):
        raise ValueError(f"{who}: packed inverse map disagrees")
    if np.any(canonical_group < 0) or np.any(canonical_group >= n_groups):
        raise ValueError(f"{who}: canonical group IDs leave the dense range")
    first = np.full((n_groups,), n_logical, dtype=np.int64)
    np.minimum.at(first, canonical_group, np.arange(n_logical))
    if first[0] != 0 or np.any(np.diff(first) <= 0):
        raise ValueError(
            f"{who}: dense group IDs do not follow first canonical appearance")
    if not np.array_equal(packed_group[active], canonical_group[packed[active]]):
        raise ValueError(f"{who}: packed and canonical group IDs disagree")
    if not np.all(packed_group[~active] == n_groups):
        raise ValueError(f"{who}: pad group sentinels disagree")
    if not np.array_equal(
            sizes, np.bincount(canonical_group, minlength=n_groups)):
        raise ValueError(f"{who}: group sizes disagree with canonical labels")
    for group in range(n_groups):
        start = int(starts[group])
        stop = start + int(sizes[group])
        if (start < 0 or stop > n_padded
                or start // shard_size != (stop - 1) // shard_size):
            raise ValueError(f"{who}: group {group} is not fine-shard local")
        if int(owners[group]) != start // shard_size or not np.all(
                packed_group[start:stop] == group):
            raise ValueError(f"{who}: group {group} owner/interval disagrees")
    measured_load = active.reshape(n_shards, shard_size).sum(axis=1)
    if not np.array_equal(arrays["shard_load"], measured_load):
        raise ValueError(f"{who}: shard loads disagree with active rows")
    local_rows = np.arange(shard_size)[None, :]
    if not np.array_equal(
            active.reshape(n_shards, shard_size),
            local_rows < arrays["shard_load"][:, None]):
        raise ValueError(f"{who}: pads are not shard-local suffixes")
    return layout


@dataclass(frozen=True, init=False)
class SquareGroupedShardLayout:
    """One finest-grain layout reused by every view of a square mesh.

    ``fine`` has ``side**2`` equal contiguous shards.  Both one-axis views
    split the identical packed sequence into ``side`` chunks, each comprising
    ``side`` consecutive fine shards.  ``axis_group_owner`` is therefore a
    numerical owner shared by X and Y; it does not claim that X and Y name the
    same physical devices.
    """

    side: int
    fine: GroupedShardLayout

    def __init__(self, *args, **kwargs):
        del args, kwargs
        raise TypeError("use build_square_grouped_shard_layout()")

    @property
    def axis_shard_size(self) -> int:
        return self.side * self.fine.shard_size

    @property
    def axis_group_owner(self) -> np.ndarray:
        return _readonly(self.fine.group_owner // self.side, np.int32)

    @property
    def gram_extent_multiplier(self) -> float:
        """Dense square-storage multiplier caused by fine-shard padding."""
        ratio = self.fine.n_padded / self.fine.n_logical
        return ratio * ratio

    def pack_axis_local_permutations_host(self, permutations) -> np.ndarray:
        """Return local gather offsets for either identical X/Y axis view."""
        packed = self.fine.pack_permutations_host(permutations)
        return (packed % self.axis_shard_size).astype(np.int32)


def build_grouped_shard_layout(group_id, n_shards: int) -> GroupedShardLayout:
    """Greedily balance complete groups across equal-capacity shards.

    Longest groups are placed first on the currently least-loaded shard
    (LPT scheduling).  Ties use the lowest canonical group and shard IDs.
    Rows retain canonical order inside each group.  Each shard is then padded
    at its own tail to the largest realized load, producing one static local
    shape without splitting any group.
    """
    labels = np.asarray(group_id)
    if labels.ndim != 1 or labels.dtype.kind not in "iu":
        raise ValueError(
            "build_grouped_shard_layout: group_id must be a one-dimensional "
            f"integer array; got shape={labels.shape}, dtype={labels.dtype}.")
    if labels.size == 0:
        raise ValueError("build_grouped_shard_layout: group_id may not be empty.")
    nshard = int(n_shards)
    if nshard < 1:
        raise ValueError(
            f"build_grouped_shard_layout: n_shards must be >= 1; got {nshard}.")

    # Dense IDs follow first canonical appearance, not the numerical spelling
    # of an upstream label.  That makes the layout stable under label renames.
    _, first, inverse = np.unique(
        labels, return_index=True, return_inverse=True)
    old_in_first_order = np.argsort(first, kind="stable")
    dense_of_old = np.empty_like(old_in_first_order)
    dense_of_old[old_in_first_order] = np.arange(old_in_first_order.size)
    dense = dense_of_old[inverse].astype(np.int32, copy=False)
    n_groups = int(old_in_first_order.size)
    stable_rows = np.argsort(dense, kind="stable").astype(np.int64)
    sizes = np.bincount(dense, minlength=n_groups).astype(np.int32)
    stops = np.cumsum(sizes, dtype=np.int64)
    members = np.split(stable_rows, stops[:-1])

    # LPT is deterministic and has a useful worst-case load bound, while the
    # number of centroids here makes its host cost negligible.
    packing_order = sorted(range(n_groups), key=lambda g: (-int(sizes[g]), g))
    shard_groups: list[list[int]] = [[] for _ in range(nshard)]
    loads = np.zeros((nshard,), dtype=np.int64)
    group_owner = np.empty((n_groups,), dtype=np.int32)
    load_heap = [(0, shard) for shard in range(nshard)]
    heapq.heapify(load_heap)
    for group in packing_order:
        load, owner = heapq.heappop(load_heap)
        shard_groups[owner].append(group)
        group_owner[group] = owner
        loads[owner] = load + int(sizes[group])
        heapq.heappush(load_heap, (int(loads[owner]), owner))

    shard_size = int(loads.max())
    n_padded = nshard * shard_size
    packed_to_canonical = np.full((n_padded,), -1, dtype=np.int64)
    packed_group_id = np.full((n_padded,), n_groups, dtype=np.int32)
    group_start = np.empty((n_groups,), dtype=np.int64)
    for owner, groups in enumerate(shard_groups):
        cursor = owner * shard_size
        for group in groups:
            rows = members[group]
            stop = cursor + int(rows.size)
            packed_to_canonical[cursor:stop] = rows
            packed_group_id[cursor:stop] = group
            group_start[group] = cursor
            cursor = stop
        if cursor != owner * shard_size + int(loads[owner]):
            raise AssertionError("grouped layout cursor/load disagreement")

    canonical_to_packed = np.empty((labels.size,), dtype=np.int64)
    active = packed_to_canonical >= 0
    canonical_to_packed[packed_to_canonical[active]] = np.flatnonzero(active)

    out = object.__new__(GroupedShardLayout)
    fields = {
        "n_logical": int(labels.size),
        "n_groups": n_groups,
        "n_shards": nshard,
        "shard_size": shard_size,
        "n_padded": n_padded,
        "packed_to_canonical": _readonly(packed_to_canonical, np.int64),
        "canonical_to_packed": _readonly(canonical_to_packed, np.int64),
        "canonical_group_id": _readonly(dense, np.int32),
        "packed_group_id": _readonly(packed_group_id, np.int32),
        "active_mask": _readonly(active, bool),
        "group_owner": _readonly(group_owner, np.int32),
        "group_start": _readonly(group_start, np.int64),
        "group_size": _readonly(sizes, np.int32),
        "shard_load": _readonly(loads, np.int64),
    }
    for name, value in fields.items():
        object.__setattr__(out, name, value)
    return validate_grouped_shard_layout(out)


def build_square_grouped_shard_layout(
    group_id,
    mesh_shape: tuple[int, int],
) -> SquareGroupedShardLayout:
    """Build the single hierarchical group layout for a square 2-D mesh."""
    shape = tuple(int(x) for x in mesh_shape)
    if len(shape) != 2 or shape[0] < 1 or shape[0] != shape[1]:
        raise ValueError(
            "build_square_grouped_shard_layout requires a positive square "
            f"mesh shape; got {shape}.")
    side = shape[0]
    fine = build_grouped_shard_layout(group_id, side * side)
    out = object.__new__(SquareGroupedShardLayout)
    object.__setattr__(out, "side", side)
    object.__setattr__(out, "fine", fine)
    return out


__all__ = [
    "GroupedShardLayout",
    "SquareGroupedShardLayout",
    "build_grouped_shard_layout",
    "build_square_grouped_shard_layout",
    "validate_grouped_shard_layout",
]

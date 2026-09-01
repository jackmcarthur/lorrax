"""Deterministic whole-group packing for a sharded array axis.

Scientific indices stay in their canonical, process-count-independent order.
This module derives a runtime view in which every labelled group occupies one
contiguous interval on one shard.  It knows no centroid or symmetry physics:
``group_id`` may describe any partition whose members must remain local.

The layout is deliberately per array axis.  A two-dimensional ``(x, y)``
kernel may use one layout made for ``mesh.shape['x']`` on its row index and a
second layout made for ``mesh.shape['y']`` on its column index.  A flattened
selector can independently request ``mesh.size`` shards.  Files and user-
visible indices never inherit any of those runtime orders.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _readonly(array, dtype=None) -> np.ndarray:
    out = np.asarray(array, dtype=dtype)
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
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

    @property
    def n_pad(self) -> int:
        return self.n_padded - self.n_logical

    @property
    def pad_fraction(self) -> float:
        return self.n_pad / self.n_padded

    def pack(self, array, *, axis: int = 0, fill_value=0):
        """Permute one canonical host axis and insert this layout's pads."""
        src = np.asarray(array)
        axis = int(axis) % src.ndim
        if int(src.shape[axis]) != self.n_logical:
            raise ValueError(
                "GroupedShardLayout.pack: canonical axis has extent "
                f"{src.shape[axis]}, expected {self.n_logical}.")
        src0 = np.moveaxis(src, axis, 0)
        dst0 = np.full(
            (self.n_padded,) + src0.shape[1:], fill_value, dtype=src.dtype)
        active = self.packed_to_canonical >= 0
        dst0[active] = src0[self.packed_to_canonical[active]]
        return np.moveaxis(dst0, 0, axis)

    def unpack(self, array, *, axis: int = 0):
        """Drop pads and restore one packed host axis to canonical order."""
        src = np.asarray(array)
        axis = int(axis) % src.ndim
        if int(src.shape[axis]) != self.n_padded:
            raise ValueError(
                "GroupedShardLayout.unpack: packed axis has extent "
                f"{src.shape[axis]}, expected {self.n_padded}.")
        src0 = np.moveaxis(src, axis, 0)
        dst0 = src0[self.canonical_to_packed]
        return np.moveaxis(dst0, 0, axis)

    def pack_permutations(self, permutations) -> np.ndarray:
        """Conjugate canonical gather maps into the packed, local view.

        ``permutations[s, i]`` is the source row gathered for target ``i``.
        Pad rows map to themselves.  A map that crosses a shard boundary is
        refused: that means the supplied group partition is not closed under
        the action and a supposedly local symmetry gather would communicate.
        """
        perm = np.asarray(permutations)
        if perm.ndim != 2 or int(perm.shape[1]) != self.n_logical:
            raise ValueError(
                "GroupedShardLayout.pack_permutations: expected shape "
                f"(n_map, {self.n_logical}); got {perm.shape}.")
        if perm.dtype.kind not in "iu":
            raise ValueError(
                "GroupedShardLayout.pack_permutations: maps must be integer; "
                f"got dtype {perm.dtype}.")
        perm = perm.astype(np.int64, copy=False)
        if perm.size and (int(perm.min()) < 0
                          or int(perm.max()) >= self.n_logical):
            raise ValueError(
                "GroupedShardLayout.pack_permutations: a map leaves the "
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
                    "GroupedShardLayout.pack_permutations: canonical map row "
                    f"{row_i} is not a permutation.")
        owner = target // self.shard_size
        crossing = owner[None, :] != owner[out]
        if np.any(crossing):
            row_i, packed_i = np.argwhere(crossing)[0]
            canonical_i = int(self.packed_to_canonical[packed_i])
            raise ValueError(
                "GroupedShardLayout.pack_permutations: map row "
                f"{int(row_i)} moves canonical row {canonical_i} from shard "
                f"{int(owner[packed_i])} to {int(owner[out[row_i, packed_i]])}; "
                "the group partition is not closed under the map.")
        return out.astype(np.int32)


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
    first_by_label: dict[int, int] = {}
    for row, raw in enumerate(labels.tolist()):
        first_by_label.setdefault(int(raw), row)
    ordered_labels = sorted(first_by_label, key=first_by_label.get)
    dense_of = {raw: dense for dense, raw in enumerate(ordered_labels)}
    dense = np.fromiter((dense_of[int(raw)] for raw in labels),
                        dtype=np.int32, count=labels.size)
    n_groups = len(ordered_labels)
    members = [np.flatnonzero(dense == group).astype(np.int64)
               for group in range(n_groups)]
    sizes = np.asarray([rows.size for rows in members], dtype=np.int32)

    # LPT is deterministic and has a useful worst-case load bound, while the
    # number of centroids here makes its host cost negligible.
    packing_order = sorted(range(n_groups), key=lambda g: (-int(sizes[g]), g))
    shard_groups: list[list[int]] = [[] for _ in range(nshard)]
    loads = np.zeros((nshard,), dtype=np.int64)
    group_owner = np.empty((n_groups,), dtype=np.int32)
    for group in packing_order:
        owner = int(np.argmin(loads))
        shard_groups[owner].append(group)
        group_owner[group] = owner
        loads[owner] += int(sizes[group])

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

    # Construction assertions are cheap and make the descriptor trustworthy
    # to communication kernels that cannot afford defensive host reasoning.
    if np.unique(packed_to_canonical[active]).size != labels.size:
        raise AssertionError("grouped layout lost or duplicated a canonical row")
    for group in range(n_groups):
        start = int(group_start[group])
        stop = start + int(sizes[group])
        if not np.all(packed_group_id[start:stop] == group):
            raise AssertionError("grouped layout split a group")
        if start // shard_size != (stop - 1) // shard_size:
            raise AssertionError("grouped layout crossed a shard boundary")

    return GroupedShardLayout(
        n_logical=int(labels.size),
        n_groups=n_groups,
        n_shards=nshard,
        shard_size=shard_size,
        n_padded=n_padded,
        packed_to_canonical=_readonly(packed_to_canonical, np.int64),
        canonical_to_packed=_readonly(canonical_to_packed, np.int64),
        canonical_group_id=_readonly(dense, np.int32),
        packed_group_id=_readonly(packed_group_id, np.int32),
        active_mask=_readonly(active, bool),
        group_owner=_readonly(group_owner, np.int32),
        group_start=_readonly(group_start, np.int64),
        group_size=_readonly(sizes, np.int32),
        shard_load=_readonly(loads, np.int64),
    )


__all__ = ["GroupedShardLayout", "build_grouped_shard_layout"]

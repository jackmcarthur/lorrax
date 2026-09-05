"""The one in-memory centroid order: whole symmetry orbits per shard.

Every centroid axis a gwjax run computes on (ψ faces, Z_q, C_q, ζ(μ,G), V,
χ0, W, pole residues, G, Σ) is in the ORBIT-PACKED order of
:mod:`common.grouped_layout`: complete symmetry orbits live on one X (or Y)
shard so every symmetry action is a rank-local gather, and each shard ends in
exact-zero pad slots. Files keep the CANONICAL centroid-file order at the
logical extent. Their in-memory I/O staging carrier is suffix padded to the
mesh multiple (``runtime.padding.padded_mu_axis``); no processor padding is
persisted.  This object owns the packed order
and the two conversions, which are legal only at the I/O seam: a reader
packs what it read, a writer unpacks what it is about to write.  Nothing
between the seams converts.

Each device conversion uses an all-to-all round trip per axis. The extent
change is a rank-local prefix pad/crop; a temporary split-axis pad can be
needed for divisibility. No all-gather of the input carrier is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec as P

from common.grouped_layout import (
    SquareGroupedShardLayout,
    build_square_grouped_shard_layout,
    identity_square_grouped_shard_layout,
)
from runtime.padding import PaddedAxis, padded_mu_extent, spec_divisor


def _permute_sharded_axis(arr, axis, source_map, mesh, spec, *, pad_to=None,
                          crop_to=None):
    """Permute one mesh-sharded axis by a global destination→source map
    without replicating it.  ``pad_to``/``crop_to`` are LOCAL extents applied
    inside the shard before/after the permutation."""
    from common.shard_map import shard_map
    arr = jnp.asarray(arr)
    ndim = int(arr.ndim)
    axis = int(axis) % ndim
    names = spec[axis] if axis < len(spec) else None
    if names is None:
        raise ValueError(
            f"centroid basis: axis {axis} of spec {spec} is not sharded.")
    names = (names,) if isinstance(names, str) else tuple(names)
    n_shards = int(np.prod([int(mesh.shape[n]) for n in names]))
    axis_name = names[0] if len(names) == 1 else names
    local = [int(arr.shape[i]) // int(spec_divisor(mesh, spec, i))
             for i in range(ndim)]
    if ndim < 2:
        raise ValueError("centroid basis: sharded conversion needs a second axis")
    split = max((i for i in range(ndim) if i != axis), key=lambda i: local[i])
    source = np.asarray(source_map, dtype=np.int32)

    def _pad(x, ax, target):
        widths = [(0, 0)] * ndim
        widths[ax] = (0, int(target) - int(x.shape[ax]))
        return jnp.pad(x, widths)

    def body(x):
        if pad_to is not None:
            x = _pad(x, axis, pad_to)
        n_split = int(x.shape[split])
        n_split_pad = -(-n_split // n_shards) * n_shards
        if n_split_pad != n_split:
            x = _pad(x, split, n_split_pad)
        x = jax.lax.all_to_all(x, axis_name, split_axis=split,
                               concat_axis=axis, tiled=True)
        x = jnp.take(x, jnp.asarray(source), axis=axis)
        x = jax.lax.all_to_all(x, axis_name, split_axis=axis,
                               concat_axis=split, tiled=True)
        if n_split_pad != n_split:
            x = jax.lax.slice_in_dim(x, 0, n_split, axis=split)
        if crop_to is not None:
            x = jax.lax.slice_in_dim(x, 0, int(crop_to), axis=axis)
        return x

    return shard_map(body, mesh=mesh, in_specs=spec, out_specs=spec,
                     check_vma=False)(arr)


@dataclass(frozen=True, eq=False)
class PackedCentroidBasis:
    """Canonical centroid table, its orbit-packed runtime layout, and the
    conversions between the two carriers.  ``eq=False``: identity-keyed
    orchestration metadata, like the k-unfold plan that borrows its layout."""

    mesh_xy: Mesh
    layout: SquareGroupedShardLayout
    canonical_indices: np.ndarray
    n_canonical: int

    @classmethod
    def build(cls, centroid_indices, sym, fft_grid, mesh_xy: Mesh, *,
              identity: bool = False) -> "PackedCentroidBasis":
        """Orbits of ``centroid_indices`` under ``sym``'s spatial operations
        (time reversal extended).  A set that is not orbit-closed, a trivial
        group, or ``identity=True`` gives the identity layout: the packed
        order IS the canonical suffix-padded order and every conversion is a
        no-op."""
        idx = np.array(jax.device_get(centroid_indices), dtype=np.int32,
                       order="C", copy=True)
        idx.setflags(write=False)
        shape = tuple(int(mesh_xy.shape[a]) for a in ('x', 'y'))
        groups = None
        if not identity:
            from symmetry_maps import (
                centroid_source_map_and_wrap, permutation_orbit_labels)
            n_spatial = int(np.asarray(sym.sym_matrices).shape[0])
            try:
                perm, _ = centroid_source_map_and_wrap(
                    idx, np.asarray(sym.sym_matrices)[:n_spatial],
                    np.asarray(sym.translations)[:n_spatial],
                    np.asarray(fft_grid, dtype=np.int32), extend_trs=True)
                groups = permutation_orbit_labels(perm)
            except (ValueError, RuntimeError):
                # not orbit-closed: nothing to pack, keep the canonical order
                groups = None
            if groups is not None and int(groups.max()) + 1 == int(idx.shape[0]):
                groups = None
        n_canonical = int(padded_mu_extent(int(idx.shape[0]), mesh_xy))
        layout = (
            identity_square_grouped_shard_layout(
                int(idx.shape[0]), n_canonical, shape)
            if groups is None
            else build_square_grouped_shard_layout(groups, shape))
        return cls(mesh_xy=mesh_xy, layout=layout, canonical_indices=idx,
                   n_canonical=n_canonical)

    # ---- extents and tables ------------------------------------------------
    @property
    def n_logical(self) -> int:
        return int(self.layout.axis.n_logical)

    @property
    def n_packed(self) -> int:
        return int(self.layout.axis.n_padded)

    @property
    def is_identity(self) -> bool:
        p2c = self.layout.axis.packed_to_canonical
        return (self.n_packed == self.n_canonical
                and bool(np.array_equal(p2c[:self.n_logical],
                                        np.arange(self.n_logical))))

    @property
    def active_mask(self) -> np.ndarray:
        return self.layout.axis.active_mask

    @property
    def packed_indices(self) -> np.ndarray:
        """The centroid table in packed order.  Pad slots repeat row 0; the
        loader zeroes them (:attr:`active_mask`)."""
        out = self.layout.axis.pack_host(
            self.canonical_indices, axis=0, fill_value=0)
        out[~self.active_mask] = self.canonical_indices[0]
        return out

    def describe(self) -> str:
        ax = self.layout.axis
        if self.is_identity:
            return (f"centroid order: canonical (identity layout), "
                    f"{self.n_logical} centroids on a carrier of {self.n_packed}")
        return (f"centroid order: orbit-packed, {ax.n_groups} orbits on "
                f"{ax.n_shards} shards of {ax.shard_size} slots "
                f"({self.n_logical} centroids, {ax.n_pad} pad slots, "
                f"canonical carrier {self.n_canonical})")

    def pack_tables(self, sym_perm, L_table):
        """Canonical centroid permutations/wraps → packed coordinates."""
        return (self.layout.axis.pack_permutations_host(sym_perm),
                self.layout.axis.pack_host(L_table, axis=1, fill_value=0))

    def unpack_tables(self, sym_perm_packed, L_table_packed):
        ax = self.layout.axis
        perm = np.asarray(sym_perm_packed)
        canonical = ax.packed_to_canonical[perm[:, ax.canonical_to_packed]]
        return canonical.astype(np.int32), ax.unpack_host(L_table_packed, axis=1)

    # ---- host arrays -------------------------------------------------------
    def pack_host(self, array, *, axis: int = -1):
        """Canonical (logical or suffix-padded) host axis → packed."""
        src = np.asarray(array)
        axis = int(axis) % src.ndim
        if int(src.shape[axis]) != self.n_logical:
            src = np.take(src, np.arange(self.n_logical), axis=axis)
        return self.layout.axis.pack_host(src, axis=axis, fill_value=0)

    def unpack_host(self, array, *, axis: int = -1):
        """Packed host axis → canonical logical extent."""
        return self.layout.axis.unpack_host(np.asarray(array), axis=axis)

    # ---- device arrays (the I/O seam) --------------------------------------
    @property
    def solve_axis(self) -> PaddedAxis:
        """Dense-solve receipt; interleaved pads are inside the solve extent.

        The active-slot mask remains owned by the layout. A prefix receipt
        must never reinterpret it as the first ``n_logical`` packed slots.
        """
        return PaddedAxis(
            name="centroid solve", logical=(self.n_logical if self.is_identity
                                            else self.n_packed),
            carrier=self.n_packed, divisor=int(self.mesh_xy.size))

    @lru_cache(maxsize=None)
    def _maps(self, n_shards: int):
        """Inverse permutations on a common, locally padded I/O carrier.

        Either order can have the larger extent. Embed each original shard
        as its own prefix in the common carrier, map physical rows by the
        layout, and pair the remaining zero slots bijectively. Both extent
        changes are then local pad/crop operations around one permutation.
        """
        extent = max(self.n_canonical, self.n_packed)
        if any(n % n_shards for n in (self.n_canonical, self.n_packed)):
            raise ValueError("centroid basis: carrier must divide the shard count")
        slots = np.arange(extent, dtype=np.int32).reshape(n_shards, -1)
        canonical = slots[:, :self.n_canonical // n_shards].reshape(-1)
        packed = slots[:, :self.n_packed // n_shards].reshape(-1)
        dst = canonical[:self.n_logical]
        src = packed[self.layout.axis.canonical_to_packed]
        dst_pad = np.ones(extent, dtype=bool)
        src_pad = np.ones(extent, dtype=bool)
        dst_pad[dst] = False
        src_pad[src] = False
        unpack = np.empty(extent, dtype=np.int32)
        unpack[dst] = src
        unpack[dst_pad] = np.flatnonzero(src_pad)
        return unpack, np.argsort(unpack).astype(np.int32)

    @staticmethod
    def _spec(arr, spec):
        if spec is not None:
            return spec
        return arr.sharding.spec

    def _shards(self, spec, axis):
        return int(spec_divisor(self.mesh_xy, spec, axis))

    def pack_axis(self, arr, axis: int, *, spec=None):
        """Canonical suffix-padded carrier → packed, on one sharded axis."""
        if self.is_identity:
            return arr
        spec = self._spec(arr, spec)
        axis = int(axis) % int(arr.ndim)
        if int(arr.shape[axis]) != self.n_canonical:
            raise ValueError(
                f"centroid basis: pack_axis expects the canonical carrier "
                f"{self.n_canonical} on axis {axis}; got {arr.shape}.")
        n_shards = self._shards(spec, axis)
        _, pack = self._maps(n_shards)
        return _permute_sharded_axis(
            arr, axis, pack, self.mesh_xy, spec,
            pad_to=max(self.n_canonical, self.n_packed) // n_shards,
            crop_to=self.n_packed // n_shards)

    def unpack_axis(self, arr, axis: int, *, spec=None):
        """Packed → canonical suffix-padded carrier, on one sharded axis."""
        if self.is_identity:
            return arr
        spec = self._spec(arr, spec)
        axis = int(axis) % int(arr.ndim)
        if int(arr.shape[axis]) != self.n_packed:
            raise ValueError(
                f"centroid basis: unpack_axis expects the packed carrier "
                f"{self.n_packed} on axis {axis}; got {arr.shape}.")
        n_shards = self._shards(spec, axis)
        unpack, _ = self._maps(n_shards)
        return _permute_sharded_axis(
            arr, axis, unpack, self.mesh_xy, spec,
            pad_to=max(self.n_canonical, self.n_packed) // n_shards,
            crop_to=self.n_canonical // n_shards)

    def pack_operator(self, op, *, spec=None):
        """Both centroid axes (the last two) of a ``(..., mu, nu)`` operator."""
        spec = self._spec(op, spec)
        return self.pack_axis(self.pack_axis(op, -2, spec=spec), -1, spec=spec)

    def unpack_operator(self, op, *, spec=None):
        spec = self._spec(op, spec)
        return self.unpack_axis(
            self.unpack_axis(op, -2, spec=spec), -1, spec=spec)


__all__ = ["PackedCentroidBasis"]

"""Raw-parent k contractions on an orbit-packed centroid basis.

This module is the GW adapter between three existing owners:

* :class:`symmetry_maps.SymMaps` owns the typed full-k parent/action tables;
* :mod:`symmetry_maps` owns centroid pullbacks and operator transport;
* :mod:`common.grouped_layout` owns reversible whole-orbit packing.

It contains no independent symmetry algebra.  A plan is immutable host
metadata plus small device helpers: pack a centroid axis once, contract on
raw WFN parent k rows, then unfold the resulting two-endpoint operator.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.grouped_layout import (
    SquareGroupedShardLayout,
    build_square_grouped_shard_layout,
)
from symmetry_maps import (
    centroid_source_map_and_wrap,
    permutation_orbit_labels,
    unfold_spin_centroid_operator,
)


def _readonly(value, dtype) -> np.ndarray:
    out = np.array(value, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True, eq=False)
class CentroidKUnfoldPlan:
    """Authenticated raw-parent/full-k transport for one centroid basis.

    ``eq=False`` deliberately gives identity hashing.  A plan is orchestration
    metadata closed over by compiled kernels, not numerical pytree data; two
    independently authenticated plans must not share a specialization merely
    because their small host tables happen to compare equal.
    """

    mesh_xy: Mesh
    layout: SquareGroupedShardLayout
    irr_idx: np.ndarray
    sym_idx: np.ndarray
    sym_perm: np.ndarray
    L_table: np.ndarray
    k_parent_frac: np.ndarray
    spin_action_full: np.ndarray
    n_sym_spatial: int
    nspinor: int

    @property
    def n_parent(self) -> int:
        return int(self.k_parent_frac.shape[0])

    @property
    def n_full(self) -> int:
        return int(self.irr_idx.shape[0])

    @property
    def n_centroid_logical(self) -> int:
        return int(self.layout.axis.n_logical)

    @property
    def n_centroid_packed(self) -> int:
        return int(self.layout.axis.n_padded)

    def parent_rows(self, array, *, axis: int = 0):
        """Select raw-parent rows from a full-k scalar table.

        This helper is only for quantities such as energies and occupations
        that are invariant within a star.  Wavefunctions must be loaded from
        the raw WFN parent rows; selecting file-wedge rows from an unfolded
        wavefunction is not equivalent when that row carries a nonidentity or
        antiunitary action.
        """
        src = jnp.asarray(array)
        axis = int(axis) % src.ndim
        if int(src.shape[axis]) != self.n_full:
            raise ValueError(
                "CentroidKUnfoldPlan.parent_rows: full-k axis has extent "
                f"{src.shape[axis]}, expected {self.n_full}.")
        # irr_idx maps full rows to raw WFN parent rows.  Several raw rows may
        # be symmetry-redundant, but every parent used by a child carries the
        # same scalar value.  Scatter picks one child per raw parent without
        # imposing a star-wedge gauge on wavefunctions.
        source = np.full((self.n_parent,), -1, dtype=np.int32)
        for full, parent in enumerate(self.irr_idx):
            source[int(parent)] = int(full)
        unused = source < 0
        if np.any(unused):
            # Unused raw WFN rows have no full-zone consumer.  Their values
            # are immaterial but keeping the parent carrier rectangular makes
            # the direct irr_idx gather simple and stable.  Fill from row zero.
            source[unused] = 0
        return jnp.take(src, jnp.asarray(source), axis=axis)

    def pack_centroid_axis(self, array, *, axis: int, spec: P):
        """Pack one canonical centroid axis and zero every padding row."""
        src = jnp.asarray(array)
        axis = int(axis) % src.ndim
        logical = self.n_centroid_logical
        if int(src.shape[axis]) < logical:
            raise ValueError(
                "CentroidKUnfoldPlan.pack_centroid_axis: source centroid "
                f"extent {src.shape[axis]} is smaller than {logical}.")
        packed_to_canonical = self.layout.axis.packed_to_canonical
        active = packed_to_canonical >= 0
        gather = np.where(active, packed_to_canonical, 0).astype(np.int32)
        out = jnp.take(src, jnp.asarray(gather), axis=axis)
        mask_shape = [1] * out.ndim
        mask_shape[axis] = self.n_centroid_packed
        out = jnp.where(
            jnp.asarray(active).reshape(mask_shape), out,
            jnp.asarray(0, dtype=out.dtype))
        return jax.lax.with_sharding_constraint(
            out, NamedSharding(self.mesh_xy, spec))

    def pack_face_pair(self, psi_nmu, psi_mun):
        """Pack raw-parent face operands without changing their roles."""
        if int(psi_nmu.shape[0]) != self.n_parent:
            raise ValueError(
                "CentroidKUnfoldPlan.pack_face_pair: psi_nmu k extent "
                f"{psi_nmu.shape[0]} != n_parent={self.n_parent}.")
        if int(psi_mun.shape[0]) != self.n_parent:
            raise ValueError(
                "CentroidKUnfoldPlan.pack_face_pair: psi_mun k extent "
                f"{psi_mun.shape[0]} != n_parent={self.n_parent}.")
        return (
            self.pack_centroid_axis(
                psi_nmu, axis=3, spec=P(None, 'x', None, 'y')),
            self.pack_centroid_axis(
                psi_mun, axis=2, spec=P(None, None, 'x', 'y')),
        )

    def unfold_operator(self, operator_parent):
        """Transport ``(k_parent,s,mu,s,nu)`` to full k locally."""
        return unfold_spin_centroid_operator(
            operator_parent,
            irr_idx=self.irr_idx,
            sym_idx=self.sym_idx,
            sym_perm=self.sym_perm,
            L_table=self.L_table,
            k_irr_frac=self.k_parent_frac,
            spin_action_full=self.spin_action_full,
            n_sym_spatial=self.n_sym_spatial,
            mesh_xy=self.mesh_xy,
            logical_centroid_extent=self.n_centroid_logical,
            axis_local=True,
        )


def build_centroid_k_unfold_plan(
    sym,
    centroid_fft_idx,
    fft_grid,
    mesh_xy: Mesh,
    *,
    nspinor: int,
    parent_k_frac=None,
) -> CentroidKUnfoldPlan:
    """Bind canonical symmetry tables to one orbit-packed centroid basis.

    ``parent_k_frac`` is the raw WFN k table.  When omitted, the exact file
    wedge rows owned by ``SymMaps.kirr_fullids`` are used; that mapping is
    coordinate-authenticated and ordered like the raw WFN.  It is used only
    for Bloch phases here, never as a source of parent wavefunctions.
    """
    shape = tuple(int(mesh_xy.shape[a]) for a in ('x', 'y'))
    if shape[0] != shape[1]:
        raise ValueError(
            "build_centroid_k_unfold_plan requires the GW square mesh; "
            f"got {shape}.")
    ns = int(nspinor)
    if ns not in (1, 2):
        raise ValueError(
            "build_centroid_k_unfold_plan currently supports scalar and "
            f"two-component spinor wavefunctions; got nspinor={ns}.  "
            "Four-component kinetic-balance states require their exact "
            "Dirac representation at this seam and remain on full k.")

    n_spatial = int(np.asarray(sym.sym_matrices).shape[0])
    sym_perm, wraps = centroid_source_map_and_wrap(
        np.asarray(centroid_fft_idx, dtype=np.int32),
        np.asarray(sym.sym_matrices)[:n_spatial],
        np.asarray(sym.translations)[:n_spatial],
        np.asarray(fft_grid, dtype=np.int32),
        extend_trs=True,
    )
    groups = permutation_orbit_labels(sym_perm)
    layout = build_square_grouped_shard_layout(groups, shape)
    packed_perm = layout.axis.pack_permutations_host(sym_perm)
    packed_wraps = layout.axis.pack_host(wraps, axis=1, fill_value=0)

    irr = np.asarray(sym.irr_idx_k, dtype=np.int32)
    sym_idx = np.asarray(sym.sym_idx_k, dtype=np.int32)
    if irr.ndim != 1 or sym_idx.shape != irr.shape:
        raise ValueError(
            "build_centroid_k_unfold_plan: SymMaps k tables must have the "
            f"same rank-one shape; got {irr.shape}/{sym_idx.shape}.")
    if parent_k_frac is None:
        parent_k_frac = np.asarray(sym.unfolded_kpts)[
            np.asarray(sym.kirr_fullids, dtype=np.int32)]
    parent_k = np.asarray(parent_k_frac, dtype=np.float64)
    if parent_k.ndim != 2 or parent_k.shape[1] != 3:
        raise ValueError(
            "build_centroid_k_unfold_plan: parent_k_frac must be "
            f"(n_parent,3); got {parent_k.shape}.")
    if irr.size and (int(irr.min()) < 0 or int(irr.max()) >= parent_k.shape[0]):
        raise ValueError(
            "build_centroid_k_unfold_plan: irr_idx_k addresses outside the "
            f"raw parent table of length {parent_k.shape[0]}.")
    spin = np.asarray(sym.spinor_action(sym_idx, nspinor=ns),
                      dtype=np.complex128)

    return CentroidKUnfoldPlan(
        mesh_xy=mesh_xy,
        layout=layout,
        irr_idx=_readonly(irr, np.int32),
        sym_idx=_readonly(sym_idx, np.int32),
        sym_perm=_readonly(packed_perm, np.int32),
        L_table=_readonly(packed_wraps, np.int64),
        k_parent_frac=_readonly(parent_k, np.float64),
        spin_action_full=_readonly(spin, np.complex128),
        n_sym_spatial=n_spatial,
        nspinor=ns,
    )


__all__ = ["CentroidKUnfoldPlan", "build_centroid_k_unfold_plan"]

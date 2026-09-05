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

from ffi import _services

_services.ensure_on_path()

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.grouped_layout import (
    SquareGroupedShardLayout,
    build_grouped_shard_layout,
    build_square_grouped_shard_layout,
)
from symmetry_maps import (
    centroid_source_map_and_wrap,
    permutation_orbit_labels,
    real_space_orbit_labels,
    reorder_isdf_operator_basis,
    unfold_spin_centroid_operator,
)


# P=4 A100 crossover after the fixed-size spin action was made dot-free and
# canonicalization was postponed to one completed chi operator.  This is an
# internal scheduling policy, not a user-tunable physics/runtime knob.  The
# lower boundary is the measured 64->8, four-band case; require a separate 2x
# k-reduction floor so an almost-unreduced grid cannot satisfy the scalar work
# proxy merely by carrying many bands.
_MIN_AVOIDED_BAND_WORK = 3.5
_MIN_PARENT_K_REDUCTION = 2.0


def parent_k_contraction_profitable(
    *, n_full: int, n_parent: int, n_bands: int,
) -> bool:
    """Conservative automatic admission for parent-k Green contractions.

    The saved band contraction is proportional to
    ``n_bands * (1 - n_parent/n_full)``; the required full-k operator
    transport is not.  On the real Si P=4 64->8 geometry, the dot-free
    transport kept complete eight-node chi builds faster at 4, 8 and 16
    bands.  Four bands gives the boundary score 3.5.  A distinct 2x reduction
    floor prevents extrapolation to grids with almost no symmetry reduction.
    Callers additionally restrict this measured policy to GPU execution.
    """
    full = int(n_full)
    parent = int(n_parent)
    bands = int(n_bands)
    if full < 1 or parent < 1 or parent >= full or bands < 1:
        return False
    if full / parent < _MIN_PARENT_K_REDUCTION:
        return False
    avoided = bands * (full - parent) / full
    return avoided >= _MIN_AVOIDED_BAND_WORK


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
    canonical_centroid_extent: int
    #: The spatial Seitz rows and FFT grid the centroid tables were built
    #: from, kept so real-grid tiles for the ζ-fit RHS are closed under the
    #: SAME action.  ``None`` only on hand-assembled test plans.
    spatial_ops: np.ndarray | None = None
    translations: np.ndarray | None = None
    fft_grid: np.ndarray | None = None
    #: The full-k row that IS raw parent ``i`` (``SymMaps.kirr_fullids``): a
    #: full-k operator selected on these rows is the raw-gauge operator a
    #: band projection with the raw parent wavefunctions expects.  ``sym``
    #: is the typed table source for the band-index broadcast back to full
    #: k (``symmetry_maps.unfold_file_wedge_band_operator``).  ``None`` only on
    #: hand-assembled test plans.
    parent_full_rows: np.ndarray | None = None
    sym: object = None

    @property
    def n_parent(self) -> int:
        return int(self.k_parent_frac.shape[0])

    @property
    def centroid_local_perm(self) -> np.ndarray:
        """Owner-local gather offsets of the packed centroid source map.

        ``sym_perm`` never crosses an X shard (the grouped layout refused
        any orbit that would), so ``sym_perm % shard`` is the offset the
        manual-mode local unfold gathers with.  Same reduction
        :func:`unfold_spin_centroid_operator` performs for ``axis_local``.
        """
        return (self.sym_perm % int(self.layout.axis_shard_size)).astype(
            np.int32)

    def real_grid_tiles(self, *, target_width: int) -> "RealGridOrbitTiles":
        """Orbit-closed real-grid tiles for this plan's Y axis and group."""
        if self.spatial_ops is None or self.fft_grid is None:
            raise ValueError(
                "CentroidKUnfoldPlan.real_grid_tiles: this plan carries no "
                "spatial operations/FFT grid (hand-assembled test plan).")
        return build_real_grid_orbit_tiles(
            self.spatial_ops, self.translations, self.fft_grid,
            n_y=int(self.mesh_xy.shape['y']),
            target_width=int(target_width),
            shard_multiple=int(self.mesh_xy.shape['x']))

    def restore_left_basis(self, operator_packed):
        """Packed→canonical on the LEFT (centroid) axis of ``(q, mu, r)``.

        The ζ-fit RHS leaves its r endpoint in tile order (the solve does not
        care which r is which) and needs only its centroid axis in the
        canonical order the factor ``C`` was formed in.  One x-axis exchange,
        then the known-zero owner-pad rows are cropped, exactly as
        :meth:`finish_green` does on both axes.
        """
        packed = jnp.asarray(operator_packed)
        if packed.ndim != 3:
            raise ValueError(
                "CentroidKUnfoldPlan.restore_left_basis requires a rank-three "
                f"(q, mu, r) operator; got {packed.shape}.")
        reordered = reorder_isdf_operator_basis(
            packed,
            left_source_map=self._canonical_source_map(),
            right_source_map=None,
            mesh_xy=self.mesh_xy)
        canonical = reordered[:, :int(self.canonical_centroid_extent), :]
        return jax.lax.with_sharding_constraint(
            canonical, NamedSharding(self.mesh_xy, P(None, 'x', 'y')))

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

    @property
    def supports_canonical_bridge(self) -> bool:
        """Whether the canonical basis is a prefix after packed reordering.

        The square grouped layout pads to a complete-mesh multiple, while the
        canonical carrier uses the smallest complete-mesh multiple covering
        the logical rows.  Orbit imbalance can therefore make the packed
        extent larger, never smaller.  Reorder at the packed extent first;
        the surplus rows are then known-zero shard padding and may be cropped
        with both operator axes still distributed.
        """
        canonical = int(self.canonical_centroid_extent)
        complete_mesh = int(self.mesh_xy.size)
        from runtime.padding import padded_axis
        return (
            canonical >= self.n_centroid_logical
            and canonical <= self.n_centroid_packed
            and padded_axis(
                canonical, complete_mesh,
                name="canonical centroid bridge").carrier == canonical
            and padded_axis(
                self.n_centroid_packed, complete_mesh,
                name="packed centroid bridge").carrier
            == self.n_centroid_packed
        )

    def _canonical_source_map(self) -> np.ndarray:
        """Complete destination-to-source map, including padding rows."""
        canonical = int(self.canonical_centroid_extent)
        if (canonical < self.n_centroid_logical
                or canonical > self.n_centroid_packed):
            raise ValueError(
                "CentroidKUnfoldPlan: canonical centroid extent must cover "
                "the logical basis and not exceed the orbit-packed extent; "
                "got logical/packed/canonical="
                f"{self.n_centroid_logical}/{self.n_centroid_packed}/"
                f"{self.canonical_centroid_extent}.")
        from runtime.padding import authenticate_padded_axis
        authenticate_padded_axis(
            self.n_centroid_logical, canonical, self.mesh_xy,
            name="canonical centroid unfold carrier")
        authenticate_padded_axis(
            self.n_centroid_packed, self.n_centroid_packed, self.mesh_xy,
            name="orbit-packed centroid unfold carrier")
        extent = self.n_centroid_packed
        logical = self.n_centroid_logical
        source = np.empty((extent,), dtype=np.int32)
        source[:logical] = self.layout.axis.canonical_to_packed
        packed_pad = np.flatnonzero(~self.layout.axis.active_mask)
        if packed_pad.size != extent - logical:
            raise AssertionError("packed/canonical padding cardinality drift")
        source[logical:] = packed_pad
        return source

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
            # Grouped-layout padding is a suffix of EACH owner shard, not a
            # single global suffix.  The packed source maps permute those pad
            # rows among themselves and pack_centroid_axis made them exactly
            # zero, so the prefix-mask convention of the generic service is
            # neither needed nor correct here.  Treat the complete packed
            # extent as active for transport; finish_green removes the known
            # pad rows after restoring canonical order.
            logical_centroid_extent=self.n_centroid_packed,
            axis_local=True,
        )

    def finish_green(self, operator_parent):
        """Unfold parent k locally, then restore today's canonical basis.

        The one-time basis move is the bring-up bridge while V/W and
        projection remain in canonical centroid order.  It uses the symmetry
        service's volume-preserving two-axis permutation: every intermediate
        remains ``P(None,'x','y')`` and no rank materializes a complete
        centroid axis.  Once those consumers use this same packed plan, the
        bridge disappears without changing the Green contraction or symmetry
        algebra.
        """
        packed = self.unfold_operator(operator_parent)
        ns = int(self.nspinor)
        extent = self.n_centroid_packed
        source_mu = self._canonical_source_map()
        spin = np.arange(ns, dtype=np.int32)
        source_endpoint = (
            source_mu[:, None] * ns + spin[None, :]).reshape(extent * ns)
        flat_sharding = NamedSharding(self.mesh_xy, P(None, 'x', 'y'))
        flat = jax.lax.with_sharding_constraint(
            jnp.transpose(packed, (0, 2, 1, 4, 3)).reshape(
                self.n_full, extent * ns, extent * ns),
            flat_sharding)
        canonical_flat = self.restore_operator_basis(
            flat,
            source_map=source_endpoint,
            canonical_extent=int(self.canonical_centroid_extent) * ns)
        canonical_extent = int(self.canonical_centroid_extent)
        canonical = jnp.transpose(
            canonical_flat.reshape(
                self.n_full, canonical_extent, ns,
                canonical_extent, ns),
            (0, 2, 1, 4, 3))
        return jax.lax.with_sharding_constraint(
            canonical,
            NamedSharding(self.mesh_xy, P(None, None, 'x', None, 'y')))

    def restore_operator_basis(
        self,
        operator_packed,
        *,
        source_map=None,
        canonical_extent=None,
    ):
        """Restore one packed ``P(batch,X,Y)`` operator to canonical order.

        Reordering happens before cropping, at the complete packed extent.
        Thus the only nonlocal operation is the symmetry service's
        volume-preserving all-to-all backend; surplus owner-padding rows are
        discarded only after they have been moved to the global suffix.
        ``source_map``/``canonical_extent`` generalize the scalar-centroid
        operation to the merged ``(mu,spin)`` endpoints used by Green's
        functions.
        """
        packed = jnp.asarray(operator_packed)
        if packed.ndim != 3 or packed.shape[1] != packed.shape[2]:
            raise ValueError(
                "CentroidKUnfoldPlan.restore_operator_basis requires a "
                f"square rank-three operator; got {packed.shape}.")
        if source_map is None:
            source = self._canonical_source_map()
        else:
            source = np.asarray(source_map, dtype=np.int32)
        extent = int(packed.shape[1])
        if source.shape != (extent,):
            raise ValueError(
                "CentroidKUnfoldPlan.restore_operator_basis source map "
                f"must have shape ({extent},); got {source.shape}.")
        target = (
            int(self.canonical_centroid_extent)
            if canonical_extent is None else int(canonical_extent))
        if not 0 < target <= extent:
            raise ValueError(
                "CentroidKUnfoldPlan.restore_operator_basis canonical "
                f"extent must lie in (0,{extent}]; got {target}.")
        reordered = reorder_isdf_operator_basis(
            packed,
            left_source_map=source,
            right_source_map=source,
            mesh_xy=self.mesh_xy)
        canonical = reordered[:, :target, :target]
        return jax.lax.with_sharding_constraint(
            canonical, NamedSharding(self.mesh_xy, P(None, 'x', 'y')))


def build_centroid_k_unfold_plan(
    sym,
    centroid_fft_idx,
    fft_grid,
    mesh_xy: Mesh,
    *,
    nspinor: int,
    parent_k_frac=None,
    canonical_centroid_extent=None,
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
    canonical_extent = (
        int(len(centroid_fft_idx)) if canonical_centroid_extent is None
        else int(canonical_centroid_extent))
    from runtime.padding import padded_axis
    canonical_axis = padded_axis(
        int(len(centroid_fft_idx)), mesh_xy,
        name="canonical centroid unfold carrier")
    if canonical_extent != canonical_axis.carrier:
        raise ValueError(
            "build_centroid_k_unfold_plan: canonical centroid carrier is "
            f"{canonical_extent}, expected {canonical_axis.carrier} for "
            f"logical extent {canonical_axis.logical} and divisor "
            f"{canonical_axis.divisor}.")

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
        canonical_centroid_extent=canonical_extent,
        spatial_ops=_readonly(
            np.asarray(sym.sym_matrices)[:n_spatial], np.int64),
        translations=_readonly(
            np.asarray(sym.translations)[:n_spatial], np.float64),
        fft_grid=_readonly(np.asarray(fft_grid).reshape(3), np.int64),
        parent_full_rows=(
            _readonly(np.asarray(sym.kirr_fullids), np.int32)
            if getattr(sym, 'kirr_fullids', None) is not None
            and int(np.asarray(sym.kirr_fullids).shape[0]) == parent_k.shape[0]
            else None),
        sym=sym,
    )


@dataclass(frozen=True, eq=False)
class RealGridOrbitTiles:
    """Fixed-width real-grid tiles made of complete symmetry orbits.

    The ζ-fit RHS ``Z_q(mu, r)`` is streamed over its r endpoint.  A
    contiguous r slab is not closed under the point group, so a parent-k
    build of it would need sources outside the slab (recomputed, retained,
    or communicated — all three lose the parent-k saving).  These tiles are
    unions of complete orbits placed so that every orbit lies inside one Y
    owner: ``r_index[t, slot]`` is the flat C-order grid index held in
    packed slot ``slot`` of tile ``t`` (``-1`` marks an owner-local pad
    slot), and slot ``slot`` belongs to Y owner ``slot // shard_size``.
    The symmetry gather on the r endpoint is then local to each Y shard,
    exactly as the orbit-packed centroid gather is local to each X shard.
    Pads trail within each owner, hold exact zeros in every carrier and map
    to themselves under every operation.  All tiles share one width so one
    compiled kernel serves them; the tables below are runtime operands.
    """

    fft_grid: np.ndarray
    spatial_ops: np.ndarray
    translations: np.ndarray
    n_y: int
    shard_size: int
    r_index: np.ndarray

    @property
    def n_tiles(self) -> int:
        return int(self.r_index.shape[0])

    @property
    def width(self) -> int:
        return int(self.n_y) * int(self.shard_size)

    @property
    def n_rtot(self) -> int:
        return int(np.prod(self.fft_grid))

    def active_mask(self, tile: int) -> np.ndarray:
        return self.r_index[int(tile)] >= 0

    def source_tables(self, tile: int) -> tuple[np.ndarray, np.ndarray]:
        """Owner-local source map and lattice wrap for one tile.

        Returns ``(local_perm, wraps)``: ``(2·n_sym, width)`` int32 gather
        offsets inside each Y owner and ``(2·n_sym, width, 3)`` int8 wraps,
        rows ``[n_sym:]`` duplicating ``[:n_sym]`` (time reversal fixes r).
        The tables come from :func:`symmetry_maps.centroid_source_map_and_wrap`
        on the tile's active points — the one source-map owner — which
        also refuses if the tile is not closed.  Pads are fixed points.
        """
        row = self.r_index[int(tile)]
        slots = np.flatnonzero(row >= 0)
        flat = row[slots]
        fg = self.fft_grid
        pts = np.stack(
            [flat // (fg[1] * fg[2]), (flat // fg[2]) % fg[1], flat % fg[2]],
            axis=1).astype(np.int32)
        perm_active, wrap_active = centroid_source_map_and_wrap(
            pts, self.spatial_ops, self.translations, fg, extend_trs=True)
        n_rows = int(perm_active.shape[0])
        width = self.width
        perm = np.broadcast_to(
            np.arange(width, dtype=np.int64), (n_rows, width)).copy()
        perm[:, slots] = slots[perm_active]
        wraps = np.zeros((n_rows, width, 3), dtype=np.int8)
        wraps[:, slots] = wrap_active
        owner = np.arange(width) // int(self.shard_size)
        if np.any(owner[perm] != owner[None, :]):
            raise AssertionError(
                "RealGridOrbitTiles: an orbit crosses a Y owner; the "
                "grouped layout should have made that impossible.")
        return (perm % int(self.shard_size)).astype(np.int32), wraps


def build_real_grid_orbit_tiles(
    spatial_ops, translations, fft_grid, *, n_y: int, target_width: int,
    shard_multiple: int = 1,
) -> RealGridOrbitTiles:
    """Partition the FFT grid into orbit-closed tiles of one fixed width.

    Orbits are labelled by :func:`symmetry_maps.real_space_orbit_labels`
    (O(n_rtot) host memory) and placed by the same LPT packing that places
    centroid orbits (:func:`common.grouped_layout.build_grouped_shard_layout`)
    over ``n_tiles·n_y`` equal Y-owner shards; tile ``t`` is shards
    ``[t·n_y, (t+1)·n_y)``.  ``target_width`` is the memory planner's r
    chunk; the realized width is the smallest LPT placement whose padded
    owner extent fits it (one more tile is added while it does not), and it
    is a multiple of ``n_y·shard_multiple`` so the r axis stays a legal
    carrier for every downstream reshard.
    """
    ops = np.asarray(spatial_ops, dtype=np.int64)
    tau = np.asarray(translations, dtype=np.float64)
    fg = np.asarray(fft_grid, dtype=np.int64).reshape(3)
    n_y = int(n_y)
    multiple = int(shard_multiple)
    labels = real_space_orbit_labels(ops, tau, fg)
    n_rtot = int(labels.size)
    width = max(int(target_width), n_y * multiple)
    n_tiles = max(1, -(-n_rtot // width))
    while True:
        layout = build_grouped_shard_layout(
            labels, n_tiles * n_y, shard_size_multiple=multiple)
        if n_y * int(layout.shard_size) <= width or n_tiles * n_y >= n_rtot:
            break
        n_tiles += 1
    r_index = layout.packed_to_canonical.reshape(
        n_tiles, n_y * int(layout.shard_size))
    return RealGridOrbitTiles(
        fft_grid=_readonly(fg, np.int64),
        spatial_ops=_readonly(ops, np.int64),
        translations=_readonly(tau, np.float64),
        n_y=n_y,
        shard_size=int(layout.shard_size),
        r_index=_readonly(r_index, np.int64),
    )


__all__ = [
    "CentroidKUnfoldPlan",
    "RealGridOrbitTiles",
    "build_centroid_k_unfold_plan",
    "build_real_grid_orbit_tiles",
    "parent_k_contraction_profitable",
]

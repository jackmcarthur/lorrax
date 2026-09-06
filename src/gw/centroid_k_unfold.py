"""Raw-parent k contractions on an orbit-packed centroid basis.

This module is the GW adapter between three existing owners:

* :class:`symmetry_maps.SymMaps` owns the typed full-k parent/action tables;
* :mod:`symmetry_maps` owns centroid pullbacks and operator transport;
* :mod:`common.grouped_layout` owns reversible whole-orbit packing.

It contains no independent symmetry algebra.  A plan is immutable host
metadata plus one device helper: contract on raw WFN parent k rows in the
run's orbit-packed centroid order (``common.centroid_basis``), then unfold
the resulting two-endpoint operator to full k, still in that order.
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
    unfold_spin_centroid_operator,
    unfold_wavefunction_local,
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

    def wavefunction_unfold_tables(self) -> dict:
        """Host tables for ``symmetry_maps.unfold_wavefunction_local`` on a
        packed face: the parents' rows and operations of every full-k row,
        the parents' k, the owner-local centroid offsets and wraps over the
        COMPLETE packed endpoint, and the spinor representation.  A consumer
        streams children of a packed parent face from these without a full-k
        face ever being resident (the fractional-occupation pair scans)."""
        return dict(
            irr_idx=self.irr_idx, sym_idx=self.sym_idx,
            k_irr_frac=self.k_parent_frac,
            local_perm=self.centroid_local_perm, L_table=self.L_table,
            spin_action_full=self.spin_action_full,
            n_sym_spatial=int(self.n_sym_spatial))

    def unfold_face(self, face, *, vertex=0, spin_axis, mu_axis,
                    mesh_axis=None, tables=None):
        """Unfold a raw-parent face by the typed action, then apply its Lorentz vertex."""
        from common.gamma_matrices import gamma_apply, gamma_perm_phase

        t = self.wavefunction_unfold_tables() if tables is None else tables
        child = unfold_wavefunction_local(
            face, irr_idx=t["irr_idx"], sym_idx=t["sym_idx"],
            k_irr_frac=t["k_irr_frac"], local_perm=t["local_perm"],
            L_table=t["L_table"], spin_action_full=t["spin_action_full"],
            n_sym_spatial=t["n_sym_spatial"], spin_axis=spin_axis,
            mu_axis=mu_axis, mesh_axis=mesh_axis)
        if vertex:
            child = gamma_apply(child, *gamma_perm_phase(vertex), axis=spin_axis)
        return child

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
            # rows among themselves and the loader made them exactly zero, so
            # the prefix-mask convention of the generic service is neither
            # needed nor correct here: the complete packed extent is active.
            logical_centroid_extent=self.n_centroid_packed,
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
    layout=None,
) -> CentroidKUnfoldPlan:
    """Bind canonical symmetry tables to one orbit-packed centroid basis.

    ``layout`` is the run's :class:`SquareGroupedShardLayout`
    (``meta.mu_basis.layout``): the plan's tables are conjugated into THAT
    order, so its unfold acts directly on the arrays the run computes on.
    Omitted (tests), the layout is built here from the same orbits.

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
    if ns not in (1, 2, 4):
        raise ValueError(
            f"build_centroid_k_unfold_plan: nspinor must be 1, 2 or 4; got {ns}.")

    n_spatial = int(np.asarray(sym.sym_matrices).shape[0])
    sym_perm, wraps = centroid_source_map_and_wrap(
        np.asarray(centroid_fft_idx, dtype=np.int32),
        np.asarray(sym.sym_matrices)[:n_spatial],
        np.asarray(sym.translations)[:n_spatial],
        np.asarray(fft_grid, dtype=np.int32),
        extend_trs=True,
    )
    groups = permutation_orbit_labels(sym_perm)
    if layout is None:
        layout = build_square_grouped_shard_layout(groups, shape)
    elif int(layout.axis.n_logical) != int(sym_perm.shape[-1]):
        raise ValueError(
            "build_centroid_k_unfold_plan: the run's centroid layout holds "
            f"{layout.axis.n_logical} centroids, the table {sym_perm.shape[-1]}.")
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
]

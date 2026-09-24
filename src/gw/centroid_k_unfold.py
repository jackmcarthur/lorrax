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
from typing import NamedTuple

from ffi import _services

_services.ensure_on_path()

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
    #: from, kept so route G's typed ψ(G) children (``isdf.zeta_mubatch``)
    #: transport under the SAME action.  ``None`` only on hand-assembled test
    #: plans.
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
        return np.where(self.sym_perm < 0, -1,
                        self.sym_perm % int(self.layout.axis_shard_size)).astype(np.int32)

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

    def unfold_operator(self, operator_parent, *, operator_transpose=None, right_plan=None,
                        conjugate=False):
        """Transport the centroid-major ``(k_parent,mu,s,nu,s)`` Green to full k locally, in that order.

        ``conjugate=True`` returns the conjugated full-k operator in the same pass.
        """
        right = self if right_plan is None else right_plan
        return unfold_spin_centroid_operator(
            operator_parent,
            right_sym_perm=None if right_plan is None else right.sym_perm,
            right_L_table=None if right_plan is None else right.L_table,
            operator_transpose=operator_transpose,
            conjugate=conjugate,
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

    def unfold_load_tables(self):
        """:meth:`unfold_operator` as load tables (``symmetry_maps.unfold_load_tables``), same arguments.

        For a consumer that reads the parent Green and does the typed unfold
        on its own load (``ffi.fft.make_kconv_klead_unfold``).
        """
        from symmetry_maps import unfold_load_tables
        return unfold_load_tables(
            irr_idx=self.irr_idx, sym_idx=self.sym_idx, sym_perm=self.sym_perm,
            L_table=self.L_table, k_irr_frac=self.k_parent_frac,
            spin_action_full=self.spin_action_full, n_sym_spatial=self.n_sym_spatial,
            mesh_xy=self.mesh_xy, logical_centroid_extent=self.n_centroid_packed)



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
        extend_trs=True, required_rows=np.asarray(sym.sym_idx_k),
    )
    available = np.all(sym_perm >= 0, axis=1)
    groups = permutation_orbit_labels(sym_perm[available])
    if layout is None:
        layout = build_square_grouped_shard_layout(groups, shape)
    elif int(layout.axis.n_logical) != int(sym_perm.shape[-1]):
        raise ValueError(
            "build_centroid_k_unfold_plan: the run's centroid layout holds "
            f"{layout.axis.n_logical} centroids, the table {sym_perm.shape[-1]}.")
    packed_perm = np.full((sym_perm.shape[0], layout.axis.n_padded), -1, dtype=np.int32)
    packed_perm[available] = layout.axis.pack_permutations_host(sym_perm[available])
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


# ---------------------------------------------------------------------------
# μ-batch ζ fit: whole-orbit centroid batches and orbit-closed rank r blocks
# (docs/architecture/zeta_fit_mubatch.md, "Symmetry: parent k").  The batch
# loop computes pair projectors on raw parent k for one batch B of centroids
# (replicated) and one r block of its own rank, then unfolds them to full k
# with route G's k-convolution (``ffi.fft.make_fused_conv_kplane``).  Both
# endpoint gathers are local exactly when B and the r block are unions of
# whole orbits; these builders make them so and hand back the endpoint tables
# in the local frame.
# ---------------------------------------------------------------------------

class MuOrbitBatches(NamedTuple):
    """Whole-orbit centroid batches of one plan (see :func:`orbit_mu_batches`).

    ``mu[β, slot]`` is the PACKED centroid in slot ``slot`` of batch ``β``
    (``-1`` pad).  Slot ``p·c + j`` (``c = b / n_ranks``) is owned by rank
    ``p`` after the batch transpose.  ``left_perm[β, row, slot]`` is the
    batch-local slot of the source centroid of action row ``row`` and
    ``left_L[β, row, slot]`` its lattice wrap: ``plan.centroid_local_perm``
    / ``plan.L_table`` semantics with the batch as the local extent.  Rows
    outside ``rows`` (the ones ``plan.sym_idx`` never selects) are ``-1``;
    pad slots map to themselves with zero wrap.
    """
    mu: np.ndarray
    left_perm: np.ndarray
    left_L: np.ndarray
    rows: np.ndarray
    n_ranks: int

    @property
    def n_batch(self) -> int:
        return int(self.mu.shape[0])

    @property
    def b(self) -> int:
        return int(self.mu.shape[1])

    @property
    def c(self) -> int:
        return self.b // int(self.n_ranks)

    @property
    def rank_mu(self) -> np.ndarray:
        """``(n_batch, n_ranks, c)``: each rank's rows after the transpose."""
        return self.mu.reshape(self.n_batch, int(self.n_ranks), self.c)

    def packed_to_slot(self, mu_pad: int) -> np.ndarray:
        """``(mu_pad,)`` flat slot ``β·b + slot`` of every packed centroid; ``-1``
        for the layout's pad slots, whose Z rows are exactly zero."""
        out = np.full((int(mu_pad),), -1, dtype=np.int64)
        flat = self.mu.reshape(-1)
        hit = flat >= 0
        out[flat[hit]] = np.flatnonzero(hit)
        return out


def mu_batch_tables(k_unfold_plan, mu) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batch-local endpoint tables for given batches; refuses a split orbit.

    ``mu`` is ``(n_batch, b)`` packed centroids (``-1`` pad).  Returns
    ``(left_perm, left_L, rows)`` in :class:`MuOrbitBatches` semantics.  A
    batch not closed under every row ``plan.sym_idx`` selects raises naming
    the batch, row and centroid: the unfold would need a source outside it.
    """
    plan = k_unfold_plan
    mu = np.asarray(mu, dtype=np.int64)
    n_rows_all = int(plan.sym_perm.shape[0])
    rows = np.unique(np.asarray(plan.sym_idx, dtype=np.int64))
    sym_perm = np.asarray(plan.sym_perm, dtype=np.int64)
    L = np.asarray(plan.L_table, dtype=np.int64)
    if np.any(sym_perm[rows] < 0):
        raise ValueError(
            "mu_batch_tables: a row plan.sym_idx selects has no centroid "
            "action; the plan should have refused it.")
    n_batch, b = mu.shape
    left_perm = np.full((n_batch, n_rows_all, b), -1, dtype=np.int32)
    left_L = np.zeros((n_batch, n_rows_all, b, 3), dtype=np.int32)
    seen = np.zeros((int(plan.n_centroid_packed),), dtype=np.int64)
    for beta in range(n_batch):
        slots = np.flatnonzero(mu[beta] >= 0)
        members = mu[beta, slots]
        seen[members] += 1
        slot_of = np.full((int(plan.n_centroid_packed),), -1, dtype=np.int64)
        slot_of[members] = slots
        for row in rows:
            src = slot_of[sym_perm[row, members]]
            if np.any(src < 0):
                bad = int(np.flatnonzero(src < 0)[0])
                raise ValueError(
                    f"mu_batch_tables: batch {beta} is not a union of whole "
                    f"orbits: action row {int(row)} takes packed centroid "
                    f"{int(members[bad])} (slot {int(slots[bad])}) from "
                    f"{int(sym_perm[row, members[bad]])}, which is outside "
                    "the batch.  Fix: build the batches with orbit_mu_batches.")
            left_perm[beta, row] = np.arange(b)
            left_perm[beta, row, slots] = src
            left_L[beta, row, slots] = L[row, members]
    active = np.asarray(plan.layout.axis.active_mask, dtype=bool)
    if np.any(seen[active] != 1) or np.any(seen[~active] != 0):
        raise ValueError(
            "mu_batch_tables: every active packed centroid must sit in "
            f"exactly one slot and no layout pad in any; got counts "
            f"{np.unique(seen[active]).tolist()} (active) / "
            f"{np.unique(seen[~active]).tolist()} (pads).")
    return left_perm, left_L, rows.astype(np.int32)


def orbit_mu_batches(k_unfold_plan, mu_pad: int, n_ranks: int, *,
                     b_target: int) -> MuOrbitBatches:
    """Pack the plan's centroid orbits whole into batches of about ``b_target``.

    Orbits are those of the rows ``plan.sym_idx`` selects (the only ones the
    k unfold applies), in the packed order.  The batch width ``b`` is a
    multiple of ``n_ranks`` and at least the largest orbit: when the target
    is smaller the batch is widened to hold one orbit, and the caller prices
    the returned ``b``.  The batch count is the smallest at which
    largest-first placement onto the least-loaded batch fits every orbit
    (LPT), so the active centroids are spread evenly over the batches and the
    realized ``b`` is the largest load rounded up to ``n_ranks``.  Inside a
    batch the members are in packed order and each rank's ``c`` slots get
    ``⌊n/P⌋`` or ``⌈n/P⌉`` of them, pads trailing.  An orbit may span ranks:
    the batch is replicated and unfolded before the transpose.  The layout's
    pad slots belong to no batch (their face rows, hence Z rows, are zero).
    """
    import heapq

    from symmetry_maps import permutation_orbit_labels

    plan = k_unfold_plan
    P_ = int(n_ranks)
    if int(mu_pad) != int(plan.n_centroid_packed):
        raise ValueError(
            f"orbit_mu_batches: mu_pad={mu_pad} is not the plan's packed "
            f"centroid extent {plan.n_centroid_packed}.")
    if P_ < 1 or int(b_target) < 1:
        raise ValueError(
            f"orbit_mu_batches: need n_ranks, b_target >= 1; got {P_}, {b_target}.")
    rows = np.unique(np.asarray(plan.sym_idx, dtype=np.int64))
    labels = permutation_orbit_labels(np.asarray(plan.sym_perm)[rows])
    act = np.flatnonzero(np.asarray(plan.layout.axis.active_mask, dtype=bool))
    lab = labels[act]
    order = np.argsort(lab, kind="stable")
    _, start, sizes = np.unique(lab[order], return_index=True,
                                return_counts=True)
    members = np.split(act[order], start[1:])            # packed order inside
    up = lambda v: -(-int(v) // P_) * P_
    b_cap = max(up(int(sizes.max())), (int(b_target) // P_) * P_)
    by_size = sorted(range(len(members)),
                     key=lambda g: (-int(sizes[g]), int(members[g][0])))
    n_batch = max(1, -(-int(act.size) // b_cap))
    while True:
        heap = [(0, beta) for beta in range(n_batch)]
        owner = np.empty((len(members),), dtype=np.int64)
        loads = np.zeros((n_batch,), dtype=np.int64)
        ok = True
        for g in by_size:
            load, beta = heapq.heappop(heap)
            if load + int(sizes[g]) > b_cap:
                ok = False
                break
            owner[g] = beta
            loads[beta] = load + int(sizes[g])
            heapq.heappush(heap, (int(loads[beta]), beta))
        if ok:
            break
        n_batch += 1
    b = up(int(loads.max()))
    c = b // P_
    mu = np.full((n_batch, b), -1, dtype=np.int32)
    for beta in range(n_batch):
        mem = np.sort(np.concatenate(
            [members[g] for g in np.flatnonzero(owner == beta)]))
        q, r = divmod(int(mem.size), P_)
        cursor = 0
        for p in range(P_):
            take = q + (1 if p < r else 0)
            mu[beta, p * c:p * c + take] = mem[cursor:cursor + take]
            cursor += take
    left_perm, left_L, rows = mu_batch_tables(plan, mu)
    return MuOrbitBatches(mu=mu, left_perm=left_perm, left_L=left_L,
                          rows=rows, n_ranks=P_)


__all__ = [
    "CentroidKUnfoldPlan",
    "MuOrbitBatches",
    "build_centroid_k_unfold_plan",
    "mu_batch_tables",
    "orbit_mu_batches",
]

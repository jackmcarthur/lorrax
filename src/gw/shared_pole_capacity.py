"""Device-byte accounting for one shared-pole construction (SP 17).

The constructor's physics is the Hermite/Ritz chain in
``gw.shared_pole_constructor``. Everything that prices it -- the carrier byte
model, the native eigh/GEMM workspace queries, the map ledger reservations and
the panels that stay live across a stage -- is here, so the driver reads as the
equations and this file reads as the budget.

One :class:`ConstructorCapacity` belongs to one construction. It owns no array:
``retained_panels`` is a view of arrays the driver keeps alive, and the ledger
(``gw.shared_pole_recipe.CapacityLedger``) owns admission and refusal.
"""

from __future__ import annotations

import math

import numpy as np


def shared_pole_byte_terms(meta, *, mesh_xy, resolution, pencil_side,
                           parent_batch, sample_batch, phase="reduction"):
    """Price constructor carriers; the map CapacityLedger owns admission.

    Selection holds samples, current narrow actions and the n/2n direction
    solve; it has no R-by-R pencil. Reduction holds the actual selected
    pencil. Model checks hold factors and bounded samples, with no pencil.
    Native workspace is separately supplied by the service. No threshold
    or independent capacity policy lives in this constructor helper.
    """
    p = int(mesh_xy.shape["x"]) * int(mesh_xy.shape["y"])
    # Constructor carriers are mu x mu charge operators on every admitted deck.
    packed = int(meta.n_rmu_padded)
    b, a, r = int(parent_batch), int(sample_batch), int(pencil_side)
    if min(packed, b, a) <= 0 or r < 0:
        raise ValueError("GATE shared_pole_capacity: got: invalid extents; want: positive basis/batches and nonnegative pencil; why: live-set pricing")
    dense_copies = math.ceil(b / p) if resolution.layout == "local" else b / p
    if phase == "selection":
        dense = 24 * packed**2
        sample_faces = max(2*a, 2)
    elif phase == "reduction":
        dense = 14 * r*r + 12 * packed * r
        sample_faces = 0
    elif phase == "model":
        dense = 8 * packed**2 + 4 * packed * r
        sample_faces = max(2*a, 2)
    else:
        raise ValueError(f"unknown shared-pole capacity phase: {phase}")
    terms = {
        "sample_or_moment_batch": math.ceil(16*b*sample_faces*packed**2/p),
        "narrow_actions": math.ceil(16*b*3*packed*r/p),
        "replicated_scalars": 8*b*(12*r+4*packed),
        "phase_dense_temporaries": math.ceil(16*dense_copies*dense),
    }
    return {"terms_bytes_per_rank": terms,
            "resident_bytes_per_rank": sum(terms.values()),
            "layout": resolution.layout, "phase": phase, "pencil_side": r,
            "parent_batch": b, "sample_batch": a}


def _shard_bytes(array):
    return int(np.prod(array.sharding.shard_shape(array.shape))) * array.dtype.itemsize


class ConstructorCapacity:
    """Plans, native workspace and ledger rows for one construction.

    Parameters
    ----------
    meta : Meta
        Current packed centroid basis; supplies ``n_rmu_padded`` to the byte
        model and to the eigh extents.
    resolution : LinalgResolution
        The once-resolved dense policy (layout, eigh backend, batched route).
    mesh_xy : Mesh
        The run's named x/y mesh, never reconstructed here.
    ledger : CapacityLedger
        The map's ledger. It owns the budget, the refusal and the receipt.
    upstream : tuple of str
        Stage names already live when the construction starts; every
        reservation is concurrent with exactly these.

    The driver keeps two facts current as it moves through the chain:

    ``retained_panels``
        the narrow panels other parents still hold, charged on every row so
        they are visible storage rather than headroom hidden in a limit;
    ``batch_width``
        the parent batch the next price is for.
    """

    def __init__(self, meta, resolution, *, mesh_xy, ledger, upstream):
        self._meta = meta
        self._resolution = resolution
        self._mesh_xy = mesh_xy
        self._ledger = ledger
        self._upstream = tuple(upstream)
        self._n = int(meta.n_rmu_padded)
        self._plans = {}
        self.native_queries = {}
        self._native_maxima = {"eigh": 0, "gemm": 0}
        self._workspace = 0
        self._side = 0
        self._phase = "selection"
        self.retained_panels = ()
        self.batch_width = 1

    def eigenplan(self, side):
        """The resolved eigh plan for one side, built once per side."""
        import distrib_la

        if side not in self._plans:
            self._plans[side] = distrib_la.plan(
                "eigh", self._mesh_xy, n=side, backend=self._resolution.eigh_backend,
                batched_route=self._resolution.batched_route)
        return self._plans[side]

    def query_workspace(self, op, shapes, plan=None):
        """Native workspace bytes per rank for one op at one shape, cached."""
        import distrib_la

        key = (op, shapes)
        if key not in self.native_queries:
            self.native_queries[key] = (
                distrib_la.matmul_workspace_bytes_per_rank(
                    self._mesh_xy, shapes, np.complex128, backend="auto",
                    batched_route=self._resolution.batched_route) if op == "gemm" else
                distrib_la.workspace_bytes_per_rank(plan, op, shapes, np.complex128))
        if op == "gemm":
            self._native_maxima[op] = max(self._native_maxima[op], self.native_queries[key])
            self._workspace = sum(self._native_maxima.values())
        return self.native_queries[key]

    def plan(self, side=None, *, phase=None, sample_batch=1, transpose_staging=0):
        """Admit this phase's actual live set before allocating it.

        ``side`` is the pencil side this price is for; omit it to reprice the
        side already in hand, as a mid-GEMM staging charge does.
        """
        if phase is not None:
            self._phase = phase
        if side is None:
            side = self._side
        self._side = side
        n = self._n
        extents = {n, 2*n} if self._phase == "selection" else (
            {side} if self._phase == "reduction" else {n})
        # Eigh scratch is transient: replace it at each phase boundary.
        # Only the actually used GEMM context workspace persists.
        self._native_maxima["eigh"] = max(self.query_workspace(
            "eigh", ((self.batch_width, extent, extent),), self.eigenplan(extent))
            for extent in sorted(extents))
        self._workspace = sum(self._native_maxima.values())
        price = shared_pole_byte_terms(
            self._meta, mesh_xy=self._mesh_xy, resolution=self._resolution,
            pencil_side=side, parent_batch=self.batch_width,
            sample_batch=sample_batch, phase=self._phase)
        # Other parents' narrow inputs survive selection and each model's
        # checks; they are additional live storage, never hidden in a limit.
        extra = sum(_shard_bytes(a)
                    for a in {id(a): a for a in self.retained_panels}.values())
        price["terms_bytes_per_rank"]["retained_parent_panels"] = extra
        price["terms_bytes_per_rank"]["gemm_transpose_staging"] = transpose_staging
        price["resident_bytes_per_rank"] += extra + transpose_staging
        row = self._reserve("constructor.plan", price["resident_bytes_per_rank"])
        return dict(row, price=price, native_workspace=dict(self._native_maxima))

    def live(self, arrays):
        """Bind the ledger's ambient lifetimes to exactly these live arrays."""
        # Callees price only their additional allocations. Supply their
        # exact current inputs instead of the future dense-phase envelope,
        # so a store read does not count its own returned arrays twice.
        unique = {id(array): array for array in (*arrays, *self.retained_panels)}
        row = self._reserve("constructor.live",
                            sum(_shard_bytes(a) for a in unique.values()))
        self._ledger.live_stages = (*self._upstream, row["stage"])

    def matmul(self, a, b, **kwargs):
        """The resolved service GEMM, with its native workspace charged first."""
        import distrib_la

        # Workspace belongs to the matmul route, not the eigh backend.
        # Its query excludes operand-sized endpoint transpose staging;
        # charge those transient faces separately before execution.
        shapes = tuple(value.shape[:-2] + (value.shape[-2:][::-1]
                       if kwargs.get(trans, "N") != "N" else value.shape[-2:])
                       for value, trans in ((a, "transa"), (b, "transb")))
        previous = self._workspace
        self.query_workspace("gemm", shapes)
        staging = (sum(int(np.prod(value.shape)) * value.dtype.itemsize
                       // int(self._mesh_xy.size)
                       for value, trans in ((a, "transa"), (b, "transb"))
                       if kwargs.get(trans, "N") != "N")
                   if self._resolution.batched_route != "batch_reshard" else 0)
        if self._workspace != previous or staging:
            self.plan(transpose_staging=staging)
        return distrib_la.matmul(a, b, mesh=self._mesh_xy, backend="auto",
                                 batched_route=self._resolution.batched_route, **kwargs)

    def _reserve(self, kind, resident_bytes_per_rank):
        return self._ledger.reserve(
            f"{kind}.{len(self._ledger.entries)}",
            resident_bytes_per_rank=resident_bytes_per_rank,
            workspace_bytes_per_rank=self._workspace,
            concurrent_with=self._upstream)

"""Device-byte accounting for one shared-pole construction (I 1).

The constructor's physics is the Hermite/Ritz chain in
``gw.shared_pole_constructor``. Everything that prices it -- the carrier byte
model, the native eigh workspace queries, the map ledger reservations and
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
                           parent_batch, sample_batch, phase="reduction",
                           selection_faces=None):
    """Price constructor carriers; the map CapacityLedger owns admission.

    Selection holds samples, current narrow actions and the n/2n direction
    solve; it has no R-by-R pencil. ``selection_faces`` is an internal count
    of the caller's already resident sample and moment faces when that count
    differs from the ordered scalar default. Reduction holds the actual selected
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
        # Ordered scalar selection keeps its historical two sample faces.
        # A sector caller can supply the number of resident sample AND
        # moment faces from the actual read dictionaries; this prices the
        # four-field photon bank without introducing a user capacity dial.
        sample_faces = (max(2*a, 2) if selection_faces is None
                        else int(selection_faces))
        if sample_faces < 2*a:
            raise ValueError(
                'GATE shared_pole_capacity: selection_faces underprices '
                f'the {2*a} base sample faces; got {sample_faces}')
    elif phase == "reduction":
        if selection_faces is not None:
            raise ValueError('selection_faces applies only to selection')
        dense = 14 * r*r + 12 * packed * r
        sample_faces = 0
    elif phase == "model":
        if selection_faces is not None:
            raise ValueError('selection_faces applies only to selection')
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
            "parent_batch": b, "sample_batch": a,
            "sample_face_count": sample_faces}


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
        The once-resolved dense policy; its layout prices the dense temporaries.
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

    def __init__(self, meta, resolution, *, mesh_xy, ledger, upstream, execution="local"):
        self.execution = execution
        self._meta = meta
        self._resolution = resolution
        self._mesh_xy = mesh_xy
        self._ledger = ledger
        self._upstream = tuple(upstream)
        self._n = int(meta.n_rmu_padded)
        self._plans = {}
        self.native_queries = {}
        self._native_maxima = {"eigh": 0}
        self._workspace = 0
        self._side = 0
        self._phase = "selection"
        self.retained_panels = ()
        self.batch_width = 1

    def eigenplan(self, side):
        """One service plan per configured execution layout and actual side."""
        import distrib_la

        key = self.execution, side
        if key not in self._plans:
            self._plans[key] = distrib_la.plan("eigh", self._mesh_xy, n=side,
                backend="distributed" if self.execution == "face" else "off",
                batched_route="auto" if self.execution == "face" else "batch_reshard")
        return self._plans[key]

    def query_workspace(self, op, shapes, plan):
        """Native workspace bytes per rank for one op at one shape, cached."""
        import distrib_la

        key = (self.execution, op, shapes)
        if key not in self.native_queries:
            self.native_queries[key] = distrib_la.workspace_bytes_per_rank(plan, op, shapes, np.complex128)
        return self.native_queries[key]

    def plan(self, side=None, *, phase=None, sample_batch=1,
             selection_faces=None):
        """Admit this phase's actual live set before allocating it.

        ``side`` is the pencil side this price is for; omit it to reprice the
        side already in hand.
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
        self._native_maxima["eigh"] = max(self.query_workspace(
            "eigh", ((1 if self.execution == "face" else self.batch_width, extent, extent),), self.eigenplan(extent))
            for extent in sorted(extents))
        if self.execution == 'face':
            extent=max(n,*extents)
            import distrib_la
            shapes=((1,extent,extent),(1,extent,extent))
            key=('face','matmul',shapes)
            if key not in self.native_queries:
                self.native_queries[key]=distrib_la.matmul_workspace_bytes_per_rank(
                    self._mesh_xy,shapes,np.complex128,backend='distributed',batched_route='auto')
            self._native_maxima['gemm']=self.native_queries[key]
        self._workspace = sum(self._native_maxima.values())
        from types import SimpleNamespace
        pricing_resolution = SimpleNamespace(layout="distributed" if self.execution == "face" else "local")
        price = shared_pole_byte_terms(
            self._meta, mesh_xy=self._mesh_xy, resolution=pricing_resolution,
            pencil_side=side, parent_batch=1 if self.execution == "face" else self.batch_width,
            sample_batch=sample_batch, phase=self._phase,
            selection_faces=selection_faces)
        # Other parents' narrow inputs survive selection and each model's
        # checks; they are additional live storage, never hidden in a limit.
        extra = sum(_shard_bytes(a)
                    for a in {id(a): a for a in self.retained_panels}.values())
        price["terms_bytes_per_rank"]["retained_parent_panels"] = extra
        price["resident_bytes_per_rank"] += extra
        row = self._reserve("constructor.plan", price["resident_bytes_per_rank"])
        row['execution'] = self.execution
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

    def _reserve(self, kind, resident_bytes_per_rank):
        return self._ledger.reserve(
            f"{kind}.{len(self._ledger.entries)}",
            resident_bytes_per_rank=resident_bytes_per_rank,
            workspace_bytes_per_rank=self._workspace,
            concurrent_with=self._upstream)

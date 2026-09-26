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
from functools import lru_cache

import numpy as np


def shared_pole_byte_terms(meta, *, mesh_xy, resolution, pencil_side,
                           parent_batch, sample_batch, phase="reduction",
                           selection_faces=None, cross_original_sides=None,
                           padding_output_bytes_per_rank=0):
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
    padding_output_bytes_per_rank = int(padding_output_bytes_per_rank)
    if padding_output_bytes_per_rank < 0 or (phase != "reduction" and padding_output_bytes_per_rank):
        raise ValueError("GATE shared_pole_capacity: round padding outputs require reduction and nonnegative bytes")
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
    elif phase == "cross_reduction":
        if selection_faces is not None or cross_original_sides is None or len(cross_original_sides) != 2:
            raise ValueError('cross reduction requires two original sides and no selection faces')
        c, t = map(int, cross_original_sides)
        if min(c, t) <= 0:
            raise ValueError('cross reduction original sides must be positive')
        # The original CT pencils are rectangular C-by-T. They are projected
        # on the two retained diagonal spans before the joint square is made.
        # The original assembly's largest rectangular live set has five
        # finite blocks, three top, three bottom and four corner blocks,
        # plus at most four whole-rectangle concatenation/output buffers:
        # at most nine C-by-T equivalents. Allow one more for overlap with
        # projection, and keep the full fourteen-copy joint-square envelope.
        dense = 10 * c*t + 14 * r*r + 12 * packed * (c+t)
        sample_faces = 0
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
    action_side = sum(map(int, cross_original_sides)) if phase == 'cross_reduction' else r
    terms = {
        "sample_or_moment_batch": math.ceil(16*b*sample_faces*packed**2/p),
        "narrow_actions": math.ceil(16*b*3*packed*action_side/p),
        # jnp.pad creates new arrays while all source panels still live. The
        # reduction envelope covers the old actions, not these new outputs.
        "round_padding_outputs": padding_output_bytes_per_rank,
        "replicated_scalars": 8*b*(12*action_side+4*packed),
        "phase_dense_temporaries": math.ceil(16*dense_copies*dense),
    }
    return {"terms_bytes_per_rank": terms,
            "resident_bytes_per_rank": sum(terms.values()),
            "layout": resolution.layout, "phase": phase, "pencil_side": r,
            "parent_batch": b, "sample_batch": a,
            "sample_face_count": sample_faces}


@lru_cache(maxsize=None)
def constructor_eigenplan(mesh_xy, side, execution):
    """The eigh service plan of a constructor layout: whole parents per rank
    ('local', the q-local kernel) or the complete mesh ('face')."""
    import distrib_la

    return distrib_la.plan("eigh", mesh_xy, n=int(side),
        backend="distributed" if execution == "face" else "off",
        batched_route="auto" if execution == "face" else "batch_reshard")


def _shard_bytes(array):
    return int(np.prod(array.sharding.shard_shape(array.shape))) * array.dtype.itemsize


def round_padding_output_bytes(states, infinity, widths, infinity_width):
    """Per-rank bytes of new padded panels while every input remains live."""
    def output_bytes(array, width):
        if int(array.shape[-1]) == int(width):
            return 0
        shape = (*array.shape[:-1], int(width))
        return int(np.prod(array.sharding.shard_shape(shape))) * array.dtype.itemsize

    return (sum(output_bytes(array, width)
                for state, width in zip(states, widths) for array in state[1:])
            + sum(output_bytes(array, infinity_width) for array in infinity))


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
        self.native_queries = {}
        self._native_maxima = {"eigh": 0}
        self._workspace = 0
        self._side = 0
        self._phase = "selection"
        self.retained_panels = ()
        self.batch_width = 1

    def eigenplan(self, side):
        """One service plan per configured execution layout and actual side."""
        return constructor_eigenplan(self._mesh_xy, int(side), self.execution)

    def query_workspace(self, op, shapes, plan):
        """Native workspace bytes per rank for one op at one shape, cached."""
        import distrib_la

        # Receipt schema owns a stable ``(op, shapes)`` key. Execution is
        # recorded on every capacity row and one ConstructorCapacity never
        # mixes layouts, so it does not belong in this map key.
        key = (op, shapes)
        if key not in self.native_queries:
            self.native_queries[key] = distrib_la.workspace_bytes_per_rank(plan, op, shapes, np.complex128)
        return self.native_queries[key]

    def plan(self, side=None, *, phase=None, sample_batch=1,
             selection_faces=None, eigen_side=None, cross_original_sides=None,
             padding_output_bytes_per_rank=0):
        """Admit this phase's actual live set before allocating it.

        ``side`` is the pencil side this price is for; omit it to reprice the
        side already in hand.
        """
        if phase is not None:
            self._phase = phase
        if side is None:
            side = self._side
        self._side = side
        price, native = self.quote(side, phase=self._phase,
                                   sample_batch=sample_batch,
                                   selection_faces=selection_faces, eigen_side=eigen_side,
                                   cross_original_sides=cross_original_sides,
                                   padding_output_bytes_per_rank=padding_output_bytes_per_rank)
        self._workspace = sum(native.values())
        row = self._reserve("constructor.plan", price["resident_bytes_per_rank"])
        row['execution'] = self.execution
        return dict(row, price=price, native_workspace=dict(native))

    def quote(self, side, *, phase, sample_batch=1, selection_faces=None, eigen_side=None,
              cross_original_sides=None, padding_output_bytes_per_rank=0):
        """Return an unrecorded phase price for route selection/preflight."""
        side = int(side)
        n = self._n
        price = self.resident_quote(
            side, phase=phase, sample_batch=sample_batch,
            selection_faces=selection_faces, cross_original_sides=cross_original_sides,
            padding_output_bytes_per_rank=padding_output_bytes_per_rank)
        extents = {n, 2*n} if phase == "selection" else (
            {side if eigen_side is None else int(eigen_side)} if phase in ("reduction", "cross_reduction") else {n})
        # Eigh scratch is transient: replace it at each phase boundary.
        self._native_maxima["eigh"] = max(self.query_workspace(
            "eigh", ((self.batch_width, extent, extent),), self.eigenplan(extent))
            for extent in sorted(extents))
        if self.execution == 'face':
            extent=max(n,side,*extents)
            import distrib_la
            shapes=((self.batch_width,extent,extent),(self.batch_width,extent,extent))
            # Receipts use the public workspace operation name, matching
            # the capacity maximum and the distrib_la service vocabulary.
            key=('gemm',shapes)
            if key not in self.native_queries:
                self.native_queries[key]=distrib_la.matmul_workspace_bytes_per_rank(
                    self._mesh_xy,shapes,np.complex128,backend='distributed',batched_route='auto')
            self._native_maxima['gemm']=self.native_queries[key]
        self._workspace = sum(self._native_maxima.values())

        return price, dict(self._native_maxima)

    def resident_quote(self, side, *, phase, sample_batch=1,
                       selection_faces=None, cross_original_sides=None,
                       padding_output_bytes_per_rank=0):
        """Price the live arrays without invoking a native workspace query.

        This is an optimistic admission bound. Route selection uses it first
        because a local provider need not support an extent whose resident
        arrays already exceed the device budget; any native workspace can only
        make that route larger.
        """
        side = int(side)
        from types import SimpleNamespace
        pricing_resolution = SimpleNamespace(layout="distributed" if self.execution == "face" else "local")
        price = shared_pole_byte_terms(
            self._meta, mesh_xy=self._mesh_xy, resolution=pricing_resolution,
            pencil_side=side, parent_batch=self.batch_width,
            sample_batch=sample_batch, phase=phase,
            selection_faces=selection_faces,
            cross_original_sides=cross_original_sides,
            padding_output_bytes_per_rank=padding_output_bytes_per_rank)
        # Other parents' narrow inputs survive selection and each model's
        # checks; they are additional live storage, never hidden in a limit.
        extra = sum(_shard_bytes(a)
                    for a in {id(a): a for a in self.retained_panels}.values())
        price["terms_bytes_per_rank"]["retained_parent_panels"] = extra
        price["resident_bytes_per_rank"] += extra
        return price

    def preview(self, side, *, phase, sample_batch=1, selection_faces=None,
                cross_original_sides=None, padding_output_bytes_per_rank=0):
        """Preview device admission without appending a ledger row."""
        # A candidate carrier may be rejected in favour of the current
        # round's smaller one.  Its workspace must not become a high-water
        # charge on that fallback: only an admitted plan advances maxima.
        maxima, workspace = dict(self._native_maxima), self._workspace
        try:
            price, native = self.quote(side, phase=phase, sample_batch=sample_batch,
                                       selection_faces=selection_faces,
                                       cross_original_sides=cross_original_sides,
                                       padding_output_bytes_per_rank=padding_output_bytes_per_rank)
        finally:
            self._native_maxima, self._workspace = maxima, workspace
        return self._ledger.preview(
            resident_bytes_per_rank=price['resident_bytes_per_rank'],
            workspace_bytes_per_rank=sum(native.values()),
            concurrent_with=self._upstream)

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

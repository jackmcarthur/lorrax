"""Provider resolution and distributed active-subspace algebra.

Only row coefficients are reduced. Vector storage remains partitioned over
all mesh axes declared by the caller; no vector gather is permitted here.
The CPU compatibility provider uses active NumPy BLAS/LAPACK callbacks, so
it promises fixed JAX geometry, not GPU-like memory or transfer performance.
The XLA provider stages the same algebra as device products on the active
window and :func:`distrib_la.eigh_local` on the active block: no host callback.
Which provider a platform leg takes is :data:`SUBSPACE_ROUTE`.
"""
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from lxkit import device_vendor, host_pure_callback
from lxkit import ordered_psum

from distrib_la._active_local import interval_rows
from distrib_la.active_subspace import (LocalSubspacePlan, _orthogonalization_range, native_fields,
                                       native_leg, plan_local_subspace)
from distrib_la.loader import probe_target
from distrib_la.local_eigh import eigh_local


#: Subspace provider by ``lxkit`` leg key -- a backend preference of this
#: service (as ``local_eigh.COMPLEX_EIGH_ROUTE``), so it lives here and not in
#: ``lxkit.LEGS``.  ``"native"``: the ``lorrax_active_subspace_*`` handlers;
#: ``"xla"``: :class:`XlaSubspacePlan` (device products and ``eigh_local``, the
#: XLA reference route); ``"host"``: :class:`CpuSubspacePlan` (NumPy LAPACK in
#: ``pure_callback``).  A vendor with no leg runs the XLA route
#: (``docs/architecture/decisions.md#xla-reference``).  ROCM: ``"native"`` -- the
#: CUDA handlers' leg-neutral port (rocBLAS/rocSOLVER, ``src/ffi/cpp/active_subspace``),
#: gated against ``"xla"`` and ``"host"`` on the same GCD by
#: tests/test_subspace_native_gate.py.  (Wave 1: the host route cost bse 517 callbacks,
#: 1.52 s of 5.8 s, and made the solver jits uncacheable; the XLA route followed.)
SUBSPACE_ROUTE = {"cpu": "host", "CUDA": "native", "ROCM": "native"}


def subspace_route(leg_key=None):
    """The provider this process's leg (or ``leg_key``) takes: native | xla | host."""
    if leg_key is None:
        from lxkit import leg_for_vendor
        leg = leg_for_vendor(device_vendor(jax.local_devices()[0]))
        leg_key = None if leg is None else leg.key
    return SUBSPACE_ROUTE.get(leg_key, "xla")



@dataclass(frozen=True)
class CpuSubspacePlan:
    capacity: int
    n_eig: int
    max_block_size: int | None = None

    @property
    def workspace_specs(self):
        # LAPACK workspace is host-owned on this compatibility provider.
        return {}

    def eigh(self, h, active):
        def work(h, m):
            m = int(m)
            e, c = np.linalg.eigh(h[:m, :m])
            values = np.full(self.n_eig, np.inf, np.float64)
            vectors = np.zeros((self.capacity, self.n_eig), np.complex128)
            n = min(m, self.n_eig)
            values[:n], vectors[:m, :n] = e[:n], c[:, :n]
            return values, vectors
        return host_pure_callback(work, (
            jax.ShapeDtypeStruct((self.n_eig,), jnp.float64),
            jax.ShapeDtypeStruct((self.capacity, self.n_eig), jnp.complex128)), h, active)

    def gram(self, v, p, active, *, start=0):
        def work(v, p, start, count):
            start, count = int(start), int(count)
            out = np.zeros((self.capacity, p.shape[0]), np.complex128)
            out[start:start+count] = v[start:start+count].reshape(count, -1).conj() @ p.reshape(p.shape[0], -1).T if count else 0
            return out
        return host_pure_callback(work, jax.ShapeDtypeStruct((self.capacity, p.shape[0]), jnp.complex128), v, p, start, active)

    def reconstruct(self, v, hv, c, active, template, *, columns=None, compute_image=True, start=0):
        columns = template.shape[0] if columns is None else columns
        def work(v, hv, c, count, columns, start):
            count, columns, start = int(count), int(columns), int(start)
            out, hout = np.zeros(template.shape, np.complex128), np.zeros(template.shape, np.complex128)
            if count and columns:
                coeff = c[start:start+count, :columns].T
                out[:columns] = (coeff @ v[start:start+count].reshape(count, -1)).reshape(out[:columns].shape)
                if compute_image:
                    hout[:columns] = (coeff @ hv[start:start+count].reshape(count, -1)).reshape(hout[:columns].shape)
            return out, hout
        spec = jax.ShapeDtypeStruct(template.shape, jnp.complex128)
        return host_pure_callback(work, (spec, spec), v, hv, c, active, columns, start)

    def orthogonalize(self, v, p, active, *, start=0):
        start, active = _orthogonalization_range(start, active, self.capacity)
        def work(v, p, start, count):
            start, count = int(start), int(count)
            if start < 0 or count < 0 or start+count > self.capacity:
                raise ValueError('orthogonalization range outside capacity')
            out = np.array(p, copy=True).reshape(p.shape[0], -1)
            if count:
                basis = v[start:start+count].reshape(count, -1)
                for _ in range(2):
                    coefficients = basis.conj() @ out.T
                    out -= coefficients.T @ basis
            return out.reshape(p.shape)
        return host_pure_callback(work, jax.ShapeDtypeStruct(p.shape, p.dtype),
                                 v, p, start, active)

    def store(self, v, hv, p, hp, start, count):
        def body(i, state):
            vv, hh = state
            return vv.at[start+i].set(p[i]), hh.at[start+i].set(hp[i])
        return jax.lax.fori_loop(0, count, body, (v, hv))

    def project(self, v, hv, active, h, start, count):
        def work(v, hv, h, m, start, count):
            m, start, count = int(m), int(start), int(count)
            out = np.array(h, copy=True)
            if count:
                vv, hh = v[:m].reshape(m, -1), hv[start:start+count].reshape(count, -1)
                panel = vv.conj() @ hh.T
                out[:m, start:start+count] = panel
                out[start:start+count, :m] = panel.conj().T
            return out
        return host_pure_callback(work, jax.ShapeDtypeStruct(h.shape, h.dtype), v, hv, h, active, start, count)

    def qr(self, rows):
        flat = rows.reshape(rows.shape[0], -1)
        if flat.shape[1] < rows.shape[0]:
            raise ValueError('QR needs at least as many vector entries as rows')
        q, r = jnp.linalg.qr(flat.T, mode='reduced')
        return q.T.reshape(rows.shape), r

    constrain = LocalSubspacePlan.constrain
    normalize = LocalSubspacePlan.normalize


def _rows(array):
    return array.reshape(array.shape[0], -1)


def _eigh_sizes(capacity):
    """Static eigensolve extents: powers of two from 64 below capacity, then capacity."""
    sizes, size = [], 64
    while size < capacity:
        sizes.append(size)
        size *= 2
    return tuple(sizes) + (capacity,)


@dataclass(frozen=True)
class XlaSubspacePlan:
    """The XLA route: active-window device products and ``eigh_local``; no callback.

    Window products slice only the active rows (``interval_rows``): inactive
    rows may hold NaN and never enter arithmetic. ``project`` multiplies the
    new image rows against the whole basis and selects the first ``active``
    rows of the result (a GEMM output row reads only its own basis row).
    ``eigh`` solves the smallest static extent of :func:`_eigh_sizes` that
    holds the active block, its inactive diagonal shifted above the active
    spectrum (twice its Gershgorin bound), so arithmetic stays within a factor
    two of the active size above 64. Invalid runtime windows return NaN,
    never a clamped slice.
    """
    capacity: int
    n_eig: int
    max_block_size: int | None = None

    @property
    def workspace_specs(self):
        # Scratch is XLA-owned temporaries; read it from memory_analysis().
        return {}

    def _window(self, start, count):
        """int32 [lo, hi) and its validity, judged in the operands' own width.

        An invalid window comes back empty (no loop trip, no clamped slice);
        callers return NaN for it.
        """
        start, count = jnp.asarray(start), jnp.asarray(count)
        cap = self.capacity
        valid = ((start >= 0) & (start <= cap) & (count >= 0) & (count <= cap)
                 & (start + count <= cap))
        lo = jnp.where(valid, start, 0).astype(jnp.int32)
        return lo, lo + jnp.where(valid, count, 0).astype(jnp.int32), valid

    def _gram_rows(self, vf, pf, lo, hi):
        """C[r] = conj(V[r]) . P^T for r in [lo, hi), zero elsewhere."""
        def block(s, w, out):
            vs = jax.lax.dynamic_slice_in_dim(vf, s, w, axis=0)
            return jax.lax.dynamic_update_slice_in_dim(
                out, jnp.conj(jnp.conj(pf) @ vs.T).T, s, axis=0)
        return interval_rows(block, lo, hi, self.capacity,
                             jnp.zeros((self.capacity, pf.shape[0]), jnp.complex128))

    def _combine(self, c, vf, lo, hi):
        """sum_{r in [lo, hi)} c[r]^T V[r]: (c.shape[1], n)."""
        def block(s, w, out):
            vs = jax.lax.dynamic_slice_in_dim(vf, s, w, axis=0)
            cs = jax.lax.dynamic_slice_in_dim(c, s, w, axis=0)
            return out + cs.T @ vs
        return interval_rows(block, lo, hi, self.capacity,
                             jnp.zeros((c.shape[1], vf.shape[1]), jnp.complex128))

    def eigh(self, h, active):
        cap, b = self.capacity, self.n_eig
        if h.shape != (cap, cap) or h.dtype != jnp.complex128:
            raise ValueError('projected Hamiltonian differs from the declared plan')
        m = jnp.asarray(active, jnp.int32)
        sizes = _eigh_sizes(cap)

        def solve(size):
            def branch(h):
                inside = jnp.arange(size) < m
                a = jnp.where(inside[:, None] & inside[None, :], h[:size, :size], 0)
                # The lower triangle is the matrix (LAPACK's UPLO='L', as the host route).
                a = jnp.tril(a) + jnp.conj(jnp.tril(a, -1)).T
                bound = jnp.max(jnp.sum(jnp.abs(a), axis=1))
                shift = jnp.where(bound > 0, 2 * bound, 1.0)
                a = a + jnp.diag(jnp.where(inside, 0.0, shift)).astype(a.dtype)
                w, v = eigh_local(a, UPLO='L', symmetrize_input=False)
                n = min(size, b)
                values = jnp.full((b,), jnp.inf, jnp.float64).at[:n].set(w[:n])
                vectors = jnp.zeros((cap, b), jnp.complex128).at[:size, :n].set(v[:, :n])
                return values, vectors
            return branch

        index = jnp.minimum(jnp.sum(jnp.asarray(sizes) < m), len(sizes) - 1)
        values, vectors = jax.lax.switch(index, tuple(solve(n) for n in sizes), h)
        column = jnp.arange(b) < m
        values = jnp.where(column, values, jnp.inf)
        vectors = jnp.where((jnp.arange(cap) < m)[:, None] & column[None, :], vectors, 0)
        return values, vectors

    def gram(self, v, p, active, *, start=0):
        if v.shape[0] != self.capacity or v.shape[1:] != p.shape[1:]:
            raise ValueError('active Gram buffer geometry differs from plan')
        lo, hi, valid = self._window(start, active)
        out = self._gram_rows(_rows(v), _rows(p), lo, hi)
        return jnp.where(valid, out, jnp.nan)

    def reconstruct(self, v, hv, c, active, template, *, columns=None, compute_image=True, start=0):
        if v.shape != hv.shape or v.shape[0] != self.capacity or c.shape != (self.capacity, template.shape[0]):
            raise ValueError('active reconstruction buffer geometry differs from plan')
        if v.shape[1:] != template.shape[1:]:
            raise ValueError('active reconstruction vector shapes differ')
        columns = template.shape[0] if columns is None else columns
        lo, hi, valid = self._window(start, active)
        keep = (jnp.arange(template.shape[0]) < columns)[:, None]

        def image(vectors):
            out = jnp.where(keep, self._combine(c, _rows(vectors), lo, hi), 0)
            return jnp.where(valid, out, jnp.nan).reshape(template.shape)
        x = image(v)
        hx = image(hv) if compute_image else jnp.zeros(template.shape, jnp.complex128)
        return x, hx

    def orthogonalize(self, v, p, active, *, start=0):
        if p.shape[0] > (self.max_block_size or self.n_eig):
            raise ValueError('orthogonalization block exceeds planned scratch width')
        if v.shape[0] != self.capacity or v.shape[1:] != p.shape[1:]:
            raise ValueError('active orthogonalization buffer geometry differs from plan')
        start, active = _orthogonalization_range(start, active, self.capacity)
        lo, hi, valid = self._window(start, active)
        vf, pf = _rows(v), _rows(p)
        for _ in range(2):
            pf = pf - self._combine(self._gram_rows(vf, pf, lo, hi), vf, lo, hi)
        return jnp.where(valid, pf, jnp.nan).reshape(p.shape).astype(p.dtype)

    def project(self, v, hv, active, h, start, count):
        if v.shape != hv.shape or v.shape[0] != self.capacity or h.shape != (self.capacity, self.capacity):
            raise ValueError('active projection buffer geometry differs from plan')
        cap = self.capacity
        vf, hf = _rows(v), _rows(hv)
        lo, hi, valid = self._window(start, count)

        def block(s, w, out):
            hs = jax.lax.dynamic_slice_in_dim(hf, s, w, axis=0)
            return jax.lax.dynamic_update_slice(out, jnp.conj(jnp.conj(hs) @ vf.T).T, (jnp.int32(0), s))
        panel = interval_rows(block, lo, hi, cap, jnp.zeros((cap, cap), jnp.complex128))
        row = jnp.arange(cap)
        new, old = (row >= lo) & (row < hi), row < active
        upper = old[:, None] & new[None, :]
        out = jnp.where(upper, panel, h)
        out = jnp.where(upper.T, jnp.conj(jnp.where(upper, panel, 0)).T, out)
        return jnp.where(valid, out, jnp.nan)

    store = CpuSubspacePlan.store
    qr = CpuSubspacePlan.qr
    constrain = LocalSubspacePlan.constrain
    normalize = LocalSubspacePlan.normalize


@dataclass(frozen=True)
class DistributedSubspacePlan:
    local: object
    vector_sharding: NamedSharding
    orthogonalization_context: int | None = None

    @property
    def capacity(self):
        return self.local.capacity

    @property
    def n_eig(self):
        return self.local.n_eig

    @property
    def workspace_specs(self):
        width = self.local.max_block_size or self.n_eig
        return {**self.local.workspace_specs,
                **({'orthogonalization_ranges': jax.ShapeDtypeStruct(
                    (2*self.vector_sharding.mesh.size,), jnp.int32)}
                   if self.orthogonalization_context is not None else {}),
                'qr_stacked_r': jax.ShapeDtypeStruct(
                    (self.vector_sharding.mesh.size*width, width), jnp.complex128)}

    @property
    def axes(self):
        return tuple(a for item in self.vector_sharding.spec[1:] if item is not None
                     for a in (item if isinstance(item, tuple) else (item,)))

    def qr(self, rows):
        """Stable TSQR: gather only small R factors, never vector rows.

        Local tiles may have fewer entries than the block width. Their
        reduced QR contributes min(local_entries, width) rows to the stack.
        """
        width = rows.shape[0]
        if width > (self.local.max_block_size or self.n_eig):
            raise ValueError('QR block exceeds planned scratch width')
        def body(rows):
            flat = rows.reshape(width, -1)
            q_local, r_local = jnp.linalg.qr(flat.T, mode='reduced')
            r_stack = jax.lax.all_gather(r_local, self.axes, axis=0, tiled=True)
            q_small, r = jnp.linalg.qr(r_stack, mode='reduced')
            height = r_local.shape[0]
            start = jax.lax.axis_index(self.axes)*height
            factor = jax.lax.dynamic_slice_in_dim(q_small, start, height, axis=0)
            q = (q_local @ factor).T.reshape(rows.shape)
            return q, r
        return self._map(body, (self.vector_sharding.spec,),
                         (self.vector_sharding.spec, P()))(rows)

    def constrain(self, vectors):
        return jax.lax.with_sharding_constraint(vectors, self.vector_sharding)

    def _map(self, fn, specs, outputs):
        return jax.shard_map(fn, mesh=self.vector_sharding.mesh,
                             in_specs=specs, out_specs=outputs, check_vma=False)

    def eigh(self, h, active):
        return self._map(self.local.eigh, (P(), P()), (P(), P()))(h, active)

    def store(self, v, hv, p, hp, start, count):
        spec = self.vector_sharding.spec
        return self._map(self.local.store, (spec, spec, spec, spec, P(), P()),
                         (spec, spec))(v, hv, p, hp, start, count)

    def gram(self, v, p, active, *, start=0):
        def body(v, p, active, start):
            c = self.local.gram(v, p, active, start=start)
            return ordered_psum(c, self.axes)
        spec = self.vector_sharding.spec
        return self._map(body, (spec, spec, P(), P()), P())(v, p, active, start)

    def project(self, v, hv, active, h, start, count):
        # Reduce only the new panel. Reducing the previously global H again
        # would multiply its retained block by the processor count.
        def body(v, hv, active, start, count):
            delta = self.local.project(v, hv, active, jnp.zeros((self.capacity, self.capacity), jnp.complex128), start, count)
            return ordered_psum(delta, self.axes)
        spec = self.vector_sharding.spec
        delta = self._map(body, (spec, spec, P(), P(), P()), P())(v, hv, active, start, count)
        row = jnp.arange(self.capacity)
        new = (row >= start) & (row < start+count)
        mask = ((new[:, None] & (row[None, :] < active)) |
                (new[None, :] & (row[:, None] < active)))
        return jnp.where(mask, delta, h)

    def reconstruct(self, v, hv, c, active, template, *, columns=None, compute_image=True, start=0):
        columns = template.shape[0] if columns is None else columns
        def body(v, hv, c, active, template, columns, start):
            return self.local.reconstruct(v, hv, c, active, template,
                columns=columns, compute_image=compute_image, start=start)
        spec = self.vector_sharding.spec
        return self._map(body, (spec, spec, P(), P(), spec, P(), P()), (spec, spec))(
            v, hv, c, active, template, columns, start)

    def orthogonalize(self, v, p, active, *, start=0):
        if isinstance(self.local, LocalSubspacePlan) and self.orthogonalization_context is None:
            def body(v, p, active, start):
                start, active = _orthogonalization_range(start, active, self.capacity)
                c = ordered_psum(self.local.gram(v, p, active, start=start), self.axes)
                # Form the second pass's local overlaps while updating p in
                # place, avoiding a separate vector-sized projection buffer.
                # They still need a global sum before the second subtraction.
                p, c = self.local.subtract_projection(
                    v, p, c, active, start=start, next_gram=True)
                c = ordered_psum(c, self.axes)
                return self.local.subtract_projection(v, p, c, active, start=start)
            spec = self.vector_sharding.spec
            return self._map(body, (spec, spec, P(), P()), spec)(v, p, active, start)
        if self.orthogonalization_context is None:
            # Each callback sees only its local vector tile. Only coefficients
            # cross the JAX mesh; never call the CPU callback on global vectors.
            start, active = _orthogonalization_range(start, active, self.capacity)
            for _ in range(2):
                c = self.gram(v, p, active, start=start)
                x, _ = self.reconstruct(v, v, c, active, p,
                                        start=start, compute_image=False)
                p = p-x
            return p
        def body(v, p, active, start):
            return self.local.distributed_orthogonalize(
                v, p, active, start=start, context=self.orthogonalization_context,
                world=self.vector_sharding.mesh.size)
        spec = self.vector_sharding.spec
        return self._map(body, (spec, spec, P(), P()), spec)(v, p, active, start)

    normalize = LocalSubspacePlan.normalize


def _validate_plan_geometry(capacity, n_eig, vector_sharding, max_block_size):
    if not isinstance(capacity, int) or not isinstance(n_eig, int):
        raise TypeError('capacity and n_eig must be Python integers')
    if not 1 <= n_eig <= capacity < 2**31:
        raise ValueError('require 1 <= n_eig <= capacity < 2**31')
    if max_block_size is not None and (not isinstance(max_block_size, int) or not 1 <= max_block_size < 2**31):
        raise ValueError('max_block_size must be a positive Python integer below 2**31')
    if not jax.config.x64_enabled:
        raise ValueError('active subspace requires JAX x64')
    if vector_sharding is not None:
        if not isinstance(vector_sharding, NamedSharding):
            raise TypeError('vector_sharding must be NamedSharding')
        if vector_sharding.spec and vector_sharding.spec[0] is not None:
            raise ValueError('subspace row axis must be unsharded')
        used = {axis for item in vector_sharding.spec[1:] if item is not None
                for axis in (item if isinstance(item, tuple) else (item,))}
        required = {axis for axis, size in vector_sharding.mesh.shape.items() if size > 1}
        if not required.issubset(used):
            raise ValueError('vector axes must partition storage over every nontrivial mesh axis')


def _distribute(local, vector_sharding, *, native_collectives=False):
    if vector_sharding is None or vector_sharding.mesh.size == 1:
        return local
    context = None
    if isinstance(local, LocalSubspacePlan):
        local = replace(local, local_shard=True)
        mesh = vector_sharding.mesh
        leg = native_leg().key
        if native_collectives:
            if not (mesh.size == jax.process_count() and jax.local_device_count() == 1
                    and set(mesh.axis_names) == {'x', 'y'}):
                raise ValueError('native orthogonalization collectives need one GPU per process on the complete X/Y mesh')
            # Only a leg with a native communicator context builds this handler (CUDA:
            # cuSolverMp's NCCL comm); elsewhere the opt-in refuses, it never degrades.
            target = probe_target('lorrax_active_subspace_distributed_ortho', leg)
            if not target.ok:
                raise RuntimeError('GATE subspace_native_collectives: distributed orthogonalization '
                                   f'provider unavailable on the {leg} leg: {target}; '
                                   'fix: native_collectives=False (JAX psum of the coefficients)')
            from distrib_la._cusolvermp import context_key
            context = context_key(mesh, col_major=False)
        else:
            for name in ('gram', 'subtract', 'subtract_gram'):
                target = probe_target('lorrax_active_subspace_'+name, leg)
                if not target.ok:
                    raise RuntimeError(f'orthogonalization provider unavailable: {target}')
    return DistributedSubspacePlan(local, vector_sharding, context)


def _local_plan(route, capacity, n_eig, max_block_size):
    if route == 'native':
        return plan_local_subspace(capacity=capacity, n_eig=n_eig, max_block_size=max_block_size)
    if route == 'xla':
        return XlaSubspacePlan(capacity, n_eig, max_block_size)
    if route == 'host':
        return CpuSubspacePlan(capacity, n_eig, max_block_size)
    raise ValueError(f'GATE subspace_route: unknown subspace route {route!r}')


def plan_subspace(*, capacity, n_eig, vector_sharding=None, max_block_size=None,
                  native_collectives=False):
    """Resolve before tracing; ``vector_sharding`` includes the unsharded row axis.

    The provider is the leg's :data:`SUBSPACE_ROUTE`: CUDA and ROCm runtime-sized
    native BLAS/LAPACK (cuBLAS/cuSOLVER, rocBLAS/rocSOLVER), a vendor with no leg
    XLA device products, CPU active host LAPACK. All accept count/start windows
    without changing the compiled array sizes. Native collectives are an explicit
    opt-in because their context costs memory (native route; legs without the
    handler refuse by GATE subspace_native_collectives).
    """
    _validate_plan_geometry(capacity, n_eig, vector_sharding, max_block_size)
    local = _local_plan(subspace_route(), capacity, n_eig, max_block_size)
    return _distribute(local, vector_sharding, native_collectives=native_collectives)


def plan_orthogonalization(*, capacity, max_block_size, vector_sharding=None,
                           native_collectives=False):
    """Plan reusable CGS2 without allocating/querying an eigensolver workspace.

    Resolve before tracing (collectively with native_collectives=True). Arrays retain
    fixed maximum shapes; runtime start/count select only the active basis.
    CPU callbacks and JAX-collective compatibility routes retain their existing
    semantics. ``backend`` and ``workspace_specs`` describe the resolved plan.
    """
    if not isinstance(max_block_size, int) or max_block_size < 1:
        raise ValueError('max_block_size must be a positive Python integer')
    _validate_plan_geometry(capacity, 1, vector_sharding, max_block_size)
    route = subspace_route()
    if route == 'native':
        leg = native_leg()
        result = probe_target('lorrax_active_subspace_ortho', leg.key)
        if not result.ok:
            raise RuntimeError(f'orthogonalization provider unavailable: {result}')
        local = LocalSubspacePlan(capacity, 1, 0, max_block_size=max_block_size,
                                  **native_fields(leg))
    else:
        local = _local_plan(route, capacity, 1, max_block_size)
    implementation = _distribute(local, vector_sharding,
                                  native_collectives=native_collectives)
    def orthogonalize(basis, block, active, *, start=0):
        """Remove the selected basis components twice; do not normalize block."""
        if (basis.ndim < 2 or block.ndim != basis.ndim or
                basis.shape[0] != capacity or basis.shape[1:] != block.shape[1:] or
                not 1 <= block.shape[0] <= max_block_size):
            raise ValueError('orthogonalization operands differ from planned geometry')
        if basis.dtype != jnp.complex128 or block.dtype != jnp.complex128:
            raise TypeError('orthogonalization requires complex128')
        return implementation.orthogonalize(basis, block, active, start=start)

    orthogonalize.workspace_specs = {
        k: v for k, v in implementation.workspace_specs.items()
        if k in ('gram', 'blas', 'orthogonalization_ranges')}
    if isinstance(implementation, DistributedSubspacePlan):
        orthogonalize.backend = (f'{local.vendor}_nccl'
            if implementation.orthogonalization_context is not None
            else (f'{local.vendor}_jax_collectives' if isinstance(local, LocalSubspacePlan)
                  else 'xla_jax_collectives' if isinstance(local, XlaSubspacePlan)
                  else 'jax_collectives'))
    else:
        orthogonalize.backend = (local.vendor if isinstance(local, LocalSubspacePlan)
                                 else 'xla' if isinstance(local, XlaSubspacePlan)
                                 else 'cpu_callback')
    return orthogonalize

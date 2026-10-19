"""Fixed-capacity, active-prefix algebra for one-device iterative solvers.

The native provider (CUDA: cuBLAS/cuSOLVER; ROCM: rocBLAS/rocSOLVER, the same
leg-neutral handlers in ``src/ffi/cpp/active_subspace``) uses runtime dimensions in
BLAS/LAPACK. It never gathers
vectors from a distributed mesh. Size scalars are copied to the host inside
the FFI handlers: JIT-compatible does not mean host-synchronization-free.
The arrays returned as scratch by FFI are XLA-owned, fixed at compilation.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from distrib_la.loader import active_eigh_workspace, probe_target
from lxkit import device_vendor


def _spec(array):
    return jax.ShapeDtypeStruct(array.shape, array.dtype)


def _orthogonalization_range(start, count, capacity):
    """Refuse invalid wide metadata instead of wrapping it into a valid int32."""
    start, count = jnp.asarray(start), jnp.asarray(count)
    if (start.ndim or count.ndim or
            not jnp.issubdtype(start.dtype, jnp.integer) or
            not jnp.issubdtype(count.dtype, jnp.integer)):
        raise TypeError('orthogonalization start/count must be integer scalars')
    valid = ((start >= 0) & (start <= capacity) & (count >= 0) &
             (count <= capacity) & (start+count <= capacity))
    descriptor = jnp.stack((start.astype(jnp.int32), count.astype(jnp.int32)))
    return jnp.where(valid, descriptor, jnp.full((2,), -1, jnp.int32))


def _flat(array, *, local_shard=False):
    if array.dtype != jnp.complex128:
        raise TypeError('active subspace currently requires complex128')
    # Refuse a distributed eager operand before any reshape/custom call.
    sharding = getattr(array, 'sharding', None)
    abstract = getattr(getattr(array, 'aval', None), 'sharding', None)
    if not local_shard and getattr(getattr(abstract, 'mesh', None), 'size', 1) > 1:
        raise ValueError('active subspace cannot trace distributed operands')
    if not local_shard and sharding is not None and len(sharding.device_set) != 1:
        raise ValueError('active subspace requires one-device operands')
    return array.reshape(array.shape[0], -1)


@dataclass(frozen=True)
class LocalSubspacePlan:
    """Resolved complex128 native operations; vector axes are process-local.

    ``vendor`` is the device leg's vendor (``lxkit.LEGS[...].vendor``: ``cuda`` or
    ``rocm``); it names the plan's backend and never selects code (the lowering
    platform picks the leg's registered handler).  ``handlers`` is the leg's
    implementation (:data:`NATIVE_HANDLERS`): ``"blas"`` (CUDA: cuBLAS/cuSOLVERDn
    on the active dimensions, descriptors read on the host) or ``"devsize"``
    (ROCM: run-time compiled kernels reading the descriptors on the device -- no
    per-call host sync -- and the capacity eigensolve, :meth:`_capacity_eigh`).
    """
    capacity: int
    n_eig: int
    lwork: int
    local_shard: bool = False
    max_block_size: int | None = None
    vendor: str = 'cuda'
    handlers: str = 'blas'
    leg_key: str = 'CUDA'
    cubin_dir: str = ''

    def qr(self, rows):
        """Reduced QR of a row-vector block; no capacity buffer is involved."""
        if rows.shape[0] > (self.max_block_size or self.n_eig):
            raise ValueError('QR block exceeds planned scratch width')
        flat = self._flat(rows)
        if flat.shape[1] < rows.shape[0]:
            raise ValueError('QR needs at least as many vector entries as rows')
        q, r = jnp.linalg.qr(flat.T, mode='reduced')
        return q.T.reshape(rows.shape), r

    def constrain(self, vectors):
        return vectors

    def _ffi_call(self, *args, **kwargs):
        """``jax.ffi.ffi_call`` with the leg's handler attributes bound (devsize: the
        run-time compiled kernels' image cache directory; blas: none)."""
        call = jax.ffi.ffi_call(*args, **kwargs)
        if self.handlers != 'devsize':
            return call
        return lambda *a, **k: call(*a, cubin_dir=self.cubin_dir, **k)

    def _flat(self, array):
        return _flat(array, local_shard=self.local_shard)

    @property
    def workspace_specs(self):
        """Declared eigensolve and Gram scratch, excluding the caller's arrays."""
        cap, b = self.capacity, self.n_eig
        if self.handlers == 'devsize':   # the capacity eigensolve's scratch is XLA's allocator
            return {
                'gram': jax.ShapeDtypeStruct((cap, self.max_block_size or b), jnp.complex128),
                'blas': jax.ShapeDtypeStruct((4*1024*1024,), jnp.uint8),
                'qr_r': jax.ShapeDtypeStruct(((self.max_block_size or b),)*2, jnp.complex128),
            }
        return {
            'eigh': jax.ShapeDtypeStruct((cap*cap+self.lwork,), jnp.complex128),
            'eigenvalues': jax.ShapeDtypeStruct((cap,), jnp.float64),
            'info': jax.ShapeDtypeStruct((1,), jnp.int32),
            'gram': jax.ShapeDtypeStruct((cap, self.max_block_size or b), jnp.complex128),
            'blas': jax.ShapeDtypeStruct((4*1024*1024,), jnp.uint8),
            'qr_r': jax.ShapeDtypeStruct(((self.max_block_size or b),)*2, jnp.complex128),
        }

    def eigh(self, h, active):
        """Lowest active pairs; missing values are inf and unused columns zero."""
        cap, b = self.capacity, self.n_eig
        if h.shape != (cap, cap) or h.dtype != jnp.complex128:
            raise ValueError('projected Hamiltonian differs from the declared plan')
        if self.handlers == 'devsize':
            return self._capacity_eigh(h, active)
        outputs = (
            jax.ShapeDtypeStruct((b,), jnp.float64),
            jax.ShapeDtypeStruct((cap, b), jnp.complex128),
            *tuple(self.workspace_specs[k] for k in ('eigh', 'eigenvalues', 'info')),
        )
        call = self._ffi_call(
            'lorrax_active_subspace_eigh', outputs,
            input_layouts=[(1, 0), ()],
            output_layouts=[(0,), (1, 0), (0,), (0,), (0,)],
            vmap_method='sequential')
        e, c, _, _, info = call(self._flat(h), jnp.asarray(active, jnp.int32))
        e = jnp.where(jnp.arange(b) < active, e, jnp.inf)
        return jnp.where(info[0] == 0, e, jnp.nan), c

    def _capacity_eigh(self, h, active):
        """The eigensolve with no size read: one rocSOLVER call on the CAPACITY matrix.

        A is the active block's lower triangle mirrored (the host and XLA routes' matrix);
        the inactive diagonal holds distinct values in [shift, 2 shift), shift twice the
        block's Gershgorin bound, so every active eigenvalue sorts first and no padded
        cluster is degenerate.  rocSOLVER zheevd fails on structured (block-diagonal,
        real-valued) matrices (``local_eigh`` module docstring), so the solve is of
        B = H A H with ``local_eigh``'s fixed dense reflector, through the batched target
        at batch 1 (``lorrax_rocsolver_batched_eigh``: workspace from XLA's allocator,
        a failed or non-finite solve NaN).  C = H C_B, masked to the active block; a
        scalar active block is answered exactly.  Cost O(capacity^3) per call.
        """
        from distrib_la.local_eigh import BATCHED_EIGH_ROUTE, _native_batched_eigh, _reflector
        cap, b = self.capacity, self.n_eig
        m = jnp.asarray(active, jnp.int32)
        idx = jnp.arange(cap)
        inside = idx < m
        block = inside[:, None] & inside[None, :]
        a = jnp.where(block, h, 0)
        a = jnp.tril(a) + jnp.conj(jnp.tril(a, -1)).T
        bound = jnp.max(jnp.sum(jnp.abs(a), axis=1))
        shift = jnp.where(bound > 0, 2 * bound, 1.0)
        a = a + jnp.diag(jnp.where(inside, 0.0, shift * (1.0 + idx / cap))).astype(a.dtype)
        u = _reflector(cap, jnp.complex128)
        au = a @ u
        uau = jnp.real(jnp.vdot(u, au))
        bm = (a - 2 * u[:, None] * jnp.conj(au)[None, :] - 2 * au[:, None] * jnp.conj(u)[None, :]
              + 4 * uau * (u[:, None] * jnp.conj(u)[None, :]))
        w, vb = _native_batched_eigh(BATCHED_EIGH_ROUTE[self.leg_key], bm[None])
        w, vb = w[0], vb[0]
        c = (vb - 2 * u[:, None] * (jnp.conj(u) @ vb)[None, :])[:, :b]
        e = w[:b]
        column = jnp.arange(b) < m
        alpha = a[0, 0]
        scalar = jnp.all(jnp.where(block, a == alpha * jnp.eye(cap, dtype=a.dtype), True))
        e = jnp.where(scalar, jnp.real(alpha), e)
        c = jnp.where(scalar, jnp.eye(cap, b, dtype=c.dtype), c)
        keep = inside[:, None] & column[None, :]
        c = jnp.where(keep, c, 0)
        ok = scalar | (jnp.all(jnp.isfinite(e)) & jnp.all(jnp.isfinite(c)))
        e = jnp.where(column, e, jnp.inf)
        return jnp.where(ok, e, jnp.nan), c

    def store(self, v, hv, p, hp, start, count):
        """Update only an active row interval, with XLA-declared buffer aliases."""
        if v.shape != hv.shape or p.shape != hp.shape or v.shape[0] != self.capacity or v.shape[1:] != p.shape[1:]:
            raise ValueError('active store buffer geometry differs from plan')
        vf, hf = self._flat(v), self._flat(hv)
        call = self._ffi_call(
            'lorrax_active_subspace_store', (_spec(vf), _spec(hf)),
            input_layouts=[(0, 1), (0, 1), (0, 1), (0, 1), (0,)],
            output_layouts=[(0, 1), (0, 1)], input_output_aliases={0: 0, 1: 1},
            vmap_method='sequential')
        vv, hh = call(vf, hf, self._flat(p), self._flat(hp), jnp.array([start, count], jnp.int32))
        return vv.reshape(v.shape), hh.reshape(hv.shape)

    def project(self, v, hv, active, h, start, count):
        if v.shape != hv.shape or v.shape[0] != self.capacity or h.shape != (self.capacity, self.capacity):
            raise ValueError('active projection buffer geometry differs from plan')
        call = self._ffi_call(
            'lorrax_active_subspace_project', (_spec(h), self.workspace_specs['blas']),
            input_layouts=[(0, 1), (0, 1), (1, 0), (0,)],
            output_layouts=[(1, 0), (0,)], input_output_aliases={2: 0}, vmap_method='sequential')
        result, _ = call(self._flat(v), self._flat(hv), self._flat(h),
                         jnp.array([active, start, count], jnp.int32))
        return result

    def reconstruct(self, v, hv, c, active, template, *, columns=None, compute_image=True, start=0):
        if v.shape != hv.shape or v.shape[0] != self.capacity or c.shape != (self.capacity, template.shape[0]):
            raise ValueError('active reconstruction buffer geometry differs from plan')
        if v.shape[1:] != template.shape[1:]:
            raise ValueError('active reconstruction vector shapes differ')
        flat = self._flat(template)
        call = self._ffi_call(
            'lorrax_active_subspace_reconstruct', (_spec(flat), _spec(flat), self.workspace_specs['blas']),
            input_layouts=[(0, 1), (0, 1), (1, 0), (0,)],
            output_layouts=[(0, 1), (0, 1), (0,)], vmap_method='sequential')
        columns = template.shape[0] if columns is None else columns
        x, hx, _ = call(self._flat(v), self._flat(hv), self._flat(c), jnp.array([active, columns, compute_image, start], jnp.int32))
        return x.reshape(template.shape), hx.reshape(template.shape)

    def orthogonalize(self, v, p, active, *, start=0):
        if p.shape[0] > (self.max_block_size or self.n_eig):
            raise ValueError("orthogonalization block exceeds planned scratch width")
        if v.shape[0] != self.capacity or v.shape[1:] != p.shape[1:]:
            raise ValueError('active orthogonalization buffer geometry differs from plan')
        work = jax.ShapeDtypeStruct((v.shape[0], p.shape[0]), v.dtype)
        call = self._ffi_call(
            'lorrax_active_subspace_ortho', (_spec(self._flat(p)), work, self.workspace_specs['blas']),
            input_layouts=[(0, 1), (0, 1), (0,)],
            output_layouts=[(0, 1), (1, 0), (0,)], input_output_aliases={1: 0},
            vmap_method='sequential')
        result, _, _ = call(self._flat(v), self._flat(p),
                             _orthogonalization_range(start, active, self.capacity))
        return result.reshape(p.shape)

    def distributed_orthogonalize(self, v, p, active, *, start, context, world):
        """Two native CGS passes; only active coefficients cross the mesh."""
        if p.shape[0] > (self.max_block_size or self.n_eig):
            raise ValueError('orthogonalization block exceeds planned scratch width')
        if v.shape[0] != self.capacity or v.shape[1:] != p.shape[1:]:
            raise ValueError('active orthogonalization buffer geometry differs from plan')
        flat = self._flat(p)
        outputs = (_spec(flat),
                   jax.ShapeDtypeStruct((self.capacity*p.shape[0],), jnp.complex128),
                   jax.ShapeDtypeStruct((2*world,), jnp.int32),
                   self.workspace_specs['blas'])
        call = self._ffi_call(
            'lorrax_active_subspace_distributed_ortho', outputs,
            input_layouts=[(0, 1), (0, 1), (0,)],
            output_layouts=[(0, 1), (0,), (0,), (0,)],
            input_output_aliases={1: 0}, has_side_effect=True,
            vmap_method='sequential')
        result, _, _, _ = call(self._flat(v), flat,
            _orthogonalization_range(start, active, self.capacity), ctx_key=context)
        return result.reshape(p.shape)

    def subtract_projection(self, v, p, coefficients, active, *, start=0,
                            next_gram=False):
        """Subtract in place, optionally forming the next local overlap matrix."""
        if (v.shape[0] != self.capacity or v.shape[1:] != p.shape[1:] or
                coefficients.shape != (self.capacity, p.shape[0]) or
                not 1 <= p.shape[0] <= (self.max_block_size or self.n_eig)):
            raise ValueError('projection subtraction differs from planned geometry')
        flat = self._flat(p)
        if next_gram:
            outputs = (_spec(flat), _spec(coefficients), self.workspace_specs['blas'])
            layouts = [(0, 1), (1, 0), (0,)]
            aliases = {1: 0, 2: 1}
        else:
            outputs = (_spec(flat), self.workspace_specs['blas'])
            layouts, aliases = [(0, 1), (0,)], {1: 0}
        call = self._ffi_call(
            'lorrax_active_subspace_subtract'+('_gram' if next_gram else ''),
            outputs, input_layouts=[(0, 1), (0, 1), (1, 0), (0,)],
            output_layouts=layouts, input_output_aliases=aliases,
            vmap_method='sequential')
        result = call(self._flat(v), flat, self._flat(coefficients),
                      _orthogonalization_range(start, active, self.capacity))
        return ((result[0].reshape(p.shape), result[1]) if next_gram
                else result[0].reshape(p.shape))


    def gram(self, v, p, active, *, start=0):
        if p.shape[0] > (self.max_block_size or self.n_eig):
            raise ValueError("Gram block exceeds planned scratch width")
        if v.shape[0] != self.capacity or v.shape[1:] != p.shape[1:]:
            raise ValueError('active Gram buffer geometry differs from plan')
        work = jax.ShapeDtypeStruct((self.capacity, p.shape[0]), v.dtype)
        call = self._ffi_call(
            'lorrax_active_subspace_gram', (work, self.workspace_specs['blas']),
            input_layouts=[(0, 1), (0, 1), (0,)],
            output_layouts=[(1, 0), (0,)], vmap_method='sequential')
        result, _ = call(self._flat(v), self._flat(p), jnp.array([start, active], jnp.int32))
        return result

    def normalize(self, p, rank):
        """Second Gram whitening of only the retained correction prefix.

        Rank discovery happened before this call. No zero tail enters a
        Gram product, eigensolve, or vector reconstruction here.
        """
        if p.shape[0] != self.capacity or self.n_eig != self.capacity:
            raise ValueError('normalization requires a square block-sized plan')
        def retained(p):
            h = jnp.zeros((self.capacity, self.capacity), p.dtype)
            h = self.project(p, p, rank, h, 0, rank)
            e, c = self.eigh(h, rank)
            c = c / jnp.sqrt(e)[None, :]
            result, _ = self.reconstruct(p, p, c, rank, p, columns=rank, compute_image=False)
            return result
        return self.constrain(jax.lax.cond(
            rank > 0, lambda p: self.constrain(retained(p)),
            self.constrain, p))


#: The handlers every native local plan calls (``distributed_ortho`` is the
#: opt-in collective one, probed by ``subspace._distribute``).
NATIVE_LOCAL_OPS = ('eigh', 'project', 'reconstruct', 'ortho', 'store', 'gram',
                    'subtract', 'subtract_gram')

#: The native implementation per device leg -- a backend preference of this service (as
#: ``subspace.SUBSPACE_ROUTE``).  ``"blas"``: cuBLAS/cuSOLVERDn on the active dimensions
#: (``active_ops.cc``, ``active_eigh.cc``; each handler reads its descriptor on the host).
#: ``"devsize"``: ``active_devsize.cc``, run-time compiled kernels that read the descriptor on
#: the device (no per-call host sync, owner rule), and :meth:`LocalSubspacePlan._capacity_eigh`
#: (no ``lorrax_active_subspace_eigh``).
NATIVE_HANDLERS = {"CUDA": "blas", "ROCM": "devsize"}


def native_fields(leg):
    """The plan fields of ``leg``'s native implementation (and probe its targets)."""
    handlers = NATIVE_HANDLERS[leg.key]
    ops = NATIVE_LOCAL_OPS if handlers == 'blas' else NATIVE_LOCAL_OPS[1:]
    for op in ops:
        result = probe_target('lorrax_active_subspace_'+op, leg.key)
        if not result.ok:
            raise RuntimeError(f'active subspace provider unavailable: {result}')
    cubin_dir = ''
    if handlers == 'devsize':
        from distrib_la.local_eigh import BATCHED_EIGH_ROUTE
        result = probe_target(BATCHED_EIGH_ROUTE[leg.key].target, leg.key)
        if not result.ok:
            raise RuntimeError(f'active subspace capacity eigensolve unavailable: {result}')
        from lxkit import user_cache_dir
        cubin_dir = str(user_cache_dir('active_subspace'))
    return dict(vendor=leg.vendor, handlers=handlers, leg_key=leg.key, cubin_dir=cubin_dir)


def native_leg():
    """The process's device leg (``lxkit.LEGS`` entry) for the native provider.

    Refuses by GATE on a process without a device leg: the native handlers are
    device code (the host leg's provider is ``CpuSubspacePlan``).
    """
    from lxkit import leg_for_vendor
    vendor = device_vendor(jax.local_devices()[0])
    leg = leg_for_vendor(vendor)
    if leg is None or leg.key == 'cpu':
        raise ValueError(f'GATE subspace_native: the native active-subspace provider needs a '
                         f'device leg (CUDA or ROCM); this process runs {vendor!r}')
    return leg


def plan_local_subspace(*, capacity: int, n_eig: int, max_block_size: int | None = None) -> LocalSubspacePlan:
    """Resolve the native provider of this process's device leg once, before tracing.

    No CPU/distributed fallback is implied. Changing the maximum capacity or
    block size creates a new plan; changing the active dimension does not.
    """
    if not isinstance(capacity, int) or not isinstance(n_eig, int):
        raise TypeError('capacity and n_eig must be Python integers')
    if not 1 <= n_eig <= capacity < 2**31:
        raise ValueError('require 1 <= n_eig <= capacity < 2**31')
    if not jax.config.x64_enabled:
        raise ValueError('local active subspace requires JAX x64')
    leg = native_leg()
    fields = native_fields(leg)
    if max_block_size is not None and (not isinstance(max_block_size, int) or not 1 <= max_block_size < 2**31):
        raise ValueError('max_block_size must be a positive Python integer below 2**31')
    lwork = active_eigh_workspace(capacity, leg.key) if fields['handlers'] == 'blas' else 0
    return LocalSubspacePlan(capacity, n_eig, lwork, max_block_size=max_block_size, **fields)

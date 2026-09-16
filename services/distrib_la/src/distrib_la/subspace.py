"""Provider resolution and distributed active-subspace algebra.

Only row coefficients are reduced. Vector storage remains partitioned over
all mesh axes declared by the caller; no vector gather is permitted here.
The CPU compatibility provider uses active NumPy BLAS/LAPACK callbacks, so
it promises fixed JAX geometry, not GPU-like memory or transfer performance.
"""
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from distrib_la.active_subspace import LocalSubspacePlan, _orthogonalization_range, plan_local_subspace
from distrib_la.loader import probe_target


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
        return jax.pure_callback(work, (
            jax.ShapeDtypeStruct((self.n_eig,), jnp.float64),
            jax.ShapeDtypeStruct((self.capacity, self.n_eig), jnp.complex128)), h, active)

    def gram(self, v, p, active, *, start=0):
        def work(v, p, start, count):
            start, count = int(start), int(count)
            out = np.zeros((self.capacity, p.shape[0]), np.complex128)
            out[start:start+count] = v[start:start+count].reshape(count, -1).conj() @ p.reshape(p.shape[0], -1).T if count else 0
            return out
        return jax.pure_callback(work, jax.ShapeDtypeStruct((self.capacity, p.shape[0]), jnp.complex128), v, p, start, active)

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
        return jax.pure_callback(work, (spec, spec), v, hv, c, active, columns, start)

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
        return jax.pure_callback(work, jax.ShapeDtypeStruct(p.shape, p.dtype),
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
        return jax.pure_callback(work, jax.ShapeDtypeStruct(h.shape, h.dtype), v, hv, h, active, start, count)

    def qr(self, rows):
        flat = rows.reshape(rows.shape[0], -1)
        if flat.shape[1] < rows.shape[0]:
            raise ValueError('QR needs at least as many vector entries as rows')
        q, r = jnp.linalg.qr(flat.T, mode='reduced')
        return q.T.reshape(rows.shape), r

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
            return jax.lax.psum(c, self.axes)
        spec = self.vector_sharding.spec
        return self._map(body, (spec, spec, P(), P()), P())(v, p, active, start)

    def project(self, v, hv, active, h, start, count):
        # Reduce only the new panel. Reducing the previously global H again
        # would multiply its retained block by the processor count.
        def body(v, hv, active, start, count):
            delta = self.local.project(v, hv, active, jnp.zeros((self.capacity, self.capacity), jnp.complex128), start, count)
            return jax.lax.psum(delta, self.axes)
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
                c = jax.lax.psum(self.local.gram(v, p, active, start=start), self.axes)
                # Form the second pass's local overlaps while updating p in
                # place, avoiding a separate vector-sized projection buffer.
                # They still need a global sum before the second subtraction.
                p, c = self.local.subtract_projection(
                    v, p, c, active, start=start, next_gram=True)
                c = jax.lax.psum(c, self.axes)
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
        if native_collectives:
            if not (mesh.size == jax.process_count() and jax.local_device_count() == 1
                    and set(mesh.axis_names) == {'x', 'y'}):
                raise ValueError('native orthogonalization collectives need one GPU per process on the complete X/Y mesh')
            target = probe_target('lorrax_active_subspace_distributed_ortho', 'CUDA')
            if not target.ok:
                raise RuntimeError(f'distributed orthogonalization provider unavailable: {target}')
            from distrib_la._cusolvermp import get_or_init_context
            context = get_or_init_context(mesh, col_major=False)
        else:
            for name in ('gram', 'subtract', 'subtract_gram'):
                target = probe_target('lorrax_active_subspace_'+name, 'CUDA')
                if not target.ok:
                    raise RuntimeError(f'orthogonalization provider unavailable: {target}')
    return DistributedSubspacePlan(local, vector_sharding, context)


def plan_subspace(*, capacity, n_eig, vector_sharding=None, max_block_size=None,
                  native_collectives=False):
    """Resolve before tracing; ``vector_sharding`` includes the unsharded row axis.

    CUDA uses runtime-sized native BLAS/LAPACK. CPU uses active host LAPACK.
    Both accept count/start windows without changing the compiled array sizes.
    Native collectives are an explicit opt-in because their context costs memory.
    """
    _validate_plan_geometry(capacity, n_eig, vector_sharding, max_block_size)
    local = (plan_local_subspace(capacity=capacity, n_eig=n_eig, max_block_size=max_block_size)
             if jax.local_devices()[0].platform == 'gpu' else CpuSubspacePlan(capacity, n_eig, max_block_size))
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
    if jax.local_devices()[0].platform == 'gpu':
        result = probe_target('lorrax_active_subspace_ortho', 'CUDA')
        if not result.ok:
            raise RuntimeError(f'orthogonalization provider unavailable: {result}')
        local = LocalSubspacePlan(capacity, 1, 0, max_block_size=max_block_size)
    else:
        local = CpuSubspacePlan(capacity, 1, max_block_size)
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
        orthogonalize.backend = ('cuda_nccl'
            if implementation.orthogonalization_context is not None
            else ('cuda_jax_collectives' if isinstance(local, LocalSubspacePlan)
                  else 'jax_collectives'))
    else:
        orthogonalize.backend = ('cuda' if isinstance(local, LocalSubspacePlan)
                                 else 'cpu_callback')
    return orthogonalize

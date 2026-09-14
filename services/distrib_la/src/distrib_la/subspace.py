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

from distrib_la.active_subspace import LocalSubspacePlan, plan_local_subspace


def _merge_projection(h, delta, start, count, active):
    """Retain old global entries; only the newly reduced panel replaces them."""
    row = jnp.arange(h.shape[0])
    new = (row >= start) & (row < start+count)
    mask = ((new[:, None] & (row[None, :] < active)) |
            (new[None, :] & (row[:, None] < active)))
    return jnp.where(mask, delta, h)


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
        for _ in range(2):
            c = self.gram(v, p, active, start=start)
            x, _ = self.reconstruct(v, v, c, active, p, start=start, compute_image=False)
            p = p-x
        return p

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

    def store_project(self, v, hv, p, hp, h, start, count):
        v, hv = self.store(v, hv, p, hp, start, count)
        return v, hv, self.project(v, hv, start+count, h, start, count)

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
        return _merge_projection(h, delta, start, count, active)

    def store_project(self, v, hv, p, hp, h, start, count):
        def body(v, hv, p, hp, start, count):
            v, hv, delta = self.local.store_project(
                v, hv, p, hp,
                jnp.zeros((self.capacity, self.capacity), jnp.complex128),
                start, count)
            return v, hv, jax.lax.psum(delta, self.axes)
        spec = self.vector_sharding.spec
        v, hv, delta = self._map(
            body, (spec, spec, spec, spec, P(), P()), (spec, spec, P()))(
                v, hv, p, hp, start, count)
        return v, hv, _merge_projection(h, delta, start, count, start+count)

    def reconstruct(self, v, hv, c, active, template, *, columns=None, compute_image=True, start=0):
        columns = template.shape[0] if columns is None else columns
        def body(v, hv, c, active, template, columns, start):
            return self.local.reconstruct(v, hv, c, active, template,
                columns=columns, compute_image=compute_image, start=start)
        spec = self.vector_sharding.spec
        return self._map(body, (spec, spec, P(), P(), spec, P(), P()), (spec, spec))(
            v, hv, c, active, template, columns, start)

    orthogonalize = CpuSubspacePlan.orthogonalize
    normalize = LocalSubspacePlan.normalize


def plan_subspace(*, capacity, n_eig, vector_sharding=None, max_block_size=None):
    """Resolve before tracing; ``vector_sharding`` includes the unsharded row axis.

    CUDA uses runtime-sized native BLAS/LAPACK. CPU uses active host LAPACK.
    Both accept count/start windows without changing the compiled array sizes.
    """
    if not isinstance(capacity, int) or not isinstance(n_eig, int):
        raise TypeError('capacity and n_eig must be Python integers')
    if not 1 <= n_eig <= capacity < 2**31:
        raise ValueError('require 1 <= n_eig <= capacity < 2**31')
    if max_block_size is not None and (not isinstance(max_block_size, int) or max_block_size < 1):
        raise ValueError('max_block_size must be a positive Python integer')
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
    local = (plan_local_subspace(capacity=capacity, n_eig=n_eig, max_block_size=max_block_size)
             if jax.local_devices()[0].platform == 'gpu' else CpuSubspacePlan(capacity, n_eig, max_block_size))
    if vector_sharding is None or vector_sharding.mesh.size == 1:
        return local
    if isinstance(local, LocalSubspacePlan):
        local = replace(local, local_shard=True)
    return DistributedSubspacePlan(local, vector_sharding)

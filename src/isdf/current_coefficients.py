"""Consumed radial packets solved by the actual three current fit factors.

This internal operand carries separate final coefficient banks. It is
not an alternate factor, fit or current-admission policy. The ordinary raw
RHS route remains available; a completed stream authenticates the same
ZetaG objects, factor leaves, q rows, centroid basis and field identities
before the grouped provider can consume it.
"""
from __future__ import annotations

import numpy as np


def radial_packet_store(mesh, atoms, harmonics, radial_points, radial_packet,
                        feature_count):
    """One authoritative atom/LM/radial map, donating the final bank."""
    import jax
    import jax.numpy as jnp
    from jax import shard_map
    from jax.sharding import PartitionSpec as P
    qs = P(('x', 'y'), None, None)
    zero = jnp.int32(0)

    def store(out, packet, first):
        values = packet[..., :feature_count].reshape(
            packet.shape[0], packet.shape[1], atoms, harmonics, radial_packet)
        view = out.reshape(out.shape[0], out.shape[1], atoms, harmonics, radial_points)
        view = jax.lax.dynamic_update_slice(view, values, (zero, zero, zero, zero, first))
        return view.reshape(out.shape)

    return jax.jit(shard_map(store, mesh=mesh, in_specs=(qs, qs, P()),
                            out_specs=qs, check_vma=False), donate_argnums=(0,))


def _factor_leaves(zeta):
    factor = zeta.factor
    leaves = factor if isinstance(factor, tuple) else (factor,)
    if any(value is None for value in leaves):
        raise ValueError('streamed current coefficients require actual live OWN factor leaves')
    return leaves


def _fit_binding(zeta):
    return (zeta.fit_vertex_mu_L, zeta.fit_augmentation_identity,
            zeta.fit_current_angular_support_identity,
            tuple(np.asarray(zeta.fit_q_full_indices, dtype=int)),
            zeta.solver_kind, int(zeta.n_rmu_solve),
            id(zeta.mu_basis), id(zeta.mesh))


class CurrentCoefficientBanks:
    """Sealed OWN3 coefficient payload, produced only by a completed stream."""
    __slots__ = ('_zetas', '_factors', '_bindings', '_banks', '_bank_ids',
                 'shape', 'radial_shape', 'identity', 'angular_identity',
                 '_operand_kind', '_smooth_support')

    @property
    def operand_kind(self):
        return self._operand_kind

    @property
    def smooth_support(self):
        return self._smooth_support

    def __init__(self, stream):
        if not isinstance(stream, CurrentCoefficientStream) or not stream._complete():
            raise ValueError('current coefficients require complete nonoverlapping radial packets')
        stream._authenticate()
        self._zetas, self._factors, self._bindings = (
            stream._zetas, stream._factors, stream._bindings)
        self._banks = tuple(tuple(pair) for pair in stream._banks)
        self._bank_ids = tuple(id(v) for pair in self._banks for v in pair)
        self.shape, self.radial_shape = stream.shape, stream.radial_shape
        self.identity, self.angular_identity = stream.identity, stream.angular_identity
        self._operand_kind, self._smooth_support = stream.operand_kind, stream.smooth_support
        stream._banks = None

    def authenticate(self, zetas):
        from isdf.zeta_mubatch import _require_q_owned
        if (self._banks is None
                or tuple(id(z) for z in zetas) != tuple(id(z) for z in self._zetas)
                or tuple(_fit_binding(z) for z in zetas) != self._bindings
                or any(any(a is not b for a, b in zip(_factor_leaves(z), saved))
                       or len(_factor_leaves(z)) != len(saved)
                       for z, saved in zip(zetas, self._factors))
                or tuple(id(v) for pair in self._banks for v in pair) != self._bank_ids):
            raise ValueError('streamed current factor/source/q/basis or bank owner changed')
        if self.smooth_support is not None:
            self.smooth_support.authenticate(zetas)
        for pair in self._banks:
            for bank in pair:
                _require_q_owned(bank, zetas[0].mesh, self.shape,
                                 name='streamed solved current half-bank')
        return self._banks

    def close(self):
        """Release banks explicitly even if procedural provider closures cycle."""
        if self._smooth_support is not None:
            self._smooth_support.close()
        self._banks = None
        self._zetas = self._factors = ()
        self._smooth_support = None


class CurrentCoefficientStream:
    """Consume compressed packets through canonical OWN solves and donated stores.

    Construction compiles the exact solve/store owners and admits the final
    final banks plus their temporary and caller-priced external live set
    before allocating. The caller owns the external lifetime bound; this
    is not admission of the later pair/onsite contraction.
    Each append consumes its packet; callers must drop aliases to that RHS.
    The default carries delta and PS. A private delta-only same-Z hybrid
    requires a certificate derived from actual source/body reciprocal support.
    """
    def __init__(self, zetas, *, atoms, harmonics, radial_points, radial_packet,
                 identity, angular_identity, resident_bytes=0., packet_features=None,
                 operand_kind='delta_and_ps', smooth_support=None):
        import jax
        import jax.numpy as jnp
        from jax.sharding import NamedSharding, PartitionSpec as P
        from isdf.zeta_mubatch import _local_coefficient_solve
        self._zetas = tuple(zetas)
        if len(self._zetas) != 3:
            raise ValueError('streamed current coefficients require the complete OWN3 family')
        if operand_kind not in ('delta_and_ps', 'delta'):
            raise ValueError('unsupported streamed current operand kind')
        if operand_kind == 'delta':
            from isdf.current_support import SmoothCurrentSupport
            if not isinstance(smooth_support, SmoothCurrentSupport):
                raise ValueError('delta-only current banks require actual same-Z support certification')
            smooth_support.authenticate(self._zetas)
        elif smooth_support is not None:
            raise ValueError('six-bank incumbent does not consume a same-Z specialization')
        self.operand_kind, self.smooth_support = operand_kind, smooth_support
        self._kinds = 2 if operand_kind == 'delta_and_ps' else 1
        z0 = self._zetas[0]
        st = z0.store
        self.radial_shape = tuple(map(int, (atoms, harmonics, radial_points, radial_packet)))
        na, nh, nr, rp = self.radial_shape
        if min(self.radial_shape) < 1 or nr % rp:
            raise ValueError('streamed current radial packets must exactly cover the radial axis')
        self.identity, self.angular_identity = identity, angular_identity
        self.shape = (int(st.Q_pad), int(st.mu_pad), na * nh * nr)
        self.packet_features = na * nh * rp
        self.packet_width = self.packet_features if packet_features is None else int(packet_features)
        if self.packet_width < self.packet_features or (packet_features is not None
                and self.packet_width != packet_features):
            raise ValueError('streamed current packet capacity must cover its exact physical features')
        self._bindings = tuple(_fit_binding(z) for z in self._zetas)
        for channel, z in enumerate(self._zetas, 1):
            if (z.mesh is not z0.mesh or z.mu_basis is not z0.mu_basis
                    or (z.store.Q, z.store.Q_pad, z.store.mu_pad) != (st.Q, st.Q_pad, st.mu_pad)
                    or z.fit_vertex_mu_L != channel
                    or z.fit_augmentation_identity != identity
                    or z.fit_current_angular_support_identity != angular_identity
                    or not np.array_equal(z.fit_q_full_indices, z0.fit_q_full_indices)):
                raise ValueError('streamed current coefficients require exact OWN/source/q/basis binding')
        self._factors = tuple(_factor_leaves(z) for z in self._zetas)
        self._solves = tuple(_local_coefficient_solve(z.mesh, z.solver_kind,
            z.n_rmu_solve, st.Q, donate_rhs=True) for z in self._zetas)
        self._store = radial_packet_store(z0.mesh, na, nh, nr, rp, self.packet_features)
        sharding = NamedSharding(z0.mesh, P(('x', 'y'), None, None))
        # Compile the exact owners before any final cloud exists. In
        # particular a nonaliased full-bank update would defeat streaming.
        packet = jax.ShapeDtypeStruct((self.shape[0], self.shape[1], self.packet_width),
                                     jnp.complex128, sharding=sharding)
        cloud = jax.ShapeDtypeStruct(self.shape, jnp.complex128, sharding=sharding)
        self.aot_quotes = []
        for name, fn, args in (
                *((f'OWN{c+1}_packet_solve', fn, (z.factor, packet))
                  for c, (z, fn) in enumerate(zip(self._zetas, self._solves))),
                ('donated_final_bank_store', self._store, (cloud, packet, jnp.int32(0)))):
            stats = fn.lower(*args).compile().memory_analysis()
            quote = dict(owner=name, arguments=int(stats.argument_size_in_bytes),
                outputs=int(stats.output_size_in_bytes), temporary=int(stats.temp_size_in_bytes),
                aliases=int(stats.alias_size_in_bytes))
            self.aot_quotes.append(quote)
        bank_bytes = 16*np.prod(self.shape)/z0.mesh.size
        if self.aot_quotes[-1]['aliases'] < bank_bytes:
            raise MemoryError('streamed current store must alias the entire final half-bank')
        from common.gpu_utils import device_budget_bytes, record_stage_price
        if not np.isfinite(resident_bytes) or resident_bytes < 0:
            raise ValueError('streamed current external resident bound must be finite and nonnegative')
        scratch = max(q['arguments'] + q['outputs'] + q['temporary'] - q['aliases']
            - (bank_bytes if q['owner'] == 'donated_final_bank_store' else 0.)
            for q in self.aot_quotes)
        self.memory_receipt = dict(final_half_banks=3*self._kinds*bank_bytes,
            external_resident_bytes=float(resident_bytes),
            exact_solve_store_AOT_scratch=scratch,
            planned_bytes_per_rank=3*self._kinds*bank_bytes+float(resident_bytes)+scratch,
            budget_bytes_per_rank=float(device_budget_bytes()))
        if self.memory_receipt['planned_bytes_per_rank'] > self.memory_receipt['budget_bytes_per_rank']:
            raise MemoryError('streamed current final banks and solve/store AOT exceed live budget')
        record_stage_price('isdf.current_coefficient_banks',self.memory_receipt['planned_bytes_per_rank'])
        self._banks = [[jnp.zeros(self.shape, jnp.complex128, device=sharding)
                        for _ in range(self._kinds)] for _ in range(3)]
        self._coverage = np.zeros((3, self._kinds, nr // rp), dtype=bool)

    def _authenticate(self):
        if self._banks is None or tuple(_fit_binding(z) for z in self._zetas) != self._bindings:
            raise ValueError('current coefficient stream is consumed or its fit binding changed')
        if any(len(_factor_leaves(z)) != len(saved)
               or any(a is not b for a, b in zip(_factor_leaves(z), saved))
               for z, saved in zip(self._zetas, self._factors)):
            raise ValueError('current coefficient stream requires the actual unchanged OWN factor leaves')

    def append(self, channel, kind, packet, first):
        import jax.numpy as jnp
        from isdf.zeta_mubatch import _require_q_owned
        self._authenticate()
        c, k, r0 = int(channel), int(kind), int(first)
        rp = self.radial_shape[-1]
        if (c != channel or k != kind or r0 != first or c not in (0, 1, 2)
                or k < 0 or k >= self._kinds or r0 < 0 or r0 >= self.radial_shape[2] or r0 % rp
                or self._coverage[c, k, r0 // rp]):
            raise ValueError('current packet channel/kind/radial coverage is invalid or repeated')
        if packet.ndim != 3 or packet.shape[:2] != self.shape[:2] or packet.shape[2] != self.packet_width:
            raise ValueError('compressed current packet shape disagrees with the radial map')
        _require_q_owned(packet, self._zetas[0].mesh, packet.shape,
                         name='consumed compressed current RHS packet')
        solved = self._solves[c](self._zetas[c].factor, packet)
        self._banks[c][k] = self._store(self._banks[c][k], solved, jnp.int32(r0))
        self._banks[c][k].block_until_ready()
        self._coverage[c, k, r0 // rp] = True

    def _complete(self):
        return self._banks is not None and bool(np.all(self._coverage))

    def finish(self):
        return CurrentCoefficientBanks(self)

"""Deferred atomic-current packets after the charge fit has been consumed.

The endpoint sampler, angular compressor, ordered normal source and current
factor remain the existing owners. This module changes their lifetime:
charge closes, OWN3 fits finish, then radial packets solve into six final
half-banks. It supplies no physical current or compact-target admission.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DeferredCurrentRequest:
    """Internal execution request from the charge-first public fitting owner."""
    request: dict


def physical_current_writer_workspace(*, q_logical, q_carrier, mu_packed,
                                      mu_canonical, g_tile, shards):
    """Physical output plus canonical-axis transport, priced before stores.

    Three emitted current tiles coexist, but the canonical writer transports
    only one channel at a time. The same shape price is used by planning and
    by the live writer; it excludes the current contraction's other leaves.
    """
    from runtime.padding import padded_axis
    emitted = 16 * q_carrier * mu_packed * g_tile / shards
    split = padded_axis(max(q_logical, g_tile), shards,
                        name='current writer transport split').carrier
    canonical = 16 * split * min(q_logical, g_tile) * max(mu_packed, mu_canonical) / shards
    return dict(emitted_tile=emitted, canonical_transport_tile=canonical,
                device_extra=3 * emitted + 5 * canonical, host_pending=canonical)


def live_device_bytes(*objects):
    """Count unique addressable device leaves, excluding donated buffers."""
    import jax
    seen = set()
    def count(value):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        if isinstance(value, jax.Array):
            if value.is_deleted():
                return 0
            return sum(shard.data.size*shard.data.dtype.itemsize
                       for shard in value.addressable_shards)
        if isinstance(value, dict):
            return sum(count(v) for v in value.values())
        if isinstance(value, (tuple, list)):
            return sum(count(v) for v in value)
        return 0
    return sum(count(value) for value in objects)


def _shared_current_packet_plan(mesh, vertex_terms, packet):
    """Reuse matching first/second channel primitives with bounded packets.

    Other bases and the compensated two-output route keep their ordinary
    scalar execution. Only program metadata is retained, never a point RHS.
    """
    import jax
    import jax.numpy as jnp
    from isdf.local_rhs import _projected_rhs_add
    from runtime.aot_memory import compiled_new_bytes
    if len(vertex_terms) != 3:
        return None
    first, second = (tuple(terms) for terms in vertex_terms[:2])
    pairs = tuple((int(i), int(j)) for _, i, j in first)
    if (len(pairs) < 2 or len(set(pairs)) != len(pairs)
            or pairs != tuple((int(i), int(j)) for _, i, j in second)):
        return None
    weights = tuple((complex(left[0]), complex(right[0]))
                    for left, right in zip(first, second))
    initialize = jax.jit(lambda: tuple(jnp.zeros(packet.shape, packet.dtype)
                                      for _ in range(2)),
                         out_shardings=(packet.sharding,) * 2)
    adds = tuple(_projected_rhs_add(mesh, weight) for weight in weights)
    quotes = [dict(owner='shared_current_packet_initialize',
        resident_increment=int(compiled_new_bytes(initialize.lower().compile())))]
    for weight, fn in dict(zip(weights, adds)).items():
        quotes.append(dict(owner='shared_current_packet_add',
            weights=[[w.real, w.imag] for w in weight],
            resident_increment=int(compiled_new_bytes(
                fn.lower((packet, packet), packet).compile()))))
    packet_bytes = packet.size * packet.dtype.itemsize / mesh.size
    increment = max(quotes[0]['resident_increment'],
        3 * packet_bytes + max(q['resident_increment'] for q in quotes[1:]))
    return dict(pairs=pairs, initialize=initialize, adds=adds,
        shape=packet.shape, mesh=mesh,
        receipt=dict(channels=[1, 2], unique_vertex_pairs=[list(p) for p in pairs],
            compact_accumulators_bytes=2 * packet_bytes,
            additional_workspace_bytes=increment, actual_AOT=quotes,
            policy='Completion before complex weights; projected primitives only; distinct OWN factors'))


def _shared_current_rhs_packets(plan, project, **operands):
    """Project each Cartesian delta once, then fan out its channel weights."""
    import jax
    from isdf.local_rhs import _iter_local_density_rhs_primitives
    from isdf.zeta_mubatch import _require_q_owned
    accumulators = plan['initialize']()
    jax.block_until_ready(accumulators)
    primitives = _iter_local_density_rhs_primitives(
        **operands, vertex_pairs=plan['pairs'], return_smooth=False)
    try:
        index = 0
        for _, value, _ in primitives:
            packet = project(value)
            _require_q_owned(packet, plan['mesh'], plan['shape'],
                             name='shared completed current primitive')
            packet.block_until_ready()
            accumulators = plan['adds'][index](accumulators, packet)
            jax.block_until_ready(accumulators)
            del packet, value
            index += 1
    finally:
        # In particular, drop a yielded raw RHS if projection/solve refused.
        primitives.close()
    return accumulators


class DeferredCurrentDensity:
    """Bounded source/C replay; each raw current packet is consumed once."""
    def __init__(self, current, charge_state, *, source, coefficients, parameters):
        self.current, self.charge_state = current, charge_state
        self.source, self.coefficients = source, tuple(coefficients)
        self.parameters = parameters
        self._source_id = id(source)
        self._coefficient_ids = tuple(id(v) for v in coefficients)
        self._identity = current['identity']
        self._angular_identity = current['angular_support_identity']
        self._qrows = np.array(current['q_full_indices'], copy=True)
        self._charge_zeta = None
        self._consumed = False

    def bind_charge_zeta(self, zeta):
        if self._charge_zeta is not None or zeta.mesh is not self.current['plan'].mesh_xy:
            raise ValueError('deferred current charge lifetime must bind its actual charge zeta once')
        self._charge_zeta = zeta

    def after_charge_close(self):
        """Price live current fit operands only after the actual charge closes."""
        self._authenticate()
        self.current['fit_resident_extra_bytes_per_rank'] = live_device_bytes(
            self.current,self.charge_state,self.source,self.coefficients)
        self.current['phase_prices']['charge_closed_before_current_fit'] = True

    def _authenticate(self):
        current = self.current
        if (self._consumed or id(self.source) != self._source_id
                or tuple(id(v) for v in self.coefficients) != self._coefficient_ids
                or current['identity'] != self._identity
                or current['angular_support_identity'] != self._angular_identity
                or not np.array_equal(current['q_full_indices'], self._qrows)):
            raise ValueError('deferred current source/C/field/angular/q binding changed or was consumed')
        if any(key in self.charge_state for key in ('rhs', 'smooth_rhs', 'monopole_rhs')):
            raise ValueError('deferred current replay requires charge RHS consumption before final banks')
        charge = self._charge_zeta
        if charge is None:
            raise ValueError('deferred current replay requires its actual charge zeta lifetime binding')
        if any(getattr(charge, key) is not None
                for key in ('L_q', 'lu_piv', 'local_augmentation', 'shell', 'q0_response_rhs')):
            raise ValueError('deferred current replay requires the actual charge zeta to be closed')

    def replay(self, channels, *, smooth_support=None):
        import jax
        import jax.numpy as jnp
        from jax.sharding import NamedSharding, PartitionSpec as P
        from common import timing
        from common.gpu_utils import device_budget_bytes, warn_over_budget, record_stage_price
        from common.staged_reshard import face_to_batch_reshard
        from psp.augmented_samples import make_sample_correction
        from isdf.local_rhs import local_density_rhs
        from isdf.current_coefficients import CurrentCoefficientStream
        from gw import isdf_augmentation as stage
        self._authenticate()
        if smooth_support is not None:
            from isdf.current_support import SmoothCurrentSupport
            if not isinstance(smooth_support, SmoothCurrentSupport):
                raise ValueError('same-Z replay requires the actual support witness')
            smooth_support.authenticate(channels)
        kinds = 2 if smooth_support is None else 1
        s, p = self.current, self.parameters
        mesh, plan = s['plan'].mesh_xy, s['plan']
        if (len(channels) != 3 or int(plan.n_centroid_packed) != int(channels[0].store.mu_pad)
                or int(s['qpad']) != int(channels[0].store.Q_pad)
                or len(s['q_full_indices']) != int(channels[0].store.Q)):
            raise ValueError('deferred current plan q/centroid extent differs from the actual OWN stores')
        na, nh, nr, rp = len(p['centers']), len(s['lm']), len(s['radius']), p['radial_packet']
        pc, bc, ng = p['parent_chunk'], p['band_chunk'], self.source.shape[-1]
        npar, nb = self.source.shape[:2]
        relative = s['radius'][:rp,None,None]*p['directions'][None]
        raw_points = (p['centers'][:,None,None] + relative[None]@np.linalg.inv(p['lattice'])).reshape(-1,3)
        right_plan = stage._packet_plan(plan,raw_points,None,
            fractional_lookup_budget=p['atomic_lookup'])
        npoint = int(right_plan.n_centroid_packed)
        phase = stage._point_phase_kernel(mesh)
        sample = stage._point_samples_kernel(mesh,pc,bc,npoint,p['g_block'],s['fft_points'])
        kernels = stage._tile_kernels(mesh,pc,bc)
        compressor,nf,nfp = stage._compress_rhs_kernel(mesh,right_plan,na,nh,rp,p['angular_projection'])
        ids = s['q_full_indices'] if s['indexed_q'] is None else np.arange(len(s['q_full_indices']))
        select,_ = stage._rhs_storage_kernels(mesh,ids,s['qpad'],na,nh,nr,rp,nf)
        to_q = face_to_batch_reshard(mesh)
        shared = None
        if kinds == 1:
            packet = jax.ShapeDtypeStruct((s['qpad'], plan.n_centroid_packed, nfp),
                jnp.complex128, sharding=NamedSharding(mesh, P(('x', 'y'), None, None)))
            shared = _shared_current_packet_plan(mesh, s['vertex_terms'], packet)
        gs,fs,qs = P(None,None,None,('x','y')), P(None,'x',None,'y'), P(('x','y'),None,None)
        shape = lambda dims,dtype,spec:jax.ShapeDtypeStruct(dims,dtype,
            sharding=NamedSharding(mesh,spec))
        K = shape((pc,ng,3),jnp.float64,P(None,('x','y'),None))
        cart = shape((npoint,3),jnp.float64,P())
        table = shape((pc,ng,npoint),jnp.complex128,P(None,('x','y'),None))
        source = shape((pc,bc,4,ng),jnp.complex128,gs)
        active = shape((npoint,),jnp.float64,P())
        quotes = []
        for label,fn,args in (('phase_packet',phase,(K,cart)),
                             ('point_sample_packet',sample,(source,table,active))):
            m = fn.lower(*args).compile().memory_analysis()
            quotes.append(dict(owner=label,arguments=int(m.argument_size_in_bytes),
                outputs=int(m.output_size_in_bytes),temporary=int(m.temp_size_in_bytes),
                aliases=int(m.alias_size_in_bytes)))
        final_banks = 3*kinds*16*channels[0].store.Q_pad*channels[0].store.mu_pad*na*nh*nr/mesh.size
        external = live_device_bytes(s,self.charge_state,self.source,self.coefficients,
                                     tuple(z.factor for z in channels))
        points = 4*16*npar*nb*4*npoint/mesh.size
        angular = s['angular_workspace']['total']
        # Full current normal-source and angular workspaces retain their
        # existing conservative shape bounds. Exact sampler AOT additionally
        # accounts for its compiled intermediates; no multiplier is relaxed.
        packet_work = (points+s['workspace']['total']+angular
            +max(q['arguments']+q['outputs']+q['temporary']-q['aliases'] for q in quotes))
        if shared is not None:
            packet_work += shared['receipt']['additional_workspace_bytes']
        price = final_banks+external+packet_work
        budget = float(device_budget_bytes())
        s['replay_preflight'] = dict(final_banks=final_banks,actual_live_device_leaves=external,
            packet_shape_bound=packet_work,actual_sampler_AOT=quotes,planned_bytes_per_rank=price,
            budget_bytes_per_rank=budget,scope='Sampler AOT plus conservative current RHS bounds; not pair/onsite AOT admission')
        if shared is not None:
            s['replay_preflight']['shared_current_primitives'] = shared['receipt']
        if price > budget:
            warn_over_budget('isdf.current_packet_replay',price,budget)
            raise MemoryError('deferred current packet replay exceeds its live device budget')
        record_stage_price('isdf.current_packet_replay',price)
        stream = CurrentCoefficientStream(channels,atoms=na,harmonics=nh,
            radial_points=nr,radial_packet=rp,identity=s['identity'],
            angular_identity=s['angular_support_identity'],resident_bytes=external+packet_work,
            packet_features=nfp,operand_kind='delta_and_ps' if kinds==2 else 'delta',
            smooth_support=smooth_support)
        s['replay_preflight']['actual_solve_store_AOT'] = stream.aot_quotes
        s['replay_preflight']['solve_store_memory_receipt'] = stream.memory_receipt
        # Only final output banks exist, with no charge cloud or full raw
        # current/paired coefficient cloud. Solves use these exact OWN leaves.
        correct = make_sample_correction(mesh,face_sharding=NamedSharding(mesh,fs))
        with timing.section('augmentation.current_atomic_packet_replay'):
            for r0 in range(0,nr,rp):
                relative = s['radius'][r0:r0+rp,None,None]*p['directions'][None]
                raw_points = (p['centers'][:,None,None]+relative[None]@np.linalg.inv(p['lattice'])).reshape(-1,3)
                right_plan = stage._packet_plan(plan,raw_points,right_plan,
                    fractional_lookup_budget=p['atomic_lookup'])
                positions = right_plan.layout.axis.pack_host(raw_points,axis=0,fill_value=0.)
                live = np.asarray(right_plan.layout.axis.active_mask)
                ps = jnp.zeros((npar,nb,4,npoint),jnp.complex128,device=NamedSharding(mesh,fs))
                ae = jnp.zeros((npar,nb,4,npoint),jnp.complex128,device=NamedSharding(mesh,fs))
                cart = stage._put(positions@p['lattice'],mesh,P())
                active = stage._put(live.astype(float),mesh,P())
                for p0 in range(0,npar,pc):
                    K = stage._put(p['wavevectors'][p0:p0+pc],mesh,P(None,('x','y'),None))
                    phases = phase(K,cart)
                    for b0 in range(0,nb,bc):
                        tile = sample(kernels['source'](self.source,jnp.int32(p0),jnp.int32(b0)),phases,active)
                        ps = kernels['update_face'](ps,tile,jnp.int32(p0),jnp.int32(b0))
                        ae = kernels['update_face'](ae,tile,jnp.int32(p0),jnp.int32(b0))
                    ae.block_until_ready()
                for atom,(delta,image_phase) in enumerate(stage._sample_geometry(
                        positions,live,p['centers'],p['lattice'],p['caches'],p['atom_types'],
                        p['kfrac'],s['support_radius'],p['scale'])):
                    delta = stage._put(delta,mesh,P(None,None,'y'))
                    for p0 in range(0,npar,pc):
                        image = stage._put(image_phase[p0:p0+pc],mesh,P(None,'y'))
                        for b0 in range(0,nb,bc):
                            C = kernels['read_coeff'](self.coefficients[atom],jnp.int32(p0),jnp.int32(b0))
                            tile = kernels['face'](ae,jnp.int32(p0),jnp.int32(b0))
                            ae = kernels['update_face'](ae,correct(tile,C,delta,image),jnp.int32(p0),jnp.int32(b0))
                ae_faces,ps_faces = (ae,stage._face_complement(ae,mesh)),(ps,stage._face_complement(ps,mesh))
                common = dict(centroid_faces=s['parent_faces'],
                    atom_ae_faces=ae_faces,atom_ps_faces=ps_faces,left_plan=plan,right_plan=right_plan,
                    weight_l=p['weight_l'],weight_r=p['weight_r'],kgrid=s['kgrid'],mesh_xy=mesh,
                    q_neg_idx=s['qneg'],q_indices=s['indexed_q'])
                if shared is not None:
                    packets = list(_shared_current_rhs_packets(shared,
                        lambda value: to_q(compressor(select(value))), **common))
                    for channel in range(2):
                        packet, packets[channel] = packets[channel], None
                        stream.append(channel, 0, packet, r0)
                        del packet
                    del packets
                for channel,terms in enumerate(s['vertex_terms']):
                    if shared is not None and channel < 2:
                        continue
                    values = local_density_rhs(**common,vertex_terms=terms,return_smooth=kinds==2)
                    values = values if kinds==2 else (values,)
                    for kind,value in enumerate(values):
                        packet = to_q(compressor(select(value)))
                        stream.append(channel,kind,packet,r0)
                        del packet
                    del values,value
                del common,ae_faces,ps_faces,ae,ps,phases,tile,C,image
        result = stream.finish()
        self._consumed = True
        self.source = self.coefficients = self.parameters = self.charge_state = self._charge_zeta = None
        s['replay_preflight']['raw_clouds_retained'] = 0
        s['replay_preflight']['paired_clouds_retained'] = 0
        return result

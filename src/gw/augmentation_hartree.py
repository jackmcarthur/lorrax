"""Bind and serve the reconstructed fixed-source Hartree band operator.

The numerical receiving contraction runs before the fit donates its orbital
store. Only its small FILE-wedge band matrix survives. File format validation
belongs to file_io; source occupations and transport retain their existing
owners. This module never reads wavefunction coefficients or rebuilds A.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


def _identity(binding):
    return hashlib.sha256(json.dumps(binding, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _current_operator_binding(provenance, current_operator_record):
    """Bind the current owner's live generation to the same occupied source.

    ``current_operator_record`` comes only from the canonical V-generation
    reader or that producer's identical fresh record. It is not an admission
    flag: absent, torn or changed contracts refuse. Its physical owner must
    authenticate the source frame and current fit before publishing it.
    """
    if (not isinstance(current_operator_record, dict)
            or set(current_operator_record) != {
                'operator_contract', 'operator_identity', 'generation_identity'}
            or not isinstance(current_operator_record['operator_contract'], dict)
            or not current_operator_record['operator_contract']
            or current_operator_record['operator_identity'] !=
                _identity(current_operator_record['operator_contract'])
            or not isinstance(current_operator_record['generation_identity'], str)
            or len(current_operator_record['generation_identity']) != 64
            or any(c not in '0123456789abcdef' for c in
                   current_operator_record['generation_identity'])):
        raise ValueError('GATE resident_current_operator: requires the canonical physical current generation')
    source = provenance['source_binding']
    source_keys = ('augmentation_identity', 'physical_bands', 'source_frame_policy',
        'physical_frame_shape', 'physical_frame_band_domain', 'physical_frame_convention',
        'full150_frame_sha256', 'fft_grid', 'cell_volume', 'wfn_fingerprint_scheme',
        'wfn_fingerprint', 'wfn_sha256')
    if any(key not in source for key in source_keys):
        raise ValueError('GATE resident_current_source: paired current requires a complete physical source/frame binding')
    expected_source = {key:source[key] for key in source_keys}
    expected_source.update(FILE_parent_rows=provenance['parent_full_rows'],
                           FILE_parent_k_fractional=provenance['parent_k_frac'])
    contract = current_operator_record['operator_contract']
    if (contract.get('schema') != 'lorrax.current.physical_Slab_mixed_operator.v1'
            or contract.get('operator') != 'mixed_reciprocal_point_Slab_plus_bulk_local_infinite_remainder'
            or int(source.get('sys_dim', 3)) != 2):
        raise ValueError('GATE resident_current_operator: requires the physical mixed Slab current owner')
    if contract.get('endpoint_source_binding') != expected_source:
        raise ValueError('GATE resident_current_source: current endpoints differ from the charge WFN/full-source frame/FILE chart')
    geometry = contract.get('current_geometry_binding')
    if (not isinstance(geometry, dict) or not {
            'logical_mu', 'centroid_hash', 'q_full_indices', 'q_fractional', 'kgrid'} <= set(geometry)):
        raise ValueError('GATE resident_current_operator: current centroid/q geometry contract is missing')
    return dict(schema='lorrax.resident_current_hartree_operator.v1',
        **{key: provenance[key] for key in (
            'source_identity', 'band_range', 'parent_full_rows', 'parent_k_frac',
            'units', 'k_domain', 'trs_rule')},
        physical_current_operator_contract=current_operator_record['operator_contract'],
        physical_current_operator_identity=current_operator_record['operator_identity'],
        current_generation_identity=current_operator_record['generation_identity'],
        variational_convention='hermitian_density_current_energy_derivative_v1')


def resident_hartree_for_io(record):
    """Transport named direct fields on the existing versioned band axes."""
    if 'parent_kij_ry' in record:
        return dict(hartree_parent_kij_ry=record['parent_kij_ry'],
                    hartree_provenance=record['provenance'])
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    charge, transverse = (record[name+'_parent_kij_ry'] for name in ('charge', 'transverse'))
    if (charge.shape != transverse.shape or charge.sharding != transverse.sharding
            or charge.ndim != 3 or charge.dtype != jnp.complex128
            or transverse.dtype != jnp.complex128):
        raise ValueError('Paired resident direct fields require identical native band carriers')
    spec = NamedSharding(charge.sharding.mesh, P(None, 'x', 'y', None))
    payload = jax.jit(lambda a, b: jnp.stack((a, b), axis=-1),
                      out_shardings=spec)(charge, transverse)
    return dict(hartree_parent_kij_ry=payload, hartree_provenance=record['provenance'])


def _operator_binding(*, wfn, sym, artifact, source_identity, band_range, sys_dim=3):
    from . import augmentation_hartree_receiving as receiving
    from . import isdf_augmentation as stage
    from isdf.atomic_hartree import charge_hartree_operator_contract

    array_hash = lambda value: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
    directions, weights, lm, Y, _ = stage._orbit_angular_quadrature(
        artifact['angular'], np.asarray(sym.R_cart))
    radius, weights_dr, support_radius = stage._radial_grid(artifact['radial'])
    radial = dict(radius_sha256=array_hash(radius),
        weights_dr_sha256=array_hash(weights_dr), support_radius=support_radius)
    radial.update({key: artifact['radial'][key] for key in
        ('interpolation_degree', 'quadrature_order', 'fourier_points') if key in artifact['radial']})

    return dict(schema='lorrax.resident_charge_hartree_operator.v1',
        source_identity=source_identity, augmentation_identity=artifact['identity'],
        **charge_hartree_operator_contract(wfn, sys_dim=sys_dim),
        band_range=list(map(int, band_range)),
        parent_full_rows=np.asarray(sym.kirr_fullids, int).tolist(),
        parent_k_frac=np.asarray(wfn.kvecs(k=sym.parent_k_domain), float).tolist(),
        units='Ry', k_domain='file_wedge', trs_rule='conj',
        fft_grid=list(map(int, wfn.fft_grid)), cell_volume=float(wfn.cell_volume),
        lattice_cart=(float(wfn.alat)*np.asarray(wfn.avec)).tolist(),
        atom_crys=np.asarray(wfn.atom_crys).tolist(),
        atom_types=np.asarray(wfn.atom_types, int).tolist(),
        radial=radial, angular=dict(directions_sha256=array_hash(directions),
            weights_sha256=array_hash(weights), lm_sha256=array_hash(lm), Y_sha256=array_hash(Y)),
        owner_sources_sha256=receiving.numerical_owner_sources_sha256())


def require_resident_hartree_source(provenance, *, wfn, sym, plan, artifact,
                                   wfn_fingerprint_binding, band_range,
                                   current_operator_record=None):
    """Authenticate physical primitives before reading or serving a matrix.

    A restart authenticates the producer's full-WFN factor chain; it does not
    claim to recompute that factor. A different physical density requires a
    fresh source, even if the requested receiving interval is unchanged.
    """
    from common.parallel_transport import fingerprint_from_binding, WFN_FINGERPRINT_SCHEME
    from file_io.tagged_arrays import normalize_resident_hartree_provenance
    from .isdf_augmentation import (_hartree_source_request,
        _physical_full_wfn_frame_binding)

    record = normalize_resident_hartree_provenance(
        provenance, persisted='payload_sha256' in provenance)
    paired = record['schema'] == 'lorrax.resident_direct_hartree.v2'
    source = record['source_binding']
    from isdf.atomic_hartree import charge_hartree_operator_contract
    dimension = source.get('sys_dim', 3)
    contract = charge_hartree_operator_contract(wfn, sys_dim=dimension)
    if (('kernel' in contract and source.get('hartree_kernel') != contract['kernel'])
            or ('kernel' not in contract and 'hartree_kernel' in source)):
        raise ValueError('GATE resident_hartree_source: physical Coulomb kernel changed')
    raw = source.get('prepared_raw_binding', {})
    prepared_raw = (artifact.get('raw_parent_binding') or
        (artifact.get('raw_parent_moments') or {}).get('metadata', {}).get('binding'))
    frame = source.get('full150_frame_sha256')
    count = int(wfn.nbands)
    lo, hi = map(int, band_range)
    stored_lo, stored_hi = record['band_range']
    rows = np.asarray(sym.kirr_fullids, int)
    coords = np.asarray(wfn.kvecs(k=sym.parent_k_domain), float)
    expected_fingerprint = fingerprint_from_binding(wfn_fingerprint_binding, wfn)
    compact = artifact.get('compact_target')
    compact_binding = (compact['binding'] if compact is not None
                       else artifact.get('compact_target_binding'))
    if compact_binding is None:
        source_policy_matches = (
            source.get('source_frame_policy') == 'same_actual_served_four_spinor_full_WFN_Lowdin'
            and prepared_raw is not None and raw == prepared_raw
            and raw.get('carrier') == 'normalized_rkb'
            and raw.get('wfn_fingerprint_scheme') == WFN_FINGERPRINT_SCHEME
            and raw.get('wfn_fingerprint') == expected_fingerprint
            and raw.get('band_range') == [0, count]
            and np.array_equal(raw.get('k_parent_frac'), coords))
    else:
        from psp.augmentation_cache import paired_field_policy_contract
        contract = paired_field_policy_contract(compact_binding.get('field_policy'))
        saved_frame = (artifact.get('compact_frame_sha256') if compact is None else
            _physical_full_wfn_frame_binding(compact['frame']['arrays']['inverse_sqrt'], count)
            ['full150_frame_sha256'])
        public_identity = artifact.get('public_source_identity', {})
        source_policy_matches = (
            compact_binding.get('model') == contract['frame_model']
            and compact_binding.get('carrier') in ('normalized_rkb', 'pauli2embed4')
            and compact_binding.get('physical_bands') == count
            and compact_binding.get('complete_all_FILE_parents') is True
            and compact_binding.get('normalization') == 'one compact target A; no represented-field renormalization'
            and source.get('source_frame_policy') == contract['source_frame_policy']
            and compact_binding.get('source_frame_policy') == source['source_frame_policy']
            and source.get('compact_target_binding') == compact_binding
            and raw == compact_binding and frame == saved_frame
            and public_identity.get('wfn_sha256') == compact_binding.get('wfn_sha256')
            and source.get('wfn_sha256') == compact_binding.get('wfn_sha256')
            and public_identity.get('wfn_fingerprint_scheme') == WFN_FINGERPRINT_SCHEME
            and public_identity.get('wfn_fingerprint') == expected_fingerprint
            and source.get('wfn_fingerprint_scheme') == WFN_FINGERPRINT_SCHEME
            and source.get('wfn_fingerprint') == expected_fingerprint)
    if (plan.sym is not sym or plan.nspinor != 4 or plan.n_parent != int(sym.nk_red)
            or not np.array_equal(plan.parent_full_rows, rows)
            or not np.array_equal(plan.k_parent_frac, coords)
            or not np.array_equal(record['parent_full_rows'], rows)
            or not np.array_equal(record['parent_k_frac'], coords)
            or not stored_lo <= lo < hi <= stored_hi
            or source.get('augmentation_identity') != artifact['identity']
            or source.get('physical_bands') != count
            or source.get('physical_frame_shape') != [plan.n_parent, count, count]
            or source.get('physical_frame_band_domain') != [0, count]
            or source.get('physical_frame_convention') != 'same full-WFN inverse_sqrt columns on reciprocal and atomic band rows'
            or not isinstance(frame, str) or len(frame) != 64
            or any(c not in '0123456789abcdef' for c in frame)
            or not source_policy_matches
            or source.get('fft_grid') != list(map(int, wfn.fft_grid))
            or source.get('cell_volume') != float(wfn.cell_volume)):
        raise ValueError('GATE resident_hartree_source: reconstructed source, frame or FILE receiving domain changed')
    physical = _hartree_source_request(
        {'occupations': None, 'full_kweights': None, 'spin_degeneracy': 1.},
        wfn=wfn, plan=plan, public_range=source['public_band_range'])
    if any(source.get(name) != physical[name] for name in
           ('occupations_sha256', 'full_kweights_sha256', 'spin_degeneracy')):
        raise ValueError('GATE resident_hartree_source: physical occupations or quadrature changed')
    expected = _operator_binding(wfn=wfn, sym=sym, artifact=artifact,
        source_identity=record['source_identity'], band_range=record['band_range'],
        sys_dim=dimension)
    charge_member = (record['operator_binding']['components']['charge'] if paired else record)
    if (charge_member['operator_binding'] != expected
            or charge_member['operator_identity'] != _identity(expected)):
        raise ValueError('GATE resident_hartree_operator: geometry, field recipe or numerical owner changed')
    if paired:
        expected_current = _current_operator_binding(record, current_operator_record)
        actual = record['operator_binding']['components']['transverse']
        if (actual['operator_binding'] != expected_current
                or actual['operator_identity'] != _identity(expected_current)):
            raise ValueError('GATE resident_current_operator: current fit, source/frame or generation changed')
    return record


def bind_resident_hartree(record, *, wfn, sym, plan, artifact,
                         wfn_fingerprint_binding, band_range,
                         current_operator_record=None):
    """Bind the two-member IO carrier to this invocation's actual source."""
    paired_keys = {'charge_parent_kij_ry', 'transverse_parent_kij_ry', 'provenance'}
    if (not isinstance(record, dict)
            or set(record) not in ({'parent_kij_ry', 'provenance'}, paired_keys)):
        raise ValueError('resident Hartree requires one paired native matrix and provenance')
    provenance = require_resident_hartree_source(record['provenance'],
        wfn=wfn, sym=sym, plan=plan, artifact=artifact,
        wfn_fingerprint_binding=wfn_fingerprint_binding, band_range=band_range,
        current_operator_record=current_operator_record)
    if (set(record) == paired_keys) != (provenance['schema'] == 'lorrax.resident_direct_hartree.v2'):
        raise ValueError('Resident direct field names disagree with their versioned schema')
    thin_artifact = {key:artifact[key] for key in ('identity', 'radial', 'angular')}
    compact = artifact.get('compact_target')
    if compact is None:
        thin_artifact['raw_parent_binding'] = (artifact.get('raw_parent_binding') or
            artifact['raw_parent_moments']['metadata']['binding'])
    else:
        from .isdf_augmentation import _physical_full_wfn_frame_binding
        thin_artifact.update(compact_target_binding=compact['binding'],
            compact_frame_sha256=_physical_full_wfn_frame_binding(
                compact['frame']['arrays']['inverse_sqrt'], int(wfn.nbands))['full150_frame_sha256'],
            public_source_identity=artifact['public_source_identity'])
    result = {key: value for key, value in record.items() if key != 'provenance'}
    context = dict(wfn_fingerprint_binding=wfn_fingerprint_binding,
                   artifact=thin_artifact, plan=plan)
    if provenance['schema'] == 'lorrax.resident_direct_hartree.v2':
        context['current_operator_record'] = current_operator_record
    return dict(result, provenance=provenance, source_context=context)


def pair_resident_hartree(charge_record, transverse, current_operator_record):
    """Pair a completed current derivative with the independently bound charge.

    The caller supplies the canonical current producer's live operator record
    and its source-bound sampled derivative. This never constructs a source,
    changes charge bytes, or supplies a synthetic current for a missing owner.
    Final bind/serve re-authenticate both actual operators.
    """
    from file_io.tagged_arrays import normalize_resident_hartree_provenance
    if (set(charge_record) != {'parent_kij_ry', 'provenance', 'source_context'}
            or charge_record['provenance']['schema'] != 'lorrax.resident_charge_hartree.v1'):
        raise ValueError('Paired direct fields require an independently bound native charge field')
    charge = charge_record['provenance']
    current = _current_operator_binding(charge, current_operator_record)
    operator = dict(schema='lorrax.resident_paired_direct_operator.v1',
        source_identity=charge['source_identity'], components={
            'charge': {key:charge[key] for key in ('operator_identity', 'operator_binding')},
            'transverse': dict(operator_binding=current, operator_identity=_identity(current))})
    provenance = {key:value for key,value in charge.items() if key != 'payload_sha256'}
    provenance.update(schema='lorrax.resident_direct_hartree.v2',
                      operator_binding=operator, operator_identity=_identity(operator))
    provenance = normalize_resident_hartree_provenance(provenance)
    context = dict(charge_record['source_context'], current_operator_record=current_operator_record)
    return dict(charge_parent_kij_ry=charge_record['parent_kij_ry'],
        transverse_parent_kij_ry=transverse, provenance=provenance, source_context=context)


def prepare_resident_hartree(*, wfn, sym, mesh, plan, state, artifact,
                            wfn_fingerprint_binding, band_range):
    """Finish the receiving contraction while the shared orbital store lives."""
    from file_io.tagged_arrays import normalize_resident_hartree_provenance
    from .augmentation_hartree_receiving import build_resident_receiving_J

    matrix, diagnostics = build_resident_receiving_J(
        wfn=wfn, mesh=mesh, state=state, artifact=artifact,
        receiving_range=band_range, band_tile=int(np.gcd(8, band_range[1]-band_range[0])))
    # The numerical owner serves the charge matrix at its distributed boundary.
    source = state['hartree_source']
    binding = _operator_binding(wfn=wfn, sym=sym, artifact=artifact,
        source_identity=source['source_identity'], band_range=band_range,
        sys_dim=state.get('sys_dim', 3))
    provenance = normalize_resident_hartree_provenance(dict(
        schema='lorrax.resident_charge_hartree.v1',
        source_identity=source['source_identity'], source_binding=source['source_binding'],
        operator_identity=_identity(binding), operator_binding=binding,
        band_range=list(map(int, band_range)),
        parent_full_rows=binding['parent_full_rows'], parent_k_frac=binding['parent_k_frac'],
        units='Ry', k_domain='file_wedge', trs_rule='conj'))
    record = bind_resident_hartree(dict(parent_kij_ry=matrix, provenance=provenance),
        wfn=wfn, sym=sym, plan=plan, artifact=artifact,
        wfn_fingerprint_binding=wfn_fingerprint_binding, band_range=band_range)
    return record, diagnostics


def serve_resident_hartree(record, *, config, wfn, sym, mesh, band_range,
                          occupation_state=None):
    """Select the fixed-source time-even field before the existing QP rotation."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from runtime.padding import pad_square
    from .ppm_sigma import sigma_band_axis
    from .gw_config import (BispinorGWMode, QPSolver, coerce_bispinor_gw_mode,
                            uses_transverse_interaction)
    from symmetry_maps import unfold_file_wedge_band_operator

    paired = (isinstance(record, dict) and isinstance(record.get('provenance'), dict)
              and record['provenance'].get('schema') == 'lorrax.resident_direct_hartree.v2')
    keys = ({'charge_parent_kij_ry', 'transverse_parent_kij_ry', 'provenance', 'source_context'}
            if paired else {'parent_kij_ry', 'provenance', 'source_context'})
    requested_current = uses_transverse_interaction(config)
    if (not isinstance(record, dict) or set(record) != keys
            or not config.bispinor or paired != requested_current
            or (not paired and coerce_bispinor_gw_mode(config.bispinor_gw)
                is not BispinorGWMode.COULOMB_ONLY)
            or config.qp_solver is not QPSolver.ONE_SHOT_DFT):
        raise ValueError('GATE resident_hartree_fixed_source: requires fixed-source Coulomb-only charge or authenticated paired current fields')
    context = record['source_context']
    provenance = require_resident_hartree_source(record['provenance'],
        wfn=wfn, sym=sym, band_range=band_range, **context)
    if int(getattr(config, 'sys_dim', 3)) != int(provenance['source_binding'].get('sys_dim', 3)):
        raise ValueError('GATE resident_hartree_fixed_source: requested Coulomb kernel differs from its bound source')
    matrices = ({name:record[name+'_parent_kij_ry'] for name in ('charge', 'transverse')}
                if paired else {'charge':record['parent_kij_ry']})
    lo, hi = map(int, band_range)
    start, stop = provenance['band_range']
    bands = stop-start
    sharding = NamedSharding(mesh, P(None, 'x', 'y'))
    for matrix in matrices.values():
        if (not isinstance(matrix, jax.Array) or matrix.dtype != jnp.complex128
                or matrix.ndim != 3
                or matrix.shape[0] != len(provenance['parent_full_rows'])
                or matrix.shape[1] < bands or matrix.shape[2] < bands
                or matrix.shape[1] != matrix.shape[2]
                or matrix.sharding != sharding):
            raise ValueError('GATE resident_hartree_payload: native band carrier differs from its logical source')
    if occupation_state is not None:
        from .isdf_augmentation import _hartree_source_request
        physical = _hartree_source_request(
            {'occupations': None, 'full_kweights': None, 'spin_degeneracy': 1.},
            wfn=wfn, plan=context['plan'], public_range=provenance['source_binding']['public_band_range'])
        occupations = np.asarray(occupation_state.f_kn)
        unfolded = physical['occupations'][np.asarray(context['plan'].irr_idx)]
        common = min(occupations.shape[1], unfolded.shape[1]) if occupations.ndim == 2 else 0
        if (occupations.ndim != 2 or occupations.shape[0] != unfolded.shape[0]
                or not np.array_equal(occupations[:, :common], unfolded[:, :common])
                or np.any(occupations[:, common:]) or np.any(unfolded[:, common:])):
            raise ValueError('GATE resident_hartree_fixed_source: one-shot occupations changed the reconstructed density')
    # Normalize ghosts without publishing an indivisible logical matrix.
    native_axis = sigma_band_axis(bands, mesh, ansatz='static')
    output_axis = sigma_band_axis(hi-lo, mesh, ansatz='static')
    def select(value):
        selected = value[:, lo-start:hi-start, lo-start:hi-start]
        return pad_square(selected, output_axis)
    fields = {}
    for name, matrix in matrices.items():
        native = jax.jit(lambda value: pad_square(value, native_axis),
                         out_shardings=sharding)(matrix)
        if not bool(jax.device_get(jnp.all(jnp.isfinite(native)))):
            raise ValueError('GATE resident_hartree_payload: nonfinite physical matrix')
        full = unfold_file_wedge_band_operator(sym, native, trs_rule='conj')
        fields[name] = jax.jit(select, out_shardings=sharding)(full)
    return fields if paired else fields['charge']


def place_resident_hartree_rotation(rotation, *, mesh, logical_bands, dtype):
    """Normalize its ghosts through the Sigma band owner before normal rotation."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from runtime.padding import pad_square
    from .ppm_sigma import sigma_band_axis
    from .sigma_dispatch import _place_band_rotation

    axis = sigma_band_axis(logical_bands, mesh, ansatz='static')
    if rotation.ndim != 3 or min(rotation.shape[-2:]) < axis.logical:
        raise ValueError('resident Hartree rotation cannot cover the physical receiving window')
    if isinstance(rotation, jax.Array):
        prepared = jax.jit(lambda value: pad_square(value, axis),
            out_shardings=NamedSharding(mesh, P(None, 'x', 'y')))(rotation)
    else:
        # A host U already belongs to the caller. Pad on that side, then let
        # the incumbent owner transfer process-local tiles; jnp.asarray(U)
        # here would replicate the complete band square on every GPU.
        source = np.asarray(rotation, dtype=dtype)
        prepared = np.zeros((source.shape[0], axis.carrier, axis.carrier), dtype=source.dtype)
        prepared[:, :axis.logical, :axis.logical] = source[:, :axis.logical, :axis.logical]
    return _place_band_rotation(prepared, mesh, dtype)

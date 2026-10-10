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
                                   wfn_fingerprint_binding, band_range):
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
        saved_frame = (artifact.get('compact_frame_sha256') if compact is None else
            _physical_full_wfn_frame_binding(compact['frame']['arrays']['inverse_sqrt'], count)
            ['full150_frame_sha256'])
        public_identity = artifact.get('public_source_identity', {})
        source_policy_matches = (
            compact_binding.get('model') == 'compact_native_pauli_common_frame_v1'
            and compact_binding.get('carrier') in ('normalized_rkb', 'pauli2embed4')
            and compact_binding.get('physical_bands') == count
            and compact_binding.get('complete_all_FILE_parents') is True
            and compact_binding.get('field_policy') == 'unwindowed_U_of_compact_native_pauli'
            and compact_binding.get('normalization') == 'one compact target A; no represented-field renormalization'
            and source.get('source_frame_policy') == 'compact_native_pauli_common_A_before_U'
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
    if record['operator_binding'] != expected or record['operator_identity'] != _identity(expected):
        raise ValueError('GATE resident_hartree_operator: geometry, field recipe or numerical owner changed')
    return record


def bind_resident_hartree(record, *, wfn, sym, plan, artifact,
                         wfn_fingerprint_binding, band_range):
    """Bind the two-member IO carrier to this invocation's actual source."""
    if not isinstance(record, dict) or set(record) != {'parent_kij_ry', 'provenance'}:
        raise ValueError('resident Hartree requires one paired native matrix and provenance')
    provenance = require_resident_hartree_source(record['provenance'],
        wfn=wfn, sym=sym, plan=plan, artifact=artifact,
        wfn_fingerprint_binding=wfn_fingerprint_binding, band_range=band_range)
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
    return dict(parent_kij_ry=record['parent_kij_ry'], provenance=provenance,
        source_context=dict(wfn_fingerprint_binding=wfn_fingerprint_binding,
            artifact=thin_artifact, plan=plan))


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
    from .gw_config import BispinorGWMode, QPSolver, coerce_bispinor_gw_mode
    from symmetry_maps import unfold_file_wedge_band_operator

    if (not isinstance(record, dict)
            or set(record) != {'parent_kij_ry', 'provenance', 'source_context'}
            or not config.bispinor
            or coerce_bispinor_gw_mode(config.bispinor_gw) is not BispinorGWMode.COULOMB_ONLY
            or config.qp_solver is not QPSolver.ONE_SHOT_DFT):
        raise ValueError('GATE resident_hartree_fixed_source: the reconstructed field requires fixed-source Coulomb-only GW')
    context = record['source_context']
    provenance = require_resident_hartree_source(record['provenance'],
        wfn=wfn, sym=sym, band_range=band_range, **context)
    if int(getattr(config, 'sys_dim', 3)) != int(provenance['source_binding'].get('sys_dim', 3)):
        raise ValueError('GATE resident_hartree_fixed_source: requested Coulomb kernel differs from its bound source')
    matrix = record['parent_kij_ry']
    lo, hi = map(int, band_range)
    start, stop = provenance['band_range']
    bands = stop-start
    sharding = NamedSharding(mesh, P(None, 'x', 'y'))
    if (not isinstance(matrix, jax.Array) or matrix.dtype != jnp.complex128
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
    native = jax.jit(lambda value: pad_square(value, native_axis),
                     out_shardings=sharding)(matrix)
    if not bool(jax.device_get(jnp.all(jnp.isfinite(native)))):
        raise ValueError('GATE resident_hartree_payload: nonfinite physical matrix')
    full = unfold_file_wedge_band_operator(sym, native, trs_rule='conj')
    output_axis = sigma_band_axis(hi-lo, mesh, ansatz='static')
    def select(value):
        selected = value[:, lo-start:hi-start, lo-start:hi-start]
        return pad_square(selected, output_axis)
    return jax.jit(select, out_shardings=sharding)(full)


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

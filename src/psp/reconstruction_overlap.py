"""Explicit reconstruction metric and finite-WFN symmetric Lowdin convention.

U=[I,X](I+X^dagger X)^(-1/2) is isometric. The full reconstructed Gram
therefore follows from Pauli overlaps with raw atomic differences, without
four-spinor quadrature or a normalized-tail cutoff. Atomic spheres must
not overlap; the caller authenticates geometry and atomic source identity.

Lowdin is opt-in at the fitting owner. Retaining the original energy labels
after nonunitary band mixing defines an effective vertex model, not exact
all-electron eigenstate equivalence. One factor must rotate the smooth
carrier, all atomic coefficients, and every sample endpoint coherently.
"""
from __future__ import annotations

import hashlib
import numpy as np

from psp.augmentation_spinors import (
    atomic_pauli_fourier, build_pauli_fourier_cache,
    evaluate_pauli_fourier_cache, spinor_function_labels,
)


def _delta_radial(data):
    r = np.asarray(data['r'], dtype=np.float64)
    w = np.asarray(data['weights_dr'], dtype=np.float64)
    u = np.asarray(data['delta_u'], dtype=np.complex128)
    ell, kappa = np.asarray(data['l']), np.asarray(data['kappa'])
    spinor_function_labels(ell, kappa)
    if (r.ndim != 1 or len(r) < 2 or w.shape != r.shape
            or u.shape != (len(r), len(ell)) or not np.all(np.isfinite(r))
            or not np.all(np.isfinite(w)) or not np.all(np.isfinite(u))
            or np.any(r <= 0) or np.any(np.diff(r) <= 0) or np.any(w <= 0)):
        raise ValueError("invalid raw atomic correction radial arrays")
    return r, w, u, ell.astype(int), kappa.astype(int)


def _delta_identity(data):
    digest = hashlib.sha256()
    for name in ('r', 'weights_dr', 'delta_u', 'l', 'kappa'):
        value = np.ascontiguousarray(data[name])
        digest.update(name.encode())
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def atomic_delta_gram(data):
    """Raw Pauli B_ij=<delta_i|delta_j>, with exact angular selection.

    Radial data use u=rR. The returned constant-size block includes all mj
    members, and never mixes distinct kappa even when radial shapes overlap.
    """
    _, w, u, ell, kappa = _delta_radial(data)
    labels = spinor_function_labels(ell, kappa)
    opf, m2 = labels[:, 0], labels[:, 1]
    radial = (u.conj().T*w) @ u
    allowed = (kappa[opf, None] == kappa[None, opf]) & (m2[:, None] == m2[None, :])
    return radial[opf[:, None], opf[None, :]]*allowed


def build_delta_radial_cache(data, **controls):
    """Species raw-delta transform cache, authenticated against its arrays."""
    r, w, u, ell, kappa = _delta_radial(data)
    cache = build_pauli_fourier_cache(u/r[:, None], r, w, ell, kappa, **controls)
    cache['source_identity'] = _delta_identity(data)
    return cache


def atomic_delta_overlap_table(data, wavevectors_cart, *, center_cart,
                               cell_volume, normalized_rkb_source=False,
                               radial_cache=None):
    """Conjugated Fourier table for d_ni=<delta_i|psi_n>, (function,2,G).

    Cell-volume normalization is included exactly once. For smooth normalized
    RKB upper components, the inverse canonical r(K) restores the original
    Pauli source before projection. The delta is never dualized or normalized.
    """
    r, w, u, ell, kappa = _delta_radial(data)
    volume = float(cell_volume)
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError("atomic delta overlap requires positive cell volume")
    if radial_cache is None:
        table = atomic_pauli_fourier(u/r[:, None], r, w, ell, kappa,
                                     wavevectors_cart, center_cart=center_cart)
    else:
        if (radial_cache.get('source_identity') != _delta_identity(data)
                or not np.array_equal(radial_cache['ell'], ell)
                or not np.array_equal(radial_cache['kappa'], kappa)):
            raise ValueError("atomic delta Fourier cache source mismatch")
        table = evaluate_pauli_fourier_cache(radial_cache, wavevectors_cart,
                                             center_cart=center_cart)
    table = table.conj()/np.sqrt(volume)
    if normalized_rkb_source:
        from common.bispinor_init import _normalized_rkb_factor
        table = table/np.asarray(_normalized_rkb_factor(wavevectors_cart))[None, None, :]
    return table


def reconstruction_gram_correction(coefficients, delta_overlaps, delta_gram):
    r"""One atom's full band-Gram correction, including the smooth residual.

    C and D have shape (...,band,function), D_ni=<delta_i|psi_n>, and
    B_ij=<delta_i|delta_j>. Returns D* C^T + C* D^T + C* B C^T in band-row
    convention. Stream atomic blocks; no orbital sample cloud is required.
    """
    c = np.asarray(coefficients, dtype=np.complex128)
    d = np.asarray(delta_overlaps, dtype=np.complex128)
    b = np.asarray(delta_gram, dtype=np.complex128)
    if (c.ndim < 2 or d.shape != c.shape or b.shape != (c.shape[-1],)*2
            or not np.all(np.isfinite(c)) or not np.all(np.isfinite(d))
            or not np.all(np.isfinite(b))):
        raise ValueError("reconstruction Gram requires paired coefficients and an atomic Gram block")
    _hermitian(b, 'atomic delta Gram')
    cross = np.einsum('...ni,...mi->...nm', d.conj(), c)
    return (cross+cross.conj().swapaxes(-1, -2)
            + np.einsum('...ni,ij,...mj->...nm', c.conj(), b, c, optimize=True))


def reconstruction_gram(source_gram, coefficients, delta_overlaps, delta_gram):
    """Add one atomic correction to the actual smooth-source Gram.

    Pass the previous accumulated Gram when streaming several nonoverlapping
    atoms. The source Gram is measured from source coefficients, not assumed
    to be identity, so physical source drift and zero padding remain visible.
    """
    source = np.asarray(source_gram, dtype=np.complex128)
    correction = reconstruction_gram_correction(coefficients, delta_overlaps, delta_gram)
    if source.shape != correction.shape or not np.all(np.isfinite(source)):
        raise ValueError("smooth-source and reconstructed band Gram shapes differ")
    _hermitian(source, 'smooth-source Gram')
    return source+correction


def _hermitian(value, name):
    if (value.ndim < 2 or value.shape[-1] != value.shape[-2]
            or not value.shape[-1] or not np.all(np.isfinite(value))):
        raise ValueError(f"{name} must be a finite nonempty square matrix")
    scale = np.maximum(1., np.max(np.abs(value), axis=(-2, -1)))
    error = np.max(np.abs(value-value.conj().swapaxes(-1, -2)), axis=(-2, -1))
    if np.any(error > 128*np.finfo(float).eps*value.shape[-1]*scale):
        raise ValueError(f"{name} is not Hermitian within floating-point accumulation tolerance")
    return error


def lowdin_factor(gram, *, physical_bands):
    """Explicit symmetric inverse-root and finite-WFN metric receipts.

    Physical bands form the leading block. Padded source rows/columns must
    be zero; they receive identity only in the factorization matrix and
    rotation factor. Nonpositive or numerically unresolved physical metrics
    refuse rather than truncate an orbital or choose a hidden rank threshold.
    Returned arrays have the same parent batch axes as the input Gram.
    """
    full = np.asarray(gram, dtype=np.complex128)
    hermitian_error = _hermitian(full, 'reconstructed band Gram')
    n = int(physical_bands)
    nb = full.shape[-1]
    if n != physical_bands or n <= 0 or n > nb:
        raise ValueError("physical band count must be a positive integer within the carrier")
    scale = max(1., float(np.max(np.abs(full))))
    tolerance = 128*np.finfo(float).eps*nb*scale
    if n < nb and (np.max(np.abs(full[..., n:, :])) > tolerance
                   or np.max(np.abs(full[..., :, n:])) > tolerance):
        raise ValueError("padded reconstruction Gram rows must remain zero")
    block = (full[..., :n, :n]+full[..., :n, :n].conj().swapaxes(-1, -2))/2
    eigenvalue, eigenvector = np.linalg.eigh(block)
    minimum, maximum = eigenvalue[..., 0], eigenvalue[..., -1]
    resolution = 128*np.finfo(float).eps*n*np.maximum(1., maximum)
    if np.any(minimum <= resolution):
        raise ValueError("reconstructed physical band metric is nonpositive or numerically unresolved")
    factor = np.broadcast_to(np.eye(nb, dtype=np.complex128), full.shape).copy()
    factor[..., :n, :n] = ((eigenvector*eigenvalue[..., None, :]**-.5)
                           @ eigenvector.conj().swapaxes(-1, -2))
    padded_metric = np.zeros_like(full)
    padded_metric[..., :n, :n] = block
    if n < nb:
        padded_metric[..., n:, n:] = np.eye(nb-n)
    restored = factor.conj().swapaxes(-1, -2) @ padded_metric @ factor
    defect = block-np.eye(n)
    diagonal = np.diagonal(defect, axis1=-2, axis2=-1)
    offdiagonal = defect.copy()
    index = np.arange(n)
    offdiagonal[..., index, index] = 0
    return dict(gram=full, inverse_sqrt=factor, physical_bands=n,
                eigenvalue_min=minimum, eigenvalue_max=maximum,
                condition_number=maximum/minimum,
                max_diagonal_defect=np.max(np.abs(diagonal), axis=-1),
                max_offdiagonal_defect=np.max(np.abs(offdiagonal), axis=(-2, -1)),
                hermitian_error=hermitian_error,
                factor_isometry_error=np.max(np.abs(restored-np.eye(nb)), axis=(-2, -1)),
                convention='full_wfn_lowdin_effective_vertex')


def rotate_band_rows(values, inverse_sqrt, *, band_axis=1, conjugated=False):
    r"""JAX rotation out_n=sum_m values_m A_mn, preserving caller sharding.

    Parent is axis zero; A is (parent,band,band), or a shared (band,band)
    matrix. Band axis may also be the last axis of complementary sample
    faces. Conjugated orbital faces require A* and explicitly set that flag.
    The caller owns compilation, layouts and donation; no host gather occurs.
    """
    import jax.numpy as jnp

    value, factor = jnp.asarray(values), jnp.asarray(inverse_sqrt)
    if value.ndim < 2 or int(band_axis) != band_axis or not -value.ndim <= band_axis < value.ndim:
        raise ValueError("band rotation requires a valid parent/band array axis")
    axis = int(band_axis) % value.ndim
    if axis == 0:
        raise ValueError("band rotation requires a parent axis and a distinct band axis")
    bands = value.shape[axis]
    if (factor.ndim not in (2, 3) or factor.shape[-2:] != (bands, bands)
            or (factor.ndim == 3 and factor.shape[0] != value.shape[0])):
        raise ValueError("band rotation factor disagrees with parent/band carrier shape")
    if conjugated:
        factor = factor.conj()
    row = jnp.moveaxis(value, axis, 1)
    if factor.ndim == 2:
        rotated = jnp.einsum('mn,pm...->pn...', factor, row)
    else:
        rotated = jnp.einsum('pmn,pm...->pn...', factor, row)
    return jnp.moveaxis(rotated, 1, axis)


COMPACT_PAULI_FRAME_MODEL = 'compact_native_pauli_common_frame_v1'
COMPACT_PAULI_FRAME_SCHEMA = 'lorrax.dev.compact_pauli_common_frame.v1'
COMPACT_PAULI_SUPPORT_POLICY = (
    'B and D use the SAME authenticated native positive Hermite-cell interval; '
    'zero contribution outside that interval; no analytic nucleus extrapolation')
AE_LARGE_SUPPORT_POLICY = (
    'B uses compact construction large and its free sigma-gradient; '
    'D uses chi=R_inverse L; finite-K served sphere/tail/cross are diagnostics')
AE_LARGE_TARGET_WITNESS = dict(
    schema='lorrax.dev.ae_large_free_graph_target_witness.v1',
    kind='compact_construction_large_sobolev')


def native_hermite_delta_gram(data, *, quadrature_order=4):
    """Integrate the declared compact native u-Hermite target with its owner.

    The interval starts at the first positive native radius: no origin
    continuation is invented. Four GL nodes integrate each cubic-u product
    exactly up to rounding. A higher order is an independent quadrature
    control, not a different physical target.
    """
    from psp.atomic_reconstruction import evaluate_radial_correction

    r = np.asarray(data['r'], np.float64)
    order = int(quadrature_order)
    if order != quadrature_order or order < 4:
        raise ValueError('compact native Gram requires integer GL order >=4')
    x, w = np.polynomial.legendre.leggauss(order)
    mid, half = (r[1:]+r[:-1])/2, np.diff(r)/2
    query = (mid[:, None]+half[:, None]*x).ravel()
    weights = (half[:, None]*w).ravel()
    radial = evaluate_radial_correction(data, query)[0]
    return atomic_delta_gram(dict(r=query, weights_dr=weights,
        delta_u=query[:, None]*radial, l=data['l'], kappa=data['kappa']))


def _ae_large_target_gram_witness(entry, data, pins, *, support_radius):
    """Remeasure the construction L/free-small target, not served bank norms."""
    import importlib.util
    import json
    from pathlib import Path
    from psp.augmentation_cache import _payload_hash, _target_inputs
    from psp.augmentation_spinors import free_graph_small_from_large

    path, expected = entry.get('target_witness_file'), entry.get('target_witness_sha256')
    if (not isinstance(path, str) or not Path(path).is_absolute()
            or pins.get(path) != expected
            or hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected):
        raise ValueError('AE-large target requires the independently pinned construction witness')
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata_json']))
        arrays = {k:archive[k].copy() for k in archive.files if k != 'metadata_json'}
    required = {'source_radius', 'source_weights_dr', 'source_large_R',
        'source_dlarge_R_dr', 'source_lower_free_graph_R', 'ell', 'kappa',
        'labels', 'target_B', 'momentum', 'weights_dK', 'pauli_radial_spectrum',
        'normalized_large_spectrum', 'normalized_small_spectrum', 'target_large_spectrum',
        'native_small_diagnostic_spectrum', 'source_native_small_R', 'represented_spectral_B'}
    if (set(arrays) != required
            or any(metadata.get(k) != v for k,v in AE_LARGE_TARGET_WITNESS.items())
            or metadata.get('payload_sha256') != _payload_hash(arrays)):
        raise ValueError('AE-large construction witness schema/kind/payload mismatch')
    control = metadata.get('construction_controls')
    if not isinstance(control, dict):
        raise ValueError('AE-large construction witness requires exact construction controls')
    inputs = _target_inputs(data, control, support_radius)
    if (inputs is None or metadata.get('target_binding') != inputs[3]
            or entry.get('target_binding') != inputs[3]
            or entry.get('construction_controls') != control):
        raise ValueError('AE-large witness/frame and primitive constructor targets differ')
    descriptor = inputs[3]['descriptor']
    if (metadata.get('construction_window') !=
            [descriptor['dirac_window_start'], descriptor['dirac_window_stop']]
            or metadata.get('target_radial_domain') != [0.,descriptor['dirac_window_stop']]):
        raise ValueError('AE-large exact construction support differs from its input window')
    owners = metadata.get('source_owners_sha256')
    if not isinstance(owners, dict) or not owners:
        raise ValueError('AE-large construction witness requires numerical owner pins')
    for module in ('psp.augmentation_cache', 'psp.augmentation_spinors', 'common.bispinor_init'):
        origin = importlib.util.find_spec(module).origin
        actual = hashlib.sha256(Path(origin).read_bytes()).hexdigest()
        if {pin for name,pin in owners.items() if Path(name).name == Path(origin).name} != {actual}:
            raise ValueError('AE-large construction witness numerical owner changed')
    r, w = arrays['source_radius'], arrays['source_weights_dr']
    ell, kappa = arrays['ell'], arrays['kappa']
    if (r.dtype != np.float64 or w.dtype != np.float64 or w.shape != r.shape
            or r.ndim != 1 or len(r) < 2 or np.any(r <= 0)
            or np.any(np.diff(r) <= 0) or np.any(w <= 0)
            or not np.isfinite(r).all() or not np.isfinite(w).all()
            or np.any(r >= descriptor['dirac_window_stop'])
            or not np.array_equal(ell, data['l']) or not np.array_equal(kappa, data['kappa'])
            or not np.array_equal(arrays['labels'], spinor_function_labels(ell, kappa))):
        raise ValueError('AE-large construction witness radial/angular support mismatch')
    for key in ('source_large_R', 'source_dlarge_R_dr', 'source_lower_free_graph_R'):
        value = arrays[key]
        if (value.dtype != np.complex128 or value.shape != (len(r),len(ell))
                or not np.isfinite(value).all()):
            raise ValueError('AE-large construction witness fields must be finite complex128')
    lower = free_graph_small_from_large(r, arrays['source_large_R'],
        arrays['source_dlarge_R_dr'], kappa)
    if not np.array_equal(lower, arrays['source_lower_free_graph_R']):
        raise ValueError('AE-large target lower is not the same free sigma-gradient including window derivative')
    common = dict(r=r, weights_dr=w)
    upper_B = atomic_delta_gram(dict(common, delta_u=r[:,None]*arrays['source_large_R'], l=ell, kappa=kappa))
    lower_B = atomic_delta_gram(dict(common, delta_u=r[:,None]*lower,
        l=2*np.abs(kappa)-1-ell, kappa=-kappa))
    B = upper_B+lower_B
    saved = arrays['target_B']
    if (saved.shape != B.shape or saved.dtype != np.complex128
            or not np.isfinite(saved).all()
            or np.max(abs(saved-B)) > 2e-10*max(1.,float(np.max(abs(B))))):
        raise ValueError('AE-large target B is not its construction large/free-small Sobolev Gram')
    return B


def _authenticate_ae_large_frame_witness(metadata, arrays, pins, *, wfn_sha256):
    """Join an independently contracted full-source witness to actual arrays."""
    import json
    from pathlib import Path

    bound = metadata.get('independent_target_witness')
    required = {'schema', 'program_file', 'program_sha256', 'spec_file', 'spec_sha256',
        'receipt_file', 'receipt_sha256', 'arrays_file', 'arrays_sha256',
        'target_binding_sha256_by_species'}
    if (not isinstance(bound, dict) or set(bound) != required
            or bound['schema'] != 'lorrax.dev.ae_large_target_witness.v1'):
        raise ValueError('AE-large frame requires its independent full-source target witness')
    for stem in ('program', 'spec', 'receipt', 'arrays'):
        name, expected = bound[stem+'_file'], bound[stem+'_sha256']
        if (not isinstance(name, str) or not Path(name).is_absolute()
                or pins.get(name) != expected
                or hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected):
            raise ValueError('Independent AE-large witness file identity differs: '+stem)
    receipt = json.loads(Path(bound['receipt_file']).read_text())
    if (receipt.get('schema') != bound['schema']
            or receipt.get('status') != 'PASS_INDEPENDENT_CONTINUUM_AE_LARGE_TARGET_FULL_SOURCE'
            or receipt.get('program_sha256') != bound['program_sha256']
            or receipt.get('spec_sha256') != bound['spec_sha256']
            or receipt.get('arrays_sha256') != bound['arrays_sha256']
            or receipt.get('wfn_sha256') != wfn_sha256
            or receipt.get('physical_bands') != metadata['physical_bands']
            or receipt.get('complete_all_FILE_parents') is not True):
        raise ValueError('Independent AE-large target witness receipt/domain differs')
    for key in ('source_files_sha256', 'source_input_files_sha256'):
        recorded = receipt.get(key)
        if (not isinstance(recorded, dict) or not recorded
                or any(pins.get(name) != expected for name,expected in recorded.items())):
            raise ValueError('Independent AE-large witness and frame have different source/input pins')
    spec = json.loads(Path(bound['spec_file']).read_text())
    math_file, math_sha = spec.get('math_owner_file'), spec.get('math_owner_sha256')
    if (not isinstance(math_file, str) or not Path(math_file).is_absolute()
            or receipt.get('math_owner_sha256') != math_sha
            or receipt['source_files_sha256'].get(math_file) != math_sha):
        raise ValueError('Independent AE-large witness literal math owner differs')
    species = metadata['atomic_species_inputs']
    descriptors = {z:entry['target_binding'] for z,entry in species.items()}
    descriptor_hashes = {z:hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',',':'), allow_nan=False).encode()).hexdigest() for z,value in descriptors.items()}
    if (receipt.get('target_binding_by_species') != descriptors
            or bound['target_binding_sha256_by_species'] != descriptor_hashes
            or receipt.get('common_spectrum_sha256_by_species') !=
                {z:entry['spectral_witness_sha256'] for z,entry in species.items()}):
        raise ValueError('Independent AE-large witness primitive targets differ')
    with np.load(bound['arrays_file'], allow_pickle=False) as archive:
        witness = {k:archive[k].copy() for k in archive.files}
    controls = {f'species_B_gradient_control_{z}' for z in species}
    if set(witness) != set(arrays)|controls:
        raise ValueError('Independent AE-large witness must cover every physical frame array')
    for key,value in arrays.items():
        other = witness[key]
        if other.shape != value.shape or other.dtype != value.dtype:
            raise ValueError('Independent AE-large witness array shape/dtype differs: '+key)
        if value.dtype.kind in 'fc':
            if (not np.isfinite(other).all()
                    or np.max(abs(other-value)) > 2e-10*max(1.,float(np.max(abs(other))))):
                raise ValueError('Independent AE-large witness physical array differs: '+key)
        elif not np.array_equal(other, value):
            raise ValueError('Independent AE-large witness geometry/source chart differs: '+key)
    for z in species:
        B, control = witness[f'species_B_{z}'], witness[f'species_B_gradient_control_{z}']
        if (control.shape != B.shape or control.dtype != B.dtype
                or not np.isfinite(control).all()
                or np.max(abs(B-control)) > 2e-10*max(1.,float(np.max(abs(B))))):
            raise ValueError('Independent AE-large graph/Sobolev quadrature control differs')


def load_compact_pauli_frame(path, *, expected_file_sha256, expected_wfn_sha256,
                             tables, parent_k_frac, gvecs, ngk_valid,
                             atom_types, centers_cart, physical_bands,
                             field_policy='unwindowed_U_of_compact_native_pauli'):
    """Authenticate a complete fixed-frame endpoint bundle, never WFN arrays.

    The file contains only physical FILE-parent C/D/Gram arrays and one A.
    The caller supplies the authoritative whole-WFN identity and actual
    loader geometry. Live source/action checks still precede using A in the
    fitting stage; a prepared frame alone is not fitting admission.
    """
    import json
    from pathlib import Path
    from psp.atomic_reconstruction import load_atomic_reconstruction
    from psp.augmentation_cache import _payload_hash
    from psp.augmentation_cache import paired_field_policy_contract, PAIRED_AE_LARGE_FIELD_POLICY

    contract = paired_field_policy_contract(field_policy)
    ae_large = field_policy == PAIRED_AE_LARGE_FIELD_POLICY

    def file_hash(name):
        return hashlib.sha256(Path(name).read_bytes()).hexdigest()

    def digest_ok(value):
        return (isinstance(value, str) and len(value) == 64
                and all(c in '0123456789abcdef' for c in value))

    if not digest_ok(expected_file_sha256) or not digest_ok(expected_wfn_sha256):
        raise ValueError('compact frame requires explicit file and whole-WFN SHA256')
    if file_hash(path) != expected_file_sha256:
        raise ValueError('compact frame file identity mismatch')
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata_json']))
        arrays = {key: archive[key].copy() for key in archive.files if key != 'metadata_json'}
    if (metadata.get('schema') != contract['frame_schema']
            or metadata.get('model') != contract['frame_model']
            or metadata.get('source_frame') != 'original WFN Pauli band labels'
            or metadata.get('complete_all_FILE_parents') is not True
            or metadata.get('physical_bands') != physical_bands
            or metadata.get('wfn_sha256') != expected_wfn_sha256
            or metadata.get('target_support_policy') !=
                (AE_LARGE_SUPPORT_POLICY if ae_large else COMPACT_PAULI_SUPPORT_POLICY)
            or metadata.get('payload_sha256') != _payload_hash(arrays)):
        raise ValueError('compact target model/source/full-FILE/payload policy mismatch')
    if ae_large and metadata.get('target_witness') != AE_LARGE_TARGET_WITNESS:
        raise ValueError('AE-large frame requires its distinct construction-target witness')
    pins, species = metadata.get('source_input_files_sha256'), metadata.get('atomic_species_inputs')
    if (not isinstance(pins, dict) or not pins or not isinstance(species, dict)
            or set(species) != set(map(str, tables))):
        raise ValueError('compact target requires all source and exact atomic-species bindings')
    controls = metadata.get('fourier_controls')
    if (not isinstance(controls,dict) or set(controls) != {'momentum_max','momentum_points',
            'relative_tolerance','absolute_tolerance','validation_points'}):
        raise ValueError('compact target requires the explicit public source-dual Fourier controls')
    for name, expected in pins.items():
        if not Path(name).is_absolute() or not digest_ok(expected) or file_hash(name) != expected:
            raise ValueError('compact target source/input identity mismatch: '+str(name))
    nk, nb, na = len(parent_k_frac), int(physical_bands), len(atom_types)
    control_gram = 'target_gram_quadrature_control' if ae_large else 'target_gram_GL8_control'
    required = {'source_gram', 'target_gram', control_gram, 'inverse_sqrt',
        'parent_FILE_rows', 'parent_k_frac', 'ngk_valid_parent', 'atom_types', 'centers_cart',
        'source_pauli_sha256_by_parent'}
    required |= {f'atom_{kind}_{a:03d}' for kind in ('C', 'D') for a in range(na)}
    required |= {f'gvec_parent_{p:03d}' for p in range(nk)}
    required |= {f'species_{kind}_{z}' for kind in ('B', 'labels') for z in tables}
    if set(arrays) != required:
        raise ValueError('compact frame arrays are incomplete or have unknown fields')
    k = np.asarray(arrays['parent_k_frac'])
    residual = k-np.asarray(parent_k_frac)
    if (k.shape != (nk, 3) or not np.isfinite(k).all()
            or np.max(abs(residual-np.rint(residual))) > 2e-12
            or arrays['parent_FILE_rows'].dtype.kind not in 'iu'
            or arrays['ngk_valid_parent'].dtype.kind not in 'iu'
            or arrays['atom_types'].dtype.kind not in 'iu'
            or not np.array_equal(arrays['parent_FILE_rows'], np.arange(nk))
            or not np.array_equal(arrays['ngk_valid_parent'], ngk_valid)
            or not np.array_equal(arrays['atom_types'], atom_types)
            or not np.array_equal(arrays['centers_cart'], centers_cart)
            or arrays['source_pauli_sha256_by_parent'].shape != (nk, 32)
            or arrays['source_pauli_sha256_by_parent'].dtype != np.uint8):
        raise ValueError('compact frame FILE-parent/atomic/source-codec geometry mismatch')
    for p, count in enumerate(ngk_valid):
        if (arrays[f'gvec_parent_{p:03d}'].dtype.kind not in 'iu'
                or not np.array_equal(arrays[f'gvec_parent_{p:03d}'], np.asarray(gvecs)[p, :count])):
            raise ValueError('compact frame physical ordered G rows mismatch')
    for key in ('source_gram', 'target_gram', control_gram, 'inverse_sqrt'):
        value = arrays[key]
        if value.shape != (nk, nb, nb) or value.dtype != np.complex128:
            raise ValueError('compact frame matrices require physical complex128 bands')
        _hermitian(value, key)
    for z, data in tables.items():
        entry = species[str(z)]
        if (entry.get('upf_sha256') != data['metadata']['source_sha256']
                or pins.get(entry.get('native')) != entry.get('native_sha256')
                or pins.get(entry.get('upf')) != entry.get('upf_sha256')):
            raise ValueError('compact native/PCA source differs from the fitting atomic table')
        native = load_atomic_reconstruction(entry['native'], entry['upf'])
        if native['metadata']['payload_sha256'] != data['metadata']['payload_sha256']:
            raise ValueError('compact target native amplitudes/PCA differ from the fitting table')
        interval = metadata.get('native_target_radial_interval_by_species_bohr', {}).get(str(z))
        if not ae_large and interval != [float(data['r'][0]), float(data['r'][-1])]:
            raise ValueError('compact target native Hermite interval mismatch')
        labels = spinor_function_labels(data['l'], data['kappa'])
        B = arrays[f'species_B_{z}']
        if (not np.array_equal(arrays[f'species_labels_{z}'], labels)
                or arrays[f'species_labels_{z}'].dtype.kind not in 'iu'
                or B.shape != (len(labels), len(labels)) or B.dtype != np.complex128):
            raise ValueError('compact target function labels/B do not cover the native channel inventory')
        _hermitian(B, 'compact target B')
        target_B = (_ae_large_target_gram_witness(entry, data, pins,
            support_radius=float(metadata['represented_sphere_radius_bohr']))
            if ae_large else native_hermite_delta_gram(data))
        if np.max(abs(B-target_B)) > 2e-10*max(1., float(np.max(abs(target_B)))):
            raise ValueError('compact target B is not the declared native Hermite GL4 Gram')
    for a, z in enumerate(atom_types):
        for kind in ('C', 'D'):
            value = arrays[f'atom_{kind}_{a:03d}']
            if (value.shape != (nk, nb, len(arrays[f'species_labels_{z}']))
                    or value.dtype != np.complex128 or not np.isfinite(value).all()):
                raise ValueError('compact target C/D physical bands or function labels mismatch')
    if ae_large:
        _authenticate_ae_large_frame_witness(metadata, arrays, pins,
            wfn_sha256=expected_wfn_sha256)
    return dict(arrays=arrays, metadata=metadata, file_sha256=expected_file_sha256)


def compact_frame_factor(source_gram, coefficients, frame, *, parent_start,
                         physical_bands, tolerance=2e-10):
    """Remeasure target G and A from a live packet before accepting common A.

    This gate checks source Gram and independently projected C, reconstructs
    target G with authenticated native D/B, and recomputes the positive
    symmetric Lowdin factor. The saved common A is used only after these
    checks. Transport ghosts receive identity; no served-field metric is
    normalized away.
    """
    data = frame['arrays']; n = int(physical_bands)
    first, stop = int(parent_start), int(parent_start)+len(source_gram)
    source = np.asarray(source_gram, np.complex128)
    if (not np.isfinite(tolerance) or tolerance <= 0 or len(coefficients) != len(data['atom_types'])
            or source.ndim != 3 or source.shape[-1] != source.shape[-2]
            or source.shape[-1] < n or first < 0 or stop > len(data['source_gram'])):
        raise ValueError('compact frame tolerance must be positive and finite')
    prepared_source = data['source_gram'][first:stop]
    source_error = float(np.max(abs(source[:, :n, :n]-prepared_source)))
    if source_error > tolerance:
        raise ValueError('live source Gram differs from the compact Pauli source frame')
    gram = source.copy(); C_error = 0.
    for a, (z, C) in enumerate(zip(data['atom_types'], coefficients)):
        prepared_C = data[f'atom_C_{a:03d}'][first:stop]
        C_error = max(C_error, float(np.max(abs(np.asarray(C)[:, :n]-prepared_C))))
        D = np.pad(data[f'atom_D_{a:03d}'][first:stop], ((0,0),(0,source.shape[-1]-n),(0,0)))
        gram = reconstruction_gram(gram, C, D, data[f'species_B_{z}'])
    if C_error > tolerance:
        raise ValueError('live pseudo-dual C differs from the compact original Pauli band frame')
    target_error = float(np.max(abs(gram[:, :n, :n]-data['target_gram'][first:stop])))
    if target_error > tolerance:
        raise ValueError('live reconstructed Gram differs from the compact native target')
    receipt = lowdin_factor(gram, physical_bands=n)
    saved = data['inverse_sqrt'][first:stop]
    factor_error = float(np.max(abs(receipt['inverse_sqrt'][:, :n, :n]-saved)))
    isometry_error = float(np.max(abs(saved.conj().swapaxes(-1,-2) @ gram[:, :n, :n] @ saved-np.eye(n))))
    if factor_error > tolerance or isometry_error > tolerance:
        raise ValueError('saved common A does not match the freshly measured target Lowdin/isometry')
    receipt['inverse_sqrt'][:, :n, :n] = saved
    receipt.update(compact_source_gram_error=np.full(len(source), source_error),
        compact_coefficient_error=np.full(len(source), C_error),
        compact_target_gram_error=np.full(len(source), target_error),
        compact_factor_error=np.full(len(source), factor_error),
        compact_factor_isometry_error=np.full(len(source), isometry_error))
    return receipt

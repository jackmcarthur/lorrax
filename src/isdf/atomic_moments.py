"""Served-field monopoles as signed charge fitting functionals.

The radial operator here is the already served cubic Hermite four-spinor
cache, including its declared support. It neither imposes a native overlap
identity nor applies a second kinetic-balance normalization. Auxiliary point
fields are explicitly charge functionals, not reconstructed wavefunctions.
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np

SCHEMA = 'lorrax.served_moment_cache.v1'
CARRIER = 'auxiliary_charge_functional'


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _served_operator_identity():
    """Bind the field and its overlap operator, independently of auxiliaries."""
    owners = {}
    for module in ('common.bispinor_init', 'common.gamma_matrices',
                   'psp.augmentation_spinors', 'psp.atomic_reconstruction',
                   'psp.augmentation_cache'):
        owners[module] = hashlib.sha256(Path(importlib.util.find_spec(module).origin).read_bytes()).hexdigest()
    source = inspect.getsource(build_served_overlap_cache) + inspect.getsource(served_overlap_table)
    return dict(operator='compact_cubic_hermite_four_spinor_gl5_v1',
                operator_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                owner_sources_sha256=owners)


def build_served_overlap_cache(normalized_cache, *, support_radius,
                               momentum_max=16., momentum_points=4097,
                               source_quad_order=5):
    """Prepare exact Hermite-cell B and Fourier bra tables once, offline.

    GL5 exactly integrates products of served cubic radial functions times
    r². Fourier rows have their own independent interpolation validation.
    No native partial-wave metric or omitted tail is substituted for this B.
    """
    from scipy.interpolate import CubicHermiteSpline
    from psp.augmentation_spinors import build_pauli_fourier_cache, spinor_function_labels

    radius = np.asarray(normalized_cache['radius'])
    support, order = float(support_radius), int(source_quad_order)
    if (radius.ndim != 1 or len(radius) < 2 or not np.isfinite(radius).all()
            or radius[0] != 0. or np.any(np.diff(radius) <= 0)
            or order != source_quad_order or order < 5 or not np.isfinite(support)
            or not 0 < support <= radius[-1]):
        raise ValueError('served moments require a radial cache from zero, support within cache, and GL order >=5')
    edges = np.concatenate((radius[radius < support], [support]))
    x, w = np.polynomial.legendre.leggauss(order)
    half, mid = np.diff(edges)/2, (edges[:-1]+edges[1:])/2
    r, wr = (mid[:, None]+half[:, None]*x).ravel(), (half[:, None]*w).ravel()
    large = CubicHermiteSpline(radius, normalized_cache['large_R'],
                              normalized_cache['dlarge_R_dr'], axis=0)(r)
    small = CubicHermiteSpline(radius, normalized_cache['small_R'],
                              normalized_cache['dsmall_R_dr'], axis=0)(r)
    ell, kappa = np.asarray(normalized_cache['ell']), np.asarray(normalized_cache['kappa'])
    controls = dict(momentum_max=momentum_max, momentum_points=momentum_points,
                    relative_tolerance=1e-10, absolute_tolerance=1e-12)
    upper = build_pauli_fourier_cache(large, r, wr, ell, kappa, **controls)
    lower = build_pauli_fourier_cache(small, r, wr, 2*np.abs(kappa)-1-ell, -kappa, **controls)
    labels = spinor_function_labels(ell, kappa)
    opf, mj = labels.T
    radial = (large.conj().T*(wr*r*r)) @ large + (small.conj().T*(wr*r*r)) @ small
    metric = radial[opf[:, None], opf[None, :]]
    metric *= ((kappa[opf, None] == kappa[None, opf]) & (mj[:, None] == mj[None, :]))
    return dict(upper=upper, lower=lower, B=metric, labels=labels,
                support_radius=support, lower_radius=0., source_quad_order=order,
                source_identity=_served_operator_identity())


def served_overlap_table(cache, K_cart, *, center_cart, cell_volume):
    """Bra (function,4,G) of the served field, divided by sqrt(cell volume).

    Source coefficients are already U psi(G). An additional R(K) would change
    this overlap. The caller must mask every ghost source G coefficient.
    """
    from psp.augmentation_spinors import evaluate_pauli_fourier_cache

    volume = float(cell_volume)
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError('served moment overlap requires a positive physical volume')
    upper = evaluate_pauli_fourier_cache(cache['upper'], K_cart, center_cart=center_cart)
    lower = evaluate_pauli_fourier_cache(cache['lower'], K_cart, center_cart=center_cart)
    return np.concatenate((upper, lower), axis=1).conj()/np.sqrt(volume)


def _cache_arrays(cache):
    arrays = dict(B=np.asarray(cache['B']), labels=np.asarray(cache['labels']))
    for leg in ('upper', 'lower'):
        for name in ('momentum', 'radial', 'ell', 'kappa'):
            arrays[f'{leg}_{name}'] = np.asarray(cache[leg][name])
    return arrays


def _payload_hash(arrays):
    digest = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        value = np.asarray(value)
        digest.update(key.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.dtype.str.encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _validate_cache_arrays(arrays):
    from psp.augmentation_spinors import spinor_function_labels

    expected = {'B', 'labels'} | {f'{leg}_{name}' for leg in ('upper', 'lower')
                               for name in ('momentum', 'radial', 'ell', 'kappa')}
    if set(arrays) != expected:
        raise ValueError('served moment cache array schema mismatch')
    for leg in ('upper', 'lower'):
        momentum, radial, ell, kappa = (arrays[f'{leg}_{name}'] for name in ('momentum', 'radial', 'ell', 'kappa'))
        if (momentum.dtype != np.float64 or momentum.ndim != 1 or len(momentum) < 8
                or momentum[0] != 0. or not np.isfinite(momentum).all()
                or np.any(np.diff(momentum) <= 0) or ell.dtype.kind not in 'iu'
                or kappa.dtype.kind not in 'iu' or ell.ndim != 1 or kappa.shape != ell.shape
                or np.any(ell < 0) or np.any(kappa == 0)
                or np.any((kappa != ell) & (kappa != -ell-1))
                or radial.shape != (len(momentum), len(ell))
                or radial.dtype != np.complex128 or not np.isfinite(radial).all()):
            raise ValueError('served moment cache radial Fourier arrays are malformed')
    if (not np.array_equal(arrays['upper_momentum'], arrays['lower_momentum'])
            or not np.array_equal(arrays['lower_kappa'], -arrays['upper_kappa'])
            or not np.array_equal(arrays['lower_ell'], 2*np.abs(arrays['upper_kappa'])-1-arrays['upper_ell'])):
        raise ValueError('served moment upper/lower angular or momentum labels disagree')
    labels = spinor_function_labels(arrays['upper_ell'], arrays['upper_kappa'])
    B = arrays['B']
    if (not np.array_equal(arrays['labels'], labels) or B.shape != (len(labels), len(labels))
            or B.dtype != np.complex128 or not np.isfinite(B).all()
            or np.max(abs(B-B.conj().T)) > 2e-12*max(1., np.max(abs(B)))):
        raise ValueError('served moment metric is not finite Hermitian in canonical label order')
    return arrays


def write_served_moment_cache(path, cache, *, normalized_cache_sha256):
    """Write a new immutable source-bound artifact; never rebuild while fitting."""
    target = Path(path)
    if target.exists():
        raise FileExistsError(f'preserve served moment artifact: {target}')
    if cache['source_identity'] != _served_operator_identity():
        raise ValueError('served moment operator source identity mismatch')
    arrays = _validate_cache_arrays(_cache_arrays(cache))
    metadata = dict(schema=SCHEMA, normalized_cache_sha256=str(normalized_cache_sha256),
        payload_sha256=_payload_hash(arrays), source_identity=cache['source_identity'],
        support_radius=float(cache['support_radius']), lower_radius=float(cache['lower_radius']),
        source_quad_order=int(cache['source_quad_order']),
        validation={leg: {key: cache[leg][key] for key in
            ('maximum_absolute_error', 'maximum_scaled_error', 'validation_points')}
                    for leg in ('upper', 'lower')})
    if metadata['lower_radius'] != 0. or metadata['source_quad_order'] < 5:
        raise ValueError('served moment artifact must cover the full sphere with resolved Hermite products')
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays, metadata_json=np.asarray(_json_bytes(metadata).decode()))
    return metadata


def load_served_moment_cache(path, *, normalized_cache_sha256, support_radius):
    """Refuse a changed source, field support, payload, or angular ordering."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata_json']))
        arrays = {key: archive[key].copy() for key in archive.files if key != 'metadata_json'}
    if (metadata.get('schema') != SCHEMA
            or metadata.get('normalized_cache_sha256') != str(normalized_cache_sha256)
            or metadata.get('support_radius') != float(support_radius)
            or metadata.get('lower_radius') != 0. or metadata.get('source_quad_order', 0) < 5
            or metadata.get('source_identity') != _served_operator_identity()):
        raise ValueError('served moment artifact source or field identity mismatch')
    _validate_cache_arrays(arrays)
    if metadata.get('payload_sha256') != _payload_hash(arrays):
        raise ValueError('served moment artifact payload identity mismatch')
    cache = {key: metadata[key] for key in ('support_radius', 'lower_radius', 'source_quad_order', 'source_identity')}
    cache.update(B=arrays['B'], labels=arrays['labels'], metadata=metadata)
    for leg in ('upper', 'lower'):
        cache[leg] = {name: arrays[f'{leg}_{name}'] for name in ('momentum', 'radial', 'ell', 'kappa')}
        cache[leg].update(metadata['validation'][leg])
    return cache


def exact_pair_moments(C_left, D_left, C_right, D_right, B, *, array_api=np):
    """Finite-q served sphere integral, for independently chosen band windows."""
    xp = array_api
    c, d, cc, dd, b = map(xp.asarray, (C_left, D_left, C_right, D_right, B))
    if (c.ndim != 2 or d.shape != c.shape or cc.ndim != 2 or dd.shape != cc.shape
            or c.shape[1] != cc.shape[1] or b.shape != (c.shape[1], c.shape[1])):
        raise ValueError('served moment coefficient and radial overlap axes disagree')
    return d.conj() @ cc.T + c.conj() @ dd.T + c.conj() @ b @ cc.T


def auxiliary_charge_geometry(cache, *, radial_points=8, angular_order=7, auxiliary_radius=.7):
    """Signed H functionals on true points with exact κ/magnetic covariance.

    H=[[B,I],[I,0]] is factored per κ and repeated for every magnetic label.
    No inverse of B, fitted rank threshold, or eigenvalue truncation occurs.
    GL8 × Lebedev26 is exact for native SPD radial rank at most five.
    """
    from scipy.integrate import lebedev_rule
    from scipy.special import eval_jacobi
    from psp.augmentation_spinors import spinor_spherical_harmonic

    R, nrad = float(auxiliary_radius), int(radial_points)
    if not np.isfinite(R) or R <= 0 or nrad != radial_points or nrad < 1:
        raise ValueError('auxiliary charge geometry requires a positive radius and integer radial count')
    ell, kappa = cache['upper']['ell'], cache['upper']['kappa']
    labels = np.asarray(cache['labels'])
    opf, mj = labels.T
    x, w = np.polynomial.legendre.leggauss(nrad)
    r, wr = R*(x+1)/2, R*w/2
    directions, wa = lebedev_rule(int(angular_order))
    directions = directions.T
    # The declared Lebedev degree must integrate every retained orbital product.
    if int(angular_order) != angular_order or int(angular_order) < 2*int(np.max(ell)):
        raise ValueError('auxiliary angular quadrature does not integrate retained κ products')
    table = np.zeros((2, 2*len(labels), 4, len(r)*len(directions)), np.complex128)
    factor_receipts = []
    for k in np.unique(kappa):
        radial_ids = np.flatnonzero(kappa == k)
        l, nr = int(ell[radial_ids[0]]), len(radial_ids)
        if nrad < nr+l+1:
            raise ValueError('auxiliary radial quadrature does not integrate the full retained rank')
        mvals = np.arange(-2*abs(k)+1, 2*abs(k), 2)
        ids = np.asarray([[np.flatnonzero((opf == i) & (mj == m))[0] for m in mvals] for i in radial_ids])
        b = np.asarray(cache['B'])[np.ix_(ids[:, 0], ids[:, 0])]
        if np.max(abs(b.imag)) > 2e-12*max(1., np.max(abs(b))):
            raise ValueError('auxiliary κ factor requires the authenticated real atomic radial metric')
        for column in range(len(mvals)):
            if not np.allclose(np.asarray(cache['B'])[np.ix_(ids[:, column], ids[:, column])], b, rtol=1e-12, atol=2e-12):
                raise ValueError('served radial metric is not magnetic-label covariant')
        h = np.block([[b.real, np.eye(nr)], [np.eye(nr), np.zeros((nr, nr))]])
        eigenvalues, vectors = np.linalg.eigh(h)
        if np.count_nonzero(eigenvalues > 0) != nr or np.count_nonzero(eigenvalues < 0) != nr:
            raise ValueError('auxiliary signed factor lost exact positive/negative inertia')
        radial = np.stack([(r/R)**l/R**1.5*np.sqrt(2*n+2*l+3)
                           *eval_jacobi(n, 0, 2*l+2, 2*r/R-1) for n in range(nr)], axis=1)
        omega = np.stack([spinor_spherical_harmonic(k, m, directions) for m in mvals])
        for sign, positive in enumerate((True, False)):
            v = vectors[:, eigenvalues > 0 if positive else eigenvalues < 0]
            lam = eigenvalues[eigenvalues > 0 if positive else eigenvalues < 0]
            radial_map = np.einsum('ia,ra->ir', v*np.sqrt(abs(lam))[None], radial)
            for i in range(2*nr):
                for m, function_id in enumerate(ids[i % nr]):
                    destination = function_id + (len(labels) if i >= nr else 0)
                    values = radial_map[i, :, None, None]*omega[m][None]
                    table[sign, destination, :2] = values.reshape(-1, 2).T
        factor_receipts.append(dict(kappa=int(k), radial_rank=nr,
            minimum_absolute_eigenvalue=float(np.min(abs(eigenvalues)))))
    return dict(carrier=CARRIER, relative_points=(r[:, None, None]*directions[None]).reshape(-1, 3),
                integration_weights=((wr*r*r)[:, None]*wa[None]).ravel(),
                signed_tables=table, radial_points=nrad, angular_order=int(angular_order),
                auxiliary_radius=R, function_count=len(labels), factor_receipts=factor_receipts)


def evaluate_auxiliary_charge(C, D, geometry, *, grid_sample_scale, array_api=np):
    """Return plus/minus upper-two charge-functional fields, with zero lower.

    C,D are already mixed by the same full-window band factor as the source.
    grid_sample_scale=sqrt(Omega/Nfft) makes point RHS grid units identical
    to the physical radial-density RHS. Divide an integrated signed RHS by
    sqrt(4π) only to obtain its radial Y00 monopole.
    """
    xp = array_api
    c, d = xp.asarray(C), xp.asarray(D)
    if (geometry.get('carrier') != CARRIER or c.ndim < 2 or d.shape != c.shape
            or c.shape[-1] != geometry['function_count']
            or not np.isfinite(grid_sample_scale) or float(grid_sample_scale) <= 0):
        raise ValueError('auxiliary charge carrier, coefficients, or grid scale disagree')
    table = xp.asarray(geometry['signed_tables'])
    n = c.shape[-1]
    return tuple(float(grid_sample_scale)*(xp.einsum('...ni,isp->...nsp', c, table[sign, :n])
                 + xp.einsum('...ni,isp->...nsp', d, table[sign, n:])) for sign in range(2))


def integrated_auxiliary_monopole(signed_point_rhs, packed_weights, *, array_api=np):
    """Integrate the canonical signed point RHS to radial Y00 moments."""
    xp = array_api
    values, weights = xp.asarray(signed_point_rhs), xp.asarray(packed_weights)
    if values.ndim != 3 or weights.shape != (values.shape[-1],):
        raise ValueError('auxiliary monopole RHS and packed integration weights disagree')
    return xp.einsum('qmp,p->qm', values, weights)/np.sqrt(4*np.pi)


def make_auxiliary_monopole_compressor(mesh, point_plan, canonical_atom_weights):
    """Integrate genuine canonical point faces, retaining atomic features on Y.

    The caller selects its physical q rows before the standard face-to-q
    redistribution. Point packing and image phases remain owned by the
    supplied fractional-point plan; this helper introduces no virtual labels.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local
    from common.shard_map import shard_map

    weights = np.asarray(canonical_atom_weights, np.float64)
    axis = point_plan.layout.axis
    if (getattr(point_plan, 'coordinate_kind', None) != 'fractional'
            or point_plan.nspinor != 4 or weights.ndim != 2 or weights.shape[0] == 0
            or weights.shape[1] != int(axis.n_logical) or not np.isfinite(weights).all()
            or np.any(weights < 0) or np.any(np.sum(weights, axis=1) <= 0)):
        raise ValueError('auxiliary monopoles require positive atomic weights on a genuine fractional four-spinor point plan')
    packed = axis.pack_host(weights, axis=1, fill_value=0.)/np.sqrt(4*np.pi)
    py = int(mesh.shape['y'])
    atom_pad = ((len(weights)+py-1)//py)*py
    packed = np.pad(packed, ((0, atom_pad-len(weights)), (0, 0)))
    table = device_put_process_local(packed, NamedSharding(mesh, P(None, 'y')))

    def integrate(z, row):
        partial = jnp.einsum('qmp,ap->qma', z, row)
        return jax.lax.psum_scatter(partial, 'y', scatter_dimension=2, tiled=True)
    kernel = jax.jit(shard_map(integrate, mesh=mesh,
        in_specs=(P(None, 'x', 'y'), P(None, 'y')), out_specs=P(None, 'x', 'y'), check_vma=False))
    return dict(carrier=CARRIER, compress=lambda rhs: kernel(rhs, table),
                atom_count=len(weights), atom_pad=atom_pad)


def raw_parent_moment_binding(wfn, *, k_parent_frac, gvecs, ngk_valid, centers_cart,
                              atom_types, cell_volume, physical_bands,
                              served_cache_sha256_by_species, wfn_fingerprint_binding=None):
    """Point-independent identity of cached unrotated full-window D_R.

    The canonical WFN fingerprint is a bounded gauge identity, not a full
    coefficient checksum. The generator additionally records each complete
    source-row checksum in the payload; fitting does not reread all source
    coefficients to recompute it at startup.
    """
    from common.parallel_transport import WFN_FINGERPRINT_SCHEME, fingerprint_from_binding, wfn_fingerprint
    from common.bispinor_init import NORMALIZED_RKB_LIFT_PROVENANCE

    k = np.asarray(k_parent_frac, np.float64)
    g = np.asarray(gvecs)
    ng = np.asarray(ngk_valid)
    centers, species = np.asarray(centers_cart, np.float64), np.asarray(atom_types)
    nb, volume = int(physical_bands), float(cell_volume)
    if (k.ndim != 2 or k.shape[1] != 3 or not np.isfinite(k).all()
            or g.ndim != 3 or g.shape[0] != len(k) or g.shape[2] != 3
            or not np.isfinite(g).all() or not np.all(g == np.round(g))
            or ng.shape != (len(k),) or not np.all(ng == np.round(ng))
            or np.any(ng <= 0) or np.any(ng > g.shape[1])
            or centers.shape != (len(species), 3) or len(species) == 0
            or not np.isfinite(centers).all() or not np.all(species == np.round(species))
            or np.any(species <= 0) or nb != physical_bands or nb != int(wfn.nbands)
            or nb < 1 or not np.isfinite(volume) or volume <= 0):
        raise ValueError('raw served-moment cache requires full physical WFN bands and authentic parent/atom geometry')
    hashes = {str(int(z)): str(sha) for z, sha in served_cache_sha256_by_species.items()}
    if (set(hashes) != {str(int(z)) for z in species}
            or any(len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha) for sha in hashes.values())):
        raise ValueError('raw served-moment species cache identities disagree with atoms')
    identity = _served_operator_identity()
    identity['owner_sources_sha256']['wfn_loader.loader'] = hashlib.sha256(
        Path(importlib.util.find_spec('wfn_loader.loader').origin).read_bytes()).hexdigest()
    identity['owner_sources_sha256']['common.parallel_transport'] = hashlib.sha256(
        Path(importlib.util.find_spec('common.parallel_transport').origin).read_bytes()).hexdigest()
    fingerprint = (wfn_fingerprint(wfn) if wfn_fingerprint_binding is None
                   else fingerprint_from_binding(wfn_fingerprint_binding, wfn))
    # Zero the G ghosts before hashing; their stored scratch values are inert.
    logical_g = np.asarray(g[:, :int(np.max(ng))], np.int64).copy()
    logical_g[np.arange(logical_g.shape[1])[None] >= ng[:, None]] = 0
    return dict(schema='lorrax.raw_parent_served_moments.v1', carrier='normalized_rkb',
        carrier_provenance=NORMALIZED_RKB_LIFT_PROVENANCE, band_range=[0, nb],
        wfn_fingerprint_scheme=WFN_FINGERPRINT_SCHEME, wfn_fingerprint=fingerprint,
        k_parent_frac=k.tolist(), ngk_valid=ng.astype(int).tolist(),
        gvecs_shape=list(logical_g.shape), gvecs_sha256=_payload_hash({'gvecs': logical_g}),
        centers_cart=centers.tolist(), atom_types=species.astype(int).tolist(), cell_volume=volume,
        served_cache_sha256_by_species=hashes, source_identity=identity,
        coefficient_convention='D_ni = <served_delta_i | normalized4_source_n>; unrotated full physical bands')


def _raw_parent_arrays(atom_D, raw_source_sha256, binding):
    nb = int(binding['band_range'][1])
    nparent = len(binding['k_parent_frac'])
    source_sha = np.asarray(raw_source_sha256)
    if source_sha.dtype != np.uint8 or source_sha.shape != (nparent, 32):
        raise ValueError('raw served-moment cache requires complete normalized source-row SHA256 bytes')
    if len(atom_D) != len(binding['atom_types']):
        raise ValueError('raw served-moment overlap tables disagree with atom count')
    arrays = dict(raw_source_sha256=source_sha)
    for atom, value in enumerate(atom_D):
        value = np.asarray(value)
        if (value.ndim != 3 or value.shape[:2] != (nparent, nb) or value.shape[2] == 0
                or value.dtype != np.complex128 or not np.isfinite(value).all()):
            raise ValueError('raw served-moment D_R must be finite complex128 on unpadded full physical bands')
        arrays[f'D_atom{atom}'] = value
    return arrays


def write_raw_parent_moments(path, atom_D, raw_source_sha256, *, binding):
    """Persist one-time source overlaps; interpolation points are absent."""
    target = Path(path)
    if target.exists():
        raise FileExistsError(f'preserve raw-parent served moments: {target}')
    arrays = _raw_parent_arrays(atom_D, raw_source_sha256, binding)
    metadata = dict(binding=binding, payload_sha256=_payload_hash(arrays))
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays, metadata_json=np.asarray(_json_bytes(metadata).decode()))
    return metadata


def load_raw_parent_moments(path, *, expected_binding, expected_file_sha256):
    """Strict artifact load with no on-demand source projection or rebuild."""
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != str(expected_file_sha256):
        raise ValueError('raw-parent served-moment file identity mismatch')
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata_json']))
        arrays = {key: archive[key].copy() for key in archive.files if key != 'metadata_json'}
    if metadata.get('binding') != expected_binding:
        raise ValueError('raw-parent served-moment source, geometry, or full-window identity mismatch')
    wanted = {'raw_source_sha256'} | {f'D_atom{atom}' for atom in range(len(expected_binding['atom_types']))}
    if set(arrays) != wanted:
        raise ValueError('raw-parent served-moment payload schema mismatch')
    atom_D = tuple(arrays[f'D_atom{atom}'] for atom in range(len(expected_binding['atom_types'])))
    _raw_parent_arrays(atom_D, arrays['raw_source_sha256'], expected_binding)
    if metadata.get('payload_sha256') != _payload_hash(arrays):
        raise ValueError('raw-parent served-moment payload identity mismatch')
    return dict(atom_D=atom_D, raw_source_sha256=arrays['raw_source_sha256'], metadata=metadata)

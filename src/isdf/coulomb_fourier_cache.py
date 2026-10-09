"""Authenticate prepared reciprocal tables for the existing local charge model.

The artifact preserves the incumbent CubicSpline coefficients exactly. It
changes preparation cost, never quadrature, compensation, or interpolation.
An explicit mismatch refuses; this module does not rebuild a missing cache.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np


SCHEMA = 'lorrax.local_coulomb_fourier_cache.v1'
_DIAGNOSTICS = ('points', 'maximum_wavevector', 'table_bytes', 'retained_spline_bytes',
                'max_density_validation_error', 'max_compensation_validation_error',
                'validation_points')
_FIELD_ARRAYS = ('degrees', 'quadrature_radius', 'quadrature_weights_dr',
                 'interpolation_map', 'origin_factors', 'moments',
                 'compensation_quadrature_shapes', 'compensation_self')


def _array_digest(arrays):
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(array.dtype.str.encode())
        digest.update(json.dumps(list(array.shape)).encode())
        digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _source_binding():
    import scipy
    from isdf.atomic_coulomb import _radial_fourier_cache, atomic_radial_metrics
    from isdf.augmentation import radial_coulomb_metric_interpolated

    owners = (_radial_fourier_cache, atomic_radial_metrics,
              radial_coulomb_metric_interpolated)
    return dict(scipy_version=scipy.__version__, owner_sources_sha256={
        function.__module__+'.'+function.__name__:
        hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
        for function in owners})


def _field_binding(tables):
    try:
        arrays = {key: np.asarray(tables[key]) for key in _FIELD_ARRAYS}
        controls = {key: int(tables[key]) for key in
                    ('origin_row_count', 'interpolation_degree', 'quadrature_order')}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('prepared Fourier cache requires the physical density interpolant') from exc
    return dict(arrays_sha256=_array_digest(arrays), controls=controls)


def build_coulomb_fourier_cache(tables, maximum_wavevector, fourier_points=4097):
    """Run the unchanged validated factory once for explicit preparation."""
    from isdf.atomic_coulomb import _radial_fourier_cache

    cache = _radial_fourier_cache(tables, maximum_wavevector, fourier_points)
    return dict(cache, source_binding=_source_binding(), field_binding=_field_binding(tables))


def _spline_arrays(cache):
    return dict(density_coefficients=np.asarray(cache['density'].c),
                density_knots=np.asarray(cache['density'].x),
                compensation_coefficients=np.asarray(cache['compensation'].c),
                compensation_knots=np.asarray(cache['compensation'].x))


def write_coulomb_fourier_cache(path, cache):
    """Write one immutable artifact; source identity must still match."""
    if cache.get('source_binding') != _source_binding():
        raise ValueError('prepared Fourier source identity changed before write')
    arrays = _spline_arrays(cache)
    metadata = dict(schema=SCHEMA, source_binding=cache['source_binding'],
        field_binding=cache['field_binding'], payload_sha256=_array_digest(arrays),
        density_axis=int(cache['density'].axis), compensation_axis=int(cache['compensation'].axis),
        density_extrapolate=bool(cache['density'].extrapolate),
        compensation_extrapolate=bool(cache['compensation'].extrapolate),
        diagnostics={key: cache[key] for key in _DIAGNOSTICS})
    with Path(path).open('xb') as stream:
        np.savez(stream, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)), **arrays)


def load_coulomb_fourier_cache(path, *, expected_file_sha256):
    """Read an explicitly pinned artifact before fitting; never regenerate."""
    from scipy.interpolate import CubicSpline

    path = Path(path)
    if (not isinstance(expected_file_sha256, str) or len(expected_file_sha256) != 64
            or _file_digest(path) != expected_file_sha256):
        raise ValueError('prepared Fourier file identity mismatch')
    with np.load(path, allow_pickle=False) as stream:
        names = {'density_coefficients', 'density_knots', 'compensation_coefficients',
                 'compensation_knots'}
        if set(stream.files) != names | {'metadata_json'}:
            raise ValueError('prepared Fourier artifact schema mismatch')
        metadata = json.loads(str(stream['metadata_json']))
        arrays = {name: np.array(stream[name]) for name in names}
    if (metadata.get('schema') != SCHEMA or metadata.get('source_binding') != _source_binding()
            or metadata.get('density_axis') != 1 or metadata.get('compensation_axis') != 1
            or metadata.get('density_extrapolate') is not True
            or metadata.get('compensation_extrapolate') is not True):
        raise ValueError('prepared Fourier source or spline identity mismatch')
    if metadata.get('payload_sha256') != _array_digest(arrays):
        raise ValueError('prepared Fourier payload identity mismatch')
    diagnostics = metadata['diagnostics']
    count = diagnostics['points']; maximum = diagnostics['maximum_wavevector']
    density, compensation = arrays['density_coefficients'], arrays['compensation_coefficients']
    x, gx = arrays['density_knots'], arrays['compensation_knots']
    if (not isinstance(count, int) or count < 4 or not np.isfinite(maximum) or maximum < 0
            or any(value.dtype != np.dtype(np.float64) or not np.all(np.isfinite(value))
                   for value in arrays.values())
            or x.shape != (count,) or not np.array_equal(x, gx)
            or not np.array_equal(x, np.linspace(0., maximum if maximum > 0 else 1., count))
            or density.ndim != 4 or density.shape[:2] != (4, count-1)
            or compensation.shape != density.shape[:3]
            or density.shape[2] == 0 or density.shape[3] == 0):
        raise ValueError('prepared Fourier spline dimensions or extent mismatch')
    for array in arrays.values():
        array.flags.writeable = False
    return dict(diagnostics,
        density=CubicSpline.construct_fast(density, x, extrapolate=True, axis=1),
        compensation=CubicSpline.construct_fast(compensation, gx, extrapolate=True, axis=1),
        source_binding=metadata['source_binding'], field_binding=metadata['field_binding'],
        prepared_file=str(path.resolve()), prepared_file_sha256=expected_file_sha256)


def validate_coulomb_fourier_cache(cache, tables, maximum_wavevector, fourier_points):
    """Authenticate the actual model/extent and repeat incumbent direct pins."""
    from scipy.special import spherical_jn

    if (cache.get('source_binding') != _source_binding()
            or cache.get('field_binding') != _field_binding(tables)):
        raise ValueError('prepared Fourier field or source identity mismatch')
    count = int(fourier_points); maximum = float(maximum_wavevector)
    if (count != fourier_points or count != cache['points'] or not np.isfinite(maximum)
            or maximum < 0 or maximum > cache['maximum_wavevector']):
        raise ValueError('prepared Fourier points or consuming extent mismatch')
    grid = cache['density'].x
    qr, qw = tables['quadrature_radius'], tables['quadrature_weights_dr']
    mapping, origin = tables['interpolation_map'], tables['origin_row_count']
    degrees, shapes = tables['degrees'], tables['compensation_quadrature_shapes']
    if cache['density'].c.shape[2:] != (len(degrees), mapping.shape[1]):
        raise ValueError('prepared Fourier coefficient dimensions mismatch')
    cells = np.unique(np.linspace(0, count-2, min(129, count-1)).astype(int))
    fraction = .5+.2*np.sin(np.arange(len(cells))*1.618033988749895)
    probes = grid[cells]+fraction*(grid[1]-grid[0])
    predicted, predicted_comp = cache['density'](probes), cache['compensation'](probes)
    values, comp = cache['density'](grid), cache['compensation'](grid)
    if any(not np.all(np.isfinite(value)) for value in (predicted, predicted_comp, values, comp)):
        raise ValueError('prepared Fourier interpolation contains nonfinite coefficients')
    scale, comp_scale = np.max(np.abs(values), axis=1), np.max(np.abs(comp), axis=1)
    errors, comp_errors = [], []
    for row, l in enumerate(degrees):
        weighted = spherical_jn(l, probes[:, None]*qr)*(qw*qr*qr)
        exact = weighted @ mapping
        exact[:, 0] += weighted[:, :origin] @ (tables['origin_factors'][row]-1.)
        error = np.max(np.abs(predicted[row]-exact), axis=0)
        comp_error = np.max(np.abs(predicted_comp[row]-weighted @ shapes[row]))
        if (np.any(error > 1e-12+1e-10*scale[row])
                or comp_error > 1e-12+1e-10*comp_scale[row]):
            raise ValueError('prepared Fourier interpolation failed direct-quadrature tolerance')
        target = tables['moments'][row] if l == 0 else np.zeros(mapping.shape[1])
        if (not np.allclose(values[row, 0], target, rtol=1e-10, atol=1e-12)
                or not np.isclose(comp[row, 0], 1. if l == 0 else 0., rtol=1e-10, atol=1e-12)):
            raise ValueError('prepared Fourier zero-momentum multipole pin failed')
        errors.append(float(error.max())); comp_errors.append(float(comp_error))
    return dict(cache, max_density_validation_error=max(errors),
                max_compensation_validation_error=max(comp_errors), validation_points=len(probes))


PERIODIC_SCHEMA = 'lorrax.periodic_compensation_metric.v1'
_PERIODIC_DATASET = 'periodic_compensation_gram_ry'
_PERIODIC_MODEL = dict(compensation_power=6,
    coulomb='bare_periodic_8pi_over_Omega_K2_Ry', gamma_zero='excluded',
    phase='exp_minus_i_K_dot_center',
    units='physical_Ry_unit_harmonic_multipoles')


def _periodic_geometry_binding(geometry):
    """Bind the ordered geometry; no orbital or centroid identity enters it."""
    names = ('reciprocal_rows_bohr_inverse', 'cell_volume_bohr3',
             'atom_centres_bohr', 'operator_q_fractional', 'support_radius_bohr')
    if not isinstance(geometry, dict) or set(geometry) != set(names):
        raise ValueError('periodic compensation geometry fields mismatch')
    bvec, volume, atoms, q, radius = (np.asarray(geometry[name], float) for name in names)
    if (bvec.shape != (3, 3) or volume.shape != () or radius.shape != ()
            or atoms.ndim != 2 or atoms.shape[1:] != (3,) or len(atoms) == 0
            or q.ndim != 2 or q.shape[1:] != (3,) or len(q) == 0
            or any(not np.isfinite(value).all() for value in (bvec, volume, atoms, q, radius))
            or volume <= 0 or radius <= 0 or abs(np.linalg.det(bvec)) == 0
            or not np.isclose(volume, (2*np.pi)**3/abs(np.linalg.det(bvec)), rtol=2e-13, atol=0)
            or len(np.unique(q, axis=0)) != len(q)):
        raise ValueError('invalid periodic compensation geometry')
    return {name: value.tolist() for name, value in zip(names, (bvec, volume, atoms, q, radius))}


def _periodic_harmonics(lm):
    """Require atom-major rows of the canonical complete complex Y_lm space."""
    channels = np.asarray(lm)
    if (channels.ndim != 2 or channels.shape[1:] != (2,) or len(channels) == 0
            or not np.isfinite(channels).all() or not np.equal(channels, np.round(channels)).all()):
        raise ValueError('invalid periodic compensation harmonic rows')
    channels = channels.astype(np.int64)
    wanted = np.asarray([(l, m) for l in range(int(channels[:, 0].max())+1)
                         for m in range(-l, l+1)], np.int64)
    if not np.array_equal(channels, wanted):
        raise ValueError('periodic compensation requires canonical complete harmonic order')
    return channels.tolist()


def _periodic_sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(char in '0123456789abcdef' for char in value))


def _periodic_preparation_binding(preparation, nq, *, authenticate_receipt=False):
    """Bind self-contained producer evidence; only preparation reads its receipt."""
    names = {'receipt_sha256', 'payload_sha256', 'producer_sources_sha256',
             'cutoffs', 'refinement_max'}
    if authenticate_receipt:
        names.add('receipt_path')
    if not isinstance(preparation, dict) or set(preparation) != names:
        raise ValueError('periodic compensation preparation fields mismatch')
    sources = preparation['producer_sources_sha256']
    if (not _periodic_sha(preparation['receipt_sha256'])
            or not _periodic_sha(preparation['payload_sha256'])
            or not isinstance(sources, dict) or not sources
            or any(not isinstance(name, str) or not name or not _periodic_sha(digest)
                   for name, digest in sources.items())
            or (authenticate_receipt and
                _file_digest(preparation['receipt_path']) != preparation['receipt_sha256'])):
        raise ValueError('periodic compensation preparation identity mismatch')
    cutoffs = np.asarray(preparation['cutoffs'], float)
    refinement = np.asarray(preparation['refinement_max'], float)
    if (cutoffs.ndim != 1 or len(cutoffs) < 2 or not np.isfinite(cutoffs).all()
            or np.any(cutoffs <= 0) or np.any(np.diff(cutoffs) <= 0)
            or refinement.shape != (nq, len(cutoffs)-1)
            or not np.isfinite(refinement).all() or np.any(refinement < 0)):
        raise ValueError('invalid periodic compensation finite-cutoff evidence')
    return dict(receipt_sha256=preparation['receipt_sha256'],
        payload_sha256=preparation['payload_sha256'], producer_sources_sha256=sources,
        cutoffs=cutoffs.tolist(), refinement_max=refinement.tolist())


def _periodic_face(gram, mesh, logical_shape):
    """Authenticate bulk ownership without converting the tensor to NumPy."""
    from jax.sharding import NamedSharding, PartitionSpec as P
    from runtime.padding import padded_axis, authenticate_axis
    face = NamedSharding(mesh, P(None, 'x', 'y'))
    if gram.dtype != np.dtype('complex128') or gram.ndim != 3 or gram.sharding != face:
        raise ValueError('periodic compensation Gram must be complex128 P(None,x,y)')
    if int(gram.shape[0]) != logical_shape[0]:
        raise ValueError('periodic compensation q domain mismatch')
    axis = padded_axis(logical_shape[-1], mesh, name='periodic moment',
        specs=((face.spec, 1), (face.spec, 2)))
    authenticate_axis(gram, axis, axis=1, where='periodic compensation rows')
    authenticate_axis(gram, axis, axis=2, where='periodic compensation columns')
    return face, axis


def write_periodic_compensation_cache(path, gram, *, mesh, geometry, lm, preparation):
    """Persist a prepared periodic multipole Gram using collective tile IO.

    Parameters
    ----------
    gram : complex128 JAX array, (nq, moment_carrier, moment_carrier)
        Physical Ry metric for unit harmonic multipoles, sharded P(None,x,y).
        Atom-major canonical harmonic axes have logical size Nat * (L+1)^2.
        SlabIO stores only those logical axes; padded values are not physical.
    geometry, preparation : dict
        Ordered cell/atoms/q/support and authenticated finite-cutoff evidence.
        This function does not construct a Fourier metric or certify its tail.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import agree_io_error
    from file_io.slab_io import SlabIO
    from runtime.padding import pad_square

    binding = _periodic_geometry_binding(geometry)
    harmonics = _periodic_harmonics(lm)
    nq = len(binding['operator_q_fractional'])
    n = len(binding['atom_centres_bohr'])*len(harmonics)
    evidence = _periodic_preparation_binding(preparation, nq, authenticate_receipt=True)
    face, axis = _periodic_face(gram, mesh, (nq, n, n))
    clean = jax.jit(lambda value: pad_square(value, axis),
        in_shardings=face, out_shardings=face)(gram)
    finite = jax.jit(lambda value: jnp.all(jnp.isfinite(value)),
        in_shardings=face, out_shardings=NamedSharding(mesh, P()))(clean)
    if not bool(np.asarray(finite.addressable_shards[0].data)):
        raise ValueError('nonfinite physical periodic compensation Gram')
    path = Path(path)
    agree_io_error(FileExistsError('immutable periodic compensation cache exists')
        if path.exists() else None, path=path, stage='periodic compensation fresh write')
    metadata = dict(schema=PERIODIC_SCHEMA, model=_PERIODIC_MODEL,
        geometry=binding, lm=harmonics, logical_shape=[nq, n, n],
        row_order='atom_major_canonical_complex_lm', preparation=evidence,
        source_binding=dict(physical_model=_PERIODIC_MODEL,
            producer_sources_sha256=evidence['producer_sources_sha256']),
        preparation_payload_scope='Producer asserts receipt/payload-to-input equality at preparation; loader authenticates persisted payload by the externally pinned whole-file digest.')
    encoded = json.dumps(metadata, sort_keys=True).encode('utf-8')
    if len(encoded) > 4*1024*1024:
        raise ValueError('periodic compensation metadata exceeds its bounded domain')
    with SlabIO(path, mode='w', mesh=mesh) as stream:
        stream.create_dataset(_PERIODIC_DATASET, shape=(nq, n, n), dtype=np.complex128)
        stream.write_slab(_PERIODIC_DATASET, clean)
        stream.write_attr('periodic_compensation_metadata_json',
                          np.asarray(encoded, dtype=f'S{len(encoded)}'))
    return dict(path=str(path.resolve()), file_sha256=_file_digest(path), metadata=metadata)


def load_periodic_compensation_cache(path, *, mesh, expected_file_sha256, geometry, lm):
    """Read a pinned geometry metric directly onto the two-dimensional face.

    Only bounded metadata is read through h5py. The metric is collectively
    loaded through SlabIO, retaining P(None,x,y) with exact inert carrier tails.
    The whole-file digest pins the full payload in addition to its upstream
    preparation receipt; no replicated global tensor or axis permutation occurs.
    """
    import h5py
    from jax.sharding import PartitionSpec as P
    from file_io.commit_state import assert_committed, agree_io_refusal, COMMIT_STATE
    from file_io.slab_io import SlabIO

    path = Path(path)
    error = None
    try:
        if not _periodic_sha(expected_file_sha256) or _file_digest(path) != expected_file_sha256:
            raise ValueError('periodic compensation file identity mismatch')
        binding, harmonics = _periodic_geometry_binding(geometry), _periodic_harmonics(lm)
        nq = len(binding['operator_q_fractional'])
        n = len(binding['atom_centres_bohr'])*len(harmonics)
        with h5py.File(path, 'r') as stream:
            assert_committed(stream, path=path)
            if COMMIT_STATE not in stream:
                raise ValueError('periodic compensation cache has no collective commit receipt')
            record = stream['periodic_compensation_metadata_json']
            if (record.shape != () or record.dtype.kind != 'S'
                    or record.dtype.itemsize < 1 or record.dtype.itemsize > 4*1024*1024):
                raise ValueError('periodic compensation metadata must be bounded fixed scalar bytes')
            text = record[()]
            if isinstance(text, bytes):
                text = text.decode('utf-8')
            if len(text) > 4*1024*1024:
                raise ValueError('periodic compensation metadata exceeds its bounded domain')
            metadata = json.loads(text)
            if (metadata.get('schema') != PERIODIC_SCHEMA or metadata.get('model') != _PERIODIC_MODEL
                    or metadata.get('geometry') != binding or metadata.get('lm') != harmonics
                    or metadata.get('logical_shape') != [nq, n, n]
                    or metadata.get('row_order') != 'atom_major_canonical_complex_lm'
                    or stream[_PERIODIC_DATASET].shape != (nq, n, n)
                    or stream[_PERIODIC_DATASET].dtype != np.dtype('complex128')):
                raise ValueError('periodic compensation model, geometry or axis identity mismatch')
            if _periodic_preparation_binding(metadata['preparation'], nq) != metadata['preparation']:
                raise ValueError('periodic compensation preparation binding mismatch')
            if metadata.get('source_binding') != dict(physical_model=_PERIODIC_MODEL,
                    producer_sources_sha256=metadata['preparation']['producer_sources_sha256']):
                raise ValueError('periodic compensation physical producer binding mismatch')
    except Exception as exc:
        error = exc
    agree_io_refusal(error, path=path, stage='periodic compensation metadata authentication')
    with SlabIO(path, mode='r', mesh=mesh) as stream:
        gram = stream.read_slab(_PERIODIC_DATASET, partition_spec=P(None, 'x', 'y'))
    _, axis = _periodic_face(gram, mesh, (nq, n, n))
    agree_io_refusal(ValueError('periodic compensation file changed during collective read')
        if _file_digest(path) != expected_file_sha256 else None,
        path=path, stage='periodic compensation post-read file authentication')
    return dict(gram=gram, moment_axis=axis, metadata=metadata,
        path=str(path.resolve()), file_sha256=expected_file_sha256)

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

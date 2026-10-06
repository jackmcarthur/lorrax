"""Prepared local Fourier tables preserve exact spline fields and refuse drift."""
from __future__ import annotations


def _tables():
    import numpy as np
    from isdf.atomic_coulomb import atomic_radial_metrics

    radius = np.linspace(.003, 1.6, 12)
    weights = np.diff(np.r_[0., (radius[1:]+radius[:-1])/2, 1.6])
    return atomic_radial_metrics(radius, weights, np.arange(3), support_radius=1.6,
        fft_points=80, cell_volume=107., interpolation_degree=5, quadrature_order=16)


def _write(tmp_path):
    from isdf.coulomb_fourier_cache import (build_coulomb_fourier_cache,
        write_coulomb_fourier_cache, load_coulomb_fourier_cache, _file_digest)

    tables = _tables()
    cache = build_coulomb_fourier_cache(tables, 5., 2049)
    path = tmp_path/'fourier.npz'
    write_coulomb_fourier_cache(path, cache)
    loaded = load_coulomb_fourier_cache(path, expected_file_sha256=_file_digest(path))
    return tables, cache, path, loaded


def test_exact_roundtrip_complete_field(tmp_path):
    import numpy as np
    from isdf.coulomb_fourier_cache import validate_coulomb_fourier_cache

    tables, cache, _, loaded = _write(tmp_path)
    for key in ('density', 'compensation'):
        np.testing.assert_array_equal(loaded[key].c, cache[key].c)
        np.testing.assert_array_equal(loaded[key].x, cache[key].x)
        assert loaded[key].axis == cache[key].axis
        assert not loaded[key].c.flags.writeable
    query = np.r_[0., np.linspace(.001, 4.999, 173), 5.]
    for key in ('density', 'compensation'):
        np.testing.assert_array_equal(loaded[key](query), cache[key](query))
    # A smaller consuming domain may use the prepared spline without extrapolation.
    verified = validate_coulomb_fourier_cache(loaded, tables, 4.8, 2049)
    assert verified['validation_points'] == 129
    coefficients = np.random.default_rng(610206).normal(size=(3, 12, 9))*(1+.31j)
    expected = np.einsum('lkr,lrf->lkf', cache['density'](query), coefficients)
    actual = np.einsum('lkr,lrf->lkf', verified['density'](query), coefficients)
    np.testing.assert_array_equal(actual, expected)


def test_extent_points_field_and_source_refuse(tmp_path):
    import numpy as np
    import pytest
    from isdf.coulomb_fourier_cache import validate_coulomb_fourier_cache

    tables, _, _, loaded = _write(tmp_path)
    with pytest.raises(ValueError, match='extent'):
        validate_coulomb_fourier_cache(loaded, tables, 5.0001, 2049)
    with pytest.raises(ValueError, match='points'):
        validate_coulomb_fourier_cache(loaded, tables, 5., 1025)
    wrong = dict(tables, origin_factors=tables['origin_factors'].copy())
    wrong['origin_factors'][1, 0] *= 1.0001
    with pytest.raises(ValueError, match='field'):
        validate_coulomb_fourier_cache(loaded, wrong, 5., 2049)
    with pytest.raises(ValueError, match='source'):
        validate_coulomb_fourier_cache(dict(loaded, source_binding={}), tables, 5., 2049)
    with pytest.raises(ValueError, match='physical density'):
        validate_coulomb_fourier_cache(loaded, {'degrees': np.arange(3)}, 5., 2049)


def test_immutable_file_and_payload_guards(tmp_path):
    import json
    import numpy as np
    import pytest
    from isdf.coulomb_fourier_cache import (write_coulomb_fourier_cache,
        load_coulomb_fourier_cache, _file_digest)

    _, cache, path, _ = _write(tmp_path)
    with pytest.raises(FileExistsError):
        write_coulomb_fourier_cache(path, cache)
    with pytest.raises(ValueError, match='file identity'):
        load_coulomb_fourier_cache(path, expected_file_sha256='0'*64)
    with np.load(path, allow_pickle=False) as stream:
        payload = {key: np.array(stream[key]) for key in stream.files}
    payload['density_coefficients'][0, 0, 0, 0] += .01
    damaged = tmp_path/'damaged.npz'
    np.savez(damaged, **payload)
    with pytest.raises(ValueError, match='payload'):
        load_coulomb_fourier_cache(damaged, expected_file_sha256=_file_digest(damaged))
    metadata = json.loads(str(payload['metadata_json']))
    metadata['source_binding'] = {}
    payload['metadata_json'] = np.asarray(json.dumps(metadata))
    source = tmp_path/'source.npz'
    np.savez(source, **payload)
    with pytest.raises(ValueError, match='source'):
        load_coulomb_fourier_cache(source, expected_file_sha256=_file_digest(source))


def test_repeated_direct_pins_detect_bad_preparation(tmp_path):
    import numpy as np
    import pytest
    from scipy.interpolate import CubicSpline
    from isdf.coulomb_fourier_cache import validate_coulomb_fourier_cache

    tables, _, _, loaded = _write(tmp_path)
    # Simulate a producer error independently of the reader's payload hash.
    # The normal direct quadrature still rejects the wrong represented field.
    coefficients = loaded['density'].c.copy()
    coefficients[3, :, 1, 3] += 1e-4
    wrong = dict(loaded, density=CubicSpline.construct_fast(coefficients,
        loaded['density'].x, extrapolate=True, axis=1))
    with pytest.raises(ValueError, match='direct-quadrature'):
        validate_coulomb_fourier_cache(wrong, tables, 5., 2049)
    coefficients[0, 71, 1, 3] = np.nan
    wrong = dict(loaded, density=CubicSpline.construct_fast(coefficients,
        loaded['density'].x, extrapolate=True, axis=1))
    with pytest.raises(ValueError, match='nonfinite'):
        validate_coulomb_fourier_cache(wrong, tables, 5., 2049)

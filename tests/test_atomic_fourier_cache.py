"""Persistence must preserve dual/delta distinction and reject stale artifacts."""
import json
import numpy as np
import pytest

from runtime import bootstrap
bootstrap()

from psp.atomic_fourier_cache import (build_atomic_fourier_caches,
    write_atomic_fourier_caches, load_atomic_fourier_caches)
from psp.augmented_samples import atomic_projection_table
from psp.reconstruction_overlap import atomic_delta_overlap_table


def fixture_data():
    radius = np.linspace(.001, 1.8, 161)
    weights = np.full(len(radius), radius[1] - radius[0])
    weights[[0, -1]] *= .5
    pseudo = np.column_stack((radius*np.exp(-radius), radius**2*np.exp(-1.3*radius)))
    delta = np.column_stack((.07*radius*(1-radius/1.8)**3,
                             -.03*radius**2*(1-radius/1.8)**3))
    data = dict(r=radius, weights_dr=weights, ps_u=pseudo, delta_u=delta,
                l=np.asarray((0, 1)), kappa=np.asarray((-1, 1)),
                metadata=dict(source_sha256="a"*64, payload_sha256="b"*64,
                              operator_comparison={"authenticated": True}, phase_branch_validated=True))
    controls = dict(momentum_max=5., momentum_points=513,
                    relative_tolerance=1e-9, absolute_tolerance=1e-11, validation_points=64)
    return data, controls


def test_roundtrip_preserves_independent_transforms_phase_and_carrier(tmp_path):
    data, controls = fixture_data()
    caches = build_atomic_fourier_caches(data, controls)
    path = tmp_path / "paired.npz"
    write_atomic_fourier_caches(path, caches, data, controls)
    restored = load_atomic_fourier_caches(path, data, controls, required_momentum_max=4.99)
    momenta = np.asarray(((0., 0., 0.), (.43, -.78, 1.13), (-2.04, .91, .34)))
    for kind, function in (("projection", atomic_projection_table), ("delta", atomic_delta_overlap_table)):
        for key in ("momentum", "radial", "ell", "kappa"):
            np.testing.assert_array_equal(restored[kind][key], caches[kind][key])
        options = dict(center_cart=(.29, -.47, .11), cell_volume=73., normalized_rkb_source=True)
        direct = function(data, momenta, **options)
        actual = function(data, momenta, radial_cache=restored[kind], **options)
        np.testing.assert_allclose(actual, direct, rtol=2e-9, atol=2e-11)
    assert not np.allclose(restored["projection"]["radial"], restored["delta"]["radial"])


def test_refuses_metadata_controls_source_and_momentum_changes(tmp_path):
    data, controls = fixture_data()
    path = tmp_path / "paired.npz"
    caches = build_atomic_fourier_caches(data, controls)
    write_atomic_fourier_caches(path, caches, data, controls)
    load_atomic_fourier_caches(path, data, dict(controls, species_files={"1": "unused"}))
    for changed_data, changed_control in (
            (dict(data, metadata=dict(data["metadata"], frozen_configuration="different")), controls),
            (data, dict(controls, momentum_max=5.01)),
            (data, dict(controls, relative_tolerance=1e-8))):
        with pytest.raises(ValueError, match="provenance mismatch"):
            load_atomic_fourier_caches(path, changed_data, changed_control)
    changed_data = dict(data, ps_u=data["ps_u"]*1.00001)
    with pytest.raises(ValueError, match="source identity"):
        load_atomic_fourier_caches(path, changed_data, controls)
    with pytest.raises(ValueError, match="cached envelope"):
        load_atomic_fourier_caches(path, data, controls, required_momentum_max=5.000001)
    with pytest.raises(ValueError, match="cached envelope"):
        load_atomic_fourier_caches(path, data, controls, required_momentum_max=np.nan)


def test_corruption_and_missing_explicit_file_never_build_replacements(tmp_path):
    data, controls = fixture_data()
    path = tmp_path / "paired.npz"
    write_atomic_fourier_caches(path, build_atomic_fourier_caches(data, controls), data, controls)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["delta_radial"] = arrays["delta_radial"] + .01
    poisoned = tmp_path / "poisoned.npz"
    np.savez(poisoned, **arrays)
    with pytest.raises(ValueError, match="payload checksum"):
        load_atomic_fourier_caches(poisoned, data, controls)
    metadata = json.loads(str(arrays["metadata_json"]))
    metadata["binding"]["representation"] = "normalized_four_spinor"
    arrays["metadata_json"] = np.asarray(json.dumps(metadata))
    np.savez(poisoned, **arrays)
    with pytest.raises(ValueError, match="metadata checksum"):
        load_atomic_fourier_caches(poisoned, data, controls)
    missing = tmp_path / "missing.npz"
    with pytest.raises(FileNotFoundError):
        load_atomic_fourier_caches(missing, data, controls)
    assert not missing.exists()


def test_writer_refuses_wrong_labels_grid_nonfinite_and_overwrite(tmp_path):
    data, controls = fixture_data()
    caches = build_atomic_fourier_caches(data, controls)
    for index, (name, value) in enumerate((
            ("momentum", caches["delta"]["momentum"]*1.001),
            ("ell", np.asarray((0, 2))),
            ("radial", caches["delta"]["radial"].astype(np.complex64)),
            ("radial", np.full_like(caches["delta"]["radial"], np.nan)))):
        changed = dict(caches, delta=dict(caches["delta"], **{name: value}))
        with pytest.raises(ValueError):
            write_atomic_fourier_caches(tmp_path / f"bad{index}.npz", changed, data, controls)
    path = tmp_path / "paired.npz"
    write_atomic_fourier_caches(path, caches, data, controls)
    with pytest.raises(FileExistsError):
        write_atomic_fourier_caches(path, caches, data, controls)

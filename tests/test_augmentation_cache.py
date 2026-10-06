"""Immutable artifact checks; normalized carrier mathematics has its own tests."""
import json
import numpy as np
import pytest

from runtime import bootstrap
bootstrap()

from psp.augmentation_cache import write_normalized_cache, load_normalized_cache
from psp.augmentation_spinors import COMPACT_GRAPH_FIELD_MODEL
from common.bispinor_init import HALFALPHA


def fixture_cache():
    control = dict(momentum_max=30., momentum_points=24, momentum_quadrature="gauss_legendre",
                   radius_kind="log", radius_min=1e-7, radius_max=1.8, radius_points=32,
                   tail_relative_tolerance=1., taper_start=1.)
    radius = np.concatenate(([0.], np.geomspace(1e-7, 1.8, 31)))
    data = dict(r=np.asarray((.1, .9)), l=np.asarray((0, 1)), kappa=np.asarray((-1, 1)),
                metadata=dict(source_sha256="a"*64, payload_sha256="b"*64,
                              operator_comparison={"authenticated": True}, phase_branch_validated=True))
    value = np.asarray(np.exp(-radius[:, None])*np.asarray((1., 0.7))[None], dtype=np.complex128)
    cache = dict(radius=radius, ell=data["l"], kappa=data["kappa"], large_R=value,
                 dlarge_R_dr=-value, small_R=0.1j*value, dsmall_R_dr=-0.1j*value,
                 field_model=np.asarray(COMPACT_GRAPH_FIELD_MODEL), taper_start=np.asarray(1.),
                 support_radius=np.asarray(1.2), half_alpha=np.asarray(float(HALFALPHA)))
    return data, control, cache


def test_cache_roundtrip_refuses_different_atomic_metadata_controls_and_support(tmp_path):
    data, control, cache = fixture_cache()
    path = tmp_path / "normalized.npz"
    write_normalized_cache(path, cache, data, control, support_radius=1.2)
    actual = load_normalized_cache(path, data, dict(control, species_files={"1": "elsewhere"}), support_radius=1.2)
    for key in cache:
        np.testing.assert_array_equal(actual[key], cache[key])
    changed = dict(data, metadata=dict(data["metadata"], source_sha256="c"*64))
    with pytest.raises(ValueError, match="provenance mismatch"):
        load_normalized_cache(path, changed, control, support_radius=1.2)
    changed = dict(data, metadata=dict(data["metadata"], extra_generator_information="changed"))
    with pytest.raises(ValueError, match="provenance mismatch"):
        load_normalized_cache(path, changed, control, support_radius=1.2)
    with pytest.raises(ValueError, match="provenance mismatch"):
        load_normalized_cache(path, data, dict(control, momentum_max=31.), support_radius=1.2)
    with pytest.raises(ValueError, match="provenance mismatch"):
        load_normalized_cache(path, data, control, support_radius=1.3)


def test_cache_refuses_payload_and_metadata_poison_without_rebuild(tmp_path):
    data, control, cache = fixture_cache()
    path = tmp_path / "normalized.npz"
    write_normalized_cache(path, cache, data, control, support_radius=1.2)
    with np.load(path, allow_pickle=False) as source:
        changed = {key: source[key] for key in source.files}
    changed["large_R"] = changed["large_R"] + 0.1
    poisoned = tmp_path / "poisoned.npz"
    np.savez(poisoned, **changed)
    with pytest.raises(ValueError, match="payload checksum"):
        load_normalized_cache(poisoned, data, control, support_radius=1.2)
    metadata = json.loads(str(changed["metadata_json"]))
    metadata["binding"]["carrier"] = "raw"
    changed["metadata_json"] = np.asarray(json.dumps(metadata))
    poisoned_metadata = tmp_path / "poisoned_metadata.npz"
    np.savez(poisoned_metadata, **changed)
    with pytest.raises(ValueError, match="metadata checksum"):
        load_normalized_cache(poisoned_metadata, data, control, support_radius=1.2)
    missing = tmp_path / "missing.npz"
    with pytest.raises(FileNotFoundError):
        load_normalized_cache(missing, data, control, support_radius=1.2)
    assert not missing.exists()


def test_cache_writer_rejects_wrong_grid_labels_dtype_and_nonfinite_values(tmp_path):
    data, control, cache = fixture_cache()
    changes = (("radius", cache["radius"]*1.001), ("ell", np.asarray((0, 2))),
               ("large_R", cache["large_R"].astype(np.complex64)),
               ("small_R", np.full_like(cache["small_R"], np.nan)))
    for index, (name, replacement) in enumerate(changes):
        with pytest.raises(ValueError):
            write_normalized_cache(tmp_path / f"wrong{index}.npz", dict(cache, **{name: replacement}),
                                   data, control, support_radius=1.2)


def test_cache_cannot_be_overwritten(tmp_path):
    data, control, cache = fixture_cache()
    path = tmp_path / "normalized.npz"
    write_normalized_cache(path, cache, data, control, support_radius=1.2)
    with pytest.raises(FileExistsError):
        write_normalized_cache(path, cache, data, control, support_radius=1.2)


def test_cache_requires_explicit_native_preserving_compact_descriptor(tmp_path):
    data, control, cache = fixture_cache()
    with pytest.raises(ValueError, match='explicit taper_start'):
        write_normalized_cache(tmp_path/'missing.npz', cache, data,
            {key: value for key, value in control.items() if key != 'taper_start'}, support_radius=1.2)
    with pytest.raises(ValueError, match='native reconstruction sphere'):
        write_normalized_cache(tmp_path/'core.npz', cache, data, dict(control, taper_start=.8), support_radius=1.2)
    for key, value in (('field_model', np.asarray('hard_mask')), ('half_alpha', np.asarray(2*HALFALPHA)),
                       ('taper_start', np.asarray(.7))):
        with pytest.raises(ValueError, match='descriptor'):
            write_normalized_cache(tmp_path/f'{key}.npz', dict(cache, **{key: value}), data, control, support_radius=1.2)

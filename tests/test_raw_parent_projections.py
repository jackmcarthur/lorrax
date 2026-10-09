"""Prepared pseudo C must preserve its physical source and projection identity."""
from types import SimpleNamespace
import hashlib
import json

import numpy as np
import pytest

from isdf.atomic_moments import (raw_parent_moment_binding,
    raw_parent_projection_binding, write_raw_parent_moments, load_raw_parent_moments)


def fixture(monkeypatch):
    import common.parallel_transport as owner
    monkeypatch.setattr(owner, 'wfn_fingerprint', lambda wfn: 'a'*64)
    wfn = SimpleNamespace(nbands=6, blat=1.4, bvec=np.eye(3))
    tables = {47: {'metadata': {'payload_sha256': 'b'*64, 'reference': 'Ag'}},
              53: {'metadata': {'payload_sha256': 'c'*64, 'reference': 'I'}}}
    projection = raw_parent_projection_binding(tables, {47: 'd'*64, 53: 'e'*64})
    geometry = dict(k_parent_frac=np.array([[0., 0., 0.], [.25, .25, .5]]),
        gvecs=np.array([[[0, 0, 0], [1, 0, 0]], [[0, 0, 0], [0, 1, 0]]]),
        ngk_valid=np.array([2, 2]), centers_cart=np.array([[0., 0., 0.], [.1, .2, .3]]),
        atom_types=np.array([47, 53]), cell_volume=8., physical_bands=6,
        served_cache_sha256_by_species={47: 'f'*64, 53: '1'*64})
    binding = raw_parent_moment_binding(wfn, **geometry, projection_binding=projection)
    rng = np.random.default_rng(78)
    def values(n):
        return rng.normal(size=(2, 6, n))+1j*rng.normal(size=(2, 6, n))
    C, D = (values(3), values(4)), (values(3), values(4))
    source = np.arange(64, dtype=np.uint8).reshape(2, 32)
    return wfn, tables, geometry, projection, binding, C, D, source


def test_roundtrip_complete_unrotated_C_and_D(monkeypatch, tmp_path):
    _, _, _, _, binding, C, D, source = fixture(monkeypatch)
    target = tmp_path/'raw.npz'
    write_raw_parent_moments(target, D, source, binding=binding, atom_C=C)
    loaded = load_raw_parent_moments(target, expected_binding=binding,
        expected_file_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    assert binding['schema'] == 'lorrax.raw_parent_served_moments.v2'
    for actual, expected in zip(loaded['atom_C']+loaded['atom_D'], C+D):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(loaded['raw_source_sha256'], source)


def test_wrong_duals_geometry_and_full_window_refuse(monkeypatch, tmp_path):
    wfn, tables, geometry, projection, binding, C, D, source = fixture(monkeypatch)
    target = tmp_path/'raw.npz'
    write_raw_parent_moments(target, D, source, binding=binding, atom_C=C)
    sha = hashlib.sha256(target.read_bytes()).hexdigest()
    changed_duals = raw_parent_projection_binding(tables, {47: '0'*64, 53: 'e'*64})
    wrong = raw_parent_moment_binding(wfn, **geometry, projection_binding=changed_duals)
    with pytest.raises(ValueError, match='source, geometry, or full-window'):
        load_raw_parent_moments(target, expected_binding=wrong, expected_file_sha256=sha)
    changed_lattice = SimpleNamespace(nbands=6, blat=1.5, bvec=np.eye(3))
    wrong = raw_parent_moment_binding(changed_lattice, **geometry, projection_binding=projection)
    with pytest.raises(ValueError, match='source, geometry, or full-window'):
        load_raw_parent_moments(target, expected_binding=wrong, expected_file_sha256=sha)
    with pytest.raises(ValueError, match='full physical WFN'):
        raw_parent_moment_binding(wfn, **dict(geometry, physical_bands=5),
                                  projection_binding=projection)
    with pytest.raises(ValueError, match='file identity'):
        load_raw_parent_moments(target, expected_binding=binding, expected_file_sha256='0'*64)


def test_v2_never_rebuilds_missing_or_padded_C(monkeypatch, tmp_path):
    _, _, _, _, binding, C, D, source = fixture(monkeypatch)
    with pytest.raises(ValueError, match='complete prepared C'):
        write_raw_parent_moments(tmp_path/'missing.npz', D, source, binding=binding)
    bad = (np.pad(C[0], ((0, 0), (0, 2), (0, 0))), C[1])
    with pytest.raises(ValueError, match='unpadded full-window'):
        write_raw_parent_moments(tmp_path/'padded.npz', D, source, binding=binding, atom_C=bad)
    target = tmp_path/'raw.npz'
    write_raw_parent_moments(target, D, source, binding=binding, atom_C=C)
    with np.load(target) as f:
        arrays = {key: f[key] for key in f.files if key != 'C_atom1'}
    poisoned = tmp_path/'partial.npz'
    np.savez_compressed(poisoned, **arrays)
    with pytest.raises(ValueError, match='payload schema'):
        load_raw_parent_moments(poisoned, expected_binding=binding,
            expected_file_sha256=hashlib.sha256(poisoned.read_bytes()).hexdigest())


def test_stale_projection_owner_and_payload_refuse(monkeypatch, tmp_path):
    _, _, _, _, binding, C, D, source = fixture(monkeypatch)
    target = tmp_path/'raw.npz'
    write_raw_parent_moments(target, D, source, binding=binding, atom_C=C)
    with np.load(target) as f:
        arrays = {key: f[key].copy() for key in f.files}
    metadata = json.loads(str(arrays['metadata_json']))
    metadata['binding']['projection_binding']['source_identity']['owner_sources_sha256'][
        'psp.augmented_samples'] = '0'*64
    arrays['metadata_json'] = np.asarray(json.dumps(metadata))
    poisoned = tmp_path/'owner.npz'
    np.savez_compressed(poisoned, **arrays)
    with pytest.raises(ValueError, match='owner identity'):
        load_raw_parent_moments(poisoned, expected_binding=metadata['binding'],
            expected_file_sha256=hashlib.sha256(poisoned.read_bytes()).hexdigest())
    with np.load(target) as f:
        arrays = {key: f[key].copy() for key in f.files}
    arrays['C_atom0'][0, 0, 0] += .01j
    poisoned = tmp_path/'payload.npz'
    np.savez_compressed(poisoned, **arrays)
    with pytest.raises(ValueError, match='payload identity'):
        load_raw_parent_moments(poisoned, expected_binding=binding,
            expected_file_sha256=hashlib.sha256(poisoned.read_bytes()).hexdigest())


def test_explicit_v1_retains_canonical_runtime_projection(monkeypatch, tmp_path):
    wfn, _, geometry, _, _, _, D, source = fixture(monkeypatch)
    binding = raw_parent_moment_binding(wfn, **geometry)
    target = tmp_path/'historical.npz'
    write_raw_parent_moments(target, D, source, binding=binding)
    got = load_raw_parent_moments(target, expected_binding=binding,
        expected_file_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    assert got['atom_C'] is None
    assert binding['schema'] == 'lorrax.raw_parent_served_moments.v1'

"""Centroid subgrid quadrature respects full-grid coordinates and IBZ stars."""
from types import SimpleNamespace

import numpy as np
import pytest

from centroid.sampling_metric import _quadrature_tables


def fixture(ibz):
    coords = np.indices((4, 4, 4)).reshape(3, -1).T
    coords = coords[np.random.default_rng(7).permutation(64)]
    parents = np.arange(64) // 2 if ibz else np.arange(64)
    wfn = SimpleNamespace(kgrid=np.array([4, 4, 4]), nkpts=32 if ibz else 64,
                          kweights=np.ones(32 if ibz else 64))
    sym = SimpleNamespace(nk_tot=64, kvecs_asints=coords, irr_idx_k=parents,
                          sym_idx_k=np.arange(64) % 2,
                          sym_mats_k=np.zeros((4, 3, 3)))
    return wfn, sym


@pytest.mark.parametrize('ibz', [False, True])
def test_coordinate_selection_and_star_weight_conservation(ibz):
    wfn, sym = fixture(ibz)
    parents, stars, weights = _quadrature_tables(wfn, sym, k_stride=2)
    selected = np.all(sym.kvecs_asints % 2 == 0, axis=1)
    np.testing.assert_array_equal(weights > 0, selected)
    np.testing.assert_allclose(weights[selected], np.full(8, 1/8))
    np.testing.assert_array_equal(parents, np.unique(sym.irr_idx_k[selected]))
    assert sum(float(w.sum()) for _, w in stars.values()) == pytest.approx(1)
    for parent, (rows, w) in stars.items():
        members = np.flatnonzero((sym.irr_idx_k == parent) & selected)
        np.testing.assert_array_equal(rows, sym.sym_idx_k[members])
        np.testing.assert_array_equal(w, weights[members])


@pytest.mark.parametrize('stride', [0, -1, 3])
def test_invalid_stride_refuses(stride):
    with pytest.raises(ValueError, match='stride'):
        _quadrature_tables(*fixture(False), k_stride=stride)


def test_stride_one_keeps_original_quadrature():
    _, _, weights = _quadrature_tables(*fixture(True), k_stride=1)
    np.testing.assert_array_equal(weights, np.full(64, 1/64))

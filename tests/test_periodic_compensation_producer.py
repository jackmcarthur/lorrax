"""Independent finite geometry and Fourier-unit checks for preparation."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


_path = Path(__file__).resolve().parents[1]/'tools/generate_periodic_compensation_cache.py'
_spec = importlib.util.spec_from_file_location('periodic_compensation_producer', _path)
producer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(producer)


@pytest.mark.parametrize('tile', [1, 7, 64])
def test_streamed_skew_reciprocal_sphere_has_exact_once_coverage(tile):
    B = np.array([[1.2, .1, 0.], [.1, .9, 0.], [0., 0., .4]])
    q, cutoff = np.array([.25, -.3, 0.]), 2.1
    # This small oversized box is an independent exhaustive reference;
    # production never materializes the full box or reciprocal sphere.
    rows = np.indices((17, 17, 17)).reshape(3, -1).T-8
    K = (rows+q)@B
    wanted = set(map(tuple, rows[np.sum(K*K, axis=1) <= cutoff*cutoff]))
    blocks = list(producer.reciprocal_tiles(B, q, cutoff, tile))
    actual = [tuple(row) for block, valid in blocks for row in block[:valid]]
    assert len(actual) == len(set(actual))
    assert set(actual) == wanted
    assert all(block.shape == (tile, 3) and 0 < valid <= tile for block, valid in blocks)
    assert np.all(blocks[-1][0][blocks[-1][1]:] == 0)


def test_compensation_fourier_units_reality_and_ghosts():
    lm = np.array([(l, m) for l in range(3) for m in range(-l, l+1)])
    centers = np.array([[.2, -.1, .3], [1., .5, -.2]])
    K = np.array([[0., 0., 0.], [.3, -.2, .4], [-.4, .1, .7]])
    rows = np.arange(len(centers)*len(lm)+2)
    F = producer.compensation_fourier(K, rows, centers=centers, lm=lm, support_radius=1.5)
    minus = producer.compensation_fourier(-K, rows, centers=centers, lm=lm, support_radius=1.5)
    assert np.all(F[-2:] == 0)
    for atom in range(len(centers)):
        for h, (l, m) in enumerate(lm):
            partner = int(np.flatnonzero(np.all(lm == [l, -m], axis=1))[0])
            np.testing.assert_allclose(minus[atom*len(lm)+h],
                (-1.)**int(m)*F[atom*len(lm)+partner].conj(), rtol=2e-14, atol=2e-14)
            np.testing.assert_allclose(F[atom*len(lm)+h, 0],
                np.sqrt(4*np.pi) if l == 0 else 0., rtol=2e-14, atol=2e-14)


def test_shard_interval_handles_replicated_axes_and_refuses_strides():
    assert producer._interval(slice(None), 7) == (0, 7)
    assert producer._interval(slice(2, 6), 7) == (2, 6)
    with pytest.raises(ValueError, match='contiguous'):
        producer._interval(slice(0, 7, 2), 7)

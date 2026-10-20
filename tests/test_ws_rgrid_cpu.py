"""Wigner–Seitz R images of the htransform off-grid Fourier sum (CPU).

``fh_interp.build_ws_R_images_np`` replaces the lattice-coordinate box of
``build_R_grid_np`` for every OFF-grid phase sum.  Checked here, on the FCC
(Si) and hexagonal (MoS2) metrics:

* the weighted image set is invariant under every lattice point operation and
  under R -> -R, and each class's weights sum to 1;
* on the coarse grid the Wigner–Seitz phase row equals the box phase row
  (on-grid results unchanged);
* a point-group-symmetric scalar band sampled on the coarse grid interpolates
  to a symmetric off-grid band through ``ws_fourier_phase``, while the box
  (the control variant) breaks the star degeneracy.
"""
import itertools

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")

from bandstructure.fh_interp import (  # noqa: E402
    build_R_grid_np, build_ws_R_images_np, ws_fourier_phase)

FCC = 10.26 * np.array([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
HEX = 5.97 * np.array([[1, 0, 0], [-.5, np.sqrt(3) / 2, 0], [0, 0, 6.0]])
CASES = [((4, 4, 4), FCC), ((8, 8, 8), FCC), ((9, 9, 1), HEX),
         ((12, 12, 1), HEX)]


def _point_ops(lat):
    """Integer lattice-coordinate matrices M (R -> M R) preserving the metric."""
    G = lat @ lat.T
    ops = []
    for m in itertools.product((-1, 0, 1), repeat=9):
        M = np.array(m).reshape(3, 3)
        if abs(round(np.linalg.det(M))) == 1 and np.allclose(M.T @ G @ M, G):
            ops.append(M)
    return ops


def _image_weights(kgrid, lat):
    R_tab, w_tab = build_ws_R_images_np(kgrid, lat)
    out = {}
    for r, d in zip(*np.nonzero(w_tab)):
        out[tuple(int(round(x)) for x in R_tab[r, d])] = w_tab[r, d]
    return R_tab, w_tab, out


@pytest.mark.parametrize("kgrid, lat", CASES)
def test_image_set_is_point_group_invariant(kgrid, lat):
    R_tab, w_tab, S = _image_weights(kgrid, lat)
    ops = _point_ops(lat)
    assert len(ops) == (48 if lat is FCC else 24)
    np.testing.assert_allclose(w_tab.sum(axis=1), 1.0, atol=1e-15)
    for M in ops:
        for R, w in S.items():
            assert S.get(tuple(M @ np.array(R))) == pytest.approx(w, abs=1e-15)
    for R, w in S.items():
        assert S.get(tuple(-np.array(R))) == pytest.approx(w, abs=1e-15)
    if kgrid[2] == 1:                       # no image ever leaves the plane
        assert np.all(R_tab[..., 2] == 0)


@pytest.mark.parametrize("kgrid, lat", CASES)
def test_on_grid_phase_equals_box(kgrid, lat):
    R_tab, w_tab = build_ws_R_images_np(kgrid, lat)
    box = build_R_grid_np(kgrid)
    k = box / np.asarray(kgrid)             # every coarse point (wrapped)
    ws = np.asarray(ws_fourier_phase(jnp.asarray(k), R_tab, w_tab))
    np.testing.assert_allclose(ws, np.exp(-2j * np.pi * k @ box.T),
                               atol=1e-12)


def _scalar_band(k_frac, lat):
    """eps(k) = sum over a long-range shell set of cos(2 pi k.R), symmetric."""
    R = np.array(list(itertools.product(range(-6, 7), repeat=3)), float)
    if lat is HEX:
        R = R[R[:, 2] == 0]
    d = np.linalg.norm(R @ lat, axis=1)
    a = np.linalg.norm(lat[0])
    R, d = R[d <= 4.5 * a], d[d <= 4.5 * a]        # a sphere: symmetric shells
    t = np.exp(-d / (0.9 * a))
    return np.cos(2 * np.pi * k_frac @ R.T) @ t


@pytest.mark.parametrize("kgrid, lat", [CASES[0], CASES[2]])
def test_offgrid_interpolant_is_symmetric_and_box_is_not(kgrid, lat):
    box = build_R_grid_np(kgrid)
    kc = box / np.asarray(kgrid)
    eR = np.exp(2j * np.pi * kc @ box.T).T @ _scalar_band(kc, lat) / len(kc)
    R_tab, w_tab = build_ws_R_images_np(kgrid, lat)
    rng = np.random.default_rng(7)
    k = rng.random((16, 3))
    if kgrid[2] == 1:
        k[:, 2] = 0
    spread_ws, spread_box = 0.0, 0.0
    ops = _point_ops(lat)
    for M in ops:
        kM = k @ np.linalg.inv(M)           # k -> k M^-1 keeps k.(M R)
        e_ws = lambda q: np.real(np.asarray(
            ws_fourier_phase(jnp.asarray(q), R_tab, w_tab)) @ eR)
        e_box = lambda q: np.real(np.exp(-2j * np.pi * q @ box.T) @ eR)
        spread_ws = max(spread_ws, np.abs(e_ws(kM) - e_ws(k)).max())
        spread_box = max(spread_box, np.abs(e_box(kM) - e_box(k)).max())
    assert spread_ws < 1e-12
    assert spread_box > 1e-4

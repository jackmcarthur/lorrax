"""Focused contracts for metric-aligned centroid sampling weights."""

from __future__ import annotations

import ast
import inspect
from types import MethodType

import jax.numpy as jnp
import numpy as np
import pytest

from common.bispinor_init import ALPHA_FS
from common.gamma_matrices import gammas
from centroid.sampling_metric import (
    build_feature_metric_diagonal,
    feature_metric_diagonal_from_psi_r,
)


def _direct_pairs(psi, mask_left, mask_right, scale, mode):
    """Tiny reference that deliberately materializes state-pair features."""
    left = psi[np.asarray(mask_left, dtype=bool)]
    right = psi[np.asarray(mask_right, dtype=bool)]
    vertices = ((np.eye(psi.shape[1]),) if mode == "charge" else
                tuple(np.asarray(gammas[mu]) / ALPHA_FS
                      for mu in (1, 2, 3)))
    out = np.zeros(psi.shape[-3:], dtype=np.float64)
    for vertex in vertices:
        features = np.einsum(
            "maxyz,ab,nbxyz->mnxyz", np.conj(left), vertex, right,
            optimize=True)
        out += np.sum(np.abs(features) ** 2, axis=(0, 1))
    return out * float(scale) ** 4


def _diagonal_only(psi):
    """Rejected gauge-dependent current candidate, retained as a red twin."""
    out = np.zeros(psi.shape[-3:], dtype=np.float64)
    for mu in (1, 2, 3):
        diag = np.einsum(
            "naxyz,ab,nbxyz->nxyz", np.conj(psi),
            np.asarray(gammas[mu]), psi, optimize=True)
        out += np.sum(np.abs(diag) ** 2, axis=0)
    return out


def test_transverse_projectors_match_pairs_and_are_gauge_invariant():
    rng = np.random.default_rng(20260830)
    psi = (rng.standard_normal((3, 4, 2, 1, 2))
           + 1j * rng.standard_normal((3, 4, 2, 1, 2)))
    raw = (rng.standard_normal((3, 3))
           + 1j * rng.standard_normal((3, 3)))
    unitary, _ = np.linalg.qr(raw)
    rotated = np.einsum("mn,naxyz->maxyz", unitary, psi, optimize=True)
    mask = np.ones(3)
    got = np.asarray(feature_metric_diagonal_from_psi_r(
        jnp.asarray(psi), mask, mask, 1.0, gamma_mode="transverse"))
    got_rot = np.asarray(feature_metric_diagonal_from_psi_r(
        jnp.asarray(rotated), mask, mask, 1.0, gamma_mode="transverse"))
    expected = _direct_pairs(psi, mask, mask, 1.0, "transverse")
    np.testing.assert_allclose(got, expected, rtol=3e-14, atol=3e-14)
    np.testing.assert_allclose(got_rot, expected, rtol=3e-14, atol=3e-14)
    rejected, rejected_rot = _diagonal_only(psi), _diagonal_only(rotated)
    assert np.max(np.abs(rejected - rejected_rot)) > 1e-3 * np.max(rejected)


def test_asymmetric_charge_projectors_match_pairs():
    rng = np.random.default_rng(19)
    psi = (rng.standard_normal((4, 2, 2, 1, 1))
           + 1j * rng.standard_normal((4, 2, 2, 1, 1)))
    left = np.asarray([1, 1, 0, 0], dtype=np.float64)
    right = np.asarray([0, 1, 1, 1], dtype=np.float64)
    got = np.asarray(feature_metric_diagonal_from_psi_r(
        jnp.asarray(psi), left, right, 1.7, gamma_mode="charge"))
    expected = _direct_pairs(psi, left, right, 1.7, "charge")
    np.testing.assert_allclose(got, expected, rtol=3e-14, atol=3e-14)


def test_one_k_scalar_equal_window_row_norm_is_the_band_density():
    rng = np.random.default_rng(31)
    psi = (rng.standard_normal((3, 1, 2, 2, 1))
           + 1j * rng.standard_normal((3, 1, 2, 2, 1)))
    mask = np.asarray([1, 0, 1], dtype=np.float64)
    scale = 0.8
    diagonal = np.asarray(feature_metric_diagonal_from_psi_r(
        jnp.asarray(psi), mask, mask, scale, gamma_mode="charge"))
    density = scale ** 2 * np.sum(
        np.abs(psi[np.asarray(mask, dtype=bool)]) ** 2, axis=(0, 1))
    np.testing.assert_allclose(
        np.sqrt(diagonal), density, rtol=3e-14, atol=3e-14)


def _fake_loader(psi_by_parent_band, kweights, *, raw_nspinor, bispinor):
    from wfn_loader import WfnLoader

    psi = np.asarray(psi_by_parent_band, dtype=np.complex128)
    loader = object.__new__(WfnLoader)
    loader.nspinor, loader.nkpts = int(raw_nspinor), int(psi.shape[0])
    loader.nbands = int(psi.shape[1])
    loader.fft_grid = tuple(int(v) for v in psi.shape[-3:])
    loader.cell_volume = float(np.prod(loader.fft_grid))
    loader.kweights = np.asarray(kweights, dtype=np.float64)
    loader.requests = []
    bispinor_expected = bool(bispinor)

    def box_index(self, *, k):
        return np.zeros((1,) + self.fft_grid, dtype=np.int32)

    def load_process_local(self, *, bands, k, bispinor):
        assert bispinor is bispinor_expected
        parent, (lo, hi) = int(k.rows[0]), tuple(map(int, bands))
        self.requests.append((parent, lo, hi))
        return jnp.asarray(psi[parent:parent + 1, lo:hi])

    loader.box_index = MethodType(box_index, loader)
    loader.load_process_local = MethodType(load_process_local, loader)
    return loader


class _Star:
    def __init__(self, parents, rows):
        self.irr_idx_k = np.asarray(parents, dtype=np.int32)
        self.sym_idx_k = np.asarray(rows, dtype=np.int32)
        self.nk_tot = len(parents)
        n = max(1, max(rows, default=0) + 1)
        self.sym_mats_k = np.stack((np.eye(3, dtype=np.int32),) * (2 * n))
        self.pullback_calls = []

    def fft_grid_pullback(self, rows, fft_grid, *, validate):
        assert validate
        self.pullback_calls.extend(map(int, rows))
        nr = int(np.prod(fft_grid))
        return np.broadcast_to(np.arange(nr), (len(rows), nr)).copy()


def _current_psi(nparent=2, nbands=3):
    psi = np.zeros((nparent, nbands, 4, 1, 1, 1), np.complex128)
    for p in range(nparent):
        for n in range(nbands):
            psi[p, n, :, 0, 0, 0] = (
                1 + p + 0.5 * n) * np.array([1, 0, 0, 1])
    return psi


def test_parent_stream_partitions_across_ranks_and_keeps_union(monkeypatch):
    import common.collectives as C
    import common.wfn_transforms as T

    psi = _current_psi()
    monkeypatch.setattr(T, "to_rbox", lambda values, *a, **k: values)
    monkeypatch.setattr(C, "process_rank_world", lambda: (0, 1))
    serial_loader = _fake_loader(
        psi, [0.4, 0.6], raw_nspinor=2, bispinor=True)
    serial = build_feature_metric_diagonal(
        serial_loader, _Star([0, 1], [0, 0]), (0, 2), (1, 3),
        gamma_mode="transverse", verbose=False)
    assert serial_loader.requests == [(0, 0, 3), (1, 0, 3)]

    monkeypatch.setattr(C, "psum_replicate", lambda x, mesh: np.asarray(x))
    partials, requests = [], []
    for rank in (0, 1):
        loader = _fake_loader(
            psi, [0.4, 0.6], raw_nspinor=2, bispinor=True)
        monkeypatch.setattr(
            C, "process_rank_world", lambda rank=rank: (rank, 2))
        partials.append(build_feature_metric_diagonal(
            loader, _Star([0, 1], [0, 0]), (0, 2), (1, 3),
            gamma_mode="transverse", dist_mesh=object(), verbose=False))
        requests.append(loader.requests)
    assert requests == [[(0, 0, 3)], [(1, 0, 3)]]
    np.testing.assert_allclose(
        partials[0] + partials[1], serial, rtol=3e-14, atol=3e-14)


def test_nonuniform_parent_weights_are_divided_over_unequal_stars(monkeypatch):
    import common.collectives as C
    import common.wfn_transforms as T

    psi = _current_psi(2, 1)
    loader = _fake_loader(
        psi, [0.7, 0.3], raw_nspinor=2, bispinor=True)
    sym = _Star([0, 1, 1, 1], [0, 0, 1, 2])
    monkeypatch.setattr(T, "to_rbox", lambda values, *a, **k: values)
    monkeypatch.setattr(C, "process_rank_world", lambda: (0, 1))
    got = build_feature_metric_diagonal(
        loader, sym, (0, 1), (0, 1), gamma_mode="transverse",
        verbose=False)
    p0 = _direct_pairs(psi[0], [1], [1], 1.0, "transverse")
    p1 = _direct_pairs(psi[1], [1], [1], 1.0, "transverse")
    np.testing.assert_allclose(got, 0.7 * p0 + 0.3 * p1,
                               rtol=3e-14, atol=3e-14)
    assert np.max(np.abs(got - (0.25 * p0 + 0.75 * p1))) > 1.0
    assert sym.pullback_calls == [0, 0, 1, 2]


def test_distributed_build_requires_mesh(monkeypatch):
    import common.collectives as C

    loader = _fake_loader(
        _current_psi(1, 1), [1.0], raw_nspinor=2, bispinor=True)
    monkeypatch.setattr(C, "process_rank_world", lambda: (0, 2))
    with pytest.raises(ValueError, match="requires dist_mesh at P>1"):
        build_feature_metric_diagonal(
            loader, _Star([0], [0]), (0, 1), (0, 1),
            gamma_mode="transverse", verbose=False)
    assert loader.requests == []


def test_builder_has_no_occupation_or_persistent_pullback_cache():
    tree = ast.parse(inspect.getsource(build_feature_metric_diagonal))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    forbidden = {"occ", "occs", "occupation", "occupations", "n_occ",
                 "pullback_cache"}
    assert names.isdisjoint(forbidden) and attrs.isdisjoint(forbidden)
    bad = ast.parse("n_occ=wfn.nelec\nx=wfn.occs\npullback_cache={}")
    bad_names = {n.id for n in ast.walk(bad) if isinstance(n, ast.Name)}
    bad_attrs = {n.attr for n in ast.walk(bad) if isinstance(n, ast.Attribute)}
    assert not bad_names.isdisjoint(forbidden)
    assert not bad_attrs.isdisjoint(forbidden)

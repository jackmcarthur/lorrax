"""Focused contracts for transverse-current centroid sampling weights."""

from __future__ import annotations

import ast
import inspect
from types import MethodType

import jax.numpy as jnp
import numpy as np
import pytest

from common.bispinor_init import ALPHA_FS
from common.gamma_matrices import gammas
from centroid.current_density import (
    build_current_density,
    transverse_current_sampling_weight_from_psi_r,
)


def _direct_pairs(psi, scale):
    """Tiny reference that deliberately materializes state-pair features."""
    out = np.zeros(psi.shape[-3:], dtype=np.float64)
    for mu in (1, 2, 3):
        mtx = np.einsum(
            "naxyz,ab,mbxyz->nmxyz", np.conj(psi),
            np.asarray(gammas[mu]), psi, optimize=True)
        out += np.sum(np.abs(mtx) ** 2, axis=(0, 1))
    return out * (float(scale) ** 2 / ALPHA_FS) ** 2


def _diagonal_only(psi):
    """Rejected gauge-dependent candidate, retained as the red twin."""
    out = np.zeros(psi.shape[-3:], dtype=np.float64)
    for mu in (1, 2, 3):
        diag = np.einsum(
            "naxyz,ab,nbxyz->nxyz", np.conj(psi),
            np.asarray(gammas[mu]), psi, optimize=True)
        out += np.sum(np.abs(diag) ** 2, axis=0)
    return out


def test_density_matrix_matches_pairs_and_is_subspace_gauge_invariant():
    rng = np.random.default_rng(20260830)
    psi = (rng.standard_normal((3, 4, 2, 1, 2))
           + 1j * rng.standard_normal((3, 4, 2, 1, 2)))
    raw = (rng.standard_normal((3, 3))
           + 1j * rng.standard_normal((3, 3)))
    unitary, _ = np.linalg.qr(raw)
    rotated = np.einsum("mn,naxyz->maxyz", unitary, psi, optimize=True)
    scale = np.sqrt(ALPHA_FS)
    got = np.asarray(transverse_current_sampling_weight_from_psi_r(
        jnp.asarray(psi), np.ones(3), scale))
    got_rot = np.asarray(transverse_current_sampling_weight_from_psi_r(
        jnp.asarray(rotated), np.ones(3), scale))
    expected = _direct_pairs(psi, scale)
    np.testing.assert_allclose(got, expected, rtol=3e-14, atol=3e-14)
    np.testing.assert_allclose(got_rot, expected, rtol=3e-14, atol=3e-14)
    # Red twin: the retired diagonal-only expression changes under this U.
    rejected, rejected_rot = _diagonal_only(psi), _diagonal_only(rotated)
    assert np.max(np.abs(rejected - rejected_rot)) > 1e-3 * np.max(rejected)


def _fake_loader(psi_by_parent_band, kweights):
    from wfn_loader import WfnLoader

    psi = np.asarray(psi_by_parent_band, dtype=np.complex128)
    loader = object.__new__(WfnLoader)
    loader.nspinor, loader.nkpts = 2, int(psi.shape[0])
    loader.nbands = int(psi.shape[1])
    loader.fft_grid = tuple(int(v) for v in psi.shape[-3:])
    loader.cell_volume = float(np.prod(loader.fft_grid))
    loader.kweights = np.asarray(kweights, dtype=np.float64)
    loader.requests = []

    def box_index(self, *, k):
        return np.zeros((1,) + self.fft_grid, dtype=np.int32)

    def load_process_local(self, *, bands, k, bispinor):
        assert bispinor
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


def _psi(nparent=2, nbands=3):
    psi = np.zeros((nparent, nbands, 4, 1, 1, 1), np.complex128)
    for p in range(nparent):
        for n in range(nbands):
            psi[p, n, :, 0, 0, 0] = (
                1 + p + 0.5 * n) * np.array([1, 0, 0, 1])
    return psi


def test_parent_stream_partitions_across_ranks_and_keeps_all_bands(monkeypatch):
    import common.collectives as C
    import common.wfn_transforms as T

    psi = _psi()
    monkeypatch.setattr(T, "to_rbox", lambda values, *a, **k: values)
    monkeypatch.setattr(C, "process_rank_world", lambda: (0, 1))
    serial_loader = _fake_loader(psi, [0.4, 0.6])
    serial = build_current_density(
        serial_loader, _Star([0, 1], [0, 0]), (0, 3), verbose=False)
    assert serial_loader.requests == [(0, 0, 3), (1, 0, 3)]

    monkeypatch.setattr(C, "psum_replicate", lambda x, mesh: np.asarray(x))
    partials, requests = [], []
    for rank in (0, 1):
        loader = _fake_loader(psi, [0.4, 0.6])
        monkeypatch.setattr(
            C, "process_rank_world", lambda rank=rank: (rank, 2))
        partials.append(build_current_density(
            loader, _Star([0, 1], [0, 0]), (0, 3),
            dist_mesh=object(), verbose=False))
        requests.append(loader.requests)
    assert requests == [[(0, 0, 3)], [(1, 0, 3)]]
    np.testing.assert_allclose(
        partials[0] + partials[1], serial, rtol=3e-14, atol=3e-14)


def test_nonuniform_parent_weights_are_divided_over_unequal_stars(monkeypatch):
    import common.collectives as C
    import common.wfn_transforms as T

    psi = _psi(2, 1)
    loader = _fake_loader(psi, [0.7, 0.3])
    sym = _Star([0, 1, 1, 1], [0, 0, 1, 2])
    monkeypatch.setattr(T, "to_rbox", lambda values, *a, **k: values)
    monkeypatch.setattr(C, "process_rank_world", lambda: (0, 1))
    got = build_current_density(loader, sym, (0, 1), verbose=False)
    p0 = np.asarray(transverse_current_sampling_weight_from_psi_r(
        jnp.asarray(psi[0]), np.ones(1), 1.0))
    p1 = np.asarray(transverse_current_sampling_weight_from_psi_r(
        jnp.asarray(psi[1]), np.ones(1), 1.0))
    np.testing.assert_allclose(got, 0.7 * p0 + 0.3 * p1,
                               rtol=3e-14, atol=3e-14)
    assert np.max(np.abs(got - (0.25 * p0 + 0.75 * p1))) > 1.0
    assert sym.pullback_calls == [0, 0, 1, 2]


def test_distributed_build_requires_mesh(monkeypatch):
    import common.collectives as C

    loader = _fake_loader(_psi(1, 1), [1.0])
    monkeypatch.setattr(C, "process_rank_world", lambda: (0, 2))
    with pytest.raises(ValueError, match="requires dist_mesh at P>1"):
        build_current_density(loader, _Star([0], [0]), (0, 1), verbose=False)
    assert loader.requests == []


def test_builder_has_no_occupation_or_persistent_pullback_cache():
    tree = ast.parse(inspect.getsource(build_current_density))
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

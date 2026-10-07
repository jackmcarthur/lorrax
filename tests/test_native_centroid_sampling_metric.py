"""Independent finite-PW/native-window centroid metric and input controls."""
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding

from common.wfn_layout import band_sphere_spec
from centroid import sampling_metric as metric
from centroid.kmeans_cli import _validate_native_selection_request


@pytest.fixture(scope="module")
def mesh():
    assert jax.process_count() == 1 and len(jax.devices()) == 4
    return Mesh(np.asarray(jax.devices()).reshape(2, 2), ("x", "y"))


def plant(mesh, monkeypatch, ns=1, ragged=True):
    import wfn_loader
    ng = np.asarray([5, 3 if ragged else 5], np.int64)
    counts = ns * ng
    box = np.asarray([[0, 16, 48, 4, 12]] * 2, np.int32)
    coefficient = np.zeros((2, int(counts.max()), ns, 5), np.complex128)
    rng = np.random.default_rng(31)
    for k, n in enumerate(counts):
        a = rng.normal(size=(n, n)) + 1j*rng.normal(size=(n, n))
        unitary = np.linalg.qr(a)[0]
        coefficient[k, :n, :, :ng[k]] = unitary.reshape(n, ns, ng[k])

    class Wfn:
        nbands = int(counts.max())
        nkpts = 2
        nspinor = ns
        ngkmax = 5
        fft_grid = (4, 4, 4)
        cell_volume = 37.

        def box_index(self, *, k):
            return box[np.asarray(k.rows)]

        def ngk_valid(self, *, k):
            return ng[np.asarray(k.rows)]

        def load(self, *, bands, k, sharding, bispinor):
            assert not bispinor and sharding == band_sphere_spec()
            lo, hi = bands
            return read(band_range=bands, pad_to=4*((hi-lo+3)//4), parent_ids=k.rows)[0]

    def read(*, band_range, pad_to, parent_ids):
        lo, hi = band_range
        p = np.asarray(parent_ids)
        ids = lo + np.arange(pad_to)
        valid = (ids[None] < counts[p, None]) & (ids[None] < hi)
        out = np.zeros((len(p), pad_to, ns, 5), np.complex128)
        for slot, parent in enumerate(p):
            for b, i in enumerate(ids):
                if valid[slot, b]:
                    out[slot, b] = coefficient[parent, i]
        return jax.device_put(out, NamedSharding(mesh, band_sphere_spec())), valid

    class Sym:
        def fft_grid_pullback(self, rows, fft_grid, validate):
            assert validate and fft_grid == (4, 4, 4)
            ids = np.arange(64, dtype=np.int32)
            return np.stack([ids if row == 0 else np.roll(ids, 3) for row in rows])

    stars = {0: (np.asarray([0, 1]), np.asarray([.2, .05])),
             1: (np.asarray([0, 1]), np.asarray([.3, .45]))}
    monkeypatch.setattr(wfn_loader, "WfnLoader", Wfn)
    monkeypatch.setattr(metric, "_quadrature_tables",
        lambda w, s: (np.arange(2), stars, np.asarray([.25, .75])))
    monkeypatch.setattr(metric, "_metric_chunk_plan", lambda **kw: (1, 3, 2 << 30))
    return Wfn(), Sym(), read, counts, coefficient, stars


def literal(coefficient, stars, left, right):
    points = np.stack(np.meshgrid(*(np.arange(4) for _ in range(3)), indexing="ij"), -1).reshape(-1, 3)
    g = np.asarray([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
    phase = np.exp(2j*np.pi*(g @ points.T)/4.)/np.sqrt(37.)
    psi = np.einsum("knsg,gr->knsr", coefficient, phase)
    out = np.zeros(64)
    for k, (rows, weights) in stars.items():
        a = psi[k, slice(*left)]
        b = psi[k, slice(*right)]
        da = np.einsum("nsr,ntr->str", a, a.conj())
        db = np.einsum("nsr,ntr->str", b, b.conj())
        field = np.einsum("str,tsr->r", da, db).real
        for row, weight in zip(rows, weights):
            ids = np.arange(64) if row == 0 else np.roll(np.arange(64), 3)
            out += weight * field[ids]
    return out.reshape(4, 4, 4)


@pytest.mark.parametrize("ns", [1, 2])
def test_complete_ragged_native_metric_uniform_and_spin_trace(mesh, monkeypatch, ns):
    w, s, read, counts, coefficients, stars = plant(mesh, monkeypatch, ns)
    w.nbands = 1  # actual stored prefix differs; never relabel it for the native call
    window = (0, int(counts.max()))
    got = metric.build_feature_metric_diagonal(w, s, window, window,
        gamma_mode="charge", dist_mesh=mesh, verbose=False,
        band_reader=read, native_parent_band_counts=counts)
    expected = ns*(.25*5**2 + .75*3**2)/37.**2
    np.testing.assert_allclose(got, expected, rtol=5e-13, atol=2e-14)
    np.testing.assert_allclose(got, literal(coefficients, stars, window, window), rtol=5e-13, atol=2e-14)
    assert w.nbands == 1
    # Missing native tails and squaring the spin trace are active wrong oracles.
    assert np.max(abs(got-literal(coefficients, stars, (0, 1), (0, 1)))) > 1e-3
    if ns == 2:
        assert np.max(abs(got-2*expected)) > 1e-3


def test_distinct_windows_and_typed_scalar_pullbacks(mesh, monkeypatch):
    w, s, read, counts, coefficients, stars = plant(mesh, monkeypatch, 2)
    left, right = (1, 5), (2, 9)
    got = metric.build_feature_metric_diagonal(w, s, left, right,
        gamma_mode="charge", dist_mesh=mesh, verbose=False,
        band_reader=read, native_parent_band_counts=counts)
    np.testing.assert_allclose(got, literal(coefficients, stars, left, right), rtol=5e-13, atol=2e-14)
    assert np.max(abs(got-literal(coefficients, stars, left, left))) > 1e-3


@pytest.mark.parametrize("ns", [1, 2])
def test_default_wfn_metric_and_native_same_input_bitwise(mesh, monkeypatch, ns):
    w, s, read, counts, _, _ = plant(mesh, monkeypatch, ns, ragged=False)
    window = (0, int(counts.max()))
    old = metric.build_feature_metric_diagonal(w, s, window, window,
        gamma_mode="charge", dist_mesh=mesh, verbose=False)
    new = metric.build_feature_metric_diagonal(w, s, window, window,
        gamma_mode="charge", dist_mesh=mesh, verbose=False,
        band_reader=read, native_parent_band_counts=counts)
    np.testing.assert_array_equal(new, old)


@pytest.mark.parametrize("poison", ["native_tail", "G_tail", "nan", "mask"])
def test_public_metric_reuses_native_input_refusals(mesh, monkeypatch, poison):
    w, s, clean, counts, _, _ = plant(mesh, monkeypatch)
    def read(**kw):
        value, valid = clean(**kw)
        if kw["parent_ids"] == (1,) and kw["band_range"] == (0, 3):
            a = np.asarray(value).copy()
            if poison == "G_tail":a[0, 0, 0, 4] = 1e-200
            elif poison == "native_tail":a[0, 3, 0, 0] = 1e-200
            elif poison == "nan":a[0, 0, 0, 0] = np.nan
            else:valid = ~valid
            value = jax.device_put(a, value.sharding)
        return value, valid
    with pytest.raises(ValueError, match="native_parent_reader"):
        metric.build_feature_metric_diagonal(w, s, (0, 5), (0, 5),
            gamma_mode="charge", dist_mesh=mesh, verbose=False,
            band_reader=read, native_parent_band_counts=counts)


@pytest.mark.parametrize("window", [(False, 5), (0., 5), (0, 6)])
def test_native_window_identity_before_loading(mesh, monkeypatch, window):
    w, s, read, counts, _, _ = plant(mesh, monkeypatch)
    with pytest.raises(ValueError, match="integer|subset"):
        metric.build_feature_metric_diagonal(w, s, window, (0, 5),
            gamma_mode="charge", dist_mesh=mesh, verbose=False,
            band_reader=read, native_parent_band_counts=counts)


@pytest.mark.parametrize("change", ["prune", "current", "prefix", "provenance", "sha"])
def test_reference_cli_preflight_refuses_unshared_pruning_and_unbound_input(change):
    args = SimpleNamespace(fit_window="0:5,0:5", oversample=1., density_mode="scalar")
    proof = dict(complete_native_basis=True, archive_sha256="1"*64, seed_wfn_sha256="2"*64)
    if change == "prune":args.oversample = 1.5
    elif change == "current":args.density_mode = "current"
    elif change == "prefix":args.fit_window = None
    elif change == "provenance":proof["complete_native_basis"] = False
    else:proof["archive_sha256"] = "stale"
    with pytest.raises(ValueError, match="native centroid"):
        _validate_native_selection_request(args, lambda **kw: None, [5], proof)


def test_default_and_explicit_native_reference_preflight():
    args = SimpleNamespace(fit_window="0:5,0:5", oversample=1., density_mode="scalar")
    _validate_native_selection_request(args, None, None, None)
    _validate_native_selection_request(args, lambda **kw: None, [5],
        dict(complete_native_basis=True, archive_sha256="1"*64, seed_wfn_sha256="2"*64))
    with pytest.raises(ValueError, match="requires the optional reader"):
        _validate_native_selection_request(args, None, [5], None)

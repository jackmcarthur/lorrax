"""The distributed downfold plan on a CPU mesh, each check with its red twin.

Run with ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4``.

Every kernel is built inside ``ffi.fft.xla_reference()``: these cells check
the Gram's column stream, its +q labelling, the q stream of D3-D5, the pool
spectrum and the deck's ``plan`` key, not the FFT vendor route, which
``tests/test_kconv_xla_gate.py`` owns.  The tiles are forced through
``runtime.tiles.TILE_BYTES``, the one bound every streamed loop sizes from.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update("jax_enable_x64", True)

QUIET = dict(print_fn=lambda *a, **k: None)


@pytest.fixture
def mesh():
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 host devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))


def _psi(rng, nk, nb, ns, mu):
    """Plain ψ at centroids, ``(nk, nb, ns, μ)`` — the restart's ``psi_rmu_Y`` order."""
    return rng.normal(size=(nk, nb, ns, mu)) + 1j * rng.normal(size=(nk, nb, ns, mu))


def _faces(psi, mesh):
    """The restart loader's two faces: conj ψ at ``P(None,'x',None,None)``, ψ at ``P(None,None,None,'y')``."""
    X = jax.device_put(np.conj(psi).transpose(0, 3, 1, 2),
                       NamedSharding(mesh, P(None, "x", None, None)))
    Y = jax.device_put(psi, NamedSharding(mesh, P(None, None, None, "y")))
    return X, Y


def _hpsd(rng, nq, mu, mesh):
    a = rng.normal(size=(nq, mu, 2 * mu)) + 1j * rng.normal(size=(nq, mu, 2 * mu))
    return jax.device_put(np.einsum("qik,qjk->qij", a, a.conj()),
                          NamedSharding(mesh, P(None, "x", "y")))


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def _dense_gram(psi, left, right, kgrid):
    """``S[q][μ,ν] = Σ_{k, m∈left, n∈right} X[(m,n),μ] conj(X[(m,n),ν])``,
    ``X[(m,n),μ] = Σ_s conj(ψ_ms(k, r_μ)) ψ_ns(k+q, r_μ)`` — the definition in
    ``gw.downfold.BandWindow``, labelled by +q, as a double loop."""
    nk = int(np.prod(kgrid))
    (l0, l1), (r0, r1) = left, right
    mu = psi.shape[-1]
    S = np.zeros((nk, mu, mu), complex)
    for q in range(nk):
        qv = np.unravel_index(q, kgrid)
        for k in range(nk):
            kv = np.unravel_index(k, kgrid)
            kq = np.ravel_multi_index(tuple((kv[i] + qv[i]) % kgrid[i] for i in range(3)), kgrid)
            X = np.einsum("msu,nsu->mnu", np.conj(psi[k, l0:l1]), psi[kq, r0:r1]).reshape(-1, mu)
            S[q] += X.T @ np.conj(X)
    return S


def test_gram_column_stream_is_the_unstreamed_kernel(mesh, monkeypatch):
    """Streaming the Gram's local columns (clamped last tile) changes nothing."""
    from ffi.fft import xla_reference
    from isdf.core import c_q_downfold, downfold_gram_col_tile
    rng = np.random.default_rng(1)
    kgrid, nk, nb, ns, mu, col = (2, 2, 1), 4, 3, 2, 8, 14      # local face (4, 7)
    Xl, _ = _faces(_psi(rng, nk, nb, ns, mu), mesh)
    _, Yc = _faces(_psi(rng, nk, nb, ns, col), mesh)

    def gram():
        with xla_reference():
            return np.asarray(c_q_downfold(Xl, Yc, Xl, Yc, kgrid=kgrid, mesh_xy=mesh))

    assert downfold_gram_col_tile(nk, ns, 4, 7) == (7, (0,))
    one = gram()
    per_col = (3 * ns * ns + 2) * nk * 4 * 16
    monkeypatch.setattr("runtime.tiles.TILE_BYTES", 3 * per_col)
    assert downfold_gram_col_tile(nk, ns, 4, 7) == (3, (0, 3, 4))   # overlap at columns 4, 5
    assert _rel(gram(), one) <= 1e-14

    # RED TWIN: a stream that never writes the last tile leaves column 6 at zero.
    monkeypatch.setattr("isdf.core.downfold_gram_col_tile", lambda *a: (3, (0, 3)))
    assert _rel(gram(), one) > 1e-3


def test_gram_is_labelled_by_plus_q_on_an_asymmetric_window(mesh, monkeypatch):
    """The streamed Gram equals the closed form at +q; the raw kernel is S(-q)."""
    from ffi.fft import xla_reference
    from gw.downfold import BandWindow, pair_density_gram
    from isdf.core import c_q_downfold
    rng = np.random.default_rng(2)
    kgrid, nk, nb, ns, mu = (3, 2, 1), 6, 4, 2, 8          # -q != q on the 3-axis
    psi = _psi(rng, nk, nb, ns, mu)
    X, Y = _faces(psi, mesh)
    win = BandWindow(left=(0, 2), right=(1, 4))
    monkeypatch.setattr("runtime.tiles.TILE_BYTES", 1)          # one column per tile
    ref = _dense_gram(psi, win.left, win.right, kgrid)
    with xla_reference():
        S = pair_density_gram(X, Y, win, kgrid=kgrid, mesh_xy=mesh)
        raw = c_q_downfold(X[:, :, 1:4, :], Y[:, 1:4], X[:, :, 0:2, :], Y[:, 0:2],
                           kgrid=kgrid, mesh_xy=mesh)
    assert _rel(S, ref) <= 1e-12
    # RED TWIN: the kernel's own labelling is S(-q), which only an asymmetric
    # window on a grid with -q != q can tell apart from S(+q).
    assert _rel(raw, ref) > 1e-3


def test_gram_column_stream_bounds_the_compiled_module(mesh, monkeypatch):
    """The compiled Gram's scratch scales with the column tile, not with the face."""
    from ffi.fft import xla_reference
    from isdf.core import c_q_downfold
    rng = np.random.default_rng(5)
    kgrid, nk, nb, ns, mu, col = (3, 2, 1), 6, 3, 2, 16, 64      # local face (8, 32)
    Xl, _ = _faces(_psi(rng, nk, nb, ns, mu), mesh)
    _, Yc = _faces(_psi(rng, nk, nb, ns, col), mesh)

    def temp_bytes():
        with xla_reference():
            f = jax.jit(lambda a, b: c_q_downfold(a, b, a, b, kgrid=kgrid, mesh_xy=mesh))
            return int(f.lower(Xl, Yc).compile().memory_analysis().temp_size_in_bytes)

    monkeypatch.setattr("runtime.tiles.TILE_BYTES", 1 << 40)
    whole = temp_bytes()
    monkeypatch.setattr("runtime.tiles.TILE_BYTES", 4 * (3 * ns * ns + 2) * nk * 8 * 16)
    tiled = temp_bytes()
    pair_face = 2 * nk * ns * ns * 8 * 32 * 16                     # both all-column pair densities
    assert whole >= pair_face          # RED TWIN: one tile holds every column's pair densities
    assert tiled <= whole // 3


def test_q_stream_is_the_one_tile_chain(mesh, monkeypatch):
    """D3-D5 one q at a time equal D3-D5 on every q at once."""
    from ffi.fft import xla_reference
    from gw.downfold import (BandWindow, pair_density_gram, slice_psi_to_centroids,
                             stream_transfer)
    rng = np.random.default_rng(3)
    kgrid, nk, nb, ns, muL = (3, 2, 1), 6, 4, 2, 16
    X, Y = _faces(_psi(rng, nk, nb, ns, muL), mesh)
    win = BandWindow(left=(0, 4), right=(0, 4))
    keep = np.array([0, 2, 3, 5, 8, 9, 12, 15])
    with xla_reference():
        S_LL = pair_density_gram(X, Y, win, kgrid=kgrid, mesh_xy=mesh)
        XS, YS = slice_psi_to_centroids(X, Y, keep, 8, mesh)
        S_cross = pair_density_gram(XS, Y, win, kgrid=kgrid, mesh_xy=mesh)
        S_SS = pair_density_gram(XS, YS, win, kgrid=kgrid, mesh_xy=mesh)
    tensors = {"V_qmunu": _hpsd(rng, nk, muL, mesh), "W0_qmunu": _hpsd(rng, nk, muL, mesh)}
    g0 = jax.device_put(rng.normal(size=muL) + 1j * rng.normal(size=muL),
                        NamedSharding(mesh, P()))

    def run():
        return stream_transfer(S_LL, S_SS, S_cross, tensors, mesh, rcond=1e-8,
                               g0_L=g0, **QUIET)

    monkeypatch.setattr("runtime.tiles.TILE_BYTES", 1 << 40)
    one = run()
    monkeypatch.setattr("runtime.tiles.TILE_BYTES", 1)
    streamed = run()
    assert (one["n_tiles"], streamed["width"], streamed["n_tiles"]) == (1, 1, nk)
    for name in tensors:
        assert _rel(streamed["small"][name], one["small"][name]) <= 1e-12
        np.testing.assert_allclose(streamed["eps"][name], one["eps"][name], rtol=1e-10)
    assert _rel(streamed["T"], one["T"]) <= 1e-12
    assert _rel(streamed["g0_S"], one["g0_S"]) <= 1e-12
    assert [r.rank_criterion for r in streamed["reports"]] == \
        [r.rank_criterion for r in one["reports"]]

    # RED TWIN: a stream that never visits the last q leaves that child q empty.
    monkeypatch.setattr("gw.downfold.transfer_q_tile",
                        lambda *a, **k: (1, tuple(range(nk - 1))))
    assert _rel(run()["small"]["W0_qmunu"], one["small"]["W0_qmunu"]) > 1e-3


def test_pool_spectrum_is_the_logical_block(mesh):
    """The device spectrum of a zero-padded Gram is its logical block's."""
    from gw.downfold import pool_gram_spectrum
    rng = np.random.default_rng(4)
    n, pad = 6, 2
    a = rng.normal(size=(n, 3 * n)) + 1j * rng.normal(size=(n, 3 * n))
    G = a @ a.conj().T
    Gp = np.zeros((n + pad, n + pad), complex)
    Gp[:n, :n] = G
    S = jax.device_put(Gp, NamedSharding(mesh, P("x", "y")))
    ref = np.linalg.eigvalsh(G)
    ev = pool_gram_spectrum(S, mesh, n)
    assert ev.shape == (n,)
    np.testing.assert_allclose(ev, ref, rtol=1e-10, atol=1e-12 * ref.max())
    # RED TWIN: read at the padded extent, the pad's zeros enter the spectrum.
    bad = pool_gram_spectrum(S, mesh, n + pad)
    assert bad.shape == (n + pad,) and int(np.sum(np.abs(bad) < 1e-12 * ref.max())) == pad


def test_plan_key_has_one_plan(tmp_path):
    """``auto`` and ``distributed`` load; ``local`` and anything else refuse by GATE."""
    from gw.downfold_config import DownfoldConfig

    def deck(plan):
        p = tmp_path / f"d_{plan}.in"
        p.write_text("[downfold]\nsource_restart = /nonexistent/parent\n"
                     "output_restart = /nonexistent/child\nband_range_left = 0:4\n"
                     "band_range_right = 0:4\nmu_small = 8\n"
                     + (f"plan = {plan}\n" if plan else ""))
        return str(p)

    for ok in (None, "auto", "distributed"):
        assert DownfoldConfig.from_input_file(deck(ok), **QUIET).plan in ("auto", "distributed")
    with pytest.raises(ValueError, match=r"GATE downfold_plan_local: got: plan = local; "
                                         r"want: auto or distributed.*Fix: delete the key"):
        DownfoldConfig.from_input_file(deck("local"), **QUIET)
    with pytest.raises(ValueError, match=r"GATE downfold_plan: got: plan = 'gspmd'"):
        DownfoldConfig.from_input_file(deck("gspmd"), **QUIET)

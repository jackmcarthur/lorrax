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


@pytest.fixture
def cpu_mesh():
    """``mesh`` on host devices, for a bound on the host compiler's ``memory_analysis()``.

    On a GPU node (``lx test``: four CUDA devices in one process) ``jax.devices()``
    is the GPUs, and the GPU compiler's temp for this Gram (296200 B) is over the
    one-tile budget the test states for the host (172032 B; CRI3TD INTEG stage 2),
    so this mesh is built from ``jax.devices("cpu")`` and skips by name unless four
    host devices exist.
    """
    try:
        cpus = jax.devices("cpu")
    except RuntimeError:                      # a platform list without cpu
        cpus = []
    if len(cpus) < 4:
        pytest.skip("CPU only: needs 4 host devices (JAX_PLATFORMS=cpu "
                    "XLA_FLAGS=--xla_force_host_platform_device_count=4); the one-buffer "
                    "bound is the host compiler's memory_analysis()")
    return Mesh(np.array(cpus[:4]).reshape(2, 2), ("x", "y"))


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


def test_gram_holds_one_output_buffer(cpu_mesh, monkeypatch):
    """The Gram writes into one donated output buffer; only the column tile is scratch."""
    mesh = cpu_mesh                                                 # the host compiler's bound
    import isdf.core as core
    from ffi.fft import xla_reference
    from gw.downfold import negate_q_index
    rng = np.random.default_rng(7)
    kgrid, nk, nb, ns, mu = (3, 2, 1), 6, 3, 2, 128                # local face (64, 64)
    X, Y = _faces(_psi(rng, nk, nb, ns, mu), mesh)
    out = nk * 64 * 64 * 16
    tile = 2 * (3 * ns * ns + 2) * nk * 64 * 16                    # two columns per tile
    monkeypatch.setattr("runtime.tiles.TILE_BYTES", tile)
    monkeypatch.setattr(core, "_pair_pipeline_sm_cache", {})
    with xla_reference():
        core.c_q_downfold(X, Y, X, Y, kgrid=kgrid, mesh_xy=mesh, q_order=negate_q_index(kgrid))
        (zeros, fn), = core._pair_pipeline_sm_cache.values()
        ident = (jnp.arange(ns, dtype=jnp.int32), jnp.ones(ns, jnp.complex128))
        m = fn.lower(zeros(), X, Y, X, Y, *ident, *ident).compile().memory_analysis()
    assert m.alias_size_in_bytes == out          # the output IS the donated buffer
    assert m.temp_size_in_bytes <= tile          # the relabel and the carry add nothing
    # RED TWIN: relabelling the finished Gram as a separate step (the pattern this
    # replaced, measured at 57 GB/device on the 6000 parent) costs a second output.
    C = jax.device_put(np.zeros((nk, mu, mu), complex), NamedSharding(mesh, P(None, "x", "y")))
    relabel = jax.jit(lambda c: jnp.take(c, jnp.asarray(negate_q_index(kgrid)), axis=0))
    r = relabel.lower(C).compile().memory_analysis()
    assert r.output_size_in_bytes == out and r.alias_size_in_bytes == 0


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


def test_wedge_round_trip_streams_on_the_device(mesh, monkeypatch):
    """The child-wedge gate's residual, q-streamed on the device, is 0 on an exact unfold."""
    from types import SimpleNamespace
    from ffi import _services
    _services.ensure_on_path()
    from symmetry_maps import unfold_isdf_operator
    from gw.downfold_run import _unfold_roundtrip_rel
    rng = np.random.default_rng(6)
    mu = 8
    perm = np.stack([np.arange(mu), np.roll(np.arange(mu), 2)]).astype(np.int32)
    L = np.zeros((2, mu, 3))
    tables = SimpleNamespace(irr_idx_q=np.array([0, 0, 1, 1]), sym_idx_q=np.array([0, 1, 0, 1]),
                             q_irr_frac=np.array([[0., 0., 0.], [0.5, 0., 0.]]), n_sym_spatial=2)
    a = rng.normal(size=(2, mu, mu)) + 1j * rng.normal(size=(2, mu, mu))
    wedge = jax.device_put(a + a.conj().transpose(0, 2, 1), NamedSharding(mesh, P(None, "x", "y")))
    full = unfold_isdf_operator(
        wedge, irr_idx=jnp.asarray(tables.irr_idx_q), sym_idx=jnp.asarray(tables.sym_idx_q),
        sym_perm=jnp.asarray(perm), L_table=jnp.asarray(L),
        q_irr_frac=jnp.asarray(tables.q_irr_frac), mesh_xy=mesh, n_sym_spatial=2)
    slots = np.array([0, 2])                       # the full-zone index of each wedge q
    monkeypatch.setattr("runtime.tiles.TILE_BYTES", 1)          # one q per tile
    assert _unfold_roundtrip_rel(full, tables, perm, L, slots, mesh, mu) <= 1e-15
    # RED TWIN: a full-zone q that is not the unfold of its wedge q is caught.
    bad = np.asarray(full).copy()
    bad[3] *= 1.0 + 1e-6
    bad = jax.device_put(bad, NamedSharding(mesh, P(None, "x", "y")))
    assert _unfold_roundtrip_rel(bad, tables, perm, L, slots, mesh, mu) > 1e-7

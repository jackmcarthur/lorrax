"""Finite-Q bsemat bookkeeping (``bse.bsemat_export.finite_q_legs``, the exchange tile momentum and
the writer's tile, the off-grid refusal, the header).

CPU clause of the four-GPU rule: index arithmetic and file layout; the writer's exchange tile on a
2 x 2 host mesh. The finite-Q kernel itself is verified end to end by BerkeleyGW absorption on the
file against BerkeleyGW's own kernel at the same Q (docs/theory/bgw-bsemat-export.md#finite-q).
"""
import os

import h5py
import numpy as np
import pytest

GRID = np.array([3, 3, 1])
K = np.array([[i / 3, j / 3, 0.0] for i in range(3) for j in range(3)])
FIXTURE_WFN = os.path.join(os.path.dirname(__file__), "hsuite", "fixture", "WFN.h5")


def _gi(frac):
    from bse.bsemat_export import grid_index
    return grid_index(np.asarray(frac, float), GRID)


def test_q_zero_and_integer_q_are_qflag_1():
    from bse.bsemat_export import finite_q_legs
    for Q in ((0, 0, 0), (1, 0, 0), (0, -1, 0)):
        qflag, c, v = finite_q_legs(K, GRID, Q)
        assert qflag == 1 and np.array_equal(c, v) and np.array_equal(c, _gi(K))


def test_on_grid_q_pairs_are_exciton_bands_pairs():
    """qflag 2: row j holds (hole k_j − Q, electron k_j); as a set this is |v k, c k+Q⟩."""
    from bse.bsemat_export import finite_q_legs
    Q = np.array([1 / 3, 0.0, 0.0])
    qflag, c, v = finite_q_legs(K, GRID, Q)
    assert qflag == 2
    assert np.array_equal(v, _gi(K - Q)) and np.array_equal(c, _gi(K))
    file_pairs = set(zip(v.tolist(), c.tolist()))
    exb_pairs = set(zip(_gi(K).tolist(), _gi(K + Q).tolist()))       # (v at k, c at k+Q)
    assert file_pairs == exb_pairs
    # Control variant: the opposite sign (hole at k + Q) is the exciton at −Q, a different pair set.
    wrong = set(zip(_gi(K + Q).tolist(), _gi(K).tolist()))
    assert wrong != exb_pairs


def _toy_tile(q, r_mu, ns, v):
    """V_q of a 1-D toy whose centroids are every grid point (ISDF exact on the band window)."""
    n = np.arange(-ns // 2, ns // 2)
    p = q + 2 * np.pi * n                                         # plane waves of momentum q + G
    zh = np.exp(-1j * np.outer(p, r_mu)) / ns                     # FT of zeta_{q mu}: (1/N) e^{-i p r_mu}
    return (zh.conj().T * v(p)) @ zh, p


def test_exchange_tile_momentum_is_plus_q_against_plane_waves():
    """E12: conj(rho_vc) V_q rho_v'c' with rho = psi*_{v,k-Q} psi_{c,k} reproduces the plane-wave exchange only
    at q = exchange_tile_momentum(Q) = +Q. Control variant: the tile at -Q (the conjugate label) does not."""
    from bse.bsemat_export import exchange_tile_momentum
    rng = np.random.default_rng(7)
    N, Q = 16, np.array([0.25, 0.0, 0.0])
    r = np.arange(N) / N
    v = lambda p: 1.0 / (p ** 2 + 0.3)
    def bloch(k):                                                 # band-limited u, |G| <= 2
        c = rng.normal(size=5) + 1j * rng.normal(size=5)
        return np.exp(1j * k * r) * (c @ np.exp(2j * np.pi * np.outer(np.arange(-2, 3), r)))
    ks = 2 * np.pi * np.array([0.0, 0.25, 0.5])
    rho = np.array([bloch(k - 2 * np.pi * Q[0]).conj() * bloch(k) for k in ks])   # momentum +Q
    # plane-wave exchange: L sum_p conj(rho(p)) v(p) rho'(p), p in Q + 2 pi Z
    n = np.arange(-N // 2, N // 2)
    p = 2 * np.pi * (Q[0] + n)
    rh = rho @ np.exp(-1j * np.outer(r, p)) / N
    X_pw = (rh.conj() * v(p)) @ rh.T
    q = exchange_tile_momentum(Q)
    assert np.allclose(q, [0.25, 0, 0], rtol=0, atol=1e-15)
    V_plus, _ = _toy_tile(2 * np.pi * q[0], r, N, v)
    V_minus, _ = _toy_tile(-2 * np.pi * q[0], r, N, v)
    X_plus = rho.conj() @ V_plus @ rho.T
    X_minus = rho.conj() @ V_minus @ rho.T
    assert np.linalg.norm(X_plus - X_pw) / np.linalg.norm(X_pw) < 1e-12
    assert np.linalg.norm(X_minus - X_pw) / np.linalg.norm(X_pw) > 1e-2, "the -Q tile must not reproduce it"
    assert np.allclose(exchange_tile_momentum([0, 0, 0]), 0.0) and np.allclose(exchange_tile_momentum([1.25, 0, 0]), [0.25, 0, 0])


def _cpu_mesh():
    import jax
    from bse.bse_ring_comm import create_mesh_2d
    cpu = jax.devices("cpu")
    if len(cpu) < 4:
        pytest.skip("needs 4 host CPU devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return create_mesh_2d(devices=cpu[:4])


def test_writer_exchange_tile_is_the_stored_tile_at_wrap_plus_q():
    """E12 through the writer: ``exchange_tile`` (route ``ongrid``) returns the stored V at wrap(+Q).

    The call goes through ``bse.exchange_path.exchange_tiles``, which labels its argument's tile
    wrap(-arg); the test fails if the writer's -Q argument flips or if that shared routine is
    relabelled. Control variant: the stored tile at wrap(-Q) differs at every Q. Q = 0 returns
    the head-less Gamma tile itself. A synthetic Hermitian tile stack, distinct at every q of a
    3 x 3 grid, on a 2 x 2 host mesh.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from bse import bsemat_export as X
    mesh = _cpu_mesh()
    pad = 8
    up = np.triu(np.ones((pad, pad)), 1)
    V = np.zeros((pad, pad, 3, 3, 1), complex)
    for ix in range(3):
        for iy in range(3):
            V[:, :, ix, iy, 0] = (ix * 3 + iy + 1) * np.eye(pad) + 1j * (ix - 2 * iy + 0.5) * (up - up.T)
    data = {"V_q0": jax.device_put(jnp.eye(pad, dtype=jnp.complex128), NamedSharding(mesh, P("x", "y"))),
            "V_q_full": jax.device_put(V, NamedSharding(mesh, P("x", "y", None, None, None)))}
    rep = jax.jit(lambda a: a, out_shardings=NamedSharding(mesh, P()))
    at = lambda q: V[(slice(None), slice(None)) + tuple(int(i) for i in np.round(q * GRID).astype(int) % GRID)]
    for Q in ([1 / 3, 0, 0], [2 / 3, 0, 0], [0, 1 / 3, 0], [-1 / 3, 2 / 3, 0], [4 / 3, -1 / 3, 0]):
        Q = np.asarray(Q, dtype=np.float64)
        Vt, rec = X.exchange_tile(Q, 2, data=data, input_file=None, restart=None, grid=GRID,
                                  n_rmu=pad, n_rmu_pad=pad, mesh_xy=mesh)
        Vt = np.asarray(rep(Vt))
        assert np.abs(Vt - at(Q)).max() < 1e-14, (Q.tolist(), "the writer's tile is not V[wrap(+Q)]")
        assert np.abs(Vt - at(-Q)).max() > 1e-3, (Q.tolist(), "the control variant must differ")
        assert "wrap(+Q)" in rec
    V0, rec0 = X.exchange_tile(np.zeros(3), 1, data=data, input_file=None, restart=None, grid=GRID,
                               n_rmu=pad, n_rmu_pad=pad, mesh_xy=mesh)
    assert V0 is data["V_q0"] and "head-less" in rec0


def test_off_grid_route_refuses_without_a_full_zone_zeta(tmp_path, monkeypatch):
    """``GATE bsemat_finite_q_offgrid``: qflag 0 runs only when the restart's zeta_q.h5 stores every q."""
    import zeta_loader
    from bse.bsemat_export import require_offgrid_zeta
    restart = str(tmp_path / "isdf_tensors_8.h5")
    with pytest.raises(ValueError, match="GATE bsemat_finite_q_offgrid.*absent"):
        require_offgrid_zeta(restart, 9, (1 / 6, 0, 0))
    (tmp_path / "zeta_q.h5").write_bytes(b"")

    def header(layout, nq):
        class Header:
            def __init__(self, path, **kw):
                self.q_layout, self.n_q_on_disk = layout, nq
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                return False
        return Header
    monkeypatch.setattr(zeta_loader, "ZetaLoader", header("ibz", 5))
    with pytest.raises(ValueError, match="GATE bsemat_finite_q_offgrid.*stores 5 of the grid's 9"):
        require_offgrid_zeta(restart, 9, (1 / 6, 0, 0))
    monkeypatch.setattr(zeta_loader, "ZetaLoader", header("full_bz", 9))
    assert require_offgrid_zeta(restart, 9, (1 / 6, 0, 0)) == 9


def test_off_grid_q_needs_the_shifted_wfn_co():
    from bse.bsemat_export import finite_q_legs
    Q = np.array([1 / 6, 0.0, 0.0])
    qflag, c, v = finite_q_legs(K + Q, GRID, Q)                    # WFN_co = grid + Q
    assert qflag == 0 and c is None and np.array_equal(v, _gi(K))
    with pytest.raises(ValueError, match="GATE bsemat_finite_q_grid"):
        finite_q_legs(K, GRID, Q)                                  # the unshifted WFN_co


@pytest.mark.skipif(not os.path.exists(FIXTURE_WFN), reason="hsuite fixture WFN absent")
def test_header_carries_qflag_and_minus_q(tmp_path):
    from file_io import bgw_bsemat as F
    kw = dict(wfn_file=FIXTURE_WFN, kpts=K, kgrid=GRID, nvb=1, ncb=1, extended=False, icutv=0,
              ecuts=5.0, ecutg=5.0, efermi=0.0, nspinor=1)
    F.create_bsemat_file(str(tmp_path / "q0.h5"), **kw)
    F.create_bsemat_file(str(tmp_path / "q.h5"), qflag=2, exciton_Q_shift=(-1 / 3, 0, 0), **kw)
    h0, h = F.read_bsemat_header(str(tmp_path / "q0.h5")), F.read_bsemat_header(str(tmp_path / "q.h5"))
    assert int(h0["kpoints/qflag"]) == 1 and np.all(h0["kpoints/exciton_Q_shift"] == 0)
    assert int(h["kpoints/qflag"]) == 2
    assert np.allclose(h["kpoints/exciton_Q_shift"], [-1 / 3, 0, 0], rtol=0, atol=1e-15)
    with h5py.File(str(tmp_path / "q.h5"), "r") as f:
        assert f["bse_header/kpoints/exciton_Q_shift"].dtype == np.float64

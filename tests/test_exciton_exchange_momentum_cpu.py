"""Finite-Q native/export exchange against a complete-basis plane-wave sum."""
import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


def _functions(path, names, namespace):
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


def _host_owners(monkeypatch, metric):
    # Only placement and the expensive fit/evaluator are substituted. The
    # native route selection, certification rows and exporter call are real.
    root = Path(__file__).parents[1]
    package = ModuleType("bse")
    package.__path__ = []
    owner = ModuleType("bse.exchange_path")
    vq = ModuleType("bse.vq_interp")
    vq.refit_vq = lambda _zx, _rst, q, *_a, **_kw: metric(q)
    package.vq_interp = vq
    for name, module in (("bse", package), ("bse.exchange_path", owner),
                          ("bse.vq_interp", vq)):
        monkeypatch.setitem(sys.modules, name, module)
    put = lambda x, *_a, **_kw: np.asarray(x)
    namespace = dict(__package__="bse", np=np, jnp=np,
                     jax=SimpleNamespace(device_put=put, jit=lambda fn, **_kw: fn),
                     NamedSharding=lambda *_a: None, P=lambda *_a: None,
                     device_put_process_local=put, log=SimpleNamespace(info=lambda *_: None))
    _functions(root/"src/bse/exchange_path.py",
               {"exchange_tile_momentum", "_tile_index", "_host_tile", "_hermitize", "exchange_tiles"},
               namespace)
    owner.__dict__.update(namespace)
    exporter = dict(namespace, _log=lambda *_: None)
    _functions(root/"src/bse/bsemat_export.py", {"exchange_tile_momentum", "exchange_tile"}, exporter)
    return owner, exporter["exchange_tile"]


def _plane_wave_fixture():
    # Complete real-space centroid basis: no ISDF approximation. G are a 3D
    # plane-wave subset (Gx,0,0); the fields are constant in y,z. The smooth
    # transition densities fit below Nyquist, so the discrete Fourier map is exact.
    n = 11
    x = np.arange(n)/n
    g = np.fft.fftfreq(n, d=1/n)
    fourier = np.exp(-2j*np.pi*g[:, None]*x[None, :])/np.sqrt(n)
    rho = np.stack((np.exp(2j*np.pi*x)+.4j*np.exp(6j*np.pi*x),
                    .5*np.exp(-4j*np.pi*x)+np.exp(2j*np.pi*x)))

    def coulomb(q):
        k2 = (g+q[0])**2 + q[1]**2 + q[2]**2
        values = np.zeros(n)
        np.divide(4*np.pi, k2, out=values, where=k2 > 1e-15)
        return values

    def metric(q):
        # The producer's declared V_mu,nu(q)=sum_G conj(zeta_mu) v(q+G) zeta_nu.
        return fourier.conj().T @ (coulomb(q)[:, None]*fourier)

    def literal(q):
        density_g = rho @ fourier.T
        return (density_g.conj()*coulomb(q)[None, :]) @ density_g.T

    return n, rho.conj(), metric, literal


@pytest.mark.parametrize("route", ["ongrid", "refit", "interp"])
def test_native_exchange_and_certificate_match_plane_wave_sum(monkeypatch, route):
    n, pair, metric, literal = _plane_wave_fixture()
    owner, _ = _host_owners(monkeypatch, metric)
    grid = np.array([4, 1, 1])
    stored = np.stack([metric(np.array([q, 0., 0.]))
                       for q in np.fft.fftfreq(4)], axis=-1)[:, :, :, None, None]
    q = np.array([.25, 0., 0.])
    gamma = 3*np.eye(n)  # The exact-Q=0 production shortcut must remain intact.
    rows, _, _, count = owner.exchange_tiles(
        np.array([q, np.zeros(3)]), route=route, mesh_xy=None,
        V_q0=gamma, V_ongrid=stored, kgrid_vq=grid, n_rmu=n, n_rmu_pad=n,
        refit=(None, None), interp=(None, {"V_SRc": None},
            lambda momentum, *_a: metric(momentum), None, None),
        cert_idx=(0,), both_refit=(None, None), both_idx=(0,))
    for index in (0, 2, 3):  # Native solve, on-grid certificate and refit spot.
        actual = pair @ rows[index] @ pair.conj().T
        np.testing.assert_allclose(actual, literal(q), rtol=2e-14, atol=1e-12)
    np.testing.assert_array_equal(rows[1], gamma)
    assert count == 1
    # A real signed-momentum oracle: the old tile gives a different operator.
    assert np.max(np.abs(pair @ metric(-q) @ pair.conj().T-literal(q))) > 1.


def test_bgw_exporter_preserves_its_positive_q_plane_wave_contract(monkeypatch):
    n, pair, metric, literal = _plane_wave_fixture()
    _, export = _host_owners(monkeypatch, metric)
    stored = np.stack([metric(np.array([q, 0., 0.]))
                       for q in np.fft.fftfreq(4)], axis=-1)[:, :, :, None, None]
    q = np.array([.25, 0., 0.])
    actual, _ = export(q, 2, data={"V_q0":np.eye(n), "V_q_full":stored},
                       input_file="unused", restart="unused", grid=(4,1,1),
                       n_rmu=n, n_rmu_pad=n, mesh_xy=None)
    np.testing.assert_allclose(pair @ actual @ pair.conj().T,
                               literal(q), rtol=2e-14, atol=1e-12)

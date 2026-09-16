"""Metal chi0 kernels on a time-reversal-broken lattice with fractional occupations: orientation gate.

A tiny 2-D lattice with complex hoppings (inversion and time reversal both broken, so e_k != e_-k
and the pair densities are complex) and Fermi-Dirac occupations with mu inside a band.  Both metal
kernels are evaluated on a CPU 1x1 mesh and compared with the supercell Kubo sum in the stream's
convention (TRINT B1, validated against Sigma by the Casida plant):

    chi(r, r'; z) = sum_ab (f_a - f_b) / (e_a - e_b + z) conj(rho_ab(r)) rho_ab(r'),  rho_ab = psi_a conj psi_b,

through the four candidates FT_{+-q}[chi], FT_{+-q}[chi^T], FT_q[f](mu, nu) = sum_R f(r_mu, r_nu + R) e^{iq.R}.
The incumbent trace (ordered=False) is FT_q[chi^T]; ordered=True is the physical FT_q[chi] and equals the
incumbent row -q transposed.  CPU stand-ins for the flat-k FFT, the GEMM plan and the accumulator are
installed with monkeypatch and stated here as the scope (sandbox run 440_metal_20260916, 2026-09-16).
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

jax.config.update("jax_enable_x64", True)


class _Lattice:
    """s sites per cell on a 2-D oblique lattice, generic complex Hermitian hoppings, n1 x n2 k grid."""

    def __init__(self, n1=3, n2=3, n_sites=3, seed=20260915, hop=0.35, gap_shift=2.0, real_hopping=False):
        rng = np.random.default_rng(seed)
        self.n1, self.n2, self.ns = n1, n2, n_sites
        self.a = np.array([[1.0, 0.0], [0.5, np.sqrt(3) / 2]])
        self.tau = rng.uniform(0.1, 0.9, size=(n_sites, 2)) @ self.a
        shifts = [(0, 0), (1, 0), (0, 1), (1, -1)]
        onsite = np.diag(np.linspace(-gap_shift, gap_shift, n_sites) + 0.1 * rng.normal(size=n_sites))
        h0 = hop * (rng.normal(size=(n_sites, n_sites)) + 1j * rng.normal(size=(n_sites, n_sites)))
        self.hop = {(0, 0): onsite + 0.5 * (h0 + h0.conj().T)}
        for R in shifts[1:]:
            self.hop[R] = hop * (rng.normal(size=(n_sites, n_sites)) + 1j * rng.normal(size=(n_sites, n_sites)))
        if real_hopping:
            self.hop = {R: np.real(m) for R, m in self.hop.items()}
        self.kfrac = np.array([(i / n1, j / n2) for i in range(n1) for j in range(n2)])
        self.nk = n1 * n2
        self.cells = np.array([(i, j) for i in range(n1) for j in range(n2)])
        self.b = 2 * np.pi * np.linalg.inv(self.a).T

    def kvec(self, kf):
        return np.asarray(kf, float) @ self.b

    def h_k(self, kf):
        k = self.kvec(kf)
        h = np.zeros((self.ns, self.ns), complex)
        for R, block in self.hop.items():
            for sign, mat in ((1, block), (-1, block.conj().T)):
                if R == (0, 0) and sign == -1:
                    continue
                Rv = sign * (R[0] * self.a[0] + R[1] * self.a[1])
                h += mat * np.exp(1j * ((Rv[None, None, :] + self.tau[None, :, :] - self.tau[:, None, :]) @ k))
        return 0.5 * (h + h.conj().T)

    def bands(self):
        e, u = zip(*(np.linalg.eigh(self.h_k(kf)) for kf in self.kfrac))
        return np.asarray(e), np.asarray(u)

    def bloch_states(self, u):
        pos = np.asarray([c[0] * self.a[0] + c[1] * self.a[1] + self.tau[s]
                          for c in self.cells for s in range(self.ns)])
        site_of = np.tile(np.arange(self.ns), self.nk)
        psi = np.zeros((self.nk, self.ns, self.nk * self.ns), complex)
        for ik, kf in enumerate(self.kfrac):
            psi[ik] = (np.exp(1j * pos @ self.kvec(kf))[None, :] * u[ik][site_of, :].T) / np.sqrt(self.nk)
        return psi

    def ft_q(self, f, q_frac):
        q = self.kvec(q_frac)
        out = np.zeros((self.ns, self.ns), complex)
        for c, cell in enumerate(self.cells):
            R = cell[0] * self.a[0] + cell[1] * self.a[1]
            out += f[:self.ns, c * self.ns:(c + 1) * self.ns] * np.exp(1j * R @ q)
        return out

    def minus(self):
        return [((-i) % self.n1) * self.n2 + ((-j) % self.n2) for i in range(self.n1) for j in range(self.n2)]

    def kminq(self, iq):
        i0, j0 = divmod(iq, self.n2)
        return np.asarray([((i - i0) % self.n1) * self.n2 + ((j - j0) % self.n2)
                           for i in range(self.n1) for j in range(self.n2)], np.int32)


def _kubo(psi_sc, e, f, z):
    N = psi_sc.shape[-1]
    X = np.zeros((N, N), complex)
    states = [(ik, n) for ik in range(e.shape[0]) for n in range(e.shape[1])]
    for a in states:
        for b in states:
            rho = psi_sc[a] * np.conj(psi_sc[b])
            X += (f[a] - f[b]) / (e[a] - e[b] + z) * np.outer(np.conj(rho), rho)
    return X


def _candidates(lat, per_out):
    out = {}
    for name, sign, transpose in (("FT_q[chi]", +1, False), ("FT_q[chi^T]", +1, True),
                                  ("FT_-q[chi]", -1, False), ("FT_-q[chi^T]", -1, True)):
        arr = np.zeros((lat.nk, len(per_out), lat.ns, lat.ns), complex)
        for iq, qf in enumerate(lat.kfrac):
            for o, X in enumerate(per_out):
                arr[iq, o] = lat.ft_q(X.T if transpose else X, sign * np.asarray(qf))
        out[name] = arr
    return out


def _resid(got, cand):
    g, h = got.ravel(), cand.ravel()
    c = np.vdot(h, g) / np.vdot(h, h)
    return float(np.linalg.norm(g - c * h) / np.linalg.norm(g))


@pytest.fixture
def cpu_standins(monkeypatch):
    import common.fft_helpers as fh
    import distrib_la
    import gw.contour_accumulator as ca

    def emulated(kind):
        def factory(mesh_, kgrid, spec, *, norm="ortho", out_spec=None):
            maker = fh.make_sharded_fftn_3d if kind == "fftn" else fh.make_sharded_ifftn_3d
            fft3 = maker(mesh_, spec, spec, axes=(0, 1, 2), norm=norm)
            return lambda x: fft3(jnp.reshape(x, tuple(kgrid) + x.shape[1:])).reshape(x.shape)
        return factory
    monkeypatch.setattr(fh, "make_flat_k_fftn", emulated("fftn"))
    monkeypatch.setattr(fh, "make_flat_k_ifftn", emulated("ifftn"))
    monkeypatch.setattr(distrib_la, "gemm_plan", lambda *a, **k: (lambda A, B: A @ B))
    monkeypatch.setattr(ca, "contour_accumulator", lambda mesh_: (lambda acc, c, p: acc + p[:, None, None, None] * c[None]))
    from gw import w_isdf
    w_isdf._chi_minimax_kernel_cache.clear()
    yield
    w_isdf._chi_minimax_kernel_cache.clear()


def _setup(real_hopping):
    lat = _Lattice(real_hopping=real_hopping)
    e, u = lat.bands()
    psi_sc = lat.bloch_states(u)
    band = e[:, 0]
    mu, kT = float(np.median(band)), float(np.ptp(band) / 3.0)
    f = 1.0 / (1.0 + np.exp((e - mu) / kT))
    return lat, e, psi_sc, mu, f, f * (1 - f) / kT


def _pair_outputs(lat, e, psi_sc, f, surface, z, ordered):
    from gw import w_isdf
    mesh = Mesh(np.asarray(jax.devices("cpu")[:1]).reshape(1, 1), ("x", "y"))
    put = lambda a: jax.device_put(jnp.asarray(a), NamedSharding(mesh, P()))
    psi_cell = np.sqrt(lat.nk) * psi_sc[:, :, :lat.ns]
    psi_mun = put(psi_cell.transpose(0, 2, 1)[:, None, :, :])
    psi_nmu = put(psi_cell[:, :, None, :])
    kern = w_isdf._get_chi_fractional_q_kernel_face(mesh, nb_full=lat.ns, nb_logical=lat.ns, pair_tile=2,
                                                    n_z=z.size, layout="face", ordered=ordered)
    out = np.zeros((lat.nk, z.size, lat.ns, lat.ns), complex)
    for iq in range(lat.nk):
        row = lat.kminq(iq)
        if ordered:
            row = np.argsort(row, kind="stable").astype(np.int32)
        out[iq] = np.asarray(kern(psi_mun, psi_nmu, put(row), put(e), put(f), put(surface), put(z)))
    return out


def _contour_outputs(lat, e, psi_sc, f, ordered, times, e_ref):
    from gw import w_isdf
    mesh = Mesh(np.asarray(jax.devices("cpu")[:1]).reshape(1, 1), ("x", "y"))
    put = lambda a: jax.device_put(jnp.asarray(a), NamedSharding(mesh, P()))
    psi_cell = np.sqrt(lat.nk) * psi_sc[:, :, :lat.ns]
    psi_mun = put(psi_cell.transpose(0, 2, 1)[:, None, :, :])
    psi_nmu = put(psi_cell[:, :, None, :])
    kern = w_isdf._get_chi_fractional_contour_kernel_face(
        mesh, (lat.n1, lat.n2, 1), times.size, (lat.nk, lat.ns, lat.ns, 1), pair_mode="retarded", ordered=ordered)
    got = kern(put(times), put(np.eye(times.size, dtype=complex)), psi_mun, psi_nmu, put(e),
               put(f.astype(complex)), put((1 - f).astype(complex)), put(np.float64(e_ref)))
    return np.stack([np.asarray(v) for v in got], axis=1)


Z = np.asarray([0.31j, 0.17 + 0.31j, -0.09 + 0.05j])
TIMES = np.asarray([0.37, 1.13, 2.9])


@pytest.mark.parametrize("real_hopping", [False, True], ids=["tr_broken", "trs_control"])
def test_pair_kernel_orientation(cpu_standins, real_hopping):
    lat, e, psi_sc, mu, f, surface = _setup(real_hopping)
    cands = _candidates(lat, [_kubo(psi_sc, e, f, z) for z in Z])
    incumbent = _pair_outputs(lat, e, psi_sc, f, surface, Z, ordered=False)
    physical = _pair_outputs(lat, e, psi_sc, f, surface, Z, ordered=True)
    assert _resid(physical, cands["FT_q[chi]"]) < 1e-12
    assert _resid(incumbent, cands["FT_q[chi^T]"]) < 1e-12
    # ordered = incumbent row -q transposed (same products, summed in another order: roundoff).
    rel = np.linalg.norm(physical - np.swapaxes(incumbent[lat.minus()], -1, -2)) / np.linalg.norm(physical)
    assert rel < 1e-14
    if real_hopping:
        assert _resid(physical, incumbent) < 1e-12
    else:
        assert _resid(incumbent, cands["FT_q[chi]"]) > 1e-3


@pytest.mark.parametrize("real_hopping", [False, True], ids=["tr_broken", "trs_control"])
def test_full_q_contour_orientation(cpu_standins, real_hopping):
    lat, e, psi_sc, mu, f, _ = _setup(real_hopping)
    N = psi_sc.shape[-1]

    def g_sc(weight, t):
        G = np.zeros((N, N), complex)
        for ik in range(lat.nk):
            for n in range(lat.ns):
                G += weight[ik, n] * np.exp(-1j * (e[ik, n] - 0.21) * t) * np.outer(psi_sc[ik, n], psi_sc[ik, n].conj())
        return G
    per_node = []
    for t in TIMES:
        C = g_sc(1 - f, t) * np.conj(g_sc(f, t))
        per_node.append(-1j * (C - np.conj(C)))
    cands = _candidates(lat, per_node)
    incumbent = _contour_outputs(lat, e, psi_sc, f, False, TIMES, 0.21)
    physical = _contour_outputs(lat, e, psi_sc, f, True, TIMES, 0.21)
    assert _resid(physical, cands["FT_q[chi]"]) < 1e-12
    assert _resid(incumbent, cands["FT_q[chi^T]"]) < 1e-12
    rel = np.linalg.norm(physical - np.swapaxes(incumbent[lat.minus()], -1, -2)) / np.linalg.norm(physical)
    assert rel < 1e-13
    if not real_hopping:
        assert _resid(incumbent, cands["FT_q[chi]"]) > 1e-3

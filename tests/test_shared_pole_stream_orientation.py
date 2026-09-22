"""The ordered response stream stores the physical orientation W_q = FT_q[W] on a time-reversal-broken lattice.

Sigma's G_{k-q} W_q contraction is exact for FT_q[f](mu, nu) = sum_R f(r_mu, r_nu+R) e^{iq.R}. The
incumbent trace returns FT_q[chi^T]. On a TR-broken deck that hands each Green's-function branch the
other branch's residues, Sigma[W^even] - Sigma^odd. Lattice orbitals at the sites are the face carrier.
CPU stand-ins, stated as scope: the jnp flat-k FFT emulation, a plain GEMM and a plain accumulator;
the stream kernel itself is production code. Oracle: the TRINT stream plant (sandbox run
425_trint_20260915, harness/stream_orientation_plant.py).
"""
from types import SimpleNamespace

import numpy as np
import pytest

from tr_broken_lattice import Lattice, cpu_flat_k_fft, ft_q

TIMES = np.asarray([0.37, 1.13, 2.9])


def _resid(got, want, scale=None):
    g, h = got.ravel(), want.ravel()
    c = np.vdot(h, g) / np.vdot(h, h) if scale is None else scale
    return float(np.linalg.norm(g - c * h) / np.linalg.norm(g)), c


def _stack(lat, nodes, *, transpose=False, sign=1):
    return np.stack([np.stack([ft_q(X.T if transpose else X, lat, sign * qf) for X in nodes])
                     for qf in lat.kfrac])


@pytest.fixture
def stream(monkeypatch):
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    import distrib_la
    import gw.contour_accumulator as accumulator
    from gw import w_isdf

    cpu_flat_k_fft(monkeypatch)
    monkeypatch.setattr(distrib_la, "gemm_plan", lambda *a, **k: (lambda A, B: A @ B))
    monkeypatch.setattr(accumulator, "contour_accumulator",
                        lambda mesh: (lambda acc, c, p: acc + p[:, None, None, None] * c[None]))
    lat = Lattice(3, 3, 3, n_occ=1)
    e, u = lat.bands()
    psi = lat.bloch_states(e, u)
    nk, nb, N = psi.shape
    cell = np.sqrt(nk) * psi[:, :, :lat.ns]
    mesh = Mesh(np.asarray(jax.devices("cpu")[:1]).reshape(1, 1), ("x", "y"))
    put = lambda a: jax.device_put(jnp.asarray(a), NamedSharding(mesh, P()))
    f = np.zeros((nk, nb))
    f[:, :lat.n_occ] = 1.0

    def g_supercell(weight, phase):
        return sum(weight[k, n] * phase(e[k, n]) * np.outer(psi[k, n], psi[k, n].conj())
                   for k in range(nk) for n in range(nb) if weight[k, n])

    def kernel(pair_mode, *, n_out=len(TIMES), **options):
        return w_isdf._get_chi_fractional_contour_kernel_face(
            mesh, (lat.n1, lat.n2, 1), n_out, (nk, nb, lat.ns, 1),
            selected_q=tuple(range(nk)), pair_mode=pair_mode, **options)

    return SimpleNamespace(lat=lat, e=e, psi=psi, f=f, u=1.0 - f, put=put, g=g_supercell, kernel=kernel,
                           psi_mun=put(cell.transpose(0, 2, 1)[:, None, :, :]),
                           psi_nmu=put(cell[:, :, None, :]))


def test_ordered_retarded_stream_is_the_physical_orientation(stream):
    s, e_ref = stream, 0.21
    nodes = []
    for t in TIMES:
        phase = lambda E, t=t: np.exp(-1j * (E - e_ref) * t)
        C = s.g(s.u, phase) * np.conj(s.g(s.f, phase))
        nodes.append(-1j * (C - np.conj(C)))
    physical, transposed = _stack(s.lat, nodes), _stack(s.lat, nodes, transpose=True)
    args = (s.put(TIMES), s.put(np.eye(len(TIMES), dtype=complex)), s.psi_mun, s.psi_nmu,
            s.put(s.e), s.put(s.f.astype(complex)), s.put(s.u.astype(complex)), s.put(np.float64(e_ref)))
    ordered = np.asarray(s.kernel("retarded", ordered=True)(*args))
    incumbent = np.asarray(s.kernel("retarded")(*args))
    assert _resid(ordered, physical)[0] <= 1e-10
    assert _resid(ordered, transposed)[0] >= 1e-3
    # The TRS trace keeps the transposed orientation; on this lattice the two differ at O(1).
    assert _resid(incumbent, transposed)[0] <= 1e-10
    assert _resid(incumbent, physical)[0] >= 1e-3


def test_ordered_laplace_rows_are_the_physical_orientation(stream):
    s, refs, n_t = stream, np.asarray([0.0, 0.3]), len(TIMES)
    lower = np.stack([s.f, np.zeros_like(s.f)]).astype(complex)
    upper = np.stack([s.u, np.zeros_like(s.f)]).astype(complex)
    kernel = s.kernel("laplace_ordered")
    zero, eye = np.zeros((n_t, n_t)), np.eye(n_t)
    got = np.concatenate([np.asarray(kernel(
        s.put(TIMES), s.put(np.concatenate(rows).astype(complex)), s.psi_mun, s.psi_nmu,
        s.put(s.e), s.put(lower), s.put(upper), s.put(refs))) for rows in ((eye, zero), (zero, eye))], axis=1)
    even, odd = [], []
    for t in TIMES:
        P = s.g(s.u, lambda E: np.exp(-t * (E - refs[1]))) * np.conj(s.g(s.f, lambda E: np.exp(t * (E - refs[0]))))
        even.append(P + np.conj(P))
        odd.append(P - np.conj(P))
    # Real Laplace phases make P^T = conj(P): the even node is symmetric and the odd node's transpose
    # is minus itself. The even rows fix the q label (q against -q) and the scale; the odd rows are
    # judged at that scale, where the transposed candidate is off by a factor -1.
    resid, scale = _resid(got[:, :n_t], _stack(s.lat, even))
    assert resid <= 1e-10
    assert _resid(got[:, :n_t], _stack(s.lat, even, sign=-1))[0] >= 1e-3
    assert _resid(got[:, n_t:], _stack(s.lat, odd), scale)[0] <= 1e-10
    assert _resid(got[:, n_t:], _stack(s.lat, odd, transpose=True), scale)[0] >= 1e-3


def test_compact_rule_through_ordered_stream_matches_exact_denominators(stream):
    import minimax
    s = stream
    # Choose occupied/empty windows with a positive global gap in this lattice.
    ef, eu = s.e[s.f != 0], s.e[s.u != 0]
    refs = np.array([ef.max(), eu.min()])
    gap = refs[1]-refs[0]
    assert gap > 0
    z = gap*np.array([.2+.1j, 1.7j])
    rule = minimax.response_laplace_rule(gap, eu.max()-ef.min(), z, ordered=True,
                                        reference_ry=gap)
    rows = -np.concatenate([rule[k] for k in (
        'projection_value', 'projection_derivative',
        'odd_projection_value', 'odd_projection_derivative')])
    lower = np.stack([s.f, np.zeros_like(s.f)]).astype(complex)
    upper = np.stack([s.u, np.zeros_like(s.u)]).astype(complex)
    got = np.asarray(s.kernel('laplace_ordered', n_out=2*len(z))(
        s.put(rule['t']), s.put(rows), s.psi_mun, s.psi_nmu,
        s.put(s.e), s.put(lower), s.put(upper), s.put(refs)))
    matrices = np.zeros((2*len(z), s.psi.shape[-1], s.psi.shape[-1]), complex)
    for i in zip(*np.nonzero(s.f)):
        for a in zip(*np.nonzero(s.u)):
            d = s.e[a]-s.e[i]
            density = s.psi[a]*s.psi[i].conj()
            forward = np.outer(density, density.conj())
            matrices[:len(z)] += (forward/(z-d)[:, None, None]
                                    - forward.conj()/(z+d)[:, None, None])
            matrices[len(z):] += (-forward/(2*z*(z-d)**2)[:, None, None]
                                    + forward.conj()/(2*z*(z+d)**2)[:, None, None])
    exact = _stack(s.lat, matrices)
    # Each orthonormal Green FFT is sqrt(nk) times the normalized supercell
    # Green; their product and the final orthonormal FFT leave sqrt(nk).
    np.testing.assert_allclose(got, np.sqrt(s.lat.nk)*exact, rtol=1e-7, atol=1e-9)
    wrong = rows.copy()
    wrong[2*len(z):] = 0
    even_only = np.asarray(s.kernel('laplace_ordered', n_out=2*len(z))(
        s.put(rule['t']), s.put(wrong), s.psi_mun, s.psi_nmu,
        s.put(s.e), s.put(lower), s.put(upper), s.put(refs)))
    assert np.max(abs(even_only-got)) > 1e-3

"""Ordered shared-pole orientation on a time-reversal-broken lattice (TRINT, 2026-09-15).

A 3x3 lattice with 3 sites per cell and complex hoppings breaks time reversal and inversion; the sites are the
centroids. With FT_q[f](mu, nu) = sum_R f(r_mu, r_nu + R) exp(i q.R):
1. the response-bank stream kernel with ``ordered=True`` returns chi_q = FT_q[chi] (the incumbent time-reversal-symmetric
   trace returns FT_q[chi^T]);
2. ordered Sigma (production synthesis, hole kernel and tau kernel) on a store in that orientation reproduces the
   real-space Sigma = iGW to 1e-10, and the swapped branch routing does not.
CPU stand-ins: flat-k FFT through ``make_sharded_fftn_3d`` (numpy sign, as the FFI), GEMM plan -> matmul, contour
accumulator -> jnp. Sandbox evidence: runs/frequency_integration_sandbox/425_trint_20260915 (orientation_* logs).
"""
import numpy as np
import pytest

N1, N2, NS, NOCC = 3, 3, 3, 1


def _lattice(seed=20260915, hop=0.35, gap=2.0):
    rng = np.random.default_rng(seed)
    a = np.array([[1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    tau = rng.uniform(0.1, 0.9, size=(NS, 2)) @ a
    onsite = np.diag(np.linspace(-gap, gap, NS) + 0.1 * rng.normal(size=NS))
    h0 = hop * (rng.normal(size=(NS, NS)) + 1j * rng.normal(size=(NS, NS)))
    hops = {(0, 0): onsite + 0.5 * (h0 + h0.conj().T)}
    for R in ((1, 0), (0, 1), (1, -1)):
        hops[R] = hop * (rng.normal(size=(NS, NS)) + 1j * rng.normal(size=(NS, NS)))
    kfrac = np.array([(i / N1, j / N2) for i in range(N1) for j in range(N2)])
    cells = np.array([(i, j) for i in range(N1) for j in range(N2)])
    b = 2 * np.pi * np.linalg.inv(a).T
    e, u = [], []
    for kf in kfrac:
        k = kf @ b
        h = np.zeros((NS, NS), complex)
        for R, m in hops.items():
            for sign, mat in ((1, m), (-1, m.conj().T)):
                if R == (0, 0) and sign == -1:
                    continue
                Rv = sign * (R[0] * a[0] + R[1] * a[1])
                h += mat * np.exp(1j * ((Rv[None, None, :] + tau[None, :, :] - tau[:, None, :]) @ k))
        w, v = np.linalg.eigh(0.5 * (h + h.conj().T))
        e.append(w)
        u.append(v)
    e, u = np.asarray(e), np.asarray(u)
    pos = np.asarray([c[0] * a[0] + c[1] * a[1] + tau[s] for c in cells for s in range(NS)])
    nk, N = len(kfrac), len(pos)
    site_of = np.tile(np.arange(NS), nk)
    psi = np.stack([np.exp(1j * pos @ (kf @ b))[None, :] * u[ik][site_of, :].T for ik, kf in enumerate(kfrac)]) / np.sqrt(nk)
    L = np.array([N1 * a[0], N2 * a[1]])
    v = np.zeros((N, N))
    for i in range(N):
        d = pos - pos[i]
        dist = np.min([np.linalg.norm(d + m1 * L[0] + m2 * L[1], axis=1) for m1 in (-1, 0, 1) for m2 in (-1, 0, 1)], axis=0)
        v[i] = np.exp(-dist / 0.6) / (1.0 + dist)
    v = 0.5 * (v + v.T)
    v += max(0.0, 1e-3 - np.linalg.eigvalsh(v)[0]) * np.eye(N)
    return dict(e=e, psi=psi, v=v, a=a, cells=cells, b=b, kfrac=kfrac, nk=nk, N=N)


def _modes(M, s3, c):
    """Positive modes of (z s3 - M) through M^1/2 s3 M^1/2; R+ = a a^H."""
    lam, U = np.linalg.eigh(0.5 * (M + M.conj().T))
    half, inv_half = (U * np.sqrt(lam)) @ U.conj().T, (U / np.sqrt(lam)) @ U.conj().T
    w, y = np.linalg.eigh(half @ s3 @ half)
    keep = w > 0
    return w[keep], (c @ (inv_half @ y[:, keep])) * np.sqrt(w[keep])[None, :]


def _flat(i, j):
    return (i % N1) * N2 + (j % N2)


def _minus():
    return [_flat(-i, -j) for i in range(N1) for j in range(N2)]


def _rpa_by_momentum(lat):
    """Exact RPA modes of definite momentum p: particle (i at k, a at k + p) and hole conj(rho) of momentum -p."""
    e, psi, v, nk = lat["e"], lat["psi"], lat["v"], lat["nk"]
    kint = [(i, j) for i in range(N1) for j in range(N2)]
    trans = [((ik, n), (ka, m)) for ik in range(nk) for n in range(NOCC) for ka in range(nk) for m in range(NOCC, NS)]
    mom = [_flat(kint[a[0]][0] - kint[i[0]][0], kint[a[0]][1] - kint[i[0]][1]) for i, a in trans]
    rho = np.array([np.conj(psi[i]) * psi[a] for i, a in trans]).T
    D = np.array([e[a] - e[i] for i, a in trans])
    minus, out = _minus(), {}
    for p in range(nk):
        part = [t for t in range(len(trans)) if mom[t] == p]
        hole = [t for t in range(len(trans)) if mom[t] == minus[p]]
        phi = np.hstack([rho[:, part], rho[:, hole].conj()])
        M = np.diag(np.r_[D[part], D[hole]]).astype(complex) + phi.conj().T @ v @ phi
        s3 = np.diag(np.r_[np.ones(len(part)), -np.ones(len(hole))]).astype(complex)
        out[p] = _modes(M, s3, v @ phi)
    return out


def _ft_q(f, lat, qf):
    q = np.asarray(qf) @ lat["b"]
    out = np.zeros((NS, NS), complex)
    for c, cell in enumerate(lat["cells"]):
        R = cell[0] * lat["a"][0] + cell[1] * lat["a"][1]
        out += f[:NS, c * NS:(c + 1) * NS] * np.exp(1j * R @ q)
    return out


def _green(lat, weight, phase):
    psi, e = lat["psi"], lat["e"]
    G = np.zeros((lat["N"], lat["N"]), complex)
    for ik in range(lat["nk"]):
        for n in range(NS):
            if weight[ik, n]:
                G += weight[ik, n] * phase(e[ik, n]) * np.outer(psi[ik, n], psi[ik, n].conj())
    return G


@pytest.fixture
def cpu_mesh(monkeypatch):
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh
    import common.fft_helpers as fh
    import distrib_la
    import gw.contour_accumulator as ca
    import gw.ppm_tau_kernel as tk

    def emulated(kind):
        def factory(mesh_, kgrid, spec, *, norm="ortho", out_spec=None):
            maker = fh.make_sharded_fftn_3d if kind == "fftn" else fh.make_sharded_ifftn_3d
            fft3 = maker(mesh_, spec, spec, axes=(0, 1, 2), norm=norm)
            return lambda x: fft3(jnp.reshape(x, tuple(kgrid) + x.shape[1:])).reshape(x.shape)
        return factory

    monkeypatch.setattr(fh, "make_flat_k_fftn", emulated("fftn"))
    monkeypatch.setattr(fh, "make_flat_k_ifftn", emulated("ifftn"))
    monkeypatch.setattr(tk, "_fft_ffi_fused_enabled", lambda: False)
    monkeypatch.setattr(distrib_la, "gemm_plan", lambda *a, **k: (lambda A, B: A @ B))
    monkeypatch.setattr(ca, "contour_accumulator", lambda mesh_: (lambda acc, c, p: acc + p[:, None, None, None] * c[None]))
    monkeypatch.setenv("LORRAX_BANDS_GEMM_FFI", "0")
    return Mesh(np.asarray(jax.devices("cpu")[:1]).reshape(1, 1), ("x", "y"))


def _put(mesh, x):
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    return jax.device_put(jnp.asarray(x), NamedSharding(mesh, P()))


def test_ordered_stream_kernel_returns_the_physical_orientation(cpu_mesh):
    from gw import w_isdf
    lat = _lattice()
    nk, e = lat["nk"], lat["e"]
    cell = np.sqrt(nk) * lat["psi"][:, :, :NS]
    f = np.zeros((nk, NS)); f[:, :NOCC] = 1.0
    times, e_ref = np.asarray([0.37, 1.13]), 0.21
    args = (_put(cpu_mesh, times), _put(cpu_mesh, np.eye(2, dtype=complex)),
            _put(cpu_mesh, cell.transpose(0, 2, 1)[:, None]), _put(cpu_mesh, cell[:, :, None]), _put(cpu_mesh, e),
            _put(cpu_mesh, f.astype(complex)), _put(cpu_mesh, (1 - f).astype(complex)), _put(cpu_mesh, np.float64(e_ref)))
    want = np.zeros((nk, 2, NS, NS), complex)
    want_T = np.zeros_like(want)
    for o, t in enumerate(times):
        ph = lambda E: np.exp(-1j * (E - e_ref) * t)
        C = _green(lat, 1 - f, ph) * np.conj(_green(lat, f, ph))
        chi = -1j * (C - np.conj(C))
        for iq, qf in enumerate(lat["kfrac"]):
            want[iq, o], want_T[iq, o] = _ft_q(chi, lat, qf), _ft_q(chi.T, lat, qf)
    rel = lambda g, h: np.linalg.norm(g - h) / np.linalg.norm(h)
    for ordered, target, other in ((True, want, want_T), (False, want_T, want)):
        kernel = w_isdf._get_chi_fractional_contour_kernel_face(
            cpu_mesh, (N1, N2, 1), 2, (nk, NS, NS, 1), selected_q=tuple(range(nk)), ordered=ordered)
        got = np.asarray(kernel(*args)) / np.sqrt(nk)
        assert rel(got, target) < 1e-10
        assert rel(got, other) > 1e-2


def test_ordered_sigma_reproduces_real_space_igw_and_swapped_routing_does_not(cpu_mesh):
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    import gw.ppm_tau_kernel as tk
    from gw.mpa.sigma import shared_pole_hole_kernel, synthesize_shared_pole_parents

    lat = _lattice()
    nk, N, e, psi = lat["nk"], lat["N"], lat["e"], lat["psi"]
    blocks = _rpa_by_momentum(lat)
    minus = _minus()
    # Store in the physical orientation W_q = FT_q[W]: parent q holds its own momentum block, b = sqrt(2 Omega) c.
    parents = [(np.sqrt(2 * blocks[q][0])[None, :] * np.sqrt(nk) * blocks[q][1][:NS], blocks[q][0] ** 2) for q in range(nk)]
    width = max(b.shape[1] for b, _ in parents)
    factors = np.stack([np.pad(b, ((0, 0), (0, width - b.shape[1]))) for b, _ in parents])[:, :, None, :]
    poles2 = np.stack([np.pad(p2, (0, width - p2.size), constant_values=1.0) for _, p2 in parents])
    bounds = np.asarray([[0, b.shape[1]] for b, _ in parents], np.int32)
    mesh = cpu_mesh
    put = lambda x: _put(mesh, x)
    gemm = jax.jit(lambda x, y: x @ y, out_shardings=NamedSharding(mesh, P()))
    hole, minus_dev = shared_pole_hole_kernel(mesh), put(np.asarray(minus, np.int32))
    routing = dict(mode="production")

    def build(space, _omega, _indices, _bounds, _phase_real, E_ref_B, t):
        plus, _ = synthesize_shared_pole_parents(put(factors), put(factors), put(poles2), put(bounds),
                                                 E_ref_B, t, mesh_xy=mesh, gemm=gemm)
        cond_hole = routing["mode"] == "swapped"
        return hole(plus, minus_dev) if (space == "val") != cond_hole else plus

    kernel = tk.get_shared_sigma_tau_kernel(mesh_xy=mesh, kgrid=(N1, N2, 1), w_synthesis=build)
    cell = np.sqrt(nk) * psi[:, :, :NS]
    ops = [put(cell.transpose(0, 2, 1)[:, None]), put(cell[:, :, None]), put(cell[:, :, None]), put(cell.transpose(0, 2, 1)[:, None])]
    occ = np.zeros((nk, NS)); occ[:, :NOCC] = 1.0
    mu = 0.5 * (e[:, 0].max() + e[:, 1].min())
    t, e_ref_a, e_ref_b = 0.35 - 0.8j, 0.3, -0.2
    all_modes = [(om, vec) for om, vec in blocks.values()]
    for space in ("cond", "val"):
        sel = (occ == 0) if space == "cond" else (occ > 0)
        E_A = (e - mu) if space == "cond" else (mu - e)
        G = np.zeros((N, N), complex)
        for ik in range(nk):
            for n in range(NS):
                if sel[ik, n]:
                    G += np.outer(psi[ik, n], psi[ik, n].conj()) * np.exp(-1j * t * (E_A[ik, n] - e_ref_a))
        W = sum(((vec if space == "cond" else vec.conj()) * np.exp(-1j * (om - e_ref_b) * t)[None, :])
                @ (vec.conj().T if space == "cond" else vec.T) for om, vec in all_modes)
        hedin = np.diagonal(-np.einsum("kmr,rs,kns->kmn", psi.conj(), G * W, psi), axis1=-2, axis2=-1)
        results = {}
        for mode in ("production", "swapped"):
            routing["mode"] = mode
            got = np.asarray(kernel(*ops, put(E_A), put(sel), space, None, put(np.zeros(1, np.int32)),
                                    put(np.zeros((1, 6))), put(np.zeros(1, bool)), put(np.float64(e_ref_a)),
                                    put(np.float64(e_ref_b)), put(np.complex128(t))))[..., :NS, :NS]
            results[mode] = np.max(np.abs(np.diagonal(got, axis1=-2, axis2=-1) - hedin)) / np.max(np.abs(hedin))
        assert results["production"] < 1e-10, (space, results)
        assert results["swapped"] > 1e-2, (space, results)

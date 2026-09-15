"""A tiny time-reversal-broken tight-binding lattice with exact supercell RPA modes (test oracle only).

s sites per cell with generic complex Hermitian hoppings, H(-R) = H(R)^H, so inversion and time
reversal are both broken (e_k != e_-k). Bloch states live on the Born-von Karman supercell. The
supercell RPA pencil (z s3 - M) gives W_c(z) = C (z s3 - M)^-1 C^H, whose modes are
W_c(z) = sum_j R+_j/(z - Omega_j) - R-_j/(z + Omega_j), R+_j = a_j a_j^H, R-_j = conj(a_j) a_j^T.
Real-space convention: FT_q[f](mu, nu) = sum_R f(r_mu, r_nu + R) exp(+i q.R), mu, nu home-cell sites.
Source: the TRINT orientation plant (sandbox run 425_trint_20260915, harness/lattice_tr_broken.py).
"""
import numpy as np


class Lattice:
    def __init__(self, n1=3, n2=3, n_sites=3, n_occ=1, seed=20260915, hop=0.35, gap_shift=2.0):
        rng = np.random.default_rng(seed)
        self.n1, self.n2, self.ns, self.n_occ = n1, n2, n_sites, n_occ
        self.a = np.array([[1.0, 0.0], [0.5, np.sqrt(3) / 2]])
        self.tau = rng.uniform(0.1, 0.9, size=(n_sites, 2)) @ self.a
        onsite = np.diag(np.linspace(-gap_shift, gap_shift, n_sites) + 0.1 * rng.normal(size=n_sites))
        h0 = hop * (rng.normal(size=(n_sites, n_sites)) + 1j * rng.normal(size=(n_sites, n_sites)))
        self.hop = {(0, 0): onsite + 0.5 * (h0 + h0.conj().T)}
        for R in ((1, 0), (0, 1), (1, -1)):
            self.hop[R] = hop * (rng.normal(size=(n_sites, n_sites)) + 1j * rng.normal(size=(n_sites, n_sites)))
        self.kfrac = np.array([(i / n1, j / n2) for i in range(n1) for j in range(n2)])
        self.nk = n1 * n2
        self.cells = np.array([(i, j) for i in range(n1) for j in range(n2)])
        self.b = 2 * np.pi * np.linalg.inv(self.a).T

    def kvec(self, kf):
        return np.asarray(kf, float) @ self.b

    def bands(self):
        """Band energies [nk, nb] and cell eigenvectors [nk, s, nb]."""
        e, u = [], []
        for kf in self.kfrac:
            k = self.kvec(kf)
            h = np.zeros((self.ns, self.ns), complex)
            for R, block in self.hop.items():
                for sign, mat in ((1, block), (-1, block.conj().T)):
                    if R == (0, 0) and sign == -1:
                        continue
                    Rv = sign * (R[0] * self.a[0] + R[1] * self.a[1])
                    h += mat * np.exp(1j * ((Rv[None, None, :] + self.tau[None, :, :] - self.tau[:, None, :]) @ k))
            w, v = np.linalg.eigh(0.5 * (h + h.conj().T))
            e.append(w)
            u.append(v)
        return np.asarray(e), np.asarray(u)

    def site_positions(self):
        """Supercell sites in canonical order: index = cell_index * ns + s."""
        return np.asarray([c[0] * self.a[0] + c[1] * self.a[1] + self.tau[s]
                           for c in self.cells for s in range(self.ns)])

    def bloch_states(self, e, u):
        """psi[nk, nb, N] on the supercell sites, normalized over the supercell."""
        pos = self.site_positions()
        site_of = np.tile(np.arange(self.ns), self.nk)
        return np.stack([np.exp(1j * pos @ self.kvec(kf))[None, :] * u[ik][site_of, :].T
                         for ik, kf in enumerate(self.kfrac)]) / np.sqrt(self.nk)

    def coulomb(self, v0=1.0, screen=0.6):
        """Real, symmetric, positive-definite, translation-invariant (minimum image) interaction."""
        pos = self.site_positions()
        L = np.array([self.n1 * self.a[0], self.n2 * self.a[1]])
        v = np.zeros((len(pos), len(pos)))
        for i in range(len(pos)):
            d = pos - pos[i]
            best = np.min([np.linalg.norm(d + m1 * L[0] + m2 * L[1], axis=1)
                           for m1 in (-1, 0, 1) for m2 in (-1, 0, 1)], axis=0)
            v[i] = v0 * np.exp(-best / screen) / (1.0 + best)
        v = 0.5 * (v + v.T)
        w = np.linalg.eigvalsh(v)
        return v + max(0.0, 1e-3 - w[0]) * np.eye(len(pos))


def rpa_modes(psi, e, n_occ, v):
    """Exact supercell RPA modes: Omega_j > 0, residue vectors a [N, J], and the pencil for checks."""
    nk, nb, _ = psi.shape
    occ = [(ik, n) for ik in range(nk) for n in range(n_occ)]
    emp = [(ik, n) for ik in range(nk) for n in range(n_occ, nb)]
    rho = np.array([np.conj(psi[i]) * psi[a] for i in occ for a in emp]).T
    D = np.array([e[a] - e[i] for i in occ for a in emp])
    phi = np.hstack([rho, rho.conj()])
    M = np.diag(np.r_[D, D]).astype(complex) + phi.conj().T @ v @ phi
    M = 0.5 * (M + M.conj().T)
    s3 = np.diag(np.r_[np.ones(len(D)), -np.ones(len(D))]).astype(complex)
    # Positive modes through the Hermitian form M^1/2 s3 M^1/2 y = w y, so degenerate
    # multiplets stay s3-orthogonal.
    lam, U = np.linalg.eigh(M)
    assert lam[0] > 0, "unstable RPA"
    half, inv_half = (U * np.sqrt(lam)) @ U.conj().T, (U / np.sqrt(lam)) @ U.conj().T
    w, y = np.linalg.eigh(half @ s3 @ half)
    keep = w > 0
    a = (v @ phi @ inv_half @ y[:, keep]) * np.sqrt(w[keep])[None, :]
    order = np.argsort(w[keep])
    return w[keep][order], a[:, order], dict(c=v @ phi, s3=s3, M=M)


def w_c(z, pencil):
    """W_c(z) = C (z s3 - M)^-1 C^H."""
    c, s3, M = pencil["c"], pencil["s3"], pencil["M"]
    return c @ np.linalg.solve(z * s3 - M, c.conj().T)


def ft_q(f, lat, q_frac):
    """FT_q[f](mu, nu) = sum_R f(r_mu, r_nu + R) exp(+i q.R) for f on the supercell sites."""
    q = lat.kvec(q_frac)
    ns = lat.ns
    return sum(f[:ns, c * ns:(c + 1) * ns] * np.exp(1j * (cell[0] * lat.a[0] + cell[1] * lat.a[1]) @ q)
               for c, cell in enumerate(lat.cells))


def minus_index(lat):
    """Canonical flat index of -q for every q."""
    return [((-i) % lat.n1) * lat.n2 + ((-j) % lat.n2) for i in range(lat.n1) for j in range(lat.n2)]


def mode_momenta(lat, a):
    """Momentum index of each supercell mode vector a[:, j], a(r + R) = exp(i p.R) a(r), and its residual."""
    ns, home = lat.ns, a[:lat.ns]
    shifts = [np.exp(1j * (cell[0] * lat.a[0] + cell[1] * lat.a[1]) @ lat.kvec(pf))
              for pf in lat.kfrac for cell in lat.cells]
    phases = np.asarray(shifts).reshape(lat.nk, lat.nk)              # [p, cell]
    index, residual = [], []
    for j in range(a.shape[1]):
        rows = [np.linalg.norm(np.kron(phases[p], home[:, j]) - a[:, j]) / np.linalg.norm(a[:, j])
                for p in range(lat.nk)]
        index.append(int(np.argmin(rows)))
        residual.append(float(min(rows)))
    assert ns == home.shape[0]
    return np.asarray(index), np.asarray(residual)


def cpu_flat_k_fft(monkeypatch):
    """Route the flat-k FFT factories through the jnp emulation the CPU tests use."""
    import jax.numpy as jnp
    import common.fft_helpers as fh

    def emulated(maker):
        def factory(mesh_, kgrid, spec, *, norm="ortho", out_spec=None):
            fft3 = maker(mesh_, spec, spec, axes=(0, 1, 2), norm=norm)
            return lambda x: fft3(jnp.reshape(x, tuple(kgrid) + x.shape[1:])).reshape(x.shape)
        return factory

    monkeypatch.setattr(fh, "make_flat_k_fftn", emulated(fh.make_sharded_fftn_3d))
    monkeypatch.setattr(fh, "make_flat_k_ifftn", emulated(fh.make_sharded_ifftn_3d))

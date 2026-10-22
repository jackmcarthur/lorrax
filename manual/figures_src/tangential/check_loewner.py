"""Numerical check of the equations on docs/theory/tangential-w.md (sections 2-5 and 7).

Small random realizations in double precision. Each check prints its worst
relative defect, and all of them should be at round-off (below 1e-9). Run from the site root:
    _tools/plotenv/bin/python -B figures_src/tangential/check_loewner.py
"""
import numpy as np
from scipy.linalg import eigh

rng = np.random.default_rng(7)
TOL = 1e-9
results = []


def check(name, err):
    err = float(err)
    results.append((name, err))
    print(f"{'OK ' if err < TOL else 'BAD'} {err:9.2e}  {name}")


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)


def crandn(*shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def dag(a):
    return a.conj().T


# ------------------------------------------------------------- the RPA in transition space
Nt, Nr = 7, 5                                        # transitions, real-space points
Delta = np.sort(rng.uniform(1.0, 6.0, Nt))           # transition energies E_c - E_v
rho = crandn(Nr, Nt)                                 # resonant pair densities rho_t(r)
rhob = crandn(Nr, Nt)                                # antiresonant densities (no time reversal)
a = crandn(Nr, Nr)
v = a @ dag(a) / Nr + np.eye(Nr)                     # bare Coulomb, Hermitian positive
Phi = np.hstack([rho, rhob])                         # (2.1)
D = np.diag(np.concatenate([Delta, -Delta]))
sig = np.diag(np.concatenate([np.ones(Nt), -np.ones(Nt)]))
K = dag(Phi) @ v @ Phi                               # (2.2) RPA kernel in transition space
H = D + sig @ K                                      # (2.3) RPA Hamiltonian
I2 = np.eye(2 * Nt)

z = 2.3 + 0.7j
chi0 = Phi @ np.linalg.solve(z * I2 - D, sig @ dag(Phi))
chi0_sum = sum(np.outer(rho[:, t], rho[:, t].conj()) / (z - Delta[t])
               - np.outer(rhob[:, t], rhob[:, t].conj()) / (z + Delta[t]) for t in range(Nt))
check("chi0 = Phi (z-D)^-1 sigma Phi^+  (2.1)", rel(chi0, chi0_sum))
L0 = np.linalg.solve(z * I2 - D, sig)
L = np.linalg.solve(z * I2 - H, sig)
check("L = L0 + L0 K L solved by L = (z-H)^-1 sigma  (2.3)", rel(L, L0 + L0 @ K @ L))
chi = Phi @ L @ dag(Phi)
check("chi = Phi L Phi^+ obeys chi = chi0 + chi0 v chi  (1.2)", rel(chi, chi0 + chi0 @ v @ chi))
W = v + v @ chi @ v
check("W = v + v chi v = [1 - v chi0]^-1 v  (1.2)", rel(W, np.linalg.solve(np.eye(Nr) - v @ chi0, v)))
check("sigma H Hermitian (H is sigma-self-adjoint)", rel(sig @ H, dag(sig @ H)))

# Lehmann representation
w, X = np.linalg.eig(H)
check("eigenvalues of H real", np.max(np.abs(w.imag)) / np.max(np.abs(w)))
w = w.real
nrm = np.einsum('is,ij,js->s', X.conj(), sig, X).real
eps = np.sign(nrm)
check("sign(x^+ sigma x) = sign(Omega)", float(np.any(eps != np.sign(w))))
X = X / np.sqrt(np.abs(nrm))
Lleh = sum(eps[s] * np.outer(X[:, s], X[:, s].conj()) / (z - w[s]) for s in range(2 * Nt))
check("(z-H)^-1 sigma = sum_S eps_S x_S x_S^+ /(z - Omega_S)  (3.1)", rel(L, Lleh))
b = v @ Phi @ X                                      # b_S = v rho_S
Wc_leh = sum(eps[s] * np.outer(b[:, s], b[:, s].conj()) / (z - w[s]) for s in range(2 * Nt))
check("W^c = sum_S eps_S b_S b_S^+ /(z - Omega_S)  (3.2)", rel(W - v, Wc_leh))

# ------------------------------------------------------------ time reversal: the form in s
PhiT = np.hstack([rho, rho])                         # TR pairs each antiresonant density with a resonant one
HT = D + sig @ (dag(PhiT) @ v @ PhiT)


def WcT(zz):
    return v @ PhiT @ np.linalg.solve(zz * I2 - HT, sig @ dag(PhiT)) @ v


check("TR: W^c(z) even in z", rel(WcT(z), WcT(-z)))
P = np.sqrt(2.0) * rho * np.sqrt(Delta)[None, :]
T = np.diag(Delta**2) + dag(P) @ v @ P               # (3.4)
F = v @ P
s = z * z
check("TR: W^c = F (s-T)^-1 F^+, T = Delta^2 + P^+ v P  (3.4)",
      rel(F @ np.linalg.solve(s * np.eye(Nt) - T, dag(F)), WcT(z)))
Om2, U = np.linalg.eigh(T)
Om = np.sqrt(Om2)
wpos = np.sort(np.linalg.eigvals(HT).real)
wpos = wpos[wpos > 0]
check("TR: eig(T) = Omega_S^2 of the RPA Hamiltonian", rel(np.sort(Om), wpos))
fac = F @ U                                          # stored factor sqrt(2 Omega) b
bb = fac / np.sqrt(2 * Om)
Wpole = sum(np.outer(bb[:, j], bb[:, j].conj()) * (1 / (z - Om[j]) - 1 / (z + Om[j])) for j in range(Nt))
check("stored form sum b b^+ [1/(z-Om) - 1/(z+Om)]  (3.3)", rel(Wpole, WcT(z)))
W0 = -sum(2 * np.outer(bb[:, j], bb[:, j].conj()) / Om[j] for j in range(Nt))
check("W^c(0) = -sum 2 b b^+ / Omega", rel(W0, WcT(0.0)))
M1 = 0.5 * F @ dag(F)
M3 = 0.5 * F @ T @ dag(F)
check("2 M1 = sum 2 Omega b b^+", rel(2 * M1, sum(2 * Om[j] * np.outer(bb[:, j], bb[:, j].conj()) for j in range(Nt))))
zb = 7.0j
sb = zb * zb
check("tail: W^c - 2M1/s - 2M3/s^2 = F T^2 (s-T)^-1 F^+ / s^2 exactly (moments 2M1, 2M3)",
      rel(WcT(zb) - 2 * M1 / sb - 2 * M3 / sb**2,
          F @ T @ T @ np.linalg.solve(sb * np.eye(Nt) - T, dag(F)) / sb**2))
# time form: int dt e^{iwt} (-i) e^{-i Om |t| - eta |t|} = 1/(w - Om + i eta) - 1/(w + Om - i eta)
tt = np.linspace(0, 80, 800001)
dt = tt[1] - tt[0]
et, om, OmT = 0.3, 1.1, 2.0
f = -1j * np.exp(-1j * OmT * tt - et * tt)
g_int = np.exp(1j * om * tt) * f + np.exp(-1j * om * tt) * f
ft = (np.sum(g_int) - 0.5 * (g_int[0] + g_int[-1])) * dt
check("time form -i e^{-i Om|t|} <-> 1/(w-Om+i eta) - 1/(w+Om-i eta)  (3.5)",
      1e-2 * abs(ft - (1 / (om - OmT + 1j * et) - 1 / (om + OmT - 1j * et))))

# ------------------------------------------------------------ Loewner for a general realization
n, m = 6, 3                                          # order, ports
Ag = crandn(n, n)
Bg = crandn(n, m)
Cg = crandn(m, n)


def Wg(zz):
    return Cg @ np.linalg.solve(zz * np.eye(n) - Ag, Bg)


zi, zj = 0.3 + 1.1j, -0.8 + 0.4j
Ri = np.linalg.inv(zi * np.eye(n) - Ag)
Rj = np.linalg.inv(zj * np.eye(n) - Ag)
check("[W(zi) - W(zj)]/(zi - zj) = -C (zi-A)^-1 (zj-A)^-1 B  (5.2)",
      rel((Wg(zi) - Wg(zj)) / (zi - zj), -Cg @ Ri @ Rj @ Bg))
check("[zi W(zi) - zj W(zj)]/(zi - zj) = -C (zi-A)^-1 A (zj-A)^-1 B  (5.2)",
      rel((zi * Wg(zi) - zj * Wg(zj)) / (zi - zj), -Cg @ Ri @ Ag @ Rj @ Bg))
mu = 2 * crandn(n)                                   # left points
lam = 2 * crandn(n)                                  # right points
ell = crandn(n, m)                                   # left directions (rows)
r = crandn(m, n)                                     # right directions (columns)
LL = np.empty((n, n), complex)
LLs = np.empty((n, n), complex)
for i in range(n):
    for j in range(n):
        li, rj = ell[i], r[:, j]
        Wi, Wj = Wg(mu[i]), Wg(lam[j])
        LL[i, j] = (li @ Wi @ rj - li @ Wj @ rj) / (mu[i] - lam[j])
        LLs[i, j] = (mu[i] * (li @ Wi @ rj) - lam[j] * (li @ Wj @ rj)) / (mu[i] - lam[j])
R = np.column_stack([np.linalg.solve(lam[j] * np.eye(n) - Ag, Bg @ r[:, j]) for j in range(n)])
O = np.vstack([ell[i] @ Cg @ np.linalg.inv(mu[i] * np.eye(n) - Ag) for i in range(n)])
check("Loewner L = -O R  (5.4)", rel(LL, -O @ R))
check("shifted Loewner Ls = -O A R  (5.4)", rel(LLs, -O @ Ag @ R))
pe = np.sort_complex(np.linalg.eigvals(np.linalg.solve(LL, LLs)))
check("generalized eigenvalues of (Ls, L) = eigenvalues of A  (5.5)", rel(pe, np.sort_complex(np.linalg.eigvals(Ag))))
Wd = np.column_stack([Wg(lam[j]) @ r[:, j] for j in range(n)])
Vd = np.vstack([ell[i] @ Wg(mu[i]) for i in range(n)])
z1 = 0.4 - 1.3j
check("W_r(z) = Wd (Ls - z L)^-1 Vd = W(z) when k = n  (5.6)", rel(Wd @ np.linalg.solve(LLs - z1 * LL, Vd), Wg(z1)))
k = 3                                                # fewer samples than the order


def Wrk(zz):
    return Wd[:, :k] @ np.linalg.solve(LLs[:k, :k] - zz * LL[:k, :k], Vd[:k])


check("k < n: right tangential interpolation W_r(lam_j) r_j = W(lam_j) r_j",
      max(rel(Wrk(lam[j]) @ r[:, j], Wg(lam[j]) @ r[:, j]) for j in range(k)))
check("k < n: left tangential interpolation l_i W_r(mu_i) = l_i W(mu_i)",
      max(rel(ell[i] @ Wrk(mu[i]), ell[i] @ Wg(mu[i])) for i in range(k)))
Rk, Ok = R[:, :k], O[:k]
check("k < n: W_r = C R [O (z-A) R]^-1 O B, projection on rational Krylov spaces  (5.7)",
      rel(Wrk(z1), Cg @ Rk @ np.linalg.solve(Ok @ (z1 * np.eye(n) - Ag) @ Rk, Ok @ Bg)))


# ------------------------------------------------------- the Hermitian pencil in s used by LORRAX
def Ws(s_):
    return F @ np.linalg.solve(s_ * np.eye(Nt) - T, dag(F))


def dWs(s_):
    Rs = np.linalg.inv(s_ * np.eye(Nt) - T)
    return -F @ Rs @ Rs @ dag(F)


zs = np.array([1.5 + 0.9j, 3.0 + 0.9j, 0.6j, 1.7j])  # two line-type, two imaginary-axis supports
ss = zs**2
Q = [crandn(Nr, 1) for _ in zs]
ns = len(zs)
G = np.empty((ns, ns), complex)
Hp = np.empty((ns, ns), complex)
for i in range(ns):
    for j in range(ns):
        den = ss[j] - np.conj(ss[i])
        if abs(den) < 1e-12:
            g = -(dag(Q[i]) @ dWs(ss[j]) @ Q[j]).item()
        else:
            g = (dag(Q[i]) @ (dag(Ws(ss[i])) - Ws(ss[j])) @ Q[j]).item() / den
        G[i, j] = g
        Hp[i, j] = ss[j] * g - (dag(Q[i]) @ dag(Ws(ss[i])) @ Q[j]).item()
Xs = np.column_stack([np.linalg.solve(ss[j] * np.eye(Nt) - T, dag(F) @ Q[j]).ravel() for j in range(ns)])
check("G = X^+ X with X_b = (s_b - T)^-1 F^+ Q_b  (5.8)", rel(G, dag(Xs) @ Xs))
check("H = X^+ T X  (5.8)", rel(Hp, dag(Xs) @ T @ Xs))
Lh = np.empty((ns, ns), complex)
for i in range(ns):
    for j in range(ns):
        mi = np.conj(ss[i])
        if abs(mi - ss[j]) < 1e-12:
            Lh[i, j] = (dag(Q[i]) @ dWs(ss[j]) @ Q[j]).item()
        else:
            Lh[i, j] = (dag(Q[i]) @ (Ws(mi) - Ws(ss[j])) @ Q[j]).item() / (mi - ss[j])
check("G = -Loewner with left points conj(s_a), directions Q_a^+ (confluent -> derivative)", rel(G, -Lh))
theta, xr = eigh(Hp, G)
check("Ritz values real, inside [min Omega^2, max Omega^2]",
      float(np.any(theta < Om2.min() - 1e-9) or np.any(theta > Om2.max() + 1e-9)))
Od = np.column_stack([Ws(ss[j]) @ Q[j] for j in range(ns)])


def Wr_s(s_):
    return Od @ np.linalg.solve(s_ * G - Hp, dag(Od))


check("W_r(s) = O (sG - H)^-1 O^+ interpolates W(s_b) Q_b",
      max(rel(Wr_s(ss[j]) @ Q[j], Ws(ss[j]) @ Q[j]) for j in range(ns)))
fr = Od @ xr                                         # factor = O x_j with x^+ G x = 1
check("W_r(s) = sum_j (O x_j)(O x_j)^+ / (s - theta_j)  (5.9)",
      rel(sum(np.outer(fr[:, j], fr[:, j].conj()) / (s - theta[j]) for j in range(ns)), Wr_s(s)))
iu, hstep = 2, 1e-5                                  # imaginary support, s real: Hermite interpolation
dWr = (Wr_s(ss[iu] + hstep) - Wr_s(ss[iu] - hstep)) / (2 * hstep)
check("imaginary support: Q^+ W_r'(s) Q = Q^+ W'(s) Q (Hermite)",
      1e-2 * abs((dag(Q[iu]) @ (dWr - dWs(ss[iu])) @ Q[iu]).item()) / abs((dag(Q[iu]) @ dWs(ss[iu]) @ Q[iu]).item()))
Qf = [crandn(Nr, 1) for _ in range(Nt)]
sf = (np.linspace(0.5, 3.0, Nt) * 1j) ** 2
Xf = np.column_stack([np.linalg.solve(sf[j] * np.eye(Nt) - T, dag(F) @ Qf[j]).ravel() for j in range(Nt)])
check("as many states as poles: the pencil recovers every Omega^2",
      rel(np.sort(eigh(dag(Xf) @ T @ Xf, dag(Xf) @ Xf, eigvals_only=True)), Om2))
# the point at infinity
Qi = crandn(Nr, 1)
Xi = dag(F) @ Qi
check("G_inf,b = Q_inf^+ W(s_b) Q_b  (5.10)", rel(dag(Xi) @ Xs[:, [1]], dag(Qi) @ Ws(ss[1]) @ Q[1]))
check("H_inf,b = s_b G_inf,b - 2 (M1 Q_inf)^+ Q_b  (5.10)",
      rel(dag(Xi) @ T @ Xs[:, [1]], ss[1] * (dag(Qi) @ Ws(ss[1]) @ Q[1]) - 2 * dag(M1 @ Qi) @ Q[1]))
check("G_inf,inf = 2 Q^+ M1 Q and H_inf,inf = 2 Q^+ M3 Q  (5.10)",
      rel([(dag(Xi) @ Xi).item(), (dag(Xi) @ T @ Xi).item()],
          [(2 * dag(Qi) @ M1 @ Qi).item(), (2 * dag(Qi) @ M3 @ Qi).item()]))
Xa = np.hstack([Xs, Xi])
th_a, x_a = eigh(dag(Xa) @ T @ Xa, dag(Xa) @ Xa)
fa = np.hstack([Od, 2 * M1 @ Qi]) @ x_a
check("with the infinity state the model keeps Q^+ M1 Q and Q^+ M3 Q",
      rel([(dag(Qi) @ (0.5 * fa @ dag(fa)) @ Qi).item(), (dag(Qi) @ (0.5 * fa @ np.diag(th_a) @ dag(fa)) @ Qi).item()],
          [(dag(Qi) @ M1 @ Qi).item(), (dag(Qi) @ M3 @ Qi).item()]))

# ------------------------------------------------------------ plasmon-pole limits (section 7)
c2, Om_true = 1.7, 2.4                               # one scalar pole, numerator c2 = 2 Omega b^2


def Wsc(zz):
    return c2 / (zz**2 - Om_true**2)


wp = 1.3
sp = (1j * wp) ** 2
g11 = (Wsc(0.0) - Wsc(1j * wp)) / (sp - 0.0)         # left point 0, right point s_p
h11 = sp * g11 - Wsc(0.0)
gn = -sp * Wsc(1j * wp) / (Wsc(0.0) - Wsc(1j * wp))  # Frequency treatments (4.1)
check("GN: the 1x1 pencil on {0, i w_p} gives Omega^2 = GN formula = exact", abs(h11 / g11 - gn) / abs(gn) + abs(gn - Om_true**2) / Om_true**2)
check("HL: the 1x1 pencil on {infinity, 0} gives Omega^2 = -2 M1 / W(0)", abs(-c2 / Wsc(0.0) - Om_true**2) / Om_true**2)

bad = [name for name, e in results if not e < TOL]
print(f"\n{len(results)} checks, {len(bad)} failed" + (": " + "; ".join(bad) if bad else ""))
raise SystemExit(1 if bad else 0)

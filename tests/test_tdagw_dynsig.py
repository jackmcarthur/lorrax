"""The instantaneous GW self-energy (tdagw.dynsig) against an independent dense reference.

A synthetic bundle (k grid 3x2x1, 9 bands = 3 frozen occupied + a 2v2c active
window + 2 frozen empty, two spinor components, 7 centroids padded to the
mesh) is compared with a NumPy evaluation of docs/theory/td_agw_dynamical.md
that shares no code with tdagw.dynsig (``_dense_*``: the Adler-Wiser chi0
over ordered band pairs, the Dyson solve, the COHSEX self-energy as explicit
sums; the TD kernel as in tests/test_tdagw_model.py).

Runs as a script on a 2x2 mesh, every k-convolution built inside
``ffi.fft.xla_reference()``.  CPU, one process with four host devices::

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \\
        python3 tests/test_tdagw_dynsig.py

CUDA, four ranks with one GPU each (the runtime's communicator stack)::

    lx run -N 1 -G 4 -n 4 -- python3 tests/test_tdagw_dynsig.py

Under pytest it skips.  Gates (G) and red twins (R); each twin feeds a
deliberately wrong ingredient to the SAME checker as the gate it shadows:

G1  chi0^ff + chi0^fa + chi0^aa (t = 0, the run's rule) == dense chi0     rel <= 1e-8
G2  Sigma^SX, Sigma^COH at rho_eq from W0 == dense COHSEX                 rel <= 1e-12
G3  dchi0 of an excited state (holes, carriers, unequal same-side
    populations, coherences) == dense chi0(t) - chi0(0)                  rel <= 1e-7
G4  dSigma (static_rpa, dress none) == dense Sigma[rho] - Sigma[rho_eq]   rel <= 1e-7
G4b the same with dress = instantaneous: the population-dressed basis, the
    fixed point of diag(eps^G) + K drho_pop (drho_pop = U diag(f) U^H - rho_eq),
    iterated to convergence in NumPy                                     rel <= 1e-7
G4c the same on a hot-carrier state (populations only, up to 0.39)       rel <= 1e-7
G5  rho = rho_eq gives dSigma == 0 exactly;  G6 dSigma_k Hermitian        <= 1e-12
G7  literal Eq. (1) in a rotated instantaneous basis, frozen W:
    Sigma^SX[rho] - Sigma^SX[rho_eq] == -K^W drho (TD model)              rel <= 1e-12
G8  frozen mode through the propagator on a pulse == isdf_model           rel <= 1e-13
G9  static_rpa (cross-gap member, same_side = False) through the propagator:
    finite, and its departure from the frozen run scales as E0^2 (dW ~
    populations ~ E0^2): log2 slope of max|P_rpa - P_frozen|, E0 -> 2 E0  in [1.6, 2.4]
D9  (diagnostic, not a gate) the same with the exact same-side pairs: the
    static carrier screening (f_a - f_b)/(E_a - E_b) of near-degenerate
    pairs at different k makes the departure grow faster than E0^2
G10 linear response: drho = V rho_eq V^H - rho_eq, V = exp(-i eps A) (A
    valence-conduction), eps = 2.5, 5, 10 x 1e-4: the dynamic correction with
    dress = instantaneous scales as eps^2 (fit slope in [1.9, 2.1]), and on the
    pure coherence drho = eps X, X = -i[A, rho_eq], it is round-off (<= 1e-9 of
    the coherent twin's)
G11 the dressing's fixed-point residual after DRESS_STEPS steps on the driven
    state of G3 and on the hot-carrier state of G4c                      <= 1e-9
D10 (diagnostic, not a gate) the slope of G10 with dress = none
R1  populations read in the fixed basis (occupation = fixed_basis) fails G4b
R6  the coherent dressing (dress = coherent: diag(eps^G) + K drho, the whole drho)
    fails G4b
R7  the coherent dressing scales as eps (slope ~1) and fails G10
R2  chi0^fa dropped fails G1
R3  the Pauli-blocking product dropped fails G3
R4  coherences dropped from G^< fails G7
R5  the exact same-side pairs dropped fails G3
"""
import json
import sys

import runtime  # noqa: F401  (before numpy: the runtime tunes BLAS spin at import)
import numpy as np

KGRID = (3, 2, 1)
NBF, NS, NMU, NG = 9, 2, 7, 13
B_LO, NA = 3, 4                 # window [3, 7): 2 valence + 2 conduction
NOCC = 5


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))


def _synthetic(rng):
    nk = int(np.prod(KGRID))
    c = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)
    psi = 0.6 * c(nk, NBF, NS, NMU)
    e = np.empty((nk, NBF))
    for k in range(nk):
        e[k, :NOCC] = np.sort(rng.uniform(-1.2, -0.25, NOCC))
        e[k, NOCC:] = np.sort(rng.uniform(0.25, 1.4, NBF - NOCC))
    f = np.zeros((nk, NBF)); f[:, :NOCC] = 1.0
    A = c(nk, NMU, NMU)
    # scales of a screened insulator: |V chi0| ~ 10, |K drho| / |drho| < 1 Ry
    V = 0.05 * (np.einsum("qij,qkj->qik", A, np.conj(A)) / NMU + 0.5 * np.eye(NMU)[None])
    half = (NG - 1) // 2
    zeta = c(NMU, NG)
    zeta[:, 0] = zeta[:, 0].real
    zeta[:, 1 + half:] = np.conj(zeta[:, 1:1 + half])
    v = 0.005 * (rng.random(NG) + 0.1)
    v[1 + half:] = v[1:1 + half]
    v[0] = 0.0
    return psi, e, f, V, zeta, v


def _kmq():
    nk = int(np.prod(KGRID))
    ijk = np.stack(np.unravel_index(np.arange(nk), KGRID), axis=1)
    return np.stack([np.ravel_multi_index(tuple(((ijk - ijk[q]) % np.asarray(KGRID)).T), KGRID)
                     for q in range(nk)])


def _dense_chi(psi, e, f, *, x_sep=0.0, mode="all"):
    """chi_q(mu,nu) = (1/N_k) sum_k sum_{a in k, b in k-q} (f_a - f_b)/(E_a - E_b) rho_ab rho_ab^*.

    mode: "all" | "cross" (opposite sides of x_sep only) — g_s = 1 (spinors).
    """
    nk = psi.shape[0]
    kmq = _kmq()
    chi = np.zeros((nk, NMU, NMU), complex)
    for q in range(nk):
        for k in range(nk):
            kb = kmq[q, k]
            rho = np.einsum("asu,bsu->abu", psi[k], np.conj(psi[kb]))
            dE = e[k][:, None] - e[kb][None, :]
            df = f[k][:, None] - f[kb][None, :]
            ok = np.abs(dE) > 1e-12
            if mode == "cross":
                ok &= ((e[k][:, None] < x_sep) != (e[kb][None, :] < x_sep))
            cab = np.where(ok, df / np.where(ok, dE, 1.0), 0.0)
            chi[q] += np.einsum("abu,ab,abv->uv", rho, cab, np.conj(rho))
    return chi / nk


def _dense_w_apply(psi, Wq, X):
    """(w X)_{k,ij} = (1/N_k) sum_{k' n m} [psi*_ik psi_nk'](mu) W_{k-k'}(mu,nu) [psi*_mk' psi_jk](nu) X_{k',nm}."""
    nk = psi.shape[0]
    out = np.zeros((nk, psi.shape[1], psi.shape[1]), complex)
    ijk = np.stack(np.unravel_index(np.arange(nk), KGRID), axis=1)
    for k in range(nk):
        for kp in range(nk):
            q = np.ravel_multi_index(tuple((ijk[k] - ijk[kp]) % np.asarray(KGRID)), KGRID)
            Aa = np.einsum("itu,ntu->inu", np.conj(psi[k]), psi[kp])
            Bb = np.einsum("msv,jsv->mjv", np.conj(psi[kp]), psi[k])
            out[k] += np.einsum("inu,uv,mjv,nm->ij", Aa, Wq[q], Bb, X[kp])
    return out / nk


def _dense_dyson(V, chi):
    return np.stack([np.linalg.solve(np.eye(NMU) - V[q] @ chi[q], V[q]) for q in range(len(V))])


def _dense_hartree(psi_w, zeta, v, x):
    nk = psi_w.shape[0]
    M = np.einsum("knsu,kmsu->knmu", np.conj(psi_w), psi_w)
    Vbar = np.einsum("uG,G,vG->uv", np.conj(zeta), v, zeta)
    dn = np.einsum("knmu,knm->u", np.conj(M), x)
    return np.einsum("knmu,uv,v->knm", M, Vbar, dn) / nk


def _dense_sigma(psi, e, f_full_t, rho_win, W, V):
    """Window COHSEX Sigma^SX + Sigma^COH for rho_tot = diag(f_frozen) (+) rho_win and W."""
    nk = psi.shape[0]
    X = np.zeros((nk, NBF, NBF), complex)
    idx = np.arange(NBF)
    X[:, idx, idx] = f_full_t
    X[:, B_LO:B_LO + NA, B_LO:B_LO + NA] = rho_win
    one = np.broadcast_to(np.eye(NBF), (nk, NBF, NBF)).astype(complex)
    S = -_dense_w_apply(psi, W, X) + 0.5 * _dense_w_apply(psi, W - V, one)
    return S[:, B_LO:B_LO + NA, B_LO:B_LO + NA]


def _excited(rng, nk):
    """drho: holes in both valence bands, carriers in both conduction bands with unequal
    populations (same-side pairs), and coherences; Hermitian, model order."""
    x = np.zeros((nk, NA, NA), complex)
    pops = np.array([-0.06, -0.035, 0.05, 0.045])
    for k in range(nk):
        x[k, np.arange(NA), np.arange(NA)] = pops * (1.0 + 0.3 * rng.standard_normal(NA))
    g = 0.04 * (rng.standard_normal((nk, NA, NA)) + 1j * rng.standard_normal((nk, NA, NA)))
    g = g - np.einsum("kii->ki", g)[:, :, None] * np.eye(NA)[None]
    return x + g + np.conj(np.swapaxes(g, -1, -2))


def _dense_kernel(psi_w, zeta, v, W0):
    """K x = K^H x - K^W x on the window (Ry): the TD-aGW kernel of the frozen W0."""
    return lambda x: _dense_hartree(psi_w, zeta, v, x) - _dense_w_apply(psi_w, W0, x)


def _dense_dressed(e_w, f_w, x, K, tol=1e-15, cap=4000):
    """The population-dressed basis by plain iteration to convergence: U, e, f, (steps, residual).

    h = diag(e_w) + K drho_pop, drho_pop = U diag(f) U^H - rho_eq, f = Re diag(U^H rho U).
    """
    rho_eq = np.stack([np.diag(fk) for fk in f_w]).astype(complex)
    rho = rho_eq + x
    h0 = np.stack([np.diag(ek) for ek in e_w]).astype(complex)
    P = np.stack([np.diag(np.real(np.diag(xk))) for xk in x]).astype(complex)
    for n in range(1, cap + 1):
        ev, U = np.linalg.eigh(h0 + K(P))
        fr = np.real(np.einsum("kmi,kmn,kni->ki", np.conj(U), rho, U))
        Pn = np.einsum("kmi,ki,kni->kmn", U, fr, np.conj(U)) - rho_eq
        r, P = float(np.max(np.abs(Pn - P))), Pn
        if r < tol:
            break
    ev, U = np.linalg.eigh(h0 + K(P))
    return U, ev, np.real(np.einsum("kmi,kmn,kni->ki", np.conj(U), rho, U)), (n, r)


def _hot(rng, nk):
    """drho of hot carriers: populations only (holes up to 0.39, carriers up to 0.36)."""
    pops = np.array([-0.30, -0.22, 0.28, 0.24])
    x = np.zeros((nk, NA, NA), complex)
    for k in range(nk):
        x[k, np.arange(NA), np.arange(NA)] = pops * (1.0 + 0.3 * np.tanh(rng.standard_normal(NA)))
    return x


def _drive_family(rng, nk, f_w):
    """(A, unitary(eps), X): A Hermitian valence-conduction (nk, NA, NA); unitary(eps) =
    V rho_eq V^H - rho_eq with V = exp(-i eps A); X = -i[A, rho_eq], its O(eps) part."""
    nv = int(np.sum(f_w[0] > 0.5))
    A = np.zeros((nk, NA, NA), complex)
    b = rng.standard_normal((nk, nv, NA - nv)) + 1j * rng.standard_normal((nk, nv, NA - nv))
    A[:, :nv, nv:] = b
    A[:, nv:, :nv] = np.conj(np.swapaxes(b, -1, -2))
    rho_eq = np.stack([np.diag(fk) for fk in f_w]).astype(complex)

    def unitary(eps):
        w, Q = np.linalg.eigh(A)
        Vu = np.einsum("kab,kb,kcb->kac", Q, np.exp(-1j * eps * w), np.conj(Q))
        return np.einsum("kab,kbc,kdc->kad", Vu, rho_eq, np.conj(Vu)) - rho_eq
    return A, unitary, -1j * (A @ rho_eq - rho_eq @ A)


def run_checks(mesh):
    import jax
    import jax.numpy as jnp
    from jax.experimental import multihost_utils
    from jax.sharding import NamedSharding, PartitionSpec as P
    from lxkit import device_put_process_local
    from ffi import fft as F
    from runtime.padding import padded_mu_extent, round_up
    from tdagw.model import ZETA_G_SPEC, V_G_SPEC, assemble_td_model, build_td_agw_apply
    from tdagw import dynsig as D
    from tdagw import propagate as tp

    rng = np.random.default_rng(20261016)
    psi, e, f, V, zeta, v = _synthetic(rng)
    nk = int(np.prod(KGRID))
    p = int(np.prod(mesh.devices.shape))
    nmu_pad = padded_mu_extent(NMU, mesh)
    nG_pad = round_up(NG, p)
    put = lambda a, spec: device_put_process_local(np.asarray(a), NamedSharding(mesh, spec))
    gather = lambda x: np.asarray(multihost_utils.process_allgather(x, tiled=True))
    rows = {}

    with F.xla_reference():
        rule = D.static_rule(e, f, target_error=1e-11, max_nodes=64)
        chi_eq = _dense_chi(psi, e, f)
        W0 = _dense_dyson(V, chi_eq)
        pad_mu = lambda a: np.pad(a, [(0, 0)] * (a.ndim - 1) + [(0, nmu_pad - NMU)])
        sq = lambda a: np.pad(a, ((0, 0), (0, nmu_pad - NMU), (0, nmu_pad - NMU)))
        W_p = np.moveaxis(sq(W0), 0, -1).reshape(nmu_pad, nmu_pad, *KGRID)
        W_R = np.fft.ifftn(W_p, axes=(2, 3, 4), norm="ortho")
        z_p = np.zeros((nmu_pad, nG_pad), complex); z_p[:NMU, :NG] = zeta
        v_p = np.zeros(nG_pad); v_p[:NG] = v
        psi_w = psi[:, B_LO:B_LO + NA]
        kf = np.stack(np.unravel_index(np.arange(nk), KGRID), axis=1) / np.asarray(KGRID, float)
        td = assemble_td_model(
            put(pad_mu(psi_w), P(None, None, None, "x")), put(W_R, P("x", "y", None, None, None)),
            put(z_p, ZETA_G_SPEC), put(v_p, V_G_SPEC), mesh_xy=mesh, kgrid=KGRID,
            k_frac=kf, band_lo=B_LO, n_bands=NA, n_rmu=NMU)
        dyn = D.assemble_dynsig_bundle(
            td, put(pad_mu(psi), P(None, None, None, "x")), e, f, put(sq(W0), D.CHI_SPEC),
            mesh_xy=mesh, rule=rule, V_q=put(sq(V), D.CHI_SPEC))
        kern = D.make_dynsig_kernels(mesh, KGRID)
        A, S = D.split_bundle(dyn)                  # arrays are jit operands (multi-process)
        strip = lambda a: np.asarray(gather(a))[..., :NMU, :NMU]

        # G1 / R2: the split at t = 0
        sp = jax.jit(lambda A_: D.chi0_split({**S, **A_}, kern))(A)
        ff, fa, aa = (strip(sp[k]) for k in ("ff", "fa", "aa"))
        rows["G1"] = _rel(ff + fa + aa, chi_eq)
        rows["R2"] = _rel(ff + aa, chi_eq)
        rows["G1_fa_share"] = float(np.max(np.abs(fa)) / np.max(np.abs(chi_eq)))
        rows["rule_nodes"] = int(len(rule["tau"]))

        # G2: equilibrium COHSEX from W0
        sx, coh = jax.jit(lambda A_: D.equilibrium_sigma({**S, **A_}, kern, A_["W0_q"]))(A)
        ref = _dense_sigma(psi, e, f, np.diag(f[0, B_LO:B_LO + NA])[None].repeat(nk, 0), W0, V)
        rows["G2"] = _rel(np.asarray(gather(sx)) + np.asarray(gather(coh)), ref)

        # the excited state and its dense references
        x = _excited(rng, nk)
        f_t = f.copy()
        f_t[:, B_LO:B_LO + NA] = f[:, B_LO:B_LO + NA] + np.real(np.einsum("kii->ki", x))
        chi_t = _dense_chi(psi, e, f_t)
        dchi_ref = chi_t - chi_eq
        xd = put(x, P())
        U1 = put(np.broadcast_to(np.eye(NA, dtype=complex), (nk, NA, NA)), P())
        eA, fA = put(e[:, B_LO:B_LO + NA], P()), put(f_t[:, B_LO:B_LO + NA], P())
        for tag, kw in (("G3", {}), ("R3", dict(pauli=False)), ("R5", dict(same_side=False))):
            dchi, flags = jax.jit(lambda A_, U_, e_, f_: D.delta_chi0(
                {**S, **A_}, kern, U_, e_, f_, **kw))(A, U1, eA, fA)
            rows[tag] = _rel(strip(dchi), dchi_ref)
        rows["G3_flags"] = {k: int(gather(v)) for k, v in flags.items()}

        # G4: total dSigma, dress none, through the model (TDGW order = model order here)
        rho_eq_h = np.zeros((round_up(nk, p), NA, NA), complex)
        rho_eq_h[:nk] = np.diag(f[0, B_LO:B_LO + NA])[None]
        rho_eq = tp._place(rho_eq_h, mesh, 0)
        rho_h = np.zeros((1, round_up(nk, p), NA, NA), complex)
        rho_h[0, :nk] = rho_eq_h[:nk] + x
        rho = tp.place_state(rho_h, mesh)
        W_t = _dense_dyson(V, chi_t)
        ref_dyn = (_dense_sigma(psi, e, f, rho_h[0, :nk], W_t, V)
                   - _dense_sigma(psi, e, f, rho_eq_h[:nk], W0, V)
                   + _dense_hartree(psi_w, zeta, v, x))
        m_dyn = D.dynsig_model(dyn, mesh, k_frac=kf, rho_eq=rho_eq, screening="static_rpa")
        got = gather(jax.jit(lambda prm, r: m_dyn.apply(prm, r, 0.0))(m_dyn.params, rho))
        rows["G4"] = _rel(2.0 * got[0, :nk], ref_dyn)                # Ha -> Ry
        rows["G6"] = float(np.max(np.abs(got[0, :nk] - np.conj(np.swapaxes(got[0, :nk], -1, -2)))))
        z = gather(jax.jit(lambda prm, r: m_dyn.apply(prm, r, 0.0))(
            m_dyn.params, tp.place_state(rho_eq_h[None], mesh)))
        rows["G5"] = float(np.max(np.abs(z)))

        # G4b / G4c / R1 / R6: dress = instantaneous; the reference iterates the population
        # dressing diag(eps^G) + K drho_pop to its fixed point (dense K, plain iteration)
        K_dense = _dense_kernel(psi_w, zeta, v, W0)
        e_w, f_w = e[:, B_LO:B_LO + NA], f[:, B_LO:B_LO + NA]

        def dressed_ref(xx):
            Uv, ev, fr, conv = _dense_dressed(e_w, f_w, xx, K_dense)
            psi_r = psi.copy()
            psi_r[:, B_LO:B_LO + NA] = np.einsum("kmsu,kmi->kisu", psi_w, Uv)
            e_r, f_r = e.copy(), f.copy()
            e_r[:, B_LO:B_LO + NA], f_r[:, B_LO:B_LO + NA] = ev, fr
            W_r = _dense_dyson(V, _dense_chi(psi_r, e_r, f_r, x_sep=rule["e_sep"]))
            return (_dense_sigma(psi, e, f, rho_eq_h[:nk] + xx, W_r, V)
                    - _dense_sigma(psi, e, f, rho_eq_h[:nk], W0, V)
                    + _dense_hartree(psi_w, zeta, v, xx)), conv
        ref_b, rows["G4b_ref_steps_residual"] = dressed_ref(x)
        # the dressed levels move: the rule is planned with a margin (static_rule docstring)
        dyn_m = D.assemble_dynsig_bundle(
            td, put(pad_mu(psi), P(None, None, None, "x")), e, f, put(sq(W0), D.CHI_SPEC),
            mesh_xy=mesh, rule=D.static_rule(e, f, target_error=1e-11, margin=0.5),
            V_q=put(sq(V), D.CHI_SPEC))
        rng2 = np.random.default_rng(20261019)       # the new states; rng's stream is unchanged
        x_hot = _hot(rng2, nk)
        rho_hot = np.zeros_like(rho_h)
        rho_hot[0, :nk] = rho_eq_h[:nk] + x_hot
        ref_c, rows["G4c_ref_steps_residual"] = dressed_ref(x_hot)
        for tag, kw, r_, ref_ in (
                ("G4b", dict(dress="instantaneous"), rho, ref_b),
                ("R1", dict(dress="instantaneous", occupation="fixed_basis"), rho, ref_b),
                ("R6", dict(dress="coherent"), rho, ref_b),
                ("G4c", dict(dress="instantaneous"), tp.place_state(rho_hot, mesh), ref_c)):
            mb = D.dynsig_model(dyn_m, mesh, k_frac=kf, rho_eq=rho_eq, screening="static_rpa",
                                **kw)
            gb = gather(jax.jit(lambda prm, r: mb.apply(prm, r, 0.0))(mb.params, r_))
            rows[tag] = _rel(2.0 * gb[0, :nk], ref_)

        # G10 / R7 / G11: the dynamic correction itself (Ry), K = the TD model (static_dsigma)
        rows.update(_dress_checks(mesh, td, dyn_m, kern, x, x_hot, f_w, nk, rng2))

        # G7 / R4: the literal definition in a rotated basis, frozen W
        hrot = np.stack([np.diag(e[k, B_LO:B_LO + NA]) for k in range(nk)]) + 0.3 * (
            rng.standard_normal((nk, NA, NA)))
        hrot = put(0.5 * (hrot + np.swapaxes(hrot, -1, -2)), P())
        WR0 = D._w_R(dyn["W0_q"], kern)
        sex = build_td_agw_apply(mesh, td, sex=True, hartree=False)
        from tdagw.model import pad_td_state, strip_td_state, td_state_sharding
        xs = device_put_process_local(pad_td_state(x[None], td), td_state_sharding(mesh))
        kw_ref = strip_td_state(gather(sex(xs)), td)[0]                 # -K^W drho, Ry
        for tag, coh_ in (("G7", True), ("R4", False)):
            lit = jax.jit(lambda A_, x_, w_, h_: D.literal_sx({**S, **A_}, kern, x_, w_, h=h_,
                                                             coherences=coh_)
                          - D.literal_sx({**S, **A_}, kern, jnp.zeros_like(x_), w_, h=h_))(
                A, xd, WR0, hrot)
            rows[tag] = _rel(gather(lit), kw_ref)

        # G8 / G9: the propagator on a pulse
        rows.update(_pulse_checks(mesh, td, dyn, e, kf, rho_eq, rho_eq_h, nk, p, rng))
    return rows


def _dress_checks(mesh, td, dyn_m, kern, x_drv, x_hot, f_w, nk, rng):
    """G10 / R7 (eps scaling of the dynamic correction) and G11 (fixed-point residuals)."""
    import functools
    import jax
    import jax.numpy as jnp
    from jax.experimental import multihost_utils
    from jax.sharding import NamedSharding, PartitionSpec as P
    from lxkit import device_put_process_local
    from tdagw import dynsig as D
    from tdagw.model import build_td_agw_apply
    from tdagw.self_energy import _ISDF_OPERANDS
    gather = lambda a: np.asarray(multihost_utils.process_allgather(a, tiled=True))
    put = lambda a: device_put_process_local(np.asarray(a), NamedSharding(mesh, P()))
    A_m, S_m = D.split_bundle(dyn_m)
    ops = tuple(td[k] for k in _ISDF_OPERANDS)

    def static_ds(ops_):                         # K drho on the window, Ry (as _dyn_apply)
        app = build_td_agw_apply(mesh, dict(zip(_ISDF_OPERANDS, ops_), kgrid=td["kgrid"],
                                            nspinor=td["nspinor"]))
        pad = int(td["nb_pad"]) - NA
        return lambda y: app(jnp.pad(y, ((0, 0), (0, pad), (0, pad)))[None])[0, :, :NA, :NA]

    def corr(A_, ops_, xx, dress):
        d, fl = D.dynamic_correction({**S_m, **A_}, kern, xx, static_dsigma=static_ds(ops_),
                                     dress=dress)
        return d, fl["dress_residual"], fl["dress_unconverged"]

    def basis(A_, ops_, xx, steps):
        dd = {**S_m, **A_}
        fw = dd["f_full"][:, B_LO:B_LO + NA]
        return D._population_basis(dd["e_full"][:, B_LO:B_LO + NA], xx + D._band_diag(fw),
                                   D._band_diag(fw), static_ds(ops_), steps=steps)
    fns = {dr: jax.jit(functools.partial(corr, dress=dr))
           for dr in ("instantaneous", "coherent", "none")}
    rows = {}
    A, unitary, X = _drive_family(rng, nk, f_w)
    epss = (2.5e-4, 5e-4, 1e-3)
    for tag, dr in (("G10", "instantaneous"), ("R7", "coherent"), ("D10", "none")):
        amp = [float(np.max(np.abs(gather(fns[dr](A_m, ops, put(unitary(s)))[0])))) for s in epss]
        rows[f"{tag}_amp"] = amp
        rows[f"{tag}_slope"] = float(np.polyfit(np.log(epss), np.log(amp), 1)[0])
    pure = {dr: float(np.max(np.abs(gather(fns[dr](A_m, ops, put(epss[1] * X))[0]))))
            for dr in ("instantaneous", "coherent")}
    rows["G10_pure"] = pure["instantaneous"] / pure["coherent"]
    rows["G10_pure_amp"] = [pure["instantaneous"], pure["coherent"]]
    steps = D.DRESS_STEPS
    basis_prev = jax.jit(functools.partial(basis, steps=steps - 1))
    basis_last = jax.jit(functools.partial(basis, steps=steps))
    for tag, xx in (("driven", x_drv), ("hot", x_hot)):
        _, res, unc = fns["instantaneous"](A_m, ops, put(xx))
        f_prev = gather(basis_prev(A_m, ops, put(xx))[2])
        f_last = gather(basis_last(A_m, ops, put(xx))[2])
        rows[f"G11_{tag}"] = float(gather(res))
        rows[f"G11_{tag}_df"] = float(np.max(np.abs(f_last - f_prev)))
        rows[f"G11_{tag}_unconverged"] = int(gather(unc))
    rows["G11_steps"] = steps
    return rows


def _pulse_checks(mesh, td, dyn, e, kf, rho_eq, rho_eq_h, nk, p, rng):
    import jax
    from tdagw import dynsig as D
    from tdagw import propagate as tp
    from tdagw.self_energy import isdf_model
    e_qp = 0.5 * e[:, B_LO:B_LO + NA]                                   # Ha
    e_qp = e_qp + np.array([0.0, 0.0, 0.05, 0.05])[None]
    r = 0.4 * (rng.standard_normal((3, nk, NA, NA)) + 1j * rng.standard_normal((3, nk, NA, NA)))
    r = r + np.conj(np.swapaxes(r, -1, -2))
    r[:, :, np.arange(NA), np.arange(NA)] = 0.0
    host = dict(e_qp=e_qp, f_kn=np.real(np.einsum("kii->ki", rho_eq_h[:nk])), r_inter=r,
                gamma=np.zeros((NA, NA)), w_k=np.full(nk, 1.0 / nk),
                v0_inter=1j * (e_qp[None, :, :, None] - e_qp[None, :, None, :]) * r,
                v0_intra=None, intra=None)
    ops, nk_pad = tp.place_operators(host, mesh)
    dt, n_sub, n_meas = 1.0, 4, 40                       # a.u.
    cfg = tp.PropagatorConfig(method="rk4", t0=0.0, tf=dt * n_sub * (n_meas - 1),
                              dt_measure=dt * n_sub, dt=dt, n_corr=1, atol=1e-10,
                              rtol=1e-8, max_step=np.inf, chunk=n_meas, remainder=False)
    spec = tp.SampleSpec(kinds=(("polarization", "polarization", ()),))
    rho0 = tp.place_state(rho_eq_h[None], mesh)

    def run(model, e0):
        def drive(t):
            g = e0 * jax.numpy.exp(-((t - 60.0) / 20.0) ** 2) * jax.numpy.cos(0.4 * t)
            return jax.numpy.stack([g, 0.5 * g, 0.0 * g])[None]
        out = []
        tp.run_propagation(cfg, spec, rho0, ops, model, drive, mesh, cell_volume=100.0,
                           nspinor=NS, count=1, mtsteps=np.arange(n_meas) * dt * n_sub,
                           on_chunk=lambda i0, i1, s, f_, t: out.append(
                               np.asarray(s["polarization"])))
        return np.concatenate(out)

    iso = isdf_model(td, mesh, k_frac=kf, rho_eq=rho_eq)
    frz = D.dynsig_model(dyn, mesh, k_frac=kf, rho_eq=rho_eq, screening="frozen")
    rows = {}
    e1 = 0.002
    p_iso = run(iso, e1)
    fields = dict(G9=(2.5e-4, 5e-4), D9=(e1, 2 * e1))       # G9 well inside E0^2 (pop <= 3e-4)
    p_frz = {e0: run(frz, e0) for pair in fields.values() for e0 in pair}
    rows["G8"] = _rel(p_frz[e1], p_iso)
    for tag, ss in (("G9", False), ("D9", True)):
        rpa = D.dynsig_model(dyn, mesh, k_frac=kf, rho_eq=rho_eq, screening="static_rpa",
                             same_side=ss)
        dep, finite = [], True
        for e0 in fields[tag]:
            pr = run(rpa, e0)
            finite &= bool(np.all(np.isfinite(pr)))
            dep.append(float(np.max(np.abs(pr - p_frz[e0]))))
        rows[f"{tag}_finite"] = finite
        rows[f"{tag}_departure"] = dep
        rows[f"{tag}_slope"] = float(np.log(dep[1] / dep[0]) / np.log(2.0)) if dep[0] > 0 else 0.0
        rows[f"{tag}_rel_departure"] = float(dep[0] / np.max(np.abs(p_frz[fields[tag][0]])))
        rows[f"{tag}_fields"] = list(fields[tag])
    return rows


GATES = dict(G1=1e-8, G2=1e-12, G3=1e-7, G4=1e-7, G4b=1e-7, G4c=1e-7, G5=1e-12, G6=1e-12,
             G7=1e-12, G8=1e-13, G10_pure=1e-9)
TWINS = dict(R1="G4b", R2="G1", R3="G3", R4="G7", R5="G3", R6="G4b")
G10_SLOPE, G11_TOL = (1.9, 2.1), 1e-9


def verdict(rows):
    ok = {g: rows[g] <= tol for g, tol in GATES.items()}
    ok["G9"] = rows["G9_finite"] and 1.6 <= rows["G9_slope"] <= 2.4
    ok["G10"] = G10_SLOPE[0] <= rows["G10_slope"] <= G10_SLOPE[1]
    ok["G11"] = all(rows[f"G11_{s}"] <= G11_TOL and not rows[f"G11_{s}_unconverged"]
                    for s in ("driven", "hot"))
    for r, g in TWINS.items():
        ok[r] = rows[r] > GATES[g]
    ok["R7"] = not G10_SLOPE[0] <= rows["R7_slope"] <= G10_SLOPE[1]
    return ok


def main() -> int:
    """Two launch modes: under ``lx run -n 4`` (a coordinator address in the
    environment) the runtime's communicator stack; else the CPU unit-test
    clause, one process with four host devices (``resolve_mesh``)."""
    import os
    multi = bool(os.environ.get("JAX_COORDINATOR_ADDRESS"))
    if multi:
        mesh = runtime.initialize_communicator_stack().mesh   # before any JAX call
    else:
        import jax
        jax.config.update("jax_enable_x64", True)
        from common.collectives import resolve_mesh
        mesh = resolve_mesh()
    import jax
    finalize = runtime.finalize_process if multi else (lambda rc: None)
    if tuple(mesh.devices.shape) != (2, 2):
        print(f"SKIP: needs a 2x2 mesh, got {tuple(mesh.devices.shape)}", flush=True)
        finalize(0)
        return 0
    rows = run_checks(mesh)
    ok = verdict(rows)
    if jax.process_index() == 0:
        print("RESULT " + json.dumps(dict(rows=rows, ok=ok, all=all(ok.values()),
                                          devices=jax.device_count(),
                                          platform=jax.devices()[0].platform), default=float),
              flush=True)
    rc = 0 if all(ok.values()) else 1
    finalize(rc)
    return rc


def test_tdagw_dynsig_needs_2x2_mesh():
    import pytest
    pytest.skip("script on a 2x2 mesh: JAX_PLATFORMS=cpu XLA_FLAGS="
                "--xla_force_host_platform_device_count=4 python3 tests/test_tdagw_dynsig.py")


if __name__ == "__main__":
    sys.exit(main())

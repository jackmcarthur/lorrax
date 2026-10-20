"""The GN-PPM groundwork of the instantaneous self-energy against an independent dense reference.

docs/theory/td_agw_dynamical.md §9 items 4-5: ``gw.ppm_sigma.screening_update``
(Eq. 6 at the GN supports 0 and i w_p, then the ordered pole fit the GW driver
runs) and ``gw.ppm_sigma.window_sigma_c`` (the GW run's Sigma tau sweep on the
TD window: frozen bands at equilibrium, the active window rotated with its
populations).  Synthetic magnet (random complex spinors: no time reversal) as
tests/test_tdagw_dynsig.py: k grid 3x2x1, 9 bands = 3 frozen occupied + a 2v2c
window + 2 frozen empty, two spinor components, 8 centroids, one pad band on
the GW carrier.  The dense reference (``_dense_*``) shares no code with either
entry: Adler-Wiser chi0 over ordered band pairs at z = 0 and z = i w_p, the
Dyson solve, the elementwise GN fit of gn-ppm-nonhermitian.md §3 and the
GN-PPM Sigma_c of its §4 as an explicit (q, band, mu, nu) sum with the
retarded broadening of sigma-quadrature-problem.md §1 (valence branches R_-,
conduction R_+, invalid modes in their static COHSEX limit).

Runs as a script on a 2x2 mesh.  CPU, one process with four host devices::

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \\
        python3 tests/test_tdagw_dynsig_sigma.py

CUDA, four ranks with one GPU each::

    lx run -N 1 -G 4 -n 4 -- python3 tests/test_tdagw_dynsig_sigma.py

Under pytest it skips.  Gates (G) and red twins (R); each twin feeds one wrong
ingredient to the checker of the gate it shadows:

G0  the GW driver's own screening on the synthetic bundle (compute_chi0,
    compute_chi0_imag_ordered, solve_w) == dense Dyson, both supports     rel <= 1e-8
G1  chi0^ff + chi0^fa + chi0^aa at i w_p (tdagw.dynsig.chi0_split on the
    probe rule) == dense ordered chi0(i w_p); the same at z = 0            rel <= 1e-8
G2  screening_update with V in W0's place and the split chi0 at both
    supports == the driver's W at both supports                           rel <= 1e-8
    and its poles == fit_pole_model on the driver's W                       rel <= 1e-8
    and == the dense elementwise fit (Omega, B, D, valid)                   rel <= 1e-8
G3  screening_update with W0 held and dchi0 of an excited state at both
    supports == dense [1 - W0 dchi0]^-1 W0 and its dense fit               rel <= 1e-8
G4  window_sigma_c at rho_eq: pair Sigma_ij(e_i), Sigma_ij(e_j) == dense    rel <= 1e-5
G5  window_sigma_c at an excited state in a rotated instantaneous basis
    (populations f of U^H rho U) == dense with rotated orbitals            rel <= 1e-5
G6  its on-shell operator U [1/2 (pair_0 + pair_1)] U^H (complex, no Hermitian
    projection) == dense; Sigma^c_ij(omega) at requested energies == dense  rel <= 1e-5
G7  branches=True at the same state, one sweep: cond + val + static == pair
    (rel <= 1e-12) and pair unchanged by keeping the branches (rel <= 1e-12);
    each branch == the dense branch (particle R_+ retarded, hole R_- time
    ordered, invalid-pole static term)                                      rel <= 1e-5
    tdagw.collision.branch_operators: gamma, injection and m_h from the dense
    particle column and adjoint time-ordered hole row; the complex PV terms
    survive the one-sided sampling                                       rel <= 1e-5
G8  static_limit=True: sx == the dense -sum_s f_s <s| W^c(0) |s> exchange on
    every mode (window block, instantaneous basis)                          rel <= 1e-8
    and the head part is zero (no head on this bundle)
G9  the split band sum (td_agw_dynamical.md §1.6) at the excited state of G5:
    window_sigma_c(internal="frozen") + window_sigma_c(internal="active")
    == the whole sum for the pair, each branch, the static term and the
    static limit (separately planned: within the rules' eps)                rel <= 1e-5;
    a window-only plan == the prefix plan                                   rel <= 1e-5;
    on one held session (every active state live in both branches) the
    active part is affine in f: A[(f1 + f2)/2] == (A[f1] + A[f2])/2         rel <= 1e-12
G10 tdagw.dynsig.delta_chi0 of the excited populations of G3 (dress none) at both
    supports == the dense ordered dchi0 (same-side pairs exact at i w_p)    rel <= 1e-8;
    at rho_eq it is exactly zero at both supports
    with same_side_gamma (the static support's principal value) == the dense
    chi0 with the same-side pairs at their principal value                rel <= 1e-8
G11 the refreshed screening of the collision term (dynsig.collision_screening,
    screen_populations) on the run's own W samples and fit plan: its rules ==
    the driver's; at rho_eq dchi0 = 0, W == W0 and the poles == the run's   rel <= 1e-14;
    at the excited state W(t) == dense [1 - W0 dchi0]^-1 W0 at both supports rel <= 1e-8;
    window_sigma_c on the new poles == dense, frozen + active == whole       rel <= 1e-5;
    its static limit on the run's W0 (static_limit_wc0) == dense            rel <= 1e-8;
    with every mode invalid (the static-W limit) the held Hermitian update ==
    the static_rpa member's dynamic_correction (Eq. 5) for the populations   rel <= 1e-8
R0  the conjugate particle-hole orientation at i w_p (z -> -z) fails G0
R1  chi0^fa dropped fails G2 (W)
R2  pole order swapped (D -> -D: R_+ <-> R_-) fails G5
R3  frozen bands omitted from G fails G5
R4  populations read in the fixed basis fails G5
R5  the hole sign flipped (G~< = +i(S~_h - S~_h^H)) fails G7 (G~<)
R6  a stale active part (at f_eq, not f) fails G9 (pair)
R7  the time-reversal-even same-side coefficient at i w_p, (f_a - f_b)(E_a - E_b)/((E_a -
    E_b)^2 + w_p^2), fails G10 (probe)
R8  the static limit subtracted at W(t) instead of W0 fails G11 (static-W limit)
"""
import json
import sys
from types import SimpleNamespace

import runtime  # noqa: F401  (before numpy: the runtime tunes BLAS spin at import)
import numpy as np

KGRID = (3, 2, 1)
NBF, NB_PAD, NS, NMU, NG = 9, 10, 2, 8, 13
B_LO, NA = 3, 4                 # window [3, 7): 2 valence + 2 conduction
NOCC = 5
OMEGA_P = 2.0                   # Ry, the GN probe (ppm_omega_p default)
ETA_EV = 0.25                   # sigma_regularization_ev default
EPS_Q = 1e-7                    # sigma_quadrature_eps (tight: the reference is exact)
TARGET = 1e-10                  # minimax target error of both chi0 rules
RYD = None                      # filled from common.units in run_checks


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))


def _kmq():
    nk = int(np.prod(KGRID))
    ijk = np.stack(np.unravel_index(np.arange(nk), KGRID), axis=1)
    return np.stack([np.ravel_multi_index(tuple(((ijk - ijk[q]) % np.asarray(KGRID)).T), KGRID)
                     for q in range(nk)])


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
    V = 0.05 * (np.einsum("qij,qkj->qik", A, np.conj(A)) / NMU + 0.5 * np.eye(NMU)[None])
    half = (NG - 1) // 2
    zeta = c(NMU, NG)
    zeta[:, 0] = zeta[:, 0].real
    zeta[:, 1 + half:] = np.conj(zeta[:, 1:1 + half])
    v = 0.005 * (rng.random(NG) + 0.1)
    v[1 + half:] = v[1:1 + half]
    v[0] = 0.0
    return psi, e, f, V, zeta, v


# ---------------------------------------------------------------- dense reference

def _dense_chi(psi, e, f, z=0.0):
    """chi_q(z)(mu,nu) = (1/N_k) sum_k sum_{a in k, b in k-q} (f_a - f_b)/(E_a - E_b - z) rho_ab rho_ab^*.

    g_s = 1 (spinors); rho_ab(mu) = sum_s psi_a,s(mu) psi*_b,s(mu).  At z = i w_p
    this is the ordered (time-reversal-odd channel kept) chi0 of the GW run:
    the sign of z is the one ``gw.w_isdf.compute_chi0_imag_ordered`` computes
    (G0; the other sign is red twin R0, which fails by the odd channel).
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
            den = dE - z
            ok = (np.abs(den) > 1e-12) & (np.abs(df) > 0)
            cab = np.where(ok, df / np.where(ok, den, 1.0), 0.0)
            chi[q] += np.einsum("abu,ab,abv->uv", rho, cab, np.conj(rho))
    return chi / nk


def _dense_chi_pv(psi, e, f, gamma, e_sep):
    """chi_q(0) with the same-side pairs at their principal value of width ``gamma``, Ry^-1.

    Cross-gap pairs exact; pairs with both legs on one side of ``e_sep`` take
    (f_a - f_b)(E_a - E_b)/((E_a - E_b)^2 + gamma^2) (tdagw.dynsig.delta_chi0's
    ``same_side_gamma``).
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
            same = ((e[k][:, None] - e_sep) < 0.0) == ((e[kb][None, :] - e_sep) < 0.0)
            ok = (np.abs(dE) > 1e-12) & (np.abs(df) > 0)
            exact = np.where(ok, df / np.where(ok, dE, 1.0), 0.0)
            cab = np.where(same, df * dE / (dE * dE + gamma * gamma), exact)
            chi[q] += np.einsum("abu,ab,abv->uv", rho, cab, np.conj(rho))
    return chi / nk


def _dense_dyson(K, chi):
    return np.stack([np.linalg.solve(np.eye(NMU) - K[q] @ chi[q], K[q]) for q in range(len(K))])


def _dense_fit(W0, Wp, V, fallback=2.0):
    """gn-ppm-nonhermitian.md §3, elementwise: (Omega, B, D, valid)."""
    Wc0, Wcp = W0 - V, Wp - V
    adj = np.conj(np.swapaxes(Wcp, -1, -2))
    h, a = 0.5 * (Wcp + adj), 0.5 * (Wcp - adj)
    den = Wc0 - h
    with np.errstate(all="ignore"):
        om2 = np.real(OMEGA_P ** 2 * h / den)
    good = (np.abs(den) > 1e-14) & np.isfinite(om2) & (om2 > 0.0)
    Om = np.where(good, np.sqrt(np.where(good, om2, 1.0)), fallback)
    B = -0.5 * Wc0 * Om
    D = 1j * a * (OMEGA_P ** 2 + Om ** 2) / (2.0 * OMEGA_P)
    return Om, B, D, good, Wc0


def _dense_sigma(psi_g, e_g, f_g, proj, E_eval, poles, eta, part=None, hole_retarded=False):
    """Sigma^c_ij(k, E) of gn-ppm-nonhermitian.md §4 with the retarded broadening, Ry.

    psi_g/e_g/f_g: the Green orbitals of every band (nk, nb, ns, mu), energies,
    populations (valence weight f with R_-, conduction 1 - f with R_+);
    proj (nk, nP, ns, mu): the projector orbitals; E_eval (nk, nP, n_E): the
    absolute energies Sigma_i. is read at.  Invalid modes (static_limit): the
    Omega -> inf limit, weight (1/2 - f) on W^c(0).  Returns (nk, n_E, nP, nP)
    with entry [k, w, i, j] = Sigma_ij(k, E_eval[k, i or j, w]) per ``mode``.
    ``part``: None (all), "val", "cond" or "static" alone; ``hole_retarded``:
    the valence branch with +i eta (the retarded hole part) instead.
    """
    Om, B, D, good, Wc0 = poles
    nk = psi_g.shape[0]
    kmq = _kmq()
    Rp = np.where(good, B + D, 0.0)
    Rm = np.where(good, B - D, 0.0)
    Ws = np.where(good, 0.0, Wc0)
    out = {}
    for k in range(nk):
        for q in range(nk):
            kb = kmq[q, k]
            a = np.einsum("isu,asu->iau", np.conj(proj[k]), psi_g[kb])     # (nP, nb, mu)
            for iE in range(E_eval.shape[-1]):
                for i in range(proj.shape[1]):
                    E = E_eval[k, i, iE]
                    dm = (E - e_g[kb][:, None, None] + Om[q][None]
                          + (1j if hole_retarded else -1j) * eta)               # valence
                    dp = E - e_g[kb][:, None, None] - Om[q][None] + 1j * eta     # conduction
                    if part is None:
                        Wa = (f_g[kb][:, None, None] * Rm[q][None] / dm
                              + (1.0 - f_g[kb][:, None, None]) * Rp[q][None] / dp
                              + (0.5 - f_g[kb][:, None, None]) * Ws[q][None])   # (nb, mu, nu)
                    elif part == "val":
                        Wa = f_g[kb][:, None, None] * Rm[q][None] / dm
                    elif part == "cond":
                        Wa = (1.0 - f_g[kb][:, None, None]) * Rp[q][None] / dp
                    else:
                        Wa = np.broadcast_to((0.5 - f_g[kb][:, None, None]) * Ws[q][None],
                                             (e_g.shape[1],) + Ws[q].shape)
                    row = np.einsum("au,auv,jav->j", a[i], Wa, np.conj(a))
                    out.setdefault((k, iE, i), 0.0)
                    out[(k, iE, i)] = out[(k, iE, i)] + row / nk
    nP = proj.shape[1]
    S = np.zeros((nk, E_eval.shape[-1], nP, nP), complex)
    for (k, iE, i), row in out.items():
        S[k, iE, i] = row
    return S


def _dense_pair(psi_g, e_g, f_g, proj, e_proj, poles, eta, **kw):
    """(2, nk, nP, nP): [0] Sigma_ij(e_i), [1] Sigma_ij(e_j) (``kw``: :func:`_dense_sigma`'s)."""
    nk, nP = e_proj.shape
    S = np.zeros((2, nk, nP, nP), complex)
    for j in range(nP):
        Ej = np.repeat(e_proj[:, j][:, None, None], nP, axis=1)               # (nk, nP, 1)
        S[1, :, :, j] = _dense_sigma(psi_g, e_g, f_g, proj, Ej, poles, eta, **kw)[:, 0, :, j]
    S[0] = _dense_sigma(psi_g, e_g, f_g, proj, e_proj[:, :, None], poles, eta, **kw)[:, 0]
    return S


def _dense_sx(psi_g, f_g, proj, Wc0):
    """-1/nk sum_q sum_s f_s <i|s> W^c0_q <s|j> on every mode: (nk, nP, nP) Ry."""
    nk = psi_g.shape[0]
    kmq = _kmq()
    S = np.zeros((nk, proj.shape[1], proj.shape[1]), complex)
    for k in range(nk):
        for q in range(nk):
            kb = kmq[q, k]
            a = np.einsum("isu,asu->iau", np.conj(proj[k]), psi_g[kb])         # (nP, nb, mu)
            S[k] -= np.einsum("iau,a,uv,jav->ij", a, f_g[kb], Wc0[q], np.conj(a)) / nk
    return S


def _branch_reference(pair_cond, pair_val, static):
    """Independent NumPy column-shell retarded operators, including the PV part."""
    dagger = lambda a: np.conj(np.swapaxes(a, -1, -2))
    rp, rh = pair_cond[1], dagger(pair_val[0])
    total = rp + rh
    return (1j * (total - dagger(total)), 1j * rh + dagger(1j * rh),
            0.5 * (total + static + dagger(total + static)))


def _excited(rng, nk):
    """drho on the window: holes, carriers with unequal populations, coherences (Hermitian)."""
    x = np.zeros((nk, NA, NA), complex)
    pops = np.array([-0.06, -0.035, 0.05, 0.045])
    for k in range(nk):
        x[k, np.arange(NA), np.arange(NA)] = pops * (1.0 + 0.3 * rng.standard_normal(NA))
    g = 0.04 * (rng.standard_normal((nk, NA, NA)) + 1j * rng.standard_normal((nk, NA, NA)))
    g = g - np.einsum("kii->ki", g)[:, :, None] * np.eye(NA)[None]
    return x + g + np.conj(np.swapaxes(g, -1, -2))


# ---------------------------------------------------------------- the GW carrier

class _WfnStub:
    """The identity-grid facts ``symmetry_maps.SymMaps`` reads from a WFN (a magnet)."""

    def __init__(self, kf):
        self.trs_holds = False
        self.ntran = 1
        self.kgrid = np.asarray(KGRID)
        self.nkpts = int(kf.shape[0])
        self.kpoints = kf
        self.sym_matrices = np.eye(3, dtype=np.int64)[None]
        self.shift = np.zeros(3)


FFT_GRID = (4, 4, 4)


def _plan(mesh, kf, rng):
    """The identity-grid k-unfold plan and its packed centroid order (the run's in-memory order)."""
    import warnings
    from symmetry_maps import SymMaps
    from gw.centroid_k_unfold import build_centroid_k_unfold_plan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sym = SymMaps(_WfnStub(kf))
    flat = np.sort(rng.choice(int(np.prod(FFT_GRID)), NMU, replace=False))
    cidx = np.stack(np.unravel_index(flat, FFT_GRID), axis=1).astype(np.int32)
    kplan = build_centroid_k_unfold_plan(sym, cidx, FFT_GRID, mesh, nspinor=NS, parent_k_frac=kf)
    packed = np.asarray(kplan.layout.axis.pack_host(np.arange(NMU)[None], axis=1))[0]
    if int(kplan.layout.axis.n_padded) != NMU or sorted(packed.tolist()) != list(range(NMU)):
        raise RuntimeError(f"fixture: the packed centroid order is not a permutation ({packed})")
    return kplan, packed


def _gw_bundle(mesh, psi, e, f, kplan):
    """A parents-only ``Wavefunctions`` bundle of the GW Sigma stage on an identity grid.

    ``psi`` is already in the plan's packed centroid order (every operator of
    the fixture is), so ``meta.mu_basis`` is None.
    """
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local
    from common.meta import Meta
    from common.wfn_layout import PSI_MUN_SPEC, PSI_NMU_SPEC
    from gw.wavefunction_bundle import BandSlices, ParentGreenCarrier, Wavefunctions
    nk = int(np.prod(KGRID))
    fft_grid = FFT_GRID
    psi_p = np.zeros((nk, NB_PAD, NS, NMU), complex); psi_p[:, :NBF] = psi
    e_p = np.full((nk, NB_PAD), float(e.max()) + 1.0); e_p[:, :NBF] = e
    f_p = np.zeros((nk, NB_PAD)); f_p[:, :NBF] = f
    put = lambda a, spec: device_put_process_local(np.ascontiguousarray(a), NamedSharding(mesh, spec))
    nmu = put(psi_p, PSI_NMU_SPEC)
    mun = put(np.transpose(psi_p, (0, 2, 3, 1)), PSI_MUN_SPEC)
    enk, occ = put(e_p, P(None, None)), put(f_p, P(None, None))
    carrier = ParentGreenCarrier(psi_nmu=nmu, psi_mun=mun, enk=enk, occ=occ, plan=kplan,
                                 layout="face")
    slices = BandSlices.from_band_edges(0, 0, NOCC, NBF, NB_PAD, b4_logical=NBF)
    wfns = Wavefunctions(enk=enk, occ=occ, slices=slices, green_parent=carrier, layout="face")
    meta = Meta(rank=0, n_proc=4, b_id_0=0, b_id_1=0, b_id_2=NOCC, b_id_3=NBF, b_id_4=NB_PAD,
                fft_grid=fft_grid, cell_volume=100.0, n_rtot=int(np.prod(fft_grid)), n_rmu=NMU,
                npol=NS, nfreq=1, nspin=1, nspinor=NS, nspinor_wfnfile=NS, nkx=KGRID[0],
                nky=KGRID[1], nkz=KGRID[2], nk_tot=nk, n_rmu_padded=NMU, b_id_4_user=NBF)
    return wfns, meta


def _config(omega_ry):
    return SimpleNamespace(
        ppm=SimpleNamespace(invalid_mode="static_limit", fallback_omega=2.0),
        sigma=SimpleNamespace(fermi_reference="midgap", regularization_ev=ETA_EV,
                              window_edge_factor=1.5, quadrature_eps=EPS_Q, omega_step_ev=0.25,
                              out_of_grid="clamp"),
        mpa=SimpleNamespace(pole_batch_size=4), compute_mode="gn_ppm",
        screening=SimpleNamespace(diagrams="w_rpa"),
        omega_grid_ry=omega_ry, omega_grid_ev=omega_ry * RYD)


# ---------------------------------------------------------------- checks

def run_checks(mesh):
    global RYD
    import jax
    import jax.numpy as jnp
    from jax.experimental import multihost_utils
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local
    from common.units import RYD_TO_EV
    from ffi import fft as F
    from runtime.padding import padded_mu_extent, round_up
    from symmetry_maps import q_negation_index
    from gw import ppm_sigma as PS
    from gw.minimax_screening import build_imag_probe_response_rule, build_static_quadrature
    from gw.w_isdf import compute_chi0, compute_chi0_imag_ordered, solve_w
    from tdagw.model import ZETA_G_SPEC, V_G_SPEC, assemble_td_model
    from tdagw import dynsig as D
    RYD = RYD_TO_EV

    rng = np.random.default_rng(20261017)
    psi, e, f, V, zeta, v = _synthetic(rng)
    nk = int(np.prod(KGRID))
    kf = np.stack(np.unravel_index(np.arange(nk), KGRID), axis=1) / np.asarray(KGRID, float)
    kplan, perm = _plan(mesh, kf, rng)
    # every operator in the packed centroid order (the GW run's in-memory order)
    psi, zeta, V = psi[..., perm], zeta[perm], V[:, perm][:, :, perm]
    rows = {"packed_order": perm.tolist()}
    p = int(np.prod(mesh.devices.shape))
    if padded_mu_extent(NMU, mesh) != NMU:
        raise RuntimeError("fixture: 8 centroids must need no mesh padding")
    nG_pad = round_up(NG, p)
    gather = lambda x: np.asarray(multihost_utils.process_allgather(x, tiled=True))
    putc = lambda a: device_put_process_local(np.ascontiguousarray(a), NamedSharding(mesh, D.CHI_SPEC))
    q_neg = np.asarray(q_negation_index(KGRID))
    z = 1j * OMEGA_P

    with F.xla_reference():
        # ---- dense screening at both supports
        chi0_d = _dense_chi(psi, e, f, 0.0)
        chip_d = _dense_chi(psi, e, f, z)
        W0_d, Wp_d = _dense_dyson(V, chi0_d), _dense_dyson(V, chip_d)

        # ---- G0: the GW driver's own screening on the synthetic bundle
        wfns, meta = _gw_bundle(mesh, psi, e, f, kplan)
        mm = SimpleNamespace(target_error=TARGET, max_nodes=64, energy_reference="midgap")
        quad, e_ref = build_static_quadrature(wfns, mm)
        quad_p = build_imag_probe_response_rule(quad, OMEGA_P, mm, with_odd_kernel=True)
        V_dev = putc(V)
        W0_g = solve_w(V_dev, compute_chi0(wfns, quad, meta, mesh, energy_reference=e_ref),
                       meta, mesh)
        Wp_g = solve_w(V_dev, compute_chi0_imag_ordered(wfns, quad_p, meta, mesh, q_neg_index=q_neg,
                                                        energy_reference=e_ref), meta, mesh)
        W0_gh, Wp_gh = gather(W0_g), gather(Wp_g)
        rows["G0_W0"] = _rel(W0_gh, W0_d)
        rows["G0_Wp"] = _rel(Wp_gh, Wp_d)
        rows["R0"] = _rel(Wp_gh, _dense_dyson(V, _dense_chi(psi, e, f, -z)))
        rows["G0_Wp_antihermitian"] = float(np.max(np.abs(Wp_d - np.conj(np.swapaxes(Wp_d, -1, -2))))
                                            / np.max(np.abs(Wp_d)))

        # ---- the TD bundle (tdagw.dynsig) of the same system
        rule = D.static_rule(e, f, target_error=TARGET, max_nodes=64)
        prule = D.probe_rule(rule, OMEGA_P, target_error=TARGET)
        rows["rule_nodes"] = [int(len(rule["tau"])), int(len(prule["tau"]))]
        rows["rule_vs_driver"] = [float(np.max(np.abs(np.asarray(quad.tau) - rule["tau"]))),
                                  float(np.max(np.abs(np.asarray(quad_p.tau) - prule["tau"])))]
        sq = lambda a: a
        W_p5 = np.moveaxis(W0_d, 0, -1).reshape(NMU, NMU, *KGRID)
        W_R = np.fft.ifftn(W_p5, axes=(2, 3, 4), norm="ortho")
        z_p = np.zeros((NMU, nG_pad), complex); z_p[:, :NG] = zeta
        v_p = np.zeros(nG_pad); v_p[:NG] = v
        putr = lambda a, spec: device_put_process_local(np.asarray(a), NamedSharding(mesh, spec))
        td = assemble_td_model(
            putr(psi[:, B_LO:B_LO + NA], P(None, None, None, "x")),
            putr(W_R, P("x", "y", None, None, None)), putr(z_p, ZETA_G_SPEC), putr(v_p, V_G_SPEC),
            mesh_xy=mesh, kgrid=KGRID, k_frac=kf, band_lo=B_LO, n_bands=NA, n_rmu=NMU)
        dyn = D.assemble_dynsig_bundle(td, putr(psi, P(None, None, None, "x")), e, f, putc(W0_d),
                                       mesh_xy=mesh, rule=rule, V_q=V_dev)
        kern = D.make_dynsig_kernels(mesh, KGRID)
        A_, S_ = D.split_bundle(dyn)

        # ---- G1 / R1: the split at both supports
        split = {}
        for tag, r in (("0", None), ("p", prule)):
            sp = jax.jit(lambda A__: D.chi0_split({**S_, **A__}, kern, rule=r))(A_)
            split[tag] = {k_: gather(sp[k_]) for k_ in ("ff", "fa", "aa")}
        rows["G1_static"] = _rel(sum(split["0"].values()), chi0_d)
        rows["G1_probe"] = _rel(sum(split["p"].values()), chip_d)
        rows["G1_probe_fa_share"] = float(np.max(np.abs(split["p"]["fa"])) / np.max(np.abs(chip_d)))

        # ---- G2: screening_update with V in W0's place == the driver's W, poles
        plan = PS.pole_fit_plan(V_dev, z, mesh, n_mu_logical=NMU, fallback_omega=2.0, ordered=True,
                                coarsen_extreme_tails=True, q_neg_index=q_neg,
                                n_nodes_static=len(rule["tau"]))
        chis = [putc(sum(split[t].values())) for t in ("0", "p")]
        poles, (W0_u, Wp_u) = PS.screening_update((V_dev, V_dev), chis, plan)
        rows["G2_W0"] = _rel(gather(W0_u), W0_gh)
        rows["G2_Wp"] = _rel(gather(Wp_u), Wp_gh)
        ref = PS.fit_pole_model(W0_g, Wp_g, plan)
        po = {k_: gather(getattr(poles, k_)) for k_ in ("Omega_q", "B_q", "B_odd_q", "valid_mask_q")}
        pr = {k_: gather(getattr(ref, k_)) for k_ in ("Omega_q", "B_q", "B_odd_q", "valid_mask_q")}
        rows["G2_poles_vs_driver"] = max(_rel(po[k_], pr[k_]) for k_ in ("Omega_q", "B_q", "B_odd_q"))
        rows["G2_valid_vs_driver"] = int(np.sum(po["valid_mask_q"] != pr["valid_mask_q"]))
        dfit = _dense_fit(W0_d, Wp_d, V)
        live = dfit[3]
        rows["G2_valid_vs_dense"] = int(np.sum(po["valid_mask_q"] != live))
        rows["G2_poles_vs_dense"] = max(_rel(np.where(live, po["Omega_q"], 0), np.where(live, dfit[0], 0)),
                                        _rel(np.where(live, po["B_q"], 0), np.where(live, dfit[1], 0)),
                                        _rel(np.where(live, po["B_odd_q"], 0), np.where(live, dfit[2], 0)))
        rows["n_invalid_modes"] = int(np.sum(~live))
        rows["odd_even_ratio"] = float(np.max(np.abs(dfit[2][live])) / np.max(np.abs(dfit[1][live])))
        # R1: chi0^fa dropped at both supports
        chis_bad = [putc(split[t]["ff"] + split[t]["aa"]) for t in ("0", "p")]
        _, (W0_b, Wp_b) = PS.screening_update((V_dev, V_dev), chis_bad, plan)
        rows["R1"] = max(_rel(gather(W0_b), W0_gh), _rel(gather(Wp_b), Wp_gh))

        # ---- G3: W0 held, dchi0 of an excited state at both supports
        x = _excited(rng, nk)
        f_t = f.copy()
        f_t[:, B_LO:B_LO + NA] += np.real(np.einsum("kii->ki", x))
        dchi = [_dense_chi(psi, e, f_t, zz) - _dense_chi(psi, e, f, zz) for zz in (0.0, z)]
        poles3, (W0_3, Wp_3) = PS.screening_update((putc(W0_d), putc(Wp_d)), [putc(d) for d in dchi],
                                                   plan)
        W3_d = [_dense_dyson(W, d) for W, d in zip((W0_d, Wp_d), dchi)]
        rows["G3_W"] = max(_rel(gather(W0_3), W3_d[0]), _rel(gather(Wp_3), W3_d[1]))
        dfit3 = _dense_fit(W3_d[0], W3_d[1], V)
        l3 = dfit3[3]
        rows["G3_valid_vs_dense"] = int(np.sum(gather(poles3.valid_mask_q) != l3))
        rows["G3_poles_vs_dense"] = max(
            _rel(np.where(l3, gather(poles3.Omega_q), 0), np.where(l3, dfit3[0], 0)),
            _rel(np.where(l3, gather(poles3.B_odd_q), 0), np.where(l3, dfit3[2], 0)))

        # ---- G4: window_sigma_c at rho_eq
        eta = ETA_EV / RYD_TO_EV
        omega_ry = np.arange(-120, 121) * 0.25 / RYD_TO_EV
        splan = PS.window_sigma_plan(_config(omega_ry))
        eye = np.broadcast_to(np.eye(NA, dtype=complex), (nk, NA, NA)).copy()
        win_eq = dict(b_lo=B_LO, U=eye, e=e[:, B_LO:B_LO + NA], f=f[:, B_LO:B_LO + NA])
        out_eq = PS.window_sigma_c(wfns, poles, win_eq, splan, meta=meta, mesh_xy=mesh)
        dpo = (po["Omega_q"], po["B_q"], po["B_odd_q"], po["valid_mask_q"],
               gather(poles.Wc0_q))
        ref_eq = _dense_pair(psi, e, f, psi[:, B_LO:B_LO + NA], e[:, B_LO:B_LO + NA], dpo, eta)
        rows["G4"] = _rel(out_eq["pair"], ref_eq)
        rows["G4_max_abs_sigma_ev"] = float(np.max(np.abs(ref_eq))) * RYD_TO_EV

        # ---- G5 / G6: an excited state in a rotated instantaneous basis
        rho = x + np.stack([np.diag(f[k, B_LO:B_LO + NA]) for k in range(nk)])
        h = np.stack([np.diag(e[k, B_LO:B_LO + NA]) for k in range(nk)]) + 0.02 * (
            rng.standard_normal((nk, NA, NA)) + 1j * rng.standard_normal((nk, NA, NA)))
        h = 0.5 * (h + np.conj(np.swapaxes(h, -1, -2)))
        ev, Uv = np.linalg.eigh(h)
        rt = np.einsum("kmi,kmn,knj->kij", np.conj(Uv), rho, Uv)
        fA = np.real(np.einsum("kii->ki", rt))
        psi_r = psi.copy()
        psi_r[:, B_LO:B_LO + NA] = np.einsum("kmsu,kmi->kisu", psi[:, B_LO:B_LO + NA], Uv)
        e_r, f_r = e.copy(), f.copy()
        e_r[:, B_LO:B_LO + NA], f_r[:, B_LO:B_LO + NA] = ev, fA
        win = dict(b_lo=B_LO, U=Uv, e=ev, f=fA)
        req = (np.array([-1.0, 0.3, 1.1]) / RYD_TO_EV)         # Sigma(omega) at requested energies
        out = PS.window_sigma_c(wfns, poles, win, splan, meta=meta, mesh_xy=mesh, omega_ry=req)
        ref = _dense_pair(psi_r, e_r, f_r, psi_r[:, B_LO:B_LO + NA], ev, dpo, eta)
        rows["G5"] = _rel(out["pair"], ref)
        Sd = 0.5 * (ref[0] + ref[1])
        rows["G6_onshell"] = _rel(out["onshell"], Uv @ Sd @ np.conj(np.swapaxes(Uv, -1, -2)))
        Ew = np.broadcast_to((req + out["efermi_ry"])[None, None, :], (nk, NA, req.size))
        Sw = _dense_sigma(psi_r, e_r, f_r, psi_r[:, B_LO:B_LO + NA], Ew, dpo, eta)
        Sw = np.moveaxis(Sw, 1, 0)                                              # (w, k, i, j)
        Uh = np.conj(np.swapaxes(Uv, -1, -2))
        rows["G6_omega"] = _rel(out["omega"], Uv[None] @ Sw @ Uh[None])
        # ---- G7 / G8: the branches of the same sweep, the rates, the static limit
        from tdagw import collision as C
        outb = PS.window_sigma_c(wfns, poles, win, splan, meta=meta, mesh_xy=mesh,
                                 branches=True, static_limit=True)
        tabs = outb["branch_pair"]
        rows["G7_split"] = _rel(tabs["cond"] + tabs["val"] + tabs["static"][None], outb["pair"])
        rows["G7_pair_same"] = _rel(outb["pair"], out["pair"])
        args = (psi_r, e_r, f_r, psi_r[:, B_LO:B_LO + NA], ev, dpo, eta)
        ref_c, ref_v = _dense_pair(*args, part="cond"), _dense_pair(*args, part="val")
        ref_s = _dense_pair(*args, part="static")[0]
        rows["G7_cond"], rows["G7_val"] = _rel(tabs["cond"], ref_c), _rel(tabs["val"], ref_v)
        rows["G7_static"] = _rel(tabs["static"], ref_s)
        gamma, injection, m_h = map(np.asarray, C.branch_operators(
            tabs["cond"], tabs["val"], tabs["static"]))
        gamma_ref, injection_ref, m_h_ref = _branch_reference(ref_c, ref_v, ref_s)
        rows["G7_gamma"] = _rel(gamma, gamma_ref)
        rows["G7_injection"] = _rel(injection, injection_ref)
        rows["G7_m_h"] = _rel(m_h, m_h_ref)
        rows["G7_min_diag_rate_ev"] = float(min(np.real(np.einsum("kii->ki", g)).min()
                                                for g in (gamma, injection))) * RYD_TO_EV
        rows["R5"] = _rel(-injection, injection_ref)
        rows["G8_sx"] = _rel(outb["static_limit"]["sx"],
                             _dense_sx(psi_r, f_r, psi_r[:, B_LO:B_LO + NA], gather(poles.Wc0_q)))
        rows["G8_head"] = float(np.max(np.abs(outb["static_limit"]["head"])))
        # ---- G9: the split band sum, the window-only plan, the held active session
        kw = dict(meta=meta, mesh_xy=mesh, branches=True, static_limit=True)
        fro = PS.window_sigma_c(wfns, poles, win, splan, internal="frozen", **kw)
        act = PS.window_sigma_c(wfns, poles, win, splan, internal="active", **kw)
        tot = PS.add_window_parts(fro, act)
        rows["G9_pair"] = _rel(tot["pair"], outb["pair"])
        rows["G9_branches"] = max(_rel(tot["branch_pair"][k_], outb["branch_pair"][k_])
                                  for k_ in ("cond", "val", "static"))
        rows["G9_static_limit"] = _rel(tot["static_limit"]["sx"], outb["static_limit"]["sx"])
        rows["G9_tau"] = [int(fro["sweep"]["distinct_tau"]), int(act["sweep"]["distinct_tau"]),
                          int(outb["sweep"]["distinct_tau"])]
        wplan = PS.window_sigma_plan(_config(omega_ry), window_only=True)
        own = PS.window_sigma_c(wfns, poles, win, wplan, **kw)
        rows["G9_window_only"] = _rel(own["pair"], outb["pair"])
        rows["G9_window_only_tau"] = int(own["sweep"]["distinct_tau"])
        held = PS.window_sigma_plan(_config(omega_ry), session={}, window_only=True)
        frac = lambda c: np.clip(c + 0.1 * rng.standard_normal((nk, NA)), 0.05, 0.95)
        PS.window_sigma_c(wfns, poles, dict(win, f=np.full((nk, NA), 0.5)), held,
                          internal="active", **kw)                         # the session's plan
        f1, f2 = frac(np.full((nk, NA), 0.4)), frac(np.full((nk, NA), 0.6))
        a1, a2, am = (PS.window_sigma_c(wfns, poles, dict(win, f=ff), held, internal="active",
                                        **kw) for ff in (f1, f2, 0.5 * (f1 + f2)))
        rows["G9_affine"] = _rel(am["pair"], 0.5 * (a1["pair"] + a2["pair"]))
        rows["G9_affine_rebuilds"] = sum(int(o["sweep"]["rebuilds"]) for o in (a1, a2, am))
        stale = PS.window_sigma_c(wfns, poles, dict(win, f=f[:, B_LO:B_LO + NA]), splan,
                                  internal="active", **kw)
        rows["R6"] = _rel(PS.add_window_parts(fro, stale)["pair"], outb["pair"])
        # R2: pole order swapped; R3: frozen bands omitted; R4: fixed-basis populations
        import dataclasses
        swapped = dataclasses.replace(poles, B_odd_q=-poles.B_odd_q)
        rows["R2"] = _rel(PS.window_sigma_c(wfns, swapped, win, splan, meta=meta,
                                            mesh_xy=mesh)["pair"], ref)
        rows["R3"] = _rel(PS.window_sigma_c(wfns, poles, win, splan, meta=meta, mesh_xy=mesh,
                                            omit_frozen=True)["pair"], ref)
        win_fixed = dict(win, f=np.real(np.einsum("kii->ki", rho)))
        rows["R4"] = _rel(PS.window_sigma_c(wfns, poles, win_fixed, splan, meta=meta,
                                            mesh_xy=mesh)["pair"], ref)

        # ---- G10 / R7: dchi0 at both supports (dress none: U = 1, e = eps^G), the fixed point
        putrep = lambda a: putr(np.ascontiguousarray(a), P())
        wsl = slice(B_LO, B_LO + NA)

        def _dmap(r, odd):
            return jax.jit(lambda A__, U_, e_, f_: D.delta_chi0(
                {**S_, **A__}, kern, U_, e_, f_, rule=r, odd_same_side=odd)[0])

        dfn = {"0": _dmap(None, True), "p": _dmap(prule, True), "p_even": _dmap(prule, False)}
        args10 = (A_, putrep(eye), putrep(e[:, wsl]))
        d10 = {t_: gather(fn(*args10, putrep(f_t[:, wsl]))) for t_, fn in dfn.items()}
        rows["G10_static"] = _rel(d10["0"], dchi[0])
        rows["G10_probe"] = _rel(d10["p"], dchi[1])
        rows["R7"] = _rel(d10["p_even"], dchi[1])
        rows["G10_eq"] = max(float(np.max(np.abs(gather(dfn[t_](*args10, putrep(f[:, wsl]))))))
                             for t_ in ("0", "p"))
        GAMMA_PV = 0.05                                    # Ry: wide enough to matter here
        d_pv = gather(jax.jit(lambda A__, U_, e_, f_: D.delta_chi0(
            {**S_, **A__}, kern, U_, e_, f_, same_side_gamma=GAMMA_PV)[0])(
                *args10, putrep(f_t[:, wsl])))
        ref_pv = _dense_chi_pv(psi, e, f_t, GAMMA_PV, rule["e_sep"]) - _dense_chi(psi, e, f, 0.0)
        rows["G10_pv"] = _rel(d_pv, ref_pv)
        rows["G10_pv_vs_exact"] = _rel(d_pv, dchi[0])
        # ---- G11 / R8: the refreshed screening of the collision term
        src = dict(wfns=wfns, meta=meta, poles=poles, fit_plan=plan, W_supports=(W0_u, Wp_u),
                   quad=quad, config=SimpleNamespace(minimax_config=mm))
        scr = D.collision_screening(src, mesh=mesh, b_lo=B_LO, n_active=NA)
        rows["G11_rules"] = max(
            float(np.max(np.abs(scr["rules"]["static"]["tau"] - rule["tau"]))),
            float(np.max(np.abs(scr["rules"]["probe"]["tau"] - prule["tau"]))),
            float(np.max(np.abs(scr["rules"]["probe"]["beta"] - prule["beta"]))))
        p_eq, _, r_eq = D.screen_populations(scr, f[:, wsl])
        rows["G11_eq_dchi"] = max(r_eq["dchi_max"])
        rows["G11_eq_W"] = max(r_eq["dW"])
        rows["G11_eq_poles"] = max(_rel(gather(getattr(p_eq, k_)), gather(getattr(poles, k_)))
                                   for k_ in ("Omega_q", "B_q", "B_odd_q", "Wc0_q"))
        rows["G11_eq_valid"] = int(np.sum(gather(p_eq.valid_mask_q) != gather(poles.valid_mask_q)))
        p_t, W_t, r_t = D.screen_populations(scr, f_t[:, wsl])
        rows["G11_W"] = max(_rel(gather(W_t[0]), W3_d[0]), _rel(gather(W_t[1]), W3_d[1]))
        rows["G11_census"] = dict(n_invalid=r_t["n_invalid"], n_logical=r_t["n_logical"],
                                  flips_eq=r_t["flips_eq"], d_omega_mev=r_t["d_omega_mev"],
                                  dW=r_t["dW"])
        win_t = dict(b_lo=B_LO, U=eye, e=e[:, wsl], f=f_t[:, wsl])
        kw11 = dict(meta=meta, mesh_xy=mesh, branches=True, static_limit=True,
                    static_limit_wc0=poles.Wc0_q)
        o11 = PS.window_sigma_c(wfns, p_t, win_t, splan, **kw11)
        dpo_t = tuple(gather(getattr(p_t, k_)) for k_ in ("Omega_q", "B_q", "B_odd_q",
                                                          "valid_mask_q", "Wc0_q"))
        rows["G11_sigma"] = _rel(o11["pair"], _dense_pair(psi, e, f_t, psi[:, wsl], e[:, wsl],
                                                          dpo_t, eta))
        rows["G11_static_limit_W0"] = _rel(o11["static_limit"]["sx"],
                                           _dense_sx(psi, f_t, psi[:, wsl], gather(poles.Wc0_q)))
        fro_t, act_t = (PS.window_sigma_c(wfns, p_t, win_t, splan, internal=i_, **kw11)
                        for i_ in ("frozen", "active"))
        rows["G11_split"] = _rel(PS.add_window_parts(fro_t, act_t)["pair"], o11["pair"])
        dead = lambda pp: dataclasses.replace(pp, valid_mask_q=jnp.zeros_like(pp.valid_mask_q))
        stl = lambda o: o["static_limit"]["sx"] + np.stack(
            [np.diag(r_) for r_ in o["static_limit"]["head"]])
        s_t = PS.window_sigma_c(wfns, dead(p_t), win_t, splan, **kw11)
        s_0 = PS.window_sigma_c(wfns, dead(poles), win_eq, splan, **kw11)
        h11 = np.asarray(C.hermitian_update(s_t["onshell"], s_0["onshell"], stl(s_t), stl(s_0)))
        x_d = np.stack([np.diag(np.diag(x[k_])) for k_ in range(nk)])
        dsig = gather(jax.jit(lambda A__, xx: D.dynamic_correction({**S_, **A__}, kern, xx)[0])(
            A_, putrep(x_d)))
        rows["G11_static_W_limit"] = _rel(h11, dsig)
        kw8 = dict(kw11, static_limit_wc0=None)
        s_t8 = PS.window_sigma_c(wfns, dead(p_t), win_t, splan, **kw8)
        rows["R8"] = _rel(np.asarray(C.hermitian_update(s_t8["onshell"], s_0["onshell"], stl(s_t8),
                                                        stl(s_0))), dsig)
    return rows


GATES = dict(G0_W0=1e-8, G0_Wp=1e-8, G1_static=1e-8, G1_probe=1e-8, G2_W0=1e-8, G2_Wp=1e-8,
             G2_poles_vs_driver=1e-8, G2_poles_vs_dense=1e-8, G3_W=1e-8, G3_poles_vs_dense=1e-8,
             G4=1e-5, G5=1e-5, G6_onshell=1e-5, G6_omega=1e-5, G7_split=1e-12,
             G7_pair_same=1e-12, G7_cond=1e-5, G7_val=1e-5, G7_static=1e-5, G7_gamma=1e-5,
             G7_injection=1e-5, G7_m_h=1e-5, G8_sx=1e-8, G8_head=0.0, G9_pair=1e-5, G9_branches=1e-5,
             G9_static_limit=1e-5, G9_window_only=1e-5, G9_affine=1e-12, G10_static=1e-8,
             G10_probe=1e-8, G10_eq=0.0, G10_pv=1e-8, G11_rules=0.0, G11_eq_dchi=0.0, G11_eq_W=1e-14,
             G11_eq_poles=1e-14, G11_W=1e-8, G11_sigma=1e-5, G11_static_limit_W0=1e-8,
             G11_split=1e-5, G11_static_W_limit=1e-8)
COUNTS = ("G2_valid_vs_driver", "G2_valid_vs_dense", "G3_valid_vs_dense", "G9_affine_rebuilds",
          "G11_eq_valid")
TWINS = dict(R0="G0_Wp", R1="G2_W0", R2="G5", R3="G5", R4="G5", R5="G7_injection", R6="G9_pair",
             R7="G10_probe", R8="G11_static_W_limit")


def verdict(rows):
    ok = {g: rows[g] <= tol for g, tol in GATES.items()}
    ok.update({c: rows[c] == 0 for c in COUNTS})
    for r, g in TWINS.items():
        ok[r] = rows[r] > 100.0 * GATES[g]
    return ok


def main() -> int:
    import os
    multi = bool(os.environ.get("JAX_COORDINATOR_ADDRESS"))
    if multi:
        mesh = runtime.initialize_communicator_stack().mesh
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


def test_tdagw_dynsig_sigma_needs_2x2_mesh():
    import pytest
    pytest.skip("script on a 2x2 mesh: JAX_PLATFORMS=cpu XLA_FLAGS="
                "--xla_force_host_platform_device_count=4 python3 tests/test_tdagw_dynsig_sigma.py")


if __name__ == "__main__":
    sys.exit(main())

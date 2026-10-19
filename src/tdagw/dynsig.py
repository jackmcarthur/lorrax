"""tdagw/dynsig.py — the instantaneous GW self-energy of TD-aGW (groundwork).

``dynsig_model`` is one more :class:`tdagw.self_energy.SelfEnergyModel`: a pure
function ``apply(params, rho, t) -> dSigma`` of the instantaneous state,

    dSigma(t) = Sigma[rho(t)] - Sigma[rho_eq],                                (1)
    Sigma[rho] = Sigma^SX[G^<, W] + Sigma^COH[W] + V_H[rho],   W = W[rho]      (2)

the static-screening (COHSEX) member of the instantaneous family of
``docs/theory/td_agw_dynamical.md``.  With ``screening = "frozen"`` W is the
GW run's W0 and (1) is exactly the TD-aGW kernel K drho of :mod:`tdagw.model`
(that page, "static limit"); with ``screening = "static_rpa"`` W is rebuilt
from the instantaneous populations at every call:

    W(t) = [1 - W0 dchi0(t)]^{-1} W0,                                         (3)
    dchi0 = [chi0^fa(t) - chi0^fa(0)] + [chi0^aa(t) - chi0^aa(0)]              (4)
          = -2 Re sum_l alpha_l [G^h_eq * dG^p* + dG^h * G^p_eq* + dG^h * dG^p*
                                 - G^p_L * G^h_U*]_(R, tau_l)  (+ same-side pairs)

chi0^ff never changes and is folded once into W0 (the GW run's own W).
G_eq(tau) = G_frozen(tau) + G_active(tau, 0) is composed at every minimax node
from the centroid wavefunctions of EVERY band (never stored per tau); only the
active window enters dG.  Products ``*`` are elementwise in (R, mu, nu) and
spin-traced (``Gc_R x conj(Gv_R)``, as ``gw.w_isdf``'s chi0 kernel).

The instantaneous state (the approximation, stated)
---------------------------------------------------
* The active window's G Hamiltonian H^G_A(t) = diag(eps^G) [+ dSigma^static]
  is diagonalized per k, H^G = U diag(e) U^H (``dress``: ``"none"`` keeps U = 1
  and e = eps^G; ``"instantaneous"`` adds the frozen-W update K drho).
  eps^G are the energies the GW run's G used (DFT for one_shot_dft), so that
  Sigma[rho_eq] is the run's own Sigma.
* Screening: chi0 sees the populations f_i = Re (U^H rho U)_ii of the rotated
  active orbitals psi_A U; the coherences (U^H rho U)_{i != j} are dropped
  (a non-stationary state has no frequency-domain chi0).
* Exchange: Sigma^SX uses the full equal-time G^< = rho (coherences kept), so
  the static limit is exact.
* Cross-gap pairs (one leg below the separator e_sep, one above) take the GW
  run's Laplace minimax rule with Pauli blocking (the inverted product
  G^p_L * G^h_U*); same-side pairs whose populations differ (carrier
  intraband, active x frozen on one side) are summed exactly (Adler-Wiser,
  omega = 0).  A pair with |dE| < ``deg_tol_ry`` and different populations,
  an active level crossing e_sep, or a transition outside the rule's interval
  poisons dSigma with NaN (``GATE dynsig_*``, :func:`check_state` names it).

Layouts (P ranks on the square ('x','y') mesh; k flat C order of ``kgrid``):
``psi_full_X/Y`` (nk, nbf_pad, ns, nmu_pad) at P(None,None,None,'x'|'y');
Green tiles (nk, mu, s, nu, t) at P(None,'x',None,'y',None); chi0, W
(nq, mu, nu) at P(None,'x','y'); W_R (mu, nu, kx, ky, kz) at
P('x','y',None,None,None).  Every N_mu^2-class object is 2-D sharded.

Prototype scope: static screening only (the GN-PPM / shared-pole members, the
head of dchi0, the in-jit re-plan of the rule are design items of the theory
page); the chi0 products are a reference implementation of the weight-row
generalization that ``gw.w_isdf._get_chi_minimax_kernel_face`` needs.
"""
from __future__ import annotations

import dataclasses
import functools
import json
import os
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from distrib_la import eigh_local

__all__ = ["DynsigKernels", "assemble_dynsig_bundle", "check_state", "chi0_split", "split_bundle",
           "dynamic_correction", "dynsig_model", "equilibrium_sigma", "exchange_sigma",
           "instantaneous_basis", "literal_sx", "make_dynsig_kernels", "probe_rule",
           "screened_update", "static_rule", "gw_window_source", "window_collision",
           "window_branch_rates", "collision_screening", "screen_populations",
           "spm_screening", "spm_refit", "pole_census", "census_summary",
           "COLLISION_SCREENINGS"]

GREEN_SPEC = P(None, "x", None, "y", None)        # (k, mu, s, nu, t)
CHI_SPEC = P(None, "x", "y")                      # (q, mu, nu)
WR_SPEC = P("x", "y", None, None, None)           # (mu, nu, kx, ky, kz)


def _refuse(gate: str, got, want, fix: str):
    raise ValueError(f"GATE {gate}: got: {got}; want: {want}; fix: {fix} "
                     f"(docs/theory/td_agw_dynamical.md)")


# ===========================================================================
# Kernels built once per (mesh, k grid, k-convolution backend)
# ===========================================================================

class DynsigKernels(NamedTuple):
    """The callables of one (mesh, k grid): flat-k FFTs on Green and chi tiles and the W term.

    ``g_ifft`` (nk, mu, s, nu, t) -> R, ortho, the router's flat-k transform
    (``common.fft_helpers.make_flat_k_fft``); ``c_fft`` / ``c_ifft`` the same on
    (nq, mu, nu); ``w_term`` the BSE screened-direct term
    (``bse.bse_stack_matvec.make_w_term``): ``w_term(X, psi_X, psi_Y, W_R)``
    = (1/N_k) sum psi*_{nk}(mu) psi_{n'k'}(mu) W(k-k')(mu,nu) psi*_{m'k'}(nu)
    psi_{mk}(nu) X_{k'n'm'}, X (b, n, m, k).
    """
    g_ifft: object
    c_fft: object
    c_ifft: object
    w_term: object
    kgrid: tuple
    mesh: Mesh
    kmq: np.ndarray            # (nq, nk) flat row of k - q, C order


@functools.lru_cache(maxsize=None)
def _kernels(mesh_xy: Mesh, kgrid: tuple, backend: str) -> DynsigKernels:
    from common.fft_helpers import make_flat_k_fft
    from bse.bse_stack_matvec import make_w_term
    del backend                                   # part of the key only
    g_spec = P(None, None, None, "x", None, "y", None)
    c_spec = P(None, None, None, "x", "y")
    nk = int(np.prod(kgrid))
    ijk = np.stack(np.unravel_index(np.arange(nk), kgrid), axis=1)
    kmq = np.stack([np.ravel_multi_index(tuple(((ijk - ijk[iq]) % np.asarray(kgrid)).T), kgrid)
                    for iq in range(nk)])
    return DynsigKernels(
        g_ifft=make_flat_k_fft(mesh_xy, kgrid, g_spec, kind="ifftn", norm="ortho"),
        c_fft=make_flat_k_fft(mesh_xy, kgrid, c_spec, kind="fftn", norm="ortho"),
        c_ifft=make_flat_k_fft(mesh_xy, kgrid, c_spec, kind="ifftn", norm="ortho"),
        w_term=make_w_term(mesh_xy, *kgrid), kgrid=tuple(kgrid), mesh=mesh_xy,
        kmq=kmq.astype(np.int32))


def make_dynsig_kernels(mesh_xy: Mesh, kgrid) -> DynsigKernels:
    """:class:`DynsigKernels` for ``mesh_xy`` and ``kgrid`` (cached; backend in the key)."""
    from ffi.fft import kconv_backend
    kgrid = tuple(int(n) for n in kgrid)
    return _kernels(mesh_xy, kgrid, kconv_backend(mesh_xy))


# ===========================================================================
# The bundle
# ===========================================================================

def static_rule(e_full, f_full, *, target_error: float, max_nodes: int = 64, margin: float = 0.0):
    """The GW run's static Laplace rule 1/x ~ sum alpha_l exp(-tau_l x), Ry.

    Interval [c_min - v_max, c_max - v_min] over the occupied (f = 1) and empty
    (f = 0) bands of ``e_full`` (nk, nbf), as ``gw.minimax_screening.
    build_static_quadrature`` plans it; ``margin`` widens it by a relative
    factor for a propagation whose transitions move (0 reproduces the run).
    """
    from gw.minimax_screening import solve_laplace_minimax_interval
    e, f = np.asarray(e_full, np.float64), np.asarray(f_full, np.float64)
    occ, emp = f > 0.5, f < 0.5
    vmin, vmax = float(e[occ].min()), float(e[occ].max())
    cmin, cmax = float(e[emp].min()), float(e[emp].max())
    x_min, x_max = (cmin - vmax) * (1.0 - margin), (cmax - vmin) * (1.0 + margin)
    q = solve_laplace_minimax_interval(x_min, x_max, target_error=float(target_error),
                                       max_nodes=int(max_nodes))
    return dict(tau=np.asarray(q.tau), alpha=np.asarray(q.alpha), x_min=float(q.x_min),
                x_max=float(q.x_max), max_error=float(q.max_error),
                e_sep=0.5 * (vmax + cmin), provenance=q.provenance)


def assemble_dynsig_bundle(td_model: dict, psi_full, e_full, f_full, W0_q, *, mesh_xy: Mesh,
                           rule: dict, V_q=None, deg_tol_ry: float = 1e-6,
                           occ_tol: float = 1e-12, receipts=None) -> dict:
    """The dynamical-Sigma bundle beside the TD model of the same window.

    Parameters
    ----------
    td_model : the :func:`tdagw.model.assemble_td_model` / ``load_td_model`` bundle
        of the active window [band_lo, band_lo + n_bands) (static part, Ry).
    psi_full : (nk, nbf, ns, nmu_pad) c128 at P(None,None,None,'x') — psi of
        EVERY band of the GW run's sums at the centroids, model k order (flat C).
    e_full, f_full : (nk, nbf) host float — the G energies eps^G (Ry) and the
        equilibrium step occupations of every band.
    W0_q : (nq, nmu_pad, nmu_pad) at P(None,'x','y') — the run's static W (Ry,
        body), flat C-order q, the convention of the TD model's W_R.
    rule : :func:`static_rule` of ``e_full``/``f_full``.
    V_q : optional bare V (nq, nmu_pad, nmu_pad) — needed only by
        :func:`equilibrium_sigma` (the t = 0 check), never per step.
    deg_tol_ry, occ_tol : the same-side degeneracy refusal (module docstring).
    """
    from bse.bse_ring_comm import make_bse_shardings
    from runtime.padding import pad_axis
    kgrid = tuple(int(n) for n in td_model["kgrid"])
    nk = int(np.prod(kgrid))
    e_full = np.asarray(e_full, np.float64)
    f_full = np.asarray(f_full, np.float64)
    nbf = int(e_full.shape[1])
    b_lo, nA = int(td_model["band_lo"]), int(td_model["n_bands"])
    if psi_full.shape[0] != nk or psi_full.shape[1] < nbf or e_full.shape != (nk, nbf):
        _refuse("dynsig_bundle_shapes", f"psi_full {tuple(psi_full.shape)}, e_full "
                f"{e_full.shape}", f"(nk={nk}, >= nbf) and ({nk}, nbf)",
                "sample every band of the GW sums on the model's k grid")
    if not (0 <= b_lo and b_lo + nA <= nbf):
        _refuse("dynsig_window_in_bands", f"window [{b_lo}, {b_lo + nA})",
                f"inside the {nbf} bands", "load the full band set of the GW run")
    if np.any((f_full != 0.0) & (f_full != 1.0)):
        _refuse("dynsig_step_reference", "fractional equilibrium occupations",
                "an insulating step reference (0/1)", "metals take the fractional "
                "contour route (design page, 'metals'); not in this prototype")
    if int(W0_q.shape[0]) != nk or int(W0_q.shape[1]) != int(psi_full.shape[3]):
        _refuse("dynsig_w0_shape", f"W0_q {tuple(W0_q.shape)}",
                f"({nk}, {int(psi_full.shape[3])}, {int(psi_full.shape[3])})",
                "pass the run's full-q W0 on the model's μ carrier")
    sh = make_bse_shardings(mesh_xy)
    px = int(mesh_xy.shape["x"])
    psi_X = jax.jit(lambda a: pad_axis(a[:, :nbf], px, axis=1).array,
                    out_shardings=sh.psi_x)(psi_full)
    psi_Y = jax.jit(lambda a: a, out_shardings=sh.psi_y)(psi_X)
    rep = NamedSharding(mesh_xy, P())
    from common.collectives import device_put_process_local
    put = lambda a: device_put_process_local(np.asarray(a), rep)
    nbf_pad = int(psi_X.shape[1])
    e_pad = np.zeros((nk, nbf_pad)); e_pad[:, :nbf] = e_full
    e_pad[:, nbf:] = e_full.max() + 1.0           # pad bands: inert (zero psi, zero weight)
    f_pad = np.zeros((nk, nbf_pad)); f_pad[:, :nbf] = f_full
    live = np.zeros((nk, nbf_pad)); live[:, :nbf] = 1.0
    return dict(
        td=td_model, psi_X=psi_X, psi_Y=psi_Y, e_full=put(e_pad), f_full=put(f_pad),
        live=put(live), W0_q=W0_q, V_q=V_q, tau=put(rule["tau"]), alpha=put(rule["alpha"]),
        rule=dict(rule), kgrid=kgrid, nspinor=int(psi_full.shape[2]), nbf=nbf,
        nbf_pad=nbf_pad, band_lo=b_lo, n_active=nA, n_rmu=int(td_model["n_rmu"]),
        nmu_pad=int(psi_full.shape[3]), g_s=2.0 if int(psi_full.shape[2]) == 1 else 1.0,
        deg_tol_ry=float(deg_tol_ry), occ_tol=float(occ_tol), receipts=dict(receipts or {}))


def split_bundle(dyn: dict):
    """(arrays, static): the bundle's device arrays and the rest.

    A multi-process jit may not close over an array that spans other processes'
    devices, so a caller jits ``lambda arr: f({**static, **arr}, ...)`` and passes
    ``arrays`` as the operand (the propagator's model does this through its params).
    """
    arrays = {k: v for k, v in dyn.items() if isinstance(v, jax.Array)}
    return arrays, {k: v for k, v in dyn.items() if k not in arrays}


# ===========================================================================
# The instantaneous basis and the Green tiles
# ===========================================================================

def instantaneous_basis(h, rho):
    """Eigenpairs of the window's G Hamiltonian and the populations in that basis.

    h, rho : (nk, nA, nA) Hermitian, Ry / dimensionless.  Returns U (nk, nA, nA)
    (columns = instantaneous orbitals in the fixed basis), e (nk, nA) Ry, and
    f = Re diag(U^H rho U) (nk, nA): the populations the screening sees.
    """
    h = 0.5 * (h + jnp.conj(jnp.swapaxes(h, -1, -2)))
    e, U = eigh_local(h)
    rho_t = jnp.conj(jnp.swapaxes(U, -1, -2)) @ rho @ U
    return U, e, jnp.real(jnp.diagonal(rho_t, axis1=-2, axis2=-1))


def _green(mesh, psi_X, psi_Y, w):
    """G_k(mu s, nu t) = sum_n w_kn psi_kn,s(mu) psi*_kn,t(nu) — (nk, mu, s, nu, t) at GREEN_SPEC.

    psi_X/psi_Y (nk, nb, ns, nmu) on 'x'/'y'; w (nk, nb) real or complex.  The
    band sum is local (both faces carry every band); no N_mu^2 object is
    replicated.
    """
    g = jnp.einsum("knsu,kn,kntv->kusvt", psi_X, w.astype(psi_X.dtype), jnp.conj(psi_Y))
    return lax.with_sharding_constraint(g, NamedSharding(mesh, GREEN_SPEC))


def _chi_zero(mesh, nk, nmu):
    z = jnp.zeros((nk, nmu, nmu), jnp.complex128)
    return lax.with_sharding_constraint(z, NamedSharding(mesh, CHI_SPEC))


def _traced_product(A_R, B_R):
    """sum_st A_R(mu s, nu t) conj(B_R(mu s, nu t)) -> (nR, mu, nu): the spin-traced chi0 product."""
    return jnp.einsum("kusvt,kusvt->kuv", A_R, jnp.conj(B_R))


def _weights(e, f, x_sep, tau, live):
    """The four population weight rows at one node (nk, nb): hole/particle x lower/upper.

    x = e - e_sep; lower (x < 0): h = f e^{x tau}, p = (1 - f) e^{x tau};
    upper: h = f e^{-x tau}, p = (1 - f) e^{-x tau}.  Every factor is <= 1.
    """
    x = e - x_sep
    low = x < 0.0
    ph = jnp.exp(-jnp.abs(x) * tau) * live
    hL = jnp.where(low, f * ph, 0.0)
    pL = jnp.where(low, (1.0 - f) * ph, 0.0)
    hU = jnp.where(low, 0.0, f * ph)
    pU = jnp.where(low, 0.0, (1.0 - f) * ph)
    return hL, pL, hU, pU


# ===========================================================================
# chi0 at omega = 0 with the frozen / active split
# ===========================================================================

def probe_rule(rule: dict, omega_p: float, *, target_error: float, ordered: bool = True) -> dict:
    """The GW run's chi0(i omega_p) rule on the static rule's interval, Ry.

    ``gw.minimax_screening.build_imag_probe_response_rule`` (the GN-PPM probe
    of ``gw.screening.compute_screening``): sum_l alpha_l e^{-tau_l x} ~
    x/(x^2 + w^2) and, ``ordered``, sum_l beta_l e^{-tau_l x} ~ w/(x^2 + w^2) on
    the same times (docs/theory/gn-ppm-nonhermitian.md §2).  Returns
    ``dict(tau, alpha, beta, omega_p, max_error, provenance)``.
    """
    from types import SimpleNamespace
    from gw.minimax_screening import build_imag_probe_response_rule
    q = build_imag_probe_response_rule(
        SimpleNamespace(x_min=float(rule["x_min"]), x_max=float(rule["x_max"])), float(omega_p),
        SimpleNamespace(target_error=float(target_error)), with_odd_kernel=bool(ordered))
    beta = np.zeros_like(np.asarray(q.tau)) if q.alpha_odd is None else np.asarray(q.alpha_odd)
    return dict(tau=np.asarray(q.tau), alpha=np.asarray(q.alpha), beta=beta,
                omega_p=float(omega_p), max_error=float(q.max_error), provenance=q.provenance)


def chi0_split(dyn: dict, kern: DynsigKernels, rule: dict | None = None):
    """chi0^ff, chi0^fa, chi0^aa at equilibrium on the run's rule, (nq, mu, nu) each, Ry^-1.

    chi0_q(z) = (g_s / N_k) sum_k sum_ab (f_a - f_b)/(E_a - E_b - z) rho_ab(mu) rho_ab*(nu)
    on the Laplace rule; G_frozen and G_active(0) are separate Green tiles at each
    node (the t = 0 consistency check of the design page: their sum must be
    the run's chi0).  ``rule=None`` is the static support z = 0 (the bundle's
    rule).  An imaginary-axis support z = i omega_p (:func:`probe_rule`) weights
    each node by alpha_l - i beta_l, so the R-space sum is -2 Re[sum_l (alpha_l -
    i beta_l) G^h * G^p*] and keeps the time-reversal-odd channel of a magnet:
    the ordered chi0(i omega_p) of ``gw.w_isdf.compute_chi0_imag_ordered``
    (docs/theory/gn-ppm-nonhermitian.md §2), whose orientation the formula above
    states (tests/test_tdagw_dynsig_sigma.py G0-G2; the conjugate orientation
    is red twin R0).  Returns a dict with the three terms.
    """
    nk, mesh = int(np.prod(kern.kgrid)), kern.mesh
    b_lo, nA = dyn["band_lo"], dyn["n_active"]
    act = jnp.zeros(dyn["nbf_pad"]).at[b_lo:b_lo + nA].set(1.0)[None, :]
    e, f, live = dyn["e_full"], dyn["f_full"], dyn["live"]
    x_sep = jnp.asarray(dyn["rule"]["e_sep"])
    psi_X, psi_Y = dyn["psi_X"], dyn["psi_Y"]
    if rule is None:
        taus, weights = dyn["tau"], dyn["alpha"]
    else:
        taus = jnp.asarray(np.asarray(rule["tau"], np.float64))
        weights = jnp.asarray(np.asarray(rule["alpha"], np.float64)
                              - 1j * np.asarray(rule["beta"], np.float64))

    def body(acc, node):
        tau, alpha = node
        hL, _, _, pU = _weights(e, f, x_sep, tau, live)
        GhLf, GpUf = (kern.g_ifft(_green(mesh, psi_X, psi_Y, w * (1.0 - act))) for w in (hL, pU))
        GhLa, GpUa = (kern.g_ifft(_green(mesh, psi_X, psi_Y, w * act)) for w in (hL, pU))
        ff = _traced_product(GhLf, GpUf)
        fa = _traced_product(GhLf, GpUa) + _traced_product(GhLa, GpUf)
        aa = _traced_product(GhLa, GpUa)
        return tuple(a + alpha * p for a, p in zip(acc, (ff, fa, aa))), None

    zero = _chi_zero(mesh, nk, dyn["nmu_pad"])
    acc, _ = lax.scan(body, (zero, zero, zero), (taus, weights), unroll=1)
    scale = -2.0 * dyn["g_s"] / np.sqrt(nk)
    return {name: kern.c_fft(scale * jnp.real(a).astype(jnp.complex128))
            for name, a in zip(("ff", "fa", "aa"), acc)}


def _same_side_chi(dyn, kern, psi_A, e_A, f_A, x_sep, omega_p: float = 0.0,
                   odd: bool = True, gamma: float = 0.0):
    """Exact sum over ordered same-side pairs with an active leg, (nq, mu, nu), Ry^-1.

    chi^ss_q(z) = (g_s/N_k) sum_k sum_ab c_ab rho_ab(mu) rho_ab*(nu), a at k, b at k-q,
    rho_ab(mu) = sum_s psi_a,s(mu) psi*_b,s(mu), c_ab = (f_a - f_b)/(E_a - E_b - z) where
    both legs lie on one side of e_sep (cross-gap pairs are on the Laplace rule).  E1: a
    in the window at k, b any band; E2: a frozen at k, b in the window.  Every q is one
    ``lax.map`` step, so one q's pair densities (nk, nA, nbf, mu) are live at a time.
    ``omega_p`` 0: the static support z = 0; > 0: z = i omega_p, the ordered orientation of
    the GW run's chi0(i omega_p), c_ab = (f_a - f_b)(E_a - E_b + i omega_p)/((E_a - E_b)^2 +
    omega_p^2); ``odd = False`` keeps its real part alone (the time-reversal-even
    coefficient; a comparison, not the run's chi0 on a magnet).  ``gamma`` > 0 (z = 0
    only): the principal value of width gamma, c_ab = (f_a - f_b)(E_a - E_b)/((E_a -
    E_b)^2 + gamma^2): exact for |E_a - E_b| >> gamma, zero for a degenerate pair (a
    driven carrier distribution makes symmetry-equivalent k unequal).  Returns (chi,
    bad, n_deg): bad counts near-degenerate same-side pairs whose populations differ
    at z = 0 with gamma = 0 (their static limit is a Drude term this prototype
    refuses); n_deg counts them under the principal value (not refused); none at
    i omega_p.
    """
    nk, mesh = int(np.prod(kern.kgrid)), kern.mesh
    b_lo, nA = dyn["band_lo"], dyn["n_active"]
    win = slice(b_lo, b_lo + nA)
    e_all = dyn["e_full"].at[:, win].set(e_A)
    f_all = dyn["f_full"].at[:, win].set(f_A)
    live = dyn["live"]
    frozen = live.at[:, win].set(0.0)
    psi_all = dyn["psi_X"].at[:, win].set(psi_A)
    tol, otol, gs = dyn["deg_tol_ry"], dyn["occ_tol"], dyn["g_s"]

    def coeff(Ea, Eb, fa, fb, mask):
        same = ((Ea - x_sep) < 0.0) == ((Eb - x_sep) < 0.0)
        dE, df = Ea - Eb, fa - fb
        zero = jnp.zeros((), jnp.int32)
        if omega_p > 0.0:
            w = float(omega_p)
            c = df * ((dE + 1j * w) if odd else dE) / (dE * dE + w * w)
            return jnp.where(same & mask, c, 0.0), zero, zero
        deg = same & (jnp.abs(dE) <= tol) & (jnp.abs(df) > otol) & mask
        if gamma > 0.0:
            g2 = float(gamma) ** 2
            return jnp.where(same & mask, df * dE / (dE * dE + g2), 0.0), zero, jnp.sum(deg)
        ok = same & (jnp.abs(dE) > tol) & mask
        return jnp.where(ok, df / jnp.where(ok, dE, 1.0), 0.0), jnp.sum(deg), zero

    def pair_chi(r, c):
        return jnp.einsum("kabu,kab,kabv->uv", r, c.astype(r.dtype), jnp.conj(r))

    def per_q(kmq):
        tk = lambda a: jnp.take(a, kmq, axis=0)
        r1 = jnp.einsum("kasu,kbsu->kabu", psi_A, jnp.conj(tk(psi_all)))
        c1, n1, d1 = coeff(e_A[:, :, None], tk(e_all)[:, None, :], f_A[:, :, None],
                           tk(f_all)[:, None, :], tk(live)[:, None, :] > 0)
        r2 = jnp.einsum("kasu,kbsu->kabu", psi_all, jnp.conj(tk(psi_A)))
        c2, n2, d2 = coeff(e_all[:, :, None], tk(e_A)[:, None, :], f_all[:, :, None],
                           tk(f_A)[:, None, :], frozen[:, :, None] > 0)
        return (pair_chi(r1, c1) + pair_chi(r2, c2)) * (gs / nk), n1 + n2, d1 + d2

    chis, bads, degs = lax.map(per_q, jnp.asarray(kern.kmq))
    return (lax.with_sharding_constraint(chis, NamedSharding(mesh, CHI_SPEC)), jnp.sum(bads),
            jnp.sum(degs))


def _gate_flags(dyn, e_A, x_sep):
    """Traced counters: active levels that crossed e_sep, cross-gap transitions outside the rule."""
    b_lo, nA = dyn["band_lo"], dyn["n_active"]
    win = slice(b_lo, b_lo + nA)
    e0 = dyn["e_full"][:, win]
    crossed = jnp.sum(((e_A - x_sep) < 0.0) != ((e0 - x_sep) < 0.0))
    e_all = dyn["e_full"].at[:, win].set(e_A)
    live = dyn["live"] > 0
    low = live & ((e_all - x_sep) < 0.0)
    up = live & ~((e_all - x_sep) < 0.0)
    big = 1e30
    x_lo = jnp.min(jnp.where(up, e_all, big)) - jnp.max(jnp.where(low, e_all, -big))
    x_hi = jnp.max(jnp.where(up, e_all, -big)) - jnp.min(jnp.where(low, e_all, big))
    rule = dyn["rule"]
    out = (x_lo < rule["x_min"] * (1.0 - 1e-12)) | (x_hi > rule["x_max"] * (1.0 + 1e-12))
    return crossed.astype(jnp.int32), out.astype(jnp.int32)


def delta_chi0(dyn: dict, kern: DynsigKernels, U, e_A, f_A, *, pauli: bool = True,
               same_side: bool = True, rule: dict | None = None, odd_same_side: bool = True,
               same_side_gamma: float = 0.0):
    """dchi0 of Eq. (4) for the instantaneous window state (U, e_A, f_A), (nq, mu, nu), Ry^-1.

    G_eq(tau) of every band (G_frozen + G_active(0)) is composed at each node and
    never stored; the active tiles carry the rotated orbitals psi_A U.  The
    regrouping G_eq = G_f + G_a(0) makes Eq. (4) free of cancellation: no term is
    a difference of two equilibrium-sized products.  ``pauli=False`` /
    ``same_side=False`` drop the inverted product / the same-side pairs (red twins).
    ``rule=None`` is the static support (the bundle's rule); :func:`probe_rule` is
    z = i omega_p, the ordered orientation of :func:`chi0_split`: each node weighs
    alpha_l - i beta_l (the Pauli product is the same kernel with the opposite
    sign at any support) and the same-side pairs take (f_a - f_b)/(E_a - E_b -
    i omega_p) (``odd_same_side=False``: its real part alone, a comparison).
    ``same_side_gamma`` > 0 (Ry, static support): the same-side pairs' principal value
    of that width (:func:`_same_side_chi`; ``flags['degenerate_pv']`` counts the
    degenerate pairs it zeroes, which ``gamma = 0`` refuses).
    Returns (dchi, flags) with flags the NaN-gate counters.
    """
    nk, mesh = int(np.prod(kern.kgrid)), kern.mesh
    b_lo, nA = dyn["band_lo"], dyn["n_active"]
    win = slice(b_lo, b_lo + nA)
    psi_X, psi_Y = dyn["psi_X"], dyn["psi_Y"]
    psiA0_X, psiA0_Y = psi_X[:, win], psi_Y[:, win]
    psiA_X = jnp.einsum("kmsu,kmi->kisu", psiA0_X, U.astype(psi_X.dtype))
    psiA_Y = jnp.einsum("kmsu,kmi->kisu", psiA0_Y, U.astype(psi_Y.dtype))
    e_full, f_full, live = dyn["e_full"], dyn["f_full"], dyn["live"]
    e0, f0 = e_full[:, win], f_full[:, win]
    x_sep = jnp.asarray(dyn["rule"]["e_sep"])
    onesA = jnp.ones_like(e0)
    G = functools.partial(_green, mesh)
    if rule is None:
        taus, weights, omega_p = dyn["tau"], dyn["alpha"], 0.0
    else:
        taus = jnp.asarray(np.asarray(rule["tau"], np.float64))
        weights = jnp.asarray(np.asarray(rule["alpha"], np.float64)
                              - 1j * np.asarray(rule["beta"], np.float64))
        omega_p = float(rule["omega_p"])

    def body(acc, node):
        tau, alpha = node
        hL, _, _, pU = _weights(e_full, f_full, x_sep, tau, live)
        GhL_eq = kern.g_ifft(G(psi_X, psi_Y, hL))
        GpU_eq = kern.g_ifft(G(psi_X, psi_Y, pU))
        ahL, apL, ahU, apU = _weights(e_A, f_A, x_sep, tau, onesA)
        a0hL, _, _, a0pU = _weights(e0, f0, x_sep, tau, onesA)
        dGh = kern.g_ifft(G(psiA_X, psiA_Y, ahL) - G(psiA0_X, psiA0_Y, a0hL))
        dGp = kern.g_ifft(G(psiA_X, psiA_Y, apU) - G(psiA0_X, psiA0_Y, a0pU))
        p = (_traced_product(GhL_eq, dGp) + _traced_product(dGh, GpU_eq)
             + _traced_product(dGh, dGp))
        if pauli:
            p = p - _traced_product(kern.g_ifft(G(psiA_X, psiA_Y, apL)),
                                    kern.g_ifft(G(psiA_X, psiA_Y, ahU)))
        return acc + alpha * p, None

    acc, _ = lax.scan(body, _chi_zero(mesh, nk, dyn["nmu_pad"]), (taus, weights), unroll=1)
    dchi = kern.c_fft((-2.0 * dyn["g_s"] / np.sqrt(nk)) * jnp.real(acc).astype(jnp.complex128))
    crossed, out = _gate_flags(dyn, e_A, x_sep)
    bad = n_pv = jnp.zeros((), jnp.int32)
    if same_side:
        chi_ss, bad, n_pv = _same_side_chi(dyn, kern, psiA_X, e_A, f_A, x_sep, omega_p=omega_p,
                                           odd=odd_same_side, gamma=float(same_side_gamma))
        dchi = dchi + chi_ss
    return dchi, dict(crossed=crossed, out_of_rule=out, degenerate=bad.astype(jnp.int32),
                      degenerate_pv=n_pv.astype(jnp.int32))


# ===========================================================================
# W and Sigma
# ===========================================================================

def screened_update(W0_q, dchi, mesh_xy: Mesh, n_rmu: int):
    """W(t) = [1 - W0 dchi]^{-1} W0, Eq. (3): the run's Dyson solve with W0 in V's place.

    ``gw.w_isdf.solve_w`` (meta=None route, pref = 1: dchi is physical, Ry^-1);
    (nq, mu, nu) at CHI_SPEC in and out.  chi0^ff is never formed again.
    """
    from gw.w_isdf import solve_w
    from runtime.padding import padded_mu_axis
    axis = padded_mu_axis(int(n_rmu), mesh_xy)
    return solve_w(W0_q, dchi, None, mesh_xy, pref=1.0, axis=axis)


def _w_R(Wq, kern: DynsigKernels):
    """(nq, mu, nu) -> W_R (mu, nu, kx, ky, kz), ortho inverse k-FFT (``bse_feast.ensure_W_R``'s convention)."""
    R = kern.c_ifft(Wq)
    R = jnp.moveaxis(R, 0, -1).reshape(R.shape[1], R.shape[2], *kern.kgrid)
    return lax.with_sharding_constraint(R, NamedSharding(kern.mesh, WR_SPEC))


def _w_term_full(dyn, kern, X_full, W_R):
    """w_term on the full band set: X_full (b, nbf_pad, nbf_pad, nk) -> same, Ry."""
    from bse.bse_ring_comm import make_bse_shardings
    X = lax.with_sharding_constraint(X_full, make_bse_shardings(kern.mesh).X)
    return kern.w_term(X, dyn["psi_X"], dyn["psi_Y"], W_R)


def _band_diag(d):
    """(nk, nb) -> (nk, nb, nb) diagonal, complex."""
    return jax.vmap(jnp.diag)(d.astype(jnp.complex128))


def _sigma_window(dyn, kern, X_win, d_full, W_R):
    """Window block of w_term[W](X), X = diag(d_full) + X_win on the window, (nk, nA, nA) Ry."""
    b_lo, nA = dyn["band_lo"], dyn["n_active"]
    Xf = _band_diag(d_full).at[:, b_lo:b_lo + nA, b_lo:b_lo + nA].add(X_win)
    Y = _w_term_full(dyn, kern, jnp.moveaxis(Xf, 0, -1)[None], W_R)[0]
    return jnp.moveaxis(Y, -1, 0)[:, b_lo:b_lo + nA, b_lo:b_lo + nA]


def _active_state(dyn, x, *, dress, static_dsigma):
    """U, e, f of the window for drho = x (nk, nA, nA) model order (Ry)."""
    b_lo, nA = dyn["band_lo"], dyn["n_active"]
    e_win = dyn["e_full"][:, b_lo:b_lo + nA]
    f_eq = dyn["f_full"][:, b_lo:b_lo + nA]
    rho = x + _band_diag(f_eq)
    if dress == "none":
        U = jnp.broadcast_to(jnp.eye(nA, dtype=x.dtype), x.shape)
        return U, e_win, jnp.real(jnp.diagonal(rho, axis1=-2, axis2=-1))
    if dress != "instantaneous":
        _refuse("dynsig_dress", repr(dress), "'none' | 'instantaneous'", "pick one")
    return instantaneous_basis(_band_diag(e_win) + static_dsigma(x), rho)


def dynamic_correction(dyn: dict, kern: DynsigKernels, x, *, static_dsigma=None,
                       dress: str = "none", pauli: bool = True, same_side: bool = True,
                       occupation: str = "instantaneous"):
    """dSigma[rho] - dSigma^frozen[rho] for one density matrix: the terms carrying dW, Ry.

    x : (nk, nA, nA) drho in the model's k order.  Returns (dsig (nk, nA, nA),
    flags).  With dW = W(t) - W0 (Eq. 3 on dchi of Eq. 4),

        dSigma^dyn = w_term[dW](-rho^eq_tot - drho + 1/2) |_window,            (5)

    the change of Sigma^SX = -w_term[W](rho_tot) and Sigma^COH = +1/2 w_term[W - V](1)
    (``gw.cohsex_sigma``: occupied and band-sum completeness Greens) at fixed V.
    ``occupation = "fixed_basis"`` (red twin) reads the populations from diag(rho)
    in the given gauge instead of the instantaneous basis.  A raised gate makes
    dsig NaN (:func:`check_state` names it on the host).
    """
    U, e_A, f_A = _active_state(dyn, x, dress=dress, static_dsigma=static_dsigma)
    if occupation == "fixed_basis":
        b_lo, nA = dyn["band_lo"], dyn["n_active"]
        f_A = jnp.real(jnp.diagonal(x, axis1=-2, axis2=-1)) + dyn["f_full"][:, b_lo:b_lo + nA]
    elif occupation != "instantaneous":
        _refuse("dynsig_occupation", repr(occupation), "'instantaneous'",
                "'fixed_basis' exists only as the red twin")
    dchi, flags = delta_chi0(dyn, kern, U, e_A, f_A, pauli=pauli, same_side=same_side)
    W = screened_update(dyn["W0_q"], dchi, kern.mesh, dyn["n_rmu"])
    dW_R = _w_R(W - dyn["W0_q"], kern)
    dsig = _sigma_window(dyn, kern, -x, 0.5 * dyn["live"] - dyn["f_full"], dW_R)
    bad = flags["crossed"] + flags["out_of_rule"] + flags["degenerate"]
    return jnp.where(bad > 0, jnp.nan, 1.0) * dsig, dict(flags, dchi=dchi, W=W)


def equilibrium_sigma(dyn: dict, kern: DynsigKernels, W_q, *, heads=None):
    """Sigma^SX and Sigma^COH of the window at rho_eq for a static W, (nk, nA, nA) each, Ry.

    Sigma^SX = -w_term[W](diag f_eq), Sigma^COH = +1/2 w_term[W - V](1) on the
    full band set (``gw.cohsex_sigma``'s occupied and band-sum completeness
    Greens), plus, with ``heads = (w_h, v_h, cell_volume)``, the exact
    band-diagonal q -> 0 shifts -w_h f/(Omega N_k) and (w_h - v_h)/(2 Omega N_k)
    (docs/theory/four-current-head-corrections.md#static-head).
    """
    if dyn["V_q"] is None:
        _refuse("dynsig_equilibrium_needs_v", "V_q = None", "the run's bare V",
                "pass V_q to assemble_dynsig_bundle for the t = 0 check")
    nk, nA = int(np.prod(kern.kgrid)), dyn["n_active"]
    zero = jnp.zeros((nk, nA, nA), jnp.complex128)
    sx = -_sigma_window(dyn, kern, zero, dyn["f_full"], _w_R(W_q, kern))
    coh = 0.5 * _sigma_window(dyn, kern, zero, dyn["live"], _w_R(W_q - dyn["V_q"], kern))
    if heads is not None:
        w_h, v_h, omega = (float(v) for v in heads)
        b_lo = dyn["band_lo"]
        f = dyn["f_full"][:, b_lo:b_lo + nA]
        sx = sx - (w_h / (omega * nk)) * _band_diag(f)
        coh = coh + ((w_h - v_h) / (2.0 * omega * nk)) * jnp.eye(nA)[None]
    return sx, coh


def exchange_sigma(dyn: dict, kern: DynsigKernels, x, *, heads=None):
    """Sigma^x[rho] = -w_term[V](rho_tot) on the window, (nk, nA, nA) Ry.

    The bare exchange of Eq. (1) of docs/theory/td_agw_dynamical.md with the
    full equal-time G^< = rho (coherences kept, §1.1): rho_tot = diag(f_frozen)
    (+) (diag f_eq + x) on the window, x = drho (nk, nA, nA) in the model's k
    order.  ``heads = (w_h, v_h, cell_volume)`` adds the q -> 0 term
    -v_h rho/(Omega N_k) of the window (the bare head; ``equilibrium_sigma``'s
    screened one is -w_h f/(Omega N_k)).
    """
    if dyn["V_q"] is None:
        _refuse("dynsig_exchange_needs_v", "V_q = None", "the run's bare V",
                "pass V_q to assemble_dynsig_bundle")
    nk = int(np.prod(kern.kgrid))
    b_lo, nA = dyn["band_lo"], dyn["n_active"]
    rho = x + _band_diag(dyn["f_full"][:, b_lo:b_lo + nA])
    f_frozen = dyn["f_full"].at[:, b_lo:b_lo + nA].set(0.0)
    sx = -_sigma_window(dyn, kern, rho, f_frozen, _w_R(dyn["V_q"], kern))
    if heads is not None:
        _, v_h, omega = (float(v) for v in heads)
        sx = sx - (v_h / (omega * nk)) * rho
    return sx


def literal_sx(dyn: dict, kern: DynsigKernels, x, W_R, *, h=None, coherences: bool = True):
    """Sigma^SX[rho] = -w_term[W](rho_tot) on the window, built in the instantaneous basis, Ry.

    rho_tot = diag(f_frozen) (+) U rho~ U^H, rho~ = U^H rho U, U the eigenvectors
    of ``h`` (nk, nA, nA; default diag eps^G).  ``coherences=False`` keeps only
    diag(rho~) (red twin: the static limit then fails at O(drho^coh)).  The
    literal Eq. (1) is literal_sx(x) - literal_sx(0); its frozen-W limit is
    -K^W drho of the TD model.
    """
    b_lo, nA = dyn["band_lo"], dyn["n_active"]
    f_eq = dyn["f_full"][:, b_lo:b_lo + nA]
    rho = x + _band_diag(f_eq)
    if h is None:
        h = _band_diag(dyn["e_full"][:, b_lo:b_lo + nA])
    U, _, _ = instantaneous_basis(h, rho)
    Uh = jnp.conj(jnp.swapaxes(U, -1, -2))
    rt = Uh @ rho @ U
    if not coherences:
        rt = _band_diag(jnp.real(jnp.diagonal(rt, axis1=-2, axis2=-1)))
    f_frozen = dyn["f_full"].at[:, b_lo:b_lo + nA].set(0.0)
    return -_sigma_window(dyn, kern, U @ rt @ Uh, f_frozen, W_R)


# ===========================================================================
# The pluggable model (tdagw.self_energy contract)
# ===========================================================================

_DYN_OPERANDS = ("psi_X", "psi_Y", "e_full", "f_full", "live", "W0_q", "tau", "alpha")


def _dyn_apply(static, params, rho, t):
    """dSigma = the ISDF static model + (1/2) P^T dSigma^dyn[P drho] (Ry -> Ha), Eq. (5)."""
    from tdagw.model import build_td_agw_apply
    from tdagw.self_energy import _ISDF_OPERANDS
    static_apply, kern, meta, rows, perm, nk, screening, dress, opts = static
    iso_params, dyn_ops = params
    out = static_apply(iso_params, rho, t)                          # Ha, TDGW order
    if screening == "frozen":
        return out
    ops, rho_eq = iso_params
    nA, nb_pad = meta["n_active"], meta["nb_pad"]
    model = dict(zip(_ISDF_OPERANDS, ops), kgrid=kern.kgrid, nspinor=meta["nspinor"])
    td_apply = build_td_agw_apply(kern.mesh, model)
    dyn = dict(meta, **dict(zip(_DYN_OPERANDS, dyn_ops)))

    def static_dsigma(x):                                           # Ry, model order
        xp = jnp.pad(x, ((0, 0), (0, nb_pad - nA), (0, nb_pad - nA)))[None]
        return td_apply(xp)[0, :, :nA, :nA]

    d = (rho - rho_eq[None])[:, :, :nA, :nA]
    x = jnp.take(d, jnp.asarray(perm), axis=1)                     # (ntr, nk, nA, nA)
    corr = lax.map(lambda xb: dynamic_correction(
        dyn, kern, xb, static_dsigma=static_dsigma, dress=dress, **opts)[0], x)
    corr = jnp.take(corr, jnp.asarray(rows), axis=1)
    return out + 0.5 * jnp.pad(corr, ((0, 0), (0, rho.shape[1] - nk), (0, 0), (0, 0)))


def dynsig_model(dyn: dict, mesh_xy: Mesh, *, k_frac, rho_eq, screening: str = "static_rpa",
                 dress: str = "none", **opts):
    """The instantaneous GW self-energy behind the dSigma contract, Ha.

    Parameters
    ----------
    dyn : :func:`assemble_dynsig_bundle`'s bundle (its ``td`` is the window model).
    k_frac, rho_eq : as :func:`tdagw.self_energy.isdf_model` (TDGW k order).
    screening : ``"frozen"`` (W = W0: exactly the ISDF TD-aGW model) |
        ``"static_rpa"`` (W rebuilt each call, Eqs. 3-5).
    dress : ``"none"`` | ``"instantaneous"`` (module docstring).
    opts : ``pauli``, ``same_side``, ``occupation`` (red twins only).
    """
    from tdagw.model import flat_k_rows
    from tdagw.self_energy import SelfEnergyModel, isdf_model
    if screening not in ("frozen", "static_rpa"):
        _refuse("dynsig_screening", repr(screening), "'frozen' | 'static_rpa'",
                "the GN-PPM / shared-pole members are design items (theory page)")
    iso = isdf_model(dyn["td"], mesh_xy, k_frac=k_frac, rho_eq=rho_eq)
    kern = make_dynsig_kernels(mesh_xy, dyn["kgrid"])
    rows = np.asarray(flat_k_rows(k_frac, dyn["kgrid"]), np.int64)
    meta = {k: v for k, v in dyn.items() if k not in _DYN_OPERANDS + ("td", "V_q", "receipts")}
    meta["nb_pad"] = int(dyn["td"]["nb_pad"])
    static = (iso.apply, kern, meta, tuple(rows.tolist()), tuple(np.argsort(rows).tolist()),
              int(rows.size), screening, dress, dict(opts))
    params = (iso.params, tuple(dyn[k] for k in _DYN_OPERANDS))
    return SelfEnergyModel(apply=functools.partial(_dyn_apply, static), params=params,
                           linear=(screening == "frozen"),
                           name=f"instantaneous GW ({screening}, dress={dress}) on {iso.name}")


def check_state(dyn: dict, flags: dict) -> None:
    """Refuse by name on the host when a correction's flags were raised (its dSigma is NaN)."""
    f = {k: int(np.asarray(flags[k])) for k in ("crossed", "out_of_rule", "degenerate")}
    if f["crossed"]:
        _refuse("dynsig_level_crossed_separator", f"{f['crossed']} active levels crossed e_sep",
                "a gapped instantaneous spectrum", "the transient metal needs the fractional "
                "contour route (design page)")
    if f["out_of_rule"]:
        _refuse("dynsig_transition_outside_rule", "an instantaneous transition outside "
                f"[{dyn['rule']['x_min']:.4g}, {dyn['rule']['x_max']:.4g}] Ry", "inside",
                "re-plan with static_rule(..., margin>0)")
    if f["degenerate"]:
        _refuse("dynsig_degenerate_population", f"{f['degenerate']} same-side pairs with "
                "|dE| <= deg_tol and different populations", "none", "the static limit of "
                "such a pair is a Drude term (design page, 'carriers')")



# ===========================================================================
# The GN-PPM member's on-shell collision term (§1.5): W frozen at the run's own poles
# ===========================================================================

def _copy_gw_deck(deck: str, workdir: str, overrides: dict | None = None) -> str:
    """The GW deck in ``workdir`` with ``write_restart_tensors = false`` (rank 0 writes).

    ``overrides`` {key: value text} replace (or append) those deck keys: a source
    at another sigma_regularization_ev, for instance.
    """
    os.makedirs(workdir, exist_ok=True)
    sets = dict(write_restart_tensors="false", **{str(k): str(v) for k, v in (overrides or {}).items()})
    lines = open(deck).read().splitlines()
    key_of = lambda ln: ln.split("=", 1)[0].strip() if "=" in ln and not ln.lstrip().startswith("#") else None
    seen = set()
    for i, ln in enumerate(lines):
        if key_of(ln) in sets:
            seen.add(key_of(ln))
            lines[i] = f"{key_of(ln)} = {sets[key_of(ln)]}"
    lines += [f"{k} = {v}" for k, v in sets.items() if k not in seen]
    out = os.path.join(workdir, os.path.basename(deck))
    open(out, "w").write("\n".join(lines) + "\n")
    for name in ("dipole.h5",):              # the deck's implicit companions (not on the deck)
        src = os.path.join(os.path.dirname(os.path.abspath(deck)), name)
        dst = os.path.join(workdir, name)
        if os.path.exists(src) and not os.path.lexists(dst):
            os.symlink(os.path.realpath(src), dst)
    return out


def gw_window_source(gw_deck: str, mesh: Mesh, *, workdir: str, screening: bool = False,
                     overrides: dict | None = None, say=print) -> dict:
    """The collision term's source: the GW run's own carrier, screening model and head.

    A GN-PPM run persists W(0) but neither W(i w_p) nor its pole model, and a
    shared-pole one-shot keeps its ``model.h5`` but not its W stage's inputs, so the
    GW driver (``gw.gw_jax.main``) runs in this process on a copy of ``gw_deck`` in
    ``workdir`` (``write_restart_tensors = false``: the run's own restart is never
    written; ``overrides`` set deck keys of the copy) and
    ``gw.ppm_pipeline.observe_window_sigma_source`` reads the Sigma stage at its
    return: no GW stage is re-implemented, and the window Sigma at rho_eq is the
    run's (§6.1).  Collective: every process calls it.  Must run outside another
    driver's ``RunSession`` (the GW driver owns the timing table and stdout while
    it runs).  Returns the observer's dict (``member`` "gn_ppm" or "shared_pole",
    ``wfns``, ``meta``, ``config``, ``head``, ``efermi_ry``, ``request``; GN-PPM:
    ``poles``, ``V_q``, ``quad`` and with ``screening`` ``fit_plan``,
    ``W_supports``; shared pole: ``model``, ``occupation_state``, ``linalg`` and
    with ``screening`` ``screening_inputs``, ``recipe_inputs``) plus ``poles`` (the
    window Sigma's model: the ``PPMBuildResult``, or
    ``gw.ppm_sigma.shared_pole_window_source`` of the run's model), ``plan``
    (``gw.ppm_sigma.window_sigma_plan`` on the run's own on-shell frequency list),
    ``deck``, ``workdir``, ``rc``, ``wall_s``.
    """
    import time
    import gw.gw_jax as gw_driver
    from jax.experimental import multihost_utils
    from gw.ppm_pipeline import observe_window_sigma_source
    from gw.ppm_sigma import shared_pole_window_source, window_sigma_plan
    t0 = time.perf_counter()
    workdir = os.path.abspath(workdir)
    copy = os.path.join(workdir, os.path.basename(gw_deck))
    if jax.process_index() == 0:
        _copy_gw_deck(gw_deck, workdir, overrides)
    multihost_utils.sync_global_devices("dynsig gw_window_source deck")
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        with observe_window_sigma_source(screening=screening) as src:
            rc = gw_driver.main(["-i", os.path.basename(copy)])
    finally:
        os.chdir(cwd)
    member = src.get("member")
    if member not in ("gn_ppm", "shared_pole"):
        _refuse("dynsig_window_source", f"no window Sigma stage ran in {gw_deck}",
                "a compute_mode = gn_ppm deck or a scalar sigma_w_model = shared_pole deck",
                "point the source at a GN-PPM or shared-pole GW run")
    mode = str(getattr(src["config"].compute_mode, "value", src["config"].compute_mode))
    if member == "gn_ppm" and mode != "gn_ppm":
        _refuse("dynsig_gnppm_mode", f"compute_mode = {mode}", "gn_ppm (the ordered GN fit the "
                "branch rates are measured on)", "use a GN-PPM run")
    if member == "shared_pole":
        if screening and ("screening_inputs" not in src or "recipe_inputs" not in src):
            _refuse("dynsig_spm_source", f"no shared-pole W stage observed in {gw_deck}",
                    "a one-shot shared-pole run (its W stage runs in this process)",
                    "drop restart from the deck")
        src["poles"] = shared_pole_window_source(src["model"], src["meta"],
                                                 efermi_ry=src["efermi_ry"], linalg=src["linalg"])
    req = src["request"]
    freqs = None if req is None else np.asarray(req.frequencies_ry) + float(src["efermi_ry"])
    src = dict(src, plan=window_sigma_plan(src["config"], head=src["head"],
                                           frequencies_abs_ry=freqs),
               deck=os.path.abspath(gw_deck), workdir=workdir, rc=rc,
               wall_s=time.perf_counter() - t0)
    what = ("GN-PPM poles at the run's own W_0" if member == "gn_ppm" else
            f"shared-pole model K {min(src['model']['K'])}-{max(src['model']['K'])}")
    say(f"  collision source: GW driver on {gw_deck} in {workdir} (rc {rc}, "
        f"{src['wall_s']:.1f} s): {what}, {int(src['meta'].nk_tot)} k, N_mu {int(src['meta'].n_rmu)}")
    return src


#: ``window_collision``'s screening (``[lorrax] collision_screening``).
COLLISION_SCREENINGS = ("frozen", "refreshed", "shared_pole")


def collision_screening(source: dict, *, mesh: Mesh, b_lo: int, n_active: int,
                        odd_same_side: bool = True, gamma_ry: float = 0.0) -> dict:
    """The held objects of the refreshed screening of the GN-PPM collision term (§7.1).

    W(z_s, t) = [1 - W0(z_s) dchi0(z_s, t)]^-1 W0(z_s) at the GN supports z_s = 0 and
    i omega_p (Eq. 6, ``gw.ppm_sigma.screening_update``), then the run's own pole fit.
    Everything comes from ``source`` (:func:`gw_window_source` with ``screening=True``):
    the run's two W samples and fit plan, its carrier (psi of every band of the chi0
    sum at the centroids in the run's packed order, eps^G, the step occupations), its
    static Laplace rule ``quad`` and, built by the driver's own call, its probe rule.
    dchi0 (:func:`delta_chi0`, dress none: U = 1, e = eps^G) is the change of the
    run's chi0 when the window populations move: cross-gap pairs on the rules,
    same-side pairs exact at each support, chi0^ff never formed.  k rows go to flat
    C order (the run's q convention); the screening bundle is not the propagator's.
    ``gamma_ry`` > 0: the static support's same-side pairs take the principal value of
    that width (:func:`delta_chi0`'s ``same_side_gamma``; a driven distribution makes
    symmetry-equivalent k unequal, whose exact static term is a Drude pole); 0 refuses
    such pairs by name.

    Returns dict: ``dchi`` (the two jitted dchi0 maps (arrays, U, e, f) -> (dchi,
    flags), static then probe), ``arrays``, ``dyn`` (static part), ``U``, ``e_A``
    (device, replicated), ``flat_rows`` (flat k row j <- carrier row flat_rows[j]),
    ``W0`` (the two supports), ``plan`` (``fit_plan``), ``rules``, ``census`` (the
    jitted pole census; logical modes are those with Omega_0 != 0), ``nbf``, ``poles0``.
    """
    from gw.minimax_screening import build_imag_probe_response_rule
    from tdagw.model import flat_k_rows
    from common.collectives import device_put_process_local
    if "fit_plan" not in source or "W_supports" not in source:
        _refuse("dynsig_collision_screening_source", "a source without the run's W samples",
                "gw_window_source(..., screening=True)", "rebuild the source with screening=True")
    wfns, meta, poles0 = source["wfns"], source["meta"], source["poles"]
    plan, quad = source["fit_plan"], source["quad"]
    carrier = wfns.green_parent
    kplan = carrier.plan
    rows = getattr(kplan, "parent_full_rows", None)
    nk = int(meta.nk_tot)
    if rows is not None and not np.array_equal(np.asarray(rows), np.arange(nk)):
        _refuse("dynsig_collision_screening_rows", "a carrier whose parent rows are permuted",
                "parent rows = the full zone in the bundle's order", "run the GW parent "
                "without symmetry (as runs/CrI3/516)")
    kgrid = tuple(int(n) for n in meta.kgrid)
    rows_b = np.asarray(flat_k_rows(np.asarray(kplan.k_parent_frac, np.float64), kgrid))
    flat_rows = np.argsort(rows_b)                       # flat row j <- carrier row flat_rows[j]
    sl = wfns.slices                     # the logical chi0 bands (the world-size pads are inert)
    nbf = min(int(sl.b4_chi), int(sl.b4_logical or sl.b4_chi)) - int(sl.b0)
    if int(wfns.slices.b0) != 0:
        _refuse("dynsig_collision_screening_bands", f"chi0 sum from band {int(wfns.slices.b0)}",
                "every band from 0", "run the GN-PPM bundle on the full band set")
    e_all = np.asarray(jax.device_get(wfns.enk), np.float64)[flat_rows, :nbf]
    f_all = np.asarray(jax.device_get(wfns.occ), np.float64)[flat_rows, :nbf]
    occ, emp = f_all > 0.5, f_all < 0.5
    rule = dict(tau=np.asarray(quad.tau, np.float64), alpha=np.asarray(quad.alpha, np.float64),
                x_min=float(quad.x_min), x_max=float(quad.x_max),
                max_error=float(quad.max_error), provenance=quad.provenance,
                e_sep=0.5 * (float(e_all[occ].max()) + float(e_all[emp].min())))
    z = complex(plan["probe_omega"])
    if abs(z.real) > 0.0 or z.imag <= 0.0:
        _refuse("dynsig_collision_screening_probe", f"probe {z!r}", "i omega_p (GN-PPM)",
                "use a GN-PPM run")
    ordered = bool(plan["fit"]["ordered_orientations"])
    q_p = build_imag_probe_response_rule(quad, float(z.imag), source["config"].minimax_config,
                                         print_fn=None, with_odd_kernel=ordered)
    prule = dict(tau=np.asarray(q_p.tau, np.float64), alpha=np.asarray(q_p.alpha, np.float64),
                 beta=(np.zeros(len(q_p.tau)) if q_p.alpha_odd is None
                       else np.asarray(q_p.alpha_odd, np.float64)),
                 omega_p=float(z.imag), max_error=float(q_p.max_error),
                 provenance=q_p.provenance)
    take = jax.jit(lambda a: jnp.take(a, jnp.asarray(flat_rows), axis=0)[:, :nbf],
                   out_shardings=NamedSharding(mesh, P(None, None, None, "x")))
    psi_flat = take(carrier.psi_nmu)
    W0 = tuple(source["W_supports"])
    td_min = dict(kgrid=kgrid, band_lo=int(b_lo), n_bands=int(n_active), n_rmu=int(meta.n_rmu))
    dyn = assemble_dynsig_bundle(td_min, psi_flat, e_all, f_all, W0[0], mesh_xy=mesh,
                                 rule=rule)
    del psi_flat
    kern = make_dynsig_kernels(mesh, kgrid)
    arrays, static = split_bundle(dyn)
    win = slice(int(b_lo), int(b_lo) + int(n_active))
    rep_ = NamedSharding(mesh, P())
    put = lambda a: device_put_process_local(np.ascontiguousarray(a), rep_)

    def dchi_map(r):
        g = float(gamma_ry) if r is None else 0.0
        return jax.jit(lambda A, U, e, f: delta_chi0(
            {**static, **A}, kern, U, e, f, rule=r, odd_same_side=odd_same_side,
            same_side_gamma=g))

    @jax.jit
    def census(om, valid, valid_prev, w_t0, w_tp, om0, valid0, w00, w0p):
        """Pole census against the run's (every operand an argument: no global closure)."""
        live = om0 != 0.0                         # logical modes (pad modes are born dead)
        both = valid & valid0
        mx = lambda a: jnp.max(jnp.abs(a))
        return dict(
            n_invalid=jnp.sum(live & ~valid), flips_prev=jnp.sum(live & (valid != valid_prev)),
            flips_eq=jnp.sum(live & (valid != valid0)), n_logical=jnp.sum(live),
            d_omega=jnp.max(jnp.where(both, jnp.abs(om - om0), 0.0)),
            dW_static=mx(w_t0 - w00) / mx(w00), dW_probe=mx(w_tp - w0p) / mx(w0p))

    return dict(dchi=(dchi_map(None), dchi_map(prule)), arrays=arrays, dyn=static, kern=kern,
                U=put(np.broadcast_to(np.eye(int(n_active), dtype=complex),
                                      (nk, int(n_active), int(n_active))).copy()),
                e_A=put(e_all[:, win]), flat_rows=flat_rows, W0=W0, plan=plan,
                rules=dict(static=rule, probe=prule), census=census, nbf=nbf, poles0=poles0,
                gamma_ry=float(gamma_ry))


def screen_populations(scr: dict, f_rows, *, valid_prev=None):
    """W(t) at both supports and its pole model for the window populations ``f_rows``.

    ``f_rows`` (nk, n_A) on the carrier's rows.  Returns (poles, W, rec): ``rec`` has
    the walls (``dchi_s``: dchi0 at both supports; ``fit_s``: the two Dyson solves and
    the fit, ``gw.ppm_sigma.screening_update``), the pole census against the run's
    poles (``n_invalid``, ``flips_prev`` against ``valid_prev``, ``flips_eq``,
    ``d_omega_mev`` over the modes valid in both, ``dW`` = max|W(z) - W0(z)|/max|W0(z)|
    at 0 and i omega_p), ``dchi_max`` (max|dchi0| at both supports, Ry^-1); a raised
    dchi0 gate refuses by name.
    """
    import time
    from common.collectives import device_put_process_local
    from common.units import RYD_TO_EV
    from gw.ppm_sigma import screening_update
    f_flat = np.asarray(f_rows, np.float64)[scr["flat_rows"]]
    rep_ = NamedSharding(scr["kern"].mesh, P())
    f_dev = device_put_process_local(np.ascontiguousarray(f_flat), rep_)
    t0 = time.perf_counter()
    outs = [fn(scr["arrays"], scr["U"], scr["e_A"], f_dev) for fn in scr["dchi"]]
    jax.block_until_ready([o[0] for o in outs])
    t1 = time.perf_counter()
    for _, flags in outs:
        check_state(scr["dyn"], flags)
    dchi_max = [float(v) for v in jax.device_get([jnp.max(jnp.abs(o[0])) for o in outs])]
    poles, W = screening_update(scr["W0"], [o[0] for o in outs], scr["plan"])
    jax.block_until_ready((poles.Omega_q, W[0], W[1]))
    t2 = time.perf_counter()
    p0 = scr["poles0"]
    prev = p0.valid_mask_q if valid_prev is None else valid_prev
    c = {k: float(v) for k, v in jax.device_get(scr["census"](
        poles.Omega_q, poles.valid_mask_q, prev, W[0], W[1], p0.Omega_q, p0.valid_mask_q,
        scr["W0"][0], scr["W0"][1])).items()}
    rec = dict(dchi_s=t1 - t0, fit_s=t2 - t1, n_invalid=int(c["n_invalid"]),
               flips_prev=int(c["flips_prev"]), flips_eq=int(c["flips_eq"]),
               d_omega_mev=c["d_omega"] * RYD_TO_EV * 1e3, dW=[c["dW_static"], c["dW_probe"]],
               n_logical=int(c["n_logical"]), dchi_max=dchi_max,
               degenerate_pv=int(jax.device_get(outs[0][1]["degenerate_pv"])))
    return poles, W, rec


# ===========================================================================
# The shared-pole member: the W stage of the GW run re-run at the populations
# ===========================================================================

#: Residue-weighted census cuts (eV): the weight of the poles below each.
CENSUS_CUTS_EV = (0.3, 1.0)
#: The census histogram's pole-energy bin edges (eV), log spaced.
CENSUS_BINS_EV = np.geomspace(1.0e-3, 2.0e2, 64)


def pole_census(path, *, cuts_ev=CENSUS_CUTS_EV, bins_ev=CENSUS_BINS_EV) -> dict:
    """Residue-weighted census of a shared-pole model file (host, h5py ``'r'``).

    For the ordered (W 16) or even (W 14) store, parent q's pole j has energy
    Omega_j = sqrt(Lambda_j) and residue R_j = b_j b_j^H / (2 Omega_j) on the
    positive-frequency side (the negative side is the partner parent's, the
    same set over the zone), so -Im W^c(q, omega > 0) = pi sum_j R_j
    delta(omega - Omega_j).  The census weighs pole j by tr R_j = |b_j|^2 /
    (2 Omega_j) (its spectral weight in the centroid basis, one basis for every
    state of a run) and by its share of the static value, |b_j|^2 / (2 Omega_j^2)
    (W^c(0) = -sum_j b_j b_j^H / (2 Omega_j^2) per side).  Returns
    :func:`census_summary`.
    """
    import h5py
    from common.units import RYD_TO_EV
    with h5py.File(path, "r") as f:
        raw = f["header_json"][()]
        header = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
        K = np.asarray(header["K"], np.int64)
        om, wr, ws = [], [], []
        for q in range(int(f["poles2_ry2"].shape[0])):
            lam = np.asarray(f["poles2_ry2"][q, :K[q]], np.float64)
            b = np.asarray(f["factor"][q, :, :, :K[q]])
            norm2 = np.sum(np.abs(b) ** 2, axis=(0, 1))
            om.append(np.sqrt(lam) * RYD_TO_EV)
            wr.append(norm2 / (2.0 * np.sqrt(lam)))
            ws.append(norm2 / (2.0 * lam))
    return census_summary(om, wr, ws, cuts_ev=cuts_ev, bins_ev=bins_ev)


def census_summary(om, wr, ws, *, cuts_ev=CENSUS_CUTS_EV, bins_ev=CENSUS_BINS_EV) -> dict:
    """The census of per-q pole energies ``om`` (eV) with residue and static weights ``wr``, ``ws``.

    Returns dict: ``K`` (poles per q), ``omega_min_ev``, ``frac_residue_below`` and
    ``frac_static_below`` (q-summed fractions per cut, eV), ``n_below`` (poles per
    cut), ``hist_residue`` / ``hist_static`` (q-summed weights in ``bins_ev``),
    ``hist_residue_q`` (per q), ``bins_ev``, ``residue_total``, ``static_total``.
    """
    o, r, s = (np.concatenate([np.asarray(v, np.float64) for v in x]) if len(x) else np.zeros(0)
               for x in (om, wr, ws))
    rt, st = float(r.sum()), float(s.sum())
    return dict(
        K=[int(np.size(v)) for v in om], omega_min_ev=float(o.min()) if o.size else None,
        frac_residue_below={f"{c:g}": float(r[o < c].sum() / rt) for c in cuts_ev},
        frac_static_below={f"{c:g}": float(s[o < c].sum() / st) for c in cuts_ev},
        n_below={f"{c:g}": int(np.sum(o < c)) for c in cuts_ev},
        hist_residue=np.histogram(o, bins=bins_ev, weights=r)[0].tolist(),
        hist_static=np.histogram(o, bins=bins_ev, weights=s)[0].tolist(),
        hist_residue_q=[np.histogram(np.asarray(oq, np.float64), bins=bins_ev,
                                     weights=np.asarray(rq, np.float64))[0].tolist()
                        for oq, rq in zip(om, wr)],
        bins_ev=np.asarray(bins_ev, np.float64).tolist(), residue_total=rt, static_total=st)


def _occupation_duck(f_kn, mu_ry: float):
    """The occupation state of an instantaneous TD state (duck-typed ``gw.efermi.OccupationState``).

    ``f_kn`` (nk, nb) every band's population on the GW run's W carrier: the
    equilibrium step occupations of the frozen bands, diag rho of the window
    (dress none).  ``mu_ry`` is the run's chemical potential, so the census frame
    and the bank's references do not move with the populations.
    """
    from types import SimpleNamespace
    f = np.asarray(f_kn, np.float64)
    return SimpleNamespace(f_kn=f, mu_ry=float(mu_ry), smearing_family="TD-aGW instantaneous",
                           smearing_width_ry=0.0, occ_hash=None, n_electrons=float(f.sum()))


def spm_screening(source: dict, *, mesh: Mesh, b_lo: int, n_active: int,
                  support_sites: str = "", scratch: str) -> dict:
    """The held objects of the shared-pole refit of the collision term (§1.4, §7.1).

    A refit is the GW run's own W stage (``gw.shared_pole_screening.
    screen_shared_poles``: response bank -> moments -> constructor -> model)
    re-run at the instantaneous populations, with the recipe re-resolved at the
    instantaneous census (``gw.shared_pole_recipe.bind_shared_pole_census`` +
    ``resolve_shared_pole_recipe``).  Every input is the run's, taken at its seams
    (``gw_window_source(screening=True)``: ``screening_inputs``,
    ``recipe_inputs``): the W carrier (psi of every band at the centroids, eps^G,
    the step occupations), V, the symmetry tables and the run's occupation state
    (its mu).  The q -> 0 head is not rebuilt (the run's head is frozen, as in the
    GN-PPM refresh).  ``support_sites`` "" is the run's production ladder (V1);
    otherwise a ``sigma_w_support_sites_ev`` override (V2, low-energy supports:
    ``"<line eV> | <imaginary eV> [| <line height eV>]"``), resolved at the run's
    census.  Models go to ``scratch``; a refit's directory is removed when its
    label is reused.

    Returns dict: ``inputs`` (the run's W-stage arguments), ``recipe_inputs``,
    ``meta`` (the refit's metadata copy), ``config`` (the run's with the head off
    and the support override), ``f_eq`` (the run's occupation table on the W
    carrier), ``rows`` (window k rows of the collision bundle -> W-carrier rows),
    ``win``, ``mu_ry``, ``recipe0`` (the variant's recipe at the run's census),
    ``variant``, ``support_sites``, ``scratch``, ``efermi_ry``, ``linalg``, ``n``.
    """
    import copy as _copy
    from jax.experimental import multihost_utils
    from gw.gw_config import HeadCorrection
    from gw.shared_pole_recipe import parse_support_sites
    if "screening_inputs" not in source or "recipe_inputs" not in source:
        _refuse("dynsig_spm_source", "a source without the run's W-stage inputs",
                "gw_window_source(..., screening=True) on a shared-pole deck",
                "rebuild the source with screening=True")
    inp = source["screening_inputs"]
    wfns_w, meta_run, config_run, kw = inp["wfns"], inp["meta"], inp["config"], inp["kwargs"]
    state = kw.get("occupation_state")
    f_eq = np.asarray(jax.device_get(wfns_w.occ if state is None else state.f_kn), np.float64)
    mu = (float(meta_run.shared_pole_census["mu_ry"]) if state is None else float(state.mu_ry))
    win = slice(int(b_lo), int(b_lo) + int(n_active))
    enk_w = np.asarray(jax.device_get(wfns_w.enk), np.float64)
    enk_s = np.asarray(jax.device_get(source["wfns"].enk), np.float64)
    if enk_s.shape[0] != enk_w.shape[0] or not np.array_equal(enk_s[:, win], enk_w[:, win]):
        _refuse("dynsig_spm_carriers", "the Sigma and W carriers' window energies differ",
                "one k order and band axis", "run the shared-pole source on an identity-only grid")
    if np.any((f_eq[:, win] != 0.0) & (f_eq[:, win] != 1.0)):
        _refuse("dynsig_spm_step", "fractional equilibrium occupations in the window",
                "an insulating step reference (0/1)", "metals are not in this member")
    override = parse_support_sites(support_sites)
    sigma = config_run.sigma
    if override is not None:
        sigma = dataclasses.replace(sigma, w_support_sites_ev=override["text"])
    config = dataclasses.replace(
        config_run, sigma=sigma,
        head=dataclasses.replace(config_run.head, correction=HeadCorrection.OFF))
    scr = dict(inputs=inp, recipe_inputs=dict(source["recipe_inputs"]), meta=_copy.copy(meta_run),
               config=config, f_eq=f_eq, rows=np.arange(enk_w.shape[0]), win=win, mu_ry=mu,
               variant=("production" if override is None else "override"),
               support_sites="" if override is None else override["text"], scratch=scratch,
               efermi_ry=float(source["efermi_ry"]), linalg=source.get("linalg"), n=0)
    # The production variant's reference is the run's own recipe; an override is
    # resolved at the run's census (the same resolver, another support set).
    scr["recipe0"] = meta_run.shared_pole_recipe if override is None else _spm_recipe(scr, f_eq)
    if jax.process_index() == 0:
        os.makedirs(scratch, exist_ok=True)
    multihost_utils.sync_global_devices("dynsig spm_screening scratch")
    return scr


def _spm_recipe(scr: dict, f_kn) -> dict:
    """The refit's recipe at the census of the populations ``f_kn``: the run's resolver on a meta copy.

    Also installs that census and the resolver's fresh capacity ledger on ``scr['meta']``.
    """
    import copy as _copy
    from gw.shared_pole_recipe import bind_shared_pole_census, resolve_shared_pole_recipe
    inp, rin = scr["inputs"], scr["recipe_inputs"]
    meta = _copy.copy(inp["meta"])
    bind_shared_pole_census(inp["wfns"], meta, occupation_state=_occupation_duck(f_kn, scr["mu_ry"]),
                            trs_allowed=rin["trs_allowed"], state_capacity=rin["state_capacity"],
                            kweights=rin["kweights"])
    recipe = resolve_shared_pole_recipe(scr["config"], inp["wfns"], meta,
                                        mesh_xy=inp["kwargs"]["mesh_xy"],
                                        print_fn=lambda *a, **k: None,
                                        support_reads_ev=rin["support_reads_ev"])
    scr["meta"].shared_pole_census = meta.shared_pole_census
    scr["meta"].shared_pole_capacity = meta.shared_pole_capacity
    return recipe


def _recipe_sites_equal(a: dict, b: dict) -> bool:
    """Whether two recipes sample the same supports (every role's z, the held set, the widths)."""
    keys = ("z_ry", "role", "distinct_id", "held", "support_pair")
    return (all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in keys)
            and all(a[k] == b[k] for k in ("imaginary_width", "infinity_width",
                                             "line_direction_cap", "pole_budget", "height_ev")))


def spm_refit(scr: dict, f_rows, *, label: str, census: bool = True, say=None):
    """The shared-pole W of the window populations ``f_rows`` (nk, n_A) on the bundle's k rows.

    The run's W stage (``screen_shared_poles``) at the occupation table f_eq with the
    window block set to ``f_rows`` (dress none: U = 1, e = eps^G, f = diag rho), after
    its recipe is re-resolved at that census (whether a support moved is recorded).
    Returns (poles, rec): ``poles`` is the window Sigma's source
    (``gw.ppm_sigma.shared_pole_window_source``: the new model with the refit's
    metadata), ``rec`` the walls (``recipe_s``, ``screen_s``; rank 0: the bank,
    moments and constructor receipts' seconds and ``census_s``), ``K`` per q, the
    constructor's gate rows that did not pass, ``sites_moved`` and, with
    ``census``, the residue-weighted :func:`pole_census` (rank 0).  A gate that
    refuses the model raises here by name: no refresh proceeds on a model the
    constructor did not certify.
    """
    import shutil
    import time
    from jax.experimental import multihost_utils
    from gw.ppm_sigma import shared_pole_window_source
    from gw.shared_pole_screening import screen_shared_poles
    inp = scr["inputs"]
    f = scr["f_eq"].copy()
    f[scr["rows"], scr["win"]] = np.asarray(f_rows, np.float64)
    t0 = time.perf_counter()
    recipe = _spm_recipe(scr, f)
    moved = not _recipe_sites_equal(recipe, scr["recipe0"])
    scr["meta"].shared_pole_recipe = recipe
    t1 = time.perf_counter()
    root = os.path.join(scr["scratch"], label + "_shared_pole")
    if jax.process_index() == 0 and os.path.isdir(root):
        shutil.rmtree(root)
    multihost_utils.sync_global_devices("dynsig spm_refit slot")
    kw = dict(inp["kwargs"], run_dir=scr["scratch"], label=label, tensors_filename=None,
              occupation_state=_occupation_duck(f, scr["mu_ry"]), head_resolver=None,
              print_fn=(say if say is not None else (lambda *a, **k: None)))
    out = screen_shared_poles(inp["wfns"], inp["V_q"], scr["meta"], scr["config"], **kw)
    t2 = time.perf_counter()
    handle = out["shared_pole"]
    rec = dict(recipe_s=t1 - t0, screen_s=t2 - t1, K=[int(k) for k in handle["K"]],
               sites_moved=bool(moved), label=label)
    if jax.process_index() == 0:
        rec.update(_refit_receipts(root))
        if census:
            t3 = time.perf_counter()
            rec["census"] = pole_census(handle["path"])
            rec["census_s"] = time.perf_counter() - t3
    scr["n"] += 1
    import copy as _copy
    poles = shared_pole_window_source(handle, _copy.copy(scr["meta"]),
                                      efermi_ry=scr["efermi_ry"], linalg=scr["linalg"])
    return poles, rec


def _refit_receipts(root: str) -> dict:
    """Walls and the non-passing gate rows of one refit, from its receipts (rank 0)."""
    out = {}
    for stage in ("bank", "moments", "constructor"):
        p = os.path.join(root, stage + "_receipt.json")
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        sec = r.get("seconds") or {}
        if isinstance(sec, dict):
            out[stage + "_seconds"] = {k: float(v) for k, v in sec.items()
                                       if isinstance(v, (int, float))}
        if stage == "bank":
            rule = r.get("rule") or {}
            out["bank_rule"] = dict(interval_ry=rule.get("interval_ry"), nodes=r.get("nodes"),
                                    reused=rule.get("reused"),
                                    decay_rate=rule.get("decay_rate_ry_inv"))
        if stage == "constructor":
            # The gate rows are per parent q (``q_receipts[q]["gates"]``): count each
            # (name, status) that is not PASS / NOT_MEASURED over q, and keep the worst
            # value of the accuracy rows (held-site W, moment defects, passivity).
            rows = [g for qr in (r.get("q_receipts") or []) for g in (qr.get("gates") or [])]
            bad = {}
            for g in rows:
                if g.get("status") not in ("PASS", "NOT_MEASURED"):
                    key = f"{g.get('name')}:{g.get('status')}"
                    bad[key] = bad.get(key, 0) + 1
            out["gates_not_pass"] = [dict(gate=k, n_q=n) for k, n in sorted(bad.items())]
            out["gates_counted"] = len(rows)
            worst = {}
            for g in rows:
                v = g.get("value")
                if g.get("name") in _REFIT_GATE_VALUES and isinstance(v, (int, float)):
                    worst[g["name"]] = max(worst.get(g["name"], float(v)), float(v))
            out["gate_values_max"] = worst
    return out


#: The constructor gate rows whose worst value over q a refit records.
_REFIT_GATE_VALUES = ("held_w", "full_m1_defect", "full_m3_defect", "passivity",
                      "retained_subspace_moments", "sc_rebuild")


def _model_payload_cmp(path_a: str, path_b: str):
    """Two shared-pole models' payloads (``factor``, ``poles2_ry2``, ``K``): bitwise and max |a - b| (rank 0)."""
    if jax.process_index() != 0:
        return None
    import h5py
    out = {}
    with h5py.File(path_a, "r") as fa, h5py.File(path_b, "r") as fb:
        for key in ("factor", "poles2_ry2"):
            a, b = fa[key][()], fb[key][()]
            same = a.shape == b.shape
            out[key] = dict(bitwise=bool(same and np.array_equal(a, b)),
                            max_abs=float(np.max(np.abs(a - b))) if same else None,
                            shapes=[list(a.shape), list(b.shape)])
        ka = json.loads(fa["header_json"][()])["K"]
        kb = json.loads(fb["header_json"][()])["K"]
        out["K_equal"] = bool(list(ka) == list(kb))
    out["bitwise"] = all(out[k]["bitwise"] for k in ("factor", "poles2_ry2")) and out["K_equal"]
    return out


def _refit_walls(sr: dict) -> dict:
    """One refit's walls (s): the bank, the moments, the constructor (its rounds excluded)."""
    sec = lambda key: sr.get(key + "_seconds") or {}
    con = {k: v for k, v in sec("constructor").items() if k != "rounds"}
    return dict(bank=float(sec("bank").get("total", 0.0)),
                bank_rule=float(sec("bank").get("rule_construction", 0.0)),
                bank_direct=float(sec("bank").get("direct_dispatch", 0.0)),
                bank_dyson=float(sec("bank").get("sample_dyson_dispatch", 0.0)),
                bank_line=float(sec("bank").get("line_selection", 0.0)),
                moments=float(sec("moments").get("total", 0.0)),
                constructor=float(sum(con.values())),
                gram_reduction=float(con.get("gram_reduction", 0.0)),
                direction_selection=float(con.get("direction_selection", 0.0)),
                writer=float(con.get("writer", 0.0)))


def spm_record(screens: list, eq: dict | None = None) -> dict:
    """One shared-pole refresh's refit summary over its trials (``spm_refit`` records).

    Walls are summed over the trials; ``K`` is the first trial's per-q pole count and
    ``dK_eq`` its change against the equilibrium refit; the census (rank 0) is the
    first trial's.
    """
    walls = [_refit_walls(sr) for sr in screens]
    total = lambda key: sum(w[key] for w in walls)
    r0 = screens[0]
    out = dict(member="shared_pole", refit_s=sum(sr["recipe_s"] + sr["screen_s"] for sr in screens),
               recipe_s=sum(sr["recipe_s"] for sr in screens),
               walls={k: total(k) for k in walls[0]}, K=list(r0["K"]),
               K_min=min(min(sr["K"]) for sr in screens), K_max=max(max(sr["K"]) for sr in screens),
               sites_moved=any(sr["sites_moved"] for sr in screens),
               gates_not_pass=sorted({g["gate"] for sr in screens for g in sr.get("gates_not_pass", [])}),
               gate_values_max={k: max(sr.get("gate_values_max", {}).get(k, float("-inf"))
                                       for sr in screens)
                                for k in (screens[0].get("gate_values_max") or {})},
               bank_rule=r0.get("bank_rule"), census_s=sum(sr.get("census_s", 0.0) for sr in screens))
    if eq is not None:
        out["dK_eq"] = int(np.sum(np.abs(np.asarray(r0["K"]) - np.asarray(eq["K"]))))
    if r0.get("census") is not None:
        c = r0["census"]
        out["census"] = {k: c[k] for k in ("omega_min_ev", "frac_residue_below", "frac_static_below",
                                           "n_below", "hist_residue", "hist_static",
                                           "residue_total", "static_total")}
    return out


def spm_line(rec: dict) -> str:
    """The shared-pole part of a refresh line (``window_collision``'s log)."""
    sc = rec["screen"]
    w = sc["walls"]
    c = sc.get("census") or {}
    nb1, fr1, fs1 = ((c.get("n_below") or {}).get("1"), (c.get("frac_residue_below") or {}).get("1"),
                     (c.get("frac_static_below") or {}).get("1"))
    return (f"; W(t) full refit {sc['refit_s']:.2f} s (bank {w['bank']:.2f} [rules {w['bank_rule']:.2f}, "
            f"chi0 {w['bank_direct']:.2f}, Dyson {w['bank_dyson']:.2f}, lines {w['bank_line']:.2f}], "
            f"moments {w['moments']:.2f}, constructor {w['constructor']:.2f} [pencil/Gram "
            f"{w['gram_reduction']:.2f}, directions {w['direction_selection']:.2f}, writer "
            f"{w['writer']:.2f}], recipe {sc['recipe_s']:.2f})"
            + ("" if "frozen_s" not in rec else
               f", frozen sweep {rec['frozen_s']:.2f} s ({rec['frozen_nodes']['distinct_tau']} tau), "
               f"active {rec['active_s']:.2f} s")
            + (f", rebase {rec['rebase_s']:.2f} s" if rec.get("rebase_s") else "")
            + f"; model K {sc['K_min']}-{sc['K_max']}"
            + (f" (sum |dK| vs rho_eq {sc['dK_eq']})" if "dK_eq" in sc else "")
            + f", supports {'MOVED' if sc['sites_moved'] else 'fixed'}, gates not PASS: "
            f"{', '.join(sc['gates_not_pass']) or 'none'}"
            + ("" if nb1 is None else
               f"; poles below 1 eV {nb1} ({fr1:.2e} of the residue weight, {fs1:.2e} of the "
               f"static), Omega_min {c['omega_min_ev']:.3f} eV"))


def spm_t0_line(screen_eq: dict) -> str:
    """The shared-pole part of the t0 line: the refit at rho_eq against the run's model."""
    cmp = screen_eq.get("model_vs_run") or {}
    w = _refit_walls(screen_eq)
    c = screen_eq.get("census") or {}
    return (f"; screening shared_pole ({screen_eq['variant']}"
            + (f" supports {screen_eq['support_sites']}" if screen_eq.get("support_sites") else "")
            + f"): refit at rho_eq {screen_eq['recipe_s'] + screen_eq['screen_s']:.2f} s (bank "
            f"{w['bank']:.2f}, moments {w['moments']:.2f}, constructor {w['constructor']:.2f}), model "
            + ("bitwise the run's" if cmp.get("bitwise") else
               "NOT bitwise the run's" if cmp else "vs the run's on rank 0")
            + f", K {min(screen_eq['K'])}-{max(screen_eq['K'])}, supports "
            f"{'MOVED' if screen_eq['sites_moved'] else 'fixed'}"
            + ("" if not c else f", Omega_min {c['omega_min_ev']:.3f} eV"))


def window_branch_rates(tabs: dict):
    """(G~>, G~<, S~) host (nk, n, n) Ry from ``window_sigma_c``'s ``branch_pair`` (Eq. 8)."""
    from tdagw import collision as C
    g_out, g_in = (np.asarray(g) for g in C.rates(jnp.asarray(tabs["cond"]),
                                                   jnp.asarray(tabs["val"])))
    s = 0.5 * (tabs["cond"][0] + tabs["cond"][1] + tabs["val"][0] + tabs["val"][1]
               ) + tabs["static"]
    return g_out, g_in, s


def _shift_window_part(base: dict, new: dict, old: dict) -> dict:
    """base + (new - old) on every table of ``window_sigma_c`` outputs (frame and census: base/new).

    The truncated frozen part of a refreshed-screening refresh: Sigma^F[W0] held from t0
    plus the change of the kept frozen bands' part, Sigma^F_kept[W(t)] - Sigma^F_kept[W0].
    """
    for part in (new, old):
        if float(part["efermi_ry"]) != float(base["efermi_ry"]):
            raise ValueError("GATE dynsig_collision_parts: the parts' omega frames differ")
    out = dict(base)
    for key in ("pair", "diag", "onshell"):
        out[key] = base[key] + (new[key] - old[key])
    out["branch_pair"] = {n: base["branch_pair"][n] + (new["branch_pair"][n] - old["branch_pair"][n])
                          for n in base["branch_pair"]}
    out["sweep"] = new.get("sweep")
    return out


#: ``window_collision``'s refresh sweeps (``[lorrax] collision_refresh_sweep``).
REFRESH_SWEEPS = ("full", "active")


def _sweep_nodes(out: dict) -> dict:
    """The tau-node census of one ``window_sigma_c`` output (its ``sweep``)."""
    s = out.get("sweep") or {}
    return dict(distinct_tau=int(s.get("distinct_tau", 0)),
                window_tau_pairs=int(s.get("window_tau_pairs", 0)),
                n_windows=int(s.get("n_windows", 0)), rebuilds=int(s.get("rebuilds", 0)),
                windows=list(s.get("windows", ())))


def window_collision(source: dict, ops, *, mesh: Mesh, k_frac, b_lo: int, ntr: int,
                     form: str = "subtracted", hermitian: str = "dynamic", policy=None,
                     sweep: str = "full", check_full: bool = False, screening: str = "frozen",
                     frozen_bands=None, frozen_plan: str = "window",
                     check_truncation: bool = False, odd_same_side: bool = True,
                     screening_gamma_ev: float = 0.0, rebase: bool | None = None,
                     spm_support_sites: str = "", spm_scratch: str | None = None,
                     say=print, record=None):
    """The held collision term at t0 and its refresh hook (§1.5, §7.1), GN-PPM or shared pole.

    Parameters
    ----------
    source : :func:`gw_window_source` (a GN-PPM or a scalar shared-pole GW run).
    ops : the propagator's ``tdagw.eom.TdOperators`` of the window [b_lo, b_lo + nb)
        (its ``rho_eq``, ``w_k`` and k padding); ``k_frac`` (nk, 3) its k list.
    b_lo : the window's first band, 0-based on the GW run's band axis.
    ntr : the propagator's trials (each gets its own rates).
    form : ``tdagw.collision.FORMS``; ``hermitian`` : ``tdagw.collision.HERMITIAN_MODES``.
    policy : ``tdagw.collision.RefreshPolicy`` (None: the rates of t0 throughout).
    sweep : :data:`REFRESH_SWEEPS`.  ``"full"``: one window sweep over every
        intermediate band per refresh, planned on the GW run's on-shell list (the
        incumbent).  ``"active"`` (td_agw_dynamical.md §1.6): with W frozen and
        dress none every branch is linear in the intermediate weights and the
        frozen bands keep their populations, so their part (``window_sigma_c(
        internal="frozen")``) is evaluated once at t0 and a refresh sweeps the
        window's own intermediate bands (``internal="active"``); each part on its
        own held quadrature session, planned on the window's on-shell energies.
    check_full : ``sweep = "active"`` only (diagnostic): also run the full sweep at
        t0 and at every refresh and record the differences of the rates and of
        Delta Sigma^h (``full_vs_split`` in each record).
    screening : :data:`COLLISION_SCREENINGS`.  ``"frozen"``: the run's poles (or shared-pole
        model) throughout (the incumbent).  ``"shared_pole"`` (td_agw_dynamical.md §1.4,
        §7.1; a shared-pole source): each refresh re-runs the GW run's W stage at the
        window populations (:func:`spm_screening`, :func:`spm_refit`: response bank,
        moments, recipe and constructor, a full refit) and sweeps the window Sigma on
        the new model; as with ``"refreshed"`` the frozen part is swept per refresh,
        the head stays the run's and the static limit is the equilibrium model's
        own W^c(0) (``gw.mpa.sigma.shared_pole_static_wc``).  ``"refreshed"``
        (GN-PPM source; td_agw_dynamical.md §7.1): each refresh
        re-screens W at both GN supports from the window populations
        (:func:`collision_screening`, :func:`screen_populations`: dchi0, the Dyson
        solves and the run's pole fit) and sweeps the window Sigma on the new poles;
        the q -> 0 head stays the run's.  The frozen bands' part then changes too and
        is swept at every refresh (``sweep = "active"``) on the held session of t0;
        the static-W limit subtracted from Delta Sigma^h stays the run's W0 (K drho
        holds -W^c_0(0) df), so the static and dynamic parts of dW are both kept.
    frozen_bands : ``(n_below, n_above)`` or None, refreshed + active only: the frozen
        part's change is swept over the frozen bands [b_lo - n_below, b_lo) and
        [b_hi, b_hi + n_above) alone, Sigma^F[W(t)] ~ Sigma^F[W0] + Sigma^F_kept[W(t)]
        - Sigma^F_kept[W0] (exact at rho_eq; its error is measured, ``check_truncation``).
    frozen_plan : ``"window"`` (the window's on-shell energies, a held session) or
        ``"gw_list"`` (the GW run's on-shell list, one-shot rules): the frozen part's
        windows (``sweep = "active"``).
    check_truncation : with ``frozen_bands`` (diagnostic): also sweep the whole frozen
        part at each refresh and record the differences (``truncation`` in each record).
    odd_same_side : :func:`delta_chi0`'s (False: the time-reversal-even same-side
        coefficient at i omega_p, a comparison).
    screening_gamma_ev : the static support's same-side principal-value width
        (:func:`collision_screening`), eV; 0 refuses degenerate pairs.
    rebase : refreshed or shared-pole screening only: the subtracted form's C[rho_eq] at
        the rates of rho_eq on the refresh's W(t) (one more window sweep at f_eq) instead
        of the t0 rates, so rho_eq stays a fixed point of each refresh's dissipator and the
        regularization tails of C[rho_eq] cancel at every W.  None: on for
        ``"shared_pole"`` with ``form = "subtracted"``, off otherwise (the GN-PPM
        member's measured option; ``"raw"`` subtracts nothing).
    spm_support_sites : ``"shared_pole"`` only: :func:`spm_screening`'s support set (""
        the run's production ladder, re-resolved at every state).
    spm_scratch : ``"shared_pole"`` only: the refits' model directory (required).
    say : the stdout line of each refresh; ``record`` (or None) gets each
        refresh's dict (the run log's section, the case scripts' tables).

    The instantaneous basis is the band basis (``dress = "none"``: U = 1, e =
    eps^G of the run, f = diag rho of the window), the coherences are not in the
    rates.  Every rate and table is per trial and per k on the propagator's k
    rows; the window Sigma runs one tau sweep per trial with both branches kept
    (``gw.ppm_sigma.window_sigma_c(branches=True)``).  Returns dict: ``diss`` the
    t0 :class:`tdagw.collision.Dissipator` (k-sharded), ``held`` the t0 held
    Delta Sigma^h (zeros; None when ``hermitian == "none"``), ``refresh`` the hook of
    ``tdagw.propagate.run_propagation``, ``t0`` the equilibrium receipt and
    ``state`` (the hook's history), ``eq`` the t0 sweep (:func:`window_branch_rates`
    and ``window_sigma_c``'s raw output) and ``bundle_of`` (TD k row -> GW row).
    The basis is the band basis (U = 1), so the window outputs' instantaneous
    basis and the propagator's coincide.
    """
    import time
    from common.collectives import gather_to_host
    from common.units import RYD_TO_EV
    from gw.ppm_sigma import add_window_parts, window_sigma_c, window_sigma_plan
    from tdagw import collision as C
    from tdagw import propagate as tp
    from tdagw.self_energy import held_term
    if form not in C.FORMS:
        _refuse("dynsig_collision_form", repr(form), " | ".join(C.FORMS), "pick one")
    if hermitian not in C.HERMITIAN_MODES:
        _refuse("dynsig_collision_hermitian", repr(hermitian), " | ".join(C.HERMITIAN_MODES),
                "pick one")
    if sweep not in REFRESH_SWEEPS:
        _refuse("dynsig_collision_sweep", repr(sweep), " | ".join(REFRESH_SWEEPS), "pick one")
    if check_full and sweep != "active":
        _refuse("dynsig_collision_check", "check_full with sweep = 'full'", "sweep = 'active'",
                "the check compares the split refresh with the full one")
    if screening not in COLLISION_SCREENINGS:
        _refuse("dynsig_collision_screening", repr(screening), " | ".join(COLLISION_SCREENINGS),
                "pick one")
    if frozen_plan not in ("window", "gw_list"):
        _refuse("dynsig_collision_frozen_plan", repr(frozen_plan), "'window' | 'gw_list'",
                "pick one")
    refreshed = screening in ("refreshed", "shared_pole")
    if frozen_bands is not None and (not refreshed or sweep != "active"):
        _refuse("dynsig_collision_frozen_bands", f"frozen_bands with screening = {screening!r}, "
                f"sweep = {sweep!r}", "screening = 'refreshed' or 'shared_pole', sweep = 'active'",
                "the truncation is of the frozen part's change under W(t)")
    rebase = (screening == "shared_pole" and form != "raw") if rebase is None else bool(rebase)
    if rebase and not refreshed:
        _refuse("dynsig_collision_rebase", "rebase with screening = 'frozen'",
                "screening = 'refreshed' or 'shared_pole'", "the t0 rates are rho_eq's at W0")
    member = source.get("member", "gn_ppm")
    if ((screening == "refreshed" and member != "gn_ppm")
            or (screening == "shared_pole" and member != "shared_pole")):
        _refuse("dynsig_collision_member", f"screening = {screening!r} on a {member} source",
                "refreshed on a GN-PPM source, shared_pole on a shared-pole source",
                "point collision_source at the matching GW run")
    if screening == "shared_pole" and not spm_scratch:
        _refuse("dynsig_collision_spm_scratch", "no refit directory", "spm_scratch",
                "pass the refits' model directory (outdir/collision_refits)")
    if check_truncation and frozen_bands is None:
        _refuse("dynsig_collision_check", "check_truncation without frozen_bands",
                "frozen_bands", "the check compares the truncated frozen part with the whole")
    policy = C.RefreshPolicy() if policy is None else policy
    wfns, meta, poles, splan = source["wfns"], source["meta"], source["poles"], source["plan"]
    nk_pad, nb = int(ops.e_qp.shape[0]), int(ops.e_qp.shape[1])
    kplan = wfns.green_parent.plan
    kb_frac = np.asarray(kplan.k_parent_frac, np.float64)
    irr = np.asarray(getattr(kplan, "irr_idx", np.arange(kb_frac.shape[0])))
    if (kb_frac.shape[0] != np.asarray(k_frac).shape[0]
            or not np.array_equal(irr, np.arange(irr.size))):
        _refuse("dynsig_collision_kplan", f"GW carrier of {kb_frac.shape[0]} k (irreducible "
                f"map identity: {np.array_equal(irr, np.arange(irr.size))}) for "
                f"{np.asarray(k_frac).shape[0]} TD k", "the full k grid of the TD system, "
                "no symmetry reduction", "run the GN-PPM bundle without symmetry")
    kgrid = tuple(int(n) for n in meta.kgrid)
    from tdagw.model import flat_k_rows
    rows_td, rows_b = flat_k_rows(np.asarray(k_frac), kgrid), flat_k_rows(kb_frac, kgrid)
    nk = int(rows_td.size)
    bundle_of = np.argsort(rows_b)[rows_td]             # TD k row i <- bundle row bundle_of[i]
    win = slice(int(b_lo), int(b_lo) + nb)
    e_g = np.asarray(jax.device_get(wfns.enk), np.float64)[:, win]          # Ry, bundle rows
    f_b = np.asarray(jax.device_get(wfns.occ), np.float64)[:, win]
    f_td = np.real(np.einsum("kii->ki", np.asarray(gather_to_host(ops.rho_eq))[:nk]))
    if not np.array_equal(f_b[bundle_of], f_td):
        _refuse("dynsig_collision_occupations", "the GW run's window occupations differ from "
                "the TD system's", "one equilibrium", "build the TD system from the same bundle "
                "and window")
    eye = np.broadcast_to(np.eye(nb, dtype=complex), (kb_frac.shape[0], nb, nb)).copy()
    dynamic = hermitian == "dynamic"
    half = 0.5                                                              # Ry -> Ha

    def to_td(a):                         # bundle rows (nk, ...) -> TD rows
        return np.asarray(a)[bundle_of]

    def window(f_bundle, plan, internal=None, *, at=None, keep=None, wc0=None, static=None):
        """One window Sigma (both branches) at window populations f (bundle rows), poles ``at``."""
        t = time.perf_counter()
        st = dynamic if static is None else bool(static)
        out = window_sigma_c(wfns, poles if at is None else at,
                             dict(b_lo=int(b_lo), U=eye, e=e_g, f=f_bundle), plan, meta=meta,
                             mesh_xy=mesh, branches=True, static_limit=st,
                             internal=internal, keep_frozen=keep,
                             static_limit_wc0=(W0c if (spm and st and wc0 is None) else wc0))
        out["wall_s"] = time.perf_counter() - t
        return out

    spm = member == "shared_pole"
    scr, screen_eq = None, None
    if screening == "shared_pole":
        t_s = time.perf_counter()
        scr = spm_screening(source, mesh=mesh, b_lo=int(b_lo), n_active=nb,
                            support_sites=spm_support_sites, scratch=spm_scratch)
        t_b0 = time.perf_counter()
        poles_eq, screen_eq = spm_refit(scr, f_b, label="eq")
        if scr["variant"] == "override":
            poles = poles_eq                     # V2: the equilibrium model of the support set
        screen_eq.update(setup_s=t_b0 - t_s, variant=scr["variant"],
                         support_sites=scr["support_sites"],
                         model_vs_run=_model_payload_cmp(source["model"]["path"],
                                                         poles_eq["model"]["path"]))
    if spm:                                             # the model's own W^c(0) (owner ruling)
        from gw.mpa.sigma import shared_pole_static_wc
        from gw.ppm_sigma import fresh_shared_pole_capacity
        W0c = shared_pole_static_wc(poles["model"], fresh_shared_pole_capacity(poles["meta"], mesh),
                                    mesh_xy=mesh, rows=np.arange(int(meta.nk_tot)))
    else:
        W0c = poles.Wc0_q                               # the run's W^c(0): the static limit's
    keep = None
    if frozen_bands is not None:
        n_lo, n_hi = (int(v) for v in frozen_bands)
        nfull = int(np.asarray(jax.device_get(wfns.occ))[:, wfns.slices.full].shape[1])
        keep = np.zeros(nfull, bool)
        keep[max(0, int(b_lo) - n_lo):int(b_lo)] = True
        keep[int(b_lo) + nb:min(nfull, int(b_lo) + nb + n_hi)] = True
    if screening == "refreshed":
        t_s = time.perf_counter()
        from common.units import RYD_TO_EV as _ev
        scr = collision_screening(source, mesh=mesh, b_lo=int(b_lo), n_active=nb,
                                  odd_same_side=odd_same_side,
                                  gamma_ry=float(screening_gamma_ev) / _ev)
        t_b0 = time.perf_counter()
        poles_eq, _, screen_eq = screen_populations(scr, f_b)
        same = lambda a, b: (a is None and b is None) or (
            a is not None and b is not None and bool(jax.device_get(jnp.array_equal(a, b))))
        screen_eq.update(
            setup_s=t_b0 - t_s, poles_bitwise=all(same(getattr(poles_eq, k), getattr(poles, k))
                                                 for k in ("Omega_q", "B_q", "B_odd_q",
                                                           "valid_mask_q", "Wc0_q")),
            rule_nodes=[len(scr["rules"]["static"]["tau"]), len(scr["rules"]["probe"]["tau"])],
            frozen_bands=None if keep is None else int(keep.sum()), frozen_plan=frozen_plan,
            gamma_ev=float(screening_gamma_ev))
        del poles_eq

    held_parts, plans = {}, {}
    t_split = time.perf_counter()
    if sweep == "active":
        # A held session's first call is served by one-shot rules (gw.sigma_box_plan), so each
        # part's session is planned by a call before t0: every value the run uses then comes
        # from held rules, and a refresh at rho_eq reproduces t0.  The active part is planned
        # with every active state live in both branches (f = 1/2), the superset of the live
        # sets a run reaches; the frozen part's states never change.
        cfg = source["config"]
        if frozen_plan == "gw_list":
            req = source["request"]
            fplan = window_sigma_plan(cfg, head=None, frequencies_abs_ry=(
                None if req is None else np.asarray(req.frequencies_ry) + float(source["efermi_ry"])))
        else:
            fplan = window_sigma_plan(cfg, head=None, session={}, window_only=True)
        plans = dict(frozen=fplan, active=window_sigma_plan(cfg, head=source["head"], session={},
                                                            window_only=True))
        if frozen_plan == "window":
            window(f_b, plans["frozen"], "frozen")
        held_parts["frozen"] = window(f_b, plans["frozen"], "frozen")
        window(np.full_like(f_b, 0.5), plans["active"], "active")
        if keep is not None:
            plans["frozen_keep"] = window_sigma_plan(cfg, head=None, session={}, window_only=True)
            window(f_b, plans["frozen_keep"], "frozen", keep=keep, static=False)
            held_parts["frozen_keep"] = window(f_b, plans["frozen_keep"], "frozen", keep=keep,
                                               static=False)
    t_split = time.perf_counter() - t_split

    def tables(out):
        """The rates, S~ and the static-W limit of one window output, on the TD k rows."""
        g_out, g_in, s = window_branch_rates(out["branch_pair"])
        static = None
        if dynamic:
            idx = np.arange(nb)
            static = out["static_limit"]["sx"].copy()
            static[:, idx, idx] += out["static_limit"]["head"]
        return dict(g_out=to_td(g_out), g_in=to_td(g_in), s=to_td(s),
                    static=None if static is None else to_td(static), raw=out,
                    wall_s=out["wall_s"])

    def full_sweep(f_bundle, at=None):
        out = window(f_bundle, splan, at=at, wc0=None if at is None else W0c)
        return dict(tables(out), nodes=_sweep_nodes(out))

    scratch = {}

    def frozen_part(f_bundle, at):
        """The frozen bands' part on the poles ``at`` (static limit: the held one at W0)."""
        if keep is None:
            fz = window(f_bundle, plans["frozen"], "frozen", at=at, static=False)
            out = dict(fz)
        else:
            fz = window(f_bundle, plans["frozen_keep"], "frozen", at=at, keep=keep, static=False)
            out = _shift_window_part(held_parts["frozen"], fz, held_parts["frozen_keep"])
        if "static_limit" in held_parts["frozen"]:
            out["static_limit"] = held_parts["frozen"]["static_limit"]
        return out, fz

    def sweep_tables(f_bundle, at=None):
        """The refresh's window Sigma at window populations f (bundle rows), poles ``at``."""
        if sweep == "full":
            return full_sweep(f_bundle, at=at)
        extra = {}
        frozen = held_parts["frozen"]
        if at is not None:
            frozen, fz = frozen_part(f_bundle, at)
            extra = dict(frozen_s=fz["wall_s"], frozen_nodes=_sweep_nodes(fz))
        act = window(f_bundle, plans["active"], "active", at=at,
                     wc0=None if at is None else W0c)
        scratch["act"] = act
        out = add_window_parts(frozen, act)
        out["wall_s"] = act["wall_s"] + extra.get("frozen_s", 0.0)
        return dict(tables(out), nodes=_sweep_nodes(act), active_s=act["wall_s"], **extra)

    put = lambda a: tp.place_window_blocks(a, mesh, nk_pad)
    stack = lambda key, parts: np.stack([p[key] for p in parts]) * half
    program = tp.collision_program(mesh, raw=(form == "raw"))
    pops = jax.jit(lambda r: jnp.real(jnp.diagonal(r, axis1=-2, axis2=-1)))

    def populations(rho):
        return np.asarray(gather_to_host(pops(rho)))[:, :nk]               # (ntr, nk, nb)

    ry_mev = RYD_TO_EV * 1e3

    def compare(a, b, h_a=None, h_b=None):
        """Split vs full tables at one state: rates (G~>, G~<), S~ and Delta Sigma^h, meV."""
        mx = lambda x: float(np.max(np.abs(x)))
        scale = max(mx(b["g_out"]), mx(b["g_in"]), 1e-300)
        d = dict(rate_out_mev=mx(a["g_out"] - b["g_out"]) * ry_mev,
                 rate_in_mev=mx(a["g_in"] - b["g_in"]) * ry_mev,
                 rate_rel=max(mx(a["g_out"] - b["g_out"]), mx(a["g_in"] - b["g_in"])) / scale,
                 s_mev=mx(a["s"] - b["s"]) * ry_mev, full_tau=b["nodes"]["distinct_tau"],
                 full_s=b["wall_s"])
        if h_a is not None:
            d["herm_mev"] = mx(np.asarray(h_a) - np.asarray(h_b)) * 2.0 * ry_mev   # Ha -> meV
        return d

    # ---- t0: the equilibrium rates, C[rho_eq], the first dissipator
    t_a = time.perf_counter()
    eq = sweep_tables(f_b)
    t_b = time.perf_counter()
    eq_full = full_sweep(f_b) if check_full else None
    g_out_eq, g_in_eq = put(stack("g_out", [eq] * ntr)), put(stack("g_in", [eq] * ntr))
    rho0 = put(np.broadcast_to(np.asarray(gather_to_host(ops.rho_eq))[:nk], (ntr, nk, nb, nb)))
    res = program(g_out_eq, g_in_eq, rho0, ops.rho_eq, g_out_eq, g_in_eq, ops.w_k)
    c_eq = res["c_eq"]
    jax.block_until_ready(res["diss"])
    t_c = time.perf_counter()
    geq_h = np.stack([eq["g_out"], eq["g_in"]])                            # Ry
    d_out, d_in = (np.real(np.einsum("kii->ki", g)) for g in geq_h)
    receipt = dict(
        t_fs=0.0, why="t0", wall_s=t_c - t_a, sweep_s=t_b - t_a, build_s=t_c - t_b,
        min_rate_mev=float(res["min_rate"]) * 2.0 * ry_mev,
        rate_out_diag_mev=[float(d_out.min() * ry_mev), float(d_out.max() * ry_mev)],
        rate_in_diag_mev=[float(d_in.min() * ry_mev), float(d_in.max() * ry_mev)],
        min_eig_out_mev=float(np.linalg.eigvalsh(0.5 * (eq["g_out"] + np.conj(
            np.swapaxes(eq["g_out"], -1, -2)))).min() * ry_mev),
        min_eig_in_mev=float(np.linalg.eigvalsh(0.5 * (eq["g_in"] + np.conj(
            np.swapaxes(eq["g_in"], -1, -2)))).min() * ry_mev),
        c_eq_diag=float(res["c_eq_diag"]), c_eq_offdiag=float(res["c_eq_offdiag"]),
        trace=[float(v) for v in np.asarray(res["trace"])], form=form, hermitian=hermitian,
        sweep=sweep, nodes=eq["nodes"],
        frozen=(None if not held_parts else dict(
            nodes=_sweep_nodes(held_parts["frozen"]), wall_s=held_parts["frozen"]["wall_s"],
            setup_s=t_split)), screening=screening, screen_eq=screen_eq)
    if eq_full is not None:
        receipt["full_vs_split"] = compare(eq, eq_full)
    s_eq = np.stack([eq["s"]] * ntr)
    static_eq = None if not dynamic else np.stack([eq["static"]] * ntr)
    if eq_full is not None:
        s_eq_full = np.stack([eq_full["s"]] * ntr)
        static_eq_full = None if not dynamic else np.stack([eq_full["static"]] * ntr)
    held0 = None if hermitian == "none" else put(np.zeros((ntr, nk, nb, nb), complex))
    state = dict(t_last=0.0, f_last=np.broadcast_to(f_td, (ntr, nk, nb)).copy(), n=0,
                 trace_last=np.asarray(res["trace"]).copy(), history=[receipt],
                 held_max_mev=0.0, valid_prev=[None] * ntr)
    line = (f"  collision t0     : {form}, hermitian {hermitian}; rates at rho_eq: diag "
            f"out [{receipt['rate_out_diag_mev'][0]:.3g}, {receipt['rate_out_diag_mev'][1]:.3g}] meV, "
            f"in [{receipt['rate_in_diag_mev'][0]:.3g}, {receipt['rate_in_diag_mev'][1]:.3g}] meV, "
            f"min eig G~ {receipt['min_rate_mev']:.3g} meV (G~> {receipt['min_eig_out_mev']:.3g}, "
            f"G~< {receipt['min_eig_in_mev']:.3g}); ||C[rho_eq]|| diag {receipt['c_eq_diag']:.3e} "
            f"off-diag {receipt['c_eq_offdiag']:.3e} Ha/hbar; wall {receipt['wall_s']:.2f} s "
            f"(sweep {receipt['sweep_s']:.2f}, build {receipt['build_s']:.2f}); sweep {sweep}: "
            f"{eq['nodes']['distinct_tau']} tau in {eq['nodes']['n_windows']} windows"
            + ("" if not held_parts else
               f" (frozen part {'at t0' if not refreshed else 'per refresh'}: "
               f"{receipt['frozen']['nodes']['distinct_tau']} tau, "
               f"{receipt['frozen']['wall_s']:.2f} s)")
            + ("" if screen_eq is None or screening != "shared_pole" else spm_t0_line(screen_eq))
            + ("" if screen_eq is None or screening == "shared_pole" else
               f"; screening refreshed: at rho_eq max|dchi0| {screen_eq['dchi_max'][0]:.1e}/"
               f"{screen_eq['dchi_max'][1]:.1e}, max|W - W0|/|W0| {screen_eq['dW'][0]:.1e}/"
               f"{screen_eq['dW'][1]:.1e} (0 / i w_p), poles "
               f"{'bitwise' if screen_eq['poles_bitwise'] else 'NOT bitwise'} the run's, "
               f"{screen_eq['n_invalid']}/{screen_eq['n_logical']} invalid modes; dchi0 "
               f"{screen_eq['dchi_s']:.2f} s, Dyson+fit {screen_eq['fit_s']:.2f} s; rules "
               f"{screen_eq['rule_nodes'][0]}+{screen_eq['rule_nodes'][1]} nodes"
               + ("" if keep is None else f"; frozen change over {int(keep.sum())} bands")))
    say(line)
    if record is not None:
        record(dict(receipt, line=line))
    fs = 1000.0 / 24.188843265857                       # a.u. of time per fs

    def refresh(t, rho, ops_now, params):
        f = populations(rho)
        due, df, why = C.refresh_due(policy, t, state["t_last"], f, state["f_last"])
        if not due:
            return None
        t1 = time.perf_counter()
        parts, f_rows, screens, poles_now = [], [], [], []
        for tr in range(ntr):
            fb = np.empty((kb_frac.shape[0], nb))
            fb[bundle_of] = f[tr]
            f_rows.append(fb)
            at = None
            if screening == "shared_pole":       # two slots per trial: this refresh's and the last
                at, srec = spm_refit(scr, fb, label=f"s{(state['n'] + 1) % 2}t{tr}")
                screens.append(srec)
                poles_now.append(at)
            elif refreshed:
                at, _, srec = screen_populations(scr, fb, valid_prev=state["valid_prev"][tr])
                state["valid_prev"][tr] = at.valid_mask_q
                screens.append(srec)
                poles_now.append(at)
            parts.append(sweep_tables(fb, at=at))
            if check_truncation:
                fz = window(fb, plans["frozen"], "frozen", at=at, static=False)
                whole = dict(fz)
                if "static_limit" in held_parts["frozen"]:
                    whole["static_limit"] = held_parts["frozen"]["static_limit"]
                alt = dict(tables(add_window_parts(whole, scratch["act"])), nodes=_sweep_nodes(fz),
                           wall_s=fz["wall_s"])
                parts[-1]["_truncation"] = alt
        t2 = time.perf_counter()
        g_out, g_in = put(stack("g_out", parts)), put(stack("g_in", parts))
        g_eq_now, rebase_s = (g_out_eq, g_in_eq), 0.0
        if rebase:
            tr0 = time.perf_counter()
            eqs = [sweep_tables(f_b, at=p_) for p_ in poles_now]
            g_eq_now = (put(stack("g_out", eqs)), put(stack("g_in", eqs)))
            rebase_s = time.perf_counter() - tr0
        t3 = time.perf_counter()
        out = program(g_out, g_in, rho, ops_now.rho_eq, *g_eq_now, ops_now.w_k)
        new_ops = ops_now._replace(collision=out["diss"])
        jax.block_until_ready(out["diss"])
        t4 = time.perf_counter()
        new_params, delta, held_max, h = params, None, 0.0, None
        if hermitian != "none":
            s_t = np.stack([p["s"] for p in parts])
            st_t = None if not dynamic else np.stack([p["static"] for p in parts])
            h = np.asarray(C.hermitian_update(s_t, s_eq, st_t, static_eq)) * half   # Ha
            held_max = float(np.max(np.abs(h))) * 2.0 * ry_mev
            term, set_term = held_term(params)
            new = put(h)
            delta = new - term
            new_params = set_term(new)
        t5 = time.perf_counter()
        trace = np.asarray(out["trace"])
        state["n"] += 1
        rec = dict(n=state["n"], t_fs=float(t) / fs, why=why, df=df,
                   wall_s=t5 - t1, sweep_s=t2 - t1, rates_s=t3 - t2, build_s=t4 - t3,
                   hermitian_s=t5 - t4, out_per_fs=[float(v) * fs for v in np.asarray(out["out"])],
                   in_per_fs=[float(v) * fs for v in np.asarray(out["inn"])],
                   d_trace=[float(v) for v in (trace - state["trace_last"])],
                   min_rate_mev=float(out["min_rate"]) * 2.0 * ry_mev, held_max_mev=held_max,
                   c_eq_diag=receipt["c_eq_diag"], c_eq_offdiag=receipt["c_eq_offdiag"],
                   sweep=sweep, nodes=parts[0]["nodes"], screening=screening,
                   rebase_s=rebase_s, c_eq_now=(None if not rebase else
                                                [float(out["c_eq_diag"]), float(out["c_eq_offdiag"])]))
        if screening == "shared_pole":
            rec.update(screen=spm_record(screens, screen_eq))
        elif refreshed:
            worst = lambda key: max(sr[key] for sr in screens)
            rec.update(screen=dict(
                dchi_s=sum(sr["dchi_s"] for sr in screens), fit_s=sum(sr["fit_s"] for sr in screens),
                n_invalid=worst("n_invalid"), n_logical=screens[0]["n_logical"],
                flips_prev=worst("flips_prev"), flips_eq=worst("flips_eq"),
                d_omega_mev=worst("d_omega_mev"),
                dW=[max(sr["dW"][0] for sr in screens), max(sr["dW"][1] for sr in screens)],
                dchi_max=[max(sr["dchi_max"][0] for sr in screens),
                          max(sr["dchi_max"][1] for sr in screens)],
                degenerate_pv=max(sr["degenerate_pv"] for sr in screens)))
        if refreshed and sweep == "active":
            rec.update(frozen_s=sum(p_["frozen_s"] for p_ in parts),
                       active_s=sum(p_["active_s"] for p_ in parts),
                       frozen_nodes=parts[0]["frozen_nodes"])
        if check_truncation:                 # diagnostic, outside the comparison's own wall
            h_alt = None
            if hermitian != "none":
                h_alt = np.asarray(C.hermitian_update(
                    np.stack([p_["_truncation"]["s"] for p_ in parts]), s_eq,
                    None if not dynamic else np.stack([p_["_truncation"]["static"] for p_ in parts]),
                    static_eq)) * half
            cmp = [compare(p_, p_["_truncation"], None if h is None else h[i],
                           None if h_alt is None else h_alt[i]) for i, p_ in enumerate(parts)]
            rec["truncation"] = {k: max(c_[k] for c_ in cmp) for k in cmp[0]}
        if check_full:                       # diagnostic, outside the refresh's wall
            fulls = [full_sweep(fb, at=(poles_now[i] if refreshed else None))
                     for i, fb in enumerate(f_rows)]
            h_full = None
            if hermitian != "none":
                h_full = np.asarray(C.hermitian_update(
                    np.stack([p["s"] for p in fulls]), s_eq_full,
                    None if not dynamic else np.stack([p["static"] for p in fulls]),
                    static_eq_full)) * half
            cmp = [compare(p, q, None if h is None else h[i], None if h_full is None else h_full[i])
                   for i, (p, q) in enumerate(zip(parts, fulls))]
            rec["full_vs_split"] = {k: max(c[k] for c in cmp) for k in cmp[0]}
        state.update(t_last=float(t), f_last=f, trace_last=trace)
        state["history"].append(rec)
        line = (f"  collision refresh {rec['n']:3d} at {rec['t_fs']:9.4f} fs ({why}): wall "
                f"{rec['wall_s']:.2f} s (sweep {rec['sweep_s']:.2f}, rates {rec['rates_s']:.3f}, "
                f"dissipator {rec['build_s']:.3f}, hermitian {rec['hermitian_s']:.3f}), max|df| "
                f"{df:.3e}, dTr rho {max(rec['d_trace'], key=abs):+.3e} since last; d(Tr rho)/dt "
                f"out {min(rec['out_per_fs']):+.3e} in {max(rec['in_per_fs']):+.3e} per fs; min "
                f"eig G~ {rec['min_rate_mev']:.3g} meV; max|dSigma^h| {held_max:.3g} meV; "
                f"||C[rho_eq]|| diag {rec['c_eq_diag']:.2e} off {rec['c_eq_offdiag']:.2e}; "
                f"{sweep} sweep {rec['nodes']['distinct_tau']} tau"
                + (f" ({rec['nodes']['rebuilds']} rules rebuilt)" if rec["nodes"]["rebuilds"] else "")
                + ("" if "screen" not in rec or screening != "shared_pole" else
                   spm_line(rec))
                + ("" if "screen" not in rec or screening == "shared_pole" else
                   f"; W(t): dchi0 {rec['screen']['dchi_s']:.2f} s, Dyson+fit "
                   f"{rec['screen']['fit_s']:.2f} s"
                   + ("" if "frozen_s" not in rec else
                      f", frozen sweep {rec['frozen_s']:.2f} s ({rec['frozen_nodes']['distinct_tau']} "
                      f"tau" + (f", {rec['frozen_nodes']['rebuilds']} rules rebuilt"
                                if rec['frozen_nodes']['rebuilds'] else "")
                      + f"), active {rec['active_s']:.2f} s")
                   + f"; invalid {rec['screen']['n_invalid']}/{rec['screen']['n_logical']} modes "
                   f"({rec['screen']['flips_prev']} flips since last, {rec['screen']['flips_eq']} vs "
                   f"rho_eq), max|dOmega| {rec['screen']['d_omega_mev']:.3g} meV, max|dW|/|W0| "
                   f"{rec['screen']['dW'][0]:.2e} (0) {rec['screen']['dW'][1]:.2e} (i w_p)"
                   + (f"; {rec['screen']['degenerate_pv']} degenerate same-side pairs at "
                      f"their principal value" if rec['screen']['degenerate_pv'] else ""))
                + ("" if "truncation" not in rec else
                   f"; vs whole frozen part: rates {rec['truncation']['rate_rel']:.2e} rel, "
                   f"dSigma^h {rec['truncation'].get('herm_mev', 0.0):.2e} meV")
                + ("" if "full_vs_split" not in rec else
                   f"; vs full: rates {rec['full_vs_split']['rate_rel']:.2e} rel, "
                   f"dSigma^h {rec['full_vs_split'].get('herm_mev', 0.0):.2e} meV"))
        say(line)
        if record is not None:
            record(dict(rec, line=line))
        return dict(ops=new_ops, params=new_params, delta=delta)

    return dict(diss=res["diss"], held=held0, refresh=refresh, t0=receipt, state=state,
                c_eq=c_eq, rates_eq=(g_out_eq, g_in_eq), sweep=sweep_tables, program=program,
                eq=eq, bundle_of=bundle_of, parts=held_parts, plans=plans, screening=scr,
                window=window, tables=tables, keep=keep, rebase=rebase, W0c=W0c, poles=poles)

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
* The active window's G Hamiltonian H^G_A(t) = diag(eps^G) [+ K drho_car]
  is diagonalized per k, H^G = U diag(e) U^H (``dress``: ``"none"`` keeps U = 1
  and e = eps^G; ``"instantaneous"`` adds the frozen-W update of the carriers,
  drho_car = U diag(f - f_eq) U^H, a fixed point per call (:func:`_population_basis`):
  the coherences, the field and the rotation of the equilibrium occupations stay
  out, so U and e move at O(E^2) and linear response is the BSE.  Red twins:
  ``"coherent"`` adds K drho of the whole drho, whose coherent part moves them at
  O(E); ``"sea_rotation"`` dresses with U diag(f) U^H - rho_eq, whose rotation of
  the equilibrium occupations feeds back through K).
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

Prototype scope: static screening only (the GN-PPM / tangential members, the
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
           "window_branch_operators", "collision_screening", "screen_populations",
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


#: ``dress = "instantaneous"`` (docs/theory/td_agw_dynamical.md §1.1): the fixed point's static
#: number of plain steps, and the residual (population units) above which a correction is
#: refused (GATE dynsig_dress_not_converged).  Measured on CrI3 516 2v2c: round-off in 3, 5
#: and 23 steps at populations 5e-6, 5e-4 and 1.2e-2 (contraction per step ~ p^1.1; the gate
#: holds to populations ~2e-2, extrapolated).  On tests/test_tdagw_dynsig.py's bundle, to 1e-12:
#: 4 steps at 1.6e-3, 17 at 7e-2, 36 at 0.39; 32 steps keep its hot state's dSigma within 2e-9
#: of the converged one.
DRESS_STEPS, DRESS_TOL = 32, 1e-8


def _population_basis(e_win, rho, f_eq, static_dsigma, steps: int = DRESS_STEPS,
                      sea: bool = False):
    """The carrier-dressed window basis, Eq. (2): U, e, f and the fixed point's residual.

    h = diag(e_win) + K drho_car, drho_car = U diag(f - f_eq) U^H, (U, e, f) =
    :func:`instantaneous_basis` (h, rho): ``steps`` plain evaluations of the map
    P -> U[P] diag(f[P] - f_eq) U[P]^H from P = diag(rho) - diag(f_eq) (U = 1, f = diag rho).
    ``sea=True`` is the red twin ``dress = "sea_rotation"``, P -> U diag(f) U^H - rho_eq: its
    rotation of the equilibrium occupations feeds back through K, which on CrI3 516 stalls
    the plain iteration (design page §1.1).  rho (nk, nA, nA), f_eq (nk, nA); ``static_dsigma``
    is K (Ry, model order).  Returns U, e, f of the last evaluation and its residual
    max |P_out - P_in| (population units; 0 at the fixed point).
    """
    h0 = _band_diag(e_win)
    f0 = jnp.real(jnp.diagonal(rho, axis1=-2, axis2=-1))

    def step(c, _):
        P, _ = c
        U, e, f = instantaneous_basis(h0 + static_dsigma(P), rho)
        P_out = (U * (f if sea else f - f_eq)[..., None, :]) @ jnp.conj(jnp.swapaxes(U, -1, -2))
        if sea:
            P_out = P_out - _band_diag(f_eq)
        return (P_out, (U, e, f, jnp.max(jnp.abs(P_out - P)))), None

    out0 = (jnp.broadcast_to(jnp.eye(rho.shape[-1], dtype=rho.dtype), rho.shape), e_win, f0,
            jnp.zeros((), f0.dtype))
    c, _ = lax.scan(step, (_band_diag(f0 - f_eq), out0), None, length=steps)
    return c[1]


def _active_state(dyn, x, *, dress, static_dsigma):
    """U, e, f of the window for drho = x (nk, nA, nA) model order (Ry), and the dressing's
    fixed-point residual (:func:`_population_basis`; 0 for ``"none"`` and the twin)."""
    b_lo, nA = dyn["band_lo"], dyn["n_active"]
    e_win = dyn["e_full"][:, b_lo:b_lo + nA]
    f_eq = dyn["f_full"][:, b_lo:b_lo + nA]
    rho = x + _band_diag(f_eq)
    zero = jnp.zeros((), e_win.dtype)
    if dress == "none":
        U = jnp.broadcast_to(jnp.eye(nA, dtype=x.dtype), x.shape)
        return U, e_win, jnp.real(jnp.diagonal(rho, axis1=-2, axis2=-1)), zero
    if dress == "coherent":                # red twin: the whole drho, e and U move at O(E)
        return (*instantaneous_basis(_band_diag(e_win) + static_dsigma(x), rho), zero)
    if dress not in ("instantaneous", "sea_rotation"):
        _refuse("dynsig_dress", repr(dress), "'none' | 'instantaneous'",
                "'coherent' and 'sea_rotation' exist only as red twins")
    return _population_basis(e_win, rho, f_eq, static_dsigma, sea=(dress == "sea_rotation"))


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
    dsig NaN (:func:`check_state` names it on the host); ``flags['dress_residual']``
    is the dressing's fixed-point residual (0 for ``dress = "none"``).
    """
    U, e_A, f_A, res = _active_state(dyn, x, dress=dress, static_dsigma=static_dsigma)
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
    flags = dict(flags, dress_residual=res,
                 dress_unconverged=jnp.logical_not(res <= DRESS_TOL).astype(jnp.int32))
    bad = flags["crossed"] + flags["out_of_rule"] + flags["degenerate"] + flags["dress_unconverged"]
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
    dress : ``"none"`` | ``"instantaneous"`` (module docstring; ``"coherent"`` and
        ``"sea_rotation"`` are red twins).
    opts : ``pauli``, ``same_side``, ``occupation`` (red twins only).
    """
    from tdagw.model import flat_k_rows
    from tdagw.self_energy import SelfEnergyModel, isdf_model
    if screening not in ("frozen", "static_rpa"):
        _refuse("dynsig_screening", repr(screening), "'frozen' | 'static_rpa'",
                "the GN-PPM / tangential members are design items (theory page)")
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
    if int(np.asarray(flags.get("dress_unconverged", 0))):
        _refuse("dynsig_dress_not_converged", f"population-dressing residual "
                f"{float(np.asarray(flags['dress_residual'])):.3g} after {DRESS_STEPS} steps",
                f"<= {DRESS_TOL:g}", "raise DRESS_STEPS, or dress = 'none' (design page §1.1)")



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
    tangential one-shot keeps its ``model.h5`` but not its W stage's inputs, so the
    GW driver (``gw.gw_jax.main``) runs in this process on a copy of ``gw_deck`` in
    ``workdir`` (``write_restart_tensors = false``: the run's own restart is never
    written; ``overrides`` set deck keys of the copy) and
    ``gw.ppm_pipeline.observe_window_sigma_source`` reads the Sigma stage at its
    return: no GW stage is re-implemented, and the window Sigma at rho_eq is the
    run's (§6.1).  Collective: every process calls it.  Must run outside another
    driver's ``RunSession`` (the GW driver owns the timing table and stdout while
    it runs).  Returns the observer's dict (``member`` "gn_ppm" or "tangential",
    ``wfns``, ``meta``, ``config``, ``head``, ``efermi_ry``, ``request``; GN-PPM:
    ``poles``, ``V_q``, ``quad`` and with ``screening`` ``fit_plan``,
    ``W_supports``; shared pole: ``model``, ``occupation_state``, ``linalg`` and
    with ``screening`` ``screening_inputs``, ``recipe_inputs``) plus ``poles`` (the
    window Sigma's model: the ``PPMBuildResult``, or
    ``gw.ppm_sigma.tangential_window_source`` of the run's model), ``plan``
    (``gw.ppm_sigma.window_sigma_plan`` on the run's own on-shell frequency list),
    ``deck``, ``workdir``, ``rc``, ``wall_s``.
    """
    import time
    import gw.gw_jax as gw_driver
    from jax.experimental import multihost_utils
    from gw.ppm_pipeline import observe_window_sigma_source
    from gw.ppm_sigma import tangential_window_source, window_sigma_plan
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
    if member not in ("gn_ppm", "tangential"):
        _refuse("dynsig_window_source", f"no window Sigma stage ran in {gw_deck}",
                "a compute_mode = gn_ppm deck or a scalar sigma_w_model = tangential deck",
                "point the source at a GN-PPM or tangential GW run")
    mode = str(getattr(src["config"].compute_mode, "value", src["config"].compute_mode))
    if member == "gn_ppm" and mode != "gn_ppm":
        _refuse("dynsig_gnppm_mode", f"compute_mode = {mode}", "gn_ppm (the ordered GN fit the "
                "branch rates are measured on)", "use a GN-PPM run")
    if member == "tangential":
        if screening and ("screening_inputs" not in src or "recipe_inputs" not in src):
            _refuse("dynsig_spm_source", f"no tangential W stage observed in {gw_deck}",
                    "a one-shot tangential run (its W stage runs in this process)",
                    "drop restart from the deck")
        src["poles"] = tangential_window_source(src["model"], src["meta"],
                                                 efermi_ry=src["efermi_ry"], linalg=src["linalg"])
    req = src["request"]
    freqs = None if req is None else np.asarray(req.frequencies_ry) + float(src["efermi_ry"])
    src = dict(src, plan=window_sigma_plan(src["config"], head=src["head"],
                                           frequencies_abs_ry=freqs),
               deck=os.path.abspath(gw_deck), workdir=workdir, rc=rc,
               wall_s=time.perf_counter() - t0)
    what = ("GN-PPM poles at the run's own W_0" if member == "gn_ppm" else
            f"tangential model K {min(src['model']['K'])}-{max(src['model']['K'])}")
    say(f"  collision source: GW driver on {gw_deck} in {workdir} (rc {rc}, "
        f"{src['wall_s']:.1f} s): {what}, {int(src['meta'].nk_tot)} k, N_mu {int(src['meta'].n_rmu)}")
    return src


#: ``window_collision``'s screening (``[lorrax] collision_screening``).
COLLISION_SCREENINGS = ("frozen", "refreshed", "tangential")


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
# The tangential member: the W stage of the GW run re-run at the populations
# ===========================================================================

#: Residue-weighted census cuts (eV): the weight of the poles below each.
CENSUS_CUTS_EV = (0.3, 1.0)
#: The census histogram's pole-energy bin edges (eV), log spaced.
CENSUS_BINS_EV = np.geomspace(1.0e-3, 2.0e2, 64)


def pole_census(path, *, cuts_ev=CENSUS_CUTS_EV, bins_ev=CENSUS_BINS_EV) -> dict:
    """Residue-weighted census of a tangential model file (host, h5py ``'r'``).

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


def chi0_active_group(enk, win: slice, request="auto"):
    """The refits' chi0 active bands A: the smallest isolated band group containing ``request``.

    ``request`` is ``"auto"`` (the window ``win``) or ``(a0, a1)``.  A band group
    [a0, a1) is isolated when the gap below a0 and the gap above a1 - 1 are both
    positive, min_k E(a0) - max_k E(a0 - 1) > 0 and min_k E(a1) - max_k E(a1 - 1) > 0:
    then the frozen-leg blocks FA and AF (``gw.response_bank.response_blocks``) are
    strictly positive.  Each edge moves outward to the first positive gap; the extra
    bands sit at their 0/1 step inside A (the AA block).  ``enk`` [nk, nb] are the
    W carrier's physical bands.  Returns (a0, a1, (gap below, gap above) in Ry; a
    missing side is +inf).  Refuses only when no isolated group exists (A would be
    every band).
    """
    enk = np.asarray(enk, np.float64)
    nb = enk.shape[1]
    lo, hi = ((win.start, win.stop) if isinstance(request, str) else
              (int(request[0]), int(request[1])))
    if isinstance(request, str) and request != "auto":
        _refuse("dynsig_spm_chi0_active", f"chi0_active = {request!r}", "'auto' or (a0, a1)",
                "pick one")
    lo, hi = min(lo, win.start), max(hi, win.stop)
    if not (0 <= lo < hi <= nb):
        _refuse("dynsig_spm_chi0_active", f"active bands [{lo}, {hi})",
                f"a range of the W carrier's {nb} bands", "the window's populations are A's")
    gap = lambda b: float(enk[:, b].min() - enk[:, b - 1].max())   # below band b
    a0 = lo
    while a0 > 0 and not gap(a0) > 0:
        a0 -= 1
    a1 = hi
    while a1 < nb and not gap(a1) > 0:
        a1 += 1
    if a0 == 0 and a1 == nb:
        _refuse("dynsig_spm_chi0_active", f"no isolated band group contains [{lo}, {hi})",
                "a positive gap below and above an active range", "use the one-rule chi0 "
                "(leave collision_spm_chi0_active unset)")
    return a0, a1, (gap(a0) if a0 > 0 else float("inf"), gap(a1) if a1 < nb else float("inf"))


def spm_screening(source: dict, *, mesh: Mesh, b_lo: int, n_active: int,
                  support_sites: str = "", scratch: str, chi0_active=None) -> dict:
    """The held objects of the tangential refit of the collision term (§1.4, §7.1).

    A refit is the GW run's own W stage (``gw.tangential_screening.
    screen_tangential_poles``: response bank -> moments -> constructor -> model)
    re-run at the instantaneous populations, with the recipe re-resolved at the
    instantaneous census (``gw.tangential_recipe.bind_tangential_census`` +
    ``resolve_tangential_recipe``).  Every input is the run's, taken at its seams
    (``gw_window_source(screening=True)``: ``screening_inputs``,
    ``recipe_inputs``): the W carrier (psi of every band at the centroids, eps^G,
    the step occupations), V, the symmetry tables and the run's occupation state
    (its mu).  The q -> 0 head is not rebuilt (the run's head is frozen, as in the
    GN-PPM refresh).  ``support_sites`` "" is the run's production ladder (V1);
    otherwise a ``sigma_w_support_sites_ev`` override (V2, low-energy supports:
    ``"<line eV> | <imaginary eV> [| <line height eV>]"``), resolved at the run's
    census.  Models go to ``scratch``; a refit's directory is removed when its
    label is reused.

    ``chi0_active`` (None: chi0 over all pairs on one rule, the GW run's path):
    ``"auto"`` or ``(a0, a1)``, extended to the smallest isolated band group
    containing it and the window (:func:`chi0_active_group`).  The refits then take chi0 as four
    (hole x electron) blocks FF, FA, AF, AA with F the other bands at their
    0/1 step (``gw.response_bank.response_blocks``): only AA crosses zero,
    every block's interval comes from band energies, and its rule is built at
    the first refit and held for the run (``meta.tangential_response_blocks``).

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
    from gw.tangential_recipe import parse_support_sites
    from common.units import RYD_TO_EV
    if "screening_inputs" not in source or "recipe_inputs" not in source:
        _refuse("dynsig_spm_source", "a source without the run's W-stage inputs",
                "gw_window_source(..., screening=True) on a tangential deck",
                "rebuild the source with screening=True")
    inp = source["screening_inputs"]
    wfns_w, meta_run, config_run, kw = inp["wfns"], inp["meta"], inp["config"], inp["kwargs"]
    state = kw.get("occupation_state")
    f_eq = np.asarray(jax.device_get(wfns_w.occ if state is None else state.f_kn), np.float64)
    mu = (float(meta_run.tangential_census["mu_ry"]) if state is None else float(state.mu_ry))
    win = slice(int(b_lo), int(b_lo) + int(n_active))
    enk_w = np.asarray(jax.device_get(wfns_w.enk), np.float64)
    enk_s = np.asarray(jax.device_get(source["wfns"].enk), np.float64)
    if enk_s.shape[0] != enk_w.shape[0] or not np.array_equal(enk_s[:, win], enk_w[:, win]):
        _refuse("dynsig_spm_carriers", "the Sigma and W carriers' window energies differ",
                "one k order and band axis", "run the tangential source on an identity-only grid")
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
    if chi0_active is not None:
        stop = min(int(getattr(meta_run, "b_id_4_chi_user", enk_w.shape[1])),
                   int(wfns_w.slices.b4_logical), enk_w.shape[1])
        a0, a1, edge_gaps = chi0_active_group(enk_w[:, :stop], win, chi0_active)
    scr = dict(inputs=inp, recipe_inputs=dict(source["recipe_inputs"]), meta=_copy.copy(meta_run),
               config=config, f_eq=f_eq, rows=np.arange(enk_w.shape[0]), win=win, mu_ry=mu,
               variant=("production" if override is None else "override"),
               support_sites="" if override is None else override["text"], scratch=scratch,
               efermi_ry=float(source["efermi_ry"]), linalg=source.get("linalg"), n=0)
    if chi0_active is not None:
        # Read by gw.response_bank (response_support, response_quadrature): the
        # refit's own meta copy only; the block rules are held in "held".
        scr["meta"].tangential_response_blocks = dict(active=(a0, a1), held={})
    scr["chi0_active"] = None if chi0_active is None else dict(
        requested=chi0_active if isinstance(chi0_active, str) else [int(x) for x in chi0_active],
        bands=[a0, a1], edge_gaps_ev=[g * RYD_TO_EV for g in edge_gaps])
    # The production variant's reference is the run's own recipe; an override is
    # resolved at the run's census (the same resolver, another support set).
    scr["recipe0"] = meta_run.tangential_recipe if override is None else _spm_recipe(scr, f_eq)
    if jax.process_index() == 0:
        os.makedirs(scratch, exist_ok=True)
    multihost_utils.sync_global_devices("dynsig spm_screening scratch")
    return scr


def _spm_recipe(scr: dict, f_kn) -> dict:
    """The refit's recipe at the census of the populations ``f_kn``: the run's resolver on a meta copy.

    Also installs that census and the resolver's fresh capacity ledger on ``scr['meta']``.
    """
    import copy as _copy
    from gw.tangential_recipe import bind_tangential_census, resolve_tangential_recipe
    inp, rin = scr["inputs"], scr["recipe_inputs"]
    meta = _copy.copy(inp["meta"])
    bind_tangential_census(inp["wfns"], meta, occupation_state=_occupation_duck(f_kn, scr["mu_ry"]),
                            trs_allowed=rin["trs_allowed"], state_capacity=rin["state_capacity"],
                            kweights=rin["kweights"])
    recipe = resolve_tangential_recipe(scr["config"], inp["wfns"], meta,
                                        mesh_xy=inp["kwargs"]["mesh_xy"],
                                        print_fn=lambda *a, **k: None,
                                        support_reads_ev=rin["support_reads_ev"])
    scr["meta"].tangential_census = meta.tangential_census
    scr["meta"].tangential_capacity = meta.tangential_capacity
    return recipe


def _recipe_sites_equal(a: dict, b: dict) -> bool:
    """Whether two recipes sample the same supports (every role's z, the held set, the widths)."""
    keys = ("z_ry", "role", "distinct_id", "held", "support_pair")
    return (all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in keys)
            and all(a[k] == b[k] for k in ("imaginary_width", "infinity_width",
                                             "line_direction_cap", "pole_budget", "height_ev")))


def spm_refit(scr: dict, f_rows, *, label: str, census: bool = True, say=None):
    """The tangential W of the window populations ``f_rows`` (nk, n_A) on the bundle's k rows.

    The run's W stage (``screen_tangential_poles``) at the occupation table f_eq with the
    window block set to ``f_rows`` (dress none: U = 1, e = eps^G, f = diag rho), after
    its recipe is re-resolved at that census (whether a support moved is recorded).
    Returns (poles, rec): ``poles`` is the window Sigma's source
    (``gw.ppm_sigma.tangential_window_source``: the new model with the refit's
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
    from gw.ppm_sigma import tangential_window_source
    from gw.tangential_screening import screen_tangential_poles
    inp = scr["inputs"]
    f = scr["f_eq"].copy()
    f[scr["rows"], scr["win"]] = np.asarray(f_rows, np.float64)
    t0 = time.perf_counter()
    recipe = _spm_recipe(scr, f)
    moved = not _recipe_sites_equal(recipe, scr["recipe0"])
    scr["meta"].tangential_recipe = recipe
    t1 = time.perf_counter()
    root = os.path.join(scr["scratch"], label + "_tangential")
    if jax.process_index() == 0 and os.path.isdir(root):
        shutil.rmtree(root)
    multihost_utils.sync_global_devices("dynsig spm_refit slot")
    kw = dict(inp["kwargs"], run_dir=scr["scratch"], label=label, tensors_filename=None,
              occupation_state=_occupation_duck(f, scr["mu_ry"]), head_resolver=None,
              print_fn=(say if say is not None else (lambda *a, **k: None)))
    out = screen_tangential_poles(inp["wfns"], inp["V_q"], scr["meta"], scr["config"], **kw)
    t2 = time.perf_counter()
    handle = out["tangential"]
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
    poles = tangential_window_source(handle, _copy.copy(scr["meta"]),
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
    """Two tangential models' payloads (``factor``, ``poles2_ry2``, ``K``): bitwise and max |a - b| (rank 0)."""
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
    """One tangential refresh's refit summary over its trials (``spm_refit`` records).

    Walls are summed over the trials; ``K`` is the first trial's per-q pole count and
    ``dK_eq`` its change against the equilibrium refit; the census (rank 0) is the
    first trial's.
    """
    walls = [_refit_walls(sr) for sr in screens]
    total = lambda key: sum(w[key] for w in walls)
    r0 = screens[0]
    out = dict(member="tangential", refit_s=sum(sr["recipe_s"] + sr["screen_s"] for sr in screens),
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
    """The tangential part of a refresh line (``window_collision``'s log)."""
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
    """The tangential part of the t0 line: the refit at rho_eq against the run's model."""
    cmp = screen_eq.get("model_vs_run") or {}
    w = _refit_walls(screen_eq)
    c = screen_eq.get("census") or {}
    return (f"; screening tangential ({screen_eq['variant']}"
            + (f" supports {screen_eq['support_sites']}" if screen_eq.get("support_sites") else "")
            + f"): refit at rho_eq {screen_eq['recipe_s'] + screen_eq['screen_s']:.2f} s (bank "
            f"{w['bank']:.2f}, moments {w['moments']:.2f}, constructor {w['constructor']:.2f}), model "
            + ("bitwise the run's" if cmp.get("bitwise") else
               "NOT bitwise the run's" if cmp else "vs the run's on rank 0")
            + f", K {min(screen_eq['K'])}-{max(screen_eq['K'])}, supports "
            f"{'MOVED' if screen_eq['sites_moved'] else 'fixed'}"
            + ("" if not c else f", Omega_min {c['omega_min_ev']:.3f} eV"))


def window_branch_operators(tabs: dict):
    """(gamma, injection, m_h), host (nk, n, n) Ry, from complex branch samples."""
    from tdagw import collision as C
    return tuple(np.asarray(g) for g in C.branch_operators(
        jnp.asarray(tabs["cond"]), jnp.asarray(tabs["val"]), jnp.asarray(tabs["static"])))


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
#: The internal line of the active collision coefficients (``[lorrax] collision_in_scattering``):
#: the populations (diag rho) or the full window rho (td_agw_dynamical.md §1.5, Eq. 8a).
IN_SCATTERINGS = ("populations", "full")
#: The intermediate states of Gamma and D (``[lorrax] collision_rate_intermediates``):
#: every band, or the window's own (the frozen part then enters M_H only).
RATE_INTERMEDIATES = ("all", "active")
#: The basis and energies of the lines (``[lorrax] collision_basis``): the band basis at the
#: bundle's energies (the incumbent), or the eigenbasis of the propagator's slow Hamiltonian
#: at its QP energies (td_agw_dynamical.md §1.5, Eq. 8b).
COLLISION_BASES = ("band", "instantaneous")
#: The full-ρ internal line's energy pairing (``[lorrax] collision_internal_pairing``;
#: ``gw.ppm_sigma.LINE_PAIRINGS``, Eq. 8c).
INTERNAL_PAIRINGS = ("symmetric", "vertex")


def slow_basis_step(apply, rho_eq, e_qp, f_eq):
    """Diagonalize the full-density reference Hamiltonian of a snapshot, in Ha.

    ``step(params, rho, P, t)`` returns ``(U, E, f, rho-rho_eq, 0, dSigma)``
    for ``H = diag(e_qp) + apply(params, rho, t)``.  The full Hermitian rho,
    including coherences, enters the mean field and the collision internal
    line.  ``P`` and ``f_eq`` retain the existing basis-helper call contract;
    there is no population projection or inner fixed point.  The coherent
    collision correction is converged by :func:`collision_snapshot_state`.

    Arrays use the propagator's k sharding: rho, P (ntr, nk, nb, nb),
    rho_eq (nk, nb, nb), e_qp, f_eq (nk, nb).  All distributed arrays enter
    the compiled program as operands, including the reference operators.
    """
    nb = int(e_qp.shape[-1])

    def _step(params, rho, P, t, rho_eq, e_qp, f_eq):
        del P, f_eq
        idx = jnp.arange(nb)
        h0 = jnp.zeros(rho_eq.shape, rho_eq.dtype).at[:, idx, idx].set(e_qp.astype(rho_eq.dtype))
        dsig = apply(params, rho, t)
        U, e, f = instantaneous_basis(h0[None] + dsig, rho)
        return U, e, f, rho - rho_eq[None], jnp.asarray(0.0), dsig

    def _start(rho, f_eq):
        del f_eq
        return jnp.zeros_like(rho)

    run, start = jax.jit(_step), jax.jit(_start)

    def step(params, rho, P, t):
        return run(params, rho, P, t, rho_eq, e_qp, f_eq)
    step.start = lambda rho: start(rho, f_eq)
    return step


def collision_basis_state(step, params, rho, t, *, steps: int = DRESS_STEPS,
                          tol: float = DRESS_TOL):
    """Evaluate the full-density snapshot basis at the supplied coherent term.

    Returns U, E (Ha), f, the dSigma of H_slow (device, the propagator's layout) and
    dict(residual, iterations).
    """
    del steps, tol
    U, E, f, _, residual, dsig = step(params, rho, step.start(rho), t)
    return U, E, f, dsig, dict(residual=float(residual), iterations=1)


@jax.jit
def _snapshot_residual(new, old):
    """Maximum matrix Frobenius norm of the Hamiltonian update, Ha (all ranks)."""
    return jnp.max(jnp.sqrt(jnp.sum(jnp.abs(new - old) ** 2, axis=(-2, -1))))


@jax.jit
def _snapshot_mix(old, new, mixing):
    return old + mixing * (new - old)


def collision_snapshot_state(evaluate, held, *, steps: int = 12,
                             tol_ha: float = 1e-9, mixing: float = 0.5,
                             acceleration: str = "linear", residual_history=None):
    """Solve the frozen-Hamiltonian Markov snapshot with full-rho lines.

    Parameters
    ----------
    evaluate : callable
        ``evaluate(held) -> (target, tables)`` diagonalizes the Hamiltonian
        with this held coherent term and uses those same eigenpairs for
        the internal phases and external columns.  ``target`` is the new
        equilibrium-subtracted Hermitian correction, in Ha, including the
        static counterterm exactly once.  The full current rho stays fixed.
    held : (ntr, nk, nb, nb) complex device array, propagator k sharding
        Predictor (usually the preceding interval's held correction), Ha.
    steps, tol_ha, mixing : bounded iterations, maximum matrix Frobenius
        residual in Ha, and linear mixing fraction in (0, 1].
    acceleration : {"linear", "anderson"}
        Linear mixing, or the existing single-evaluation type-II Anderson
        owner with at most five history entries.  ``mixing`` scales the
        residual passed to Anderson; the physical acceptance test remains
        the unscaled maximum matrix Frobenius residual in Ha.
    residual_history : list, optional
        Caller-owned sink for scalar unscaled residuals, including when
        convergence refuses. It must be empty on entry.

    Returns the converged held term, its tables, and a residual receipt.
    The accepted held term is the *input* of its final sweep, so the
    installed Hamiltonian and the tables' eigenpairs agree exactly.  Its
    difference from the newly calculated correction is bounded by tol_ha.
    An unconverged or nonfinite snapshot raises a named gate before any
    propagator state or parameters are replaced.  Every rank evaluates the
    same global residual, with distributed arrays passed as jit operands.
    """
    if int(steps) < 1 or not np.isfinite(tol_ha) or tol_ha <= 0.0 or not (
            np.isfinite(mixing) and 0.0 < mixing <= 1.0) or acceleration not in ("linear", "anderson"):
        _refuse("dynsig_collision_snapshot_controls", (steps, tol_ha, mixing, acceleration),
                "steps >= 1, finite tol_ha > 0, 0 < mixing <= 1, acceleration linear or anderson",
                "set bounded snapshot iteration controls")
    if residual_history is not None and residual_history:
        raise ValueError("snapshot residual_history must be empty on entry")
    history = [] if residual_history is None else residual_history
    if acceleration == "anderson":
        from mixing.acceleration import anderson_nojit

        latest = {}

        def residual_fn(candidate):
            target, tables = evaluate(candidate)
            residual = float(_snapshot_residual(target, candidate))
            history.append(residual)
            latest.update(held=candidate, tables=tables, residual=residual)
            if not np.isfinite(residual):
                _refuse("dynsig_collision_snapshot_convergence",
                        f"nonfinite residual after {len(history)} sweeps",
                        f"finite maximum Frobenius residual <= {tol_ha:g} Ha",
                        "inspect the screening and self-energy snapshot")
            # A zero marks physical convergence for the service's global
            # norm without changing the accepted input or evaluating again.
            return (jnp.zeros_like(candidate) if residual <= tol_ha
                    else mixing * (target - candidate))

        depth = min(5, int(steps))
        anderson_nojit(residual_fn, held, m=depth, maxit=int(steps) - 1,
                       tol=mixing * tol_ha, entry_sharding=getattr(held, "sharding", None))
        if latest["residual"] <= tol_ha:
            return latest["held"], latest["tables"], dict(
                iterations=len(history), residual=latest["residual"],
                tolerance_ha=float(tol_ha), mixing=float(mixing),
                residuals_ha=history, converged=True, acceleration=acceleration,
                history_depth=depth)
        _refuse("dynsig_collision_snapshot_convergence",
                f"residual {history[-1]:.6g} Ha after {len(history)} sweeps",
                f"finite maximum Frobenius residual <= {tol_ha:g} Ha",
                "increase snapshot_steps or improve the screening fit; no unconverged snapshot is installed")
    for it in range(1, int(steps) + 1):
        target, tables = evaluate(held)
        residual = float(_snapshot_residual(target, held))
        history.append(residual)
        if not np.isfinite(residual):
            break
        if residual <= tol_ha:
            return held, tables, dict(iterations=it, residual=residual,
                                      tolerance_ha=float(tol_ha), mixing=float(mixing),
                                      residuals_ha=history, converged=True,
                                      acceleration=acceleration)
        if it < int(steps):
            held = _snapshot_mix(held, target, mixing)
    _refuse("dynsig_collision_snapshot_convergence",
            f"residual {history[-1]:.6g} Ha after {len(history)} sweeps",
            f"finite maximum Frobenius residual <= {tol_ha:g} Ha",
            "increase snapshot_steps or reduce snapshot_mixing; no unconverged snapshot is installed")


def _slow_line(sl: dict) -> str:
    """The slow Hamiltonian's size (``window_collision``'s basis receipt) as a log fragment."""
    return (f"slow H: |E - eps^G| max {sl['de_max_mev']:.3g} meV ({sl['de_max_pct_gap']:.2g}% "
            f"E_g) rms {sl['de_rms_mev']:.3g} [K {sl['de_K_max_mev']:.3g}/{sl['de_K_rms_mev']:.3g}, "
            f"held {sl['de_held_max_mev']:.3g}/{sl['de_held_rms_mev']:.3g}], max|offdiag| "
            f"{sl['off_max_mev']:.3g} meV ({sl['off_max_pct_gap']:.2g}%) [K "
            f"{sl['off_K_max_mev']:.3g}, held {sl['off_held_max_mev']:.3g}]")


def _track_bands(U, e, f):
    """Host U (..., n, n), e, f with each column on its band and U_jj real >= 0.

    Column j becomes the eigenvector of largest weight on band j (an assignment over
    |U_mj|^2 per k).  Order and phase are a gauge of the lines (the rates are the same);
    this one makes U = 1 at a diagonal Hamiltonian whatever the order of its energies.
    """
    from scipy.optimize import linear_sum_assignment
    U, e, f = np.array(U), np.array(e), np.array(f)
    Uf, ef, ff = (U.reshape((-1,) + U.shape[-2:]), e.reshape((-1, e.shape[-1])),
                  f.reshape((-1, f.shape[-1])))
    for b in range(Uf.shape[0]):
        _, cols = linear_sum_assignment(-np.abs(Uf[b]) ** 2)
        Uf[b], ef[b], ff[b] = Uf[b][:, cols], ef[b][cols], ff[b][cols]
        d = np.diagonal(Uf[b]).copy()
        a = np.abs(d)
        Uf[b] = Uf[b] * np.where(a > 0.0, np.conj(d) / np.where(a > 0.0, a, 1.0), 1.0)[None, :]
    return Uf.reshape(U.shape), ef.reshape(e.shape), ff.reshape(f.shape)


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
                     spm_chi0_active=None, in_scattering: str = "populations",
                     rate_intermediates: str = "all", basis: str = "band",
                     pairing: str = "symmetric", model=None, snapshot_steps: int = 12,
                     snapshot_tol_ha: float = 1e-9, snapshot_mixing: float = 0.5,
                     snapshot_acceleration: str = "anderson",
                     say=print, record=None):
    """The held collision term at t0 and its refresh hook (§1.5, §7.1), GN-PPM or shared pole.

    Parameters
    ----------
    source : :func:`gw_window_source` (a GN-PPM or a scalar tangential GW run).
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
    screening : :data:`COLLISION_SCREENINGS`.  ``"frozen"``: the run's poles (or tangential
        model) throughout (the incumbent).  ``"tangential"`` (td_agw_dynamical.md §1.4,
        §7.1; a tangential source): each refresh re-runs the GW run's W stage at the
        window populations (:func:`spm_screening`, :func:`spm_refit`: response bank,
        moments, recipe and constructor, a full refit) and sweeps the window Sigma on
        the new model; as with ``"refreshed"`` the frozen part is swept per refresh,
        the head stays the run's and the static limit is the equilibrium model's
        own W^c(0) (``gw.mpa.sigma.tangential_static_wc``).  ``"refreshed"``
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
    rebase : refreshed or tangential screening only: the subtracted form's C[rho_eq] at
        the rates of rho_eq on the refresh's W(t) (one more window sweep at f_eq) instead
        of the t0 rates, so rho_eq stays a fixed point of each refresh's dissipator and the
        regularization tails of C[rho_eq] cancel at every W.  None: on for
        ``"tangential"`` with ``form = "subtracted"``, off otherwise (the GN-PPM
        member's measured option; ``"raw"`` subtracts nothing).
    spm_support_sites : ``"tangential"`` only: :func:`spm_screening`'s support set (""
        the run's production ladder, re-resolved at every state).
    spm_scratch : ``"tangential"`` only: the refits' model directory (required).
    spm_chi0_active : ``"tangential"`` only: :func:`spm_screening`'s ``chi0_active``
        (None: one rule on all pairs; ``(a0, a1)``: chi0 as four blocks).
    in_scattering : :data:`IN_SCATTERINGS`, ``sweep = "active"`` for ``"full"``: the
        active part's internal line is the populations (diag rho, the incumbent) or the
        full window rho, hole rho_mn and particle (1 - rho)_mn with the energy pairing
        ``pairing`` (``window_sigma_c``'s ``rho``; td_agw_dynamical.md §1.5, Eqs. 8a, 8c);
        each line sweep has its own held session, planned at f = 1/2, at t0 on rho_eq.
    rate_intermediates : :data:`RATE_INTERMEDIATES`, ``sweep = "active"`` for
        ``"active"``: the dissipative Gamma and D coefficients use every intermediate
        band (the incumbent) or the window's own; the frozen part then enters the coherent
        M_H correction (Delta Sigma^h) only, so frozen populations, which
        never update, are no particle source or sink of the window.
    basis : :data:`COLLISION_BASES`, ``in_scattering = "full"`` for ``"instantaneous"``:
        the lines' basis and energies.  ``"band"`` (the incumbent): U = 1, e = eps^G of
        the run, f = diag rho of the window.  ``"instantaneous"`` (frozen
        screening): each refresh solves H = diag(e_ref) + model[rho] + held,
        held = Delta M_H - Delta M_H_static for ``hermitian = "dynamic"``.
        The model receives the full current rho.  The coherent correction and
        eigenpairs converge together (:func:`collision_snapshot_state`); the
        accepted eigenvectors U and energies E are the lines' and
        the external states', rho' = U^H rho U the internal line, the frozen bands sit at
        eps^G + Delta_edge(k) (the QP correction of the window's lowest band below the
        window, of its highest band above) with their step occupation, and the frozen
        part is swept at every refresh (its external states rotate; its static limit is
        t0's rotated).  The tables are rotated back, U X U^H, before the dissipator.
    pairing : :data:`INTERNAL_PAIRINGS`, ``in_scattering = "full"`` only (``window_sigma_c``'s).
    model : the propagator's ``tdagw.self_energy.SelfEnergyModel`` (without the held
        term), read with ``basis = "instantaneous"`` only.
        The current ISDF model is a reference Hartree plus screened-exchange
        increment; e_ref is the supplied propagator spectrum.  This convention
        does not reconstruct absolute bare HF or a missing DFT Vxc counterterm.
    snapshot_steps, snapshot_tol_ha, snapshot_mixing, snapshot_acceleration : instantaneous basis only
        Maximum outer sweeps, maximum Hamiltonian-update Frobenius norm in Ha,
        and mixing fraction.  A failed solve refuses the refresh.
    say : the stdout line of each refresh; ``record`` (or None) gets each
        refresh's dict (the run log's section, the case scripts' tables).

    With ``basis = "band"`` the coherences are in the rates with ``in_scattering =
    "full"`` only.  Every rate and table is per trial and per k on the propagator's k
    rows; the window Sigma runs one tau sweep per trial with both branches kept
    (``gw.ppm_sigma.window_sigma_c(branches=True)``).  Returns dict: ``diss`` the
    t0 :class:`tdagw.collision.Dissipator` (k-sharded), ``held`` the t0 held
    Delta Sigma^h (zeros; None when ``hermitian == "none"``), ``refresh`` the hook of
    ``tdagw.propagate.run_propagation``, ``t0`` the equilibrium receipt and
    ``state`` (the hook's history), ``eq`` the t0 sweep (:func:`window_branch_operators`
    and ``window_sigma_c``'s raw output) and ``bundle_of`` (TD k row -> GW row).
    With ``basis = "band"`` the window outputs' instantaneous basis and the
    propagator's coincide (U = 1).
    """
    import time
    from common.collectives import gather_to_host
    from common.units import RYD_TO_EV
    from gw.ppm_sigma import add_window_parts, window_sigma_c, window_sigma_plan
    from tdagw import collision as C
    from tdagw import propagate as tp
    from tdagw.self_energy import held_term, with_held_term
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
    if in_scattering not in IN_SCATTERINGS:
        _refuse("dynsig_collision_in_scattering", repr(in_scattering), " | ".join(IN_SCATTERINGS),
                "pick one")
    if rate_intermediates not in RATE_INTERMEDIATES:
        _refuse("dynsig_collision_rate_intermediates", repr(rate_intermediates),
                " | ".join(RATE_INTERMEDIATES), "pick one")
    full_rho = in_scattering == "full"
    active_rates = rate_intermediates == "active"
    if (full_rho or active_rates) and (sweep != "active" or check_full):
        _refuse("dynsig_collision_in_scattering", f"in_scattering = {in_scattering!r}, "
                f"rate_intermediates = {rate_intermediates!r} with sweep = {sweep!r}"
                + (", check_full" if check_full else ""), "sweep = 'active', no check_full",
                "both act on the active part of the split sweep (the full sweep has no parts)")
    if basis not in COLLISION_BASES:
        _refuse("dynsig_collision_basis", repr(basis), " | ".join(COLLISION_BASES), "pick one")
    if pairing not in INTERNAL_PAIRINGS:
        _refuse("dynsig_collision_pairing", repr(pairing), " | ".join(INTERNAL_PAIRINGS),
                "pick one")
    instant = basis == "instantaneous"
    if (instant or pairing != "symmetric") and not full_rho:
        _refuse("dynsig_collision_basis", f"basis = {basis!r}, pairing = {pairing!r} with "
                f"in_scattering = {in_scattering!r}", "in_scattering = 'full'",
                "the basis and the pairing are the full-ρ internal line's")
    if not (sweep == "active" and full_rho and active_rates and pairing == "vertex"):
        _refuse(
            "dynsig_collision_mgkba_closure",
            f"sweep = {sweep!r}, in_scattering = {in_scattering!r}, "
            f"rate_intermediates = {rate_intermediates!r}, pairing = {pairing!r}",
            "sweep = 'active', in_scattering = 'full', rate_intermediates = 'active', "
            "pairing = 'vertex'",
            "the column-sampled full-matrix MGKBA operator conserves the closed active "
            "window only with the full-rho vertex line; the other kernels are diagnostics",
        )
    if instant and (screening != "frozen" or model is None):
        _refuse("dynsig_collision_basis", f"basis = 'instantaneous' with screening = "
                f"{screening!r}" + ("" if model is not None else ", no model"),
                "screening = 'frozen' and the propagator's model",
                "the slow Hamiltonian is the propagator's at fixed W (Eq. 8b)")
    if screening not in COLLISION_SCREENINGS:
        _refuse("dynsig_collision_screening", repr(screening), " | ".join(COLLISION_SCREENINGS),
                "pick one")
    if frozen_plan not in ("window", "gw_list"):
        _refuse("dynsig_collision_frozen_plan", repr(frozen_plan), "'window' | 'gw_list'",
                "pick one")
    refreshed = screening in ("refreshed", "tangential")
    if refreshed:
        _refuse("dynsig_collision_keldysh_screening", f"screening = {screening!r}",
                "screening = 'frozen' until a joint retarded/lesser W realization exists",
                "the refreshed paths rebuild retarded W from populations but do not supply "
                "the nonequilibrium W< required by the column-sampled MGKBA operator")
    if frozen_bands is not None and (not refreshed or sweep != "active"):
        _refuse("dynsig_collision_frozen_bands", f"frozen_bands with screening = {screening!r}, "
                f"sweep = {sweep!r}", "screening = 'refreshed' or 'tangential', sweep = 'active'",
                "the truncation is of the frozen part's change under W(t)")
    rebase = (screening == "tangential" and form != "raw") if rebase is None else bool(rebase)
    if rebase and not refreshed:
        _refuse("dynsig_collision_rebase", "rebase with screening = 'frozen'",
                "screening = 'refreshed' or 'tangential'", "the t0 rates are rho_eq's at W0")
    member = source.get("member", "gn_ppm")
    if ((screening == "refreshed" and member != "gn_ppm")
            or (screening == "tangential" and member != "tangential")):
        _refuse("dynsig_collision_member", f"screening = {screening!r} on a {member} source",
                "refreshed on a GN-PPM source, tangential on a tangential source",
                "point collision_source at the matching GW run")
    if screening == "tangential" and not spm_scratch:
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

    nkb = int(kb_frac.shape[0])
    e_frozen = frozen_rule = step = None
    if instant:
        from tdagw.tdgw_io import HA_IN_EV
        ha_ry = HA_IN_EV / RYD_TO_EV            # Ha -> Ry, the inverse of the header's (td_system)
        e_qp_b = np.empty((nkb, nb))
        e_qp_b[bundle_of] = np.asarray(gather_to_host(ops.e_qp), np.float64)[:nk] * ha_ry
        enk_all = np.asarray(jax.device_get(wfns.enk), np.float64)
        b_hi = int(b_lo) + nb
        d_lo = e_qp_b[:, 0] - enk_all[:, int(b_lo)]           # Delta_edge(k), lowest window band
        d_hi = e_qp_b[:, -1] - enk_all[:, b_hi - 1]           # and highest
        e_frozen = enk_all.copy()
        e_frozen[:, :int(b_lo)] += d_lo[:, None]
        e_frozen[:, b_hi:] += d_hi[:, None]
        e_frozen[:, win] = e_qp_b
        n_v = int(np.sum(f_td[0] > 0.5))
        gap_ry = float(e_qp_b[:, n_v:].min() - e_qp_b[:, :n_v].max())
        mev = RYD_TO_EV * 1e3

        def edge_stat(d, band, n_bands):
            return dict(band=int(band), n_bands=int(n_bands), mean_mev=float(d.mean() * mev),
                        std_mev=float(d.std() * mev), min_mev=float(d.min() * mev),
                        max_mev=float(d.max() * mev), mean_pct_gap=float(100.0 * d.mean() / gap_ry))
        frozen_rule = dict(
            rule="frozen bands at eps^G(k) + Delta_edge(k), Delta_edge = e - eps^G of the "
                 "window's edge band on their side (e the propagator's window energies); window "
                 "bands at e",
            below=edge_stat(d_lo, int(b_lo), int(b_lo)),
            above=edge_stat(d_hi, b_hi - 1, int(enk_all.shape[1]) - b_hi),
            gap_e_mev=gap_ry * mev, n_val=n_v)
        gap_g = float(e_g[:, n_v:].min() - e_g[:, :n_v].max()) * mev       # eps^G, fundamental
        gap_g_direct = float((e_g[:, n_v:].min(1) - e_g[:, :n_v].max(1)).min()) * mev
        frozen_rule.update(gap_eps_g_mev=gap_g, gap_eps_g_direct_mev=gap_g_direct)
        e_g_td = to_td(e_g)
        rho_eq_td = np.asarray(gather_to_host(ops.rho_eq))[:nk]
        slow_apply = model.apply if hermitian == "none" else with_held_term(model, None).apply
        f_eq_dev = jax.jit(lambda r: jnp.real(jnp.diagonal(r, axis1=-2, axis2=-1)))(ops.rho_eq)
        step = slow_basis_step(slow_apply, ops.rho_eq, ops.e_qp, f_eq_dev)

    def basis_at(rho_dev, params, t):
        """The collision basis of each trial (bundle rows): dict(U, e (Ry), rho', f), receipt.

        The receipt's ``slow`` (meV; % of the eps^G gap): the size of H_slow - diag(e),
        split into K = dSigma - held and the held Delta Sigma^h: max and rms over k and
        window states of |E - eps^G| and of diag(U^H X U), and max |offdiag X| in the
        band basis."""
        U, E, f, dsig, rec = collision_basis_state(step, params, rho_dev, t)
        U, E, f = _track_bands(np.asarray(gather_to_host(U))[:, :nk],
                               np.asarray(gather_to_host(E))[:, :nk],
                               np.asarray(gather_to_host(f))[:, :nk])
        dsig_h = np.asarray(gather_to_host(dsig))[:, :nk]
        held_h = (np.zeros_like(dsig_h) if hermitian == "none" else
                  np.asarray(gather_to_host(held_term(params)[0]))[:, :nk])
        ha_mev = HA_IN_EV * 1e3
        Uc = np.conj(U)
        on = lambda X: np.real(np.einsum("tkji,tkjl,tkli->tki", Uc, X, U)) * ha_mev
        off = lambda X: float(np.max(np.abs(X - np.einsum("tkii->tki", X)[..., None]
                                            * np.eye(nb)))) * ha_mev
        mx = lambda a: float(np.max(np.abs(a)))
        rms = lambda a: float(np.sqrt(np.mean(np.abs(a) ** 2)))
        de = (E * ha_ry - e_g_td[None]) * mev
        dK, dH = on(dsig_h - held_h), on(held_h)
        slow = dict(de_max_mev=mx(de), de_rms_mev=rms(de), de_K_max_mev=mx(dK),
                    de_K_rms_mev=rms(dK), de_held_max_mev=mx(dH), de_held_rms_mev=rms(dH),
                    off_max_mev=off(dsig_h), off_K_max_mev=off(dsig_h - held_h),
                    off_held_max_mev=off(held_h))
        slow.update({k_.replace("_mev", "_pct_gap"): 100.0 * v_ / gap_g for k_, v_ in list(slow.items())})
        rec["slow"] = slow
        rho_h = np.asarray(gather_to_host(rho_dev))[:, :nk]
        out = []
        for tr in range(U.shape[0]):
            Ub = np.empty((nkb, nb, nb), complex)
            Ub[bundle_of] = U[tr]
            Eb = np.empty((nkb, nb))
            Eb[bundle_of] = E[tr] * ha_ry
            rb = np.empty((nkb, nb, nb), complex)
            rb[bundle_of] = rho_h[tr]
            rp = np.conj(np.swapaxes(Ub, -1, -2)) @ rb @ Ub
            rp = 0.5 * (rp + np.conj(np.swapaxes(rp, -1, -2)))
            out.append(dict(U=Ub, e=Eb, rho=rp, f=np.real(np.einsum("kii->ki", rp))))
        rec.update(max_u_minus_1=float(max(np.max(np.abs(o["U"] - np.eye(nb))) for o in out)),
                   max_de_mev=float(max(np.max(np.abs(o["e"] - e_qp_b)) for o in out) * mev))
        return out, rec

    def window(f_bundle, plan, internal=None, *, at=None, keep=None, wc0=None, static=None,
               rho=None, basis_now=None):
        """One window Sigma (both branches) at window populations f (bundle rows), poles ``at``.

        ``rho`` (bundle rows, nk, nb, nb): the full-ρ internal line (active part only);
        ``basis_now`` (:func:`basis_at`'s, or None: the band basis): U, e of the lines."""
        t = time.perf_counter()
        st = dynamic if static is None else bool(static)
        w = dict(b_lo=int(b_lo), U=eye, e=e_g, f=f_bundle)
        if basis_now is not None:
            w.update(U=basis_now["U"], e=basis_now["e"], e_frozen=e_frozen)
        if rho is not None:
            w["rho"] = rho
        out = window_sigma_c(wfns, poles if at is None else at, w, plan, meta=meta,
                             mesh_xy=mesh, branches=True, static_limit=st,
                             internal=internal, keep_frozen=keep, pairing=pairing,
                             static_limit_wc0=(W0c if (spm and st and wc0 is None) else wc0))
        out["wall_s"] = time.perf_counter() - t
        return out

    spm = member == "tangential"
    scr, screen_eq = None, None
    if screening == "tangential":
        t_s = time.perf_counter()
        scr = spm_screening(source, mesh=mesh, b_lo=int(b_lo), n_active=nb,
                            support_sites=spm_support_sites, scratch=spm_scratch,
                            chi0_active=spm_chi0_active)
        t_b0 = time.perf_counter()
        poles_eq, screen_eq = spm_refit(scr, f_b, label="eq")
        if scr["variant"] == "override":
            poles = poles_eq                     # V2: the equilibrium model of the support set
        screen_eq.update(setup_s=t_b0 - t_s, variant=scr["variant"],
                         support_sites=scr["support_sites"],
                         model_vs_run=_model_payload_cmp(source["model"]["path"],
                                                         poles_eq["model"]["path"]))
    if spm:                                             # the model's own W^c(0) (owner ruling)
        from gw.mpa.sigma import tangential_static_wc
        from gw.ppm_sigma import fresh_tangential_capacity
        W0c = tangential_static_wc(poles["model"], fresh_tangential_capacity(poles["meta"], mesh),
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

    def diag_rho(f_bundle):
        """diag(f) as a window rho (bundle rows): the full-ρ line at populations alone."""
        return np.einsum("ki,ij->kij", np.asarray(f_bundle, np.float64),
                         np.eye(nb)).astype(np.complex128)

    b0, basis_t0 = [None], None
    if instant:
        t_b = time.perf_counter()
        rho_eq_h = np.asarray(gather_to_host(ops.rho_eq))[:nk]
        params0 = model.params if hermitian == "none" else (
            model.params, tp.place_window_blocks(np.zeros((ntr, nk, nb, nb), complex), mesh, nk_pad))
        b0, basis_t0 = basis_at(tp.place_window_blocks(
            np.broadcast_to(rho_eq_h, (ntr, nk, nb, nb)), mesh, nk_pad), params0, 0.0)
        basis_t0["basis_s"] = time.perf_counter() - t_b
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
            window(f_b, plans["frozen"], "frozen", basis_now=b0[0])
        held_parts["frozen"] = window(f_b, plans["frozen"], "frozen", basis_now=b0[0])
        half_f = np.full_like(f_b, 0.5)
        window(half_f, plans["active"], "active", rho=diag_rho(half_f) if full_rho else None,
               basis_now=b0[0])
        if keep is not None:
            plans["frozen_keep"] = window_sigma_plan(cfg, head=None, session={}, window_only=True)
            window(f_b, plans["frozen_keep"], "frozen", keep=keep, static=False)
            held_parts["frozen_keep"] = window(f_b, plans["frozen_keep"], "frozen", keep=keep,
                                               static=False)
    t_split = time.perf_counter() - t_split

    def tables(out, rates_of=None, U=None):
        """Gamma, injection D, M_H and the static-W limit, on the TD k rows.

        ``rates_of``: the output whose branches give Gamma and D (the active part, with
        ``rate_intermediates = "active"``); M_H and the static limit stay ``out``'s.
        ``U`` (bundle rows): the outputs' basis, rotated back to the band basis."""
        gamma, injection, s = window_branch_operators(out["branch_pair"])
        if rates_of is not None:
            gamma, injection, _ = window_branch_operators(rates_of["branch_pair"])
        static = None
        if dynamic:
            idx = np.arange(nb)
            static = out["static_limit"]["sx"].copy()
            static[:, idx, idx] += out["static_limit"]["head"]
        if U is not None:
            Uh = np.conj(np.swapaxes(U, -1, -2))
            gamma, injection, s = U @ gamma @ Uh, U @ injection @ Uh, U @ s @ Uh
            static = None if static is None else U @ static @ Uh
        return dict(gamma=to_td(gamma), injection=to_td(injection), m_h=to_td(s),
                    static=None if static is None else to_td(static), raw=out,
                    wall_s=out["wall_s"])

    def full_sweep(f_bundle, at=None):
        out = window(f_bundle, splan, at=at, wc0=None if at is None else W0c)
        return dict(tables(out), nodes=_sweep_nodes(out))

    scratch = {}

    def frozen_part(f_bundle, at, basis_now=None):
        """The frozen bands' part on the poles ``at`` (static limit: the held one at W0),
        in the basis ``basis_now`` (its static limit: t0's rotated)."""
        if keep is None:
            fz = window(f_bundle, plans["frozen"], "frozen", at=at, static=False,
                        basis_now=basis_now)
            out = dict(fz)
        else:
            fz = window(f_bundle, plans["frozen_keep"], "frozen", at=at, keep=keep, static=False)
            out = _shift_window_part(held_parts["frozen"], fz, held_parts["frozen_keep"])
        if "static_limit" in held_parts["frozen"]:
            sl = held_parts["frozen"]["static_limit"]
            if basis_now is not None:            # frequency independent: t0's, rotated
                V = np.conj(np.swapaxes(b0[0]["U"], -1, -2)) @ basis_now["U"]
                sl = dict(sl, sx=np.conj(np.swapaxes(V, -1, -2)) @ sl["sx"] @ V)
            out["static_limit"] = sl
        return out, fz

    def sweep_tables(f_bundle, at=None, rho=None, basis_now=None):
        """The refresh's window Sigma at window populations f (bundle rows), poles ``at``.

        ``rho`` (bundle rows): the window's density matrix, read with ``in_scattering =
        "full"`` (None: diag f, the populations' state), in the basis ``basis_now``
        (:func:`basis_at`'s; None: the band basis)."""
        if sweep == "full":
            return full_sweep(f_bundle, at=at)
        extra = {}
        frozen = held_parts["frozen"]
        if at is not None or basis_now is not None:
            frozen, fz = frozen_part(f_bundle, at, basis_now=basis_now)
            extra = dict(frozen_s=fz["wall_s"], frozen_nodes=_sweep_nodes(fz))
        r_in = None
        if full_rho:
            r_in = diag_rho(f_bundle) if rho is None else rho
        act = window(f_bundle, plans["active"], "active", at=at,
                     wc0=None if at is None else W0c, rho=r_in, basis_now=basis_now)
        scratch["act"] = act
        out = add_window_parts(frozen, act)
        out["wall_s"] = act["wall_s"] + extra.get("frozen_s", 0.0)
        return dict(tables(out, rates_of=act if active_rates else None,
                           U=None if basis_now is None else basis_now["U"]),
                    nodes=_sweep_nodes(act), active_s=act["wall_s"], **extra)

    put = lambda a: tp.place_window_blocks(a, mesh, nk_pad)
    stack = lambda key, parts: np.stack([p[key] for p in parts]) * half
    program = tp.collision_program(mesh, raw=(form == "raw"))
    pops = jax.jit(lambda r: jnp.real(jnp.diagonal(r, axis1=-2, axis2=-1)))
    fs = 1000.0 / 24.188843265857                       # a.u. of time per fs

    def number_balance(result):
        """Per-trial instantaneous number flux and residual, in electrons/fs.

        This is the freshly rebuilt generator evaluated on the refresh state.
        It is distinct from the accumulated trace drift while the coefficients
        are held until the next refresh.
        """
        out = np.asarray(result["out"], np.float64) * fs
        inn = np.asarray(result["inn"], np.float64) * fs
        residual = out + inn
        flux = 0.5 * (np.abs(out) + np.abs(inn))
        rel = np.full_like(flux, np.inf)
        np.divide(np.abs(residual), flux, out=rel, where=flux > 0.0)
        rel[(flux == 0.0) & (residual == 0.0)] = 0.0
        return dict(out_per_fs=out.tolist(), in_per_fs=inn.tolist(),
                    number_flux_per_fs=flux.tolist(),
                    number_residual_per_fs=residual.tolist(),
                    number_residual_rel=rel.tolist())

    def populations(rho):
        return np.asarray(gather_to_host(pops(rho)))[:, :nk]               # (ntr, nk, nb)

    ry_mev = RYD_TO_EV * 1e3

    def edge_rates(parts_):
        """min/max over k and trials of diag Gamma and injection D at the VBM and CBM
        bands (band basis), meV."""
        nv = int(np.sum(f_td[0] > 0.5))
        out = {}
        for name, j in (("vbm", nv - 1), ("cbm", nv)):
            for g, tag in (("gamma", "out"), ("injection", "in")):
                d = np.concatenate([np.real(np.einsum("kii->ki", p_[g]))[:, j] for p_ in parts_])
                out[f"{name}_{tag}"] = [float(d.min() * ry_mev), float(d.max() * ry_mev)]
        return out

    def compare(a, b, h_a=None, h_b=None):
        """Split vs full Gamma, injection D, M_H and coherent correction, meV."""
        mx = lambda x: float(np.max(np.abs(x)))
        scale = max(mx(b["gamma"]), mx(b["injection"]), 1e-300)
        d = dict(rate_out_mev=mx(a["gamma"] - b["gamma"]) * ry_mev,
                 rate_in_mev=mx(a["injection"] - b["injection"]) * ry_mev,
                 rate_rel=max(mx(a["gamma"] - b["gamma"]), mx(a["injection"] - b["injection"])) / scale,
                 s_mev=mx(a["m_h"] - b["m_h"]) * ry_mev, full_tau=b["nodes"]["distinct_tau"],
                 full_s=b["wall_s"])
        if h_a is not None:
            d["herm_mev"] = mx(np.asarray(h_a) - np.asarray(h_b)) * 2.0 * ry_mev   # Ha -> meV
        return d

    # ---- t0: the equilibrium rates, C[rho_eq], the first dissipator
    t_a = time.perf_counter()
    eq = (sweep_tables(f_b) if not instant else
          sweep_tables(b0[0]["f"], rho=b0[0]["rho"], basis_now=b0[0]))
    t_b = time.perf_counter()
    eq_full = full_sweep(f_b) if check_full else None
    gamma_eq, injection_eq = put(stack("gamma", [eq] * ntr)), put(stack("injection", [eq] * ntr))
    rho0 = put(np.broadcast_to(np.asarray(gather_to_host(ops.rho_eq))[:nk], (ntr, nk, nb, nb)))
    res = program(gamma_eq, injection_eq, rho0, ops.rho_eq, gamma_eq, injection_eq, ops.w_k)
    c_eq = res["c_eq"]
    jax.block_until_ready(res["diss"])
    t_c = time.perf_counter()
    balance0 = number_balance(res)
    geq_h = np.stack([eq["gamma"], eq["injection"]])                            # Ry
    d_out, d_in = (np.real(np.einsum("kii->ki", g)) for g in geq_h)
    receipt = dict(
        t_fs=0.0, why="t0", wall_s=t_c - t_a, sweep_s=t_b - t_a, build_s=t_c - t_b,
        min_rate_mev=float(res["min_rate"]) * 2.0 * ry_mev,
        min_injection_mev=float(res["min_injection"]) * 2.0 * ry_mev,
        min_blocking_mev=float(res["min_blocking"]) * 2.0 * ry_mev,
        rho_min=float(res["rho_min"]), rho_max=float(res["rho_max"]),
        rate_out_diag_mev=[float(d_out.min() * ry_mev), float(d_out.max() * ry_mev)],
        rate_in_diag_mev=[float(d_in.min() * ry_mev), float(d_in.max() * ry_mev)],
        min_eig_out_mev=float(np.linalg.eigvalsh(0.5 * (eq["gamma"] + np.conj(
            np.swapaxes(eq["gamma"], -1, -2)))).min() * ry_mev),
        min_eig_in_mev=float(np.linalg.eigvalsh(0.5 * (eq["injection"] + np.conj(
            np.swapaxes(eq["injection"], -1, -2)))).min() * ry_mev),
        c_eq_diag=float(res["c_eq_diag"]), c_eq_offdiag=float(res["c_eq_offdiag"]),
        trace=[float(v) for v in np.asarray(res["trace"])], form=form, hermitian=hermitian,
        sweep=sweep, nodes=eq["nodes"], in_scattering=in_scattering,
        rate_intermediates=rate_intermediates,
        frozen=(None if not held_parts else dict(
            nodes=_sweep_nodes(held_parts["frozen"]), wall_s=held_parts["frozen"]["wall_s"],
            setup_s=t_split)), screening=screening, screen_eq=screen_eq)
    receipt.update(balance0)
    if instant:
        receipt.update(basis=basis, pairing=pairing, basis_t0=basis_t0,
                       frozen_energies=frozen_rule, edges=edge_rates([eq]),
                       frozen_s=eq.get("frozen_s"), active_s=eq.get("active_s"),
                       hamiltonian_convention="reference_increment: diag(e_ref) + "
                       "model[rho] + equilibrium-subtracted Hermitian correction",
                       snapshot_controls=dict(steps=int(snapshot_steps),
                                              tolerance_ha=float(snapshot_tol_ha),
                                              mixing=float(snapshot_mixing),
                                              acceleration=snapshot_acceleration))
    elif pairing != "symmetric":
        receipt.update(pairing=pairing)
    if eq_full is not None:
        receipt["full_vs_split"] = compare(eq, eq_full)
    s_eq = np.stack([eq["m_h"]] * ntr)
    static_eq = None if not dynamic else np.stack([eq["static"]] * ntr)
    if eq_full is not None:
        s_eq_full = np.stack([eq_full["m_h"]] * ntr)
        static_eq_full = None if not dynamic else np.stack([eq_full["static"]] * ntr)
    held0 = None if hermitian == "none" else put(np.zeros((ntr, nk, nb, nb), complex))
    state = dict(t_last=0.0, f_last=np.broadcast_to(f_td, (ntr, nk, nb)).copy(), n=0,
                 trace_last=np.asarray(res["trace"]).copy(), history=[receipt],
                 held_max_mev=0.0, valid_prev=[None] * ntr,
                 basis_last=b0 if instant else None, snapshot_held=held0)
    line = (f"  collision t0     : {form}, hermitian {hermitian}, in-scattering {in_scattering}, "
            f"rate intermediates {rate_intermediates}; operators at rho_eq: diag "
            f"Gamma [{receipt['rate_out_diag_mev'][0]:.3g}, {receipt['rate_out_diag_mev'][1]:.3g}] meV, "
            f"D [{receipt['rate_in_diag_mev'][0]:.3g}, {receipt['rate_in_diag_mev'][1]:.3g}] meV, "
            f"min eig Gamma {receipt['min_rate_mev']:.3g} meV, D_eff "
            f"{receipt['min_injection_mev']:.3g} meV, Gamma-D_eff "
            f"{receipt['min_blocking_mev']:.3g} meV; rho bounds "
            f"[{receipt['rho_min']:.3e}, {receipt['rho_max']:.3e}]; ||C[rho_eq]|| diag "
            f"{receipt['c_eq_diag']:.3e} "
            f"off-diag {receipt['c_eq_offdiag']:.3e} Ha/hbar; number residual "
            f"{max(receipt['number_residual_per_fs'], key=abs):+.3e} e/fs "
            f"({max(receipt['number_residual_rel']):.3e} of flux); wall "
            f"{receipt['wall_s']:.2f} s "
            f"(sweep {receipt['sweep_s']:.2f}, build {receipt['build_s']:.2f}); sweep {sweep}: "
            f"{eq['nodes']['distinct_tau']} tau in {eq['nodes']['n_windows']} windows"
            + ("" if not held_parts else
               f" (frozen part {'at t0' if not refreshed else 'per refresh'}: "
               f"{receipt['frozen']['nodes']['distinct_tau']} tau, "
               f"{receipt['frozen']['wall_s']:.2f} s)")
            + ("" if not instant else
               f"; {pairing} pairing, instantaneous basis: t0 fixed point "
               f"{basis_t0['iterations']} it (residual {basis_t0['residual']:.1e}, max|U - 1| "
               f"{basis_t0['max_u_minus_1']:.1e}, max|E - e| {basis_t0['max_de_mev']:.1e} meV); "
               f"frozen bands at eps^G + Delta_edge: below (band {frozen_rule['below']['band']}) "
               f"{frozen_rule['below']['mean_mev']:.1f} meV mean "
               f"({frozen_rule['below']['mean_pct_gap']:.1f}% E_g), above (band "
               f"{frozen_rule['above']['band']}) {frozen_rule['above']['mean_mev']:.1f} meV "
               f"({frozen_rule['above']['mean_pct_gap']:.1f}% E_g), E_g(e) "
               f"{frozen_rule['gap_e_mev']:.1f} meV, eps^G {frozen_rule['gap_eps_g_mev']:.1f} "
               f"(direct {frozen_rule['gap_eps_g_direct_mev']:.1f}) meV; "
               + _slow_line(basis_t0["slow"])
               + "; edge diagonals VBM Gamma "
               f"[{receipt['edges']['vbm_out'][0]:.3g}, {receipt['edges']['vbm_out'][1]:.3g}] D "
               f"[{receipt['edges']['vbm_in'][0]:.3g}, {receipt['edges']['vbm_in'][1]:.3g}], CBM Gamma "
               f"[{receipt['edges']['cbm_out'][0]:.3g}, {receipt['edges']['cbm_out'][1]:.3g}] D "
               f"[{receipt['edges']['cbm_in'][0]:.3g}, {receipt['edges']['cbm_in'][1]:.3g}] meV")
            + ("" if screen_eq is None or screening != "tangential" else spm_t0_line(screen_eq))
            + ("" if screen_eq is None or screening == "tangential" else
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

    def refresh(t, rho, ops_now, params):
        f = populations(rho)
        due, df, why = C.refresh_due(policy, t, state["t_last"], f, state["f_last"])
        if not due:
            return None
        t1 = time.perf_counter()
        rho_h = np.asarray(gather_to_host(rho))[:, :nk] if full_rho and not instant else None
        bnow, brec, basis_s = [None] * ntr, None, 0.0
        parts, f_rows, screens, poles_now = [], [], [], []
        snapshot_held, snapshot_rec = None, None
        if instant:
            def evaluate_snapshot(candidate):
                nonlocal basis_s
                t_basis = time.perf_counter()
                candidate_params = params
                if hermitian != "none":
                    _, set_term = held_term(params)
                    candidate_params = set_term(candidate)
                bases, basis_rec = basis_at(rho, candidate_params, t)
                basis_s += time.perf_counter() - t_basis
                rows = [sweep_tables(b["f"], rho=b["rho"], basis_now=b) for b in bases]
                target = candidate
                if hermitian != "none":
                    target = put(np.asarray(C.hermitian_update(
                        np.stack([p["m_h"] for p in rows]), s_eq,
                        None if not dynamic else np.stack([p["static"] for p in rows]),
                        static_eq)) * half)
                return target, (rows, bases, basis_rec)

            predictor = (jnp.zeros_like(rho) if hermitian == "none" else held_term(params)[0])
            snapshot_held, snapshot_payload, snapshot_rec = collision_snapshot_state(
                evaluate_snapshot, predictor, steps=snapshot_steps,
                tol_ha=snapshot_tol_ha, mixing=snapshot_mixing,
                acceleration=snapshot_acceleration)
            parts, bnow, brec = snapshot_payload
            # The outer residual is the coherent self-consistency condition;
            # the basis at a supplied held term is one direct diagonalization.
            brec.update(snapshot_rec)
            f_rows = [b["f"] for b in bnow]
        for tr in range(0 if instant else ntr):
            fb = np.empty((kb_frac.shape[0], nb))
            fb[bundle_of] = f[tr]
            rb = None
            if full_rho:
                rb = np.empty((kb_frac.shape[0], nb, nb), np.complex128)
                rb[bundle_of] = rho_h[tr]
            f_rows.append(fb)
            at = None
            if screening == "tangential":       # two slots per trial: this refresh's and the last
                at, srec = spm_refit(scr, fb, label=f"s{(state['n'] + 1) % 2}t{tr}")
                screens.append(srec)
                poles_now.append(at)
            elif refreshed:
                at, _, srec = screen_populations(scr, fb, valid_prev=state["valid_prev"][tr])
                state["valid_prev"][tr] = at.valid_mask_q
                screens.append(srec)
                poles_now.append(at)
            parts.append(sweep_tables(fb, at=at, rho=rb, basis_now=bnow[tr]))
            if check_truncation:
                fz = window(fb, plans["frozen"], "frozen", at=at, static=False)
                whole = dict(fz)
                if "static_limit" in held_parts["frozen"]:
                    whole["static_limit"] = held_parts["frozen"]["static_limit"]
                alt = dict(tables(add_window_parts(whole, scratch["act"]),
                                  rates_of=scratch["act"] if active_rates else None),
                           nodes=_sweep_nodes(fz), wall_s=fz["wall_s"])
                parts[-1]["_truncation"] = alt
        t2 = time.perf_counter()
        gamma, injection = put(stack("gamma", parts)), put(stack("injection", parts))
        g_eq_now, rebase_s = (gamma_eq, injection_eq), 0.0
        if rebase:
            tr0 = time.perf_counter()
            eqs = [sweep_tables(f_b, at=p_) for p_ in poles_now]
            g_eq_now = (put(stack("gamma", eqs)), put(stack("injection", eqs)))
            rebase_s = time.perf_counter() - tr0
        t3 = time.perf_counter()
        out = program(gamma, injection, rho, ops_now.rho_eq, *g_eq_now, ops_now.w_k)
        new_ops = ops_now._replace(collision=out["diss"])
        jax.block_until_ready(out["diss"])
        t4 = time.perf_counter()
        new_params, delta, held_max, h = params, None, 0.0, None
        if hermitian != "none":
            s_t = np.stack([p["m_h"] for p in parts])
            st_t = None if not dynamic else np.stack([p["static"] for p in parts])
            h = (np.asarray(C.hermitian_update(s_t, s_eq, st_t, static_eq)) * half
                 if snapshot_held is None else
                 np.asarray(gather_to_host(snapshot_held))[:, :nk])            # Ha
            held_max = float(np.max(np.abs(h))) * 2.0 * ry_mev
            term, set_term = held_term(params)
            new = put(h) if snapshot_held is None else snapshot_held
            delta = new - term
            new_params = set_term(new)
        t5 = time.perf_counter()
        trace = np.asarray(out["trace"])
        balance = number_balance(out)
        elapsed_fs = (float(t) - state["t_last"]) / fs
        trace_drift_per_fs = ((trace - state["trace_last"]) / elapsed_fs).tolist()
        state["n"] += 1
        rec = dict(n=state["n"], t_fs=float(t) / fs, why=why, df=df,
                   wall_s=t5 - t1, sweep_s=t2 - t1, rates_s=t3 - t2, build_s=t4 - t3,
                   hermitian_s=t5 - t4,
                   d_trace=[float(v) for v in (trace - state["trace_last"])],
                   trace_drift_per_fs=trace_drift_per_fs,
                   min_rate_mev=float(out["min_rate"]) * 2.0 * ry_mev,
                   min_injection_mev=float(out["min_injection"]) * 2.0 * ry_mev,
                   min_blocking_mev=float(out["min_blocking"]) * 2.0 * ry_mev,
                   rho_min=float(out["rho_min"]), rho_max=float(out["rho_max"]),
                   held_max_mev=held_max,
                   c_eq_diag=receipt["c_eq_diag"], c_eq_offdiag=receipt["c_eq_offdiag"],
                   sweep=sweep, nodes=parts[0]["nodes"], screening=screening,
                   rebase_s=rebase_s, c_eq_now=(None if not rebase else
                                                [float(out["c_eq_diag"]), float(out["c_eq_offdiag"])]))
        rec.update(balance)
        if screening == "tangential":
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
        if instant:
            rec.update(basis=dict(brec, basis_s=basis_s), edges=edge_rates(parts))
            rec.update(snapshot=snapshot_rec,
                       hamiltonian_convention="reference_increment: diag(e_ref) + "
                       "model[rho] + equilibrium-subtracted Hermitian correction")
        if instant and state["n"] == 1:          # once, at the first refresh after t0
            # C[rho_eq] at the t0 rates (the subtracted constant) and at this refresh's
            # rates, Ha/hbar (Frobenius over k and the window)
            coll = lambda gamma, injection: 0.5 * (
                injection - 0.5 * (gamma @ rho_eq_td + rho_eq_td @ gamma))
            offd = lambda X: X - np.einsum("...ii->...i", X)[..., None] * np.eye(nb)
            fro = lambda X: float(np.linalg.norm(X))
            c0 = coll(eq["gamma"], eq["injection"])
            shift = max(fro(offd(coll(p_["gamma"], p_["injection"]) - c0)) for p_ in parts)
            rec["c_eq_split"] = dict(
                c_eq_fro=fro(c0), c_eq_offdiag_frac=fro(offd(c0)) / max(fro(c0), 1e-300),
                shift_offdiag_fro=shift, shift_offdiag_rel=shift / max(fro(c0), 1e-300))
        if (refreshed or instant) and sweep == "active":
            rec.update(frozen_s=sum(p_["frozen_s"] for p_ in parts),
                       active_s=sum(p_["active_s"] for p_ in parts),
                       frozen_nodes=parts[0]["frozen_nodes"])
        if check_truncation:                 # diagnostic, outside the comparison's own wall
            h_alt = None
            if hermitian != "none":
                h_alt = np.asarray(C.hermitian_update(
                    np.stack([p_["_truncation"]["m_h"] for p_ in parts]), s_eq,
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
                    np.stack([p["m_h"] for p in fulls]), s_eq_full,
                    None if not dynamic else np.stack([p["static"] for p in fulls]),
                    static_eq_full)) * half
            cmp = [compare(p, q, None if h is None else h[i], None if h_full is None else h_full[i])
                   for i, (p, q) in enumerate(zip(parts, fulls))]
            rec["full_vs_split"] = {k: max(c[k] for c in cmp) for k in cmp[0]}
        state.update(t_last=float(t), f_last=f, trace_last=trace)
        if instant:
            state.update(basis_last=bnow, snapshot_held=snapshot_held)
        state["history"].append(rec)
        line = (f"  collision refresh {rec['n']:3d} at {rec['t_fs']:9.4f} fs ({why}): wall "
                f"{rec['wall_s']:.2f} s (sweep {rec['sweep_s']:.2f}, rates {rec['rates_s']:.3f}, "
                f"dissipator {rec['build_s']:.3f}, hermitian {rec['hermitian_s']:.3f}), max|df| "
                f"{df:.3e}, dTr rho {max(rec['d_trace'], key=abs):+.3e} since last; d(Tr rho)/dt "
                f"out {min(rec['out_per_fs']):+.3e} in {max(rec['in_per_fs']):+.3e} per fs; "
                f"instantaneous number residual {max(rec['number_residual_per_fs'], key=abs):+.3e} "
                f"e/fs ({max(rec['number_residual_rel']):.3e} of flux), held trace drift "
                f"{max(rec['trace_drift_per_fs'], key=abs):+.3e} e/fs; min "
                f"eig Gamma {rec['min_rate_mev']:.3g} meV, D_eff "
                f"{rec['min_injection_mev']:.3g} meV, Gamma-D_eff "
                f"{rec['min_blocking_mev']:.3g} meV; rho bounds "
                f"[{rec['rho_min']:.3e}, {rec['rho_max']:.3e}]; max|dSigma^h| "
                f"{held_max:.3g} meV; "
                f"||C[rho_eq]|| diag {rec['c_eq_diag']:.2e} off {rec['c_eq_offdiag']:.2e}; "
                f"{sweep} sweep {rec['nodes']['distinct_tau']} tau"
                + (f" ({rec['nodes']['rebuilds']} rules rebuilt)" if rec["nodes"]["rebuilds"] else "")
                + ("" if "screen" not in rec or screening != "tangential" else
                   spm_line(rec))
                + ("" if "screen" not in rec or screening == "tangential" else
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
                + ("" if not instant else
                   f"; basis {rec['basis']['iterations']} it (residual "
                   f"{rec['basis']['residual']:.1e}, max|U - 1| {rec['basis']['max_u_minus_1']:.2e}, "
                   f"max|E - e| {rec['basis']['max_de_mev']:.3g} meV, {basis_s:.2f} s; "
                   + _slow_line(rec["basis"]["slow"]) + "), frozen "
                   f"sweep {rec['frozen_s']:.2f} s, active {rec['active_s']:.2f} s; edge diagonals VBM "
                   f"Gamma [{rec['edges']['vbm_out'][0]:.3g}, {rec['edges']['vbm_out'][1]:.3g}] D "
                   f"[{rec['edges']['vbm_in'][0]:.3g}, {rec['edges']['vbm_in'][1]:.3g}], CBM Gamma "
                   f"[{rec['edges']['cbm_out'][0]:.3g}, {rec['edges']['cbm_out'][1]:.3g}] D "
                   f"[{rec['edges']['cbm_in'][0]:.3g}, {rec['edges']['cbm_in'][1]:.3g}] meV"
                   + ("" if "c_eq_split" not in rec else
                      f"; C[rho_eq]: offdiag share {rec['c_eq_split']['c_eq_offdiag_frac']:.3e} of "
                      f"{rec['c_eq_split']['c_eq_fro']:.3e} Ha/hbar, offdiag(C_t - C_eq)[rho_eq] "
                      f"{rec['c_eq_split']['shift_offdiag_fro']:.3e} Ha/hbar "
                      f"({rec['c_eq_split']['shift_offdiag_rel']:.3e} rel)"))
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
                c_eq=c_eq, operators_eq=(gamma_eq, injection_eq), sweep=sweep_tables, program=program,
                eq=eq, bundle_of=bundle_of, parts=held_parts, plans=plans, screening=scr,
                window=window, tables=tables, keep=keep, rebase=rebase, W0c=W0c, poles=poles,
                in_scattering=in_scattering, rate_intermediates=rate_intermediates,
                basis=basis, pairing=pairing, basis_t0=basis_t0, e_frozen=e_frozen,
                frozen_energies=frozen_rule, basis_of=basis_at if instant else None)

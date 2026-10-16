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

import functools
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from distrib_la import eigh_local

__all__ = ["DynsigKernels", "assemble_dynsig_bundle", "check_state", "chi0_split", "split_bundle",
           "dynamic_correction", "dynsig_model", "equilibrium_sigma", "instantaneous_basis",
           "literal_sx", "make_dynsig_kernels", "screened_update", "static_rule"]

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

def chi0_split(dyn: dict, kern: DynsigKernels):
    """chi0^ff, chi0^fa, chi0^aa at equilibrium on the run's rule, (nq, mu, nu) each, Ry^-1.

    chi0_q = (g_s / N_k) sum_k sum_ab (f_a - f_b)/(E_a - E_b) rho_ab(mu) rho_ab*(nu)
    on the Laplace rule; G_frozen and G_active(0) are separate Green tiles at each
    node (the t = 0 consistency check of the design page: their sum must be
    the run's chi0).  Returns a dict with the three terms.
    """
    nk, mesh = int(np.prod(kern.kgrid)), kern.mesh
    b_lo, nA = dyn["band_lo"], dyn["n_active"]
    act = jnp.zeros(dyn["nbf_pad"]).at[b_lo:b_lo + nA].set(1.0)[None, :]
    e, f, live = dyn["e_full"], dyn["f_full"], dyn["live"]
    x_sep = jnp.asarray(dyn["rule"]["e_sep"])
    psi_X, psi_Y = dyn["psi_X"], dyn["psi_Y"]

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
    acc, _ = lax.scan(body, (zero, zero, zero), (dyn["tau"], dyn["alpha"]), unroll=1)
    scale = -2.0 * dyn["g_s"] / np.sqrt(nk)
    return {name: kern.c_fft(scale * jnp.real(a).astype(jnp.complex128))
            for name, a in zip(("ff", "fa", "aa"), acc)}


def _same_side_chi(dyn, kern, psi_A, e_A, f_A, x_sep):
    """Exact omega = 0 sum over ordered same-side pairs with an active leg, (nq, mu, nu), Ry^-1.

    chi^ss_q = (g_s/N_k) sum_k sum_ab c_ab rho_ab(mu) rho_ab*(nu), a at k, b at k-q,
    rho_ab(mu) = sum_s psi_a,s(mu) psi*_b,s(mu), c_ab = (f_a - f_b)/(E_a - E_b) where both
    legs lie on one side of e_sep (cross-gap pairs are on the Laplace rule).  E1: a in
    the window at k, b any band; E2: a frozen at k, b in the window.  Every q is one
    ``lax.map`` step, so one q's pair densities (nk, nA, nbf, mu) are live at a time.
    Returns (chi, bad): bad counts near-degenerate same-side pairs whose populations
    differ (their static limit is a Drude term this prototype refuses).
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
        ok = same & (jnp.abs(dE) > tol) & mask
        bad = same & (jnp.abs(dE) <= tol) & (jnp.abs(df) > otol) & mask
        return jnp.where(ok, df / jnp.where(ok, dE, 1.0), 0.0), jnp.sum(bad)

    def pair_chi(r, c):
        return jnp.einsum("kabu,kab,kabv->uv", r, c.astype(r.dtype), jnp.conj(r))

    def per_q(kmq):
        tk = lambda a: jnp.take(a, kmq, axis=0)
        r1 = jnp.einsum("kasu,kbsu->kabu", psi_A, jnp.conj(tk(psi_all)))
        c1, n1 = coeff(e_A[:, :, None], tk(e_all)[:, None, :], f_A[:, :, None],
                       tk(f_all)[:, None, :], tk(live)[:, None, :] > 0)
        r2 = jnp.einsum("kasu,kbsu->kabu", psi_all, jnp.conj(tk(psi_A)))
        c2, n2 = coeff(e_all[:, :, None], tk(e_A)[:, None, :], f_all[:, :, None],
                       tk(f_A)[:, None, :], frozen[:, :, None] > 0)
        return (pair_chi(r1, c1) + pair_chi(r2, c2)) * (gs / nk), n1 + n2

    chis, bads = lax.map(per_q, jnp.asarray(kern.kmq))
    return lax.with_sharding_constraint(chis, NamedSharding(mesh, CHI_SPEC)), jnp.sum(bads)


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
               same_side: bool = True):
    """dchi0 of Eq. (4) for the instantaneous window state (U, e_A, f_A), (nq, mu, nu), Ry^-1.

    G_eq(tau) of every band (G_frozen + G_active(0)) is composed at each node and
    never stored; the active tiles carry the rotated orbitals psi_A U.  The
    regrouping G_eq = G_f + G_a(0) makes Eq. (4) free of cancellation: no term is
    a difference of two equilibrium-sized products.  ``pauli=False`` /
    ``same_side=False`` drop the inverted product / the same-side pairs (red twins).
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

    acc, _ = lax.scan(body, _chi_zero(mesh, nk, dyn["nmu_pad"]), (dyn["tau"], dyn["alpha"]),
                      unroll=1)
    dchi = kern.c_fft((-2.0 * dyn["g_s"] / np.sqrt(nk)) * jnp.real(acc).astype(jnp.complex128))
    crossed, out = _gate_flags(dyn, e_A, x_sep)
    bad = jnp.zeros((), jnp.int32)
    if same_side:
        chi_ss, bad = _same_side_chi(dyn, kern, psiA_X, e_A, f_A, x_sep)
        dchi = dchi + chi_ss
    return dchi, dict(crossed=crossed, out_of_rule=out, degenerate=bad.astype(jnp.int32))


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

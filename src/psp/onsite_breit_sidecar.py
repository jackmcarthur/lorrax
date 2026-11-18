r"""psp/onsite_breit_sidecar.py -- write ``<El>.onsite_breit.json`` (``lorrax.onsite_breit_sidecar/v1``).

The generator of the sidecar that ``psp.onsite_breit.read_sidecar`` reads
(``onsite_breit = true``).  Physics library (L1) beside ``psp.onsite_breit``;
theory, conventions and file format: ``docs/dev/onsite_breit.md``.

Run::

    python -m psp.onsite_breit_sidecar --upf Cr.upf --shell 3D --ae Cr_3d_ae.dat [-o Cr.onsite_breit.json]
    python -m psp.onsite_breit_sidecar --upf Cr.upf --shell 3D --ae-oncvpsp Cr.out --ae-n 3
    python -m psp.onsite_breit_sidecar --upf Cr.upf --shell 3D --null      (AE block := PS block)

Inputs.  A fully relativistic norm-conserving UPF (PP_PSWFC with
PP_SPIN_ORB) and the all-electron radial function of the shell, either a
two-column table ``r [bohr], u(r) = r R(r)`` (any normalisation and sign;
``#`` comments) or the plot blocks of a fully relativistic ONCVPSP output
file (j-averaged here).  The pseudo function and the projector come from the
UPF; ``--null`` needs no all-electron input.

PUBLIC API
----------
``radial_integral(r, f, *, cumulative=False)``
    ``int f dr`` (or its running value), cubic-spline quadrature in the grid index.
``radial_moments(r, u) -> dict``
    ``norm``, ``r3`` = <r^-3>, Marvin ``I0``, ``I2`` of ``u`` (not renormalised).
``prepare_radial(r, u) -> (r, u, tail_kappa)``
    tail continuation (grid ending before ``u`` decays) and normalisation.
``shell_tensor(r, u, l) -> Lam (3, n, n, n, n) complex128``
    ``Lam[c, a, a2, b, d] = <a(1) a2(2)| (r12 x p1)_c / r12^3 |b(1) d(2)>`` for ``phi_m = (u/r) Y_lm``.
``tensor_block(r, u, l, reference) -> dict``
    one sidecar tensor block (``lam_re``, ``lam_im``, ``I0``, ``r3``, ``I2``, ``b1_mev_per_muB``,
    ``x_closed_mev``) after the closed-shell, Hermiticity and classical-limit checks.
``upf_shell(upf_path, label) -> dict``
    element, l, mesh, the projector chi (plain j-average) and the PS function (aligned j-average).
``javerage(r, u_hi, u_lo, l) -> u``
    ``[(l+1) u_{l+1/2} + l u_{l-1/2}]/(2l+1)`` after giving ``u_lo`` positive overlap with ``u_hi``.
``read_radial_table(path) -> (r, u)``;  ``read_oncvpsp_ae(path, n, l) -> (r, u)``
``make_sidecar(upf_path, label, *, ae=None, null=False, ...) -> dict``;  ``write_sidecar(side, path)``

THE TENSOR (docs/dev/onsite_breit.md section 7)
-----------------------------------------------
``(r12 x p1)/r12^3 = i (grad_1 V) x grad_1`` acting on electron 1, with
``V(r1) = int rho_{a2 d}(r2) / r12``, ``rho_{a2 d} = conj(phi_a2) phi_d``.
Expand ``conj(Y_a2) Y_d = sum_LM c^LM_{a2 d} Y_LM``; then
``V = sum_LM c^LM v_L(r) Y_LM`` with::

    v_L(r)  = 4 pi/(2L+1) [ r^-(L+1) P_L(r) + r^L Q_L(r) ],   P_L = int_0^r u^2 s^L ds,  Q_L = int_r^inf u^2 s^-(L+1) ds
    v_L'(r) = 4 pi/(2L+1) [ -(L+1) r^-(L+2) P_L(r) + L r^(L-1) Q_L(r) ]

and, with ``grad = r^ d_r + (1/r) grad_Omega``, ``grad_Omega Y = -i r^ x (L Y)``::

    Lam^c[a, a2, b, d] = i sum_LM c^LM_{a2 d} ( A_L X1^c + B_L X2^c + C_L X3^c )[a, LM, b]
    A_L = int u^2 v_L' / r dr,   C_L = int u^2 v_L / r^2 dr,   B_L = int r R R' v_L dr = -(A_L + C_L)/2
    X1^c = int conj(Y_a) Y_LM (r^ x grad_Omega Y_b)_c   = int conj(Y_a) Y_LM  i (L_c Y_b)
    X2^c = int conj(Y_a) Y_b (grad_Omega Y_LM x r^)_c    = int conj(Y_a) Y_b (-i) (L_c Y_LM)
    X3^c = int conj(Y_a) (grad_Omega Y_LM x grad_Omega Y_b)_c

``B_L`` is integrated by parts (no derivative of the tabulated u); the
angular integrals are exact (Gauss-Legendre x uniform-phi rule above the
polynomial degree 4l+2).  The L = 0 term gives ``A_0 = -4 pi I0``: the l.m
coefficient of d1 per unit (N_up - N_dn) equals ``-(alpha^2/2) I0`` to
rounding, which the generator checks.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np

from psp import onsite_breit as ob

GENERATOR = "psp.onsite_breit_sidecar"
TAIL_R_OUT = 40.0          # bohr: a function not decayed at the grid end is continued to here
TAIL_DECAYED = 1.0e-7      # |u(r_end)| below this fraction of max|u|: no continuation
CHECK_RTOL = 1.0e-10       # closed-shell identity, Hermiticity (the reader's CLOSED_SHELL_RTOL)
CLASSICAL_RTOL = 1.0e-8    # b1 from the tensor against -(alpha^2/2) I0 (same quadrature)

CONVENTIONS = (
    "Shell basis |a> = phi_{I mu}(r) x |sigma> on atom I: complex Y_lm (Condon-Shortley), m = -l..l, pure spinors, "
    "a = sigma*(2l+1) + (m+l), sigma = 0 the first spinor component of the WFN (s_z = +1/2 along Cartesian z of the "
    "cell). Tensor Lam[c, a, a2, b, d] = <a(1) a2(2)| Lam^c |b(1) d(2)> over the 2l+1 spatial orbitals only, "
    "Lam(1;2) = (r12 x p1)/r12^3, c = x, y, z; stored as lam_re / lam_im = flat lists in C (row-major) order of an "
    f"array of shape [3, n, n, n, n], n = 2l+1. Hartree atomic units, WITHOUT the alpha^2 prefactor (alpha = "
    f"{ob.ALPHA!r}). Spin-orbital two-body tensor g[p,q,r,s] = -alpha^2 sum_c Lam[c, a_p, a_q, a_r, a_s] "
    "delta(sigma_p, sigma_r) (sigma^c/2)[sigma_q, sigma_s] = <p(1) q(2)|g|r(1) s(2)>, H = sum_pqrs g[pqrs] "
    "a+_p a+_q a_s a_r. Density matrix D[a,b] = <a|rho|b> = sum_n f_n <a|psi_n><psi_n|b> (QE ns is its transpose). "
    "Mean field: d1[p,r] = sum_qs g[p,q,r,s] D[s,q]; d2[p,r] = sum_qs g[q,p,s,r] D[s,q]; "
    "x1[p,r] = -sum_qs g[p,q,s,r] D[s,q]; x2[p,r] = -sum_qs g[q,p,r,s] D[s,q]; h_dir = d1 + d2, h_x = x1 + x2 (Ha). "
    "I0 = int_0^inf dr2 u(r2)^2 r2^-3 int_0^r2 dr1 u(r1)^2 (Marvin), r3 = <r^-3>, I2 Marvin I^2, all of the "
    "normalised radial function u = rR by cubic-spline quadrature in the grid index. b1_mev_per_muB = -(alpha^2/2) I0 "
    "in meV: l.m coefficient of d1 per unit (N_up - N_dn) of a spherical collinear shell. x_closed_mev = c in "
    "x1 + x2 = c l.s for D = identity (meV). 1 Ha = 2 Ry, 1 Ry = common.units.RYD_TO_EV eV. projector.chi = r*R(r) "
    "on projector.r (bohr), the plain j-average [(l+1) chi_{l+1/2} + l chi_{l-1/2}]/(2l+1) of the UPF PP_CHI "
    "functions (psp/hubbard_ops.py::j_averaged_atomic_wfcs), no sign flip, no renormalisation. "
    "Generator and theory: psp.onsite_breit_sidecar, docs/dev/onsite_breit.md.")


# ---------------------------------------------------------------------------
# Radial quadrature
# ---------------------------------------------------------------------------

def radial_integral(r, f, *, cumulative: bool = False):
    """``int_{r_0}^{r_end} f dr`` (or its running value at every grid point) on any smooth monotone grid.

    The integrand is carried to the grid index i, ``int f dr = int f(r(i))
    J(i) di`` with ``J = dr/di`` the derivative of a cubic spline of r(i),
    and ``f J`` is integrated as a cubic spline in i (not-a-knot): fourth
    order in the spacing on uniform (UPF) and logarithmic (all-electron)
    grids alike, so the running integrals inside the Marvin and Slater-type
    double integrals are as accurate as the single ones.

    Parameters
    ----------
    r : (N,) float, strictly increasing.
    f : (N,) float, integrand on r.
    cumulative : bool
        Return the (N,) running integral from r_0 instead of the total.
    """
    from scipy.interpolate import CubicSpline
    r = np.asarray(r, dtype=np.float64)
    i = np.arange(len(r), dtype=np.float64)
    J = CubicSpline(i, r)(i, 1)
    F = CubicSpline(i, np.asarray(f, dtype=np.float64) * J).antiderivative()(i)
    return F if cumulative else float(F[-1])


def _pow(r, p):
    out = np.zeros_like(r)
    m = r > 0
    out[m] = r[m] ** p
    return out


def _marvin(r, u, k):
    """Marvin ``I^k = int dr2 int_0^r2 dr1 u1^2 u2^2 r1^k / r2^(k+3)``."""
    inner = radial_integral(r, u ** 2 * _pow(r, float(k)), cumulative=True)
    return radial_integral(r, inner * u ** 2 * _pow(r, -(k + 3.0)))


def radial_moments(r, u) -> dict:
    """``norm`` = int u^2, ``r3`` = int u^2 r^-3, Marvin ``I0`` and ``I2`` of u (not renormalised)."""
    r = np.asarray(r, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    return dict(norm=radial_integral(r, u ** 2), r3=radial_integral(r, u ** 2 * _pow(r, -3.0)),
                I0=_marvin(r, u, 0), I2=_marvin(r, u, 2))


def prepare_radial(r, u):
    """Strictly increasing grid, tail continuation, unit normalisation.

    A function that has not decayed at the grid end (``|u(r_end)| >=
    1e-7 max|u|``, e.g. a UPF mesh ending at ~10-16 bohr) is continued
    exponentially, ``u(r_end) exp(-kappa (r - r_end))``, kappa from a
    log-linear fit over the last bohr, at the last spacing out to 40 bohr.

    Returns
    -------
    (r, u, tail_kappa) : float64 arrays and the decay constant (0 if none).
    """
    r = np.asarray(r, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    if r.ndim != 1 or u.shape != r.shape or r.size < 16:
        raise ValueError(f"radial function: r {r.shape}, u {u.shape}; want two equal 1-d arrays (>= 16 points)")
    if not np.all(np.diff(r) > 0) or r[0] < 0:
        raise ValueError("radial function: r must be non-negative and strictly increasing")
    kappa = 0.0
    umax = float(np.abs(u).max())
    if umax == 0.0:
        raise ValueError("radial function is identically zero")
    if r[-1] < TAIL_R_OUT and abs(u[-1]) >= TAIL_DECAYED * umax:
        m = r > r[-1] - 1.0
        kappa = float(-np.polyfit(r[m], np.log(np.abs(u[m])), 1)[0])
        if kappa <= 0:
            raise ValueError(f"radial function is not decaying at the grid end r = {r[-1]:.3f} bohr")
        r_end, dr = r[-1], r[-1] - r[-2]
        rx = np.arange(r_end + dr, TAIL_R_OUT, dr)
        r = np.concatenate([r, rx])
        u = np.concatenate([u, u[-1] * np.exp(-kappa * (rx - r_end))])
    u = u / np.sqrt(radial_integral(r, u ** 2))
    return r, u, kappa


def javerage(r, u_hi, u_lo, l):
    """``[(l+1) u_hi + l u_lo]/(2l+1)``, u_lo first given positive overlap with u_hi (not normalised).

    ``u_hi`` is the j = l+1/2 function, ``u_lo`` the j = l-1/2 one (``None``
    for l = 0).  The sign alignment matters only for a file whose two
    partners carry opposite signs; the projector of ``psp.hubbard_ops``
    (``j_averaged_atomic_wfcs``) does not align, and the sidecar's
    ``projector.chi`` is that plain average.
    """
    u_hi = np.asarray(u_hi, dtype=np.float64)
    if u_lo is None or l == 0:
        return u_hi.copy()
    u_lo = np.asarray(u_lo, dtype=np.float64)
    if radial_integral(r, u_hi * u_lo) < 0:
        u_lo = -u_lo
    return ((l + 1) * u_hi + l * u_lo) / (2 * l + 1)


# ---------------------------------------------------------------------------
# The tensor
# ---------------------------------------------------------------------------

def _sphere_rule(l):
    """Gauss-Legendre in cos(theta) x uniform phi, exact above degree 4l+2 (points (N, 3), weights (N,))."""
    x, wx = np.polynomial.legendre.leggauss(2 * l + 4)
    nphi = 4 * l + 8
    ph = 2 * np.pi * np.arange(nphi) / nphi
    ct = np.repeat(x, nphi)
    st = np.sqrt(1.0 - ct ** 2)
    phr = np.tile(ph, len(x))
    pts = np.stack([st * np.cos(phr), st * np.sin(phr), ct], axis=1)
    return pts, np.repeat(wx, nphi) * (2 * np.pi / nphi)


def _angular_fields(L, pts):
    """``Y`` (N, 2L+1), ``iLY = i (L_c Y)`` (N, 2L+1, 3) and ``grad_Omega Y = -i r^ x (L Y)`` (N, 2L+1, 3)."""
    Y = ob.complex_ylm(L, pts)
    LY = np.stack([Y @ Lc for Lc in ob._lmats(L)], axis=-1)                   # (L_c Y_m) = sum_m' Y_m' (l_c)[m', m]
    grad = -1j * np.cross(pts[:, None, :], LY)
    return Y, 1j * LY, grad


def _radial_AC(r, u, L):
    """``A_L = int u^2 v_L'/r dr``, ``C_L = int u^2 v_L/r^2 dr`` (normalised u)."""
    u2 = u ** 2
    P = radial_integral(r, u2 * _pow(r, float(L)), cumulative=True)
    G = radial_integral(r, u2 * _pow(r, -(L + 1.0)), cumulative=True)
    Q = G[-1] - G
    pref = 4 * np.pi / (2 * L + 1)
    v = pref * (_pow(r, -(L + 1.0)) * P + _pow(r, float(L)) * Q)
    dv = pref * (-(L + 1) * _pow(r, -(L + 2.0)) * P + L * _pow(r, L - 1.0) * Q)
    return radial_integral(r, u2 * dv * _pow(r, -1.0)), radial_integral(r, u2 * v * _pow(r, -2.0))


def shell_tensor(r, u, l: int) -> np.ndarray:
    """Spin-other-orbit spatial tensor of one shell ``phi_m = (u/r) Y_lm`` (module docstring).

    Parameters
    ----------
    r, u : (N,) float
        Grid (bohr) and NORMALISED radial function u = r R(r) (:func:`prepare_radial`).
    l : int, 1..3

    Returns
    -------
    (3, n, n, n, n) complex128, n = 2l+1:
    ``Lam[c, a, a2, b, d] = <a(1) a2(2)| (r12 x p1)_c / r12^3 |b(1) d(2)>``,
    complex Condon-Shortley Y_lm, m = -l..l, Hartree atomic units, no alpha^2.
    """
    r = np.asarray(r, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    pts, w = _sphere_rule(l)
    Yl, iLYl, gl = _angular_fields(l, pts)
    n = 2 * l + 1
    lam = np.zeros((3, n, n, n, n), np.complex128)
    for L in range(0, 2 * l + 1, 2):                  # conj(Y_l) Y_l holds even L <= 2l only
        YL, iLYL, gL = _angular_fields(L, pts)
        c = np.einsum("N,NM,Nx,Nd->Mxd", w, YL.conj(), Yl.conj(), Yl)
        A, C = _radial_AC(r, u, L)
        B = -0.5 * (A + C)
        X1 = np.einsum("N,Na,NM,Nbc->caMb", w, Yl.conj(), YL, iLYl)
        X2 = np.einsum("N,Na,Nb,NMc->caMb", w, Yl.conj(), Yl, -iLYL)
        X3 = np.einsum("N,Na,NMbc->caMb", w, Yl.conj(), np.cross(gL[:, :, None, :], gl[:, None, :, :]))
        lam += 1j * np.einsum("Mxd,caMb->caxbd", c, A * X1 + B * X2 + C * X3)
    return lam


def _coef(op, M):
    return float((np.vdot(op, M) / np.vdot(op, op)).real)


def verify_tensor(lam, l: int, I0: float) -> dict:
    """Checks a tensor must pass before it is written (raises ``ValueError``).

    Closed shell D = 1: max|d1|, max|d2| <= 1e-10 max|x|, x1 + x2 = c l.s
    (relative residual <= 1e-10), c > 0 (the reader's gate); a random
    Hermitian D (``default_rng(7)``): d1, d2, x1 + x2 and their sum
    Hermitian to 1e-10; spherical collinear D = diag(0.9, 0.3): d1 = b L_z
    and b / (n 0.6) = -(alpha^2/2) I0 to 1e-8.  Returns the diagnostics
    (``x_closed_ha`` = c).
    """
    n = 2 * l + 1
    n2 = 2 * n
    ops = ob.shell_operators(l)
    h = ob.meanfield_pieces(lam, np.eye(n2))
    x = h["x1"] + h["x2"]
    scale = float(np.abs(x).max())
    dmax = max(float(np.abs(h["d1"]).max()), float(np.abs(h["d2"]).max())) / scale
    c = _coef(ops["ls"], x)
    res = float(np.linalg.norm(x - c * ops["ls"]) / np.linalg.norm(x))
    g = np.random.default_rng(7)
    A = g.normal(size=(n2, n2)) + 1j * g.normal(size=(n2, n2))
    hm = ob.meanfield_pieces(lam, (A + A.conj().T) / 2)
    herm = 0.0
    for M in (hm["d1"], hm["d2"], hm["x1"] + hm["x2"], sum(hm.values())):
        herm = max(herm, float(np.linalg.norm(M - M.conj().T) / np.linalg.norm(M)))
    Lz = ops["L"][2]
    d1 = ob.meanfield_pieces(lam, np.diag([0.9] * n + [0.3] * n))["d1"]
    b = _coef(Lz, d1)
    lz_res = float(np.linalg.norm(d1 - b * Lz) / np.linalg.norm(d1))
    b1_tensor = b / (n * 0.6)
    b1_formula = -ob.ALPHA ** 2 / 2 * I0
    cls = abs(b1_tensor / b1_formula - 1.0)
    out = dict(closed_shell_d_over_x=dmax, closed_shell_ls_resid=res, hermiticity_rel=herm, x_closed_ha=c,
               spherical_lz_resid=lz_res, b1_tensor_over_formula_minus_1=b1_tensor / b1_formula - 1.0)
    if not (dmax <= CHECK_RTOL and res <= CHECK_RTOL and herm <= CHECK_RTOL and c > 0
            and lz_res <= CHECK_RTOL and cls <= CLASSICAL_RTOL):
        raise ValueError(f"onsite_breit_sidecar: tensor verification failed (l = {l}): {out}")
    return out


def _flist(a):
    """Flat list of Python floats, C order (json writes repr: exact float64 round trip)."""
    return [float(x) for x in np.asarray(a, dtype=np.float64).ravel(order="C")]


def tensor_block(r, u, l: int, reference: str):
    """One sidecar tensor block of the radial function (r, u) and its check diagnostics.

    Parameters
    ----------
    r, u : (N,) float
        Grid (bohr) and radial function u = r R(r), any normalisation and sign
        (prepared by :func:`prepare_radial`).
    l : int, 1..3
    reference : str
        What the radial function is (written into the block).

    Returns
    -------
    (block, info) : the JSON-ready dict (``reference, shape, lam_re, lam_im,
    I0, r3, I2, b1_mev_per_muB, x_closed_mev``) and a dict with the checks,
    ``tail_kappa`` and the grid size.
    """
    if not 1 <= int(l) <= 3:
        raise ValueError(f"onsite_breit_sidecar: l = {l}; the on-site operator needs 1 <= l <= 3 "
                         "(an s shell has no orbital moment; the atomic rows stop at l = 3)")
    r, u, kappa = prepare_radial(r, u)
    mom = radial_moments(r, u)
    lam = shell_tensor(r, u, l)
    chk = verify_tensor(lam, l, mom["I0"])
    ha_to_mev = ob.HA_TO_RY * ob.RY_TO_MEV
    block = dict(reference=str(reference), shape=list(lam.shape), lam_re=_flist(lam.real), lam_im=_flist(lam.imag),
                 I0=mom["I0"], r3=mom["r3"], I2=mom["I2"],
                 b1_mev_per_muB=float(-ob.ALPHA ** 2 / 2 * mom["I0"] * ha_to_mev),
                 x_closed_mev=float(chk["x_closed_ha"] * ha_to_mev))
    return block, dict(chk, tail_kappa=kappa, grid_points=int(len(r)), r_max=float(r[-1]))


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def upf_shell(upf_path, label: str) -> dict:
    """The shell ``label`` of a UPF's PP_PSWFC.

    Returns
    -------
    dict
        ``element``, ``l``, ``r`` (the full PP_R mesh as read), ``chi`` (the
        projector: ``psp.hubbard_ops.j_averaged_atomic_wfcs``, the function
        ``psp.onsite_breit`` authenticates against), ``ps`` (the PS radial
        function: :func:`javerage` of the same two PP_CHI, sign-aligned),
        ``construction`` (which PP_CHI entered with which weight).
    """
    from psp.hubbard_ops import j_averaged_atomic_wfcs
    from psp.upf.load_upf import load_upf_pswfc, parse_upf_xml
    root = parse_upf_xml(upf_path)
    hdr = next((c for c in root if isinstance(c.tag, str) and c.tag.split("}")[-1] == "PP_HEADER"), None)
    element = (hdr.get("element") if hdr is not None else "") or ""
    element = element.strip()
    pswfc = load_upf_pswfc(upf_path)
    chis = pswfc["chi"]
    r = np.asarray(pswfc["r"], dtype=np.float64)
    if not any(c["j"] is not None for c in chis):
        raise ValueError(f"{upf_path}: no PP_SPIN_ORB j labels; the on-site Breit shell basis needs a fully "
                         "relativistic UPF (the projector is the j-average of the two PP_CHI partners)")
    hits = [w for w in j_averaged_atomic_wfcs(pswfc) if w["label"].upper() == label.upper()]
    if len(hits) != 1:
        raise ValueError(f"{upf_path}: {len(hits)} j-averaged PP_PSWFC functions labelled {label!r}; want exactly one "
                         f"(labels: {sorted({c['label'] for c in chis})})")
    l = int(hits[0]["l"])
    # the same pair hubbard_ops averages: c (j = l+1/2, this label), partner = first j = l-1/2 with this l
    hi = next(c for c in chis if c["label"].upper() == label.upper() and c["l"] == l
              and c["occupation"] >= 0.0 and abs(c["j"] - l + 0.5) >= 1e-4)
    lo = next((d for d in chis if d["l"] == l and abs(d["j"] - l + 0.5) < 1e-4), None) if l > 0 else None
    chi = np.asarray(hits[0]["chi"], dtype=np.float64)
    nm = min(len(chi), len(r))
    if lo is None:
        how = f"PP_CHI.{hi['index']} (j={hi['j']})"
    else:
        how = (f"PP_CHI.{hi['index']} (j={hi['j']}) x {l + 1} + PP_CHI.{lo['index']} (j={lo['j']}) x {l}, "
               f"/ {2 * l + 1}")
    ps = javerage(r[:nm], hi["chi"][:nm], None if lo is None else lo["chi"][:nm], l)
    return dict(element=element, label=hi["label"], l=l, r=r[:nm], chi=chi[:nm], ps=ps, construction=how)


def read_radial_table(path):
    """Two columns ``r [bohr]  u(r) = r R(r)`` (whitespace, ``#`` comments; extra columns ignored)."""
    a = np.loadtxt(path, comments="#", usecols=(0, 1), ndmin=2)
    return a[:, 0].copy(), a[:, 1].copy()


_ONCV_HEAD = re.compile(r"^\s*n=\s*(\d+),\s*l=\s*(\d+)\s+kap=\s*(-?\d+),\s*all-electron wave function")


def read_oncvpsp_ae(path, n: int, l: int):
    """j-averaged all-electron function of shell (n, l) from a fully relativistic ONCVPSP output file.

    ONCVPSP prints, for each valence state of the reference configuration, a
    block headed ``n= 3,  l= 2  kap=-3, all-electron wave function, pseudo
    w-f`` followed by lines ``&  <tag>  r  u_AE  u_PS`` (u = r R, the large
    component).  Both kappa blocks of (n, l) are read (kappa = -(l+1): j =
    l+1/2; kappa = l: j = l-1/2) and averaged with :func:`javerage`.  The
    first block per kappa is used (later blocks belong to test configurations).
    The plot ends at a few bohr; :func:`prepare_radial` continues the tail.

    Returns
    -------
    (r, u) : float64 arrays (u not normalised).
    """
    blocks = {}
    cur = None
    with open(path) as f:
        for line in f:
            m = _ONCV_HEAD.match(line)
            if m:
                key = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
                cur = key if key not in blocks else None
                if cur is not None:
                    blocks[cur] = []
                continue
            if cur is not None:
                s = line.split()
                if s and s[0] == "&" and len(s) >= 4:
                    blocks[cur].append((float(s[2]), float(s[3])))
                elif blocks[cur]:
                    cur = None
    hi = blocks.get((n, l, -(l + 1)))
    lo = blocks.get((n, l, l)) if l > 0 else None
    if not hi or (l > 0 and not lo):
        raise ValueError(f"{path}: no all-electron plot blocks for n={n} l={l} "
                         f"(found {sorted(blocks)}); a fully relativistic ONCVPSP output is needed")
    hi, lo = np.asarray(hi), (None if lo is None else np.asarray(lo))
    if lo is not None and (len(lo) != len(hi) or np.abs(lo[:, 0] - hi[:, 0]).max() > 0):
        raise ValueError(f"{path}: the two kappa blocks of n={n} l={l} are on different meshes")
    r = hi[:, 0]
    return r, javerage(r, hi[:, 1], None if lo is None else lo[:, 1], l)


# ---------------------------------------------------------------------------
# The sidecar
# ---------------------------------------------------------------------------

def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _md5(path) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def make_sidecar(upf_path, label: str, *, ae=None, ae_reference: str = "", ae_source: str = "",
                 null: bool = False, date: str | None = None) -> dict:
    """Build one ``lorrax.onsite_breit_sidecar/v1`` sidecar (one shell).

    Parameters
    ----------
    upf_path : str
        Fully relativistic norm-conserving UPF of the species (the one the run uses).
    label : str
        PP_CHI label of the shell (``3D``, ``3P``; case-insensitive).
    ae : (r, u) or None
        All-electron radial function of the shell (u = r R, any normalisation),
        e.g. :func:`read_radial_table` or :func:`read_oncvpsp_ae`.  Not needed
        with ``null``.
    ae_reference, ae_source : str
        Written into ``ae.reference`` / ``provenance.ae_source``.
    null : bool
        Write the null sidecar: the AE block is the PS block, so
        ``h_AE - h_PS = 0`` in every ``paw`` combination (a bit-for-bit gate
        against the run without the term).
    date : str, optional
        ISO date for the provenance (default today, UTC).

    Returns
    -------
    dict, ready for :func:`write_sidecar`.
    """
    sh = upf_shell(upf_path, label)
    l = sh["l"]
    ps, ps_info = tensor_block(sh["r"], sh["ps"], l, (
        f"PP_PSWFC {sh['label']} of {os.path.basename(str(upf_path))}: {sh['construction']}, the j=l-1/2 "
        "partner given positive overlap with j=l+1/2, normalised (projector.chi is the plain unnormalised average)"))
    checks = {"ps": ps_info}
    if null:
        ae = copy.deepcopy(ps)
        ae["reference"] = "NULL GATE: AE block replaced by the PS block (h_AE - h_PS = 0 exactly); " + ps["reference"]
        checks["ae"] = ps_info
        ae_src = None
    else:
        if ae is None:
            raise ValueError("make_sidecar: an all-electron radial function (ae=(r, u)) is needed unless null=True")
        ae, checks["ae"] = tensor_block(ae[0], ae[1], l, ae_reference or "all-electron radial function (user table)")
        ae_src = ae_source or None
    shell = dict(label=sh["label"], l=l,
                 projector=dict(type="ortho-atomic",
                                function="PP_PSWFC j-averaged [(l+1)chi_{l+1/2} + l chi_{l-1/2}]/(2l+1)",
                                construction=sh["construction"], mesh_size=int(len(sh["r"])),
                                r=_flist(sh["r"]), chi=_flist(sh["chi"])),
                 ae=ae, ps=ps)
    prov = dict(generator=GENERATOR, date=date or datetime.now(timezone.utc).date().isoformat(),
                code_sha256={"psp/onsite_breit_sidecar.py": _sha256(os.path.abspath(__file__))},
                source_upf=dict(path=os.path.basename(str(upf_path)), md5=_md5(upf_path), sha256=_sha256(upf_path)),
                ae_source=ae_src, quadrature="cubic spline in the grid index (psp.onsite_breit_sidecar.radial_integral)",
                checks=checks)
    if null:
        prov["null_gate"] = f"{sh['element']}: ae := ps"
    return dict(format=ob.SIDECAR_FORMAT, element=sh["element"], conventions=CONVENTIONS, shells=[shell],
                provenance=prov)


def write_sidecar(side: dict, path) -> str:
    """Write the sidecar (JSON, float repr, LF) and re-read it with ``psp.onsite_breit.read_sidecar``; returns sha256."""
    with open(path, "w", newline="\n") as f:
        json.dump(side, f, indent=1, allow_nan=False)
        f.write("\n")
    ob.read_sidecar(path)
    return _sha256(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m psp.onsite_breit_sidecar",
                                 description="Write <El>.onsite_breit.json (lorrax.onsite_breit_sidecar/v1); "
                                             "docs/dev/onsite_breit.md.")
    ap.add_argument("--upf", required=True, help="fully relativistic norm-conserving UPF of the species")
    ap.add_argument("--shell", required=True, help="PP_CHI label of the open shell (e.g. 3D)")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--ae", help="all-electron radial table: columns r [bohr], u = r R(r)")
    src.add_argument("--ae-oncvpsp", help="fully relativistic ONCVPSP output file (plot blocks, j-averaged)")
    ap.add_argument("--ae-n", type=int, help="principal quantum number of the shell (with --ae-oncvpsp)")
    ap.add_argument("--ae-reference", default="", help="text for ae.reference (configuration, source)")
    ap.add_argument("--null", action="store_true", help="null sidecar: AE block := PS block")
    ap.add_argument("-o", "--out", help="output path (default <El>.onsite_breit.json, <El>.null.onsite_breit.json)")
    a = ap.parse_args(argv)
    if a.null == bool(a.ae or a.ae_oncvpsp):
        ap.error("give exactly one of --ae, --ae-oncvpsp, --null")
    ae, ae_source = None, ""
    if a.ae:
        ae, ae_source = read_radial_table(a.ae), f"{os.path.basename(a.ae)} sha256 {_sha256(a.ae)}"
    elif a.ae_oncvpsp:
        if a.ae_n is None:
            ap.error("--ae-oncvpsp needs --ae-n")
        l = upf_shell(a.upf, a.shell)["l"]
        ae = read_oncvpsp_ae(a.ae_oncvpsp, a.ae_n, l)
        ae_source = (f"{os.path.basename(a.ae_oncvpsp)} n={a.ae_n} l={l} plot blocks, j-averaged, "
                     f"sha256 {_sha256(a.ae_oncvpsp)}")
    side = make_sidecar(a.upf, a.shell, ae=ae, ae_reference=a.ae_reference, ae_source=ae_source, null=a.null)
    out = a.out or f"{side['element']}{'.null' if a.null else ''}.onsite_breit.json"
    sha = write_sidecar(side, out)
    sh = side["shells"][0]
    for kind in ("ae", "ps"):
        b = sh[kind]
        print(f"{side['element']} {sh['label']} {kind}: I0 = {b['I0']:.6f}, <r^-3> = {b['r3']:.5f}, "
              f"b1 = {b['b1_mev_per_muB']:+.5f} meV/muB, x_closed = {b['x_closed_mev']:+.5f} meV")
    print(f"wrote {out} (sha256 {sha})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

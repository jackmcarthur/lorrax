r"""psp/onsite_breit.py -- on-site Breit (spin-other-orbit) mean field of one open shell per atom.

Physics library (L1) beside ``psp.hubbard_ops``; design note ``ONSITE_INCODE.md``
(breit salvage, 2026-11-11), sections 2, 3 and 6.  The assembly for the drivers
(``gw.hartree.onsite_breit_field``) and the deck keys live elsewhere; this module
holds the sidecar reader, the shell algebra, the setup and the band-sized kernels.

PUBLIC API
----------
``read_sidecar(path, *, require_ps=False) -> dict``
    parse and verify ``<El>.onsite_breit.json`` (``lorrax.onsite_breit_sidecar/v1``).
``resolve_onsite_keys(double_counting, exchange, *, bispinor=None, tt_exchange=None) -> (str, str)``
    the one check of the two keys of section 3 (values and pairings).
``shell_operators(l) -> dict``
    ``L`` (3), ``S`` (3), ``ls``, ``I`` in the shell basis (complex CS Y_lm, spin-major).
``real_to_complex_ylm(l) -> W``
    ``Y^c_mu = sum_i W[i, mu] Y^r_i`` (QE real Y_lm of ``psp.radial.solid_harmonics``).
``meanfield_pieces(lam, D) -> dict``
    ``d1, d2, x1, x2`` (Hartree) of one tensor for one density matrix.
``meanfield_kernels(lam) -> (K_dir, K_x)``
    precontracted ``h[p,r] = sum_{s,q} K[p,r,s,q] D[s,q]`` (Hartree, alpha^2 included).
``shell_kernels(sidecar_shell) -> dict``
    ``{"l", "label", "ae": {"dir", "x"}, "ps": {...} | None}`` for one sidecar shell.
``onsite_potential(D, kernels, *, double_counting, exchange) -> dict``
    pieces ``dir_ae, dir_ps, x_ae, x_ps`` and ``total`` (Ry) by the tables of section 3.
``onsite_block_potential(D, setup) -> (h_block, pieces_list)``
    the block-diagonal (R_os, R_os) potential (Ry) of every shell-atom of a setup.
``shell_moments(D, l) -> dict``
    ``N`` = Tr D, ``S`` = <S>, ``L`` = <L> (3-vectors), ``herm`` = max|D - D^dagger|.
``decompose_potential(h, l, mhat) -> dict``
    coefficients of 1, L.m, S.m, (L.m)(S.m), l.s and the remainder norm.
``time_reversal(l) -> U_T``;  ``rotation_matrix(l, R) -> M``
    Theta = U_T K;  M = U_spin(R) (x) D^l(R) (tests and checks).
``build_onsite_setup(wfn, upf_paths, sidecars, *, projector, double_counting, exchange) -> OnsiteSetup``
    projector rows, on-site row map, kernels, keys, hashes and provenance.
``onsite_projections(psi_G, gvecs, gmask, kvecs, setup, *, mesh) -> P (nk, R_os, nb)``
    ``P[k, a, n] = <phi~_a(k)|psi_nk>`` in the shell basis, replicated.
``onsite_density_matrix(P, kweights, occ, setup, *, U=None, mesh) -> D (R_os, R_os)``
    ``D = sum_k w_k (P U) f (P U)^dagger`` (host complex128).
``onsite_band_matrix(P, h_block, *, mesh) -> dH (nk, nb, nb)``
    ``dH[k] = P[k]^dagger h P[k]`` (units of h, Ry), replicated.
``onsite_report_lines(D, pieces, setup) -> list[str]``
    per shell-atom log lines (Tr D, 2<S>, <L>, max|D - D^dagger|, coefficients in meV).

CONVENTIONS (design note section 2, in order)
---------------------------------------------
Units.  Hartree atomic units inside the tensors, ``alpha = 1/c`` taken from
``common.bispinor_init.ALPHA_FS`` (the repo's one fine-structure constant,
``2 * HALFALPHA = 0.0072973525643``); Rydberg for every potential and band
matrix returned here (x2); meV only in the log lines (``1e3 * hubbard_ops.RYTOEV``).

Shell basis.  ``|a> = phi~_{I mu}(r) (x) |sigma>`` on atom I, complex Y_lm
(Condon-Shortley), m = -l..l, pure spinors::

    a = sigma (2l+1) + (m + l),     sigma = 0: first spinor component (s_z = +1/2 along Cartesian z)

Projector (key ``onsite_breit_projector``; registry ``_PROJECTORS``).  Tonight
one type, ``ortho-atomic``: the j-averaged PP_PSWFC functions, Loewdin-
orthogonalised at each k over ALL atomic functions of all atoms (QE
ortho-atomic, ``hubbard_ops.build_atwfc_setup`` + ``lowdin_rows_with_derivative``).
The rows are held in QE real Y_lm; the shell basis is reached per spin block by::

    P^c = W^dagger P^r,     Y^c_mu = sum_i W[i, mu] Y^r_i      (W numerical, unitary)

Projection table, density matrix, potential, band matrix::

    P[k, I, a, n]  = <phi~_{I a}(k) | psi_nk>
    D_I[a, b]      = sum_k w_k sum_i f_ik (P U)_{a i} conj((P U)_{b i})     (U = 1 at DFT level)
    h_I            = h^dir[D_I] + h^x[D_I]                                   (section 3 combination)
    H_OS[k, n, n'] = sum_I sum_ab conj(P[k,I,a,n]) h_I[a,b] P[k,I,b,n']      (Ry)

``D[a, b] = <a|rho|b>`` (QE's ``ns`` is its transpose).

Mean field.  Sidecar tensor ``Lam[c, a, a2, b, d] = <a(1) a2(2)|Lam^c|b(1) d(2)>``,
``Lam(1;2) = (r12 x p1)/r12^3``, spatial indices only, c = x, y, z::

    g[p,q,r,s] = -alpha^2 sum_c Lam^c[a_p, a_q, a_r, a_s] delta(s_p, s_r) (sigma^c/2)[s_q, s_s]
    d1[p,r] =  sum_qs g[p,q,r,s] D[s,q]      orbit of the test electron in the shell's spin field
    d2[p,r] =  sum_qs g[q,p,s,r] D[s,q]      spin of the test electron in the shell's orbital current
    x1[p,r] = -sum_qs g[p,q,s,r] D[s,q]
    x2[p,r] = -sum_qs g[q,p,r,s] D[s,q]
    h^dir = d1 + d2,    h^x = x1 + x2

Limits (unit tests): closed shell D = 1 -> d1 = d2 = 0, h^x = c l.s with c > 0;
spherical collinear shell -> d1 = b L.m, b = -(alpha^2/2)(N_up - N_dn) I0; one
electron -> h_aa = 0; h[Theta D Theta^-1] = Theta h[D] Theta^-1; h[M D M^+] = M h[D] M^+.

Double counting (section 3, two keys, no defaults)::

    double_counting = onsite_only : direct part = h^dir_AE              (bispinor = false)
                      paw         : direct part = h^dir_AE - h^dir_PS   (bispinor = true)
    exchange        = full        : exchange    = h^x_AE                (no TT exchange in H)
                      paw         : exchange    = h^x_AE - h^x_PS       (Sigma has TT exchange)
                      off         : exchange    = 0

Not included: i[r, H_OS] in the velocity; on-site spin-spin / orbit-orbit;
exchange with other shells; ultrasoft/PAW; nspinor = 1 (all refused or stated).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import factorial

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from common.bispinor_init import ALPHA_FS
from psp.hubbard_ops import RYTOEV

ALPHA = float(ALPHA_FS)
HA_TO_RY = 2.0
RY_TO_MEV = 1.0e3 * RYTOEV
SIDECAR_FORMAT = "lorrax.onsite_breit_sidecar/v1"
PROVENANCE_SCHEME = "lorrax.onsite_breit/v1"
DOUBLE_COUNTING_VALUES = ("onsite_only", "paw")
EXCHANGE_VALUES = ("full", "paw", "off")
CHI_RTOL = 1.0e-8            # section 6: max|chi_upf - chi_sidecar| <= 1e-8 max|chi_upf|
CLOSED_SHELL_RTOL = 1.0e-10  # section 6: d1 = d2 = 0, h^x proportional to l.s
_SIGMA = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=np.complex128)


def _gate(rule, got, want, why, fix):
    return ValueError(f"GATE {rule}: {why.split(';')[0]}\n  got:  {got}\n  want: {want}\n"
                      f"  why:  {why}\n  fix:  {fix}\n  doc:  src/psp/onsite_breit.py")


# ---------------------------------------------------------------------------
# Shell algebra (host numpy, 2(2l+1)-dimensional)
# ---------------------------------------------------------------------------

def _lmats(l: int):
    """(Lx, Ly, Lz) on complex Y_lm, m = -l..l, Condon-Shortley phases (hbar = 1)."""
    m = np.arange(-l, l + 1)
    Lp = np.zeros((2 * l + 1, 2 * l + 1), np.complex128)
    for i in range(2 * l):
        Lp[i + 1, i] = np.sqrt(l * (l + 1) - m[i] * (m[i] + 1))
    Lm = Lp.conj().T
    return 0.5 * (Lp + Lm), -0.5j * (Lp - Lm), np.diag(m).astype(np.complex128)


def shell_operators(l: int) -> dict:
    """Angular-momentum operators of one shell in the shell basis.

    Parameters
    ----------
    l : int
        Orbital angular momentum of the shell.

    Returns
    -------
    dict
        ``L`` and ``S``: lists of three (2(2l+1), 2(2l+1)) complex128 matrices
        (x, y, z), ``ls`` = L.S, ``I`` the identity.  Basis
        ``a = sigma (2l+1) + (m + l)``: ``L = 1_2 (x) l``, ``S = (sigma/2) (x) 1``.
    """
    Lc = _lmats(l)
    n = 2 * l + 1
    L = [np.kron(np.eye(2), x) for x in Lc]
    S = [np.kron(0.5 * s, np.eye(n)) for s in _SIGMA]
    return dict(L=L, S=S, ls=sum(a @ b for a, b in zip(L, S)), I=np.eye(2 * n, dtype=np.complex128))


def _assoc_legendre(l, m, x):
    """P_l^m(x), m >= 0, with the Condon-Shortley phase."""
    pmm = np.ones_like(x)
    if m > 0:
        s = np.sqrt(np.maximum(1.0 - x * x, 0.0))
        fact = 1.0
        for _ in range(m):
            pmm = -pmm * fact * s
            fact += 2.0
    if l == m:
        return pmm
    pm1 = x * (2 * m + 1) * pmm
    if l == m + 1:
        return pm1
    for ll in range(m + 2, l + 1):
        pll = ((2 * ll - 1) * x * pm1 - (ll + m - 1) * pmm) / (ll - m)
        pmm, pm1 = pm1, pll
    return pm1


def complex_ylm(l: int, xyz: np.ndarray) -> np.ndarray:
    """Complex Y_lm (Condon-Shortley) at directions ``xyz`` (N, 3): (N, 2l+1), columns m = -l..l."""
    xyz = np.atleast_2d(np.asarray(xyz, dtype=np.float64))
    r = np.linalg.norm(xyz, axis=1)
    ct = np.clip(xyz[:, 2] / np.where(r > 0, r, 1.0), -1.0, 1.0)
    ph = np.arctan2(xyz[:, 1], xyz[:, 0])
    out = np.empty((len(xyz), 2 * l + 1), np.complex128)
    for m in range(l + 1):
        nrm = np.sqrt((2 * l + 1) / (4 * np.pi) * factorial(l - m) / factorial(l + m))
        Y = nrm * _assoc_legendre(l, m, ct) * np.exp(1j * m * ph)
        out[:, l + m] = Y
        if m:
            out[:, l - m] = (-1) ** m * np.conj(Y)
    return out


def _unit_vectors(n: int, seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1)[:, None]


def real_to_complex_ylm(l: int) -> np.ndarray:
    """W with ``Y^c_mu = sum_i W[i, mu] Y^r_i``.

    ``Y^r`` are the QE real harmonics of ``psp.radial.solid_harmonics``
    (row order of the atomic rows), ``Y^c`` the complex CS harmonics of
    :func:`complex_ylm`.  Fitted by least squares on random unit vectors,
    so neither ordering nor phase of either set is assumed; the residual and
    the unitarity of W are asserted.

    Returns
    -------
    ndarray, (2l+1, 2l+1) complex128, unitary.
    """
    from psp.radial.solid_harmonics import solid_harmonics_jax
    pts = _unit_vectors(16 * (2 * l + 1), seed=20261111 + l)
    Yr = np.asarray(solid_harmonics_jax(l, jnp.asarray(pts))).T.astype(np.complex128)  # (N, n)
    Yc = complex_ylm(l, pts)
    W, *_ = np.linalg.lstsq(Yr, Yc, rcond=None)
    res = float(np.abs(Yr @ W - Yc).max())
    uni = float(np.abs(W.conj().T @ W - np.eye(2 * l + 1)).max())
    if res > 1e-12 or uni > 1e-12:
        raise AssertionError(f"real_to_complex_ylm(l={l}): residual {res:.1e}, unitarity defect {uni:.1e}")
    return W


def time_reversal(l: int) -> np.ndarray:
    """Unitary part U_T of ``Theta = U_T K`` in the shell basis (K: complex conjugation).

    ``Theta = i sigma_y (x) P``, ``P[m', m] = (-1)^m delta(m', -m)``;
    ``Theta D Theta^-1 = U_T conj(D) U_T^dagger``.
    """
    n = 2 * l + 1
    Pm = np.zeros((n, n))
    for i, m in enumerate(range(-l, l + 1)):
        Pm[n - 1 - i, i] = (-1) ** m
    return np.kron(1j * _SIGMA[1], Pm)


def _spin_rotation(R: np.ndarray) -> np.ndarray:
    """U = exp(-i theta n.sigma/2) of the proper part det(R) R; checked U^+ sigma_i U = sum_j R'_ij sigma_j."""
    Rp = np.linalg.det(R) * np.asarray(R, dtype=np.float64)
    c = np.clip((np.trace(Rp) - 1.0) / 2.0, -1.0, 1.0)
    th = float(np.arccos(c))
    if th < 1e-12:
        U = np.eye(2, dtype=np.complex128)
    else:
        if abs(np.pi - th) < 1e-6:
            w, v = np.linalg.eig(Rp)
            nv = np.real(v[:, np.argmin(np.abs(w - 1.0))])
        else:
            nv = np.array([Rp[2, 1] - Rp[1, 2], Rp[0, 2] - Rp[2, 0], Rp[1, 0] - Rp[0, 1]]) / (2 * np.sin(th))
        nv = nv / np.linalg.norm(nv)
        U = np.cos(th / 2) * np.eye(2) - 1j * np.sin(th / 2) * np.einsum("k,kab->ab", nv, _SIGMA)
    for i in range(3):
        if np.abs(U.conj().T @ _SIGMA[i] @ U - np.einsum("j,jab->ab", Rp[i], _SIGMA)).max() > 1e-8:
            raise AssertionError("spin rotation convention check failed")
    return U


def rotation_matrix(l: int, R: np.ndarray) -> np.ndarray:
    """Shell-basis matrix ``M = U_spin(R) (x) D^l(R)`` of the active rotation R (3x3 Cartesian).

    ``D^l[m', m]`` from ``Y_lm(R^-1 r) = sum_m' D^l[m', m] Y_lm'(r)`` (least
    squares on random unit vectors); ``U_spin = exp(-i theta n.sigma/2)`` of
    the proper part.  A density matrix or operator transforms as ``M X M^+``.
    """
    R = np.asarray(R, dtype=np.float64)
    pts = _unit_vectors(64, seed=12345)
    A = complex_ylm(l, pts)
    B = complex_ylm(l, pts @ np.linalg.inv(R).T)
    Dl, *_ = np.linalg.lstsq(A, B, rcond=None)
    if np.abs(A @ Dl - B).max() > 1e-10:
        raise AssertionError("rotation_matrix: D^l fit residual")
    return np.kron(_spin_rotation(R), Dl)


# ---------------------------------------------------------------------------
# Mean field (ONSITE_BREIT section 1.3, onsite_ops.g_soo / meanfield)
# ---------------------------------------------------------------------------

def _g_soo(lam: np.ndarray) -> np.ndarray:
    """g[p,q,r,s] = -alpha^2 sum_c Lam^c[a_p,a_q,a_r,a_s] delta(s_p,s_r) (sigma^c/2)[s_q,s_s]  (Hartree)."""
    lam = np.asarray(lam, dtype=np.complex128)
    n = lam.shape[1]
    g = -ALPHA ** 2 * np.einsum("cAQRS,pr,cqs->pAqQrRsS", lam, np.eye(2), _SIGMA / 2, optimize=True)
    return g.reshape(2 * n, 2 * n, 2 * n, 2 * n)


def _piece_kernels(lam: np.ndarray) -> dict:
    """K_piece[p,r,s,q] with piece[p,r] = sum_{s,q} K[p,r,s,q] D[s,q] (Hartree)."""
    g = _g_soo(lam)
    return dict(d1=np.einsum("pqrs->prsq", g), d2=np.einsum("qpsr->prsq", g),
                x1=-np.einsum("pqsr->prsq", g), x2=-np.einsum("qprs->prsq", g))


def meanfield_pieces(lam: np.ndarray, D: np.ndarray) -> dict:
    """The four mean-field pieces of one tensor for one shell density matrix.

    Parameters
    ----------
    lam : (3, n, n, n, n) complex
        ``Lam[c, a, a2, b, d]``, Hartree a.u., no alpha^2; n = 2l+1.
    D : (2n, 2n) complex
        Shell density matrix ``D[a, b] = <a|rho|b>``.

    Returns
    -------
    dict of (2n, 2n) complex128, Hartree: ``d1, d2, x1, x2`` (module docstring).
    """
    D = np.asarray(D, dtype=np.complex128)
    return {k: np.einsum("prsq,sq->pr", K, D) for k, K in _piece_kernels(lam).items()}


def meanfield_kernels(lam: np.ndarray):
    """Precontracted direct and exchange kernels of one tensor.

    ``h^dir[p, r] = sum_{s,q} K_dir[p,r,s,q] D[s,q]`` (= d1 + d2) and
    ``h^x[p, r] = sum_{s,q} K_x[p,r,s,q] D[s,q]`` (= x1 + x2), Hartree, alpha^2
    (``common.bispinor_init.ALPHA_FS``) included.

    Returns
    -------
    (K_dir, K_x) : two (2n, 2n, 2n, 2n) complex128 arrays.
    """
    K = _piece_kernels(lam)
    return K["d1"] + K["d2"], K["x1"] + K["x2"]


def _contract(K, D):
    return np.einsum("prsq,sq->pr", K, np.asarray(D, dtype=np.complex128))


def shell_kernels(sc_shell: dict) -> dict:
    """``{"l", "label", "ae": {"dir", "x"}, "ps": {"dir", "x"} | None}`` (Hartree kernels) of one sidecar shell."""
    out = {"l": int(sc_shell["l"]), "label": str(sc_shell["label"])}
    for kind in ("ae", "ps"):
        t = sc_shell.get(kind)
        if t is None:
            out[kind] = None
            continue
        Kd, Kx = meanfield_kernels(t["lam"])
        out[kind] = {"dir": Kd, "x": Kx}
    return out


def resolve_onsite_keys(double_counting, exchange, *, bispinor=None, tt_exchange=None):
    """The one check of ``onsite_breit_double_counting`` / ``onsite_breit_exchange``.

    Values must be named (no defaults).  With ``bispinor`` given:
    ``onsite_only`` needs ``bispinor = false`` and ``paw`` needs ``true``.
    With ``tt_exchange`` (whether Sigma carries the bare transverse exchange on
    pseudo-orbitals) given: ``exchange = paw`` needs it, ``exchange = full``
    must not have it.

    Returns
    -------
    (double_counting, exchange) : the validated values.
    """
    dc = None if double_counting is None else str(double_counting).strip()
    ex = None if exchange is None else str(exchange).strip()
    if dc not in DOUBLE_COUNTING_VALUES:
        raise _gate("onsite_breit_keys", f"onsite_breit_double_counting = {double_counting!r}",
                    " | ".join(DOUBLE_COUNTING_VALUES),
                    "the direct term's double counting with the smooth alpha.A field has no default; "
                    "a wrong choice counts it twice or subtracts a term that is not there",
                    "set onsite_breit_double_counting (onsite_only for bispinor = false, paw for bispinor = true)")
    if ex not in EXCHANGE_VALUES:
        raise _gate("onsite_breit_keys", f"onsite_breit_exchange = {exchange!r}", " | ".join(EXCHANGE_VALUES),
                    "the exchange part has no default (full: no TT exchange in H; paw: Sigma has it; off)",
                    "set onsite_breit_exchange")
    if bispinor is not None:
        want = "paw" if bool(bispinor) else "onsite_only"
        if dc != want:
            raise _gate("onsite_breit_keys", f"onsite_breit_double_counting = {dc}, bispinor = {bool(bispinor)}",
                        f"onsite_breit_double_counting = {want}",
                        "onsite_only with bispinor = true counts the direct term twice; paw with bispinor = false "
                        "subtracts a pseudo term the run does not have",
                        f"set onsite_breit_double_counting = {want}")
    if tt_exchange is not None:
        if ex == "paw" and not tt_exchange:
            raise _gate("onsite_breit_keys", "onsite_breit_exchange = paw, Sigma without TT exchange",
                        "exchange = paw only when Sigma carries the bare transverse exchange",
                        "paw subtracts a pseudo exchange that is not in H", "use onsite_breit_exchange = full or off")
        if ex == "full" and tt_exchange:
            raise _gate("onsite_breit_keys", "onsite_breit_exchange = full, Sigma with TT exchange",
                        "exchange = full only when no transverse exchange on pseudo-orbitals is in H",
                        "the pseudo part of the on-site exchange would be counted twice",
                        "use onsite_breit_exchange = paw")
    return dc, ex


def onsite_potential(D, kernels: dict, *, double_counting: str, exchange: str) -> dict:
    """On-site Breit potential of one shell-atom, by piece (design note section 3).

    Parameters
    ----------
    D : (2n, 2n) complex
        Shell density matrix of this atom (shell basis).
    kernels : dict
        :func:`shell_kernels` of the atom's element and shell.
    double_counting, exchange : str
        The two keys (:func:`resolve_onsite_keys`).

    Returns
    -------
    dict of (2n, 2n) complex128 in Ry: ``dir_ae`` = h^dir_AE[D], ``dir_ps``,
    ``x_ae``, ``x_ps`` (raw pieces; a ``*_ps`` piece is ``None`` when the
    sidecar has no ``ps``), and ``total``::

        total = h^dir_AE - [dc = paw] h^dir_PS + {full: h^x_AE, paw: h^x_AE - h^x_PS, off: 0}
    """
    dc, ex = resolve_onsite_keys(double_counting, exchange)
    ps = kernels.get("ps")
    if ps is None and (dc == "paw" or ex == "paw"):
        raise _gate("onsite_breit_sidecar", f"{kernels.get('label')}: ps = null",
                    "a 'ps' tensor when either key says paw",
                    "the PAW-style combination subtracts the pseudo on-site term", "regenerate the sidecar with ps")
    ae = kernels["ae"]
    out = {"dir_ae": HA_TO_RY * _contract(ae["dir"], D), "x_ae": HA_TO_RY * _contract(ae["x"], D),
           "dir_ps": None if ps is None else HA_TO_RY * _contract(ps["dir"], D),
           "x_ps": None if ps is None else HA_TO_RY * _contract(ps["x"], D)}
    total = out["dir_ae"].copy()
    if dc == "paw":
        total = total - out["dir_ps"]
    if ex == "full":
        total = total + out["x_ae"]
    elif ex == "paw":
        total = total + out["x_ae"] - out["x_ps"]
    out["total"] = total
    return out


def onsite_block_potential(D, setup: "OnsiteSetup"):
    """Block-diagonal potential of every shell-atom of ``setup``.

    ``D`` (R_os, R_os) from :func:`onsite_density_matrix`; only the diagonal
    shell-atom blocks are used.  Returns ``(h_block (R_os, R_os) complex128
    Ry, pieces)`` with ``pieces[i]`` the :func:`onsite_potential` dict of
    ``setup.shell_atoms[i]``.
    """
    D = np.asarray(D, dtype=np.complex128)
    h = np.zeros((setup.R_os, setup.R_os), np.complex128)
    pieces = []
    for sa in setup.shell_atoms:
        sl = slice(sa["start"], sa["stop"])
        pc = onsite_potential(D[sl, sl], setup.kernels[(sa["element"], sa["label"])],
                              double_counting=setup.double_counting, exchange=setup.exchange)
        h[sl, sl] = pc["total"]
        pieces.append(pc)
    return h, pieces


def shell_moments(D, l: int) -> dict:
    """``N = Tr D``, ``S = Tr(D S_c)``, ``L = Tr(D L_c)`` (Cartesian, hbar = 1) and ``herm = max|D - D^+|``."""
    D = np.asarray(D, dtype=np.complex128)
    o = shell_operators(l)
    return dict(N=float(np.trace(D).real), S=np.array([np.trace(D @ s).real for s in o["S"]]),
                L=np.array([np.trace(D @ x).real for x in o["L"]]),
                herm=float(np.abs(D - D.conj().T).max()))


def decompose_potential(h, l: int, mhat) -> dict:
    """Real coefficients of ``1, L.m, S.m, (L.m)(S.m), l.s`` in a shell matrix (least squares, Frobenius).

    Returns ``{"I", "Lm", "Sm", "LmSm", "ls"}`` (units of h), ``rest`` (Frobenius
    norm of the remainder) and ``rest_rel`` (rest / |h|).
    """
    h = np.asarray(h, dtype=np.complex128)
    m = np.asarray(mhat, dtype=np.float64)
    m = m / np.linalg.norm(m)
    o = shell_operators(l)
    Lm = sum(mi * x for mi, x in zip(m, o["L"]))
    Sm = sum(mi * x for mi, x in zip(m, o["S"]))
    names = ("I", "Lm", "Sm", "LmSm", "ls")
    B = np.stack([o["I"], Lm, Sm, Lm @ Sm, o["ls"]]).reshape(5, -1).T
    A = np.concatenate([B.real, B.imag])
    y = np.concatenate([h.reshape(-1).real, h.reshape(-1).imag])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    rest = h.reshape(-1) - B @ c
    nh = np.linalg.norm(h)
    out = {k: float(v) for k, v in zip(names, c)}
    out["rest"] = float(np.linalg.norm(rest))
    out["rest_rel"] = float(out["rest"] / nh) if nh > 0 else 0.0
    return out


# ---------------------------------------------------------------------------
# Sidecar (design note section 6)
# ---------------------------------------------------------------------------

def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _array(x, shape, what, path):
    """Decode a JSON array: nested lists, a flat list, or {"shape", "data"}; check the shape."""
    if isinstance(x, dict):
        shp = tuple(int(s) for s in x.get("shape", ()))
        data = x.get("data", x.get("flat"))
        a = np.asarray(data, dtype=np.float64)
        if shp and a.size == int(np.prod(shp)):
            a = a.reshape(shp)
    else:
        a = np.asarray(x, dtype=np.float64)
    if shape is not None:
        if a.ndim == 1 and a.size == int(np.prod(shape)) and len(shape) > 1:
            a = a.reshape(shape)
        if a.shape != tuple(shape):
            raise _gate("onsite_breit_sidecar", f"{path}: {what} has shape {a.shape}", f"shape {tuple(shape)}",
                        "a tensor of the wrong shape is a different basis or a truncated file",
                        "regenerate the sidecar")
    return a


def _closed_shell_check(lam, l, where):
    """d1 = d2 = 0 and x1 + x2 = c l.s, c > 0, at D = 1 (relative 1e-10); returns c (Hartree)."""
    n = 2 * l + 1
    I = np.eye(2 * n)
    h = meanfield_pieces(lam, I)
    x = h["x1"] + h["x2"]
    ls = shell_operators(l)["ls"]
    scale = float(np.abs(x).max())
    dmax = max(float(np.abs(h["d1"]).max()), float(np.abs(h["d2"]).max()))
    c = float(np.vdot(ls, x).real / np.vdot(ls, ls).real) if l > 0 else 0.0
    resid = float(np.linalg.norm(x - c * ls) / np.linalg.norm(x)) if scale > 0 else np.inf
    if not (scale > 0 and dmax <= CLOSED_SHELL_RTOL * scale and resid <= CLOSED_SHELL_RTOL and c > 0):
        raise _gate("onsite_breit_sidecar",
                    f"{where}: closed shell max|d1|,|d2| = {dmax:.2e} Ha, |x| = {scale:.2e} Ha, "
                    f"|x - c l.s|/|x| = {resid:.2e}, c = {c:+.3e} Ha",
                    f"d1 = d2 = 0 and x1 + x2 = c l.s with c > 0, to {CLOSED_SHELL_RTOL:g} relative",
                    "a tensor that fails the closed-shell identity is corrupted or in another index order",
                    "regenerate the sidecar from its tensor npz")
    return c


def read_sidecar(path, *, require_ps: bool = False) -> dict:
    """Parse and verify one ``<El>.onsite_breit.json`` (``lorrax.onsite_breit_sidecar/v1``).

    Checks (``GATE onsite_breit_sidecar``): the format string; every shell's
    l, projector mesh/chi lengths and tensor shapes ``(3, n, n, n, n)``,
    n = 2l+1; ``ps`` present when ``require_ps``; the closed-shell identity
    of every tensor (D = 1: d1 = d2 = 0, x1 + x2 = c l.s, c > 0, 1e-10
    relative).  The UPF authentication of ``projector.chi`` needs the deck's
    UPF and is done by :func:`build_onsite_setup`.

    Returns
    -------
    dict
        ``format, element, conventions, provenance, path, sha256`` and
        ``shells``: a list of ``{"label", "l", "projector": {"type",
        "function", "r", "chi"}, "ae": T, "ps": T | None}`` with
        ``T = {"reference", "lam" (3,n,n,n,n) complex128, "I0", "r3",
        "b1_mev_per_muB", "x_closed_mev", "x_closed_ha_check"}``.
    """
    path = str(path)
    sha = _sha256_file(path)
    with open(path) as f:
        raw = json.load(f)
    fmt = raw.get("format")
    if fmt != SIDECAR_FORMAT:
        raise _gate("onsite_breit_sidecar", f"{path}: format = {fmt!r}", f"format = {SIDECAR_FORMAT!r}",
                    "an unknown sidecar format has unknown conventions", "regenerate with make_sidecar.py")
    shells = []
    for ish, s in enumerate(raw.get("shells") or []):
        l = int(s["l"])
        if not 0 <= l <= 3:
            raise _gate("onsite_breit_sidecar", f"{path}: shell {ish} l = {l}", "0 <= l <= 3",
                        "the atomic rows exist only to l = 3", "fix the sidecar")
        n = 2 * l + 1
        pj = s.get("projector") or {}
        r = _array(pj.get("r", []), None, "projector.r", path)
        chi = _array(pj.get("chi", []), None, "projector.chi", path)
        # type "none" (test fixtures: the shell IS the radial function) carries no
        # projector; build_onsite_setup refuses it by type.
        if str(pj.get("type")) != "none" and (r.ndim != 1 or chi.shape != r.shape or r.size < 2):
            raise _gate("onsite_breit_sidecar", f"{path}: projector r {r.shape}, chi {chi.shape}",
                        "two equal-length 1-d arrays", "chi authenticates the projector against the UPF",
                        "regenerate the sidecar")
        entry = {"label": str(s["label"]), "l": l,
                 "projector": {"type": str(pj.get("type")), "function": str(pj.get("function", "")),
                               "r": r, "chi": chi}}
        for kind in ("ae", "ps"):
            t = s.get(kind)
            if t is None:
                entry[kind] = None
                continue
            if "shape" in t and [int(x) for x in t["shape"]] != [3, n, n, n, n]:
                raise _gate("onsite_breit_sidecar", f"{path}: shell {ish} {kind}.shape = {t['shape']}",
                            f"[3, {n}, {n}, {n}, {n}] (l = {l})",
                            "a tensor of the wrong shape is a different basis or a truncated file",
                            "regenerate the sidecar")
            lam = (_array(t["lam_re"], (3, n, n, n, n), f"shell {ish} {kind}.lam_re", path)
                   + 1j * _array(t["lam_im"], (3, n, n, n, n), f"shell {ish} {kind}.lam_im", path))
            c = _closed_shell_check(lam, l, f"{path}: {s['label']} {kind}")
            entry[kind] = {"reference": str(t.get("reference", "")), "lam": lam,
                           "I0": float(t["I0"]), "r3": float(t["r3"]),
                           "b1_mev_per_muB": float(t["b1_mev_per_muB"]),
                           "x_closed_mev": float(t["x_closed_mev"]), "x_closed_ha_check": c}
        if entry["ae"] is None:
            raise _gate("onsite_breit_sidecar", f"{path}: {s['label']} ae = null", "an 'ae' tensor",
                        "the all-electron tensor is the operator", "regenerate the sidecar")
        if require_ps and entry["ps"] is None:
            raise _gate("onsite_breit_sidecar", f"{path}: {s['label']} ps = null",
                        "a 'ps' tensor (a key says paw)",
                        "the PAW-style combination subtracts the pseudo on-site term",
                        "regenerate the sidecar with ps, or use onsite_only / full")
        shells.append(entry)
    if not shells:
        raise _gate("onsite_breit_sidecar", f"{path}: no shells", "at least one shell",
                    "an empty sidecar adds nothing", "regenerate the sidecar")
    return {"format": fmt, "element": str(raw.get("element")), "conventions": raw.get("conventions", ""),
            "provenance": raw.get("provenance", {}), "path": path, "sha256": sha, "shells": shells}


# ---------------------------------------------------------------------------
# Projector types (registry) and the setup
# ---------------------------------------------------------------------------

def _ortho_atomic_build(wfn, upf_paths, nspinor):
    """``(rows_setup, row_labels)``: every j-averaged PP_PSWFC function of every atom (QE natomwfc)."""
    from psp.hubbard_ops import build_atwfc_setup
    return build_atwfc_setup(wfn, dict(upf_paths), nspinor=nspinor)


def _ortho_atomic_rows_at_k(kvec, gvec, gmask, rows_setup, idx):
    """Loewdin-orthogonalised ket rows ``Z~[idx]`` (n_sel, nG) at one k (traced); ``P = conj(Z~) psi``.

    The rows are MASKED before the overlap, as in
    ``hubbard_ops.apply_hubbard_velocity_to_ket``: vnl_ops' Z is finite on a
    pad column, and ``O = Z^+ Z`` never sees psi's mask.
    """
    from psp import vnl_ops
    from psp.hubbard_ops import lowdin_rows_with_derivative
    kd = vnl_ops.build_vnl_kdata_traced(kvec, gvec, rows_setup, compute_dZ=False)
    Z = kd.Z * gmask.astype(kd.Z.dtype)[None, :]
    # The one Loewdin owner; its derivative branch is fed one zero component.
    Zt, _, _ = lowdin_rows_with_derivative(Z, jnp.zeros((1,) + Z.shape, Z.dtype), idx)
    return Zt


#: Projector types: name -> {"build": (wfn, upf_paths, nspinor) -> (rows_setup, row_labels),
#: "rows_at_k": (kvec, gvec, gmask, rows_setup, idx) -> (n_sel, nG) ket rows}.  row_labels[r] =
#: (atom, element, label, l, m) with m the QE real-Y_lm index.  A second type is one more entry.
_PROJECTORS = {
    "ortho-atomic": {"build": _ortho_atomic_build, "rows_at_k": _ortho_atomic_rows_at_k},
}


@dataclass(frozen=True)
class OnsiteSetup:
    """k-independent on-site Breit data (one per run).

    ``rows`` is the projector type's row setup (ortho-atomic: a
    ``vnl_ops.VNLSetup`` over all atomic functions); ``os_idx`` (n_sp,) int32
    the spatial rows of every shell-atom, concatenated in ``shell_atoms``
    order, m in QE real order; ``T`` (R_os, 2, n_sp) complex128 maps
    ``<Z~_i|psi_s>`` to the shell basis (``conj(W)`` blocks, spin-major);
    ``shell_atoms[i]`` = ``{"atom", "element", "label", "l", "start", "stop",
    "rows", "W"}`` with ``[start, stop)`` its slice of the flat on-site axis
    (length 2(2l+1)); ``kernels[(element, label)]`` the :func:`shell_kernels`;
    ``sidecar_sha256[element]``; ``provenance`` a canonical JSON string.
    """
    projector: str
    rows: object
    row_labels: tuple
    os_idx: np.ndarray
    T: np.ndarray
    shell_atoms: tuple
    kernels: dict
    double_counting: str
    exchange: str
    sidecar_sha256: dict
    provenance: str

    @property
    def R_os(self) -> int:
        return int(self.T.shape[0])


def _upf_is_us_or_paw(path) -> bool:
    from psp.upf.load_upf import parse_upf_xml
    root = parse_upf_xml(path)
    hdr = next((c for c in root if (c.tag.split("}")[-1] if isinstance(c.tag, str) else "") == "PP_HEADER"), None)
    if hdr is None:
        return False

    def true(v):
        return str(v or "").strip().strip(".").upper() in ("T", "TRUE")
    return (true(hdr.get("is_ultrasoft")) or true(hdr.get("is_paw"))
            or str(hdr.get("pseudo_type", "")).strip().upper() in ("US", "USPP", "PAW"))


def _authenticate_chi(sc, sh, upf_path):
    """The sidecar's projector chi equals the UPF's j-averaged PP_PSWFC (max dev <= 1e-8 max|chi|)."""
    from psp.hubbard_ops import j_averaged_atomic_wfcs
    from psp.upf.load_upf import load_upf_pswfc
    pswfc = load_upf_pswfc(upf_path)
    wf = [w for w in j_averaged_atomic_wfcs(pswfc)
          if w["label"].upper() == sh["label"].upper() and int(w["l"]) == sh["l"]]
    r_upf = np.asarray(pswfc["r"], dtype=np.float64)
    r_sc, chi_sc = sh["projector"]["r"], sh["projector"]["chi"]
    if len(wf) != 1:
        raise _gate("onsite_breit_sidecar", f"{sc['path']}: shell {sh['label']} l={sh['l']}; UPF {upf_path} "
                    f"has {len(wf)} such PP_PSWFC functions", "exactly one",
                    "the projector is the UPF's own atomic function", "use the UPF the sidecar was made from")
    chi = np.asarray(wf[0]["chi"], dtype=np.float64)
    n = min(len(chi), len(r_upf))
    ok_mesh = (len(r_sc) == n and np.abs(r_sc - r_upf[:n]).max() <= 1e-10 * np.abs(r_upf[:n]).max())
    dev = float(np.abs(chi_sc - chi[:n]).max()) if len(chi_sc) == n else np.inf
    ref = float(np.abs(chi[:n]).max())
    if not ok_mesh or not dev <= CHI_RTOL * ref:
        raise _gate("onsite_breit_sidecar",
                    f"{sc['path']}: {sh['label']} chi vs {upf_path}: mesh match {bool(ok_mesh)}, "
                    f"max|dchi| = {dev:.3e} (max|chi| = {ref:.3e})",
                    f"same mesh and max|dchi| <= {CHI_RTOL:g} max|chi|",
                    "the radial integrals belong to another atomic function than the projector of this run",
                    "regenerate the sidecar from this UPF (a Gaunt-corrected UPF keeps PP_PSWFC and passes)")


def build_onsite_setup(wfn, upf_paths: dict, sidecars, *, projector: str, double_counting: str,
                       exchange: str) -> OnsiteSetup:
    """Projector rows, on-site row map, kernels and provenance for one run.

    Parameters
    ----------
    wfn
        WFN loader/header (``nspinor``, ``atom_types``, ``atom_crys``,
        ``blat``, ``bvec``, ``cell_volume``, ``ecutwfc``).
    upf_paths : dict
        element -> UPF path, for EVERY species of the WFN (the ortho-atomic
        Loewdin runs over all atomic functions of all atoms).
    sidecars : dict or sequence
        element -> sidecar path, or a sequence of paths (element read from
        each file), or already-read :func:`read_sidecar` dicts.
    projector, double_counting, exchange : str
        Deck keys ``onsite_breit_projector`` / ``_double_counting`` / ``_exchange``.

    Refusals: ``GATE onsite_breit_projector`` (nspinor != 2, ultrasoft/PAW,
    unknown projector name, missing species UPF), ``GATE onsite_breit_sidecar``
    (sidecar checks of :func:`read_sidecar`, element not in the WFN,
    projector type, chi authentication, missing shell rows),
    ``GATE onsite_breit_keys``.
    """
    from psp.pseudos import symbol_to_Z
    dc, ex = resolve_onsite_keys(double_counting, exchange)
    need_ps = dc == "paw" or ex == "paw"
    nspinor = int(getattr(wfn, "nspinor", 1))
    if nspinor != 2:
        raise _gate("onsite_breit_projector", f"nspinor = {nspinor}", "nspinor = 2 (noncollinear WFN)",
                    "the shell basis has pure spinors on both components; a collinear WFN has no spin-flip "
                    "density matrix", "run the DFT noncollinear")
    if projector not in _PROJECTORS:
        raise _gate("onsite_breit_projector", f"onsite_breit_projector = {projector!r}",
                    " | ".join(sorted(_PROJECTORS)),
                    "implementation limit: only the named projector types are built",
                    "use ortho-atomic, or add a type to psp.onsite_breit._PROJECTORS")
    atom_types = np.asarray(wfn.atom_types, dtype=int)
    have_Z = {int(symbol_to_Z(el)) for el in upf_paths}
    missing = sorted(set(int(z) for z in atom_types) - have_Z)
    if missing:
        raise _gate("onsite_breit_projector", f"UPFs for {sorted(upf_paths)}; WFN species Z = {missing} missing",
                    "a UPF with PP_PSWFC for every species", "the ortho-atomic Loewdin runs over all atomic "
                    "functions of all atoms; a subset is another projector", "set onsite_breit_pseudo_dir")
    for el, p in upf_paths.items():
        if _upf_is_us_or_paw(p):
            raise _gate("onsite_breit_projector", f"{el}: {p} is ultrasoft/PAW", "norm-conserving (S = 1)",
                        "the ortho-atomic overlap here assumes S = 1", "use norm-conserving pseudopotentials")

    if isinstance(sidecars, dict):
        items = list(sidecars.values())
    else:
        items = list(sidecars)
    scs = [s if isinstance(s, dict) else read_sidecar(s, require_ps=need_ps) for s in items]
    by_el = {}
    for sc in scs:
        el = sc["element"]
        if symbol_to_Z(el) not in set(int(z) for z in atom_types) or el not in upf_paths:
            raise _gate("onsite_breit_sidecar", f"{sc['path']}: element {el}",
                        f"a species of the WFN with a UPF ({sorted(upf_paths)})",
                        "a sidecar for an absent species is a deck error", "fix onsite_breit_files")
        if el in by_el:
            raise _gate("onsite_breit_sidecar", f"two sidecars for {el}", "one per species",
                        "two operators for one shell", "fix onsite_breit_files")
        by_el[el] = sc
        for sh in sc["shells"]:
            if sh["projector"]["type"] != projector:
                raise _gate("onsite_breit_sidecar", f"{sc['path']}: projector.type = {sh['projector']['type']!r}",
                            f"{projector!r} (onsite_breit_projector)",
                            "the radial integrals were authenticated for another projector",
                            "use the matching sidecar or key")
            if need_ps and sh["ps"] is None:
                raise _gate("onsite_breit_sidecar", f"{sc['path']}: {sh['label']} ps = null",
                            "a 'ps' tensor (a key says paw)",
                            "the PAW-style combination subtracts the pseudo on-site term",
                            "regenerate the sidecar with ps")
            _authenticate_chi(sc, sh, upf_paths[el])

    ptype = _PROJECTORS[projector]
    rows, labels = ptype["build"](wfn, upf_paths, nspinor)
    labels = tuple(tuple(x) for x in labels)
    entries = []
    for el, sc in by_el.items():
        z = symbol_to_Z(el)
        for sh in sc["shells"]:
            for a in [i for i in range(len(atom_types)) if atom_types[i] == z]:
                rr = [i for i, (aa, e, lab, ll, m) in enumerate(labels)
                      if aa == a and e == el and lab.upper() == sh["label"].upper() and ll == sh["l"]]
                rr = sorted(rr, key=lambda i: labels[i][4])
                if len(rr) != 2 * sh["l"] + 1:
                    raise _gate("onsite_breit_sidecar", f"atom {a} ({el}): {len(rr)} rows of {sh['label']} "
                                f"l={sh['l']}", f"{2 * sh['l'] + 1} rows", "the shell must be a PP_PSWFC "
                                "function of this atom's UPF", "use the UPF the sidecar was made from")
                entries.append((a, el, sh["label"], sh["l"], rr))
    entries.sort(key=lambda e: (e[0], e[3], e[2]))
    Ws = {l: real_to_complex_ylm(l) for l in {e[3] for e in entries}}
    n_sp = sum(2 * e[3] + 1 for e in entries)
    R_os = 2 * n_sp
    T = np.zeros((R_os, 2, n_sp), np.complex128)
    os_idx, shell_atoms = [], []
    cur_r, cur_i = 0, 0
    for a, el, lab, l, rr in entries:
        n = 2 * l + 1
        W = Ws[l]
        for s in range(2):
            # <phi^c_mu, s|psi> = sum_i conj(W[i, mu]) <phi^r_i|psi_s>
            T[cur_r + s * n: cur_r + (s + 1) * n, s, cur_i:cur_i + n] = W.conj().T
        shell_atoms.append({"atom": int(a), "element": el, "label": lab, "l": int(l),
                            "start": cur_r, "stop": cur_r + 2 * n, "rows": tuple(int(i) for i in rr), "W": W})
        os_idx.extend(rr)
        cur_r += 2 * n
        cur_i += n
    kernels = {(el, sh["label"]): shell_kernels(sh) for el, sc in by_el.items() for sh in sc["shells"]}
    sha = {el: sc["sha256"] for el, sc in by_el.items()}
    prov = json.dumps({
        "scheme": PROVENANCE_SCHEME, "projector": projector, "double_counting": dc, "exchange": ex,
        "alpha": ALPHA, "sidecars": [{"element": el, "path": sc["path"], "sha256": sc["sha256"]}
                                     for el, sc in sorted(by_el.items())],
        "shell_atoms": [{k: sa[k] for k in ("atom", "element", "label", "l")} for sa in shell_atoms]},
        sort_keys=True)
    return OnsiteSetup(projector=projector, rows=rows, row_labels=labels,
                       os_idx=np.asarray(os_idx, dtype=np.int32), T=T, shell_atoms=tuple(shell_atoms),
                       kernels=kernels, double_counting=dc, exchange=ex, sidecar_sha256=sha, provenance=prov)


# ---------------------------------------------------------------------------
# Band-sized kernels (JAX)
# ---------------------------------------------------------------------------

_PROJ_KERNELS: dict = {}
_DM_KERNELS: dict = {}
_BAND_KERNELS: dict = {}


def _axes_of(entry) -> tuple:
    if entry is None:
        return ()
    return (entry,) if isinstance(entry, str) else tuple(entry)


def onsite_projections(psi_G, gvecs, gmask, kvecs, setup: OnsiteSetup, *, mesh):
    """Projection table ``P[k, a, n] = <phi~_a(k)|psi_nk>`` in the shell basis.

    Parameters
    ----------
    psi_G : (nk, nb, 2, ngkmax) complex128
        Two-component G-sphere psi of the loader, band-sharded over the whole
        mesh (``common.wfn_layout.band_sphere_spec()``); a band count that
        does not divide the mesh is zero-padded here (pad bands give P = 0).
    gvecs : (nk, ngkmax, 3) int, gmask : (nk, ngkmax) float, kvecs : (nk, 3) float
        ``psp.dft_operators.padded_gvectors`` (1 on physical G, 0 on the pad).
    setup : OnsiteSetup
    mesh : jax.sharding.Mesh
        The run's mesh with the axes named in ``band_sphere_spec``.

    Returns
    -------
    (nk, R_os, nb) complex128, REPLICATED (``P(None, None, None)``): the table
    is small (R_os << nb) and every consumer (density matrix, band matrix)
    contracts it whole.

    One ``shard_map`` over the band axis with a ``lax.scan`` over k inside;
    per k the projector type builds its orthonormal rows (ortho-atomic:
    ``vnl_ops.build_vnl_kdata_traced``, masked, Loewdin), this rank's bands
    are projected, and ``T`` (``W^dagger`` per spin block) maps to the shell
    basis.  The jitted program is cached by its static signature.
    """
    from jax import shard_map
    from common.wfn_layout import band_sphere_spec
    from runtime.padding import pad_axis

    spec = band_sphere_spec()
    psi = jnp.asarray(psi_G, dtype=jnp.complex128)
    nk, nb_in, ns, ngkmax = psi.shape
    if ns != 2:
        raise ValueError(f"onsite_projections: psi_G must be two-component (nk, nb, 2, ngkmax); got {psi.shape}")
    psi = pad_axis(psi, mesh, axis=1, name="on-site projection bands", spec=spec).array
    band_axes = _axes_of(spec[1])
    rows_at_k = _PROJECTORS[setup.projector]["rows_at_k"]
    key = (tuple(psi.shape), int(ngkmax), mesh, id(setup))
    hit = _PROJ_KERNELS.get(key)
    if hit is None:
        idx = jnp.asarray(setup.os_idx, dtype=jnp.int32)
        T = jnp.asarray(setup.T)
        rows = setup.rows

        def per_rank(psi_l, g, m, kv):
            def one(c, x):
                psi_k, g_k, m_k, kv_k = x
                Zt = rows_at_k(kv_k, g_k, m_k, rows, idx)                         # (n_sp, nG)
                Pr = jnp.einsum("rG,nsG->rsn", jnp.conj(Zt), psi_k)               # (n_sp, 2, nb_l)
                return c, jnp.einsum("asr,rsn->an", T, Pr)                         # (R_os, nb_l)
            _, out = jax.lax.scan(one, None, (psi_l, g, m, kv))
            return out

        def run(psi_, g_, m_, kv_):
            psi_ = jax.lax.with_sharding_constraint(psi_, NamedSharding(mesh, spec))
            f = shard_map(per_rank, mesh=mesh, in_specs=(spec, P(), P(), P()),
                          out_specs=P(None, None, band_axes), check_vma=False)
            return f(psi_, g_, m_, kv_)

        fn = jax.jit(run, out_shardings=NamedSharding(mesh, P(None, None, None)))
        hit = _PROJ_KERNELS[key] = (fn, setup)
    Pt = hit[0](psi, jnp.asarray(gvecs, dtype=jnp.int32), jnp.asarray(gmask, dtype=jnp.float64),
                jnp.asarray(kvecs, dtype=jnp.float64))
    if not bool(jnp.all(jnp.isfinite(Pt))):
        raise FloatingPointError("onsite_projections: non-finite projection table (Loewdin overlap singular?)")
    return Pt[:, :, :nb_in] if Pt.shape[2] != nb_in else Pt


def onsite_density_matrix(P_table, kweights, occ, setup: OnsiteSetup, *, U=None, mesh) -> np.ndarray:
    """On-site density matrix ``D[a, b] = sum_k w_k sum_i f_ki (P U)_{k a i} conj((P U)_{k b i})``.

    Parameters
    ----------
    P_table : (nk, R_os, nb) complex128, replicated (:func:`onsite_projections`).
    kweights : (nk,) float, w_k (sum 1 on the full BZ).
    occ : (nk, nb_qp) float, occupations f of the states i (QP or DFT).
    U : (nk, nb, nb_qp) complex, optional
        ``U[k, n, i] = <DFT_n|QP_i>``; ``None`` = identity.  Replicated or
        mesh-sharded (any NamedSharding); it is contracted on device in one
        ``shard_map`` at its own layout and never gathered.
    mesh : jax.sharding.Mesh

    Returns
    -------
    (R_os, R_os) complex128 numpy array (only the shell-atom diagonal blocks
    are used downstream).
    """
    from jax import shard_map
    rep = NamedSharding(mesh, P())
    Pt = jnp.asarray(P_table, dtype=jnp.complex128)
    w = jnp.asarray(kweights, dtype=jnp.float64)
    f = jnp.asarray(occ, dtype=jnp.float64)
    if U is None:
        key = ("identity", mesh)
        fn = _DM_KERNELS.get(key)
        if fn is None:
            fn = _DM_KERNELS[key] = jax.jit(
                lambda Pt_, w_, f_: jnp.einsum("k,kn,kan,kbn->ab", w_, f_, Pt_, jnp.conj(Pt_)),
                out_shardings=rep)
        nb = f.shape[1]
        return np.asarray(fn(Pt[:, :, :nb], w, f))
    Ua = U if isinstance(U, jax.Array) else jnp.asarray(U)
    sh = getattr(Ua, "sharding", None)
    spec = tuple(sh.spec) if isinstance(sh, NamedSharding) and sh.mesh == mesh else ()
    if not spec:
        Ua = jax.device_put(Ua, rep)
    spec = tuple(spec) + (None,) * (3 - len(spec))
    kax, rax, cax = (_axes_of(e) for e in spec)
    nb_r, nb_c = int(Ua.shape[1]), int(Ua.shape[2])
    if Pt.shape[2] < nb_r:
        Pt = jnp.pad(Pt, ((0, 0), (0, 0), (0, nb_r - Pt.shape[2])))
    Pt = Pt[:, :, :nb_r]
    if f.shape[1] < nb_c:
        f = jnp.pad(f, ((0, 0), (0, nb_c - f.shape[1])))
    f = f[:, :nb_c]
    pspec = lambda *e: P(*[(a if len(a) != 1 else a[0]) if a else None for a in e])  # noqa: E731
    key = ("rotated", mesh, spec)
    fn = _DM_KERNELS.get(key)
    if fn is None:
        def per_rank(P_l, U_l, w_l, f_l):
            PU = jnp.einsum("kan,kni->kai", P_l, U_l)
            if rax:
                PU = jax.lax.psum(PU, rax)
            D = jnp.einsum("k,ki,kai,kbi->ab", w_l, f_l, PU, jnp.conj(PU))
            red = kax + cax
            return jax.lax.psum(D, red) if red else D

        fn = _DM_KERNELS[key] = jax.jit(shard_map(
            per_rank, mesh=mesh,
            in_specs=(pspec(kax, (), rax), pspec(kax, rax, cax), pspec(kax), pspec(kax, cax)),
            out_specs=P(), check_vma=False), out_shardings=rep)
    return np.asarray(fn(Pt, Ua, w, f))


def onsite_band_matrix(P_table, h_block, *, mesh):
    """Band matrix ``dH[k, n, n'] = sum_ab conj(P[k,a,n]) h[a,b] P[k,b,n']`` (units of h: Ry).

    Parameters
    ----------
    P_table : (nk, R_os, nb) complex128, replicated.
    h_block : (R_os, R_os) complex, the block-diagonal potential
        (:func:`onsite_block_potential`).
    mesh : jax.sharding.Mesh

    Returns
    -------
    (nk, nb, nb) complex128 at ``P(None, None, None)``: it is added to the
    replicated ``kin_ion`` / H_T carry of the same size
    (``gw.sc_iteration.qp_hamiltonian_sum`` keeps that layout), so a
    sharded result would only be resharded there.
    """
    fn = _BAND_KERNELS.get(mesh)
    if fn is None:
        fn = _BAND_KERNELS[mesh] = jax.jit(
            lambda Pt, h: jnp.einsum("kan,ab,kbm->knm", jnp.conj(Pt), h, Pt),
            out_shardings=NamedSharding(mesh, P(None, None, None)))
    return fn(jnp.asarray(P_table, dtype=jnp.complex128), jnp.asarray(h_block, dtype=jnp.complex128))


def onsite_report_lines(D, pieces, setup: OnsiteSetup) -> list:
    """Log lines per shell-atom: Tr D, 2<S>, <L>, max|D - D^+|, and per piece the
    coefficients of L.m, S.m, (L.m)(S.m), l.s (meV) with m = <S>/|<S>| (z if |<S>| < 1e-6)."""
    D = np.asarray(D, dtype=np.complex128)
    lines = []
    for sa, pc in zip(setup.shell_atoms, pieces):
        sl = slice(sa["start"], sa["stop"])
        mo = shell_moments(D[sl, sl], sa["l"])
        nS = np.linalg.norm(mo["S"])
        mhat = mo["S"] / nS if nS > 1e-6 else np.array([0.0, 0.0, 1.0])
        lines.append(
            f"onsite_breit atom {sa['atom']} {sa['element']} {sa['label']} (l={sa['l']}): Tr D = {mo['N']:.6f}  "
            f"2<S> = ({2 * mo['S'][0]:+.6f}, {2 * mo['S'][1]:+.6f}, {2 * mo['S'][2]:+.6f})  "
            f"<L> = ({mo['L'][0]:+.6f}, {mo['L'][1]:+.6f}, {mo['L'][2]:+.6f})  max|D-D^+| = {mo['herm']:.1e}  "
            f"m = ({mhat[0]:+.3f}, {mhat[1]:+.3f}, {mhat[2]:+.3f})")
        for name in ("dir_ae", "dir_ps", "x_ae", "x_ps", "total"):
            h = pc.get(name)
            if h is None:
                continue
            c = decompose_potential(h, sa["l"], mhat)
            lines.append(
                f"  {name:7s} [meV] L.m {c['Lm'] * RY_TO_MEV:+.5f}  S.m {c['Sm'] * RY_TO_MEV:+.5f}  "
                f"(L.m)(S.m) {c['LmSm'] * RY_TO_MEV:+.5f}  l.s {c['ls'] * RY_TO_MEV:+.5f}  "
                f"1 {c['I'] * RY_TO_MEV:+.5f}  rest {c['rest_rel']:.1e} rel  max|h| {np.abs(h).max() * RY_TO_MEV:.5f}")
    return lines

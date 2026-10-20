"""Column-sampled GW on general snapshot Keldysh transfer functions.

The screening coordinates have unit bare interaction: a Coulomb factor
``F(mu,r)`` maps a reduced interaction ``w(r,s)`` to ``F w F^H``.
``vertices[k,a,n,r] = sum_mu,spin psi[k,a]^* psi[k-q,n] F[mu,r]``
therefore contracts the same ISDF Hadamard product as the spatial Sigma
executor.  Electronic phases always multiply the right internal index.

The dense routines here operate on the small projected coordinates and one
q block at a time. They do not infer lesser fluctuations from retarded W.
All energy arguments and returned self-energies use the same supplied unit.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from tdagw import keldysh_rational as kr


def _dagger(a):
    return np.conj(np.swapaxes(a, -1, -2))


def _hermitian(a):
    return 0.5 * (a + _dagger(a))


def rational_factors(system, *, condition_limit=1e10, tolerance=1e-7):
    """Resolve a small transfer realization into checked biorthogonal poles.

    Parameters
    ----------
    system : tuple
        ``(A,B,C,D)`` with ``D=0``, in the convention of
        :mod:`tdagw.keldysh_rational`. General signed complex residues are
        admitted. A defective realization is refused instead of splitting a
        repeated pole by roundoff.
    condition_limit, tolerance : float
        Maximum eigenvector condition number and relative reconstruction
        error of the transfer at three upper-half-plane supports.

    Returns
    -------
    dict
        ``poles[p]``, ``left[n_out,p]``, ``right[p,n_in]``, so that
        ``W(z)=sum_p left[:,p] right[p,:]/(z-poles[p])``. Neither factor is
        assumed to be the adjoint of the other.
    """
    A, B, C, D = (np.asarray(a, np.complex128) for a in system)
    if np.max(np.abs(D), initial=0.0) > 1e-12:
        raise ValueError("GATE td_snapshot_sigma_direct: correlation/lesser W must be strictly proper")
    if A.shape[0] == 0:
        return dict(poles=np.empty(0, complex), left=np.empty((C.shape[0], 0), complex),
                    right=np.empty((0, B.shape[1]), complex), condition=1.0, residual=0.0)
    poles, vectors = sla.eig(A)
    condition = float(np.linalg.cond(vectors))
    if not np.isfinite(condition) or condition > condition_limit:
        raise ValueError("GATE td_snapshot_sigma_poles: ill-conditioned or repeated-pole "
                         f"realization (eigenvector condition {condition:.3g}); "
                         "use the state-space time executor or improve the joint reduction")
    left, right = C @ vectors, sla.solve(vectors, B)
    span = max(float(np.max(np.abs(poles))), 1.0)
    residual = 0.0
    for z in span * np.asarray([-0.37 + 0.51j, 0.23 + 0.87j, 1.13 + 0.43j]):
        reference = kr.evaluate(system, z)
        got = (left / (z - poles)[None]) @ right
        residual = max(residual, float(np.linalg.norm(got - reference)
                                       / max(np.linalg.norm(reference), 1e-30)))
    if not np.isfinite(residual) or residual > tolerance:
        raise ValueError(f"GATE td_snapshot_sigma_poles: transfer reconstruction {residual:.3g} "
                         f"exceeds {tolerance:g}")
    return dict(poles=poles, left=left, right=right, condition=condition, residual=residual)


def joint_rational_factors(retarded, lesser, *, condition_limit=1e10, tolerance=1e-7):
    """Factor one exact common union realization of response and noise.

    Both inputs are strictly proper ``(A,B,C,D)`` square transfer functions
    in the same reduced coordinates.  Their block-diagonal common A/B
    realizes the stacked output ``[WcR;Wless]``.  One checked factorization
    returns matching response/noise left factors and a common right factor.
    Greater amplitudes are their sum, preserving ``Wgreater=WcR+Wless``
    algebraically.  No second fit, pole merging or residue truncation occurs.
    Defective or unresolved repeated poles retain the explicit refusal of
    ``rational_factors``.
    """
    Ac, Bc, Cc, Dc = (np.asarray(x, complex) for x in retarded)
    Al, Bl, Cl, Dl = (np.asarray(x, complex) for x in lesser)
    if Dc.shape != Dl.shape or Dc.shape[0] != Dc.shape[1]:
        raise ValueError("GATE td_snapshot_sigma_shapes: response/noise coordinate axes differ")
    rank = Dc.shape[0]
    joint = (sla.block_diag(Ac, Al), np.vstack((Bc, Bl)),
             sla.block_diag(Cc, Cl), np.vstack((Dc, Dl)))
    factors = rational_factors(joint, condition_limit=condition_limit, tolerance=tolerance)
    factors["left_retarded"] = factors["left"][:rank]
    factors["left_lesser"] = factors["left"][rank:]
    factors["left_greater"] = factors["left_retarded"] + factors["left_lesser"]
    return factors


def _line_pairs(vertices, density, energies, external_energies, factors, eta):
    """Sum ordered lesser/greater electronic lines on the SAME union bank.

    Both results have shape ``[nk,na,na]``.  Scanning k bounds all
    ``[na,ni,pole_block]`` intermediates independently of the full k count.
    Each bank block synthesizes its right vertex and denominators once,
    then contracts the two left amplitudes with rho and I-rho respectively.
    """
    nk, na, ni, _ = vertices.shape
    lin = np.zeros((nk, na, na), np.complex128)
    lout = np.zeros_like(lin)
    if factors["poles"].size == 0:
        return lin, lout
    # At most 16 MiB per complex contraction tile when one pole fits; the
    # one-k scan also removes the previous nk-fold temporary amplification.
    width = min(64, max(1, (16 * 1024 ** 2) // (16 * max(na * ni, 1))))
    for k in range(nk):
        conjugate = vertices[k].conj()
        hole_vertices = np.einsum("amr,mn->anr", vertices[k], density[k], optimize=True)
        particle_vertices = vertices[k] - hole_vertices
        for start in range(0, factors["poles"].size, width):
            stop = start + width
            right = factors["right"][start:stop]
            poles = factors["poles"][start:stop]
            rv = np.einsum("ps,bns->bnp", right, conjugate, optimize=True)
            denominator = (external_energies[k, :, None, None] - energies[k, None, :, None]
                           - poles[None, None, :] + 1j * eta)
            rv /= denominator
            vl = np.einsum("anr,rp->anp", hole_vertices,
                            factors["left_lesser"][:, start:stop], optimize=True)
            lin[k] += np.einsum("anp,bnp->ab", vl, rv, optimize=True)
            vl = np.einsum("anr,rp->anp", particle_vertices,
                            factors["left_greater"][:, start:stop], optimize=True)
            lout[k] += np.einsum("anp,bnp->ab", vl, rv, optimize=True)
    return lin, lout


def q_sigma_operators(vertices, energies, density, external_energies, screening, *,
                      eta, active=None, static_wc0=None):
    """One q contribution to the full-matrix Markov coefficients.

    Parameters
    ----------
    vertices : ndarray (nk,na,ni,r)
        Instantaneous external/internal density vertices in Coulomb coordinates.
    energies : ndarray (nk,ni)
        Internal energies at k-q; all frozen virtual bands remain included.
    density : ndarray (nk,ni,ni)
        Hermitian internal density, including the full active coherence block.
    external_energies : ndarray (nk,na)
        Column shell energies at external k.
    screening : dict
        ``retarded`` and ``lesser`` strictly proper state-space interactions.
        ``retarded`` is W_c^R. W^> is formed as their sum algebraically.
    eta : float
        Positive collision regularization, independent of screening regulators.
    active : slice or array, optional
        Real-transition internal space. Only it contributes Gamma and D;
        all supplied internal states contribute M_H.
    static_wc0 : ndarray (r,r), optional
        Equilibrium W_c^R(0), fixed during the run. Its screened-exchange
        contraction supplies the separate coherent double-counting term.

    Returns
    -------
    dict
        ``gamma``, ``injection``, ``m_h``, ``static``, ``ain``, ``aout``;
        all ``(nk,na,na)`` in supplied energy units, unweighted in q.
    """
    vertices = np.asarray(vertices, np.complex128)
    density = np.asarray(density, np.complex128)
    energies, external_energies = (np.asarray(x, float) for x in (energies, external_energies))
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("GATE td_snapshot_sigma_eta: collision eta must be positive")
    nk, na, ni, rank = vertices.shape
    if (density.shape != (nk, ni, ni) or energies.shape != (nk, ni)
            or external_energies.shape != (nk, na)):
        raise ValueError("GATE td_snapshot_sigma_shapes: inconsistent electronic axes")
    factors = joint_rational_factors(screening["retarded"], screening["lesser"])
    # Sigma^< = -line[rho,W<], Sigma^> = +line[I-rho,W>].
    lin, lout = _line_pairs(vertices, density, energies, external_energies, factors, eta)
    m_h = _hermitian(lin + lout)
    if active is not None:
        indices = np.arange(ni)[active]
        va = vertices[:, :, indices]
        ra = density[:, indices][:, :, indices]
        ea = energies[:, indices]
        lin, lout = _line_pairs(va, ra, ea, external_energies, factors, eta)
    ain, aout = 1j * lin, 1j * lout
    m = lin + lout
    gamma = 1j * (m - _dagger(m))
    injection = ain + _dagger(ain)
    static = np.zeros_like(m_h)
    if static_wc0 is not None:
        static = -np.einsum("kamr,kmn,rs,kbns->kab", vertices, density,
                            static_wc0, np.conj(vertices), optimize=True)
        static = _hermitian(static)
    return dict(gamma=gamma, injection=injection, m_h=m_h, static=static,
                ain=ain, aout=aout)
